#!/usr/bin/env python3
"""
generate_gulp_inputs.py

Writes GULP .gin inputs for AB structures and their isolated components.
Components are scoped PER-PAIR (case_id + framework name), never shared
across different pairs that happen to contain the same framework --
confirmed tonight that sharing caused silent wrong-registry substitution
(AEI/CHA pulled in a stale CHA component built for a different pair,
with b=23.46 A instead of the real 37.43 A).

Usage:
    python generate_gulp_inputs.py --pairs-csv pairs_20_total.csv --gulp-dir gulp-6.4
"""
import argparse
import sys
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-csv", required=True)
    ap.add_argument("--gulp-dir", required=True)
    ap.add_argument("--epizeo-src", default="Supplementary_Software1/epizeo_si/src")
    ap.add_argument("--ab-source-dir", default="Oishi_External_All_SiO2/structures")
    ap.add_argument("--comp-root", default="MATPES_Oishi_Component_References")
    args = ap.parse_args()

    sys.path.insert(0, args.epizeo_src)
    from ase.io import read as ase_read
    from pymatgen.io.ase import AseAtomsAdaptor
    from epizeo.gulp_io import GulpIO

    adaptor = AseAtomsAdaptor()
    gulp_dir = Path(args.gulp_dir)
    gulp_dir.mkdir(parents=True, exist_ok=True)

    pairs = pd.read_csv(args.pairs_csv)
    written_components = set()

    for _, row in pairs.iterrows():
        case_id, pa, pb = row["case_id"], row["parent_A"], row["parent_B"]

        ab_gin_path = gulp_dir / f"{case_id}_slc.gin"
        if ab_gin_path.exists():
            print(f"{case_id}: AB .gin already exists, skipping")
        else:
            ab_source = Path(args.ab_source_dir) / f"{case_id}.extxyz"
            if not ab_source.exists():
                print(f"{case_id}: WARNING AB source not found at {ab_source}, skipping AB")
            else:
                atoms = ase_read(ab_source)
                s = adaptor.get_structure(atoms)
                gin = GulpIO.slc_input(s)
                ab_gin_path.write_text(gin)
                print(f"{case_id}: {len(s)} atoms, wrote {ab_gin_path}")

        comp_dir = Path(args.comp_root) / f"{case_id}-IZA-OISHI-EXACT"
        for name in (pa, pb):
            comp_gin_path = gulp_dir / f"{case_id}_{name}_component_fixedcell_slc.gin"
            cache_key = (case_id, name)
            if cache_key in written_components or comp_gin_path.exists():
                written_components.add(cache_key)
                continue
            comp_source = comp_dir / f"{name}_coherent_initial.extxyz"
            if not comp_source.exists():
                print(f"  {name}: WARNING component source not found at {comp_source}, skipping")
                continue
            atoms = ase_read(comp_source)
            s = adaptor.get_structure(atoms)
            gin = GulpIO.slc_input(s)
            if not gin.startswith("opti conp\n"):
                print(f"  {name}: WARNING unexpected gin start, skipping to avoid a bad edit")
                continue

            body = gin[len("opti conp\n"):]
            lines = body.split("\n")
            assert lines[0] == "cell", f"{name}: expected 'cell' as first body line, got {lines[0]!r}"
            cell_params = lines[1].split()
            assert len(cell_params) == 6, f"{name}: expected 6 cell parameters, got {cell_params}"
            lines[1] = lines[1].rstrip() + "  0 0 1 0 0 0"
            assert lines[2] == "frac", f"{name}: expected 'frac' as third body line, got {lines[2]!r}"

            coord_end = None
            for i in range(3, len(lines)):
                if lines[i].strip() == "":
                    coord_end = i
                    break
            assert coord_end is not None, f"{name}: could not find end of coordinate block"

            n_coord_lines = coord_end - 3
            expected = s.composition.get("Si", 0) + 2 * s.composition.get("O", 0)
            assert n_coord_lines == expected, (
                f"{name}: expected {expected} coordinate lines (Si once, O core+shell), "
                f"found {n_coord_lines}"
            )

            for i in range(3, coord_end):
                lines[i] = lines[i].rstrip() + "  0 1 1 1 1"

            gin_fixed = "opti\n" + "\n".join(lines)
            comp_gin_path.write_text(gin_fixed)
            written_components.add(cache_key)
            print(f"  {name}: {len(s)} atoms, wrote {comp_gin_path} (atoms-only, fixed cell)")

    print(f"\nDone. {len(written_components)} distinct (pair, framework) components covered.")


if __name__ == "__main__":
    main()
