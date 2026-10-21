#!/usr/bin/env python3
"""Write path-free copies of the holdout-40 split tables into the repository.

The dataset folder (data/expanded_holdout40_v1) holds the split definition, but
two of its files contain absolute cluster paths and per-structure label values
(manifest.csv / train.csv / val.csv / test.csv: column `source_label_file`;
split_summary.json: key `label_run`).  This script writes copies that are safe
to commit:

  holdout40_records.csv         one row per structure (512): identifiers, split,
                                atom count, geometry hash, data-file hash.
                                Dropped: source_label_file (absolute path),
                                energy_eV_per_atom and max_force_eV_A (labels).
  holdout40_groups.csv          group_assignments.csv (group -> split, record ids)
  holdout40_split_summary.json  split_summary.json with paths made relative
  holdout_split_n40.csv         the family-level split table passed to
                                prepare_expanded_moe_splits_holdout.py --split-table

Nothing is read from or written to anywhere except the three paths you give.
Any cell or JSON string that still looks like a cluster path is rewritten to
`<path>/<name>` (or `$MOE_HET/...`) and reported, so a missed column cannot
leak a path silently.

Usage (from anywhere):
  python3 export_split_tables.py \\
      --data-dir  $P/data/expanded_holdout40_v1 \\
      --holdout-table $E/holdout_split_n40.csv \\
      --out  <repository>/data_splits
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, OrderedDict
from pathlib import Path

DROP_COLUMNS = ("source_label_file", "energy_eV_per_atom", "max_force_eV_A")
EXPECTED_TOTALS = {"train": 369, "val": 38, "test": 105}

PATH_TOKEN = re.compile(
    r"(?:/pscratch|/global/(?:cfs|homes|common|u\d)|/home/[\w.\-]+|/Users/[\w.\-]+)"
    r"/[^\s\"'`:;,)}\]]*"
)
HET_PREFIX = re.compile(r"(?:/pscratch|/global/[^\s\"']*?)/[^\s\"']*?/HETEROSTRUCTURES")

REPORT = Counter()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def scrub_string(s: str, where: str) -> str:
    if not PATH_TOKEN.search(s):
        return s
    out = HET_PREFIX.sub("$MOE_HET", s)
    out = PATH_TOKEN.sub(lambda m: "<path>/" + m.group(0).rstrip("/").rsplit("/", 1)[-1], out)
    REPORT[where] += 1
    return out


def scrub_json(obj, where: str):
    if isinstance(obj, str):
        return scrub_string(obj, where)
    if isinstance(obj, list):
        return [scrub_json(v, where) for v in obj]
    if isinstance(obj, dict):
        return OrderedDict((k, scrub_json(v, where + "." + str(k))) for k, v in obj.items())
    return obj


def read_csv(path: Path):
    with path.open(newline="") as f:
        r = csv.DictReader(f)
        return list(r.fieldnames or []), list(r)


def write_csv(path: Path, fields, rows) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def export_records(src: Path, dst: Path) -> None:
    fields, rows = read_csv(src)
    keep = [c for c in fields if c not in DROP_COLUMNS]
    missing = [c for c in DROP_COLUMNS if c not in fields]
    if missing:
        print("note: columns not present in %s: %s" % (src.name, ", ".join(missing)))
    for row in rows:
        for c in keep:
            row[c] = scrub_string(row.get(c) or "", "holdout40_records.csv:" + c)
    write_csv(dst, keep, rows)
    by = Counter((r.get("source", "?"), r.get("split", "?")) for r in rows)
    sources = sorted({s for s, _ in by})
    splits = ["train", "val", "test"]
    print("\nholdout40_records.csv: %d rows, %d columns (dropped: %s)"
          % (len(rows), len(keep), ", ".join(c for c in DROP_COLUMNS if c in fields)))
    print("  %-18s %6s %6s %6s" % ("source", *splits))
    for s in sources:
        print("  %-18s %6d %6d %6d" % (s, *[by.get((s, sp), 0) for sp in splits]))
    totals = {sp: sum(by.get((s, sp), 0) for s in sources) for sp in splits}
    print("  %-18s %6d %6d %6d" % ("total", *[totals[sp] for sp in splits]))
    if totals != EXPECTED_TOTALS:
        print("  WARNING: expected %s, found %s" % (EXPECTED_TOTALS, totals))
    else:
        print("  matches the paper split 369 / 38 / 105")
    ids = [r["record_id"] for r in rows]
    if len(ids) != len(set(ids)):
        print("  WARNING: duplicate record_id values")
    geo = [r["geometry_sha256"] for r in rows]
    if len(geo) != len(set(geo)):
        print("  WARNING: duplicate geometry_sha256 values")


def copy_checked_csv(src: Path, dst: Path) -> None:
    fields, rows = read_csv(src)
    for row in rows:
        for c in fields:
            row[c] = scrub_string(row.get(c) or "", dst.name + ":" + c)
    write_csv(dst, fields, rows)
    print("%s: %d rows, %d columns" % (dst.name, len(rows), len(fields)))


def export_summary(src: Path, dst: Path) -> None:
    data = json.loads(src.read_text(), object_pairs_hook=OrderedDict)
    data = scrub_json(data, "holdout40_split_summary.json")
    dst.write_text(json.dumps(data, indent=2) + "\n")
    print("holdout40_split_summary.json: %d top-level keys" % len(data))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-dir", type=Path, required=True,
                    help="data/expanded_holdout40_v1 (needs manifest.csv, group_assignments.csv, split_summary.json)")
    ap.add_argument("--holdout-table", type=Path, required=True,
                    help="holdout_split_n40.csv (family-level split table)")
    ap.add_argument("--out", type=Path, required=True, help="repository data_splits/ folder")
    a = ap.parse_args()

    need = [a.data_dir / n for n in ("manifest.csv", "group_assignments.csv", "split_summary.json")]
    need.append(a.holdout_table)
    bad = [str(p.name) for p in need if not p.is_file()]
    if bad:
        print("missing input file(s): " + ", ".join(bad), file=sys.stderr)
        return 2
    a.out.mkdir(parents=True, exist_ok=True)

    export_records(a.data_dir / "manifest.csv", a.out / "holdout40_records.csv")
    copy_checked_csv(a.data_dir / "group_assignments.csv", a.out / "holdout40_groups.csv")
    export_summary(a.data_dir / "split_summary.json", a.out / "holdout40_split_summary.json")
    copy_checked_csv(a.holdout_table, a.out / "holdout_split_n40.csv")

    if REPORT:
        print("\npath-like text was rewritten in:")
        for k, v in sorted(REPORT.items()):
            print("  %-60s %d" % (k, v))
    else:
        print("\nno path-like text found in the retained columns")

    print("\nsha256 of the written files:")
    for n in ("holdout40_records.csv", "holdout40_groups.csv",
              "holdout40_split_summary.json", "holdout_split_n40.csv"):
        print("  %s  %s" % (sha256(a.out / n)[:16], n))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
