#!/usr/bin/env python3
"""
build_corrected_snapshot.py -- build a source snapshot carrying the CORRECTED
descriptor, in the exact form prepare_expanded_moe_features.py verifies.

Why this is not a copy-and-edit job.  That script does:

    SOAP = "torchsoap_autograd_module_periodic_regnorm.py"
    source_index = {str(Path(r["snapshot_path"]).resolve()): r
                    for r in io.read_csv(snapshot / "snapshot_provenance.csv")}
    paths = {name: snapshot / "src_snapshot" / name for name in (TRAINER, COMMON, SOAP)}
    ... raise ValueError("Source snapshot did not verify: " + str(path))

so three things must all line up in the NEW snapshot directory:

  1. the corrected module must sit at src_snapshot/<SOAP>, under the ORIGINAL
     filename.  A file named ..._rotinv_v1.py is simply not found;
  2. every row's `snapshot_path` must point into the NEW directory, because the
     lookup is keyed on the resolved snapshot_path -- a plain directory copy
     leaves them pointing at the old one and the lookup misses;
  3. the swapped file's `sha256` and `bytes` must match the corrected module,
     or verification fails.

Getting this wrong has two failure modes.  The loud one is a ValueError.  The
quiet one is worse: reusing the ORIGINAL snapshot succeeds and silently
computes features with the UNCORRECTED descriptor while every log says the run
was corrected.  This script exists so that cannot happen by accident, and it
verifies its own output before exiting.

    python build_corrected_snapshot.py --source-snapshot DIR --corrected-module FILE \
        --out DIR

The source snapshot is only read.
"""
import argparse
import csv
import hashlib
import shutil
import sys
from pathlib import Path

SOAP_NAME = "torchsoap_autograd_module_periodic_regnorm.py"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-snapshot", required=True,
                    help="the existing verified snapshot directory (read only)")
    ap.add_argument("--corrected-module", required=True,
                    help="the corrected descriptor module to install as %s" % SOAP_NAME)
    ap.add_argument("--out", required=True, help="new snapshot directory to create")
    ap.add_argument("--force", action="store_true", help="overwrite --out if it exists")
    args = ap.parse_args()

    src = Path(args.source_snapshot).expanduser().resolve()
    mod = Path(args.corrected_module).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()

    if not src.is_dir():
        raise SystemExit("FAIL: source snapshot is not a directory: %s" % src)
    if out == src:
        raise SystemExit("FAIL: --out is the source snapshot; refusing to modify it in place")
    prov_src = src / "snapshot_provenance.csv"
    if not prov_src.is_file():
        raise SystemExit("FAIL: no snapshot_provenance.csv in %s" % src)
    soap_src = src / "src_snapshot" / SOAP_NAME
    if not soap_src.is_file():
        raise SystemExit("FAIL: %s not found in %s/src_snapshot" % (SOAP_NAME, src))
    if not mod.is_file():
        raise SystemExit("FAIL: corrected module not found: %s" % mod)

    print("source snapshot   : %s" % src)
    print("  its %s" % SOAP_NAME)
    print("    sha256        : %s" % sha256_file(soap_src))
    print("corrected module  : %s" % mod)
    print("    sha256        : %s" % sha256_file(mod))
    print("    bytes         : %d" % mod.stat().st_size)
    print("new snapshot      : %s" % out)

    if out.exists():
        if not args.force:
            raise SystemExit("FAIL: %s already exists; pass --force to replace it" % out)
        shutil.rmtree(out)

    # 1. copy the whole snapshot, dropping __pycache__ so no stale bytecode of the
    #    uncorrected module can shadow the corrected source
    shutil.copytree(src, out, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    print("\ncopied, __pycache__ excluded")

    # 2. install the corrected module under the ORIGINAL filename
    soap_out = out / "src_snapshot" / SOAP_NAME
    shutil.copy2(mod, soap_out)
    new_sha = sha256_file(soap_out)
    new_bytes = soap_out.stat().st_size
    print("installed %s  sha256 %s  %d bytes" % (SOAP_NAME, new_sha, new_bytes))

    # 3. rewrite snapshot_provenance.csv: repoint every snapshot_path into the new
    #    directory, and update the swapped file's sha256 and bytes
    with open(prov_src, newline="") as f:
        rdr = csv.DictReader(f)
        rows = [dict(r) for r in rdr]
        cols = list(rdr.fieldnames or [])
    for need in ("snapshot_path", "sha256", "bytes"):
        if need not in cols:
            raise SystemExit("FAIL: provenance file lacks a %r column; columns are %s"
                             % (need, cols))

    s_src, s_out = str(src), str(out)
    repointed = updated = 0
    for r in rows:
        p = r["snapshot_path"]
        if p.startswith(s_src):
            r["snapshot_path"] = s_out + p[len(s_src):]
            repointed += 1
        if Path(r["snapshot_path"]).name == SOAP_NAME and \
                Path(r["snapshot_path"]).parent.name == "src_snapshot":
            r["sha256"] = new_sha
            r["bytes"] = str(new_bytes)
            updated += 1
    if updated != 1:
        raise SystemExit("FAIL: expected exactly 1 provenance row for src_snapshot/%s, "
                         "found %d" % (SOAP_NAME, updated))
    with open(out / "snapshot_provenance.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print("provenance: %d of %d rows repointed, 1 row's sha256/bytes updated"
          % (repointed, len(rows)))

    # 4. verify the way prepare_expanded_moe_features.py will
    print("\nverifying as prepare_expanded_moe_features.py does:")
    index = {str(Path(r["snapshot_path"]).resolve()): r
             for r in csv.DictReader(open(out / "snapshot_provenance.csv", newline=""))}
    ok = True
    names = [SOAP_NAME] + sorted(p.name for p in (out / "src_snapshot").iterdir()
                                 if p.is_file() and p.name != SOAP_NAME
                                 and p.suffix == ".py")
    for name in names:
        path = (out / "src_snapshot" / name).resolve()
        row = index.get(str(path))
        if row is None:
            print("  %-62s NOT IN PROVENANCE" % name)
            ok = False
            continue
        actual = sha256_file(path)
        good = (actual == row["sha256"] and str(path.stat().st_size) == str(row["bytes"]))
        ok = ok and good
        print("  %-62s %s" % (name, "verified" if good else "MISMATCH"))
        if not good:
            print("      provenance %s / %s bytes" % (row["sha256"][:16], row["bytes"]))
            print("      on disk    %s / %s bytes" % (actual[:16], path.stat().st_size))
    if sha256_file(out / "src_snapshot" / SOAP_NAME) != sha256_file(mod):
        print("  installed module does not match --corrected-module")
        ok = False
    if sha256_file(soap_src) == sha256_file(out / "src_snapshot" / SOAP_NAME):
        print("  REFUSED: the installed module is identical to the ORIGINAL one. "
              "This snapshot would compute UNCORRECTED features.")
        ok = False

    print("\nsource snapshot unchanged: %s sha256 %s"
          % (SOAP_NAME, sha256_file(soap_src)))

    if not ok:
        # Do not leave a half-built snapshot on disk.  A directory that exists
        # but did not verify is the one thing that could be picked up by a later
        # command and compute features with the wrong descriptor.
        shutil.rmtree(out, ignore_errors=True)
        print("RESULT: FAILED -- removed %s so it cannot be used by mistake" % out)
        return 1
    print("RESULT: snapshot ready")
    print("Pass this directory as --snapshot to prepare_expanded_moe_features.py:")
    print("  %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
