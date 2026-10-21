#!/usr/bin/env python3
"""Read-only: do the scripts in the repository match the scripts that produced the paper runs?

Every training / data run folder keeps a `src_snapshot/` with the exact scripts it
executed.  The repository's MANIFEST.tsv records the sha256 of the original file each
repository copy came from (`source_sha256`).  This compares the two:

  SAME          snapshot file == the original the repository copy was made from
  DIFFERENT     the repository was built from another version of this file
  NOT IN REPO   a script the run used that the repository does not contain
  (and, at the end) repository scripts that no snapshot vouches for

Nothing is written.  Usage:
  python3 verify_provenance.py --manifest <repository>/MANIFEST.tsv  RUN_DIR [RUN_DIR ...]
RUN_DIR may be a run folder (it looks for RUN_DIR/src_snapshot) or the snapshot folder itself.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from collections import defaultdict
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(path: Path):
    by_name = defaultdict(list)          # basename -> [(repo_path, source_sha, repo_sha)]
    with path.open() as f:
        header = f.readline().rstrip("\n").split("\t")
        ix = {k: i for i, k in enumerate(header)}
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 3:
                continue
            rel = p[ix["file"]]
            by_name[Path(rel).name].append((rel, p[ix["source_sha256"]], p[ix["repo_sha256"]]))
    return by_name


def same(a: str, b: str) -> bool:
    n = min(len(a), len(b))
    return n >= 8 and a[:n] == b[:n]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("runs", nargs="+", type=Path)
    a = ap.parse_args()
    if not a.manifest.is_file():
        print("no such manifest: %s" % a.manifest, file=sys.stderr)
        return 2
    manifest = load_manifest(a.manifest)
    vouched = set()
    totals = defaultdict(int)

    for run in a.runs:
        snap = run / "src_snapshot" if (run / "src_snapshot").is_dir() else run
        label = run.name or str(run)
        if not snap.is_dir():
            print("\n=== %s\n  (no src_snapshot folder)" % label)
            totals["runs without snapshot"] += 1
            continue
        files = sorted(p for p in snap.rglob("*") if p.is_file() and p.suffix in (".py", ".sh", ".slurm"))
        print("\n=== %s  (%d script file(s) in the snapshot)" % (label, len(files)))
        for p in files:
            h = sha256(p)
            ent = manifest.get(p.name)
            if not ent:
                status, where = "NOT IN REPO", ""
            else:
                ok = [e for e in ent if same(h, e[1])]
                if ok:
                    status, where = "SAME", ok[0][0]
                    vouched.add(ok[0][0])
                else:
                    status, where = "DIFFERENT", ent[0][0]
            totals[status] += 1
            print("  %-12s %-62s %s" % (status, p.relative_to(snap), where))

    print("\nsummary: " + ", ".join("%s %d" % (k, v) for k, v in sorted(totals.items())))
    repo_files = sorted({e[0] for v in manifest.values() for e in v if e[0].endswith((".py", ".sh", ".slurm"))})
    unvouched = [f for f in repo_files if f not in vouched]
    print("\nrepository scripts vouched for by a snapshot: %d of %d" % (len(vouched), len(repo_files)))
    print("not covered by any snapshot you gave (not necessarily a problem):")
    for f in unvouched:
        print("  " + f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
