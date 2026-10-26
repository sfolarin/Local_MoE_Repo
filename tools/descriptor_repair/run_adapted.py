#!/usr/bin/env python3
"""
run_adapted.py -- invoke an adapted stage with the executed path instrumented.

A standalone split-guard test proves the guard works in isolation; it does not
prove the stage that actually ran never touched a test shard.  This wrapper
instruments the real execution: it wraps numpy.load and builtins.open, records
every data file the stage opens, and aborts immediately if one of them belongs
to the test partition.

The allow/deny decision is made from the stage manifest, which is read once and
is the same metadata the pilot preserves untouched.  Record ids on the test side
are matched by file stem, so a shard is refused whatever directory it sits in --
filename-based guessing protects nothing on a flat stage.

    python3 run_adapted.py --adapter DIR/c1b_holdout40_adapter.py \
        --manifest STAGE/manifest.csv --audit LOG.jsonl -- <the stage's own args>

Exit status is the adapted stage's own, or 3 if a test file was opened.
"""
import argparse
import builtins
import importlib.util
import json
import sys
import time
from pathlib import Path

DATA_SUFFIX = {".npz", ".npy", ".pt", ".pth"}


class TestPartitionAccess(RuntimeError):
    pass


def read_manifest(path):
    import csv
    with open(path, newline="", errors="replace") as f:
        rdr = csv.DictReader(f)
        rows = [dict(r) for r in rdr]
        cols = list(rdr.fieldnames or [])
    sc = next((c for c in ("split", "split_group") if c in cols), None)
    if sc is None:
        raise SystemExit("FAIL: no split column in %s" % path)
    ids = {"train": set(), "val": set(), "test": set()}
    for r in rows:
        sp = (r[sc] or "").strip()
        if sp not in ids:
            continue
        for c in ("record_id", "source_file", "output_file", "shard_file", "teacher_file"):
            if r.get(c):
                ids[sp].add(Path(r[c]).stem)
    return ids


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--adapter", required=True)
    a.add_argument("--manifest", required=True)
    a.add_argument("--audit", default="file_access_audit.jsonl")
    a.add_argument("--allow-test", action="store_true",
                   help="never use in this pilot; present so the refusal is explicit")
    a.add_argument("rest", nargs=argparse.REMAINDER)
    args = a.parse_args()

    ids = read_manifest(args.manifest)
    print("=" * 94)
    print("INSTRUMENTED STAGE RUN -- file access audited")
    print("=" * 94)
    print("adapter  : %s" % args.adapter)
    print("manifest : %s" % args.manifest)
    print("allow-list: train %d, val %d  |  deny-list: test %d"
          % (len(ids["train"]), len(ids["val"]), len(ids["test"])))
    print("audit    : %s" % args.audit)

    audit = open(args.audit, "a", buffering=1)
    # numpy.load calls open() internally, so raw open counts double-count every
    # shard.  Distinct files per split is the figure that means something.
    counts = {"train": 0, "val": 0, "test": 0, "other": 0}
    distinct = {"train": set(), "val": set(), "test": set(), "other": set()}
    seen = set()

    def classify(p):
        st = Path(p).stem
        for k in ("test", "train", "val"):
            if st in ids[k]:
                return k
        return "other"

    def record(p, how):
        p = str(p)
        if Path(p).suffix not in DATA_SUFFIX:
            return
        k = classify(p)
        counts[k] += 1
        distinct[k].add(p)
        if p not in seen:
            seen.add(p)
            audit.write(json.dumps(dict(t=time.time(), how=how, split=k, path=p)) + "\n")
        if k == "test" and not args.allow_test:
            audit.write(json.dumps(dict(t=time.time(), how=how, split="test",
                                        path=p, fatal=True)) + "\n")
            audit.flush()
            raise TestPartitionAccess(
                "TEST PARTITION ACCESS: %s was opened by the executed path. "
                "This is outside the pilot's scope; aborting rather than "
                "continuing." % p)

    import numpy as np
    real_load, real_open = np.load, builtins.open

    def guarded_load(file, *aa, **kk):
        try:
            record(file, "numpy.load")
        except TypeError:
            pass
        return real_load(file, *aa, **kk)

    def guarded_open(file, *aa, **kk):
        try:
            record(file, "open")
        except TypeError:
            pass
        return real_open(file, *aa, **kk)

    np.load = guarded_load
    builtins.open = guarded_open

    spec = importlib.util.spec_from_file_location("_adapted_stage", args.adapter)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_adapted_stage"] = mod

    rest = args.rest[1:] if args.rest and args.rest[0] == "--" else args.rest
    sys.argv = [args.adapter] + rest
    print("argv     : %s\n" % " ".join(sys.argv))

    rc = 0
    try:
        spec.loader.exec_module(mod)
        if hasattr(mod, "main"):
            r = mod.main()
            rc = int(r) if isinstance(r, int) else 0
    except TestPartitionAccess as e:
        print("\n%s" % e, file=sys.stderr)
        rc = 3
    except SystemExit as e:
        rc = int(e.code) if isinstance(e.code, int) else (0 if e.code is None else 1)
    finally:
        np.load, builtins.open = real_load, real_open
        print("\n" + "=" * 94)
        print("FILE ACCESS AUDIT")
        print("=" * 94)
        for k in ("train", "val", "test", "other"):
            print("  %-6s distinct files %6d   (raw opens %d)"
                  % (k, len(distinct[k]), counts[k]))
        print("  distinct data files opened: %d" % len(seen))
        print("  test-partition files: %d  -> %s"
              % (len(distinct["test"]),
                 "CLEAN" if not distinct["test"] else "SCOPE VIOLATION"))
        print("  audit log: %s" % Path(args.audit).resolve())
        audit.close()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
