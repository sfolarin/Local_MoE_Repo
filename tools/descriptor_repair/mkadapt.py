#!/usr/bin/env python3
"""mkadapt.py -- build train/validation-only adapters for the three stages that
cannot be restricted from their command line.  Anchored replacements; every
anchor must match exactly once or the script aborts.  Originals are read only.

Why adapters and not filtered manifests: both trainers assert split SIZES in
main(), so deleting test rows makes the assertion fail before training starts.
The row filter and the assertion must change together.  Filtering happens inside
the manifest loop, before any shard is opened -- which matters for the router,
whose load_dataset() materialises StructureData.X in that loop.

  python3 mkadapt.py --repo $REPO --proj $PROJ --out $REPAIR/adapters
"""
import argparse, hashlib
from pathlib import Path

# (tag, description, anchor, replacement).  Anchors are the shortest text that
# is unique in the file, so surrounding whitespace cannot break the match.
PREP = [("B1", "historical B1 equality -> recorded difference",
 '''                if not np.allclose(x, old_x, atol=1e-5, rtol=1e-5):
                    raise ValueError("Live features disagree with existing B1 features; investigate descriptor provenance")
''',
 '''                # PILOT ADAPTER.  The original required equality with features
                # cached by the UNCORRECTED descriptor; a corrected descriptor
                # must disagree, so that requirement is inappropriate here and
                # becomes a recorded difference.  Retained: the shape and
                # finiteness checks, the [Si, O] species-order check, the
                # geometry hash written into the shard, and the common
                # trainer's own live-vs-cache agreement check.
                status["old_B1_comparison_policy"] = (
                    "recorded_not_enforced_corrected_descriptor_expected_to_differ")
                status["old_B1_allclose_1e-5"] = bool(
                    np.allclose(x, old_x, atol=1e-5, rtol=1e-5))
                print("B1 historical diff {} max_abs={:.6e} allclose={}".format(
                    row["record_id"], status["old_B1_max_abs_feature_difference"],
                    status["old_B1_allclose_1e-5"]), flush=True)
''')]

ROUTER = [("R1", "skip test rows before the shard is opened",
 '''        if split not in out:
            raise RuntimeError(f"Unexpected split {split!r}")
''',
 '''        if split not in out:
            raise RuntimeError(f"Unexpected split {split!r}")

        # PILOT ADAPTER: exclude test HERE; the loop body loads StructureData.X,
        # so skipping any later would still have read the file.
        if split == "test":
            continue
'''),
("R2", "split assertion now requires only that test be empty",
 '''    if (len(train_set), len(val_set), len(test_set)) != (112, 22, 22):
''',
 '''    # PILOT ADAPTER: the original expected (112, 22, 22), which does NOT match
    # the holdout-40 distillation dataset (369/38/105 in its manifest).  The
    # mismatch is reported, and the condition now holds iff test is empty.
    print("PILOT ADAPTER original expectation (112, 22, 22); actual "
          f"({len(train_set)}, {len(val_set)}, {len(test_set)})", flush=True)
    if (len(train_set), len(val_set), len(test_set)) != (len(train_set), len(val_set), 0):
'''),
("R3", "suppress test-prediction output",
 '''    pred_root = output_root / "test_predictions"
''',
 '''    # PILOT ADAPTER: writing test predictions is out of scope.  The function is
    # kept so any call site resolves, but it writes nothing and says so.
    print("PILOT ADAPTER: save_test_predictions suppressed", flush=True)
    return
    pred_root = output_root / "test_predictions"
''')]

C1B = [("X1", "skip test rows before the shard is opened",
 '''        if split not in records:
            raise RuntimeError(f"Unexpected split {split!r}")
''',
 '''        if split not in records:
            raise RuntimeError(f"Unexpected split {split!r}")

        # PILOT ADAPTER: exclude test HERE, before a record is built from it.
        if split == "test":
            continue
'''),
("X2", "expected split sizes 369/38/0 (split metadata itself unchanged)",
 '''    ) != (369, 38, 105):  # run-specific: expanded_holdout40_v1
''',
 '''    # PILOT ADAPTER: the stage manifest still carries all three partitions --
    # the split is unchanged -- but this run loads train and val only.
    ) != (369, 38, 0):  # PILOT: test excluded before shard access
'''),
("X3", "drop test from the evaluation loop",
 '''        ("test", test_records),
''',
 '''        # PILOT ADAPTER: the test partition is not evaluated.
''')]

TARGETS = [("prep_features_adapter.py", "repo",
            "scripts/prepare_expanded_moe_features.py", PREP),
           ("router_distill_adapter.py", "proj",
            "runs/train_local_torchsoap_gate_distillation.py", ROUTER),
           ("c1b_holdout40_adapter.py", "repo",
            "scripts/train_c1b_holdout40.py", C1B)]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--repo", required=True)
    a.add_argument("--proj", required=True)
    a.add_argument("--out", required=True)
    g = a.parse_args()
    roots = {"repo": Path(g.repo).resolve(), "proj": Path(g.proj).resolve()}
    out = Path(g.out).resolve(); out.mkdir(parents=True, exist_ok=True)

    print("=" * 92); print("TRAIN/VALIDATION-ONLY ADAPTERS"); print("=" * 92)
    print("mkadapt.py sha256 %s" % sha(__file__))
    print("originals read only; adapters written to %s\n" % out)

    for name, root, rel, patches in TARGETS:
        src = roots[root] / rel
        print(rel)
        if not src.is_file():
            raise SystemExit("FAIL: missing %s" % src)
        if out == src.parent:
            raise SystemExit("FAIL: --out is the source directory")
        print("  source sha256 %s" % sha(src))
        text = src.read_text(); ok = True
        for tag, desc, old, new in patches:
            n = text.count(old)
            print("  %-4s %-58s %d" % (tag, desc, n))
            if n != 1:
                print("       ANCHOR MATCHED %d TIMES, expected 1" % n)
                print("       begins: %r" % old.strip().splitlines()[0][:88])
                ok = False; continue
            text = text.replace(old, new, 1)
        if not ok:
            raise SystemExit("FAIL: %s -- the source differs from the excerpts these "
                             "adapters were written against. Nothing written." % rel)
        dst = out / name
        dst.write_text("# PILOT ADAPTER (train/validation only) -- mkadapt.py\n"
                       "# adapted from: %s\n# original sha256: %s\n%s"
                       % (src, sha(src), text))
        try:
            compile(dst.read_text(), str(dst), "exec"); syn = "parses cleanly"
        except SyntaxError as e:
            syn = "SYNTAX ERROR line %s: %s" % (e.lineno, e.msg)
        print("  adapter %-28s sha256 %s\n  %s\n" % (dst.name, sha(dst), syn))
        if syn != "parses cleanly":
            raise SystemExit("FAIL: adapter does not parse")
        if sha(dst) == sha(src):
            raise SystemExit("FAIL: adapter identical to original")

    print("originals unchanged:")
    for name, root, rel, _ in TARGETS:
        print("  %-58s %s" % (rel, sha(roots[root] / rel)[:14]))
    print("\nInvoke adapters through run_adapted.py so every shard opened is audited.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
