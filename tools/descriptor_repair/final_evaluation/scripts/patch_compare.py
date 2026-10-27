#!/usr/bin/env python3
"""
patch_compare.py -- build a patched copy of eval_test_regions_common_vs_full.py whose
"reproduction of existing numbers" section tolerates a C1b run that has no recorded
TEST numbers.

WHY THIS IS A DEFECT IN THE ORIGINAL, NOT A WORKAROUND

The original guards its two reproduction inputs with Path.is_file():

    rf = rd / "test_region_force_metrics.csv"
    ff = rd / "final_split_metrics.csv"
    if rf.is_file():
        for row in pd.read_csv(rf).itertuples():
        ...
    if ff.is_file():
        fs = pd.read_csv(ff)
        fs = fs[fs.split == "test"].iloc[0]

is_file() is True for a ZERO-BYTE file, and pandas raises EmptyDataError on one.  It
is likewise True for a file that exists but contains no `test` row, where .iloc[0] on
the empty selection raises IndexError.  The pilot C1b run produced both states on
purpose: the trainer's write_csv() writes an empty file when given no rows, and the
pilot excluded the test partition, so test_region_force_metrics.csv is 0 bytes and
final_split_metrics.csv holds train and val rows only.

The original already encodes the correct behaviour for missing reproduction data --
it prints "(no ...; its reproduction is skipped)" and carries on.  These patches
extend that same behaviour to the empty and test-less cases.

WHAT IS AND IS NOT AFFECTED

Not affected: the comparison itself.  _per_structure.csv and _regions.csv are built
from the fresh evaluation that already ran to 105/105 before the crash.  Also not
affected: the COMMON side's reproduction against metrics.csv, which stays fully
active because the frozen evaluation wrote a test/ALL/ALL row.

Affected: only the cross-check of the C1b run's test numbers against values recorded
at training time.  For the pilot those values do not exist, and their absence is
itself the evidence that the test partition was never touched during fitting.

The original file is READ ONLY.  Every patch is anchored on exact text and must match
exactly once, or this script refuses to write anything.

    python3 patch_compare.py --original <path> --out <path>
"""
import argparse
import hashlib
import sys
from pathlib import Path

# (label, old, new) -- each `old` must occur EXACTLY once in the original.
PATCHES = [
    (
        "P1 region file: treat a zero-byte file as absent",
        "        if rf.is_file():\n"
        "            for row in pd.read_csv(rf).itertuples():\n",
        "        if rf.is_file() and rf.stat().st_size > 0:\n"
        "            for row in pd.read_csv(rf).itertuples():\n",
    ),
    (
        "P2 region file: make the skip message accurate",
        '            print(f"(no {rf}; its reproduction is skipped)")\n',
        '            print(f"(no usable {rf}: absent or empty -- '
        'its reproduction is skipped)")\n',
    ),
    (
        "P3 split metrics: require an actual test row before selecting it",
        "        if ff.is_file():\n"
        "            fs = pd.read_csv(ff)\n"
        '            fs = fs[fs.split == "test"].iloc[0]\n',
        "        _ff_has_test = False\n"
        "        if ff.is_file() and ff.stat().st_size > 0:\n"
        '            _ff_has_test = "test" in set(pd.read_csv(ff)["split"].astype(str))\n'
        "        if _ff_has_test:\n"
        "            fs = pd.read_csv(ff)\n"
        '            fs = fs[fs.split == "test"].iloc[0]\n',
    ),
    (
        "P4 split metrics: make the skip message accurate",
        '            print(f"(no {ff}; its reproduction is skipped)")\n',
        '            print(f"(no usable {ff}: absent, empty, or no test row -- '
        'its reproduction is skipped)")\n',
    ),
]


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    orig = args.original.expanduser().resolve()
    out = args.out.expanduser().resolve()
    if orig == out:
        raise SystemExit("FAIL: --out is the original; refusing to modify it in place")
    if out.exists() and not args.force:
        raise SystemExit("FAIL: %s exists; pass --force to replace it" % out)

    src = orig.read_text()
    print("original : %s" % orig)
    print("  sha256 : %s" % sha(src.encode()))
    print("  lines  : %d" % len(src.splitlines()))
    print()

    text = src
    ok = True
    for label, old, new in PATCHES:
        n = text.count(old)
        status = "matched once" if n == 1 else "MATCHED %d TIMES" % n
        print("  %-58s %s" % (label, status))
        if n != 1:
            ok = False
            continue
        text = text.replace(old, new, 1)

    if not ok:
        print("\nRESULT: FAILED -- an anchor did not match exactly once.")
        print("The original has changed since these patches were written.  Nothing")
        print("was written: a partially patched evaluator would produce numbers that")
        print("look right.  Re-derive the anchors against the current original.")
        return 1

    # Confirm the patched file parses before it is allowed to exist on disk.
    try:
        compile(text, str(out), "exec")
    except SyntaxError as exc:
        print("\nRESULT: FAILED -- patched source does not compile: %s" % exc)
        return 1

    header = (
        "# adapted from: %s\n"
        "# adapter: patch_compare.py -- tolerate a C1b run with no recorded test\n"
        "#   numbers (0-byte test_region_force_metrics.csv, test-less\n"
        "#   final_split_metrics.csv).  Reproduction of those values is skipped;\n"
        "#   the comparison itself and the common-side reproduction are unchanged.\n"
        "# original sha256: %s\n"
    ) % (orig, sha(src.encode()))

    lines = text.split("\n")
    if lines and lines[0].startswith("#!"):
        text = lines[0] + "\n" + header + "\n".join(lines[1:])
    else:
        text = header + text

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print("\npatched  : %s" % out)
    print("  sha256 : %s" % sha(text.encode()))
    print("  lines  : %d" % len(text.splitlines()))
    print("\noriginal unchanged: sha256 %s" % sha(orig.read_bytes()))
    print("RESULT: patched evaluator ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
