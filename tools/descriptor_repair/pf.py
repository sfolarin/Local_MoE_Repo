#!/usr/bin/env python3
"""pf.py -- compact pilot preflight. Read-only, no GPU. Resolves the six unknowns
that block the pilot, and writes pf.json.   python3 pf.py        (uses $PROJ/$REPO)"""
import csv, hashlib, json, os, re, sys
from pathlib import Path

NL = "30acc0838fb689cb7274e5261baf265d26b1dc2ec1649d0341e8dbb09a114c86"
ORIG = "186453838f89f7e85a4f40d182359c4bef19b5b2516a116d62dd02cc1a79624c"
REFCK = "a03a4f2b488ccfd92cbc1bb7282c575fcac3f5ae8f74907f65663fd3db636875"
EREF = -9.668443844547909
WANT = {"train": 369, "val": 38, "test": 105}
R, B = {}, []


def sh(p):
    try:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for c in iter(lambda: f.read(1 << 20), b""):
                h.update(c)
        return h.hexdigest()
    except OSError:
        return None


def bl(t, m):
    B.append({"tag": t, "detail": m}); print("  BLOCKER [%s] %s" % (t, m))


def rows(p):
    with open(p, newline="", errors="replace") as f:
        r = csv.DictReader(f); return [dict(x) for x in r], list(r.fieldnames or [])


P = Path(os.environ.get("PROJ", "")).resolve()
Q = Path(os.environ.get("REPO", str(P))).resolve()
print("=" * 86); print("PILOT PREFLIGHT (compact) -- read-only, no GPU"); print("=" * 86)
print("pf.py sha256 %s" % sh(__file__)); print("PROJ %s" % P); print("REPO %s" % Q)
R["proj"], R["repo"] = str(P), str(Q)
if not P.is_dir():
    bl("proj", "PROJ does not exist"); print(json.dumps(R)); raise SystemExit(1)

# ---- 1. snapshot + e_ref ----------------------------------------------------
print("\n--- 1. snapshot and e_ref ---")
snaps = sorted(x for x in (P / "runs").glob("size_pilot_prep_*") if x.is_dir())
for s in snaps:
    print("  snapshot: %s" % s)
S = snaps[-1] if snaps else None
if S is None:
    bl("snapshot", "no size_pilot_prep_* under %s/runs" % P)
R["snapshot_root"] = str(S) if S else None
if S:
    bd = S / "baseline"
    found = None
    if bd.is_dir():
        for f in sorted(bd.iterdir()):
            if f.is_file():
                s2 = sh(f)
                print("  baseline/%-28s %s%s" % (f.name, s2[:14],
                      "  <-- recorded reference_checkpoint" if s2 == REFCK else ""))
                if s2 == REFCK or f.name == "common.pt":
                    found = f
    else:
        print("  no baseline/ dir")
    R["eref_checkpoint"] = str(found) if found else None
    if not found:
        bl("eref", "no baseline checkpoint under %s/baseline" % S)
    else:
        try:
            import torch
            ck = torch.load(found, map_location="cpu", weights_only=False)
            e = ck.get("e_ref_eV_per_atom") if isinstance(ck, dict) else None
            print("  e_ref in checkpoint = %r" % e)
            R["eref_value"] = None if e is None else float(e)
            if e is not None and abs(float(e) - EREF) > 1e-9:
                bl("eref-value", "e_ref %r is not the baseline constant %r" % (e, EREF))
            if isinstance(ck, dict):
                print("  checkpoint keys: %s" % sorted(ck)[:14])
        except Exception as ex:
            print("  could not read checkpoint: %s: %s" % (type(ex).__name__, ex))

# ---- 2. dataset root --------------------------------------------------------
print("\n--- 2. feature-prep dataset root (--dataset) ---")
prep = Q / "scripts" / "prepare_expanded_moe_features.py"
if prep.is_file():
    txt = prep.read_text(errors="replace")
    need = sorted(set(re.findall(r'["\']([A-Za-z0-9_.\-]+\.(?:json|csv))["\']', txt)))
    print("  prep script sha256 %s" % sh(prep)); print("  metadata it names: %s" % need)
    R["prepare_features_sha256"] = sh(prep); R["prep_metadata_files"] = need
else:
    bl("prep", "missing %s" % prep); txt = ""
M = ["split_summary.json", "all_source_records.csv"]
cands = []
for d in sorted(x for x in (P / "data").iterdir() if x.is_dir()) if (P / "data").is_dir() else []:
    have = [m for m in M if (d / m).is_file()]
    if not have:
        continue
    n, spl = None, None
    a = d / "all_source_records.csv"
    if a.is_file():
        try:
            rr, cc = rows(a); n = len(rr)
            k = next((x for x in ("split", "split_group") if x in cc), None)
            if k:
                spl = {}
                for r0 in rr:
                    spl[r0[k]] = spl.get(r0[k], 0) + 1
        except OSError:
            pass
    cands.append(dict(path=str(d), markers=have, n=n, splits=spl))
    print("  %-50s %d/%d  n=%s  %s" % (d.name[:50], len(have), len(M), n, spl))
R["dataset_candidates"] = cands
full = [c for c in cands if len(c["markers"]) == len(M)]
hit = [c for c in full if c["splits"] == WANT]
if hit:
    R["dataset_root"] = hit[0]["path"]; print("  RESOLVED --dataset = %s" % hit[0]["path"])
else:
    bl("dataset-root", "no root under %s/data has both %s with splits %s (the C1b stage "
       "is NOT a dataset root)" % (P, M, WANT))

# ---- 3. which descriptor module to correct ----------------------------------
print("\n--- 3. descriptor module to correct ---")
cd = {"scripts_nl": P / "scripts_nl" / "torchsoap_autograd_module_periodic_regnorm_nl.py",
      "scripts": P / "scripts" / "torchsoap_autograd_module_periodic_regnorm.py"}
if S:
    cd["snapshot"] = S / "src_snapshot" / "torchsoap_autograd_module_periodic_regnorm.py"
seen = {}
for k, v in cd.items():
    s2 = sh(v)
    idn = "MISSING" if s2 is None else ("= known PRODUCTION NL" if s2 == NL else
          ("= known ORIGINAL pre-NL" if s2 == ORIG else "unrecognised"))
    print("  %-11s %-14s %s" % (k, (s2[:14] if s2 else "--"), idn))
    seen[k] = dict(path=str(v), sha256=s2, id=idn)
R["descriptor_candidates"] = seen
fc = P / "data" / "expanded_holdout40_v1_features" / "feature_contract.json"
csha = None
if fc.is_file():
    try:
        c = json.load(open(fc))
        csha = c.get("source_sha256") or c.get("soap_sha256")
        print("  feature_contract source_sha256 = %s" % csha)
        R["contract_source_sha256"] = csha
        R["contract_baseline_keys"] = {k: v for k, v in c.items() if "baseline" in k.lower()}
        if R["contract_baseline_keys"]:
            print("  contract baseline keys: %s" % R["contract_baseline_keys"])
    except (OSError, ValueError) as ex:
        print("  cannot parse contract: %s" % ex)
else:
    print("  no feature_contract.json at %s" % fc)
if csha:
    m = [k for k, v in seen.items() if v["sha256"] == csha]
    if m:
        R["descriptor_to_correct"] = seen[m[0]]["path"]
        print("  RESOLVED correct this file -> %s" % seen[m[0]]["path"])
        if S and seen.get("snapshot", {}).get("sha256") not in (csha, None):
            bl("snapshot-mismatch", "snapshot module %s != contract %s; the feature "
               "snapshot must be built from the contract module"
               % (seen["snapshot"]["sha256"][:14], csha[:14]))
    else:
        bl("descriptor-identity", "contract records %s but no candidate has that hash; "
           "do not correct any module until reconciled" % csha)
else:
    bl("descriptor-identity", "could not read source_sha256 from the feature contract")

# ---- 4. checkpoint basenames + device args ----------------------------------
print("\n--- 4. checkpoint basenames and --device forms ---")
base, shown = {}, set()
for kind, pat in (("common", "common_holdout40*"), ("router", "*routing*"),
                  ("router", "*distill*"), ("c1b", "*c1b*holdout40*")):
    for d in sorted((P / "runs").glob(pat)) if (P / "runs").is_dir() else []:
        if not d.is_dir() or (kind, d) in shown:
            continue
        shown.add((kind, d))
        pts = sorted(x.name for x in d.iterdir() if x.suffix == ".pt")
        if pts:
            print("  %-7s %-42s %s" % (kind, d.name[:42], pts))
            base.setdefault(kind, set()).update(pts)
R["checkpoint_basenames"] = {k: sorted(v) for k, v in base.items()}
if not base:
    bl("basenames", "no .pt files found in any production run root")
for lb, p2 in (("router", P / "runs" / "train_local_torchsoap_gate_distillation.py"),
               ("c1b", Q / "scripts" / "train_c1b_holdout40.py"),
               ("common", P / "scripts" / "train_expanded_moe_common.py")):
    if not p2.is_file():
        print("  %-7s trainer MISSING %s" % (lb, p2)); continue
    t2 = p2.read_text(errors="replace")
    ch, df = [], []
    for call in re.findall(r'add_argument\(\s*["\']--device["\'](.*?)\)\s*$', t2, re.M | re.S):
        g = re.search(r'choices\s*=\s*[\(\[]([^\)\]]*)[\)\]]', call)
        if g:
            ch = re.findall(r'["\']([^"\']+)["\']', g.group(1))
        d2 = re.search(r'default\s*=\s*["\']([^"\']+)["\']', call)
        if d2:
            df = [d2.group(1)]
        break
    print("  %-7s --device choices=%s default=%s  sha256 %s"
          % (lb, ch or "unrestricted", df or "none", sh(p2)[:14]))
    R.setdefault("device_args", {})[lb] = dict(choices=ch, default=df, sha256=sh(p2))

# ---- 5. test access + baseline check ----------------------------------------
print("\n--- 5. test-partition access and B1 baseline check ---")
for lb, p2 in (("prep-feat", prep),
               ("prep-dist", Q / "scripts" / "prepare_expanded_torchsoap_distillation_dataset.py"),
               ("common", P / "scripts" / "train_expanded_moe_common.py"),
               ("router", P / "runs" / "train_local_torchsoap_gate_distillation.py"),
               ("c1b", Q / "scripts" / "train_c1b_holdout40.py")):
    if not p2 or not p2.is_file():
        continue
    ls = p2.read_text(errors="replace").splitlines()
    h = [(i + 1, l.strip()[:96]) for i, l in enumerate(ls)
         if re.search(r'["\']test["\']|\btest_|_test\b|SPLITS', l)]
    print("  %s (%d hits)" % (lb, len(h)))
    for n, t in h[:8]:
        print("      %5d %s" % (n, t))
    R.setdefault("test_access", {})[lb] = [{"line": n, "text": t} for n, t in h]
if txt:
    bh = [(i + 1, l.strip()[:96]) for i, l in enumerate(txt.splitlines())
          if re.search(r'baseline|b1_|_b1|historical|previous_features', l, re.I)]
    fl = re.findall(r'add_argument\(\s*["\'](--[a-z0-9\-]*baseline[a-z0-9\-]*)["\']', txt, re.I)
    print("  B1 baseline-check references in prep script: %d; flags: %s" % (len(bh), fl or "NONE"))
    for n, t in bh[:10]:
        print("      %5d %s" % (n, t))
    R["baseline_check_hits"] = [{"line": n, "text": t} for n, t in bh]
    R["baseline_check_flags"] = fl
    if bh and not fl:
        bl("baseline-check", "prep script compares against baseline B1 features but has no "
           "flag to adapt it; an adapter must keep geometry/membership/species/hash and "
           "live-vs-cache checks and drop only equality-with-uncorrected")

# ---- summary ----------------------------------------------------------------
print("\n" + "=" * 86)
for k in ("dataset_root", "descriptor_to_correct", "snapshot_root", "eref_checkpoint",
          "eref_value"):
    print("  %-22s %s" % (k, R.get(k, "UNRESOLVED")))
print("  %-22s %s" % ("checkpoint_basenames", R.get("checkpoint_basenames")))
print("\n  BLOCKERS: %d" % len(B))
for b in B:
    print("    [%s] %s" % (b["tag"], b["detail"]))
R["blockers"] = B
R["pf_sha256"] = sh(__file__)
Path("pf.json").write_text(json.dumps(R, indent=2, default=str))
print("\nwrote %s/pf.json" % Path.cwd())
print("RESULT: %s" % ("CLEAN" if not B else "BLOCKED (%d)" % len(B)))
raise SystemExit(1 if B else 0)
