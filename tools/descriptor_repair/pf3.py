#!/usr/bin/env python3
"""pf3.py -- bind descriptor generation to the source the generator can actually
patch, and hard-block the items that must not be deferred. Read-only, no GPU."""
import hashlib, json, os, re
from pathlib import Path

P = Path(os.environ["PROJ"]).resolve()
Q = Path(os.environ.get("REPO", str(P))).resolve()
NL, ORIG = "30acc0838fb689", "186453838f89f7"

def sh(p):
    try:
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    except OSError:
        return None

# the anchors the generator needs; C2 is the discriminating one
PROBES = {
    "C3 norm(d + eps)":        r"torch\.linalg\.norm\(\s*d\s*\+\s*eps",
    "def _power_spectrum":     r"def\s+_power_spectrum",
    "def _real_sph_harm_basis": r"def\s+_real_sph_harm_basis",
    "def compute_per_center":  r"def\s+compute_per_center\s*\(",
    "def compute_per_center_dense": r"def\s+compute_per_center_dense",
    "C2 contraction inside _power_spectrum": None,   # computed below
}

cands = {
  "scripts_nl (NL)": P/"scripts_nl"/"torchsoap_autograd_module_periodic_regnorm_nl.py",
}
for s in sorted((P/"runs").glob("size_pilot_prep_*")):
    f = s/"src_snapshot"/"torchsoap_autograd_module_periodic_regnorm.py"
    if f.is_file():
        cands["snapshot %s" % s.name[-22:]] = f
f = P/"data"/"expanded_holdout40_v1_features"/"src_snapshot"/"torchsoap_autograd_module_periodic_regnorm.py"
if f.is_file():
    cands["features src_snapshot"] = f

print("=" * 88); print("DESCRIPTOR BINDING -- which source can the generator patch?"); print("=" * 88)
R = {"candidates": {}}
for lb, f in cands.items():
    t = f.read_text(errors="replace")
    h = sh(f)
    # does the power-spectrum function body contain the m-contraction?
    m = re.search(r"def\s+_power_spectrum\b.*?(?=\ndef\s|\nclass\s|\Z)", t, re.S)
    body = m.group(0) if m else ""
    c2 = bool(re.search(r"einsum|sum\s*\(\s*-1|\*\s*b\[|contract", body))
    got = {k: bool(re.search(v, t)) for k, v in PROBES.items() if v}
    got["C2 contraction inside _power_spectrum"] = c2
    ok = all(got.values())
    print("\n  %-34s %s" % (lb, h[:14] if h else "--"))
    print("    identity: %s" % ("PRODUCTION NL" if h and h.startswith(NL)
          else ("ORIGINAL pre-NL" if h and h.startswith(ORIG) else "unrecognised")))
    for k, v in got.items():
        print("      %-40s %s" % (k, "yes" if v else "NO"))
    print("    generator-compatible: %s" % ("YES" if ok else "no"))
    R["candidates"][lb] = dict(path=str(f), sha256=h, probes=got, compatible=ok)

comp = [k for k, v in R["candidates"].items() if v["compatible"]]
print("\n  compatible sources: %s" % (comp or "NONE"))
R["compatible"] = comp
B = []
if not comp:
    B.append("no candidate contains all generator anchors; the anchors must be "
             "adapted before any module is corrected")
print("\n" + "=" * 88); print("HARD BLOCKERS (must be empty before any GPU stage)"); print("=" * 88)
for name, path in (("test-exclusion adapters", Path("adapters")),
                   ("wired checkpoint assembly", Path("validate_pilot_checkpoints.py")),
                   ("B1 feature-check adapter", Path("adapters/prep_features_adapter.py"))):
    present = path.exists()
    if name == "wired checkpoint assembly" and present:
        present = "ASSEMBLY_WIRED = True" in Path(path).read_text(errors="replace")
    print("  %-30s %s" % (name, "present" if present else "MISSING -> hard blocker"))
    if not present:
        B.append("%s not present" % name)
R["hard_blockers"] = B
print("\n  hard blockers: %d" % len(B))
for b in B:
    print("    - %s" % b)
Path("pf3.json").write_text(json.dumps(R, indent=2, default=str))
print("\nwrote pf3.json\nRESULT: %s" % ("CLEAN" if not B else "BLOCKED (%d)" % len(B)))
raise SystemExit(1 if B else 0)
