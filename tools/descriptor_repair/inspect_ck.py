#!/usr/bin/env python3
"""inspect_ck.py -- report what is actually inside the three checkpoints and
which of the trainer's own functions to wire assembly to.  Read-only, no GPU."""
import hashlib, json, os, re, sys
from pathlib import Path
import torch

EREF = -9.668443844547909
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

rep = {}
for lb in ("common", "router", "residual"):
    p = os.environ.get(lb.upper())
    print("=" * 92); print("%s  %s" % (lb.upper(), p)); print("=" * 92)
    if not p or not Path(p).is_file():
        print("  MISSING"); rep[lb] = {"present": False, "path": p}; continue
    info = {"present": True, "path": p, "sha256": sha(p)}
    print("  sha256 %s" % info["sha256"])
    ck = torch.load(p, map_location="cpu", weights_only=False)
    if not isinstance(ck, dict):
        print("  not a dict: %s" % type(ck).__name__); rep[lb] = info; continue
    info["top_level_keys"] = sorted(ck)
    print("  top-level keys (%d): %s" % (len(ck), sorted(ck)))
    for k in sorted(ck):
        v = ck[k]
        if isinstance(v, dict) and v and all(hasattr(x, "shape") for x in list(v.values())[:3]):
            sh = {n: list(t.shape) for n, t in v.items()}
            print("  %s: state dict, %d tensors" % (k, len(sh)))
            for n, s in list(sh.items())[:14]:
                print("      %-46s %s" % (n, s))
            if len(sh) > 14: print("      ... %d more" % (len(sh) - 14))
            info.setdefault("state_dicts", {})[k] = sh
        elif isinstance(v, (int, float, str, bool)) or v is None:
            print("  %-34s %r" % (k, v)); info.setdefault("scalars", {})[k] = v
        elif hasattr(v, "shape"):
            print("  %-34s tensor %s" % (k, list(v.shape)))
            info.setdefault("tensors", {})[k] = list(v.shape)
        elif isinstance(v, (list, tuple)):
            print("  %-34s %s len=%d" % (k, type(v).__name__, len(v)))
    if "e_ref_eV_per_atom" in ck:
        e = float(ck["e_ref_eV_per_atom"]); ok = abs(e - EREF) < 1e-9
        print("  e_ref = %.15g  %s" % (e, "(baseline constant)" if ok else "<-- NOT baseline"))
        info["e_ref_ok"] = ok; info["e_ref"] = e
    rep[lb] = info
    print()

t = os.environ.get("TRAINER")
print("=" * 92); print("TRAINER  %s" % t); print("=" * 92)
if t and Path(t).is_file():
    txt = Path(t).read_text(errors="replace")
    print("  sha256 %s" % sha(t))
    defs = re.findall(r'^def\s+(\w+)\s*\(([^)]*)\)', txt, re.M)
    want = [d for d in defs if re.search(
        r'load|build|assemble|model|energy|force|checkpoint|predict|evaluate|descriptor',
        d[0], re.I)]
    print("  module-level callables relevant to loading / assembly / energy:")
    for n, s in want:
        print("      %-40s (%s)" % (n, " ".join(s.split())[:84]))
    rep["trainer"] = {"path": t, "sha256": sha(t), "candidates": [n for n, _ in want]}
    for key in ("def load_common_checkpoint", "def load_routing_checkpoint",
                "def evaluate_split", "def total_energy", "def energy_and_forces",
                "def build_descriptor", "def import_torchsoap"):
        m = re.search(re.escape(key) + r".*?(?=\n(?:def |class )|\Z)", txt, re.S)
        if m:
            body = m.group(0).splitlines()
            print("\n  --- %s  (%d lines) ---" % (key[4:], len(body)))
            for ln in body[:26]: print("    %s" % ln[:104])
            if len(body) > 26: print("    ... %d more lines" % (len(body) - 26))
else:
    print("  MISSING")
Path("P3_inspect.json").write_text(json.dumps(rep, indent=2, default=str))
print("\nwrote P3_inspect.json")
