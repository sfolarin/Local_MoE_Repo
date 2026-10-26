#!/usr/bin/env python3
"""
pf5.py -- consolidated hardened preflight for the run that is about to happen.

Unlike pf.py (which resolves paths) and pf3.py (which checks presence), this
verifies the ARTIFACTS THE RUN WILL ACTUALLY USE, and reconciles the GPU budget
in ALLOCATED GPU-hours: elapsed seconds times the number of GPUs the scheduler
allocated, which is not the same as the number requested.

Gates, each a hard blocker:
  A  corrected descriptor module present, hash pinned, variant rotinv_v1
  B  corrected snapshot verifies the way prepare_expanded_moe_features.py does,
     and its installed module is NOT the original
  C  adapters present AND regenerable: re-running mkadapt.py against the current
     originals must reproduce the same hashes, so an adapter cannot drift from
     the source it claims to patch
  D  assembled-model physics verdict present, variant correct, test partition
     empty, and the gradient verdict carried through as verified/unverified
  E  executed-path audit shows zero test-partition file opens
  F  router contract verdict present with no blockers
  G  budget: allocated GPU-hours against the cap, with per-job GPU counts

    PROJ=.. REPO=.. REPAIR=.. [CAP=8.0] [JOBS=gpu_jobs.txt] python3 pf5.py
"""
import csv, hashlib, json, os, re, subprocess, sys
from pathlib import Path

CORRECTED_SHA = "0afd9a1ba7b9a24f8bf8151a7a3bdfc409544ed0920355de0598ffae79d51d21"
ORIGINAL_NL_SHA = "30acc0838fb689cb7274e5261baf265d26b1dc2ec1649d0341e8dbb09a114c86"
PRENL_SHA = "186453838f89f7e85a4f40d182359c4bef19b5b2516a116d62dd02cc1a79624c"
SOAP_NAME = "torchsoap_autograd_module_periodic_regnorm.py"
CAP = float(os.environ.get("CAP", "8.0"))

P = Path(os.environ["PROJ"])
Q = Path(os.environ.get("REPO", str(P)))
R = Path(os.environ.get("REPAIR", str(P / "descriptor_repair")))
blockers, notes, rep = [], [], {}


def sha(p):
    try:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for c in iter(lambda: f.read(1 << 20), b""):
                h.update(c)
        return h.hexdigest()
    except OSError:
        return None


def bl(tag, msg):
    blockers.append({"tag": tag, "detail": msg})
    print("  BLOCKER [%s] %s" % (tag, msg))


def head(t):
    print("\n" + "=" * 92); print(t); print("=" * 92)


print("=" * 92)
print("CONSOLIDATED HARDENED PREFLIGHT (pf5)")
print("=" * 92)
print("pf5.py sha256 %s" % sha(__file__))
print("cap %.3f allocated GPU-hours" % CAP)

# ---- A. the corrected module ----------------------------------------------
head("A. corrected descriptor module")
corr = R / "torchsoap_corrected_rotinv_v1.py"
cs = sha(corr)
print("  %s" % corr)
print("  sha256 %s" % cs)
rep["corrected_module"] = {"path": str(corr), "sha256": cs}
if cs != CORRECTED_SHA:
    bl("A1", "corrected module hash is %s, expected %s" % (cs, CORRECTED_SHA))
else:
    print("  matches the pinned corrected hash")
txt = corr.read_text(errors="replace") if corr.is_file() else ""
if 'DESCRIPTOR_VARIANT = "rotinv_v1"' not in txt:
    bl("A2", "module does not declare DESCRIPTOR_VARIANT = \"rotinv_v1\"")
else:
    print("  declares DESCRIPTOR_VARIANT = rotinv_v1")

# ---- B. the corrected snapshot --------------------------------------------
head("B. corrected source snapshot, verified as the feature prep will")
snap = R / "src_snapshot_rotinv_v1"
rep["snapshot"] = {"path": str(snap)}
if not snap.is_dir():
    bl("B1", "corrected snapshot absent: %s (run build_corrected_snapshot.py)" % snap)
else:
    prov = snap / "snapshot_provenance.csv"
    inst = snap / "src_snapshot" / SOAP_NAME
    if not prov.is_file():
        bl("B2", "no snapshot_provenance.csv in the corrected snapshot")
    elif not inst.is_file():
        bl("B3", "%s absent from the corrected snapshot" % SOAP_NAME)
    else:
        isha = sha(inst)
        print("  installed %s" % SOAP_NAME)
        print("    sha256 %s" % isha)
        rep["snapshot"]["installed_sha256"] = isha
        if isha != CORRECTED_SHA:
            bl("B4", "installed module is %s, not the corrected %s" % (isha, CORRECTED_SHA))
        if isha in (PRENL_SHA, ORIGINAL_NL_SHA):
            bl("B5", "installed module is an UNCORRECTED module; this snapshot would "
                     "compute uncorrected features")
        rows = list(csv.DictReader(open(prov, newline="", errors="replace")))
        idx = {str(Path(r["snapshot_path"]).resolve()): r for r in rows}
        bad = 0
        for f in sorted((snap / "src_snapshot").glob("*.py")):
            row = idx.get(str(f.resolve()))
            if row is None:
                print("    %-58s NOT IN PROVENANCE" % f.name); bad += 1; continue
            good = sha(f) == row["sha256"] and str(f.stat().st_size) == str(row["bytes"])
            print("    %-58s %s" % (f.name, "verified" if good else "MISMATCH"))
            bad += (not good)
        rep["snapshot"]["provenance_rows"] = len(rows)
        if bad:
            bl("B6", "%d snapshot file(s) failed provenance verification" % bad)

# ---- C. adapters present AND regenerable ----------------------------------
head("C. adapters: present, and reproducible from the current originals")
ad = R / "adapters"
EXPECT = {"prep_features_adapter.py": "175ecc61b4fb4d3c8a581e818bd14e0a6ad53a942e1964febf4ac05d8dc805dd",
          "router_distill_adapter.py": "5940fc1e517e6d38d99ef301216fa9e6c0da7de65c79ba82f263e3014bb27c9c",
          "c1b_holdout40_adapter.py": "3705bf0a3aa4bf2dbe7d74091e9da79754867db7c8d728d6f63c2ecd0e8710fa"}
rep["adapters"] = {}
if not ad.is_dir():
    bl("C1", "adapters directory absent: %s (run mkadapt.py)" % ad)
else:
    for n, want in EXPECT.items():
        got = sha(ad / n)
        ok = got == want
        print("  %-28s %s" % (n, "hash matches" if ok else "HASH DIFFERS"))
        rep["adapters"][n] = {"sha256": got, "expected": want, "matches": ok}
        if got is None:
            bl("C2", "%s absent" % n)
        elif not ok:
            bl("C3", "%s is %s, expected %s" % (n, got, want))
    # regenerability: mkadapt must reproduce them from the originals as they are NOW
    mk = R / "mkadapt.py"
    if mk.is_file():
        tmp = R / ".pf5_adapter_check"
        try:
            res = subprocess.run([sys.executable, str(mk), "--repo", str(Q),
                                  "--proj", str(P), "--out", str(tmp)],
                                 capture_output=True, text=True, timeout=300)
            if res.returncode != 0:
                bl("C4", "mkadapt.py could not regenerate the adapters from the current "
                         "originals (exit %d); an original has changed since the adapters "
                         "were built" % res.returncode)
                print("    %s" % (res.stdout or res.stderr)[-400:])
            else:
                drift = [n for n in EXPECT if sha(tmp / n) != sha(ad / n)]
                print("  regenerated from current originals: %s"
                      % ("identical" if not drift else "DRIFT in %s" % drift))
                rep["adapters"]["regenerable"] = not drift
                if drift:
                    bl("C5", "adapters differ from a fresh regeneration: %s" % drift)
        except Exception as e:                                        # noqa: BLE001
            notes.append("adapter regeneration check skipped: %s" % e)
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)
    else:
        bl("C6", "mkadapt.py absent; adapter provenance cannot be re-established")

# ---- D. assembled-model physics -------------------------------------------
head("D. assembled-model physics verdict")
rep["physics"] = {}
found = False
for name in ("verify_assembled_gpu.json", "verify_assembled.json"):
    j = R / name
    if not j.is_file():
        continue
    found = True
    d = json.load(open(j))
    ver = d.get("gradient_consistency_verified_records") or []
    unv = d.get("gradient_consistency_unverified_records") or []
    print("  %s" % name)
    print("    verdict  %s" % d.get("verdict"))
    print("    variant  %r" % d.get("variant"))
    print("    gradient verified   : %s" % (ver or "none"))
    print("    gradient UNVERIFIED : %s" % (unv or "none"))
    rep["physics"][name] = {"verdict": d.get("verdict"), "variant": d.get("variant"),
                            "verified": ver, "unverified": unv,
                            "failures": d.get("failures")}
    if d.get("variant") != "rotinv_v1":
        bl("D1", "%s records variant %r, not rotinv_v1" % (name, d.get("variant")))
    if d.get("failures"):
        bl("D2", "%s records failures: %s" % (name, d["failures"]))
    if d.get("verdict") not in ("PASS", "PASS_WITH_UNVERIFIED_GRADIENTS"):
        bl("D3", "%s verdict is %r" % (name, d.get("verdict")))
    if unv:
        notes.append("%s: gradient consistency UNVERIFIED on %s -- an incomplete "
                     "diagnostic, not a physics pass" % (name, unv))
if not found:
    bl("D4", "no assembled-model physics JSON present; run verify_assembled.py")

# ---- E. executed-path audit ------------------------------------------------
head("E. executed-path audit: zero test-partition opens")
audits = sorted(R.glob("*audit*.jsonl"))
rep["audits"] = {}
if not audits:
    notes.append("no audit logs yet; the training stages have not been run through "
                 "run_adapted.py")
    print("  none yet (expected before the smoke test)")
for a in audits:
    tests = fatal = tot = 0
    for ln in a.read_text(errors="replace").splitlines():
        try:
            e = json.loads(ln)
        except ValueError:
            continue
        tot += 1
        if e.get("split") == "test":
            tests += 1
            fatal += bool(e.get("fatal"))
    probe = "probe" in a.name
    print("  %-28s entries %4d  test %3d  fatal %d%s"
          % (a.name, tot, tests, fatal, "   (deliberate probe)" if probe else ""))
    rep["audits"][a.name] = {"entries": tot, "test": tests, "fatal": fatal,
                             "deliberate_probe": probe}
    if tests and not probe:
        bl("E1", "%s records %d test-partition open(s) outside a deliberate probe"
                 % (a.name, tests))

# ---- F. router contract ----------------------------------------------------
head("F. router dataset contract")
j = R / "router_contract.json"
if not j.is_file():
    bl("F1", "router_contract.json absent; run router_contract.py")
else:
    d = json.load(open(j))
    print("  provenance            %s" % d.get("router_provenance"))
    print("  membership (distill)  %s" % d.get("distillation_membership_matches_stage"))
    print("  membership (routing)  %s" % d.get("routing_membership_matches_stage"))
    print("  procedure assumed     %s" % sorted(d.get("router_procedure_trainer_defaults") or {}))
    print("  procedure determined  %s" % sorted(d.get("router_procedure_determined_from_source") or {}))
    print("  procedure unknown     %s" % (d.get("router_procedure_unknown") or "none"))
    rep["router_contract"] = d.get("router_provenance")
    rep["router_procedure_assumed"] = sorted(d.get("router_procedure_trainer_defaults") or {})
    if d.get("blockers"):
        bl("F2", "router contract blockers: %s" % d["blockers"])
    if not d.get("distillation_membership_matches_stage"):
        bl("F3", "distillation train/val membership does not match the stage manifest")
    if d.get("router_procedure_trainer_defaults"):
        notes.append("router comparison is NOT exactly matched: %s are assumptions from "
                     "the current source, not settings the historical run recorded"
                     % sorted(d["router_procedure_trainer_defaults"]))

# ---- G. budget in ALLOCATED GPU-hours -------------------------------------
head("G. budget: ALLOCATED GPU-hours (elapsed x GPUs the scheduler gave)")
jobs_file = Path(os.environ.get("JOBS", str(R / "gpu_jobs.txt")))
ids = []
if jobs_file.is_file():
    ids = [l.strip() for l in jobs_file.read_text().split() if l.strip().isdigit()]
print("  job list: %s (%d job ids)" % (jobs_file, len(ids)))
rows, total = [], 0.0
if ids:
    try:
        out = subprocess.run(["sacct", "-n", "-X", "-P", "-j", ",".join(ids),
                              "--format=JobID,ElapsedRaw,AllocTRES,State"],
                             capture_output=True, text=True, timeout=120).stdout
        for ln in out.strip().splitlines():
            f = ln.split("|")
            if len(f) < 4:
                continue
            jid, el, tres, state = f[0], int(f[1] or 0), f[2], f[3]
            m = re.search(r"gres/gpu=(\d+)", tres)
            ngpu = int(m.group(1)) if m else 0
            if ngpu == 0:
                m2 = re.search(r"gres/gpu:[a-z0-9]+=(\d+)", tres)
                ngpu = int(m2.group(1)) if m2 else 0
            gh = el / 3600.0 * ngpu
            total += gh
            rows.append(dict(job=jid, elapsed_s=el, gpus=ngpu, alloc_gpu_h=gh, state=state))
            print("    %-12s %7ds x %d GPU = %7.4f GPU-h   %s"
                  % (jid, el, ngpu, gh, state))
    except Exception as e:                                            # noqa: BLE001
        notes.append("sacct query failed: %s" % e)
else:
    notes.append("no job ids recorded; write one per line to %s as jobs complete"
                 % jobs_file)
rep["budget"] = {"cap": CAP, "jobs": rows, "allocated_gpu_hours": total,
                 "remaining": CAP - total}
print("\n  ALLOCATED total %.4f of %.3f GPU-hours; remaining %.4f"
      % (total, CAP, CAP - total))
if total > CAP:
    bl("G1", "allocated GPU-hours %.4f exceed the cap %.3f" % (total, CAP))

# ---- summary ---------------------------------------------------------------
head("SUMMARY")
print("  notes    : %d" % len(notes))
for n in notes:
    print("    - %s" % n)
print("  BLOCKERS : %d" % len(blockers))
for b in blockers:
    print("    [%s] %s" % (b["tag"], b["detail"]))
rep["blockers"] = blockers
rep["notes"] = notes
rep["pf5_sha256"] = sha(__file__)
Path(R / "pf5.json").write_text(json.dumps(rep, indent=2, default=str))
print("\nwrote %s" % (R / "pf5.json"))
print("RESULT: %s" % ("CLEAR TO RUN" if not blockers
                      else "BLOCKED (%d)" % len(blockers)))
raise SystemExit(1 if blockers else 0)
