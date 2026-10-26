#!/usr/bin/env python3
"""router_contract.py -- resolve the router dataset contract before training.

The adapter's R2 patch only asserts that the test partition is empty.  That says
nothing about WHICH 369/38 records are present, so this verifies membership
against the preserved stage manifest, record by record.

It also tries to identify what produced the existing router checkpoint, whose
script asserts (112, 22, 22) while the holdout-40 distillation manifest is
369/38/105.  If no dataset with those counts exists, the provenance is marked
unresolved and no exactly-matched retraining comparison may be claimed.

Read-only.  Exit 0 if membership verifies, 1 otherwise."""
import csv, json, os, re, sys
from pathlib import Path

P = Path(os.environ["PROJ"])
STAGE = P / "data" / "expanded_holdout40_v1_c1b_stage"
DIST = P / "data" / "expanded_holdout40_v1_distillation"
ROUTING = P / "data" / "expanded_holdout40_v1_routing"
blockers, rep = [], {}


def rows(p):
    with open(p, newline="", errors="replace") as f:
        r = csv.DictReader(f)
        return [dict(x) for x in r], list(r.fieldnames or [])


def membership(path, idcols=("record_id", "source_file", "output_file", "shard_file")):
    rr, cc = rows(path)
    sc = next((c for c in ("split", "split_group") if c in cc), None)
    ic = next((c for c in idcols if c in cc), None)
    if sc is None or ic is None:
        raise SystemExit("FAIL: %s lacks split or id column (cols=%s)" % (path, cc))
    out = {}
    for r in rr:
        out.setdefault((r[sc] or "").strip(), set()).add(Path(r[ic]).stem)
    return out, ic, sc


print("=" * 90); print("ROUTER DATASET CONTRACT"); print("=" * 90)
st, st_ic, st_sc = membership(STAGE / "manifest.csv")
print("stage manifest      %-28s id=%s split=%s" % (STAGE.name, st_ic, st_sc))
print("  counts: %s" % {k: len(v) for k, v in sorted(st.items())})
rep["stage_counts"] = {k: len(v) for k, v in st.items()}

for label, root in (("distillation", DIST), ("routing", ROUTING)):
    man = root / "manifest.csv"
    if not man.is_file():
        man = root / "dataset_manifest.csv"
    if not man.is_file():
        print("\n%s: NO MANIFEST under %s" % (label, root))
        blockers.append("%s manifest missing (looked for manifest.csv and "
                        "dataset_manifest.csv under %s)" % (label, root))
        continue
    ds, ic, sc = membership(man)
    print("\n%s manifest %-22s id=%s split=%s" % (label, man.name, ic, sc))
    print("  counts: %s" % {k: len(v) for k, v in sorted(ds.items())})
    rep["%s_counts" % label] = {k: len(v) for k, v in ds.items()}
    ok = True
    for sp in ("train", "val"):
        a, b = st.get(sp, set()), ds.get(sp, set())
        extra, lost = b - a, a - b
        same = not extra and not lost
        ok = ok and same
        print("  %-5s membership vs stage: %s  (only-here %d, only-stage %d)"
              % (sp, "IDENTICAL" if same else "DIFFERS", len(extra), len(lost)))
        if not same:
            print("      e.g. only-here %s | only-stage %s"
                  % (sorted(extra)[:3], sorted(lost)[:3]))
    rep["%s_membership_matches_stage" % label] = ok
    if not ok:
        blockers.append("%s train/val membership differs from the preserved stage manifest"
                        % label)
    if ds.get("train") and len(ds["train"]) != 369 or ds.get("val") and len(ds["val"]) != 38:
        blockers.append("%s counts are not 369/38" % label)

print("\n" + "=" * 90); print("PROVENANCE OF THE EXISTING ROUTER CHECKPOINT"); print("=" * 90)
print("the router script asserts (112, 22, 22); searching every manifest for those counts")
hits = []
for man in sorted(P.glob("data/*/*manifest*.csv")):
    try:
        ds, _, _ = membership(man)
    except SystemExit:
        continue
    except Exception:
        continue
    c = (len(ds.get("train", ())), len(ds.get("val", ())), len(ds.get("test", ())))
    if c == (112, 22, 22):
        hits.append(str(man.relative_to(P)))
        print("  MATCH (112,22,22): %s" % man.relative_to(P))
rep["counts_112_22_22_found_in"] = hits

# The run directory records its own inputs; that is authoritative and outranks
# any inference from the script's current assertion, which may be stale.
RUN = P / "runs" / "torchsoap_gate_distillation_holdout40_v1"
info = RUN / "training_info.json"
recorded = {}
if info.is_file():
    recorded = json.load(open(info))
    print("\n%s records its own provenance:" % info.name)
    print("  dataset_root = %s" % recorded.get("dataset_root"))
    print("  best_epoch   = %s" % recorded.get("best_epoch"))
    dr = str(recorded.get("dataset_root", ""))
    same = Path(dr).resolve() == DIST.resolve() if dr else False
    print("  dataset_root is the holdout-40 distillation dataset: %s" % same)
    rep["router_training_info"] = recorded
    rep["router_dataset_root_matches_distillation"] = same
    # split sizes as recorded by the run's own metrics, not inferred
    met = RUN / "routing_distillation_metrics.csv"
    sizes = {}
    if met.is_file():
        for r in csv.DictReader(open(met, newline="", errors="replace")):
            if r.get("region") == "all":
                sizes[r["split"]] = int(r["n_structures"])
    tf = RUN / "test_force_mixture_metrics.csv"
    if tf.is_file():
        for r in csv.DictReader(open(tf, newline="", errors="replace")):
            if r.get("region") == "all" and r.get("predictor") == "teacher":
                sizes["test"] = int(r["n_structures"])
    print("  split sizes recorded by the run: %s" % sizes)
    rep["router_recorded_split_sizes"] = sizes
    expected = {k: len(v) for k, v in st.items()}
    if sizes and sizes == expected:
        print("  -> MATCHES the preserved stage manifest %s" % expected)
        rep["router_provenance"] = "resolved_from_training_info_json"
        print("  -> the (112, 22, 22) assertion in the script on disk is STALE;")
        print("     that file was edited after this run, or another version produced it.")
    elif sizes:
        print("  -> DIFFERS from the stage manifest %s" % expected)
        rep["router_provenance"] = "recorded_sizes_differ_from_stage"
        blockers.append("router recorded split sizes %s differ from the stage "
                        "manifest %s" % (sizes, expected))
    # procedure settings: membership alone does not match a training procedure
    print("\n  recorded training procedure settings:")
    PROC = ("lr", "weight_decay", "max_epochs", "patience", "min_delta", "grad_clip",
            "seed", "dropout", "hidden_dims", "optimizer", "scheduler", "loss",
            "selection_metric", "normalization", "feature_space")
    present, absent = {}, []
    for k in PROC:
        hit = next((kk for kk in recorded if kk.lower() == k or k in kk.lower()), None)
        if hit:
            present[hit] = recorded[hit]
            print("    %-26s %s" % (hit, recorded[hit]))
        else:
            absent.append(k)
    print("    NOT RECORDED: %s" % (absent or "none"))
    rep["router_procedure_recorded"] = present
    # the checkpoint carries settings that training_info.json omits
    ck = RUN / "best_torchsoap_routing_gate.pt"
    from_ck = {}
    if ck.is_file():
        try:
            import torch
            d = torch.load(ck, map_location="cpu", weights_only=False)
            for k in sorted(d):
                if k in ("model_state_dict", "feature_mean", "feature_std",
                         "expert_names"):
                    continue
                from_ck[k] = d[k]
            print("\n  additionally recorded in the checkpoint:")
            for k, v in from_ck.items():
                print("    %-26s %r" % (k, v))
            if "feature_mean" in d and "feature_std" in d:
                print("    %-26s present (%d-dim), i.e. normalisation is stored with"
                      % ("feature_mean/std", len(d["feature_mean"])))
                print("    %-26s the checkpoint, not recomputed at load time" % "")
                from_ck["normalization_stored_in_checkpoint"] = True
        except Exception as e:                                       # noqa: BLE001
            print("  could not read the checkpoint: %s: %s" % (type(e).__name__, e))
    rep["router_procedure_from_checkpoint"] = {
        k: (v if isinstance(v, (int, float, str, bool, list)) else str(v))
        for k, v in from_ck.items()}
    still = [k for k in absent
             if not any(k in kk.lower() for kk in from_ck)]
    rep["router_procedure_not_recorded"] = still

    # what the trainer's own defaults would supply for the remainder
    TRAINER = P / "runs" / "train_local_torchsoap_gate_distillation.py"
    defaults = {}
    if TRAINER.is_file():
        txt = TRAINER.read_text(errors="replace")
        for k in still:
            m = re.search(r'add_argument\(\s*["\']--%s["\'][^)]*?default\s*=\s*'
                          r'([^,)\n]+)' % re.escape(k), txt, re.S)
            if m:
                defaults[k] = m.group(1).strip()
        if defaults:
            print("\n  the trainer's argparse defaults would supply:")
            for k, v in sorted(defaults.items()):
                print("    %-26s %s   (ASSUMED, not recorded by the run)" % (k, v))
    rep["router_procedure_trainer_defaults"] = defaults

    # Some settings are not argparse flags but are fixed in the trainer source.
    # Those are determinable by reading the code, so they are neither "recorded"
    # nor "unknown": they are determined, and reported as such.
    SOURCE = {
        "optimizer":        r"torch\.optim\.(\w+)\s*\(",
        "scheduler":        r"lr_scheduler\.(\w+)\s*\(",
        "scheduler_params": r"lr_scheduler\.\w+\s*\([^)]*?factor\s*=\s*([\d.]+)[^)]*?"
                            r"patience\s*=\s*(\d+)[^)]*?min_lr\s*=\s*([\deE.\-+]+)",
        "loss":             r'Loss:\s*\{?["\']?([^"\'}\n]*KL[^"\'}\n]*)',
        "selection_metric": r"best_val_(\w+)",
        "normalization":    r"(compute_train_normalization)\s*\(",
    }
    determined = {}
    if TRAINER.is_file():
        for k, pat in SOURCE.items():
            m = re.search(pat, txt, re.S)
            if m:
                determined[k] = " ".join(x for x in m.groups() if x)
        if determined:
            print("\n  determined by reading the trainer source:")
            for k, v in sorted(determined.items()):
                print("    %-26s %s" % (k, v))
    rep["router_procedure_determined_from_source"] = determined
    unknown = [k for k in still
               if k not in defaults and not any(k in d for d in determined)]
    rep["router_procedure_unknown"] = unknown

    print("\n  PROCEDURE MATCH SUMMARY")
    print("    recorded by the run        : %s" % (sorted(present) or "none"))
    print("    recorded in the checkpoint : %s" % (sorted(from_ck) or "none"))
    print("    assumed from trainer flags : %s" % (sorted(defaults) or "none"))
    print("    determined from source     : %s" % (sorted(determined) or "none"))
    print("    neither recorded nor found : %s" % (unknown or "none"))
    if still:
        print("    -> the PROCEDURE is matched only to the extent above.  Settings in")
        print("       the 'assumed' row are the trainer's defaults, which the run did")
        print("       not record; they must be reported as assumed, not verified.")
    if unknown:
        blockers.append("router procedure settings neither recorded nor available as a "
                        "trainer default: %s" % unknown)
else:
    print("\n%s absent; cannot resolve provenance from the run directory" % info)
    blockers.append("router training_info.json absent; provenance unresolved")

if not hits and not recorded:
    print("  no manifest anywhere under data/ has (112, 22, 22)")
    print("  -> the dataset that produced the existing router checkpoint is NOT present.")
    print("     Provenance UNRESOLVED. No exactly-matched retraining comparison may be")
    print("     claimed against that checkpoint; the pilot's router must be described as")
    print("     trained on the holdout-40 distillation dataset, compared only to itself.")
    rep["router_provenance"] = "unresolved_dataset_absent"
elif hits:
    rep.setdefault("router_provenance", "candidate_datasets_found")

print("\nblockers: %d" % len(blockers))
for b in blockers:
    print("  - %s" % b)
rep["blockers"] = blockers
Path("router_contract.json").write_text(json.dumps(rep, indent=2, default=str))
print("wrote router_contract.json")
print("RESULT: %s" % ("membership verified" if not blockers else "BLOCKED"))
raise SystemExit(1 if blockers else 0)
