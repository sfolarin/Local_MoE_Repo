# Adapters are generated, not committed

`mkadapt.py` produces three adapters by anchored patching of the original
trainers.  The adapters are full copies of those trainers and are therefore NOT
committed here; regenerate them with:

    python3 mkadapt.py --repo $REPO --proj $PROJ --out <out dir>

Verified hashes (seven anchors, each matched exactly once):

| adapter                    | sha256         | from original  |
|----------------------------|----------------|----------------|
| prep_features_adapter.py   | 175ecc61b4fb4d | f7facba6abb67f |
| router_distill_adapter.py  | 5940fc1e517e6d | d34dfd668b5ccd |
| c1b_holdout40_adapter.py   | 3705bf0a3aa4bf | 333dd835069596 |
