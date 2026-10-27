# Frozen artefacts and executed code

All four frozen artefacts were verified against these hashes before the
evaluation and again after it, unchanged.

| artefact | sha256 |
|---|---|
| common checkpoint | `5b17849752a535739b16cfc8e390e9dd53bbe27c8e50b975235142cf16e346e1` |
| router checkpoint | `9e8e0ca6af4976ef8c74c34e1ded97b2aebeabb51c278037ab3ecab1145e1c66` |
| residual checkpoint | `c7ad2a89c570549309ad8906e477a42adad298b51251da5aa33a09c878ade451` |
| corrected descriptor | `0afd9a1ba7b9a24f8bf8151a7a3bdfc409544ed0920355de0598ffae79d51d21` |

## Project scripts used, unmodified

| script | sha256 |
|---|---|
| `export_c1b_parity_predictions_holdout40.py` | `40f7c795dc46423dac4eed0be89a3d8db37e15f69e0a6620b33f3f0acb7efe75` |
| `plot_energy_force_parity_holdout40.py` | `cbca3f795b3358a4c37876d53430d0aa8f915259dff2780ecf0857216e3adf36` |
| `export_common_parity_predictions_v2.py` | `2ecc8d404c0b93283efbe5b4bb9539fb95da7cbd10a99d973af4a3c18d15cbd3` |
| `eval_test_regions_common_vs_full.py` | `42845f3c68fdc51189bb09a33af4c207b097ea11c7068f0b3128404e56cae8e1` |
| `train_c1b_holdout40.py` | `333dd835069596d81d43b264a9c95b46152b44cc8a8897de1b9d2af427ac477f` |
| `train_expanded_moe_common.py` | `dba8920e80bff68731ddfad9fbf804a208066ab070877d0a8bb395974e59294a` |

## Code written for this evaluation

| script | sha256 |
|---|---|
| `final_eval.sh` | `803d9389bf8a636452fe162b54a80796a1f0bdccdc72482e0066f22a5f97957c` |
| `finish_eval.sh` | `96cd49cbaa815450e65d972ebe8424e628fe984318b6d1ac2cadc2cf2ced0085` |
| `patch_compare.py` | `090dc85d6008f8e8a568b5d3a044782621ae9a0f8a16b68894585d24b27fe201` |
| `paired_test.py` | `53af1cfa29bc73b248d882ee5ed4e9d91e127f42fc4b983a30262c20077785e3` |
| `iface_test.py` | `a414068f5cec8d26736497c396654d36f9237ab7db6baab7f3a19931570e7f29` |
| `eval_compare_patched.py` | `05926b2949ddd6ce93a1094054c50741ce2cb7e7182c4995db1a57d96e11b900` |

`eval_compare_patched.py` is a patched copy of
`eval_test_regions_common_vs_full.py`; the original is read-only and
unchanged.  See section 8 of the report for the defect and the four anchored
patches.
