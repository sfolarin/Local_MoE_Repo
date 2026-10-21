# Expected results

What each stage should reproduce. Values marked (verify) were taken from the analysis during writing and must be
checked against the final output of the script named in the last column before submission.

| quantity | value | script |
|---|---|---|
| Structures per split (train / val / test) | 369 / 38 / 105 | `data_splits/make_oishi_holdout_split.py` |
| Pairs or geometry hashes spanning splits | 0 / 0 | `tests/check_gate_split.py` |
| Energy RMSE, full model (train / val / test) | 2.02 / 5.28 / 4.55 meV/atom (verify) | `evaluation/plot_energy_force_parity.py` |
| Energy MAE, full model, test | 2.275 meV/atom (verify) | same |
| Force structure-balanced RMSE (train / val / test) | 0.060 / 0.132 / 0.073 eV/A (verify) | same |
| Force pooled RMSE, test | 0.0915 eV/A (verify) | same |
| Interfacial energy, 40 held-out interface families (66 rows: 26 train + 40 test), full Local-MoE (checkpoint `c1b_holdout40_e10000_v1`): MAE / bias / RMSE / Pearson r | 0.0111 / +0.0108 / 0.0134 J/m2 / 0.8765 | `interface_energy/eval_gamma66_fullmoe.py` |
| Same, common-network part only | 0.0107 / +0.0099 / 0.0132 J/m2 / 0.8403 | same |
| Spearman rho on the 40 held-out families, full / common | 0.765 / 0.724 (verify) | `interface_energy/make_gamma_parity_figure.py` |
| Speedup vs MACE-MatPES-r2SCAN, mean / median | float32: 9.74x / 7.29x; float64: 12.32x / 9.09x (verify) | `cost/plot_computational_cost_holdout40.py` |
| Local-MoE time and memory scaling | N^0.07 and N^0.94 (verify) | same |
| NVE (10 ps, 0.5 fs, float64): structures energy-conserving | 52 / 52 (limit 0.1 meV/atom); maximum excursion 0.0077 meV/atom (verify) | `md/plot_local_moe_nve_drift.py` |
| NVE 500 ps | [pending] | `md/run_local_moe_long_nve.py` |
