# Raw prediction archive (not in Git)

The per-structure prediction arrays are kept out of Git and distributed as a
single archive.

| field | value |
|---|---|
| file | `local_moe_raw_predictions_20261026.tar.gz` |
| size | 49 MB |
| .npz members | 1024 (512 configurations x 2 models) |
| sha256 | `5d475ba75df4398dc161628062eb5c1b4cf7f615f38321e680525a0904296dd0` |

Layout: `moe/raw_predictions/<split>/<record_id>.npz` for the full Local-MoE
and `common/raw_predictions/<split>/<record_id>.npz` for the common network.
Splits are train (369), val (38), test (105).

Each .npz holds `record_id`, `n_atoms`, `E_reference_eV`,
`E_predicted_eV`, `reference_forces` and `predicted_forces`, all float64.

Per-configuration energies are also committed as
`metrics/local_moe_per_configuration.csv`, so the energy results are
reproducible from Git alone; the archive is required only for force arrays.
