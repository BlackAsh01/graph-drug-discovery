# Full-data HGDR run (`configs/default.yaml`, seed 0)

**This is the full-MIMIC run**, not the published 30 % subset ablation.
It does **not** replace `results/runs/full_seed0` (that directory is the subset-protocol `full` seed).

Command: `python scripts/train.py --config configs/default.yaml --seed 0`  
Run directory: `results/runs/fulldata_full_seed0/` (`metrics.json`, `history.csv`, `config.yaml`).  
Checkpoint `best.pt` was deleted after evaluation (git-ignored; trained on credentialed data).

## Protocol

| | this run | published ablation (`full`) |
|---|---|---|
| config | `configs/default.yaml` | `configs/ablation_base.yaml` |
| patients (train/val/test) | 31,268 / 3,961 / 4,010 | 9,349 / 1,182 / 1,196 |
| admissions | 39,992 / 5,032 / 5,061 | 11,954 / 1,471 / 1,485 |
| max epochs / patience | 30 / 5 (early stop) | 15 / 4 |
| batch / lr | 128 / 1e-3 | 256 / 2e-3 |
| seeds | 0 only | 0 and 1 |
| hardware | GTX 1650 Ti 4 GB, CUDA | same |

## Test metrics (seed 0)

Source: `results/runs/fulldata_full_seed0/metrics.json`.

| | Jaccard ↑ | PR-AUC ↑ | F1 ↑ | DDI rate ↓ | #drugs | P@10 ↑ | Jaccard@10 ↑ | HitRate@10 ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **HGDR full data (test)** | 0.3561 | 0.6033 | 0.4969 | **0.0724** | 15.54 | 0.7065 | 0.2565 | 0.9949 |
| val (best checkpoint) | 0.3543 | 0.6036 | 0.4951 | 0.0723 | 15.21 | 0.7042 | 0.2582 | 0.9946 |
| *ground-truth test prescriptions* | – | – | – | 0.0866 | – | – | – | – |

- best epoch: **19** (early stopped at epoch 24, patience 5)
- wall time: **12.3 min** (738 s, includes data loading and the final test evaluation)
- trainable params: 493,022
- device: cuda
- test admissions: 5,061

## Comparison (honest)

These rows are **not** the same protocol. Ablation numbers are 2-seed means on a 30 % patient
subset with a 15-epoch cap. The multi-hot MLP was **not** re-run on full data.

| run | protocol | Jaccard ↑ | F1 ↑ | DDI rate ↓ |
|---|---|---:|---:|---:|
| HGDR `full` (this run) | **full data**, ≤ 30 epochs, seed 0 | **0.3561** | 0.4969 | **0.0724** |
| HGDR `full` (published ablation) | 30 % patients, ≤ 15 epochs, 2 seeds | 0.3233 ± 0.0013 | 0.4590 ± 0.0020 | 0.0779 ± 0.0020 |
| MLP multi-hot (published ablation) | 30 % patients, ≤ 15 epochs, 2 seeds | 0.3633 ± 0.0004 | 0.5097 ± 0.0003 | 0.1003 ± 0.0006 |

**Takeaway.** Full-data training improved HGDR Jaccard by about +0.033 over the 15-epoch 30 %
ablation and made recommendations safer (DDI 0.072 vs 0.078, still below the 0.087 of real
prescriptions). It did **not** overtake the published *subset* MLP on Jaccard (0.356 vs 0.363).
Whether a full-data MLP would stay ahead is unknown; that baseline was not re-run here.
