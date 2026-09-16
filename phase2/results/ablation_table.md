# HGDR ablation study (MIMIC-III)

Mean ± std over seeds (`n_seeds` column). Ground-truth DDI rate of the test prescriptions: 0.0869.
Arrows: ↑ higher is better, ↓ lower is better. `#drugs` is the average recommended-set size.

| Variant | Description | Jaccard ↑ | PR-AUC ↑ | F1 ↑ | DDI rate ↓ | #drugs | Runtime (min) | Seeds |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `full` | HGDR (full model) | 0.3233 ± 0.0013 | 0.5621 ± 0.0016 | 0.4590 ± 0.0020 | 0.0779 ± 0.0020 | 15.07 ± 0.20 | 4.2 | 2 |
| `no_ddi` | - DDI edges & DDI loss | 0.3242 ± 0.0024 | 0.5654 ± 0.0030 | 0.4611 ± 0.0023 | 0.1152 ± 0.0015 | 14.83 ± 0.16 | 4.3 | 2 |
| `no_ddi_loss` | - DDI loss (edges kept) | 0.3331 ± 0.0007 | 0.5687 ± 0.0009 | 0.4722 ± 0.0006 | 0.1094 ± 0.0018 | 15.74 ± 0.20 | 4.3 | 2 |
| `no_mol` | - molecular (SMILES) drug encoder | 0.3104 ± 0.0005 | 0.5488 ± 0.0017 | 0.4444 ± 0.0001 | 0.0841 ± 0.0003 | 14.08 ± 0.21 | 3.5 | 2 |
| `homo_graph` | homogeneous graph (types collapsed) | 0.3081 ± 0.0030 | 0.5592 ± 0.0011 | 0.4389 ± 0.0036 | 0.0853 ± 0.0016 | 13.90 ± 0.52 | 3.4 | 2 |
| `no_proc` | - procedure nodes | 0.2914 ± 0.0006 | 0.5336 ± 0.0047 | 0.4216 ± 0.0015 | 0.0885 ± 0.0016 | 13.48 ± 0.10 | 2.8 | 2 |
| `no_history` | - visit-history GRU | 0.3237 ± 0.0024 | 0.5695 ± 0.0003 | 0.4585 ± 0.0029 | 0.0778 ± 0.0009 | 14.68 ± 0.18 | 2.4 | 2 |
| `no_gnn` | - graph message passing (0 layers) | 0.2594 ± 0.0297 | 0.4811 ± 0.0498 | 0.3906 ± 0.0273 | 0.0901 ± 0.0008 | 13.01 ± 0.46 | 2.2 | 2 |
| `gnn_1layer` | 1 GNN layer (instead of 2) | 0.3190 ± 0.0001 | 0.5616 ± 0.0007 | 0.4535 ± 0.0003 | 0.0850 ± 0.0024 | 14.55 ± 0.19 | 3.6 | 2 |
| `baseline_mlp` | MLP on multi-hot codes | 0.3633 ± 0.0004 | 0.6261 ± 0.0012 | 0.5097 ± 0.0003 | 0.1003 ± 0.0006 | 15.15 ± 0.02 | 1.5 | 2 |
| `baseline_lr` | logistic regression on multi-hot codes | 0.3608 ± 0.0001 | 0.6224 ± 0.0001 | 0.5076 ± 0.0001 | 0.1000 ± 0.0000 | 14.78 ± 0.00 | 1.6 | 2 |
| `baseline_nearest` | copy previous admission | 0.2100 ± 0.0000 | 0.4474 ± 0.0000 | 0.3348 ± 0.0000 | 0.0959 ± 0.0000 | 12.44 ± 0.00 | 0.2 | 2 |
