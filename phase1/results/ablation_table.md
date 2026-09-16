# Ablation study (KIBA, subset protocol)

Protocol: `kiba_subset20_seed42` — train/val/test = 18826/2353/2353 pairs; batch 128, ≤ 12 epochs, early stopping patience 4 on val MSE, AdamW lr 0.0005; seeds = [0, 1]; best-validation-MSE checkpoint evaluated on the test split. Values are mean ± std over seeds; percentages are relative MSE change vs. the full model. Runs were executed sequentially on one GTX 1650 Ti (4 GB).

| Variant | Description | n | MSE ↓ | CI ↑ | r_m² ↑ | Pearson ↑ | AUROC ↑ | Params | Time / run |
|---|---|---|---|---|---|---|---|---|---|
| `full` | Full model (edge-aware GT, 4 layers, 8 heads, degree enc., ProtBERT-BFD, cross-attention fusion) | 2 | 0.4684 ± 0.0296 | 0.7343 ± 0.0088 | 0.3625 ± 0.0416 | 0.6079 ± 0.0326 | 0.7758 ± 0.0098 | 1.54 M | 5.4 min |
| `no_edge_features` | - bond features in attention / edge channel | 2 | 0.4543 ± 0.0086 (-3.0%) | 0.7443 ± 0.0014 | 0.3702 ± 0.0170 | 0.6268 ± 0.0074 | 0.7879 ± 0.0066 | 1.14 M | 4.2 min |
| `no_degree_encoding` | - centrality (degree) encoding | 2 | 0.4477 ± 0.0044 (-4.4%) | 0.7428 ± 0.0009 | 0.3875 ± 0.0108 | 0.6397 ± 0.0022 | 0.7907 ± 0.0045 | 1.54 M | 4.8 min |
| `depth_1` | 1 transformer layer instead of 4 | 2 | 0.4787 ± 0.0153 (+2.2%) | 0.7306 ± 0.0004 | 0.3437 ± 0.0178 | 0.5979 ± 0.0176 | 0.7643 ± 0.0038 | 0.85 M | 2.5 min |
| `heads_1` | 1 attention head instead of 8 | 2 | 0.4731 ± 0.0082 (+1.0%) | 0.7304 ± 0.0040 | 0.3559 ± 0.0022 | 0.6064 ± 0.0104 | 0.7726 ± 0.0030 | 1.54 M | 4.4 min |
| `no_residual_norm` | - residual connections, - normalisation | 2 | 0.6537 ± 0.0016 (+39.6%) | 0.6731 ± 0.0007 | 0.1183 ± 0.0007 | 0.3582 ± 0.0015 | 0.6485 ± 0.0007 | 1.54 M | 4.4 min |
| `protein_cnn` | Protein: learned 1-D CNN on sequence (instead of ProtBERT-BFD) | 2 | 0.4550 ± 0.0078 (-2.9%) | 0.7458 ± 0.0003 | 0.3826 ± 0.0108 | 0.6268 ± 0.0046 | 0.7788 ± 0.0047 | 1.66 M | 11.6 min |
| `protein_composition` | Protein: amino-acid composition MLP (instead of ProtBERT-BFD) | 2 | 0.6028 ± 0.0146 (+28.7%) | 0.6386 ± 0.0059 | 0.1836 ± 0.0202 | 0.4444 ± 0.0132 | 0.7141 ± 0.0083 | 1.27 M | 5.4 min |
| `fusion_concat` | Fusion: concat (instead of cross-attention) | 2 | 0.4511 ± 0.0029 (-3.7%) | 0.7402 ± 0.0001 | 0.3903 ± 0.0036 | 0.6272 ± 0.0032 | 0.7778 ± 0.0100 | 1.44 M | 4.8 min |
| `backbone_gcn` | Drug encoder: GCN (original notebook backbone) | 2 | 0.4604 ± 0.0053 (-1.7%) | 0.7336 ± 0.0002 | 0.3866 ± 0.0006 | 0.6237 ± 0.0012 | 0.7751 ± 0.0038 | 0.68 M | 2.5 min |
| `backbone_gat` | Drug encoder: GAT | 2 | 0.4742 ± 0.0028 (+1.2%) | 0.7260 ± 0.0065 | 0.3547 ± 0.0083 | 0.6042 ± 0.0031 | 0.7621 ± 0.0073 | 0.68 M | 2.3 min |
