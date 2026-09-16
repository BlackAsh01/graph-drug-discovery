# Comparison with published KIBA results

All literature numbers below are **as reported in the cited papers — they were not re-run here**. Protocols differ
(DeepDTA/GraphDTA use the Öztürk et al. 5-fold train / held-out test partition of 118,254 pairs with up to 1,000
training epochs; the numbers of this work come from the **20 % subset protocol** of the 117,657-pair TDC copy —
18,826 training pairs, seed-42 split, ≤ 12 epochs on a 4 GB laptop GPU), so the table is indicative only: our
model saw ~5 × less data and ~100 × fewer epochs than the published baselines.

## Regression (KIBA score)

| Model | Drug repr. | Protein repr. | MSE ↓ | CI ↑ | r_m² ↑ | Source |
|---|---|---|---|---|---|---|
| KronRLS (Pahikkala 2015) | PubChem-Sim | Smith–Waterman | 0.411 | 0.782 | 0.342 | reported in [1] |
| SimBoost (He 2017) | PubChem-Sim | Smith–Waterman | 0.222 | 0.836 | 0.629 | reported in [1] |
| DeepDTA (Öztürk 2018) | 1-D CNN (SMILES) | 1-D CNN (sequence) | 0.194 | 0.863 | 0.673 | [1], r_m² via [3] |
| WideDTA (Öztürk 2019) | 1-D + PDM | 1-D + LMCS | 0.179 | 0.875 | – | reported in [2] |
| GraphDTA – GAT (Nguyen 2021) | GAT | 1-D CNN | 0.179 | 0.866 | – | [2] |
| GraphDTA – GIN | GIN | 1-D CNN | 0.147 | 0.882 | – | [2] |
| GraphDTA – GCN | GCN | 1-D CNN | 0.139 | 0.889 | – | [2] |
| GraphDTA – GAT_GCN | GAT + GCN | 1-D CNN | 0.139 | 0.891 | 0.780 | [2], r_m² via [4] |
| MGPLI (2022) | Transformer (SMILES) | Transformer | 0.159 | 0.891 | 0.753 | reported in [3] |
| GRA-DTA (2024) | GNN + attention | – | 0.142 | 0.890 | 0.784 | [3] |
| **This work — full model, 20 % subset protocol (2 seeds, ≤ 12 epochs)** | edge-aware graph transformer, 4 × 128 | ProtBERT-BFD (frozen) + MLP, cross-attention fusion | **0.4684 ± 0.0296** | **0.7343 ± 0.0088** | **0.3625 ± 0.0416** | `results/ablation_table.md` |
| This work — full model, full KIBA (`configs/default.yaml`, ≤ 40 epochs) | same | same | – | – | – | *not run for this release (≈ 1.5–2 h on the 4 GB GPU); `python scripts/train.py --config configs/default.yaml`* |

## Binarised interaction classification (KIBA ≥ 12.1 → positive)

GraphormerDTI itself is a classifier and its paper/README do not report KIBA numbers in a form that maps to the
regression setting. The most directly comparable published figures are from the EviDTI study, which re-ran
GraphormerDTI and other classifiers on binarised KIBA (5 replications) [5]. Our regressor's predictions are
thresholded at 12.1 post hoc (no classification training), so this row is a sanity check rather than a fair contest.

| Model | Accuracy | F1 | AUROC | AUPRC | Source |
|---|---|---|---|---|---|
| GraphDTA (as classifier) | 88.92 | 71.24 | 91.41 | 77.61 | [5] |
| MolTrans | 88.91 | 71.94 | 92.32 | 79.49 | [5] |
| HyperAttentionDTI | 89.33 | 74.21 | 93.51 | 81.41 | [5] |
| GraphormerDTI (Gao 2024) | 86.54 | 70.87 | 92.58 | 81.95 | [5] |
| EviDTI (2025) | 91.27 | 75.48 | 93.57 | 83.03 | [5] |
| **This work — regressor thresholded at 12.1, 20 % subset protocol (2 seeds)** | – | – | **0.7758 ± 0.0098** | **0.5752 ± 0.0227** | `results/ablation_results.csv` |

## Notes on comparability

* **Hardware and budget.** All results here were obtained on an NVIDIA GTX 1650 Ti (4 GB) / 16 GB RAM / Windows 11
  with ≤ 12 epochs on the 20 % subset (mean 5.4 min per full-model run; 1.8 h for the 22-run study). GraphDTA
  reports 1,000 epochs on the full data; the master thesis in [6] documents that GraphDTA reproductions with a
  small budget land at MSE ≈ 0.6–0.8, i.e. the budget matters a lot on KIBA. The full-data configuration
  (`configs/default.yaml`) is provided but was not run for this release.
* **Split.** Random pair-level (transductive) splits as in DeepDTA/GraphDTA; drugs and proteins can appear in
  both train and test. Inductive (cold-drug / cold-target) evaluation is out of scope for Phase 1.
* **Metrics.** CI and r_m² follow the DeepDTA definitions (`src/dti_gt/utils/metrics.py`; the CI implementation
  is unit-tested against a naive O(n²) reference).
* **Subset protocol.** The ablation numbers use 20 % of KIBA and a 12-epoch budget and are therefore *not*
  comparable with the literature rows; they are only comparable with each other.

## References

1. Öztürk, H., Özgür, A. & Ozkirimli, E. *DeepDTA: deep drug–target binding affinity prediction.* Bioinformatics 34, i821–i829 (2018).
2. Nguyen, T. et al. *GraphDTA: predicting drug–target binding affinity with graph neural networks.* Bioinformatics 37, 1140–1147 (2021), Table 3.
3. Zhang, X. et al. *Prediction of Drug-Target Affinity Using Attention Neural Network.* Int. J. Mol. Sci. 25, 5126 (2024), Table 3.
4. *Drug-Target Affinity Prediction Based on Graph Representation and Attention Fusion Mechanism* (2024), KIBA comparison table (r_m² for GraphDTA GAT_GCN).
5. *EviDTI* — Nature Communications 16 (2025), Table 2: "Comparison results of EviDTI and baselines on the KIBA dataset" (https://www.nature.com/articles/s41467-025-62235-6/tables/2).
6. Gálvez Rísquez, N. *Master thesis*, URV (2024) — GraphDTA reproduction study.
7. Gao, M. et al. *GraphormerDTI: A graph transformer-based approach for drug-target interaction prediction.* Comput. Biol. Med. 173, 108339 (2024).
