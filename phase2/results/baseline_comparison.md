# Comparison with published MIMIC-III medication-recommendation results

**Read this first.** The numbers below come from two *different* evaluation protocols
and must not be compared cell by cell:

| | literature (GAMENet / SafeDrug protocol) | this repository (HGDR) |
|---|---|---|
| label space | 131 ATC level-3 drug classes (NDC -> RXCUI -> ATC-4 -> ATC-3) | 606 normalised MIMIC drug names (active ingredient) |
| admissions | patients with >= 2 admissions only (6,350 patients / 15,032 visits) | all admissions with >= 1 diagnosis and >= 1 drug (39,239 patients / 50,085 visits; 30 % patient subset for the ablation) |
| procedures | `PROCEDURES_ICD` (ICD-9) | `PROCEDUREEVENTS_MV` item ids (ICD-9 procedures were not in the local copy) |
| DDI source | TWOSIDES, top-40 side effects | TWOSIDES, top-40 side effects (same convention) |
| split | random 2/3 - 1/6 - 1/6 by patient | hash-based 80/10/10 by patient |
| ground-truth DDI rate | 0.0808 | 0.087 (test split) |

A 606-way label space with rarer, more specific labels makes Jaccard/F1 *systematically
lower* than a 131-class ATC-3 space for the same underlying quality; the DDI rate and the
relative ordering of variants are the meaningful things to compare.

## Reported numbers (not re-run here)

Taken from the SafeDrug paper (Yang et al., IJCAI 2021, Table 2) and the respective
follow-up papers on the same MIMIC-III protocol. "reported" = copied from the paper.

| method | Jaccard ↑ | PR-AUC ↑ | F1 ↑ | DDI rate ↓ | #drugs | source |
|---|---:|---:|---:|---:|---:|---|
| LR (multi-hot) | 0.4865 | 0.7509 | 0.6434 | 0.0829 | 16.18 | reported (SafeDrug, Tab. 2) |
| ECC | 0.4996 | 0.6844 | 0.6569 | 0.0846 | 18.07 | reported |
| RETAIN (Choi et al. 2016) | 0.4887 | 0.7556 | 0.6481 | 0.0835 | 20.41 | reported |
| LEAP (Zhang et al. 2017) | 0.4521 | 0.6549 | 0.6138 | 0.0731 | 18.71 | reported |
| DMNC (Le et al. 2018) | 0.4864 | 0.7580 | 0.6529 | 0.0842 | 20.00 | reported |
| GAMENet (Shang et al. 2019) | 0.5067 | 0.7631 | 0.6626 | 0.0864 | 27.21 | reported |
| SafeDrug (Yang et al. 2021) | 0.5213 | 0.7647 | 0.6768 | 0.0589 | 19.92 | reported |
| MICRON (Yang et al. 2021b) | 0.5219 | 0.7489 | 0.6761 | 0.0640 | 17.93 | reported |
| COGNet (Wu et al. 2022) | 0.5336 | 0.7739 | 0.6869 | 0.0852 | 28.09 | reported |
| MoleRec (Yang et al. 2023) | 0.5305 | 0.7736 | 0.6843 | 0.0726 | 22.22 | reported |

DRecHGR (the DGL/HAN heterogeneous-graph recommender vendored in the research folder) and
DrugDAGT (Wang et al., a dual-attention graph transformer for DDI-*type* prediction) were
adapted in the original notebooks but could not be executed in the release environment
(DGL unavailable for torch 2.5 / Windows; DrugDAGT solves a different task - pairwise DDI
classification on DrugBank - and was only used as a safety scorer). They are acknowledged
in the README; no numbers are claimed for them.

## This repository (ablation protocol: 30 % patient subset, 2 seeds, <= 15 epochs)

See `ablation_table.md` for the full table with standard deviations. The rows that have
a literature counterpart are:

| method | protocol | Jaccard ↑ | PR-AUC ↑ | F1 ↑ | DDI rate ↓ | #drugs |
|---|---|---:|---:|---:|---:|---:|
| LR on multi-hot codes (`baseline_lr`) | ours | 0.3608 ± 0.0001 | 0.6224 ± 0.0001 | 0.5076 ± 0.0001 | 0.1000 ± 0.0000 | 14.78 |
| MLP on multi-hot codes (`baseline_mlp`) | ours | 0.3633 ± 0.0004 | 0.6261 ± 0.0012 | 0.5097 ± 0.0003 | 0.1003 ± 0.0006 | 15.15 |
| copy previous admission (`baseline_nearest`) | ours | 0.2100 | 0.4474 | 0.3348 | 0.0959 | 12.44 |
| **HGDR full** | ours | 0.3233 ± 0.0013 | 0.5621 ± 0.0016 | 0.4590 ± 0.0020 | **0.0779 ± 0.0020** | 15.07 |
| *ground-truth prescriptions (test)* | ours | - | - | - | 0.0869 | 26.1 |

Mean ± std over seeds 0 and 1, test split of 1,485 admissions (copied from `ablation_table.md` at
release time; if you re-run the study the authoritative numbers are in `ablation_table.md` /
`ablation_results.csv`). Note the same qualitative picture as in the literature table: the LR
baseline is a strong accuracy reference (in SafeDrug's Table 2 it is within 0.035 Jaccard of the
best method), and DDI-aware models buy their lower DDI rate with some accuracy. Under our
budget-limited protocol (<= 15 epochs) HGDR has not yet overtaken the multi-hot baselines on
accuracy; it is the only learned model whose recommendations are safer than the real prescriptions.

## This repository (full data, seed 0, ≤ 30 epochs)

A follow-up run of `configs/default.yaml` on **all** 39,239 patients / 50,085 admissions
(not a 30 % subset). One seed. Early-stopped at epoch 24 (best epoch 19), 12.3 min on a
GTX 1650 Ti. Authoritative numbers: `fulldata_full_seed0_summary.md` and
`runs/fulldata_full_seed0/metrics.json`. The subset ablation directories were not overwritten.

| method | protocol | Jaccard ↑ | PR-AUC ↑ | F1 ↑ | DDI rate ↓ | #drugs |
|---|---|---:|---:|---:|---:|---:|
| **HGDR full** | full data, seed 0 | 0.3561 | 0.6033 | 0.4969 | **0.0724** | 15.54 |
| HGDR full (from the table above) | 30 % subset, 2 seeds | 0.3233 ± 0.0013 | 0.5621 ± 0.0016 | 0.4590 ± 0.0020 | 0.0779 ± 0.0020 | 15.07 |
| MLP on multi-hot codes (from the table above) | 30 % subset, 2 seeds | 0.3633 ± 0.0004 | 0.6261 ± 0.0012 | 0.5097 ± 0.0003 | 0.1003 ± 0.0006 | 15.15 |
| *ground-truth prescriptions (full-data test)* | full data | - | - | - | 0.0866 | - |

Full-data training improved HGDR Jaccard by about +0.033 and lowered DDI to 0.072 (still
below real prescriptions). It did **not** overtake the published *subset* MLP (0.356 vs
0.363). The MLP was not re-run on full data, so that comparison is across protocols.

## References

- Shang, Xiao, Ma, Sun. *GAMENet: Graph Augmented MEmory Networks for Recommending Medication Combination.* AAAI 2019.
- Yang, Xiao, Ma, Glass, Sun. *SafeDrug: Dual Molecular Graph Encoders for Recommending Effective and Safe Drug Combinations.* IJCAI 2021.
- Yang, Xiao, Glass, Sun. *Change Matters: Medication Change Prediction with Recurrent Residual Networks (MICRON).* IJCAI 2021.
- Wu, Qiu, Wang, Yin, Tang, Wang. *Conditional Generation Net for Medication Recommendation (COGNet).* WWW 2022.
- Yang, Wu, Chen, Wang, Chen, Huang. *MoleRec: Combinatorial Drug Recommendation with Substructure-Aware Molecular Representation Learning.* WWW 2023.
- Tatonetti, Ye, Daneshjou, Altman. *Data-driven prediction of drug effects and interactions (TWOSIDES).* Sci. Transl. Med. 2012.
- Johnson et al. *MIMIC-III, a freely accessible critical care database.* Sci. Data 2016.
- Wang et al. *DrugDAGT: a dual-attention graph transformer with contrastive learning improves drug-drug interaction prediction.* (vendored as `dagt/` in the research code.)
