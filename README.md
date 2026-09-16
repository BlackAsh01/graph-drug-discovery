# Graph models for drug discovery

[![python](https://img.shields.io/badge/python-3.10%20%7C%203.12-blue)](#installation)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![phase1 tests](https://img.shields.io/badge/phase1-pytest%2022-brightgreen)](phase1/README.md#tests)
[![phase2 tests](https://img.shields.io/badge/phase2-pytest%2034-brightgreen)](phase2/README.md#tests)

Publication-ready release of a two-phase graph-ML thesis by **Ashwin Prabhu M**
(Register No. 2023176029), M.Tech. Information Technology (AI & DS),
Department of Information Science and Technology, College of Engineering,
Guindy, Anna University (guide: Dr. T. Mala).

Two AUIST project reports back this repository. The December 2024 cover title
is *Graph Transformer Based Personalized Drug Recommendation System for
Cardiovascular Disease* (report body: KIBA DTI / graph transformer / ProtBERT)
and is the source for **Phase 1**. The May 2025 cover title is
**GRAND: Graph-based Recommendation with Attention Network for Personalized Drug Therapy**
and is the source for **Phase 2**. Both PDFs self-describe as an AUIST
“Phase-I” project report; that label is **not** the GitHub folder numbering
(see [Citation](#citation)).

| Phase | Package | Task | Data | Headline (honest, budget-limited) |
|---|---|---|---|---|
| **[1](phase1/README.md)** | `dti_gt` | Drug–target **binding-affinity** regression | [KIBA](https://doi.org/10.1021/ci400709d) (TDC) | Residuals + layer-norm and a real protein encoder matter; Graphormer-style extras did not pay off on a 20 % subset / 12-epoch protocol |
| **[2](phase2/README.md)** | `hgdr` | **Medication recommendation** with DDI control (EHR+DDI subset of GRAND) | [MIMIC-III](https://physionet.org/content/mimiciii/1.4/) (not shipped) | HGDR is the **safest** learned model on DDI rate; multi-hot MLP / logistic regression are **more accurate** under the 15-epoch protocol |

Both packages are from-scratch, DGL-free re-implementations of the research notebooks (bugs fixed; see each phase's `CHANGELOG.md`). They train on a laptop **GTX 1650 Ti (4 GB)** / 16 GB RAM. This repository is new and independent of `BlackAsh01/DTI_GraphTransformer` and `BlackAsh01/DAGT_drug_recommendation`.

---

## Contents

1. [Abstracts](#abstracts)
2. [Architecture](#architecture)
3. [Repository layout](#repository-layout)
4. [Installation](#installation)
5. [Quick start](#quick-start)
6. [Headline results](#headline-results)
7. [Reproducibility and hardware](#reproducibility-and-hardware)
8. [Data and licensing](#data-and-licensing)
9. [Citation](#citation)
10. [License](#license)

---

## Abstracts

**Phase 1 — edge-aware graph transformer for KIBA drug–target affinity.**
Given a drug SMILES and a kinase sequence, the model regresses the continuous KIBA score. The drug is an RDKit molecular graph encoded by an edge-aware graph transformer (bond-feature-biased sparse attention, Graphormer-style degree encoding, residual + BatchNorm blocks) adapted from [GraphormerDTI](https://github.com/mengmeng34/GraphormerDTI); the protein is a frozen ProtBERT-BFD embedding (or a learned 1-D CNN); the two are fused by protein-conditioned cross-attention over atoms and scored by a 3-layer MLP. Every component is a config flag. Under a fixed 20 % stratified KIBA subset (18,826 training pairs, ≤ 12 epochs, 2 seeds) the full 1.54 M model reaches **test MSE 0.4684 ± 0.0296 / CI 0.7343 ± 0.0088 / r_m² 0.3625 ± 0.0416** in **5.4 min** per run. Dropping residuals/norm inflates MSE by **+40 %**; a bag-of-residues protein encoder costs **+29 %**. Bond features, degree encoding and cross-attention fusion did not help under this budget; a 0.68 M GCN backbone is competitive. Full-KIBA training (`phase1/configs/default.yaml`) is provided but was not run for this release. Details: [`phase1/README.md`](phase1/README.md).

**Phase 2 — GRAND / HGDR medication recommendation on MIMIC-III.**
The official architecture (EHR diagnose / genome / medication channels, meta-paths
P–D–P / P–M–P / M–P–M / M–G–M, semantic fusion, Dual-attention Graph Transformer
+ InfoNCE DDI branch, recommendation loss \(\mathcal{L}_1\)) is in
[`phase2/docs/figures/architecture_final.jpg`](phase2/docs/figures/architecture_final.jpg)
and is summarised in [`phase2/docs/thesis_notes.md`](phase2/docs/thesis_notes.md).
The **released `hgdr` package trains the EHR+DDI subset that the ablation
measured**: diagnosis / procedure / drug nodes (no genomes, no patient nodes),
a TWOSIDES DDI graph with a differentiable DDI-rate penalty (the shipped
\(\mathcal{L}_1\)), a GAT/GIN molecular encoder on PubChem SMILES (not the
InfoNCE Dual-attention Graph Transformer), and attention-pooled admissions plus
a visit-history GRU. The task is the GAMENet / SafeDrug multi-label setting.
Under a 30 % patient subset / ≤ 15 epochs / 2-seed protocol, **HGDR is safest**
(DDI rate **0.0779 ± 0.0020**, below the 0.087 of the real prescriptions) but a
**multi-hot MLP (Jaccard 0.363)** and **logistic regression (0.361)** are more
accurate than HGDR (**0.323**). A later full-data seed-0 run (≤ 30 epochs)
reached Jaccard **0.356** / DDI **0.072**. Report Tables 4.1–4.2 (DDI MAE,
substitute Jaccard 0.967) are a **different protocol** and are not reproduced
here. **MIMIC-III patient data are not in this repo.** Details and the
thesis-vs-code gap table: [`phase2/README.md`](phase2/README.md).

---

## Architecture

<p align="center">
  <img src="phase1/docs/figures/thesis_architecture.jpg" alt="Thesis architecture (Phase 1 + Phase 2 overview)" width="95%">
  <br><em>Thesis overview. Phase 1 is the drug graph-transformer + protein + fusion + affinity head.
  Phase 2 is the heterogeneous EHR graph + DDI penalty + medication head.
  The released Phase 1 default uses 4 transformer layers rather than the 12 drawn.</em>
</p>

<p align="center">
  <img src="phase2/docs/figures/architecture_final.jpg" alt="Official GRAND architecture (Phase 2)" width="95%">
  <br><em>Phase 2 official architecture (AUIST report / thesis figure): EHR channels,
  meta-paths, semantic fusion, Dual-attention Graph Transformer + InfoNCE DDI branch.
  The released <code>hgdr</code> model is the EHR+DDI subset of this drawing
  (no genome channel, no InfoNCE DAGT head). See
  <a href="phase2/README.md#thesis-architecture-vs-released-implementation">the gap table</a>.</em>
</p>

Per-phase figures, ablation charts and training curves live in
[`phase1/docs/figures/`](phase1/docs/figures/),
[`phase1/results/figures/`](phase1/results/figures/),
[`phase2/docs/figures/`](phase2/docs/figures/) and
[`phase2/results/figures/`](phase2/results/figures/).

---

## Repository layout

```
.
├── README.md              ← this file
├── LICENSE                MIT
├── NOTICE                 attributions (GraphormerDTI, GAMENet/SafeDrug, DRecHGR, …)
├── CITATION.cff
├── CITATIONS.bib          both AUIST @mastersthesis entries
├── .gitignore             unions both phases + venvs / MIMIC / checkpoints
├── phase1/                package dti_gt  — KIBA DTI (ships compact data, ~6 MB)
│   ├── README.md, CHANGELOG.md, requirements.txt, environment.yml, pyproject.toml
│   ├── configs/           default.yaml (full KIBA) + ablation/*.yaml
│   ├── src/dti_gt/        data / models / utils
│   ├── scripts/           download, preprocess, train, evaluate, ablation, figures
│   ├── data/              compact KIBA + ProtBERT-BFD npz + versioned splits
│   ├── results/           11-variant × 2-seed ablation tables and figures
│   └── tests/             22 pytest cases
└── phase2/                package hgdr    — MIMIC-III GRAND/HGDR (ships mappings only, ~2 MB)
    ├── README.md, CHANGELOG.md, requirements.txt, environment.yml
    ├── configs/           default.yaml (full data, 30 epochs) + ablation_*.yaml
    ├── src/hgdr/          data / models / utils
    ├── scripts/           preprocess, graph build, train, ablation, demo_mode.ps1
    ├── data/mappings/     TWOSIDES CID pairs + PubChem name map (no patient data)
    ├── docs/              thesis_notes.md + official architecture_final.jpg
    ├── results/           12-variant × 2-seed ablation tables and figures
    └── tests/             34 pytest cases
```

Virtualenvs at `.venv_phase1` / `.venv_phase2`, MIMIC paths, checkpoints (`*.pt`),
and `phase2/data/processed/` are git-ignored and never pushed.

---

## Installation

The two phases have **separate** environments (different packages, same torch / PyG stack).
Tested on Windows 11 / Python 3.12.3 / PyTorch 2.5.1 + CUDA 12.1 / RDKit 2024.03.5 /
PyTorch Geometric 2.6.1. Linux and macOS use the same commands (swap the venv activate line).
No DGL and no compiled `torch_scatter` / `torch_sparse` wheels are required.

### Phase 1 (`dti_gt`)

```powershell
git clone https://github.com/BlackAsh01/graph-drug-discovery.git
cd graph-drug-discovery/phase1
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # Linux/macOS: source .venv/bin/activate
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .                      # optional
python -m pytest tests/               # 22 tests, ~20 s on CPU
```

### Phase 2 (`hgdr`)

```powershell
cd graph-drug-discovery/phase2
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python -m pytest -q                   # 34 tests, ~15–30 s on CPU
```

CPU-only: `pip install torch==2.5.1` (no extra index) then `requirements.txt`.
A conda alternative is in each phase's `environment.yml`.

---

## Quick start

### Phase 1 — train the full model on the 20 % KIBA subset (~5–7 min)

```powershell
cd phase1
python scripts/preprocess.py          # writes git-ignored drug_graphs.pt (~12 MB)
python scripts/train.py --config configs/ablation/full.yaml --seed 0
python scripts/evaluate.py --summarize results/runs
python scripts/make_figures.py
# optional 11 × 2 ablation (~1.8 h sequential):  python scripts/run_ablation.py --seeds 0 1
```

A 1 MB compact KIBA copy is already in `phase1/data/kiba/`; `download_kiba.py` is only needed
to refresh the raw TDC table. Full-KIBA training (`configs/default.yaml`, ≤ 40 epochs,
≈ 1.5–2 h on a 4 GB GPU) was **not** run for this release.

### Phase 2 — demo (no PhysioNet credentials, ~3–5 min)

```powershell
cd phase2
powershell -ExecutionPolicy Bypass -File scripts/demo_mode.ps1
```

Downloads the public MIMIC-III demo, preprocesses it, trains a smoke-test model and writes
git-ignored `results_demo/`. **Numbers on 100 patients are not meaningful.**

### Phase 2 — full pipeline on *your* MIMIC-III copy

```powershell
cd phase2
# 1. point at a local credentialed download (nothing is copied)
powershell -File scripts/link_local_data.ps1 -MimicDir "D:\path\to\mimic-iii-clinical-database-1.4"
python scripts/preprocess_mimic.py --mimic_dir data/raw/mimic-iii --out data/processed/mimic3
python scripts/build_hetero_graph.py --processed_dir data/processed/mimic3
# 2. subset protocol (matches the published ablation, ~5 min)
python scripts/train.py --config configs/ablation_base.yaml --seed 0
# 3. full-data 30-epoch run (~12 min on a GTX 1650 Ti; seed 0 completed)
python scripts/train.py --config configs/default.yaml --seed 0
```

See [`phase1/README.md`](phase1/README.md) and [`phase2/README.md`](phase2/README.md) for the
full command lists, config flags and ablation variants.

---

## Headline results

All numbers below are **test-set** metrics from the released ablation tables. They are
**not** comparable to published DeepDTA / GraphDTA / GAMENet / SafeDrug numbers: those
papers use full data, longer training, and (for Phase 2) a ≥ 2-visit / ATC-3 protocol.
Literature tables with that caveat are in
[`phase1/results/baseline_comparison.md`](phase1/results/baseline_comparison.md) and
[`phase2/results/baseline_comparison.md`](phase2/results/baseline_comparison.md).

### Phase 1 — KIBA 20 % subset, ≤ 12 epochs, seeds {0, 1}

Source: [`phase1/results/ablation_table.md`](phase1/results/ablation_table.md).

| Variant | MSE ↓ | CI ↑ | r_m² ↑ | What it shows |
|---|---|---|---|---|
| **full** (edge-aware GT, ProtBERT-BFD, cross-attention) | **0.4684 ± 0.0296** | **0.7343 ± 0.0088** | **0.3625 ± 0.0416** | 1.54 M params, 5.4 min/run, ~485 MB GPU |
| `no_residual_norm` | 0.6537 ± 0.0016 (**+40 %**) | 0.673 | 0.118 | Residuals + BatchNorm are required to train |
| `protein_composition` | 0.6028 ± 0.0146 (**+29 %**) | 0.639 | 0.184 | A bag-of-residues MLP is not a protein encoder |
| `protein_cnn` | 0.4550 ± 0.0078 | 0.746 | 0.383 | Learned CNN matches frozen ProtBERT, 2× slower |
| `backbone_gcn` | 0.4604 ± 0.0053 | 0.734 | 0.387 | 0.68 M GCN is competitive with the transformer |
| `no_edge_features` / `no_degree_encoding` / `fusion_concat` | ~0.45 | ~0.74 | ~0.37 | Graphormer extras / cross-attention did not help here |

**Takeaway.** Keep residuals/norm and a real protein encoder. A small GCN + concat fusion
is a strong, cheaper default on this 4 GB / 12-epoch / 20 % budget. *n* = 2 is not enough
to rank 1–4 % deltas (the two full-model seeds already differ by 0.06 MSE).

### Phase 2 — MIMIC-III 30 % patients, ≤ 15 epochs, seeds {0, 1}

Source: [`phase2/results/ablation_table.md`](phase2/results/ablation_table.md).
Cohort rebuilt locally: 39,239 patients / 50,085 admissions, 606 drugs, ground-truth DDI rate 0.087.
Subset protocol: 11,954 / 1,471 / 1,485 train/val/test admissions.

| Model | Jaccard ↑ | PR-AUC ↑ | F1 ↑ | DDI rate ↓ | #drugs |
|---|---|---|---|---|---|
| **HGDR (full)** | 0.3233 ± 0.0013 | 0.5621 ± 0.0016 | 0.4590 ± 0.0020 | **0.0779 ± 0.0020** | 15.07 |
| MLP on multi-hot codes | **0.3633 ± 0.0004** | **0.6261 ± 0.0012** | **0.5097 ± 0.0003** | 0.1003 ± 0.0006 | 15.15 |
| logistic regression on multi-hot codes | 0.3608 ± 0.0001 | 0.6224 ± 0.0001 | 0.5076 ± 0.0001 | 0.1000 ± 0.0000 | 14.78 |
| copy previous admission | 0.2100 | 0.4474 | 0.3348 | 0.0959 | 12.44 |
| *ground-truth prescriptions* | – | – | – | 0.0869 | 26.1 |

**Takeaway.** HGDR is the safest recommender (DDI 22 % below the multi-hot baselines and
below real prescriptions) but not the most accurate under this cap. Two reasons are
visible in the logs: (i) every `full` seed was still improving at epoch 15 (20-epoch
runs of the same config reached Jaccard 0.326–0.345); (ii) the DDI penalty costs about
+0.010 Jaccard for +0.032 DDI rate (`no_ddi_loss`). Graph message passing is the
largest HGDR-internal effect (`no_gnn` drops Jaccard by 0.064). The visit-history GRU
does not help on this mostly single-admission cohort.

### Phase 2 — full MIMIC-III, ≤ 30 epochs, seed 0

Source: [`phase2/results/fulldata_full_seed0_summary.md`](phase2/results/fulldata_full_seed0_summary.md).
Same 39,239-patient cohort, no 30 % subset. One seed. Early-stopped at epoch 24 (best 19), 12.3 min.

| Model | Jaccard ↑ | PR-AUC ↑ | F1 ↑ | DDI rate ↓ | #drugs |
|---|---|---|---|---|---|
| **HGDR (full data, seed 0)** | 0.3561 | 0.6033 | 0.4969 | **0.0724** | 15.54 |
| HGDR (30 % subset, 2-seed mean) | 0.3233 ± 0.0013 | 0.5621 ± 0.0016 | 0.4590 ± 0.0020 | 0.0779 ± 0.0020 | 15.07 |
| MLP (30 % subset, 2-seed mean) | **0.3633 ± 0.0004** | 0.6261 ± 0.0012 | 0.5097 ± 0.0003 | 0.1003 ± 0.0006 | 15.15 |

**Takeaway.** Full-data training improved HGDR Jaccard by about +0.033 and lowered DDI to 0.072
(still below real prescriptions). It did not overtake the published *subset* MLP (0.356 vs 0.363).
The MLP was not re-run on full data.

---

## Reproducibility and hardware

| | Phase 1 | Phase 2 |
|---|---|---|
| GPU used | NVIDIA GTX 1650 Ti **4 GB**, 16 GB RAM, Windows 11 | same (GPU sometimes shared) |
| Protocol | 20 % stratified KIBA, batch 128, ≤ 12 epochs, patience 4, 2 seeds | 30 % hash-selected patients, batch 256, ≤ 15 epochs, patience 4, 2 seeds |
| Wall-clock | 22 runs sequential, **1.8 h**; ~20–40 s/epoch; peak **~485 MB** | 24 runs sequential, **68 min**; 11–20 s/epoch idle; peak **~1.4 GB** |
| Seeds | `dti_gt.utils.set_seed` (Python / NumPy / PyTorch); CUDA scatter is not bit-deterministic | `hgdr.utils.set_seed`; same caveat — report mean ± std |
| Splits | Versioned index files in `phase1/data/splits/` | `md5(split_seed:subject_id)` — reproducible without shipping IDs |
| Checkpoints | `best.pt` git-ignored (~6 MB); regenerate with `train.py` | never shipped (trained on credentialed data) |
| Full-scale config | `phase1/configs/default.yaml` (94 k pairs, ≤ 40 epochs, ~1.5–2 h) — **not run** | `phase2/configs/default.yaml` (50 k admissions, ≤ 30 epochs) — **seed 0 done**, 12.3 min, Jaccard 0.3561 / DDI 0.0724 |

Phase 2 full-data write-up: [`phase2/results/fulldata_full_seed0_summary.md`](phase2/results/fulldata_full_seed0_summary.md).
Reproduce with `python scripts/train.py --config configs/default.yaml --seed 0`.

---

## Data and licensing

| Asset | In this repo? | Notes |
|---|---|---|
| Compact KIBA (`phase1/data/kiba/`) | yes, ~1 MB | Public benchmark (Tang et al. 2014 / TDC). |
| ProtBERT-BFD embeddings `[229, 1024]` | yes, 0.55 MB | Derived from Rostlab MIT weights. |
| MIMIC-III v1.4 (full) | **no** | PhysioNet credentialed DUA — **must not be redistributed**. |
| MIMIC-III demo | **no** | Downloaded by `demo_mode.ps1` (ODbL 1.0). |
| TWOSIDES CID pairs / PubChem map | yes, `phase2/data/mappings/` | No patient identifiers. |
| Checkpoints / `data/processed/` / venvs | **no** | Git-ignored. |

How to obtain MIMIC-III and rebuild Phase 2 artefacts:
[`phase2/data/README.md`](phase2/data/README.md).

---

## Citation

Please cite **both** AUIST project reports when using this repository.
University “Phase-I” labels on the PDFs are **not** the GitHub folder names:

- GitHub `phase1/` ↔ December 2024 AUIST report (DTI / graph transformer / KIBA)
- GitHub `phase2/` ↔ May 2025 GRAND report (the BibTeX `note` still says “AUIST Phase-I”)
- **University AUIST Phase-I report (GRAND) ↔ repository folder `phase2/`**

Machine-readable copies: [`CITATION.cff`](CITATION.cff), [`CITATIONS.bib`](CITATIONS.bib).
Please also cite the papers the architectures adapt (GraphormerDTI; GAMENet / SafeDrug; DrugDAGT).

```bibtex
@mastersthesis{prabhu2024dti,
  author   = {Ashwin Prabhu M},
  title    = {Graph Transformer Based Personalized Drug Recommendation
              System for Cardiovascular Disease},
  school   = {Department of Information Science and Technology,
              College of Engineering, Guindy, Anna University},
  year     = {2024},
  month    = dec,
  note     = {M.Tech. (Information Technology -- AI \& DS) AUIST Phase-I
              project report, December 2024. Register No. 2023176029.
              Supervisor: Dr. T. Mala. Cover title uses the CVD wording;
              the report body is KIBA drug--target interaction prediction
              with a graph transformer and ProtBERT (repository folder
              phase1/). PDF has no Keywords section; abstract terms:
              DTI, Graph Transformer, ProtBERT, KIBA, SMILES, RDKit,
              multi-head attention, binding affinity.},
  url      = {https://github.com/BlackAsh01/graph-drug-discovery}
}

@mastersthesis{prabhu2025grand,
  author  = {Ashwin Prabhu M},
  title   = {GRAND: Graph-based Recommendation with Attention Network
             for Personalized Drug Therapy},
  school  = {Department of Information Science and Technology,
             College of Engineering, Guindy, Anna University},
  year    = {2025},
  note    = {M.Tech. (Information Technology -- AI \& DS) AUIST Phase-I
             project report, Register No. 2023176029. Supervisor: Dr. T. Mala},
  url     = {https://github.com/BlackAsh01/graph-drug-discovery}
}

@article{gao2024graphormerdti,
  title   = {GraphormerDTI: A graph transformer-based approach for drug-target interaction prediction},
  author  = {Gao, Mengmeng and Zhang, Daokun and Chen, Yi and Zhang, Yiwen and Wang, Zhikang
             and Wang, Xiaoyu and Li, Shanshan and Guo, Yuming and Webb, Geoffrey I.
             and Nguyen, Anh T. N. and May, Lauren and Song, Jiangning},
  journal = {Computers in Biology and Medicine},
  volume  = {173},
  pages   = {108339},
  year    = {2024},
  doi     = {10.1016/j.compbiomed.2024.108339}
}

@inproceedings{shang2019gamenet,
  title     = {GAMENet: Graph Augmented MEmory Networks for Recommending Medication Combination},
  author    = {Shang, Junyuan and Xiao, Cao and Ma, Tengfei and Li, Hongyan and Sun, Jimeng},
  booktitle = {AAAI},
  year      = {2019}
}

@inproceedings{yang2021safedrug,
  title     = {SafeDrug: Dual Molecular Graph Encoders for Recommending Effective and Safe Drug Combinations},
  author    = {Yang, Chaoqi and Xiao, Cao and Ma, Fenglong and Glass, Lucas and Sun, Jimeng},
  booktitle = {IJCAI},
  year      = {2021}
}

@article{chen2024drugdagt,
  title   = {DrugDAGT: a dual-attention graph transformer with contrastive learning
             improves drug-drug interaction prediction},
  author  = {Chen, Yaojia and Wang, Jiacheng and Zou, Quan and Niu, Mengting
             and Ding, Yijie and Song, Jiangning and Wang, Yansu},
  journal = {BMC Biology},
  volume  = {22},
  pages   = {233},
  year    = {2024}
}
```

KIBA: Tang et al., *J. Chem. Inf. Model.* 2014. DeepDTA: Öztürk et al., *Bioinformatics* 2018.
ProtTrans: Elnaggar et al., *IEEE TPAMI* 2021. MIMIC-III: Johnson et al., *Sci. Data* 2016.
TWOSIDES: Tatonetti et al., *Sci. Transl. Med.* 2012.

---

## License

Source code is **MIT** — see [LICENSE](LICENSE) and the attribution file [NOTICE](NOTICE).

* KIBA / TDC tables: terms of the original publication and Therapeutics Data Commons.
* ProtBERT-BFD weights: MIT (Rostlab).
* MIMIC-III: PhysioNet DUA; **not redistributed**.
* TWOSIDES / PubChem derived tables: original open terms.

Adapted baselines (GraphormerDTI, GAMENet, SafeDrug, DRecHGR, DrugDAGT) are **not**
vendored. This tree is an independent PyG re-implementation with those papers cited.
