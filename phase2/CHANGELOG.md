# Changelog - from research notebooks to the release package

## 1.0.1 (2026-09-17) - full-data training results

- Completed `configs/default.yaml` seed 0 on all 50,085 admissions (early stop epoch 24,
  best epoch 19, 12.3 min). Test Jaccard 0.3561 / PR-AUC 0.6033 / F1 0.4969 / DDI 0.0724.
- Write-up in `results/fulldata_full_seed0_summary.md`; subset ablation runs are unchanged.

## 1.0.0 (2026-09) - first reproducible release

### What the original research code looked like

The Phase-2 folder was a flat collection of notebooks and copies
(`drug_recom*.ipynb`, `drug_rec_1strev.ipynb`, `dr*.ipynb`, `bew*.ipynb`, `dagt.ipynb`,
`drug_recommendation_deepseek.ipynb`, `data_display.ipynb`), a script
(`drug_interaction_model.py`), two vendored baselines (`DRecHGR/` - a DGL/HAN model;
`dagt/` - the DrugDAGT dual-attention graph transformer) and ~30 GB of intermediate
artefacts. Four different modelling ideas coexisted:

| notebook family | graph library | idea |
|---|---|---|
| `drug_recom*` / `dr*` | NetworkX -> DGL (`hetero_graph.bin`) | patient-diagnosis-drug heterogeneous graph, HAN/GAT encoder, per-patient drug ranking |
| `drug_rec_1strev` | PyTorch Geometric | LightGCN-style hetero propagation + BERT text features, top-k drug ranking |
| `bew*` | PyTorch Geometric | `LightGCNHetero` over patient/drug nodes, DDI graph from TWOSIDES, molecular features from PubChem SMILES |
| `dagt` | plain PyTorch | DrugDAGT (SMILES -> DDI type prediction) re-used for drug-pair safety scoring |

None of them ran end-to-end from a clean checkout: absolute Windows paths, DGL binaries
built for another torch version, 17 GB merged CSVs loaded with `pd.read_csv`, evaluation
at the *patient* level with test drugs leaking into the graph, and hard-coded PubChem
look-ups inside training loops.

### What changed

**Structure**
- Everything lives in an importable package `src/hgdr/` (`data/`, `models/`, `utils/`,
  `train.py`, `evaluate.py`, `config.py`) with thin CLI wrappers in `scripts/`.
- YAML configs with inheritance (`base:`) and dotted CLI overrides (`--set a.b=c`);
  one config per ablation variant.
- `set_seed()` everywhere, `logging` to `results/runs/<run>/train.log`, per-epoch
  `history.csv`, resolved `config.yaml` and `metrics.json` per run.

**Data pipeline (rewritten)**
- `preprocess_mimic.py` rebuilds admission-level records directly from the raw
  PRESCRIPTIONS / DIAGNOSES_ICD / PROCEDURE* tables in **chunks** (never loads the
  17 GB `merged_data_all_full.csv`); accepts flat, `.gz` and nested `X.csv/X.csv`
  layouts and case-insensitive headers.
- Drug names are normalised to active-ingredient keys (dose / formulation / salt /
  route / neonatal-prefix stripping, non-drug fluid filtering) instead of raw
  free-text strings; 606 keys cover >= 50 admissions each on the full data.
- Deterministic **patient-level** split by `md5(split_seed:subject_id)` - no ID lists
  are shipped, yet the split is bit-for-bit reproducible.
- Optional **subset protocol** (`data.subset_fraction`) - hash-selected patients,
  identical for every ablation variant.
- TWOSIDES is reduced to a CID-pair table (top-40 side effects, the SafeDrug
  convention) once, in chunks; the shipped file is 0.7 MB.
- PubChem mapping consolidated into one JSON, extended through the PUG-REST API
  (fixed: PubChem renamed `CanonicalSMILES` -> `SMILES` in 2025).
- Graph edges (diagnosis-drug, procedure-drug, drug-drug co-prescription) are built
  from **training** admissions only (no label leakage); DDI edges from TWOSIDES;
  molecular graphs from SMILES via RDKit.

**Model (consolidated and made ablatable)**
- Single `HGDRecommender`: entity embeddings (+ GAT/GIN molecular encoder for drug
  nodes) -> relation-aware `HeteroConv(SAGE|GAT)` message passing -> attention-pooled
  admission representation (+ GRU over previous admissions) -> dot-product scoring of
  all drugs. Every component is a config flag (`use_mol`, `gnn_type`, `gnn_layers`,
  `use_proc`, `use_history`, `use_ddi_edges`, `use_coprescription_edges`).
- DDI-awareness moved from a post-hoc filter into the loss (soft-DDI-rate penalty),
  which makes it ablatable and comparable with GAMENet/SafeDrug.
- **DGL -> PyTorch Geometric.** DGL wheels for torch 2.5 / CUDA 12.1 on Windows were
  unavailable; PyG 2.6.1 works with the pre-installed torch (the optional
  `torch_scatter` extension is deliberately *not* used - its wheel was ABI-incompatible).
- Cheap baselines that actually run (LR / MLP on multi-hot codes, copy-previous-visit).

**Evaluation**
- Admission-level multi-label metrics standard for this task (Jaccard, PR-AUC, F1,
  DDI rate, avg #drugs, following GAMENet/SafeDrug) *plus* the top-k ranking metrics
  the original notebooks reported (Precision@k, Jaccard@k, HitRate@k).
- Ground-truth DDI rate of the test prescriptions reported alongside.

**Experiment protocol**
- `scripts/run_ablation.py` runs the 12 variants x 2 seeds strictly one at a time (shared 4 GB
  GPU), seed-major, skips runs that already have a `metrics.json`, rewrites
  `results/ablation_results.csv` after every finished run and survives a crashed run, so the
  study can be launched as a detached process and resumed after an interruption
  (`--summary_only` rebuilds the table). Whole study: 24 runs / 68 min.
- Subset protocol fixed at 30 % of patients, <= 15 epochs, patience 4, 2 seeds (see README
  Section 6); the full-data `configs/default.yaml` run is documented in 1.0.1.

**Hygiene**
- No patient-level MIMIC-derived files, checkpoints or files > 10 MB in the release;
  `data/processed/**` (MIMIC-derived vocabularies / records, also the demo) is git-ignored and
  kept only locally.
- `tests/` (34 pytest cases, ~15 s CPU) cover normalisation, splits, graph building,
  every ablation flag's forward/backward pass, losses, metrics, preprocessing on a
  synthetic MIMIC-shaped sample and a two-epoch training loop.
- `scripts/demo_mode.ps1` runs the entire pipeline on the public MIMIC-III demo.

### Known limitations carried over / introduced
- Only the MIMIC-III tables present in the local research copy were used
  (`PROCEDURES_ICD` was absent, so procedures come from `PROCEDUREEVENTS_MV` item
  ids, which exist only for MetaVision-era admissions). The code prefers
  `PROCEDURES_ICD` automatically when it is available.
- Labels are normalised drug names (606), not ATC-3 classes (131) as in
  GAMENet/SafeDrug, so absolute numbers are not directly comparable to the literature
  (see `results/baseline_comparison.md`).
- The DRecHGR (DGL) and DrugDAGT baselines vendored in the research folder are
  acknowledged but not re-run.
