# Changelog — from research notebooks to release

## 1.0.0 — first public release

Source material: `Phase1/scripts/*.ipynb` (nine notebooks, Sep–Oct 2024), `Phase1/scripts/*.py` (copies of
GraphormerDTI's DGL modules) and the GraphormerDTI baseline repository. The latest complete pipeline was the
`new_kiba*.ipynb` family (TDC KIBA → RDKit graphs → ProtBERT embeddings → graph network + MLP regression).

### Bugs found in the original code and fixed here

1. **Degenerate protein embeddings (critical).** Sequences were fed to ProtBERT as one unbroken string
   (`"MTVKTEAAK…"`). ProtBERT's vocabulary is single space-separated residues, so every sequence tokenised to
   `[UNK]`s and *all 229 targets received the identical 1024-d vector* (verified: per-dimension std across the
   229 saved tensors is exactly 0). The protein branch therefore carried no information, and the notebook's
   reported test MSE of 0.70 equals the variance of the KIBA score (0.834² = 0.70) — i.e. the model predicted the
   mean. `scripts/embed_proteins.py` now formats sequences correctly (`"M T V K …"`, rare residues → `X`, mean over
   residue tokens only) and the new embeddings have a per-dimension std of 0.023 across targets.
2. **Unusable DGL graph transformer on Windows.** `graph_transformer_net.py` / `graph_transformer_edge_layer.py`
   required DGL, whose `dgl.dll` failed to load in the original environment; the notebooks fell back to a
   `12 × GCNConv` stack with a post-hoc attention layer that indexed `Q[edge_index[0]] · K[edge_index[1]]` and
   then returned a per-*edge* tensor (`drug_features shape [1894,128]` vs `batch [861]` → "Mismatch … adjusting"
   hack). The edge-aware graph transformer is re-implemented in pure PyTorch Geometric
   (`src/dti_gt/models/graph_transformer.py`), including the bond-feature attention bias, edge channel, degree
   (centrality) encoding, residuals and BatchNorm of GraphormerDTI, with a proper per-destination softmax.
3. **Raw numeric atom features.** Atomic number, hybridisation enum, etc. were cast to `float` and fed to a
   linear layer. They are now categorical indices embedded per attribute (OGB/Graphormer style).
4. **`loss.backward()` inside `torch.no_grad()` evaluation**, evaluation with `shuffle=True`, MSE computed on the
   first batch only in some cells, training loops without validation/early stopping, hard-coded Colab / Windows
   paths, `eval()`-based deserialisation of graph strings, and a 1 GB pickled DataFrame as the data format —
   all removed. Data are stored as a 1 MB compact CSV set + cached tensors; there is a fixed, versioned split.
5. **No metrics beyond MSE.** Added CI, r_m², Pearson/Spearman, RMSE/MAE and AUROC/AUPRC at the 12.1 threshold
   (`src/dti_gt/utils/metrics.py`, unit-tested against a naive O(n²) implementation).
6. **Original `.py` modules were unused copies** of GraphormerDTI (`model.py`, `hyperparameter.py`,
   `drugparameter.py`, `process_data.py` for PDB files); `code_final.ipynb` was a PubChem-bioassay data-wrangling
   experiment that never produced a training set (its final protein/compound merge was empty). Not carried over.

### Engineering changes

* Package layout `src/dti_gt/{data,models,utils}`, config-driven (`configs/*.yaml` with inheritance and
  `key=value` overrides), argparse CLIs in `scripts/`, pytest smoke tests (22 tests, ~20 s on CPU).
* Deterministic seeding, per-run artefacts (`config.yaml`, `history.csv`, `metrics.json`, `test_predictions.csv`,
  `train.log`), early stopping on validation MSE, ReduceLROnPlateau, gradient clipping, fused AdamW, automatic
  CPU fallback on CUDA OOM, target standardisation (metrics always reported in KIBA units).
* Every architectural component is a config flag → the ablation study in `results/` (`scripts/run_ablation.py`,
  resumable, parallelisable across processes).
* No DGL, no torch-scatter/torch-sparse binaries needed (PyG 2.6 fallbacks), works with torch 2.5 / RDKit 2024.
