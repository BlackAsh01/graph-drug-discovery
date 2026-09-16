# Data

## What is versioned here (all files < 1 MB)

| Path | Content | Size |
|---|---|---|
| `kiba/pairs.csv.gz`, `kiba/drugs.csv`, `kiba/targets.csv` | **Compact copy of the full KIBA benchmark** (117,657 drug–target pairs, 2,068 drugs, 229 targets) in normalised form: pair table (`drug_id,target_id,y`), unique SMILES, unique sequences. Written by `scripts/preprocess.py` from the raw TDC table and read automatically when `data/raw/kiba.tab` is absent. | ~0.9 MB |
| `protein_embeddings/kiba_prot_bert_bfd.npz` | Frozen **ProtBERT-BFD** (`Rostlab/prot_bert_bfd`) mean-pooled embeddings, `[229, 1024]` float32, one row per KIBA target (`target_ids` array gives the order). Produced by `scripts/embed_proteins.py`; regenerating them requires the 1.7 GB model download. | 0.55 MB |
| `splits/kiba_random_seed42/` | The fixed **full-data split** (train/val/test = 94,125 / 11,766 / 11,766 row indices into `pairs.csv.gz` / `kiba.tab`, seed 42). | 0.8 MB |
| `splits/kiba_subset20_seed42/` | The fixed **ablation subset split**: a 20 % subset (23,532 pairs) stratified over 10 quantile bins of `y`, then split 80/10/10 (18,826 / 2,353 / 2,353), seed 42. | 0.2 MB |
| `sample/kiba_sample.tab` | 200 random rows of the raw TDC table, used by the unit tests. | 0.15 MB |

Git-ignored (recreated by the scripts): `raw/kiba.tab` (96 MB), `processed/drug_graphs.pt` (featurised molecular graphs cache).

## KIBA

KIBA (Tang et al., *J. Chem. Inf. Model.* 2014) integrates Ki, Kd and IC50 measurements of kinase inhibitors
into a single *KIBA score*; the benchmark version used by DeepDTA (Öztürk et al. 2018) and virtually all
later DTA papers keeps drugs with ≥ 10 interactions and applies the transform `y = -log10(KIBA/1e9)`-style
rescaling so that **higher score = stronger binding** (range 0.0–17.2, mean 11.72, std 0.83; the conventional
binarisation threshold for "interacting" is 12.1). We use the Therapeutics Data Commons (TDC) copy,
`DTI(name="KIBA")`, which contains 117,657 pairs after TDC's de-duplication.

Columns of the raw TDC file (`ID1, X1, ID2, X2, Y`): ChEMBL drug id, SMILES, UniProt target id, amino-acid
sequence, KIBA score. The loaders also accept TDC's `Drug_ID, Drug, Target_ID, Target, Y` naming.

## Getting the raw file

```powershell
# option A - download the TDC copy (Harvard Dataverse, ~96 MB, no extra dependencies)
python scripts/download_kiba.py

# option B - with PyTDC installed
python -c "from tdc.multi_pred import DTI; DTI(name='KIBA', path='data/raw')"   # then rename kiba.tab if needed

# option C - copy an existing download (author's machine: D:\CLG\AU\projects\Phase1\scripts\data\kiba.tab)
powershell -ExecutionPolicy Bypass -File scripts\link_local_data.ps1
python scripts/download_kiba.py --from-local path\to\kiba.tab
```

Then `python scripts/preprocess.py` rebuilds the compact copy, the graph cache and the splits
(identical to the versioned ones for seed 42).

## Preprocessing summary

* **Drugs** – RDKit parses each SMILES (all 2,068 parse). Atoms become 8 categorical features
  (atomic number, degree, formal charge, #H, hybridisation, aromatic, in-ring, chirality), bonds become
  3 categorical features (type, conjugated, in-ring); edges are stored in both directions, degree is kept as a
  separate tensor for the centrality encoding. Mean 28.7 atoms, max 268.
* **Proteins** – three interchangeable encodings (see `src/dti_gt/data/proteins.py`): frozen ProtBERT-BFD
  embeddings (default), integer residue tokens truncated/padded to 1,000 for the CNN encoder, or
  amino-acid composition vectors. ProtBERT input is space-separated with rare residues mapped to `X`.
* **Splits** – random *pair-level* (transductive / "warm") split; drugs and targets can appear in every split,
  as in DeepDTA/GraphDTA. Cold-drug / cold-target splits are not part of Phase 1.
