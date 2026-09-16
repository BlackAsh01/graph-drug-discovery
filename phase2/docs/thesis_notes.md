# AUIST report notes (maps to this package)

This page summarises the May 2025 AUIST Phase-I project report that this
repository's **Phase 2** (`hgdr`) implements a subset of. It is a reading
guide, not a substitute for the PDF.

**Report file:** `2023176029_AshwinPrabhuM_auistreport_PG_AI_DS_2024_.pdf`
(68 pages, LaTeX / pdfTeX, May 2025).

Academic “Phase-I report” here means the university project-report label on
the PDF. It is **not** the same split as this repo’s Phase 1 (KIBA DTI) vs
Phase 2 (medication recommendation). **University AUIST Phase-I report
(GRAND) ↔ repository folder `phase2/`.** The report is entirely about
DDI prediction and personalized drug recommendation; that work lives in
`phase2/`.

The earlier **December 2024** AUIST report (file
`2023176029_AshwinPrabhuM_auistreport_PG_AI_DS_2024.pdf`, 62 pages,
pdfTeX 2024-12-18, cover date DEC 2024) uses the CVD wording as its
**cover** title and is a KIBA DTI / graph-transformer study. That PDF
maps to repository **`phase1/`**, not this folder.

---

## Bibliographic facts

| Field | Value in the PDF |
|---|---|
| Cover title | **GRAND: Graph-based Recommendation with Attention Network for Personalized Drug Therapy** |
| Certificate title | Graph Transformer Based Personalized Drug Recommendation System for Cardiovascular Disease |
| Author | Ashwin Prabhu M |
| Register no. | 2023176029 |
| Degree | M.Tech. Information Technology (AI & DS) |
| Department | Information Science and Technology, College of Engineering, Guindy, Anna University, Chennai 600 025 |
| Guide | Dr. T. Mala, Professor, Dept. of IST |
| HOD | Dr. S. Swamynathan |
| Committee | Dr. S. Sridhar, Dr. G. Geetha, Dr. D. Narashiman |
| Date on cover | May 2025 |
| Self-description | “A report for the phase-I of the project” |

The cover title is treated as official. The certificate title is a
cardiovascular-disease wording of the same project and is recorded here
only as an alternate.

---

## Problem and claimed method

**Problem (Ch. 1).** Critical-care polypharmacy is hard to manage from EHR
data alone. Existing tools do not jointly forecast harmful drug–drug
interactions and recommend a safer, patient-specific regimen.

**Proposed system (Ch. 3–4 + official architecture figure).** Two coupled
branches:

1. **EHR heterogeneous network** over patients, diagnoses, medications
   (and, in the official figure, genomes). Typed GCN channels
   (diagnose / genome / medication, LeakyReLU) produce embeddings
   \(E^{(d)}, E^{(v)}, E^{(m)}\). HAN-style **meta-paths**
   **P–D–P**, **P–M–P**, **M–P–M**, and (figure only) **M–G–M** feed
   multi-head node-projection attention and **semantic fusion**
   (Linear → Tanh → Linear) to patient / medication vectors
   \(Z^{(p)}, Z^{(m)}\). A recommendation head is trained with the
   figure’s \(\mathcal{L}_1\).
2. **Molecular Dual-attention Graph Transformer** (DrugDAGT-style): each
   drug SMILES becomes an RDKit graph; intra-graph and inter-graph
   attention (and, in the figure, **InfoNCE**) produce a pairwise
   **DDI score** through an FFN.

The report text also describes a BERT clinical-notes encoder, LightGCN
on a patient–diagnosis–drug DGL heterograph, a local substitute table
plus FDA / DailyMed fallbacks, and a ranking score
\(\mathrm{cosine}(\tilde h_p, h_{dr}) - \lambda\,\rho_{dr}\).
Those pieces sit beside the official figure rather than replacing it.

**Data named in the report.** MIMIC-III (notes, demographics, ICD
diagnoses, prescriptions), TWOSIDES (DDI ground truth), RDKit / PubChem
SMILES, optional FDA and DailyMed APIs. Genomes appear in the figure
only; MIMIC-III does not ship a genome table used here.

---

## Results the report claims (not the released ablation)

These numbers are from report Tables 4.1–4.2. They evaluate a
**pairwise DDI regressor** and a **substitute-retrieval** pipeline, not
the GAMENet / SafeDrug multi-label admission task used in `results/`.

**Table 4.1 — Dual-attention Graph Transformer (DDI probability).**

| Metric | Value |
|---|---|
| Validation loss (BCE) | 0.0835 |
| Validation MAE | 0.2506 |
| Test loss | 0.0815 |
| Test MAE | 0.2468 |

**Table 4.2 — Recommendation framework vs curated substitutes.**

| Metric | Value |
|---|---|
| Cohen’s κ (top-1 vs substitute) | 0.693 |
| Top-1 / Top-2 / Top-3 hit rate | 92.9% / 93.1% / 93.1% |
| Weighted κ (linear / quadratic) | 0.974 / 0.972 |
| Multi-label Jaccard vs substitutes | 0.967 |

Hardware named in the report: MSI GF75 Thin, GTX 1660 Ti 6 GB;
DAGT grid search settled on hidden 128, 4 heads, 2 layers, lr \(5\times10^{-4}\).
LightGCN: 64 channels, 3 layers.

**Do not compare Table 4.2’s Jaccard 0.967 to the released HGDR Jaccard
0.356.** They are different tasks, label spaces, and splits.

The numbers this repo actually trained and published (30% subset and
full-data GAMENet-style metrics) are in
[`../README.md`](../README.md#5-results) and
[`../results/`](../results/).

---

## Conclusions and future work (Ch. 5)

The report concludes that stacking GATConv molecular encoders, BERT
context, and a substitute-filtered ranker can produce calibrated pairwise
DDI scores and high overlap with expert substitutes. Limitations it
lists: retrospective MIMIC-III only; pairwise (not n-ary) DDI; cost of
large heterographs and long notes; ICU vs outpatient shift.

Future work it names: prospective clinical validation; hypergraph /
tensor models for higher-order interactions; temporal EHR; extra
modalities including **patient genomics**; interpretability
(Integrated Gradients, rule extraction); distillation / federated
learning; UMLS / DrugBank knowledge graphs.

---

## How this release maps to the report

| Report | In this package |
|---|---|
| Ch. 1 problem / objectives | Phase 2 README §1 (admission-level multi-label formulation actually trained) |
| Ch. 2 literature (GAMENet-adjacent graphs, DrugDAGT, Trans-GAHNet, …) | `results/baseline_comparison.md`; NOTICE |
| Ch. 3.1 MIMIC-III + TWOSIDES + PubChem | `scripts/preprocess_mimic.py`, `data/mappings/` |
| Ch. 3.1.3 FDA / DailyMed APIs | **not shipped** |
| Ch. 3.2.1 Clinical notes NLP / BERT | **not shipped** |
| Official figure: diagnose / medication channels, P–D–P / P–M–P / M–P–M | Approximated by the diag–proc–drug `HeteroConv` entity graph (no patient nodes, no explicit HAN meta-path graphs) |
| Official figure: genome channel, M–G–M | **Thesis-proposed only** — no genome data in the MIMIC copy used here |
| Official figure: semantic fusion Linear–Tanh–Linear | Same MLP shape is used as **admission-code attention pooling**, not HAN semantic attention over meta-path embeddings |
| Ch. 3.2.3 / 4.2 Dual-attention GT + InfoNCE DDI | **not shipped**; `MolEncoder` is a GAT/GIN stand-in; DDI is a soft-rate penalty, not pairwise InfoNCE |
| Ch. 4.3 LightGCN + substitute ranker | **not shipped**; scoring is \(qK^\top/\sqrt{d}\) over the drug vocabulary |
| Ch. 4.5 Tables 4.1–4.2 | Cited above; **not reproduced** in `results/` |
| Released ablation / full-data Jaccard | New protocol, documented in the Phase 2 README |

A component-by-component gap list is in the Phase 2 README section
**Thesis architecture vs. released implementation**.

---

## Acknowledgements and citations (from the report)

Acknowledgements: Dr. T. Mala; Dr. S. Swamynathan; committee members
Dr. S. Sridhar, Dr. G. Geetha, Dr. D. Narashiman; IST faculty and staff
at CEG, Anna University.

Selected references the report uses: Bhoi et al. 2021 (PREMIER);
Zhang et al. 2023/24 (heterogeneous EHR graphs); Li et al. 2024
(Trans-GAHNet); Mulyadi & Suk (KindMed); Chen et al. 2024 (**DrugDAGT**,
dual-attention graph transformer + contrastive learning); He et al. 2024;
Kuang & Xie 2024 (DrugDoctor). GAMENet / SafeDrug are the evaluation
convention this *release* follows; they are cited in NOTICE even though
the May 2025 report’s Table 4.2 uses a substitute-overlap protocol.
