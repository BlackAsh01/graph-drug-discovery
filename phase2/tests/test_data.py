"""Data utilities: drug-name normalisation, deterministic splits, graph construction, SMILES parsing,
and preprocessing on a synthetic 20-admission MIMIC-shaped sample."""

import csv
from pathlib import Path

import numpy as np
import pytest
import torch

from hgdr.data.drug_mapping import cid_to_twosides_id, normalize_drug_name, twosides_id_to_cid
from hgdr.data.graph import smiles_to_graph
from hgdr.data.mimic import preprocess_mimic
from hgdr.data.splits import split_records
from hgdr.data.ddi import build_ddi_adjacency
from tests.conftest import N_DIAG, N_DRUG, N_PROC


@pytest.mark.parametrize("raw, expected", [
    ("Metoprolol Tartrate", "METOPROLOL"),
    ("Heparin Flush (10 units/ml)", "HEPARIN"),
    ("Potassium Chloride", "POTASSIUM CHLORIDE"),
    ("Insulin Human Regular", "INSULIN"),
    ("NEO*IV*Gentamicin", "GENTAMICIN"),
    ("Aspirin EC 81mg", "ASPIRIN"),
    ("Magnesium Sulfate Replacement (Oncology)", "MAGNESIUM SULFATE"),
    ("0.9% Sodium Chloride", "SODIUM CHLORIDE"),
])
def test_normalize_drug_name(raw, expected):
    assert normalize_drug_name(raw) == expected


def test_twosides_id_roundtrip():
    assert twosides_id_to_cid(cid_to_twosides_id(2244)) == 2244
    assert twosides_id_to_cid("CID000002244") == 2244


def test_split_is_deterministic_and_disjoint(records):
    a = split_records(records, [0.6, 0.2, 0.2], split_seed=0)
    b = split_records(records, [0.6, 0.2, 0.2], split_seed=0)
    assert [len(x) for x in a] == [len(x) for x in b]
    assert a[0] == b[0] and a[2] == b[2]
    assert sum(len(x) for x in a) == len(records)
    ids = [set(p[0]["subject_id"] for p in part) for part in a]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    c = split_records(records, [0.6, 0.2, 0.2], split_seed=1)
    assert a[0] != c[0]


def test_graph_structure(graph):
    assert graph.num_nodes == {"diag": N_DIAG, "proc": N_PROC, "drug": N_DRUG}
    assert ("diag", "treated_by", "drug") in graph.edge_index
    assert ("drug", "interacts", "drug") in graph.edge_index
    assert graph.ddi_adj.shape == (N_DRUG, N_DRUG)
    assert torch.equal(graph.ddi_adj, graph.ddi_adj.T)
    assert int(graph.has_mol().sum()) == N_DRUG - 3
    for k, ei in graph.edge_index.items():
        assert ei.shape[0] == 2
        assert ei[0].max() < graph.num_nodes[k[0]] and ei[1].max() < graph.num_nodes[k[2]]


def test_smiles_to_graph():
    x, ei = smiles_to_graph("CC(=O)Oc1ccccc1C(=O)O")  # aspirin: 13 heavy atoms, 13 bonds
    assert x.shape[0] == 13 and ei.shape == (2, 26)
    assert smiles_to_graph("not a smiles") is None


def test_ddi_adjacency_from_pairs():
    import pandas as pd
    pairs = pd.DataFrame({"cid1": [10, 20], "cid2": [20, 30]})
    words = ["A", "B", "C", "D"]
    adj = build_ddi_adjacency(words, {"A": 10, "B": 20, "D": 30}, pairs)
    assert adj.shape == (4, 4) and adj[0, 1] == 1 and adj[1, 0] == 1 and adj[1, 3] == 1 and adj[0, 3] == 0
    assert adj[2].sum() == 0  # unmapped drug has no DDI edges


def _write_csv(path: Path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def test_preprocess_synthetic_mimic(tmp_path: Path):
    """20 admissions / 12 subjects shaped like MIMIC-III tables (nested X.csv/X.csv layout)."""
    rng = np.random.default_rng(0)
    drugs = ["Heparin", "Aspirin EC", "Metoprolol Tartrate", "Insulin Human Regular", "Furosemide",
             "Potassium Chloride", "Acetaminophen", "Docusate Sodium", "Vancomycin", "Pantoprazole"]
    icd = ["4019", "42731", "5849", "2724", "51881", "486", "2859", "25000"]
    pres, diag, proc = [], [], []
    hadm = 100
    for subj in range(12):
        for _ in range(1 + (subj % 3 == 0) + (subj % 4 == 0)):
            hadm += 1
            day = f"2150-01-{(hadm % 27) + 1:02d} 00:00:00"
            for d in rng.choice(drugs, size=5, replace=False):
                pres.append([len(pres) + 1, subj, hadm, "", day, day, "MAIN", d, d, d, "", "", "", "", "", "", "", "", "PO"])
            for i, c in enumerate(rng.choice(icd, size=3, replace=False)):
                diag.append([len(diag) + 1, subj, hadm, i + 1, c])
            proc.append([len(proc) + 1, subj, hadm, "", day, day, int(rng.integers(224000, 224005)), 1, "", ""])
    (tmp_path / "PRESCRIPTIONS.csv").mkdir()
    _write_csv(tmp_path / "PRESCRIPTIONS.csv" / "PRESCRIPTIONS.csv",
               ["ROW_ID", "SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "STARTDATE", "ENDDATE", "DRUG_TYPE", "DRUG",
                "DRUG_NAME_POE", "DRUG_NAME_GENERIC", "FORMULARY_DRUG_CD", "GSN", "NDC", "PROD_STRENGTH",
                "DOSE_VAL_RX", "DOSE_UNIT_RX", "FORM_VAL_DISP", "FORM_UNIT_DISP", "ROUTE"], pres)
    _write_csv(tmp_path / "DIAGNOSES_ICD.csv", ["ROW_ID", "SUBJECT_ID", "HADM_ID", "SEQ_NUM", "ICD9_CODE"], diag)
    _write_csv(tmp_path / "PROCEDUREEVENTS_MV.csv",
               ["ROW_ID", "SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "STARTTIME", "ENDTIME", "ITEMID", "VALUE",
                "VALUEUOM", "LOCATION"], proc)
    data = preprocess_mimic(tmp_path, min_drug_admissions=1, min_diag_count=1, min_proc_count=1, verbose=False)
    n_visits = sum(len(p) for p in data.records)
    assert n_visits == hadm - 100 == 19
    assert len(data.records) == 12
    assert len(data.drug_voc) == len(drugs)
    assert "METOPROLOL" in data.drug_voc.word2idx and "ASPIRIN" in data.drug_voc.word2idx
    assert len(data.diag_voc) == len(icd)
    for p in data.records:
        sids = {v["subject_id"] for v in p}
        assert len(sids) == 1  # one patient per record list; visits ordered chronologically
        for v in p:
            assert len(v["drug"]) == 5 and len(v["diag"]) == 3 and len(v["proc"]) >= 1
    out = tmp_path / "out"
    data.save(out)
    assert (out / "records.pkl").exists() and (out / "voc.json").exists()
