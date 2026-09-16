"""MIMIC-III preprocessing into admission-level records + vocabularies.

The output follows the *records / voc* convention popularised by
GAMENet and SafeDrug: ``records`` is a list of patients, each a list of
visits (admissions, chronologically ordered), each visit a dict with
integer-coded ``diag``, ``proc`` and ``drug`` lists.  Nothing in this
module depends on a fixed column case or on the raw tables being flat
files – both ``<dir>/PRESCRIPTIONS.csv`` and the nested
``<dir>/PRESCRIPTIONS.csv/PRESCRIPTIONS.csv`` layout (as produced by some
un-zippers) are supported, as are ``.csv.gz`` files.

Only the columns needed are read, in chunks, so the 735 MB
``PRESCRIPTIONS.csv`` never has to fit in memory at once.
"""

from __future__ import annotations

import gzip
import json
import pickle
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .drug_mapping import is_non_drug, normalize_drug_name


# --------------------------------------------------------------------------- #
# File helpers
# --------------------------------------------------------------------------- #
def resolve_table(mimic_dir: str | Path, table: str) -> Optional[Path]:
    """Locate ``TABLE.csv`` in a MIMIC-III directory (case-insensitive).

    Accepts flat files, ``.csv.gz`` files and the nested ``X.csv/X.csv``
    layout.  Returns ``None`` when the table is absent.
    """
    mimic_dir = Path(mimic_dir)
    candidates = [
        mimic_dir / f"{table}.csv",
        mimic_dir / f"{table}.csv.gz",
        mimic_dir / f"{table}.csv" / f"{table}.csv",
        mimic_dir / table.lower() / f"{table.lower()}.csv",
        mimic_dir / f"{table.lower()}.csv",
        mimic_dir / f"{table.lower()}.csv.gz",
    ]
    for c in candidates:
        if c.is_file():
            return c
    # case-insensitive scan
    if mimic_dir.is_dir():
        for p in mimic_dir.iterdir():
            if p.name.lower() in (f"{table.lower()}.csv", f"{table.lower()}.csv.gz"):
                if p.is_file():
                    return p
                inner = p / p.name
                if inner.is_file():
                    return inner
    return None


def read_table(path: Path, usecols: Sequence[str], chunksize: int = 500_000,
               dtype: Optional[dict] = None) -> pd.DataFrame:
    """Read selected columns of a (possibly huge) CSV, chunked, upper-casing headers."""
    usecols_up = [c.upper() for c in usecols]
    header = pd.read_csv(path, nrows=0)
    colmap = {c.upper(): c for c in header.columns}
    missing = [c for c in usecols_up if c not in colmap]
    if missing:
        raise KeyError(f"{path.name}: missing columns {missing}; has {list(header.columns)}")
    real_cols = [colmap[c] for c in usecols_up]
    dtype_real = {colmap[k.upper()]: v for k, v in (dtype or {}).items() if k.upper() in colmap}
    chunks = []
    for chunk in pd.read_csv(path, usecols=real_cols, chunksize=chunksize, dtype=dtype_real,
                             low_memory=False):
        chunk.columns = [c.upper() for c in chunk.columns]
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #
@dataclass
class Voc:
    """Bidirectional token <-> index vocabulary."""

    idx2word: List[str] = field(default_factory=list)
    word2idx: Dict[str, int] = field(default_factory=dict)

    def add(self, word: str) -> int:
        if word not in self.word2idx:
            self.word2idx[word] = len(self.idx2word)
            self.idx2word.append(word)
        return self.word2idx[word]

    def __len__(self) -> int:
        return len(self.idx2word)

    def to_dict(self) -> Dict[str, List[str]]:
        return {"idx2word": list(self.idx2word)}

    @classmethod
    def from_dict(cls, d: Dict[str, List[str]]) -> "Voc":
        v = cls()
        for w in d["idx2word"]:
            v.add(w)
        return v


@dataclass
class ProcessedData:
    """Container returned by :func:`preprocess_mimic`."""

    records: List[List[Dict[str, object]]]
    diag_voc: Voc
    proc_voc: Voc
    drug_voc: Voc
    drug_raw_names: Dict[str, List[str]]  # normalised key -> raw strings seen
    stats: Dict[str, object]

    def save(self, out_dir: str | Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "records.pkl", "wb") as f:
            pickle.dump(self.records, f)
        with open(out_dir / "voc.json", "w", encoding="utf-8") as f:
            json.dump({"diag_voc": self.diag_voc.to_dict(), "proc_voc": self.proc_voc.to_dict(),
                       "drug_voc": self.drug_voc.to_dict()}, f, indent=1)
        with open(out_dir / "drug_raw_names.json", "w", encoding="utf-8") as f:
            json.dump(self.drug_raw_names, f, indent=1)
        with open(out_dir / "stats.json", "w", encoding="utf-8") as f:
            json.dump(self.stats, f, indent=2)

    @classmethod
    def load(cls, out_dir: str | Path) -> "ProcessedData":
        out_dir = Path(out_dir)
        with open(out_dir / "records.pkl", "rb") as f:
            records = pickle.load(f)
        with open(out_dir / "voc.json", "r", encoding="utf-8") as f:
            v = json.load(f)
        raw_path = out_dir / "drug_raw_names.json"
        raw = json.load(open(raw_path, "r", encoding="utf-8")) if raw_path.exists() else {}
        stats_path = out_dir / "stats.json"
        stats = json.load(open(stats_path, "r", encoding="utf-8")) if stats_path.exists() else {}
        return cls(records, Voc.from_dict(v["diag_voc"]), Voc.from_dict(v["proc_voc"]),
                   Voc.from_dict(v["drug_voc"]), raw, stats)


# --------------------------------------------------------------------------- #
# Core preprocessing
# --------------------------------------------------------------------------- #
def load_prescriptions(mimic_dir: Path, main_only: bool = True) -> pd.DataFrame:
    """Load ``(SUBJECT_ID, HADM_ID, DRUG_KEY, STARTDATE)`` prescription rows."""
    path = resolve_table(mimic_dir, "PRESCRIPTIONS")
    if path is None:
        raise FileNotFoundError(f"PRESCRIPTIONS.csv not found under {mimic_dir}")
    df = read_table(path, ["SUBJECT_ID", "HADM_ID", "DRUG_TYPE", "DRUG", "STARTDATE"],
                    dtype={"DRUG": str, "DRUG_TYPE": str, "STARTDATE": str})
    df = df.dropna(subset=["SUBJECT_ID", "HADM_ID", "DRUG"])
    if main_only and "DRUG_TYPE" in df:
        df = df[df["DRUG_TYPE"].astype(str).str.upper() == "MAIN"]
    df["SUBJECT_ID"] = df["SUBJECT_ID"].astype(np.int64)
    df["HADM_ID"] = df["HADM_ID"].astype(np.int64)
    uniq = pd.Series(df["DRUG"].unique())
    key_map = dict(zip(uniq, uniq.map(normalize_drug_name)))
    df["DRUG_KEY"] = df["DRUG"].map(key_map)
    df = df[~df["DRUG_KEY"].map(is_non_drug)]
    return df[["SUBJECT_ID", "HADM_ID", "DRUG", "DRUG_KEY", "STARTDATE"]]


def load_codes(mimic_dir: Path, table: str, code_col: str) -> Optional[pd.DataFrame]:
    """Load ``(SUBJECT_ID, HADM_ID, CODE)`` from an ICD / event table, or None if absent."""
    path = resolve_table(mimic_dir, table)
    if path is None:
        return None
    df = read_table(path, ["SUBJECT_ID", "HADM_ID", code_col], dtype={code_col: str})
    df = df.dropna(subset=["SUBJECT_ID", "HADM_ID", code_col])
    df["SUBJECT_ID"] = df["SUBJECT_ID"].astype(np.int64)
    df["HADM_ID"] = df["HADM_ID"].astype(np.int64)
    df["CODE"] = df[code_col].astype(str).str.strip()
    return df[["SUBJECT_ID", "HADM_ID", "CODE"]].drop_duplicates()


def load_procedures(mimic_dir: Path) -> tuple[Optional[pd.DataFrame], str]:
    """Procedures from ``PROCEDURES_ICD`` (preferred) or ``PROCEDUREEVENTS_MV`` (ITEMID)."""
    df = load_codes(mimic_dir, "PROCEDURES_ICD", "ICD9_CODE")
    if df is not None:
        return df, "PROCEDURES_ICD.ICD9_CODE"
    df = load_codes(mimic_dir, "PROCEDUREEVENTS_MV", "ITEMID")
    if df is not None:
        df["CODE"] = "MV_" + df["CODE"].str.replace(r"\.0$", "", regex=True)
        return df, "PROCEDUREEVENTS_MV.ITEMID"
    return None, "none"


def _admission_order(pres: pd.DataFrame, mimic_dir: Path) -> pd.DataFrame:
    """Return ``HADM_ID -> (SUBJECT_ID, ADMIT_KEY)`` used to order visits.

    Uses ``ADMISSIONS.ADMITTIME`` when the table exists, otherwise the
    earliest prescription ``STARTDATE`` of the admission.
    """
    adm_path = resolve_table(mimic_dir, "ADMISSIONS")
    if adm_path is not None:
        adm = read_table(adm_path, ["SUBJECT_ID", "HADM_ID", "ADMITTIME"], dtype={"ADMITTIME": str})
        adm["ADMIT_KEY"] = pd.to_datetime(adm["ADMITTIME"], errors="coerce")
        adm["SUBJECT_ID"] = adm["SUBJECT_ID"].astype(np.int64)
        adm["HADM_ID"] = adm["HADM_ID"].astype(np.int64)
        return adm[["SUBJECT_ID", "HADM_ID", "ADMIT_KEY"]].drop_duplicates("HADM_ID")
    tmp = pres[["SUBJECT_ID", "HADM_ID", "STARTDATE"]].copy()
    tmp["ADMIT_KEY"] = pd.to_datetime(tmp["STARTDATE"], errors="coerce")
    tmp = tmp.groupby(["SUBJECT_ID", "HADM_ID"], as_index=False)["ADMIT_KEY"].min()
    return tmp


def preprocess_mimic(
    mimic_dir: str | Path,
    min_drug_admissions: int = 50,
    min_diag_count: int = 5,
    min_proc_count: int = 5,
    max_drugs: Optional[int] = None,
    use_procedures: bool = True,
    max_visits_per_patient: int = 100,
    max_admissions: Optional[int] = None,
    subset_seed: int = 0,
    verbose: bool = True,
) -> ProcessedData:
    """Build admission-level records from raw MIMIC-III tables.

    Args:
        mimic_dir: Directory with ``PRESCRIPTIONS``, ``DIAGNOSES_ICD`` and
            optionally ``PROCEDURES_ICD`` / ``PROCEDUREEVENTS_MV`` /
            ``ADMISSIONS``.
        min_drug_admissions: Keep drug keys prescribed in at least this many
            admissions (reduces the ~4k noisy free-text names to a
            recommendable vocabulary).
        min_diag_count / min_proc_count: Keep codes with at least this many
            admission occurrences; rarer codes are dropped from the visit.
        max_drugs: Optionally cap the vocabulary to the ``max_drugs`` most
            frequent keys.
        use_procedures: Whether to build the procedure node type at all.
        max_visits_per_patient: Safety cap for very long histories.
        max_admissions: If set, keep a fixed-seed random subset of this many
            admissions *after* filtering (documented subset protocol).
        subset_seed: Seed for ``max_admissions`` sampling.

    Returns:
        :class:`ProcessedData` with records, vocabularies and stats.
    """
    mimic_dir = Path(mimic_dir)
    log = print if verbose else (lambda *a, **k: None)

    log(f"[preprocess] loading PRESCRIPTIONS from {mimic_dir} ...")
    pres = load_prescriptions(mimic_dir)
    log(f"[preprocess]   {len(pres):,} MAIN prescription rows, "
        f"{pres['DRUG_KEY'].nunique():,} normalised drug keys")

    log("[preprocess] loading DIAGNOSES_ICD ...")
    diag = load_codes(mimic_dir, "DIAGNOSES_ICD", "ICD9_CODE")
    if diag is None:
        raise FileNotFoundError(f"DIAGNOSES_ICD.csv not found under {mimic_dir}")
    log(f"[preprocess]   {len(diag):,} diagnosis rows")

    proc_source = "disabled"
    proc = None
    if use_procedures:
        proc, proc_source = load_procedures(mimic_dir)
        log(f"[preprocess]   procedures source: {proc_source}"
            + (f" ({len(proc):,} rows)" if proc is not None else ""))

    # Admission-level drug sets
    pres_pairs = pres[["SUBJECT_ID", "HADM_ID", "DRUG_KEY"]].drop_duplicates()
    drug_adm_counts = pres_pairs["DRUG_KEY"].value_counts()
    keep_drugs = drug_adm_counts[drug_adm_counts >= min_drug_admissions]
    if max_drugs is not None:
        keep_drugs = keep_drugs.iloc[:max_drugs]
    keep_drug_set = set(keep_drugs.index)
    pres_pairs = pres_pairs[pres_pairs["DRUG_KEY"].isin(keep_drug_set)]
    log(f"[preprocess]   drug vocabulary: {len(keep_drug_set)} keys "
        f"(>= {min_drug_admissions} admissions)")

    # Diagnosis / procedure filtering
    diag_counts = diag["CODE"].value_counts()
    diag = diag[diag["CODE"].isin(diag_counts[diag_counts >= min_diag_count].index)]
    if proc is not None:
        proc_counts = proc["CODE"].value_counts()
        proc = proc[proc["CODE"].isin(proc_counts[proc_counts >= min_proc_count].index)]

    # Admissions with >=1 diagnosis and >=1 drug
    adm_with_drug = set(map(tuple, pres_pairs[["SUBJECT_ID", "HADM_ID"]].drop_duplicates().values))
    adm_with_diag = set(map(tuple, diag[["SUBJECT_ID", "HADM_ID"]].drop_duplicates().values))
    keep_adm = adm_with_drug & adm_with_diag
    log(f"[preprocess]   admissions with drugs & diagnoses: {len(keep_adm):,}")

    if max_admissions is not None and len(keep_adm) > max_admissions:
        rng = np.random.default_rng(subset_seed)
        keep_list = sorted(keep_adm)
        idx = rng.choice(len(keep_list), size=max_admissions, replace=False)
        keep_adm = {keep_list[i] for i in idx}
        log(f"[preprocess]   SUBSET PROTOCOL: sampled {max_admissions:,} admissions (seed={subset_seed})")

    keep_hadm = {h for _, h in keep_adm}
    pres_pairs = pres_pairs[pres_pairs["HADM_ID"].isin(keep_hadm)]
    diag = diag[diag["HADM_ID"].isin(keep_hadm)]
    if proc is not None:
        proc = proc[proc["HADM_ID"].isin(keep_hadm)]

    order = _admission_order(pres, mimic_dir)
    order = order[order["HADM_ID"].isin(keep_hadm)]

    # Vocabularies (frequency-sorted for stable ids)
    diag_voc, proc_voc, drug_voc = Voc(), Voc(), Voc()
    for code, _ in diag["CODE"].value_counts().items():
        diag_voc.add(code)
    if proc is not None:
        for code, _ in proc["CODE"].value_counts().items():
            proc_voc.add(code)
    for key, _ in pres_pairs["DRUG_KEY"].value_counts().items():
        drug_voc.add(key)

    diag_by_adm = diag.groupby("HADM_ID")["CODE"].apply(list).to_dict()
    proc_by_adm = proc.groupby("HADM_ID")["CODE"].apply(list).to_dict() if proc is not None else {}
    drug_by_adm = pres_pairs.groupby("HADM_ID")["DRUG_KEY"].apply(list).to_dict()

    order = order.sort_values(["SUBJECT_ID", "ADMIT_KEY", "HADM_ID"])
    records: List[List[Dict[str, object]]] = []
    n_visits = 0
    for sid, grp in order.groupby("SUBJECT_ID", sort=False):
        visits = []
        for _, row in grp.iterrows():
            h = int(row["HADM_ID"])
            if h not in drug_by_adm or h not in diag_by_adm:
                continue
            visits.append({
                "subject_id": int(sid),
                "hadm_id": h,
                "diag": sorted({diag_voc.word2idx[c] for c in diag_by_adm[h]}),
                "proc": sorted({proc_voc.word2idx[c] for c in proc_by_adm.get(h, [])}),
                "drug": sorted({drug_voc.word2idx[c] for c in drug_by_adm[h]}),
            })
            if len(visits) >= max_visits_per_patient:
                break
        if visits:
            records.append(visits)
            n_visits += len(visits)

    raw_names: Dict[str, List[str]] = {}
    for raw, key in pres[["DRUG", "DRUG_KEY"]].drop_duplicates().values:
        if key in drug_voc.word2idx:
            raw_names.setdefault(key, []).append(str(raw))

    visit_lens = [len(v["drug"]) for p in records for v in p]
    stats = {
        "n_patients": len(records),
        "n_visits": n_visits,
        "n_diag": len(diag_voc),
        "n_proc": len(proc_voc),
        "n_drug": len(drug_voc),
        "avg_drugs_per_visit": float(np.mean(visit_lens)) if visit_lens else 0.0,
        "avg_diag_per_visit": float(np.mean([len(v["diag"]) for p in records for v in p])) if n_visits else 0.0,
        "avg_proc_per_visit": float(np.mean([len(v["proc"]) for p in records for v in p])) if n_visits else 0.0,
        "avg_visits_per_patient": float(n_visits / max(1, len(records))),
        "procedure_source": proc_source,
        "min_drug_admissions": min_drug_admissions,
        "min_diag_count": min_diag_count,
        "min_proc_count": min_proc_count,
        "max_admissions": max_admissions,
        "subset_seed": subset_seed,
    }
    log(f"[preprocess] done: {stats['n_patients']:,} patients, {n_visits:,} visits, "
        f"{len(diag_voc)} diag / {len(proc_voc)} proc / {len(drug_voc)} drug codes, "
        f"{stats['avg_drugs_per_visit']:.1f} drugs/visit")
    return ProcessedData(records, diag_voc, proc_voc, drug_voc, raw_names, stats)


def write_demo_license_note(out_dir: Path, mimic_dir: Path) -> None:
    """Copy LICENSE.txt of the MIMIC-III demo next to processed demo data if present."""
    lic = Path(mimic_dir) / "LICENSE.txt"
    if lic.exists():
        (Path(out_dir) / "MIMIC_DEMO_LICENSE.txt").write_text(lic.read_text(encoding="utf-8", errors="ignore"),
                                                              encoding="utf-8")
