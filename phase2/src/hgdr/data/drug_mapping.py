"""Drug-name normalisation and PubChem (CID / SMILES) mapping.

MIMIC-III ``PRESCRIPTIONS.DRUG`` strings are free text such as
``"Metoprolol Succinate XL"``, ``"Heparin Flush (10 units/ml)"`` or
``"0.9% Sodium Chloride"``.  We normalise them to an *active-ingredient
key* and map the key to a PubChem CID + canonical SMILES using a shipped
offline dictionary (``data/mappings/drug_name_to_pubchem.json``), which
was built with the PubChem PUG-REST API and hand-curated fixes taken from
the original research notebooks.  Missing names can be resolved online
with :func:`lookup_pubchem_online` (requires network access).
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Dict, Iterable, Optional

# Words that indicate a formulation / vehicle rather than an ingredient.
_FORMULATION_WORDS = [
    "INHALER", "SUSPENSION", "SUSP", "OINTMENT", "OINT", "SOLUTION", "SOLN", "TABLET", "TAB",
    "CAPSULE", "CAP", "LIQUID", "LIQ", "FLUSH", "POWDER", "CREAM", "GEL", "PATCH", "NEB",
    "MDI", "ELIXIR", "SYRUP", "SPRAY", "DROPS", "OPHTH", "OPHTHALMIC", "OTIC", "NASAL",
    "TOPICAL", "ORAL", "RECTAL", "ENEMA", "SUPPOSITORY", "LOTION", "DISKUS", "HFA",
    "EXTENDED RELEASE", "EXTENDED-RELEASE", "SUSTAINED RELEASE", "IMMEDIATE RELEASE",
    "DELAYED RELEASE", "CR", "ER", "XL", "XR", "SR", "IR", "ODT", "DS", "SS", "P.F.", "PF",
    "IV", "PO", "IM", "SC", "SQ", "PREMIX", "MINI BAG PLUS", "MINI-BAG PLUS", "EXCEL BAG",
    "GLASS BOTTLE", "IRRIGATION BOTTLE", "BAG", "BOTTLE", "VIAL", "SYRINGE", "KIT", "UNITS",
    "UNIT", "GENERIC", "DESENSITIZATION", "STUDY DRUG", "PLACEBO", "*NF*", "NF",
    "CVL", "PICC", "CRRT", "LOCK", "HICKMAN", "EC", "DISINTEGRATING", "CHEWABLE", "PRESERV",
    "FREE", "CONCENTRATED", "CONCENTRATE", "INJ", "INJECTION", "INFUSION", "DRIP", "PEDIATRIC",
    "ADULT", "HUMAN", "REGULAR", "RINSE", "MOUTHWASH", "SWISH", "SWALLOW", "WASH", "SHAMPOO",
    "PORT", "DWELL", "REPLACEMENT", "TTS", "MODIFIED", "EXTENDED", "INFATAB", "WAFER",
    "DIALYSIS",
]
# Neonatal / route prefixes such as ``NEO*IV*GENTAMICIN`` -> ``GENTAMICIN``.
_NEO_PREFIX_RE = re.compile(r"^NEO\*[A-Z]{2,3}\*", re.IGNORECASE)

# Trailing salt / counter-ion words that do not change the active ingredient.
_SALT_WORDS = {
    "SODIUM", "NA", "SUCC", "SUCCINATE", "HCL", "HYDROCHLORIDE", "SULFATE", "SULPHATE",
    "TARTRATE", "CITRATE", "ACETATE", "MALEATE", "MESYLATE", "BESYLATE", "BISULFATE",
    "FUMARATE", "HYDROBROMIDE", "BROMIDE", "PHOSPHATE", "POTASSIUM", "CALCIUM", "MAGNESIUM",
    "DIHYDRATE", "MONOHYDRATE", "TROMETHAMINE", "LACTATE", "GLUCONATE", "PROPIONATE",
    "DIPROPIONATE", "PALMITATE", "PROXETIL", "AXETIL", "VALERATE", "DECANOATE", "ENANTHATE",
    "NITRATE", "HYDROXIDE", "CHLORIDE", "DISODIUM", "TRISODIUM", "ALFA", "BETA",
}
# Cations / elements that *are* the drug (never strip these leading tokens' salts).
_ELEMENT_FIRST = {
    "POTASSIUM", "MAGNESIUM", "CALCIUM", "SODIUM", "FERROUS", "FERRIC", "ZINC", "ALUMINUM",
    "LITHIUM", "IRON", "SELENIUM", "CHROMIUM", "COPPER", "MANGANESE", "AMMONIUM", "SILVER",
    "GOLD", "BARIUM", "BISMUTH",
}

# Entries that are not medications (fluids, devices, placeholders).  They are
# excluded from the drug vocabulary.
NON_DRUG_KEYS = {
    "", "VIAL", "SYRINGE", "SOLN", "SOLUTION", "SW", "NS", "D5W", "D5NS", "D5 1/2NS",
    "1/2 NS", "D5LR", "LR", "STERILE WATER", "WATER", "BAG", "EPIDURAL BAG", "SOLN.",
    "SYRINGE (IV ROOM)", "SYRINGE (CHEMO)", "IV FLUID", "IV FLUIDS", "DILUENT",
    "STERILE DILUENT", "CARRIER", "PLACEBO", "NORMAL SALINE", "TPN", "PN", "TPN D", "PN D",
    "AMINO ACIDS", "FAT EMULSION", "DEXTROSE", "ISO-OSMOTIC DEXTROSE", "SODIUM CHLORIDE",
    "ISO-OSMOTIC SODIUM CHLORIDE", "ISOTONIC SODIUM CHLORIDE", "LACTATED RINGERS",
    "STUDY DRUG", "INVESTIGATIONAL DRUG", "READI-CAT", "READI-CAT 2", "BARIUM SULFATE",
}

_PCT_RE = re.compile(r"\d+(\.\d+)?\s*%")
_DOSE_RE = re.compile(
    r"\b\d+(\.\d+)?\s*(mg|mcg|ug|g|gm|kg|meq|ml|l|units?|iu|mmol|mcg/ml|mg/ml|hr|hour|h)\b",
    re.IGNORECASE,
)
_RATIO_RE = re.compile(r"\b\d+(\.\d+)?\s*[-/:]\s*\d+(\.\d+)?\b")
_NUM_RE = re.compile(r"(?<![A-Za-z])\d+(\.\d+)?(?![A-Za-z])")
_PAREN_RE = re.compile(r"\([^)]*\)|\[[^\]]*\]")
_WS_RE = re.compile(r"\s+")


def normalize_drug_name(name: str) -> str:
    """Normalise a raw MIMIC ``DRUG`` string to an active-ingredient key.

    Steps: upper-case, drop parenthesised text, strip percentages / doses /
    ratios / free numbers, drop formulation words, collapse whitespace and
    trailing punctuation.

    >>> normalize_drug_name("Metoprolol Succinate XL")
    'METOPROLOL SUCCINATE'
    >>> normalize_drug_name("Heparin Flush (10 units/ml)")
    'HEPARIN'
    """
    if not isinstance(name, str):
        return ""
    s = name.upper().strip()
    s = _NEO_PREFIX_RE.sub("", s)
    s = _PAREN_RE.sub(" ", s)
    s = _PCT_RE.sub(" ", s)
    s = _DOSE_RE.sub(" ", s)
    s = _RATIO_RE.sub(" ", s)
    s = _NUM_RE.sub(" ", s)
    s = s.replace("*NF*", " ")
    tokens = [t for t in re.split(r"[\s,/]+", s) if t]
    kept = []
    for t in tokens:
        t_clean = t.strip(".-")
        if not t_clean or t_clean in _FORMULATION_WORDS:
            continue
        kept.append(t_clean)
    # strip trailing salt words (e.g. METOPROLOL TARTRATE -> METOPROLOL) unless the
    # compound *is* the salt (POTASSIUM CHLORIDE, MAGNESIUM SULFATE, ...)
    if kept and kept[0] not in _ELEMENT_FIRST:
        while len(kept) > 1 and kept[-1] in _SALT_WORDS:
            kept.pop()
    s = " ".join(kept)
    s = _WS_RE.sub(" ", s).strip(" -.,")
    return s


def is_non_drug(key: str) -> bool:
    """Return True for fluids / devices / placeholders."""
    if key in NON_DRUG_KEYS:
        return True
    for kw in ("SYRINGE", "VIAL", "PLACEBO", "STUDY DRUG", "DILUENT"):
        if kw in key:
            return True
    return False


def cid_to_twosides_id(cid: int | str) -> str:
    """Format a PubChem CID as the TWOSIDES ``CID000002173`` style id."""
    return f"CID{int(cid):09d}"


def twosides_id_to_cid(tid: str) -> int:
    """Inverse of :func:`cid_to_twosides_id`."""
    return int(str(tid).replace("CID", ""))


def load_mapping(path: str | Path) -> Dict[str, Dict[str, object]]:
    """Load the ``{normalised_name: {cid, smiles, name}}`` dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out: Dict[str, Dict[str, object]] = {}
    for k, v in raw.items():
        key = normalize_drug_name(k) if k != k.upper() else k
        cid = v.get("cid") if isinstance(v, dict) else None
        smiles = v.get("smiles") if isinstance(v, dict) else None
        if cid in (None, "unknown", "") or smiles in (None, "unknown", ""):
            continue
        try:
            cid_int = int(cid)
        except (TypeError, ValueError):
            continue
        out.setdefault(key, {"cid": cid_int, "smiles": smiles, "name": v.get("name", k)})
    return out


def save_mapping(mapping: Dict[str, Dict[str, object]], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dict(sorted(mapping.items())), f, indent=1)


def lookup_pubchem_online(
    names: Iterable[str],
    mapping: Dict[str, Dict[str, object]],
    sleep_s: float = 0.25,
    max_queries: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Dict[str, object]]:
    """Resolve unmapped names through the PubChem PUG-REST API.

    Tries the full key, then progressively shorter prefixes (e.g.
    ``METOPROLOL SUCCINATE`` -> ``METOPROLOL``).  Results (including
    failures, stored as ``None``) are written into ``mapping`` and returned.
    Requires network access; failures are silently skipped.
    """
    try:
        import requests
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("pip install requests to use online lookup") from e

    # PubChem renamed CanonicalSMILES -> SMILES / ConnectivitySMILES in 2025; accept any of them.
    base = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/property/SMILES/JSON"
    queried = 0
    for raw in names:
        key = normalize_drug_name(raw)
        if not key or key in mapping or is_non_drug(key):
            continue
        if max_queries is not None and queried >= max_queries:
            break
        candidates = [key]
        toks = key.split()
        for n in range(len(toks) - 1, 0, -1):
            candidates.append(" ".join(toks[:n]))
        found = None
        for cand in candidates:
            queried += 1
            try:
                r = requests.get(base.format(requests.utils.quote(cand)), timeout=15)
                if r.status_code == 200:
                    props = r.json()["PropertyTable"]["Properties"][0]
                    smi = props.get("SMILES") or props.get("ConnectivitySMILES") or props.get("CanonicalSMILES")
                    if smi:
                        found = {"cid": int(props["CID"]), "smiles": smi, "name": cand}
                        break
            except Exception:  # network hiccup / parse error
                pass
            time.sleep(sleep_s)
        if found:
            mapping[key] = found
            if verbose:
                print(f"  mapped {key!r} -> CID {found['cid']}")
        elif verbose:
            print(f"  no PubChem match for {key!r}")
    return mapping
