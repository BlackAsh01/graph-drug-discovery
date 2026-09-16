"""SMILES -> molecular graph featurisation with RDKit.

Every atom / bond attribute is encoded as a *categorical index* so that the model can
learn an embedding per attribute (OGB / Graphormer style). This replaces the original
research code, which fed raw numeric values (atomic number, hybridisation enum, ...)
straight into a linear layer.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
from rdkit import Chem, RDLogger
from torch_geometric.data import Data

RDLogger.DisableLog("rdApp.*")

# --------------------------------------------------------------------------------------
# Vocabularies (index 0 is always reserved for "unknown / other")
# --------------------------------------------------------------------------------------
_HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
_BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
MAX_ATOMIC_NUM = 118
MAX_DEGREE = 10

#: Number of categories for each atom feature column (used to size embedding tables).
ATOM_FEATURE_DIMS: List[int] = [
    MAX_ATOMIC_NUM + 1,  # atomic number (0 = unknown)
    MAX_DEGREE + 1,  # total degree (clipped)
    11,  # formal charge in [-5, 5] shifted by +5
    9,  # number of attached hydrogens (clipped at 8)
    len(_HYBRIDIZATIONS) + 1,  # hybridisation
    2,  # aromatic
    2,  # in ring
    4,  # chirality tag
]

#: Number of categories for each bond feature column. Index 0 of the bond-type column is
#: reserved for self-loops that the model may add.
BOND_FEATURE_DIMS: List[int] = [
    len(_BOND_TYPES) + 2,  # bond type: 0 = self-loop, 1..4 = types, 5 = other
    2,  # conjugated
    2,  # in ring
]
SELF_LOOP_BOND_TYPE = 0


def _index(value, vocab: list, offset: int = 1) -> int:
    try:
        return vocab.index(value) + offset
    except ValueError:
        return 0


def atom_features(atom: Chem.Atom) -> List[int]:
    """Return the categorical feature vector of an RDKit atom (length ``len(ATOM_FEATURE_DIMS)``)."""
    atomic_num = atom.GetAtomicNum()
    return [
        atomic_num if 1 <= atomic_num <= MAX_ATOMIC_NUM else 0,
        min(atom.GetTotalDegree(), MAX_DEGREE),
        max(-5, min(5, atom.GetFormalCharge())) + 5,
        min(atom.GetTotalNumHs(), 8),
        _index(atom.GetHybridization(), _HYBRIDIZATIONS),
        int(atom.GetIsAromatic()),
        int(atom.IsInRing()),
        min(int(atom.GetChiralTag()), 3),
    ]


def bond_features(bond: Chem.Bond) -> List[int]:
    """Return the categorical feature vector of an RDKit bond (length ``len(BOND_FEATURE_DIMS)``)."""
    bond_type = _index(bond.GetBondType(), _BOND_TYPES)
    if bond_type == 0:  # unknown type -> "other" (keep 0 for self loops)
        bond_type = len(_BOND_TYPES) + 1
    return [bond_type, int(bond.GetIsConjugated()), int(bond.IsInRing())]


def smiles_to_graph(smiles: str, drug_id: Optional[str] = None) -> Optional[Data]:
    """Convert a SMILES string to a :class:`torch_geometric.data.Data` object.

    Args:
        smiles: SMILES string.
        drug_id: Optional identifier stored on the returned object.

    Returns:
        A ``Data`` with ``x`` (``[N, 8]`` long), ``edge_index`` (``[2, 2*B]``),
        ``edge_attr`` (``[2*B, 3]`` long) and ``degree`` (``[N]`` long, clipped in-degree),
        or ``None`` if RDKit cannot parse the SMILES.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.long)

    src, dst, eattr = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        f = bond_features(bond)
        src += [i, j]
        dst += [j, i]
        eattr += [f, f]

    if src:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(eattr, dtype=torch.long)
    else:  # single-atom molecule
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, len(BOND_FEATURE_DIMS)), dtype=torch.long)

    degree = torch.zeros(x.size(0), dtype=torch.long)
    if edge_index.numel():
        degree = torch.bincount(edge_index[1], minlength=x.size(0)).clamp(max=MAX_DEGREE)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, degree=degree)
    if drug_id is not None:
        data.drug_id = drug_id
    return data


def featurize_drugs(smiles_by_id: Dict[str, str], verbose: bool = True) -> Dict[str, Data]:
    """Featurise a mapping ``drug_id -> SMILES`` and drop unparsable molecules."""
    from tqdm import tqdm

    graphs: Dict[str, Data] = {}
    failed = []
    iterator = smiles_by_id.items()
    if verbose:
        iterator = tqdm(iterator, total=len(smiles_by_id), desc="SMILES -> graph")
    for drug_id, smi in iterator:
        g = smiles_to_graph(smi)
        if g is None:
            failed.append(drug_id)
        else:
            graphs[drug_id] = g
    if failed and verbose:
        print(f"[featurize] dropped {len(failed)} unparsable SMILES: {failed[:5]}{'...' if len(failed) > 5 else ''}")
    return graphs
