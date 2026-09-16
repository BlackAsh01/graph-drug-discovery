"""Protein sequence encodings.

Three representations are supported, matching the three protein-encoder ablations:

* **tokens**      - integer residue tokens for a learned 1-D CNN encoder,
* **composition** - normalised amino-acid frequency vector (bag-of-residues baseline),
* **pretrained**  - frozen mean-pooled embeddings from a protein language model
  (ProtBERT-BFD by default) computed once by :mod:`scripts.embed_proteins` and stored in
  a ``.npz`` file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch

#: 20 standard amino acids + 5 ambiguous / rare codes. Index 0 is padding.
AA_VOCAB = "ACDEFGHIKLMNPQRSTVWYBUXZO"
AA_TO_IDX: Dict[str, int] = {aa: i + 1 for i, aa in enumerate(AA_VOCAB)}
PAD_IDX = 0
UNK_IDX = AA_TO_IDX["X"]


def sequence_to_tokens(sequence: str, max_len: int = 1000) -> torch.Tensor:
    """Encode a protein sequence as a fixed-length ``LongTensor`` of residue indices (0 = pad)."""
    seq = sequence.strip().upper()[:max_len]
    idx = [AA_TO_IDX.get(c, UNK_IDX) for c in seq]
    out = torch.full((max_len,), PAD_IDX, dtype=torch.long)
    out[: len(idx)] = torch.tensor(idx, dtype=torch.long)
    return out


def sequence_to_composition(sequence: str) -> torch.Tensor:
    """Return ``[len(AA_VOCAB) + 1]`` float features: residue frequencies + log10(length)."""
    seq = sequence.strip().upper()
    counts = torch.zeros(len(AA_VOCAB), dtype=torch.float)
    for c in seq:
        counts[AA_TO_IDX.get(c, UNK_IDX) - 1] += 1.0
    freq = counts / max(len(seq), 1)
    return torch.cat([freq, torch.tensor([np.log10(max(len(seq), 1))], dtype=torch.float)])


def format_for_protbert(sequence: str) -> str:
    """Space-separate residues and map rare residues to ``X`` as ProtBERT / ProtTrans expect.

    The original research code skipped this step, which made ProtBERT tokenise every
    sequence to ``[UNK]`` and produced *identical* embeddings for all 229 KIBA targets.
    """
    seq = sequence.strip().upper()
    seq = "".join(c if c in "ACDEFGHIKLMNPQRSTVWY" else "X" for c in seq)
    return " ".join(seq)


def load_protein_embeddings(path: str | Path) -> Tuple[Dict[str, int], torch.Tensor]:
    """Load a ``.npz`` produced by ``scripts/embed_proteins.py``.

    Returns:
        ``(index, matrix)`` where ``index`` maps ``target_id -> row`` and ``matrix`` is a
        float32 tensor of shape ``[num_targets, dim]``.
    """
    npz = np.load(Path(path), allow_pickle=False)
    ids = [str(t) for t in npz["target_ids"]]
    emb = torch.from_numpy(npz["embeddings"].astype(np.float32))
    if len(ids) != emb.shape[0]:
        raise ValueError(f"{path}: {len(ids)} ids but {emb.shape[0]} embedding rows")
    return {t: i for i, t in enumerate(ids)}, emb


def save_protein_embeddings(path: str | Path, target_ids: Iterable[str], embeddings: np.ndarray, model_name: str) -> None:
    """Save embeddings in the format read by :func:`load_protein_embeddings`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        target_ids=np.array(list(target_ids), dtype=str),
        embeddings=embeddings.astype(np.float32),
        model_name=np.array(model_name),
    )


def build_protein_features(
    encoder: str,
    sequences_by_id: Dict[str, str],
    max_len: int = 1000,
    embedding_file: str | Path | None = None,
) -> Dict[str, torch.Tensor]:
    """Compute the per-target input tensor required by the chosen protein encoder.

    Args:
        encoder: ``"cnn"``, ``"composition"`` or ``"pretrained"``.
        sequences_by_id: mapping ``target_id -> sequence``.
        max_len: truncation length for the token encoding.
        embedding_file: ``.npz`` file for the pretrained encoder.
    """
    if encoder == "cnn":
        return {t: sequence_to_tokens(s, max_len) for t, s in sequences_by_id.items()}
    if encoder == "composition":
        return {t: sequence_to_composition(s) for t, s in sequences_by_id.items()}
    if encoder == "pretrained":
        if embedding_file is None:
            raise ValueError("protein.embedding_file must be set for the pretrained encoder")
        index, matrix = load_protein_embeddings(embedding_file)
        missing: List[str] = [t for t in sequences_by_id if t not in index]
        if missing:
            raise KeyError(f"{len(missing)} targets missing from {embedding_file}: {missing[:5]}")
        return {t: matrix[index[t]] for t in sequences_by_id}
    raise ValueError(f"unknown protein encoder '{encoder}'")
