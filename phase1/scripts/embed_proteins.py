"""Compute frozen protein-language-model embeddings for every KIBA target.

Default model is ``Rostlab/prot_bert_bfd`` (ProtBERT-BFD, 420 M parameters, ~1.7 GB download).
Residues are space separated and rare residues mapped to ``X`` as required by ProtTrans
models - the step missing from the original research code. The mean over residue
positions (excluding special tokens) is stored as a ``[num_targets, hidden]`` matrix.

Examples::

    python scripts/embed_proteins.py                               # ProtBERT-BFD -> data/protein_embeddings/kiba_protbert_bfd.npz
    python scripts/embed_proteins.py --model facebook/esm2_t12_35M_UR50D --out data/protein_embeddings/kiba_esm2_35m.npz
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

# transformers only needs PyTorch here; prevent it from importing TensorFlow / Flax if installed
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import _bootstrap  # noqa: F401
from _bootstrap import ROOT

import numpy as np
import torch

from dti_gt.data.kiba import resolve_kiba
from dti_gt.data.proteins import format_for_protbert, save_protein_embeddings


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Rostlab/prot_bert_bfd")
    ap.add_argument("--out", default=None, help="output .npz (default derived from model name)")
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--pairs-file", default=None)
    ap.add_argument("--max-len", type=int, default=1024, help="max tokens incl. special tokens")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--fp16", action="store_true", help="run the language model in half precision on GPU")
    args = ap.parse_args()

    from transformers import AutoModel, AutoTokenizer

    df = resolve_kiba(Path(args.data_dir), args.pairs_file)
    targets = df.drop_duplicates("target_id")[["target_id", "sequence"]].reset_index(drop=True)
    print(f"[embed] {len(targets)} unique targets; model={args.model}")

    device = torch.device("cuda" if args.device in ("auto", "cuda") and torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model, do_lower_case=False)
    model = AutoModel.from_pretrained(args.model)
    if args.fp16 and device.type == "cuda":
        model = model.half()
    model = model.to(device).eval()
    is_protbert = "prot_bert" in args.model.lower() or "prot_t5" in args.model.lower()

    embeddings = []
    t0 = time.time()
    with torch.no_grad():
        for i, row in targets.iterrows():
            seq = format_for_protbert(row["sequence"]) if is_protbert else row["sequence"].strip().upper()
            enc = tokenizer(seq, return_tensors="pt", truncation=True, max_length=args.max_len)
            enc = {k: v.to(device) for k, v in enc.items()}
            try:
                hidden = model(**enc).last_hidden_state  # [1, L, H]
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                model = model.float().cpu()
                device = torch.device("cpu")
                enc = {k: v.cpu() for k, v in enc.items()}
                hidden = model(**enc).last_hidden_state
                print("[embed] CUDA OOM -> continuing on CPU")
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            # drop [CLS]/[SEP] (first/last real token) from the mean
            mask[:, 0] = 0
            last = int(enc["attention_mask"].sum().item()) - 1
            mask[:, last] = 0
            emb = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            embeddings.append(emb.squeeze(0).float().cpu().numpy())
            if (i + 1) % 10 == 0 or i + 1 == len(targets):
                print(f"[embed] {i + 1}/{len(targets)} ({time.time() - t0:.0f}s)")
            if device.type == "cuda":
                torch.cuda.empty_cache()

    matrix = np.stack(embeddings)
    out = Path(args.out) if args.out else ROOT / "data" / "protein_embeddings" / f"kiba_{args.model.split('/')[-1].replace('-', '_').lower()}.npz"
    save_protein_embeddings(out, targets["target_id"].tolist(), matrix, args.model)
    spread = matrix.std(0).mean()
    print(f"[embed] saved {matrix.shape} to {out}; mean per-dim std across targets = {spread:.4f} (0 would mean degenerate embeddings)")


if __name__ == "__main__":
    main()
