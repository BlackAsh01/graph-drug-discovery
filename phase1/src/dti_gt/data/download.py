"""Download the KIBA dataset (TDC copy hosted on Harvard Dataverse) without extra dependencies."""

from __future__ import annotations

import shutil
import sys
import urllib.request
from pathlib import Path

from dti_gt.data.kiba import KIBA_URL


def download_kiba(dest: str | Path, url: str = KIBA_URL, overwrite: bool = False) -> Path:
    """Download ``kiba.tab`` to ``dest`` (a file path). Returns the destination path."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and not overwrite:
        print(f"[download] {dest} already exists (use --overwrite to re-download)")
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[download] {url} -> {dest}")

    def _hook(blocks: int, block_size: int, total: int) -> None:
        done = blocks * block_size
        if total > 0:
            sys.stdout.write(f"\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB")
        else:
            sys.stdout.write(f"\r  {done / 1e6:7.1f} MB")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, tmp, reporthook=_hook)
    sys.stdout.write("\n")
    shutil.move(str(tmp), str(dest))
    return dest


def copy_local(src: str | Path, dest: str | Path) -> Path:
    """Copy an existing ``kiba.tab`` (e.g. a previous TDC download) into the data folder."""
    src, dest = Path(src), Path(dest)
    if not src.is_file():
        raise FileNotFoundError(src)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    print(f"[download] copied {src} -> {dest}")
    return dest
