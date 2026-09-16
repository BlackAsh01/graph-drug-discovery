"""Download (or copy) the raw KIBA table to ``data/raw/kiba.tab``.

Examples::

    python scripts/download_kiba.py                       # download from Harvard Dataverse (TDC copy, ~96 MB)
    python scripts/download_kiba.py --from-local D:/path/to/kiba.tab
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from _bootstrap import ROOT

from dti_gt.data.download import copy_local, download_kiba
from dti_gt.data.kiba import KIBA_URL, load_kiba_table, summarize


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", default=str(ROOT / "data" / "raw" / "kiba.tab"), help="destination file")
    ap.add_argument("--url", default=KIBA_URL, help="download URL (default: TDC copy on Harvard Dataverse)")
    ap.add_argument("--from-local", default=None, help="copy an existing kiba.tab instead of downloading")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    dest = copy_local(args.from_local, args.dest) if args.from_local else download_kiba(args.dest, args.url, args.overwrite)
    df = load_kiba_table(dest)
    print("[download] dataset summary:", summarize(df))


if __name__ == "__main__":
    main()
