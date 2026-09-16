"""Run every ablation variant for several seeds and aggregate the results.

Outputs (under ``results/``):

* ``ablation_results.csv``  - one row per (variant, seed) with all test metrics + runtime,
* ``ablation_table.md``     - mean +/- std per variant (markdown),
* ``runs/<variant>_seed<k>/`` - per-run artefacts (config, history, metrics.json, log).

The driver is **resumable**: a run whose ``metrics.json`` exists (and was produced under the current
protocol) is skipped, and ``ablation_results.csv`` is rewritten after every finished run, so an interrupted
study can simply be relaunched with the same command. Runs are executed sequentially, seed-major.

Examples::

    python scripts/run_ablation.py                                   # all variants in configs/ablation, seeds 0 1
    python scripts/run_ablation.py --variants full no_edge_features --seeds 0 1
    python scripts/run_ablation.py --aggregate-only                  # rebuild CSV + markdown table from disk
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import _bootstrap  # noqa: F401
from _bootstrap import ROOT

import pandas as pd
import yaml

from dti_gt.data.dataset import build_dataloaders
from dti_gt.data.kiba import split_name
from dti_gt.train import train_from_config
from dti_gt.utils.config import load_config

# Windows consoles / redirected logs default to cp1252; UTF-8 so ± ≤ etc. survive.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (OSError, ValueError):
        pass

METRICS = ["mse", "rmse", "mae", "ci", "rm2", "pearson", "spearman", "auroc", "auprc"]
TABLE_METRICS = [("mse", "MSE ↓"), ("ci", "CI ↑"), ("rm2", "r_m² ↑"), ("pearson", "Pearson ↑"), ("auroc", "AUROC ↑")]
VARIANT_DESCRIPTIONS = {
    "full": "Full model (edge-aware GT, 4 layers, 8 heads, degree enc., ProtBERT-BFD, cross-attention fusion)",
    "no_edge_features": "- bond features in attention / edge channel",
    "no_degree_encoding": "- centrality (degree) encoding",
    "depth_1": "1 transformer layer instead of 4",
    "heads_1": "1 attention head instead of 8",
    "no_residual_norm": "- residual connections, - normalisation",
    "protein_cnn": "Protein: learned 1-D CNN on sequence (instead of ProtBERT-BFD)",
    "protein_composition": "Protein: amino-acid composition MLP (instead of ProtBERT-BFD)",
    "fusion_concat": "Fusion: concat (instead of cross-attention)",
    "backbone_gcn": "Drug encoder: GCN (original notebook backbone)",
    "backbone_gat": "Drug encoder: GAT",
}


def discover_variants(cfg_dir: Path) -> List[Path]:
    return sorted(p for p in cfg_dir.glob("*.yaml") if not p.name.startswith("_"))


def result_row(res: Dict) -> Dict:
    """Flatten a ``metrics.json`` dict into one CSV row."""
    row = {
        "variant": res["variant"], "seed": res["seed"], "run_name": res["run_name"], "best_epoch": res["best_epoch"],
        "epochs_run": res["epochs_run"], "train_time_s": res["train_time_s"], "n_parameters": res["n_parameters"], "device": res["device"],
        "split_dir": res["data"]["split_dir"], "n_train": res["data"]["sizes"]["train"], "n_val": res["data"]["sizes"]["val"], "n_test": res["data"]["sizes"]["test"],
    }
    row.update({f"val_{k}": res["val"][k] for k in METRICS})
    row.update({f"test_{k}": res["test"][k] for k in METRICS})
    return row


def collect_results(runs_dir: Path, variants: List[str], seeds: List[int]) -> pd.DataFrame:
    """Read ``runs/<variant>_seed<k>/metrics.json`` for every requested (variant, seed)."""
    rows = []
    for v in variants:
        for s in seeds:
            f = runs_dir / f"{v}_seed{s}" / "metrics.json"
            if f.is_file():
                rows.append(result_row(json.loads(f.read_text())))
    return pd.DataFrame(rows)


def write_results_csv(runs_dir: Path, variants: List[str], seeds: List[int], results_csv: Path) -> pd.DataFrame:
    """(Re)write ``ablation_results.csv`` from whatever runs are finished on disk (called after every run)."""
    df = collect_results(runs_dir, variants, seeds)
    if not df.empty:
        df.sort_values(["variant", "seed"]).to_csv(results_csv, index=False)
    return df


def is_complete(run_dir: Path, cfg: Dict) -> bool:
    """A run counts as done when ``metrics.json`` exists *and* was produced under the current protocol
    (same split folder, same epoch budget); otherwise it is re-trained so the final table is consistent."""
    f = run_dir / "metrics.json"
    if not f.is_file():
        return False
    try:
        res = json.loads(f.read_text())
        dcfg = cfg["data"]
        expected_split = dcfg.get("split_name") or split_name(int(dcfg["split_seed"]), float(dcfg["subset_fraction"]))
        same_split = Path(res["data"]["split_dir"]).name == expected_split
        run_cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8")) if (run_dir / "config.yaml").is_file() else {}
        same_budget = int(run_cfg.get("training", {}).get("epochs", -1)) == int(cfg["training"]["epochs"])
    except (KeyError, ValueError, OSError):
        return False
    if not (same_split and same_budget):
        print(f"[ablation] {run_dir.name}: metrics.json is from a different protocol -> re-training", flush=True)
    return same_split and same_budget


def aggregate(results_csv: Path, table_md: Path) -> pd.DataFrame:
    """Build the mean +/- std markdown table from ``ablation_results.csv``."""
    df = pd.read_csv(results_csv)
    order = [v for v in VARIANT_DESCRIPTIONS if v in set(df["variant"])] + sorted(set(df["variant"]) - set(VARIANT_DESCRIPTIONS))
    g = df.groupby("variant")
    lines = ["| Variant | Description | n | " + " | ".join(name for _, name in TABLE_METRICS) + " | Params | Time / run |", "|---|---|---|" + "---|" * len(TABLE_METRICS) + "---|---|"]
    full_mse = g.get_group("full")["test_mse"].mean() if "full" in g.groups else None
    for v in order:
        sub = g.get_group(v)
        cells = []
        for key, _ in TABLE_METRICS:
            col = f"test_{key}"
            m, s = sub[col].mean(), sub[col].std(ddof=0) if len(sub) > 1 else 0.0
            cell = f"{m:.4f} ± {s:.4f}"
            if key == "mse" and full_mse is not None and v != "full":
                cell += f" ({(m - full_mse) / full_mse * 100:+.1f}%)"
            cells.append(cell)
        params = f"{sub['n_parameters'].iloc[0] / 1e6:.2f} M"
        t = f"{sub['train_time_s'].mean() / 60:.1f} min"
        lines.append(f"| `{v}` | {VARIANT_DESCRIPTIONS.get(v, '')} | {len(sub)} | " + " | ".join(cells) + f" | {params} | {t} |")
    meta = df.iloc[0]
    run_cfg_file = results_csv.parent / "runs" / meta["run_name"] / "config.yaml"
    tcfg = (yaml.safe_load(run_cfg_file.read_text(encoding="utf-8")) if run_cfg_file.is_file() else {}).get("training", {})
    budget = f"batch {tcfg.get('batch_size', '?')}, ≤ {tcfg.get('epochs', '?')} epochs, early stopping patience {tcfg.get('early_stopping_patience', '?')} on val MSE, AdamW lr {tcfg.get('lr', '?')}"
    header = [
        "# Ablation study (KIBA, subset protocol)",
        "",
        f"Protocol: `{Path(meta['split_dir']).name}` — train/val/test = {meta['n_train']}/{meta['n_val']}/{meta['n_test']} pairs; {budget}; "
        f"seeds = {sorted(df['seed'].unique().tolist())}; best-validation-MSE checkpoint evaluated on the test split. "
        "Values are mean ± std over seeds; percentages are relative MSE change vs. the full model. Runs were executed sequentially on one GTX 1650 Ti (4 GB).",
        "",
    ]
    table_md.write_text("\n".join(header + lines) + "\n", encoding="utf-8", newline="\n")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config-dir", default=str(ROOT / "configs" / "ablation"))
    ap.add_argument("--variants", nargs="*", default=None, help="subset of variant names (file stems)")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1])
    ap.add_argument("--results-dir", default=str(ROOT / "results"))
    ap.add_argument("--no-skip-existing", action="store_true", help="re-train runs that already have metrics.json (default: reuse them)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="overrides applied to every variant")
    ap.add_argument("--aggregate-only", action="store_true", help="only rebuild the CSV/markdown from existing runs")
    ap.add_argument("--no-aggregate", action="store_true", help="train only (useful when several workers run disjoint variants in parallel)")
    args = ap.parse_args()

    cfg_dir, results_dir = Path(args.config_dir), Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = results_dir / "runs"
    results_csv, table_md = results_dir / "ablation_results.csv", results_dir / "ablation_table.md"
    all_variants = discover_variants(cfg_dir)
    by_stem = {v.stem: v for v in all_variants}
    if args.variants:  # keep the order given on the command line (cheap / important variants first)
        unknown = [v for v in args.variants if v not in by_stem]
        if unknown:
            raise SystemExit(f"unknown variants {unknown}; available: {sorted(by_stem)}")
        variants = [by_stem[v] for v in args.variants]
    else:
        variants = [by_stem[v] for v in VARIANT_DESCRIPTIONS if v in by_stem] + [v for v in all_variants if v.stem not in VARIANT_DESCRIPTIONS]

    loader_cache: Dict[str, tuple] = {}
    all_stems = [v.stem for v in all_variants]
    t_all = time.time()
    if not args.aggregate_only:
        # seed-major order: if the study is interrupted, every variant has the same number of finished seeds
        todo = [(vpath, seed) for seed in args.seeds for vpath in variants]
        for i, (vpath, seed) in enumerate(todo, 1):
            run_name = f"{vpath.stem}_seed{seed}"
            cfg = load_config(vpath, list(args.set) + [f"training.seed={seed}"])
            cfg["results_dir"] = str(results_dir.relative_to(ROOT)) if results_dir.is_relative_to(ROOT) else str(results_dir)
            if not args.no_skip_existing and is_complete(runs_dir / run_name, cfg):
                print(f"[ablation] ({i}/{len(todo)}) skipping completed run {run_name}", flush=True)
                continue
            # loaders depend only on the data/protein config -> cache across seeds and variants
            key = json.dumps({"data": cfg["data"], "protein": cfg["protein"], "bs": cfg["training"]["batch_size"]}, sort_keys=True)
            if key not in loader_cache:
                loader_cache[key] = build_dataloaders(cfg, ROOT)
            loaders, info = loader_cache[key]
            print(f"\n=== ({i}/{len(todo)}) {run_name} | {time.strftime('%H:%M:%S')} | elapsed {(time.time() - t_all) / 60:.1f} min ===", flush=True)
            res = train_from_config(cfg, ROOT, run_name=run_name, loaders=loaders, data_info=info)
            print(f"[ablation] {run_name}: test mse {res['test']['mse']:.4f} ci {res['test']['ci']:.4f} | {res['epochs_run']} epochs, {res['train_time_s'] / 60:.1f} min", flush=True)
            # incremental checkpoint of the study: the CSV is always in sync with the finished runs on disk
            write_results_csv(runs_dir, all_stems, args.seeds, results_csv)

    if args.no_aggregate:
        return
    # aggregation always reads from disk so that several parallel workers produce one consistent table
    df = write_results_csv(runs_dir, all_stems, args.seeds, results_csv)
    if df.empty:
        print("no completed runs found")
        return
    df = aggregate(results_csv, table_md)
    table_text = table_md.read_text(encoding="utf-8")
    try:
        print(table_text)
    except UnicodeEncodeError:
        sys.stdout.buffer.write(table_text.encode("utf-8", errors="replace"))
        sys.stdout.buffer.write(b"\n")
    print(f"[ablation] {len(df)} runs, wall time {(time.time() - t_all) / 60:.1f} min -> {results_csv}, {table_md}")


if __name__ == "__main__":
    main()
