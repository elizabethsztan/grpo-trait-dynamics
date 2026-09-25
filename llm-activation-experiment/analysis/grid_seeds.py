"""Across-seed activation-trait figures from saved Price-evaluation JSONL files.

The default is the cumulative observed-vs-selection grid used for frozen
traits.  ``--decomp`` draws the corresponding four-line decomposition for
all-layers runs: observed change, selection plus transmission, selection,
and transmission.  Individual seeds are faint and their mean is bold.

Examples:

  uv run python -m analysis.grid_seeds \
      --runs real_lr1e-4 real_lr1e-4_s2 real_lr1e-4_s3

  uv run python -m analysis.grid_seeds --decomp \
      --runs real_lr1e-4_alllayers real_lr1e-4_alllayers_s2 real_lr1e-4_alllayers_s3

``--row`` draws the three-panel frozen-trait headline instead; ``--features``
selects its panels explicitly.
"""
import argparse
import logging
from pathlib import Path

from src.plotting import plot_grid_decomp_seeds, plot_grid_seeds, plot_row_seeds

LOGGER = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results", help="results root")
    ap.add_argument("--runs", nargs="+", required=True, help="run-dir names, one per seed")
    ap.add_argument("--suffix", default="", help="'' -> price_eval.jsonl, '_svamp' -> price_eval_svamp.jsonl")
    ap.add_argument("--drop", default="", help="comma-separated feature ids to exclude")
    ap.add_argument("--estimator", default=None, choices=["cov", "sn"],
                    help="Price estimator to draw: raw cov or self-normalised. Default: sn for "
                         "all-layers runs (transmission logged, degenerate omega), cov otherwise.")
    ap.add_argument("--out", default=None,
                    help="output dir (default <results>/plots_seeds_trans for --decomp, "
                         "otherwise <results>/plots_seeds)")
    ap.add_argument("--decomp", action="store_true",
                    help="all-seed cumulative selection/transmission decomposition grid")
    ap.add_argument("--row", action="store_true",
                    help="headline row: first feature of each group, 1xK, equal y-extents")
    ap.add_argument("--features", default="",
                    help="(--row only) comma-separated feature ids to show, in panel order")
    ap.add_argument("--free-y", action="store_true",
                    help="(--row only) let each panel autoscale y instead of equal extents")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    root = Path(args.results)
    paths = [root / r / f"price_eval{args.suffix}.jsonl" for r in args.runs]
    for p in paths:
        if not p.exists():
            raise SystemExit(f"no such file: {p}")
    drop = [int(x) for x in args.drop.replace(",", " ").split()]
    if args.row and args.decomp:
        raise SystemExit("--row and --decomp are mutually exclusive")
    default_out = "plots_seeds_trans" if args.decomp else "plots_seeds"
    out = Path(args.out) if args.out else root / default_out
    kind = "row_price" if args.row else ("grid_decomp" if args.decomp else "grid_price")
    stem = f"{kind}_seeds{args.suffix}"
    LOGGER.info(f"{len(paths)} seeds -> {out / stem}" + (f"  (dropping {drop})" if drop else ""))
    if args.row:
        feats = [int(x) for x in args.features.replace(",", " ").split()]
        plot_row_seeds(paths, out, drop=drop, stem=stem, estimator=args.estimator,
                       features=feats or None, same_yspan=not args.free_y)
    elif args.decomp:
        plot_grid_decomp_seeds(paths, out, drop=drop, stem=stem, estimator=args.estimator)
    else:
        plot_grid_seeds(paths, out, drop=drop, stem=stem, estimator=args.estimator)


if __name__ == "__main__":
    main()
