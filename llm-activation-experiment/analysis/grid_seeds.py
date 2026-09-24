"""grid_price across GRPO seeds: every seed's cumulative observed-ΔT / Price(cov) curve
drawn light, the across-seed mean in bold. Pure CPU -- re-reads existing price_eval.jsonl.

Panel order and labels come from the FIRST run's features.json (all seed replicates share
the same frozen SAE features). --drop excludes feature ids from the grid.

  uv run python -m analysis.grid_seeds --runs real_lr1e-4 real_lr1e-4_s2 real_lr1e-4_s3
  uv run python -m analysis.grid_seeds --runs real_lr1e-4 real_lr1e-4_s2 real_lr1e-4_s3 --drop 30882

--row draws the paper headline instead: one row of panels, by default the first feature of
each group (+#1, −#1, ctrl); --features picks them explicitly. Panels share a y-EXTENT (not a
y-range) so drifts are comparable; --free-y autoscales each panel instead.

  uv run python -m analysis.grid_seeds --runs real_lr1e-4 real_lr1e-4_s2 real_lr1e-4_s3 \
      --row --features 30882,22348,22996
"""
import argparse
import logging
from pathlib import Path

from src.plotting import plot_grid_seeds, plot_row_seeds

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
    ap.add_argument("--out", default=None, help="output dir (default <results>/plots_seeds)")
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
    out = Path(args.out) if args.out else root / "plots_seeds"
    stem = f"{'row' if args.row else 'grid'}_price_seeds{args.suffix}"
    LOGGER.info(f"{len(paths)} seeds -> {out / stem}" + (f"  (dropping {drop})" if drop else ""))
    if args.row:
        feats = [int(x) for x in args.features.replace(",", " ").split()]
        plot_row_seeds(paths, out, drop=drop, stem=stem, estimator=args.estimator,
                       features=feats or None, same_yspan=not args.free_y)
    else:
        plot_grid_seeds(paths, out, drop=drop, stem=stem, estimator=args.estimator)


if __name__ == "__main__":
    main()
