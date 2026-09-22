"""Re-render the Phase-3 plots from an existing price_eval.jsonl -- pure CPU, no re-sampling.

Use when you want to change what is DISPLAYED (drop features from the grid, retitle, restyle)
without paying for another GPU Phase-3 run. --drop applies everywhere, including the pooled
scatter's corr/slope, so the reported fit always describes exactly the features shown.

  uv run python -m analysis.replot --run results/real_lr1e-4 --drop 30882,22348,22996
  uv run python -m analysis.replot --run results/real_lr1e-4 --suffix _svamp --drop 30882
"""
import argparse
import logging
from pathlib import Path

from src.plotting import plot_from_jsonl

LOGGER = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run dir holding price_eval<suffix>.jsonl")
    # --suffix mirrors run_price_eval --out-suffix: '' -> price_eval.jsonl / plots/,
    # '_svamp' -> price_eval_svamp.jsonl / plots_svamp/.
    ap.add_argument("--suffix", default="")
    ap.add_argument("--drop", default="", help="comma-separated feature ids to exclude")
    ap.add_argument("--out", default=None, help="output dir (default <run>/plots<suffix>)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    run = Path(args.run)
    jsonl = run / f"price_eval{args.suffix}.jsonl"
    if not jsonl.exists():
        raise SystemExit(f"no such file: {jsonl}")
    drop = [int(x) for x in args.drop.replace(",", " ").split()]
    out = Path(args.out) if args.out else run / f"plots{args.suffix}"

    LOGGER.info(f"replotting {jsonl} -> {out}" + (f"  (dropping {drop})" if drop else ""))
    plot_from_jsonl(jsonl, out, drop=drop)
    LOGGER.info(f"wrote {sorted(p.name for p in out.glob('*.p*'))}")


if __name__ == "__main__":
    main()
