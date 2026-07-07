from __future__ import annotations

import argparse
from pathlib import Path

from src.plotting import plot_reliability_sweep, plot_run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    written = []
    if (run_dir / "metrics.jsonl").exists():
        written.extend(plot_run(run_dir))
    if (run_dir / "summary.json").exists():
        written.extend(plot_reliability_sweep(run_dir))
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
