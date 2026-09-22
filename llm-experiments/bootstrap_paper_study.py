"""Estimate Price-accounting uncertainty from saved measurements; no GPU required."""
import argparse
import json

from src.bootstrap import bootstrap_measurement


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--measurement', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--draws', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=2026092200)
    args = parser.parse_args()
    print(json.dumps(bootstrap_measurement(args.measurement, args.output, args.draws, args.seed), indent=2))
