import argparse
from pathlib import Path

from . import pipeline


def main():
    parser = argparse.ArgumentParser(description="Offline per-run closure diagnostics; no fitting or training.")
    parser.add_argument("--registry", type=Path, default=Path(__file__).parent / "configs/existing.yaml")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pipeline.run(args.registry, args.output)


if __name__ == "__main__":
    main()
