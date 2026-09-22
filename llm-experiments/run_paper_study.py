"""Prepare a study, train one run, or measure a saved checkpoint sequence."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.config import load_config
from src.study import prepare_study, read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="save a manifest and evaluation bank; no model loading")
    prepare.add_argument("--config", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--cohort", choices=("pilot", "paper"), required=True)
    for name in ("train", "measure"):
        command = commands.add_parser(name)
        command.add_argument("--study", required=True)
        command.add_argument("--run-id", required=True)
        if name == "measure":
            command.add_argument("--measurement-id", required=True)
            command.add_argument("--prompts", type=int, help="use a larger prefix of the saved bank without retraining")
    status = commands.add_parser("status")
    status.add_argument("--study", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        manifest = prepare_study(load_config(args.config), args.output, args.cohort)
        print(f"Prepared {len(manifest['runs'])} {args.cohort} run entries in {args.output}")
        for run in manifest["runs"]:
            print(run["run_id"])
    elif args.command == "status":
        root = Path(args.study)
        for run in read_json(root / "manifest.json")["runs"]:
            path = root / "runs" / run["run_id"] / "status.json"
            print(run["run_id"], read_json(path)["state"] if path.exists() else "not_started")
            for measurement in sorted((root / "measurements" / run["run_id"]).glob("*/status.json")):
                print("  ", measurement.parent.name, read_json(measurement)["state"])
    else:
        from src.study_runner import measure_run, train_run

        if args.command == "train":
            print(train_run(args.study, args.run_id))
        else:
            print(measure_run(args.study, args.run_id, args.measurement_id, args.prompts))


if __name__ == "__main__":
    main()
