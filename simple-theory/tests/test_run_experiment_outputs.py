import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_experiment


def test_metrics_json_includes_price_check_arrays(tmp_path, monkeypatch):
    config = {
        "TabularPolicyConfig": {"N": 32, "K": 4, "G": 2, "mode": "hidden_quality"},
        "HiddenQualityConfig": {"gamma": 0.5, "p": 0.3, "alpha": 1.0},
        "NeuralPolicyConfig": {"N_eval": 16, "d": 8},
        "TrainConfig": {"steps": 2, "batch_size": 4, "num_runs": 1, "seed": 11, "eta": 0.3},
        "NeuralTrainConfig": {"steps": 2, "batch_size": 4, "num_runs": 1, "seed": 11, "lr": 0.02},
        "OutputConfig": {"results_dir": str(tmp_path / "results"), "name": "tiny"},
        "PriceCheckConfig": {"samples": 8},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_experiment.py", "--config", str(config_path), "--mode", "tabular"],
    )

    run_experiment.main()

    metrics_path = tmp_path / "results" / "tiny" / "tabular" / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    for key in [
        "steps_axis",
        "observed_cum_mean",
        "observed_cum_sem",
        "predicted_cum_mean",
        "predicted_cum_sem",
    ]:
        assert key in metrics
        assert len(metrics[key]) == config["TrainConfig"]["steps"] + 1


def test_trait_plot_lower_band_is_not_clipped_to_zero(tmp_path, monkeypatch):
    captured_lower_bands = []

    def capture_fill_between(_self, _x, y1, _y2, *args, **kwargs):
        captured_lower_bands.append(np.asarray(y1))

    monkeypatch.setattr(run_experiment.plt.Axes, "fill_between", capture_fill_between)
    monkeypatch.setattr(run_experiment.plt, "savefig", lambda *args, **kwargs: None)

    run_experiment.plot_trait(
        steps_axis=np.array([0, 1]),
        trait_mean=np.array([0.1, -0.2]),
        trait_sem=np.array([0.3, 0.4]),
        color="blue",
        output_dir=tmp_path,
        stem="trait",
    )

    assert captured_lower_bands
    assert np.isclose(captured_lower_bands[0][0], -0.2)
    assert np.isclose(captured_lower_bands[0][1], -0.6)
