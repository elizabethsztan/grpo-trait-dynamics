import pytest

from validate_activation_probe import _group_mean_diagnostics, _natural_rollout_score_metrics, _selected_layer_result


def test_natural_rollout_score_metrics_computes_correlation_and_auc_on_valid_outputs():
    metrics = _natural_rollout_score_metrics(
        scores=[-1.0, 0.2, 1.0, 100.0],
        output_agreements=[False, True, True, False],
        strict_valid=[True, True, True, False],
    )

    assert metrics["n_valid"] == 3
    assert metrics["auc"] == pytest.approx(1.0)
    assert metrics["correlation"] > 0.0


def test_group_mean_diagnostics_reports_largest_group_gap():
    diagnostics = _group_mean_diagnostics(
        [
            {"completion_choice": "A", "score": 1.0},
            {"completion_choice": "A", "score": 3.0},
            {"completion_choice": "B", "score": -1.0},
        ],
        group_key="completion_choice",
    )

    assert diagnostics["means"] == {"A": 2.0, "B": -1.0}
    assert diagnostics["max_mean_gap"] == pytest.approx(3.0)


def test_selected_layer_result_raises_when_selected_layer_missing_from_sweep():
    with pytest.raises(ValueError, match="selected_hook_layer=12"):
        _selected_layer_result([{"layer": 4}, {"layer": 8}], selected_layer=12)
