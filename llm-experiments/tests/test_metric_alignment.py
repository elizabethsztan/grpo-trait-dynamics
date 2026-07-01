from types import SimpleNamespace

import numpy as np

from src.metrics import summarize_trait_metrics
from src.plotting import _carry_series, _summary_values
from train_grpo_price import _latest_eval_wrong_hint, _record_step_for_update


def test_record_step_for_update_labels_post_update_policy_step():
    assert _record_step_for_update(0) == 1
    assert _record_step_for_update(4) == 5


def test_invalid_hint_present_completion_counts_as_non_agreement():
    metrics = [
        SimpleNamespace(
            correct=False,
            output_agreement=True,
            hint_is_correct=True,
            invalid_output=False,
            completion_token_length=3,
            activation_agreement=None,
            parsed_choice="A",
            sycophantic_error=False,
            correct_disagreement=False,
        ),
        SimpleNamespace(
            correct=False,
            output_agreement=False,
            hint_is_correct=True,
            invalid_output=True,
            completion_token_length=3,
            activation_agreement=None,
            parsed_choice=None,
            sycophantic_error=False,
            correct_disagreement=False,
        ),
    ]

    summary = summarize_trait_metrics(metrics)

    assert summary["agreement_rate"] == 0.5


def test_no_hint_completions_do_not_enter_agreement_denominator():
    metrics = [
        SimpleNamespace(
            correct=False,
            output_agreement=False,
            hint_is_correct=None,
            invalid_output=True,
            completion_token_length=3,
            activation_agreement=None,
            parsed_choice=None,
            sycophantic_error=False,
            correct_disagreement=False,
        )
    ]

    summary = summarize_trait_metrics(metrics)

    assert summary["agreement_rate"] == 0.0


def test_carry_series_uses_nan_until_first_real_value_then_carries():
    rows = [
        {"step": 0},
        {"step": 1, "observed_eval": {"eval_wrong_hint": {"agreement_rate": 0.25}}},
        {"step": 2},
        {"step": 3, "observed_eval": {"eval_wrong_hint": {"agreement_rate": 0.75}}},
    ]

    values = _carry_series(rows, lambda row: row["observed_eval"]["eval_wrong_hint"]["agreement_rate"])

    assert np.isnan(values[0])
    assert values[1:] == [0.25, 0.25, 0.75]


def test_reliability_sweep_uses_latest_available_observed_eval():
    rows = [
        {"step": 0, "observed_eval": {"eval_wrong_hint": {"wrong_hint_agreement_rate": 0.25}}},
        {"step": 1, "observed_eval": {}},
        {"step": 2, "observed_eval": {}},
    ]

    observed_wrong = _latest_eval_wrong_hint(rows)

    assert observed_wrong["wrong_hint_agreement_rate"] == 0.25


def test_reliability_sweep_plot_missing_summary_values_as_nan():
    runs = [
        {"train_hint_correct_probability": 0.1, "final_wrong_hint_agreement_rate": 0.25},
        {"train_hint_correct_probability": 0.5},
    ]

    values = _summary_values(runs, "final_wrong_hint_agreement_rate")

    assert values[0] == 0.25
    assert np.isnan(values[1])
