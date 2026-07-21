from __future__ import annotations

import math
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

import analyze_price_fit as price_fit
from analyze_price_fit import (
    score_command_exit_code,
    score_price_fit,
    select_calibration,
    validate_reference,
    validate_run_protocol,
)
from src.config import load_config, save_config
import src.plotting as plotting
from src.plotting import price_plot_series
from train_grpo_price import _configure_cuda_allocator


DISTRIBUTIONS = ("eval_balanced_hint", "eval_wrong_hint")


def _metrics(
    observed_by_step: dict[int, float],
    predicted_by_step: dict[int, float],
    *,
    final_agreement: float = 1.0,
    final_invalid: float = 0.0,
    price_n: int = 10,
) -> list[dict]:
    rows = []
    baseline = 0.2
    final_step = max(predicted_by_step)
    for step in range(final_step + 1):
        observed = {}
        if step == 0 or step in observed_by_step:
            drift = 0.0 if step == 0 else observed_by_step[step]
            agreement = baseline + drift
            if step == final_step:
                agreement = final_agreement
                drift = final_agreement - baseline
            observed = {
                distribution: {
                    "agreement_rate": agreement,
                    "invalid_output_rate": final_invalid if step == final_step else 0.0,
                    "output_agreement_observed_drift": drift,
                }
                for distribution in DISTRIBUTIONS
            }
        price = {}
        if step > 0:
            price = {
                distribution: {
                    "output_agreement": {
                        "cov_cum": predicted_by_step[step],
                        "ess": 0.8 * price_n,
                        "n": price_n,
                        "mean_omega": 1.1,
                        "max_omega": 2.0,
                    }
                }
                for distribution in DISTRIBUTIONS
            }
        rows.append({"step": step, "observed_eval": observed, "price": price})
    return rows


def test_price_fit_uses_only_real_observed_checkpoints():
    observed = {5: 0.4, 10: 0.8}
    reference_predicted = {step: (0.4 if step <= 5 else 0.8) for step in range(1, 11)}
    candidate_predicted = dict(reference_predicted)
    candidate_predicted[2] = 99.0
    candidate_predicted[7] = -99.0

    reference = _metrics(observed, reference_predicted)
    candidate = _metrics(observed, candidate_predicted)
    report = score_price_fit(reference, candidate, "candidate", learning_rate=2e-5)

    for distribution in DISTRIBUTIONS:
        assert report["distributions"][distribution]["candidate"]["rmse"] == 0.0
        assert report["distributions"][distribution]["candidate"]["final_abs_residual"] == 0.0
        assert report["distributions"][distribution]["candidate"]["steps"] == [0, 5, 10]
    assert report["passed"] is True


def test_price_fit_restricts_reference_to_candidate_horizon():
    reference_observed = {step: min(step / 60, 0.8) for step in range(5, 101, 5)}
    reference_predicted = {
        step: min(step / 60, 0.8) if step <= 60 else 10.0
        for step in range(1, 101)
    }
    candidate_observed = {step: min(step / 60, 0.8) for step in range(5, 61, 5)}
    candidate_predicted = {step: min(step / 60, 0.8) for step in range(1, 61)}

    report = score_price_fit(
        _metrics(reference_observed, reference_predicted),
        _metrics(candidate_observed, candidate_predicted),
        "candidate",
        learning_rate=2e-5,
    )

    for distribution in DISTRIBUTIONS:
        block = report["distributions"][distribution]
        assert block["reference"]["steps"] == list(range(0, 61, 5))
        assert block["candidate"]["steps"] == list(range(0, 61, 5))
        assert block["reference"]["rmse"] == pytest.approx(0.0)


def test_price_fit_applies_reference_margin_and_behavior_gates():
    observed = {5: 0.4, 10: 0.8}
    reference = _metrics(observed, {step: (0.45 if step <= 5 else 0.85) for step in range(1, 11)})
    candidate = _metrics(observed, {step: (0.46 if step <= 5 else 0.86) for step in range(1, 11)})

    report = score_price_fit(reference, candidate, "candidate", learning_rate=2e-5, margin=0.02)
    for distribution in DISTRIBUTIONS:
        block = report["distributions"][distribution]
        assert math.isclose(block["thresholds"]["rmse"], block["reference"]["rmse"] + 0.02)
        assert math.isclose(
            block["thresholds"]["final_abs_residual"],
            block["reference"]["final_abs_residual"] + 0.02,
        )
        assert block["gates"]["rmse"] is True
        assert block["gates"]["final_abs_residual"] is True
    assert report["behavior_passed"] is True

    invalid = _metrics(
        observed,
        {step: (0.46 if step <= 5 else 0.86) for step in range(1, 11)},
        final_invalid=0.1,
    )
    invalid_report = score_price_fit(reference, invalid, "invalid", learning_rate=2e-5)
    assert invalid_report["behavior_passed"] is False
    assert invalid_report["passed"] is False


def test_calibration_selection_uses_score_then_lower_lr_tie_break(monkeypatch):
    def metadata(learning_rate):
        return {
            "seed": 290403,
            "num_steps": 60,
            "learning_rate": learning_rate,
            "layer_scope": "all",
            "lora_layer_indices": list(range(24)),
            "price_distributions": list(DISTRIBUTIONS),
            "price_samples_per_distribution": 256,
            "cuda_allocator_config": "expandable_segments:True",
            "run_dir": f"results/calibration_lr{int(learning_rate / 1e-5)}",
                "expected_config": (
                    "configs/qwen25_05b_sycophancy_full_lora_price_cal_"
                    f"lr{int(learning_rate / 1e-5)}e5_n256_seed290403_retry1.yaml"
                ),
        }

    config_dir = Path(__file__).parents[1]
    config_hashes = {
        learning_rate: hashlib.sha256(
            (config_dir / metadata(learning_rate)["expected_config"]).read_bytes()
        ).hexdigest()
        for learning_rate in (1e-5, 2e-5, 3e-5)
    }

    def report(label, learning_rate, score):
        fit_passed = score <= 1.0
        run_metadata = metadata(learning_rate)
        run_metadata["expected_config_sha256"] = config_hashes[learning_rate]
        distribution_reports = {
            distribution: {
                "reference": {"rmse": 0.98, "final_abs_residual": 0.98},
                "candidate": {"rmse": score, "final_abs_residual": score},
                "thresholds": {"rmse": 1.0, "final_abs_residual": 1.0},
                "gates": {"rmse": fit_passed, "final_abs_residual": fit_passed},
                "final_agreement": 1.0,
                "final_invalid_output_rate": 0.0,
            }
            for distribution in DISTRIBUTIONS
        }
        return {
            "label": label,
            "learning_rate": learning_rate,
            "margin": 0.02,
            "passed": fit_passed,
            "behavior_passed": True,
            "selection_score": score,
            "behavior_gates": {
                **{
                    f"{distribution}_final_agreement": True
                    for distribution in DISTRIBUTIONS
                },
                **{
                    f"{distribution}_final_invalid": True
                    for distribution in DISTRIBUTIONS
                },
            },
            "distributions": distribution_reports,
            "run_metadata": run_metadata,
            "reference_metadata": reference_metadata,
        }

    reference_metadata = {
        "reference_dir": "results/sycophancy_main",
        "manifest_sha256": (
            "d0cdfd50420be8e21611fa5ae4a79e6de94b331178e2ee770b51d140241f97fb"
        ),
        "manifest_file_count": 114,
    }

    reports = [
        report("lr3", 3e-5, 0.70),
        report("lr2", 2e-5, 0.73),
        report("lr1", 1e-5, 1.20),
    ]
    source_reports = {item["label"]: deepcopy(item) for item in reports}
    monkeypatch.setattr(
        price_fit,
        "rescore_report_from_source",
        lambda item: source_reports[item["label"]],
        raising=False,
    )
    selection = select_calibration(reports, tie_tolerance=0.05)
    assert selection["selected_label"] == "lr2"
    assert selection["fallback_label"] == "lr3"

    with pytest.raises(ValueError, match="exactly three"):
        select_calibration(reports[:1])

    wrong_seed = deepcopy(reports)
    wrong_seed[0]["run_metadata"]["seed"] = 290402
    with pytest.raises(ValueError, match="seed"):
        select_calibration(wrong_seed)

    with pytest.raises(ValueError, match="tie tolerance"):
        select_calibration(reports, tie_tolerance=1.0)

    wrong_reference = deepcopy(reports)
    wrong_reference[0]["reference_metadata"] = dict(wrong_reference[0]["reference_metadata"])
    wrong_reference[0]["reference_metadata"]["manifest_sha256"] = "wrong"
    with pytest.raises(ValueError, match="approved historical reference"):
        select_calibration(wrong_reference)

    wrong_config_hash = deepcopy(reports)
    wrong_config_hash[0]["run_metadata"]["expected_config_sha256"] = "wrong"
    with pytest.raises(ValueError, match="canonical config hash"):
        select_calibration(wrong_config_hash)

    wrong_pass = deepcopy(reports)
    wrong_pass[0]["passed"] = False
    with pytest.raises(ValueError, match="passed field"):
        select_calibration(wrong_pass)

    wrong_score = deepcopy(reports)
    wrong_score[0]["selection_score"] = 0.01
    with pytest.raises(ValueError, match="selection_score"):
        select_calibration(wrong_score)

    coherent_tamper = deepcopy(reports)
    coherent_tamper[0]["selection_score"] = 0.0
    for distribution in DISTRIBUTIONS:
        block = coherent_tamper[0]["distributions"][distribution]
        block["candidate"]["rmse"] = 0.0
        block["candidate"]["final_abs_residual"] = 0.0
        block["gates"] = {"rmse": True, "final_abs_residual": True}
    with pytest.raises(ValueError, match="source metrics"):
        select_calibration(coherent_tamper)


def test_score_command_exit_code_reflects_fit_gate():
    assert score_command_exit_code({"passed": True}) == 0
    assert score_command_exit_code({"passed": False}) == 1


def test_run_protocol_validation_checks_resolved_config_summary_and_metrics(tmp_path):
    source = (
        Path(__file__).parents[1]
        / "configs"
        / "qwen25_05b_sycophancy_full_lora_price_cal_lr2e5_n256_seed290403.yaml"
    )
    config = load_config(source)
    run_dir = tmp_path / config["RunConfig"]["name"]
    run_dir.mkdir()
    save_config(config, run_dir / "config.yaml")
    summary = {
        "run_name": config["RunConfig"]["name"],
        "num_steps": 60,
        "lora_layer_scope": "all",
        "lora_layer_indices": list(range(24)),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary))
    rows = _metrics(
        {step: min(step / 60, 0.8) for step in range(5, 61, 5)},
        {step: min(step / 60, 0.8) for step in range(1, 61)},
        price_n=256,
    )
    (run_dir / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))

    metadata = validate_run_protocol(
        run_dir,
        expected_seed=290403,
        expected_steps=60,
        expected_learning_rate=2e-5,
        expected_price_samples=256,
        expected_config=source,
    )
    assert metadata["price_samples_per_distribution"] == 256
    assert metadata["lora_layer_indices"] == list(range(24))

    with pytest.raises(ValueError, match="learning rate"):
        validate_run_protocol(
            run_dir,
            expected_seed=290403,
            expected_steps=60,
            expected_learning_rate=3e-5,
            expected_price_samples=256,
            expected_config=source,
        )

    wrong_observed_config = deepcopy(config)
    wrong_observed_config["ObservedEvalConfig"]["prompts_per_distribution"] = 1
    save_config(wrong_observed_config, run_dir / "config.yaml")
    with pytest.raises(ValueError, match="ObservedEvalConfig"):
        validate_run_protocol(
            run_dir,
            expected_seed=290403,
            expected_steps=60,
            expected_learning_rate=2e-5,
            expected_price_samples=256,
            expected_config=source,
        )
    save_config(config, run_dir / "config.yaml")

    rows[5]["observed_eval"] = {}
    (run_dir / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="observed checkpoints"):
        validate_run_protocol(
            run_dir,
            expected_seed=290403,
            expected_steps=60,
            expected_learning_rate=2e-5,
            expected_price_samples=256,
            expected_config=source,
        )
    rows = _metrics(
        {step: min(step / 60, 0.8) for step in range(5, 61, 5)},
        {step: min(step / 60, 0.8) for step in range(1, 61)},
        price_n=256,
    )

    rows[5]["observed_eval"]["eval_wrong_hint"].pop("output_agreement_observed_drift")
    (run_dir / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="scoring fields"):
        validate_run_protocol(
            run_dir,
            expected_seed=290403,
            expected_steps=60,
            expected_learning_rate=2e-5,
            expected_price_samples=256,
            expected_config=source,
        )
    rows = _metrics(
        {step: min(step / 60, 0.8) for step in range(5, 61, 5)},
        {step: min(step / 60, 0.8) for step in range(1, 61)},
        price_n=256,
    )

    rows[10]["price"].pop("eval_wrong_hint")
    (run_dir / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="Price block"):
        validate_run_protocol(
            run_dir,
            expected_seed=290403,
            expected_steps=60,
            expected_learning_rate=2e-5,
            expected_price_samples=256,
            expected_config=source,
        )


def test_reference_validation_is_anchored_to_tracked_manifest(tmp_path):
    metadata = validate_reference(
        "results/sycophancy_main",
        "results_output_lora_summary/reference_manifest.json",
    )
    assert metadata["reference_dir"] == "results/sycophancy_main"
    assert metadata["manifest_file_count"] == 114
    assert metadata["manifest_sha256"] == (
        "d0cdfd50420be8e21611fa5ae4a79e6de94b331178e2ee770b51d140241f97fb"
    )

    unapproved = tmp_path / "reference_manifest.json"
    unapproved.write_text("{}")
    with pytest.raises(ValueError, match="approved tracked manifest"):
        validate_reference("results/sycophancy_main", unapproved)


def test_price_plot_series_does_not_carry_sparse_observations():
    metrics = _metrics(
        {5: 0.4, 10: 0.8},
        {step: 0.08 * step for step in range(1, 11)},
    )
    observed, predicted, residual = price_plot_series(
        metrics,
        "eval_wrong_hint",
        "output_agreement",
        "agreement_rate",
    )
    assert math.isnan(observed[1])
    assert math.isnan(residual[1])
    assert predicted[1] == 0.08
    assert observed[5] == pytest.approx(0.4)
    assert residual[5] == pytest.approx(0.0)


def test_standard_plots_do_not_carry_sparse_observed_values(tmp_path, monkeypatch):
    metrics = _metrics(
        {5: 0.4},
        {step: 0.08 * step for step in range(1, 6)},
    )
    for step in (0, 5):
        block = metrics[step]["observed_eval"]["eval_wrong_hint"]
        block.update(
            {
                "wrong_hint_agreement_rate": 0.1 + step / 10,
                "sycophantic_error_rate": 0.2 + step / 10,
                "correct_disagreement_rate": 0.3 + step / 10,
                "mean_completion_token_length": 10 + step,
            }
        )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in metrics)
    )
    captured = {}

    def capture(path, x, series, ylabel):
        captured[path.name] = series

    monkeypatch.setattr(plotting, "_save_line_plot", capture)
    plotting.plot_run(run_dir)

    wrong_hint = dict(captured["wrong_hint_sycophancy.png"])
    length = dict(captured["length_control.png"])
    assert math.isnan(wrong_hint["Wrong-hint agreement"][1])
    assert math.isnan(wrong_hint["Sycophantic error"][1])
    assert math.isnan(length["Wrong-hint eval length"][1])


def test_calibration_configs_differ_only_by_name_and_learning_rate():
    config_dir = Path(__file__).parents[1] / "configs"
    paths = [
        config_dir / f"qwen25_05b_sycophancy_full_lora_price_cal_lr{lr}e5_n256_seed290403_retry1.yaml"
        for lr in (1, 2, 3)
    ]
    configs = [load_config(path) for path in paths]
    for config, expected_lr in zip(configs, (1e-5, 2e-5, 3e-5)):
        assert config["RunConfig"]["seed"] == 290403
        assert config["RunConfig"]["fail_if_exists"] is True
        assert config["RunConfig"]["cuda_allocator_expandable_segments"] is True
        assert config["RunConfig"]["name"].endswith("_retry1")
        assert config["LoRAConfig"]["layer_scope"] == "all"
        assert config["ActivationProbeConfig"]["enabled"] is False
        assert config["TrainConfig"]["num_steps"] == 60
        assert config["TrainConfig"]["learning_rate"] == expected_lr
        assert config["PriceConfig"]["eval_distributions"] == [
            "eval_balanced_hint",
            "eval_wrong_hint",
        ]
        assert config["PriceConfig"]["prompts_per_distribution"] == 128
        assert config["PriceConfig"]["completions_per_prompt"] == 2
        assert config["ObservedEvalConfig"]["eval_every"] == 5
        assert config["ObservedEvalConfig"]["prompts_per_distribution"] == 128

    normalized = []
    for config in configs:
        candidate = deepcopy(config)
        candidate["RunConfig"].pop("name")
        candidate["TrainConfig"].pop("learning_rate")
        normalized.append(candidate)
    assert normalized[0] == normalized[1] == normalized[2]


def test_price_calibration_smoke_uses_full_coverage_and_target_sampling():
    config_path = (
        Path(__file__).parents[1]
        / "configs"
        / "qwen25_05b_sycophancy_full_lora_price_smoke_lr2e5_n256_seed290403.yaml"
    )
    config = load_config(config_path)
    assert config["RunConfig"]["fail_if_exists"] is True
    assert config["LoRAConfig"]["layer_scope"] == "all"
    assert config["TrainConfig"]["num_steps"] == 2
    assert config["TrainConfig"]["learning_rate"] == 2e-5
    assert config["PriceConfig"]["eval_distributions"] == [
        "eval_balanced_hint",
        "eval_wrong_hint",
    ]
    assert config["PriceConfig"]["prompts_per_distribution"] == 128
    assert config["PriceConfig"]["completions_per_prompt"] == 2


def test_price_calibration_memory_smoke_reproduces_full_training_batch():
    config_path = (
        Path(__file__).parents[1]
        / "configs"
        / "qwen25_05b_sycophancy_full_lora_price_memory_smoke_lr2e5_n256_seed290403.yaml"
    )
    config = load_config(config_path)
    assert config["RunConfig"]["fail_if_exists"] is True
    assert config["LoRAConfig"]["layer_scope"] == "all"
    assert config["TrainConfig"]["num_steps"] == 4
    assert config["TrainConfig"]["train_prompts_per_step"] == 16
    assert config["TrainConfig"]["group_size"] == 8
    assert config["TrainConfig"]["learning_rate"] == 2e-5
    assert config["PriceConfig"]["eval_distributions"] == [
        "eval_balanced_hint",
        "eval_wrong_hint",
    ]
    assert config["PriceConfig"]["prompts_per_distribution"] == 128
    assert config["PriceConfig"]["completions_per_prompt"] == 2


def test_configure_cuda_allocator_runs_before_torch_import(monkeypatch):
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    effective = _configure_cuda_allocator({"cuda_allocator_expandable_segments": True})
    assert effective == "expandable_segments:True"
    assert (
        __import__("os").environ["PYTORCH_CUDA_ALLOC_CONF"]
        == "expandable_segments:True"
    )

    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", " expandable_segments:True ")
    effective = _configure_cuda_allocator({"cuda_allocator_expandable_segments": True})
    assert effective == "expandable_segments:True"
    assert __import__("os").environ["PYTORCH_CUDA_ALLOC_CONF"] == effective

    for conflicting in (
        "expandable_segments:False",
        "max_split_size_mb:128",
        "max_split_size_mb:128, expandable_segments:False",
    ):
        monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", conflicting)
        with pytest.raises(ValueError, match="requires PYTORCH_CUDA_ALLOC_CONF"):
            _configure_cuda_allocator({"cuda_allocator_expandable_segments": True})

    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    assert _configure_cuda_allocator({"cuda_allocator_expandable_segments": False}) is None
    assert __import__("os").environ["PYTORCH_CUDA_ALLOC_CONF"] == "max_split_size_mb:128"
