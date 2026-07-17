from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from compare_output_runs import build_manifest, compare_metrics, verify_manifest, write_manifest
from src.config import load_config
from src.lora_freeze import assert_only_lora_layers_trainable, resolve_lora_layers
from train_grpo_price import prepare_run_dir


class _Parameter:
    def __init__(self, requires_grad: bool):
        self.requires_grad = requires_grad


class _NamedParameterModel:
    def __init__(self, names: list[str]):
        self._parameters = [(name, _Parameter(True)) for name in names]

    def named_parameters(self):
        return list(self._parameters)


def _synthetic_metrics(
    final_agreement: float = 1.0,
    final_no_hint_accuracy: float = 0.2,
    final_no_hint_invalid: float = 0.2,
) -> list[dict]:
    rows = []
    baseline = 0.18
    for step in range(101):
        observed = {}
        if step % 5 == 0:
            progress = min(step / 25.0, 1.0)
            agreement = baseline + (final_agreement - baseline) * progress
            observed = {
                "eval_wrong_hint": {
                    "agreement_rate": agreement,
                    "accuracy": 0.1 * (1.0 - progress),
                    "invalid_output_rate": 0.58 * (1.0 - progress),
                    "output_agreement_observed_drift": agreement - baseline,
                },
                "eval_no_hint": {
                    "accuracy": final_no_hint_accuracy,
                    "invalid_output_rate": final_no_hint_invalid,
                },
            }
        price = {}
        if step > 0:
            progress = min(step / 25.0, 1.0)
            price = {
                "eval_wrong_hint": {
                    "output_agreement": {
                        "cov_cum": (final_agreement - baseline) * progress + 0.05 * progress,
                    },
                    "output_agreement_shuffled": {"cov_cum": 0.01 * progress},
                }
            }
        rows.append({"step": step, "observed_eval": observed, "price": price})
    return rows


def test_resolve_lora_layers_for_restricted_and_full_model():
    assert resolve_lora_layers(24, {"hook_layer": 12, "layer_scope": "above_hook"}) == list(range(13, 24))
    assert resolve_lora_layers(24, {"hook_layer": 12, "layer_scope": "all"}) == list(range(24))


def test_resolve_lora_layers_rejects_invalid_or_empty_selection():
    with pytest.raises(ValueError, match="unsupported"):
        resolve_lora_layers(24, {"hook_layer": 12, "layer_scope": "partial"})
    with pytest.raises(ValueError, match="selects no"):
        resolve_lora_layers(24, {"hook_layer": 23, "layer_scope": "above_hook"})


def test_trainable_parameter_assertion_enforces_lora_and_selected_layers():
    valid = _NamedParameterModel(["base_model.model.layers.13.self_attn.q_proj.lora_A.default.weight"])
    assert_only_lora_layers_trainable(valid, [13])

    outside = _NamedParameterModel(["base_model.model.layers.12.self_attn.q_proj.lora_A.default.weight"])
    with pytest.raises(AssertionError, match="outside selected"):
        assert_only_lora_layers_trainable(outside, [13])

    non_lora = _NamedParameterModel(["base_model.model.layers.13.self_attn.q_proj.weight"])
    with pytest.raises(AssertionError, match="non-LoRA"):
        assert_only_lora_layers_trainable(non_lora, [13])

    missing_layer = _NamedParameterModel(
        ["base_model.model.layers.13.self_attn.q_proj.lora_A.default.weight"]
    )
    with pytest.raises(AssertionError, match="missing selected layers"):
        assert_only_lora_layers_trainable(missing_layer, [13, 14])

    missing_module = _NamedParameterModel(
        [
            "base_model.model.layers.13.self_attn.q_proj.lora_A.default.weight",
            "base_model.model.layers.14.self_attn.q_proj.lora_A.default.weight",
        ]
    )
    with pytest.raises(AssertionError, match="missing LoRA target modules"):
        assert_only_lora_layers_trainable(
            missing_module,
            [13, 14],
            target_modules=["q_proj", "k_proj"],
        )

    unexpected_module = _NamedParameterModel(
        [
            "base_model.model.layers.13.self_attn.q_proj.lora_A.default.weight",
            "base_model.model.layers.13.self_attn.unexpected_proj.lora_A.default.weight",
        ]
    )
    with pytest.raises(AssertionError, match="unexpected LoRA target module"):
        assert_only_lora_layers_trainable(
            unexpected_module,
            [13],
            target_modules=["q_proj"],
        )


def test_main_configs_differ_only_by_name_and_lora_scope():
    config_dir = Path(__file__).parents[1] / "configs"
    restricted = load_config(config_dir / "qwen25_05b_sycophancy_output_repro.yaml")
    full = load_config(config_dir / "qwen25_05b_sycophancy_output_all_layers.yaml")
    assert restricted["ActivationProbeConfig"]["enabled"] is False
    assert full["ActivationProbeConfig"]["enabled"] is False
    assert restricted["RunConfig"]["fail_if_exists"] is True
    assert full["RunConfig"]["fail_if_exists"] is True

    restricted_common = deepcopy(restricted)
    full_common = deepcopy(full)
    restricted_common["RunConfig"].pop("name")
    full_common["RunConfig"].pop("name")
    restricted_common["LoRAConfig"].pop("layer_scope")
    full_common["LoRAConfig"].pop("layer_scope")
    assert restricted_common == full_common


def test_a100_control_differs_from_reproduction_only_by_name():
    config_dir = Path(__file__).parents[1] / "configs"
    restricted = load_config(config_dir / "qwen25_05b_sycophancy_output_repro.yaml")
    a100_control = load_config(config_dir / "qwen25_05b_sycophancy_output_repro_a100.yaml")
    restricted["RunConfig"].pop("name")
    a100_control["RunConfig"].pop("name")
    assert restricted == a100_control


def test_legacy_config_keeps_legacy_defaults():
    config_path = Path(__file__).parents[1] / "configs" / "qwen25_05b_sycophancy_main.yaml"
    config = load_config(config_path)
    assert config["LoRAConfig"]["layer_scope"] == "above_hook"
    assert config["RunConfig"]["fail_if_exists"] is False


def test_prepare_run_dir_refuses_overwrite(tmp_path):
    run_cfg = {"results_dir": str(tmp_path), "name": "protected", "fail_if_exists": True}
    run_dir = prepare_run_dir(run_cfg)
    assert run_dir.exists()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        prepare_run_dir(run_cfg)


def test_prepare_run_dir_reserves_protected_path_without_precheck(tmp_path, monkeypatch):
    run_cfg = {"results_dir": str(tmp_path), "name": "atomic", "fail_if_exists": True}
    target = tmp_path / "atomic"
    original_exists = Path.exists

    def reject_exists_precheck(path):
        if path == target:
            raise AssertionError("protected run creation must not use a racy exists precheck")
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", reject_exists_precheck)
    assert prepare_run_dir(run_cfg) == tmp_path / "atomic"


def test_comparison_gate_passes_matching_output_trajectories():
    report = compare_metrics(_synthetic_metrics(), _synthetic_metrics(), "reproduction")
    assert report["passed"] is True
    assert all(report["gates"].values())


def test_comparison_gate_rejects_different_behavior():
    report = compare_metrics(_synthetic_metrics(), _synthetic_metrics(final_agreement=0.5), "reproduction")
    assert report["passed"] is False
    assert report["gates"]["candidate_final_agreement"] is False


def test_comparison_gate_reports_no_hint_endpoint_regression():
    reference = _synthetic_metrics(final_no_hint_accuracy=0.1, final_no_hint_invalid=0.65)
    candidate = _synthetic_metrics(final_no_hint_accuracy=0.2, final_no_hint_invalid=0.2)
    report = compare_metrics(reference, candidate, "coverage")
    assert report["passed"] is False
    assert report["gates"]["final_no_hint_accuracy_similarity"] is False
    assert report["gates"]["final_no_hint_invalid_similarity"] is False
    assert report["reference"]["final_no_hint_accuracy"] == 0.1
    assert report["candidate"]["final_no_hint_invalid_output_rate"] == 0.2


def test_reference_manifest_detects_changes(tmp_path):
    reference = tmp_path / "reference"
    reference.mkdir()
    source = reference / "metrics.jsonl"
    source.write_text('{"step": 0}\n')
    manifest_path = tmp_path / "manifest.json"
    write_manifest(reference, manifest_path)
    assert verify_manifest(reference, manifest_path) is True
    source.write_text('{"step": 1}\n')
    assert verify_manifest(reference, manifest_path) is False
    assert build_manifest(reference)["files"][0]["path"] == "metrics.jsonl"
    assert json.loads(manifest_path.read_text())["files"][0]["path"] == "metrics.jsonl"
