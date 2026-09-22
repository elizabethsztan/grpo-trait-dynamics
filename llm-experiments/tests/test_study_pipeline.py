from pathlib import Path
from copy import deepcopy
import random

import numpy as np
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from src.config import load_config
from src.lora_freeze import apply_lora, extract_layer_index
from src.study import (
    DISTRIBUTIONS, PRICE_DISTRIBUTIONS, bank_examples, digest, make_bank,
    prepare_study, read_json, read_rows, verify_artifact, training_examples,
)
from src import study_runner as runner
from train_grpo_price import _generate_train_examples


@pytest.fixture
def config():
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs/qwen25_05b_sycophancy_paper.yaml")
    cfg["RunConfig"].update(device="cpu", dtype="float32")
    cfg["TrainConfig"].update(num_steps=2, train_prompts_per_step=2, group_size=8, learning_rate=0.01)
    cfg["LoRAConfig"].update(r=2, lora_alpha=4)
    cfg["GenerationConfig"].update(max_new_tokens=3)
    cfg["PriceConfig"].update(prompts_per_distribution=2)
    cfg["ObservedEvalConfig"].update(prompts_per_distribution=2, eval_every=2)
    cfg["PaperStudyConfig"].update(bank_capacity=4)
    return cfg


class TinyTokenizer:
    pad_token, eos_token = "pad", "eos"
    pad_token_id, eos_token_id = 0, 2

    def encode(self, text, add_special_tokens=True):
        return [1, 3]

    def get_chat_template(self):
        return "synthetic test chat template"

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, return_dict):
        assert tokenize and add_generation_prompt and not return_dict
        assert len(messages) == 1 and messages[0]["role"] == "user"
        assert "<answer></answer>" in messages[0]["content"]
        return [1, 4, 3]

    def decode(self, ids, skip_special_tokens=True):
        if ids and 3 <= ids[0] <= 6:
            return f"<answer>{'ABCD'[ids[0] - 3]}</answer>"
        return "invalid"


def tiny_base():
    return Qwen2ForCausalLM(Qwen2Config(
        vocab_size=8, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=32,
        bos_token_id=1, eos_token_id=2, pad_token_id=0,
    ))


@pytest.fixture
def tiny_loader(monkeypatch):
    models = []

    def load(config):
        model = apply_lora(tiny_base(), config["LoRAConfig"]).eval()
        models.append(model)
        return model, TinyTokenizer(), "cpu"

    monkeypatch.setattr(runner, "load_model", load)
    return models


def test_manifest_covers_30_runs_but_paper_execution_is_disabled(config, tmp_path):
    from train_grpo_price import run_training

    root = tmp_path / "paper"
    manifest = prepare_study(config, root, "paper")
    assert len(manifest["runs"]) == len({r["run_id"] for r in manifest["runs"]}) == 30
    assert sum(not r["has_hint"] for r in manifest["runs"]) == 5
    assert {r["hint_probability"] for r in manifest["runs"]} == {None, .1, .25, .5, .75, .9}
    with pytest.raises(ValueError, match="disabled"):
        runner.train_run(root, manifest["runs"][0]["run_id"])
    assert not (root / "runs").exists()
    with pytest.raises(ValueError, match="run_paper_study"):
        run_training(config)
    with pytest.raises(FileExistsError):
        prepare_study(config, root, "paper")


def test_bank_is_matched_reproducible_and_balanced_at_each_budget(config):
    bank = make_bank(config, 731)
    assert bank == make_bank(config, 731)
    assert bank != make_bank(config, 732)
    for row in bank:
        variants = row["variants"]
        assert len({v["problem_text"] for v in variants.values()}) == 1
        assert len({v["gold_choice"] for v in variants.values()}) == 1
        assert variants["eval_wrong_hint"]["hint_is_correct"] is False
        assert variants["eval_correct_hint"]["hint_is_correct"] is True
        assert variants["eval_no_hint"]["user_hint"] is None
        for v in variants.values():
            assert v["problem_id"] == row["group_id"]
            assert v["options"] == variants["eval_wrong_hint"]["options"]
    for size in (2, 4):
        assert sum(r["variants"]["eval_balanced_hint"]["hint_is_correct"] for r in bank[:size]) == size // 2


def test_no_hint_training_uses_existing_generator_without_hints(config):
    config["DataConfig"]["train_has_hint"] = False
    examples = _generate_train_examples(config, 0)
    assert all(e.user_hint is None and e.hint_is_correct is None for e in examples)
    assert all("No user guess is provided." in e.prompt_text for e in examples)


def test_training_inputs_match_across_sweep_with_nested_hints(config):
    config["TrainConfig"]["train_prompts_per_step"] = 128
    streams = []
    for p in (0, .1, .25, .5, .75, .9, 1, None):
        variant = deepcopy(config)
        variant["DataConfig"].update(train_has_hint=p is not None, train_hint_correct_probability=p or 0)
        examples = training_examples(variant, 7)
        assert examples == training_examples(variant, 7)
        streams.append(examples)
    for group in zip(*streams):
        first = group[0]
        for example in group:
            assert (example.problem_id, example.problem_text, example.options, example.gold_choice, example.gold_value) == (
                first.problem_id, first.problem_text, first.options, first.gold_choice, first.gold_value,
            )
        hinted = group[:-1]
        assert len({e.hint_phrase for e in hinted}) == 1
        assert len({e.user_hint for e in hinted if not e.hint_is_correct}) == 1
        correct = [e.hint_is_correct for e in hinted]
        assert correct == sorted(correct)
        assert correct[0] is False and correct[-1] is True
        assert group[-1].user_hint is None and group[-1].hint_is_correct is None
        assert "No user guess is provided." in group[-1].prompt_text
    assert streams[0] != training_examples(config, 8)
    changed = deepcopy(config)
    changed["RunConfig"]["seed"] += 1
    assert [e.problem_text for e in streams[0]] != [e.problem_text for e in training_examples(changed, 7)]
    changed = deepcopy(config)
    changed["DataConfig"]["train_hint_phrases"] = ["I choose {choice}."]
    original, reworded = training_examples(config, 7), training_examples(changed, 7)
    assert [(e.problem_text, e.options, e.user_hint) for e in original] == [
        (e.problem_text, e.options, e.user_hint) for e in reworded
    ]


@pytest.mark.parametrize("scope,expected", [("all", {0, 1}), ("above_hook", {1})])
def test_lora_scope_selects_only_requested_adapter_layers(config, scope, expected):
    config["LoRAConfig"].update(layer_scope=scope, hook_layer=0)
    model = apply_lora(tiny_base(), config["LoRAConfig"])
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    assert {extract_layer_index(name) for name in trainable} == expected
    assert all("lora_" in name and "lm_head" not in name for name in trainable)


def test_checkpoint_replay_restores_distinct_weights(config, tmp_path, tiny_loader):
    from peft import get_peft_model_state_dict

    model, _, _ = runner.load_model(config)
    before = {k: v.clone() for k, v in get_peft_model_state_dict(model).items()}
    zero = runner.save_checkpoint(model, tmp_path, 0)
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.add_(0.25)
    after = {k: v.clone() for k, v in get_peft_model_state_dict(model).items()}
    one = runner.save_checkpoint(model, tmp_path, 1)
    for reference, expected in ((zero, before), (one, after), (zero, before)):
        runner.load_checkpoint(model, tmp_path, reference)
        for key, value in get_peft_model_state_dict(model).items():
            torch.testing.assert_close(value, expected[key], atol=0, rtol=0)
    weights_file = tmp_path / "checkpoints/step_0000/adapter_model.safetensors"
    with weights_file.open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="mismatch"):
        runner.load_checkpoint(model, tmp_path, zero)


@pytest.mark.parametrize("batch_size", [1, 2])
def test_complete_training_measurement_replay_and_raw_accounting(config, tmp_path, tiny_loader, batch_size):
    root = tmp_path / "pilot"
    manifest = prepare_study(config, root, "pilot")
    run_id = manifest["runs"][0]["run_id"]
    run_dir = runner.train_run(root, run_id)
    runtime = read_json(run_dir / "runtime.json")
    assert runtime["prompt_format"] == "chat"
    assert runtime["chat_template"] == TinyTokenizer().get_chat_template()
    for step in (0, 1):
        records = list(read_rows(run_dir / f"train_{step:04d}.jsonl.gz"))
        expected_config = deepcopy(config)
        expected_config["RunConfig"]["seed"] = manifest["runs"][0]["seed"]
        expected_config["DataConfig"]["train_hint_correct_probability"] = .25
        expected = training_examples(expected_config, step)
        assert [r["example"] for r in records[::config["TrainConfig"]["group_size"]]] == [e.to_json_dict() for e in expected]
        assert all(r["prompt_ids"] == [1, 4, 3] for r in records)
    assert read_json(run_dir / "status.json")["state"] == "complete"
    checkpoints = read_json(run_dir / "checkpoints.json")["checkpoints"]
    assert [c["step"] for c in checkpoints] == [0, 1, 2]
    with pytest.raises(FileExistsError):
        runner.train_run(root, run_id)

    random.seed(19)
    np.random.seed(19)
    torch.manual_seed(19)
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    output = runner.measure_run(root, run_id, "initial", question_batch_size=batch_size)
    assert random.getstate() == python_state
    assert np.random.get_state()[0] == numpy_state[0]
    np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
    assert np.random.get_state()[2:] == numpy_state[2:]
    torch.testing.assert_close(torch.get_rng_state(), torch_state, atol=0, rtol=0)
    assert read_json(output / "status.json")["state"] == "complete"
    rows = list(read_rows(output / "metrics.jsonl"))
    assert [r["step"] for r in rows] == [0, 1, 2]
    assert rows[0]["price"] == {}
    assert rows[1]["observed_eval"] == {}
    assert set(rows[2]["observed_eval"]) == set(DISTRIBUTIONS)
    for reference in read_json(output / "files.json")["files"]:
        verify_artifact(output, reference)

    for distribution in PRICE_DISTRIBUTIONS:
        cumulative = 0.0
        for step in (0, 1):
            source_path = output / f"source_{step:04d}_{distribution}.jsonl.gz"
            source = list(read_rows(source_path))
            paired = list(read_rows(output / f"price_{step:04d}_{step + 1:04d}_{distribution}.jsonl.gz"))
            assert len(source) == len(paired) == 4
            assert len({r["group_id"] for r in paired}) == 2
            for old, new in zip(source, paired):
                assert old["likelihood_method"] == new["likelihood_method"] == "generation_capture_then_cached_replay_v1"
                assert old["generation_batch_size"] == new["generation_batch_size"] == 2 * batch_size
                assert old["prompt_ids"] == new["prompt_ids"] == [1, 4, 3]
                assert old["completion_ids"] == new["completion_ids"]
                assert old["pre_logprob"] == new["pre_logprob"]
                assert old["traits"] == new["traits"]
                assert new["source_pool_sha256"] == digest(source_path)
                assert (new["source_step"], new["target_step"]) == (step, step + 1)
                assert new["source_checkpoint_sha256"] == checkpoints[step]["sha256"]
                assert new["target_checkpoint_sha256"] == checkpoints[step + 1]["sha256"]
            weights = np.exp(np.array([r["post_logprob"] - r["pre_logprob"] for r in paired]))
            trait = np.array([r["traits"]["output_agreement"] for r in paired], dtype=float)
            covariance = np.mean(weights * trait) - np.mean(weights) * np.mean(trait)
            cumulative += covariance
            logged = rows[step + 1]["price"][distribution]["output_agreement"]
            assert logged["cov_step"] == pytest.approx(covariance, abs=1e-12)
            assert logged["cov_cum"] == pytest.approx(cumulative, abs=1e-12)
        direct = list(read_rows(output / f"observed_0002_{distribution}.jsonl.gz"))
        prevalence = np.mean([r["traits"]["output_agreement"] for r in direct])
        assert rows[2]["observed_eval"][distribution]["agreement_rate"] == prevalence

    with pytest.raises(FileExistsError):
        runner.measure_run(root, run_id, "initial", question_batch_size=batch_size)
    replay = runner.measure_run(root, run_id, "replay", question_batch_size=batch_size)
    assert list(read_rows(replay / "metrics.jsonl")) == rows
    larger = runner.measure_run(root, run_id, "larger", prompts=4, question_batch_size=batch_size)
    for distribution in PRICE_DISTRIBUTIONS:
        original = list(read_rows(output / f"source_0000_{distribution}.jsonl.gz"))
        expanded = list(read_rows(larger / f"source_0000_{distribution}.jsonl.gz"))
        assert expanded[:4] == original
        assert len(expanded) == 8
    assert read_json(larger / "config.json")["PriceConfig"]["prompts_per_distribution"] == 4
    assert read_json(run_dir / "config.json")["PriceConfig"]["prompts_per_distribution"] == 2
    from src.bootstrap import bootstrap_measurement
    assert set(bootstrap_measurement(larger, tmp_path / "bootstrap", draws=100)) == set(PRICE_DISTRIBUTIONS)


def test_failed_training_is_retained_and_not_reused(config, tmp_path, monkeypatch):
    root = tmp_path / "pilot"
    manifest = prepare_study(config, root, "pilot")
    run_id = manifest["runs"][0]["run_id"]

    def fail(config):
        raise RuntimeError("synthetic loading failure")

    monkeypatch.setattr(runner, "load_model", fail)
    with pytest.raises(RuntimeError, match="synthetic"):
        runner.train_run(root, run_id)
    state = read_json(root / "runs" / run_id / "status.json")
    assert state["state"] == "failed" and state["error_type"] == "RuntimeError"
    with pytest.raises(FileExistsError):
        runner.train_run(root, run_id)
    with pytest.raises(ValueError, match="complete"):
        runner.measure_run(root, run_id, "invalid")


def test_bank_tampering_and_invalid_budgets_are_rejected(config, tmp_path):
    root = tmp_path / "pilot"
    manifest = prepare_study(config, root, "pilot")
    with pytest.raises(ValueError, match="within"):
        bank_examples(root, manifest, 6)
    with (root / "bank.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="mismatch"):
        bank_examples(root, manifest, 2)


def test_failed_successor_scoring_retains_source_responses(config, tmp_path, tiny_loader, monkeypatch):
    root = tmp_path / "pilot"
    manifest = prepare_study(config, root, "pilot")
    run_id = manifest["runs"][0]["run_id"]
    runner.train_run(root, run_id)

    def fail(*args):
        raise RuntimeError("synthetic successor failure")

    monkeypatch.setattr(runner, "score_successor", fail)
    with pytest.raises(RuntimeError, match="synthetic"):
        runner.measure_run(root, run_id, "failed")
    output = root / "measurements" / run_id / "failed"
    assert read_json(output / "status.json")["state"] == "failed"
    assert len(list(read_rows(output / "source_0000_eval_wrong_hint.jsonl.gz"))) == 4
    assert not (output / "files.json").exists()


def test_measurement_refuses_a_changed_numeric_policy(config, tmp_path, tiny_loader, monkeypatch):
    root = tmp_path / "pilot"
    manifest = prepare_study(config, root, "pilot")
    run_id = manifest["runs"][0]["run_id"]
    runner.train_run(root, run_id)
    original_loader = runner.load_model

    def changed(config):
        model, tokenizer, device = original_loader(config)
        return model.double(), tokenizer, device

    monkeypatch.setattr(runner, "load_model", changed)
    with pytest.raises(ValueError, match="weight_dtype"):
        runner.measure_run(root, run_id, "different_precision")


def test_checkpoint_writing_does_not_change_training(config, tmp_path, monkeypatch, tiny_loader):
    from peft import get_peft_model_state_dict

    a, b = tmp_path / "with_checkpoints", tmp_path / "without_checkpoints"
    manifest = prepare_study(config, a, "pilot")
    prepare_study(config, b, "pilot")
    run_id = manifest["runs"][0]["run_id"]
    first = runner.train_run(a, run_id)
    weights = {k: v.clone() for k, v in get_peft_model_state_dict(tiny_loader[-1]).items()}
    monkeypatch.setattr(runner, "save_checkpoint", lambda model, root, step: {
        "step": step, "path": f"test_disabled_{step}", "sha256": f"{step:064x}",
    })
    second = runner.train_run(b, run_id)
    for key, value in get_peft_model_state_dict(tiny_loader[-1]).items():
        torch.testing.assert_close(value, weights[key], atol=0, rtol=0)
    assert list(read_rows(first / "training_metrics.jsonl")) == list(read_rows(second / "training_metrics.jsonl"))
    for step in (0, 1):
        left = list(read_rows(first / f"train_{step:04d}.jsonl.gz"))
        right = list(read_rows(second / f"train_{step:04d}.jsonl.gz"))
        assert [r["completion_ids"] for r in left] == [r["completion_ids"] for r in right]


def test_partial_measurement_batch_preserves_artifact_replay(config, tmp_path, tiny_loader):
    from src.generation import GeneratedCompletion
    from src.logprobs import cached_sequence_logprobs
    from src.bootstrap import bootstrap_measurement

    root = tmp_path / "partial"
    manifest = prepare_study(config, root, "pilot")
    run_id = manifest["runs"][0]["run_id"]
    directory = runner.train_run(root, run_id)
    output = runner.measure_run(root, run_id, "batch3", prompts=4, question_batch_size=3)
    model, tokenizer = tiny_loader[-1], TinyTokenizer()
    checkpoints = read_json(directory / "checkpoints.json")["checkpoints"]
    runner.load_checkpoint(model, directory, checkpoints[1])
    records = list(read_rows(output / "price_0000_0001_eval_wrong_hint.jsonl.gz"))
    assert [r["generation_batch_size"] for r in records] == [6] * 6 + [2] * 2
    assert [r["generation_batch_row"] for r in records] == list(range(6)) + [0, 1]
    assert [r["group_index"] for r in records] == [0, 0, 1, 1, 2, 2, 3, 3]
    for batch in (records[:6], records[6:]):
        samples = [GeneratedCompletion(**{k: r[k] for k in (
            "prompt_ids", "completion_ids", "completion_text", "eos_token_id", "min_new_tokens",
            "generation_batch_size", "generation_batch_row", "prompt_attention_mask")}) for r in batch]
        scores = cached_sequence_logprobs(model, samples, tokenizer.pad_token_id)
        torch.testing.assert_close(scores, torch.tensor([r["post_logprob"] for r in batch]), atol=0, rtol=0)
    assert set(bootstrap_measurement(output, tmp_path / "bootstrap_partial", draws=100)) == set(PRICE_DISTRIBUTIONS)
