"""Training and offline measurement of the paper study; no fitting or plotting."""
from __future__ import annotations

from contextlib import contextmanager
import gzip
from pathlib import Path
import random
import re
import time

import numpy as np

from .generation import configure_tokenizer_and_model, resolve_generation_config
from .study import training_examples
from .grpo import attach_logprobs, compute_price_block, sample_rollouts, train_grpo_step
from .lora_freeze import apply_lora
from .metrics import summarize_trait_metrics
from .price import CumulativePriceTracker
from .study import (
    DISTRIBUTIONS, PRICE_DISTRIBUTIONS, artifact, bank_examples, digest, load_study,
    provenance, read_json, run_config, stream_seed, timestamp, verify_artifact,
    write_json, write_row,
)


@contextmanager
def isolated_rng(seed):
    import torch

    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed)
            np.random.seed(seed % 2**32)
            torch.manual_seed(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def load_model(config):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    run, model_config = config["RunConfig"], config["ModelConfig"]
    device = run["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype_name = run["dtype"]
    if dtype_name == "auto":
        dtype_name = "bfloat16" if str(device).startswith("cuda") else "float32"
    dtypes = {"float32": torch.float32, "bfloat16": torch.bfloat16, "bf16": torch.bfloat16}
    if dtype_name not in dtypes:
        raise ValueError("paper runner supports float32 or bfloat16 model weights")
    kwargs = {"revision": model_config["revision"], "trust_remote_code": False, "local_files_only": True}
    tokenizer = AutoTokenizer.from_pretrained(model_config["model_name"], **kwargs)
    base = AutoModelForCausalLM.from_pretrained(model_config["model_name"], torch_dtype=dtypes[dtype_name], **kwargs).to(device)
    configure_tokenizer_and_model(tokenizer, base)
    if model_config.get("use_gradient_checkpointing", False):
        base.gradient_checkpointing_enable()
    model = apply_lora(base, config["LoRAConfig"]).to(device).eval()
    return model, tokenizer, device


def runtime_metadata(model, tokenizer, device, config):
    import torch

    return {
        "device": str(device), "weight_dtype": str(next(model.parameters()).dtype),
        "cuda_device_name": torch.cuda.get_device_name(device) if str(device).startswith("cuda") else None,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "model_revision": config["ModelConfig"]["revision"],
        "generation_config": resolve_generation_config(config["GenerationConfig"], tokenizer).to_dict(),
        "prompt_format": config["GenerationConfig"].get("prompt_format", "plain"),
        "chat_template": tokenizer.get_chat_template() if config["GenerationConfig"].get("prompt_format") == "chat" else None,
        "sequence_likelihood_dtype": "float32", "ratio_and_covariance_dtype": "float64",
    }


@contextmanager
def attempt(path, **metadata):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    state = {"state": "running", "started_at": timestamp(), **metadata}
    write_json(path / "status.json", state)
    started = time.monotonic()
    try:
        yield state
    except BaseException as exc:
        state.update(state="failed", finished_at=timestamp(), elapsed_seconds=time.monotonic() - started,
                     error_type=type(exc).__name__, error=str(exc))
        write_json(path / "status.json", state, update=True)
        raise
    else:
        state.update(state="complete", finished_at=timestamp(), elapsed_seconds=time.monotonic() - started)
        write_json(path / "status.json", state, update=True)


def save_checkpoint(model, run_dir, step):
    directory = run_dir / "checkpoints" / f"step_{step:04d}"
    directory.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(directory, safe_serialization=True)
    files = [artifact(directory, p) for p in sorted(directory.rglob("*")) if p.is_file()]
    metadata = {"step": step, "files": files}
    write_json(directory / "checkpoint.json", metadata)
    return artifact(run_dir, directory / "checkpoint.json", step=step)


def verify_checkpoint(run_dir, reference):
    metadata_path = verify_artifact(run_dir, reference)
    metadata = read_json(metadata_path)
    if metadata["step"] != reference["step"]:
        raise ValueError("checkpoint step mismatch")
    for file in metadata["files"]:
        verify_artifact(metadata_path.parent, file)
    return metadata_path.parent


def load_checkpoint(model, run_dir, reference):
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from peft.utils.save_and_load import load_peft_weights

    directory = verify_checkpoint(run_dir, reference)
    weights = load_peft_weights(str(directory), device="cpu")
    expected = get_peft_model_state_dict(model)
    if weights.keys() != expected.keys() or any(weights[k].shape != expected[k].shape for k in weights):
        raise ValueError("checkpoint adapter keys/shapes differ from the configured model")
    result = set_peft_model_state_dict(model, weights)
    if result.unexpected_keys:
        raise ValueError(f"unexpected adapter keys: {result.unexpected_keys}")
    model.eval()


def sample_record(sample, index, completions, *, run_id, distribution, kind, source, target=None):
    for score in (sample.pre_logprob, sample.post_logprob):
        if score is not None and not np.isfinite(score):
            raise ValueError("non-finite sequence likelihood")
    return {
        "schema_version": 1, "run_id": run_id, "kind": kind, "distribution": distribution,
        "source_step": source["step"], "target_step": None if target is None else target["step"],
        "source_checkpoint_sha256": source["sha256"],
        "target_checkpoint_sha256": None if target is None else target["sha256"],
        "group_id": sample.example.problem_id, "group_index": index // completions,
        "response_index": index % completions,
        "example": sample.example.to_json_dict(),
        "prompt_ids": sample.prompt_ids, "completion_ids": sample.completion_ids,
        "completion_text": sample.generated.completion_text,
        "eos_token_id": sample.generated.eos_token_id,
        "min_new_tokens": sample.generated.min_new_tokens,
        "stop_reason": sample.generated.stop_reason,
        "traits": sample.traits.to_json_dict(),
        "pre_logprob": sample.pre_logprob, "post_logprob": sample.post_logprob,
    }


def train_run(study_dir, run_id):
    import torch

    study_dir = Path(study_dir)
    manifest, run = load_study(study_dir, run_id)
    if manifest["cohort"] == "paper" and manifest["paper_execution_enabled"] is not True:
        raise ValueError("paper execution remains disabled pending pilot review and manifest approval")
    config = run_config(manifest, run)
    directory = study_dir / "runs" / run_id
    with attempt(directory, phase="training", run_id=run_id,
                 study_manifest_sha256=digest(study_dir / "manifest.json"), source=provenance()) as status:
        write_json(directory / "config.json", config)
        with isolated_rng(run["seed"]):
            model, tokenizer, device = load_model(config)
            write_json(directory / "runtime.json", runtime_metadata(model, tokenizer, device, config))
            train = config["TrainConfig"]
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=float(train["learning_rate"]), weight_decay=float(train["weight_decay"]),
            )
            checkpoints = [save_checkpoint(model, directory, 0)]
            with (directory / "training_metrics.jsonl").open("x") as metrics:
                for step in range(train["num_steps"]):
                    summary, samples = train_grpo_step(
                        model, tokenizer, optimizer, training_examples(config, step),
                        config["GenerationConfig"], group_size=train["group_size"], eps=float(train["eps"]),
                        max_grad_norm=float(train["max_grad_norm"]), device=device,
                    )
                    checkpoints.append(save_checkpoint(model, directory, step + 1))
                    with gzip.open(directory / f"train_{step:04d}.jsonl.gz", "xt") as raw:
                        for index, sample in enumerate(samples):
                            write_row(raw, sample_record(
                                sample, index, train["group_size"], run_id=run_id, distribution="train",
                                kind="training", source=checkpoints[step], target=checkpoints[step + 1],
                            ))
                    write_row(metrics, {"source_step": step, "target_step": step + 1, "train": summary})
                    metrics.flush()
                    status["completed_updates"] = step + 1
                    write_json(directory / "status.json", status, update=True)
            write_json(directory / "checkpoints.json", {"checkpoints": checkpoints})
            status["checkpoint_index"] = artifact(directory, directory / "checkpoints.json")
            status["runtime"] = artifact(directory, directory / "runtime.json")
            status["config"] = artifact(directory, directory / "config.json")
    return directory


def collect_pool(model, tokenizer, device, config, examples, completions, path, *, seed, run_id, kind, distribution, source):
    samples = []
    with isolated_rng(seed), gzip.open(path, "xt") as raw:
        for example in examples:
            group = sample_rollouts(model, tokenizer, [example], config["GenerationConfig"], completions, device)
            group = attach_logprobs(model, tokenizer, group, device, "pre_logprob")
            for sample in group:
                write_row(raw, sample_record(
                    sample, len(samples), completions, run_id=run_id, distribution=distribution,
                    kind=kind, source=source,
                ))
                samples.append(sample)
            raw.flush()
    return samples


def score_successor(model, tokenizer, device, samples, completions):
    # Match the source-scoring batch boundaries exactly, including padding.
    result = []
    for start in range(0, len(samples), completions):
        result.extend(attach_logprobs(model, tokenizer, samples[start:start + completions], device, "post_logprob"))
    delta = np.array([s.post_logprob - s.pre_logprob for s in result], dtype=np.float64)
    with np.errstate(over="raise", invalid="raise"):
        omega = np.exp(delta)
    if not np.isfinite(omega).all():
        raise ValueError("non-finite probability ratios")
    return result


def measure_run(study_dir, run_id, measurement_id, prompts=None):
    study_dir = Path(study_dir)
    manifest, run = load_study(study_dir, run_id)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", measurement_id):
        raise ValueError("invalid measurement ID")
    directory = study_dir / "runs" / run_id
    training = read_json(directory / "status.json")
    manifest_hash = digest(study_dir / "manifest.json")
    if training["state"] != "complete" or training["study_manifest_sha256"] != manifest_hash:
        raise ValueError("training must be complete and bound to this study manifest")
    config = read_json(verify_artifact(directory, training["config"]))
    training_runtime = read_json(verify_artifact(directory, training["runtime"]))
    if config != run_config(manifest, run):
        raise ValueError("training configuration differs from the study manifest")
    checkpoints = read_json(verify_artifact(directory, training["checkpoint_index"]))["checkpoints"]
    if [c["step"] for c in checkpoints] != list(range(config["TrainConfig"]["num_steps"] + 1)):
        raise ValueError("checkpoint sequence is incomplete or out of order")
    for checkpoint in checkpoints:
        verify_checkpoint(directory, checkpoint)
    prompts = config["PriceConfig"]["prompts_per_distribution"] if prompts is None else prompts
    examples = bank_examples(study_dir, manifest, prompts)
    config["PriceConfig"]["prompts_per_distribution"] = prompts
    config["ObservedEvalConfig"]["prompts_per_distribution"] = prompts
    output = study_dir / "measurements" / run_id / measurement_id
    with attempt(output, phase="measurement", run_id=run_id, measurement_id=measurement_id,
                 study_manifest_sha256=manifest_hash, source=provenance(),
                 training_status_sha256=digest(directory / "status.json"), bank=manifest["bank"],
                 measurement_prompts=prompts) as status:
        write_json(output / "config.json", config)
        with isolated_rng(stream_seed(run["seed"], "measurement_model")):
            model, tokenizer, device = load_model(config)
            measurement_runtime = runtime_metadata(model, tokenizer, device, config)
            write_json(output / "runtime.json", measurement_runtime)
            for key in ("weight_dtype", "model_revision", "generation_config", "prompt_format", "chat_template"):
                if measurement_runtime[key] != training_runtime[key]:
                    raise ValueError(f"measurement {key} differs from training")
            load_checkpoint(model, directory, checkpoints[0])
            tracker, baselines, files, pending = CumulativePriceTracker(), {}, [], {}
            observed_cfg, price_cfg = config["ObservedEvalConfig"], config["PriceConfig"]
            with (output / "metrics.jsonl").open("x") as metrics:
                for step, checkpoint in enumerate(checkpoints):
                    observed = {}
                    if step % observed_cfg["eval_every"] == 0 or step == len(checkpoints) - 1:
                        for distribution in DISTRIBUTIONS:
                            path = output / f"observed_{step:04d}_{distribution}.jsonl.gz"
                            samples = collect_pool(
                                model, tokenizer, device, config, examples[distribution],
                                observed_cfg["completions_per_prompt"], path,
                                seed=stream_seed(run["seed"], "observed", step, distribution),
                                run_id=run_id, kind="observed", distribution=distribution, source=checkpoint,
                            )
                            summary = summarize_trait_metrics(s.traits for s in samples)
                            baselines.setdefault(distribution, summary["agreement_rate"])
                            summary["output_agreement_observed_drift"] = summary["agreement_rate"] - baselines[distribution]
                            observed[distribution] = summary
                            files.append(artifact(output, path, rows=len(samples), kind="observed", step=step))
                    write_row(metrics, {"step": step, "price": pending, "observed_eval": observed})
                    metrics.flush()
                    if step == len(checkpoints) - 1:
                        break
                    source_pools, source_files = {}, {}
                    completions = price_cfg["completions_per_prompt"]
                    for distribution in PRICE_DISTRIBUTIONS:
                        path = output / f"source_{step:04d}_{distribution}.jsonl.gz"
                        source_pools[distribution] = collect_pool(
                            model, tokenizer, device, config, examples[distribution], completions, path,
                            seed=stream_seed(run["seed"], "price", step, distribution),
                            run_id=run_id, kind="price_source", distribution=distribution, source=checkpoint,
                        )
                        source_files[distribution] = artifact(output, path, rows=len(source_pools[distribution]), kind="price_source", step=step)
                        files.append(source_files[distribution])
                    successor = checkpoints[step + 1]
                    load_checkpoint(model, directory, successor)
                    pending = {}
                    for distribution, samples in source_pools.items():
                        paired = score_successor(model, tokenizer, device, samples, completions)
                        path = output / f"price_{step:04d}_{step + 1:04d}_{distribution}.jsonl.gz"
                        with gzip.open(path, "xt") as raw:
                            for index, sample in enumerate(paired):
                                row = sample_record(
                                    sample, index, completions, run_id=run_id, distribution=distribution,
                                    kind="price", source=checkpoint, target=successor,
                                )
                                row["source_pool_sha256"] = source_files[distribution]["sha256"]
                                write_row(raw, row)
                        files.append(artifact(output, path, rows=len(paired), kind="price", source_step=step, target_step=step + 1))
                        pending[distribution] = compute_price_block(
                            paired, tracker, distribution, random.Random(stream_seed(run["seed"], "shuffle", step, distribution)),
                            shuffled=price_cfg["compute_shuffled_null"],
                        )
                    status["completed_transitions"] = step + 1
                    write_json(output / "status.json", status, update=True)
            files.extend([artifact(output, output / name) for name in ("metrics.jsonl", "runtime.json", "config.json")])
            write_json(output / "files.json", {"files": files})
            status["files"] = artifact(output, output / "files.json")
    return output
