from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GeneratedCompletion:
    prompt_ids: list[int]
    completion_ids: list[int]
    completion_text: str
    # These define the sampling distribution used by the likelihood scorer.
    eos_token_id: int | None = None
    min_new_tokens: int = 0
    stop_reason: str | None = None

    @property
    def completion_token_length(self) -> int:
        return len(self.completion_ids)


def configure_tokenizer_and_model(tokenizer, model) -> None:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id


def truncate_after_eos(token_ids: list[int], eos_token_id: int | None, pad_token_id: int | None) -> list[int]:
    # A pad token sampled before EOS is part of the response. Only tokens after
    # the first EOS are batch padding; retain EOS itself, including when PAD=EOS.
    cleaned = []
    for token_id in token_ids:
        cleaned.append(int(token_id))
        if eos_token_id is not None and token_id == eos_token_id:
            break
    return cleaned


def resolve_generation_config(generation_config: dict, tokenizer):
    """Construct the sampling policy supported by our sequence likelihoods.

    Start from explicit defaults so model-specific generation presets cannot
    silently add a logits processor that the scorer does not account for.
    """
    from transformers import GenerationConfig

    supported = {
        "do_sample": True,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "repetition_penalty": 1.0,
        "max_new_tokens": 16,
        "min_new_tokens": 1,
    }
    generation_config = dict(generation_config)
    prompt_format = generation_config.pop("prompt_format", "plain")
    if prompt_format not in ("plain", "chat"):
        raise ValueError("prompt_format must be plain or chat")
    unknown = set(generation_config) - set(supported)
    if unknown:
        raise ValueError(f"unsupported generation settings: {sorted(unknown)}")
    resolved = {**supported, **generation_config}
    for name in ("do_sample", "temperature", "top_p", "top_k", "repetition_penalty"):
        if resolved[name] != supported[name]:
            raise ValueError(f"likelihood accounting requires {name}={supported[name]!r}")
    for name in ("min_new_tokens", "max_new_tokens"):
        if type(resolved[name]) is not int:
            raise ValueError(f"{name} must be an integer")
    if not 0 <= resolved["min_new_tokens"] <= resolved["max_new_tokens"] or resolved["max_new_tokens"] < 1:
        raise ValueError("require 0 <= min_new_tokens <= max_new_tokens and max_new_tokens >= 1")
    return GenerationConfig(
        **resolved,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )


def generate_completions(model, tokenizer, prompt_text: str, generation_config: dict, num_return_sequences: int, device):
    import torch

    configure_tokenizer_and_model(tokenizer, model)
    resolved = resolve_generation_config(generation_config, tokenizer)
    resolved.num_return_sequences = num_return_sequences
    if generation_config.get("prompt_format", "plain") == "chat":
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=True, add_generation_prompt=True, return_dict=False,
        )
    else:
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=True)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)

    # Transformers can fill unset fields from model.generation_config even when
    # an explicit config is passed. Use our policy as that fallback as well.
    previous_config = model.generation_config
    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    previous_base_config = base_model.generation_config
    try:
        model.generation_config = resolved
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=resolved,
        )
    finally:
        model.generation_config = previous_config
        base_model.generation_config = previous_base_config

    completions = []
    prompt_len = len(prompt_ids)
    for sequence in outputs:
        completion_ids = truncate_after_eos(
            sequence[prompt_len:].detach().cpu().tolist(),
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        completions.append(
            GeneratedCompletion(
                prompt_ids=list(prompt_ids),
                completion_ids=completion_ids,
                completion_text=tokenizer.decode(completion_ids, skip_special_tokens=True),
                eos_token_id=resolved.eos_token_id,
                min_new_tokens=resolved.min_new_tokens,
                stop_reason="eos" if completion_ids and completion_ids[-1] == resolved.eos_token_id else "max_new_tokens",
            )
        )
    return completions
