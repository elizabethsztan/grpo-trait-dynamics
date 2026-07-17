from __future__ import annotations

from dataclasses import dataclass

from .prompts import encode_prompt_for_generation


@dataclass
class GeneratedCompletion:
    prompt_ids: list[int]
    completion_ids: list[int]
    completion_text: str
    stop_reason: str
    stopped_on_answer_tag: bool

    @property
    def completion_token_length(self) -> int:
        return len(self.completion_ids)


def configure_tokenizer_and_model(tokenizer, model) -> None:
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(model, "generation_config"):
        model.generation_config.pad_token_id = tokenizer.pad_token_id


def truncate_after_eos(token_ids: list[int], eos_token_id: int | None, pad_token_id: int | None) -> list[int]:
    cleaned = []
    for token_id in token_ids:
        if pad_token_id is not None and token_id == pad_token_id:
            continue
        cleaned.append(int(token_id))
        if eos_token_id is not None and token_id == eos_token_id:
            break
    return cleaned


def stop_sequence_suffix(token_ids: list[int], stop_ids: list[int]) -> bool:
    return bool(stop_ids) and len(token_ids) >= len(stop_ids) and token_ids[-len(stop_ids) :] == stop_ids


def _sample_next_token(logits, generation_config: dict):
    import torch

    temperature = float(generation_config.get("temperature", 1.0))
    top_p = float(generation_config.get("top_p", 1.0))
    top_k = int(generation_config.get("top_k", 0))
    repetition_penalty = float(generation_config.get("repetition_penalty", 1.0))
    do_sample = bool(generation_config.get("do_sample", True))
    if top_p != 1.0 or top_k != 0 or repetition_penalty != 1.0 or temperature != 1.0 or not do_sample:
        raise NotImplementedError(
            "v2 sampler currently supports do_sample=True, temperature=1.0, "
            "top_p=1.0, top_k=0, repetition_penalty=1.0 so generation and logprobs use the same distribution"
        )
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def generate_completions(model, tokenizer, prompt_text: str, generation_config: dict, num_return_sequences: int, device):
    import torch

    configure_tokenizer_and_model(tokenizer, model)
    prompt_ids = encode_prompt_for_generation(
        tokenizer,
        prompt_text,
        use_chat_template=bool(generation_config.get("use_chat_template", True)),
    )
    stop_sequence = generation_config.get("stop_sequence", "</answer>")
    stop_ids = list(tokenizer.encode(stop_sequence, add_special_tokens=False)) if stop_sequence else []
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", eos_token_id)
    max_new_tokens = int(generation_config.get("max_new_tokens", 16))
    min_new_tokens = int(generation_config.get("min_new_tokens", 1))

    completions: list[list[int]] = [[] for _ in range(num_return_sequences)]
    stop_reasons: list[str | None] = [None for _ in range(num_return_sequences)]

    was_training = model.training
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            active = [idx for idx, reason in enumerate(stop_reasons) if reason is None]
            if not active:
                break
            sequences = [prompt_ids + completions[idx] for idx in active]
            max_len = max(len(sequence) for sequence in sequences)
            input_ids = torch.full((len(active), max_len), pad_token_id, dtype=torch.long, device=device)
            attention_mask = torch.zeros_like(input_ids)
            for row, sequence in enumerate(sequences):
                input_ids[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=device)
                attention_mask[row, : len(sequence)] = 1
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            last_positions = attention_mask.sum(dim=1) - 1
            logits = outputs.logits[torch.arange(len(active), device=device), last_positions]
            next_tokens = _sample_next_token(logits, generation_config).detach().cpu().tolist()
            for idx, token_id in zip(active, next_tokens):
                completions[idx].append(int(token_id))
                if len(completions[idx]) < min_new_tokens:
                    continue
                if stop_sequence_suffix(completions[idx], stop_ids):
                    stop_reasons[idx] = "answer_stop"
                elif eos_token_id is not None and int(token_id) == int(eos_token_id):
                    stop_reasons[idx] = "eos"
        for idx, reason in enumerate(stop_reasons):
            if reason is None:
                stop_reasons[idx] = "max_new_tokens"
    if was_training:
        model.train()

    generated = []
    for completion_ids, reason in zip(completions, stop_reasons):
        if reason == "eos":
            completion_ids = truncate_after_eos(completion_ids, eos_token_id=eos_token_id, pad_token_id=pad_token_id)
        generated.append(
            GeneratedCompletion(
                prompt_ids=list(prompt_ids),
                completion_ids=list(completion_ids),
                completion_text=tokenizer.decode(completion_ids, skip_special_tokens=True),
                stop_reason=str(reason),
                stopped_on_answer_tag=reason == "answer_stop",
            )
        )
    return generated
