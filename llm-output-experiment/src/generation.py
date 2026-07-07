from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GeneratedCompletion:
    prompt_ids: list[int]
    completion_ids: list[int]
    completion_text: str

    @property
    def completion_token_length(self) -> int:
        return len(self.completion_ids)


def configure_tokenizer_and_model(tokenizer, model) -> None:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
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


def generate_completions(model, tokenizer, prompt_text: str, generation_config: dict, num_return_sequences: int, device):
    import torch

    configure_tokenizer_and_model(tokenizer, model)
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=True)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        num_return_sequences=num_return_sequences,
        do_sample=bool(generation_config.get("do_sample", True)),
        temperature=float(generation_config.get("temperature", 1.0)),
        top_p=float(generation_config.get("top_p", 1.0)),
        top_k=int(generation_config.get("top_k", 0)),
        repetition_penalty=float(generation_config.get("repetition_penalty", 1.0)),
        max_new_tokens=int(generation_config.get("max_new_tokens", 16)),
        min_new_tokens=int(generation_config.get("min_new_tokens", 1)),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

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
            )
        )
    return completions
