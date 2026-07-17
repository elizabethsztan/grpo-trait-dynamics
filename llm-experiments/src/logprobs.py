from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TokenLogprobBatch:
    token_logprobs: object
    mask: object
    sequence_logprobs: object


def _model_device(model, fallback=None):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return fallback


def token_logprob_batch(model, samples, pad_token_id: int, device=None, with_grad: bool = False) -> TokenLogprobBatch:
    import torch

    if not samples:
        empty = torch.empty((0, 0), dtype=torch.float32, device=device)
        return TokenLogprobBatch(token_logprobs=empty, mask=torch.empty((0, 0), dtype=torch.bool, device=device), sequence_logprobs=empty.sum(dim=1))

    device = device or _model_device(model)
    max_total_len = max(len(sample.prompt_ids) + len(sample.completion_ids) for sample in samples)
    max_completion_len = max(len(sample.completion_ids) for sample in samples)
    input_ids = torch.full((len(samples), max_total_len), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(input_ids)
    completion_mask = torch.zeros((len(samples), max_completion_len), dtype=torch.bool, device=device)

    for row, sample in enumerate(samples):
        ids = list(sample.prompt_ids) + list(sample.completion_ids)
        input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        attention_mask[row, : len(ids)] = 1
        completion_mask[row, : len(sample.completion_ids)] = True

    context = torch.enable_grad() if with_grad else torch.no_grad()
    with context:
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        log_probs = logits.float().log_softmax(dim=-1)
        token_scores = torch.zeros((len(samples), max_completion_len), dtype=torch.float32, device=device)
        for row, sample in enumerate(samples):
            prompt_len = len(sample.prompt_ids)
            for offset, token_id in enumerate(sample.completion_ids):
                pos = prompt_len + offset
                token_scores[row, offset] = log_probs[row, pos - 1, token_id]
        sequence_scores = (token_scores * completion_mask).sum(dim=1).float()
        return TokenLogprobBatch(
            token_logprobs=token_scores,
            mask=completion_mask,
            sequence_logprobs=sequence_scores,
        )


def sequence_logprobs(model, samples, pad_token_id: int, device=None, with_grad: bool = False):
    return token_logprob_batch(model, samples, pad_token_id, device=device, with_grad=with_grad).sequence_logprobs
