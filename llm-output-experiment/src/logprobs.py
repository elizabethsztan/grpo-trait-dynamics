from __future__ import annotations


def token_logprobs(model, samples, pad_token_id: int, device=None, with_grad: bool = False):
    """Per-completion-token logprobs, padded to (B, max_completion_len), plus a 0/1 mask."""
    import torch

    if not samples:
        empty = torch.empty(0, 0, device=device)
        return empty, empty

    device = device or next(model.parameters()).device
    max_len = max(len(sample.prompt_ids) + len(sample.completion_ids) for sample in samples)
    max_completion = max(len(sample.completion_ids) for sample in samples)
    input_ids = torch.full((len(samples), max_len), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(input_ids)

    for row, sample in enumerate(samples):
        ids = list(sample.prompt_ids) + list(sample.completion_ids)
        input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        attention_mask[row, : len(ids)] = 1

    context = torch.enable_grad() if with_grad else torch.no_grad()
    with context:
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        log_probs = logits.log_softmax(dim=-1)
        rows, mask = [], torch.zeros((len(samples), max_completion), device=device, dtype=log_probs.dtype)
        for row, sample in enumerate(samples):
            prompt_len = len(sample.prompt_ids)
            scores = [log_probs[row, prompt_len + offset - 1, token_id]
                      for offset, token_id in enumerate(sample.completion_ids)]
            if scores:
                mask[row, : len(scores)] = 1.0
                pad = max_completion - len(scores)
                stacked = torch.stack(scores)
                if pad:
                    stacked = torch.cat([stacked, torch.zeros(pad, device=device, dtype=stacked.dtype)])
            else:
                stacked = torch.zeros(max_completion, device=device, dtype=log_probs.dtype)
            rows.append(stacked)
        return torch.stack(rows), mask


def sequence_logprobs(model, samples, pad_token_id: int, device=None, with_grad: bool = False):
    import torch

    if not samples:
        return torch.empty(0, device=device)
    tokens, mask = token_logprobs(model, samples, pad_token_id, device=device, with_grad=with_grad)
    return (tokens * mask).sum(dim=1)
