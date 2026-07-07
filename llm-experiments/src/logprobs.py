from __future__ import annotations


def sequence_logprobs(model, samples, pad_token_id: int, device=None, with_grad: bool = False):
    import torch

    if not samples:
        return torch.empty(0, device=device)

    device = device or next(model.parameters()).device
    max_len = max(len(sample.prompt_ids) + len(sample.completion_ids) for sample in samples)
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
        sequence_scores = []
        for row, sample in enumerate(samples):
            prompt_len = len(sample.prompt_ids)
            token_scores = []
            for offset, token_id in enumerate(sample.completion_ids):
                pos = prompt_len + offset
                token_scores.append(log_probs[row, pos - 1, token_id])
            if token_scores:
                sequence_scores.append(torch.stack(token_scores).sum())
            else:
                sequence_scores.append(torch.zeros((), device=device, dtype=log_probs.dtype))
        return torch.stack(sequence_scores)
