from __future__ import annotations


def sequence_logprobs(model, samples, pad_token_id: int, device=None, with_grad: bool = False):
    """Score generated sequences under their actual sampling distribution.

    GeneratedCompletion stores the EOS suppression settings; RolloutSample wraps
    it in ``generated``. Samples without that metadata retain raw-policy scoring
    for compatibility with existing callers constructing token-only samples.
    """
    import torch

    if not samples:
        return torch.empty(0, device=device, dtype=torch.float32)

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
        sequence_scores = []
        for row, sample in enumerate(samples):
            prompt_len = len(sample.prompt_ids)
            completion_len = len(sample.completion_ids)
            if completion_len:
                # Upcast before normalization and accumulation. Slice first to
                # avoid materializing float32 probabilities for prompt tokens.
                token_logits = logits[row, prompt_len - 1 : prompt_len + completion_len - 1].float().clone()
                generated = getattr(sample, "generated", sample)
                eos_token_id = getattr(generated, "eos_token_id", None)
                min_new_tokens = getattr(generated, "min_new_tokens", 0)
                if eos_token_id is not None and min_new_tokens:
                    token_logits[:min_new_tokens, eos_token_id] = -torch.inf
                log_probs = token_logits.log_softmax(dim=-1)
                token_ids = torch.tensor(sample.completion_ids, dtype=torch.long, device=device)
                token_scores = log_probs.gather(1, token_ids[:, None]).squeeze(1)
                sequence_scores.append(token_scores.sum(dtype=torch.float32))
            else:
                sequence_scores.append(torch.zeros((), device=device, dtype=torch.float32))
        return torch.stack(sequence_scores)
