from __future__ import annotations


def sequence_logprobs(model, samples, pad_token_id: int, device=None, with_grad: bool = False):
    """Whole-sequence scoring for the training objective and legacy callers.

    Low-precision full-sequence computation can differ from incremental
    generation. Paper measurements use cached_sequence_logprobs instead.

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


def cached_sequence_logprobs(model, samples, pad_token_id, device=None):
    """Replay one complete generation group, retaining finished rows as padding."""
    import inspect
    import torch

    device = device or next(model.parameters()).device
    if not samples:
        return torch.empty(0, dtype=torch.float32, device=device)
    generated = [getattr(sample, "generated", sample) for sample in samples]
    first = generated[0]
    if model.training:
        raise ValueError("cached measurement requires model.eval()")
    if any(len(s.prompt_ids) != len(first.prompt_ids) or s.generation_batch_size != len(samples)
           or s.eos_token_id != first.eos_token_id or s.min_new_tokens != first.min_new_tokens
           or not s.completion_ids for s in generated):
        raise ValueError("replay requires one intact generation group with matching sampling settings")
    for row, sample in enumerate(generated):
        if sample.generation_batch_row is not None and sample.generation_batch_row != row:
            raise ValueError("replay requires one intact generation group in original row order")
        mask = sample.prompt_attention_mask
        if mask is not None and (len(mask) != len(sample.prompt_ids) or not mask
                                 or mask[-1] != 1 or any(x not in (0, 1) for x in mask)
                                 or mask != sorted(mask)):
            raise ValueError("invalid left-padded prompt attention mask")
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    # Match Transformers' last-token logits optimization when supported.
    forward_kwargs = {"logits_to_keep": 1} if "logits_to_keep" in inspect.signature(base.forward).parameters else {}
    lengths = [len(s.completion_ids) for s in generated]
    tokens = torch.full((len(samples), max(lengths)), pad_token_id, dtype=torch.long, device=device)
    for row, sample in enumerate(generated):
        tokens[row, :lengths[row]] = torch.tensor(sample.completion_ids, device=device)
    inputs = torch.tensor([s.prompt_ids for s in generated], device=device)
    mask = torch.tensor([s.prompt_attention_mask or [1] * len(s.prompt_ids) for s in generated], device=device)
    cache, scores = None, []
    with torch.no_grad():
        for t in range(max(lengths)):
            positions = (mask.long().cumsum(-1) - 1).masked_fill(mask == 0, 0)
            output = model(input_ids=inputs, attention_mask=mask,
                           position_ids=positions if t == 0 else positions[:, -1:], past_key_values=cache,
                           use_cache=True, **forward_kwargs)
            cache = output.past_key_values
            if cache is None:
                raise ValueError("model did not return the cache needed for generation replay")
            logits = output.logits[:, -1].float().clone()
            if first.eos_token_id is not None and t < first.min_new_tokens:
                logits[:, first.eos_token_id] = -torch.inf
            scores.append(logits.log_softmax(-1).gather(1, tokens[:, t:t+1]).squeeze(1))
            inputs = tokens[:, t:t+1]
            mask = torch.cat((mask, torch.ones_like(inputs)), dim=1)
        token_scores = torch.stack(scores, dim=1)
        return torch.stack([token_scores[row, :length].sum() for row, length in enumerate(lengths)])
