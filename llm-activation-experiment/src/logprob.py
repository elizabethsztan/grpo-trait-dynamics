import torch
import torch.nn.functional as F


def generation_logprob_batch(policy, pairs):
    """Unwarped completion probabilities using Qwen's cached generation path.

    Keep the original generation batch: BF16 scoring can depend on batch layout.
    Count every supplied completion token, including a terminating EOS. Finished
    rows stay in the batch, as in generate(); empty completions have logprob zero.
    """
    if not pairs:
        return []
    device = policy.model.device
    pad = policy.tokenizer.pad_token_id
    width = max(len(p) for p, _ in pairs)
    lengths = torch.tensor([len(c) for _, c in pairs], device=device)
    count = int(lengths.max())
    if count == 0:
        return [0.0] * len(pairs)
    ids = torch.full((len(pairs), width), pad, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    completions = torch.full((len(pairs), count), pad, dtype=torch.long, device=device)
    for i, (p, c) in enumerate(pairs):
        ids[i, -len(p):], mask[i, -len(p):] = p.to(device), 1
        completions[i, :len(c)] = c.to(device)
    positions = (mask.cumsum(-1) - 1).masked_fill(mask == 0, 0)
    totals = torch.zeros(len(pairs), dtype=torch.float64, device=device)
    cache = None
    with torch.no_grad():
        for j in range(count):
            out = policy.model(input_ids=ids, attention_mask=mask,
                               position_ids=positions, past_key_values=cache,
                               use_cache=True, logits_to_keep=1)
            lp = F.log_softmax(out.logits[:, -1].float(), dim=-1)
            token_lp = lp.gather(1, completions[:, j:j + 1]).squeeze(1)
            totals += torch.where(j < lengths, token_lp.double(), 0.)
            cache = out.past_key_values
            ids = completions[:, j:j + 1]
            positions = positions[:, -1:] + 1
            mask = torch.cat([mask, torch.ones_like(ids)], dim=1)
    return totals.cpu().tolist()


def sequence_logprob(policy, prompt_ids, completion_ids):
    # completion_ids are the EXACT sampled tokens (EOS included iff generation
    # stopped on EOS; only emitted tokens if truncated). Logprob is summed over
    # exactly those positions, predicted from the preceding context. Computed
    # under whatever LoRA state the caller has set (pi_t or pi_{t+1}).
    device = policy.model.device
    input_ids = torch.cat([prompt_ids, completion_ids], dim=-1).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = policy.model(input_ids=input_ids).logits[0]        # (seq, vocab)
    logprobs = F.log_softmax(logits.float(), dim=-1)
    comp_start = prompt_ids.shape[-1]
    # token at position i is predicted by logits at i-1
    idx = torch.arange(comp_start, input_ids.shape[-1], device=device)
    tok = input_ids[0, idx]
    return float(logprobs[idx - 1, tok].sum().item())


def sequence_logprob_batch(policy, pairs):
    # Batched exact-span sequence logprob over a list of (prompt_ids, completion_ids).
    # Right-pad (pads after real tokens don't affect earlier logits in a causal model);
    # one forward, per-row sum of completion-token logprobs. Returns a list[float].
    # Empty-completion rows return 0.0.
    device = policy.model.device
    pad = policy.tokenizer.pad_token_id
    seqs = [torch.cat([p, c]) for p, c in pairs]
    Lmax = max(s.shape[0] for s in seqs)
    ids = torch.full((len(seqs), Lmax), pad, dtype=torch.long)
    attn = torch.zeros((len(seqs), Lmax), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :s.shape[0]] = s
        attn[i, :s.shape[0]] = 1
    with torch.no_grad():
        logits = policy.model(input_ids=ids.to(device), attention_mask=attn.to(device)).logits
    lp = F.log_softmax(logits.float(), dim=-1)
    out = []
    for i, (p, c) in enumerate(pairs):
        clen = c.shape[0]
        if clen == 0:
            out.append(0.0); continue
        pstart = p.shape[0]
        pos = torch.arange(pstart, pstart + clen, device=device)
        tok = ids[i, pos.cpu()].to(device)
        out.append(float(lp[i, pos - 1, tok].sum().item()))
    return out
