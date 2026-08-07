import torch
import torch.nn.functional as F


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
