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
