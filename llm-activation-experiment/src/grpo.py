import logging
import torch
import torch.nn.functional as F

LOGGER = logging.getLogger(__name__)


def sample_completions(policy, prompt_ids, G, gen_cfg):
    device = policy.model.device
    batch = prompt_ids.unsqueeze(0).to(device).expand(G, -1)
    with torch.no_grad():
        out = policy.model.generate(
            batch,
            attention_mask=torch.ones_like(batch),  # prompts are unpadded; explicit mask (pad==eos)
            do_sample=True,
            temperature=gen_cfg["temperature"],
            top_p=gen_cfg["top_p"],
            top_k=(gen_cfg["top_k"] or 0),
            max_new_tokens=gen_cfg["max_new_tokens"],
            pad_token_id=policy.tokenizer.pad_token_id,
        )
    plen = prompt_ids.shape[-1]
    pad = policy.tokenizer.pad_token_id
    # Exact sampled span per row, trailing pad stripped (keeps EOS if present).
    completions = []
    for row in out:
        comp = row[plen:]
        keep = (comp != pad).nonzero()
        end = (keep[-1].item() + 1) if len(keep) else 0
        completions.append(comp[:end].detach().cpu())
    return completions


def _token_logprobs(policy, prompt_ids, completion_ids):
    device = policy.model.device
    ids = torch.cat([prompt_ids, completion_ids]).unsqueeze(0).to(device)
    logits = policy.model(input_ids=ids).logits[0]
    lp = F.log_softmax(logits.float(), dim=-1)
    plen = prompt_ids.shape[-1]
    idx = torch.arange(plen, ids.shape[-1], device=device)
    return lp[idx - 1, ids[0, idx]]                     # (n_comp,) with grad


def grpo_step(policy, optimizer, batch, gen_cfg, cfg):
    grpo_cfg = cfg["GRPOConfig"]
    from src.verifier import reward as reward_fn
    G, kl_coef, eps = grpo_cfg["G"], grpo_cfg["kl_coef"], 1e-8
    losses, rewards_all, advs_all, kls = [], [], [], []
    degenerate = 0

    for ex, prompt_ids in batch:                        # ex has 'gold'; prompt_ids precomputed
        comps = sample_completions(policy, prompt_ids, G, gen_cfg)
        texts = [policy.tokenizer.decode(c, skip_special_tokens=True) for c in comps]
        r = torch.tensor([reward_fn(t, ex["gold"]) for t in texts])
        rewards_all.extend(r.tolist())
        if r.std() < eps:
            degenerate += 1
            adv = torch.zeros(G)
        else:
            adv = (r - r.mean()) / (r.std() + eps)
        advs_all.extend(adv.tolist())

        for c, a in zip(comps, adv):
            if c.numel() == 0:
                continue
            tok_lp = _token_logprobs(policy, prompt_ids, c)          # grad
            with policy.disable_lora():
                ref_lp = _token_logprobs(policy, prompt_ids, c).detach()
            log_ratio = ref_lp - tok_lp
            kl = (torch.exp(log_ratio) - log_ratio - 1.0).mean()     # k3, token-avg
            pg = -(a.to(tok_lp.device) * tok_lp.mean())
            losses.append(pg + kl_coef * kl)
            kls.append(kl.item())

    loss = torch.stack(losses).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return {
        "mean_reward": float(sum(rewards_all) / len(rewards_all)),
        "mean_adv": float(sum(advs_all) / len(advs_all)),
        "degenerate_frac": degenerate / len(batch),
        "kl": float(sum(kls) / len(kls)) if kls else 0.0,
    }
