import logging
import torch
import torch.nn.functional as F

LOGGER = logging.getLogger(__name__)


def _exact_span(comp, eos):
    # The exact sampled action: up to and INCLUDING the terminating EOS (once the model
    # emits EOS, generation stops and everything after is padding), or all emitted tokens
    # if generation was truncated at max_new_tokens (no EOS). pad_token_id == eos_token_id
    # for this model, so we detect the FIRST EOS rather than stripping trailing pads --
    # stripping trailing pads would drop the real terminating EOS and make omega the
    # ratio of the prefix probability (omitting the EOS-token importance factor).
    hit = (comp == eos).nonzero()
    end = (hit[0].item() + 1) if len(hit) else comp.shape[0]
    return comp[:end].detach().cpu()


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
    eos = policy.tokenizer.eos_token_id
    return [_exact_span(row[plen:], eos) for row in out]


def sample_completions_batch(policy, prompt_ids_list, gen_cfg):
    # Generate one completion for each (possibly different-length) prompt in ONE batched
    # generate() call. Prompts are LEFT-padded so all rows begin generating at the same
    # index; verified numerically identical to per-prompt generation for this SSM-hybrid
    # model (leading pads do not pollute the recurrent state). Returns list[Tensor] of the
    # exact sampled spans (up to and including the terminating EOS; see _exact_span).
    device = policy.model.device
    pad = policy.tokenizer.pad_token_id
    L = max(x.shape[0] for x in prompt_ids_list)
    ids = torch.full((len(prompt_ids_list), L), pad, dtype=torch.long)
    attn = torch.zeros((len(prompt_ids_list), L), dtype=torch.long)
    for i, x in enumerate(prompt_ids_list):
        ids[i, L - x.shape[0]:] = x
        attn[i, L - x.shape[0]:] = 1
    with torch.no_grad():
        out = policy.model.generate(
            ids.to(device), attention_mask=attn.to(device),
            do_sample=True, temperature=gen_cfg["temperature"], top_p=gen_cfg["top_p"],
            top_k=(gen_cfg["top_k"] or 0), max_new_tokens=gen_cfg["max_new_tokens"],
            pad_token_id=pad,
        )
    eos = policy.tokenizer.eos_token_id
    # new tokens start after the left-padded prompt block (index L)
    return [_exact_span(row[L:], eos) for row in out]


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
