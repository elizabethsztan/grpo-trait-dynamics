import torch


def _capture_hook_acts(policy, layer, input_ids, attention_mask=None):
    # Single teacher-forced forward pass; grab the hook module's OUTPUT
    # (post-block residual) for every position. This is the exact activation
    # the Qwen-Scope SAE was trained on (see src/sae.py hook convention).
    captured = {}
    module = policy.hook_module(layer)

    def hook(_m, _inp, out):
        captured["acts"] = (out[0] if isinstance(out, tuple) else out).detach()

    handle = module.register_forward_hook(hook)
    try:
        with torch.no_grad():
            policy.model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        handle.remove()
    return captured["acts"]  # (B, seq, d_model)


def collect_feature_scores(policy, sae, layer, prompt_ids, completion_ids):
    input_ids = torch.cat([prompt_ids, completion_ids], dim=-1).unsqueeze(0).to(policy.model.device)
    acts = _capture_hook_acts(policy, layer, input_ids)[0]          # (seq, d_model)
    comp_start = prompt_ids.shape[-1]
    comp_acts = acts[comp_start:].to(sae.W_enc_t.dtype)            # completion tokens only, SAE dtype
    feats = sae.encode(comp_acts)                                  # (n_comp, n_features)
    return feats.max(dim=0).values                                 # (n_features,)


def collect_feature_scores_batch(policy, sae, layer, pairs):
    # Batched teacher-forced trait scores over a list of (prompt_ids, completion_ids).
    # Right-pad the full sequences (pads sit AFTER real tokens -> real-position acts are
    # unaffected in a causal/recurrent model); read each row's completion-token acts,
    # SAE-encode, max over completion tokens. Returns a (len(pairs), n_features) tensor
    # on the SAE device. Rows with empty completions yield a zero feature vector.
    dev = policy.model.device
    pad = policy.tokenizer.pad_token_id
    seqs = [torch.cat([p, c]) for p, c in pairs]
    Lmax = max(s.shape[0] for s in seqs)
    ids = torch.full((len(seqs), Lmax), pad, dtype=torch.long)
    attn = torch.zeros((len(seqs), Lmax), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :s.shape[0]] = s
        attn[i, :s.shape[0]] = 1
    acts = _capture_hook_acts(policy, layer, ids.to(dev), attn.to(dev))   # (B, Lmax, d_model)
    out = torch.zeros(len(pairs), sae.n_features, device=acts.device, dtype=torch.float32)
    for i, (p, c) in enumerate(pairs):
        clen = c.shape[0]
        if clen == 0:
            continue
        pstart = p.shape[0]
        comp_acts = acts[i, pstart:pstart + clen].to(sae.W_enc_t.dtype)
        feats = sae.encode(comp_acts)
        out[i] = feats.max(dim=0).values.float()
    return out


def trait_scores(policy, sae, features, prompt_ids, completion_ids, layer):
    all_scores = collect_feature_scores(policy, sae, layer, prompt_ids, completion_ids)
    return all_scores[features]
