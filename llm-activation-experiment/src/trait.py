import torch


def _capture_hook_acts(policy, layer, input_ids):
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
            policy.model(input_ids=input_ids)
    finally:
        handle.remove()
    return captured["acts"]  # (1, seq, d_model)


def collect_feature_scores(policy, sae, layer, prompt_ids, completion_ids):
    input_ids = torch.cat([prompt_ids, completion_ids], dim=-1).unsqueeze(0).to(policy.model.device)
    acts = _capture_hook_acts(policy, layer, input_ids)[0]          # (seq, d_model)
    comp_start = prompt_ids.shape[-1]
    comp_acts = acts[comp_start:].to(sae.W_enc_t.dtype)            # completion tokens only, SAE dtype
    feats = sae.encode(comp_acts)                                  # (n_comp, n_features)
    return feats.max(dim=0).values                                 # (n_features,)


def trait_scores(policy, sae, features, prompt_ids, completion_ids, layer):
    all_scores = collect_feature_scores(policy, sae, layer, prompt_ids, completion_ids)
    return all_scores[features]
