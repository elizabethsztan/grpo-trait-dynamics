"""Qwen-Scope residual-stream SAE loader.

HOOK CONVENTION (confirmed via Task 2 against the actual repo
Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_100 — its README.md, config.json,
app.py demo, and the layer12.sae.pt tensors):

  - Hook point: the **post-block residual stream**, i.e. the OUTPUT of
    transformers module `model.layers.{L}`. app.py registers the hook as
    `model.model.layers[layer].register_forward_hook(...)` and reads the
    layer output. LoRA must therefore live strictly on layers > L so the
    trait score s is frozen (Delta s = 0).
  - TopK SAE, k = 100 (kept non-zero features per position), width 32768,
    d_model 2048.
  - Checkpoint file per layer: `layer{L}.sae.pt`, a plain torch dict with
    float32 tensors:
        W_enc: (n_features, d_model) = (32768, 2048)
        W_dec: (d_model, n_features) = (2048, 32768)
        b_enc: (n_features,)         = (32768,)
        b_dec: (d_model,)            = (2048,)
  - Encode (matches app.py `compute_sae_features` EXACTLY):
        pre      = hidden @ W_enc.T + b_enc        # NOTE: no b_dec subtraction
        features = topk_relu(pre, k) = scatter(top_k(relu(pre)))

  Update this docstring + hook_name/encode together if the repo changes.
"""
import logging
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

LOGGER = logging.getLogger(__name__)

DEFAULT_TOPK = 100


class SAE(nn.Module):
    def __init__(self, W_enc, b_enc, W_dec, b_dec, k=DEFAULT_TOPK):
        super().__init__()
        # W_enc stored as (n_features, d_model); keep the transposed encoder
        # (d_model, n_features) so encode is a single matmul, as in app.py.
        self.register_buffer("W_enc_t", W_enc.t().contiguous())  # (d_model, n_features)
        self.register_buffer("b_enc", b_enc)                     # (n_features,)
        self.register_buffer("W_dec", W_dec)                     # (d_model, n_features)
        self.register_buffer("b_dec", b_dec)                     # (d_model,)
        self.k = int(k)
        self.d_model = W_enc.shape[1]
        self.n_features = W_enc.shape[0]

    @torch.no_grad()
    def encode(self, acts):
        # acts: (..., d_model) -> features (..., n_features), TopK sparse.
        # Exactly app.py: pre = hidden @ W_enc.T + b_enc, then relu -> top-k scatter.
        pre = acts @ self.W_enc_t + self.b_enc
        if self.k and self.k < self.n_features:
            relu_pre = torch.relu(pre)
            topv, topi = torch.topk(relu_pre, self.k, dim=-1)
            out = torch.zeros_like(relu_pre)
            out.scatter_(-1, topi, topv)
            return out
        return torch.relu(pre)

    @staticmethod
    def hook_name(layer):
        # Module whose OUTPUT is the SAE-trained (post-block residual) activation.
        return f"model.layers.{layer}"


def load_sae(repo, layer, device, k=DEFAULT_TOPK):
    weights_path = hf_hub_download(repo, filename=f"layer{layer}.sae.pt")
    sd = torch.load(weights_path, map_location="cpu")
    sae = SAE(sd["W_enc"], sd["b_enc"], sd["W_dec"], sd["b_dec"], k=k)
    sae = sae.to(device).eval()
    LOGGER.info(f"loaded SAE repo={repo} layer={layer} d_model={sae.d_model} "
                f"n_features={sae.n_features} k={sae.k} hook={SAE.hook_name(layer)}")
    return sae
