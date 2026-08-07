"""Qwen3.5-2B-Base policy loader + downstream-only LoRA.

MODEL NOTES (confirmed via Task 5 against Qwen/Qwen3.5-2B-Base):
  - The repo advertises a multimodal Qwen3_5ForConditionalGeneration, but
    AutoModelForCausalLM resolves to the TEXT-ONLY `Qwen3_5ForCausalLM`
    (vision tower dropped). Decoder layers live at `model.layers.{i}`
    (24 layers, hidden 2048 == SAE d_model), matching SAE.hook_name.
  - Hybrid architecture: layers {3,7,11,15,19,23} are `full_attention`
    (self_attn.{q,k,v,o}_proj); the rest are `linear_attention`
    (linear_attn.{in_proj_a/b/qkv/z, out_proj} + a conv1d). Both types have
    mlp.{gate,up,down}_proj. All projections are nn.Linear.
  - LoRA target (lora_layers=">L", the default) = EVERY nn.Linear leaf under
    layers strictly > L (introspected, not hardcoded), so the trait score at
    layer L stays frozen (Delta s = 0). The frozen-trait assertion in Phase 2 is
    the runtime backstop. With lora_layers="all" the target extends down to layer
    0, so the trait itself moves (Delta s != 0) and the Price transmission term
    E[omega*Delta s] is nonzero -- used to measure how much that term drives ΔT.
"""
import logging
from contextlib import contextmanager
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

LOGGER = logging.getLogger(__name__)


class Policy:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self._has_lora = False

    def _base(self):
        # The underlying (possibly peft-wrapped) HF model.
        return self.model.get_base_model() if self._has_lora else self.model

    def n_layers(self):
        return len(self._base().get_submodule("model.layers"))

    def hook_module(self, layer):
        # The module whose OUTPUT is the SAE-trained activation (post-block residual).
        return self._base().get_submodule(f"model.layers.{layer}")

    def attach_lora(self, layer_L, rank, layers=">L"):
        # layers=">L": LoRA strictly downstream of the hook (every nn.Linear under
        #   layers > L), so the trait at L is frozen (Delta s = 0) and Price's
        #   transmission term vanishes -- the headline setup.
        # layers="all": extend the target down to layer 0, so the trait moves with
        #   the policy (Delta s != 0) and the transmission term E[omega*Delta s] is
        #   nonzero. The Phase-2 frozen assertion must be skipped for this mode.
        if layers not in (">L", "all"):
            raise ValueError(f"lora_layers must be '>L' or 'all', got {layers!r}")
        n_layers = self.n_layers()
        base = self._base()
        start = 0 if layers == "all" else layer_L + 1
        target_modules = []
        for i in range(start, n_layers):
            layer = base.get_submodule(f"model.layers.{i}")
            for name, mod in layer.named_modules():
                if isinstance(mod, nn.Linear):
                    target_modules.append(f"model.layers.{i}.{name}")
        cfg = LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.0,
                         target_modules=target_modules, bias="none")
        self.model = get_peft_model(self.model, cfg)
        self._has_lora = True
        LOGGER.info(f"attached LoRA rank={rank} (lora_layers={layers}) to layers "
                    f"{start}..{n_layers - 1} ({len(target_modules)} Linear modules)")

    @contextmanager
    def disable_lora(self):
        # Reference policy pi_ref = pi_0 = base model (LoRA off).
        if not self._has_lora:
            yield
            return
        with self.model.disable_adapter():
            yield

    def save_lora(self, path):
        self.model.save_pretrained(str(path))

    def load_lora(self, path):
        # First call wraps the base model with the saved adapter. Subsequent calls
        # SWAP the adapter weights into the SAME wrapper (set_peft_model_state_dict)
        # rather than re-wrapping — re-wrapping stacks adapters and corrupts pi_t.
        from peft import PeftModel, load_peft_weights, set_peft_model_state_dict
        if not self._has_lora:
            self.model = PeftModel.from_pretrained(self.model, str(path))
            self._has_lora = True
        else:
            sd = load_peft_weights(str(path))
            set_peft_model_state_dict(self.model, sd)


def load_policy(cfg, device):
    name = cfg["ModelConfig"]["backbone"]
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to(device).eval()
    policy = Policy(model, tok)
    LOGGER.info(f"loaded backbone={name} class={type(model).__name__} layers={policy.n_layers()}")
    return policy
