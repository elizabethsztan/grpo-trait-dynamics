from __future__ import annotations

import re

LAYER_RE = re.compile(r"(?:layers|h|blocks)\.(\d+)\.")


def infer_num_hidden_layers(model) -> int:
    if hasattr(model.config, "num_hidden_layers"):
        return int(model.config.num_hidden_layers)
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is not None:
        return len(layers)
    raise ValueError("could not infer num_hidden_layers from model")


def extract_layer_index(parameter_name: str) -> int | None:
    match = LAYER_RE.search(parameter_name)
    return int(match.group(1)) if match else None


def freeze_base_model(model, freeze_lm_head: bool = True) -> None:
    for param in model.parameters():
        param.requires_grad_(False)
    if freeze_lm_head and hasattr(model, "lm_head"):
        for param in model.lm_head.parameters():
            param.requires_grad_(False)


def apply_lora_above_hook(model, lora_config: dict):
    from peft import LoraConfig, TaskType, get_peft_model

    hook_layer = int(lora_config["hook_layer"])
    num_hidden_layers = infer_num_hidden_layers(model)
    lora_layers = list(range(hook_layer + 1, num_hidden_layers))
    if not lora_layers:
        raise ValueError("hook_layer leaves no transformer blocks available for LoRA")

    freeze_base_model(model, freeze_lm_head=bool(lora_config.get("freeze_lm_head", True)))
    peft_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(lora_config.get("r", 8)),
        lora_alpha=int(lora_config.get("lora_alpha", 16)),
        lora_dropout=float(lora_config.get("lora_dropout", 0.0)),
        bias="none",
        target_modules=list(lora_config["target_modules"]),
        layers_to_transform=lora_layers,
        layers_pattern="layers",
    )
    model = get_peft_model(model, peft_cfg)
    assert_only_lora_above_hook_trainable(model, hook_layer)
    return model


def assert_only_lora_above_hook_trainable(model, hook_layer: int) -> None:
    trainable = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    if not trainable:
        raise AssertionError("no trainable LoRA parameters found")
    for name, _param in trainable:
        if "lora_" not in name:
            raise AssertionError(f"non-LoRA trainable parameter: {name}")
        layer_index = extract_layer_index(name)
        if layer_index is None:
            raise AssertionError(f"trainable LoRA parameter has no layer index: {name}")
        if layer_index <= hook_layer:
            raise AssertionError(f"LoRA parameter at/below hook layer {hook_layer}: {name}")
        if "lm_head" in name:
            raise AssertionError(f"lm_head must not be trainable: {name}")
