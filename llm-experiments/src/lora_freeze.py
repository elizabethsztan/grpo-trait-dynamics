from __future__ import annotations

import re

LAYER_RE = re.compile(r"(?:layers|h|blocks)\.(\d+)\.")
LORA_MODULE_RE = re.compile(r"\.([^.]+)\.lora_")


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


def resolve_lora_layers(num_hidden_layers: int, lora_config: dict) -> list[int]:
    scope = str(lora_config.get("layer_scope", "above_hook"))
    if scope == "all":
        layers = list(range(num_hidden_layers))
    elif scope == "above_hook":
        hook_layer = int(lora_config["hook_layer"])
        layers = list(range(hook_layer + 1, num_hidden_layers))
    else:
        raise ValueError(f"unsupported LoRA layer_scope: {scope!r}")
    if not layers:
        raise ValueError(f"LoRA layer_scope {scope!r} selects no transformer blocks")
    return layers


def trainable_lora_layer_indices(model) -> list[int]:
    indices = set()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        layer_index = extract_layer_index(name)
        if layer_index is not None:
            indices.add(layer_index)
    return sorted(indices)


def assert_only_lora_layers_trainable(
    model,
    allowed_layers: list[int],
    target_modules: list[str] | None = None,
) -> None:
    allowed = set(allowed_layers)
    trainable = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    if not trainable:
        raise AssertionError("no trainable LoRA parameters found")
    observed_layers = set()
    observed_layer_modules = set()
    for name, _param in trainable:
        if "lora_" not in name:
            raise AssertionError(f"non-LoRA trainable parameter: {name}")
        layer_index = extract_layer_index(name)
        if layer_index is None:
            raise AssertionError(f"trainable LoRA parameter has no layer index: {name}")
        if layer_index not in allowed:
            raise AssertionError(f"LoRA parameter outside selected layers: {name}")
        observed_layers.add(layer_index)
        if target_modules:
            module_match = LORA_MODULE_RE.search(name)
            module = module_match.group(1) if module_match else None
            if module not in target_modules:
                raise AssertionError(f"unexpected LoRA target module in parameter: {name}")
            observed_layer_modules.add((layer_index, module))
        if "lm_head" in name:
            raise AssertionError(f"lm_head must not be trainable: {name}")
    missing_layers = sorted(allowed - observed_layers)
    if missing_layers:
        raise AssertionError(f"missing selected layers in trainable LoRA parameters: {missing_layers}")
    if target_modules:
        expected = {(layer, module) for layer in allowed for module in target_modules}
        missing_modules = sorted(expected - observed_layer_modules)
        if missing_modules:
            raise AssertionError(f"missing LoRA target modules by layer: {missing_modules}")


def apply_lora(model, lora_config: dict):
    from peft import LoraConfig, TaskType, get_peft_model

    num_hidden_layers = infer_num_hidden_layers(model)
    lora_layers = resolve_lora_layers(num_hidden_layers, lora_config)

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
    assert_only_lora_layers_trainable(
        model,
        lora_layers,
        target_modules=list(lora_config["target_modules"]),
    )
    return model


def apply_lora_above_hook(model, lora_config: dict):
    config = dict(lora_config)
    config["layer_scope"] = "above_hook"
    return apply_lora(model, config)


def assert_only_lora_above_hook_trainable(model, hook_layer: int) -> None:
    trainable_layers = trainable_lora_layer_indices(model)
    if not trainable_layers:
        raise AssertionError("no trainable LoRA parameters found")
    assert_only_lora_layers_trainable(model, [layer for layer in trainable_layers if layer > hook_layer])
