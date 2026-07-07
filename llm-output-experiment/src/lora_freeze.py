from __future__ import annotations


def freeze_base_model(model, freeze_lm_head: bool = True) -> None:
    for param in model.parameters():
        param.requires_grad_(False)
    if freeze_lm_head and hasattr(model, "lm_head"):
        for param in model.lm_head.parameters():
            param.requires_grad_(False)


def apply_lora(model, lora_config: dict):
    from peft import LoraConfig, TaskType, get_peft_model

    freeze_base_model(model, freeze_lm_head=bool(lora_config.get("freeze_lm_head", True)))
    peft_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(lora_config.get("r", 8)),
        lora_alpha=int(lora_config.get("lora_alpha", 16)),
        lora_dropout=float(lora_config.get("lora_dropout", 0.0)),
        bias="none",
        target_modules=list(lora_config["target_modules"]),
    )
    model = get_peft_model(model, peft_cfg)
    assert_only_lora_trainable(model)
    return model


def assert_only_lora_trainable(model) -> None:
    trainable = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    if not trainable:
        raise AssertionError("no trainable LoRA parameters found")
    for name, _param in trainable:
        if "lora_" not in name:
            raise AssertionError(f"non-LoRA trainable parameter: {name}")
        if "lm_head" in name:
            raise AssertionError(f"lm_head must not be trainable: {name}")
