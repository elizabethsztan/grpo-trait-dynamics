from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel, GenerationConfig

from src.generation import GeneratedCompletion, generate_completions, resolve_generation_config, truncate_after_eos
from src.logprobs import sequence_logprobs
from src.price import price_covariance


class ConstantPolicy(torch.nn.Module):
    def __init__(self, values, dtype=torch.float32):
        super().__init__()
        self.values = torch.nn.Parameter(torch.tensor(values, dtype=dtype))

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(logits=self.values.expand(*input_ids.shape, -1))


def sample(tokens, eos=2, minimum=1):
    return GeneratedCompletion([0, 1], tokens, "", eos_token_id=eos, min_new_tokens=minimum)


@pytest.mark.parametrize(
    "tokens,eos,pad,expected",
    [
        ([1, 2, 2, 2], 2, 2, [1, 2]),
        ([0, 1, 2, 0, 0], 2, 0, [0, 1, 2]),
        ([1, 0, 1], 2, 0, [1, 0, 1]),
        ([1, 0], None, 0, [1, 0]),
    ],
)
def test_stopping_retains_sampled_tokens_and_terminal_eos(tokens, eos, pad, expected):
    assert truncate_after_eos(tokens, eos, pad) == expected


def test_bfloat16_logits_are_normalized_and_summed_in_float32():
    model = ConstantPolicy([1.125, -2.375, 0.0625, 0.5625], dtype=torch.bfloat16)
    response = sample([0, 2, 1], eos=1)
    actual = sequence_logprobs(model, [response], pad_token_id=3)
    reference = model.values.detach().double().repeat(3, 1)
    reference[0, 1] = -torch.inf
    expected = reference.log_softmax(-1)[range(3), response.completion_ids].sum()
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual[0].double(), expected, atol=1e-6, rtol=0)


def test_suppression_applies_to_each_required_token_and_keeps_gradients():
    model = ConstantPolicy([0.4, -0.3, 1.2])
    response = sample([0, 1, 2], minimum=2)
    actual = sequence_logprobs(model, [response], 0, with_grad=True)[0]
    actual.backward()

    values = model.values.detach().double().requires_grad_()
    masked = values.clone()
    masked[2] = -torch.inf
    expected = masked.log_softmax(-1)[0] + masked.log_softmax(-1)[1] + values.log_softmax(-1)[2]
    expected.backward()
    torch.testing.assert_close(actual.double(), expected, atol=1e-6, rtol=0)
    torch.testing.assert_close(model.values.grad.double(), values.grad, atol=1e-6, rtol=0)


def test_rollout_wrapper_carries_sampling_metadata_into_scoring():
    model = ConstantPolicy([0.2, -0.3, 1.2])
    generated = sample([1, 2])
    wrapped = SimpleNamespace(generated=generated, prompt_ids=generated.prompt_ids, completion_ids=generated.completion_ids)
    direct = sequence_logprobs(model, [generated], 0)
    indirect = sequence_logprobs(model, [wrapped], 0)
    torch.testing.assert_close(direct, indirect)


def test_token_only_samples_retain_raw_policy_scoring():
    model = ConstantPolicy([0.2, -0.3, 1.2])
    response = SimpleNamespace(prompt_ids=[0], completion_ids=[2])
    actual = sequence_logprobs(model, [response], 0)[0]
    torch.testing.assert_close(actual, model.values.log_softmax(-1)[2])


def test_empty_input_returns_float32():
    model = ConstantPolicy([0.2, -0.3, 1.2])
    scores = sequence_logprobs(model, [], 0)
    assert scores.shape == (0,)
    assert scores.dtype == torch.float32


@pytest.mark.parametrize("minimum", [0, 1, 2])
def test_exhaustive_stopped_sequences_normalize_and_satisfy_price_identity(minimum):
    responses = []

    def enumerate_responses(prefix):
        if len(prefix) == 3 or (prefix and prefix[-1] == 2):
            responses.append(sample(prefix, minimum=minimum))
            return
        for token in range(3):
            if token != 2 or len(prefix) >= minimum:
                enumerate_responses(prefix + [token])

    enumerate_responses([])
    before = ConstantPolicy([0.2, -0.4, 0.8])
    after = ConstantPolicy([0.5, 0.1, -0.2])
    p = sequence_logprobs(before, responses, 0).double().exp().numpy()
    q = sequence_logprobs(after, responses, 0).double().exp().numpy()
    trait = np.array([r.completion_ids[0] == 0 for r in responses], dtype=float)
    omega = q / p
    assert p.sum() == pytest.approx(1.0, abs=1e-6)
    assert q.sum() == pytest.approx(1.0, abs=1e-6)
    assert np.sum(p * omega) == pytest.approx(1.0, abs=1e-6)
    observed = np.sum(q * trait) - np.sum(p * trait)
    assert price_covariance(omega, trait, weights=p) == pytest.approx(observed, abs=1e-6)


class TinyTokenizer:
    pad_token = "<pad>"
    pad_token_id = 0
    eos_token = "<eos>"
    eos_token_id = 2

    def encode(self, text, add_special_tokens=True):
        return [4, 5]

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(i) for i in ids if i not in (self.pad_token_id, self.eos_token_id))


@pytest.mark.parametrize("prompt_format", ["plain", "chat"])
@pytest.mark.parametrize("pad", [0, 2])
def test_real_transformers_generation_scores_match_teacher_forced_likelihoods(pad, prompt_format):
    torch.manual_seed(73)
    tokenizer = TinyTokenizer()
    tokenizer.pad_token_id = pad
    if prompt_format == "chat":
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import WhitespaceSplit
        from transformers import PreTrainedTokenizerFast

        backend = Tokenizer(WordLevel({"<pad>": 0, "<user>": 1, "<eos>": 2,
                                       "<assistant>": 3, "test": 4, "prompt": 5, "<unk>": 6}, unk_token="<unk>"))
        backend.pre_tokenizer = WhitespaceSplit()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="<eos>", pad_token="<pad>", unk_token="<unk>")
        tokenizer.pad_token_id = pad
        tokenizer.chat_template = "{% for message in messages %}<{{ message['role'] }}> {{ message['content'] }} <eos> {% endfor %}{% if add_generation_prompt %}<assistant>{% endif %}"
    model = GPT2LMHeadModel(GPT2Config(
        vocab_size=7, n_positions=16, n_embd=16, n_layer=1, n_head=2,
        resid_pdrop=0, embd_pdrop=0, attn_pdrop=0, bos_token_id=4,
        eos_token_id=2, pad_token_id=pad,
    )).eval()
    # An inherited processor would invalidate raw likelihoods. Our explicit
    # sampling policy must not inherit this model-specific suppression list.
    original_config = model.generation_config
    original_config.suppress_tokens = [3]
    captures = []
    original_generate = model.generate

    def capture(**kwargs):
        config = deepcopy(kwargs["generation_config"])
        config.return_dict_in_generate = True
        config.output_scores = True
        kwargs["generation_config"] = config
        result = original_generate(**kwargs)
        captures.append(result)
        return result.sequences

    model.generate = capture
    responses = generate_completions(model, tokenizer, "test prompt", {"max_new_tokens": 5, "prompt_format": prompt_format}, 16, "cpu")
    assert all(r.prompt_ids == ([1, 4, 5, 2, 3] if prompt_format == "chat" else [4, 5]) for r in responses)
    assert model.generation_config is original_config
    assert original_config.suppress_tokens == [3]
    result = captures[0]
    assert torch.isneginf(result.scores[0][:, 2]).all()
    assert torch.isfinite(result.scores[0][:, 3]).all()
    actual = sequence_logprobs(model, responses, pad)
    expected = torch.stack([
        torch.stack([
            result.scores[t][row].float().log_softmax(-1)[token]
            for t, token in enumerate(response.completion_ids)
        ]).sum()
        for row, response in enumerate(responses)
    ])
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=0)
    assert all(r.stop_reason in ("eos", "max_new_tokens") for r in responses)


def test_generation_restores_model_config_on_failure():
    config = GenerationConfig()

    def fail(**kwargs):
        raise RuntimeError("synthetic generation failure")

    model = SimpleNamespace(generation_config=config, generate=fail)
    with pytest.raises(RuntimeError, match="synthetic"):
        generate_completions(model, TinyTokenizer(), "prompt", {}, 1, "cpu")
    assert model.generation_config is config


@pytest.mark.parametrize("fail", [False, True])
def test_lora_generation_restores_wrapper_and_base_configs(fail):
    from peft import LoraConfig, get_peft_model

    base = GPT2LMHeadModel(GPT2Config(
        vocab_size=7, n_positions=16, n_embd=8, n_layer=1, n_head=1,
        bos_token_id=4, eos_token_id=2, pad_token_id=0,
    ))
    model = get_peft_model(base, LoraConfig(
        task_type="CAUSAL_LM", r=2, target_modules=["c_attn"], fan_in_fan_out=True,
    )).eval()
    base_config = base.generation_config
    base_config.suppress_tokens = [3]
    wrapper_config = deepcopy(base_config)
    wrapper_config.suppress_tokens = [4]
    model.generation_config = wrapper_config
    original_generate = base.generate

    def generate(**kwargs):
        assert base.generation_config is model.generation_config
        assert base.generation_config is not base_config
        if fail:
            raise RuntimeError("synthetic LoRA generation failure")
        return original_generate(**kwargs)

    base.generate = generate
    if fail:
        with pytest.raises(RuntimeError, match="synthetic LoRA"):
            generate_completions(model, TinyTokenizer(), "prompt", {"max_new_tokens": 2}, 1, "cpu")
    else:
        generate_completions(model, TinyTokenizer(), "prompt", {"max_new_tokens": 2}, 1, "cpu")
    assert model.generation_config is wrapper_config
    assert base.generation_config is base_config
    assert wrapper_config.suppress_tokens == [4]
    assert base_config.suppress_tokens == [3]


@pytest.mark.parametrize("settings", [
    {"do_sample": False}, {"temperature": 0.7}, {"top_p": 0.9},
    {"top_k": 20}, {"repetition_penalty": 1.1}, {"min_new_tokens": -1},
    {"max_new_tokens": 0}, {"min_new_tokens": 3, "max_new_tokens": 2},
    {"max_new_tokens": 1.5}, {"forced_bos_token_id": 1},
    {"prompt_format": "unknown"},
])
def test_unsupported_sampling_settings_are_rejected(settings):
    with pytest.raises(ValueError):
        resolve_generation_config(settings, TinyTokenizer())
