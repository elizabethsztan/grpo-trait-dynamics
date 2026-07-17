from types import SimpleNamespace

import numpy as np
import torch

from src.activation_probe import ActivationAgreementProbe
from src import activation_probe
from src.generation import GeneratedCompletion
from src.grpo import RolloutSample
from train_grpo_price import _activation_scores_to_numpy


def test_activation_scores_to_numpy_casts_bfloat16_to_float32():
    scores = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)

    result = _activation_scores_to_numpy(scores)

    assert result.dtype == np.float32
    assert np.allclose(result, np.array([1.0, 2.0], dtype=np.float32))


class TokenIdHiddenModel(torch.nn.Module):
    def forward(self, input_ids, attention_mask=None, output_hidden_states=False):
        hidden = input_ids.float().unsqueeze(-1)
        return SimpleNamespace(hidden_states=[torch.zeros_like(hidden), hidden])


def _sample(prompt_ids, completion_ids):
    return RolloutSample(
        example=SimpleNamespace(),
        generated=GeneratedCompletion(
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            completion_text="ignored by score_samples",
            stop_reason="answer_stop",
            stopped_on_answer_tag=True,
        ),
        traits=SimpleNamespace(reward=0.0),
    )


def test_activation_score_samples_uses_exact_sample_token_ids():
    model = TokenIdHiddenModel()
    probe = ActivationAgreementProbe(
        hook_layer=0,
        vector=torch.tensor([1.0]),
        score_mean=0.0,
        score_std=1.0,
    )
    samples = [
        _sample(prompt_ids=[1000, 2000], completion_ids=[2, 4]),
        _sample(prompt_ids=[3000], completion_ids=[6]),
    ]

    scores = probe.score_samples(model, samples, device="cpu")

    assert torch.allclose(scores, torch.tensor([3.0, 6.0]))


def test_activation_answer_tag_pooling_ignores_tokens_after_stop_suffix():
    model = TokenIdHiddenModel()
    probe = ActivationAgreementProbe(
        hook_layer=0,
        vector=torch.tensor([1.0]),
        score_mean=0.0,
        score_std=1.0,
        pooling="answer_tag_tokens",
        stop_token_ids=(4,),
    )
    samples = [
        _sample(prompt_ids=[1000], completion_ids=[2, 4, 100]),
    ]

    scores = probe.score_samples(model, samples, device="cpu")

    assert torch.allclose(scores, torch.tensor([3.0]))


def test_activation_whole_completion_pooling_includes_all_completion_tokens():
    model = TokenIdHiddenModel()
    probe = ActivationAgreementProbe(
        hook_layer=0,
        vector=torch.tensor([1.0]),
        score_mean=0.0,
        score_std=1.0,
        pooling="whole_completion_tokens",
        stop_token_ids=(4,),
    )
    samples = [
        _sample(prompt_ids=[1000], completion_ids=[2, 4, 100]),
    ]

    scores = probe.score_samples(model, samples, device="cpu")

    assert torch.allclose(scores, torch.tensor([(2.0 + 4.0 + 100.0) / 3.0]))


def test_activation_mean_completion_tokens_alias_matches_whole_completion_pooling():
    model = TokenIdHiddenModel()
    probe = ActivationAgreementProbe(
        hook_layer=0,
        vector=torch.tensor([1.0]),
        score_mean=0.0,
        score_std=1.0,
        pooling="mean_completion_tokens",
        stop_token_ids=(4,),
    )
    samples = [
        _sample(prompt_ids=[1000], completion_ids=[2, 4, 100]),
    ]

    scores = probe.score_samples(model, samples, device="cpu")

    assert torch.allclose(scores, torch.tensor([(2.0 + 4.0 + 100.0) / 3.0]))


def test_build_activation_probe_passes_configured_pooling_through_construction(monkeypatch):
    calls = []

    def fake_pool_completion_hidden(model, tokenizer, prompt_texts, completion_texts, hook_layer, device, pooling="answer_tag_tokens"):
        calls.append(pooling)
        n = len(prompt_texts)
        call_index = len(calls)
        if call_index == 1:
            return torch.ones((n, 2))
        if call_index == 2:
            return torch.zeros((n, 2))
        return torch.arange(float(n)).unsqueeze(1).repeat(1, 2)

    monkeypatch.setattr(activation_probe, "pool_completion_hidden", fake_pool_completion_hidden)

    probe = activation_probe.build_activation_probe(
        model=None,
        tokenizer=SimpleNamespace(encode=lambda text, add_special_tokens=False: [4]),
        cfg={
            "hook_layer": 0,
            "pooling": "whole_completion_tokens",
            "num_probe_pairs": 2,
            "normalization_pairs": 2,
        },
        data_cfg={
            "difficulty": "easy",
            "train_hint_phrases": ["hint {choice}"],
        },
        seed=1,
        device="cpu",
    )

    assert probe.pooling == "whole_completion_tokens"
    assert calls == ["whole_completion_tokens", "whole_completion_tokens", "whole_completion_tokens"]
