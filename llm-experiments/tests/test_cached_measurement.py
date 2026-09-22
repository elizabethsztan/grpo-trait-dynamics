from copy import deepcopy
from dataclasses import replace

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from src.generation import generate_completions
from src.logprobs import cached_sequence_logprobs


class Tokenizer:
    pad_token, eos_token = 'pad', 'eos'
    pad_token_id, eos_token_id = 0, 2

    def encode(self, text, add_special_tokens=True):
        return [1, 3, 4]

    def decode(self, ids, skip_special_tokens=True):
        return str(ids)


def model_and_tokenizer(pad=0):
    torch.manual_seed(52)
    tokenizer = Tokenizer()
    tokenizer.pad_token_id = pad
    model = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=8, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=32,
        eos_token_id=2, pad_token_id=pad, attention_dropout=0,
    )).eval()
    return model, tokenizer


@pytest.mark.parametrize('batch', [1, 2, 8])
@pytest.mark.parametrize('pad', [0, 2])
@pytest.mark.parametrize('minimum', [0, 2])
def test_capture_and_replay_preserve_stopping_and_generation_batch(batch, pad, minimum):
    model, tokenizer = model_and_tokenizer(pad)
    config = {'max_new_tokens': 6, 'min_new_tokens': minimum}
    samples = generate_completions(model, tokenizer, 'prompt', config, batch, 'cpu', capture_logprobs=True)
    replay = cached_sequence_logprobs(model, samples, pad, 'cpu')
    torch.testing.assert_close(replay, torch.tensor([s.sampling_logprob for s in samples]), atol=1e-6, rtol=0)
    assert all(s.generation_batch_size == batch for s in samples)
    if batch == 8:
        assert len({len(s.completion_ids) for s in samples}) > 1
        assert any(s.stop_reason == 'eos' for s in samples)
        assert any(s.stop_reason == 'max_new_tokens' for s in samples)
    assert not replay.requires_grad


def test_capture_does_not_change_samples_rng_or_model_generation_configuration():
    model, tokenizer = model_and_tokenizer()
    original = model.generation_config
    before = deepcopy(original.to_dict())
    torch.manual_seed(73)
    plain = generate_completions(model, tokenizer, 'prompt', {'max_new_tokens': 6}, 8, 'cpu')
    rng = torch.get_rng_state().clone()
    torch.manual_seed(73)
    captured = generate_completions(model, tokenizer, 'prompt', {'max_new_tokens': 6}, 8, 'cpu', True)
    assert [s.completion_ids for s in plain] == [s.completion_ids for s in captured]
    assert all(s.sampling_logprob is None for s in plain)
    torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
    assert model.generation_config is original
    # configure_tokenizer_and_model sets the already-identical pad ID.
    assert original.to_dict() == before


def test_post_update_replay_matches_generation_on_the_old_responses(monkeypatch):
    from peft import LoraConfig, get_peft_model

    model, tokenizer = model_and_tokenizer()
    model = get_peft_model(model, LoraConfig(task_type='CAUSAL_LM', r=2, target_modules=['q_proj','v_proj'])).eval()
    samples = generate_completions(model, tokenizer, 'prompt', {'max_new_tokens': 6}, 8, 'cpu', True)
    old = torch.tensor([s.sampling_logprob for s in samples])
    # Exercise a real adapter gradient/update, then evaluate the original responses.
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.1)
    loss = -model(input_ids=torch.tensor([[1, 3, 4]])).logits[0, -1].log_softmax(-1)[3]
    loss.backward()
    optimizer.step()
    actual = cached_sequence_logprobs(model, samples, tokenizer.pad_token_id)
    assert not torch.allclose(actual, old, atol=1e-5, rtol=0)
    # Force the sampler's choices, preserving the updated model's original scores.
    # This is a diagnostic reference only, not the production replay method.
    step = 0
    def choose(probs, num_samples, **kwargs):
        nonlocal step
        ids = [s.completion_ids[step] if step < len(s.completion_ids) else tokenizer.pad_token_id for s in samples]
        step += 1
        return torch.tensor(ids, device=probs.device)[:, None]
    monkeypatch.setattr(torch, 'multinomial', choose)
    reference = generate_completions(model, tokenizer, 'prompt', {'max_new_tokens': 6}, 8, 'cpu', True)
    assert [s.completion_ids for s in reference] == [s.completion_ids for s in samples]
    torch.testing.assert_close(actual, torch.tensor([s.sampling_logprob for s in reference]), atol=1e-6, rtol=0)


def test_replay_rejects_regrouped_samples_and_training_mode():
    model, tokenizer = model_and_tokenizer()
    samples = generate_completions(model, tokenizer, 'prompt', {'max_new_tokens': 3}, 2, 'cpu', True)
    with pytest.raises(ValueError, match='intact'):
        cached_sequence_logprobs(model, samples[:1], tokenizer.pad_token_id)
    with pytest.raises(ValueError, match='intact'):
        cached_sequence_logprobs(model, [samples[0], replace(samples[1], prompt_ids=[1, 5])], tokenizer.pad_token_id)
    model.train()
    with pytest.raises(ValueError, match='eval'):
        cached_sequence_logprobs(model, samples, tokenizer.pad_token_id)


def test_exhaustive_cached_distribution_normalizes_and_obeys_price_identity():
    from src.generation import GeneratedCompletion
    from src.price import price_covariance
    import numpy as np

    model, tokenizer = model_and_tokenizer()
    # All traces at max_new_tokens=2, min_new_tokens=1: seven non-EOS
    # first tokens followed by any token (including terminal EOS).
    samples = [GeneratedCompletion([1, 3, 4], [a, b], '', eos_token_id=2,
               min_new_tokens=1, generation_batch_size=1) for a in range(8) if a != 2 for b in range(8)]
    def probabilities():
        return np.array([cached_sequence_logprobs(model, [s], 0)[0].double().exp().item() for s in samples])
    p = probabilities()
    with torch.no_grad():
        model.lm_head.weight[3].add_(.3)
    q = probabilities()
    assert p.sum() == pytest.approx(1, abs=1e-6)
    assert q.sum() == pytest.approx(1, abs=1e-6)
    trait = np.array([s.completion_ids[0] == 3 for s in samples], dtype=float)
    assert price_covariance(q/p, trait, weights=p) == pytest.approx(np.sum((q-p)*trait), abs=1e-6)
