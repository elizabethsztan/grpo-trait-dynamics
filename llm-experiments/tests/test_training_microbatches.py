from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from src import grpo
from src.data import generate_examples
from src.generation import GeneratedCompletion
from src.traits import evaluate_completion_traits


@pytest.mark.parametrize('microbatch', [1, 2, 8])
def test_microbatches_match_full_batch_gradient_and_one_clipped_update(monkeypatch, microbatch):
    torch.manual_seed(64)
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=8, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, attention_dropout=0)).eval()
    examples = generate_examples(5, 72, 'train')
    samples = []
    for group, example in enumerate(examples):
        for response in range(4):
            choice = example.gold_choice if response < group % 4 else next(c for c in 'ABCD' if c != example.gold_choice)
            generated = GeneratedCompletion([1, 3], [3+response] * (1+group%3) + [2],
                f'<answer>{choice}</answer>', eos_token_id=2, min_new_tokens=1)
            samples.append(grpo.RolloutSample(example, generated, evaluate_completion_traits(generated.completion_text, example)))
    sampling_calls = []
    def sample(*args, **kwargs):
        sampling_calls.append(1)
        return samples
    monkeypatch.setattr(grpo, 'sample_rollouts', sample)
    reference = deepcopy(model)
    ref_optimizer = torch.optim.AdamW(reference.parameters(), lr=.01)
    rewards = np.array([s.traits.reward for s in samples]).reshape(5, 4)
    advantages = torch.tensor(grpo.compute_group_advantages(rewards, 1e-8).ravel(), dtype=torch.float32)
    reference_logp = grpo.sequence_logprobs(reference, samples, 0, with_grad=True)
    reference_loss = -(advantages * reference_logp).mean()
    ref_optimizer.zero_grad()
    reference_loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), .01)
    assert norm > .01  # Ensure the test catches clipping separately per chunk.
    expected_grads = [p.grad.clone() for p in reference.parameters()]
    ref_optimizer.step()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    counts = {'zero': 0, 'step': 0}
    zero, step = optimizer.zero_grad, optimizer.step
    def counted_zero():
        counts['zero'] += 1
        zero()
    def counted_step():
        counts['step'] += 1
        step()
    monkeypatch.setattr(optimizer, 'zero_grad', counted_zero)
    monkeypatch.setattr(optimizer, 'step', counted_step)
    summary, returned = grpo.train_grpo_step(model, SimpleNamespace(pad_token_id=0), optimizer, examples,
        {}, 4, 1e-8, .01, 'cpu', microbatch_prompts=microbatch)
    assert counts == {'zero': 1, 'step': 1} and len(sampling_calls) == 1
    assert summary['loss'] == pytest.approx(reference_loss.item(), abs=2e-6)
    assert [s.completion_ids for s in returned] == [s.completion_ids for s in samples]
    np.testing.assert_allclose([s.pre_logprob for s in returned], reference_logp.detach().numpy(), atol=2e-6, rtol=0)
    for p, expected, grad in zip(model.parameters(), reference.parameters(), expected_grads):
        torch.testing.assert_close(p.grad, grad, atol=1e-7, rtol=1e-5)
        torch.testing.assert_close(p, expected, atol=2e-6, rtol=1e-5)


@pytest.mark.parametrize('value', [0, -1, 1.5, True])
def test_invalid_microbatch_is_rejected_before_sampling(value):
    with pytest.raises(ValueError, match='microbatch'):
        grpo.train_grpo_step(None, None, None, [], {}, 4, 1e-8, 1, 'cpu', microbatch_prompts=value)
