from types import SimpleNamespace
from contextlib import contextmanager

import numpy as np
import pytest
import torch

from src.generation import GeneratedCompletion
from src.grpo import (
    RolloutSample,
    _disable_adapters_if_available,
    clipped_grpo_loss,
    compute_group_advantages,
    pad_token_logprobs_to_width,
)
from src.logprobs import token_logprob_batch


class TinyLogitModel(torch.nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.param = torch.nn.Parameter(torch.tensor(0.0))
        self.dtype = dtype

    def forward(self, input_ids, attention_mask=None):
        logits = torch.zeros((*input_ids.shape, 8), dtype=self.dtype, device=input_ids.device)
        logits[..., 3] = self.param.to(self.dtype)
        logits[..., 4] = -self.param.to(self.dtype)
        return SimpleNamespace(logits=logits)


def _sample(prompt_ids, completion_ids):
    return RolloutSample(
        example=SimpleNamespace(),
        generated=GeneratedCompletion(prompt_ids=prompt_ids, completion_ids=completion_ids, completion_text="x", stop_reason="max_new_tokens", stopped_on_answer_tag=False),
        traits=SimpleNamespace(reward=0.0),
    )


def test_token_logprob_batch_masks_padding_and_sums_sequences_in_float32():
    model = TinyLogitModel(dtype=torch.bfloat16)
    samples = [_sample([1, 2], [3, 4]), _sample([1], [3])]

    batch = token_logprob_batch(model, samples, pad_token_id=0, device="cpu", with_grad=False)

    assert batch.token_logprobs.dtype == torch.float32
    assert batch.sequence_logprobs.dtype == torch.float32
    assert batch.mask.tolist() == [[True, True], [True, False]]
    assert torch.allclose(batch.sequence_logprobs, (batch.token_logprobs * batch.mask).sum(dim=1))


def test_equal_rewards_in_group_have_zero_advantages():
    rewards = np.array([[1.0, 1.0, 1.0]])

    advantages = compute_group_advantages(rewards, eps=1e-8)

    assert np.allclose(advantages, 0.0)


def test_clipped_grpo_loss_uses_clipped_ratios_outside_range():
    current = torch.log(torch.tensor([[2.0, 0.5]]))
    old = torch.zeros_like(current)
    mask = torch.tensor([[True, True]])
    advantages = torch.tensor([1.0])

    loss, stats = clipped_grpo_loss(current, old, mask, advantages, clip_range=0.2)

    assert torch.isclose(loss, torch.tensor(-0.85), atol=1e-6)
    assert stats["clip_fraction"] == 1.0


def test_reference_kl_penalty_is_nonnegative_and_added_to_loss():
    current = torch.log(torch.tensor([[0.8, 0.2]]))
    old = current.detach().clone()
    reference = torch.log(torch.tensor([[0.5, 0.5]]))
    mask = torch.tensor([[True, True]])
    advantages = torch.tensor([0.0])

    loss, stats = clipped_grpo_loss(
        current,
        old,
        mask,
        advantages,
        clip_range=0.2,
        reference_token_logprobs=reference,
        kl_coef=0.3,
    )

    expected_kl = (torch.exp(reference - current) - (reference - current) - 1.0).mean()
    assert expected_kl > 0
    assert torch.isclose(loss, 0.3 * expected_kl, atol=1e-6)
    assert stats["reference_kl"] == pytest.approx(expected_kl.item(), abs=1e-6)
    assert stats["policy_loss"] == pytest.approx(0.0, abs=1e-6)


def test_disable_adapters_helper_uses_model_context_when_available():
    class AdapterModel:
        def __init__(self):
            self.disabled = False
            self.entered = False
            self.exited = False

        @contextmanager
        def disable_adapter(self):
            self.entered = True
            self.disabled = True
            try:
                yield
            finally:
                self.disabled = False
                self.exited = True

    model = AdapterModel()

    with _disable_adapters_if_available(model):
        assert model.disabled

    assert model.entered
    assert model.exited
    assert not model.disabled


def test_zero_advantage_grpo_update_does_not_change_parameters():
    model = TinyLogitModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    samples = [_sample([1, 2], [3]), _sample([1, 2], [4])]
    old = token_logprob_batch(model, samples, pad_token_id=0, device="cpu", with_grad=False)
    before = model.param.detach().clone()

    current = token_logprob_batch(model, samples, pad_token_id=0, device="cpu", with_grad=True)
    loss, _ = clipped_grpo_loss(current.token_logprobs, old.token_logprobs, old.mask, torch.zeros(2), clip_range=0.2)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert torch.equal(before, model.param.detach())


def test_pad_token_logprobs_to_width_matches_old_policy_batch_width():
    current = torch.ones((2, 3))

    padded = pad_token_logprobs_to_width(current, width=5)

    assert padded.shape == (2, 5)
    assert torch.equal(padded[:, :3], current)
    assert torch.equal(padded[:, 3:], torch.zeros((2, 2)))
