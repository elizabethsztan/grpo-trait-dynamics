import numpy as np
import torch

from train_grpo_price import _activation_scores_to_numpy


def test_activation_scores_to_numpy_casts_bfloat16_to_float32():
    scores = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)

    result = _activation_scores_to_numpy(scores)

    assert result.dtype == np.float32
    assert np.allclose(result, np.array([1.0, 2.0], dtype=np.float32))
