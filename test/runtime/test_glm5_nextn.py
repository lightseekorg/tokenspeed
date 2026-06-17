import torch

from tokenspeed.runtime.models.glm5_nextn import (
    _mask_glm5_nextn_position_zero_embeddings,
)


def test_glm5_nextn_masks_position_zero_embeddings():
    hidden_states = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    positions = torch.tensor([0, 1, 0, 7], dtype=torch.int32)

    masked = _mask_glm5_nextn_position_zero_embeddings(hidden_states, positions)

    expected = hidden_states.clone()
    expected[0] = 0
    expected[2] = 0
    torch.testing.assert_close(masked, expected)


def test_glm5_nextn_mask_handles_empty_inputs():
    hidden_states = torch.empty((0, 3), dtype=torch.float16)
    positions = torch.empty((0,), dtype=torch.int64)

    masked = _mask_glm5_nextn_position_zero_embeddings(hidden_states, positions)

    assert masked.shape == hidden_states.shape
    assert masked.dtype == hidden_states.dtype
