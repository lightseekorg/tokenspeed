# Copyright (c) 2026 LightSeek Foundation

import pytest
from tokenspeed_kernel_amd.ops.gfx950.moe.fp16.latent_input_small_batch import (
    _BLOCK_K,
    _split_k,
)


@pytest.mark.parametrize("hidden_size", [64, 128, 192, 256, 7168])
def test_split_k_covers_every_tile(hidden_size: int) -> None:
    split_k = _split_k(tokens=2, total_n=6016, hidden=hidden_size, block_m=16)
    assert (hidden_size // _BLOCK_K) % split_k == 0


def test_split_k_does_not_drop_k_tiles() -> None:
    assert _split_k(tokens=2, total_n=6016, hidden=192, block_m=16) == 1
