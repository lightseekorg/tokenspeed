"""CUDA storage tests for the MiniMax-style indexed MHA cache."""

from __future__ import annotations

import pytest
import torch

from tokenspeed.runtime.layers.attention.kv_cache.indexed_mha import (
    IndexedMHATokenToKVPool,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Indexed MHA cache storage requires CUDA.",
)


@pytest.mark.parametrize(
    ("dtype", "store_dtype", "bytes_per_element"),
    [
        (torch.bfloat16, torch.bfloat16, 2),
        (torch.float8_e4m3fn, torch.uint8, 1),
    ],
)
def test_indexed_mha_cache_storage_move_clear_and_bytes(
    dtype: torch.dtype,
    store_dtype: torch.dtype,
    bytes_per_element: int,
) -> None:
    size = 128
    page_size = 128
    layers = 2
    head_dim = 128
    pool = IndexedMHATokenToKVPool(
        size=size,
        dtype=dtype,
        head_num=1,
        head_dim=head_dim,
        layer_num=layers,
        device="cuda",
        enable_memory_saver=False,
        max_batch_size=1,
        max_context_len=size,
        page_size=page_size,
        rank=0,
        index_head_dim=head_dim,
        index_dtype=dtype,
        enable_alt_stream=False,
    )

    rows = size + page_size
    assert pool.index_dtype is dtype
    assert pool.index_store_dtype is store_dtype
    assert pool.index_k_buffer[0].shape == (rows, head_dim)
    assert pool.index_k_buffer[0].dtype is store_dtype
    assert pool.get_index_k_buffer(0).dtype is dtype

    source = torch.tensor([7], dtype=torch.int64, device="cuda")
    target = torch.tensor([11], dtype=torch.int64, device="cuda")
    values = torch.linspace(-2, 2, head_dim, dtype=torch.float32, device="cuda").to(
        dtype
    )
    raw_values = values if store_dtype is dtype else values.view(store_dtype)
    for raw_cache in pool.index_k_buffer:
        raw_cache[source] = raw_values
    pool.move_kv_cache(target, source)
    for layer_id in range(layers):
        torch.testing.assert_close(
            pool.get_index_k_buffer(layer_id)[target],
            values.unsqueeze(0),
            rtol=0,
            atol=0,
        )

    key_bytes, value_bytes = pool.get_kv_size_bytes()
    tensor_bytes = rows * head_dim * layers * bytes_per_element
    assert key_bytes == tensor_bytes * 2  # main K plus index K
    assert value_bytes == tensor_bytes

    pool.clear_kv_buffers()
    assert all(not raw_cache.any() for raw_cache in pool.index_k_buffer)
