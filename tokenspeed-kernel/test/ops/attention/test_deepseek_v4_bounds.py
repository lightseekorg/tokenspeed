# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel import (
    deepseek_v4_csa_indexer_fp8_cache_insert,
    deepseek_v4_paged_selected_attention,
    deepseek_v4_selected_attention,
    deepseek_v4_swa_cache_insert,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")

PAGE_SIZE = 64
HEAD_DIM = 512
TOKEN_BYTES = 576
SCALE_BYTES = 8


def _constant_swa_cache(value: float) -> torch.Tensor:
    cache = torch.zeros(
        (1, PAGE_SIZE * (TOKEN_BYTES + SCALE_BYTES)),
        dtype=torch.uint8,
        device="cuda",
    )
    rows = cache[:, : PAGE_SIZE * TOKEN_BYTES].view(1, PAGE_SIZE, TOKEN_BYTES)
    rows[:, :, :448].copy_(
        torch.full((1, PAGE_SIZE, 448), value, device="cuda")
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    rows[:, :, 448:].view(torch.bfloat16).fill_(value)
    cache[:, PAGE_SIZE * TOKEN_BYTES :].fill_(127)
    return cache


@pytest.mark.parametrize("metadata_dtype", [torch.int32, torch.int64])
def test_selected_attention_masks_positive_out_of_range_rows(
    metadata_dtype: torch.dtype,
) -> None:
    q = torch.zeros((1, 1, HEAD_DIM), device="cuda", dtype=torch.bfloat16)
    kv = torch.stack(
        (
            torch.ones((HEAD_DIM,), device="cuda", dtype=torch.bfloat16),
            torch.full((HEAD_DIM,), 3.0, device="cuda", dtype=torch.bfloat16),
        )
    )
    indices = torch.tensor([[0, 2, 1]], device="cuda", dtype=metadata_dtype)
    lens = torch.tensor([3], device="cuda", dtype=metadata_dtype)
    sink = torch.zeros((1,), device="cuda", dtype=torch.float32)

    actual = deepseek_v4_selected_attention(
        q,
        kv,
        indices,
        lens,
        sink,
        1.0,
        override="triton_deepseek_v4_selected_attention",
    )

    torch.testing.assert_close(
        actual,
        torch.full_like(actual, 4.0 / 3.0),
        rtol=2e-3,
        atol=2e-3,
    )


@pytest.mark.parametrize("metadata_dtype", [torch.int32, torch.int64])
def test_paged_selected_attention_masks_positive_out_of_range_slots(
    metadata_dtype: torch.dtype,
) -> None:
    q = torch.zeros((1, 1, HEAD_DIM), device="cuda", dtype=torch.bfloat16)
    cache = _constant_swa_cache(1.0)
    slots = torch.tensor([[0, PAGE_SIZE]], device="cuda", dtype=metadata_dtype)
    lens = torch.tensor([2], device="cuda", dtype=metadata_dtype)
    sink = torch.zeros((1,), device="cuda", dtype=torch.float32)

    actual = deepseek_v4_paged_selected_attention(
        q,
        cache,
        slots,
        lens,
        PAGE_SIZE,
        sink,
        1.0,
        override="triton_deepseek_v4_paged_selected_attention",
    )

    torch.testing.assert_close(
        actual,
        torch.full_like(actual, 0.5),
        rtol=2e-3,
        atol=2e-3,
    )


def test_paged_selected_attention_masks_extreme_negative_int64_slot() -> None:
    q = torch.zeros((1, 1, HEAD_DIM), device="cuda", dtype=torch.bfloat16)
    cache = _constant_swa_cache(1.0)
    slots = torch.tensor([[0, -(1 << 32)]], device="cuda", dtype=torch.int64)
    lens = torch.tensor([2], device="cuda", dtype=torch.int64)
    sink = torch.zeros((1,), device="cuda", dtype=torch.float32)

    actual = deepseek_v4_paged_selected_attention(
        q,
        cache,
        slots,
        lens,
        PAGE_SIZE,
        sink,
        1.0,
        override="triton_deepseek_v4_paged_selected_attention",
    )

    torch.testing.assert_close(
        actual,
        torch.full_like(actual, 0.5),
        rtol=2e-3,
        atol=2e-3,
    )


def test_swa_cache_insert_ignores_positive_out_of_range_slot() -> None:
    q = torch.zeros((1, 1, HEAD_DIM), device="cuda", dtype=torch.bfloat16)
    kv = torch.ones((1, HEAD_DIM), device="cuda", dtype=torch.bfloat16)
    cache = torch.full(
        (1, PAGE_SIZE * (TOKEN_BYTES + SCALE_BYTES)),
        0xA5,
        device="cuda",
        dtype=torch.uint8,
    )
    original = cache.clone()
    cos_sin = torch.zeros((1, 64), device="cuda", dtype=torch.float32)
    cos_sin[:, :32] = 1.0

    deepseek_v4_swa_cache_insert(
        q,
        kv,
        cache,
        torch.tensor([PAGE_SIZE], device="cuda", dtype=torch.int64),
        torch.tensor([0], device="cuda", dtype=torch.int64),
        cos_sin,
        1e-6,
        PAGE_SIZE,
        override="triton_deepseek_v4_swa_cache_insert",
    )

    assert torch.equal(cache, original)


def test_swa_cache_insert_validates_strided_metadata_and_cache() -> None:
    q = torch.zeros((2, 1, HEAD_DIM), device="cuda", dtype=torch.bfloat16)
    kv = torch.zeros((2, HEAD_DIM), device="cuda", dtype=torch.bfloat16)
    cache = torch.empty(
        (1, PAGE_SIZE * (TOKEN_BYTES + SCALE_BYTES)),
        device="cuda",
        dtype=torch.uint8,
    )
    slots = torch.zeros((2,), device="cuda", dtype=torch.int64)
    positions = torch.zeros((2,), device="cuda", dtype=torch.int64)
    cos_sin = torch.zeros((2, 64), device="cuda", dtype=torch.float32)

    with pytest.raises(
        ValueError, match="positions and slot_mapping must be contiguous"
    ):
        deepseek_v4_swa_cache_insert(
            q,
            kv,
            cache,
            slots,
            torch.zeros((2, 2), device="cuda", dtype=torch.int64)[:, 0],
            cos_sin,
            1e-6,
            PAGE_SIZE,
        )

    strided_cache = torch.empty(
        (1, 2 * cache.shape[1]), device="cuda", dtype=torch.uint8
    )[:, ::2]
    with pytest.raises(ValueError, match="page-planar cache"):
        deepseek_v4_swa_cache_insert(
            q,
            kv,
            strided_cache,
            slots,
            positions,
            cos_sin,
            1e-6,
            PAGE_SIZE,
        )


@pytest.mark.parametrize("position", [-1, 2])
def test_swa_cache_insert_validates_position_bounds(position: int) -> None:
    q = torch.zeros((1, 1, HEAD_DIM), dtype=torch.bfloat16)
    kv = torch.zeros((1, HEAD_DIM), dtype=torch.bfloat16)
    cache = torch.empty((1, PAGE_SIZE * (TOKEN_BYTES + SCALE_BYTES)), dtype=torch.uint8)

    with pytest.raises(ValueError, match="positions entries must index cos_sin_cache"):
        deepseek_v4_swa_cache_insert(
            q,
            kv,
            cache,
            torch.zeros((1,), dtype=torch.int64),
            torch.tensor([position], dtype=torch.int64),
            torch.zeros((2, 64), dtype=torch.float32),
            1e-6,
            PAGE_SIZE,
        )


@pytest.mark.parametrize(
    ("state_slot", "kv_slot"),
    [(PAGE_SIZE, 0), (0, PAGE_SIZE)],
)
def test_csa_cache_insert_ignores_positive_out_of_range_slots(
    state_slot: int,
    kv_slot: int,
) -> None:
    state_cache = torch.zeros((1, PAGE_SIZE, 256), device="cuda", dtype=torch.float32)
    cache = torch.full(
        (1, PAGE_SIZE * (128 + 4)),
        0xA5,
        device="cuda",
        dtype=torch.uint8,
    )
    original = cache.clone()

    deepseek_v4_csa_indexer_fp8_cache_insert(
        state_cache=state_cache,
        token_to_req_indices=torch.tensor([0], device="cuda", dtype=torch.int32),
        positions=torch.tensor([3], device="cuda", dtype=torch.int32),
        compressor_slot_mapping=torch.tensor(
            [state_slot], device="cuda", dtype=torch.int32
        ),
        block_table=torch.tensor([[0]], device="cuda", dtype=torch.int32),
        compressor_block_size=PAGE_SIZE,
        rms_norm_weight=torch.ones((128,), device="cuda", dtype=torch.float32),
        rms_norm_eps=1e-6,
        cos_sin_cache=torch.zeros((4, 64), device="cuda", dtype=torch.float32),
        kv_cache_2d=cache,
        kv_slot_mapping=torch.tensor([kv_slot], device="cuda", dtype=torch.int32),
        kv_cache_block_size=PAGE_SIZE,
        override="triton_deepseek_v4_csa_indexer_fp8_cache_insert",
    )

    assert torch.equal(cache, original)
