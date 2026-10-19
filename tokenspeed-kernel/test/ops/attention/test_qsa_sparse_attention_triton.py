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
from tokenspeed_kernel.numerics.torch.attention.qsa import (
    torch_qsa_selected_attention,
)
from tokenspeed_kernel.ops.attention.qsa.triton import (
    _triton_qsa_sparse_attention_kernel,
    _triton_qsa_sparse_attention_merge_kernel,
    triton_qsa_sparse_attention,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="QSA sparse attention requires CUDA or ROCm"
)

_SLOTS = 4096
_WIDTH = 2051


def _reference(q, k_cache, v_cache, selected_slots, scale):
    return torch_qsa_selected_attention(
        q, k_cache, v_cache, selected_slots, scale, 1.0, 1.0
    )


def _inputs(rows, num_heads, num_kv_heads, cache_dtype, seed=0):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(
        (rows, num_heads, 256), generator=generator, device="cuda"
    ).bfloat16()
    k_cache = torch.randn(
        (_SLOTS, num_kv_heads, 256), generator=generator, device="cuda"
    ).to(cache_dtype)
    v_cache = torch.randn(
        (_SLOTS, num_kv_heads, 256), generator=generator, device="cuda"
    ).to(cache_dtype)
    # Slot 0 holds NaN padding and must never be read as a valid slot.
    k_cache[0] = float("nan")
    v_cache[0] = float("nan")
    selected = torch.randint(
        2, _SLOTS, (rows, _WIDTH), generator=generator, device="cuda"
    )
    # Mask a tail of every row and leave the last row without valid slots.
    selected[:, _WIDTH - 300 :] = -1
    selected[:, 7] = 0
    selected[-1] = -1
    return q, k_cache, v_cache, selected.int()


@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, torch.float8_e4m3fn])
# Few rows split each row across programs; many rows take the direct path.
@pytest.mark.parametrize("rows", [3, 600])
@pytest.mark.parametrize(("num_heads", "num_kv_heads"), [(12, 1), (24, 2)])
def test_triton_qsa_sparse_attention_matches_reference(
    cache_dtype, rows, num_heads, num_kv_heads
) -> None:
    q, k_cache, v_cache, selected = _inputs(rows, num_heads, num_kv_heads, cache_dtype)
    scale = 256**-0.5
    output = triton_qsa_sparse_attention(
        q,
        k_cache,
        v_cache,
        selected,
        scale=scale,
        max_seqlen_q=1,
        metadata_capacity_rows=None,
        k_scale=1.0 if cache_dtype == torch.float8_e4m3fn else None,
        v_scale=1.0 if cache_dtype == torch.float8_e4m3fn else None,
    )
    check = min(rows, 16)
    expected = _reference(q[-check:], k_cache, v_cache, selected[-check:], scale)
    torch.testing.assert_close(output[-check:], expected, atol=1e-2, rtol=1e-2)
    assert torch.all(output[-1] == 0)


def test_triton_qsa_sparse_attention_does_not_recompile_per_batch() -> None:
    from utils import assert_no_triton_compile

    def run(rows):
        q, k_cache, v_cache, selected = _inputs(rows, 12, 1, torch.bfloat16)
        triton_qsa_sparse_attention(
            q,
            k_cache,
            v_cache,
            selected,
            scale=0.0625,
            max_seqlen_q=None,
            metadata_capacity_rows=None,
            k_scale=None,
            v_scale=None,
        )

    for rows in (1, 16, 1024):
        run(rows)
    with assert_no_triton_compile(
        _triton_qsa_sparse_attention_kernel,
        _triton_qsa_sparse_attention_merge_kernel,
    ):
        for rows in (2, 5, 17, 33, 1000):
            run(rows)
