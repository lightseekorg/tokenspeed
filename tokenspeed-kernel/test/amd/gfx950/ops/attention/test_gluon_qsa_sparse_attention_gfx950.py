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

"""GFX950 Gluon QSA sparse attention checks."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.numerics.torch.attention.qsa import (
    torch_qsa_selected_attention,
)
from utils import assert_no_triton_compile, is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)

import tokenspeed_kernel.ops.attention.qsa  # noqa: E402,F401
from tokenspeed_kernel_amd.ops.gfx950.attention.qsa.sparse_attention import (  # noqa: E402
    gluon_qsa_sparse_attention_gfx950,
    gluon_qsa_sparse_attention_merge_gfx950,
    launch_gluon_qsa_sparse_attention_gfx950,
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
    # Invalid runs in the middle (as block top-k leaves them), scattered
    # invalid entries, a valid tail, a short row and a row without valid
    # slots.
    selected[:, 900:1700] = -1
    selected[:, 7] = 0
    selected[:, 31] = -5
    selected[0, 20:] = -1
    selected[-1] = -1
    return q, k_cache, v_cache, selected.int()


@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, torch.float8_e4m3fn])
# Few rows split each row's slots over workgroups; many rows take one each.
@pytest.mark.parametrize("rows", [3, 1500])
# 24 query heads on one KV head span two 16-head blocks.
@pytest.mark.parametrize(("num_heads", "num_kv_heads"), [(12, 1), (24, 2), (24, 1)])
def test_gluon_qsa_sparse_attention_matches_reference(
    cache_dtype, rows, num_heads, num_kv_heads
) -> None:
    q, k_cache, v_cache, selected = _inputs(rows, num_heads, num_kv_heads, cache_dtype)
    scale = 256**-0.5
    output = launch_gluon_qsa_sparse_attention_gfx950(
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
    check = [0, 1, rows // 2, rows - 2, rows - 1]
    expected = _reference(q[check], k_cache, v_cache, selected[check], scale)
    torch.testing.assert_close(output[check], expected, atol=4e-3, rtol=1e-2)
    assert torch.all(output[-1] == 0)


@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_gluon_qsa_sparse_attention_reads_caches_beyond_2gib(cache_dtype) -> None:
    # Serving caches exceed the 32-bit byte reach of buffer loads; select
    # slots at the end of > 2 GiB K and V caches.
    rows, small = 5, 8192
    q, k_small, v_small, selected = _inputs(rows, 12, 1, cache_dtype)
    slots = 2**31 // (256 * k_small.element_size()) + 2 * small
    k_cache = torch.empty((slots, 1, 256), dtype=cache_dtype, device="cuda")
    v_cache = torch.empty_like(k_cache)
    k_cache[:2] = k_small[:2]
    v_cache[:2] = v_small[:2]
    k_cache[-small:] = torch.randn((small, 1, 256), device="cuda").to(cache_dtype)
    v_cache[-small:] = torch.randn((small, 1, 256), device="cuda").to(cache_dtype)
    selected = torch.where(selected > 0, selected + (slots - _SLOTS), selected)
    scale = 256**-0.5
    output = launch_gluon_qsa_sparse_attention_gfx950(
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
    expected = _reference(q, k_cache, v_cache, selected, scale)
    torch.testing.assert_close(output, expected, atol=4e-3, rtol=1e-2)


def test_gluon_qsa_sparse_attention_does_not_recompile_per_batch() -> None:
    def run(rows):
        q, k_cache, v_cache, selected = _inputs(rows, 12, 1, torch.bfloat16)
        launch_gluon_qsa_sparse_attention_gfx950(
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

    # Warm the split (small batches) and unsplit (large batches) variants.
    for rows in (1, 16, 2048):
        run(rows)
    with assert_no_triton_compile(
        gluon_qsa_sparse_attention_gfx950,
        gluon_qsa_sparse_attention_merge_gfx950,
    ):
        for rows in (2, 5, 17, 33, 64, 100, 3000):
            run(rows)
