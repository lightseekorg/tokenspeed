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

"""The MXFP4 KV write: the GQA prologue's fused quantize-on-write and the
standalone store, bit-exact against the torch codec applied to the rows the
BF16 prologue writes."""

from __future__ import annotations

import pytest
import torch
from attention_prologue_helpers import BF16, cos_sin_cache, head_norm, qkv, split
from mxfp4_kv_reference import mxfp4_quantize
from tokenspeed_kernel.ops.attention.mxfp4_kv import (
    mxfp4_cache_planes,
    mxfp4_row_bytes,
    mxfp4_store_kv,
)
from tokenspeed_kernel.ops.attention.prologue import (
    HeadKVCache,
    HeadNorm,
    KVCacheFormat,
    MXFP4Scales,
    RopeStyle,
    Rotary,
    gqa_prologue,
)
from tokenspeed_kernel.selection import NoKernelFoundError

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

HQ, HKV, PAGE, PAGES = 16, 2, 64, 12


def _field(head_dim: int, fill: int = 0x5A) -> torch.Tensor:
    """A poisoned cache field of PAGES kernel pages."""
    return torch.full(
        (PAGES * PAGE, HKV, mxfp4_row_bytes(head_dim)),
        fill,
        dtype=torch.uint8,
        device="cuda",
    )


def _target(field: torch.Tensor, head_dim: int, slots: torch.Tensor):
    k_planes = mxfp4_cache_planes(field[0], PAGE, head_dim)
    v_planes = mxfp4_cache_planes(field[1], PAGE, head_dim)
    cache = HeadKVCache(
        k_cache=k_planes[0],
        v_cache=v_planes[0],
        scales=MXFP4Scales(k=k_planes[1], v=v_planes[1]),
        slots=slots,
    )
    return cache, k_planes, v_planes


def _expected(field: torch.Tensor, rows: torch.Tensor, slots: torch.Tensor, dim: int):
    """``field`` with the reference codec's bytes of ``rows`` at ``slots``."""
    want = field.clone()
    data, scales = mxfp4_cache_planes(want, PAGE, dim)
    packed, exponent = mxfp4_quantize(rows)
    page, row = slots // PAGE, slots % PAGE
    data[page, row] = packed
    scales[page, row] = exponent
    return want


def _rotary(style: str | None, dim: int, tokens: int) -> Rotary | None:
    if style is None:
        return None
    positions = torch.arange(5, 5 + tokens, device="cuda") * 37 % 4000
    return Rotary(cos_sin_cache(dim), positions, RopeStyle(style), None)


@pytest.mark.parametrize("tokens,written", [(1, 1), (7, 7), (64, 64), (300, 290)])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("rope", ["neox", "gptj", None])
@pytest.mark.parametrize("norm", [None, "norm", "round_before_rope"])
@pytest.mark.parametrize("return_kv", [False, True])
def test_fused_write_is_the_bf16_rows_quantized(
    tokens, written, head_dim, rope, norm, return_kv
):
    inputs = qkv(tokens, HQ, HKV, head_dim, seed=tokens + head_dim)
    inputs[:, HQ * head_dim : (HQ + 1) * head_dim] *= 40  # an outlier key head
    g = torch.Generator(device="cuda").manual_seed(tokens)
    slots = torch.randperm(PAGES * PAGE, device="cuda", generator=g)[:written]
    head = None
    if norm is not None:
        head = head_norm(head_dim, 0.0, seed=3)
        head = HeadNorm(
            head.q_weight,
            head.k_weight,
            0.0,
            1e-6,
            round_before_rope=norm == "round_before_rope",
        )
    rotary = _rotary(rope, head_dim, tokens)

    # The BF16 prologue's rows.
    q, k, v = split(inputs.clone(), HQ, HKV, head_dim)
    bf16_k = torch.zeros(PAGES * PAGE, HKV, head_dim, dtype=BF16, device="cuda")
    bf16_v = torch.zeros_like(bf16_k)
    native = gqa_prologue(
        q,
        k,
        v,
        norm=head,
        rotary=rotary,
        cache=HeadKVCache(bf16_k, bf16_v, None, slots),
        query_dtype=BF16,
        return_kv=return_kv,
        solution="triton",
        override=None,
    )

    q, k, v = split(inputs.clone(), HQ, HKV, head_dim)
    field = torch.stack((_field(head_dim), _field(head_dim, 0xA5)))
    cache, _, _ = _target(field, head_dim, slots)
    assert cache.format is KVCacheFormat.MXFP4
    fused = gqa_prologue(
        q,
        k,
        v,
        norm=head,
        rotary=rotary,
        cache=cache,
        query_dtype=BF16,
        return_kv=return_kv,
        solution=None,
        override="triton_mxfp4_gqa_prologue",
    )
    assert torch.equal(fused.q.flatten(1), native.q.flatten(1))
    if return_kv:
        assert torch.equal(fused.k.flatten(1), native.k.flatten(1))
        assert torch.equal(fused.v.flatten(1), native.v.flatten(1))
    for i, (rows, fill) in enumerate(((bf16_k, 0x5A), (bf16_v, 0xA5))):
        want = _expected(_field(head_dim, fill), rows[slots], slots, head_dim)
        assert torch.equal(field[i], want), "KV"[i]


@pytest.mark.parametrize("head_dim", [64, 128])
def test_standalone_store_matches_the_codec(head_dim):
    tokens = 333
    g = torch.Generator(device="cuda").manual_seed(head_dim)
    k = torch.randn(tokens, HKV, head_dim, dtype=BF16, device="cuda", generator=g)
    v = torch.randn(tokens, HKV, head_dim, dtype=BF16, device="cuda", generator=g)
    k[:, :, 7] *= 1e3
    v[3] = 0.0
    slots = torch.randperm(PAGES * PAGE, device="cuda", generator=g)[:tokens]
    k_field, v_field = _field(head_dim), _field(head_dim)
    mxfp4_store_kv(
        k,
        v,
        slots,
        mxfp4_cache_planes(k_field, PAGE, head_dim),
        mxfp4_cache_planes(v_field, PAGE, head_dim),
    )
    assert torch.equal(k_field, _expected(_field(head_dim), k, slots, head_dim))
    assert torch.equal(v_field, _expected(_field(head_dim), v, slots, head_dim))


def test_planes_view_the_field_pages():
    field = _field(64)
    data, scales = mxfp4_cache_planes(field, PAGE, 64)
    assert data.shape == (PAGES, PAGE, HKV, 32)
    assert scales.shape == (PAGES, PAGE, HKV, 2)
    page_bytes = PAGE * HKV * 34
    assert data.stride(0) == scales.stride(0) == page_bytes
    assert scales.data_ptr() - data.data_ptr() == PAGE * HKV * 32
    with pytest.raises(ValueError):
        mxfp4_cache_planes(field, PAGE, 128)


def test_mxfp4_write_refuses_a_partial_rotary():
    tokens, head_dim = 4, 64
    q, k, v = split(qkv(tokens, HQ, HKV, head_dim, seed=0), HQ, HKV, head_dim)
    cache, _, _ = _target(
        torch.stack((_field(head_dim), _field(head_dim))),
        head_dim,
        torch.arange(tokens, device="cuda"),
    )
    with pytest.raises(NoKernelFoundError):
        gqa_prologue(
            q,
            k,
            v,
            norm=None,
            rotary=_rotary("neox", 32, tokens),
            cache=cache,
            query_dtype=BF16,
            return_kv=False,
            solution=None,
            override=None,
        )


def test_mxfp4_write_checks_the_scale_planes():
    tokens, head_dim = 4, 64
    q, k, v = split(qkv(tokens, HQ, HKV, head_dim, seed=0), HQ, HKV, head_dim)
    cache, _, _ = _target(
        torch.stack((_field(head_dim), _field(head_dim))),
        head_dim,
        torch.arange(tokens, device="cuda"),
    )
    bad = HeadKVCache(
        cache.k_cache,
        cache.v_cache,
        MXFP4Scales(cache.scales.k.contiguous(), cache.scales.v),
        cache.slots,
    )
    with pytest.raises(ValueError, match="MXFP4 scale planes"):
        gqa_prologue(
            q,
            k,
            v,
            norm=None,
            rotary=None,
            cache=bad,
            query_dtype=BF16,
            return_kv=False,
            solution=None,
            override=None,
        )
