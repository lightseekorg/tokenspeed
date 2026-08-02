# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""MXFP8 KV pool: quantize -> store -> verify layout and roundtrip error.

Covers both scale layouts (interleaved for page 128, flat otherwise), the
size accounting, the Flat LCM heterogeneous-group layout, and an end-to-end
quantize_mxfp8 -> set_kv_buffer -> manual dequant roundtrip against the
original bf16 K/V.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, suite="runtime-1gpu")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)

HEADS = 4
HEAD_DIM = 128
SF_DIM = HEAD_DIM // 32
LAYERS = 2


def _make_pool(page_size: int, size: int = 512):
    from tokenspeed.runtime.layers.attention.kv_cache.mha import (
        MHATokenToKVPoolMXFP8,
    )

    return MHATokenToKVPoolMXFP8(
        size=size,
        dtype=torch.bfloat16,
        head_num=HEADS,
        head_dim=HEAD_DIM,
        layer_num=LAYERS,
        device="cuda",
        enable_memory_saver=False,
        max_batch_size=8,
        max_context_len=size,
        page_size=page_size,
        rank=0,
    )


def _quantize(x: torch.Tensor):
    """[T, H, D] bf16 -> (fp8 data, [T, H, sf] e8m0 scales) via the kernel op."""
    from tokenspeed_kernel import quantize_mxfp8

    t, h, d = x.shape
    q, sf = quantize_mxfp8(x.reshape(t * h, d))
    return q.reshape(t, h, d), sf.view(torch.float8_e8m0fnu).reshape(t, h, SF_DIM)


def _dequant(q: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    """Blockwise dequant: e8m0 scale s applies to 32 consecutive elements."""
    t, h, d = q.shape
    scale = sf.to(torch.float32).repeat_interleave(32, dim=-1)
    return q.to(torch.float32) * scale


@pytest.mark.parametrize("page_size", [128, 64])
def test_store_and_roundtrip(page_size: int):
    torch.manual_seed(0)
    pool = _make_pool(page_size)
    layer = SimpleNamespace(layer_id=1)

    T = 96
    kv = torch.randn(T, HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    k_q, k_sf = _quantize(kv)
    v_q, v_sf = _quantize(kv * 0.5)
    loc = torch.randperm(pool.size, device="cuda")[:T].to(torch.int64)

    pool.set_kv_buffer(layer, loc, k_q, v_q, k_scale=k_sf, v_scale=v_sf)

    # Data lands at loc.
    assert torch.equal(pool.k_buffer[1][loc].view(torch.uint8), k_q.view(torch.uint8))
    assert torch.equal(pool.v_buffer[1][loc].view(torch.uint8), v_q.view(torch.uint8))

    # Scales land in the layout's documented position.
    k_sfbuf, v_sfbuf = pool.get_kv_scale_buffer(1)
    if page_size == 128:
        u32 = k_sfbuf.view(torch.uint8).reshape(-1, HEADS, 128, 4).view(torch.int32)
        src = (
            k_sf.view(torch.uint8)
            .reshape(T, HEADS, 4)
            .contiguous()
            .view(torch.int32)
            .reshape(T, HEADS)
        )
        for t in range(0, T, 17):  # sample positions
            slot = int(loc[t])
            page, off = divmod(slot, 128)
            pos = (off % 32) * 4 + (off // 32)
            assert torch.equal(u32[page, :, pos, 0], src[t])
    else:
        assert torch.equal(k_sfbuf[loc].view(torch.uint8), k_sf.view(torch.uint8))
        assert torch.equal(v_sfbuf[loc].view(torch.uint8), v_sf.view(torch.uint8))

    # Roundtrip: dequantized K matches original within fp8 blockscale error.
    k_rt = _dequant(pool.k_buffer[1][loc], k_sf)
    rel = (k_rt - kv.float()).abs().max() / kv.float().abs().max()
    assert rel < 0.13, f"roundtrip rel err {rel:.4f}"  # e4m3 mantissa ~2^-3


def test_requires_prequantized_and_scales():
    pool = _make_pool(128)
    layer = SimpleNamespace(layer_id=0)
    loc = torch.arange(4, device="cuda", dtype=torch.int64)
    bf16 = torch.randn(4, HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(AssertionError):
        pool.set_kv_buffer(layer, loc, bf16, bf16, None, None)
    q, sf = _quantize(bf16)
    with pytest.raises(AssertionError):
        pool.set_kv_buffer(layer, loc, q, q, None, None)


def test_size_accounting_includes_scales():
    pool = _make_pool(128)
    k_size, v_size = pool.get_kv_size_bytes()
    slots = pool.size + pool.page_size
    expect_data = slots * HEADS * HEAD_DIM * LAYERS  # 1 byte/elem
    expect_sf = slots * HEADS * SF_DIM * LAYERS
    assert k_size == expect_data + expect_sf
    assert v_size == expect_data + expect_sf


# -----------------------------------------------------------------------------
# LCM arena mode (flat ext): heterogeneous groups share byte-uniform planes
# -----------------------------------------------------------------------------

LCM_LAYER_TYPES = (
    "full_attention",
    "sliding_attention_0",
    "full_attention",
    "sliding_attention_0",
)
# A byte-uniform LCM parent: full layers serve half the heads, so the parent
# packs twice as many 128-token child pages for that group.
LCM_KV_HEADS = (2, 4, 2, 4)


def _make_lcm_pool(num_lcm_blocks: int = 4):
    from unittest import mock

    from tokenspeed.runtime.configs import paged_cache_spec
    from tokenspeed.runtime.configs.lcm_layouts import draft_history_lcm_fields
    from tokenspeed.runtime.configs.lcm_memory_plan import plan_lcm_fields
    from tokenspeed.runtime.layers.attention.kv_cache.lcm_mha import (
        LcmMHATokenToKVPoolMXFP8,
    )

    fields = draft_history_lcm_fields(
        layer_group_ids=LCM_LAYER_TYPES,
        enabled_layer_ids=range(len(LCM_LAYER_TYPES)),
        logical_block_tokens=128,
        layer_kv_heads=LCM_KV_HEADS,
        head_dim=HEAD_DIM,
        kv_element_size=1,
        kv_scale_block_size=32,
        kv_scale_element_size=1,
    )
    plan = plan_lcm_fields(
        fields,
        logical_block_tokens=128,
        num_lcm_blocks=num_lcm_blocks,
        alignment=256,
        max_padding_fraction=1.0,
    )
    max_packing = max(group.cache_blocks_per_lcm_block for group in plan.groups)
    size = num_lcm_blocks * max_packing * plan.logical_block_tokens
    with mock.patch.object(
        paged_cache_spec, "scheduler_ext_flat_kvcache", return_value=True
    ):
        return LcmMHATokenToKVPoolMXFP8(
            size=size,
            dtype=torch.float8_e4m3fn,
            head_num=HEADS,
            head_dim=HEAD_DIM,
            layer_num=len(LCM_LAYER_TYPES),
            device="cuda",
            enable_memory_saver=False,
            max_batch_size=1,
            max_context_len=size,
            page_size=128,
            rank=0,
            layer_types=LCM_LAYER_TYPES,
            sliding_window_tokens=512,
            max_scheduled_tokens=128,
            layer_kv_head_counts=LCM_KV_HEADS,
            kv_alloc_head_count=HEADS,
            memory_plan=plan,
            layer_group_ids=LCM_LAYER_TYPES,
            state_field_dtypes={},
        )


def test_lcm_geometry_and_aliasing():
    pool = _make_lcm_pool()
    by_group = {group.group_id: group for group in pool._lcm_memory_plan.groups}

    # Both groups use the scheduler's 128-token logical page domain. Their
    # byte ratio is expressed by child-page packing, not a coarser group page.
    assert by_group["full_attention"].cache_blocks_per_lcm_block == 2
    assert by_group["sliding_attention_0"].cache_blocks_per_lcm_block == 1
    assert all(spec.block_size == 128 for spec in pool.paged_cache_group_specs)
    assert pool.paged_cache_group_page_counts == {
        "full_attention": 9,
        "sliding_attention_0": 5,
    }

    assert pool.k_buffer[0].dtype == torch.float8_e4m3fn

    # Every child page represents 128 tokens; the narrow full-attention group
    # gets twice as many child page IDs from each LCM parent.
    k_full, _ = pool.get_kv_scale_buffer(0)
    k_swa, _ = pool.get_kv_scale_buffer(1)
    full_rows = pool.get_key_buffer(0)
    swa_rows = pool.get_key_buffer(1)
    assert k_full.shape == (9, 2, 1, 32, SF_DIM, SF_DIM)
    assert k_swa.shape == (5, 4, 1, 32, SF_DIM, SF_DIM)
    assert full_rows.shape == (9 * 128, 2, HEAD_DIM)
    assert swa_rows.shape == (5 * 128, 4, HEAD_DIM)

    # Paired group fields overlay the same LCM parent at page 1 while the
    # second occurrence of the group is assigned a separate plane.
    assert full_rows[128].data_ptr() == swa_rows[128].data_ptr()
    assert k_full[1].data_ptr() == k_swa[1].data_ptr()
    assert full_rows[128].data_ptr() != pool.get_key_buffer(2)[128].data_ptr()
    assert k_full[1].data_ptr() != pool.get_kv_scale_buffer(2)[0][1].data_ptr()


@pytest.mark.parametrize("layer_id, heads_l", [(0, 2), (1, 4)])
def test_lcm_store_matches_standalone_scatter(layer_id: int, heads_l: int):
    """LCM child-page writes match an independent interleaved SF scatter."""
    from tokenspeed_kernel.ops.kvcache.triton import store_sf_interleaved

    torch.manual_seed(2)
    pool = _make_lcm_pool()
    layer = SimpleNamespace(layer_id=layer_id)
    page_tokens = pool._layer_page_tokens(layer_id)
    num_ids = pool.k_scale_buffer[layer_id].shape[0]

    T = 80
    kv = torch.randn(T, heads_l, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    k_q, k_sf = _quantize(kv)
    v_q, v_sf = _quantize(kv * 0.5)
    loc = torch.randperm(num_ids * page_tokens, device="cuda")[:T].to(torch.int64)

    pool.set_kv_buffer(layer, loc, k_q, v_q, k_scale=k_sf, v_scale=v_sf)

    # Data: readable back through the layer's row view at the same locs.
    rows = pool.get_key_buffer(layer_id)
    assert rows.shape[1] == heads_l
    assert torch.equal(rows[loc].view(torch.uint8), k_q.view(torch.uint8))

    # Scales: byte-equal to an independent scatter at the layer page size.
    ref = torch.zeros(
        num_ids,
        heads_l,
        page_tokens // 128,
        32,
        SF_DIM,
        SF_DIM,
        dtype=torch.float8_e8m0fnu,
        device="cuda",
    )
    store_sf_interleaved(k_sf, ref, loc, page_size=page_tokens)
    k_view, _ = pool.get_kv_scale_buffer(layer_id)
    assert torch.equal(k_view.view(torch.uint8), ref.view(torch.uint8))


def _make_config(page_size: int):
    from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig

    return MHAConfig(
        device="cuda",
        backend_name="mha",
        num_attention_heads=16,
        num_kv_heads=HEADS,
        head_dim=HEAD_DIM,
        attn_tp_size=1,
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.float8_e4m3fn,
        kv_cache_mxfp8=True,
        page_size=page_size,
        context_len=4096,
        max_bs=8,
        max_graph_bs=8,
        kv_cache_quant_method="none",
    )


def test_config_selects_mxfp8_pool_and_sizes():
    from tokenspeed.runtime.layers.attention.kv_cache.mha import (
        MHATokenToKVPoolMXFP8,
    )

    config = _make_config(page_size=128)
    # fp8 data + 1 scale byte per 32: 33/32 of the fp8 cell.
    assert (
        config.cache_cell_size() == HEADS * HEAD_DIM * 2 + (HEADS * HEAD_DIM * 2) // 32
    )
    pool = config.create_pool(
        num_layers=LAYERS,
        max_total_num_tokens=512,
        rank=0,
        enable_memory_saver=False,
    )
    assert isinstance(pool, MHATokenToKVPoolMXFP8)


def test_config_rejects_non_128_page():
    config = _make_config(page_size=64)
    with pytest.raises(AssertionError, match="block-size 128"):
        config.create_pool(
            num_layers=LAYERS,
            max_total_num_tokens=512,
            rank=0,
            enable_memory_saver=False,
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
