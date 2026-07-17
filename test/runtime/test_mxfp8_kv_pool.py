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
size accounting, and an end-to-end quantize_mxfp8 -> set_kv_buffer ->
manual dequant roundtrip against the original bf16 K/V.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

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
# Hybrid-slab mode (flat ext): fp8 data slabs + parallel SF slabs
# -----------------------------------------------------------------------------

SLAB_LAYER_TYPES = (
    "full_attention",
    "sliding_attention_0",
    "full_attention",
    "sliding_attention_0",
)
# Byte-uniform hetero slots: full layers serve half the heads at twice the
# tokens per page (slot bytes equal).
SLAB_KV_HEADS = (2, 4, 2, 4)


def _make_slab_pool(size: int = 512):
    from unittest import mock

    from tokenspeed.runtime.configs import paged_cache_spec
    from tokenspeed.runtime.layers.attention.kv_cache import mha as mha_mod
    from tokenspeed.runtime.layers.attention.kv_cache.mha import (
        MHATokenToKVPoolMXFP8,
    )

    with mock.patch.object(
        paged_cache_spec, "scheduler_ext_flat_kvcache", return_value=True
    ), mock.patch.object(
        mha_mod.paged_cache_spec, "scheduler_ext_flat_kvcache", return_value=True
    ):
        return MHATokenToKVPoolMXFP8(
            size=size,
            dtype=torch.float8_e4m3fn,
            head_num=HEADS,
            head_dim=HEAD_DIM,
            layer_num=len(SLAB_LAYER_TYPES),
            device="cuda",
            enable_memory_saver=False,
            max_batch_size=8,
            max_context_len=size,
            page_size=128,
            rank=0,
            layer_types=SLAB_LAYER_TYPES,
            sliding_window_tokens=512,
            max_scheduled_tokens=size,
            slot_tokens=256,
            group_page_sizes={"full_attention": 256},
            layer_kv_head_counts=SLAB_KV_HEADS,
        )


def test_slab_geometry_and_aliasing():
    pool = _make_slab_pool()
    num_ids = (pool.size + pool._slot_tokens) // pool.page_size

    # Paired layers alias data AND scale slabs.
    assert pool.k_buffer[0] is pool.k_buffer[1]
    assert pool.k_scale_buffer[0] is pool.k_scale_buffer[1]
    assert pool.k_buffer[0] is not pool.k_buffer[2]
    assert pool.k_scale_buffer[0] is not pool.k_scale_buffer[2]
    assert pool.k_buffer[0].dtype == torch.float8_e4m3fn

    # One byte-uniform SF slot per id: slot_bytes / 32 e8m0 each.
    slot_sf = 128 * HEADS * HEAD_DIM // 32
    assert pool.k_scale_buffer[0].shape == (num_ids, slot_sf)

    # Layer views factorize the same bytes: full (h/2, k=2), swa (h, k=1).
    k_full, _ = pool.get_kv_scale_buffer(0)
    k_swa, _ = pool.get_kv_scale_buffer(1)
    assert k_full.shape == (num_ids, 2, 2, 32, SF_DIM, SF_DIM)
    assert k_swa.shape == (num_ids, 4, 1, 32, SF_DIM, SF_DIM)


@pytest.mark.parametrize("layer_id, heads_l", [(0, 2), (1, 4)])
def test_slab_store_matches_standalone_scatter(layer_id: int, heads_l: int):
    """set_kv_buffer on a slab layer must land data in the layer's hetero
    row view and scales exactly where a standalone store_sf_interleaved at
    the layer's page size puts them."""
    from tokenspeed_kernel.ops.kvcache.triton import store_sf_interleaved

    torch.manual_seed(2)
    pool = _make_slab_pool()
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


def test_slab_conv_views_are_bf16_over_fp8_slots():
    """bf16 conv columns over fp8 slots: half the token capacity, zero-copy,
    and writes through the view land in the slab's leading slot bytes."""
    pool = _make_slab_pool()
    swa_layer = 1

    # 64-token kvconv block fills the 64 KB fp8 slot byte-exactly at the
    # swa width; 128 no longer fits.
    ch = HEADS * HEAD_DIM
    k_cols, v_cols = pool.kvconv_slot_views_for_layer(swa_layer, 64)
    assert k_cols.dtype == torch.bfloat16 and k_cols.shape[1:] == (64, ch)
    with pytest.raises(AssertionError):
        pool.kvconv_slot_views_for_layer(swa_layer, 128)

    # Zero-copy: writing via the view mutates the slab's slot bytes.
    k_cols[3].fill_(1.0)
    slot_bytes = 128 * HEADS * HEAD_DIM  # fp8: 1 byte/elem
    slab_bytes = pool.k_buffer[swa_layer].view(torch.uint8).reshape(-1, slot_bytes)
    assert slab_bytes[3, : 64 * ch * 2].view(torch.bfloat16).eq(1.0).all()

    # Hetero full layer: half-width columns, leading half of the slot.
    k_full, _ = pool.kvconv_slot_views_for_layer(0, 64)
    assert k_full.shape[1:] == (64, ch // 2)


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
