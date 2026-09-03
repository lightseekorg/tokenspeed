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

Covers the interleaved scale layout (uniform and per-layer head counts), the
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
    from cache_pool_test_utils import make_arena, make_mha_memory_plan

    from tokenspeed.runtime.layers.attention.kv_cache.mha import (
        MHATokenToKVPoolMXFP8,
    )

    arena = make_arena(
        make_mha_memory_plan(
            size=size,
            prefix_granularity=page_size,
            layer_num=LAYERS,
            kv_heads=HEADS,
            head_dim=HEAD_DIM,
            dtype=torch.bfloat16,
            mxfp8=True,
        )
    )
    return MHATokenToKVPoolMXFP8(
        arena,
        dtype=torch.bfloat16,
        head_num=HEADS,
        head_dim=HEAD_DIM,
        layer_num=LAYERS,
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
    _t, _h, _d = q.shape
    scale = sf.to(torch.float32).repeat_interleave(32, dim=-1)
    return q.to(torch.float32) * scale


def test_store_and_roundtrip():
    """The mxfp8 pool is a P=128 pool; no recipe can plan its scales at any
    other grain (test_config_rejects_non_128_page asserts the rejection)."""
    torch.manual_seed(0)
    pool = _make_pool(128)
    layer = SimpleNamespace(layer_id=1)

    T = 96
    kv = torch.randn(T, HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    k_q, k_sf = _quantize(kv)
    v_q, v_sf = _quantize(kv * 0.5)
    loc = torch.randperm(pool.arena.size, device="cuda")[:T].to(torch.int64)

    pool.set_kv_buffer(layer, loc, k_q, v_q, k_scale=k_sf, v_scale=v_sf)

    # Data lands at loc.
    assert torch.equal(pool.k_buffer[1][loc].view(torch.uint8), k_q.view(torch.uint8))
    assert torch.equal(pool.v_buffer[1][loc].view(torch.uint8), v_q.view(torch.uint8))

    # Scales land in the layout's documented position.
    k_sfbuf, _v_sfbuf = pool.get_kv_scale_buffer(1)
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
    q, _sf = _quantize(bf16)
    with pytest.raises(AssertionError):
        pool.set_kv_buffer(layer, loc, q, q, None, None)


def test_size_accounting_includes_scales():
    pool = _make_pool(128)
    k_size, v_size = pool.get_kv_size_bytes()
    slots = pool.arena.size + pool.arena.prefix_granularity
    expect_data = slots * HEADS * HEAD_DIM * LAYERS  # 1 byte/elem
    expect_sf = slots * HEADS * SF_DIM * LAYERS
    assert k_size == expect_data + expect_sf
    assert v_size == expect_data + expect_sf


def test_rejects_scale_planes_outside_the_interleaved_layout():
    """The pool has one scale layout; a plan declaring any other shape for a
    scale field fails when the planes are bound, not at the first store."""
    from cache_pool_test_utils import make_arena, plan_fields

    from tokenspeed.runtime.layers.attention.kv_cache.mha import (
        MHATokenToKVPoolMXFP8,
    )
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
        CacheFieldSpec,
    )

    kv_shape = (128, HEADS, HEAD_DIM)
    per_token_scales = (128, HEADS, SF_DIM)
    plan = plan_fields(
        {
            "full_attention": (
                CacheFieldSpec("layer.0.k", "unit.0.k", kv_shape, "float8_e4m3fn"),
                CacheFieldSpec("layer.0.v", "unit.0.v", kv_shape, "float8_e4m3fn"),
                CacheFieldSpec(
                    "layer.0.k_scale",
                    "unit.0.k_scale",
                    per_token_scales,
                    "float8_e8m0fnu",
                ),
                CacheFieldSpec(
                    "layer.0.v_scale",
                    "unit.0.v_scale",
                    per_token_scales,
                    "float8_e8m0fnu",
                ),
            )
        },
        prefix_granularity=128,
        num_lcm_blocks=2,
        cache_blocks_per_lcm_block={"full_attention": 1},
        alignment=1,
        max_padding_fraction=1.0,
    )
    with pytest.raises(ValueError, match="interleaved"):
        MHATokenToKVPoolMXFP8(
            make_arena(plan),
            dtype=torch.bfloat16,
            head_num=HEADS,
            head_dim=HEAD_DIM,
            layer_num=1,
            rank=0,
        )


# -----------------------------------------------------------------------------
# Plan-aliased MHA fields: FP8 data and scale fields share physical units.
# -----------------------------------------------------------------------------

SHARED_LAYER_TYPES = (
    "full_attention",
    "sliding_attention_0",
    "full_attention",
    "sliding_attention_0",
)
# Byte-uniform hetero slots: full layers serve half the heads at twice the
# tokens per page (slot bytes equal).
SHARED_KV_HEADS = (2, 4, 2, 4)


def _make_shared_pool(size: int = 512):
    from cache_pool_test_utils import make_arena, make_mha_memory_plan

    from tokenspeed.runtime.layers.attention.kv_cache.mha import (
        MHATokenToKVPoolMXFP8,
    )

    arena = make_arena(
        make_mha_memory_plan(
            size=size,
            prefix_granularity=128,
            layer_num=len(SHARED_LAYER_TYPES),
            kv_heads=HEADS,
            head_dim=HEAD_DIM,
            dtype=torch.float8_e4m3fn,
            layer_types=SHARED_LAYER_TYPES,
            sliding_window_tokens=512,
            mxfp8=True,
        )
    )
    return MHATokenToKVPoolMXFP8(
        arena,
        dtype=torch.float8_e4m3fn,
        head_num=HEADS,
        head_dim=HEAD_DIM,
        layer_num=len(SHARED_LAYER_TYPES),
        rank=0,
        layer_types=SHARED_LAYER_TYPES,
        layer_kv_head_counts=SHARED_KV_HEADS,
    )


def test_shared_field_geometry_and_aliasing():
    pool = _make_shared_pool()
    num_blocks = pool.k_scale_buffer[0].shape[0]

    # Layer-local views alias data and scale fields through the plan.
    assert pool.k_buffer[0] is not pool.k_buffer[1]
    assert pool.k_scale_buffer[0] is not pool.k_scale_buffer[1]
    assert pool.k_buffer[0].data_ptr() == pool.k_buffer[1].data_ptr()
    assert pool.k_scale_buffer[0].data_ptr() == pool.k_scale_buffer[1].data_ptr()
    assert pool.k_buffer[0].data_ptr() != pool.k_buffer[2].data_ptr()
    assert pool.k_scale_buffer[0].data_ptr() != pool.k_scale_buffer[2].data_ptr()
    assert pool.k_buffer[0].dtype == torch.float8_e4m3fn

    # One byte-uniform scale field per block: data bytes / 32 e8m0 each. The
    # plan's own rank is the interleaved layout's; only the bytes are pinned.
    slot_sf = 128 * HEADS * HEAD_DIM // 32
    assert pool.k_scale_buffer[0].shape[0] == num_blocks
    assert pool.k_scale_buffer[0][0].numel() == slot_sf

    # Layer views factorize the same bytes: full (h/2, k=2), swa (h, k=1).
    k_full, _ = pool.get_kv_scale_buffer(0)
    k_swa, _ = pool.get_kv_scale_buffer(1)
    assert k_full.shape == (num_blocks, 2, 2, 32, SF_DIM, SF_DIM)
    assert k_swa.shape == (num_blocks, 4, 1, 32, SF_DIM, SF_DIM)


@pytest.mark.parametrize("layer_id, heads_l", [(0, 2), (1, 4)])
def test_shared_field_store_matches_standalone_scatter(layer_id: int, heads_l: int):
    """A layer view must match an independent scale scatter."""
    from tokenspeed_kernel.ops.kvcache.triton import store_sf_interleaved

    torch.manual_seed(2)
    pool = _make_shared_pool()
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


def _make_config(prefix_granularity: int):
    from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
    from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig

    spec = MHAConfig(
        backend_name="mha",
        num_attention_heads=16,
        num_kv_heads=HEADS,
        head_dim=HEAD_DIM,
        attn_tp_size=1,
    )
    return AttnConfig(
        device="cuda",
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.float8_e4m3fn,
        kv_cache_mxfp8=True,
        prefix_granularity=prefix_granularity,
        kernel_page_size=prefix_granularity,
        context_len=4096,
        max_bs=8,
        kv_cache_quant_method="none",
        components=(spec,),
    )


def _create_config_pool(config):
    from tokenspeed.runtime.layers.attention.kv_cache.factory import (
        create_cache_arena,
        create_cache_pool,
    )
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.setup import (
        prepare_cache_setup,
    )

    setup = prepare_cache_setup(
        family="mha",
        server_args=SimpleNamespace(max_total_tokens=512),
        model_config=SimpleNamespace(num_attention_layers=LAYERS),
        attn_config=config,
        draft_model_config=None,
        draft_attn_config=None,
        cache_budget_bytes=config.cache_cell_size() * LAYERS * 512,
        decode_input_tokens=1,
        overlap_schedule_depth=0,
    )
    arena = create_cache_arena(
        setup.spec, device=config.device, enable_memory_saver=False
    )
    return create_cache_pool(
        setup.spec,
        config,
        arena,
        num_layers=LAYERS,
        rank=0,
    )


def test_config_selects_mxfp8_pool_and_sizes():
    from tokenspeed.runtime.layers.attention.kv_cache.mha import (
        MHATokenToKVPoolMXFP8,
    )

    config = _make_config(prefix_granularity=128)
    # fp8 data + 1 scale byte per 32: 33/32 of the fp8 cell.
    assert (
        config.cache_cell_size() == HEADS * HEAD_DIM * 2 + (HEADS * HEAD_DIM * 2) // 32
    )
    pool = _create_config_pool(config)
    assert isinstance(pool, MHATokenToKVPoolMXFP8)


def test_config_rejects_non_128_page():
    config = _make_config(prefix_granularity=64)
    with pytest.raises(AssertionError, match="prefix-granularity 128"):
        _create_config_pool(config)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
