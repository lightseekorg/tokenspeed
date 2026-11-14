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
size accounting, and an end-to-end set_kv_buffer -> manual dequant roundtrip
against the original bf16 K/V, with the stored bytes equal to quantize_mxfp8's
on finite inputs. Also the BF16 dequant view the Inkling prefix extend reads,
and that extend split into request groups.
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
    from test.runtime.cache_pool_test_utils import make_arena, make_mha_memory_plan

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
    from tokenspeed_kernel.ops.quantization import quantize_mxfp8

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
    v_q, _ = _quantize(kv * 0.5)
    loc = torch.randperm(pool.arena.size, device="cuda")[:T].to(torch.int64)

    pool.set_kv_buffer(layer, loc, kv, kv * 0.5)

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
    from test.runtime.cache_pool_test_utils import make_arena, plan_fields

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
    from test.runtime.cache_pool_test_utils import make_arena, make_mha_memory_plan

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
    loc = torch.randperm(num_ids * page_tokens, device="cuda")[:T].to(torch.int64)

    pool.set_kv_buffer(layer, loc, kv, kv * 0.5)

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


def _make_config(kernel_page_size: int | None):
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
        kernel_page_size=kernel_page_size,
        context_len=4096,
        max_bs=8,
        max_resident_requests=8,
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
        server_args=SimpleNamespace(max_total_tokens=512, kv_offload_config=None),
        model_config=SimpleNamespace(num_attention_layers=LAYERS),
        attn_config=config,
        draft_model_config=None,
        draft_attn_config=None,
        cache_budget_bytes=config.cache_cell_size() * LAYERS * 512,
        probe_batch_rows=None,
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


@pytest.mark.parametrize("layer_id", range(len(SHARED_KV_HEADS)))
def test_the_prologue_writes_through_the_pool_write_target(layer_id: int):
    """The prologue's rows land where the pool's own quantize-and-store puts
    them, for every per-layer head count of a shared field."""
    from tokenspeed_kernel.ops.attention.prologue import gqa_prologue
    from tokenspeed_kernel.ops.kvcache.triton import quantize_store_kv_mxfp8

    heads, tokens = SHARED_KV_HEADS[layer_id], 37
    g = torch.Generator(device="cuda").manual_seed(layer_id)
    q, k, v = (
        torch.randn(tokens, n * heads * HEAD_DIM, generator=g, device="cuda").bfloat16()
        for n in (2, 1, 1)
    )
    prologue_pool, reference_pool = _make_shared_pool(), _make_shared_pool()
    loc = torch.randperm(reference_pool.arena.size, generator=g, device="cuda")[:tokens]
    quantize_store_kv_mxfp8(
        k.view(tokens, heads, HEAD_DIM),
        v.view(tokens, heads, HEAD_DIM),
        reference_pool._layer_row_view(reference_pool.k_buffer[layer_id], layer_id),
        reference_pool._layer_row_view(reference_pool.v_buffer[layer_id], layer_id),
        reference_pool.k_scale_buffer[layer_id],
        reference_pool.v_scale_buffer[layer_id],
        loc,
        page_tokens=reference_pool._layer_page_tokens(layer_id),
    )
    gqa_prologue(
        q,
        k,
        v,
        norm=None,
        rotary=None,
        cache=prologue_pool.kv_write_target(layer_id, loc, None),
        query_dtype=q.dtype,
        return_kv=False,
        solution=None,
        override=None,
    )
    for name in ("k_buffer", "v_buffer", "k_scale_buffer", "v_scale_buffer"):
        for lid in range(len(SHARED_KV_HEADS)):
            got = getattr(prologue_pool, name)[lid].view(torch.uint8)
            assert torch.equal(
                got, getattr(reference_pool, name)[lid].view(torch.uint8)
            )


def test_config_selects_mxfp8_pool_and_sizes():
    from tokenspeed.runtime.layers.attention.kv_cache.mha import (
        MHATokenToKVPoolMXFP8,
    )

    config = _make_config(kernel_page_size=None)
    # fp8 data + 1 scale byte per 32: 33/32 of the fp8 cell.
    assert (
        config.cache_cell_size() == HEADS * HEAD_DIM * 2 + (HEADS * HEAD_DIM * 2) // 32
    )
    pool = _create_config_pool(config)
    assert isinstance(pool, MHATokenToKVPoolMXFP8)
    # The recipe widens its 64-token default grain to the scale tile.
    assert pool.arena.prefix_granularity == 128


def test_config_rejects_a_page_wider_than_the_scale_tile():
    config = _make_config(kernel_page_size=256)
    with pytest.raises(AssertionError, match="128-row pages"):
        _create_config_pool(config)


def _scratch(rows: int) -> tuple[torch.Tensor, torch.Tensor]:
    """BF16 key and value dequant scratch rows for HEADS heads."""
    return tuple(
        torch.zeros(rows, HEADS * HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        for _ in range(2)
    )


@pytest.mark.parametrize("window_left", [-1, 200])
def test_dequant_view_serves_the_stored_values(window_left):
    """BF16 pages of the view equal the dequantized cache rows; a sliding
    window gathers only the pages it reads; rows past the length are zero."""
    torch.manual_seed(3)
    pool = _make_pool(128, size=4096)
    layer = SimpleNamespace(layer_id=1)
    seq_lens = [300, 1000]
    width = 8
    perm = torch.randperm(pool.arena.size // 128, device="cuda")[: 3 + width]
    page_table = torch.zeros(len(seq_lens), width, dtype=torch.int32, device="cuda")
    page_table[0, :3] = perm[:3].to(torch.int32)
    page_table[1, :width] = perm[3:].to(torch.int32)
    k_ref, v_ref = [], []
    for b, n in enumerate(seq_lens):
        k = torch.randn(n, HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(n, HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16) * 2
        tok = torch.arange(n, device="cuda")
        loc = page_table[b, tok // 128].long() * 128 + tok % 128
        pool.set_kv_buffer(layer, loc, k, v)
        k_ref.append(_dequant(*_quantize(k)).to(torch.bfloat16))
        v_ref.append(_dequant(*_quantize(v)).to(torch.bfloat16))

    k_view, v_view, view_table = pool.dequant_kv_view(
        1,
        page_table,
        torch.tensor(seq_lens, dtype=torch.int32, device="cuda"),
        1,
        window_left,
        128,
        scratch=_scratch(4096),
    )
    assert k_view.dtype == torch.bfloat16 and k_view.shape[1:] == (128, HEADS, HEAD_DIM)
    for b, n in enumerate(seq_lens):
        pages = -(-n // 128)
        # The window's first query sits at n - 1 and sees keys back to n - 2 - window.
        first = 0 if window_left < 0 else max(0, (n - 2 - window_left) // 128 - 1)
        for i in range(width):
            if not first <= i < pages:
                assert int(view_table[b, i]) == 0
                continue
            rows = range(i * 128, min(n, (i + 1) * 128))
            got_k = k_view[int(view_table[b, i])]
            got_v = v_view[int(view_table[b, i])]
            assert torch.equal(got_k[: len(rows)], k_ref[b][rows.start : rows.stop])
            assert torch.equal(got_v[: len(rows)], v_ref[b][rows.start : rows.stop])
            assert not got_k[len(rows) :].any() and not got_v[len(rows) :].any()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="the relative extend kernel needs SM100",
)
@pytest.mark.parametrize("return_lse", [False, True])
# A sliding window spans the relative-logit extent: window_left + 1 == 128.
@pytest.mark.parametrize("window_left", [-1, 127])
def test_extend_in_request_groups_matches_one_call(window_left, return_lse):
    """An Inkling prefix extend whose pages overflow the dequant scratch runs
    in request groups: their rows equal one call over a scratch that holds
    the batch, and padding rows are zero."""
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.layers.attention.backends.paged.mha import (
        MHAAttnBackend,
    )
    from tokenspeed.runtime.layers.attention.backends.specific.inkling import (
        rel_mha_extend_mxfp8,
    )

    torch.manual_seed(5)
    config = _make_config(kernel_page_size=128)
    spec = config.components[0]
    leaf = MHAAttnBackend(config, spec, kernel_page_size=128)
    pool = _make_pool(128, size=8192)
    layer = SimpleNamespace(
        layer_id=1, sliding_window_size=window_left, scaling=HEAD_DIM**-0.5
    )
    prefix_lens, query_lens = [300, 900, 37], [64, 100, 1]
    seq_lens = [p + n for p, n in zip(prefix_lens, query_lens)]
    width = 8
    page_table = (
        torch.randperm(pool.arena.size // 128, device="cuda")[: 3 * width]
        .to(torch.int32)
        .view(3, width)
    )
    for b, n in enumerate(seq_lens):
        tok = torch.arange(n, device="cuda")
        loc = page_table[b, tok // 128].long() * 128 + tok % 128
        k = torch.randn(n, HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        pool.set_kv_buffer(layer, loc, k, torch.randn_like(k))
    leaf.init_forward_metadata(
        3,
        3,
        torch.tensor(seq_lens, dtype=torch.int32, device="cuda"),
        page_table,
        ForwardMode.EXTEND,
        extend_seq_lens=torch.tensor(query_lens, dtype=torch.int32, device="cuda"),
        extend_seq_lens_cpu=torch.tensor(query_lens),
        extend_prefix_lens=torch.tensor(prefix_lens, dtype=torch.int32, device="cuda"),
        extend_prefix_lens_cpu=torch.tensor(prefix_lens),
        extend_with_prefix=True,
        query_shard=None,
        page_table_cpu=None,
    )
    real = sum(query_lens)
    heads = spec.num_attention_heads
    # Two padding rows, as in a bucket-shaped extend.
    q = torch.randn(real + 2, heads, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    rel_logits = torch.randn(real + 2, heads, 128, device="cuda", dtype=torch.bfloat16)
    tau = torch.rand(real + 2, device="cuda") + 0.5

    def extend(scratch_rows: int):
        result = rel_mha_extend_mxfp8(
            leaf,
            layer,
            pool,
            _scratch(scratch_rows),
            q,
            rel_logits,
            tau,
            output_dtype=torch.float32,
            return_lse=return_lse,
        )
        return result if return_lse else (result,)

    # The longest request spans 8 pages: one request per group, then one call.
    split, whole = extend(8 * 128), extend(3 * 8 * 128)
    for got, want in zip(split, whole):
        assert torch.equal(got[:real], want[:real])
        assert not got[real:].any()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
