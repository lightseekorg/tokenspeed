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

"""MiMo-V2: pre-sharded fused QKV loading and the mixed-width KV cache."""

import math
import os
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

from tokenspeed.runtime.configs.model_config import configure_mimo_v2_attention
from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
from tokenspeed.runtime.layers.attention.kv_cache.factory import (
    create_cache_arena,
    create_cache_pool,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.setup import (
    prepare_cache_setup,
)
from tokenspeed.runtime.layers.quantization.fp8 import Fp8Config
from tokenspeed.runtime.models.mimo_v2 import MiMoV2QKVProjection

register_cuda_ci(
    est_time=10,
    suite="runtime-1gpu",
    nightly=False,
    disabled=None,
    disabled_on_runners=None,
    disabled_on_runners_reason=None,
)

# A scaled-down MiMo-V2.6-Flash attention: 4 checkpoint chunks, 2 query heads
# (128 rows) per chunk, 64-wide keys and 32-wide values.
_HIDDEN, _HEADS, _HEAD_DIM, _V_HEAD_DIM, _CHUNKS = 256, 8, 64, 32, 4


def _checkpoint_rows(num_kv_heads: int) -> list[tuple[str, int, int]]:
    """(part, chunk, rows) in checkpoint order: [Q_c | K_c | V_c] per chunk."""
    rows = []
    for chunk in range(_CHUNKS):
        rows.append(("q", chunk, _HEADS // _CHUNKS * _HEAD_DIM))
        rows.append(("k", chunk, num_kv_heads // _CHUNKS * _HEAD_DIM))
        rows.append(("v", chunk, num_kv_heads // _CHUNKS * _V_HEAD_DIM))
    return rows


def _projection(num_kv_heads, tp_rank, tp_size, quant_config):
    return MiMoV2QKVProjection(
        hidden_size=_HIDDEN,
        num_heads=_HEADS,
        num_kv_heads=num_kv_heads,
        head_dim=_HEAD_DIM,
        v_head_dim=_V_HEAD_DIM,
        num_chunks=_CHUNKS,
        tp_rank=tp_rank,
        tp_size=tp_size,
        quant_config=quant_config,
        prefix="model.layers.0.self_attn.qkv_proj",
    )


@pytest.mark.parametrize("num_kv_heads", [4, 8], ids=["full", "sliding"])
@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_qkv_projection_regroups_presharded_chunks(num_kv_heads, tp_size):
    weight = torch.randn(sum(n for _, _, n in _checkpoint_rows(num_kv_heads)), _HIDDEN)
    x = torch.randn(5, _HIDDEN)
    reference = {}
    start = 0
    for part, chunk, rows in _checkpoint_rows(num_kv_heads):
        reference[part, chunk] = x @ weight[start : start + rows].T
        start += rows

    chunks_per_rank = _CHUNKS // tp_size
    for tp_rank in range(tp_size):
        proj = _projection(num_kv_heads, tp_rank, tp_size, quant_config=None)
        proj.load("weight", weight)
        q, k, v = proj(x)
        local = range(tp_rank * chunks_per_rank, (tp_rank + 1) * chunks_per_rank)
        for part, out in (("q", q), ("k", k), ("v", v)):
            expected = torch.cat([reference[part, c] for c in local], dim=-1)
            torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("num_kv_heads", [4, 8], ids=["full", "sliding"])
@pytest.mark.parametrize("tp_size", [1, 4])
def test_qkv_projection_keeps_fp8_block_scales(num_kv_heads, tp_size):
    """FP8 rows and their 128-row block scales move together, unchanged."""
    quant_config = Fp8Config(
        is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128]
    )
    layout = _checkpoint_rows(num_kv_heads)
    rows_per_chunk = sum(n for _, c, n in layout if c == 0)
    blocks_per_chunk = -(-rows_per_chunk // 128)
    weight = torch.randn(_CHUNKS * rows_per_chunk, _HIDDEN).to(torch.float8_e4m3fn)
    # Scales tile per chunk, as the checkpoint stores them.
    scale = torch.rand(_CHUNKS * blocks_per_chunk, _HIDDEN // 128)

    for tp_rank in range(tp_size):
        proj = _projection(num_kv_heads, tp_rank, tp_size, quant_config)
        proj.load("weight", weight)
        proj.load("weight_scale_inv", scale)
        dst_w = proj.proj.weight.view(torch.uint8)
        src_w = weight.view(torch.uint8).view(_CHUNKS, rows_per_chunk, _HIDDEN)
        src_s = scale.view(_CHUNKS, blocks_per_chunk, -1)
        q_rows, q_blocks = proj.q_rows, proj.q_rows // 128
        for j in range(proj.num_local_chunks):
            c = proj.first_chunk + j
            torch.testing.assert_close(
                dst_w[j * q_rows : (j + 1) * q_rows], src_w[c, :q_rows]
            )
            torch.testing.assert_close(
                proj.proj.weight_scale_inv[j * q_blocks : (j + 1) * q_blocks],
                src_s[c, :q_blocks],
            )
            kv = proj.q_size + j * proj.kv_stride
            kv_rows = rows_per_chunk - q_rows
            torch.testing.assert_close(dst_w[kv : kv + kv_rows], src_w[c, q_rows:])
            assert not dst_w[kv + kv_rows : kv + proj.kv_stride].any()
            kv_block = proj.num_local_chunks * q_blocks + j * proj.kv_blocks
            torch.testing.assert_close(
                proj.proj.weight_scale_inv[kv_block : kv_block + proj.kv_blocks],
                src_s[c, q_blocks:],
            )


def test_qkv_projection_rejects_tp_beyond_checkpoint_chunks():
    with pytest.raises(ValueError, match="must divide the 4 pre-sharded"):
        _projection(8, tp_rank=0, tp_size=8, quant_config=None)


def _flash_like_hf_config():
    pattern = [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0]
    return SimpleNamespace(
        hybrid_layer_pattern=pattern,
        num_hidden_layers=len(pattern),
        num_attention_heads=64,
        swa_num_attention_heads=64,
        num_key_value_heads=4,
        swa_num_key_value_heads=8,
        head_dim=192,
        swa_head_dim=192,
        v_head_dim=128,
        swa_v_head_dim=128,
    )


def test_configure_mimo_v2_attention_records_layer_geometry():
    hf = _flash_like_hf_config()
    model_config = SimpleNamespace(hf_text_config=hf)
    configure_mimo_v2_attention(model_config, server_args=None)

    assert (model_config.head_dim, model_config.v_head_dim) == (192, 128)
    assert model_config.layer_num_key_value_heads == tuple(
        8 if s else 4 for s in hf.hybrid_layer_pattern
    )
    assert hf.layer_types[:2] == ["full_attention", "sliding_attention"]
    # 9 sliding + 3 full layers: 3 sub-groups of 3, assigned round-robin.
    sliding = [label for label in hf.cache_layer_types if label != "full_attention"]
    assert sliding[:4] == [
        "sliding_attention_0",
        "sliding_attention_1",
        "sliding_attention_2",
        "sliding_attention_0",
    ]
    assert {sliding.count(f"sliding_attention_{k}") for k in range(3)} == {3}


def test_configure_mimo_v2_attention_rejects_mismatched_heads():
    hf = _flash_like_hf_config()
    hf.swa_v_head_dim = 192
    with pytest.raises(ValueError, match="share head widths"):
        configure_mimo_v2_attention(SimpleNamespace(hf_text_config=hf), None)


def test_mixed_width_kv_cache_keeps_groups_apart():
    """Full layers with half the sliding layers' KV heads and values narrower
    than keys: the planner packs the narrow group two blocks per parent and
    every layer sees its own rows."""
    layer_types = (
        "full_attention",
        "sliding_attention_0",
        "sliding_attention_1",
        "full_attention",
        "sliding_attention_0",
        "sliding_attention_1",
    )
    layer_heads = tuple(2 if t == "full_attention" else 4 for t in layer_types)
    spec = MHAConfig(
        backend_name="triton",
        num_attention_heads=8,
        num_kv_heads=4,
        head_dim=64,
        v_head_dim=32,
        layer_num_kv_heads=layer_heads,
        attn_tp_size=1,
        cache_layer_types=layer_types,
        sliding_window_tokens=128,
    )
    config = AttnConfig(
        components=(spec,),
        device="cpu",
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        kv_cache_quant_method="none",
        kv_cache_mxfp8=False,
        prefix_granularity=64,
        kernel_page_size=64,
        context_len=1024,
        max_bs=2,
    )
    setup = prepare_cache_setup(
        family="mha",
        server_args=SimpleNamespace(kv_offload_config=None, max_total_tokens=None),
        model_config=SimpleNamespace(
            num_attention_layers=len(layer_types), hf_config=SimpleNamespace()
        ),
        attn_config=config,
        draft_model_config=None,
        draft_attn_config=None,
        cache_budget_bytes=1 << 22,
        probe_batch_rows=None,
        decode_input_tokens=1,
        overlap_schedule_depth=0,
    )
    assert setup.spec.layer_kv_head_counts == layer_heads
    arena = create_cache_arena(setup.spec, device="cpu", enable_memory_saver=False)
    pool = create_cache_pool(
        setup.spec, config, arena, num_layers=len(layer_types), rank=0
    )

    for layer_id, heads in enumerate(layer_heads):
        k = pool.get_key_buffer(layer_id)
        v = pool.get_value_buffer(layer_id)
        assert k.shape[1:] == (heads, 64)
        assert v.shape[1:] == (heads, 32)

    # Layers 0 (full) and 1 (sliding) share slab planes at different page
    # payloads. Take a block of each group whose bytes the plan places in
    # different parents and check the writes stay apart.
    plan = setup.spec.memory_plan
    payload = {f.field_id: math.prod(f.shape) * 2 for f in plan.fields}

    def byte_span(field_id: str, block: int) -> tuple[int, int]:
        start = arena.field_block_byte_offset(field_id, block)
        return start, start + payload[field_id]

    full_block, sliding_block = 1, 2
    full_span = byte_span("layer.0.k", full_block)
    sliding_span = byte_span("layer.1.k", sliding_block)
    assert full_span[1] <= sliding_span[0] or sliding_span[1] <= full_span[0]
    page = 64
    full_rows = slice(full_block * page, (full_block + 1) * page)
    sliding_rows = slice(sliding_block * page, (sliding_block + 1) * page)
    pool.get_key_buffer(0)[full_rows].fill_(1)
    pool.get_value_buffer(0)[full_rows].fill_(1)
    pool.get_key_buffer(1)[sliding_rows].fill_(2)
    pool.get_value_buffer(1)[sliding_rows].fill_(2)
    assert (pool.get_key_buffer(0)[full_rows] == 1).all()
    assert (pool.get_value_buffer(0)[full_rows] == 1).all()
    assert (pool.get_key_buffer(1)[sliding_rows] == 2).all()
    assert (pool.get_value_buffer(1)[sliding_rows] == 2).all()
    # The narrow group packs two blocks per parent.
    packing = {g.group_id: g.cache_blocks_per_lcm_block for g in plan.groups}
    assert packing == {
        "full_attention": 2,
        "sliding_attention_0": 1,
        "sliding_attention_1": 1,
    }
