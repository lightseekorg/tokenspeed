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

"""Inkling v1's cache parent holds only K/V pages; conv checkpoints borrow them."""

import os
import sys
from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.configs.inkling_config import InklingModelConfig
from tokenspeed.runtime.layers.attention.backends.specific.inkling import (
    conv_columns_for_pool,
)
from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
from tokenspeed.runtime.layers.attention.kv_cache.factory import (
    create_cache_arena,
    create_cache_pool,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
    cache_dtype_bytes,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.setup import cache_recipe

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

# The cache-relevant text config of the Inkling v1 checkpoints: every sixth
# layer is a full-attention layer, the other 55 slide over 512 tokens.
_INKLING_V1 = dict(
    hidden_size=6144,
    num_hidden_layers=66,
    num_attention_heads=64,
    num_key_value_heads=8,
    head_dim=128,
    swa_num_attention_heads=64,
    swa_num_key_value_heads=16,
    swa_head_dim=128,
    local_layer_ids=[i for i in range(66) if i % 6 != 5],
    sliding_window_size=512,
    use_sconv=True,
    sconv_kernel_size=4,
)
_HIDDEN_STREAMS = ("attnconv", "mlpconv")


def _setup(tp: int, kv_cache_dtype: torch.dtype, mxfp8: bool, budget_bytes: int):
    text = InklingModelConfig(**_INKLING_V1)
    attn = AttnConfig(
        components=(
            MHAConfig(
                backend_name="fa4",
                num_attention_heads=text.num_attention_heads,
                num_kv_heads=text.num_key_value_heads,
                head_dim=text.head_dim,
                attn_tp_size=tp,
                cache_layer_types=tuple(text.cache_layer_types),
                sliding_window_tokens=text.sliding_window_size,
            ),
        ),
        device="cpu",
        dtype=torch.bfloat16,
        kv_cache_dtype=kv_cache_dtype,
        kv_cache_quant_method="none",
        kv_cache_mxfp8=mxfp8,
        kernel_page_size=64,
        context_len=4096,
        max_bs=4,
        max_resident_requests=4,
    )
    recipe = cache_recipe(
        "inkling",
        server_args=SimpleNamespace(
            max_total_tokens=None,
            speculative_num_steps=0,
            speculative_num_draft_tokens=1,
            kv_offload_config=None,
            disaggregation_mode="null",
            chunked_prefill_size=8192,
            enable_prefix_caching=True,
        ),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(get_text_config=lambda: text),
            num_attention_layers=text.num_hidden_layers,
        ),
        attn_config=attn,
        draft_model_config=None,
        draft_attn_config=None,
        cache_budget_bytes=budget_bytes,
        probe_batch_rows=None,
        decode_input_tokens=1,
        overlap_schedule_depth=0,
    )
    return recipe.setup(), attn, text


def _plan(tp: int, kv_cache_dtype: torch.dtype, mxfp8: bool):
    return _setup(tp, kv_cache_dtype, mxfp8, 40 << 30)[0].spec.memory_plan


def _is_hidden(field) -> bool:
    return field.field_id.endswith(_HIDDEN_STREAMS)


def _layer(field) -> str:
    return field.field_id.rsplit(".", 1)[0]


@pytest.mark.parametrize(
    "tp,kv_cache_dtype,mxfp8",
    [
        (4, torch.bfloat16, False),
        (4, torch.float8_e4m3fn, True),
        (8, torch.bfloat16, False),
    ],
    ids=["tp4-bf16", "tp4-mxfp8", "tp8-bf16"],
)
def test_parent_holds_only_kv_pages(tp, kv_cache_dtype, mxfp8):
    plan = _plan(tp, kv_cache_dtype, mxfp8)
    kv_fields = [f for f in plan.fields if f.field_id.endswith((".k", ".v"))]
    hidden = [f for f in plan.fields if _is_hidden(f)]
    attention_group = {_layer(f): f.group_id for f in kv_fields}
    attention_planes = {
        f.plane_id for f in plan.fields if f.group_id in attention_group.values()
    }

    assert len(hidden) == 2 * _INKLING_V1["num_hidden_layers"]
    assert {p.plane_id for p in plan.planes} == attention_planes
    # Each attention group's layers checkpoint into one hiddenconv group of their own.
    for field in hidden:
        assert field.group_id == f"hiddenconv_{attention_group[_layer(field)]}"


@pytest.mark.parametrize(
    "kv_cache_dtype,mxfp8",
    [(torch.bfloat16, False), (torch.float8_e4m3fn, True)],
    ids=["bf16", "mxfp8"],
)
def test_cache_groups_do_not_depend_on_tp(kv_cache_dtype, mxfp8):
    # P/D peers of different TP widths must agree on every group's fields, in order.
    group_fields = [
        {
            group.group_id: [
                f.field_id for f in plan.fields if f.group_id == group.group_id
            ]
            for group in plan.groups
        }
        for plan in (_plan(tp, kv_cache_dtype, mxfp8) for tp in (4, 8))
    ]
    assert group_fields[0] == group_fields[1]


def test_rows_wider_than_the_kv_planes_take_one_plane_each():
    # At TP8 an MXFP8 K/V plane (32 KiB) is narrower than a bf16 hidden row (36 KiB).
    plan = _plan(8, torch.float8_e4m3fn, True)
    hidden = [f for f in plan.fields if _is_hidden(f)]
    hidden_planes = {f.plane_id for f in hidden}
    plane_bytes = {p.plane_id: p.bytes_per_lcm_block for p in plan.planes}
    row_bytes = cache_dtype_bytes(hidden[0].dtype) * 3 * _INKLING_V1["hidden_size"]

    assert not hidden_planes & {f.plane_id for f in plan.fields if not _is_hidden(f)}
    # The hiddenconv groups share these planes, so a parent carries one row per plane.
    assert {plane_bytes[p] for p in hidden_planes} == {row_bytes}


def test_hidden_checkpoints_in_distinct_parents_are_disjoint():
    # Enough parents for the scheduler's per-request reservations at max_bs=4.
    setup, attn, text = _setup(4, torch.bfloat16, False, 1 << 30)
    plan = setup.spec.memory_plan
    arena = create_cache_arena(setup.spec, device="cpu", enable_memory_saver=False)
    pool = create_cache_pool(
        setup.spec, attn, arena, num_layers=text.num_hidden_layers, rank=0
    )
    groups = conv_columns_for_pool(pool)["groups"]
    streams = [
        pool.hiddenconv_checkpoint(layer_id, component)
        for layer_id in range(text.num_hidden_layers)
        for component in _HIDDEN_STREAMS
    ]
    # A parent belongs to one group at a time: give each group's block its own parent.
    block = {
        group_id: 1 + parent * plan.group(group_id).cache_blocks_per_lcm_block
        for parent, group_id in enumerate(groups)
    }

    for value, (group_id, view) in enumerate(streams):
        view[block[group_id]].fill_(value)
    for value, (group_id, view) in enumerate(streams):
        assert torch.all(view[block[group_id]] == value)
