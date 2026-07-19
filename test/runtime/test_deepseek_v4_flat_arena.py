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

"""Focused CPU contracts for the DeepSeek V4 flat device-cache arena."""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("numpy")

from tokenspeed.runtime.configs.deepseek_v4_cache_spec import (  # noqa: E402
    V4_SWA_KV_GROUP_ID,
)
from tokenspeed.runtime.layers.attention.kv_cache.deepseek_v4 import (  # noqa: E402
    DeepseekV4TokenToKVPool,
    V4FlatArenaSet,
    build_deepseek_v4_flat_memory_plan,
    deepseek_v4_cache_layout_from_config,
)

PAGE_SIZE = 64


def _hf_config() -> SimpleNamespace:
    return SimpleNamespace(
        compress_ratios=(1, 4, 128),
        head_dim=512,
        qk_rope_head_dim=64,
        index_head_dim=128,
        sliding_window=128,
    )


def _layout(layer_indices=None):
    return deepseek_v4_cache_layout_from_config(
        _hf_config(),
        page_size=PAGE_SIZE,
        use_fp4_indexer_cache=True,
        layer_indices=layer_indices,
    )


def _plan():
    return build_deepseek_v4_flat_memory_plan(
        target_layout=_layout(),
        target_hf_config=_hf_config(),
        target_layer_num=3,
        target_max_live_requests=2,
        target_max_context_len=256,
        target_max_graph_bs=2,
        max_scheduled_tokens=64,
        max_total_tokens=256,
        draft_layout=_layout((0, 1)),
        draft_hf_config=_hf_config(),
        draft_layer_num=2,
        draft_max_live_requests=3,
        draft_max_context_len=256,
        draft_max_graph_bs=2,
    )


def _component_tensors(arena: V4FlatArenaSet):
    return tuple(
        arena.tensor(
            component.owner,
            component.group_id,
            component.layer,
            component.component,
        )
        for pool in arena.plan.pools
        for component in pool.tensors
    )


def _pool(
    *,
    owner: str,
    arena: V4FlatArenaSet,
    layout,
    layer_num: int,
    max_batch_size: int,
) -> DeepseekV4TokenToKVPool:
    return DeepseekV4TokenToKVPool(
        size=arena.plan.max_total_tokens,
        model_dtype=torch.float32,
        layout=layout,
        layer_num=layer_num,
        device="cpu",
        enable_memory_saver=False,
        max_batch_size=max_batch_size,
        max_context_len=256,
        page_size=PAGE_SIZE,
        rank=0,
        hf_config=_hf_config(),
        max_scheduled_tokens=64,
        cache_owner=owner,
        flat_arena_set=arena,
    )


def test_arena_owns_exact_plan_and_target_draft_are_nonallocating_views():
    plan = _plan()
    arena = V4FlatArenaSet(plan, device="cpu", enable_memory_saver=False)
    tensors = _component_tensors(arena)

    assert len(tensors) == sum(len(pool.tensors) for pool in plan.pools)
    assert sum(tensor.nbytes for tensor in tensors) == plan.payload_bytes
    for pool in plan.pools:
        for component in pool.tensors:
            tensor = arena.tensor(
                component.owner,
                component.group_id,
                component.layer,
                component.component,
            )
            assert tensor.shape == (pool.total_blocks, *component.shape_per_block)
            assert torch.count_nonzero(tensor[0]).item() == 0

    # The shared arena is the sole allocator. Pool construction only binds
    # owner-local views and must never create a second component plane.
    with mock.patch.object(
        torch,
        "empty",
        side_effect=AssertionError("flat pool allocated outside its arena"),
    ), mock.patch.object(
        torch,
        "zeros",
        side_effect=AssertionError("flat pool allocated outside its arena"),
    ):
        target = _pool(
            owner="target",
            arena=arena,
            layout=_layout(),
            layer_num=3,
            max_batch_size=2,
        )
        draft = _pool(
            owner="draft",
            arena=arena,
            layout=_layout((0, 1)),
            layer_num=2,
            max_batch_size=3,
        )

    assert target.device_cache_arena is draft.device_cache_arena is arena
    assert target.flat_memory_plan is draft.flat_memory_plan is plan
    assert target.scheduler_group_specs == draft.scheduler_group_specs
    assert target.owner_group_specs == plan.target_owner_group_specs
    assert draft.owner_group_specs == plan.draft_owner_group_specs
    assert plan.runtime_metadata.graph_batch_rows == 2
    assert target.get_swa_kv_buffer(0) is arena.tensor(
        "target", V4_SWA_KV_GROUP_ID, 0, "swa_kv"
    )
    assert draft.get_swa_kv_buffer(0) is arena.tensor(
        "draft", V4_SWA_KV_GROUP_ID, 0, "swa_kv"
    )
    assert (
        target.get_swa_kv_buffer(0).data_ptr() != draft.get_swa_kv_buffer(0).data_ptr()
    )


@pytest.mark.parametrize(
    ("expected_generation", "error"),
    ((1, None), (2, "generation mismatch")),
    ids=("repairs-next-generation", "rejects-drift-before-mutation"),
)
def test_arena_wake_repair_is_generation_guarded(expected_generation, error):
    arena = V4FlatArenaSet(_plan(), device="cpu", enable_memory_saver=False)
    tensors = _component_tensors(arena)
    for tensor in tensors:
        tensor[0].fill_(1)

    if error is not None:
        snapshots = tuple(tensor[0].clone() for tensor in tensors)
        with pytest.raises(RuntimeError, match=error):
            arena.repair_after_wake(expected_generation=expected_generation)
        assert arena.arena_generation == 0
        assert all(
            torch.equal(tensor[0], snapshot)
            for tensor, snapshot in zip(tensors, snapshots, strict=True)
        )
        return

    assert arena.repair_after_wake(expected_generation=expected_generation) == 1
    assert arena.arena_generation == 1
    assert all(torch.count_nonzero(tensor[0]).item() == 0 for tensor in tensors)
