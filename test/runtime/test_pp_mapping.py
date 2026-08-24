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

from tokenspeed.runtime.distributed.mapping import Mapping


def _mapping(rank: int, world_size: int, pp_size: int, **kw) -> Mapping:
    m = Mapping(world_size=world_size, pp_size=pp_size, **kw)
    m.rank = rank
    return m


def test_pp_disabled_by_default():
    m = _mapping(0, 8, 1, attn_tp_size=8)
    assert not m.has_pp
    assert m.pp_size == 1
    assert m.pp_rank == 0
    assert m.is_first_pp_rank and m.is_last_pp_rank
    assert m.stage_world_size == 8
    assert m.attn.tp_size == 8


def test_pp2_stage_split_and_groups():
    # world 8 = 2 stages x TP4. Ranks 0-3 stage 0, ranks 4-7 stage 1.
    for rank in range(8):
        m = _mapping(rank, 8, 2, attn_tp_size=4)
        assert m.stage_world_size == 4
        assert m.attn.tp_size == 4
        assert m.pp_rank == rank // 4
        assert m.pp_group == (rank % 4, rank % 4 + 4)
        assert m.is_first_pp_rank == (rank < 4)
        assert m.is_last_pp_rank == (rank >= 4)
    # Intra-stage TP group stays inside the stage.
    m5 = _mapping(5, 8, 2, attn_tp_size=4)
    assert m5.attn.tp_group == (4, 5, 6, 7)
    assert m5.attn.tp_rank == 1
    assert m5.pp_prev_rank == 1


def test_pp_next_prev_rank():
    m0 = _mapping(2, 8, 2, attn_tp_size=4)
    assert m0.pp_next_rank == 6
    with pytest.raises(AssertionError):
        _ = m0.pp_prev_rank
    m1 = _mapping(6, 8, 2, attn_tp_size=4)
    assert m1.pp_prev_rank == 2
    with pytest.raises(AssertionError):
        _ = m1.pp_next_rank


def test_pp_moe_ep_resolves_inside_stage():
    # EP consumes the stage world, not the whole world.
    m = _mapping(0, 8, 2, attn_tp_size=4, moe_ep_size=4, moe_tp_size=1)
    assert m.moe.ep_size == 4
    assert m.moe.tp_ep_size == 4


def test_pp_world_size_divisibility():
    with pytest.raises(AssertionError):
        Mapping(world_size=6, pp_size=4, attn_tp_size=1)


def test_narrow_to_layers_drops_other_stage_planes():
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
        CacheFieldLayout,
        CacheGroupLayout,
        CacheMemoryPlan,
        CachePlaneLayout,
    )

    plan = CacheMemoryPlan(
        prefix_granularity=64,
        lcm_block_bytes=1000,
        num_lcm_blocks=9,
        groups=(
            CacheGroupLayout(group_id="g", cache_blocks_per_lcm_block=1, page_count=10),
        ),
        planes=(
            CachePlaneLayout(
                plane_id="unit.0", bytes_per_lcm_block=600, arena_offset_bytes=0
            ),
            CachePlaneLayout(
                plane_id="unit.1", bytes_per_lcm_block=400, arena_offset_bytes=6000
            ),
        ),
        fields=(
            CacheFieldLayout(
                group_id="g",
                field_id="layer.0.kv",
                plane_id="unit.0",
                shape=(600,),
                dtype="uint8",
                field_offset_bytes=0,
                page_stride_bytes=600,
            ),
            CacheFieldLayout(
                group_id="g",
                field_id="layer.1.kv",
                plane_id="unit.1",
                shape=(400,),
                dtype="uint8",
                field_offset_bytes=0,
                page_stride_bytes=400,
            ),
        ),
    )
    assert plan.arena_bytes == 10 * 1000

    stage1 = plan.narrow_to_layers(1, 2)
    # Logical geometry unchanged; physical arena holds only layer 1's plane.
    assert stage1.num_lcm_blocks == 9
    assert stage1.lcm_block_bytes == 1000
    assert stage1.arena_bytes == 10 * 400
    assert [f.field_id for f in stage1.fields] == ["layer.1.kv"]
    assert [p.plane_id for p in stage1.planes] == ["unit.1"]
    # The retained plane re-packs to offset 0 of the smaller arena.
    assert stage1.planes[0].arena_offset_bytes == 0
    # Field addressing stays within the narrowed arena.
    top = stage1.field_page_byte_offset("layer.1.kv", 9)
    assert 0 <= top < stage1.arena_bytes

    stage0 = plan.narrow_to_layers(0, 1)
    assert stage0.arena_bytes == 10 * 600
    assert [f.field_id for f in stage0.fields] == ["layer.0.kv"]


def test_pp_layer_partition_explicit_windows():
    from tokenspeed.runtime.distributed.pp_stage import (
        pp_layer_window,
        pp_stage_windows,
    )

    windows = pp_stage_windows(38, 4, (8, 11, 11, 8))
    assert windows == [(0, 8), (8, 19), (19, 30), (30, 38)]

    # The mapping's partition flows through pp_layer_window.
    m = _mapping(6, 8, 4, attn_tp_size=2)
    m.pp_layer_partition = (8, 11, 11, 8)
    assert m.pp_rank == 3
    assert pp_layer_window(38, m) == (30, 38)

    # No partition -> the even split with the remainder in front.
    assert pp_stage_windows(38, 4) == [(0, 10), (10, 20), (20, 29), (29, 38)]


def test_pp_layer_partition_validation():
    from tokenspeed.runtime.distributed.pp_stage import pp_stage_windows

    with pytest.raises(ValueError, match="entries"):
        pp_stage_windows(38, 4, (8, 30))
    with pytest.raises(ValueError, match="sums to"):
        pp_stage_windows(38, 4, (8, 11, 11, 9))
    with pytest.raises(ValueError, match="at least one layer"):
        pp_stage_windows(38, 4, (0, 19, 11, 8))


def test_mapping_accepts_pp_layer_partition():
    m = Mapping(
        world_size=8, pp_size=4, attn_tp_size=2, pp_layer_partition=[8, 11, 11, 8]
    )
    assert m.pp_layer_partition == (8, 11, 11, 8)
    m2 = Mapping(world_size=8, pp_size=4, attn_tp_size=2)
    assert m2.pp_layer_partition is None


def test_k3_attn_res_boundary_blocks_agree():
    """K3 PP wire contract: the upstream stage ships exactly the snapshot rows
    the downstream stage expects, at every cut point and block size (the two
    ceil_divs coincide because the windows abut)."""
    from tokenspeed.runtime.distributed.pp_stage import pp_stage_windows
    from tokenspeed.runtime.utils import ceil_div

    for num_layers in (7, 48, 61):
        for block in (1, 3, 4, 7):
            for pp_size in (2, 3, 4):
                windows = pp_stage_windows(num_layers, pp_size)
                for (_, up_end), (down_start, _) in zip(windows, windows[1:]):
                    assert up_end == down_start
                    sent = ceil_div(up_end, block)
                    expected = ceil_div(down_start, block)
                    assert sent == expected
                    # Every snapshot the upstream wrote is included: the last
                    # block-write layer before the cut has index < sent.
                    last_written = (up_end - 1) // block
                    assert last_written < sent


def test_pp_stage_state_block_residual_round_trip():
    import torch

    from tokenspeed.runtime.distributed.pp_stage import PPStageState

    state = PPStageState(
        hidden_states=torch.zeros(5, 8),
        block_residual=torch.zeros(3, 5, 8),
    )
    tensors = state.tensors()
    assert len(tensors) == 2
    rebuilt = PPStageState.from_tensors(tensors, ["hidden_states", "block_residual"])
    assert rebuilt.block_residual.shape == (3, 5, 8)
    assert rebuilt.hc_x is None
