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

"""CPU tests for live MoE rows within the captured physical TP layout."""

import pytest
import torch

from tokenspeed.runtime.distributed import comm_manager
from tokenspeed.runtime.distributed.comm_manager import CommManager
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.context import ForwardContext, ForwardMode


def _manager(rank: int, tp_size: int, dp_size: int, ep_size: int) -> CommManager:
    world_size = tp_size * dp_size
    mapping = Mapping(
        rank=rank,
        world_size=world_size,
        attn_tp_size=tp_size,
        attn_cp_size=1,
        attn_dp_size=dp_size,
        dense_tp_size=world_size,
        moe_tp_size=1,
        moe_ep_size=ep_size,
    )
    return CommManager(
        mapping=mapping,
        layer_id=0,
        is_moe=True,
        prev_is_moe=True,
        input_layernorm=None,
        post_attn_layernorm=None,
    )


def _context(bucket: int, global_counts: list[int] | None) -> ForwardContext:
    return ForwardContext(
        attn_backend=None,
        token_to_kv_pool=None,
        bs=1,
        num_extends=1,
        input_num_tokens=bucket,
        forward_mode=ForwardMode.EXTEND,
        global_num_tokens=global_counts,
    )


@pytest.mark.parametrize(
    "bucket, real_rows, tp_size, expected",
    [
        (8, 5, 2, [4, 1]),
        (8, 1, 2, [1, 0]),
        (256, 130, 2, [128, 2]),
        (8, 3, 4, [2, 1, 0, 0]),
        (7, 5, 4, [2, 2, 1, 0]),
        (9, 8, 4, [3, 2, 2, 1]),
        (1, 1, 4, [1, 0, 0, 0]),
        (0, 0, 4, [0, 0, 0, 0]),
    ],
)
def test_reduce_scatter_uses_padded_offsets(bucket, real_rows, tp_size, expected):
    world_size = tp_size * 2
    ctx = _context(bucket, [bucket] * world_size)
    actual = [
        _manager(rank, tp_size, 2, world_size).moe_num_valid_rows(ctx, real_rows)
        for rank in range(tp_size)
    ]
    assert actual == expected
    assert sum(actual) == real_rows


@pytest.mark.parametrize("tp_size", [1, 2, 4])
@pytest.mark.parametrize("bucket", [0, 1, 7, 8, 9, 32])
def test_live_prefix_preserves_unique_rows_after_post_attn_comm(
    monkeypatch, tp_size, bucket
):
    """Exercise the real comm policy with CPU collective stand-ins.

    Unique live row IDs and poisoned padding establish which rows actually
    arrive at each MoE input, rather than reimplementing the prefix formula.
    """
    world_size = tp_size * 2
    ctx = _context(bucket, [bucket] * world_size)
    for real_rows in range(bucket + 1):
        inputs = torch.full((bucket, 1), -1, dtype=torch.int64)
        inputs[:real_rows, 0] = torch.arange(real_rows)
        routed = []
        for rank in range(tp_size):
            manager = _manager(rank, tp_size, 2, world_size)

            def reduce_scatter(tensor, group, scattered_num_tokens):
                assert group == manager.mapping.attn.tp_group
                shard = torch.tensor_split(tensor, tp_size)[rank]
                assert shard.shape[0] == scattered_num_tokens[rank]
                return shard

            monkeypatch.setattr(comm_manager, "token_reduce_scatter", reduce_scatter)
            physical, _ = manager.post_attn_comm(inputs, inputs, ctx)
            live_rows = manager.moe_num_valid_rows(ctx, real_rows)
            routed.extend(physical[:live_rows, 0].tolist())
            assert torch.all(physical[live_rows:] == -1)
            assert manager.moe_num_valid_rows(ctx, None) == physical.shape[0]
        assert routed == list(range(real_rows))


@pytest.mark.parametrize("rank", [0, 1, 2, 3])
@pytest.mark.parametrize("tp_size, ep_size", [(1, 4), (2, 2), (4, 4)])
@pytest.mark.parametrize("real_rows", [0, 1, 3, 5, 7, 8, None])
def test_no_tp_and_all_reduce_keep_full_rows(rank, tp_size, ep_size, real_rows):
    manager = _manager(rank, tp_size, 4 // tp_size, ep_size)
    assert not manager.mapping.has_attn_tp or manager.use_all_reduce(is_moe=True)
    ctx = _context(8, [8] * 4)
    assert manager.moe_num_valid_rows(ctx, real_rows) == (
        8 if real_rows is None else real_rows
    )


def test_dp_replica_uses_its_own_physical_capacity_and_real_prefix():
    bucket_by_dp = [8, 9]
    real_by_dp = [5, 6]
    actual = [
        _manager(rank, 2, 2, 4).moe_num_valid_rows(
            _context(bucket_by_dp[rank // 2], [8, 8, 9, 9]), real_by_dp[rank // 2]
        )
        for rank in range(4)
    ]
    assert actual == [4, 1, 5, 1]


def test_rs_uses_the_collective_layout_when_it_is_overridden():
    ctx = _context(16, [16] * 4)
    ctx.collective_num_tokens = 8
    ctx.collective_global_num_tokens = [8] * 4
    assert [
        _manager(rank, 2, 2, 4).moe_num_valid_rows(ctx, 5) for rank in range(2)
    ] == [4, 1]


def test_rs_without_dp_can_size_from_local_context():
    ctx = _context(8, None)
    # EP1 makes the TP2 post-attention operation a reduce-scatter.
    assert [
        _manager(rank, 2, 1, 1).moe_num_valid_rows(ctx, 5) for rank in range(2)
    ] == [4, 1]


@pytest.mark.parametrize("rank", [0, 1, 2, 3])
def test_dp_rs_requires_global_metadata_on_every_rank(rank):
    manager = _manager(rank, 2, 2, 4)
    with pytest.raises(ValueError, match="require global token counts"):
        manager.moe_num_valid_rows(_context(8, None), 5)


@pytest.mark.parametrize("real_rows", [-1, 9])
@pytest.mark.parametrize("tp_size, ep_size", [(1, 4), (2, 2), (2, 4)])
def test_invalid_real_prefix_cannot_silently_drop_rows(real_rows, tp_size, ep_size):
    manager = _manager(0, tp_size, 4 // tp_size, ep_size)
    with pytest.raises(ValueError, match="physical token range"):
        manager.moe_num_valid_rows(_context(8, [8] * 4), real_rows)
