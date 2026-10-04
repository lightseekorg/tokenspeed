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

"""The expert load counters count real rows only: layout, mask, graph owners."""

from __future__ import annotations

import ast
import inspect
import textwrap

import pytest
import torch

from tokenspeed.runtime.distributed.comm_manager import moe_input_row_segments
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution import forward_step, prefill_graph
from tokenspeed.runtime.execution.query_shard import scatter_count
from tokenspeed.runtime.moe.expert_load_rows import ExpertLoadRowMask, LayerExpertLoad

# ----------------------------------------------------------------------
# Row layout of the MoE input, mirrored from CommManager.
# ----------------------------------------------------------------------


def test_dp_all_gather_lays_one_segment_per_tp_ep_rank_in_group_order():
    # Attention DP4 (TP1), EP4: the MoE input is the all-gather of the four
    # ranks' padded rows; each rank's live rows lead its segment.
    mapping = Mapping(
        rank=2, world_size=4, attn_tp_size=1, attn_dp_size=4, moe_ep_size=4
    )
    segments = moe_input_row_segments(
        mapping,
        padded_global_num_tokens=[8, 8, 8, 8],
        live_global_num_tokens=[8, 3, 0, 5],
    )
    assert segments == [(8, 8), (8, 3), (8, 0), (8, 5)]


def test_attention_tp_shards_split_the_live_prefix_contiguously():
    # Attention TP2 x DP2, MoE TP-EP over all four ranks: each DP group's
    # padded rows are reduce-scattered over its two TP ranks; the live prefix
    # of 5 rows fills shard 0 (4 rows) and one row of shard 1.
    mapping = Mapping(
        rank=0, world_size=4, attn_tp_size=2, attn_dp_size=2, moe_ep_size=4
    )
    assert scatter_count(7, 2) == [4, 3]
    segments = moe_input_row_segments(
        mapping,
        padded_global_num_tokens=[7, 7, 7, 7],
        live_global_num_tokens=[5, 5, 2, 2],
    )
    assert segments == [(4, 4), (3, 1), (4, 2), (3, 0)]


def test_moe_all_reduce_keeps_the_whole_dp_group_rows_as_one_segment():
    # Attention TP2 == MoE TP-EP size 2 (EP2 inside each DP group): the MoE
    # input is the all-reduced rows of this rank's own DP group, nothing
    # gathered.
    mapping = Mapping(
        rank=3,
        world_size=4,
        attn_tp_size=2,
        attn_dp_size=2,
        moe_tp_size=1,
        moe_ep_size=2,
        moe_dp_size=2,
    )
    assert mapping.attn.tp_size == mapping.moe.tp_ep_size
    segments = moe_input_row_segments(
        mapping,
        padded_global_num_tokens=[6, 6, 6, 6],
        live_global_num_tokens=[6, 6, 4, 4],
    )
    assert segments == [(6, 4)]


def test_without_dp_or_tp_the_input_is_this_ranks_rows():
    mapping = Mapping(rank=0, world_size=1)
    assert moe_input_row_segments(
        mapping, padded_global_num_tokens=[4], live_global_num_tokens=[1]
    ) == [(4, 1)]
    with pytest.raises(ValueError, match="do not fit"):
        moe_input_row_segments(
            mapping, padded_global_num_tokens=[4], live_global_num_tokens=[5]
        )
    with pytest.raises(ValueError, match="world_size"):
        moe_input_row_segments(
            mapping, padded_global_num_tokens=[4, 4], live_global_num_tokens=[1]
        )


# ----------------------------------------------------------------------
# The mask and the per-layer counter view.
# ----------------------------------------------------------------------


def test_mask_marks_filler_rows_then_clears_and_the_counters_follow():
    mapping = Mapping(
        rank=1, world_size=2, attn_tp_size=1, attn_dp_size=2, moe_ep_size=2
    )
    rows = ExpertLoadRowMask(mapping)
    with pytest.raises(RuntimeError, match="not been reserved"):
        rows.rows(1)
    rows.reserve(8, "cpu")
    assert rows.max_rows == 8 and rows.rows(8).all()
    with pytest.raises(RuntimeError, match="already reserved"):
        rows.reserve(8, "cpu")

    rows.mark_padded(padded_global_num_tokens=[3, 3], live_global_num_tokens=[2, 1])
    assert rows.rows(6).tolist() == [True, True, False, True, False, False]
    # Rows past this forward's input are untouched.
    assert rows.rows(8)[6:].all()

    load = LayerExpertLoad(torch.zeros(4, dtype=torch.int64), rows)
    load.record(torch.tensor([[0, 1], [1, 2], [3, 3], [2, -1], [0, 0], [3, 0]]))
    assert load.physical_load.tolist() == [1, 2, 2, 0]
    rows.clear()
    assert rows.rows(8).all()
    load.record(torch.tensor([[0, 1], [1, 2], [3, 3], [2, -1], [0, 0], [3, 0]]))
    assert load.physical_load.tolist() == [5, 4, 4, 3]


def test_mask_rejects_a_forward_wider_than_the_reservation():
    rows = ExpertLoadRowMask(Mapping(rank=0, world_size=1))
    rows.reserve(2, "cpu")
    with pytest.raises(ValueError, match="exceeds the reserved"):
        rows.mark_padded(padded_global_num_tokens=[3], live_global_num_tokens=[1])
    with pytest.raises(ValueError, match="positive"):
        ExpertLoadRowMask(Mapping(rank=0, world_size=1)).reserve(0, "cpu")


# ----------------------------------------------------------------------
# The graph owners bracket every padded replay: mark before, clear after.
# ----------------------------------------------------------------------


def _call_lines(source: str, attr: str) -> list[int]:
    tree = ast.parse(textwrap.dedent(source))
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == attr
    ]


@pytest.mark.parametrize(
    ("owner", "replay_attr"),
    [
        (forward_step.ForwardStepRunner.__call__, "replay"),
        (prefill_graph.PrefillGraph.replay, "replay"),
    ],
)
def test_graph_owners_mark_filler_rows_before_the_replay_and_clear_after(
    owner, replay_attr
):
    source = inspect.getsource(owner)
    marks = _call_lines(source, "mark_padded")
    clears = _call_lines(source, "clear")
    replays = [
        line
        for line in _call_lines(source, replay_attr)
        # The DeepEP adapter's replay() hook precedes the graph's replay.
        if "deepep_adapter" not in source.splitlines()[line - 1]
    ]
    assert len(marks) == 1 and len(clears) == 1 and replays
    assert marks[0] < min(replays) < clears[0]
