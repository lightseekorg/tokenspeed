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

"""``CommManager.post_moe_comm`` under ``--moe-combine-order slot``: no reduction.

The slot-order MoE leaf already holds the complete routed rows on every rank;
the one MoE reduction point must not sum them again, whichever model runs the
layer. In the RSAG layout the reduce-scatter also handed each rank its own
token rows, so that slice still happens; and the fused all-reduce+norm that
``post_mlp_fused`` would otherwise defer to is vetoed by ``should_fuse``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.distributed import comm_manager as comm_manager_module
from tokenspeed.runtime.distributed.comm_manager import CommManager
from tokenspeed.runtime.utils.env import global_server_args_dict


def _manager(
    monkeypatch,
    *,
    attn_tp_size: int,
    tp_ep_size: int,
    tp_ep_rank: int,
    combine_order: str = "slot",
):
    monkeypatch.setitem(global_server_args_dict, "layer_boundary_norm", "fused")
    monkeypatch.setitem(global_server_args_dict, "moe_combine_order", combine_order)
    # As if resolve_communication had auto-enabled the fusion and nothing
    # upstream had vetoed it: the manager must veto it itself.
    monkeypatch.setitem(global_server_args_dict, "enable_allreduce_fusion", True)
    monkeypatch.setitem(global_server_args_dict, "comm_fusion_max_num_tokens", 1024)
    mapping = SimpleNamespace(
        has_attn_tp=attn_tp_size > 1,
        attn=SimpleNamespace(
            tp_rank=0,
            tp_group=tuple(range(attn_tp_size)),
            tp_size=attn_tp_size,
            dp_size=tp_ep_size // attn_tp_size,
            qcp_size=1,
            has_qcp=False,
            has_dp=tp_ep_size > attn_tp_size,
            scatter_index=lambda rank: rank,
        ),
        dense=SimpleNamespace(tp_size=attn_tp_size),
        moe=SimpleNamespace(
            tp_ep_size=tp_ep_size,
            has_tp_ep=tp_ep_size > 1,
            tp_ep_rank=tp_ep_rank,
            tp_ep_group=tuple(range(tp_ep_size)),
        ),
    )
    return CommManager(
        mapping=mapping,
        layer_id=0,
        is_moe=True,
        prev_is_moe=False,
        dense_batch_invariant=False,
        query_sharded=False,
    )


def _forbid_collectives(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("a slot-order combine must not be reduced again")

    monkeypatch.setattr(comm_manager_module, "all_reduce", boom)
    monkeypatch.setattr(comm_manager_module, "token_reduce_scatter", boom)


def test_all_reduce_layout_returns_the_rows_untouched(monkeypatch):
    _forbid_collectives(monkeypatch)
    manager = _manager(monkeypatch, attn_tp_size=4, tp_ep_size=4, tp_ep_rank=2)
    rows = torch.arange(12.0).view(6, 2)
    residual = torch.zeros(6, 2)
    ctx = SimpleNamespace(
        query_shard=None,
        collective_global_num_tokens=None,
        global_num_tokens=None,
        collective_num_tokens=None,
        input_num_tokens=6,
    )

    out, out_residual = manager.post_mlp_fused(rows, residual, ctx)

    assert out is rows and out_residual is residual


@pytest.mark.parametrize("rank", [0, 1, 2, 3])
def test_rsag_layout_takes_back_this_ranks_rows(monkeypatch, rank):
    # Attention DP over 4 ranks with EP 4: pre_moe_comm gathered every DP
    # rank's tokens (counts 3, 2, 2, 2); the reduce-scatter would have
    # returned rows [sum(counts[:rank]), +counts[rank]) of the summed tensor.
    _forbid_collectives(monkeypatch)
    manager = _manager(monkeypatch, attn_tp_size=1, tp_ep_size=4, tp_ep_rank=rank)
    counts = [3, 2, 2, 2]
    rows = torch.arange(float(sum(counts) * 2)).view(sum(counts), 2)
    residual = torch.zeros(counts[rank], 2)
    ctx = SimpleNamespace(
        query_shard=None,
        collective_global_num_tokens=None,
        global_num_tokens=counts,
        collective_num_tokens=None,
        input_num_tokens=counts[rank],
    )

    out, out_residual = manager.post_mlp_fused(rows, residual, ctx)

    start = sum(counts[:rank])
    assert torch.equal(out, rows[start : start + counts[rank]])
    assert out_residual is residual


def test_rank_order_still_reduces_and_slot_order_vetoes_the_fusion(monkeypatch):
    reduced = []
    monkeypatch.setattr(
        comm_manager_module,
        "all_reduce",
        lambda rows, group: reduced.append(group) or rows * 2,
    )
    ctx = SimpleNamespace(
        query_shard=None,
        collective_global_num_tokens=None,
        global_num_tokens=None,
        collective_num_tokens=None,
        input_num_tokens=6,
    )
    rows = torch.ones(6, 2)

    rank = _manager(
        monkeypatch, attn_tp_size=4, tp_ep_size=4, tp_ep_rank=1, combine_order="rank"
    )
    # Under "rank" the fused all-reduce+norm is allowed (post_mlp_fused then
    # leaves the reduction to the next norm) and post_moe_comm reduces.
    assert rank.should_fuse(6)
    out, _ = rank.post_moe_comm(rows, None, ctx)
    assert reduced == [(0, 1, 2, 3)] and torch.equal(out, rows * 2)

    slot = _manager(monkeypatch, attn_tp_size=4, tp_ep_size=4, tp_ep_rank=1)
    assert not slot.should_fuse(6)
    out, _ = slot.post_moe_comm(rows, None, ctx)
    assert out is rows and len(reduced) == 1
