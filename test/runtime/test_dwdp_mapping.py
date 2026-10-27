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

import pytest

from tokenspeed.runtime.distributed.comm_manager import CommManager
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.engine.rust_host import dp_coordination_ranks
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.execution.types import DpForwardMetadata
from tokenspeed.runtime.utils.server_args import prepare_server_args


def _mapping(rank, *, sharing_size):
    return Mapping(
        rank=rank,
        world_size=16,
        attn_tp_size=1,
        attn_dp_size=16,
        dense_tp_size=1,
        moe_tp_size=1,
        moe_ep_size=1,
        moe_weight_sharing_size=sharing_size,
    )


@pytest.mark.parametrize("rank", range(16))
def test_storage_group_does_not_change_compute_or_routing(rank):
    mapping = _mapping(rank, sharing_size=16)
    assert mapping.moe_weight_group == tuple(range(16))
    assert mapping.attn.dp_rank == rank
    assert mapping.attn.tp_group == mapping.moe.tp_ep_group == (rank,)
    assert not mapping.requires_forward_coordination
    manager = CommManager(mapping, 0, True, False, False, query_sharded=False)
    ctx = ForwardContext(
        attn_backend=None,
        token_to_kv_pool=None,
        bs=1,
        num_extends=1,
        input_num_tokens=11 + rank,
        forward_mode=ForwardMode.EXTEND,
        output_layout=None,
        global_num_tokens=None,
    )
    ctx.apply_dp_metadata(None, required=mapping.requires_forward_coordination)
    assert ctx.global_num_tokens is None
    assert manager.get_num_tokens(ctx) == (11 + rank, 11 + rank)
    assert manager.attn_tp_group_scattered_num_tokens(ctx) == [11 + rank]
    assert manager.dense_tp_group_scattered_num_tokens(ctx) == [11 + rank]
    assert manager.moe_tp_ep_group_scattered_num_tokens(ctx) == [11 + rank]


@pytest.mark.parametrize("rank", range(16))
def test_weight_groups_can_be_smaller_than_instance(rank):
    mapping = _mapping(rank, sharing_size=4)
    assert mapping.moe_weight_group == tuple(range(rank // 4 * 4, rank // 4 * 4 + 4))
    assert mapping.attn.dp_size == 16


def test_collective_compute_still_requires_coordination():
    mapping = Mapping(
        rank=7,
        world_size=16,
        attn_tp_size=1,
        attn_dp_size=16,
        dense_tp_size=1,
        moe_tp_size=1,
        moe_ep_size=16,
    )
    assert mapping.requires_forward_coordination
    with pytest.raises(ValueError, match="local TP1/EP1"):
        Mapping(
            rank=7,
            world_size=16,
            attn_tp_size=1,
            attn_dp_size=16,
            dense_tp_size=1,
            moe_tp_size=1,
            moe_ep_size=16,
            moe_weight_sharing_size=16,
        )


BASE = [
    "--model",
    "x",
    "--world-size",
    "16",
    "--nprocs-per-node",
    "4",
    "--nnodes",
    "4",
    "--attn-tp-size",
    "1",
    "--data-parallel-size",
    "16",
    "--moe-tp-size",
    "1",
    "--ep-size",
    "1",
    "--moe-weight-sharing-size",
    "16",
    "--disaggregation-mode",
    "prefill",
    "--prefill-graph-max-tokens",
    "0",
    "--enforce-eager",
]


def test_cli_selects_storage_separately_from_execution():
    args = prepare_server_args(BASE)
    args.mapping.rank = 11
    assert args.mapping.moe_weight_group == tuple(range(16))
    assert args.mapping.moe.ep_group == (11,)
    assert not args.mapping.requires_forward_coordination


@pytest.mark.parametrize(
    "replacement",
    [
        ["--disaggregation-mode", "decode"],
        ["--prefill-graph-max-tokens", "8192"],
        ["--moe-weight-sharing-size", "3"],
        ["--moe-tp-size", "16"],
        ["--kvstore-storage-backend", "mooncake"],
        ["--enable-eplb"],
        ["--enable-memory-saver"],
        ["--load-format", "dummy"],
        ["--load-format", "extensible"],
    ],
)
def test_unsupported_layouts_fail_before_loading(replacement):
    with pytest.raises(ValueError):
        prepare_server_args(BASE + replacement)


@pytest.mark.parametrize("required", [False, True])
def test_forward_context_preserves_supplied_dp_metadata(required):
    ctx = ForwardContext(None, None, 1, 1, 17, ForwardMode.EXTEND, None)
    metadata = DpForwardMetadata(
        global_num_tokens=[17, 29],
        global_batch_size=[1, 2],
        global_num_decodes=[0, 0],
        global_forward_mode=[ForwardMode.EXTEND.value] * 2,
        all_decode_or_idle=False,
        all_extend=True,
        need_idle_forward=False,
    )
    ctx.apply_dp_metadata(metadata, required=required)
    assert ctx.global_num_tokens == [17, 29]
    assert ctx.global_bs == [1, 2]
    assert ctx.global_num_decodes == [0, 0]
    assert not ctx.all_decode_or_idle
    assert ctx.all_extend


def test_forward_context_requires_metadata_for_collective_execution():
    ctx = ForwardContext(None, None, 1, 1, 17, ForwardMode.EXTEND, None)
    with pytest.raises(RuntimeError, match="DP forward metadata"):
        ctx.apply_dp_metadata(None, required=True)


@pytest.mark.parametrize("rank", [0, 7])
def test_independent_workers_get_no_control_plane_dp_group(rank):
    """Workers that only share expert weights advance independently: the
    host hands the control plane no DP group, so no per-round metadata
    gather, idle forward, or flush rendezvous waits for a peer."""
    mapping = _mapping(rank, sharing_size=16)
    assert not mapping.requires_forward_coordination
    assert (
        dp_coordination_ranks(
            mapping,
            requires_forward_coordination=mapping.requires_forward_coordination,
            shared_l3=False,
        )
        is None
    )
    assert mapping.attn.dp_rank == rank


def test_shared_l3_keeps_the_flush_rendezvous():
    mapping = _mapping(3, sharing_size=16)
    assert dp_coordination_ranks(
        mapping, requires_forward_coordination=False, shared_l3=True
    ) == tuple(range(16))


@pytest.mark.parametrize("ep_size", [1, 2])
def test_collective_execution_keeps_the_control_plane_dp_group(ep_size):
    mapping = Mapping(
        rank=0,
        world_size=2,
        attn_tp_size=1,
        attn_dp_size=2,
        dense_tp_size=1,
        moe_tp_size=1,
        moe_ep_size=ep_size,
        moe_weight_sharing_size=2 if ep_size == 1 else None,
    )
    ranks = dp_coordination_ranks(
        mapping,
        requires_forward_coordination=mapping.requires_forward_coordination,
        shared_l3=False,
    )
    assert ranks == ((0, 1) if ep_size == 2 else None)


def test_no_attention_dp_means_no_dp_group():
    mapping = Mapping(
        rank=0, world_size=2, attn_tp_size=2, moe_tp_size=2, moe_ep_size=1
    )
    assert (
        dp_coordination_ranks(
            mapping, requires_forward_coordination=True, shared_l3=True
        )
        is None
    )


@pytest.mark.parametrize("load_format", ["auto", "safetensors"])
def test_peer_weights_accept_supported_checkpoint_loaders(load_format):
    assert (
        prepare_server_args(BASE + ["--load-format", load_format]).load_format
        == load_format
    )
