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

from unittest import mock

import pytest
import torch

from tokenspeed.runtime.distributed.comm_manager import CommManager
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.distributed.process_group_manager import process_group_manager
from tokenspeed.runtime.engine.event_loop import EventLoop
from tokenspeed.runtime.engine.io_struct import (
    FlushCacheReqInput,
    PauseSchedulerReqInput,
    ResumeSchedulerReqInput,
)
from tokenspeed.runtime.engine.pause import PauseController, PauseHooks
from tokenspeed.runtime.engine.request_handler import RequestHandler
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
    "--disaggregation-decode-allocation-policy",
    "independent",
    "--disaggregation-layerwise-interval",
    "0",
    "--prefill-graph-max-tokens",
    "0",
    "--enforce-eager",
    "--load-balance-method",
    "round_robin",
]


def test_copy_kernel_ctas_fill_whole_clusters():
    args = prepare_server_args(BASE + ["--moe-weight-sharing-copy-ctas", "24"])
    assert args.moe_weight_sharing_copy_ctas == 24


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
        ["--moe-weight-sharing-copy-ctas", "0"],
        ["--moe-weight-sharing-copy-ctas", "12"],
    ],
)
def test_unsupported_layouts_fail_before_loading(replacement):
    prepare_server_args(BASE)
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


class _Replies:
    def __init__(self):
        self.items = []

    def send_pyobj(self, item):
        self.items.append(item)


def test_independent_worker_processes_requests_and_flushes_without_peers():
    args = prepare_server_args(BASE)
    args.mapping.rank = 7
    replies = _Replies()
    cleared = []

    def clear_cache():
        cleared.append(True)
        return True

    with (
        mock.patch(
            "tokenspeed.runtime.engine.request_handler.get_tokenizer",
            return_value=None,
        ),
        mock.patch.object(
            process_group_manager, "get_process_group", return_value=object()
        ),
        mock.patch.object(
            torch.distributed,
            "all_reduce",
            side_effect=AssertionError("An independent worker must not wait for peers"),
        ),
    ):
        handler = RequestHandler(
            args,
            hf_eos_token_id=0,
            max_req_len=2048,
            vocab_size=32000,
            recv_func=None,
            send_func=replies,
            can_clear_cache_fn=lambda: True,
            clear_cache_fn=clear_cache,
            tokenizer_kwargs={},
            l3=None,
        )
        handler.process_requests([])
        assert not cleared and not replies.items
        handler.process_requests([FlushCacheReqInput()])
        assert cleared == [True]
        assert len(replies.items) == 1 and replies.items[0].success


@pytest.mark.parametrize("load_format", ["auto", "safetensors"])
def test_peer_weights_accept_supported_checkpoint_loaders(load_format):
    assert (
        prepare_server_args(BASE + ["--load-format", load_format]).load_format
        == load_format
    )


class _PausedExecution:
    _dp_sync_and_check = EventLoop._dp_sync_and_check

    def __init__(self, mapping):
        self.has_dp = mapping.has_attn_dp
        self.requires_dp_sync = mapping.requires_forward_coordination
        self.world_cpu_group = object()
        self._dp_local_info = torch.zeros((1, 4), dtype=torch.int64)
        self._dp_global_info = torch.zeros((2, 4), dtype=torch.int64)


class _IdleDevice:
    def __init__(self):
        self.forwards = []

    def run_idle_forward(self, metadata):
        self.forwards.append(metadata)


@pytest.mark.parametrize("ep_size", [1, 2])
def test_paused_workers_coordinate_only_for_collective_execution(ep_size):
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
    loop = _PausedExecution(mapping)
    replies = _Replies()
    controller = PauseController(replies)
    device = _IdleDevice()
    hooks = PauseHooks(loop, controller, device)

    def gather(output, local, *, group):
        if not mapping.requires_forward_coordination:
            pytest.fail("Independent paused workers must not wait for peers")
        output[0].copy_(local[0])
        output[1].copy_(torch.tensor([4, 1, int(ForwardMode.DECODE), 1]))

    controller.handle_pause(PauseSchedulerReqInput(mode="keep"))
    assert controller.forward_blocked
    with mock.patch.object(torch.distributed, "all_gather_single", side_effect=gather):
        hooks.paused_idle_step()
    if mapping.requires_forward_coordination:
        assert len(device.forwards) == 1
        assert device.forwards[0].global_num_tokens == [0, 4]
    else:
        assert not device.forwards
    controller.handle_resume(ResumeSchedulerReqInput())
    assert not controller.forward_blocked
    assert replies.items[-1].success
