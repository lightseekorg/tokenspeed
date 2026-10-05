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

"""CPU tests for --emulate-rank-zero: the local collectives, the one-process
world behind every logical group, and the layouts the flag accepts."""

import os
import sys
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.distributed as dist

# CPU-only tests scheduled in runtime-1gpu because they import the full runtime.
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=20, suite="runtime-1gpu")

from tokenspeed.runtime.distributed.comm_backend.emulated import (  # noqa: E402
    EmulatedRankBackend,
)
from tokenspeed.runtime.distributed.process_group_manager import (  # noqa: E402
    ProcessGroupManager,
)
from tokenspeed.runtime.grammar import (  # noqa: E402
    grammar_manager as grammar_manager_module,
)
from tokenspeed.runtime.layers import shared_expert_tp  # noqa: E402
from tokenspeed.runtime.utils import server_args as server_args_module  # noqa: E402
from tokenspeed.runtime.utils.network import get_free_port  # noqa: E402
from tokenspeed.runtime.utils.server_args import ServerArgs  # noqa: E402

TP8 = tuple(range(8))


def test_reduce_scatters_keep_this_ranks_slice():
    backend = EmulatedRankBackend(rank=2)
    tensor = torch.arange(24.0).reshape(8, 3)
    group = (0, 1, 2, 3)

    scattered = backend.reduce_scatter(tensor, group)
    assert torch.equal(scattered, tensor[4:6])
    scattered.zero_()
    assert torch.equal(tensor, torch.arange(24.0).reshape(8, 3))

    tokens = backend.token_reduce_scatter(tensor[:7], group, [2, 1, 3, 1])
    assert torch.equal(tokens, tensor[3:6])


def test_token_all_gather_returns_every_ranks_rows():
    backend = EmulatedRankBackend(rank=0)
    tensor = torch.arange(6.0).reshape(2, 3)

    gathered = backend.token_all_gather(tensor, TP8, [2, 3, 1, 1, 1, 1, 1, 1])
    assert gathered.shape == (11, 3)
    assert torch.equal(gathered, tensor.repeat(6, 1)[:11])

    empty = backend.token_all_gather(tensor[:0], (0, 1), [0, 2])
    assert empty.shape == (2, 3)


def test_uneven_all_to_all_fills_every_received_row():
    backend = EmulatedRankBackend(rank=0)
    tensor = torch.arange(6.0).reshape(2, 3)

    received = torch.empty((5, 3))
    backend.all_to_all_single(
        received, tensor, (0, 1), output_split_sizes=[2, 3], input_split_sizes=[2, 0]
    )
    assert torch.equal(received, tensor.repeat(3, 1)[:5])

    received = torch.empty((3, 3))
    backend.all_to_all_single(
        received,
        tensor[:0],
        (0, 1),
        output_split_sizes=[0, 3],
        input_split_sizes=[0, 0],
    )
    assert torch.equal(received, torch.zeros((3, 3)))


@pytest.fixture
def emulated_manager():
    manager = ProcessGroupManager()
    manager.init_emulated_rank_zero(
        distributed_init_method=f"tcp://127.0.0.1:{get_free_port()}",
        backend="gloo",
        timeout=60,
        device_id=None,
    )
    try:
        yield manager
    finally:
        dist.destroy_process_group()


def test_one_process_world_backs_every_logical_group(emulated_manager):
    emulated_manager.init_process_group(TP8)
    emulated_manager.init_process_group((0, 2, 4, 6))
    group = emulated_manager.get_process_group("gloo", TP8)
    assert group.size() == 1
    assert emulated_manager.get_process_group("gloo", (0, 2, 4, 6)) is group

    tensor = torch.ones(4)
    dist.all_reduce(tensor, group=group)
    assert torch.equal(tensor, torch.ones(4))


def test_shared_expert_check_is_sized_by_its_process_group(emulated_manager):
    mapping = SimpleNamespace(world_size=8, world_group=TP8)
    with mock.patch.object(shared_expert_tp, "pg_manager", emulated_manager):
        assert shared_expert_tp.validate_shared_expert_settings(mapping, "1") is None


def test_grammar_sync_is_sized_by_its_process_group():
    args = SimpleNamespace(
        grammar_backend="xgrammar",
        grammar_compile_timeout_secs=1.0,
        grammar_compile_max_retries=1,
        skip_tokenizer_init=False,
        disable_any_whitespace=False,
        mapping=SimpleNamespace(attn=SimpleNamespace(tp_group=TP8)),
    )
    group = SimpleNamespace(size=lambda: 1)
    pg_manager = grammar_manager_module.pg_manager
    with (
        mock.patch.object(
            grammar_manager_module, "create_grammar_backend", return_value=object()
        ),
        mock.patch.object(pg_manager, "has_process_group", return_value=True),
        mock.patch.object(pg_manager, "get_process_group", return_value=group),
    ):
        manager = grammar_manager_module.GrammarManager(args, object(), vocab_size=32)

    assert manager.grammar_sync_group is group
    assert manager.grammar_sync_size == 1


def _server_args(**overrides) -> ServerArgs:
    platform = SimpleNamespace(is_amd=True, is_nvidia=False, is_hopper_plus=False)
    with (
        mock.patch.object(
            server_args_module, "current_platform", return_value=platform
        ),
        mock.patch.object(
            server_args_module, "get_amdgpu_memory_capacity", return_value=288_000
        ),
    ):
        return ServerArgs(model="x", **overrides)


def test_emulation_keeps_the_layout_without_fused_all_reduce():
    args = _server_args(attn_tp_size=8, emulate_rank_zero=True)

    assert args.mapping.world_size == 8
    assert args.mapping.attn.tp_size == 8
    assert args.mapping.nprocs_per_node == 8
    # The non-emulated layout auto-enables the fusion an emulated rank skips.
    assert _server_args(attn_tp_size=8).enable_allreduce_fusion
    assert not args.enable_allreduce_fusion


@pytest.mark.parametrize(
    ("overrides", "rejected"),
    [
        ({"moe_tp_size": 4}, "an MoE TP x EP size other than the attention TP size"),
        ({"mm_encoder_tp_mode": "data"}, "--mm-encoder-tp-mode data"),
        (
            {
                "prefill_context_parallel_size": 8,
                "disaggregation_mode": "prefill",
                "disable_prefill_graph": True,
                "attention_backend": "dsa",
            },
            "query context parallelism",
        ),
        ({"enable_allreduce_fusion": True}, "--enable-allreduce-fusion"),
        (
            {
                "enable_expert_parallel": True,
                "enable_eplb": True,
                "expert_distribution_recorder_mode": "stat",
                "ep_dispatch_algorithm": "static",
            },
            "--enable-eplb",
        ),
    ],
)
def test_emulation_rejects_layouts_that_need_real_peers(overrides, rejected):
    with pytest.raises(ValueError, match=rejected):
        _server_args(attn_tp_size=8, emulate_rank_zero=True, **overrides)
