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

"""Ordinary reductions are ready before a small prefill first selects Iris."""

import os
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tokenspeed.runtime.distributed.mapping import Mapping

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=60, suite="runtime-2gpu")


@pytest.mark.parametrize("rank", range(8))
def test_all_reduce_groups_preserve_role_order_across_pipeline_stages(rank):
    mapping = Mapping(
        rank=rank,
        world_size=8,
        pp_size=2,
        attn_tp_size=2,
        dense_tp_size=4,
        moe_ep_size=4,
    )
    stage = tuple(range(rank // 4 * 4, rank // 4 * 4 + 4))
    attention = tuple(range(rank // 2 * 2, rank // 2 * 2 + 2))
    # Rank 2 must prepare (2, 3) before the wider (0, 1, 2, 3), just as
    # rank 0 prepares its narrow group first. No cross-stage group participates.
    assert mapping.all_reduce_groups == (attention, stage)


@pytest.mark.parametrize("rank", range(8))
def test_all_reduce_groups_keep_narrow_dense_groups_after_attention(rank):
    mapping = Mapping(
        rank=rank,
        world_size=8,
        pp_size=2,
        attn_tp_size=4,
        dense_tp_size=2,
        dense_dp_size=2,
        moe_ep_size=4,
    )
    first = rank // 4 * 4
    stage = tuple(range(first, first + 4))
    dense = tuple(range(rank // 2 * 2, rank // 2 * 2 + 2))
    assert mapping.all_reduce_groups == (stage, dense)


def _check_all_reduce_startup(rank, world_size, device, group):
    from tokenspeed.runtime.distributed.comm_backend.auto import AutoBackend
    from tokenspeed.runtime.distributed.comm_ops import prepare_all_reduce_runtime
    from tokenspeed.runtime.utils.env import global_server_args_dict

    global_server_args_dict["force_deterministic_rsag"] = False
    expected_sum = world_size * (world_size + 1) // 2
    for model_prepared in (False, True):
        backend = AutoBackend()
        backend.configure(use_pynccl=True)
        if model_prepared:
            assert backend.prepare_all_reduce_buffers(
                group,
                staged_max_numel=65536,
                producer_direct_max_numel=16384,
                dtype=torch.bfloat16,
            )
        groups = global_server_args_dict["mapping"].all_reduce_groups
        assert prepare_all_reduce_runtime(groups, torch.bfloat16, backend)
        state = backend.iris.state(group, torch.bfloat16)
        pointers = (state.peers.bases.data_ptr(), state.producer.data.data_ptr())
        if model_prepared:
            assert state.staged_max_numel == 65536
            assert state.producer_direct_max_numel == 16384

        # A large warmup takes RCCL; a short prefix hit later selects Iris.
        large = torch.full((300 * 1024,), rank + 1, dtype=torch.bfloat16, device=device)
        assert not backend.iris.can_run(large, group)
        torch.testing.assert_close(
            backend.all_reduce(large, group),
            torch.full_like(large, expected_sum),
            atol=0,
            rtol=0,
        )
        for rows in (9, 21, 9):
            small = torch.full(
                (rows, 2880), rank + 1, dtype=torch.bfloat16, device=device
            )
            assert backend.iris.can_run(small, group)
            previous = torch.cuda.get_sync_debug_mode()
            torch.cuda.set_sync_debug_mode("error")
            try:
                assert prepare_all_reduce_runtime(groups, torch.bfloat16, backend)
                result = backend.all_reduce(small, group)
            finally:
                torch.cuda.set_sync_debug_mode(previous)
            torch.testing.assert_close(
                result, torch.full_like(result, expected_sum), atol=0, rtol=0
            )
            assert backend.iris.state(group, torch.bfloat16) is state
            assert pointers == (
                state.peers.bases.data_ptr(),
                state.producer.data.data_ptr(),
            )

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = backend.all_reduce(small, group)
        for factor in (2, 3):
            small.fill_(factor * (rank + 1))
            graph.replay()
            torch.testing.assert_close(
                result, torch.full_like(result, factor * expected_sum), atol=0, rtol=0
            )


def _worker(rank, world_size, rendezvous):
    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager as pg_manager,
    )

    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    mapping = Mapping(rank=rank, world_size=world_size, attn_tp_size=world_size)
    pg_manager.init_distributed(
        mapping, distributed_init_method=rendezvous, backend="nccl", device_id=device
    )
    group = tuple(range(world_size))
    pg_manager.init_process_group(group)
    try:
        from tokenspeed.runtime.utils.env import global_server_args_dict

        global_server_args_dict["mapping"] = mapping
        _check_all_reduce_startup(rank, world_size, device, group)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.version.hip is None, reason="Iris runs on AMD")
@pytest.mark.parametrize("world_size", [2, 4])
def test_small_prefills_use_prepared_iris_without_host_synchronization(
    tmp_path, world_size
):
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"needs {world_size} GPUs")
    mp.spawn(
        _worker,
        args=(world_size, f"file://{tmp_path / 'rendezvous'}"),
        nprocs=world_size,
        join=True,
    )
