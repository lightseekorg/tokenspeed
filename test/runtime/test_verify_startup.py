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

"""Startup profiling must never alias cache parents or exceed the arena."""

import os

import pytest
import torch

from tokenspeed.runtime.execution.drafter.verify_startup import (
    agree_startup_profile,
    measure_startup_graph,
    plan_profile_cache,
    preserve_profile_tensors,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import CacheGroupSpec


def groups():
    return (
        CacheGroupSpec(
            "history",
            "full_history",
            rows_per_page=16,
            entry_stride_tokens=4,
            replayable=False,
        ),
        CacheGroupSpec(
            "swa",
            "sliding_window",
            rows_per_page=16,
            entry_stride_tokens=1,
            sliding_window_tokens=32,
            replayable=False,
        ),
        CacheGroupSpec("state", "state", replayable=False),
    )


def plan(counts):
    return plan_profile_cache(
        groups(),
        virtual_packing={"history": 2, "swa": 4, "state": 1},
        page_counts=counts,
        batch_size=2,
        context_length=128,
        verify_width=6,
        overlap_schedule_depth=1,
    )


def test_disjoint_requests_and_physical_parents():
    packing = {"history": 2, "swa": 4, "state": 1}
    result = plan({s.group_id: 1000 for s in groups()})
    assert result is not None
    parents = set()
    for group in result:
        pages = list(range(group.first_page, group.page_stop))
        owned = {(p - 1) // packing[group.group_id] for p in pages}
        assert not parents.intersection(owned)
        parents.update(owned)
        rows = group.block_tables()
        live = [{p for p in row if p} for row in rows]
        assert not live[0].intersection(live[1])
        assert set.union(*live) == set(pages)
    assert result[0].pages_per_request == 3
    assert result[1].pages_per_request == 4
    assert result[2].pages_per_request == 2


def test_exact_capacity_and_one_page_short():
    generous = plan({s.group_id: 1000 for s in groups()})
    exact = {g.group_id: g.page_stop for g in generous}
    assert plan(exact) == generous
    for group in generous:
        short = dict(exact)
        short[group.group_id] -= 1
        assert plan(short) is None


def test_sharded_cache_is_not_silently_profiled():
    spec = CacheGroupSpec(
        "kv",
        "full_history",
        rows_per_page=16,
        entry_stride_tokens=1,
        replayable=False,
        shard_count=2,
    )
    assert (
        plan_profile_cache(
            [spec],
            virtual_packing={"kv": 2},
            page_counts={"kv": 100},
            batch_size=1,
            context_length=16,
            verify_width=6,
            overlap_schedule_depth=0,
        )
        is None
    )


def test_invalid_dimensions():
    with pytest.raises(ValueError, match="positive"):
        plan_profile_cache(
            groups(),
            virtual_packing={},
            page_counts={},
            batch_size=0,
            context_length=128,
            verify_width=6,
            overlap_schedule_depth=0,
        )


@pytest.mark.parametrize("fail", [False, True])
def test_profile_state_restores_contents_and_addresses(fail):
    storage = torch.arange(16).reshape(4, 4)
    row = storage[1]
    scalar = torch.tensor(7)
    original = storage.clone()
    address = row.data_ptr()
    try:
        with preserve_profile_tensors([row, scalar]):
            row.fill_(-1)
            scalar.zero_()
            if fail:
                raise RuntimeError("profile failed")
    except RuntimeError as error:
        assert fail and str(error) == "profile failed"
    assert row.data_ptr() == address
    assert torch.equal(storage, original)
    assert scalar.item() == 7


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_startup_graph_measurement_replays_and_restores():
    device = torch.device("cuda", torch.cuda.current_device())
    value = torch.zeros(1024, device=device)
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        value.add_(1)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        value.add_(1)
    torch.cuda.synchronize()
    value.zero_()
    calls = []

    def prepare():
        calls.append(1)
        value.zero_()

    with preserve_profile_tensors([value]):
        times = measure_startup_graph(
            graph.replay,
            prepare,
            device=device,
            host_group=None,
            warmup=2,
            samples=3,
        )
        assert torch.equal(value, torch.ones_like(value))
    assert len(calls) == 5
    assert len(times) == 3 and all(t > 0 for t in times)
    assert torch.equal(value, torch.zeros_like(value))


@pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) < 2, reason="requires torchrun"
)
def test_startup_admission_agrees_across_real_ranks():
    import torch.distributed as dist

    if not dist.is_initialized():
        dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    group = dist.group.WORLD
    for available, bs, capacities, expected in (
        (True, 4, (4, 8, 24), True),
        (rank != world - 1, 4, (4, 8, 24), False),
        (True, 8 if rank == world - 1 else 4, (4, 8, 24), False),
        (True, 4, (4, 12, 24) if rank == world - 1 else (4, 8, 24), False),
    ):
        assert (
            agree_startup_profile(
                available=available,
                batch_size=bs,
                context_length=8192,
                capacities=capacities,
                host_group=group,
            )
            is expected
        )
    dist.barrier()
