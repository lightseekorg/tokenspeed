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

"""CPU tests for NCCL suspend capability and memory-transition coverage."""

from __future__ import annotations

import os
import sys

import pytest
import torch
from torch.distributed import distributed_c10d

from tokenspeed.runtime.distributed import nccl_suspend
from tokenspeed.runtime.distributed.process_group_manager import ProcessGroupManager

# CPU-only tests scheduled in runtime-1gpu because they import the full runtime.
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")


@pytest.mark.parametrize(
    "version, bindings, supported",
    [
        ((2, 29, 6), ("suspend", "resume"), False),
        ((2, 29, 7), (), False),
        ((2, 30, 4), ("suspend",), False),
        ((2, 30, 4), ("resume",), False),
        ((2, 29, 7), ("suspend", "resume"), True),
    ],
)
def test_nccl_region_requires_both_bindings_and_a_supported_version(
    monkeypatch, version, bindings, supported
):
    # Stand in for a torch build's bindings and CUDA runtime, without GPUs.
    backend = type("NcclBackend", (), {name: lambda self: None for name in bindings})
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_nccl_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "ProcessGroupNCCL", backend, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda.nccl, "version", lambda: version)

    assert nccl_suspend.nccl_suspend_supported() is supported


def test_nccl_suspend_without_a_process_group_is_a_no_op():
    import torch.distributed as dist

    from tokenspeed.runtime.distributed.nccl_suspend import (
        resume_nccl_communicators,
        suspend_nccl_communicators,
    )

    if dist.is_initialized():
        pytest.skip("another test initialized torch.distributed in this process")
    assert suspend_nccl_communicators() == 0
    assert resume_nccl_communicators() == 0


class _NcclBackend:
    """Stand in for a CUDA backend while exercising c10d group discovery."""

    def __init__(self, name, trace, *, initialized=True):
        self.name = name
        self.trace = trace
        self.initialized = initialized

    def _is_initialized(self):
        return self.initialized

    def suspend(self):
        self.trace.append(("suspend", self.name))

    def resume(self):
        self.trace.append(("resume", self.name))


class _ProcessGroup:
    def __init__(self, name, ranks, trace, *, initialized=True):
        self.group_name = name
        self.ranks = ranks
        self.backend = _NcclBackend(name, trace, initialized=initialized)

    def _get_backend(self, _device):
        return self.backend


@pytest.fixture
def nccl_world(monkeypatch):
    # Let CPU torch expose fake CUDA communicators through its real group map.
    trace = []
    world = _ProcessGroup("0", (0, 1, 2), trace)
    manager = ProcessGroupManager()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.distributed, "is_nccl_available", lambda: True)
    monkeypatch.setattr(
        torch.distributed, "ProcessGroupNCCL", _NcclBackend, raising=False
    )
    monkeypatch.setattr(distributed_c10d._world, "default_pg", world)
    monkeypatch.setattr(distributed_c10d, "_pg_map", {})
    monkeypatch.setattr(distributed_c10d, "_pg_group_ranks", {})
    monkeypatch.setattr(nccl_suspend, "process_group_manager", manager)

    def register(*groups, managed=(), named=True):
        for group in groups:
            distributed_c10d._world.pg_map[group] = ("nccl", None)
            distributed_c10d._world.pg_group_ranks[group] = {
                rank: index for index, rank in enumerate(group.ranks)
            }
            if group in managed:
                manager.register_process_group(
                    "nccl", group.ranks, group, name=group.group_name if named else None
                )

    return world, trace, register


def test_nccl_memory_preflight_rejects_cross_replica_world(nccl_world):
    world, trace, register = nccl_world
    register(world)

    with pytest.raises(RuntimeError, match="outside the memory coordination domain"):
        nccl_suspend.prepare_nccl_suspend((0, 1), p2p_ranks=())

    assert trace == []


def test_nccl_memory_preflight_rejects_trainer_with_overlapping_rank_ids(nccl_world):
    world, trace, register = nccl_world
    trainer = _ProcessGroup("trainer_update", (0, 1, 2), trace)
    register(world, trainer)

    with pytest.raises(RuntimeError, match="trainer_update.*outside the serving world"):
        nccl_suspend.prepare_nccl_suspend((0, 1, 2), p2p_ranks=())

    assert trace == []


def test_nccl_memory_preflight_refuses_opaque_point_to_point_order(nccl_world):
    world, trace, register = nccl_world
    pipeline = _ProcessGroup("1", (0, 1, 2), trace)
    register(world, pipeline, managed=(pipeline,), named=False)

    with pytest.raises(RuntimeError, match="point-to-point.*unknown suspend order"):
        nccl_suspend.prepare_nccl_suspend((0, 1, 2), p2p_ranks=((0, 1, 2),))

    assert trace == []


def test_nccl_memory_plan_orders_shared_groups_and_resumes_the_same_backends(
    nccl_world,
):
    world, trace, register = nccl_world
    tp = _ProcessGroup("2", (0, 1), trace)
    ep = _ProcessGroup("1", (0, 2), trace)
    lazy = _ProcessGroup("3", (0, 1), trace, initialized=False)
    register(tp, lazy, ep, world, managed=(tp, ep, lazy))

    plan = nccl_suspend.prepare_nccl_suspend((0, 1, 2), p2p_ranks=())
    assert [(group.name, group.ranks) for group in plan] == [
        ("0", (0, 1, 2)),
        ("1", (0, 2)),
        ("2", (0, 1)),
    ]
    assert trace == []

    nccl_suspend.apply_nccl_memory_plan(plan, release=True)
    lazy.backend.initialized = True
    nccl_suspend.check_nccl_resume(plan)
    nccl_suspend.apply_nccl_memory_plan(plan, release=False)
    assert trace == [
        ("suspend", "0"),
        ("suspend", "1"),
        ("suspend", "2"),
        ("resume", "0"),
        ("resume", "1"),
        ("resume", "2"),
    ]

    ep.backend.initialized = False
    with pytest.raises(RuntimeError, match="no longer initialized"):
        nccl_suspend.check_nccl_resume(plan)


def test_legacy_nccl_memory_keeps_process_group_order(nccl_world):
    world, trace, register = nccl_world
    trainer = _ProcessGroup("trainer_update", (0, 1, 2), trace)
    register(trainer, world)

    assert nccl_suspend.suspend_nccl_communicators() == 2
    assert nccl_suspend.resume_nccl_communicators() == 2
    assert trace == [
        ("suspend", "trainer_update"),
        ("suspend", "0"),
        ("resume", "trainer_update"),
        ("resume", "0"),
    ]
