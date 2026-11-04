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

"""Which decode row shapes a group serves with monokernels."""

import argparse
import dataclasses
from datetime import timedelta

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

from tokenspeed.runtime.layers.monokernel import (
    MAX_WAIT_DEADLINE_SECONDS,
    MIN_WAIT_DEADLINE_SECONDS,
    MonokernelDeployment,
    agree_served_shapes,
    deployment_decline_reasons,
    wait_deadline_seconds,
)

# The validated recipe: no host KV store, nothing that adds kernels on the side.
_RECIPE = MonokernelDeployment(
    host_kv_store=False,
    kvstore_io_backend="kernel",
    expert_load_recorder=None,
    memory_saver=False,
    forward_side_stream=False,
    environ={},
)


@pytest.mark.parametrize(
    "flags, deadline",
    [
        ([], MAX_WAIT_DEADLINE_SECONDS),  # the default watchdog, clamped
        (["--watchdog-timeout", "90"], 45.0),
        (["--watchdog-timeout", "4"], MIN_WAIT_DEADLINE_SECONDS),
        (["--watchdog-timeout", "1000"], MAX_WAIT_DEADLINE_SECONDS),
    ],
)
def test_the_wait_deadline_is_half_the_watchdog_within_its_bounds(flags, deadline):
    """The monokernel wait deadline follows ``--watchdog-timeout``: half of
    it, clamped to the bounds, so a lost hand-off fails before the watchdog
    would call the server hung, and never sooner than a late GPU may take."""
    # Imported here: the spawned ranks of the other tests import this module.
    from tokenspeed.runtime.utils.server_args import ServerArgs

    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    watchdog = parser.parse_args(["--model-path", "m", *flags]).watchdog_timeout
    assert wait_deadline_seconds(watchdog) == deadline


@pytest.mark.parametrize(
    "change, reason",
    [
        ({"host_kv_store": True}, "--kvstore-io-backend direct"),
        ({"environ": {"HSA_ENABLE_SDMA": "0"}}, "HSA_ENABLE_SDMA=0"),
        ({"environ": {"HSA_CU_MASK": "0:0-127"}}, "HSA_CU_MASK"),
        ({"environ": {"ROC_GLOBAL_CU_MASK": "0xff"}}, "ROC_GLOBAL_CU_MASK"),
        ({"expert_load_recorder": "stat"}, "expert loads"),
        ({"memory_saver": True}, "memory saver"),
        ({"forward_side_stream": True}, "side stream"),
    ],
)
def test_each_setting_that_adds_concurrent_kernels_declines(change, reason):
    assert deployment_decline_reasons(_RECIPE) == []
    reasons = deployment_decline_reasons(dataclasses.replace(_RECIPE, **change))
    assert len(reasons) == 1 and reason in reasons[0]


def test_host_kv_store_with_direct_copies_is_served():
    deployment = dataclasses.replace(
        _RECIPE, host_kv_store=True, kvstore_io_backend="direct"
    )
    assert deployment_decline_reasons(deployment) == []


def _agree(rank, path, results):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{path}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        # Rank 1 alone cannot serve two requests; no rank may serve them then.
        local = {(2, 8): ["spills at 16 rows"]} if rank == 1 else {}
        served = agree_served_shapes([(1, 8), (2, 8)], local, dist.group.WORLD)
        results[rank] = (served.shapes, dict(served.declined))
        with pytest.raises(RuntimeError, match="different shapes"):
            agree_served_shapes(
                [(1, 8)] if rank == 0 else [(8, 1)], {}, dist.group.WORLD
            )
    finally:
        dist.destroy_process_group()


def test_a_decline_on_one_rank_declines_the_shape_on_every_rank(tmp_path):
    with mp.Manager() as manager:
        results = manager.dict()
        mp.spawn(
            _agree, args=(str(tmp_path / "rendezvous"), results), nprocs=2, join=True
        )
        results = dict(results)
    expected = (frozenset({(1, 8)}), {(2, 8): ("rank 1: spills at 16 rows",)})
    assert results == {0: expected, 1: expected}
