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

"""Replicated confidence must publish identical prefixes on every rank."""

import os
import sys
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="runtime-1gpu")

from tokenspeed.runtime.execution.runtime_states import RuntimeStates
from tokenspeed.runtime.execution.verify_layout import VerifyQueryLayout


def _publish_worker(rank, store_path):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{store_path}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        states = RuntimeStates(5, 16, 4, device="cpu")
        states.init_draft_confidence(3)
        slots = torch.tensor([1, 3])
        source = torch.tensor([[0.9, 0.8, 0.7], [0.1, 0.2, 0.3]])
        local = source.clone() if rank == 0 else source.flip(0).clone()
        states.publish_draft_confidence(slots, local, group=dist.group.WORLD)
        torch.testing.assert_close(
            states.draft_confidence[slots], source, atol=0, rtol=0
        )
        layout = VerifyQueryLayout(2, 4, 4, "cpu")
        ready = torch.ones(2, dtype=torch.bool)
        layout.refresh(states.draft_confidence[slots], ready, ready)
        assert layout.query_lengths.tolist() == [3, 1]
        next_source = source.flip(0)
        local.copy_(next_source if rank == 0 else source)
        states.publish_draft_confidence(slots, local, group=dist.group.WORLD)
        torch.testing.assert_close(
            states.previous_draft_confidence[slots], source, atol=0, rtol=0
        )
        torch.testing.assert_close(
            states.draft_confidence[slots], next_source, atol=0, rtol=0
        )
        layout.refresh(states.draft_confidence[slots], ready, ready)
        assert layout.query_lengths.tolist() == [1, 3]
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo is required")
def test_conflicting_rank_scores_publish_the_same_state_and_prefixes(tmp_path):
    mp.spawn(
        _publish_worker, args=(str(tmp_path / "confidence-group"),), nprocs=2, join=True
    )
