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

"""State registration across real CPU process-group members."""

import json
import sys
from pathlib import Path
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed.runtime.pd.state_shards import StateShardField, StateShardRegistration  # noqa: E402


def _register(rank, rendezvous):
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=20),
    )
    try:
        group = dist.group.WORLD
        state = StateShardRegistration(rank, 1)
        assert state.exchange("session", "cache") is None
        buffer = torch.empty(64, dtype=torch.uint8)
        base = buffer.data_ptr()
        backing = ((base, buffer.nbytes),)
        fields = {"state": StateShardField(base, 8, 4, 2)}
        with pytest.raises(ValueError, match="backing allocation"):
            state.add(group, fields, regions=((base, 32),), registered_regions=())
        assert (
            state.add(group, fields, regions=backing, registered_regions=()) == backing
        )
        with pytest.raises(ValueError, match="duplicate"):
            state.add(group, fields, regions=backing, registered_regions=backing)
        members = json.loads(state.exchange(f"session-{rank}", f"cache-{rank}"))
        assert [(m["rank"], m["session"], m["cache"]) for m in members] == [
            (0, "session-0", "cache-0"),
            (1, "session-1", "cache-1"),
        ]
        assert members[rank]["fields"]["state"]["base"] == base
        state.confirm(None)
        with pytest.raises(ValueError, match="allocation changed"):
            state.confirm("allocation changed" if rank == 1 else None)
    finally:
        dist.destroy_process_group()


def test_both_members_validate_the_registration_before_publication(tmp_path):
    mp.spawn(_register, args=((tmp_path / "group").as_uri(),), nprocs=2, join=True)
