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

"""Exercise the memory protocol handshake with real multi-process Gloo."""

from datetime import timedelta
from test.ci_system.ci_register import register_cuda_ci

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

from tokenspeed.runtime.engine.memory_protocol import (
    check_memory_protocol,
)

# CPU regressions use the runtime job's complete dependency environment.
register_cuda_ci(est_time=60, suite="runtime-1gpu")


def _rendezvous(rank, rendezvous, protocols):
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=len(protocols),
        timeout=timedelta(seconds=15),
    )
    try:
        if len(set(protocols)) > 1:
            with pytest.raises(
                RuntimeError,
                match="Memory protocol mismatch before serving",
            ) as refused:
                check_memory_protocol(
                    dist.group.WORLD,
                    replica_memory_transitions=protocols[rank] == 3,
                )
            assert str(list(protocols)) in str(refused.value)
        else:
            check_memory_protocol(
                dist.group.WORLD,
                replica_memory_transitions=protocols[rank] == 3,
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(), reason="requires Gloo"
)
@pytest.mark.parametrize("protocols", [(2, 3), (3, 2), (2, 2), (3, 3)])
def test_memory_protocol_startup_rendezvous(tmp_path, protocols):
    mp.spawn(
        _rendezvous,
        args=((tmp_path / "rendezvous").as_uri(), protocols),
        nprocs=len(protocols),
        join=True,
    )
