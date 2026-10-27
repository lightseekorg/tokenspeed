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

"""Iris all-reduce across gfx1250 trays that share one fabric domain."""

import os

import pytest
import torch
import torch.distributed as dist
from utils import is_cdna5

if not is_cdna5():
    pytest.skip("AMD CDNA5 is required", allow_module_level=True)


def _launched_across_hosts() -> bool:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return world_size > int(os.environ.get("LOCAL_WORLD_SIZE", world_size))


@pytest.mark.skipif(
    not _launched_across_hosts(), reason="needs a torchrun launch across hosts"
)
def test_iris_all_reduce_across_two_hosts():
    """Reduce over the world from GPUs that need not start at GPU 0.

    Launch the same command on each tray, for example two ranks per four-GPU
    tray on GPUs 2-3::

        TOKENSPEED_TEST_BASE_GPU=2 torchrun --nnodes 2 --nproc-per-node 2 \\
            --rdzv-backend c10d --rdzv-endpoint <host>:<port> -m pytest \\
            test_iris_fabric_gfx1250.py
    """
    from tokenspeed_kernel.ops.communication._iris import get_or_create_context
    from tokenspeed_kernel.ops.communication.iris import (
        create_iris_state,
        iris_all_reduce,
    )

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    gpu = int(os.environ.get("TOKENSPEED_TEST_BASE_GPU", "0"))
    gpu += int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{gpu}")
    torch.cuda.set_device(device)
    dist.init_process_group(backend="gloo")
    try:
        state = create_iris_state(
            group=dist.group.WORLD,
            rank_in_group=rank,
            staged_max_numel=16 * 7168,
            producer_direct_max_numel=16 * 7168,
            dtype=torch.bfloat16,
            heap_size=None,
            device=torch.device("cuda"),
        )
        ctx = get_or_create_context(heap_size=0)
        assert torch.cuda.current_device() == ctx.gpu_id == gpu
        assert type(ctx.heap.allocator).__name__ == "VMemChunkedAllocator"
        expected = world_size * (world_size + 1) // 2
        for shape in ((1, 7168), (16, 7168)):
            local = torch.full(shape, rank + 1, dtype=torch.bfloat16, device=device)
            result = iris_all_reduce(state, local)
            torch.testing.assert_close(
                result, torch.full_like(local, expected), atol=0, rtol=0
            )
        dist.barrier()
    finally:
        dist.destroy_process_group()
