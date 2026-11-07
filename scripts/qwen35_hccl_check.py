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

"""Check real HCCL sum and rank-ordered gather on the selected chips."""

import datetime
import json
import os

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401


def main():
    rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(rank)
    dist.init_process_group("hccl", timeout=datetime.timedelta(seconds=120))
    x = torch.full((1024,), rank + 1, dtype=torch.float32, device=f"npu:{rank}")
    from tokenspeed.runtime.distributed.comm_backend.hccl import HcclBackend
    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager,
    )

    process_group_manager.register_process_group("hccl", (0, 1), dist.group.WORLD)
    backend = HcclBackend()
    backend.all_reduce(x, (0, 1))
    torch.testing.assert_close(x.cpu(), torch.full((1024,), 3.0), rtol=0, atol=0)
    source = torch.full((4,), rank, dtype=torch.int32, device=f"npu:{rank}")
    gathered = torch.empty(8, dtype=torch.int32, device=f"npu:{rank}")
    backend.all_gather_single(gathered, source, (0, 1))
    torch.testing.assert_close(
        gathered.cpu(),
        torch.tensor([0] * 4 + [1] * 4, dtype=torch.int32),
        rtol=0,
        atol=0,
    )
    for counts in ([2, 2], [1, 2], [0, 2]):
        full = torch.arange(
            sum(counts) * 4, device=f"npu:{rank}", dtype=torch.float32
        ).reshape(-1, 4)
        actual = backend.token_reduce_scatter(full + rank * 10, (0, 1), counts)
        offset = sum(counts[:rank])
        expected = (full * 2 + 10)[offset : offset + counts[rank]]
        torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)
    torch.npu.synchronize()
    print(
        json.dumps(
            dict(
                rank=rank,
                pid=os.getpid(),
                device=str(x.device),
                visible=os.environ["ASCEND_RT_VISIBLE_DEVICES"],
                all_reduce=True,
                all_gather=True,
                reduce_scatter=True,
                allocated=torch.npu.memory_allocated(),
            )
        ),
        flush=True,
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
