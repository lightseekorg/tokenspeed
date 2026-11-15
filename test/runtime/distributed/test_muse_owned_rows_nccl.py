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

"""Muse Spark's NCCL owned-row collectives on real process groups.

Usage:
    python -m pytest test/runtime/distributed/test_muse_owned_rows_nccl.py -v
"""

import socket
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(
    est_time=60,
    suite="runtime-2gpu",
    disabled_on_runners=["amd-*"],
    disabled_on_runners_reason="Muse Spark owns rows over NCCL on NVIDIA only",
)

_WIDTH_PER_RANK = 16


def _worker(rank: int, port: int, world_size: int) -> None:
    from tokenspeed.runtime.distributed.comm_backend.auto import AutoBackend
    from tokenspeed.runtime.distributed.comm_backend.registry import set_global_backend
    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager,
    )
    from tokenspeed.runtime.models.muse_spark import _NcclOwnedRows

    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        group = tuple(range(world_size))
        process_group_manager.init_process_group(group)
        set_global_backend(AutoBackend())
        device = torch.device("cuda", rank)
        owned = _NcclOwnedRows(group)
        width = _WIDTH_PER_RANK * world_size
        for rows in (4 * world_size, 4 * world_size + 1, 4097):
            local = owned.local_rows(rows)
            first = rank * local
            generator = torch.Generator().manual_seed(rows)
            # Small integers sum exactly in BF16, so every rank order agrees.
            partials = torch.randint(
                -8, 8, (world_size, rows, width), generator=generator
            ).to(device=device, dtype=torch.bfloat16)
            padding = (0, 0, 0, local * world_size - rows)
            full = partials.sum(0)
            expected = torch.nn.functional.pad(full, padding)[first : first + local]

            reduced = owned.reduce_scatter_owned_rows(partials[rank])
            torch.testing.assert_close(reduced, expected, atol=0, rtol=0)
            gathered = owned.all_gather_owned_rows(reduced, rows)
            torch.testing.assert_close(gathered, full, atol=0, rtol=0)
            columns = full[
                :, rank * _WIDTH_PER_RANK : (rank + 1) * _WIDTH_PER_RANK
            ].contiguous()
            transposed = owned.transpose_owned_columns(columns, rows)
            torch.testing.assert_close(transposed, expected, atol=0, rtol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 4])
def test_nccl_owned_rows_split_and_restore_padded_rows(world_size) -> None:
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"requires {world_size} GPUs")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    mp.spawn(_worker, args=(port, world_size), nprocs=world_size, join=True)
