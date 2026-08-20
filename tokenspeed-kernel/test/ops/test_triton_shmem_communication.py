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

"""Correctness tests for the PyTorch symmetric-memory AR+RMSNorm example."""

import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tokenspeed_kernel.platform import current_platform

_EPS = 1e-6


def _allocation_failure_worker(rank: int, port: int) -> None:
    from tokenspeed_kernel.ops.communication import triton_shmem as backend

    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=2,
    )

    def local_allocation(*_args, **_kwargs):
        if rank == 1:
            raise torch.OutOfMemoryError("injected rank-local failure")
        return torch.empty(1)

    try:
        backend.symm_mem.empty = local_allocation
        backend.symm_mem.rendezvous = lambda *_args, **_kwargs: pytest.fail(
            "failed allocation reached rendezvous"
        )
        with pytest.raises(
            backend._CollectiveInitializationError,
            match="rank 1: OutOfMemoryError: injected rank-local failure",
        ):
            backend._allocate_symmetric(
                dist.group.WORLD,
                (16, 8),
                torch.bfloat16,
                torch.device("cuda:0"),
            )
    finally:
        dist.destroy_process_group()


def test_rank_local_allocation_failure_is_exchanged_before_rendezvous():
    mp.spawn(_allocation_failure_worker, args=(_open_port(),), nprocs=2, join=True)


def _open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _skip_if_unsupported(world_size: int) -> None:
    if not torch.cuda.is_available() or not current_platform().is_amd:
        pytest.skip("AMD ROCm is required")
    if world_size > torch.cuda.device_count():
        pytest.skip(f"Need {world_size} GPUs, have {torch.cuda.device_count()}")
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    if arch != "gfx950":
        pytest.skip(f"gfx950 is required, found {arch}")


def _reference(
    input_tensor: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    reduced = input_tensor.float()
    dist.all_reduce(reduced)
    residual_out = reduced + residual.float()
    norm_out = residual_out * torch.rsqrt(
        residual_out.square().mean(dim=-1, keepdim=True) + _EPS
    )
    return norm_out * weight.float(), residual_out


def _distributed_worker(
    rank: int,
    world_size: int,
    port: int,
    hidden_size: int,
) -> None:
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        from tokenspeed_kernel.ops.communication.triton_shmem import (
            create_triton_shmem_ar_rmsnorm_state,
            triton_shmem_allreduce_residual_rmsnorm,
        )

        tokens = 7
        state = create_triton_shmem_ar_rmsnorm_state(
            dist.group.WORLD,
            rank,
            max_token_num=tokens,
            hidden_dim=hidden_size,
        )
        assert state is not None

        rows = torch.arange(tokens, device=device, dtype=torch.float32)[:, None]
        columns = torch.arange(hidden_size, device=device, dtype=torch.float32)[None, :]
        input_tensor = (rank + 1 + rows * 0.01 + columns * 0.001).to(torch.bfloat16)
        residual = (rows * 0.02 + columns * 0.002).to(torch.bfloat16)
        weight = torch.linspace(
            0.5,
            1.5,
            hidden_size,
            device=device,
            dtype=torch.float32,
        )

        norm_out, residual_out = triton_shmem_allreduce_residual_rmsnorm(
            state,
            input_tensor,
            residual,
            weight,
            eps=_EPS,
        )
        expected_norm, expected_residual = _reference(
            input_tensor,
            residual,
            weight,
        )
        torch.cuda.synchronize(device)
        torch.testing.assert_close(
            residual_out.float(), expected_residual, atol=2e-2, rtol=2e-2
        )
        torch.testing.assert_close(
            norm_out.float(), expected_norm, atol=2e-2, rtol=2e-2
        )
    finally:
        dist.destroy_process_group()


def test_triton_shmem_ar_rmsnorm():
    world_size = 4
    hidden_size = 2880
    _skip_if_unsupported(world_size)
    mp.spawn(
        _distributed_worker,
        args=(world_size, _open_port(), hidden_size),
        nprocs=world_size,
        join=True,
    )
