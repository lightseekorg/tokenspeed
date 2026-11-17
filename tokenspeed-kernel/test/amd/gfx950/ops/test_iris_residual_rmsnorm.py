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

"""Row-owned Lamport all-reduce + residual RMSNorm: numerics against the
unfused reduction and norm, mailbox reuse across row counts, interleaving
with another collective on the group, and graph replay under rank skew."""

import socket
import traceback

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tokenspeed_kernel.platform import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform().is_cdna4, reason="the fused reduction targets gfx950"
)

MAX_ROWS = 128
EPS = 1e-5


def _get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _worker(rank, world_size, port, error_dict):
    try:
        device = torch.device("cuda", rank)
        torch.cuda.set_device(device)
        dist.init_process_group(
            "nccl",
            init_method=f"tcp://localhost:{port}",
            rank=rank,
            world_size=world_size,
        )
        _check_residual_rmsnorm(rank, world_size, device)
    except Exception:
        error_dict[rank] = traceback.format_exc()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _contribution(base, negative_zero, peer, step):
    values = (base + (peer + 1) * (step + 1) / 8).bfloat16()
    values[:, negative_zero] = -0.0
    return values


def _bf16_steps_apart(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """How many representable BF16 values lie between a and b, elementwise."""

    def ordered(x: torch.Tensor) -> torch.Tensor:
        bits = x.contiguous().view(torch.int16).int()
        return torch.where(bits < 0, -(bits & 0x7FFF), bits)

    return (ordered(a) - ordered(b)).abs()


def _unfused(bases, negative_zero, residual, weight, step, world_size):
    """What the draft runs without the fusion: the group sum in FP32 rank
    order rounded once to BF16 (as the Lamport all-reduces sum), then the
    residual add and RMSNorm launch."""
    from tokenspeed_kernel.ops.layernorm import rmsnorm

    reduced = _contribution(bases, negative_zero, 0, step).float()
    for peer in range(1, world_size):
        reduced += _contribution(bases, negative_zero, peer, step).float()
    return rmsnorm(reduced.bfloat16(), weight, EPS, residual=residual)


def _check_residual_rmsnorm(rank, world_size, device):
    from tokenspeed_kernel.ops.communication.iris import (
        create_iris_state,
        iris_all_reduce,
        iris_lamport_residual_rmsnorm,
        prepare_iris_lamport_residual_rmsnorm,
    )

    all_reduce = create_iris_state(
        group=dist.group.WORLD,
        rank_in_group=rank,
        staged_max_numel=64 * 1024,
        producer_direct_max_numel=8 * 8192,
        dtype=torch.bfloat16,
        heap_size=None,
        device=device,
    )
    torch.manual_seed(7)
    for width in (2048, 16384):
        # The narrowest and widest rows one launch layout covers.
        state = prepare_iris_lamport_residual_rmsnorm(all_reduce, 16, width)
        negative_zero = torch.zeros(width, dtype=torch.bool, device=device)
        base = torch.randn(9, width, device=device)
        residual = torch.randn(9, width, device=device).bfloat16()
        weight = torch.randn(width, device=device).bfloat16()
        norm_out, residual_out = iris_lamport_residual_rmsnorm(
            state, _contribution(base, negative_zero, rank, 0), residual, weight, EPS
        )
        expected_norm, expected_residual = _unfused(
            base, negative_zero, residual, weight, 0, world_size
        )
        assert torch.equal(residual_out, expected_residual), width
        assert _bf16_steps_apart(norm_out, expected_norm).max() <= 1, width

    width = 8192
    state = prepare_iris_lamport_residual_rmsnorm(all_reduce, MAX_ROWS, width)
    scatter, gather = state.scatter.data_ptr(), state.gather.data_ptr()
    weight = torch.randn(width, device=device).bfloat16()
    # Fewer rows than ranks, uneven owner blocks, the capacity (every owner
    # spinning at once), and a row count that leaves the last owners empty,
    # in one stream of launches whose owner blocks move between calls.
    rows_sequence = (8, 1, MAX_ROWS, 13, 64, 7, 3)
    bases = [torch.randn(rows, width, device=device) for rows in rows_sequence]
    residuals = [
        torch.randn(rows, width, device=device).bfloat16() for rows in rows_sequence
    ]
    # Some columns send negative zero, which must not read as an empty word.
    negative_zero = torch.zeros(width, dtype=torch.bool, device=device)
    negative_zero[::97] = True
    inputs = [_contribution(base, negative_zero, rank, 0) for base in bases]
    other = torch.full((4, width), float(rank + 1), device=device).bfloat16()

    def sequence():
        outputs = []
        for index, (values, residual) in enumerate(zip(inputs, residuals)):
            outputs.append(
                iris_lamport_residual_rmsnorm(state, values, residual, weight, EPS)
            )
            if index == 2:
                # Another collective on the group in between, as the target's
                # reductions run between the draft's in one stream.
                outputs.append(iris_all_reduce(all_reduce, other.clone(), safe=False))
        return outputs

    def check(outputs, step):
        reduced_other = outputs.pop(3)
        assert torch.all(reduced_other == world_size * (world_size + 1) / 2)
        for base, residual, (norm_out, residual_out) in zip(bases, residuals, outputs):
            expected_norm, expected_residual = _unfused(
                base, negative_zero, residual, weight, step, world_size
            )
            assert torch.equal(residual_out, expected_residual)
            # Only the sum of squares may be accumulated in another order.
            assert _bf16_steps_apart(norm_out, expected_norm).max() <= 1

    check(sequence(), 0)
    torch.cuda.synchronize()
    dist.barrier()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = sequence()
    for step in range(1, 5):
        for base, values in zip(bases, inputs):
            values.copy_(_contribution(base, negative_zero, rank, step))
        # A late rank finds its peers' pushes already waiting, and its own
        # pushes reach peers still polling.
        if rank == step % world_size:
            torch.cuda._sleep(20000)
        graph.replay()
        check(list(outputs), step)
    torch.cuda.synchronize()
    dist.barrier()

    # More row counts on the same mailboxes before the restore check.
    for rows in (5, 100, 16):
        values = torch.ones((rows, width), device=device, dtype=torch.bfloat16)
        iris_lamport_residual_rmsnorm(
            state, values, torch.zeros_like(values), weight, EPS
        )
    torch.cuda.synchronize()
    # Replay and reuse never moved the captured mailboxes.
    assert (state.scatter.data_ptr(), state.gather.data_ptr()) == (scatter, gather)
    # Every consumer restored what it read: no payload word is left behind.
    for mailbox in (state.scatter, state.gather):
        assert torch.all(mailbox.view(torch.int32) == -(1 << 31))
    dist.barrier()


@pytest.mark.parametrize("world_size", [4, 8])
def test_lamport_residual_rmsnorm_matches_the_unfused_path(world_size):
    if world_size > torch.cuda.device_count():
        pytest.skip(f"Need {world_size} GPUs, have {torch.cuda.device_count()}")
    pytest.importorskip("iris")
    error_dict = mp.Manager().dict()
    mp.spawn(
        _worker,
        args=(world_size, _get_open_port(), error_dict),
        nprocs=world_size,
        join=True,
    )
    if error_dict:
        raise RuntimeError("\n".join(f"Rank {r}: {e}" for r, e in error_dict.items()))
