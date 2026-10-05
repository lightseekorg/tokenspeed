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

import random
import socket
import time
import traceback
from typing import List

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tokenspeed_kernel.ops.communication import triton as triton_communication
from tokenspeed_kernel.ops.communication.triton import (
    all_gather,
    all_reduce,
    all_reduce_can_run,
    allreduce_residual_rmsnorm,
    create_state,
    multimem_probe_payload,
    reduce_scatter,
    rsag_all_reduce,
)
from tokenspeed_kernel.platform import current_platform


def get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def test_alloc_symm_escapes_inference_mode(monkeypatch):
    process_group = object()
    handle = object()

    def fake_empty(shape, *, dtype, device):
        assert not torch.is_inference_mode_enabled()
        assert not torch.is_grad_enabled()
        return torch.empty(shape, dtype=dtype, device=device)

    def fake_rendezvous(tensor, *, group):
        assert tensor.shape == (2, 3)
        assert group is process_group
        return handle

    monkeypatch.setattr(triton_communication.symm_mem, "empty", fake_empty)
    monkeypatch.setattr(triton_communication.symm_mem, "rendezvous", fake_rendezvous)

    with torch.inference_mode():
        tensor, result_handle = triton_communication._alloc_symm(
            (2, 3), torch.float32, torch.device("cpu"), process_group
        )

    assert not tensor.is_inference()
    assert result_handle is handle
    tensor.copy_(torch.ones_like(tensor))


def token_cases(world_size: int) -> List[List[int]]:
    cases = [
        [8] * world_size,
        [8 + rank for rank in range(world_size)],
    ]
    if world_size >= 4:
        cases.append([1, 20, 3] + [0] * (world_size - 3))
    else:
        cases.append([3] + [0] * (world_size - 1))
    return cases


def worker_fn(rank, world_size, port, hidden_size, error_dict):
    try:
        worker_main(rank, world_size, port, hidden_size)
    except Exception:
        error_dict[rank] = traceback.format_exc()


def worker_main(rank: int, world_size: int, port: int, hidden_size: int) -> None:
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=world_size,
    )

    try:
        cases = token_cases(world_size)
        max_tokens = max(sum(tokens) for tokens in cases)
        rsag = create_state(
            enable_lamport=False,
            moe_tail_max_rows=0,
            group=dist.group.WORLD,
            rank_in_group=rank,
            attnres_max_numel=0,
            attnres_max_rows=0,
            max_tokens=max_tokens,
            hidden_size=hidden_size,
            device=None,
            max_numel=0,
            max_bytes=0,
        )

        for tokens in cases:
            check_all_gather(rsag, rank, world_size, tokens, hidden_size, device)
            check_reduce_scatter(rsag, rank, world_size, tokens, hidden_size, device)

        if current_platform().is_amd:
            check_all_reduce(rank, world_size, device)
            check_allreduce_residual_rmsnorm(rank, world_size, device)
    finally:
        dist.destroy_process_group()


def check_all_gather(
    rsag, rank: int, world_size: int, tokens: List[int], hidden_size: int, device
) -> None:
    local_tokens = tokens[rank]
    local = torch.full(
        (local_tokens, hidden_size),
        rank + 1,
        dtype=torch.bfloat16,
        device=device,
    )

    result = all_gather(rsag, local, token_list_in_group=tokens)

    expected = torch.empty(
        (sum(tokens), hidden_size), dtype=torch.bfloat16, device=device
    )
    offset = 0
    for peer, peer_tokens in enumerate(tokens):
        expected[offset : offset + peer_tokens].fill_(peer + 1)
        offset += peer_tokens

    assert result.shape == expected.shape
    torch.testing.assert_close(result, expected, atol=0, rtol=0)


def check_all_reduce(rank: int, world_size: int, device) -> None:
    max_numel = 512 * 1024 // torch.empty((), dtype=torch.bfloat16).element_size()
    state = create_state(
        enable_lamport=False,
        moe_tail_max_rows=0,
        group=dist.group.WORLD,
        rank_in_group=rank,
        attnres_max_numel=0,
        attnres_max_rows=0,
        max_tokens=0,
        hidden_size=0,
        max_numel=max_numel,
        max_bytes=0,
        device=device,
    )
    assert state.max_bytes == 0
    assert not triton_communication.symm_outputs_can_run(
        state,
        ((4,),),
        torch.bfloat16,
    )

    for numel in [2880, 20160, 23040, 92160, 184320]:
        tensor = torch.full((numel,), rank + 1, dtype=torch.bfloat16, device=device)
        assert all_reduce_can_run(state, tensor)
        result = all_reduce(state, tensor)
        assert result is tensor
        expected = torch.full_like(result, world_size * (world_size + 1) // 2)
        torch.testing.assert_close(result, expected, atol=0, rtol=0)
        torch.testing.assert_close(tensor, expected, atol=0, rtol=0)

    large = torch.full((300000,), rank + 1, dtype=torch.bfloat16, device=device)
    assert not all_reduce_can_run(state, large)


def check_allreduce_residual_rmsnorm(rank: int, world_size: int, device) -> None:
    hidden = 2880
    eps = 1e-6
    weight = torch.linspace(0.5, 1.5, hidden, dtype=torch.float32, device=device)

    for tokens in [1, 8, 32]:
        x = torch.full((tokens, hidden), rank + 1, dtype=torch.bfloat16, device=device)
        residual = (
            torch.arange(tokens * hidden, dtype=torch.float32, device=device)
            .reshape(tokens, hidden)
            .mul_(0.001)
            .to(torch.bfloat16)
        )

        norm_out, residual_out, scale, partial = allreduce_residual_rmsnorm(
            input_tensor=x,
            residual=residual,
            weight=weight,
            rank=rank,
            group=dist.group.WORLD,
            eps=eps,
            max_token_num=64,
        )
        assert scale is None
        assert partial is None

        reduced = torch.full_like(residual.float(), world_size * (world_size + 1) // 2)
        ref_residual = reduced + residual.float()
        ref_norm = ref_residual * torch.rsqrt(
            ref_residual.pow(2).mean(dim=-1, keepdim=True) + eps
        )
        ref_norm = ref_norm * weight

        torch.testing.assert_close(
            residual_out.float(), ref_residual, atol=2e-2, rtol=2e-2
        )
        torch.testing.assert_close(norm_out.float(), ref_norm, atol=2e-2, rtol=2e-2)


def check_reduce_scatter(
    rsag, rank: int, world_size: int, tokens: List[int], hidden_size: int, device
) -> None:
    full = torch.full(
        (sum(tokens), hidden_size),
        rank + 1,
        dtype=torch.bfloat16,
        device=device,
    )

    result = reduce_scatter(rsag, full, token_list_in_group=tokens)
    expected = torch.full(
        (tokens[rank], hidden_size),
        world_size * (world_size + 1) // 2,
        dtype=torch.bfloat16,
        device=device,
    )

    assert result.shape == expected.shape
    torch.testing.assert_close(result, expected, atol=0, rtol=0)


def run_rsag_test(world_size: int, hidden_size: int) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm is required for TritonRSAG tests")
    if world_size > torch.cuda.device_count():
        pytest.skip(f"Need {world_size} GPUs, have {torch.cuda.device_count()}")

    port = get_open_port()
    error_dict = mp.Manager().dict()
    mp.spawn(
        worker_fn,
        args=(world_size, port, hidden_size, error_dict),
        nprocs=world_size,
        join=True,
    )

    if error_dict:
        raise RuntimeError("\n".join(f"Rank {r}: {e}" for r, e in error_dict.items()))


def test_triton_communication_correctness_world4():
    run_rsag_test(world_size=4, hidden_size=2880)


# --- The multimem (NVLS in-switch) reduction is bitwise for a fixed issuer ---
#
# --batch-invariant-collectives routes every 2-D bf16 all-reduce on a
# multicast-reachable group through rsag_all_reduce: one fixed rank issues the
# in-switch loads for every row and multicasts the sum back. The contract it
# has to meet: a row's sum has exactly one value, whichever repetition
# computes it and however many rows ride along. The issuer is pinned because
# the in-switch association order depends on the issuing rank (measured on
# 8xH20: two issuers disagree on a few elements per 10^7), so a per-rank slice
# would move a row's bits with the slicing.

_BITWISE_REPETITIONS = 8
_BITWISE_ROWS_PER_RANK = 32
_BITWISE_ISSUER = 0


def _rank_scaled_payload(
    rank: int, rows: int, hidden_size: int, device
) -> torch.Tensor:
    """Fixed pseudo-random bf16 rows whose magnitude differs per rank, so the
    fp32 association order of the sum is visible in the rounded result."""
    gen = torch.Generator(device=device).manual_seed(1234 + rank)
    scale = 2.0 ** torch.randint(-6, 7, (rows, 1), generator=gen, device=device)
    return (torch.randn(rows, hidden_size, generator=gen, device=device) * scale).to(
        torch.bfloat16
    )


def _jittered(fn, *args, **kwargs):
    """Call ``fn`` after a random pause so the ranks arrive at the switch in a
    different order every time."""
    time.sleep(random.random() * 0.002)
    return fn(*args, **kwargs)


def _gather_rows(local: torch.Tensor, counts: List[int]) -> torch.Tensor:
    """NCCL all-gather of each rank's ``counts[rank]`` rows into row order."""
    chunks = [
        torch.empty((count, local.shape[-1]), dtype=local.dtype, device=local.device)
        for count in counts
    ]
    dist.all_gather(chunks, local.contiguous())
    return torch.cat(chunks)


def bitwise_worker_fn(rank, world_size, port, hidden_size, error_dict):
    try:
        bitwise_worker_main(rank, world_size, port, hidden_size)
    except Exception:
        error_dict[rank] = traceback.format_exc()


def bitwise_worker_main(rank: int, world_size: int, port: int, hidden_size: int):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        rows = _BITWISE_ROWS_PER_RANK * world_size
        rsag = create_state(
            enable_lamport=False,
            moe_tail_max_rows=0,
            group=dist.group.WORLD,
            rank_in_group=rank,
            attnres_max_numel=0,
            attnres_max_rows=0,
            max_tokens=rows,
            hidden_size=hidden_size,
            device=None,
            max_numel=0,
            max_bytes=0,
        )
        for payload in (
            _rank_scaled_payload(rank, rows, hidden_size, device),
            multimem_probe_payload(rank, world_size, rows, hidden_size, device, seed=7),
        ):
            check_multimem_reduce_scatter_is_bitwise(rsag, rank, world_size, payload)
            check_multimem_all_reduce_is_bitwise(rsag, rank, world_size, payload)
        check_probe_payload_exposes_the_order(rank, world_size, hidden_size, device)
    finally:
        dist.destroy_process_group()


def check_multimem_reduce_scatter_is_bitwise(
    rsag, rank: int, world_size: int, payload: torch.Tensor
) -> None:
    rows, hidden_size = payload.shape
    everything = [0] * world_size
    everything[_BITWISE_ISSUER] = rows
    half = [0] * world_size
    half[_BITWISE_ISSUER] = rows // 2

    # Run invariance: the same reduction gives the same bits every time.
    reference = _jittered(reduce_scatter, rsag, payload, token_list_in_group=everything)
    for repetition in range(1, _BITWISE_REPETITIONS):
        again = _jittered(reduce_scatter, rsag, payload, token_list_in_group=everything)
        assert torch.equal(again, reference), (
            f"rank {rank}: multimem reduce-scatter repetition {repetition} differs "
            "from the first"
        )
    # Batch invariance: a row's sum does not depend on how many rows the same
    # issuer reduces alongside it.
    fewer = _jittered(
        reduce_scatter, rsag, payload[: rows // 2], token_list_in_group=half
    )
    if rank == _BITWISE_ISSUER:
        assert torch.equal(fewer, reference[: rows // 2]), (
            f"rank {rank}: reducing half the rows gives different bits than "
            "reducing all of them"
        )
    # Correctness: the in-switch fp32 add agrees with a rank-ordered fp32 fold
    # up to the association order (both round to bf16 once).
    folded = torch.zeros(rows, hidden_size, dtype=torch.float32, device=payload.device)
    for part in _gather_rows(payload, [rows] * world_size).view(world_size, rows, -1):
        folded += part.float()
    if rank == _BITWISE_ISSUER:
        torch.testing.assert_close(
            reference.float(),
            folded.to(torch.bfloat16).float(),
            rtol=2**-7,
            atol=1e-2,
        )


def check_multimem_all_reduce_is_bitwise(
    rsag, rank: int, world_size: int, payload: torch.Tensor
) -> None:
    rows = payload.shape[0]
    everything = [0] * world_size
    everything[_BITWISE_ISSUER] = rows
    # The issuer's reduce-scatter of every row, broadcast, is what the
    # all-reduce must reproduce for any row count.
    reference = reduce_scatter(rsag, payload, token_list_in_group=everything)
    if rank != _BITWISE_ISSUER:
        reference = torch.empty_like(payload)
    dist.broadcast(reference, src=_BITWISE_ISSUER)
    for row_count in (1, world_size - 1, world_size + 1, 3 * world_size + 5, rows):
        reduced = _jittered(
            rsag_all_reduce, rsag, payload[:row_count], issuer=_BITWISE_ISSUER
        )
        assert reduced.shape == (row_count, payload.shape[1])
        assert torch.equal(reduced, reference[:row_count]), (
            f"rank {rank}: the multimem all-reduce of {row_count} rows differs from "
            "the issuer's reduce-scatter of the same rows"
        )
        # Every rank holds the same result (the all-gather leg).
        everyone = _gather_rows(reduced, [row_count] * world_size)
        for peer in range(world_size):
            assert torch.equal(
                everyone[peer * row_count : (peer + 1) * row_count], reduced
            ), f"rank {rank}: rank {peer} holds a different all-reduce result"
        for repetition in range(1, _BITWISE_REPETITIONS):
            again = _jittered(
                rsag_all_reduce, rsag, payload[:row_count], issuer=_BITWISE_ISSUER
            )
            assert torch.equal(again, reduced), (
                f"rank {rank}: multimem all-reduce repetition {repetition} of "
                f"{row_count} rows differs from the first"
            )


def check_probe_payload_exposes_the_order(
    rank: int, world_size: int, hidden_size: int, device
) -> None:
    """The startup self-check's payload must tell two association orders apart
    in software, or it could not tell two reduction orders apart in hardware."""
    rows = 16
    slices = _gather_rows(
        multimem_probe_payload(rank, world_size, rows, hidden_size, device, seed=3),
        [rows] * world_size,
    ).view(world_size, rows, hidden_size)
    forward = torch.zeros(rows, hidden_size, dtype=torch.float32, device=device)
    for part in slices:
        forward += part.float()
    backward = torch.zeros_like(forward)
    for part in reversed(slices):
        backward += part.float()
    differing = (forward.to(torch.bfloat16) != backward.to(torch.bfloat16)).float()
    if world_size < 3:
        # Two addends have one sum whatever the order; nothing to expose.
        assert differing.mean() == 0
        return
    assert differing.mean() > 0.25, (
        f"rank {rank}: the probe payload separates two fold orders on only "
        f"{differing.mean():.1%} of its elements"
    )


def run_bitwise_test(world_size: int, hidden_size: int) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the multimem reduction tests")
    if not current_platform().is_nvidia:
        pytest.skip("the multimem (NVLS) reduce-scatter is NVIDIA-only")
    if world_size > torch.cuda.device_count():
        pytest.skip(f"Need {world_size} GPUs, have {torch.cuda.device_count()}")

    port = get_open_port()
    error_dict = mp.Manager().dict()
    mp.spawn(
        bitwise_worker_fn,
        args=(world_size, port, hidden_size, error_dict),
        nprocs=world_size,
        join=True,
    )
    if error_dict:
        raise RuntimeError("\n".join(f"Rank {r}: {e}" for r, e in error_dict.items()))


def test_multimem_reduction_is_bitwise_world2():
    run_bitwise_test(world_size=2, hidden_size=6144)


def test_multimem_reduction_is_bitwise_world4():
    run_bitwise_test(world_size=4, hidden_size=6144)


def test_multimem_reduction_is_bitwise_world8():
    run_bitwise_test(world_size=8, hidden_size=6144)
