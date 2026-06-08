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


import socket
import traceback
from typing import List, Tuple

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _skip_if_unsupported(world_size: int, reason_prefix: str) -> None:
    if not torch.cuda.is_available():
        pytest.skip(f"CUDA/ROCm is required for {reason_prefix}")
    if world_size > torch.cuda.device_count():
        pytest.skip(f"Need {world_size} GPUs, have {torch.cuda.device_count()}")
    if not torch.version.hip:
        pytest.skip(f"{reason_prefix} only targets AMD ROCm")
    try:
        import iris  # noqa: F401
    except ImportError:
        pytest.skip("iris is not installed")


def _spawn_and_collect(worker_fn, args, world_size: int) -> None:
    error_dict = mp.Manager().dict()
    mp.spawn(
        worker_fn,
        args=args + (error_dict,),
        nprocs=world_size,
        join=True,
    )

    if error_dict:
        raise RuntimeError("\n".join(f"Rank {r}: {e}" for r, e in error_dict.items()))


# ---------------------------------------------------------------------------
# Suite 1: iris_all_reduce
# ---------------------------------------------------------------------------


def _ar_shape_cases() -> List[Tuple[int, ...]]:
    """Shapes covering small, vector, and 2-D cases."""
    return [
        (8,),
        (16, 64),
        (4, 7, 32),
    ]


def _ar_worker_fn(rank, world_size, port, error_dict):
    try:
        _ar_worker_main(rank, world_size, port)
    except Exception:
        error_dict[rank] = traceback.format_exc()


def _ar_worker_main(rank: int, world_size: int, port: int) -> None:
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    # Iris's example uses gloo because heap-base exchange is host-side; nccl
    # also works, but gloo avoids contending with the iris-managed device
    # memory and matches the upstream example.
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=world_size,
    )

    try:
        # Importing inside the worker avoids pulling iris into the parent
        # process (which has no distributed context).
        from tokenspeed_kernel.ops.communication.iris import create_iris_state

        max_numel = max(int(torch.tensor(s).prod()) for s in _ar_shape_cases())
        state = create_iris_state(
            group=dist.group.WORLD,
            rank_in_group=rank,
            max_numel=max_numel,
            dtype=torch.bfloat16,
        )
        for shape in _ar_shape_cases():
            _check_all_reduce(state, rank, world_size, shape, device)
    finally:
        dist.destroy_process_group()


def _check_all_reduce(state, rank: int, world_size: int, shape, device) -> None:
    from tokenspeed_kernel.ops.communication.iris import iris_all_reduce

    # Each rank contributes a tensor filled with ``rank + 1``; the reduction
    # is therefore ``sum(1..world_size) = world_size*(world_size+1)/2``.
    local = torch.full(shape, rank + 1, dtype=torch.bfloat16, device=device)

    result = iris_all_reduce(state, local)

    expected_value = world_size * (world_size + 1) // 2
    expected = torch.full(shape, expected_value, dtype=torch.bfloat16, device=device)

    assert (
        result.shape == expected.shape
    ), f"shape mismatch: {result.shape} vs {expected.shape}"
    torch.testing.assert_close(result, expected, atol=0, rtol=0)


def _run_ar_test(world_size: int) -> None:
    _skip_if_unsupported(world_size, "Iris all-reduce tests")
    port = _get_open_port()
    _spawn_and_collect(_ar_worker_fn, (world_size, port), world_size)


def test_iris_all_reduce_correctness_world2():
    _run_ar_test(world_size=2)


def test_iris_all_reduce_correctness_world4():
    _run_ar_test(world_size=4)


def test_iris_all_reduce_correctness_world8():
    _run_ar_test(world_size=8)


# ---------------------------------------------------------------------------
# Suite 2: IrisRSAG (reduce-scatter / all-gather)
# ---------------------------------------------------------------------------


def _rsag_uniform_token_cases(world_size: int) -> List[List[int]]:
    return [
        [8] * world_size,
        [16] * world_size,
        [64] * world_size,
    ]


def _rsag_worker_fn(rank, world_size, port, hidden_size, error_dict):
    try:
        _rsag_worker_main(rank, world_size, port, hidden_size)
    except Exception:
        error_dict[rank] = traceback.format_exc()


def _rsag_worker_main(rank: int, world_size: int, port: int, hidden_size: int) -> None:
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    # Match the upstream iris example - gloo for the host-side rendezvous.
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=world_size,
    )

    try:
        from tokenspeed_kernel.ops.communication.iris import create_iris_rsag_state

        cases = _rsag_uniform_token_cases(world_size)
        max_tokens = max(sum(tokens) for tokens in cases)
        rsag = create_iris_rsag_state(
            group=dist.group.WORLD,
            rank_in_group=rank,
            max_tokens=max_tokens,
            hidden_size=hidden_size,
        )

        # The generic ``all_gather`` / ``reduce_scatter`` dispatchers in
        # ``communication.triton`` route AMD calls to ``amd_rsag_*`` (which
        # require ``state.symm_mem_hdl``); we deliberately bypass that
        # dispatcher and call the iris RSAG state directly. ``rsag`` IS the
        # IrisRSAG instance now (no TritonCommState wrapper).
        ag_fn = lambda state, t, **kw: rsag.all_gather(t, **kw)  # noqa: E731
        rs_fn = lambda state, t, **kw: rsag.reduce_scatter(t, **kw)  # noqa: E731

        for tokens in cases:
            _check_all_gather(
                rsag, rank, world_size, tokens, hidden_size, device, ag_fn
            )
            _check_reduce_scatter(
                rsag, rank, world_size, tokens, hidden_size, device, rs_fn
            )
    finally:
        dist.destroy_process_group()


def _check_all_gather(rsag, rank, world_size, tokens, hidden_size, device, all_gather):
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

    assert result.shape == expected.shape, f"{result.shape} vs {expected.shape}"
    torch.testing.assert_close(result, expected, atol=0, rtol=0)


def _check_reduce_scatter(
    rsag, rank, world_size, tokens, hidden_size, device, reduce_scatter
):
    full = torch.full(
        (sum(tokens), hidden_size),
        rank + 1,
        dtype=torch.bfloat16,
        device=device,
    )

    result = reduce_scatter(rsag, full, token_list_in_group=tokens)

    expected_value = world_size * (world_size + 1) // 2
    expected = torch.full(
        (tokens[rank], hidden_size),
        expected_value,
        dtype=torch.bfloat16,
        device=device,
    )

    assert result.shape == expected.shape, f"{result.shape} vs {expected.shape}"
    torch.testing.assert_close(result, expected, atol=0, rtol=0)


def _run_rsag_test(world_size: int, hidden_size: int) -> None:
    _skip_if_unsupported(world_size, "IrisRSAG tests")
    port = _get_open_port()
    _spawn_and_collect(_rsag_worker_fn, (world_size, port, hidden_size), world_size)


def test_iris_rsag_correctness_world2():
    _run_rsag_test(world_size=2, hidden_size=2880)


def test_iris_rsag_correctness_world4():
    _run_rsag_test(world_size=4, hidden_size=2880)


def test_iris_rsag_correctness_world8():
    _run_rsag_test(world_size=8, hidden_size=2880)


# ---------------------------------------------------------------------------
# Suite 3: fused allreduce + residual + RMSNorm
# ---------------------------------------------------------------------------


# Token shapes spanning decode (1), short/long prefill (256, 1024), and
# the full ``max_token_num`` (8192) so we exercise both the small-M code
# path and the path that walks the full symmetric heap buffer. Hidden=2880
# is the gpt-oss-120b size we use elsewhere.
_ARRMS_TOKEN_CASES: List[int] = [1, 64, 256, 1024, 8192]
_ARRMS_HIDDEN_DIM = 2880
_ARRMS_EPS = 1e-6


def _arrms_worker_fn(rank, world_size, port, persistent, error_dict):
    try:
        _arrms_worker_main(rank, world_size, port, persistent)
    except Exception:
        error_dict[rank] = traceback.format_exc()


def _arrms_worker_main(rank: int, world_size: int, port: int, persistent: bool) -> None:
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    # NCCL is fine here — iris's heap-base exchange is host-side and works
    # the same over any default group.
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=world_size,
    )

    try:
        from tokenspeed_kernel.ops.communication.iris import (
            create_iris_ar_rmsnorm_state,
        )

        max_token_num = max(_ARRMS_TOKEN_CASES)
        state = create_iris_ar_rmsnorm_state(
            group=dist.group.WORLD,
            rank_in_group=rank,
            max_token_num=max_token_num,
            hidden_dim=_ARRMS_HIDDEN_DIM,
            dtype=torch.bfloat16,
            persistent=persistent,
        )

        # Use a fixed RMSNorm weight that is *not* identity, so a bug in
        # the weight load path would fail the test.
        weight = torch.linspace(
            0.5, 1.5, _ARRMS_HIDDEN_DIM, dtype=torch.bfloat16, device=device
        )

        for tokens in _ARRMS_TOKEN_CASES:
            _check_arrms_one(
                state,
                rank=rank,
                world_size=world_size,
                tokens=tokens,
                weight=weight,
                device=device,
            )
    finally:
        dist.destroy_process_group()


def _check_arrms_one(state, rank, world_size, tokens, weight, device) -> None:
    from tokenspeed_kernel.ops.communication.iris import (
        iris_allreduce_residual_rmsnorm,
    )

    # Each rank contributes ``rank + 1``; sum across ranks is therefore
    # ``world_size * (world_size + 1) / 2``. Residual is non-uniform
    # (linspace) so the kernel can't accidentally short-circuit it.
    x = torch.full(
        (tokens, _ARRMS_HIDDEN_DIM), rank + 1, dtype=torch.bfloat16, device=device
    )
    residual = (
        torch.arange(tokens * _ARRMS_HIDDEN_DIM, dtype=torch.float32, device=device)
        .reshape(tokens, _ARRMS_HIDDEN_DIM)
        .mul_(0.001)
        .to(torch.bfloat16)
    )

    norm_out, residual_out = iris_allreduce_residual_rmsnorm(
        state,
        input_tensor=x,
        residual=residual,
        weight=weight,
        eps=_ARRMS_EPS,
    )

    # Reference: do everything in fp32, mirroring the AMD test exactly so
    # tolerance differences only reflect implementation noise, not
    # reference noise.
    reduced = torch.full(
        (tokens, _ARRMS_HIDDEN_DIM),
        world_size * (world_size + 1) // 2,
        dtype=torch.float32,
        device=device,
    )
    ref_residual = reduced + residual.float()
    ref_norm = ref_residual * torch.rsqrt(
        ref_residual.pow(2).mean(dim=-1, keepdim=True) + _ARRMS_EPS
    )
    ref_norm = ref_norm * weight.float()

    torch.testing.assert_close(residual_out.float(), ref_residual, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(norm_out.float(), ref_norm, atol=2e-2, rtol=2e-2)


def _run_arrms_test(world_size: int, persistent: bool) -> None:
    _skip_if_unsupported(world_size, "Iris fused tests")
    port = _get_open_port()
    _spawn_and_collect(_arrms_worker_fn, (world_size, port, persistent), world_size)


@pytest.mark.parametrize("persistent", [False, True], ids=["per_row", "persistent"])
def test_iris_allreduce_residual_rmsnorm_world1(persistent: bool):
    # Single-rank smoke test: exercises the inline-barrier self-signal/wait
    # path (rank sends to itself) and the v1 device_barrier no-op case.
    _run_arrms_test(world_size=1, persistent=persistent)


@pytest.mark.parametrize("persistent", [False, True], ids=["per_row", "persistent"])
def test_iris_allreduce_residual_rmsnorm_world2(persistent: bool):
    _run_arrms_test(world_size=2, persistent=persistent)


@pytest.mark.parametrize("persistent", [False, True], ids=["per_row", "persistent"])
def test_iris_allreduce_residual_rmsnorm_world4(persistent: bool):
    _run_arrms_test(world_size=4, persistent=persistent)


@pytest.mark.parametrize("persistent", [False, True], ids=["per_row", "persistent"])
def test_iris_allreduce_residual_rmsnorm_world8(persistent: bool):
    _run_arrms_test(world_size=8, persistent=persistent)
