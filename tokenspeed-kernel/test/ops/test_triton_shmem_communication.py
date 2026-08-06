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

"""Correctness tests for the symm_mem fused AR+residual+RMSNorm backend.

A near-1:1 clone of ``test_iris_communication.py`` Suite 3, retargeted at the
``triton_shmem`` shim (``create_triton_shmem_ar_rmsnorm_state`` +
``triton_shmem_allreduce_residual_rmsnorm``). Same mp.spawn / fp32-reference /
non-identity linspace-weight design and 2e-2 tolerances.

World sizes and hidden dims are chosen to cover the MI350X kernel variants via
the ``recommended_kernel`` dispatch (``oneshot_max_ws=2``):

* ``hidden=2880`` (not a power of two): ws<=2 -> ``oneshot_blocked``;
  ws>=4 -> ``twoshot_blocked`` (the ws=8 production path, 3 pointer tables +
  trailing barrier -- the highest-risk kernel).
* ``hidden=4096`` (power of two) at ws=2 -> ``oneshot_wholerow``.

Also includes HIP graph capture and replay with changing inputs.
"""

import socket
import traceback
from typing import List

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tokenspeed_kernel.platform import current_platform

_TOKEN_CASES: List[int] = [1, 64, 256, 1024, 8192]
_EPS = 1e-6


def test_arch_profile_is_separate_from_device_kernels():
    from tokenspeed_kernel.ops.communication import _triton_shmem_profile as profile

    assert profile.recommended_kernel(2, 2880, profile="gfx950") == "oneshot_blocked"
    assert profile.recommended_kernel(4, 2880, profile="gfx950") == "twoshot_blocked"
    assert (
        profile.recommended_grid(
            "oneshot_wholerow",
            8,
            128,
            256,
            profile="gfx950",
        )
        == 64
    )


def test_padded_kernel_keeps_m_dynamic():
    from tokenspeed_kernel.ops.communication._triton_shmem_kernels import (
        fused_ar_rmsnorm_oneshot_wholerow_padded_kernel,
    )

    assert "M" in fused_ar_rmsnorm_oneshot_wholerow_padded_kernel.do_not_specialize


def test_configuration_guard_is_gfx950_bf16_only():
    _skip_if_unsupported(1)
    from tokenspeed_kernel.ops.communication.triton_shmem import (
        _configuration_errors,
    )

    assert (
        _configuration_errors(
            arch="gfx950",
            world_size=4,
            max_token_num=2048,
            hidden_dim=2880,
            dtype=torch.bfloat16,
        )
        == []
    )
    errors = _configuration_errors(
        arch="gfx942",
        world_size=3,
        max_token_num=0,
        hidden_dim=0,
        dtype=torch.float16,
    )
    assert len(errors) == 5


def _get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _skip_if_unsupported(world_size: int) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm is required for triton_shmem tests")
    if world_size > torch.cuda.device_count():
        pytest.skip(f"Need {world_size} GPUs, have {torch.cuda.device_count()}")
    if not current_platform().is_amd:
        pytest.skip("triton_shmem backend only targets AMD ROCm")


def _spawn_and_collect(worker_fn, args, world_size: int) -> None:
    error_dict = mp.Manager().dict()
    mp.spawn(worker_fn, args=args + (error_dict,), nprocs=world_size, join=True)
    if error_dict:
        raise RuntimeError("\n".join(f"Rank {r}: {e}" for r, e in error_dict.items()))


def _reference(x, residual, weight, world_size, hidden, eps, device):
    reduced = torch.full(
        (x.shape[0], hidden),
        world_size * (world_size + 1) // 2,
        dtype=torch.float32,
        device=device,
    )
    ref_residual = reduced + residual.float()
    ref_norm = ref_residual * torch.rsqrt(
        ref_residual.pow(2).mean(dim=-1, keepdim=True) + eps
    )
    ref_norm = ref_norm * weight.float()
    return ref_residual, ref_norm


def _make_inputs(tokens, hidden, rank, device):
    # Each rank contributes rank+1 (sum across ranks = ws*(ws+1)/2); residual is
    # non-uniform (deterministic, identical across ranks -> replicated per TP)
    # so a weight/residual bug can't be masked.
    x = torch.full((tokens, hidden), rank + 1, dtype=torch.bfloat16, device=device)
    residual = (
        torch.arange(tokens * hidden, dtype=torch.float32, device=device)
        .reshape(tokens, hidden)
        .mul_(0.001)
        .to(torch.bfloat16)
    )
    return x, residual


# ---------------------------------------------------------------------------
# Suite 1: correctness sweep over token counts
# ---------------------------------------------------------------------------
def _corr_worker_fn(rank, world_size, port, hidden, error_dict):
    try:
        _corr_worker_main(rank, world_size, port, hidden)
    except Exception:
        error_dict[rank] = traceback.format_exc()


def _corr_worker_main(rank: int, world_size: int, port: int, hidden: int) -> None:
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        from tokenspeed_kernel.ops.communication.triton_shmem import (
            create_triton_shmem_ar_rmsnorm_state,
            triton_shmem_allreduce_residual_rmsnorm,
        )

        max_token_num = max(_TOKEN_CASES)
        state = create_triton_shmem_ar_rmsnorm_state(
            group=dist.group.WORLD,
            rank_in_group=rank,
            max_token_num=max_token_num,
            hidden_dim=hidden,
            dtype=torch.bfloat16,
        )
        assert state is not None, "triton_shmem state creation returned None"

        weight = torch.linspace(0.5, 1.5, hidden, dtype=torch.bfloat16, device=device)

        for tokens in _TOKEN_CASES:
            x, residual = _make_inputs(tokens, hidden, rank, device)
            norm_out, residual_out = triton_shmem_allreduce_residual_rmsnorm(
                state,
                input_tensor=x,
                residual=residual,
                weight=weight,
                eps=_EPS,
            )
            ref_residual, ref_norm = _reference(
                x, residual, weight, world_size, hidden, _EPS, device
            )
            torch.testing.assert_close(
                residual_out.float(), ref_residual, atol=2e-2, rtol=2e-2
            )
            torch.testing.assert_close(norm_out.float(), ref_norm, atol=2e-2, rtol=2e-2)
    finally:
        dist.destroy_process_group()


def _run_corr(world_size: int, hidden: int) -> None:
    _skip_if_unsupported(world_size)
    port = _get_open_port()
    _spawn_and_collect(_corr_worker_fn, (world_size, port, hidden), world_size)


def test_triton_shmem_arrms_world1():
    # ws=1: oneshot_blocked (self-only reduce; exercises the single-rank barrier).
    _run_corr(world_size=1, hidden=2880)


def test_triton_shmem_arrms_world2():
    # ws=2: oneshot_blocked (arbitrary-N one-shot pull).
    _run_corr(world_size=2, hidden=2880)


def test_triton_shmem_arrms_world2_padded(monkeypatch):
    monkeypatch.setenv("TS_TRITON_SHMEM_ONESHOT_VARIANT", "padded")
    monkeypatch.setenv("TS_TRITON_SHMEM_PADDED_MAX_M", "64")
    monkeypatch.setenv("TS_TRITON_SHMEM_ONESHOT_NUM_WARPS", "4")
    _run_corr(world_size=2, hidden=2880)


def test_triton_shmem_arrms_world2_forced_blocked(monkeypatch):
    monkeypatch.setenv("TS_TRITON_SHMEM_ONESHOT_VARIANT", "blocked")
    _run_corr(world_size=2, hidden=4096)


def test_triton_shmem_arrms_world4():
    # ws=4: twoshot_blocked (3 pointer tables + trailing barrier).
    _run_corr(world_size=4, hidden=2880)


def test_triton_shmem_arrms_world8():
    # ws=8: twoshot_blocked -- the production path.
    _run_corr(world_size=8, hidden=2880)


def test_triton_shmem_arrms_world2_wholerow():
    # ws=2, power-of-two hidden -> oneshot_wholerow.
    _run_corr(world_size=2, hidden=4096)


# ---------------------------------------------------------------------------
# Suite 2: HIP graph capture + replay (capture-safety is the decisive reason to
# migrate off the rocSHMEM host barrier). Replays with changing input must
# recompute the reduction correctly.
# ---------------------------------------------------------------------------
def _graph_worker_fn(rank, world_size, port, hidden, error_dict):
    try:
        _graph_worker_main(rank, world_size, port, hidden)
    except Exception:
        error_dict[rank] = traceback.format_exc()


def _graph_worker_main(rank: int, world_size: int, port: int, hidden: int) -> None:
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        from tokenspeed_kernel.ops.communication.triton_shmem import (
            create_triton_shmem_ar_rmsnorm_state,
            triton_shmem_allreduce_residual_rmsnorm,
        )

        tokens = 256
        state = create_triton_shmem_ar_rmsnorm_state(
            group=dist.group.WORLD,
            rank_in_group=rank,
            max_token_num=tokens,
            hidden_dim=hidden,
            dtype=torch.bfloat16,
        )
        assert state is not None

        weight = torch.linspace(0.5, 1.5, hidden, dtype=torch.bfloat16, device=device)
        x = torch.empty((tokens, hidden), dtype=torch.bfloat16, device=device)
        residual = (
            torch.arange(tokens * hidden, dtype=torch.float32, device=device)
            .reshape(tokens, hidden)
            .mul_(0.001)
            .to(torch.bfloat16)
        )
        norm_out = torch.empty_like(x)
        residual_out = torch.empty_like(x)

        def launch():
            triton_shmem_allreduce_residual_rmsnorm(
                state,
                input_tensor=x,
                residual=residual,
                weight=weight,
                eps=_EPS,
                norm_out=norm_out,
                residual_out=residual_out,
            )

        # Warmup on a side stream (required before capture).
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            x.fill_(rank + 1)
            launch()
        torch.cuda.current_stream().wait_stream(s)
        dist.barrier()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            launch()
        dist.barrier()

        # Replay with changing input; the captured graph must recompute the AR.
        for it in range(1, 4):
            x.fill_((rank + 1) * it)
            g.replay()
            torch.cuda.synchronize()
            reduced = torch.full(
                (tokens, hidden),
                world_size * (world_size + 1) // 2 * it,
                dtype=torch.float32,
                device=device,
            )
            ref_residual = reduced + residual.float()
            ref_norm = ref_residual * torch.rsqrt(
                ref_residual.pow(2).mean(dim=-1, keepdim=True) + _EPS
            )
            ref_norm = ref_norm * weight.float()
            torch.testing.assert_close(
                residual_out.float(), ref_residual, atol=2e-2, rtol=2e-2
            )
            torch.testing.assert_close(norm_out.float(), ref_norm, atol=2e-2, rtol=2e-2)
            dist.barrier()
    finally:
        dist.destroy_process_group()


def _run_graph(world_size: int, hidden: int) -> None:
    _skip_if_unsupported(world_size)
    port = _get_open_port()
    _spawn_and_collect(_graph_worker_fn, (world_size, port, hidden), world_size)


def test_triton_shmem_arrms_graph_capture_world2():
    # oneshot_blocked under graph capture/replay.
    _run_graph(world_size=2, hidden=2880)


def test_triton_shmem_arrms_graph_capture_world8():
    # twoshot_blocked under graph capture/replay -- the production path.
    _run_graph(world_size=8, hidden=2880)


def test_triton_shmem_capture_requires_persistent_output(monkeypatch):
    _skip_if_unsupported(1)
    from tokenspeed_kernel.ops.communication.triton_shmem import (
        triton_shmem_can_run,
    )

    state = type(
        "State",
        (),
        {"_output_ring_size": 72, "_output_ring_max_m": 384},
    )()
    small = type("Input", (), {"shape": (384, 2880)})()
    large = type("Input", (), {"shape": (512, 2880)})()

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert triton_shmem_can_run(state, small)
    assert not triton_shmem_can_run(state, large)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    assert triton_shmem_can_run(state, large)
