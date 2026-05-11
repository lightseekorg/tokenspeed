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

"""Micro-benchmark for the three MI355 Gluon MoE kernels.

Compares each Gluon kernel against its upstream
``triton_kernels.matmul`` / ``triton_kernels.swiglu`` baseline on
representative shapes:

* ``gluon_bf16_gating_gemm``      vs ``torch.matmul`` / triton_kernels matmul
* ``gluon_bf16_dispatch_swiglu``  vs ``triton_kernels.matmul + swiglu``
* ``gluon_bf16_combine``          vs ``triton_kernels.matmul`` (scatter combine)

The script also dumps the static GPR / spill profile per kernel
(``static_profile``) so we can confirm at a glance that no spills sneak
in across configuration changes.
"""

from __future__ import annotations

import argparse
import time

import tokenspeed_kernel  # noqa: F401  (pre-import; see test header)
import torch

DEVICE = "cuda"


def _bench(fn, *, warmup=10, rep=50, sync=True) -> float:
    for _ in range(warmup):
        fn()
    if sync:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(rep):
        fn()
    if sync:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / rep * 1e3  # ms


def _build_ragged(M, E, *, block_m=128):
    from triton_kernels.tensor import make_ragged_tensor_metadata

    per_expert = max(block_m, (M // E) // block_m * block_m)
    M_padded = per_expert * E
    counts = torch.full((E,), per_expert, device=DEVICE, dtype=torch.int32)
    md = make_ragged_tensor_metadata(counts, M_padded)
    return md, M_padded


def _bf16_tensor(*shape):
    return torch.randn(*shape, device=DEVICE, dtype=torch.bfloat16) * 0.05


# ---------------------------------------------------------------------------
# Kernel 1: bf16 gating GEMM
# ---------------------------------------------------------------------------


def bench_gating_gemm(M, N, K, *, warmup=10, rep=50):
    """Compare Gluon dense GEMM to triton's @triton.jit upstream matmul.

    For the gating projection the upstream baseline is whatever comes
    out of ``select_kernel("gemm", "mm", ...)`` -- on AMD this is the
    triton-compiled ``triton.py`` GEMM. We avoid torch.matmul (rocBLAS)
    here because that's an entirely separate optimisation track; the
    user-facing TASK question is "does Gluon beat the upstream Triton
    bf16 GEMM" rather than "does Gluon beat rocBLAS".
    """
    from tokenspeed_kernel.ops.moe.gluon import gluon_bf16_gating_gemm

    x = _bf16_tensor(M, K)
    w = _bf16_tensor(K, N)

    gluon_ms = _bench(
        lambda: gluon_bf16_gating_gemm(x, w),  # autotuner picks block sizes
        warmup=warmup,
        rep=rep,
    )
    # Triton kernel baseline = the upstream tokenspeed triton GEMM
    # (analog of triton_kernels.matmul on a dense 2-D weight).
    from triton_kernels.matmul import PrecisionConfig, matmul

    def triton_call():
        return matmul(x, w, None, precision_config=PrecisionConfig())

    try:
        triton_ms = _bench(triton_call, warmup=warmup, rep=rep)
    except Exception:
        triton_ms = float("nan")
    torch_ms = _bench(lambda: torch.matmul(x, w), warmup=warmup, rep=rep)

    flops = 2.0 * M * N * K
    return {
        "shape": f"M={M},N={N},K={K}",
        "gluon_ms": gluon_ms,
        "baseline_ms": triton_ms,
        "torch_ms": torch_ms,
        "gluon_tflops": flops / (gluon_ms * 1e-3) / 1e12,
        "baseline_tflops": (
            flops / (triton_ms * 1e-3) / 1e12
            if triton_ms == triton_ms
            else float("nan")
        ),
        "torch_tflops": flops / (torch_ms * 1e-3) / 1e12,
        "speedup": triton_ms / gluon_ms if triton_ms == triton_ms else float("nan"),
    }


# ---------------------------------------------------------------------------
# Kernel 2: dispatch + 1st GEMM + SwiGLU
# ---------------------------------------------------------------------------


def bench_dispatch_swiglu(M, N, K, E, *, warmup=10, rep=50):
    """Compare Gluon dispatch+swiglu to the upstream triton_kernels MoE
    GEMM with the same fused-activation epilogue."""
    from tokenspeed_kernel.ops.moe.gluon import gluon_bf16_dispatch_swiglu

    md, M_padded = _build_ragged(M, E, block_m=128)
    x = _bf16_tensor(M_padded, K)
    w = _bf16_tensor(E, K, 2 * N)

    def gluon_call():
        # Defaults trigger the SwiGLU-aware autotuner (BLOCK_N=64).
        return gluon_bf16_dispatch_swiglu(
            x,
            w,
            bias=None,
            a_ragged_metadata=md,
            gather_indx=None,
            swiglu_alpha=1.0,
            swiglu_limit=0.0,
        )

    def baseline_call():
        # The most apples-to-apples baseline is the upstream
        # triton_kernels.matmul which is what we'd otherwise dispatch to.
        # We feed the SAME ragged metadata so per-expert fan-out is
        # identical.
        from triton_kernels.matmul import (
            FusedActivation,
            PrecisionConfig,
            matmul,
        )
        from triton_kernels.specialize import FnSpecs
        from triton_kernels.swiglu import swiglu_fn

        act = FusedActivation(
            FnSpecs("swiglu", swiglu_fn, ("alpha", "limit"), reduction_n=2),
            (1.0, 0.0),
        )
        return matmul(
            x,
            w,
            None,
            a_ragged_metadata=md,
            fused_activation=act,
            precision_config=PrecisionConfig(),
        )

    try:
        gluon_ms = _bench(gluon_call, warmup=warmup, rep=rep)
    except Exception as e:
        return {"shape": f"M={M_padded},N={N}*2,K={K},E={E}", "error": str(e)}
    try:
        baseline_ms = _bench(baseline_call, warmup=warmup, rep=rep)
    except Exception as e:
        baseline_ms = float("nan")

    flops = 2.0 * M_padded * (2 * N) * K + 2.0 * M_padded * N
    return {
        "shape": f"M={M_padded},N={N}*2,K={K},E={E}",
        "gluon_ms": gluon_ms,
        "baseline_ms": baseline_ms,
        "gluon_tflops": flops / (gluon_ms * 1e-3) / 1e12,
        "baseline_tflops": flops / (baseline_ms * 1e-3) / 1e12,
        "speedup": (
            baseline_ms / gluon_ms if baseline_ms == baseline_ms else float("nan")
        ),
    }


# ---------------------------------------------------------------------------
# Kernel 3: 2nd GEMM + scatter combine
# ---------------------------------------------------------------------------


def bench_combine(M, N, K, E, *, warmup=10, rep=50):
    """Compare Gluon scatter-combine to the upstream triton_kernels MoE
    GEMM with the same scatter epilogue."""
    from tokenspeed_kernel.ops.moe.gluon import gluon_bf16_combine

    md, M_padded = _build_ragged(M, E, block_m=128)
    x = _bf16_tensor(M_padded, K)
    w = _bf16_tensor(E, K, N)
    scatter_indx = type(
        "ScatterIndx",
        (),
        {"dst_indx": torch.arange(M_padded, device=DEVICE, dtype=torch.int32)},
    )()

    def gluon_call():
        return gluon_bf16_combine(
            x,
            w,
            bias=None,
            a_ragged_metadata=md,
            scatter_indx=scatter_indx,
            gate_scal=None,
            n_tokens=M_padded,
            n_expts_act=1,
            block_m=128,
            block_n=128,
            block_k=64,
        )

    # The upstream API takes ``scatter_indx`` as a plain tensor.
    upstream_scatter = scatter_indx.dst_indx

    def baseline_call():
        from triton_kernels.matmul import PrecisionConfig, matmul

        return matmul(
            x,
            w,
            None,
            a_ragged_metadata=md,
            scatter_indx=upstream_scatter,
            precision_config=PrecisionConfig(),
        )

    try:
        gluon_ms = _bench(gluon_call, warmup=warmup, rep=rep)
    except Exception as e:
        return {"shape": f"M={M_padded},N={N},K={K},E={E}", "error": str(e)}
    try:
        baseline_ms = _bench(baseline_call, warmup=warmup, rep=rep)
    except Exception as e:
        baseline_ms = float("nan")

    flops = 2.0 * M_padded * N * K
    return {
        "shape": f"M={M_padded},N={N},K={K},E={E}",
        "gluon_ms": gluon_ms,
        "baseline_ms": baseline_ms,
        "gluon_tflops": flops / (gluon_ms * 1e-3) / 1e12,
        "baseline_tflops": flops / (baseline_ms * 1e-3) / 1e12,
        "speedup": (
            baseline_ms / gluon_ms if baseline_ms == baseline_ms else float("nan")
        ),
    }


# ---------------------------------------------------------------------------
# Static profile dump
# ---------------------------------------------------------------------------


def dump_static_profiles():
    """Run each Gluon kernel once and dump its AMDGCN GPR profile."""
    from tokenspeed_kernel.ops.moe.gluon import (
        _pipelined_moe_kernel,
        gluon_bf16_combine,
        gluon_bf16_dispatch_swiglu,
        gluon_bf16_gating_gemm,
        static_profile,
    )

    print("\n=== Static GPR / spill profile ===")
    M, N, K, E = 512, 256, 256, 4
    x = _bf16_tensor(M, K)
    w_dense = _bf16_tensor(K, N)
    md, M_padded = _build_ragged(M, E)
    x_pad = _bf16_tensor(M_padded, K)
    w_moe = _bf16_tensor(E, K, N)
    w_moe_2x = _bf16_tensor(E, K, 2 * N)
    scatter = type(
        "ScatterIndx",
        (),
        {"dst_indx": torch.arange(M_padded, device=DEVICE, dtype=torch.int32)},
    )()

    gluon_bf16_gating_gemm(x, w_dense, block_m=128, block_n=128, block_k=64)
    gluon_bf16_dispatch_swiglu(
        x_pad,
        w_moe_2x,
        bias=None,
        a_ragged_metadata=md,
        gather_indx=None,
        swiglu_alpha=1.0,
        swiglu_limit=0.0,
        block_m=128,
        block_n=128,
        block_k=64,
    )
    gluon_bf16_combine(
        x_pad,
        w_moe,
        bias=None,
        a_ragged_metadata=md,
        scatter_indx=scatter,
        gate_scal=None,
        n_tokens=M_padded,
        n_expts_act=1,
        block_m=128,
        block_n=128,
        block_k=64,
    )

    device = torch.cuda.current_device()
    device_cache = _pipelined_moe_kernel.device_caches.get(device)
    if not device_cache:
        print("  no Gluon kernel cached -- did the launches fail?")
        return
    kernel_cache = device_cache[0]
    for sig, compiled in kernel_cache.items():
        prof = static_profile(compiled)
        print(
            f"  - sgpr={prof['sgpr_count']} (spill={prof['sgpr_spill_count']}) "
            f"vgpr={prof['vgpr_count']} (spill={prof['vgpr_spill_count']}) "
            f"scratch={prof['ScratchSize']} occupancy={prof['Occupancy']}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


_GATING_SHAPES = [
    (512, 1024, 2880),
    (1024, 1024, 2880),
    (4096, 1024, 2880),
]
_DISPATCH_SHAPES = [
    # (M_dispatched, N_intermediate, K_hidden, E)
    (512, 1024, 2880, 4),
    (1024, 1024, 2880, 4),
]
_COMBINE_SHAPES = [
    (512, 2880, 1024, 4),
    (1024, 2880, 1024, 4),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument("--no-static", action="store_true")
    args = parser.parse_args()

    print("=" * 80)
    print("Kernel 1: bf16 gating GEMM (Gluon vs upstream triton_kernels)")
    print("=" * 80)
    for M, N, K in _GATING_SHAPES:
        r = bench_gating_gemm(M, N, K, warmup=args.warmup, rep=args.rep)
        print(
            f"  {r['shape']:30s} "
            f"gluon={r['gluon_tflops']:6.1f} TFLOPs ({r['gluon_ms']:.3f} ms)  "
            f"triton_kernels={r['baseline_tflops']:6.1f} TFLOPs ({r['baseline_ms']:.3f} ms)  "
            f"torch.mm={r['torch_tflops']:6.1f} TFLOPs ({r['torch_ms']:.3f} ms)  "
            f"speedup_vs_triton={r['speedup']:.2f}x"
        )

    print()
    print("=" * 80)
    print("Kernel 2: dispatch + 1st GEMM + SwiGLU")
    print("=" * 80)
    for M, N, K, E in _DISPATCH_SHAPES:
        r = bench_dispatch_swiglu(M, N, K, E, warmup=args.warmup, rep=args.rep)
        print(
            f"  {r['shape']:30s} "
            f"gluon={r['gluon_tflops']:6.1f} TFLOPs ({r['gluon_ms']:.3f} ms)  "
            f"baseline={r['baseline_tflops']:6.1f} TFLOPs ({r['baseline_ms']:.3f} ms)  "
            f"speedup={r['speedup']:.2f}x"
        )

    print()
    print("=" * 80)
    print("Kernel 3: 2nd GEMM + scatter combine")
    print("=" * 80)
    for M, N, K, E in _COMBINE_SHAPES:
        r = bench_combine(M, N, K, E, warmup=args.warmup, rep=args.rep)
        print(
            f"  {r['shape']:30s} "
            f"gluon={r['gluon_tflops']:6.1f} TFLOPs ({r['gluon_ms']:.3f} ms)  "
            f"baseline={r['baseline_tflops']:6.1f} TFLOPs ({r['baseline_ms']:.3f} ms)  "
            f"speedup={r['speedup']:.2f}x"
        )

    if not args.no_static:
        dump_static_profiles()


if __name__ == "__main__":
    main()
