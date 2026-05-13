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

The shapes are derived from the gpt-oss-120b (mxfp4 weight + fp8 act)
HF model card (https://huggingface.co/amd/gpt-oss-120b-w-mxfp4-a-fp8):

  hidden_size            H  = 2880   ( = K for both MoE GEMMs )
  intermediate_size      I  = 2880   ( = N for the 1st GEMM,
                                      = K for the 2nd GEMM       )
  num_local_experts      E  = 128
  num_experts_per_tok    K  = 4

For each batch_size ``B`` (token count -- 1/32/64 cover decode,
1024/4096/8192 cover prefill / chunked prefill) we compare the three
Gluon kernels against their upstream baselines:

  * Kernel 1 (router gating): GEMM ``(B, H) x (H, E)``  vs upstream
    ``triton_kernels.matmul`` and ``torch.matmul``.
  * Kernel 2 (dispatch + 1st GEMM + SwiGLU): per-expert GEMM
    ``(M_d, H) x (E, H, 2I) -> SwiGLU -> (M_d, I)`` vs upstream
    ``triton_kernels.matmul`` with ``FusedActivation(swiglu_fn)``.
  * Kernel 3 (2nd GEMM + scatter combine): per-expert GEMM
    ``(M_d, I) x (E, I, H)`` followed by scatter combine, vs upstream
    ``triton_kernels.matmul`` with ``scatter_indx``.

For decode batch sizes only ``min(B*topk, E)`` experts are active, so
the ragged metadata is built with that exact number of active experts;
prefill batch sizes saturate all ``E`` experts.

The script also dumps the static GPR / spill profile per Gluon kernel
(``static_profile``) so any regression that introduces sgpr/vgpr spills
fails loudly.
"""

from __future__ import annotations

import argparse
import time

import tokenspeed_kernel  # noqa: F401  (import order: kernel before torch)
import torch

DEVICE = "cuda"

# gpt-oss-120b architecture (per HF model card ``config.json``).
GPTOSS_HIDDEN = 2880
GPTOSS_INTERMEDIATE = 2880
GPTOSS_NUM_EXPERTS = 128
GPTOSS_TOPK = 4


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


def _bf16_tensor(*shape):
    return torch.randn(*shape, device=DEVICE, dtype=torch.bfloat16) * 0.05


# ---------------------------------------------------------------------------
# Ragged metadata helpers
# ---------------------------------------------------------------------------


def _round_up(x: int, m: int) -> int:
    return ((x + m - 1) // m) * m


def _moe_active_layout(
    B: int,
    *,
    E: int = GPTOSS_NUM_EXPERTS,
    topk: int = GPTOSS_TOPK,
    block_m: int = 64,
):
    """Return ``(n_active_experts, per_expert_M, M_padded)`` consistent
    with how the ragged metadata is constructed inside the gpt-oss-120b
    inference path.

    For *decode* (small B) only ``min(B*topk, E)`` experts ever see a
    token, so we don't pad zero-token experts -- this matches what the
    upstream triton_kernels matmul actually sees in production.

    For *prefill* (large B) we assume all ``E`` experts are active and
    each gets ``ceil(B*topk / E)`` tokens, rounded up to ``block_m`` so
    the kernel can index them as full tiles.
    """
    M_dispatched = B * topk
    n_active = min(M_dispatched, E)
    per_expert = max(
        block_m,
        _round_up((M_dispatched + n_active - 1) // n_active, block_m),
    )
    M_padded = per_expert * n_active
    return n_active, per_expert, M_padded


def _build_ragged(B, *, E=GPTOSS_NUM_EXPERTS, topk=GPTOSS_TOPK, block_m=64):
    from triton_kernels.tensor import make_ragged_tensor_metadata

    n_active, per_expert, M_padded = _moe_active_layout(
        B, E=E, topk=topk, block_m=block_m
    )
    counts = torch.zeros((E,), device=DEVICE, dtype=torch.int32)
    counts[:n_active] = per_expert
    md = make_ragged_tensor_metadata(counts, M_padded)
    return md, M_padded, n_active, per_expert


# ---------------------------------------------------------------------------
# Kernel 1: bf16 gating GEMM   (B, H) x (H, E) -> (B, E)
# ---------------------------------------------------------------------------


def bench_gating_gemm(B, *, H=GPTOSS_HIDDEN, E=GPTOSS_NUM_EXPERTS, warmup=10, rep=50):
    from tokenspeed_kernel.ops.moe.gluon import gluon_bf16_gating_gemm

    x = _bf16_tensor(B, H)
    w = _bf16_tensor(H, E)

    def gluon_call():
        return gluon_bf16_gating_gemm(x, w)

    from triton_kernels.matmul import PrecisionConfig, matmul

    def triton_call():
        return matmul(x, w, None, precision_config=PrecisionConfig())

    gluon_ms = _bench(gluon_call, warmup=warmup, rep=rep)
    try:
        triton_ms = _bench(triton_call, warmup=warmup, rep=rep)
    except Exception:
        triton_ms = float("nan")
    torch_ms = _bench(lambda: torch.matmul(x, w), warmup=warmup, rep=rep)

    flops = 2.0 * B * E * H
    return {
        "shape": f"B={B},H={H},E={E}",
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
# (M_d, H) x (E, H, 2I) -> SwiGLU -> (M_d, I)
# ---------------------------------------------------------------------------


def bench_dispatch_swiglu(
    B,
    *,
    H=GPTOSS_HIDDEN,
    I=GPTOSS_INTERMEDIATE,
    E=GPTOSS_NUM_EXPERTS,
    topk=GPTOSS_TOPK,
    warmup=10,
    rep=50,
):
    from tokenspeed_kernel.ops.moe.gluon import gluon_bf16_dispatch_swiglu

    md, M_padded, n_active, per_expert = _build_ragged(B, E=E, topk=topk, block_m=64)
    x = _bf16_tensor(M_padded, H)
    w = _bf16_tensor(E, H, 2 * I)

    def gluon_call():
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
        return {
            "shape": (f"B={B},M_d={M_padded},I={I},H={H},E_act={n_active}/{E}"),
            "error": str(e),
        }
    try:
        baseline_ms = _bench(baseline_call, warmup=warmup, rep=rep)
    except Exception:
        baseline_ms = float("nan")

    flops = 2.0 * M_padded * (2 * I) * H + 2.0 * M_padded * I  # GEMM + SwiGLU
    return {
        "shape": f"B={B},M_d={M_padded},I={I},H={H},E_act={n_active}/{E}",
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
# (M_d, I) x (E, I, H) -> scatter -> (B, H)
# ---------------------------------------------------------------------------


def bench_combine(
    B,
    *,
    H=GPTOSS_HIDDEN,
    I=GPTOSS_INTERMEDIATE,
    E=GPTOSS_NUM_EXPERTS,
    topk=GPTOSS_TOPK,
    warmup=10,
    rep=50,
):
    from tokenspeed_kernel.ops.moe.gluon import gluon_bf16_combine

    md, M_padded, n_active, per_expert = _build_ragged(B, E=E, topk=topk, block_m=64)
    x = _bf16_tensor(M_padded, I)
    w = _bf16_tensor(E, I, H)
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
        )

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
        return {
            "shape": (f"B={B},M_d={M_padded},H={H},I={I},E_act={n_active}/{E}"),
            "error": str(e),
        }
    try:
        baseline_ms = _bench(baseline_call, warmup=warmup, rep=rep)
    except Exception:
        baseline_ms = float("nan")

    flops = 2.0 * M_padded * H * I
    return {
        "shape": f"B={B},M_d={M_padded},H={H},I={I},E_act={n_active}/{E}",
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
        _pipelined_moe_kernel_scaled as _pipelined_moe_kernel,
        gluon_bf16_combine,
        gluon_bf16_dispatch_swiglu,
        gluon_bf16_gating_gemm,
        static_profile,
    )

    print("\n=== Static GPR / spill profile (no spills allowed) ===")
    # One representative shape per phase: decode B=32, prefill B=1024.
    for B in (32, 1024):
        x_dense = _bf16_tensor(B, GPTOSS_HIDDEN)
        w_dense = _bf16_tensor(GPTOSS_HIDDEN, GPTOSS_NUM_EXPERTS)
        gluon_bf16_gating_gemm(x_dense, w_dense)

        md, M_padded, *_ = _build_ragged(B, block_m=64)
        x_pad = _bf16_tensor(M_padded, GPTOSS_HIDDEN)
        w_moe_2x = _bf16_tensor(
            GPTOSS_NUM_EXPERTS, GPTOSS_HIDDEN, 2 * GPTOSS_INTERMEDIATE
        )
        gluon_bf16_dispatch_swiglu(
            x_pad,
            w_moe_2x,
            bias=None,
            a_ragged_metadata=md,
            gather_indx=None,
            swiglu_alpha=1.0,
            swiglu_limit=0.0,
        )
        scatter = type(
            "ScatterIndx",
            (),
            {"dst_indx": torch.arange(M_padded, device=DEVICE, dtype=torch.int32)},
        )()
        x_pad_i = _bf16_tensor(M_padded, GPTOSS_INTERMEDIATE)
        w_moe = _bf16_tensor(GPTOSS_NUM_EXPERTS, GPTOSS_INTERMEDIATE, GPTOSS_HIDDEN)
        gluon_bf16_combine(
            x_pad_i,
            w_moe,
            bias=None,
            a_ragged_metadata=md,
            scatter_indx=scatter,
            gate_scal=None,
            n_tokens=M_padded,
            n_expts_act=1,
        )

    # Also exercise the scaled-MFMA kernel at the same shapes so its
    # spill profile shows up alongside the bf16 ones.
    from tokenspeed_kernel.ops.moe.gluon import (
        _pipelined_moe_kernel_scaled,
        gluon_mxfp_combine,
        gluon_mxfp_dispatch_swiglu,
        gluon_mxfp_gating_gemm,
    )

    for B in (32, 1024):
        # mxfp4 x mxfp4 gating
        a_packed, _ = _make_mxfp4(B, GPTOSS_HIDDEN, packed_dim=1)
        w_packed, _ = _make_mxfp4(GPTOSS_HIDDEN, GPTOSS_NUM_EXPERTS, packed_dim=0)
        a_scale, _ = _make_mx_scale(B, GPTOSS_HIDDEN)
        w_scale, _ = _make_mx_scale(GPTOSS_NUM_EXPERTS, GPTOSS_HIDDEN)
        gluon_mxfp_gating_gemm(
            a_packed,
            w_packed,
            w_scale,
            x_scale=a_scale,
            a_format="e2m1",
        )
        # mxfp4 dispatch+swiglu
        md, M_padded, *_ = _build_ragged(B, block_m=64)
        a_pad_packed, _ = _make_mxfp4(M_padded, GPTOSS_HIDDEN, packed_dim=1)
        a_pad_scale, _ = _make_mx_scale(M_padded, GPTOSS_HIDDEN)
        w3_packed_list = []
        w3_scale_list = []
        N_full = 2 * GPTOSS_INTERMEDIATE
        for _ in range(GPTOSS_NUM_EXPERTS):
            wp, _ = _make_mxfp4(GPTOSS_HIDDEN, N_full, packed_dim=0)
            w3_packed_list.append(wp)
            sd, _ = _make_mx_scale(N_full, GPTOSS_HIDDEN)
            w3_scale_list.append(sd)
        w3 = torch.stack(w3_packed_list)
        w3_scale = torch.stack(w3_scale_list)
        gluon_mxfp_dispatch_swiglu(
            a_pad_packed,
            w3,
            w3_scale,
            x_scale=a_pad_scale,
            a_format="e2m1",
            bias=None,
            a_ragged_metadata=md,
            gather_indx=None,
            swiglu_alpha=1.0,
            swiglu_limit=0.0,
        )
        # mxfp4 combine
        a_pad_i, _ = _make_mxfp4(M_padded, GPTOSS_INTERMEDIATE, packed_dim=1)
        a_pad_i_scale, _ = _make_mx_scale(M_padded, GPTOSS_INTERMEDIATE)
        w3_c_list = []
        w3_c_scale_list = []
        for _ in range(GPTOSS_NUM_EXPERTS):
            wp, _ = _make_mxfp4(GPTOSS_INTERMEDIATE, GPTOSS_HIDDEN, packed_dim=0)
            w3_c_list.append(wp)
            sd, _ = _make_mx_scale(GPTOSS_HIDDEN, GPTOSS_INTERMEDIATE)
            w3_c_scale_list.append(sd)
        w3_c = torch.stack(w3_c_list)
        w3_c_scale = torch.stack(w3_c_scale_list)
        scatter = type(
            "S",
            (),
            {"dst_indx": torch.arange(M_padded, device=DEVICE, dtype=torch.int32)},
        )()
        gluon_mxfp_combine(
            a_pad_i,
            w3_c,
            w3_c_scale,
            x_scale=a_pad_i_scale,
            a_format="e2m1",
            bias=None,
            a_ragged_metadata=md,
            scatter_indx=scatter,
            n_tokens=M_padded,
            n_expts_act=1,
        )

    device = torch.cuda.current_device()
    bad = 0

    def _scan(cache_attr, label):
        nonlocal bad
        cache = cache_attr.device_caches.get(device)
        if not cache:
            return
        for _, compiled in cache[0].items():
            prof = static_profile(compiled)
            spill = (
                prof["sgpr_spill_count"]
                + prof["vgpr_spill_count"]
                + prof["ScratchSize"]
            )
            marker = "OK   " if spill == 0 else "SPILL"
            if spill > 0:
                bad += 1
            print(
                f"  [{marker}] {label:8s} sgpr={prof['sgpr_count']:3d} (spill={prof['sgpr_spill_count']}) "
                f"vgpr={prof['vgpr_count']:3d} (spill={prof['vgpr_spill_count']}) "
                f"scratch={prof['ScratchSize']} occupancy={prof['Occupancy']}"
            )

    _scan(_pipelined_moe_kernel, "unified")
    if bad:
        raise SystemExit(f"{bad} kernel(s) reported spill -- aborting")


def _make_mxfp4(M, K, packed_dim, *, device="cuda"):
    """Random mxfp4 tensor; returns ``(packed_uint8, fp32_ref)``."""
    from tokenspeed_triton.tools.mxfp import MXFP4Tensor

    t = MXFP4Tensor(size=(M, K)).random()
    return (
        t.to_packed_tensor(dim=packed_dim).to(device),
        t.to(torch.float32).to(device),
    )


def _make_mx_scale(rows, k_logical, *, device="cuda"):
    """Random e8m0 block scale tensor; returns ``(uint8, fp32_broadcast)``."""
    from tokenspeed_triton.tools.mxfp import MXScaleTensor

    s = MXScaleTensor(size=(rows, k_logical // 32)).random(1 / 32, 32)
    return (
        s.data.to(device),
        s.to(torch.float32).repeat_interleave(32, dim=1).to(device),
    )


def bench_scaled_gating(B: int, warmup: int = 10, rep: int = 50, fmt: str = "e2m1"):
    """Scaled-MFMA gating GEMM ``(B, H) x (H, E)``.

    ``fmt`` selects the A operand format:
        - ``"e2m1"`` : mxfp4 activations + e8m0 block scale.
        - ``"e4m3"`` : fp8 activations + a fixed global scalar.
    The W is always mxfp4 in this version.
    """
    from tokenspeed_kernel.ops.moe.gluon import gluon_mxfp_gating_gemm

    torch.manual_seed(0)
    device = "cuda"
    H, E = GPTOSS_HIDDEN, GPTOSS_NUM_EXPERTS

    if fmt == "e2m1":
        a_packed, _ = _make_mxfp4(B, H, packed_dim=1, device=device)
        a_scale, _ = _make_mx_scale(B, H, device=device)
        a_global = 1.0
    else:
        a_packed = torch.randint(20, 40, (B, H), dtype=torch.uint8, device=device)
        a_scale = None
        a_global = 0.137

    w_packed, _ = _make_mxfp4(H, E, packed_dim=0, device=device)
    w_scale, _ = _make_mx_scale(E, H, device=device)

    def fn():
        gluon_mxfp_gating_gemm(
            a_packed,
            w_packed,
            w_scale,
            x_scale=a_scale,
            a_format=fmt,
            a_global_scale=a_global,
            out_dtype=torch.bfloat16,
        )

    flops = 2.0 * B * E * H
    t_ms = _bench(fn, warmup=warmup, rep=rep)
    label = f"B={B},H={H},E={E},fmt={fmt}/e2m1"
    return label, t_ms, flops / (t_ms * 1e9)


def bench_scaled_dispatch_swiglu(B, warmup=10, rep=50, fmt="e2m1"):
    from tokenspeed_kernel.ops.moe.gluon import gluon_mxfp_dispatch_swiglu

    torch.manual_seed(0)
    device = DEVICE
    H, I, E = GPTOSS_HIDDEN, GPTOSS_INTERMEDIATE, GPTOSS_NUM_EXPERTS
    md, M_d, n_active, _ = _build_ragged(B, block_m=64)
    gather_indx = None
    N_full = 2 * I

    if fmt == "e2m1":
        a_packed, _ = _make_mxfp4(M_d, H, packed_dim=1, device=device)
        a_scale, _ = _make_mx_scale(M_d, H, device=device)
        a_global = 1.0
    else:
        a_packed = torch.randint(20, 40, (M_d, H), dtype=torch.uint8, device=device)
        a_scale = None
        a_global = 0.137

    w_packed_list = []
    w_scale_list = []
    for _ in range(E):
        wp, _ = _make_mxfp4(H, N_full, packed_dim=0, device=device)
        w_packed_list.append(wp)
        sd, _ = _make_mx_scale(N_full, H, device=device)
        w_scale_list.append(sd)
    w3 = torch.stack(w_packed_list)
    w_scale3 = torch.stack(w_scale_list)

    def fn():
        gluon_mxfp_dispatch_swiglu(
            a_packed,
            w3,
            w_scale3,
            x_scale=a_scale,
            a_format=fmt,
            a_global_scale=a_global,
            bias=None,
            a_ragged_metadata=md,
            gather_indx=gather_indx,
            out_dtype=torch.bfloat16,
            swiglu_alpha=1.0,
            swiglu_limit=0.0,
        )

    flops = 2.0 * M_d * N_full * H
    t_ms = _bench(fn, warmup=warmup, rep=rep)
    label = f"B={B},M_d={M_d},I={I},H={H},E_act={n_active}/{E},fmt={fmt}"
    return label, t_ms, flops / (t_ms * 1e9)


def bench_scaled_combine(B, warmup=10, rep=50, fmt="e2m1"):
    from tokenspeed_kernel.ops.moe.gluon import gluon_mxfp_combine

    torch.manual_seed(0)
    device = DEVICE
    H, I, E = GPTOSS_HIDDEN, GPTOSS_INTERMEDIATE, GPTOSS_NUM_EXPERTS
    md, M_d, n_active, _ = _build_ragged(B, block_m=64)

    if fmt == "e2m1":
        a_packed, _ = _make_mxfp4(M_d, I, packed_dim=1, device=device)
        a_scale, _ = _make_mx_scale(M_d, I, device=device)
        a_global = 1.0
    else:
        a_packed = torch.randint(20, 40, (M_d, I), dtype=torch.uint8, device=device)
        a_scale = None
        a_global = 0.137

    w_packed_list = []
    w_scale_list = []
    for _ in range(E):
        wp, _ = _make_mxfp4(I, H, packed_dim=0, device=device)
        w_packed_list.append(wp)
        sd, _ = _make_mx_scale(H, I, device=device)
        w_scale_list.append(sd)
    w3 = torch.stack(w_packed_list)
    w_scale3 = torch.stack(w_scale_list)

    scatter = type(
        "S",
        (),
        {
            "dst_indx": torch.arange(M_d, device=device, dtype=torch.int32),
        },
    )()

    def fn():
        gluon_mxfp_combine(
            a_packed,
            w3,
            w_scale3,
            x_scale=a_scale,
            a_format=fmt,
            a_global_scale=a_global,
            bias=None,
            a_ragged_metadata=md,
            scatter_indx=scatter,
            n_tokens=M_d,
            n_expts_act=1,
            out_dtype=torch.bfloat16,
        )

    flops = 2.0 * M_d * H * I
    t_ms = _bench(fn, warmup=warmup, rep=rep)
    label = f"B={B},M_d={M_d},H={H},I={I},E_act={n_active}/{E},fmt={fmt}"
    return label, t_ms, flops / (t_ms * 1e9)


def _print_scaled_row(label_t_tf):
    label, t_ms, tf = label_t_tf
    print(f"  {label:56s} gluon={tf:7.1f} TFLOPs ({t_ms:.3f} ms)")


def dump_scaled_mfma_autotune():
    """Print the block sizes the autotuner would pick for the future
    scaled-MFMA (mxfp4 weight + fp8 activation) path. Per TASKS.md
    Update 3, BLOCK_K must be a multiple of 128 and >= 128.
    """
    from tokenspeed_kernel.ops.moe.gluon import _autotune_block

    print("\n=== Scaled-MFMA (mxfp4/fp8, instr 16x16x128) autotune preview ===")
    rows = [
        # (label, M, do_swiglu, ragged)
        ("gating  decode  B=1", 1, False, False),
        ("gating  prefill B=8192", 8192, False, False),
        ("dispatch+swiglu B=32", 32 * GPTOSS_TOPK, True, False),
        ("dispatch+swiglu B=8192", 8192 * GPTOSS_TOPK, True, False),
        ("combine        B=32", 32 * GPTOSS_TOPK, False, True),
        ("combine        B=8192", 8192 * GPTOSS_TOPK, False, True),
    ]
    for label, M, do_swiglu, ragged in rows:
        bm, bn, bk, nw = _autotune_block(
            M,
            GPTOSS_INTERMEDIATE,
            GPTOSS_HIDDEN,
            do_swiglu=do_swiglu,
            ragged=ragged,
            scaled_mfma=True,
        )
        assert (
            bk >= 128 and bk % 128 == 0
        ), f"scaled MFMA autotune returned BLOCK_K={bk} for {label}"
        print(f"  {label:30s} BM={bm:3d} BN={bn:3d} BK={bk:3d} NW={nw}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


# gpt-oss-120b deployment scenarios (per ``task-progress-1.md``):
#   decode  : single-token / very small batch -> B in {1, 32, 64}
#   prefill : full / chunked prefill          -> B in {1024, 4096, 8192}
GPTOSS_BATCHES = (1, 32, 64, 1024, 4096, 8192)


def _print_row(r):
    if "error" in r:
        print(f"  {r['shape']:55s}  ERROR: {r['error'][:80]}")
        return
    extras = ""
    if "torch_tflops" in r:
        extras = f"torch.mm={r['torch_tflops']:7.1f} TFLOPs ({r['torch_ms']:.3f} ms)  "
    print(
        f"  {r['shape']:55s} "
        f"gluon={r['gluon_tflops']:7.1f} TFLOPs ({r['gluon_ms']:.3f} ms)  "
        f"baseline={r['baseline_tflops']:7.1f} TFLOPs ({r['baseline_ms']:.3f} ms)  "
        f"{extras}"
        f"speedup={r['speedup']:.2f}x"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=list(GPTOSS_BATCHES),
        help=(
            "Token-batch dimensions to sweep " "(default: gpt-oss-120b decode+prefill)."
        ),
    )
    parser.add_argument("--no-static", action="store_true")
    args = parser.parse_args()

    print("=" * 110)
    print(
        f"gpt-oss-120b moe shapes: H={GPTOSS_HIDDEN}, I={GPTOSS_INTERMEDIATE}, "
        f"E={GPTOSS_NUM_EXPERTS}, topk={GPTOSS_TOPK}"
    )
    print("=" * 110)

    print("\nKernel 1: bf16 gating GEMM  (B, H) x (H, E)")
    print("-" * 110)
    for B in args.batch_sizes:
        _print_row(bench_gating_gemm(B, warmup=args.warmup, rep=args.rep))

    print("\nKernel 2: dispatch + 1st GEMM + SwiGLU  (M_d, H) x (E, H, 2I)")
    print("-" * 110)
    for B in args.batch_sizes:
        _print_row(bench_dispatch_swiglu(B, warmup=args.warmup, rep=args.rep))

    print("\nKernel 3: 2nd GEMM + scatter combine  (M_d, I) x (E, I, H)")
    print("-" * 110)
    for B in args.batch_sizes:
        _print_row(bench_combine(B, warmup=args.warmup, rep=args.rep))

    # ------ Scaled MFMA variants (mxfp4 / fp8) ------
    print("\n=== Scaled MFMA variants (instr 16x16x128) ===")
    print("\nKernel 1s: mxfp4 x mxfp4 gating GEMM  (B, H) x (H, E)")
    print("-" * 110)
    for B in args.batch_sizes:
        _print_scaled_row(
            bench_scaled_gating(B, warmup=args.warmup, rep=args.rep, fmt="e2m1")
        )

    print("\nKernel 1s.fp8: fp8 x mxfp4 gating GEMM  (B, H) x (H, E)")
    print("-" * 110)
    for B in args.batch_sizes:
        _print_scaled_row(
            bench_scaled_gating(B, warmup=args.warmup, rep=args.rep, fmt="e4m3")
        )

    print("\nKernel 2s: mxfp4 dispatch + 1st GEMM + SwiGLU")
    print("-" * 110)
    for B in args.batch_sizes:
        _print_scaled_row(
            bench_scaled_dispatch_swiglu(
                B, warmup=args.warmup, rep=args.rep, fmt="e2m1"
            )
        )

    print("\nKernel 2s.fp8: fp8 dispatch + 1st GEMM + SwiGLU")
    print("-" * 110)
    for B in args.batch_sizes:
        _print_scaled_row(
            bench_scaled_dispatch_swiglu(
                B, warmup=args.warmup, rep=args.rep, fmt="e4m3"
            )
        )

    print("\nKernel 3s: mxfp4 2nd GEMM + scatter combine")
    print("-" * 110)
    for B in args.batch_sizes:
        _print_scaled_row(
            bench_scaled_combine(B, warmup=args.warmup, rep=args.rep, fmt="e2m1")
        )

    print("\nKernel 3s.fp8: fp8 2nd GEMM + scatter combine")
    print("-" * 110)
    for B in args.batch_sizes:
        _print_scaled_row(
            bench_scaled_combine(B, warmup=args.warmup, rep=args.rep, fmt="e4m3")
        )

    if not args.no_static:
        dump_static_profiles()
        dump_scaled_mfma_autotune()


if __name__ == "__main__":
    main()
