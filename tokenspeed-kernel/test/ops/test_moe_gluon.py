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

"""Correctness + spill tests for the MI355 Gluon MoE scaled-MFMA kernels.

The Gluon MoE kernel only supports the mxfp4 / fp8 scaled-MFMA path
(``e2m1`` x ``e2m1`` and ``e4m3`` / ``e5m2`` x ``e2m1``); plain bf16 / fp16
inputs are routed to ``triton_kernels.matmul`` via the registered
``_gluon_mxfp_ragged_matmul`` adapter.

For each kernel we also assert (via ``static_profile``) that AMDGCN reports
*zero* sgpr / vgpr spills.
"""

from __future__ import annotations

import pytest

# IMPORTANT: tokenspeed_kernel must be imported before torch on this docker
# image to avoid an ABI segfault between the system torch and the bundled
# tokenspeed_triton C extension.
import tokenspeed_kernel  # noqa: F401  (must be first)
import torch
from tokenspeed_kernel.platform import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform().is_cdna4,
    reason="Gluon MoE kernel is implemented for CDNA4 (gfx950 / MI355) only.",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_ragged(M: int, E: int, *, block_m: int = 128, device: str = "cuda"):
    """Block-aligned per-expert routing fixture.

    Returns ``(metadata, gather_indx, counts, M_padded)`` where every expert
    owns a multiple of ``block_m`` rows.
    """
    from triton_kernels.tensor import make_ragged_tensor_metadata

    per_expert = max(block_m, (M // E) // block_m * block_m)
    M_padded = per_expert * E
    counts = torch.full((E,), per_expert, device=device, dtype=torch.int32)
    md = make_ragged_tensor_metadata(counts, M_padded)
    gather_indx = type(
        "GatherIndx",
        (),
        {"src_indx": torch.arange(M_padded, device=device, dtype=torch.int32)},
    )()
    return md, gather_indx, counts, M_padded


# ---------------------------------------------------------------------------
# Scaled MFMA / BLOCK_K constraint  (TASKS.md Update 3)
# ---------------------------------------------------------------------------
#
# Per ``TASKS.md`` Update 3, the scaled MFMA op on CDNA4 has instruction
# shape ``[16, 16, 128]``, so the launcher's ``BLOCK_K`` must be a multiple
# of 128 and ``>= 128``. These tests pin the autotuner contract so the
# constraint can't silently regress.

GPTOSS_H = 2880
GPTOSS_I = 2880
GPTOSS_E = 128
GPTOSS_TOPK = 4


@pytest.mark.parametrize(
    "M,N,K,do_swiglu,ragged",
    [
        (32, 128, 2880, False, False),  # decode gating GEMM
        (8192, 2880, 2880, True, False),  # prefill dispatch+swiglu
        (8192, 2880, 2880, False, True),  # prefill ragged combine
        (32768, 2880, 2880, True, False),  # very large prefill
    ],
)
def test_autotune_scaled_mfma_block_k(M, N, K, do_swiglu, ragged):
    """``_autotune_block`` must enforce CDNA4's 16x16x128 scaled MFMA shape
    on ``BLOCK_K`` for every supported (M, N, K, do_swiglu, ragged) tuple."""
    from tokenspeed_kernel.ops.moe.gluon import _autotune_block

    bm, bn, bk, nw = _autotune_block(M, N, K, do_swiglu=do_swiglu, ragged=ragged)
    assert bk >= 128, f"scaled MFMA needs BLOCK_K >= 128, got {bk}"
    assert bk % 128 == 0, f"scaled MFMA needs BLOCK_K % 128 == 0, got {bk}"
    assert bm % 16 == 0, f"BLOCK_M must be a multiple of MFMA M (16), got {bm}"


def test_launcher_rejects_bad_block_k_for_scaled_mfma():
    """The launcher must refuse a sub-128 ``BLOCK_K`` since the kernel
    only supports scaled MFMA (16x16x128) -- catches future mis-wirings.
    """
    from tokenspeed_kernel.ops.moe.gluon import _launch_kernel

    # Inputs are mxfp4-packed uint8 (the only format the launcher accepts).
    x = torch.zeros((128, 64), device="cuda", dtype=torch.uint8)
    w = torch.zeros((64, 128), device="cuda", dtype=torch.uint8)
    w_scale = torch.zeros((128, 64 // 32), device="cuda", dtype=torch.uint8)
    x_scale = torch.zeros((128, 64 // 32), device="cuda", dtype=torch.uint8)
    y = torch.empty((128, 128), device="cuda", dtype=torch.bfloat16)
    with pytest.raises(AssertionError, match=r"BLOCK_K"):
        _launch_kernel(
            x,
            w,
            y=y,
            bias=None,
            gather_indx=None,
            scatter_indx=None,
            gate_scal=None,
            a_ragged_metadata=None,
            swiglu=None,
            out_block_n=64,
            block_m=64,
            block_n=64,
            block_k=64,
            num_warps=4,
            num_buffers=2,
            x_format="e2m1",
            w_format="e2m1",
            x_scale=x_scale,
            w_scale=w_scale,
        )


# ---------------------------------------------------------------------------
# Scaled MFMA (mxfp4 / fp8) correctness + spills  (TASKS.md Update 4)
# ---------------------------------------------------------------------------
#
# These tests cover the three dtype combinations the user asked for:
#
#   1. ``e2m1`` x ``e2m1``      (A: mxfp4 + block scale, W: mxfp4 + block scale)
#   2. ``e4m3`` x ``e2m1``      (A: fp8 e4m3 + global scale, W: mxfp4 + block scale)
#   3. ``e5m2`` x ``e2m1``      (A: fp8 e5m2 + global scale, W: mxfp4 + block scale)
#
# Reference is computed in fp32 from the unpacked operands and scales.


def _mxfp4_pair(M: int, K: int, packed_dim: int, *, device="cuda"):
    """Return ``(uint8 packed tensor, fp32 reference tensor)``."""
    from tokenspeed_triton.tools.mxfp import MXFP4Tensor

    t = MXFP4Tensor(size=(M, K)).random()
    return (
        t.to_packed_tensor(dim=packed_dim).to(device),
        t.to(torch.float32).to(device),
    )


def _mx_scale_pair(rows: int, k_logical: int, *, device="cuda"):
    """Return ``(uint8 e8m0 tensor [rows, K//32], fp32 broadcast [rows, K])``."""
    from tokenspeed_triton.tools.mxfp import MXScaleTensor

    s = MXScaleTensor(size=(rows, k_logical // 32)).random(1 / 32, 32)
    return (
        s.data.to(device),
        s.to(torch.float32).repeat_interleave(32, dim=1).to(device),
    )


def _fp8_pair(M: int, K: int, fmt: str, *, device="cuda"):
    """Return ``(uint8 storage, fp32 reference)`` for fp8 e4m3 / e5m2."""
    u = torch.randint(20, 40, (M, K), dtype=torch.uint8).to(device)
    view = torch.float8_e4m3fn if fmt == "e4m3" else torch.float8_e5m2
    return u, u.view(view).to(torch.float32)


@pytest.mark.parametrize("fmt", ["e2m1", "e4m3"])
def test_mxfp_ragged_combine(fmt):
    """Per-expert combine with ragged metadata + scaled MFMA."""
    from tokenspeed_kernel.ops.moe.gluon import gluon_mxfp_combine
    from triton_kernels.tensor import make_ragged_tensor_metadata

    torch.manual_seed(0)
    device = "cuda"
    E = 2
    per_exp = 16
    M = E * per_exp
    K = 128
    N = 64

    if fmt == "e2m1":
        a_packed, a_fp32 = _mxfp4_pair(M, K, packed_dim=1)
        a_scale, a_scale_mk = _mx_scale_pair(M, K)
        a_global = 1.0
    else:
        a_packed, a_fp32 = _fp8_pair(M, K, fmt)
        a_scale = None
        a_scale_mk = None
        a_global = 0.21

    w_e_packed, w_e_fp32, w_scale_e, w_scale_e_nk = [], [], [], []
    for _ in range(E):
        wp, wf = _mxfp4_pair(K, N, packed_dim=0)
        w_e_packed.append(wp)
        w_e_fp32.append(wf)
        sd, sr = _mx_scale_pair(N, K)
        w_scale_e.append(sd)
        w_scale_e_nk.append(sr)
    w3 = torch.stack(w_e_packed)
    w_scale3 = torch.stack(w_scale_e)

    counts = torch.full((E,), per_exp, device=device, dtype=torch.int32)
    md = make_ragged_tensor_metadata(counts, M)

    y_ref = torch.zeros(M, N, dtype=torch.float32, device=device)
    for e in range(E):
        a_chunk = a_fp32[e * per_exp : (e + 1) * per_exp]
        w_scale_kn = w_scale_e_nk[e].T.contiguous()
        if fmt == "e2m1":
            a_scale_chunk = a_scale_mk[e * per_exp : (e + 1) * per_exp]
            y_ref[e * per_exp : (e + 1) * per_exp] = (a_chunk * a_scale_chunk) @ (
                w_e_fp32[e] * w_scale_kn
            )
        else:
            y_ref[e * per_exp : (e + 1) * per_exp] = a_global * (
                a_chunk @ (w_e_fp32[e] * w_scale_kn)
            )

    y = gluon_mxfp_combine(
        a_packed,
        w3,
        w_scale3,
        x_scale=a_scale,
        x_format=fmt,
        x_global_scale=a_global,
        bias=None,
        a_ragged_metadata=md,
        scatter_indx=None,
        n_tokens=M,
        n_expts_act=1,
        out_dtype=torch.float32,
        block_m=16,
        block_n=64,
        block_k=128,
        num_warps=4,
    )
    rel = (y - y_ref).abs().max().item() / max(1.0, y_ref.abs().max().item())
    assert rel < 5e-2, f"{fmt} combine rel_max={rel} too large"


@pytest.mark.parametrize("fmt", ["e2m1", "e4m3"])
def test_mxfp_dispatch_swiglu(fmt):
    """Per-expert dispatch + 1st GEMM + fused SwiGLU on scaled MFMA path."""
    from tokenspeed_kernel.ops.moe.gluon import gluon_mxfp_dispatch_swiglu
    from triton_kernels.tensor import make_ragged_tensor_metadata

    torch.manual_seed(0)
    device = "cuda"
    E = 2
    per_exp = 16
    M = E * per_exp
    K = 128
    N_full = 128  # gate || linear; output will be N_full // 2
    a_global = 1.0 if fmt == "e2m1" else 0.21

    if fmt == "e2m1":
        a_packed, a_fp32 = _mxfp4_pair(M, K, packed_dim=1)
        a_scale, a_scale_mk = _mx_scale_pair(M, K)
    else:
        a_packed, a_fp32 = _fp8_pair(M, K, fmt)
        a_scale = None
        a_scale_mk = None

    w_e_packed, w_e_fp32, w_scale_e, w_scale_e_nk = [], [], [], []
    for _ in range(E):
        wp, wf = _mxfp4_pair(K, N_full, packed_dim=0)
        w_e_packed.append(wp)
        w_e_fp32.append(wf)
        sd, sr = _mx_scale_pair(N_full, K)
        w_scale_e.append(sd)
        w_scale_e_nk.append(sr)
    w3 = torch.stack(w_e_packed)
    w_scale3 = torch.stack(w_scale_e)

    counts = torch.full((E,), per_exp, device=device, dtype=torch.int32)
    md = make_ragged_tensor_metadata(counts, M)

    y_ref = torch.zeros(M, N_full // 2, dtype=torch.float32, device=device)
    for e in range(E):
        a_chunk = a_fp32[e * per_exp : (e + 1) * per_exp]
        w_scale_kn = w_scale_e_nk[e].T.contiguous()
        if fmt == "e2m1":
            a_scale_chunk = a_scale_mk[e * per_exp : (e + 1) * per_exp]
            acc = (a_chunk * a_scale_chunk) @ (w_e_fp32[e] * w_scale_kn)
        else:
            acc = a_global * (a_chunk @ (w_e_fp32[e] * w_scale_kn))
        gate = acc[:, ::2]
        linear = acc[:, 1::2]
        s = gate / (1.0 + torch.exp(-gate))
        y_ref[e * per_exp : (e + 1) * per_exp] = s * (linear + 1.0)

    y = gluon_mxfp_dispatch_swiglu(
        a_packed,
        w3,
        w_scale3,
        x_scale=a_scale,
        x_format=fmt,
        x_global_scale=a_global,
        bias=None,
        a_ragged_metadata=md,
        gather_indx=None,
        swiglu_alpha=1.0,
        swiglu_limit=0.0,
        out_dtype=torch.float32,
        block_m=16,
        block_n=128,
        block_k=128,
        num_warps=4,
    )
    rel = (y - y_ref).abs().max().item() / max(1.0, y_ref.abs().max().item())
    assert rel < 5e-2, f"{fmt} swiglu rel_max={rel} too large"


# ---------------------------------------------------------------------------
# P0-1: fused static-FP8 quant of the SwiGLU output (mirrors AITER's
# ``_compute_static_fp8_quant`` epilogue on ``_moe_gemm_a8w4``).
# ---------------------------------------------------------------------------
#
# When ``out_quant_scale`` is provided to ``gluon_mxfp_dispatch_swiglu``,
# the kernel epilogue divides the post-SwiGLU fp32 output by the scale and
# casts to ``torch.float8_e4m3fn`` directly, eliminating the standalone
# fp32->fp8 downcast kernel between GEMM1 and GEMM2 in the W4A8 path.
#
# Reference: re-run the SAME kernel with ``out_dtype=float32`` (no quant)
# and do the divide-then-cast host-side. Both paths share the same fp32
# accumulation up to the epilogue, so the fused output must be
# bit-identical to ``(y_fp32 / scale).to(float8_e4m3fn)`` modulo the order
# of the final saturating cast.


@pytest.mark.parametrize("fmt", ["e2m1", "e4m3"])
@pytest.mark.parametrize("scale_value", [0.137, 1.0])
def test_mxfp_dispatch_swiglu_fused_fp8_quant(fmt, scale_value):
    """``out_quant_scale`` fuses GEMM1 epilogue's fp32->fp8 quant into the
    SwiGLU kernel. Compare against the same kernel with ``out_dtype=fp32``
    + host-side divide-then-fp8-cast (which goes through the same fp32
    epilogue values)."""
    from tokenspeed_kernel.ops.moe.gluon import gluon_mxfp_dispatch_swiglu
    from triton_kernels.tensor import make_ragged_tensor_metadata

    torch.manual_seed(0)
    device = "cuda"
    E = 2
    per_exp = 16
    M = E * per_exp
    K = 128
    N_full = 128
    a_global = 1.0 if fmt == "e2m1" else 0.21

    if fmt == "e2m1":
        a_packed, _ = _mxfp4_pair(M, K, packed_dim=1)
        a_scale, _ = _mx_scale_pair(M, K)
    else:
        a_packed, _ = _fp8_pair(M, K, fmt)
        a_scale = None

    w_e_packed, w_scale_e = [], []
    for _ in range(E):
        wp, _ = _mxfp4_pair(K, N_full, packed_dim=0)
        w_e_packed.append(wp)
        sd, _ = _mx_scale_pair(N_full, K)
        w_scale_e.append(sd)
    w3 = torch.stack(w_e_packed)
    w_scale3 = torch.stack(w_scale_e)

    counts = torch.full((E,), per_exp, device=device, dtype=torch.int32)
    md = make_ragged_tensor_metadata(counts, M)

    common_kwargs = dict(
        x_scale=a_scale,
        x_format=fmt,
        x_global_scale=a_global,
        bias=None,
        a_ragged_metadata=md,
        gather_indx=None,
        swiglu_alpha=1.0,
        swiglu_limit=0.0,
        block_m=16,
        block_n=128,
        block_k=128,
        num_warps=4,
    )

    y_fp32 = gluon_mxfp_dispatch_swiglu(
        a_packed,
        w3,
        w_scale3,
        out_dtype=torch.float32,
        **common_kwargs,
    )
    # AMD CDNA4's v_cvt_pk_fp8_f32 instruction saturates fp32 overflow
    # to fp8_e4m3fn MAX (= 448.0), while PyTorch's host fp32->fp8 cast
    # saturates overflow to fp8 NaN. Both are spec-valid; AITER's
    # ``_compute_static_fp8_quant`` (the AITER W4A8 reference we are
    # matching) inherits the HW saturate-to-max behavior. Clamp the
    # host reference to mirror the kernel's saturate-to-max semantics.
    _FP8_E4M3_MAX = 448.0
    y_ref_fp32 = (y_fp32 / scale_value).clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX)
    y_ref_quant_fp8 = y_ref_fp32.to(torch.float8_e4m3fn)

    scale_t = torch.tensor([scale_value], dtype=torch.float32, device=device)
    y_fused = gluon_mxfp_dispatch_swiglu(
        a_packed,
        w3,
        w_scale3,
        out_quant_scale=scale_t,
        **common_kwargs,
    )
    assert (
        y_fused.dtype == torch.float8_e4m3fn
    ), f"fused path must produce fp8e4m3 output, got {y_fused.dtype}"

    # Both paths divide by ``scale_value`` in fp32 then cast to fp8e4m3.
    # The kernel uses the AMD CDNA4 fp8 convert intrinsic (round-to-
    # nearest-even with ties-away on some lanes) while PyTorch's host
    # cast may break ties differently; this manifests as +/-1 raw byte
    # on lanes that fell exactly on a fp8 tie. Both are spec-valid fp8
    # quantizations, so we assert (a) max raw-byte diff <= 1 and (b)
    # dequantized error <= 1 fp8 ULP (= relative 12.5% within a binade,
    # capped by the absolute fp8 grain at zero of ~2^-9).
    fused_u8 = y_fused.view(torch.uint8).to(torch.int16)
    ref_u8 = y_ref_quant_fp8.view(torch.uint8).to(torch.int16)
    byte_diff = (fused_u8 - ref_u8).abs()
    max_byte_diff = int(byte_diff.max().item())
    assert max_byte_diff <= 1, (
        f"fused fp8 quant byte diff > 1 ULP: max={max_byte_diff} "
        f"differing_lanes={int((byte_diff > 0).sum().item())}/{byte_diff.numel()}"
    )

    # No NaN should appear in either side after the saturate-to-max
    # reference clamp; verify, then check finite-lane error within one
    # fp8e4m3 ULP.
    y_fused_dq = y_fused.float()
    y_ref_dq = y_ref_quant_fp8.float()
    assert not torch.isnan(y_fused_dq).any(), "kernel produced fp8 NaN unexpectedly"
    assert not torch.isnan(
        y_ref_dq
    ).any(), "host reference produced fp8 NaN unexpectedly"
    abs_err = (y_fused_dq - y_ref_dq).abs()
    # fp8e4m3 binade resolution = 1/8 of the binade value -> dequant
    # error of one ULP is at most |y| / 8 + 2^-9 (subnormal grain).
    fp8_ulp_bound = y_ref_dq.abs() / 8.0 + 2.0**-9
    violations = abs_err > fp8_ulp_bound
    assert not violations.any(), (
        f"fused fp8 quant exceeds 1 ULP dequant bound: "
        f"violating_lanes={int(violations.sum().item())}/{abs_err.numel()}, "
        f"worst abs_err={abs_err.max().item():.4g}"
    )


def test_dispatch_swiglu_out_quant_scale_requires_swiglu():
    """The fused-quant epilogue is wired only through the SwiGLU branch;
    calling the launcher with out_quant_scale but swiglu=None should
    raise a clear error instead of silently writing garbage."""
    from tokenspeed_kernel.ops.moe.gluon import _launch_kernel

    # x_format=e4m3 (1B/elem so K_phys=64 -> K_logical=64); w_format=e2m1
    # (4-bit packed so K_phys=32 -> K_logical=64). Shapes are chosen so
    # the K-mismatch assertion does not trip before our ValueError.
    x = torch.zeros((32, 64), device="cuda", dtype=torch.uint8)
    w = torch.zeros((1, 32, 32), device="cuda", dtype=torch.uint8)
    w_scale = torch.zeros((1, 32, 64 // 32), device="cuda", dtype=torch.uint8)
    y = torch.empty((32, 32), device="cuda", dtype=torch.float8_e4m3fn)
    scale_t = torch.tensor([0.5], dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match=r"out_quant_scale .* SwiGLU"):
        _launch_kernel(
            x,
            w.squeeze(0),
            y=y,
            bias=None,
            gather_indx=None,
            scatter_indx=None,
            gate_scal=None,
            a_ragged_metadata=None,
            swiglu=None,
            out_block_n=32,
            block_m=32,
            block_n=32,
            block_k=128,
            num_warps=4,
            num_buffers=2,
            x_format="e4m3",
            w_format="e2m1",
            x_scale=None,
            w_scale=w_scale,
            x_global_scale=1.0,
            out_quant_scale=scale_t,
        )


def test_scaled_kernel_no_register_spill():
    """Compile the scaled kernel at a small ragged-combine shape and
    verify the AMDGCN report contains zero spills + zero scratch.

    Uses ``gluon_mxfp_combine`` (production path) to drive the compile;
    any production launcher works since they all share the
    ``_pipelined_moe_kernel_scaled`` JIT target.
    """
    from tokenspeed_kernel.ops.moe.gluon import (
        _pipelined_moe_kernel_scaled,
        assert_no_spills,
        gluon_mxfp_combine,
        static_profile,
    )
    from triton_kernels.tensor import make_ragged_tensor_metadata

    torch.manual_seed(0)
    device = "cuda"
    E, per_exp = 2, 16
    M, K, N = E * per_exp, 256, 64
    a_packed, _ = _mxfp4_pair(M, K, packed_dim=1)
    w_packed = torch.stack([_mxfp4_pair(K, N, packed_dim=0)[0] for _ in range(E)])
    a_scale, _ = _mx_scale_pair(M, K)
    w_scale = torch.stack([_mx_scale_pair(N, K)[0] for _ in range(E)])
    counts = torch.full((E,), per_exp, device=device, dtype=torch.int32)
    md = make_ragged_tensor_metadata(counts, M)

    gluon_mxfp_combine(
        a_packed,
        w_packed,
        w_scale,
        x_scale=a_scale,
        x_format="e2m1",
        bias=None,
        a_ragged_metadata=md,
        scatter_indx=None,
        n_tokens=M,
        n_expts_act=1,
        out_dtype=torch.float32,
        block_m=16,
        block_n=64,
        block_k=128,
        num_warps=4,
    )

    cache = _pipelined_moe_kernel_scaled.device_caches.get(torch.cuda.current_device())
    assert cache, "expected the scaled Gluon kernel to JIT-compile at least once"
    compiled = next(iter(cache[0].values()))
    profile = static_profile(compiled, label="mxfp4_combine")
    assert_no_spills(profile)


# ---------------------------------------------------------------------------
# Selector / fallback regression checks
# ---------------------------------------------------------------------------


def test_gluon_kernel_selected_under_env(monkeypatch):
    """With ``TOKENSPEED_MOE_GLUON=1`` the registry picks the Gluon variant."""
    pytest.importorskip("triton_kernels")
    monkeypatch.setenv("TOKENSPEED_MOE_GLUON", "1")

    import importlib

    import tokenspeed_kernel.ops.moe as moe_pkg
    import tokenspeed_kernel.ops.moe.gluon as gluon_mod

    importlib.reload(gluon_mod)
    importlib.reload(moe_pkg)

    from tokenspeed_kernel.registry import KernelRegistry
    from tokenspeed_kernel.selection import select_kernel

    KernelRegistry.get().clear_cache()
    selected = select_kernel(
        "moe",
        "experts",
        torch.bfloat16,
        features=frozenset({"ragged_metadata", "dispatch_gemm"}),
        traits={},
    )
    selected_name = getattr(selected, "name", None) or getattr(selected, "__name__", "")
    assert "gluon" in selected_name, f"unexpected selected kernel: {selected_name}"

    monkeypatch.delenv("TOKENSPEED_MOE_GLUON", raising=False)
    importlib.reload(gluon_mod)
    importlib.reload(moe_pkg)


def test_gluon_adapter_routes_pure_bf16_to_upstream():
    """The adapter falls back to ``triton_kernels.matmul`` when neither
    fp8 ``flex_ctx`` nor mxfp4 ``x_mx_scale`` are present (i.e. the
    pure bf16 x bf16 path the Gluon kernel no longer supports natively).
    """
    pytest.importorskip("triton_kernels")
    from unittest.mock import patch

    from tokenspeed_kernel.ops.moe.gluon import _gluon_mxfp_ragged_matmul
    from triton_kernels.matmul import PrecisionConfig

    M, N, K, E = 64, 128, 128, 2
    device = "cuda"
    x = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    w_bf16 = torch.zeros(E, K, N, device=device, dtype=torch.bfloat16)
    md, _, _, _ = _build_ragged(M, E, device=device)
    pc = PrecisionConfig()

    sentinel = torch.zeros((M, N), device=device, dtype=torch.bfloat16)
    with patch(
        "tokenspeed_kernel.ops.moe.gluon._upstream_matmul", return_value=sentinel
    ) as upstream:
        out = _gluon_mxfp_ragged_matmul(
            x,
            w_bf16,
            bias=None,
            a_ragged_metadata=md,
            gather_indx=None,
            scatter_indx=None,
            precision_config=pc,
            fused_activation=None,
            n_tokens=None,
            n_expts_act=None,
        )
    assert out is sentinel, "pure bf16 path should be forwarded to upstream matmul"
    upstream.assert_called_once()


# ---------------------------------------------------------------------------
# P0-1 runtime wiring: ``out_quant_scale`` must propagate from the
# ``_gluon_mxfp_ragged_matmul`` selector kwarg through ``_try_dispatch_mxfp``
# down to ``gluon_mxfp_dispatch_swiglu``. Verifies the dispatcher seam used
# by ``triton_kernel_fp8.py``'s P0-1 wiring on the production W4A8 path.
# ---------------------------------------------------------------------------


def test_dispatch_mxfp_forwards_out_quant_scale_to_swiglu():
    """The runtime backend hands the dispatcher
    ``out_quant_scale=layer.w2_act_scale`` so the Gluon SwiGLU epilogue can
    cast its fp32 output to fp8 directly. Verify the kwarg flows through
    ``_try_dispatch_mxfp`` to ``gluon_mxfp_dispatch_swiglu`` unchanged."""
    pytest.importorskip("triton_kernels")
    from unittest.mock import patch

    from tokenspeed_kernel.ops.moe.gluon import _gluon_mxfp_ragged_matmul
    from triton_kernels.matmul import FlexCtx, InFlexData, PrecisionConfig

    M, N, K, E = 64, 128, 128, 2
    device = "cuda"
    x = torch.zeros(M, K, device=device, dtype=torch.uint8)  # fp8 view (e4m3fn)
    x = x.view(torch.float8_e4m3fn)

    # 4-bit packed weights: (E, K, N_packed)
    w_packed = torch.zeros(E, K, N // 8 * 4, device=device, dtype=torch.uint8)
    w_scale = torch.zeros(
        E, N, K // 32, device=device, dtype=torch.uint8
    )  # mxfp4 block scale
    md, gather, _, _ = _build_ragged(M, E, block_m=32, device=device)

    a_scale = torch.tensor([0.7], dtype=torch.float32, device=device)
    pc = PrecisionConfig(
        flex_ctx=FlexCtx(lhs_data=InFlexData(dtype=torch.float8_e4m3fn, scale=a_scale)),
        w_mx_scale=w_scale,
        b_microblock_size=32,
        out_dtype=torch.bfloat16,
    )

    sentinel = torch.zeros((M, N // 2), device=device, dtype=torch.float8_e4m3fn)

    out_quant_scale = torch.tensor([0.137], dtype=torch.float32, device=device)
    # Build a SwiGLU FusedActivation the dispatcher recognises.
    from tokenspeed_kernel.ops.moe.triton_kernels import (
        FnSpecs,
        FusedActivation,
        swiglu_fn,
    )

    act = FusedActivation(
        FnSpecs("swiglu", swiglu_fn, ("alpha", "limit"), reduction_n=2),
        (1.702, 7.0),
    )

    with patch(
        "tokenspeed_kernel.ops.moe.gluon.gluon_mxfp_dispatch_swiglu",
        return_value=sentinel,
    ) as patched_swiglu, patch(
        "tokenspeed_kernel.ops.moe.gluon._extract_gluon_raw_w",
        side_effect=lambda w: w_packed,
    ), patch(
        "tokenspeed_kernel.ops.moe.gluon._extract_gluon_raw_s",
        side_effect=lambda s: w_scale,
    ):
        out = _gluon_mxfp_ragged_matmul(
            x,
            w_packed,
            bias=None,
            a_ragged_metadata=md,
            gather_indx=gather,
            scatter_indx=None,
            precision_config=pc,
            fused_activation=act,
            n_tokens=None,
            n_expts_act=None,
            out_quant_scale=out_quant_scale,
        )

    assert out is sentinel
    patched_swiglu.assert_called_once()
    call_kwargs = patched_swiglu.call_args.kwargs
    assert (
        call_kwargs.get("out_quant_scale") is out_quant_scale
    ), f"out_quant_scale not forwarded: {call_kwargs.get('out_quant_scale')}"


# ---------------------------------------------------------------------------
# shuffle_weight_for_gluon_dot_layout: host-side helper that reorders W into
# the 5-D HBM byte layout consumed by the W_VIA_VGPR path. Pure CPU torch
# ops; we exercise the parametrized block dims here to pin the contract.
# ---------------------------------------------------------------------------


def _gluon_dot_oracle_offset(k, n, block_k_pk, block_n):
    """Pure-python reference for the 5-D HBM byte offset within a single
    CTA tile. Mirrors the documented (n_block, k_block, k_quad, n_in_sub,
    k_within) layout used by ``issue_global_load_to_vgpr``.
    """
    K_WIDTH, N_LANE, K_QUAD = 16, 16, 4
    sub_tile_k = K_QUAD * K_WIDTH
    stride_n_in_sub = K_WIDTH
    stride_k_quad = N_LANE * K_WIDTH
    stride_k_block = K_QUAD * stride_k_quad
    stride_n_block = (block_k_pk // sub_tile_k) * stride_k_block
    k_within = k % K_WIDTH
    k_quad = (k // K_WIDTH) % K_QUAD
    k_block = k // sub_tile_k
    n_in_sub = n % N_LANE
    n_block_in_tile = n // N_LANE
    return (
        n_block_in_tile * stride_n_block
        + k_block * stride_k_block
        + k_quad * stride_k_quad
        + n_in_sub * stride_n_in_sub
        + k_within
    )


@pytest.mark.parametrize(
    "block_k_pk,block_n",
    [
        (128, 128),  # production default; pinned by kernel static_assert
        (64, 128),  # smaller K tile (still multiple of SUB_TILE_K=64)
        (128, 256),  # larger N tile
    ],
)
def test_shuffle_weight_for_gluon_dot_layout_byte_pattern(block_k_pk, block_n):
    """The shuffled tensor at byte position ``P`` must equal the source
    byte at ``(k, n)`` per the documented 5-D HBM layout.
    """
    from tokenspeed_kernel.ops.moe.gluon import shuffle_weight_for_gluon_dot_layout

    E, K_pk, N = 2, block_k_pk * 2, block_n * 3
    g = torch.Generator(device="cpu").manual_seed(0)
    w = torch.randint(0, 256, (E, K_pk, N), dtype=torch.uint8, generator=g)

    out = shuffle_weight_for_gluon_dot_layout(w, block_k_pk=block_k_pk, block_n=block_n)

    assert out.shape == (E, K_pk, N)
    assert getattr(out, "is_shuffled_for_gluon_dot", False) is True
    assert getattr(out, "original_k_pk", None) == K_pk

    N_CTA_TILES = N // block_n
    tile_bytes = block_k_pk * block_n
    out_flat = out.reshape(E, K_pk * N)

    # Spot-check every (k, n) coordinate (cheap: K*N <= 256*384 here).
    for e in range(E):
        for k in range(K_pk):
            kt = k // block_k_pk
            k_in_tile = k % block_k_pk
            for n in range(N):
                nt = n // block_n
                n_in_tile = n % block_n
                P = (kt * N_CTA_TILES + nt) * tile_bytes + _gluon_dot_oracle_offset(
                    k_in_tile, n_in_tile, block_k_pk, block_n
                )
                assert int(out_flat[e, P]) == int(
                    w[e, k, n]
                ), f"mismatch at e={e}, k={k}, n={n}, P={P}"


def test_shuffle_weight_for_gluon_dot_layout_pads_k_pk():
    """Non-multiple K_pk must be zero-padded up to ``block_k_pk`` and
    the original size stamped on ``out.original_k_pk``."""
    from tokenspeed_kernel.ops.moe.gluon import shuffle_weight_for_gluon_dot_layout

    block_k_pk, block_n = 128, 128
    E, K_pk, N = 1, 80, 128  # K_pk=80 -> padded to 128
    g = torch.Generator(device="cpu").manual_seed(0)
    w = torch.randint(0, 256, (E, K_pk, N), dtype=torch.uint8, generator=g)

    out = shuffle_weight_for_gluon_dot_layout(w, block_k_pk=block_k_pk, block_n=block_n)
    assert out.shape == (E, block_k_pk, N)
    assert out.original_k_pk == K_pk

    # The tail bytes (k in [K_pk, block_k_pk)) must be zero-filled at
    # their target HBM positions; the head bytes (k < K_pk) must match
    # the source.
    tile_bytes = block_k_pk * block_n
    out_flat = out.reshape(E, block_k_pk * N)
    for k in range(block_k_pk):
        k_in_tile = k
        for n in range(N):
            P = _gluon_dot_oracle_offset(k_in_tile, n, block_k_pk, block_n)
            assert 0 <= P < tile_bytes
            expected = int(w[0, k, n]) if k < K_pk else 0
            assert int(out_flat[0, P]) == expected, (
                f"pad mismatch at k={k}, n={n}: got {int(out_flat[0, P])}, "
                f"expected {expected}"
            )


def test_shuffle_weight_for_gluon_dot_layout_rejects_bad_block_dims():
    """Block dims must be positive multiples of the MFMA inner factors;
    N must divide ``block_n``."""
    from tokenspeed_kernel.ops.moe.gluon import shuffle_weight_for_gluon_dot_layout

    w = torch.zeros(1, 128, 128, dtype=torch.uint8)

    # block_k_pk not a multiple of SUB_TILE_K=64
    with pytest.raises(ValueError, match=r"block_k_pk"):
        shuffle_weight_for_gluon_dot_layout(w, block_k_pk=96, block_n=128)
    # block_n not a multiple of N_LANE=16
    with pytest.raises(ValueError, match=r"block_n"):
        shuffle_weight_for_gluon_dot_layout(w, block_k_pk=128, block_n=24)
    # zero or negative block dims
    with pytest.raises(ValueError, match=r"block_k_pk"):
        shuffle_weight_for_gluon_dot_layout(w, block_k_pk=0, block_n=128)
    # N not divisible by block_n
    w_bad_n = torch.zeros(1, 128, 144, dtype=torch.uint8)  # 144 % 128 != 0
    with pytest.raises(ValueError, match=r"divisible by block_n"):
        shuffle_weight_for_gluon_dot_layout(w_bad_n, block_k_pk=128, block_n=128)
