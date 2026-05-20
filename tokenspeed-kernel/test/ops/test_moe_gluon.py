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
            a_format="e2m1",
            b_format="e2m1",
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


@pytest.mark.parametrize("scale_mode", ["bypass", "transpose", "swizzle"])
@pytest.mark.parametrize(
    "M,N,K",
    [
        (32, 32, 256),
        (64, 128, 256),
        (128, 64, 256),
        (64, 128, 2880),
    ],
)
def test_mxfp4_x_mxfp4_gating(M, N, K, scale_mode):
    """``e2m1`` x ``e2m1`` dense GEMM via scaled MFMA."""
    from tokenspeed_kernel.ops.moe.gluon import gluon_mxfp_gating_gemm

    torch.manual_seed(0)
    a_packed, a_fp32 = _mxfp4_pair(M, K, packed_dim=1)
    w_packed, w_fp32 = _mxfp4_pair(K, N, packed_dim=0)
    a_scale, a_scale_mk = _mx_scale_pair(M, K)
    w_scale, w_scale_nk = _mx_scale_pair(N, K)
    w_scale_kn = w_scale_nk.T.contiguous()

    y = gluon_mxfp_gating_gemm(
        a_packed,
        w_packed,
        w_scale,
        x_scale=a_scale,
        a_format="e2m1",
        out_dtype=torch.float32,
        scale_load_mode=scale_mode,
    )
    y_ref = (a_fp32 * a_scale_mk) @ (w_fp32 * w_scale_kn)
    rel = (y - y_ref).abs().max().item() / max(1.0, y_ref.abs().max().item())
    assert rel < 5e-2, f"rel_max={rel} too large"


@pytest.mark.parametrize("scale_mode", ["bypass", "transpose", "swizzle"])
@pytest.mark.parametrize("fmt", ["e4m3", "e5m2"])
@pytest.mark.parametrize("M,N,K", [(32, 32, 128), (64, 128, 256)])
def test_fp8_x_mxfp4_gating(fmt, M, N, K, scale_mode):
    """``e4m3``/``e5m2`` (A) x ``e2m1`` (W) GEMM with global A scale."""
    from tokenspeed_kernel.ops.moe.gluon import gluon_mxfp_gating_gemm

    if scale_mode == "swizzle" and K < 256:
        pytest.skip(
            "swizzle needs BLOCK_K >= 256 (-> K >= 256 for the " "single-tile case)."
        )

    torch.manual_seed(0)
    a_u8, a_fp32 = _fp8_pair(M, K, fmt)
    w_packed, w_fp32 = _mxfp4_pair(K, N, packed_dim=0)
    w_scale, w_scale_nk = _mx_scale_pair(N, K)
    w_scale_kn = w_scale_nk.T.contiguous()
    a_global = 0.137

    # swizzle requires block_k >= 256 (so MX_SCALE_BLOCK_K >= 8 for
    # the inner-8 reshape in `unswizzle_mx_scale_cdna4`); the other modes
    # can use the smaller block_k=128 which exercises the partial-K-tile
    # masking path. Let autotune pick block_k for swizzle.
    block_k_arg = None if scale_mode == "swizzle" else 128
    y = gluon_mxfp_gating_gemm(
        a_u8,
        w_packed,
        w_scale,
        x_scale=None,
        a_format=fmt,
        a_global_scale=a_global,
        out_dtype=torch.float32,
        block_k=block_k_arg,
        scale_load_mode=scale_mode,
    )
    y_ref = a_global * (a_fp32 @ (w_fp32 * w_scale_kn))
    rel = (y - y_ref).abs().max().item() / max(1.0, y_ref.abs().max().item())
    assert rel < 5e-2, f"rel_max={rel} too large"


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
        a_format=fmt,
        a_global_scale=a_global,
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
        a_format=fmt,
        a_global_scale=a_global,
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


def test_scaled_kernel_no_register_spill():
    """Compile the scaled kernel at a gpt-oss-sized prefill shape and
    verify the AMDGCN report contains zero spills + zero scratch."""
    from tokenspeed_kernel.ops.moe.gluon import (
        _pipelined_moe_kernel_scaled,
        assert_no_spills,
        gluon_mxfp_gating_gemm,
        static_profile,
    )

    torch.manual_seed(0)
    M, N, K = 128, 128, 256
    a_packed, _ = _mxfp4_pair(M, K, packed_dim=1)
    w_packed, _ = _mxfp4_pair(K, N, packed_dim=0)
    a_scale, _ = _mx_scale_pair(M, K)
    w_scale, _ = _mx_scale_pair(N, K)

    gluon_mxfp_gating_gemm(
        a_packed,
        w_packed,
        w_scale,
        x_scale=a_scale,
        a_format="e2m1",
        out_dtype=torch.float32,
        block_m=64,
        block_n=64,
        block_k=128,
        num_warps=4,
    )

    device = torch.cuda.current_device()
    cache = _pipelined_moe_kernel_scaled.device_caches.get(device)
    assert cache, "expected the scaled Gluon kernel to JIT-compile at least once"
    compiled = next(iter(cache[0].values()))
    profile = static_profile(compiled, label="mxfp4_gating")
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
    fp8 ``flex_ctx`` nor mxfp4 ``a_mx_scale`` are present (i.e. the
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
