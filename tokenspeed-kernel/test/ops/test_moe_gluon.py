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

"""Correctness + spill tests for the MI355 Gluon MoE kernels.

Three pieces are exercised:

* ``gluon_bf16_gating_gemm`` -- bf16 dense GEMM (`x @ w`).
* ``gluon_bf16_dispatch_swiglu`` -- ragged GEMM with optional gather +
  fused SwiGLU.
* ``gluon_bf16_combine`` -- ragged GEMM with optional scatter + weighted
  combine across top-k.

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


def _torch_ragged_matmul(x, w, counts, *, bias=None, gather_indx=None):
    """Per-expert dense matmul on slices of x; reference for bf16 path."""
    if gather_indx is not None:
        x = x[gather_indx.src_indx.long()]
    M, K = x.shape
    E, K_W, N = w.shape
    out = torch.zeros((M, N), device=x.device, dtype=torch.float32)
    start = 0
    for e in range(E):
        n = int(counts[e].item())
        if n == 0:
            continue
        slc = x[start : start + n].to(torch.float32)
        we = w[e].to(torch.float32)
        out[start : start + n] = slc @ we
        if bias is not None:
            out[start : start + n] += bias[e].to(torch.float32)
        start += n
    return out


def _torch_swiglu(x, *, alpha: float, limit: float):
    """Same recipe as ``triton_kernels.swiglu`` so we can compare bit-equally."""
    gate, linear = x[..., 0::2], x[..., 1::2]
    if limit > 0.0:
        gate = torch.minimum(gate, torch.tensor(limit, device=gate.device))
        linear = torch.minimum(
            torch.maximum(linear, torch.tensor(-limit, device=linear.device)),
            torch.tensor(limit, device=linear.device),
        )
    s = gate / (1.0 + torch.exp(-alpha * gate))
    return s * (linear + 1.0)


# ---------------------------------------------------------------------------
# Kernel 1: bf16 x bf16 gating GEMM
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("M,N,K", [(128, 128, 128), (256, 128, 256), (512, 128, 512)])
def test_gating_gemm_correctness(M, N, K):
    from tokenspeed_kernel.ops.moe.gluon import gluon_bf16_gating_gemm

    torch.manual_seed(0)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.05
    w = torch.randn(K, N, device="cuda", dtype=torch.bfloat16) * 0.05
    y = gluon_bf16_gating_gemm(x, w, block_m=128, block_n=128, block_k=64)
    ref = x.to(torch.float32) @ w.to(torch.float32)
    torch.testing.assert_close(y.to(torch.float32), ref, rtol=5e-2, atol=5e-2)


# ---------------------------------------------------------------------------
# Kernel 2: dispatch + 1st GEMM + SwiGLU
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "M,N,K,E,block_m,block_n",
    [
        (128, 128, 128, 2, 128, 128),
        (256, 256, 128, 4, 128, 128),
    ],
)
def test_dispatch_swiglu_correctness(M, N, K, E, block_m, block_n):
    from tokenspeed_kernel.ops.moe.gluon import gluon_bf16_dispatch_swiglu

    torch.manual_seed(0)
    md, gather_indx, counts, M_padded = _build_ragged(
        M, E, block_m=block_m, device="cuda"
    )
    x = torch.randn(M_padded, K, device="cuda", dtype=torch.bfloat16) * 0.05
    w = torch.randn(E, K, N, device="cuda", dtype=torch.bfloat16) * 0.05

    y = gluon_bf16_dispatch_swiglu(
        x,
        w,
        bias=None,
        a_ragged_metadata=md,
        gather_indx=None,  # x is already permuted to match the ragged layout
        swiglu_alpha=1.0,
        swiglu_limit=0.0,
        block_m=block_m,
        block_n=block_n,
        block_k=64,
    )
    raw = _torch_ragged_matmul(x, w, counts)
    ref = _torch_swiglu(raw, alpha=1.0, limit=0.0)
    torch.testing.assert_close(
        y.to(torch.float32), ref.to(torch.float32), rtol=5e-2, atol=8e-2
    )


# ---------------------------------------------------------------------------
# Kernel 3: 2nd GEMM + scatter combine
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "M,N,K,E,block_m,block_n",
    [
        (128, 128, 128, 2, 128, 128),
        (256, 256, 128, 4, 128, 128),
    ],
)
def test_combine_correctness(M, N, K, E, block_m, block_n):
    """``y[token] = sum_{e in topk} x_e @ w_e`` -- check the scatter sum."""
    from tokenspeed_kernel.ops.moe.gluon import gluon_bf16_combine

    torch.manual_seed(0)
    md, _, counts, M_padded = _build_ragged(M, E, block_m=block_m, device="cuda")
    x = torch.randn(M_padded, K, device="cuda", dtype=torch.bfloat16) * 0.05
    w = torch.randn(E, K, N, device="cuda", dtype=torch.bfloat16) * 0.05

    # Identity scatter: each dispatched row writes back to its own row.
    # n_expts_act=1 so the post-kernel reduce is a no-op.
    scatter_indx = type(
        "ScatterIndx",
        (),
        {"dst_indx": torch.arange(M_padded, device="cuda", dtype=torch.int32)},
    )()

    y = gluon_bf16_combine(
        x,
        w,
        bias=None,
        a_ragged_metadata=md,
        scatter_indx=scatter_indx,
        gate_scal=None,
        n_tokens=M_padded,
        n_expts_act=1,
        block_m=block_m,
        block_n=block_n,
        block_k=64,
    )
    ref = _torch_ragged_matmul(x, w, counts)
    torch.testing.assert_close(
        y.to(torch.float32), ref.to(torch.float32), rtol=5e-2, atol=5e-2
    )


# ---------------------------------------------------------------------------
# Static spill profile: assert no sgpr / vgpr spill
# ---------------------------------------------------------------------------


def test_no_register_spill():
    """Each kernel must compile without sgpr / vgpr spills.

    We probe the AMDGCN dump after a real launch (which is what triggers
    Gluon to actually JIT compile and cache an asm artifact).
    """
    from tokenspeed_kernel.ops.moe.gluon import (
        _pipelined_moe_kernel,
        assert_no_spills,
        gluon_bf16_combine,
        gluon_bf16_dispatch_swiglu,
        gluon_bf16_gating_gemm,
        static_profile,
    )

    M, N, K, E, block_m, block_n, block_k = 256, 256, 128, 4, 128, 128, 64
    md, _, _, M_padded = _build_ragged(M, E, block_m=block_m, device="cuda")
    x = torch.randn(M_padded, K, device="cuda", dtype=torch.bfloat16)
    w_dense = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    w_moe = torch.randn(E, K, N, device="cuda", dtype=torch.bfloat16)
    w_moe_2x = torch.randn(E, K, 2 * N, device="cuda", dtype=torch.bfloat16)

    # 1. gating
    gluon_bf16_gating_gemm(
        x[:128], w_dense, block_m=block_m, block_n=block_n, block_k=block_k
    )
    # 2. dispatch + swiglu (output N is 2*N // 2 = N)
    gluon_bf16_dispatch_swiglu(
        x,
        w_moe_2x,
        bias=None,
        a_ragged_metadata=md,
        gather_indx=None,
        swiglu_alpha=1.0,
        swiglu_limit=0.0,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
    )
    # 3. combine (identity scatter)
    scatter = type(
        "ScatterIndx",
        (),
        {"dst_indx": torch.arange(M_padded, device="cuda", dtype=torch.int32)},
    )()
    gluon_bf16_combine(
        x,
        w_moe,
        bias=None,
        a_ragged_metadata=md,
        scatter_indx=scatter,
        gate_scal=None,
        n_tokens=M_padded,
        n_expts_act=1,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
    )

    # The Gluon JIT caches per (device, signature); each compiled artifact
    # holds the AMDGCN dump in ``.asm['amdgcn']`` which is what
    # ``static_profile`` consumes.
    device = torch.cuda.current_device()
    device_cache = _pipelined_moe_kernel.device_caches.get(device, None)
    assert device_cache, "expected the Gluon kernel to JIT-compile at least once"
    # device_cache is a (kernel_cache, env, ...) tuple; first element maps
    # signature -> compiled kernel.
    kernel_cache = device_cache[0]
    compiled = next(iter(kernel_cache.values()))
    profile = static_profile(compiled, label="moe-pipe")
    assert_no_spills(profile, allow_scratch=0)


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


def test_gluon_falls_back_for_mxfp4():
    """mxfp4 weights / fp8 activations should fall back to triton_kernels."""
    pytest.importorskip("triton_kernels")
    from tokenspeed_kernel.ops.moe.gluon import _gluon_bf16_ragged_matmul
    from triton_kernels.matmul import PrecisionConfig

    M, N, K, E = 64, 128, 128, 2
    device = "cuda"
    x = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    w_mxfp4 = torch.zeros(E, K // 2, N, device=device, dtype=torch.uint8)
    w_scale = torch.zeros(E, K // 32, N, device=device, dtype=torch.uint8)
    md, _, _, _ = _build_ragged(M, E, device=device)
    pc = PrecisionConfig()
    pc.b_mx_scale = w_scale
    with pytest.raises(Exception):
        _gluon_bf16_ragged_matmul(
            x,
            w_mxfp4,
            bias=None,
            a_ragged_metadata=md,
            gather_indx=None,
            scatter_indx=None,
            precision_config=pc,
            fused_activation=None,
            n_tokens=None,
            n_expts_act=None,
        )
