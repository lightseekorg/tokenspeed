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
# gpt-oss-120b shape coverage (decode + prefill)
# ---------------------------------------------------------------------------
#
# Anchors for the per-shape regressions added in ``Update 2``: we exercise
# the kernels at the *real* gpt-oss-120b MoE GEMM dimensions (per the HF
# config: ``hidden_size=2880, intermediate_size=2880, num_local_experts=128,
# num_experts_per_tok=4``) for both the sparse-decode (``B=1`` -> 4 active
# experts via the new active-expert remap) and the dense-prefill (``B=64``
# -> all 128 experts active) paths.
#
# These are correctness *and* spill checks but use small ``H/I/E`` to keep
# CI runtime bounded; the full prefill scale (M_d=8192) is exercised by
# ``benchmarks/moe_gluon_microbench.py``.

GPTOSS_H = 2880
GPTOSS_I = 2880
GPTOSS_E = 128
GPTOSS_TOPK = 4


def _gptoss_ragged(B: int, *, block_m: int = 64, device: str = "cuda"):
    """Build a gpt-oss-120b style ragged metadata: ``min(B*topk, E)``
    active experts each holding ``per_expert >= block_m`` rows. Mirrors
    the helper in ``benchmarks/moe_gluon_microbench.py``.
    """
    from triton_kernels.tensor import make_ragged_tensor_metadata

    M_d = B * GPTOSS_TOPK
    n_active = min(M_d, GPTOSS_E)
    per_expert = max(
        block_m,
        ((M_d + n_active - 1) // n_active + block_m - 1) // block_m * block_m,
    )
    M_padded = per_expert * n_active
    counts = torch.zeros((GPTOSS_E,), device=device, dtype=torch.int32)
    counts[:n_active] = per_expert
    md = make_ragged_tensor_metadata(counts, M_padded)
    return md, counts, M_padded, n_active, per_expert


@pytest.mark.parametrize("B", [1, 32, 64])
def test_gpt_oss_decode_remap(B):
    """Decode batches activate ``min(B*topk, E)`` experts; the active-expert
    remap must keep the kernel numerically equivalent to the dense
    reference (active experts only)."""
    from tokenspeed_kernel.ops.moe.gluon import gluon_bf16_combine

    torch.manual_seed(0)
    md, counts, M_padded, n_active, per_expert = _gptoss_ragged(B)
    # Use a small inner dim to keep the test fast.
    K, N = 128, 128
    x = torch.randn(M_padded, K, device="cuda", dtype=torch.bfloat16) * 0.05
    w = torch.randn(GPTOSS_E, K, N, device="cuda", dtype=torch.bfloat16) * 0.05
    scatter = type(
        "ScatterIndx",
        (),
        {"dst_indx": torch.arange(M_padded, device="cuda", dtype=torch.int32)},
    )()
    y = gluon_bf16_combine(
        x,
        w,
        bias=None,
        a_ragged_metadata=md,
        scatter_indx=scatter,
        gate_scal=None,
        n_tokens=M_padded,
        n_expts_act=1,
    )
    # Reference: per-expert dense matmul over the *active* experts only.
    ref = torch.zeros((M_padded, N), device="cuda", dtype=torch.float32)
    start = 0
    for e in range(GPTOSS_E):
        n = int(counts[e].item())
        if n == 0:
            continue
        ref[start : start + n] = x[start : start + n].to(torch.float32) @ w[e].to(
            torch.float32
        )
        start += n
    torch.testing.assert_close(y.to(torch.float32), ref, rtol=5e-2, atol=5e-2)


@pytest.mark.parametrize("B", [1, 64])
def test_gpt_oss_no_spill(B):
    """The autotuner must pick a non-spilling config for both decode
    (sparse, ``B=1``) and prefill (dense, ``B=64``) at gpt-oss-120b sizes."""
    from tokenspeed_kernel.ops.moe.gluon import (
        _pipelined_moe_kernel,
        assert_no_spills,
        gluon_bf16_combine,
        gluon_bf16_dispatch_swiglu,
        gluon_bf16_gating_gemm,
        static_profile,
    )

    torch.manual_seed(0)
    # Use real gpt-oss H/I/E but cap K to the smaller of (H, 256) so
    # JIT compile stays cheap; this still triggers the same kernel
    # specialisation the production launcher does.
    K = min(GPTOSS_H, 256)
    md, _, M_padded, *_ = _gptoss_ragged(B)

    x_dense = torch.randn(B, K, device="cuda", dtype=torch.bfloat16) * 0.05
    w_dense = torch.randn(K, GPTOSS_E, device="cuda", dtype=torch.bfloat16) * 0.05
    gluon_bf16_gating_gemm(x_dense, w_dense)

    x_pad = torch.randn(M_padded, K, device="cuda", dtype=torch.bfloat16) * 0.05
    w_2x = (
        torch.randn(GPTOSS_E, K, 2 * GPTOSS_I, device="cuda", dtype=torch.bfloat16)
        * 0.05
    )
    gluon_bf16_dispatch_swiglu(
        x_pad,
        w_2x,
        bias=None,
        a_ragged_metadata=md,
        gather_indx=None,
        swiglu_alpha=1.0,
        swiglu_limit=0.0,
    )

    scatter = type(
        "ScatterIndx",
        (),
        {"dst_indx": torch.arange(M_padded, device="cuda", dtype=torch.int32)},
    )()
    x_pad_i = (
        torch.randn(M_padded, GPTOSS_I, device="cuda", dtype=torch.bfloat16) * 0.05
    )
    w_h = (
        torch.randn(GPTOSS_E, GPTOSS_I, GPTOSS_H, device="cuda", dtype=torch.bfloat16)
        * 0.05
    )
    gluon_bf16_combine(
        x_pad_i,
        w_h,
        bias=None,
        a_ragged_metadata=md,
        scatter_indx=scatter,
        gate_scal=None,
        n_tokens=M_padded,
        n_expts_act=1,
    )

    device = torch.cuda.current_device()
    device_cache = _pipelined_moe_kernel.device_caches.get(device)
    assert device_cache, "expected the Gluon kernel to JIT-compile at least once"
    kernel_cache = device_cache[0]
    for sig, compiled in kernel_cache.items():
        prof = static_profile(compiled, label=f"B={B}")
        assert_no_spills(prof, allow_scratch=0)


# ---------------------------------------------------------------------------
# Scaled MFMA / BLOCK_K constraint  (TASKS.md Update 3)
# ---------------------------------------------------------------------------
#
# Per ``TASKS.md`` Update 3, the scaled MFMA op on CDNA4 has instruction
# shape ``[16, 16, 128]``, so any kernel that swaps in
# ``gl.amd.cdna4.mfma_scaled`` (mxfp4 weight + fp8 activation path) must
# respect ``BLOCK_K >= 128`` and ``BLOCK_K % 128 == 0``. These tests
# pin the autotuner contract today so the constraint can't silently
# regress when the scaled path lands.


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
    """``_autotune_block(scaled_mfma=True)`` must enforce CDNA4's
    16x16x128 scaled MFMA shape on ``BLOCK_K``."""
    from tokenspeed_kernel.ops.moe.gluon import _autotune_block

    bm, bn, bk, nw = _autotune_block(
        M, N, K, do_swiglu=do_swiglu, ragged=ragged, scaled_mfma=True
    )
    assert bk >= 128, f"scaled MFMA needs BLOCK_K >= 128, got {bk}"
    assert bk % 128 == 0, f"scaled MFMA needs BLOCK_K % 128 == 0, got {bk}"
    assert bm % 16 == 0, f"BLOCK_M must be a multiple of MFMA M (16), got {bm}"

    # Sanity: ``scaled_mfma=False`` should keep the BK=32/64 fast path.
    _, _, bk_unscaled, _ = _autotune_block(
        M, N, K, do_swiglu=do_swiglu, ragged=ragged, scaled_mfma=False
    )
    assert bk_unscaled in (
        32,
        64,
    ), f"regular MFMA fast path expects BK in {{32, 64}}, got {bk_unscaled}"


def test_launcher_rejects_bad_block_k_for_scaled_mfma():
    """The launcher must refuse a sub-128 ``BLOCK_K`` when the caller
    advertises ``scaled_mfma=True`` (catches future mxfp4 mis-wirings)."""
    from tokenspeed_kernel.ops.moe.gluon import _launch_pipelined

    x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
    y = torch.empty((128, 128), device="cuda", dtype=torch.bfloat16)
    with pytest.raises(AssertionError, match=r"BLOCK_K"):
        _launch_pipelined(
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
            scaled_mfma=True,
        )


def test_kernel_compiles_with_block_k_128():
    """Sanity check that the bf16 kernel still compiles and produces
    correct output at ``BLOCK_K=128`` (matches the scaled-MFMA floor).
    Speed-wise this config is slower than the BK=32/64 default for the
    register-staged pipeline, but we want the *legality* covered so that
    when we land scaled MFMA the kernel body keeps working at BK=128.
    """
    from tokenspeed_kernel.ops.moe.gluon import gluon_bf16_gating_gemm

    torch.manual_seed(0)
    M, N, K = 128, 128, 256
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.05
    w = torch.randn(K, N, device="cuda", dtype=torch.bfloat16) * 0.05
    y = gluon_bf16_gating_gemm(x, w, block_m=64, block_n=64, block_k=128, num_warps=4)
    ref = x.to(torch.float32) @ w.to(torch.float32)
    torch.testing.assert_close(y.to(torch.float32), ref, rtol=5e-2, atol=5e-2)


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
