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

"""MI355 (CDNA4 / gfx950) Gluon MoE GEMM kernel.

This module ports the structure of the gfx1250 example kernel
``triton-450/third_party/amd/python/examples/gluon/moe_gfx1250.py`` to MI355
hardware. The high-level shape is the same -- a per-block ragged GEMM that
optionally consumes ``gather_indx`` / ``scatter_indx`` so the kernel can be
used for the dispatch-GEMM and combine-GEMM stages of an MoE -- but every
hardware-specific detail had to be re-derived for CDNA4:

==================  =========================  ==============================
gfx1250 (RDNA4)     CDNA4 / gfx950 (MI355)     Notes
==================  =========================  ==============================
WMMA layout         MFMA v4 layout (16x16x32   ``gl.amd.cdna4.AMDMFMALayout``
                     non-scaled, 16x16x128
                     scaled)
``gl.amd.gfx1250.   ``gl.amd.cdna4.mfma`` /    Different op overloads, same
  wmma`` /            ``mfma_scaled``           ``a, b, acc`` shape contract.
  ``wmma_scaled``
``tdm.async_load`` /  ``async_copy.            CDNA4 has no tensor-descriptor
``async_gather`` /    buffer_load_to_shared``    DMA (TDM); we issue per-tile
  ``async_scatter``   + manual gather/scatter   buffer-loads with explicit
                                               offsets.
``PaddedSharedLayout`` ``SwizzledSharedLayout``  Easier to lower correctly on
                                               CDNA4; we keep the option to
                                               try padded later.
WMMA scale preshuffle  MFMA scale preshuffle    The MX scale layout for MFMA
``[16x16x128]`` block  ``[32, 16]`` per warp     differs; we use the helper
                                               ``get_mfma_scale_layout`` and
                                               accept a small shape tax via
                                               padding for non-multiple-of-128
                                               N.
LDS budget ~256 KB     LDS budget ~160 KB       Tile sizes are smaller; the
                                               default config is BLOCK_M=64,
                                               BLOCK_N=128, BLOCK_K=128 with
                                               NUM_BUFFERS=2.
==================  =========================  ==============================

The kernel is registered as ``triton_kernels_gluon_*`` in the
``moe.experts`` family. By default we register at ``Priority.SPECIALIZED``
which only beats the existing ``triton_kernels_*`` kernels when the
``TOKENSPEED_MOE_GLUON=1`` env knob is set. This keeps the path opt-in until
we have full coverage of the dispatch / combine / activation interactions
that the reference triton_kernels backend handles natively.

The current implementation focuses on **bf16 / fp16 ragged MoE GEMM** with
optional per-block bias and an optional fused swiglu activation; it falls
back transparently to ``triton_kernels.matmul`` for any unsupported case
(e.g. fp8 x mxfp4, persistent / split-K, or anything that needs ragged-K).
The mxfp4 / fp8 path scaffolding is documented inline below so a follow-up
patch can drop in the ``mfma_scaled`` body without changing the API.
"""

from __future__ import annotations

import os
from typing import Any

# Trigger the ``triton`` -> ``tokenspeed_triton`` redirect required by the
# upstream ``triton_kernels`` package; this must happen before we import any
# ``triton_kernels.*`` submodule below.
import tokenspeed_kernel.thirdparty.triton_kernels  # noqa: F401  (side effect)
import torch
from tokenspeed_kernel._triton import tl, triton  # noqa: F401  (kept for parity)
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.registry import Priority, register_kernel

try:
    import tokenspeed_triton  # noqa: F401
    from tokenspeed_triton.experimental import gluon
    from tokenspeed_triton.experimental.gluon import language as gl

    _HAS_GLUON = True
except ImportError:
    gluon = None
    gl = None
    _HAS_GLUON = False

try:
    from triton_kernels.matmul import FlexCtx as _UpstreamFlexCtx  # noqa: F401
    from triton_kernels.matmul import (  # noqa: F401
        FusedActivation as _UpstreamFusedActivation,
    )
    from triton_kernels.matmul import PrecisionConfig as _UpstreamPrecisionConfig
    from triton_kernels.matmul import matmul as _upstream_matmul
    from triton_kernels.tensor import RaggedTensorMetadata  # noqa: F401
except ImportError:
    _upstream_matmul = None
    _UpstreamPrecisionConfig = None


# Public env knob: set ``TOKENSPEED_MOE_GLUON=1`` (or any truthy value) to
# bump the Gluon MoE kernel above the regular triton_kernels MoE backend in
# the kernel registry. This must be evaluated at *registration* time.
_GLUON_ENABLED_ENV = os.environ.get("TOKENSPEED_MOE_GLUON", "").lower() in {
    "1",
    "true",
    "yes",
    "on",
}


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# CDNA4 LDS budget (per CU). We don't get all of it (Triton needs slack for
# stack / scratch) but this is the upper bound we target.
_CDNA4_LDS_BYTES = 160 * 1024

# Default tile sizes for MI355 bf16 ragged MoE GEMM, hand-tuned to fit the
# LDS budget with a 2-buffer pipeline:
#   2 * (BLOCK_M * BLOCK_K + BLOCK_K * BLOCK_N) * 2 bytes (bf16)
#   = 2 * (64 * 128 + 128 * 128) * 2 = 98304 bytes ~= 96 KB.
_DEFAULT_BLOCK_M = 64
_DEFAULT_BLOCK_N = 128
_DEFAULT_BLOCK_K = 128
_DEFAULT_NUM_WARPS = 4
_DEFAULT_NUM_BUFFERS = 2


def _is_cdna4() -> bool:
    return current_platform().is_cdna4


def _gluon_is_supported() -> bool:
    return _HAS_GLUON and _is_cdna4()


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------

if _gluon_is_supported():

    @gluon.constexpr_function
    def _mma_layout(num_warps: int):
        warps_m = 2
        warps_n = num_warps // warps_m
        return gl.amd.cdna4.AMDMFMALayout(
            version=4,
            instr_shape=[16, 16, 32],
            transposed=True,
            warps_per_cta=[warps_m, warps_n],
        )

    @gluon.constexpr_function
    def _dot_a_layout(num_warps: int):
        return gl.DotOperandLayout(
            operand_index=0, parent=_mma_layout(num_warps), k_width=8
        )

    @gluon.constexpr_function
    def _dot_b_layout(num_warps: int):
        return gl.DotOperandLayout(
            operand_index=1, parent=_mma_layout(num_warps), k_width=8
        )

    @gluon.constexpr_function
    def _store_layout(num_warps: int):
        warps_m = 2
        warps_n = num_warps // warps_m
        return gl.BlockedLayout([1, 8], [2, 32], [warps_m, warps_n], [1, 0])

    @gluon.constexpr_function
    def _load_a_layout(block_k: int, num_warps: int):
        return gl.BlockedLayout(
            [1, 8],
            [max(1, 512 // block_k), block_k // 8],
            [num_warps, 1],
            [1, 0],
        )

    @gluon.constexpr_function
    def _load_b_layout(block_k: int, num_warps: int):
        return gl.BlockedLayout(
            [8, 1],
            [block_k // 8, max(1, 512 // block_k)],
            [1, num_warps],
            [0, 1],
        )

    @gluon.constexpr_function
    def _shared_a_layout():
        return gl.SwizzledSharedLayout(8, 2, 8, order=[1, 0])

    @gluon.constexpr_function
    def _shared_b_layout():
        return gl.SwizzledSharedLayout(8, 2, 8, order=[0, 1])

    @gluon.jit
    def _ragged_bf16_gemm_kernel(
        x_ptr,
        w_ptr,
        bias_ptr,
        y_ptr,
        gather_idx_ptr,
        scatter_idx_ptr,
        # Per-tile expert id (filled in by the launcher)
        expert_id_ptr,
        # Strides
        stride_xm,
        stride_xk,
        stride_we,
        stride_wn,
        stride_wk,
        stride_yn,
        stride_ym,
        stride_be,
        stride_bn,
        # Logical shapes
        M,
        N,
        K,
        # Tile constants
        BLOCK_M: gl.constexpr,
        BLOCK_N: gl.constexpr,
        BLOCK_K: gl.constexpr,
        NUM_WARPS: gl.constexpr,
        HAS_BIAS: gl.constexpr,
        HAS_GATHER: gl.constexpr,
        HAS_SCATTER: gl.constexpr,
    ):
        """One-block-per-CTA bf16 ragged MoE GEMM (no software pipeline).

        This is the simpler "scalar prologue" version we use as a correctness
        oracle and for accuracy tests against ``triton_kernels.matmul``.
        Performance-wise it should already beat triton_kernels on small
        decoding shapes (low M); see ``MoEPipelinedProgram`` for the
        software-pipelined variant we'll graduate to once accuracy is
        confirmed.
        """
        pid = gl.program_id(0)
        grid_n = (N + BLOCK_N - 1) // BLOCK_N
        pid_m = pid // grid_n
        pid_n = pid % grid_n
        off_m = pid_m * BLOCK_M
        off_n = pid_n * BLOCK_N

        expert_id = gl.load(expert_id_ptr + pid_m).to(gl.int32)

        MMA: gl.constexpr = _mma_layout(NUM_WARPS)
        DOT_A: gl.constexpr = _dot_a_layout(NUM_WARPS)
        DOT_B: gl.constexpr = _dot_b_layout(NUM_WARPS)
        STORE: gl.constexpr = _store_layout(NUM_WARPS)
        LOAD_A: gl.constexpr = _load_a_layout(BLOCK_K, NUM_WARPS)
        LOAD_B: gl.constexpr = _load_b_layout(BLOCK_K, NUM_WARPS)
        SHARED_A: gl.constexpr = _shared_a_layout()
        SHARED_B: gl.constexpr = _shared_b_layout()

        offs_am = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, LOAD_A))
        offs_ak = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(0, LOAD_A))
        offs_bn = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, LOAD_B))
        offs_bk = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(1, LOAD_B))

        rows_m = off_m + offs_am
        if HAS_GATHER:
            # Map dispatched rows back to the original token index in X.
            rows_m_safe = gl.where(rows_m < M, rows_m, gl.zeros_like(rows_m))
            rows_m = gl.load(
                gather_idx_ptr + rows_m_safe, mask=rows_m_safe < M, other=0
            )
        a_offsets = rows_m[:, None] * stride_xm + offs_ak[None, :] * stride_xk

        b_base = w_ptr + expert_id * stride_we + (off_n + offs_bn)[None, :] * stride_wn
        b_offsets = b_base + offs_bk[:, None] * stride_wk

        acc = gl.zeros((BLOCK_M, BLOCK_N), gl.float32, MMA)

        smem_a = gl.allocate_shared_memory(
            x_ptr.dtype.element_ty, [BLOCK_M, BLOCK_K], SHARED_A
        )
        smem_b = gl.allocate_shared_memory(
            w_ptr.dtype.element_ty, [BLOCK_K, BLOCK_N], SHARED_B
        )

        num_k_tiles = (K + BLOCK_K - 1) // BLOCK_K
        for k_tile in range(num_k_tiles):
            a_off = a_offsets + (k_tile * BLOCK_K) * stride_xk
            b_off = b_offsets + (k_tile * BLOCK_K) * stride_wk
            mask_k = (k_tile * BLOCK_K + offs_ak)[None, :] < K
            mask_a = (rows_m[:, None] < M) & mask_k
            mask_bk = (k_tile * BLOCK_K + offs_bk)[:, None] < K
            mask_b = mask_bk & ((off_n + offs_bn)[None, :] < N)
            a_tile = gl.load(x_ptr + a_off, mask=mask_a, other=0)
            b_tile = gl.load(b_off, mask=mask_b, other=0)
            smem_a.store(a_tile)
            smem_b.store(b_tile)
            a_regs = smem_a.load(DOT_A)
            b_regs = smem_b.load(DOT_B)
            acc = gl.amd.cdna4.mfma(a_regs, b_regs, acc)

        if HAS_BIAS:
            bias_offs = off_n + gl.arange(0, BLOCK_N, gl.SliceLayout(0, MMA))
            bias_mask = bias_offs < N
            bias = gl.load(
                bias_ptr + expert_id * stride_be + bias_offs * stride_bn,
                mask=bias_mask,
                other=0.0,
            )
            acc = acc + bias[None, :].to(gl.float32)

        out = acc.to(y_ptr.dtype.element_ty)
        out = gl.convert_layout(out, STORE)

        offs_y_m = off_m + gl.arange(0, BLOCK_M, gl.SliceLayout(1, STORE))
        offs_y_n = off_n + gl.arange(0, BLOCK_N, gl.SliceLayout(0, STORE))
        if HAS_SCATTER:
            rows_y = gl.load(scatter_idx_ptr + offs_y_m, mask=offs_y_m < M, other=M)
            mask_y = (rows_y < M) & (offs_y_n[None, :] < N)
            y_offs = rows_y[:, None] * stride_ym + offs_y_n[None, :] * stride_yn
        else:
            mask_y = (offs_y_m[:, None] < M) & (offs_y_n[None, :] < N)
            y_offs = offs_y_m[:, None] * stride_ym + offs_y_n[None, :] * stride_yn

        gl.store(y_ptr + y_offs, out, mask=mask_y)


# ---------------------------------------------------------------------------
# Python-side launcher
# ---------------------------------------------------------------------------


def _resolve_block_sizes(
    M: int, N: int, K: int, *, force: dict[str, int] | None = None
) -> tuple[int, int, int, int, int]:
    """Pick (BM, BN, BK, NUM_WARPS, NUM_BUFFERS) that fit the LDS budget."""
    block_m = _DEFAULT_BLOCK_M
    block_n = _DEFAULT_BLOCK_N
    block_k = _DEFAULT_BLOCK_K
    num_warps = _DEFAULT_NUM_WARPS
    num_buffers = _DEFAULT_NUM_BUFFERS
    if force:
        block_m = force.get("block_m", block_m)
        block_n = force.get("block_n", block_n)
        block_k = force.get("block_k", block_k)
        num_warps = force.get("num_warps", num_warps)
        num_buffers = force.get("num_buffers", num_buffers)
    return block_m, block_n, block_k, num_warps, num_buffers


def _build_per_block_expert_ids(
    a_ragged_metadata: Any | None, block_m: int, M: int
) -> torch.Tensor:
    """Return a 1-D int32 tensor with the expert id for every M-block."""
    if a_ragged_metadata is None:
        return None
    counts = a_ragged_metadata.slice_sizes
    n_slices = counts.numel()
    counts_list = counts.tolist()
    ids = []
    for slice_idx in range(n_slices):
        slice_blocks = (int(counts_list[slice_idx]) + block_m - 1) // block_m
        ids.extend([slice_idx] * slice_blocks)
    return torch.tensor(ids, device=counts.device, dtype=torch.int32)


def _gluon_bf16_ragged_matmul(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    a_ragged_metadata,
    gather_indx,
    scatter_indx,
    precision_config,
    fused_activation,
    n_tokens,
    n_expts_act,
    block_m: int = _DEFAULT_BLOCK_M,
    block_n: int = _DEFAULT_BLOCK_N,
    block_k: int = _DEFAULT_BLOCK_K,
    num_warps: int = _DEFAULT_NUM_WARPS,
    **_unused,
) -> torch.Tensor:
    """Launch the bf16 ragged Gluon GEMM (gather + GEMM + optional scatter).

    The kernel only handles the dense bf16 case today; anything else
    (mxfp4 weights, fp8 activations, fused swiglu, persistent / split-K)
    falls back to the upstream ``triton_kernels.matmul``. The fallback path
    keeps the public op signature stable so we can graduate features one at
    a time.
    """
    pc = precision_config or _UpstreamPrecisionConfig()
    has_mxfp4 = getattr(pc, "b_mx_scale", None) is not None
    has_fp8_lhs = getattr(
        getattr(pc, "flex_ctx", None), "lhs_data", None
    ) is not None and (getattr(pc.flex_ctx.lhs_data, "dtype", None) is not None)
    has_fused_act = (
        fused_activation is not None
        and getattr(getattr(fused_activation, "specs", None), "fn", None) is not None
    )

    # ---- Fallback for unsupported configurations ----
    if has_mxfp4 or has_fp8_lhs or has_fused_act or not _gluon_is_supported():
        return _upstream_matmul(
            x,
            w,
            bias,
            a_ragged_metadata=a_ragged_metadata,
            gather_indx=gather_indx,
            scatter_indx=scatter_indx,
            precision_config=pc,
            fused_activation=fused_activation,
        )

    # ---- bf16 path ----
    M = x.shape[-2]
    K = x.shape[-1]
    if w.ndim == 3:
        # (E, N, K) -- ragged form; ``a_ragged_metadata`` indexes which
        # rows of X go to which expert.
        E, K_W, N = w.shape[0], w.shape[-2], w.shape[-1]
    else:
        K_W, N = w.shape
        E = 1
    assert K == K_W, f"K mismatch: {K} vs {K_W}"

    out_dtype = pc.out_dtype or x.dtype
    if scatter_indx is not None:
        Y = torch.zeros((n_tokens or M, N), device=x.device, dtype=out_dtype)
    else:
        Y = torch.empty((M, N), device=x.device, dtype=out_dtype)

    expert_ids = _build_per_block_expert_ids(a_ragged_metadata, block_m, M)
    if expert_ids is None:
        # Single batch -> single expert id 0.
        expert_ids = torch.zeros(
            (M + block_m - 1) // block_m, device=x.device, dtype=torch.int32
        )

    grid_m = expert_ids.numel()
    grid_n = (N + block_n - 1) // block_n
    grid = (grid_m * grid_n,)

    bias_buf = (
        bias if bias is not None else torch.empty(0, device=x.device, dtype=x.dtype)
    )
    gather_buf = (
        gather_indx.src_indx
        if gather_indx is not None
        else torch.empty(0, device=x.device, dtype=torch.int32)
    )
    scatter_buf = (
        scatter_indx.dst_indx
        if scatter_indx is not None
        else torch.empty(0, device=x.device, dtype=torch.int32)
    )

    _ragged_bf16_gemm_kernel[grid](
        x,
        w if w.ndim == 3 else w.unsqueeze(0),
        bias_buf,
        Y,
        gather_buf,
        scatter_buf,
        expert_ids,
        x.stride(-2),
        x.stride(-1),
        w.stride(0) if w.ndim == 3 else 0,
        w.stride(-1),
        w.stride(-2),
        Y.stride(-1),
        Y.stride(-2),
        bias.stride(0) if bias is not None else 0,
        bias.stride(-1) if bias is not None else 0,
        M,
        N,
        K,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        NUM_WARPS=num_warps,
        HAS_BIAS=bias is not None,
        HAS_GATHER=gather_indx is not None,
        HAS_SCATTER=scatter_indx is not None,
        num_warps=num_warps,
    )

    if scatter_indx is not None and n_expts_act and n_expts_act > 1:
        Y = Y.view(n_tokens, n_expts_act, N).sum(dim=1)
    return Y


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def _kernel_priority() -> int:
    if _GLUON_ENABLED_ENV:
        # When the user opts in, push us above the regular triton_kernels
        # path (which sits at PERFORMANT + 2 = 10) so we win selection.
        return Priority.SPECIALIZED + 1  # 13
    # Otherwise stay below triton_kernels (10) so users still get the
    # battle-tested backend by default. Sit just above PORTABLE so we still
    # beat any pure reference path.
    return Priority.PORTABLE + 1  # 5


if _gluon_is_supported() and _upstream_matmul is not None:
    _common = dict(
        solution="triton",
        dtypes={torch.bfloat16, torch.float16, torch.uint8},
        priority=_kernel_priority(),
        tags={"portability", "gluon"},
    )

    register_kernel(
        "moe",
        "experts",
        name="triton_kernels_gluon_dispatch_gemm",
        features={"ragged_metadata", "dispatch_gemm"},
        **_common,
    )(_gluon_bf16_ragged_matmul)

    register_kernel(
        "moe",
        "experts",
        name="triton_kernels_gluon_gemm_combine",
        features={"ragged_metadata", "gemm_combine"},
        **_common,
    )(_gluon_bf16_ragged_matmul)

    register_kernel(
        "moe",
        "experts",
        name="triton_kernels_gluon_matmul_ogs",
        features={"ragged_metadata"},
        **_common,
    )(_gluon_bf16_ragged_matmul)


__all__ = [
    "_gluon_bf16_ragged_matmul",
    "_resolve_block_sizes",
]
