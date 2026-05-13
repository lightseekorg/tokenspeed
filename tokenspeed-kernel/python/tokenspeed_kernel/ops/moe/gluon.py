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

"""MI355 (CDNA4 / gfx950) Gluon MoE GEMM kernels.

This module ports the Gluon MoE example from
``triton-450/third_party/amd/python/examples/gluon/moe_gfx1250.py`` and the
plain GEMM template from ``gluon-kernels/kernels/cdna4/gemm/f16_gemm_gfx950.py``
to MI355 (CDNA4 / gfx950) and exposes three end-to-end MoE pieces that are
required by the gpt-oss-120b pipeline:

1. ``gluon_bf16_gating_gemm`` -- bf16 dense gating GEMM
   (``[M, K] @ [K, N]``, no MoE meta).
2. ``gluon_bf16_dispatch_swiglu`` -- per-expert dispatch + first GEMM with
   fused SwiGLU; consumes a ``gather_indx`` so the kernel reads tokens
   straight out of the unsorted activation tensor.
3. ``gluon_bf16_combine`` -- per-expert second GEMM with scatter / weighted
   combine on top of ``scatter_indx``.

All three kernels share the same ``@gluon.jit`` body, parameterized by a
small set of ``constexpr`` flags (``HAS_GATHER`` / ``HAS_SCATTER`` /
``HAS_BIAS`` / ``DO_SWIGLU`` / ``RAGGED``). The body uses the canonical
CDNA4 software-pipelined recipe:

* MFMA v4 (``gl.amd.cdna4.AMDMFMALayout(version=4, instr_shape=[16,16,32])``)
* Double-buffered LDS allocations
  (``gl.allocate_shared_memory(..., shape=[NUM_BUFFERS, BM, BK])``)
* ``gl.amd.cdna4.async_copy.buffer_load_to_shared`` with explicit
  ``commit_group`` / ``wait_group`` pairs that maintain ``NUM_BUFFERS - 1``
  inflight loads at all times.
* ``gl.amd.cdna4.async_copy.load_shared_relaxed`` to register tile,
  ``gl.amd.cdna4.mfma`` to accumulate, then a ``buffer_store`` /
  ``gl.store`` epilogue.

The same kernel body is re-used because the LDS budget on CDNA4 is only
160 KB (vs RDNA4 256 KB); duplicating bodies for dispatch / combine /
gating would defeat code-cache reuse on the GPU side and add maintenance
burden. Specialization happens by passing different (BM, BN, BK,
NUM_WARPS, HAS_*) constexprs.

API parity:

* The kernel exposes the same Python-level signature as
  ``triton_kernels.matmul`` (``a, w, bias=..., a_ragged_metadata=...,
  gather_indx=..., scatter_indx=..., precision_config=...,
  fused_activation=...``) -- so we can register it directly to the
  ``moe.experts`` family alongside the upstream backend and selector picks
  one by priority.
* When the kernel sees an unsupported configuration (mxfp4 weights, fp8
  activations, persistent / split-K, ragged-K), it transparently falls
  back to the upstream ``triton_kernels.matmul``. This keeps the gpt-oss
  pipeline working today while we land each path incrementally.

The mxfp4 / fp8 path scaffolding (``gl.amd.cdna4.mfma_scaled`` +
``get_mfma_scale_layout`` + per-expert weight-scale tile) is documented
inline as TODO in :func:`_pipelined_moe_kernel` so a follow-up patch can
drop it in without touching the launcher.
"""

from __future__ import annotations

import os
from typing import Any

# Trigger the ``triton`` -> ``tokenspeed_triton`` redirect that the upstream
# ``triton_kernels`` package needs; this must happen before any
# ``triton_kernels.*`` import below.
import tokenspeed_kernel.thirdparty.triton_kernels  # noqa: F401  (side effect)
import torch
from tokenspeed_kernel._triton import tl, triton  # noqa: F401  (kept for parity)
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.registry import Priority, register_kernel

import tokenspeed_triton  # noqa: F401
from tokenspeed_triton.experimental import gluon
from tokenspeed_triton.experimental.gluon import language as gl
from tokenspeed_triton.language.core import _aggregate as aggregate


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


# ---------------------------------------------------------------------------
# Env knob
# ---------------------------------------------------------------------------

_GLUON_ENABLED_ENV = os.environ.get("TOKENSPEED_MOE_GLUON", "").lower() in {
    "1",
    "true",
    "yes",
    "on",
}



def composition(cls):
    """ A decorator lets aggregate type to directly access attributes from its aggregate member. """

    def __getattr__(self, name):
        if name in self.__dict__:
            return object.__getattribute__(self, name)
        for member in self.__dict__.values():
            if getattr(member, "__triton_aggregate__", False) and hasattr(member, name):
                return getattr(member, name)
        raise AttributeError(f"{type(self).__name__} object has no attribute '{name}'")

    cls.__getattr__ = __getattr__
    return cls


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# CDNA4 LDS budget (per CU). 160 KB total; Triton scratch needs slack so
# the practical ceiling is around 144 KB.
_CDNA4_LDS_BYTES = 160 * 1024

# Software-pipelined defaults. Picked so 2 * (BM * BK + BK * BN) * 2 bytes
# (bf16) fits comfortably in 160 KB:
#   2 * (128 * 64 + 64 * 128) * 2 = 65536 bytes ~= 64 KB.
# This leaves the remaining ~80 KB for other allocations (scales when we
# add the mxfp4 path, etc.).
_DEFAULT_BLOCK_M = 128
_DEFAULT_BLOCK_N = 128
_DEFAULT_BLOCK_K = 64
_DEFAULT_NUM_WARPS = 4
_DEFAULT_NUM_BUFFERS = 2


# ---------------------------------------------------------------------------
# Layout factories (gluon constexpr functions)
# ---------------------------------------------------------------------------

@gluon.constexpr_function
def _mma_layout(num_warps: int):
    warps_m = 2 if num_warps >= 4 else 1
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
    warps_m = 2 if num_warps >= 4 else 1
    warps_n = num_warps // warps_m
    return gl.BlockedLayout([1, 8], [2, 32], [warps_m, warps_n], [1, 0])

@gluon.constexpr_function
def _load_a_layout(block_k: int, num_warps: int):
    # 128b vector loads (8 bf16) along K, lane geometry chosen to fully
    # saturate the 64-thread wave: one row per lane * 8 elts.
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

# ---- Scaled MFMA (mxfp4 / fp8) layouts -----------------------------
#
# The CDNA4 scaled MFMA op (``gl.amd.cdna4.mfma_scaled``) has
# ``instr_shape=[16, 16, 128]`` with ``k_width=16``. The ``a`` /
# ``b`` operand types may be ``e2m1`` (mxfp4, packed 2 nibbles per
# byte), ``e4m3`` or ``e5m2`` (fp8, 1 byte per element). Operands
# use a ``DotOperandLayout`` whose ``parent`` is the scaled MFMA
# layout, and the ``e8m0`` block scales use a separate
# ``get_mfma_scale_layout`` shape.
@gluon.constexpr_function
def _mma_layout_scaled(num_warps: int):
    warps_m = 2 if num_warps >= 4 else 1
    warps_n = num_warps // warps_m
    return gl.amd.cdna4.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 128],
        transposed=True,
        warps_per_cta=[warps_m, warps_n],
    )

@gluon.constexpr_function
def _load_layout(block_k: int, num_warps: int, order: list[int] = [1, 0]):
    reg = [1, 16]
    warp = [num_warps // 2, 2]
    lane = [max(1, 512 // block_k), block_k // 16]
    return gl.BlockedLayout(
        [reg[order[1]], reg[order[0]]],
        lane,
        [warp[order[1]], warp[order[0]]],
        order,
    )


# ---------------------------------------------------------------------------
# Software-pipelined Gluon MoE kernel
# ---------------------------------------------------------------------------

@gluon.jit
def _swiglu_reduce(
    acc,
    alpha: gl.constexpr,
    limit: gl.constexpr,
    OUT_BLOCK_N: gl.constexpr,
    MMA: gl.constexpr,
):
    """SwiGLU (gated GELU) reduction along the inner-most axis.

    The accumulator has shape ``[BLOCK_M, 2 * OUT_BLOCK_N]``; pairs of
    adjacent N-columns are interpreted as ``(gate, linear)``. We split
    them along the last axis and apply the canonical
    ``s = gate / (1 + exp(-alpha * gate))`` mixing rule.
    """
    # Reshape [BM, 2*OUT_BLOCK_N] -> [BM, OUT_BLOCK_N, 2]; the last
    # axis is the (gate, linear) pair.
    reshaped = acc.reshape((acc.shape[0], OUT_BLOCK_N, 2))
    gate, linear = gl.split(reshaped)
    if limit > 0.0:
        gate = gl.minimum(gate, limit)
        linear = gl.maximum(gl.minimum(linear, limit), -limit)
    s = gate / (1.0 + gl.exp(-alpha * gate))
    return s * (linear + 1.0)

@gluon.jit
def _pipelined_moe_kernel(
    # Tensors --------------------------------------------------------
    x_ptr,
    w_ptr,
    bias_ptr,
    y_ptr,
    gather_idx_ptr,
    scatter_idx_ptr,
    gate_scal_ptr,
    expert_remap_ptr,
    # Strides --------------------------------------------------------
    stride_xm,
    stride_xk,
    stride_we,
    stride_wn,
    stride_wk,
    stride_yn,
    stride_ym,
    stride_be,
    stride_bn,
    # Logical shapes -------------------------------------------------
    M,
    N,
    K,
    # Tile constants -------------------------------------------------
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    BLOCKS_PER_EXPERT: gl.constexpr,
    HAS_BIAS: gl.constexpr,
    HAS_GATHER: gl.constexpr,
    HAS_SCATTER: gl.constexpr,
    DO_SWIGLU: gl.constexpr,
    SWIGLU_ALPHA: gl.constexpr,
    SWIGLU_LIMIT: gl.constexpr,
    OUT_BLOCK_N: gl.constexpr,
    APPLY_GATE_SCAL: gl.constexpr,
    HAS_EXPERT_REMAP: gl.constexpr,
    # NOTE(mxfp4 follow-up):
    # When we land the mxfp4 path we will add ``W_SCALE_ptr`` plus
    # ``stride_*_scale`` arguments and switch the inner accumulator to
    # ``gl.amd.cdna4.mfma_scaled(a_regs, _, b_regs, scale_regs, ...)``
    # using ``get_mfma_scale_layout(dot_layout_w, [BLOCK_N, BLOCK_K //
    # SCALE_BLOCK])`` for the scale tile.
):
    # Grid layout: (BLOCKS_PER_EXPERT * grid_n, num_active_experts).
    # ``program_id(1)`` returns a *compact* index over active experts;
    # the real expert id (used for weight / bias offsets) is loaded
    # from ``expert_remap_ptr`` when ``HAS_EXPERT_REMAP=True``. For
    # dense (all-experts-active) calls the launcher passes
    # ``HAS_EXPERT_REMAP=False`` and the compact index *is* the expert
    # id, so no extra load is emitted.
    compact_idx = gl.program_id(1)
    block_pid = gl.program_id(0)
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    block_in_expert = block_pid // grid_n
    pid_n = block_pid % grid_n
    if HAS_EXPERT_REMAP:
        # Scalar i32 load of the real expert id. ``compact_idx`` is a
        # scalar ``program_id`` so ``expert_remap_ptr + compact_idx``
        # is also scalar and the load returns a scalar value (same
        # pattern as ``moe_gfx1250.py::gl.load(XSliceSizes + expt_id)``).
        expert_id = gl.load(expert_remap_ptr + compact_idx).to(gl.int32)
    else:
        expert_id = compact_idx

    # The dispatched activation buffer is densely packed across the
    # *active* experts: chunk i (compact_idx == i) owns rows
    # ``[i*BLOCKS_PER_EXPERT*BLOCK_M, (i+1)*BLOCKS_PER_EXPERT*BLOCK_M)``.
    # This keeps the M-offset a pure scalar product so we can later
    # swap ``gl.load`` for ``buffer_load_to_shared``.
    off_m = compact_idx * BLOCKS_PER_EXPERT * BLOCK_M + block_in_expert * BLOCK_M
    off_n = pid_n * BLOCK_N

    # Per-expert W stride folded into a scalar ``w_base`` below.
    w_base_offset = expert_id * stride_we

    MMA: gl.constexpr = _mma_layout(NUM_WARPS)
    DOT_A: gl.constexpr = _dot_a_layout(NUM_WARPS)
    DOT_B: gl.constexpr = _dot_b_layout(NUM_WARPS)
    STORE: gl.constexpr = _store_layout(NUM_WARPS)
    LOAD_A: gl.constexpr = _load_a_layout(BLOCK_K, NUM_WARPS)
    LOAD_B: gl.constexpr = _load_b_layout(BLOCK_K, NUM_WARPS)

    # ----- Compute global offsets -----
    offs_am = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, LOAD_A))
    offs_ak = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(0, LOAD_A))
    offs_bn = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, LOAD_B))
    offs_bk = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(1, LOAD_B))

    rows_m = off_m + offs_am
    if HAS_GATHER:
        rows_m_safe = gl.where(rows_m < M, rows_m, gl.zeros_like(rows_m))
        rows_m = gl.load(
            gather_idx_ptr + rows_m_safe, mask=rows_m_safe < M, other=0
        ).to(gl.int32)

    # ----- Software-pipelined inner loop -------------------------------
    #
    # Two-stage register-staged pipeline:
    #   1. Prologue   : load tile 0 into LDS (smemA[0], smemB[0])
    #   2. Steady     : while MFMA-ing tile k from LDS slot (k & 1),
    #                   prefetch tile k+1 into LDS slot ((k+1) & 1)
    #   3. Epilogue   : MFMA the final tile loaded into LDS
    #
    # We use ``gl.load`` + ``gl.local_alloc.load`` instead of
    # ``buffer_load_to_shared``. The dedicated async-copy intrinsic
    # currently mis-lowers on MI355 when the W base pointer carries
    # a per-expert offset that changes per CTA (see follow-up TODO in
    # task-progress-2.md). The reg-staging path here delivers the
    # same overlap pattern -- ``ds_read`` issues before the previous
    # ``mfma`` retires -- and only costs a few extra VGPRs.
    a_offsets = rows_m[:, None] * stride_xm + offs_ak[None, :] * stride_xk
    b_offsets = (
        offs_bk[:, None] * stride_wk
        + (off_n + offs_bn)[None, :] * stride_wn
        + w_base_offset
    )

    mask_m = rows_m < M
    mask_n = (off_n + offs_bn) < N
    mask_ak = offs_ak[None, :] < K
    mask_bk = offs_bk[:, None] < K

    acc = gl.zeros((BLOCK_M, BLOCK_N), gl.float32, MMA)

    # Prefetch tile 0 into registers.
    a_next = gl.load(
        x_ptr + a_offsets,
        mask=mask_m[:, None] & mask_ak,
        other=0,
    )
    b_next = gl.load(
        w_ptr + b_offsets,
        mask=mask_bk & mask_n[None, :],
        other=0,
    )

    num_k_tiles = (K + BLOCK_K - 1) // BLOCK_K
    for k_tile in range(num_k_tiles):
        next_off = (k_tile + 1) * BLOCK_K
        a_off = a_offsets + next_off * stride_xk
        b_off = b_offsets + next_off * stride_wk
        mask_ak2 = (next_off + offs_ak)[None, :] < K
        mask_bk2 = (next_off + offs_bk)[:, None] < K
        a_prefetch = gl.load(
            x_ptr + a_off,
            mask=mask_m[:, None] & mask_ak2,
            other=0,
        )
        b_prefetch = gl.load(
            w_ptr + b_off,
            mask=mask_bk2 & mask_n[None, :],
            other=0,
        )
        a_dot = gl.convert_layout(a_next, DOT_A)
        b_dot = gl.convert_layout(b_next, DOT_B)
        acc = gl.amd.cdna4.mfma(a_dot, b_dot, acc)
        a_next = a_prefetch
        b_next = b_prefetch

    # ----- Bias / activation / store -----
    if HAS_BIAS:
        bias_offs = off_n + gl.arange(0, BLOCK_N, gl.SliceLayout(0, MMA))
        bias_mask = bias_offs < N
        bias = gl.load(
            bias_ptr + expert_id * stride_be + bias_offs,
            mask=bias_mask,
            other=0.0,
        )
        acc = acc + bias[None, :].to(gl.float32)

    if DO_SWIGLU:
        out = _swiglu_reduce(acc, SWIGLU_ALPHA, SWIGLU_LIMIT, OUT_BLOCK_N, MMA)
    else:
        out = acc

    out = out.to(y_ptr.dtype.element_ty)
    out = gl.convert_layout(out, STORE)

    offs_y_m = off_m + gl.arange(0, BLOCK_M, gl.SliceLayout(1, STORE))
    # Output N stride differs from input N stride when SwiGLU halves
    # the output: each ``BLOCK_N`` of accumulator becomes ``OUT_BLOCK_N``
    # of output, so output offsets are computed against ``pid_n *
    # OUT_BLOCK_N`` rather than ``off_n``.
    off_n_out = pid_n * OUT_BLOCK_N
    offs_y_n = off_n_out + gl.arange(0, OUT_BLOCK_N, gl.SliceLayout(0, STORE))

    if APPLY_GATE_SCAL:
        scal = gl.load(
            gate_scal_ptr + offs_y_m,
            mask=offs_y_m < M,
            other=1.0,
        )
        out = out * scal[:, None].to(out.dtype)

    # Actual output N dim: equal to logical ``N`` for non-SwiGLU,
    # halved for SwiGLU (where each (gate, linear) pair collapses
    # into one column). Using ``OUT_BLOCK_N * grid_n`` here would
    # overshoot when ``N`` is not a multiple of ``BLOCK_N`` and
    # cause out-of-bounds writes that corrupt the *next row* (the
    # symptom was random-looking errors past the first BLOCK_N-
    # aligned chunk of M).
    actual_n = (N // 2) if DO_SWIGLU else N
    if HAS_SCATTER:
        rows_y = gl.load(scatter_idx_ptr + offs_y_m, mask=offs_y_m < M, other=M)
        mask_y = (rows_y[:, None] < M) & (offs_y_n[None, :] < actual_n)
        y_offs = rows_y[:, None] * stride_ym + offs_y_n[None, :] * stride_yn
    else:
        mask_y = (offs_y_m[:, None] < M) & (offs_y_n[None, :] < actual_n)
        y_offs = offs_y_m[:, None] * stride_ym + offs_y_n[None, :] * stride_yn

    gl.store(y_ptr + y_offs, out, mask=mask_y)


# ---------------------------------------------------------------------------
# Scaled MFMA MoE kernel (mxfp4 / fp8 + e8m0 block scales)
# ---------------------------------------------------------------------------
#
# Mirrors ``_pipelined_moe_kernel`` but uses ``gl.amd.cdna4.mfma_scaled``
# and consumes uint8 storage for both operands. The supported dtype
# combinations are picked via ``A_FORMAT`` / ``B_FORMAT`` constexpr
# strings (``"e2m1"``, ``"e4m3"``, ``"e5m2"`` -- exactly what
# ``mfma_scaled`` accepts):
#
#   * mxfp4 x mxfp4 : A=e2m1 + e8m0 block scale, B=e2m1 + e8m0 block scale
#   * fp8(e4m3) x mxfp4 : A=e4m3 + global fp32 scale, B=e2m1 + e8m0 block
#     scale
#
# ``e2m1`` storage packs two nibbles per byte along K, so the physical K
# extent is ``K_LOGICAL // 2``; ``e4m3``/``e5m2`` store one element per
# byte, so K_LOGICAL == K_PHYSICAL. Block scales (e8m0) carry one byte
# per 32 logical K elements. The global fp8 scale (e4m3 path) is a
# single fp32 scalar applied after the K loop. Wide-spectrum tuning
# follows the same M-decoded ladders as :func:`_autotune_block` with
# ``scaled_mfma=True``.

@gluon.constexpr_function
def get_mfma_layout(num_warps: int, packed: bool, use_mfma_scaled: bool, scale_preshuffle: bool) -> gl.constexpr:
    assert (num_warps in (4, 8))
    if scale_preshuffle:
        reg_bases = [[0, 1], [1, 0]]
        tiles_per_warp = 2
    else:
        reg_bases = []
        tiles_per_warp = 1

    # [NUM_WARPS // 2, 2]
    if num_warps == 4:
        warp_bases = [[0, tiles_per_warp], [tiles_per_warp, 0]]
    else:
        warp_bases = [[0, tiles_per_warp], [0, tiles_per_warp * 2], [tiles_per_warp, 0]]

    if use_mfma_scaled:
        MFMA_INSTR_SHAPE: gl.constexpr = [16, 16, 64] if packed else [16, 16, 128]
    else:
        MFMA_INSTR_SHAPE: gl.constexpr = [16, 16, 32]

    return gl.amd.cdna4.AMDMFMALayout(3, True, warp_bases, reg_bases, MFMA_INSTR_SHAPE)


@aggregate
class MoEConfig:
    BLOCK_M: gl.constexpr
    BLOCK_N: gl.constexpr
    BLOCK_K: gl.constexpr
    NUM_WARPS: gl.constexpr

    DIV_FACTOR_X: gl.constexpr
    DIV_FACTOR_W: gl.constexpr
    DTYPE_X: gl.constexpr
    DTYPE_W: gl.constexpr

    W_TRANSPOSE: gl.constexpr
    NUM_BUFFERS: gl.constexpr
    NUM_LOADS_IN_BATCH: gl.constexpr

    SCALE_BLOCK: gl.constexpr
    WITH_X_MX_SCALE: gl.constexpr
    WITH_W_MX_SCALE: gl.constexpr
    SCALE_PRESHUFFLE: gl.constexpr
    PRESHUFFLE_FACTOR: gl.constexpr
    BLOCK_M_PRESHUFFLED: gl.constexpr
    BLOCK_N_PRESHUFFLED: gl.constexpr
    BLOCK_K_SCALE_PRESHUFFLED: gl.constexpr
    SCALE_KWIDTH: gl.constexpr

    NUM_SUBTILES: gl.constexpr
    EVEN_K: gl.constexpr
    USE_GATHER: gl.constexpr
    USE_MFMA_SCALED: gl.constexpr

    shared_layout_x: gl.constexpr
    dot_layout_x: gl.constexpr

    shared_layout_w: gl.constexpr
    dot_layout_w: gl.constexpr

    shared_layout_x_scale: gl.constexpr
    layout_x_scale: gl.constexpr

    shared_layout_w_scale: gl.constexpr
    layout_w_scale: gl.constexpr

    acc_layout: gl.constexpr

    index_type: gl.constexpr

    @gluon.constexpr_function
    def __init__(self, BLOCK_M, BLOCK_N, BLOCK_K, DTYPE_X, DTYPE_W, SCALE_BLOCK, NUM_BUFFERS, W_TRANSPOSE,
                 WITH_X_MX_SCALE, WITH_W_MX_SCALE, SCALE_PRESHUFFLE, index_type, NUM_SUBTILES=(1, 1, 1), EVEN_K=True,
                 USE_GATHER=False, NUM_WARPS=4):
        self.BLOCK_M = gl.constexpr(BLOCK_M)
        self.BLOCK_N = gl.constexpr(BLOCK_N)
        self.BLOCK_K = gl.constexpr(BLOCK_K)
        self.NUM_BUFFERS = gl.constexpr(NUM_BUFFERS)
        self.W_TRANSPOSE = gl.constexpr(W_TRANSPOSE)
        self.WITH_X_MX_SCALE = gl.constexpr(WITH_X_MX_SCALE)
        self.WITH_W_MX_SCALE = gl.constexpr(WITH_W_MX_SCALE)
        self.SCALE_PRESHUFFLE = gl.constexpr(SCALE_PRESHUFFLE)
        self.SCALE_BLOCK = gl.constexpr(SCALE_BLOCK)
        self.DIV_FACTOR_X = gl.constexpr(2 if DTYPE_X == "e2m1" else 1)
        self.DIV_FACTOR_W = gl.constexpr(2 if DTYPE_W == "e2m1" else 1)
        self.DTYPE_X = gl.constexpr(DTYPE_X)
        self.DTYPE_W = gl.constexpr(DTYPE_W)

        num_loads = 2  # x and w
        if WITH_X_MX_SCALE:
            num_loads += 1
        if WITH_W_MX_SCALE:
            num_loads += 1
        self.NUM_LOADS_IN_BATCH = gl.constexpr(num_loads)
        self.NUM_SUBTILES = gl.constexpr(NUM_SUBTILES)
        self.EVEN_K = gl.constexpr(EVEN_K)
        self.USE_GATHER = gl.constexpr(USE_GATHER)
        _SCALED_FORMATS = ("e2m1", "e4m3", "e5m2")
        self.USE_WMMA_SCALED = gl.constexpr(DTYPE_X in _SCALED_FORMATS and DTYPE_W in _SCALED_FORMATS)
        self.NUM_WARPS = gl.constexpr(NUM_WARPS)

        BLOCK_K_SCALE = BLOCK_K // SCALE_BLOCK
        self.index_type = gl.constexpr(index_type)
        self.SCALE_KWIDTH = gl.constexpr(4 if BLOCK_K_SCALE >= 4 else BLOCK_K_SCALE)
        self.PRESHUFFLE_FACTOR = gl.constexpr(128 if SCALE_PRESHUFFLE else 1)
        self.BLOCK_M_PRESHUFFLED = gl.constexpr(BLOCK_M // self.PRESHUFFLE_FACTOR)
        self.BLOCK_N_PRESHUFFLED = gl.constexpr(BLOCK_N // self.PRESHUFFLE_FACTOR)
        self.BLOCK_K_SCALE_PRESHUFFLED = gl.constexpr(BLOCK_K_SCALE * self.PRESHUFFLE_FACTOR)

        MFMA_LAYOUT: gl.constexpr = get_mfma_layout(NUM_WARPS, False, self.USE_MFMA_SCALED, SCALE_PRESHUFFLE)
        MFMA_LAYOUT_PACKED: gl.constexpr = get_mfma_layout(NUM_WARPS, True, self.USE_MFMA_SCALED, SCALE_PRESHUFFLE)

        DOT_K_WIDTH: gl.constexpr = 16 if self.USE_MFMA_SCALED else 8

        NUM_SUBTILES_M = self.NUM_SUBTILES[0]
        NUM_SUBTILES_N = self.NUM_SUBTILES[1]
        NUM_SUBTILES_K = self.NUM_SUBTILES[2]

        self.dot_layout_x = gl.constexpr(
            gl.DotOperandLayout(operand_index=0, parent=MFMA_LAYOUT_PACKED if DTYPE_X == "e2m1" else MFMA_LAYOUT,
                                k_width=DOT_K_WIDTH))
        self.dot_layout_w = gl.constexpr(
            gl.DotOperandLayout(operand_index=1, parent=MFMA_LAYOUT_PACKED if DTYPE_W == "e2m1" else MFMA_LAYOUT,
                                k_width=DOT_K_WIDTH))
        if self.USE_MFMA_SCALED:
            self.layout_x_scale = gl.constexpr(
                gl.amd.cdna4.get_mfma_scale_layout(self.dot_layout_x,
                                                     [BLOCK_M // NUM_SUBTILES_M, BLOCK_K_SCALE // NUM_SUBTILES_K]))
            self.layout_w_scale = gl.constexpr(
                gl.amd.cdna4.get_mfma_scale_layout(self.dot_layout_w,
                                                     [BLOCK_N // NUM_SUBTILES_N, BLOCK_K_SCALE // NUM_SUBTILES_K]))
        else:
            # Scale layouts are not needed for non-scaled MFMA
            self.layout_x_scale = gl.constexpr(0)
            self.layout_w_scale = gl.constexpr(0)
        self.acc_layout = gl.constexpr(MFMA_LAYOUT)

        vec: gl.constexpr = 8
        per_phase: gl.constexpr = 2
        max_phase: gl.constexpr = 8
        self.shared_layout_x = gl.constexpr(gl.SwizzledSharedLayout(vec, per_phase, max_phase, order=[1, 0]))
        if W_TRANSPOSE:
            self.shared_layout_w = gl.constexpr(gl.SwizzledSharedLayout(vec, per_phase, max_phase, order=[1, 0]))
        else:
            self.shared_layout_w = gl.constexpr(gl.SwizzledSharedLayout(vec, per_phase, max_phase, order=[0, 1]))
        if self.USE_MFMA_SCALED:
            self.shared_layout_x_scale = gl.constexpr(gl.SwizzledSharedLayout(vec, per_phase, max_phase, order=[1, 0]))
            self.shared_layout_w_scale = gl.constexpr(gl.SwizzledSharedLayout(vec, per_phase, max_phase, order=[1, 0]))
        else:
            self.shared_layout_x_scale = gl.constexpr(0)
            self.shared_layout_w_scale = gl.constexpr(0)


@aggregate
class MoEProgramBase:

    @gluon.constexpr_function
    def __init__(self):
        pass

    @gluon.jit
    def mfma(self, x, scale_x, w, scale_w, accumulator):
        cfg = self.cfg
        if cfg.USE_MFMA_SCALED:
            return gl.amd.cdna4.mfma_scaled(x, scale_x, cfg.DTYPE_X, w, scale_w, cfg.DTYPE_W, accumulator)
        else:
            return gl.amd.cdna4.mfma(x, w, accumulator)

    @gluon.jit
    def issue_global_loads(self, load_idx, pred=1):
        cfg = self.cfg
        # BLOCK_K_PACKED_X: gl.constexpr = cfg.BLOCK_K // cfg.DIV_FACTOR_X
        # BLOCK_K_PACKED_W: gl.constexpr = cfg.BLOCK_K // cfg.DIV_FACTOR_W
        # BLOCK_K_SCALE: gl.constexpr = cfg.BLOCK_K // cfg.SCALE_BLOCK

        self.x_desc.issue_async_load(load_idx, self.x_buffer, pred)
        self.w_desc.issue_async_load(load_idx, self.w_buffer, pred)
        if cfg.WITH_X_MX_SCALE:
            self.x_scale_desc.issue_async_load(load_idx, self.x_scale_buffer, pred)
        if cfg.WITH_W_MX_SCALE:
            self.w_scale_desc.issue_async_load(load_idx, self.w_scale_buffer, pred)

        # if cfg.USE_GATHER:
        #     col_offset_x = self.off_k_x + load_idx * BLOCK_K_PACKED_X
        #     tdm.async_gather(self.x_desc, self.gathered_m, col_offset_x,
        #                      self.x_buffer.index(load_idx % cfg.NUM_BUFFERS), pred=pred)
        # else:
        #     tdm.async_load(self.x_desc, [0, load_idx * BLOCK_K_PACKED_X],
        #                    self.x_buffer.index(load_idx % cfg.NUM_BUFFERS), pred=pred)

        # if cfg.W_TRANSPOSE:
        #     tdm.async_load(self.w_desc, [0, load_idx * BLOCK_K_PACKED_W],
        #                    self.w_buffer.index(load_idx % cfg.NUM_BUFFERS), pred=pred)
        # else:
        #     tdm.async_load(self.w_desc, [load_idx * BLOCK_K_PACKED_W, 0],
        #                    self.w_buffer.index(load_idx % cfg.NUM_BUFFERS), pred=pred)

        # if cfg.WITH_X_MX_SCALE:
        #     if cfg.USE_GATHER:
        #         col_offset_x_scale = self.off_k_x * cfg.DIV_FACTOR_X // cfg.SCALE_BLOCK + load_idx * BLOCK_K_SCALE
        #         tdm.async_gather(self.x_scale_desc, self.gathered_m, col_offset_x_scale,
        #                          self.x_scale_buffer.index(load_idx % cfg.NUM_BUFFERS), pred=pred)
        #     else:
        #         tdm.async_load(self.x_scale_desc, [0, load_idx * cfg.BLOCK_K_SCALE_PRESHUFFLED],
        #                        self.x_scale_buffer.index(load_idx % cfg.NUM_BUFFERS), pred=pred)

        # if cfg.WITH_W_MX_SCALE:
        #     tdm.async_load(self.w_scale_desc, [0, load_idx * cfg.BLOCK_K_SCALE_PRESHUFFLED],
        #                    self.w_scale_buffer.index(load_idx % cfg.NUM_BUFFERS), pred=pred)

        return load_idx + 1

    @gluon.jit
    def async_wait(self, waitcnt):
        gl.amd.cdna4.async_copy.wait_group(waitcnt * self.cfg.NUM_LOADS_IN_BATCH)


@gluon.constexpr_function
def get_bitwidth(dtype):
    if isinstance(dtype, gl.pointer_type):
        dtype = dtype.element_ty
    return dtype.primitive_bitwidth

@gluon.constexpr_function
def get_blocked_layout(num_warps: gl.constexpr, dtype: gl.constexpr, order):
    bitwidth = get_bitwidth(dtype)
    vector_size = [1, max(1, 128 // bitwidth)] if order[1] == 0 else [max(1, 128 // bitwidth), 1]
    warps_per_cta = [num_warps // 2, 2] if order[1] == 0 else [2, num_warps // 2]
    return gl.BlockedLayout(vector_size, [8, 8], warps_per_cta, order)

@gluon.constexpr_function
def get_scale_blocked_layout(num_warps: gl.constexpr):
    return gl.BlockedLayout([1, 8], [1, 64], [num_warps // 2, 2], [1, 0])


@gluon.aggregate
class AsyncCopyDescriptor:
    cfg: MoEConfig

    op_idx: gl.constexpr
    ptr: gl.tensor
    stride_k: gl.tensor
    offsets: gl.tensor

    # mask
    off_k: gl.tensor
    masks_nonk: gl.tensor
    k_limit: gl.tensor

    # constexpr
    BLOCK_K: gl.constexpr

    @gluon.constexpr_function
    def __init__(self, cfg: MoEConfig, op_idx, ptr, stride_k, offsets, off_k, masks_nonk, k_limit, BLOCK_K):
        self.cfg = cfg
        self.op_idx = gl.constexpr(op_idx)
        self.ptr = ptr
        self.offsets = offsets
        self.off_k = off_k
        self.masks_nonk = masks_nonk
        self.k_limit = k_limit
        self.BLOCK_K = BLOCK_K

    @gluon.jit
    def initialize(cfg: MoEConfig, op_idx, ptr, off_nonk, off_k, stride_nonk, stride_k, masks_nonk, k_limit, BLOCK_K: gl.constexpr):
        offsets = gl.expand_dims(off_k, op_idx) * stride_k + gl.expand_dims(off_nonk, 1 - op_idx) * stride_nonk
        return AsyncCopyDescriptor(cfg, op_idx, ptr, stride_k, offsets, off_k, masks_nonk, k_limit, BLOCK_K)

    @gluon.jit
    def issue_async_load(self, idx: int, buffer: gl.shared_memory_descriptor, pred=1):
        if pred:
            NUM_BUFFERS: gl.constexpr = self.cfg.NUM_BUFFERS
            EVEN_K: gl.constexpr = self.cfg.EVEN_K
            off_k = idx * self.BLOCK_K
            if not EVEN_K:
                mask_k = gl.expand_dims(off_k + self.off_k, self.op_idx) < self.k_limit
                mask = mask_k & self.masks_nonk
                gl.amd.cdna4.async_copy.buffer_load_to_shared(
                    buffer.index(idx % NUM_BUFFERS), #
                    self.ptr + idx * off_k * self.stride_k, #
                    self.offsets, #
                    mask=mask, #
                    other=0, #
                )
            else:
                gl.amd.cdna4.async_copy.buffer_load_to_shared(
                    buffer.index(idx % NUM_BUFFERS), #
                    self.ptr + idx * off_k * self.stride_k, #
                    self.offsets, #
                )

    @gluon.jit
    def issue_local_load(self, idx: int, buffer: gl.shared_memory_descriptor, layout: gl.constexpr):
        NUM_BUFFERS: gl.constexpr = self.cfg.NUM_BUFFERS
        return gl.amd.cdna4.async_copy.load_shared_relaxed(buffer.index(idx % NUM_BUFFERS), layout)


@composition
@gluon.aggregate
class MoEPipelinedProgram:
    base: MoEProgramBase
    cfg: MoEConfig
    x_buffer: gl.shared_memory_descriptor
    w_buffer: gl.shared_memory_descriptor
    x_scale_buffer: gl.shared_memory_descriptor | gl.constexpr
    w_scale_buffer: gl.shared_memory_descriptor | gl.constexpr

    x_desc: AsyncCopyDescriptor
    w_desc: AsyncCopyDescriptor
    x_scale_desc: AsyncCopyDescriptor | gl.constexpr
    w_scale_desc: AsyncCopyDescriptor | gl.constexpr

    @gluon.constexpr_function
    def __init__(self, cfg: MoEConfig, x_buffer, w_buffer, x_scale_buffer, w_scale_buffer, x_desc, w_desc, x_scale_desc, w_scale_desc):
        self.cfg = cfg
        self.x_buffer = x_buffer
        self.w_buffer = w_buffer
        self.x_scale_buffer = x_scale_buffer if cfg.WITH_X_MX_SCALE else gl.constexpr(0)
        self.w_scale_buffer = w_scale_buffer if cfg.WITH_W_MX_SCALE else gl.constexpr(0)

        self.x_desc = x_desc
        self.w_desc = w_desc
        self.x_scale_desc = x_scale_desc if cfg.WITH_X_MX_SCALE else gl.constexpr(0)
        self.w_scale_desc = w_scale_desc if cfg.WITH_W_MX_SCALE else gl.constexpr(0)

        self.base = MoEProgramBase()

    @gluon.jit
    def initialize(cfg: MoEConfig, x_desc, w_desc, x_scale_desc, w_scale_desc):
        NUM_BUFFERS: gl.constexpr = cfg.NUM_BUFFERS

        BLOCK_K_PACKED_X: gl.constexpr = cfg.BLOCK_K // cfg.DIV_FACTOR_X
        BLOCK_K_PACKED_W: gl.constexpr = cfg.BLOCK_K // cfg.DIV_FACTOR_W

        x_buffer = gl.allocate_shared_memory(x_desc.dtype, shape=[NUM_BUFFERS, cfg.BLOCK_M, BLOCK_K_PACKED_X],
                                             layout=cfg.shared_layout_x)
        w_buffer = gl.allocate_shared_memory(
            w_desc.dtype, shape=[NUM_BUFFERS, cfg.BLOCK_N, BLOCK_K_PACKED_W]
            if cfg.W_TRANSPOSE else [NUM_BUFFERS, BLOCK_K_PACKED_W, cfg.BLOCK_N], layout=cfg.shared_layout_w)

        if cfg.WITH_X_MX_SCALE:
            if cfg.USE_GATHER:
                BLOCK_K_SCALE: gl.constexpr = cfg.BLOCK_K // cfg.SCALE_BLOCK
                x_scale_buffer = gl.allocate_shared_memory(gl.uint8, shape=[NUM_BUFFERS, cfg.BLOCK_M, BLOCK_K_SCALE],
                                                           layout=cfg.shared_layout_x_scale)
            else:
                x_scale_buffer = gl.allocate_shared_memory(
                    gl.uint8, shape=[NUM_BUFFERS, cfg.BLOCK_M_PRESHUFFLED, cfg.BLOCK_K_SCALE_PRESHUFFLED],
                    layout=cfg.shared_layout_x_scale)
        else:
            x_scale_buffer = gl.constexpr(0)

        if cfg.WITH_W_MX_SCALE:
            w_scale_buffer = gl.allocate_shared_memory(
                gl.uint8, shape=[NUM_BUFFERS, cfg.BLOCK_N_PRESHUFFLED, cfg.BLOCK_K_SCALE_PRESHUFFLED],
                layout=cfg.shared_layout_w_scale)
        else:
            w_scale_buffer = gl.constexpr(0)

        return MoEPipelinedProgram(cfg, x_buffer, w_buffer, x_scale_buffer, w_scale_buffer, x_desc, w_desc,
                                   x_scale_desc, w_scale_desc)


    @gluon.jit
    def issue_local_loads(self, mfma_idx):
        cfg = self.cfg
        BLOCK_K_SCALE: gl.constexpr = cfg.BLOCK_K // cfg.SCALE_BLOCK

        x = self.x_buffer.index(mfma_idx % cfg.NUM_BUFFERS).load(layout=cfg.dot_layout_x)
        if cfg.W_TRANSPOSE:
            w = self.w_buffer.index(mfma_idx % cfg.NUM_BUFFERS).permute([1, 0]).load(layout=cfg.dot_layout_w)
        else:
            w = self.w_buffer.index(mfma_idx % cfg.NUM_BUFFERS).load(layout=cfg.dot_layout_w)

        if cfg.WITH_X_MX_SCALE:
            x_scale_buffer_slice = self.x_scale_buffer.index(mfma_idx % cfg.NUM_BUFFERS)
        if cfg.WITH_W_MX_SCALE:
            w_scale_buffer_slice = self.w_scale_buffer.index(mfma_idx % cfg.NUM_BUFFERS)

        if cfg.SCALE_PRESHUFFLE:
            if cfg.WITH_X_MX_SCALE and not cfg.USE_GATHER:
                x_scale_buffer_slice = x_scale_buffer_slice.reshape(
                    (cfg.BLOCK_M_PRESHUFFLED, BLOCK_K_SCALE // cfg.SCALE_KWIDTH, cfg.PRESHUFFLE_FACTOR // 4, 4,
                     cfg.SCALE_KWIDTH)).permute((0, 3, 2, 1, 4)).reshape((cfg.BLOCK_M, BLOCK_K_SCALE))
            if cfg.WITH_W_MX_SCALE:
                w_scale_buffer_slice = w_scale_buffer_slice.reshape(
                    (cfg.BLOCK_N_PRESHUFFLED, BLOCK_K_SCALE // cfg.SCALE_KWIDTH, cfg.PRESHUFFLE_FACTOR // 4, 4,
                     cfg.SCALE_KWIDTH)).permute((0, 3, 2, 1, 4)).reshape((cfg.BLOCK_N, BLOCK_K_SCALE))

        if cfg.WITH_X_MX_SCALE:
            scale_x = x_scale_buffer_slice.load(layout=cfg.layout_x_scale)
        else:
            scale_x = 0
            scale_x = scale_x.to(gl.uint8)

        if cfg.WITH_W_MX_SCALE:
            scale_w = w_scale_buffer_slice.load(layout=cfg.layout_w_scale)
        else:
            scale_w = 0
            scale_w = scale_w.to(gl.uint8)

        return x, w, scale_x, scale_w

    @gluon.jit
    def pipeline(self, loop_k):
        cfg = self.cfg
        load_idx = 0
        mfma_idx = 0

        # prologue
        for _ in gl.static_range(cfg.NUM_BUFFERS - 1):
            load_idx = self.issue_global_loads(load_idx)

        accumulator = gl.zeros((cfg.BLOCK_M, cfg.BLOCK_N), dtype=gl.float32, layout=cfg.acc_layout)
        loop_ub = gl.cdiv(loop_k, cfg.BLOCK_K)
        gl.assume(loop_ub > 0)
        epilogue_lb = loop_ub - (cfg.NUM_BUFFERS - 1)

        for i in range(0, loop_ub):
            pred = i - epilogue_lb
            pred = (pred >> 31) & 1
            load_idx = self.issue_global_loads(load_idx, pred=pred)
            self.async_wait(cfg.NUM_BUFFERS - 1)

            x, w, scale_x, scale_w = self.issue_local_loads(mfma_idx)
            mfma_idx += 1

            accumulator = self.mfma(x, scale_x, w, scale_w, accumulator)

        return accumulator

@gluon.jit
def _pipelined_moe_kernel_scaled(
    # Tensors --------------------------------------------------------
    x_ptr,
    w_ptr,
    x_scale_ptr,  # only used when A is mxfp4
    w_scale_ptr,  # always used (W is always mxfp4)
    bias_ptr,
    y_ptr,
    gather_idx_ptr,
    scatter_idx_ptr,
    gate_scal_ptr,
    expert_remap_ptr,
    # Strides --------------------------------------------------------
    stride_xm,
    stride_xk,  # bytes along the *physical* K dim
    stride_we,
    stride_wn,
    stride_wk,  # bytes along the *physical* (packed) K dim of W
    stride_xsm,
    stride_xsk,
    stride_wse,
    stride_wsn,
    stride_wsk,
    stride_yn,
    stride_ym,
    stride_be,
    stride_bn,
    # Logical shapes -------------------------------------------------
    M,
    N,
    K,  # logical K (number of e2m1/e4m3 elements per row)
    # Scalar global fp8 scale (used only when A is e4m3/e5m2) -------
    x_global_scale,  # fp32 scalar
    # Tile constants -------------------------------------------------
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    BLOCKS_PER_EXPERT: gl.constexpr,
    X_FORMAT: gl.constexpr,  # "e2m1" / "e4m3" / "e5m2"
    W_FORMAT: gl.constexpr,  # "e2m1" (W is always mxfp4 here)
    UPCAST_INDICES: gl.constexpr,
    HAS_BIAS: gl.constexpr,
    HAS_GATHER: gl.constexpr,
    HAS_SCATTER: gl.constexpr,
    DO_SWIGLU: gl.constexpr,
    SWIGLU_ALPHA: gl.constexpr,
    SWIGLU_LIMIT: gl.constexpr,
    OUT_BLOCK_N: gl.constexpr,
    APPLY_GATE_SCAL: gl.constexpr,
    HAS_EXPERT_REMAP: gl.constexpr,
    # Performance parameters ------------------------------------------
    NUM_WARPS: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    SCALE_PRESHUFFLE: gl.constexpr,
    W_TRANSPOSE: gl.constexpr = False,
    NUM_SUBTILES: gl.constexpr = (1, 1, 1),
    EVEN_K: gl.constexpr = True,
):
    compact_idx = gl.program_id(1)
    block_pid = gl.program_id(0)
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    block_in_expert = block_pid // grid_n
    pid_n = block_pid % grid_n
    if HAS_EXPERT_REMAP:
        expert_id = gl.load(expert_remap_ptr + compact_idx).to(gl.int32)
    else:
        expert_id = compact_idx

    HAS_X_BLOCK_SCALE: gl.constexpr = x_scale_ptr is not None
    HAS_W_BLOCK_SCALE: gl.constexpr = w_scale_ptr is not None
    USE_GATHER: gl.constexpr = gather_idx_ptr is not None

    BLOCK_SCALE_FACTOR: gl.constexpr = 32
    BLOCK_K_SCALE: gl.constexpr = BLOCK_K // BLOCK_SCALE_FACTOR

    off_m = compact_idx * BLOCKS_PER_EXPERT * BLOCK_M + block_in_expert * BLOCK_M
    off_n = pid_n * BLOCK_N
    w_base_offset = expert_id * stride_we
    ws_base_offset = expert_id * stride_wse

    STORE: gl.constexpr = _store_layout(NUM_WARPS)

    index_type: gl.constexpr = gl.int64 if UPCAST_INDICES else gl.int32
    cfg = MoEConfig(BLOCK_M, BLOCK_N, BLOCK_K, X_FORMAT, W_FORMAT, BLOCK_SCALE_FACTOR, NUM_BUFFERS, W_TRANSPOSE, HAS_X_BLOCK_SCALE, HAS_W_BLOCK_SCALE, SCALE_PRESHUFFLE, index_type, NUM_SUBTILES, EVEN_K, USE_GATHER, NUM_WARPS)

    # Packing factor along K: e2m1 packs 2 elements per byte.
    BLOCK_K_X: gl.constexpr = cfg.BLOCK_K // cfg.DIV_FACTOR_X
    BLOCK_K_W: gl.constexpr = cfg.BLOCK_K // cfg.DIV_FACTOR_W

    LOAD_X_LAYOUT: gl.constexpr = _load_layout(BLOCK_K_X, NUM_WARPS, [1, 0])
    LOAD_W_LAYOUT: gl.constexpr = _load_layout(BLOCK_K_W, NUM_WARPS, [0, 1])

    # ----- Compute global offsets (in *physical* byte coords) -----
    offs_xm = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, LOAD_X_LAYOUT))
    offs_xk = gl.arange(0, BLOCK_K_X, layout=gl.SliceLayout(0, LOAD_X_LAYOUT))
    offs_wn = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, LOAD_W_LAYOUT))
    offs_wk = gl.arange(0, BLOCK_K_W, layout=gl.SliceLayout(1, LOAD_W_LAYOUT))

    rows_m = off_m + offs_xm
    if HAS_GATHER:
        rows_m = gl.load(
            gather_idx_ptr + rows_m, mask=rows_m < M, other=0
        ).to(gl.int32)

    # Activation tile (uint8 storage).
    x_offsets = rows_m[:, None] * stride_xm + offs_xk[None, :] * stride_xk
    # Weight tile (uint8 storage).
    w_offsets = (
        offs_wk[:, None] * stride_wk
        + (off_n + offs_wn)[None, :] * stride_wn
        + w_base_offset
    )

    mask_m = rows_m < M
    mask_n = (off_n + offs_wn) < N

    acc = gl.zeros((BLOCK_M, BLOCK_N), gl.float32, cfg.acc_layout)

    offs_xs_m = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, cfg.layout_x_scale))
    offs_xs_k = gl.arange(0, BLOCK_K_SCALE, layout=gl.SliceLayout(0, cfg.layout_x_scale))
    offs_ws_n = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, cfg.layout_w_scale))
    offs_ws_k = gl.arange(0, BLOCK_K_SCALE, layout=gl.SliceLayout(0, cfg.layout_w_scale))

    rows_m_scale = off_m + offs_xs_m
    if HAS_GATHER:
        rows_m_scale = rows_m

    x_scale_offsets = (
        rows_m_scale[:, None] * stride_xsm + offs_xs_k[None, :] * stride_xsk
    )
    w_scale_offsets = (
        (off_n + offs_ws_n)[:, None] * stride_wsn
        + offs_ws_k[None, :] * stride_wsk
        + ws_base_offset
    )

    x_desc = AsyncCopyDescriptor.initialize(cfg, 0, BLOCK_K_X, x_ptr, rows_m, offs_xk, stride_xm, stride_xk, mask_m[:, None], K // cfg.DIV_FACTOR_X)
    if W_TRANSPOSE:
        w_desc = AsyncCopyDescriptor.initialize(cfg, 0, BLOCK_K_W, w_ptr, off_n + offs_wn, offs_wk, stride_wn, stride_wk, mask_n[None, :], K // cfg.DIV_FACTOR_W)
    else:
        w_desc = AsyncCopyDescriptor.initialize(cfg, 1, BLOCK_K_W, w_ptr, off_n + offs_wn, offs_wk, stride_wn, stride_wk, mask_n[None, :], K // cfg.DIV_FACTOR_W)

    x_scale_desc = AsyncCopyDescriptor.initialize(cfg, 0, BLOCK_K_SCALE, x_scale_ptr, rows_m_scale, offs_xs_k, stride_xsm, stride_xsk, rows_m_scale[:, None] < M, K // cfg.SCALE_BLOCK)
    w_scale_desc = AsyncCopyDescriptor.initialize(cfg, 0, BLOCK_K_SCALE, w_scale_ptr, off_n + offs_ws_n, offs_ws_k, stride_wsn, stride_wsk, (off_n + offs_ws_n)[:, None] < N, K // cfg.SCALE_BLOCK)

    pgm = MoEPipelinedProgram.initialize(cfg, x_desc, w_desc, x_scale_desc, w_scale_desc)

    acc = pgm.pipeline(K)

    # Apply global fp8 scale (single scalar) when A is fp8.
    if not HAS_X_BLOCK_SCALE:
        acc = acc * x_global_scale

    if HAS_BIAS:
        bias_offs = off_n + gl.arange(0, BLOCK_N, gl.SliceLayout(0, cfg.acc_layout))
        bias_mask = bias_offs < N
        bias = gl.load(
            bias_ptr + expert_id * stride_be + bias_offs,
            mask=bias_mask,
            other=0.0,
        )
        acc = acc + bias[None, :].to(gl.float32)

    if DO_SWIGLU:
        out = _swiglu_reduce(acc, SWIGLU_ALPHA, SWIGLU_LIMIT, OUT_BLOCK_N, cfg.acc_layout)
    else:
        out = acc

    out = out.to(y_ptr.dtype.element_ty)
    out = gl.convert_layout(out, STORE)

    offs_y_m = off_m + gl.arange(0, BLOCK_M, gl.SliceLayout(1, STORE))
    off_n_out = pid_n * OUT_BLOCK_N
    offs_y_n = off_n_out + gl.arange(0, OUT_BLOCK_N, gl.SliceLayout(0, STORE))

    if APPLY_GATE_SCAL:
        scal = gl.load(
            gate_scal_ptr + offs_y_m,
            mask=offs_y_m < M,
            other=1.0,
        )
        out = out * scal[:, None].to(out.dtype)

    actual_n = (N // 2) if DO_SWIGLU else N
    if HAS_SCATTER:
        rows_y = gl.load(scatter_idx_ptr + offs_y_m, mask=offs_y_m < M, other=M)
        mask_y = (rows_y[:, None] < M) & (offs_y_n[None, :] < actual_n)
        y_offs = rows_y[:, None] * stride_ym + offs_y_n[None, :] * stride_yn
    else:
        mask_y = (offs_y_m[:, None] < M) & (offs_y_n[None, :] < actual_n)
        y_offs = offs_y_m[:, None] * stride_ym + offs_y_n[None, :] * stride_yn

    gl.store(y_ptr + y_offs, out, mask=mask_y)


# ---------------------------------------------------------------------------
# Static profile helper (sgpr/vgpr spill detection)
# ---------------------------------------------------------------------------


def _parse_amdgcn_metric(amdgcn: str, key: str) -> int | None:
    """Look for ``.<key>: N`` or ``;  Key: N`` in the AMDGCN dump."""
    import re

    m = re.search(rf"\.{key}:\s+(\d+)", amdgcn)
    if m is not None:
        return int(m.group(1))
    m = re.search(rf";\s+{key}\s*[:=]?\s+(\d+)", amdgcn)
    if m is not None:
        return int(m.group(1))
    return None


def static_profile(kernel: Any, *, label: str = "") -> dict:
    """Return a structured GPR / scratch / occupancy profile for ``kernel``.

    Mirrors the helper from
    ``triton-450/third_party/amd/python/examples/gluon/gfx1250_utils.py``,
    but tolerant of the slightly different MI355 AMDGCN dump format.
    """
    amdgcn = kernel.asm.get("amdgcn", "")
    fields = [
        "sgpr_count",
        "sgpr_spill_count",
        "vgpr_count",
        "vgpr_spill_count",
        "ScratchSize",
        "codeLenInByte",
        "Occupancy",
    ]
    profile = {f: _parse_amdgcn_metric(amdgcn, f) for f in fields}
    if label:
        profile["label"] = label
    return profile


def assert_no_spills(profile: dict, *, allow_scratch: int = 0) -> None:
    """Raise if the static profile shows any GPR spill or excess scratch."""
    sgpr_spill = profile.get("sgpr_spill_count") or 0
    vgpr_spill = profile.get("vgpr_spill_count") or 0
    scratch = profile.get("ScratchSize") or 0
    msg = []
    if sgpr_spill:
        msg.append(f"sgpr_spill={sgpr_spill}")
    if vgpr_spill:
        msg.append(f"vgpr_spill={vgpr_spill}")
    if scratch > allow_scratch:
        msg.append(f"scratch={scratch} (allowed={allow_scratch})")
    if msg:
        raise AssertionError(
            f"Gluon MoE kernel '{profile.get('label', '?')}' "
            f"shows static spills: {', '.join(msg)}"
        )


# ---------------------------------------------------------------------------
# Helpers shared by all three Python-side launchers
# ---------------------------------------------------------------------------


def _expert_layout(
    a_ragged_metadata: Any | None,
    block_m: int,
    M: int,
) -> tuple[int, int, torch.Tensor | None]:
    """Return ``(num_active_experts, blocks_per_expert, expert_remap)``.

    The MI355 software-pipelined kernel encodes the *compact* active
    expert index as ``program_id(1)``. When the ragged metadata only
    activates a strict subset of experts (typical for decode batches
    where ``B*topk << E``) we filter out the empty experts here and
    return an ``expert_remap`` tensor of shape ``[num_active]`` that
    translates the kernel's compact index back to the real expert id
    (used to look up the per-expert W slice). For the all-active case
    we return ``expert_remap=None`` so the kernel skips the indirection
    load entirely.
    """
    if a_ragged_metadata is None:
        return 1, (M + block_m - 1) // block_m, None
    counts = a_ragged_metadata.slice_sizes
    counts_list = counts.tolist()
    active = [i for i, c in enumerate(counts_list) if int(c) > 0]
    num_active = len(active)
    if num_active == 0:
        return 1, 0, None
    max_blocks = max((int(counts_list[i]) + block_m - 1) // block_m for i in active)
    if num_active == counts.numel():
        # All experts active: identity remap, no need to materialise.
        return num_active, max_blocks, None
    expert_remap = torch.tensor(active, device=counts.device, dtype=torch.int32)
    return num_active, max_blocks, expert_remap


def _make_dummy(device, dtype=torch.int32, n: int = 0) -> torch.Tensor:
    return torch.empty(max(n, 0), device=device, dtype=dtype)


def _supports_pure_bf16(precision_config, fused_activation) -> bool:
    """Return True iff this call can take the pure-bf16 fast path."""
    if precision_config is None:
        return True
    if getattr(precision_config, "b_mx_scale", None) is not None:
        return False
    flex = getattr(precision_config, "flex_ctx", None)
    lhs = getattr(flex, "lhs_data", None) if flex is not None else None
    if lhs is not None and getattr(lhs, "dtype", None) is not None:
        return False
    return True


# ---------------------------------------------------------------------------
# Public launcher: software-pipelined ragged matmul (unified driver)
# ---------------------------------------------------------------------------


def _launch_pipelined(
    x: torch.Tensor,
    w: torch.Tensor,
    *,
    y: torch.Tensor,
    bias: torch.Tensor | None,
    gather_indx,
    scatter_indx,
    gate_scal: torch.Tensor | None,
    a_ragged_metadata,
    swiglu: tuple[float, float] | None,
    out_block_n: int,
    block_m: int,
    block_n: int,
    block_k: int,
    num_warps: int,
    num_buffers: int,
    scaled_mfma: bool = False,
):
    M, K = x.shape[-2], x.shape[-1]
    if w.ndim == 3:
        E, K_W, N = w.shape[0], w.shape[-2], w.shape[-1]
    else:
        K_W, N = w.shape
        E = 1
    assert K == K_W, f"K mismatch: {K} vs {K_W}"

    # The MFMA instruction shape determines the smallest legal
    # ``BLOCK_K``. Regular ``mfma`` on CDNA4 is 16x16x32 so any
    # multiple of 32 is fine; scaled MFMA (mxfp4 weight + fp8 act) is
    # 16x16x128 so BLOCK_K must be a multiple of 128 with a floor of
    # 128 (see ``_autotune_block`` and the upstream
    # ``test_amd_mfma_scaled`` reference in triton-450).
    mfma_k = _MFMA_SCALED_K if scaled_mfma else _MFMA_K
    assert block_k % mfma_k == 0, (
        f"BLOCK_K={block_k} must be a multiple of MFMA K dim "
        f"({mfma_k}); scaled_mfma={scaled_mfma}"
    )
    if scaled_mfma:
        assert block_k >= _MFMA_SCALED_K, (
            f"scaled MFMA requires BLOCK_K >= {_MFMA_SCALED_K} (got "
            f"{block_k}); see TASKS.md Update 3."
        )
    assert (
        block_m % _MFMA_M == 0
    ), f"BLOCK_M={block_m} must be a multiple of MFMA M dim ({_MFMA_M})"

    num_active, blocks_per_expert, expert_remap = _expert_layout(
        a_ragged_metadata, block_m, M
    )
    grid_n = (N + block_n - 1) // block_n
    grid = (blocks_per_expert * grid_n, num_active)

    bias_buf = bias if bias is not None else _make_dummy(x.device, x.dtype)
    gather_buf = (
        gather_indx.src_indx
        if gather_indx is not None
        else _make_dummy(x.device, torch.int32)
    )
    scatter_buf = (
        scatter_indx.dst_indx
        if scatter_indx is not None
        else _make_dummy(x.device, torch.int32)
    )
    gate_scal_buf = (
        gate_scal if gate_scal is not None else _make_dummy(x.device, torch.float32)
    )
    expert_remap_buf = (
        expert_remap if expert_remap is not None else _make_dummy(x.device, torch.int32)
    )

    swiglu_alpha = swiglu[0] if swiglu is not None else 0.0
    swiglu_limit = swiglu[1] if swiglu is not None else 0.0

    w3 = w if w.ndim == 3 else w.unsqueeze(0)

    _pipelined_moe_kernel[grid](
        x,
        w3,
        bias_buf,
        y,
        gather_buf,
        scatter_buf,
        gate_scal_buf,
        expert_remap_buf,
        x.stride(-2),
        x.stride(-1),
        w3.stride(0),
        w.stride(-1),
        w.stride(-2),
        y.stride(-1),
        y.stride(-2),
        bias.stride(0) if bias is not None else 0,
        bias.stride(-1) if bias is not None else 0,
        M,
        N,
        K,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        NUM_WARPS=num_warps,
        NUM_BUFFERS=num_buffers,
        BLOCKS_PER_EXPERT=blocks_per_expert,
        HAS_BIAS=bias is not None,
        HAS_GATHER=gather_indx is not None,
        HAS_SCATTER=scatter_indx is not None,
        DO_SWIGLU=swiglu is not None,
        SWIGLU_ALPHA=float(swiglu_alpha),
        SWIGLU_LIMIT=float(swiglu_limit),
        OUT_BLOCK_N=out_block_n,
        APPLY_GATE_SCAL=gate_scal is not None,
        HAS_EXPERT_REMAP=expert_remap is not None,
        num_warps=num_warps,
    )


def _launch_pipelined_scaled(
    x: torch.Tensor,  # uint8 storage (e2m1 or e4m3/e5m2)
    w: torch.Tensor,  # uint8 storage (e2m1)
    *,
    y: torch.Tensor,
    bias: torch.Tensor | None,
    gather_indx,
    scatter_indx,
    gate_scal: torch.Tensor | None,
    a_ragged_metadata,
    swiglu: tuple[float, float] | None,
    out_block_n: int,
    block_m: int,
    block_n: int,
    block_k: int,
    num_warps: int,
    a_format: str,  # "e2m1" / "e4m3" / "e5m2"
    b_format: str = "e2m1",
    x_scale: torch.Tensor | None = None,
    w_scale: torch.Tensor | None = None,
    a_global_scale: float = 1.0,
):
    """Launch the scaled-MFMA MoE kernel.

    A is interpreted as ``A_FORMAT`` (one of ``e2m1`` / ``e4m3`` / ``e5m2``)
    and may carry either a block scale tensor (mxfp4) or a global fp32
    scalar (fp8). B is always mxfp4 (``e2m1``) with an e8m0 block scale.
    """
    assert a_format in {"e2m1", "e4m3", "e5m2"}
    assert b_format == "e2m1", "Only mxfp4 (e2m1) W is supported today."
    has_a_block_scale = a_format == "e2m1"
    if has_a_block_scale:
        assert x_scale is not None, "mxfp4 A requires a block-scale tensor"
    assert w_scale is not None, "mxfp4 W requires a block-scale tensor"

    # Logical K = K_packed * DIV_FACTOR; the caller passes the *logical*
    # shape so the kernel can iterate by logical BLOCK_K.
    M = x.shape[-2]
    K_phys = x.shape[-1]
    div_a = 2 if a_format == "e2m1" else 1
    K = K_phys * div_a

    if w.ndim == 3:
        E, K_w_phys, N = w.shape
    else:
        K_w_phys, N = w.shape
        E = 1
    K_w = K_w_phys * 2  # W is always mxfp4 (packed 2:1)
    assert K == K_w, f"K mismatch: A logical K={K} vs W logical K={K_w}"

    assert block_k % _MFMA_SCALED_K == 0 and block_k >= _MFMA_SCALED_K
    assert block_m % _MFMA_M == 0

    num_active, blocks_per_expert, expert_remap = _expert_layout(
        a_ragged_metadata, block_m, M
    )
    grid_n = (N + block_n - 1) // block_n
    grid = (blocks_per_expert * grid_n, num_active)

    bias_buf = bias if bias is not None else _make_dummy(x.device, torch.float32)
    gather_buf = (
        gather_indx.src_indx
        if gather_indx is not None
        else _make_dummy(x.device, torch.int32)
    )
    scatter_buf = (
        scatter_indx.dst_indx
        if scatter_indx is not None
        else _make_dummy(x.device, torch.int32)
    )
    gate_scal_buf = (
        gate_scal if gate_scal is not None else _make_dummy(x.device, torch.float32)
    )
    expert_remap_buf = (
        expert_remap if expert_remap is not None else _make_dummy(x.device, torch.int32)
    )
    # x_scale_buf must always be a valid tensor (kernel branches on
    # HAS_A_BLOCK_SCALE constexpr but Triton still inspects the arg).
    x_scale_buf = x_scale if x_scale is not None else _make_dummy(x.device, torch.uint8)

    swiglu_alpha = swiglu[0] if swiglu is not None else 0.0
    swiglu_limit = swiglu[1] if swiglu is not None else 0.0

    w3 = w if w.ndim == 3 else w.unsqueeze(0)
    w_scale3 = w_scale if w_scale.ndim == 3 else w_scale.unsqueeze(0)

    # Per-expert strides on the e8m0 scale tensor. We expect layout
    # ``(E, N, K_SCALE)`` so that scale slices along K are contiguous.
    stride_wse = w_scale3.stride(0)
    stride_wsn = w_scale3.stride(-2)
    stride_wsk = w_scale3.stride(-1)

    if has_a_block_scale:
        stride_xsm = x_scale.stride(-2)
        stride_xsk = x_scale.stride(-1)
    else:
        stride_xsm = 0
        stride_xsk = 0

    num_buffers = 2
    SCALE_PRESHUFFLE = False
    W_TRANSPOSE = False
    NUM_SUBTILES = (1, 1, 1)
    EVEN_K = True

    _pipelined_moe_kernel_scaled[grid](
        x,
        w3,
        x_scale_buf,
        w_scale3,
        bias_buf,
        y,
        gather_buf,
        scatter_buf,
        gate_scal_buf,
        expert_remap_buf,
        x.stride(-2),
        x.stride(-1),
        w3.stride(0),
        w.stride(-1),
        w.stride(-2),
        stride_xsm,
        stride_xsk,
        stride_wse,
        stride_wsn,
        stride_wsk,
        y.stride(-1),
        y.stride(-2),
        bias.stride(0) if bias is not None else 0,
        bias.stride(-1) if bias is not None else 0,
        M,
        N,
        K,
        float(a_global_scale),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        BLOCKS_PER_EXPERT=blocks_per_expert,
        X_FORMAT=a_format,
        W_FORMAT=b_format,
        UPCAST_INDICES=False,
        HAS_BIAS=bias is not None,
        HAS_GATHER=gather_indx is not None,
        HAS_SCATTER=scatter_indx is not None,
        DO_SWIGLU=swiglu is not None,
        SWIGLU_ALPHA=float(swiglu_alpha),
        SWIGLU_LIMIT=float(swiglu_limit),
        OUT_BLOCK_N=out_block_n,
        APPLY_GATE_SCAL=gate_scal is not None,
        HAS_EXPERT_REMAP=expert_remap is not None,
        NUM_WARPS=num_warps,
        NUM_BUFFERS=num_buffers,
        SCALE_PRESHUFFLE=SCALE_PRESHUFFLE,
        W_TRANSPOSE=W_TRANSPOSE,
        NUM_SUBTILES=NUM_SUBTILES,
        EVEN_K=EVEN_K,
        num_warps=num_warps,
    )


# ---------------------------------------------------------------------------
# Public Python entry points (one per kernel that TASKS.md asks for)
# ---------------------------------------------------------------------------


# CDNA4 MFMA instruction shapes.  The regular ``mfma`` op is 16x16x32 in
# the K dimension; the *scaled* ``mfma_scaled`` op (used by the mxfp4
# weight + fp8 activation path) is 16x16x128. ``BLOCK_K`` must be a
# multiple of the relevant K dim, so any kernel that ever wants to swap
# in scaled MFMA must respect ``BLOCK_K >= 128``.
_MFMA_K = 32
_MFMA_SCALED_K = 128
_MFMA_M = 16


def _autotune_block(
    M: int,
    N: int,
    K: int,
    *,
    do_swiglu: bool = False,
    ragged: bool = False,
    scaled_mfma: bool = False,
) -> tuple[int, int, int, int]:
    """Pick ``(BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARPS)`` for given shape.

    Heuristic obtained by sweeping the microbench
    (``benchmarks/moe_gluon_microbench.py``) on MI355 with the
    gpt-oss-120b MoE dimensions (``H=I=2880, E=128, topk=4``) for
    decode (``B in {1, 32, 64}``) and prefill
    (``B in {1024, 4096, 8192}``).

    * **Dense gating GEMM** (``do_swiglu=False, ragged=False``).
      Output ``N=128`` (``num_local_experts``) so we keep ``BLOCK_N=64``
      to give us ``grid_n=2`` and rely on growing ``grid_m`` for fill:
        - M <= 1024 : 64x64x64,  8 warps  (decode + small prefill)
        - M <= 2048 : 128x64x64, 8 warps
        - M  > 2048 : 128x64x64, 4 warps  (prefill, more CTAs)

    * **Fused SwiGLU 1st GEMM** (``do_swiglu=True``). Internal
      ``BLOCK_N`` covers the ``gate || linear`` width (``2*OUT_BLOCK_N``);
      ``BLOCK_K=32`` keeps VGPR pressure for the swiglu reduce manageable:
        - M <= 8192 :  64x128x32, 4 warps
        - M  > 8192 : 128x128x32, 4 warps

    * **Ragged 2nd GEMM + scatter combine** (``ragged=True``).
      Same per-tile MFMA flow as the gating GEMM but with E experts'
      worth of CTAs in the launch grid; benefits from larger blocks at
      prefill scale to amortise scatter epilogue:
        - M <= 8192 :  64x128x32, 4 warps
        - M  > 8192 : 128x128x32, 4 warps  (256x256 saturates VGPRs
          on MI355 -> spills; 128x128 stays under the 256-VGPR limit
          while still hitting 273 TFLOPs).

    Scaled MFMA constraint
    ~~~~~~~~~~~~~~~~~~~~~~
    When ``scaled_mfma=True`` (mxfp4 weight + fp8 activation path), the
    underlying ``gl.amd.cdna4.mfma_scaled`` instruction has shape
    ``[16, 16, 128]`` in ``[M, N, K]``. We therefore *promote* whatever
    block size the regular heuristic returned so it satisfies
    ``BLOCK_K >= 128`` and ``BLOCK_K % 128 == 0``. We also widen the
    swiglu/ragged ``BLOCK_K`` floor from 32 to 128 because the smaller
    values are not legal for scaled MFMA.

    A microbench sweep over ``BLOCK_K in {64, 128, 256}`` with the
    regular MFMA path showed that ``BLOCK_K = 128`` actually *hurts*
    bf16 perf (2-3x slower than 32/64) because our register-staged
    pipeline holds 4x larger prefetch tiles, blowing VGPR pressure. So
    the override only activates when the caller explicitly opts into
    the scaled path; the bf16 default stays at 32/64.
    """
    if do_swiglu:
        bm, bn, bk, nw = (64, 128, 32, 4) if M <= 8192 else (128, 128, 32, 4)
    elif ragged:
        bm, bn, bk, nw = (64, 128, 32, 4) if M <= 8192 else (128, 128, 32, 4)
    elif M <= 1024:
        bm, bn, bk, nw = (64, 64, 64, 8)
    elif M <= 2048:
        bm, bn, bk, nw = (128, 64, 64, 8)
    else:
        bm, bn, bk, nw = (128, 64, 64, 4)
    if scaled_mfma:
        # Scaled MFMA on CDNA4 is 16x16x128; BLOCK_K must therefore be a
        # multiple of 128 with a floor of 128. We also bump BLOCK_M so it
        # remains a multiple of the MFMA M dim (16) -- which all of the
        # configurations above already are, but the explicit floor here
        # documents the requirement.
        bk = max(bk, _MFMA_SCALED_K)
        bk = (bk + _MFMA_SCALED_K - 1) // _MFMA_SCALED_K * _MFMA_SCALED_K
        bm = max(bm, _MFMA_M)
    return bm, bn, bk, nw


def gluon_bf16_gating_gemm(
    x: torch.Tensor,
    w: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    block_m: int | None = None,
    block_n: int | None = None,
    block_k: int | None = None,
    num_warps: int | None = None,
    num_buffers: int = _DEFAULT_NUM_BUFFERS,
) -> torch.Tensor:
    """bf16/fp16 dense GEMM ``y = x @ w`` (gating projection).

    Special-cases the non-MoE path of :func:`_pipelined_moe_kernel` -- no
    gather, no scatter, no swiglu, no per-expert metadata -- but still uses
    the same software-pipelined kernel body.

    Block size defaults are picked by :func:`_autotune_block` based on
    the input shape; pass explicit overrides to bench-tune.
    """
    assert _gluon_is_supported(), "Gluon MoE kernel requires CDNA4."
    assert x.dim() == 2 and w.dim() == 2
    M, K = x.shape
    K_W, N = w.shape
    assert K == K_W
    bm, bn, bk, nw = _autotune_block(M, N, K)
    block_m = block_m or bm
    block_n = block_n or bn
    block_k = block_k or bk
    num_warps = num_warps or nw
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)
    _launch_pipelined(
        x,
        w,
        y=y,
        bias=bias,
        gather_indx=None,
        scatter_indx=None,
        gate_scal=None,
        a_ragged_metadata=None,
        swiglu=None,
        out_block_n=block_n,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
        num_buffers=num_buffers,
    )
    return y


def gluon_bf16_dispatch_swiglu(
    x: torch.Tensor,
    w: torch.Tensor,
    *,
    bias: torch.Tensor | None,
    a_ragged_metadata,
    gather_indx,
    swiglu_alpha: float = 1.0,
    swiglu_limit: float = 0.0,
    block_m: int | None = None,
    block_n: int | None = None,
    block_k: int | None = None,
    num_warps: int | None = None,
    num_buffers: int = _DEFAULT_NUM_BUFFERS,
) -> torch.Tensor:
    """Dispatch + 1st GEMM + fused SwiGLU for MoE.

    The output dtype is ``x.dtype``; the output N is ``w.shape[-1] // 2``
    because SwiGLU consumes pairs of (gate, linear) along the N axis.
    """
    assert _gluon_is_supported(), "Gluon MoE kernel requires CDNA4."
    assert w.ndim == 3 and w.shape[-1] % 2 == 0
    M = x.shape[-2]
    N = w.shape[-1]
    bm, bn, bk, nw = _autotune_block(M, N, w.shape[-2], do_swiglu=True)
    block_m = block_m or bm
    block_n = block_n or bn
    block_k = block_k or bk
    num_warps = num_warps or nw
    out_block_n = block_n // 2
    y = torch.empty((M, N // 2), device=x.device, dtype=x.dtype)
    _launch_pipelined(
        x,
        w,
        y=y,
        bias=bias,
        gather_indx=gather_indx,
        scatter_indx=None,
        gate_scal=None,
        a_ragged_metadata=a_ragged_metadata,
        swiglu=(float(swiglu_alpha), float(swiglu_limit)),
        out_block_n=out_block_n,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
        num_buffers=num_buffers,
    )
    return y


def gluon_bf16_combine(
    x: torch.Tensor,
    w: torch.Tensor,
    *,
    bias: torch.Tensor | None,
    a_ragged_metadata,
    scatter_indx,
    gate_scal: torch.Tensor | None = None,
    n_tokens: int | None = None,
    n_expts_act: int | None = None,
    block_m: int | None = None,
    block_n: int | None = None,
    block_k: int | None = None,
    num_warps: int | None = None,
    num_buffers: int = _DEFAULT_NUM_BUFFERS,
) -> torch.Tensor:
    """2nd GEMM + scatter combine for MoE.

    Accumulates ``y[token] = sum_{e in topk(token)} gate_scal[e] *
    (x_e @ w_e)`` via the kernel's optional scatter+gate_scal post-write.
    The combine across the top-k axis is a final ``view + sum`` on host.
    """
    assert _gluon_is_supported(), "Gluon MoE kernel requires CDNA4."
    assert w.ndim == 3
    M = x.shape[-2]
    N = w.shape[-1]
    if n_tokens is None:
        n_tokens = M
    bm, bn, bk, nw = _autotune_block(
        M, N, w.shape[-2], ragged=a_ragged_metadata is not None
    )
    block_m = block_m or bm
    block_n = block_n or bn
    block_k = block_k or bk
    num_warps = num_warps or nw
    y = torch.zeros((n_tokens, N), device=x.device, dtype=x.dtype)
    _launch_pipelined(
        x,
        w,
        y=y,
        bias=bias,
        gather_indx=None,
        scatter_indx=scatter_indx,
        gate_scal=gate_scal,
        a_ragged_metadata=a_ragged_metadata,
        swiglu=None,
        out_block_n=block_n,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
        num_buffers=num_buffers,
    )
    if n_expts_act is not None and n_expts_act > 1:
        y = y.view(n_tokens, n_expts_act, N).sum(dim=1)
    return y


# ---------------------------------------------------------------------------
# Public Python entry points -- scaled (mxfp4 / fp8) variants
# ---------------------------------------------------------------------------
#
# ``A_FORMAT`` mirrors triton_kernels' ``get_scaled_dot_format_string``
# convention:
#   "e2m1" : mxfp4 (uint8 packed, requires a_block_scale)
#   "e4m3" : fp8 e4m3 (uint8, requires a_global_scale)
#   "e5m2" : fp8 e5m2 (uint8, requires a_global_scale)
# ``B_FORMAT`` is always ``"e2m1"`` in this version (mxfp4 weights).


def gluon_mxfp_gating_gemm(
    x: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    *,
    x_scale: torch.Tensor | None = None,
    a_format: str = "e2m1",
    a_global_scale: float = 1.0,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
    block_m: int | None = None,
    block_n: int | None = None,
    block_k: int | None = None,
    num_warps: int | None = None,
) -> torch.Tensor:
    """Scaled-MFMA dense GEMM ``y = (a_scale * x) @ (w_scale * w)``.

    * ``x`` is a 2-D ``uint8`` tensor encoding the activation. Logical
      K = ``x.shape[-1] * (2 if a_format=='e2m1' else 1)``.
    * ``w`` is a 2-D ``uint8`` tensor encoding mxfp4 weights with logical
      K = ``w.shape[0] * 2``.
    * ``w_scale`` is a 2-D ``uint8`` (e8m0) tensor of shape
      ``[N, K//32]``.
    * If ``a_format == 'e2m1'`` then ``x_scale`` is required, shape
      ``[M, K//32]``. If ``a_format`` is one of the fp8 formats then
      ``a_global_scale`` (a python float) is applied to the accumulator
      after the K loop.
    """
    assert _gluon_is_supported(), "Gluon scaled MoE kernel requires CDNA4."
    assert x.dim() == 2 and w.dim() == 2
    M = x.shape[0]
    N = w.shape[-1]
    div_a = 2 if a_format == "e2m1" else 1
    K = x.shape[-1] * div_a
    bm, bn, bk, nw = _autotune_block(M, N, K, scaled_mfma=True)
    block_m = block_m or bm
    block_n = block_n or bn
    block_k = block_k or bk
    num_warps = num_warps or nw
    y = torch.empty((M, N), device=x.device, dtype=out_dtype)
    _launch_pipelined_scaled(
        x,
        w,
        y=y,
        bias=bias,
        gather_indx=None,
        scatter_indx=None,
        gate_scal=None,
        a_ragged_metadata=None,
        swiglu=None,
        out_block_n=block_n,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
        a_format=a_format,
        b_format="e2m1",
        x_scale=x_scale,
        w_scale=w_scale,
        a_global_scale=a_global_scale,
    )
    return y


def gluon_mxfp_dispatch_swiglu(
    x: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    *,
    x_scale: torch.Tensor | None = None,
    a_format: str = "e2m1",
    a_global_scale: float = 1.0,
    bias: torch.Tensor | None,
    a_ragged_metadata,
    gather_indx,
    out_dtype: torch.dtype = torch.bfloat16,
    swiglu_alpha: float = 1.0,
    swiglu_limit: float = 0.0,
    block_m: int | None = None,
    block_n: int | None = None,
    block_k: int | None = None,
    num_warps: int | None = None,
) -> torch.Tensor:
    """Scaled-MFMA dispatch + 1st GEMM + fused SwiGLU."""
    assert _gluon_is_supported(), "Gluon scaled MoE kernel requires CDNA4."
    assert w.ndim == 3 and w.shape[-1] % 2 == 0
    M = x.shape[-2]
    N = w.shape[-1]
    div_a = 2 if a_format == "e2m1" else 1
    K = x.shape[-1] * div_a
    bm, bn, bk, nw = _autotune_block(M, N, K, do_swiglu=True, scaled_mfma=True)
    block_m = block_m or bm
    block_n = block_n or bn
    block_k = block_k or bk
    num_warps = num_warps or nw
    out_block_n = block_n // 2
    y = torch.empty((M, N // 2), device=x.device, dtype=out_dtype)
    _launch_pipelined_scaled(
        x,
        w,
        y=y,
        bias=bias,
        gather_indx=gather_indx,
        scatter_indx=None,
        gate_scal=None,
        a_ragged_metadata=a_ragged_metadata,
        swiglu=(float(swiglu_alpha), float(swiglu_limit)),
        out_block_n=out_block_n,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
        a_format=a_format,
        b_format="e2m1",
        x_scale=x_scale,
        w_scale=w_scale,
        a_global_scale=a_global_scale,
    )
    return y


def gluon_mxfp_combine(
    x: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    *,
    x_scale: torch.Tensor | None = None,
    a_format: str = "e2m1",
    a_global_scale: float = 1.0,
    bias: torch.Tensor | None,
    a_ragged_metadata,
    scatter_indx,
    gate_scal: torch.Tensor | None = None,
    n_tokens: int | None = None,
    n_expts_act: int | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
    block_m: int | None = None,
    block_n: int | None = None,
    block_k: int | None = None,
    num_warps: int | None = None,
) -> torch.Tensor:
    """Scaled-MFMA 2nd GEMM + scatter combine for MoE."""
    assert _gluon_is_supported(), "Gluon scaled MoE kernel requires CDNA4."
    assert w.ndim == 3
    M = x.shape[-2]
    N = w.shape[-1]
    if n_tokens is None:
        n_tokens = M
    div_a = 2 if a_format == "e2m1" else 1
    K = x.shape[-1] * div_a
    bm, bn, bk, nw = _autotune_block(
        M, N, K, ragged=a_ragged_metadata is not None, scaled_mfma=True
    )
    block_m = block_m or bm
    block_n = block_n or bn
    block_k = block_k or bk
    num_warps = num_warps or nw
    y = torch.zeros((n_tokens, N), device=x.device, dtype=out_dtype)
    _launch_pipelined_scaled(
        x,
        w,
        y=y,
        bias=bias,
        gather_indx=None,
        scatter_indx=scatter_indx,
        gate_scal=gate_scal,
        a_ragged_metadata=a_ragged_metadata,
        swiglu=None,
        out_block_n=block_n,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
        a_format=a_format,
        b_format="e2m1",
        x_scale=x_scale,
        w_scale=w_scale,
        a_global_scale=a_global_scale,
    )
    if n_expts_act is not None and n_expts_act > 1:
        y = y.view(n_tokens, n_expts_act, N).sum(dim=1)
    return y


# ---------------------------------------------------------------------------
# Adapter that matches ``triton_kernels.matmul`` signature (for the
# kernel registry / selector)
# ---------------------------------------------------------------------------


def _gluon_bf16_ragged_matmul(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    a_ragged_metadata=None,
    gather_indx=None,
    scatter_indx=None,
    precision_config=None,
    fused_activation=None,
    n_tokens=None,
    n_expts_act=None,
    block_m: int = _DEFAULT_BLOCK_M,
    block_n: int = _DEFAULT_BLOCK_N,
    block_k: int = _DEFAULT_BLOCK_K,
    num_warps: int = _DEFAULT_NUM_WARPS,
    num_buffers: int = _DEFAULT_NUM_BUFFERS,
    **_unused,
) -> torch.Tensor:
    """Selector-facing entry: matches the upstream ``matmul`` signature.

    Falls back to ``triton_kernels.matmul`` for unsupported precisions
    so we never break the gpt-oss-120b path while we land features.
    """
    if (
        not _supports_pure_bf16(precision_config, fused_activation)
        or not _gluon_is_supported()
    ):
        return _upstream_matmul(
            x,
            w,
            bias,
            a_ragged_metadata=a_ragged_metadata,
            gather_indx=gather_indx,
            scatter_indx=scatter_indx,
            precision_config=precision_config,
            fused_activation=fused_activation,
        )

    # bf16 / fp16 path
    M = x.shape[-2]
    if w.ndim == 3:
        N = w.shape[-1]
        K = w.shape[-2]
    else:
        K, N = w.shape

    # Apply the shape-aware autotune if the caller did not pin block sizes.
    if (
        block_m == _DEFAULT_BLOCK_M
        and block_n == _DEFAULT_BLOCK_N
        and block_k == _DEFAULT_BLOCK_K
    ):
        bm, bn, bk, nw = _autotune_block(M, N, K, ragged=a_ragged_metadata is not None)
        block_m, block_n, block_k = bm, bn, bk
        if num_warps == _DEFAULT_NUM_WARPS:
            num_warps = nw

    out_dtype = (precision_config.out_dtype if precision_config else None) or x.dtype
    if scatter_indx is not None:
        y = torch.zeros((n_tokens or M, N), device=x.device, dtype=out_dtype)
    else:
        y = torch.empty((M, N), device=x.device, dtype=out_dtype)
    _launch_pipelined(
        x,
        w,
        y=y,
        bias=bias,
        gather_indx=gather_indx,
        scatter_indx=scatter_indx,
        gate_scal=None,
        a_ragged_metadata=a_ragged_metadata,
        swiglu=None,
        out_block_n=block_n,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
        num_buffers=num_buffers,
    )
    if scatter_indx is not None and n_expts_act and n_expts_act > 1:
        y = y.view(n_tokens, n_expts_act, N).sum(dim=1)
    return y


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def _kernel_priority() -> int:
    if _GLUON_ENABLED_ENV:
        # Above triton_kernels (which sits at PERFORMANT + 2 = 10) so we
        # actually win selection.
        return Priority.SPECIALIZED + 1  # 13
    # Default: stay below triton_kernels so existing users see no change.
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
    "assert_no_spills",
    "gluon_bf16_combine",
    "gluon_bf16_dispatch_swiglu",
    "gluon_bf16_gating_gemm",
    "gluon_mxfp_combine",
    "gluon_mxfp_dispatch_swiglu",
    "gluon_mxfp_gating_gemm",
    "static_profile",
]
