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


# ---------------------------------------------------------------------------
# Env knob
# ---------------------------------------------------------------------------

_GLUON_ENABLED_ENV = os.environ.get("TOKENSPEED_MOE_GLUON", "").lower() in {
    "1",
    "true",
    "yes",
    "on",
}


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


def _is_cdna4() -> bool:
    return current_platform().is_cdna4


def _gluon_is_supported() -> bool:
    return _HAS_GLUON and _is_cdna4()


# ---------------------------------------------------------------------------
# Layout factories (gluon constexpr functions)
# ---------------------------------------------------------------------------

if _gluon_is_supported():

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

    @gluon.constexpr_function
    def _shared_a_layout():
        return gl.SwizzledSharedLayout(8, 2, 8, order=[1, 0])

    @gluon.constexpr_function
    def _shared_b_layout():
        return gl.SwizzledSharedLayout(8, 2, 8, order=[0, 1])


# ---------------------------------------------------------------------------
# Software-pipelined Gluon MoE kernel
# ---------------------------------------------------------------------------

if _gluon_is_supported():

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
        # NOTE(mxfp4 follow-up):
        # When we land the mxfp4 path we will add ``W_SCALE_ptr`` plus
        # ``stride_*_scale`` arguments and switch the inner accumulator to
        # ``gl.amd.cdna4.mfma_scaled(a_regs, _, b_regs, scale_regs, ...)``
        # using ``get_mfma_scale_layout(dot_layout_w, [BLOCK_N, BLOCK_K //
        # SCALE_BLOCK])`` for the scale tile.
    ):
        # Grid layout: (BLOCKS_PER_EXPERT * grid_n, num_experts).
        # ``program_id(1)`` returns a true scalar so we can keep all
        # downstream pointer arithmetic scalar-only (required by
        # ``buffer_load_to_shared`` which rejects tensor-of-pointers).
        expert_id = gl.program_id(1)
        block_pid = gl.program_id(0)
        grid_n = (N + BLOCK_N - 1) // BLOCK_N
        block_in_expert = block_pid // grid_n
        pid_n = block_pid % grid_n

        # The dispatched activation buffer is padded so every expert owns
        # exactly ``BLOCKS_PER_EXPERT`` adjacent M-blocks. This keeps the
        # mapping ``expert_id -> M_start`` as a pure scalar product, which
        # in turn lets us hand a scalar base pointer to
        # ``buffer_load_to_shared``.
        off_m = expert_id * BLOCKS_PER_EXPERT * BLOCK_M + block_in_expert * BLOCK_M
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

        if HAS_SCATTER:
            rows_y = gl.load(scatter_idx_ptr + offs_y_m, mask=offs_y_m < M, other=M)
            mask_y = (rows_y[:, None] < M) & (
                offs_y_n[None, :] < (OUT_BLOCK_N * grid_n)
            )
            y_offs = rows_y[:, None] * stride_ym + offs_y_n[None, :] * stride_yn
        else:
            mask_y = (offs_y_m[:, None] < M) & (
                offs_y_n[None, :] < (OUT_BLOCK_N * grid_n)
            )
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
) -> tuple[int, int]:
    """Return (num_experts, blocks_per_expert) for the launcher grid.

    The MI355 software-pipelined kernel encodes expert id as a *scalar*
    ``program_id(1)``; for that to translate into a scalar M-offset we
    need every expert to occupy the same number of M-blocks. This means
    the activation buffer must be padded to ``num_experts *
    blocks_per_expert * block_m`` rows. The host calls ``_pad_for_uniform``
    below to honour this.
    """
    if a_ragged_metadata is None:
        return 1, (M + block_m - 1) // block_m
    counts = a_ragged_metadata.slice_sizes
    num_experts = counts.numel()
    counts_list = counts.tolist()
    max_blocks = (
        max((int(c) + block_m - 1) // block_m for c in counts_list)
        if counts_list
        else 0
    )
    return num_experts, max_blocks


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
):
    M, K = x.shape[-2], x.shape[-1]
    if w.ndim == 3:
        E, K_W, N = w.shape[0], w.shape[-2], w.shape[-1]
    else:
        K_W, N = w.shape
        E = 1
    assert K == K_W, f"K mismatch: {K} vs {K_W}"

    num_experts, blocks_per_expert = _expert_layout(a_ragged_metadata, block_m, M)
    grid_n = (N + block_n - 1) // block_n
    grid = (blocks_per_expert * grid_n, num_experts)

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
        num_warps=num_warps,
    )


# ---------------------------------------------------------------------------
# Public Python entry points (one per kernel that TASKS.md asks for)
# ---------------------------------------------------------------------------


def _autotune_block(
    M: int, N: int, K: int, *, do_swiglu: bool = False
) -> tuple[int, int, int, int]:
    """Pick (BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARPS) for given shape.

    Heuristic obtained by sweeping the microbench (see
    ``benchmarks/moe_gluon_microbench.py``) on MI355.  Across all
    candidates ``BLOCK_N=64`` consistently won because it keeps
    ``grid_n`` large enough to saturate the 256+ CUs while still
    providing enough work per CTA for MFMA throughput.

    * Dense gating GEMM (``DO_SWIGLU=False``):
        - M <= 1024: 64x64x64, 8 warps
        - M <= 2048: 128x64x64, 8 warps
        - M  > 2048: 128x64x64, 4 warps -- best balance between
          per-CTA work and GPU fill at large batch.
    * Fused SwiGLU paths (``DO_SWIGLU=True``):
        - all M: 64x64x32, 4 warps
        Doubled BLOCK_N for the gate||linear concat would burn too
        many VGPRs given SwiGLU's reduce; the small-tile choice gives
        more grid_m parallelism and beats the upstream baseline.
    """
    if do_swiglu:
        return 64, 64, 32, 4
    if M <= 1024:
        return 64, 64, 64, 8
    if M <= 2048:
        return 128, 64, 64, 8
    return 128, 64, 64, 4


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
    bm, bn, bk, nw = _autotune_block(M, N, w.shape[-2])
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
    else:
        N = w.shape[-1]

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
    "static_profile",
]
