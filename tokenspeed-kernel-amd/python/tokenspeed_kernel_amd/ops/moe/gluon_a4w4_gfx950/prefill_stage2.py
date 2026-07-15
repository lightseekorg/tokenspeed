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


"""Gluon MXFP4 MoE stage 2: down-projection GEMM with reduce-mode epilogue.

For each expert ``e``::

    out_e += topk_w_e * (inter_e @ w2_e.T)

where ``inter_e`` is stage 1's post-SwiGLU output for the rows routed
to expert ``e``, and ``topk_w_e`` is the routing weight for that
``(token, slot)`` pair.

Kernel shape ("1x4"): 4 wave64/CTA with ``warps_per_cta = [1, 4]``
(four waves split the N axis), ``BLOCK_M`` selected by the caller from
``{32, 64, 128}``, ``BLOCK_N = 256``, and ``BLOCK_K = 128``. At
production K = 256 that is 2 K-iters with a 2-buffer LDS ping-pong and
``v_mfma_scale_f32_16x16x128_f8f6f4``.
A data goes HBM -> LDS via the async copy + ping-pong; A scale, B,
B scale are direct-to-VGPR. B is consumed in the (16, 16) MFMA-tile
layout produced by :func:`_b_preshuffle_3d`.

Epilogue:
  acc (fp32) * routed weight -> bf16 -> ``convert_layout`` through
  LDS to ``BlockedLayout([128, 2], [1, 64])`` -> scatter. All 64 lanes
  hit the same output row, so each store is 2 cache-line writes
  instead of 64 scattered ones. The N axis is split into 4 chunks of
  ``BLOCK_N // 4`` columns; each chunk runs the convert_layout +
  store independently so the scheduler can hide a chunk's LDS
  shuffle behind the prior chunk's VMEM in flight.

Reduce mode (``USE_REDUCE=True``):
  the scatter target becomes ``scratch[token, topk_id, n]`` and the
  per-chunk ``atomic_add`` is replaced with a plain ``gl.store``.
  Each ``(token, slot)`` cell is written by exactly one CTA --
  ``sorted_token_ids`` packs ``(topk_id << 24 | token_id)`` and the
  block-M math sends each entry to a single ``pid_m`` -- so the
  no-atomic store is race-free. The host wrapper then launches a
  small reduce kernel (``gluon_mxfp4_moe_stage2_reduce_kernel``)
  that sums ``scratch`` over the topk dim with fp32 accumulation
  into the user-visible bf16 output. This removes the
  ``M * topk``-way contended global ``atomic_add`` on overlapping
  output rows, which dominates large-M cost when the epilogue does
  the cross-slot reduction inline.

Persistent grid (``PERSISTENT=True``): launch one CTA per N tile per
CU and walk a contiguous-M slice of the tile space inside the kernel.
The launcher always selects the non-persistent grid; the persistent
branch is retained only as a kernel-level config.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd.cdna4 import async_copy as cdna4_async_copy

_USES_FP32_ATOMIC: Optional[bool] = None


def _b_preshuffle_3d(b: torch.Tensor) -> torch.Tensor:
    """Permute a 3-D MoE weight into the (16, 16) MFMA-tile layout.

    Identical to ``stage1_kernel._b_preshuffle_3d`` but kept local so
    this file doesn't import across stage modules. The MFMA
    instruction wants a tile's 16 rows and 16 K-columns contiguous in
    HBM; this permutation arranges them that way. Numerically a no-op.
    """
    assert b.dtype == torch.uint8, "B must be packed fp4 in uint8"
    assert b.ndim == 3, f"B must be 3-D (E, N, K/2), got {tuple(b.shape)}"
    E, N, K_pk = b.shape
    assert N % 16 == 0, f"N ({N}) must be divisible by 16"
    assert K_pk % 32 == 0, f"K/2 ({K_pk}) must be divisible by 32"
    b_6d = b.view(E, N // 16, 16, K_pk // 32, 2, 16)
    return b_6d.permute(0, 1, 3, 4, 2, 5).contiguous().view(E, N, K_pk)


@gluon.jit
def _gdot128_weight_offset(k_pack, n_col, n_phys: gl.constexpr):
    """Flat byte offset for the 128x128 Gluon-dot weight layout."""
    k_in = k_pack % 128
    n_in = n_col % 128
    in_tile = (
        (n_in // 16) * 2048
        + (k_in // 64) * 1024
        + ((k_in // 16) % 4) * 256
        + (n_in % 16) * 16
        + (k_in % 16)
    )
    n_tiles: gl.constexpr = n_phys // 128
    tile = (k_pack // 128) * n_tiles + (n_col // 128)
    return tile * (128 * 128) + in_tile


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


@gluon.jit
def gluon_mxfp4_moe_stage2_1x2_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_scales_ptr,
    b_scales_ptr,
    sorted_token_ids_ptr,
    sorted_expert_ids_ptr,
    num_tokens_post_padded_ptr,
    sorted_weights_ptr,
    N,
    K,
    EM,
    num_valid_tokens,
    token_num,
    top_k,
    stride_am,
    stride_ak,
    stride_be,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    stride_ase_m,
    stride_ase_k,
    stride_bse_e,
    stride_bse_n,
    stride_bse_k,
    stride_se_n_pad,
    K_PACKED_TOTAL: gl.constexpr,
    N_PHYS: gl.constexpr,
    BLOCK_M: gl.constexpr,
    SORT_BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    B_GDOT128: gl.constexpr = False,
    USE_REDUCE: gl.constexpr = False,
    MFMA_STORE_LAYOUT: gl.constexpr = False,
    PERSISTENT: gl.constexpr = True,
    CU_NUM: gl.constexpr = 256,
    COALESCE_SCALES: gl.constexpr = False,
    DIRECT_SCALE_LAYOUT: gl.constexpr = False,
    DEFER_EPILOGUE: gl.constexpr = True,
):
    """Stage 2 1x2 kernel: 2 waves/CTA, BLOCK_N=256, 2-stage v3 pipeline.

    When ``USE_REDUCE`` is True, the epilogue writes per-(token, slot)
    partials with plain stores instead of contending atomic_adds; the
    user must separately invoke ``gluon_mxfp4_moe_stage2_reduce_kernel``
    to sum across the topk dim. ``c_ptr`` is then expected to point at
    a ``[token_num * topk, N]`` bf16 scratch buffer; ``stride_cm`` /
    ``stride_cn`` describe that shape (``stride_cm = N``).

    When ``PERSISTENT`` is True, the kernel is launched with
    ``grid = (CU_NUM,)`` and each CTA walks a contiguous-M slice of the
    ``num_pid_m * num_pid_n`` tile space. ``flat_tile = cta_id *
    tiles_per_block + tile_iter``; ``pid_m = flat_tile % num_pid_m``,
    ``pid_n = flat_tile // num_pid_m``. Iterating M-fast within a
    fixed N means consecutive iters share the same expert
    (``sorted_expert_ids`` is monotone in M), so that expert's B
    weight stays L2-resident across iters. ``PERSISTENT=False``
    (the default for this wrapper) launches the per-tile grid
    ``(num_pid_m * num_pid_n,)``.
    """

    gl.static_assert(
        BLOCK_M == 32 or BLOCK_M == 64 or BLOCK_M == 128,
        "1x2 kernel requires BLOCK_M in {32, 64, 128}",
    )
    gl.static_assert(
        SORT_BLOCK_M == 32 or SORT_BLOCK_M == 64 or SORT_BLOCK_M == 128,
        "1x2 kernel requires SORT_BLOCK_M in {32, 64, 128}",
    )
    gl.static_assert(SORT_BLOCK_M % BLOCK_M == 0)
    gl.static_assert(BLOCK_N == 256, "1x2 kernel requires BLOCK_N=256")
    gl.static_assert(BLOCK_K == 128, "1x2 kernel requires BLOCK_K=128")
    gl.static_assert(NUM_WARPS == 4, "1x2 kernel requires NUM_WARPS=4")

    BLOCK_K_PACKED: gl.constexpr = BLOCK_K // 2  # 64
    BLOCK_K_SCALE: gl.constexpr = BLOCK_K // 32  # 4
    NUM_BUFFERS: gl.constexpr = 2
    # 4 N-chunks of 64 cols each at BLOCK_N = 256. With warps_per_cta=[1, 4]
    # each wave owns 1/4 of a chunk = 16 cols, so per wave [128, 16] per chunk.
    NUM_N_CHUNKS: gl.constexpr = 4
    BLOCK_N_CHUNK: gl.constexpr = BLOCK_N // NUM_N_CHUNKS  # 64

    # MFMA + dot operand + scale layouts. warps_per_cta=[1, 4] splits the N
    # axis across 4 waves; each wave's portion of acc[BLOCK_M, BLOCK_N=256]
    # is [128, 64], and each per-chunk [BLOCK_M, BLOCK_N_CHUNK=64] tile is
    # [128, 16] per wave. Halves per-wave accumulator and B operand vs the
    # earlier 2-wave layout -- the change targets register pressure to fit
    # more than one wave per SIMD on MI350.
    # COALESCE_SCALES does not change this layout; it only changes how the
    # scale operands are loaded (address-ordered coalesced load + convert to
    # the operand layout). Supported for BLOCK_M == 64, BLOCK_N == 256 only;
    # the caller must gate the flag on that.
    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 128],
        transposed=True,
        warps_per_cta=[1, 4],
    )
    dot_a_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=16
    )
    dot_b_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=16
    )
    a_scale_layout: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(
        dot_a_layout, [BLOCK_M, BLOCK_K_SCALE]
    )
    BLOCK_K_SCALE_FULL: gl.constexpr = 2 * BLOCK_K_SCALE  # K // 32 = 8
    a_scale_layout_full: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(
        dot_a_layout, [BLOCK_M, BLOCK_K_SCALE_FULL]
    )
    # Per-chunk B-scale layout. mfma_scaled requires the scale shape to be
    # consistent with the per-chunk B operand [BLOCK_N_CHUNK, K_SCALE].
    b_scale_layout_chunk: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(
        dot_b_layout, [BLOCK_N_CHUNK, BLOCK_K_SCALE]
    )
    # Full-N, full-K B-scale operand layout (COALESCE_SCALES).
    b_scale_layout_full: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(
        dot_b_layout, [BLOCK_N, BLOCK_K_SCALE_FULL]
    )

    # A global-load layout for HBM -> LDS async copy.  The CDNA4
    # buffer_load_to_shared verifier accepts BlockedLayout/SliceLayout
    # offsets, not the arbitrary DistributedLinearLayout we use for some
    # direct VGPR loads.  Each thread reads 16 contiguous bytes along K
    # (128-bit load).  BM128 uses the gdot128 K64 split; BM32/BM64 use
    # a [2,2] warp split, which is the verifier-safe layout that preserves
    # identical numerics against BM128 for the package stage2 contract.
    # The LDS layout below remains the original padded swizzle consumed by
    # the MFMA dot operand.
    if BLOCK_M == 32:
        gload_a_layout: gl.constexpr = gl.BlockedLayout(
            [1, 16],
            [16, 4],
            [2, 2],
            [1, 0],
        )
        shared_a_bases: gl.constexpr = [
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 8],
            [0, 16],
            [0, 32],
            [1, 0],
            [2, 0],
            [4, 0],
            [8, 0],
            [16, 0],
        ]
    elif BLOCK_M == 64:
        gload_a_layout: gl.constexpr = gl.BlockedLayout(
            [1, 16],
            [16, 4],
            [2, 2],
            [1, 0],
        )
        shared_a_bases: gl.constexpr = [
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 8],
            [0, 16],
            [0, 32],
            [1, 0],
            [2, 0],
            [4, 0],
            [8, 0],
            [16, 0],
            [32, 0],
        ]
    else:
        gload_a_layout: gl.constexpr = gl.BlockedLayout(
            [1, 16],
            [16, 4],
            [4, 1],
            [1, 0],
        )
        shared_a_bases: gl.constexpr = [
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 8],
            [0, 16],
            [0, 32],
            [1, 0],
            [2, 0],
            [4, 0],
            [8, 0],
            [16, 0],
            [32, 0],
            [64, 0],
        ]

    # Padded shared layout for A data. Scales bypass LDS (direct-to-VGPR).
    shared_a: gl.constexpr = gl.PaddedSharedLayout(
        [[1024, 16]],
        shared_a_bases,
        [],
        [BLOCK_M, BLOCK_K_PACKED],
    )

    cta_id = gl.program_id(axis=0)
    num_pid_m = gl.cdiv(EM, BLOCK_M)
    num_pid_n = gl.cdiv(N, BLOCK_N)
    total_tiles = num_pid_m * num_pid_n
    num_tokens_post_padded = gl.load(num_tokens_post_padded_ptr)
    if PERSISTENT:
        worker_id = gl.program_id(axis=1)
        worker_count = gl.num_programs(axis=1)
        m_tiles_floor = num_pid_m // worker_count
        m_tiles_rem = num_pid_m - m_tiles_floor * worker_count
        m_tile_extra = gl.where(worker_id < m_tiles_rem, 1, 0)
        start_pid_m = worker_id * m_tiles_floor + gl.minimum(worker_id, m_tiles_rem)
        tiles_per_block = m_tiles_floor + m_tile_extra
    else:
        tiles_per_block = 1
    # Persistent tile loop: grid axis 0 owns a fixed N tile;
    # grid axis 1 owns a contiguous floor/ceil M-tile slice.  This keeps
    # adjacent iterations in the same or adjacent expert ranges so B rows
    # stay L2-resident across the worker's slice.
    for tile_iter in range(0, tiles_per_block):
        if PERSISTENT:
            pid_m = start_pid_m + tile_iter
            pid_n = cta_id
            flat_tile = pid_n * num_pid_m + pid_m
        else:
            flat_tile = cta_id
            pid_m = flat_tile % num_pid_m
            pid_n = flat_tile // num_pid_m
        tile_ok = (
            (pid_n < num_pid_n)
            & (pid_m < num_pid_m)
            & (pid_m * BLOCK_M < num_tokens_post_padded)
        )
        if tile_ok:
            # A row gather (topk-aware): sorted_token_ids packs (topk_id << 24 | token_id)
            m_layout: gl.constexpr = gl.SliceLayout(1, gload_a_layout)
            k_layout: gl.constexpr = gl.SliceLayout(0, gload_a_layout)
            offs_sorted_slot = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=m_layout)
            # ``buffer_load`` (SRD-relative addressing) instead of
            # ``gl.load(ptr + offs)`` (absolute 64-bit ``global_load``):
            # the latter cost ~1 M cycles of global_load stall on this
            # gather alone in the previous ATT run.
            # No runtime mask: EM is BLOCK_M-aligned (SORT_BLOCK_M % BLOCK_M == 0,
            # EM % SORT_BLOCK_M == 0), so offs_sorted_slot < EM always holds for
            # valid tiles. The mask/other lower to a v_cmp + 2 v_cndmask per
            # element (x4 per lane); padding slots carry a sentinel token_id >=
            # token_num that token_mask/tok_ok gate downstream.
            offs_token = gl.amd.cdna4.buffer_load(
                ptr=sorted_token_ids_ptr,
                offsets=offs_sorted_slot,
            )
            token_id = offs_token & 0xFFFFFF
            topk_id = offs_token >> 24
            inter_row = token_id * top_k + topk_id
            token_mask = token_id < token_num

            offs_ak = gl.arange(0, BLOCK_K_PACKED, layout=k_layout)
            offs_a = (
                inter_row[:, None].to(gl.int64) * stride_am
                + offs_ak[None, :] * stride_ak
            )

            # Expert id is indexed at the sort-block granularity.  FlyDSL
            # uses tile_m=64 with sort_block_m=128 for large Kimi shapes, so
            # two compute M-tiles share one sorted_expert_ids entry.
            sort_pid_m = (pid_m * BLOCK_M) // SORT_BLOCK_M
            off_experts = gl.load(sorted_expert_ids_ptr + sort_pid_m)
            if off_experts != -1:

                # A_scale offsets (direct-to-VGPR). Half-bit ordering
                # matches what the upstream sort kernel writes:
                # m-half at +1, y-half at +2 within a 4-byte panel.
                m_scale_layout: gl.constexpr = gl.SliceLayout(1, a_scale_layout)
                k_scale_layout: gl.constexpr = gl.SliceLayout(0, a_scale_layout)
                a_scale_rows = (
                    pid_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=m_scale_layout)
                ).to(gl.uint32)
                a_m_part = (
                    (a_scale_rows // 32) * (stride_se_n_pad * 32)
                    + (a_scale_rows % 16) * 4
                    + ((a_scale_rows % 32) // 16)
                )[:, None]
                a_k_lanes = gl.arange(0, BLOCK_K_SCALE, layout=k_scale_layout).to(
                    gl.uint32
                )

                if COALESCE_SCALES:
                    if DIRECT_SCALE_LAYOUT:
                        # Diagnostic path: load A scales directly in the MFMA
                        # scale layout.  This removes the full-layout
                        # convert_layout that emits v_cndmask near line 395,
                        # at the cost of a less coalescer-friendly load shape.
                        m_full_layout: gl.constexpr = gl.SliceLayout(
                            1, a_scale_layout_full
                        )
                        k_full_layout: gl.constexpr = gl.SliceLayout(
                            0, a_scale_layout_full
                        )
                        a_scale_rows_full = (
                            pid_m * BLOCK_M
                            + gl.arange(0, BLOCK_M, layout=m_full_layout)
                        ).to(gl.uint32)
                        a_k_full = gl.arange(
                            0, BLOCK_K_SCALE_FULL, layout=k_full_layout
                        ).to(gl.uint32)
                        a_m_part_full = (
                            (a_scale_rows_full // 32) * (stride_se_n_pad * 32)
                            + (a_scale_rows_full % 16) * 4
                            + ((a_scale_rows_full % 32) // 16)
                        )[:, None]
                        a_k_part_full = ((a_k_full % 4) * 64 + (a_k_full // 4) * 2)[
                            None, :
                        ]
                        a_scale_all = gl.amd.cdna4.buffer_load(
                            ptr=a_scales_ptr,
                            offsets=a_m_part_full + a_k_part_full,
                        )
                    else:
                        # Coalesced full-K A-scale load via a 3D [m_hi=m//32,
                        # m_lo=m%32, k] layout so the coalescer proof gets
                        # clean per-axis atoms. Then convert to the operand
                        # layout and split into the K-iters.
                        if BLOCK_M == 32:
                            gload_a3d: gl.constexpr = gl.DistributedLinearLayout(
                                reg_bases=[[0, 16, 0], [0, 0, 4]],
                                lane_bases=[
                                    [0, 1, 0],
                                    [0, 2, 0],
                                    [0, 4, 0],
                                    [0, 8, 0],
                                    [0, 0, 1],
                                    [0, 0, 2],
                                ],
                                warp_bases=[[0, 0, 0], [0, 0, 0]],
                                block_bases=[],
                                shape=[BLOCK_M // 32, 32, BLOCK_K_SCALE_FULL],
                            )
                        elif BLOCK_M == 64:
                            gload_a3d: gl.constexpr = gl.DistributedLinearLayout(
                                reg_bases=[[0, 16, 0], [0, 0, 4], [0, 1, 0]],
                                lane_bases=[
                                    [1, 0, 0],
                                    [0, 2, 0],
                                    [0, 4, 0],
                                    [0, 8, 0],
                                    [0, 0, 1],
                                    [0, 0, 2],
                                ],
                                warp_bases=[[0, 0, 0], [0, 0, 0]],
                                block_bases=[],
                                shape=[BLOCK_M // 32, 32, BLOCK_K_SCALE_FULL],
                            )
                        else:  # BLOCK_M == 128
                            gload_a3d: gl.constexpr = gl.DistributedLinearLayout(
                                reg_bases=[
                                    [0, 16, 0],
                                    [0, 0, 4],
                                    [0, 1, 0],
                                    [0, 2, 0],
                                ],
                                lane_bases=[
                                    [1, 0, 0],
                                    [2, 0, 0],
                                    [0, 4, 0],
                                    [0, 8, 0],
                                    [0, 0, 1],
                                    [0, 0, 2],
                                ],
                                warp_bases=[[0, 0, 0], [0, 0, 0]],
                                block_bases=[],
                                shape=[BLOCK_M // 32, 32, BLOCK_K_SCALE_FULL],
                            )
                        l_amhi: gl.constexpr = gl.SliceLayout(
                            1, gl.SliceLayout(2, gload_a3d)
                        )
                        l_amlo: gl.constexpr = gl.SliceLayout(
                            0, gl.SliceLayout(2, gload_a3d)
                        )
                        l_ak: gl.constexpr = gl.SliceLayout(
                            0, gl.SliceLayout(1, gload_a3d)
                        )
                        a_mhi = (
                            pid_m * (BLOCK_M // 32)
                            + gl.arange(0, BLOCK_M // 32, layout=l_amhi)
                        ).to(gl.uint32)
                        a_mlo = gl.arange(0, 32, layout=l_amlo).to(gl.uint32)
                        a_kax = gl.arange(0, BLOCK_K_SCALE_FULL, layout=l_ak).to(
                            gl.uint32
                        )
                        a_off3d = (
                            a_mhi[:, None, None] * (stride_se_n_pad * 32)
                            + ((a_mlo % 16) * 4 + (a_mlo // 16))[None, :, None]
                            + ((a_kax % 4) * 64 + (a_kax // 4) * 2)[None, None, :]
                        )
                        a_scale_all = gl.amd.cdna4.buffer_load(
                            ptr=a_scales_ptr, offsets=a_off3d
                        )
                        a_scale_all = gl.reshape(
                            a_scale_all, [BLOCK_M, BLOCK_K_SCALE_FULL]
                        )
                        a_scale_all = gl.convert_layout(
                            a_scale_all, a_scale_layout_full
                        )
                    a_scale_all = gl.reshape(a_scale_all, [BLOCK_M, 2, BLOCK_K_SCALE])
                    a_scale_all = gl.permute(a_scale_all, [0, 2, 1])
                    a_scale0_raw, a_scale1_raw = gl.split(a_scale_all)
                    a_scale0 = gl.convert_layout(a_scale0_raw, a_scale_layout)
                    a_scale1 = gl.convert_layout(a_scale1_raw, a_scale_layout)

                # B data offsets (preshuffled B, direct-to-VGPR).
                # NOTE: dot_b_layout is shape-agnostic, so we reuse it for the [K, 32]
                # per-chunk tile. SliceLayout(0,dot_b_layout) over arange(0,32) yields the
                # natural N-axis sublayout for each 32-col chunk.
                offs_bk = gl.arange(
                    0, BLOCK_K_PACKED, layout=gl.SliceLayout(1, dot_b_layout)
                )
                b_k_part = (
                    (offs_bk // 32) * 512 + ((offs_bk // 16) % 2) * 256 + (offs_bk % 16)
                )[:, None]

                bn_slice: gl.constexpr = gl.SliceLayout(0, dot_b_layout)
                # Kimi/DSv3 stage2 only dispatches this 1x2 kernel for N
                # divisible by BLOCK_N=256, so every pid_n chunk is in-bounds.
                # Avoid dynamic modulo in the B address arithmetic; the
                # stage2 kernels rely on full-N tiles for these shapes.
                offs_bn_c0 = (
                    pid_n * BLOCK_N
                    + 0 * BLOCK_N_CHUNK
                    + gl.arange(0, BLOCK_N_CHUNK, layout=bn_slice)
                )
                offs_bn_c1 = (
                    pid_n * BLOCK_N
                    + 1 * BLOCK_N_CHUNK
                    + gl.arange(0, BLOCK_N_CHUNK, layout=bn_slice)
                )
                offs_bn_c2 = (
                    pid_n * BLOCK_N
                    + 2 * BLOCK_N_CHUNK
                    + gl.arange(0, BLOCK_N_CHUNK, layout=bn_slice)
                )
                offs_bn_c3 = (
                    pid_n * BLOCK_N
                    + 3 * BLOCK_N_CHUNK
                    + gl.arange(0, BLOCK_N_CHUNK, layout=bn_slice)
                )
                if B_GDOT128:
                    offs_b_c0 = _gdot128_weight_offset(
                        offs_bk[:, None], offs_bn_c0[None, :], N_PHYS
                    )
                    offs_b_c1 = _gdot128_weight_offset(
                        offs_bk[:, None], offs_bn_c1[None, :], N_PHYS
                    )
                    offs_b_c2 = _gdot128_weight_offset(
                        offs_bk[:, None], offs_bn_c2[None, :], N_PHYS
                    )
                    offs_b_c3 = _gdot128_weight_offset(
                        offs_bk[:, None], offs_bn_c3[None, :], N_PHYS
                    )
                else:
                    offs_b_c0 = (
                        b_k_part
                        + (
                            (offs_bn_c0 // 16) * (16 * K_PACKED_TOTAL)
                            + (offs_bn_c0 % 16) * 16
                        )[None, :]
                    )
                    offs_b_c1 = (
                        b_k_part
                        + (
                            (offs_bn_c1 // 16) * (16 * K_PACKED_TOTAL)
                            + (offs_bn_c1 % 16) * 16
                        )[None, :]
                    )
                    offs_b_c2 = (
                        b_k_part
                        + (
                            (offs_bn_c2 // 16) * (16 * K_PACKED_TOTAL)
                            + (offs_bn_c2 % 16) * 16
                        )[None, :]
                    )
                    offs_b_c3 = (
                        b_k_part
                        + (
                            (offs_bn_c3 // 16) * (16 * K_PACKED_TOTAL)
                            + (offs_bn_c3 % 16) * 16
                        )[None, :]
                    )

                # B_scale offsets (direct-to-VGPR), per-chunk along N.
                b_n_layout: gl.constexpr = gl.SliceLayout(1, b_scale_layout_chunk)
                b_k_layout: gl.constexpr = gl.SliceLayout(0, b_scale_layout_chunk)
                b_rows_c0 = (
                    pid_n * BLOCK_N
                    + 0 * BLOCK_N_CHUNK
                    + gl.arange(0, BLOCK_N_CHUNK, layout=b_n_layout)
                ).to(gl.uint32)
                b_rows_c1 = (
                    pid_n * BLOCK_N
                    + 1 * BLOCK_N_CHUNK
                    + gl.arange(0, BLOCK_N_CHUNK, layout=b_n_layout)
                ).to(gl.uint32)
                b_rows_c2 = (
                    pid_n * BLOCK_N
                    + 2 * BLOCK_N_CHUNK
                    + gl.arange(0, BLOCK_N_CHUNK, layout=b_n_layout)
                ).to(gl.uint32)
                b_rows_c3 = (
                    pid_n * BLOCK_N
                    + 3 * BLOCK_N_CHUNK
                    + gl.arange(0, BLOCK_N_CHUNK, layout=b_n_layout)
                ).to(gl.uint32)
                b_n_scale_part_c0 = (
                    (b_rows_c0 // 32) * (stride_bse_n * 32)
                    + (b_rows_c0 % 16) * 4
                    + ((b_rows_c0 % 32) // 16)
                )[:, None]
                b_n_scale_part_c1 = (
                    (b_rows_c1 // 32) * (stride_bse_n * 32)
                    + (b_rows_c1 % 16) * 4
                    + ((b_rows_c1 % 32) // 16)
                )[:, None]
                b_n_scale_part_c2 = (
                    (b_rows_c2 // 32) * (stride_bse_n * 32)
                    + (b_rows_c2 % 16) * 4
                    + ((b_rows_c2 % 32) // 16)
                )[:, None]
                b_n_scale_part_c3 = (
                    (b_rows_c3 // 32) * (stride_bse_n * 32)
                    + (b_rows_c3 % 16) * 4
                    + ((b_rows_c3 % 32) // 16)
                )[:, None]

                b_k_lanes = gl.arange(0, BLOCK_K_SCALE, layout=b_k_layout).to(gl.uint32)

                # LDS holds A data only; 2 ping-pong buffers
                smem_a = gl.allocate_shared_memory(
                    a_ptr.type.element_ty,
                    [NUM_BUFFERS, BLOCK_M, BLOCK_K_PACKED],
                    shared_a,
                )

                a_base_ptr = a_ptr
                b_base_ptr = b_ptr + off_experts.to(gl.int64) * stride_be
                b_scale_base = b_scales_ptr + off_experts.to(gl.uint32) * stride_bse_e
                offs_a_i32 = offs_a.to(gl.int32)

                A_DATA_K_STEP: gl.constexpr = BLOCK_K_PACKED  # 64
                if B_GDOT128:
                    B_DATA_K_STEP1: gl.constexpr = 1024
                    B_DATA_K_STEP2: gl.constexpr = (N_PHYS // 128) * (128 * 128)
                    B_DATA_K_STEP3: gl.constexpr = B_DATA_K_STEP2 + 1024
                else:
                    B_DATA_K_STEP1: gl.constexpr = (BLOCK_K_PACKED // 32) * 512
                    B_DATA_K_STEP2: gl.constexpr = 2 * B_DATA_K_STEP1
                    B_DATA_K_STEP3: gl.constexpr = 3 * B_DATA_K_STEP1

                # 4 independent accumulators of shape [BLOCK_M, BLOCK_N_CHUNK=32].
                # mfma_layout is shape-agnostic; each acc covers 8 (M) x 2 (N) x 2 (K) =
                # 32 v_mfma_scale_f32_16x16x128_f8f6f4 instructions total, summing to 128.
                acc0 = gl.zeros(
                    (BLOCK_M, BLOCK_N_CHUNK), dtype=gl.float32, layout=mfma_layout
                )
                acc1 = gl.zeros(
                    (BLOCK_M, BLOCK_N_CHUNK), dtype=gl.float32, layout=mfma_layout
                )
                acc2 = gl.zeros(
                    (BLOCK_M, BLOCK_N_CHUNK), dtype=gl.float32, layout=mfma_layout
                )
                acc3 = gl.zeros(
                    (BLOCK_M, BLOCK_N_CHUNK), dtype=gl.float32, layout=mfma_layout
                )

                # ---- paired K=128 subtiles (production K=256) -------------------
                # Issue both K-subtiles' B/B_scale/A_scale loads before the first
                # MFMA block so tile-1 global-load latency overlaps tile-0 compute.
                cdna4_async_copy.buffer_load_to_shared(
                    smem_a.index(0),
                    a_base_ptr,
                    offs_a_i32,
                    mask=token_mask[:, None],
                )
                cdna4_async_copy.commit_group()
                cdna4_async_copy.buffer_load_to_shared(
                    smem_a.index(1),
                    a_base_ptr,
                    offs_a_i32 + A_DATA_K_STEP,
                    mask=token_mask[:, None],
                )
                cdna4_async_copy.commit_group()

                if not COALESCE_SCALES:
                    a_k_iter0 = a_k_lanes
                    a_k_part0 = (
                        (a_k_iter0 // 8) * 256
                        + (a_k_iter0 % 4) * 64
                        + ((a_k_iter0 % 8) // 4) * 2
                    )[None, :]
                    a_scale0 = gl.amd.cdna4.buffer_load(
                        ptr=a_scales_ptr, offsets=a_m_part + a_k_part0
                    )
                b0_c0 = gl.amd.cdna4.buffer_load(ptr=b_base_ptr, offsets=offs_b_c0)
                b0_c1 = gl.amd.cdna4.buffer_load(ptr=b_base_ptr, offsets=offs_b_c1)
                b0_c2 = gl.amd.cdna4.buffer_load(ptr=b_base_ptr, offsets=offs_b_c2)
                b0_c3 = gl.amd.cdna4.buffer_load(ptr=b_base_ptr, offsets=offs_b_c3)
                if not COALESCE_SCALES:
                    bsc_k_iter0 = b_k_lanes
                    bsc_k_off0 = (
                        (bsc_k_iter0 // 8) * 256
                        + (bsc_k_iter0 % 4) * 64
                        + ((bsc_k_iter0 % 8) // 4) * 2
                    )[None, :]
                    b_scale0_c0 = gl.amd.cdna4.buffer_load(
                        ptr=b_scale_base, offsets=b_n_scale_part_c0 + bsc_k_off0
                    )
                    b_scale0_c1 = gl.amd.cdna4.buffer_load(
                        ptr=b_scale_base, offsets=b_n_scale_part_c1 + bsc_k_off0
                    )
                    b_scale0_c2 = gl.amd.cdna4.buffer_load(
                        ptr=b_scale_base, offsets=b_n_scale_part_c2 + bsc_k_off0
                    )
                    b_scale0_c3 = gl.amd.cdna4.buffer_load(
                        ptr=b_scale_base, offsets=b_n_scale_part_c3 + bsc_k_off0
                    )

                if not COALESCE_SCALES:
                    a_k_iter1 = BLOCK_K_SCALE + a_k_lanes
                    a_k_part1 = (
                        (a_k_iter1 // 8) * 256
                        + (a_k_iter1 % 4) * 64
                        + ((a_k_iter1 % 8) // 4) * 2
                    )[None, :]
                    a_scale1 = gl.amd.cdna4.buffer_load(
                        ptr=a_scales_ptr, offsets=a_m_part + a_k_part1
                    )
                b1_c0 = gl.amd.cdna4.buffer_load(
                    ptr=b_base_ptr, offsets=offs_b_c0 + B_DATA_K_STEP1
                )
                b1_c1 = gl.amd.cdna4.buffer_load(
                    ptr=b_base_ptr, offsets=offs_b_c1 + B_DATA_K_STEP1
                )
                b1_c2 = gl.amd.cdna4.buffer_load(
                    ptr=b_base_ptr, offsets=offs_b_c2 + B_DATA_K_STEP1
                )
                b1_c3 = gl.amd.cdna4.buffer_load(
                    ptr=b_base_ptr, offsets=offs_b_c3 + B_DATA_K_STEP1
                )
                if not COALESCE_SCALES:
                    bsc_k_iter1 = BLOCK_K_SCALE + b_k_lanes
                    bsc_k_off1 = (
                        (bsc_k_iter1 // 8) * 256
                        + (bsc_k_iter1 % 4) * 64
                        + ((bsc_k_iter1 % 8) // 4) * 2
                    )[None, :]
                    b_scale1_c0 = gl.amd.cdna4.buffer_load(
                        ptr=b_scale_base, offsets=b_n_scale_part_c0 + bsc_k_off1
                    )
                    b_scale1_c1 = gl.amd.cdna4.buffer_load(
                        ptr=b_scale_base, offsets=b_n_scale_part_c1 + bsc_k_off1
                    )
                    b_scale1_c2 = gl.amd.cdna4.buffer_load(
                        ptr=b_scale_base, offsets=b_n_scale_part_c2 + bsc_k_off1
                    )
                    b_scale1_c3 = gl.amd.cdna4.buffer_load(
                        ptr=b_scale_base, offsets=b_n_scale_part_c3 + bsc_k_off1
                    )

                if COALESCE_SCALES:
                    # One coalesced full-N, full-K B-scale load via a 4D
                    # [n7, n6, n_lo=n%64, k] layout (clean per-axis atoms; the
                    # contiguous byte pair straddles two N-chunks so this must be
                    # a single full-N load), then convert to the full operand
                    # layout and slice into the 4 chunks x 2 K-iters. Registers =
                    # {n6->+1, k2->+2, n0->+4} (address order) -> dwordx2; n4,n5
                    # map to the N warps. Requires BLOCK_N == 256 (caller-gated).
                    gload_b4d: gl.constexpr = gl.DistributedLinearLayout(
                        reg_bases=[[0, 1, 0, 0], [0, 0, 0, 4], [0, 0, 1, 0]],
                        lane_bases=[
                            [1, 0, 0, 0],
                            [0, 0, 2, 0],
                            [0, 0, 4, 0],
                            [0, 0, 8, 0],
                            [0, 0, 0, 1],
                            [0, 0, 0, 2],
                        ],
                        warp_bases=[[0, 0, 16, 0], [0, 0, 32, 0]],
                        block_bases=[],
                        shape=[BLOCK_N // 128, 2, 64, BLOCK_K_SCALE_FULL],
                    )
                    l_bn7: gl.constexpr = gl.SliceLayout(
                        1, gl.SliceLayout(2, gl.SliceLayout(3, gload_b4d))
                    )
                    l_bn6: gl.constexpr = gl.SliceLayout(
                        0, gl.SliceLayout(2, gl.SliceLayout(3, gload_b4d))
                    )
                    l_bnlo: gl.constexpr = gl.SliceLayout(
                        0, gl.SliceLayout(1, gl.SliceLayout(3, gload_b4d))
                    )
                    l_bk: gl.constexpr = gl.SliceLayout(
                        0, gl.SliceLayout(1, gl.SliceLayout(2, gload_b4d))
                    )
                    bn7 = gl.arange(0, BLOCK_N // 128, layout=l_bn7).to(gl.uint32)
                    bn6 = gl.arange(0, 2, layout=l_bn6).to(gl.uint32)
                    bnlo = gl.arange(0, 64, layout=l_bnlo).to(gl.uint32)
                    bk = gl.arange(0, BLOCK_K_SCALE_FULL, layout=l_bk).to(gl.uint32)
                    n7abs = pid_n * (BLOCK_N // 128) + bn7
                    boff4d = (
                        n7abs[:, None, None, None] * (stride_se_n_pad * 128)
                        + bn6[None, :, None, None]
                        + (bnlo * 4)[None, None, :, None]
                        + ((bk % 4) * 256 + (bk // 4) * 2)[None, None, None, :]
                    )
                    b_all = gl.amd.cdna4.buffer_load(ptr=b_scale_base, offsets=boff4d)
                    b_all = gl.reshape(b_all, [BLOCK_N, BLOCK_K_SCALE_FULL])
                    b_all = gl.convert_layout(b_all, b_scale_layout_full)
                    b_all = gl.reshape(b_all, [BLOCK_N, 2, BLOCK_K_SCALE])
                    b_all = gl.permute(b_all, [0, 2, 1])
                    b_it0, b_it1 = gl.split(b_all)
                    t0 = gl.permute(
                        gl.reshape(b_it0, [BLOCK_N // 128, 2, 64, BLOCK_K_SCALE]),
                        [2, 3, 0, 1],
                    )
                    t0a, t0b = gl.split(t0)
                    b_scale0_c0, b_scale0_c2 = gl.split(t0a)
                    b_scale0_c1, b_scale0_c3 = gl.split(t0b)
                    t1 = gl.permute(
                        gl.reshape(b_it1, [BLOCK_N // 128, 2, 64, BLOCK_K_SCALE]),
                        [2, 3, 0, 1],
                    )
                    t1a, t1b = gl.split(t1)
                    b_scale1_c0, b_scale1_c2 = gl.split(t1a)
                    b_scale1_c1, b_scale1_c3 = gl.split(t1b)
                    b_scale0_c0 = gl.convert_layout(b_scale0_c0, b_scale_layout_chunk)
                    b_scale0_c1 = gl.convert_layout(b_scale0_c1, b_scale_layout_chunk)
                    b_scale0_c2 = gl.convert_layout(b_scale0_c2, b_scale_layout_chunk)
                    b_scale0_c3 = gl.convert_layout(b_scale0_c3, b_scale_layout_chunk)
                    b_scale1_c0 = gl.convert_layout(b_scale1_c0, b_scale_layout_chunk)
                    b_scale1_c1 = gl.convert_layout(b_scale1_c1, b_scale_layout_chunk)
                    b_scale1_c2 = gl.convert_layout(b_scale1_c2, b_scale_layout_chunk)
                    b_scale1_c3 = gl.convert_layout(b_scale1_c3, b_scale_layout_chunk)

                cdna4_async_copy.wait_group(1)
                a0 = cdna4_async_copy.load_shared_relaxed(smem_a.index(0), dot_a_layout)
                acc0 = gl.amd.cdna4.mfma_scaled(
                    a=a0,
                    a_scale=a_scale0,
                    a_format="e2m1",
                    b=b0_c0,
                    b_scale=b_scale0_c0,
                    b_format="e2m1",
                    acc=acc0,
                )
                acc1 = gl.amd.cdna4.mfma_scaled(
                    a=a0,
                    a_scale=a_scale0,
                    a_format="e2m1",
                    b=b0_c1,
                    b_scale=b_scale0_c1,
                    b_format="e2m1",
                    acc=acc1,
                )
                acc2 = gl.amd.cdna4.mfma_scaled(
                    a=a0,
                    a_scale=a_scale0,
                    a_format="e2m1",
                    b=b0_c2,
                    b_scale=b_scale0_c2,
                    b_format="e2m1",
                    acc=acc2,
                )
                acc3 = gl.amd.cdna4.mfma_scaled(
                    a=a0,
                    a_scale=a_scale0,
                    a_format="e2m1",
                    b=b0_c3,
                    b_scale=b_scale0_c3,
                    b_format="e2m1",
                    acc=acc3,
                )

                # Hoist routed-weight load while tile-1 A/MFMA work is still pending.
                # The value is consumed after the second MFMA, but issuing the HBM load
                # here gives it the A1 wait and second MFMA block to arrive.
                cm_layout: gl.constexpr = gl.SliceLayout(1, mfma_layout)
                offs_cm = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=cm_layout)
                # buffer_load (SRD) -- see prologue comment. offs_cm < EM always
                # (BLOCK_M-aligned EM); padding-row weights are gated by tok_ok in
                # the store, so no runtime mask/other is needed.
                moe_weight = gl.amd.cdna4.buffer_load(
                    ptr=sorted_weights_ptr,
                    offsets=offs_cm,
                )

                # Set up the epilogue layout + per-chunk pointer arithmetic
                # BEFORE the K-iter 1 MFMAs so the ``sorted_token_ids`` reload
                # and column-offset compute land in flight while the second
                # MFMA tile runs. Per-chunk epilogue work is then emitted
                # IMMEDIATELY after each chunk's final MFMA below, so the
                # convert + LDS-shuffle + store of chunk i hides behind the
                # MFMA of chunk i+1.
                #
                # ``store_layout`` shape per chunk = [128, 64] across
                # the 4-wave CTA. We choose ``warps_per_cta=[4, 1]``
                # (waves split M, not N like the MFMA layout) so the
                # convert_layout LDS shuffle redistributes data such
                # that within one wave the N axis is contiguous and
                # each lane can own 8 N-cols = one dwordx4. Per wave:
                # 32 M-rows x 64 N-cols, with 8 M-lanes x 8 N-lanes,
                # each lane owning [4 M, 8 N] = 4 dwordx4 stores per
                # chunk. Per simultaneous-store iteration: 8 M-rows hit
                # 16 cache lines (8 rows x 2 lines per 64-col row),
                # vs the 32+ rows the in-place wide layouts hit.
                if MFMA_STORE_LAYOUT:
                    # Small-M atomic A/B path: keep the accumulator's MFMA
                    # layout through the store to avoid the convert_layout
                    # LDS shuffle/barrier tax.  Stores are less coalesced, so
                    # reduce mode keeps the coalesced C-shuffle layout because
                    # the launcher pairs MFMA-store layout with atomic mode.
                    store_layout: gl.constexpr = mfma_layout
                else:
                    store_layout: gl.constexpr = gl.BlockedLayout(
                        size_per_thread=[4, 8],
                        threads_per_warp=[8, 8],
                        warps_per_cta=[4, 1],
                        order=[1, 0],
                    )
                sm_layout: gl.constexpr = gl.SliceLayout(1, store_layout)
                sn_layout: gl.constexpr = gl.SliceLayout(0, store_layout)
                offs_sorted_slot_s = pid_m * BLOCK_M + gl.arange(
                    0, BLOCK_M, layout=sm_layout
                )
                # buffer_load (SRD) for the store-layout reload. Same invariant as
                # the prologue gather: offs < EM always, sentinel + tok_ok handle
                # padding, so no runtime mask/other.
                offs_token_s = gl.amd.cdna4.buffer_load(
                    ptr=sorted_token_ids_ptr,
                    offsets=offs_sorted_slot_s,
                )
                token_id_s = offs_token_s & 0xFFFFFF
                topk_id_s = offs_token_s >> 24
                if USE_REDUCE:
                    dst_row_s = token_id_s * top_k + topk_id_s
                else:
                    dst_row_s = token_id_s
                offs_cn_chunk = pid_n * BLOCK_N + gl.arange(
                    0, BLOCK_N_CHUNK, layout=sn_layout
                )
                cn0 = offs_cn_chunk + 0 * BLOCK_N_CHUNK
                cn1 = offs_cn_chunk + 1 * BLOCK_N_CHUNK
                cn2 = offs_cn_chunk + 2 * BLOCK_N_CHUNK
                cn3 = offs_cn_chunk + 3 * BLOCK_N_CHUNK
                p0 = (
                    c_ptr
                    + dst_row_s[:, None].to(gl.int64) * stride_cm
                    + cn0[None, :].to(gl.int64) * stride_cn
                )
                p1 = (
                    c_ptr
                    + dst_row_s[:, None].to(gl.int64) * stride_cm
                    + cn1[None, :].to(gl.int64) * stride_cn
                )
                p2 = (
                    c_ptr
                    + dst_row_s[:, None].to(gl.int64) * stride_cm
                    + cn2[None, :].to(gl.int64) * stride_cn
                )
                p3 = (
                    c_ptr
                    + dst_row_s[:, None].to(gl.int64) * stride_cm
                    + cn3[None, :].to(gl.int64) * stride_cn
                )
                # N-mask (cn_i < N) elided: the partials buffer is exactly
                # [token_num, topk, N] and BLOCK_N divides N for our DSv3 /
                # Kimi shapes, so every chunk column is in-bounds. Keeping
                # the per-element N check would block dword->dwordx4 store
                # vectorisation because the compiler can't prove uniformity.
                # Token-row validity is kept -- padded rows have
                # token_id == num_valid_tokens >= token_num.
                tok_ok = token_id_s[:, None] < token_num
                m0 = tok_ok
                m1 = tok_ok
                m2 = tok_ok
                m3 = tok_ok

                cdna4_async_copy.wait_group(0)
                a1 = cdna4_async_copy.load_shared_relaxed(smem_a.index(1), dot_a_layout)

                if DEFER_EPILOGUE:
                    # FlyDSL-like: run the K-iter-1 MFMAs as one uninterrupted
                    # cluster, then do the whole epilogue in a single phase. The
                    # unbroken MFMA cluster keeps the matrix pipe fed (~3.6%
                    # faster than the per-chunk interleave; bit-exact).
                    acc0 = gl.amd.cdna4.mfma_scaled(
                        a=a1,
                        a_scale=a_scale1,
                        a_format="e2m1",
                        b=b1_c0,
                        b_scale=b_scale1_c0,
                        b_format="e2m1",
                        acc=acc0,
                    )
                    acc1 = gl.amd.cdna4.mfma_scaled(
                        a=a1,
                        a_scale=a_scale1,
                        a_format="e2m1",
                        b=b1_c1,
                        b_scale=b_scale1_c1,
                        b_format="e2m1",
                        acc=acc1,
                    )
                    acc2 = gl.amd.cdna4.mfma_scaled(
                        a=a1,
                        a_scale=a_scale1,
                        a_format="e2m1",
                        b=b1_c2,
                        b_scale=b_scale1_c2,
                        b_format="e2m1",
                        acc=acc2,
                    )
                    acc3 = gl.amd.cdna4.mfma_scaled(
                        a=a1,
                        a_scale=a_scale1,
                        a_format="e2m1",
                        b=b1_c3,
                        b_scale=b_scale1_c3,
                        b_format="e2m1",
                        acc=acc3,
                    )
                    if K_PACKED_TOTAL > 2 * BLOCK_K_PACKED:
                        # Kimi TP4 stage2 has logical K=512.  The original
                        # production body above covers the first K=256
                        # (two K=128 scaled-MFMA subtiles); accumulate the
                        # second K=256 pair before the epilogue.  The host
                        # wrapper gates this path to direct-scale loads for
                        # now; the coalesced full-K scale loader is currently
                        # specialized to one K=256 pair.
                        cdna4_async_copy.buffer_load_to_shared(
                            smem_a.index(0),
                            a_base_ptr,
                            offs_a_i32 + 2 * A_DATA_K_STEP,
                            mask=token_mask[:, None],
                        )
                        cdna4_async_copy.commit_group()
                        cdna4_async_copy.buffer_load_to_shared(
                            smem_a.index(1),
                            a_base_ptr,
                            offs_a_i32 + 3 * A_DATA_K_STEP,
                            mask=token_mask[:, None],
                        )
                        cdna4_async_copy.commit_group()

                        a_k_iter2 = 2 * BLOCK_K_SCALE + a_k_lanes
                        a_k_part2 = (
                            (a_k_iter2 // 8) * 256
                            + (a_k_iter2 % 4) * 64
                            + ((a_k_iter2 % 8) // 4) * 2
                        )[None, :]
                        a_scale2 = gl.amd.cdna4.buffer_load(
                            ptr=a_scales_ptr, offsets=a_m_part + a_k_part2
                        )
                        b2_c0 = gl.amd.cdna4.buffer_load(
                            ptr=b_base_ptr, offsets=offs_b_c0 + B_DATA_K_STEP2
                        )
                        b2_c1 = gl.amd.cdna4.buffer_load(
                            ptr=b_base_ptr, offsets=offs_b_c1 + B_DATA_K_STEP2
                        )
                        b2_c2 = gl.amd.cdna4.buffer_load(
                            ptr=b_base_ptr, offsets=offs_b_c2 + B_DATA_K_STEP2
                        )
                        b2_c3 = gl.amd.cdna4.buffer_load(
                            ptr=b_base_ptr, offsets=offs_b_c3 + B_DATA_K_STEP2
                        )
                        bsc_k_iter2 = 2 * BLOCK_K_SCALE + b_k_lanes
                        bsc_k_off2 = (
                            (bsc_k_iter2 // 8) * 256
                            + (bsc_k_iter2 % 4) * 64
                            + ((bsc_k_iter2 % 8) // 4) * 2
                        )[None, :]
                        b_scale2_c0 = gl.amd.cdna4.buffer_load(
                            ptr=b_scale_base, offsets=b_n_scale_part_c0 + bsc_k_off2
                        )
                        b_scale2_c1 = gl.amd.cdna4.buffer_load(
                            ptr=b_scale_base, offsets=b_n_scale_part_c1 + bsc_k_off2
                        )
                        b_scale2_c2 = gl.amd.cdna4.buffer_load(
                            ptr=b_scale_base, offsets=b_n_scale_part_c2 + bsc_k_off2
                        )
                        b_scale2_c3 = gl.amd.cdna4.buffer_load(
                            ptr=b_scale_base, offsets=b_n_scale_part_c3 + bsc_k_off2
                        )

                        a_k_iter3 = 3 * BLOCK_K_SCALE + a_k_lanes
                        a_k_part3 = (
                            (a_k_iter3 // 8) * 256
                            + (a_k_iter3 % 4) * 64
                            + ((a_k_iter3 % 8) // 4) * 2
                        )[None, :]
                        a_scale3 = gl.amd.cdna4.buffer_load(
                            ptr=a_scales_ptr, offsets=a_m_part + a_k_part3
                        )
                        b3_c0 = gl.amd.cdna4.buffer_load(
                            ptr=b_base_ptr, offsets=offs_b_c0 + B_DATA_K_STEP3
                        )
                        b3_c1 = gl.amd.cdna4.buffer_load(
                            ptr=b_base_ptr, offsets=offs_b_c1 + B_DATA_K_STEP3
                        )
                        b3_c2 = gl.amd.cdna4.buffer_load(
                            ptr=b_base_ptr, offsets=offs_b_c2 + B_DATA_K_STEP3
                        )
                        b3_c3 = gl.amd.cdna4.buffer_load(
                            ptr=b_base_ptr, offsets=offs_b_c3 + B_DATA_K_STEP3
                        )
                        bsc_k_iter3 = 3 * BLOCK_K_SCALE + b_k_lanes
                        bsc_k_off3 = (
                            (bsc_k_iter3 // 8) * 256
                            + (bsc_k_iter3 % 4) * 64
                            + ((bsc_k_iter3 % 8) // 4) * 2
                        )[None, :]
                        b_scale3_c0 = gl.amd.cdna4.buffer_load(
                            ptr=b_scale_base, offsets=b_n_scale_part_c0 + bsc_k_off3
                        )
                        b_scale3_c1 = gl.amd.cdna4.buffer_load(
                            ptr=b_scale_base, offsets=b_n_scale_part_c1 + bsc_k_off3
                        )
                        b_scale3_c2 = gl.amd.cdna4.buffer_load(
                            ptr=b_scale_base, offsets=b_n_scale_part_c2 + bsc_k_off3
                        )
                        b_scale3_c3 = gl.amd.cdna4.buffer_load(
                            ptr=b_scale_base, offsets=b_n_scale_part_c3 + bsc_k_off3
                        )

                        cdna4_async_copy.wait_group(1)
                        a2 = cdna4_async_copy.load_shared_relaxed(
                            smem_a.index(0), dot_a_layout
                        )
                        acc0 = gl.amd.cdna4.mfma_scaled(
                            a=a2,
                            a_scale=a_scale2,
                            a_format="e2m1",
                            b=b2_c0,
                            b_scale=b_scale2_c0,
                            b_format="e2m1",
                            acc=acc0,
                        )
                        acc1 = gl.amd.cdna4.mfma_scaled(
                            a=a2,
                            a_scale=a_scale2,
                            a_format="e2m1",
                            b=b2_c1,
                            b_scale=b_scale2_c1,
                            b_format="e2m1",
                            acc=acc1,
                        )
                        acc2 = gl.amd.cdna4.mfma_scaled(
                            a=a2,
                            a_scale=a_scale2,
                            a_format="e2m1",
                            b=b2_c2,
                            b_scale=b_scale2_c2,
                            b_format="e2m1",
                            acc=acc2,
                        )
                        acc3 = gl.amd.cdna4.mfma_scaled(
                            a=a2,
                            a_scale=a_scale2,
                            a_format="e2m1",
                            b=b2_c3,
                            b_scale=b_scale2_c3,
                            b_format="e2m1",
                            acc=acc3,
                        )
                        cdna4_async_copy.wait_group(0)
                        a3 = cdna4_async_copy.load_shared_relaxed(
                            smem_a.index(1), dot_a_layout
                        )
                        acc0 = gl.amd.cdna4.mfma_scaled(
                            a=a3,
                            a_scale=a_scale3,
                            a_format="e2m1",
                            b=b3_c0,
                            b_scale=b_scale3_c0,
                            b_format="e2m1",
                            acc=acc0,
                        )
                        acc1 = gl.amd.cdna4.mfma_scaled(
                            a=a3,
                            a_scale=a_scale3,
                            a_format="e2m1",
                            b=b3_c1,
                            b_scale=b_scale3_c1,
                            b_format="e2m1",
                            acc=acc1,
                        )
                        acc2 = gl.amd.cdna4.mfma_scaled(
                            a=a3,
                            a_scale=a_scale3,
                            a_format="e2m1",
                            b=b3_c2,
                            b_scale=b_scale3_c2,
                            b_format="e2m1",
                            acc=acc2,
                        )
                        acc3 = gl.amd.cdna4.mfma_scaled(
                            a=a3,
                            a_scale=a_scale3,
                            a_format="e2m1",
                            b=b3_c3,
                            b_scale=b_scale3_c3,
                            b_format="e2m1",
                            acc=acc3,
                        )
                    # Match the reference combine epilogue ordering: round the GEMM
                    # accumulator to bf16 first, then multiply by the bf16
                    # routed weight. Multiplying in fp32 and rounding once at
                    # the end changes near-zero values by one or two bf16 ULPs.
                    c0 = acc0.to(c_ptr.type.element_ty)
                    c1 = acc1.to(c_ptr.type.element_ty)
                    c2 = acc2.to(c_ptr.type.element_ty)
                    c3 = acc3.to(c_ptr.type.element_ty)
                    routed_weight = moe_weight[:, None].to(c0.dtype)
                    c0_co = gl.convert_layout(c0 * routed_weight, store_layout)
                    c1_co = gl.convert_layout(c1 * routed_weight, store_layout)
                    c2_co = gl.convert_layout(c2 * routed_weight, store_layout)
                    c3_co = gl.convert_layout(c3 * routed_weight, store_layout)
                    if USE_REDUCE:
                        # ``.cs`` (cache-streaming) on the partials write: the
                        # reduce kernel reads each cell exactly once.
                        gl.store(p0, c0_co, mask=m0, cache_modifier=".cs")
                        gl.store(p1, c1_co, mask=m1, cache_modifier=".cs")
                        gl.store(p2, c2_co, mask=m2, cache_modifier=".cs")
                        gl.store(p3, c3_co, mask=m3, cache_modifier=".cs")
                    else:
                        gl.atomic_add(p0, c0_co, mask=m0, sem="relaxed")
                        gl.atomic_add(p1, c1_co, mask=m1, sem="relaxed")
                        gl.atomic_add(p2, c2_co, mask=m2, sem="relaxed")
                        gl.atomic_add(p3, c3_co, mask=m3, sem="relaxed")
                else:
                    # K-iter 1 MFMAs and the chunked epilogue interleaved per
                    # chunk: ``mfma(chunk_i)`` -> ``cvt + store(chunk_i)`` -> ...
                    acc0 = gl.amd.cdna4.mfma_scaled(
                        a=a1,
                        a_scale=a_scale1,
                        a_format="e2m1",
                        b=b1_c0,
                        b_scale=b_scale1_c0,
                        b_format="e2m1",
                        acc=acc0,
                    )
                    c0 = acc0.to(c_ptr.type.element_ty)
                    c0 = c0 * moe_weight[:, None].to(c0.dtype)
                    c0_co = gl.convert_layout(c0, store_layout)
                    if USE_REDUCE:
                        gl.store(p0, c0_co, mask=m0, cache_modifier=".cs")
                    else:
                        gl.atomic_add(p0, c0_co, mask=m0, sem="relaxed")

                    acc1 = gl.amd.cdna4.mfma_scaled(
                        a=a1,
                        a_scale=a_scale1,
                        a_format="e2m1",
                        b=b1_c1,
                        b_scale=b_scale1_c1,
                        b_format="e2m1",
                        acc=acc1,
                    )
                    c1 = acc1.to(c_ptr.type.element_ty)
                    c1 = c1 * moe_weight[:, None].to(c1.dtype)
                    c1_co = gl.convert_layout(c1, store_layout)
                    if USE_REDUCE:
                        gl.store(p1, c1_co, mask=m1, cache_modifier=".cs")
                    else:
                        gl.atomic_add(p1, c1_co, mask=m1, sem="relaxed")

                    acc2 = gl.amd.cdna4.mfma_scaled(
                        a=a1,
                        a_scale=a_scale1,
                        a_format="e2m1",
                        b=b1_c2,
                        b_scale=b_scale1_c2,
                        b_format="e2m1",
                        acc=acc2,
                    )
                    c2 = acc2.to(c_ptr.type.element_ty)
                    c2 = c2 * moe_weight[:, None].to(c2.dtype)
                    c2_co = gl.convert_layout(c2, store_layout)
                    if USE_REDUCE:
                        gl.store(p2, c2_co, mask=m2, cache_modifier=".cs")
                    else:
                        gl.atomic_add(p2, c2_co, mask=m2, sem="relaxed")

                    acc3 = gl.amd.cdna4.mfma_scaled(
                        a=a1,
                        a_scale=a_scale1,
                        a_format="e2m1",
                        b=b1_c3,
                        b_scale=b_scale1_c3,
                        b_format="e2m1",
                        acc=acc3,
                    )
                    c3 = acc3.to(c_ptr.type.element_ty)
                    c3 = c3 * moe_weight[:, None].to(c3.dtype)
                    c3_co = gl.convert_layout(c3, store_layout)
                    if USE_REDUCE:
                        gl.store(p3, c3_co, mask=m3, cache_modifier=".cs")
                    else:
                        gl.atomic_add(p3, c3_co, mask=m3, sem="relaxed")


# ---------------------------------------------------------------------------
# Reduce kernel (sums [token, topk, N] -> [token, N], fp32 accumulate)
# ---------------------------------------------------------------------------


@gluon.jit
def gluon_mxfp4_moe_stage2_reduce_kernel(
    partials_ptr,  # bf16, shape [token_num, topk, N], contiguous
    out_ptr,  # bf16, shape [token_num, N]
    token_num,
    N,
    top_k,
    stride_pt,  # partials: stride for token dim = topk * N
    stride_ps,  # partials: stride for slot  dim = N
    stride_pn,  # partials: stride for col   dim = 1
    stride_ot,
    stride_on,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    TOP_K: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    """Sum per-(token, slot) partials over the topk dim.

    Grid: ``(cdiv(token_num, BLOCK_M) * cdiv(N, BLOCK_N),)``. Each CTA
    owns a ``[BLOCK_M, BLOCK_N]`` tile of the output and reads
    ``TOP_K`` slices from the partial buffer, accumulating in fp32 and
    casting back to bf16 at the end. ``TOP_K`` is a constexpr so the
    accumulation loop unrolls (TOP_K is small: 4-10 across the models
    we serve).
    """
    pid = gl.program_id(axis=0)
    num_pid_n = gl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # Plain blocked layout for the bf16 tile. 1 wave / CTA, NUM_WARPS=1.
    blk: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[16, 4],
        warps_per_cta=[1, 1],
        order=[1, 0],
    )
    rm_layout: gl.constexpr = gl.SliceLayout(1, blk)
    cn_layout: gl.constexpr = gl.SliceLayout(0, blk)
    offs_m = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=rm_layout)
    offs_n = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=cn_layout)
    m_mask = (offs_m[:, None] < token_num) & (offs_n[None, :] < N)

    acc = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=blk)
    base = (
        partials_ptr
        + offs_m[:, None].to(gl.int64) * stride_pt
        + offs_n[None, :].to(gl.int64) * stride_pn
    )
    for s in gl.static_range(0, TOP_K):
        p = base + s * stride_ps
        v = gl.load(p, mask=m_mask, other=0.0)
        acc += v.to(gl.float32)

    out_ptrs = (
        out_ptr
        + offs_m[:, None].to(gl.int64) * stride_ot
        + offs_n[None, :].to(gl.int64) * stride_on
    )
    gl.store(out_ptrs, acc.to(out_ptr.type.element_ty), mask=m_mask)


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------


def _record_atomic_lowering() -> None:
    global _USES_FP32_ATOMIC
    try:
        kf = gluon_mxfp4_moe_stage2_1x2_kernel
        for cache_tuple in getattr(kf, "device_caches", {}).values():
            kernel_cache = cache_tuple[0]
            for compiled in kernel_cache.values():
                asm = getattr(compiled, "asm", None)
                if asm is None or "amdgcn" not in asm:
                    continue
                amdgcn = asm["amdgcn"]
                if "pk_add_bf16" in amdgcn:
                    _USES_FP32_ATOMIC = False
                    return
                # ``pk_add_bf16`` was ruled out above, so any remaining
                # ``atomic_add_f32`` is a true fp32 atomic lowering.
                if "atomic_add_f32" in amdgcn:
                    _USES_FP32_ATOMIC = True
                    return
    except Exception:
        return


def invoke_gluon_mxfp4_moe_stage2_1x2(
    inter_states,
    w1,
    w2,
    sorted_token_ids,
    sorted_expert_ids,
    num_valid_ids,
    out,
    topk,
    kernelName="",
    w2_scale=None,
    a2_scale=None,
    block_m=None,
    sort_block_m=None,
    sorted_weights=None,
    quant_type=None,
    activation=None,
    splitk=1,
    use_non_temporal_load=False,
    dst_type=None,
    b_preshuffled: bool = False,
    b_gdot128: bool = False,
    force_reduce: bool | None = None,
):
    """Host-side launcher for Gluon MXFP4 MoE stage 2.

    Computes, for each expert ``e``::

        out_e += topk_w_e * (inter_e @ w2_e.T)

    by launching the GEMM kernel followed, in reduce mode, by a small
    reduce that sums each token's ``topk`` partial contributions into
    ``out``. The wrapper can instead use the direct-atomic path for
    small M, but the standalone Kimi comparison currently forces reduce
    mode at M512 because this Gluon atomic epilogue is still slower.

    Steps below correspond to the ``# Step N:`` comments in the body:
      1. Validate inputs; reject unsupported modes.
      2. Permute ``w2`` to the (16, 16) MFMA-tile layout (see
         :func:`_b_preshuffle_3d`), or skip if the caller already did.
      3. Flatten ``inter_states`` ``(token_num, topk, K_packed)`` to
         ``(token_num * topk, K_packed)`` -- one row per (token, slot).
      4. Allocate a ``(token_num, topk, N)`` bf16 ``partials`` scratch
         only when reduce mode is selected.
      5. Launch the GEMM. The per-CTA epilogue either writes
         ``partials[token, slot]`` with plain stores or atomically adds
         into ``out[token]``.
      6. In reduce mode, launch the reduce kernel: sum over the
         ``topk`` dim with fp32 accumulation, cast back to bf16 into
         ``out``.

    Layout contract:
        inter_states : (token_num, topk, K_packed) uint8
                       OR (M_padded, K_packed) uint8 (already flattened)
        w2           : (E, D, I_r_packed) uint8 in MFMA-tile layout
        a2_scale     : (M_padded_aligned, K // 32) uint8 e8m0
        w2_scale     : (E, D, K // 32) uint8 e8m0
        out          : (token_num, D) bf16 -- final per-token output
        sorted_*     : generated with ``sort_block_m`` when it differs
                       from compute ``block_m``; use 32/64/128 to mirror
                       FlyDSL sort_block_m

    Unsupported: ``quant_type`` and ``activation`` are signature
    stubs (kernel does the down-projection with no activation);
    ``splitk not in {0, 1, None}``; ``sorted_weights=None``.
    """
    global _USES_FP32_ATOMIC

    # Step 1: validate inputs and reject unsupported modes.
    del quant_type, activation  # signature stubs (no activation in stage 2)
    if splitk not in (0, 1, None):
        raise NotImplementedError(f"splitk in {{0, 1}} expected, got {splitk!r}")
    if sorted_weights is None or sorted_weights.numel() == 0:
        raise NotImplementedError("sorted_weights is REQUIRED for stage 2")
    if dst_type is not None and dst_type != torch.bfloat16:
        raise NotImplementedError(f"only dst_type=bfloat16 supported, got {dst_type!r}")
    del kernelName, use_non_temporal_load, w1

    assert inter_states.dtype == torch.uint8
    assert w2.dtype == torch.uint8
    # Step 2: shuffle w2 to MFMA-tile layout if the caller didn't.
    if not b_preshuffled:
        w2 = _b_preshuffle_3d(w2)
    if b_gdot128 and not b_preshuffled:
        raise ValueError("b_gdot128 requires a preshuffled gdot128 weight alias")
    assert w2_scale is not None and w2_scale.dtype == torch.uint8
    assert a2_scale is not None and a2_scale.dtype == torch.uint8
    assert out.dtype == torch.bfloat16
    assert sorted_token_ids.dtype == torch.int32
    assert sorted_expert_ids.dtype == torch.int32
    assert sorted_weights.dtype == torch.float32

    # Step 3: flatten the (token, slot) axes so each row is one routed
    # token-slot. Zero-copy via .view when inter_states is contiguous.
    if inter_states.dim() == 3:
        token_num_dim, top_k_dim, K_packed_3d = inter_states.shape
        assert top_k_dim == topk
        assert inter_states.is_contiguous()
        inter_2d = inter_states.view(token_num_dim * topk, K_packed_3d)
    elif inter_states.dim() == 2:
        inter_2d = inter_states
    else:
        raise NotImplementedError(
            f"inter_states must be 2-D or 3-D, got {tuple(inter_states.shape)}"
        )
    if out.dim() != 2:
        raise NotImplementedError(f"stage 2 out must be 2-D, got {tuple(out.shape)}")
    token_num = out.shape[0]
    M_padded, I_r_packed = inter_2d.shape
    K = I_r_packed * 2

    assert w2.dim() == 3
    E_w, D, I_r_packed_w = w2.shape
    N = D
    assert I_r_packed_w == I_r_packed
    EM = sorted_token_ids.shape[0]
    K_scale = K // 32
    assert a2_scale.dim() == 2 and a2_scale.shape[1] == K_scale
    assert w2_scale.shape == (E_w, N, K_scale)
    assert sorted_weights.shape[0] == EM

    # The kernel reads the valid extent from ``num_valid_ids_ptr`` on-device
    # (``num_tokens_post_padded = gl.load(...)``), so the ``num_valid_tokens``
    # scalar arg is vestigial. Avoid a device-to-host ``.item()`` sync here and
    # pass a placeholder; the device pointer is the single source of truth.
    if torch.is_tensor(num_valid_ids):
        num_valid_tokens = 0
        num_valid_ids_ptr = num_valid_ids
    else:
        num_valid_tokens = int(num_valid_ids)
        num_valid_ids_ptr = torch.tensor(
            [num_valid_tokens], dtype=torch.int32, device=inter_states.device
        )

    BLOCK_M = 128 if block_m is None else int(block_m)
    if BLOCK_M not in (32, 64, 128):
        raise NotImplementedError(
            f"stage2 1x2 block_m must be one of 32, 64, 128; got {BLOCK_M}"
        )
    SORT_BLOCK_M = BLOCK_M if sort_block_m is None else int(sort_block_m)
    if SORT_BLOCK_M not in (32, 64, 128) or SORT_BLOCK_M % BLOCK_M != 0:
        raise NotImplementedError(
            "stage2 1x2 sort_block_m must be one of 32, 64, 128 "
            f"and divisible by block_m; got {SORT_BLOCK_M} for block_m={BLOCK_M}"
        )
    BLOCK_N = 256
    BLOCK_K = 128
    GROUP_SIZE_M = 1
    NUM_WARPS = 4
    assert K % BLOCK_K == 0, f"K ({K}) must be divisible by BLOCK_K ({BLOCK_K})"

    num_pid_m = triton.cdiv(EM, BLOCK_M)
    num_pid_n = triton.cdiv(N, BLOCK_N)
    # Grid sizing. Persistent grid is off by default -- the persistent
    # loop structure alone (without true cross-tile prefetch) doesn't
    # win: at large M it shaves only ~3 % off stage 2 because L2 reuse
    # benefits only kick in within one CTA's iterations and the first
    # iter of each CTA is still cold; at small M it wastes most CTAs
    # on no-op iterations. Re-enabled when a follow-up adds actual
    # next-tile load overlap inside the per-tile body.
    CU_NUM = 256
    PERSISTENT_BLOCKS = CU_NUM
    total_tiles_host = num_pid_m * num_pid_n
    PERSISTENT = False
    grid = (num_pid_n, PERSISTENT_BLOCKS) if PERSISTENT else (total_tiles_host,)

    # The coalesced B-scale loader and the direct-scale layout still encode the
    # legacy stage2-specific half-bit scale layout, so they stay off for the
    # generic CDNA4/e8m0 scale contract used by the package/decode kernels.
    COALESCE_SCALES = False
    DIRECT_SCALE_LAYOUT = False
    # Deferred epilogue (run the K-iter MFMAs as one cluster then a single
    # epilogue phase) is ~3.6% faster and bit-exact, so it is always on.
    DEFER_EPILOGUE = True

    # Atomic vs reduce dispatch.
    #
    # At small M, the cross-slot reduce overhead (extra scratch write +
    # separate reduce kernel launch) dominates: at M=16 the reduce alone
    # is ~12 % of the stage 2 budget while reducing essentially nothing
    # (each output row has at most a handful of contributors and the
    # atomic_add path costs ~3-4 atomic_pk_add_bf16 per row instead).
    #
    # At large M the atomic_add path loses to scratch+reduce because
    # M * topk concurrent atomic_pk_add_bf16 ops contend on the same
    # output rows (the cost is in serialization at the HBM line, not
    # the instruction).
    #
    # With FlyDSL-like smaller sort blocks, the reduce path is faster
    # starting at M512: direct atomic avoids the reduce launch, but this
    # Gluon epilogue still loses to topk-way output-row contention.
    # Keep tiny/decode shapes on atomic and switch to scratch+reduce for
    # M>=512. The ``force_reduce`` caller argument overrides the tuned default.
    if force_reduce is not None:
        _use_reduce = bool(force_reduce)
    else:
        _use_reduce = token_num >= 512
    _mfma_store_layout = not _use_reduce

    stride_am = inter_2d.stride(0)
    stride_ak = inter_2d.stride(1)
    stride_be = w2.stride(0)
    stride_bn = w2.stride(1)
    stride_bk = w2.stride(2)
    stride_ase_m = a2_scale.stride(0)
    stride_ase_k = a2_scale.stride(1)
    stride_bse_e = w2_scale.stride(0)
    stride_bse_n = w2_scale.stride(1)
    stride_bse_k = w2_scale.stride(2)
    stride_se_n_pad = a2_scale.shape[1]
    K_packed_total = K // 2

    if _use_reduce:
        # Reduce path (large M): GEMM writes per-(token, slot) cells of
        # a scratch tensor with plain stores (race-free, each cell has
        # exactly one writer CTA), then a small reduce kernel sums the
        # topk dim into `out` with fp32 accumulation.
        partials = torch.empty(
            (token_num, topk, N), dtype=torch.bfloat16, device=inter_states.device
        )
        c_ptr = partials
        # M-row index into `partials` is the flattened (token, slot) row;
        # row stride = topk * N // topk = N elements (one slot row).
        c_stride_m = partials.stride(1)
        c_stride_n = partials.stride(2)  # = 1
    else:
        # Atomic path (small M): GEMM atomic_add_pk_bf16's directly into
        # `out`. Caller's `out` must be zero-initialised so the first
        # accumulator lands on a clean slate. We zero it here -- one
        # bf16 fill is cheap relative to skipping the scratch + reduce.
        out.zero_()
        c_ptr = out
        c_stride_m = out.stride(0)
        c_stride_n = out.stride(1)

    # Step 5: GEMM. Writes either to `partials` (reduce mode) or directly
    # to `out` (atomic mode), selected by ``USE_REDUCE``.
    gluon_mxfp4_moe_stage2_1x2_kernel[grid](
        inter_2d,
        w2,
        c_ptr,
        a2_scale,
        w2_scale,
        sorted_token_ids,
        sorted_expert_ids,
        num_valid_ids_ptr,
        sorted_weights,
        N,
        K,
        EM,
        num_valid_tokens,
        token_num,
        topk,
        stride_am,
        stride_ak,
        stride_be,
        stride_bn,
        stride_bk,
        c_stride_m,
        c_stride_n,
        stride_ase_m,
        stride_ase_k,
        stride_bse_e,
        stride_bse_n,
        stride_bse_k,
        stride_se_n_pad,
        K_PACKED_TOTAL=K_packed_total,
        N_PHYS=N,
        BLOCK_M=BLOCK_M,
        SORT_BLOCK_M=SORT_BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        NUM_WARPS=NUM_WARPS,
        B_GDOT128=bool(b_gdot128),
        USE_REDUCE=_use_reduce,
        MFMA_STORE_LAYOUT=_mfma_store_layout,
        PERSISTENT=PERSISTENT,
        COALESCE_SCALES=COALESCE_SCALES,
        DIRECT_SCALE_LAYOUT=DIRECT_SCALE_LAYOUT,
        DEFER_EPILOGUE=DEFER_EPILOGUE,
        # ``CU_NUM`` is the divisor for ``tiles_per_block`` in the
        # persistent path. When PERSISTENT=False it is unused inside
        # the kernel (branch is constexpr-pruned), so we keep it at the
        # historical 256 in that case to avoid invalidating the cached
        # binary for the small-M atomic dispatch.
        CU_NUM=PERSISTENT_BLOCKS if PERSISTENT else 256,
        num_warps=NUM_WARPS,
    )

    if _use_reduce:
        # Step 6: reduce. Sum partials[token, :, n] over the topk dim
        # (fp32 accumulate, bf16 output) into `out`. Tile (32, 256),
        # 1 wave/CTA.
        BLOCK_M_R = 16 if token_num >= 4096 else 32
        BLOCK_N_R = 256
        NUM_WARPS_R = 1
        rgrid = (triton.cdiv(token_num, BLOCK_M_R) * triton.cdiv(N, BLOCK_N_R),)
        gluon_mxfp4_moe_stage2_reduce_kernel[rgrid](
            partials,
            out,
            token_num,
            N,
            topk,
            partials.stride(0),
            partials.stride(1),
            partials.stride(2),
            out.stride(0),
            out.stride(1),
            BLOCK_M=BLOCK_M_R,
            BLOCK_N=BLOCK_N_R,
            TOP_K=topk,
            NUM_WARPS=NUM_WARPS_R,
            num_warps=NUM_WARPS_R,
        )
    if _USES_FP32_ATOMIC is None:
        _record_atomic_lowering()
    return out
