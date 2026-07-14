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


"""Gluon MXFP4 MoE stage 1: gate + up GEMM with fused SwiGLU.

For each MoE expert ``e``::

    inter_e = silu(hidden_states_e @ w1_e[:I_r, :].T)        # gate
              * (hidden_states_e @ w1_e[I_r:, :].T)          # up

where ``hidden_states_e`` is the subset of input rows routed to
expert ``e``. The kernel computes that for all experts in one launch
by walking the ``sorted_token_ids`` / ``sorted_expert_ids`` arrays
produced by the upstream routing kernel.

Kernel shape:
  BLOCK_M = BLOCK_N = 128, BLOCK_K = 256, num_warps = 4,
  warps_per_cta = [1, 4]. ``BLOCK_M`` rows split into 4 quarters of
  ``GROUP_MFMA_M = 32`` so each K-tile fires 4 (quarter) x 2 (gate/up)
  = 8 ``mfma_scaled`` instructions on the production K = 7168 shape.
  K-loop pipelining: 2-slot LDS ping-pong for A data and A scale, and
  2-slot VGPR ping-pong for B data and B scale; B[k+1] is issued at
  the top of tile k so its VMEM latency hides behind tile k's MFMAs.

B-data is consumed in a (16, 16)-tile layout so each MFMA fetches its
tile with a single 128-bit ``buffer_load``. The byte at logical
``(e, n, k_pk)`` lives at::

    e * (N * K_pk)
      + (n // 16) * (16 * K_PACKED_TOTAL)
      + (k_pk // 32) * 512
      + ((k_pk // 16) % 2) * 256
      + (n % 16) * 16
      + (k_pk % 16)

The per-expert term is folded into ``b_base_ptr``; ``K_PACKED_TOTAL``
is the full B tensor's packed-K dim (= K // 2). The host wrapper
applies this permutation via ``_b_preshuffle_3d`` when callers pass
plain B, or skips it when ``b_preshuffled=True``.

Layout contract:
  hidden_states  (M_padded, K_packed)        uint8, fp4x2 packed
  w1             (E, 2 * I_r, K_packed)      uint8, (16, 16) shuffled
  a1_scale       (M_padded_aligned, K // 32) uint8, e8m0
  w1_scale       (E, 2 * I_r, K // 32)       uint8, e8m0 (shuffled)
  out            (EM, I_r) or (token_num, topk, I_r)  bf16
  sorted_token_ids    (EM,)             int32; low 24 bits = token_id
  sorted_expert_ids   (EM // BLOCK_M,)  int32; -1 marks a padding block
  sorted_weights      (EM,)             fp32  (REQUIRED to be empty here;
                                               stage 1 doesn't fold weights)

Stage 1 output is post-SwiGLU at ``I_r`` columns (half of the un-fused
gate||up width). ``N = 2 * I_r`` is the B-operand column count and is
passed in for indexing only; the C-side is ``I_r``.
"""

from __future__ import annotations

from typing import Optional  # noqa: F401  (kept for downstream type hints)

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd.cdna4 import async_copy as cdna4_async_copy


def _b_preshuffle_3d(b: torch.Tensor) -> torch.Tensor:
    """Permute a 3-D MoE weight into the (16, 16) MFMA-tile layout.

    The kernel's MFMA instruction wants a tile's 16 rows and 16
    K-columns to be contiguous in HBM so each lane can issue a single
    128-bit ``buffer_load`` for its share. Naively-laid-out ``w1``
    forces strided gathers; this permutation does it once on the host.

    ``(E, N, K_pk)`` is viewed as ``(E, N // 16, 16, K_pk // 32, 2, 16)``
    then permuted ``(0, 1, 3, 4, 2, 5)`` and made contiguous. The
    operation is a pure data rearrangement -- numerically a no-op.

    Callers that already hold a permuted ``w1`` pass
    ``b_preshuffled=True`` to skip this pass.
    """
    assert b.dtype == torch.uint8, "B must be packed fp4 in uint8"
    assert b.ndim == 3, f"B must be 3-D (E, N, K/2), got {tuple(b.shape)}"
    E, N, K_pk = b.shape
    assert N % 16 == 0, f"N ({N}) must be divisible by 16"
    assert K_pk % 32 == 0, f"K/2 ({K_pk}) must be divisible by 32"
    b_6d = b.view(E, N // 16, 16, K_pk // 32, 2, 16)
    return b_6d.permute(0, 1, 3, 4, 2, 5).contiguous().view(E, N, K_pk)


# ---------------------------------------------------------------------------
# Module-private @gluon.jit helpers used by the K-loop pipeline.
# ``_prefetch_a_data_lds`` accepts a ``mask`` kwarg so the per-token
# A gather can plumb ``token_mask[:, None]`` through the prologue,
# steady-state, peel, and drain prefetch sites.
# ---------------------------------------------------------------------------


@gluon.jit
def _load_a_scale_vgpr(
    a_scales_ptr,
    a_scale_base_offsets,
    GROUP_IDX: gl.constexpr,
    tile_k,
    group_stride,
    A_SCALE_K_STEP: gl.constexpr,
):
    # Direct-to-VGPR A-scale load in the same CDNA4-swizzled scale contract
    # used by decode. ``a_scale_base_offsets`` is group-0, current pid_m,
    # [GROUP_MFMA_M, BLOCK_K_SCALE] in the native MFMA scale layout. Each
    # successive 32-row group advances the row block by stride_npad * 32.
    return gl.amd.cdna4.buffer_load(
        ptr=a_scales_ptr,
        offsets=(
            a_scale_base_offsets + GROUP_IDX * group_stride + tile_k * A_SCALE_K_STEP
        ),
    )


@gluon.jit
def _prefetch_a_data_lds(
    smem_a_tile,
    a_base_ptr,
    offs_a,
    a_data_k_off,
    mask=None,
):
    # A_scale path is sorted-padded so the wave-uniform K-step
    # is enough; the A data path is per-token-gathered, so callers plumb
    # ``mask=token_mask[:, None]`` here to zero-fill padded sorted slots.
    cdna4_async_copy.buffer_load_to_shared(
        smem_a_tile,
        a_base_ptr,
        offs_a + a_data_k_off,
        mask=mask,
    )


@gluon.jit
def _load_b_data_vgpr(
    b_base_ptr,
    offs_b,
    b_data_k_off,
    mask=None,
    other=None,
):
    return gl.amd.cdna4.buffer_load(
        ptr=b_base_ptr, offsets=offs_b + b_data_k_off, mask=mask, other=other
    )


@gluon.jit
def _load_b_scale_vgpr(
    b_scales_ptr,
    b_scale_static_offs,
    k,
    b_scale_k_step: gl.constexpr,
    mask=None,
    other=None,
):
    # The shuffle layout factors offset = K_PART(k_iter) + N_PART(b_rows)
    # with k_iter = k*8 + lane, lane in [0, 8). For that range the only
    # k-variant term is (k_iter//8)*1024 = k*1024; both (k_iter%4) and
    # ((k_iter%8)//4) reduce to lane-only expressions. The caller has
    # therefore precomputed the lane- and N-only piece into
    # ``b_scale_static_offs`` (a 128x8 tensor in the b_scale layout) so
    # the per-iteration arithmetic collapses to a single scalar mul +
    # tensor add. This removes ~10 tensor ops (div/mod/mul chains)
    # from each K-loop iteration and gives the AMD instruction
    # scheduler much more freedom to interleave loads with MFMAs.
    return gl.amd.cdna4.buffer_load(
        ptr=b_scales_ptr,
        offsets=b_scale_static_offs + k * b_scale_k_step,
        mask=mask,
        other=other,
    )


@gluon.jit
def _read_a_lds_group(
    smem_a_tile,
    GROUP_IDX: gl.constexpr,
    dot_a_layout: gl.constexpr,
    GROUP_M: gl.constexpr,
):
    cur_a = cdna4_async_copy.load_shared_relaxed(
        smem_a_tile.slice(GROUP_IDX * GROUP_M, GROUP_M, dim=0),
        dot_a_layout,
    )
    return cur_a


@gluon.jit
def _compute_mxfp4_group(cur_a, a_scale, cur_b, b_scale, acc):
    return gl.amd.cdna4.mfma_scaled(
        a=cur_a,
        a_scale=a_scale,
        a_format="e2m1",
        b=cur_b,
        b_scale=b_scale,
        b_format="e2m1",
        acc=acc,
    )


@gluon.jit
def _store_swiglu_tile_group(
    acc_swiglu,
    c_ptr,
    sorted_token_ids_ptr,
    pid_m,
    pid_n,
    EM,
    num_tokens,
    top_k,
    stride_cm,
    stride_cn,
    mfma_layout: gl.constexpr,
    GROUP_IDX: gl.constexpr,
    GROUP_M: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    I_r: gl.constexpr,
):
    c_val = acc_swiglu.to(c_ptr.type.element_ty)
    cm_layout: gl.constexpr = gl.SliceLayout(1, mfma_layout)
    cn_layout: gl.constexpr = gl.SliceLayout(0, mfma_layout)
    offs_cm = (
        pid_m * BLOCK_M + GROUP_IDX * GROUP_M + gl.arange(0, GROUP_M, layout=cm_layout)
    )
    offs_cn = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=cn_layout)
    offs_token_cm = gl.load(
        sorted_token_ids_ptr + offs_cm,
        mask=offs_cm < EM,
        other=num_tokens,
    )
    token_id_cm = offs_token_cm & 0xFFFFFF
    topk_id_cm = offs_token_cm >> 24
    dst_row_cm = token_id_cm * top_k + topk_id_cm
    c_ptrs = (
        c_ptr
        + dst_row_cm[:, None].to(gl.int64) * stride_cm
        + offs_cn[None, :].to(gl.int64) * stride_cn
    )
    token_mask_cm = token_id_cm < num_tokens
    c_mask = token_mask_cm[:, None] & (offs_cn[None, :] < I_r)
    gl.store(c_ptrs, c_val, mask=c_mask)


@gluon.jit
def gluon_mxfp4_moe_stage1_kernel(
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
    num_tokens,
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
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    I_r: gl.constexpr,
    B_GDOT128: gl.constexpr,
    SWIGLU_ALPHA: gl.constexpr,
    SWIGLU_LIMIT: gl.constexpr,
    SWIGLU_BETA: gl.constexpr,
):
    """Stage 1 kernel: per-token A gather + 4-deep A LDS ring + 4-deep
    A_scale LDS ring + per-quarter MFMA K-loop with fused SwiGLU.

    Tile config (fixed): ``128 x 128 x 256`` (M x N x K), ``num_warps=4``,
    ``warps_per_cta=[1, 4]``. MFMA target:
    ``v_mfma_scale_f32_16x16x128_f8f6f4`` (gfx950, AMDMFMALayout
    version=4). The launch grid covers ``num_pid_m * (I_r / BLOCK_N)``
    programs (i.e. ``num_pid_n = I_r / BLOCK_N``, NOT ``N / BLOCK_N``);
    each CTA owns one ``BLOCK_N`` slab of the ``[EM, I_r]`` SwiGLU
    output and computes the corresponding gate and up contributions
    itself.

    Each K tile fires EIGHT ``mfma_scaled`` calls (4 quarters x 2 accs);
    the four quarter-pairs share ``cur_b_gate`` / ``cur_b_up`` /
    ``b_scale_gate`` / ``b_scale_up`` loaded inline at the tile head
    (4 buffer_loads per tile). The 4x unroll matches the 4-deep A LDS
    ring depth, so each Python iter of the steady-state body cycles the
    ring back to its starting state. The K-loop opens with a 3-tile
    prologue (3 ``buffer_load_to_shared`` for A data + 3 for A_scale,
    each followed by ``commit_group``) and closes with a 3-tile drain
    epilogue (``wait_group(2)``, ``wait_group(1)``, ``wait_group(0)``).
    Mirrors dense reference lines 376 to 1305.

    Epilogue: ``silu(gate_acc_groupX) * up_acc_groupX`` for X in 0..3,
    cast to bf16, stored at sorted-row positions over ``BLOCK_N`` cols
    starting at ``pid_n * BLOCK_N`` of the ``[EM, I_r]`` output buffer.
    """

    gl.static_assert(BLOCK_M == 128, "stage1 kernel requires BLOCK_M=128")
    gl.static_assert(BLOCK_N == 128, "stage1 kernel requires BLOCK_N=128")
    gl.static_assert(BLOCK_K == 256, "stage1 kernel requires BLOCK_K=256")
    gl.static_assert(NUM_WARPS == 4, "stage1 kernel requires NUM_WARPS=4")

    SCALE_GROUP: gl.constexpr = 32
    DIV: gl.constexpr = 2
    BLOCK_K_PACKED: gl.constexpr = BLOCK_K // DIV
    BLOCK_K_SCALE: gl.constexpr = BLOCK_K // SCALE_GROUP
    NUM_BUFFERS: gl.constexpr = 2
    GROUP_MFMA_M: gl.constexpr = 32

    pid = gl.program_id(axis=0)
    num_pid_m = gl.cdiv(EM, BLOCK_M)
    # Grid covers the SwiGLU output columns ``I_r``, not the un-fused
    # gate||up GEMM columns ``N = 2 * I_r``. Each CTA owns one
    # ``BLOCK_N`` slab of the ``I_r``-wide output and internally issues
    # both the gate MFMAs (B columns ``pid_n * BLOCK_N``) and the up
    # MFMAs (B columns ``(pid_n + num_pid_n) * BLOCK_N``) against the
    # same A operand.
    num_pid_n = gl.cdiv(I_r, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = gl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_M >= num_tokens_post_padded:
        return

    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 128],
        transposed=True,
        warps_per_cta=[1, NUM_WARPS],
    )
    dot_a_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=16
    )
    dot_b_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=16
    )
    a_scale_group_layout: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(
        dot_a_layout, [GROUP_MFMA_M, BLOCK_K_SCALE]
    )
    b_scale_layout: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(
        dot_b_layout, [BLOCK_N, BLOCK_K_SCALE]
    )

    # A HBM->LDS gather and LDS storage layouts. The ``shared_a`` padding
    # rule ``[[4096, 128]]`` (insert padding every 32 rows, at M-quarter
    # boundaries) lets each per-quarter ``ds_read`` share a base VGPR with
    # immediate offsets. A-scale bypasses LDS and is loaded direct-to-VGPR
    # in the native MFMA scale layout below.
    gload_a_layout: gl.constexpr = gl.BlockedLayout(
        [1, 16],
        [8, 8],
        [4, 1],
        [1, 0],
    )
    shared_a: gl.constexpr = gl.PaddedSharedLayout(
        [[4096, 128]],
        [
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 8],
            [0, 16],
            [0, 32],
            [0, 64],
            [1, 0],
            [2, 0],
            [4, 0],
            [8, 0],
            [16, 0],
            [32, 0],
            [64, 0],
        ],
        [],
        [BLOCK_M, BLOCK_K_PACKED],
    )

    # ---- per-token A row gather -----------------------------------------
    # ``sorted_token_ids`` packs ``(topk_id << 24) | token_id`` per
    # entry, so ``offs_token & 0xFFFFFF`` is already the source row
    # index into ``hidden_states``. Do NOT divide by ``top_k``.
    #
    # Validity bound is ``token_id < num_tokens`` (NOT
    # ``< num_valid_ids[0]``): ``num_valid_ids`` counts routed slots
    # over all experts and over-shoots the per-token row range, which
    # would let the gather walk off the end of ``hidden_states`` at
    # production scale.
    m_layout: gl.constexpr = gl.SliceLayout(1, gload_a_layout)
    k_layout: gl.constexpr = gl.SliceLayout(0, gload_a_layout)
    offs_token_id = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=m_layout)
    offs_token = gl.load(
        sorted_token_ids_ptr + offs_token_id,
        mask=offs_token_id < EM,
        other=num_tokens,
    )
    token_mask = (offs_token & 0xFFFFFF) < num_tokens
    a_row = offs_token & 0xFFFFFF

    # ---- expert id (per-pid_m scalar; sentinel-checked) ----------------
    # Stage 1 writes zeros and returns when this block has no expert
    # assigned (``off_experts == -1``).
    off_experts = gl.load(sorted_expert_ids_ptr + pid_m)
    if off_experts == -1:
        cm_layout_zero: gl.constexpr = gl.SliceLayout(1, mfma_layout)
        cn_layout_zero: gl.constexpr = gl.SliceLayout(0, mfma_layout)
        offs_cm_zero = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=cm_layout_zero)
        offs_cn_zero = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=cn_layout_zero)
        zero_token = gl.load(
            sorted_token_ids_ptr + offs_cm_zero,
            mask=offs_cm_zero < EM,
            other=num_tokens,
        )
        zero_token_id = zero_token & 0xFFFFFF
        zero_topk_id = zero_token >> 24
        zero_dst_row = zero_token_id * top_k + zero_topk_id
        zero_c_ptrs = (
            c_ptr
            + zero_dst_row[:, None].to(gl.int64) * stride_cm
            + offs_cn_zero[None, :].to(gl.int64) * stride_cn
        )
        zero_mask = (zero_token_id < num_tokens)[:, None] & (
            offs_cn_zero[None, :] < I_r
        )
        zero_val = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.bfloat16, layout=mfma_layout)
        gl.store(zero_c_ptrs, zero_val, mask=zero_mask)
        return

    # ---- A data-load offsets (consumed by per-tile prefetch) ----------
    offs_ak = gl.arange(0, BLOCK_K_PACKED, layout=k_layout)
    offs_a = a_row[:, None].to(gl.int64) * stride_am + offs_ak[None, :] * stride_ak

    # ---- A_scale gather offsets ----------------------------------------
    # The scale tensor is laid out by the upstream sort kernel as::
    #
    #     addr = (x/32) * scaleN_pad * 32
    #          + (y/8)  * 256
    #          + (y%4)  * 64
    #          + (x%16) * 4
    #          + ((y%8)/4) * 2          (y-half bit  -> +2)
    #          + ((x%32)/16) * 1        (x-half bit  -> +1)
    #
    # ``x`` is the sorted-padded row index and ``y`` is the K-group
    # lane. Note the half-bit ordering: m at +1, y at +2 -- the
    # opposite convention from some dense-GEMM scale layouts.
    #
    # Row index: scales are written at the SORTED-PADDED slot, NOT at
    # the source ``token_id``, so we index directly by
    # ``pid_m * BLOCK_M + arange`` (no ``gl.load(sorted_token_ids)`` /
    # ``& 0xFFFFFF`` here).
    # Direct-to-VGPR A-scale: build group-0 offsets in the same native MFMA
    # scale layout decode uses for direct scale loads. Other 32-row groups are
    # reached by adding ``stride_se_n_pad * 32`` per group.
    m_scale_layout: gl.constexpr = gl.SliceLayout(1, a_scale_group_layout)
    k_scale_layout: gl.constexpr = gl.SliceLayout(0, a_scale_group_layout)
    a_scale_rows = (
        pid_m * BLOCK_M + gl.arange(0, GROUP_MFMA_M, layout=m_scale_layout)
    ).to(gl.uint32)
    a_m_part = (
        (a_scale_rows // 32) * (stride_se_n_pad * 32)
        + (a_scale_rows % 16) * 4
        + ((a_scale_rows % 32) // 16)
    )[:, None]
    a_k_lanes = gl.arange(0, BLOCK_K_SCALE, layout=k_scale_layout).to(gl.uint32)
    a_k_lane = ((a_k_lanes % 4) * 64 + ((a_k_lanes % 8) // 4) * 2)[None, :]
    a_scale_base_offsets = a_m_part + a_k_lane
    a_scale_group_stride = stride_se_n_pad * 32

    # ---- B data offsets (gate + up halves; preshuffle-B closed form) ---
    # B is in the (16, 16) MFMA-tile layout described in the module
    # docstring, so the byte for logical ``(n, k_pk)`` is the closed
    # form computed below (the per-expert term is folded into
    # ``b_base_ptr``). GU-fusion builds two N-offset tensors -- one
    # for the gate half starting at column ``pid_n * BLOCK_N``, one
    # for the up half starting at ``(pid_n + num_pid_n) * BLOCK_N`` --
    # against the same K offsets. The ``% N`` wrap defends a
    # degenerate ``I_r < BLOCK_N`` test path where the up rows would
    # otherwise exceed ``N``.
    offs_bk = gl.arange(0, BLOCK_K_PACKED, layout=gl.SliceLayout(1, dot_b_layout))
    offs_bn_gate = (
        pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, dot_b_layout))
    ) % N
    offs_bn_up = (
        (pid_n + num_pid_n) * BLOCK_N
        + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, dot_b_layout))
    ) % N
    if B_GDOT128:
        n_tiles: gl.constexpr = (2 * I_r) // 128
        # Tokenspeed stores gate/up rows interleaved (g0,u0,g1,u1,...),
        # while this kernel's logical N axis is concatenated gate||up.
        n_gate_phys = offs_bn_gate * 2
        n_up_phys = (offs_bn_up - I_r) * 2 + 1
        k_in = offs_bk % 128
        k_within = k_in % 16
        k_quad = (k_in // 16) % 4
        k_block = k_in // 64
        n_gate_in = n_gate_phys % 128
        n_up_in = n_up_phys % 128
        in_gate = (
            (n_gate_in // 16)[None, :] * 2048
            + k_block[:, None] * 1024
            + k_quad[:, None] * 256
            + (n_gate_in % 16)[None, :] * 16
            + k_within[:, None]
        )
        in_up = (
            (n_up_in // 16)[None, :] * 2048
            + k_block[:, None] * 1024
            + k_quad[:, None] * 256
            + (n_up_in % 16)[None, :] * 16
            + k_within[:, None]
        )
        offs_b_gate = (
            (offs_bk // 128)[:, None] * n_tiles + (n_gate_phys // 128)[None, :]
        ) * (128 * 128) + in_gate
        offs_b_up = (
            (offs_bk // 128)[:, None] * n_tiles + (n_up_phys // 128)[None, :]
        ) * (128 * 128) + in_up
    else:
        b_k_part = (
            (offs_bk // 32) * 512 + ((offs_bk // 16) % 2) * 256 + (offs_bk % 16)
        )[:, None]
        b_n_part_gate = (
            (offs_bn_gate // 16) * (16 * K_PACKED_TOTAL) + (offs_bn_gate % 16) * 16
        )[None, :]
        b_n_part_up = (
            (offs_bn_up // 16) * (16 * K_PACKED_TOTAL) + (offs_bn_up % 16) * 16
        )[None, :]
        offs_b_gate = b_k_part + b_n_part_gate
        offs_b_up = b_k_part + b_n_part_up

    # ---- B_scale offsets (gate + up halves; e8m0_shuffle_opsel_b) -----
    # Same row-part decomposition as the prior preshuffle-B port; the
    # cross-tile step is the constexpr ``B_SCALE_K_STEP = 1024`` because
    # ``BLOCK_K_SCALE = 8`` covers a full ``y/8`` block per K iter. The
    # per-expert offset folds into ``b_scale_offsets_e_*`` here so the
    # K-loop only carries the K-step.
    b_n_layout_scale: gl.constexpr = gl.SliceLayout(1, b_scale_layout)
    b_k_layout_scale: gl.constexpr = gl.SliceLayout(0, b_scale_layout)
    b_rows_gate = (pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=b_n_layout_scale)).to(
        gl.uint32
    )
    b_rows_up = (
        (pid_n + num_pid_n) * BLOCK_N + gl.arange(0, BLOCK_N, layout=b_n_layout_scale)
    ).to(gl.uint32)
    if B_GDOT128:
        b_rows_gate_phys = b_rows_gate * 2
        b_rows_up_phys = (b_rows_up - I_r) * 2 + 1
        b_n_part_gate_scale = (
            (b_rows_gate_phys // 32) * (K_PACKED_TOTAL * 2)
            + (b_rows_gate_phys % 16) * 4
            + ((b_rows_gate_phys % 32) // 16)
        )[:, None]
        b_n_part_up_scale = (
            (b_rows_up_phys // 32) * (K_PACKED_TOTAL * 2)
            + (b_rows_up_phys % 16) * 4
            + ((b_rows_up_phys % 32) // 16)
        )[:, None]
    else:
        b_n_part_gate_scale = (
            (b_rows_gate // 128) * (stride_se_n_pad * 128)
            + ((b_rows_gate % 64) // 16) * 64
            + (b_rows_gate % 16) * 4
            + ((b_rows_gate % 128) // 64) * 2
        )[:, None]
        b_n_part_up_scale = (
            (b_rows_up // 128) * (stride_se_n_pad * 128)
            + ((b_rows_up % 64) // 16) * 64
            + (b_rows_up % 16) * 4
            + ((b_rows_up % 128) // 64) * 2
        )[:, None]
    b_k_lanes = gl.arange(0, BLOCK_K_SCALE, layout=b_k_layout_scale).to(gl.uint32)
    # Loop-invariant K-part of the B-scale gather offset. The full
    # per-K offset is k*1024 + b_k_lane_offsets; the second term only
    # depends on the lane index in [0,8), so we hoist it (and its sum
    # with the per-row N-part) out of the K-loop. See _load_b_scale_vgpr
    # for the algebraic justification.
    if B_GDOT128:
        b_k_lane_offsets = ((b_k_lanes % 4) * 64 + ((b_k_lanes % 8) // 4) * 2)[None, :]
        B_SCALE_K_STEP: gl.constexpr = 256
    else:
        b_k_lane_offsets = ((b_k_lanes % 4) * 256 + (b_k_lanes // 4))[None, :]
        B_SCALE_K_STEP: gl.constexpr = 1024
    b_scale_static_offs_gate = b_n_part_gate_scale + b_k_lane_offsets
    b_scale_static_offs_up = b_n_part_up_scale + b_k_lane_offsets
    b_scales_ptr_e = b_scales_ptr + off_experts.to(gl.int64) * stride_bse_e

    # ---- LDS allocation (2 A-data slots, ping-pong) --------------------
    # A fp4 data is still gathered through LDS. A scales use direct VGPR
    # loads in the decode/CDNA4-swizzled layout, avoiding the tiny A-scale
    # async-copy path that Gluon fails to lower for prefill.
    smem_a = gl.allocate_shared_memory(
        a_ptr.type.element_ty,
        [NUM_BUFFERS, BLOCK_M, BLOCK_K_PACKED],
        shared_a,
    )

    a_base_ptr = a_ptr
    b_base_ptr = b_ptr + off_experts.to(gl.int64) * stride_be

    offs_a_i32 = offs_a.to(gl.int32)

    A_DATA_K_STEP: gl.constexpr = BLOCK_K_PACKED
    if B_GDOT128:
        B_DATA_K_STEP: gl.constexpr = ((2 * I_r) // 128) * (128 * 128)
    else:
        B_DATA_K_STEP: gl.constexpr = (BLOCK_K_PACKED // 32) * 512
    A_SCALE_K_STEP: gl.constexpr = 256

    GROUP0_IDX: gl.constexpr = 0
    GROUP1_IDX: gl.constexpr = 1
    GROUP2_IDX: gl.constexpr = 2
    GROUP3_IDX: gl.constexpr = 3

    smem_a_slot0 = smem_a.index(0)
    smem_a_slot1 = smem_a.index(1)

    # ---- 1-tile prologue ------------------------------------------------
    # Prefetch A[0] data into LDS slot 0, load B[0] into VGPRs. A-scale is
    # direct-to-VGPR in the decode/CDNA4-swizzled layout.
    _prefetch_a_data_lds(
        smem_a_slot0,
        a_base_ptr,
        offs_a_i32,
        0,
        mask=token_mask[:, None],
    )
    cdna4_async_copy.commit_group()

    cur_b_gate = _load_b_data_vgpr(
        b_base_ptr,
        offs_b_gate,
        0,
    )
    cur_b_up = _load_b_data_vgpr(
        b_base_ptr,
        offs_b_up,
        0,
    )
    b_scale_gate = _load_b_scale_vgpr(
        b_scales_ptr_e,
        b_scale_static_offs_gate,
        0,
        B_SCALE_K_STEP,
    )
    b_scale_up = _load_b_scale_vgpr(
        b_scales_ptr_e,
        b_scale_static_offs_up,
        0,
        B_SCALE_K_STEP,
    )

    cdna4_async_copy.wait_group(0)
    cur_a_group0 = _read_a_lds_group(
        smem_a_slot0,
        GROUP0_IDX,
        dot_a_layout,
        GROUP_MFMA_M,
    )
    a_scale_group0 = _load_a_scale_vgpr(
        a_scales_ptr,
        a_scale_base_offsets,
        GROUP0_IDX,
        0,
        a_scale_group_stride,
        A_SCALE_K_STEP,
    )
    cur_a_group1 = _read_a_lds_group(
        smem_a_slot0,
        GROUP1_IDX,
        dot_a_layout,
        GROUP_MFMA_M,
    )
    a_scale_group1 = _load_a_scale_vgpr(
        a_scales_ptr,
        a_scale_base_offsets,
        GROUP1_IDX,
        0,
        a_scale_group_stride,
        A_SCALE_K_STEP,
    )

    # ---- 8 per-quarter accumulators (gate + up x 4 quarters) ----------
    gate_acc_group0 = gl.zeros(
        (GROUP_MFMA_M, BLOCK_N), dtype=gl.float32, layout=mfma_layout
    )
    gate_acc_group1 = gl.zeros(
        (GROUP_MFMA_M, BLOCK_N), dtype=gl.float32, layout=mfma_layout
    )
    gate_acc_group2 = gl.zeros(
        (GROUP_MFMA_M, BLOCK_N), dtype=gl.float32, layout=mfma_layout
    )
    gate_acc_group3 = gl.zeros(
        (GROUP_MFMA_M, BLOCK_N), dtype=gl.float32, layout=mfma_layout
    )
    up_acc_group0 = gl.zeros(
        (GROUP_MFMA_M, BLOCK_N), dtype=gl.float32, layout=mfma_layout
    )
    up_acc_group1 = gl.zeros(
        (GROUP_MFMA_M, BLOCK_N), dtype=gl.float32, layout=mfma_layout
    )
    up_acc_group2 = gl.zeros(
        (GROUP_MFMA_M, BLOCK_N), dtype=gl.float32, layout=mfma_layout
    )
    up_acc_group3 = gl.zeros(
        (GROUP_MFMA_M, BLOCK_N), dtype=gl.float32, layout=mfma_layout
    )

    num_k_iter = gl.cdiv(K, BLOCK_K)
    # Steady tiles: 0 .. num_k_iter-2 (last tile handled in drain).
    steady_iters = num_k_iter - 1
    unrolled_iters = steady_iters // 2
    peel_iters = steady_iters % 2

    # ---- 2x-unrolled steady-state K-loop with B VGPR ping-pong -------
    # Per tile body: issue B[k+1] loads FIRST (overlap with MFMA),
    # then compute with A[k]+B[k], prefetch A[k+1] to LDS, swap slots.
    for unrolled_k in range(unrolled_iters):
        tile0_k = unrolled_k * 2
        tile1_k = tile0_k + 1

        # ==============================================================
        # ---- Tile 0 (even k, cur_slot=slot0, next_slot=slot1) --------
        # ==============================================================
        nxt_b_data_off = (tile0_k + 1) * B_DATA_K_STEP
        nxt_b_scale_k = tile0_k + 1

        # Issue B[k+1] loads early for overlap with MFMA below.
        next_b_gate = _load_b_data_vgpr(
            b_base_ptr,
            offs_b_gate,
            nxt_b_data_off,
        )
        next_b_up = _load_b_data_vgpr(
            b_base_ptr,
            offs_b_up,
            nxt_b_data_off,
        )
        next_bsc_gate = _load_b_scale_vgpr(
            b_scales_ptr_e,
            b_scale_static_offs_gate,
            nxt_b_scale_k,
            B_SCALE_K_STEP,
        )
        next_bsc_up = _load_b_scale_vgpr(
            b_scales_ptr_e,
            b_scale_static_offs_up,
            nxt_b_scale_k,
            B_SCALE_K_STEP,
        )

        # MFMA groups 0,1 with current A[k] + current B[k].
        gate_acc_group0 = _compute_mxfp4_group(
            cur_a_group0,
            a_scale_group0,
            cur_b_gate,
            b_scale_gate,
            gate_acc_group0,
        )
        up_acc_group0 = _compute_mxfp4_group(
            cur_a_group0,
            a_scale_group0,
            cur_b_up,
            b_scale_up,
            up_acc_group0,
        )
        gate_acc_group1 = _compute_mxfp4_group(
            cur_a_group1,
            a_scale_group1,
            cur_b_gate,
            b_scale_gate,
            gate_acc_group1,
        )
        up_acc_group1 = _compute_mxfp4_group(
            cur_a_group1,
            a_scale_group1,
            cur_b_up,
            b_scale_up,
            up_acc_group1,
        )

        # Prefetch A[k+1] data into LDS slot1. A-scale stays direct-to-VGPR.
        _prefetch_a_data_lds(
            smem_a_slot1,
            a_base_ptr,
            offs_a_i32,
            (tile0_k + 1) * A_DATA_K_STEP,
            mask=token_mask[:, None],
        )

        # Read remaining A[k] groups from current LDS slot0.
        cur_a_group2 = _read_a_lds_group(
            smem_a_slot0,
            GROUP2_IDX,
            dot_a_layout,
            GROUP_MFMA_M,
        )
        a_scale_group2 = _load_a_scale_vgpr(
            a_scales_ptr,
            a_scale_base_offsets,
            GROUP2_IDX,
            tile0_k,
            a_scale_group_stride,
            A_SCALE_K_STEP,
        )
        cur_a_group3 = _read_a_lds_group(
            smem_a_slot0,
            GROUP3_IDX,
            dot_a_layout,
            GROUP_MFMA_M,
        )
        a_scale_group3 = _load_a_scale_vgpr(
            a_scales_ptr,
            a_scale_base_offsets,
            GROUP3_IDX,
            tile0_k,
            a_scale_group_stride,
            A_SCALE_K_STEP,
        )

        cdna4_async_copy.commit_group()

        # MFMA groups 2,3 with current A[k] + current B[k].
        gate_acc_group2 = _compute_mxfp4_group(
            cur_a_group2,
            a_scale_group2,
            cur_b_gate,
            b_scale_gate,
            gate_acc_group2,
        )
        up_acc_group2 = _compute_mxfp4_group(
            cur_a_group2,
            a_scale_group2,
            cur_b_up,
            b_scale_up,
            up_acc_group2,
        )
        gate_acc_group3 = _compute_mxfp4_group(
            cur_a_group3,
            a_scale_group3,
            cur_b_gate,
            b_scale_gate,
            gate_acc_group3,
        )
        up_acc_group3 = _compute_mxfp4_group(
            cur_a_group3,
            a_scale_group3,
            cur_b_up,
            b_scale_up,
            up_acc_group3,
        )

        # Drain A[k+1] prefetch; B[k+1] should also be done by now.
        cdna4_async_copy.wait_group(0)

        # Read A[k+1] groups 0,1 from LDS slot1.
        cur_a_group0 = _read_a_lds_group(
            smem_a_slot1,
            GROUP0_IDX,
            dot_a_layout,
            GROUP_MFMA_M,
        )
        a_scale_group0 = _load_a_scale_vgpr(
            a_scales_ptr,
            a_scale_base_offsets,
            GROUP0_IDX,
            tile0_k + 1,
            a_scale_group_stride,
            A_SCALE_K_STEP,
        )
        cur_a_group1 = _read_a_lds_group(
            smem_a_slot1,
            GROUP1_IDX,
            dot_a_layout,
            GROUP_MFMA_M,
        )
        a_scale_group1 = _load_a_scale_vgpr(
            a_scales_ptr,
            a_scale_base_offsets,
            GROUP1_IDX,
            tile0_k + 1,
            a_scale_group_stride,
            A_SCALE_K_STEP,
        )

        # Swap B: next -> current.
        cur_b_gate = next_b_gate
        cur_b_up = next_b_up
        b_scale_gate = next_bsc_gate
        b_scale_up = next_bsc_up

        # Swap A LDS slots.
        smem_a_slot0, smem_a_slot1 = smem_a_slot1, smem_a_slot0

        # ==============================================================
        # ---- Tile 1 (odd k+1, cur_slot=slot0, next_slot=slot1) -------
        # ==============================================================
        nxt_b_data_off = (tile1_k + 1) * B_DATA_K_STEP
        nxt_b_scale_k = tile1_k + 1

        next_b_gate = _load_b_data_vgpr(
            b_base_ptr,
            offs_b_gate,
            nxt_b_data_off,
        )
        next_b_up = _load_b_data_vgpr(
            b_base_ptr,
            offs_b_up,
            nxt_b_data_off,
        )
        next_bsc_gate = _load_b_scale_vgpr(
            b_scales_ptr_e,
            b_scale_static_offs_gate,
            nxt_b_scale_k,
            B_SCALE_K_STEP,
        )
        next_bsc_up = _load_b_scale_vgpr(
            b_scales_ptr_e,
            b_scale_static_offs_up,
            nxt_b_scale_k,
            B_SCALE_K_STEP,
        )

        gate_acc_group0 = _compute_mxfp4_group(
            cur_a_group0,
            a_scale_group0,
            cur_b_gate,
            b_scale_gate,
            gate_acc_group0,
        )
        up_acc_group0 = _compute_mxfp4_group(
            cur_a_group0,
            a_scale_group0,
            cur_b_up,
            b_scale_up,
            up_acc_group0,
        )
        gate_acc_group1 = _compute_mxfp4_group(
            cur_a_group1,
            a_scale_group1,
            cur_b_gate,
            b_scale_gate,
            gate_acc_group1,
        )
        up_acc_group1 = _compute_mxfp4_group(
            cur_a_group1,
            a_scale_group1,
            cur_b_up,
            b_scale_up,
            up_acc_group1,
        )

        _prefetch_a_data_lds(
            smem_a_slot1,
            a_base_ptr,
            offs_a_i32,
            (tile1_k + 1) * A_DATA_K_STEP,
            mask=token_mask[:, None],
        )

        cur_a_group2 = _read_a_lds_group(
            smem_a_slot0,
            GROUP2_IDX,
            dot_a_layout,
            GROUP_MFMA_M,
        )
        a_scale_group2 = _load_a_scale_vgpr(
            a_scales_ptr,
            a_scale_base_offsets,
            GROUP2_IDX,
            tile1_k,
            a_scale_group_stride,
            A_SCALE_K_STEP,
        )
        cur_a_group3 = _read_a_lds_group(
            smem_a_slot0,
            GROUP3_IDX,
            dot_a_layout,
            GROUP_MFMA_M,
        )
        a_scale_group3 = _load_a_scale_vgpr(
            a_scales_ptr,
            a_scale_base_offsets,
            GROUP3_IDX,
            tile1_k,
            a_scale_group_stride,
            A_SCALE_K_STEP,
        )

        cdna4_async_copy.commit_group()

        gate_acc_group2 = _compute_mxfp4_group(
            cur_a_group2,
            a_scale_group2,
            cur_b_gate,
            b_scale_gate,
            gate_acc_group2,
        )
        up_acc_group2 = _compute_mxfp4_group(
            cur_a_group2,
            a_scale_group2,
            cur_b_up,
            b_scale_up,
            up_acc_group2,
        )
        gate_acc_group3 = _compute_mxfp4_group(
            cur_a_group3,
            a_scale_group3,
            cur_b_gate,
            b_scale_gate,
            gate_acc_group3,
        )
        up_acc_group3 = _compute_mxfp4_group(
            cur_a_group3,
            a_scale_group3,
            cur_b_up,
            b_scale_up,
            up_acc_group3,
        )

        cdna4_async_copy.wait_group(0)

        cur_a_group0 = _read_a_lds_group(
            smem_a_slot1,
            GROUP0_IDX,
            dot_a_layout,
            GROUP_MFMA_M,
        )
        a_scale_group0 = _load_a_scale_vgpr(
            a_scales_ptr,
            a_scale_base_offsets,
            GROUP0_IDX,
            tile1_k + 1,
            a_scale_group_stride,
            A_SCALE_K_STEP,
        )
        cur_a_group1 = _read_a_lds_group(
            smem_a_slot1,
            GROUP1_IDX,
            dot_a_layout,
            GROUP_MFMA_M,
        )
        a_scale_group1 = _load_a_scale_vgpr(
            a_scales_ptr,
            a_scale_base_offsets,
            GROUP1_IDX,
            tile1_k + 1,
            a_scale_group_stride,
            A_SCALE_K_STEP,
        )

        cur_b_gate = next_b_gate
        cur_b_up = next_b_up
        b_scale_gate = next_bsc_gate
        b_scale_up = next_bsc_up

        smem_a_slot0, smem_a_slot1 = smem_a_slot1, smem_a_slot0

    # ---- Peel loop (0 or 1 single-tile advance) ----------------------
    peel_base = unrolled_iters * 2
    for peel_k_offset in range(peel_iters):
        pk = peel_base + peel_k_offset
        nxt_b_data_off = (pk + 1) * B_DATA_K_STEP
        nxt_b_scale_k = pk + 1

        next_b_gate = _load_b_data_vgpr(
            b_base_ptr,
            offs_b_gate,
            nxt_b_data_off,
        )
        next_b_up = _load_b_data_vgpr(
            b_base_ptr,
            offs_b_up,
            nxt_b_data_off,
        )
        next_bsc_gate = _load_b_scale_vgpr(
            b_scales_ptr_e,
            b_scale_static_offs_gate,
            nxt_b_scale_k,
            B_SCALE_K_STEP,
        )
        next_bsc_up = _load_b_scale_vgpr(
            b_scales_ptr_e,
            b_scale_static_offs_up,
            nxt_b_scale_k,
            B_SCALE_K_STEP,
        )

        gate_acc_group0 = _compute_mxfp4_group(
            cur_a_group0,
            a_scale_group0,
            cur_b_gate,
            b_scale_gate,
            gate_acc_group0,
        )
        up_acc_group0 = _compute_mxfp4_group(
            cur_a_group0,
            a_scale_group0,
            cur_b_up,
            b_scale_up,
            up_acc_group0,
        )
        gate_acc_group1 = _compute_mxfp4_group(
            cur_a_group1,
            a_scale_group1,
            cur_b_gate,
            b_scale_gate,
            gate_acc_group1,
        )
        up_acc_group1 = _compute_mxfp4_group(
            cur_a_group1,
            a_scale_group1,
            cur_b_up,
            b_scale_up,
            up_acc_group1,
        )

        _prefetch_a_data_lds(
            smem_a_slot1,
            a_base_ptr,
            offs_a_i32,
            (pk + 1) * A_DATA_K_STEP,
            mask=token_mask[:, None],
        )

        cur_a_group2 = _read_a_lds_group(
            smem_a_slot0,
            GROUP2_IDX,
            dot_a_layout,
            GROUP_MFMA_M,
        )
        a_scale_group2 = _load_a_scale_vgpr(
            a_scales_ptr,
            a_scale_base_offsets,
            GROUP2_IDX,
            pk,
            a_scale_group_stride,
            A_SCALE_K_STEP,
        )
        cur_a_group3 = _read_a_lds_group(
            smem_a_slot0,
            GROUP3_IDX,
            dot_a_layout,
            GROUP_MFMA_M,
        )
        a_scale_group3 = _load_a_scale_vgpr(
            a_scales_ptr,
            a_scale_base_offsets,
            GROUP3_IDX,
            pk,
            a_scale_group_stride,
            A_SCALE_K_STEP,
        )

        cdna4_async_copy.commit_group()

        gate_acc_group2 = _compute_mxfp4_group(
            cur_a_group2,
            a_scale_group2,
            cur_b_gate,
            b_scale_gate,
            gate_acc_group2,
        )
        up_acc_group2 = _compute_mxfp4_group(
            cur_a_group2,
            a_scale_group2,
            cur_b_up,
            b_scale_up,
            up_acc_group2,
        )
        gate_acc_group3 = _compute_mxfp4_group(
            cur_a_group3,
            a_scale_group3,
            cur_b_gate,
            b_scale_gate,
            gate_acc_group3,
        )
        up_acc_group3 = _compute_mxfp4_group(
            cur_a_group3,
            a_scale_group3,
            cur_b_up,
            b_scale_up,
            up_acc_group3,
        )

        cdna4_async_copy.wait_group(0)

        cur_a_group0 = _read_a_lds_group(
            smem_a_slot1,
            GROUP0_IDX,
            dot_a_layout,
            GROUP_MFMA_M,
        )
        a_scale_group0 = _load_a_scale_vgpr(
            a_scales_ptr,
            a_scale_base_offsets,
            GROUP0_IDX,
            pk + 1,
            a_scale_group_stride,
            A_SCALE_K_STEP,
        )
        cur_a_group1 = _read_a_lds_group(
            smem_a_slot1,
            GROUP1_IDX,
            dot_a_layout,
            GROUP_MFMA_M,
        )
        a_scale_group1 = _load_a_scale_vgpr(
            a_scales_ptr,
            a_scale_base_offsets,
            GROUP1_IDX,
            pk + 1,
            a_scale_group_stride,
            A_SCALE_K_STEP,
        )

        cur_b_gate = next_b_gate
        cur_b_up = next_b_up
        b_scale_gate = next_bsc_gate
        b_scale_up = next_bsc_up

        smem_a_slot0, smem_a_slot1 = smem_a_slot1, smem_a_slot0

    # ---- 1-tile drain (last K tile, no prefetch) ---------------------
    # cur_a_group0/1 and cur_b_* hold the last tile's data.
    cur_a_group2 = _read_a_lds_group(
        smem_a_slot0,
        GROUP2_IDX,
        dot_a_layout,
        GROUP_MFMA_M,
    )
    a_scale_group2 = _load_a_scale_vgpr(
        a_scales_ptr,
        a_scale_base_offsets,
        GROUP2_IDX,
        num_k_iter - 1,
        a_scale_group_stride,
        A_SCALE_K_STEP,
    )
    cur_a_group3 = _read_a_lds_group(
        smem_a_slot0,
        GROUP3_IDX,
        dot_a_layout,
        GROUP_MFMA_M,
    )
    a_scale_group3 = _load_a_scale_vgpr(
        a_scales_ptr,
        a_scale_base_offsets,
        GROUP3_IDX,
        num_k_iter - 1,
        a_scale_group_stride,
        A_SCALE_K_STEP,
    )

    gate_acc_group0 = _compute_mxfp4_group(
        cur_a_group0,
        a_scale_group0,
        cur_b_gate,
        b_scale_gate,
        gate_acc_group0,
    )
    up_acc_group0 = _compute_mxfp4_group(
        cur_a_group0,
        a_scale_group0,
        cur_b_up,
        b_scale_up,
        up_acc_group0,
    )
    gate_acc_group1 = _compute_mxfp4_group(
        cur_a_group1,
        a_scale_group1,
        cur_b_gate,
        b_scale_gate,
        gate_acc_group1,
    )
    up_acc_group1 = _compute_mxfp4_group(
        cur_a_group1,
        a_scale_group1,
        cur_b_up,
        b_scale_up,
        up_acc_group1,
    )
    gate_acc_group2 = _compute_mxfp4_group(
        cur_a_group2,
        a_scale_group2,
        cur_b_gate,
        b_scale_gate,
        gate_acc_group2,
    )
    up_acc_group2 = _compute_mxfp4_group(
        cur_a_group2,
        a_scale_group2,
        cur_b_up,
        b_scale_up,
        up_acc_group2,
    )
    gate_acc_group3 = _compute_mxfp4_group(
        cur_a_group3,
        a_scale_group3,
        cur_b_gate,
        b_scale_gate,
        gate_acc_group3,
    )
    up_acc_group3 = _compute_mxfp4_group(
        cur_a_group3,
        a_scale_group3,
        cur_b_up,
        b_scale_up,
        up_acc_group3,
    )

    # ---- SwiGLU epilogue ------------------------------------------------
    # Per-quarter SwiGLU.  The historical path is plain
    # ``silu(gate) * up`` and is represented by
    # ``alpha=1, limit=0, beta=0``.  Kimi uses the parameterized
    # form ``gate * sigmoid(alpha * gate) * (clamp(up) + beta)`` with
    # optional gate/up clamping.
    # ``gl.sigmoid`` is not in the Gluon language module today, so we
    # build sigmoid from ``gl.exp``. Each quarter's fp32 result casts to
    # bf16 inside ``_store_swiglu_tile_group`` and lands at
    # ``dst_row = token_id * top_k + topk_id`` decoded from the
    # bit-packed ``sorted_token_ids[m]``; padding rows are mask-rejected.
    if SWIGLU_LIMIT > 0.0:
        gate_acc_group0 = gl.minimum(gate_acc_group0, SWIGLU_LIMIT)
        gate_acc_group1 = gl.minimum(gate_acc_group1, SWIGLU_LIMIT)
        gate_acc_group2 = gl.minimum(gate_acc_group2, SWIGLU_LIMIT)
        gate_acc_group3 = gl.minimum(gate_acc_group3, SWIGLU_LIMIT)
        up_acc_group0 = gl.minimum(
            gl.maximum(up_acc_group0, -SWIGLU_LIMIT), SWIGLU_LIMIT
        )
        up_acc_group1 = gl.minimum(
            gl.maximum(up_acc_group1, -SWIGLU_LIMIT), SWIGLU_LIMIT
        )
        up_acc_group2 = gl.minimum(
            gl.maximum(up_acc_group2, -SWIGLU_LIMIT), SWIGLU_LIMIT
        )
        up_acc_group3 = gl.minimum(
            gl.maximum(up_acc_group3, -SWIGLU_LIMIT), SWIGLU_LIMIT
        )
    # Match the reference exact floating-point grouping.  Computing a reciprocal
    # first and then multiplying by ``gate`` is algebraically equivalent but
    # changes sparse near-zero BF16 results by one ULP after the final cast.
    silu_g0 = gate_acc_group0 / (1.0 + gl.exp(-(SWIGLU_ALPHA * gate_acc_group0)))
    acc_swiglu_group0 = gl.fma(silu_g0, up_acc_group0, silu_g0 * SWIGLU_BETA)
    silu_g1 = gate_acc_group1 / (1.0 + gl.exp(-(SWIGLU_ALPHA * gate_acc_group1)))
    acc_swiglu_group1 = gl.fma(silu_g1, up_acc_group1, silu_g1 * SWIGLU_BETA)
    silu_g2 = gate_acc_group2 / (1.0 + gl.exp(-(SWIGLU_ALPHA * gate_acc_group2)))
    acc_swiglu_group2 = gl.fma(silu_g2, up_acc_group2, silu_g2 * SWIGLU_BETA)
    silu_g3 = gate_acc_group3 / (1.0 + gl.exp(-(SWIGLU_ALPHA * gate_acc_group3)))
    acc_swiglu_group3 = gl.fma(silu_g3, up_acc_group3, silu_g3 * SWIGLU_BETA)

    _store_swiglu_tile_group(
        acc_swiglu_group0,
        c_ptr,
        sorted_token_ids_ptr,
        pid_m,
        pid_n,
        EM,
        num_tokens,
        top_k,
        stride_cm,
        stride_cn,
        mfma_layout,
        GROUP0_IDX,
        GROUP_MFMA_M,
        BLOCK_M,
        BLOCK_N,
        I_r,
    )
    _store_swiglu_tile_group(
        acc_swiglu_group1,
        c_ptr,
        sorted_token_ids_ptr,
        pid_m,
        pid_n,
        EM,
        num_tokens,
        top_k,
        stride_cm,
        stride_cn,
        mfma_layout,
        GROUP1_IDX,
        GROUP_MFMA_M,
        BLOCK_M,
        BLOCK_N,
        I_r,
    )
    _store_swiglu_tile_group(
        acc_swiglu_group2,
        c_ptr,
        sorted_token_ids_ptr,
        pid_m,
        pid_n,
        EM,
        num_tokens,
        top_k,
        stride_cm,
        stride_cn,
        mfma_layout,
        GROUP2_IDX,
        GROUP_MFMA_M,
        BLOCK_M,
        BLOCK_N,
        I_r,
    )
    _store_swiglu_tile_group(
        acc_swiglu_group3,
        c_ptr,
        sorted_token_ids_ptr,
        pid_m,
        pid_n,
        EM,
        num_tokens,
        top_k,
        stride_cm,
        stride_cn,
        mfma_layout,
        GROUP3_IDX,
        GROUP_MFMA_M,
        BLOCK_M,
        BLOCK_N,
        I_r,
    )


def invoke_gluon_mxfp4_moe_stage1(
    hidden_states,
    w1,
    w2,
    sorted_token_ids,
    sorted_expert_ids,
    num_valid_ids,
    out,
    topk,
    kernelName="",
    w1_scale=None,
    a1_scale=None,
    block_m=32,
    sorted_weights=None,
    quant_type=None,
    activation=None,
    splitk=1,
    use_non_temporal_load=False,
    dst_type=None,
    b_preshuffled: bool = False,
    b_gdot128: bool = False,
    swiglu_alpha: float = 1.0,
    swiglu_limit: float = 0.0,
    swiglu_beta: float = 0.0,
):
    """Host-side launcher for Gluon MXFP4 MoE stage 1.

    Computes, for each expert ``e``::

        inter_e = silu(hidden_states_e @ w1_e[:I_r, :].T)
                  * (hidden_states_e @ w1_e[I_r:, :].T)

    in one kernel launch over all experts. The grid is one CTA per
    (M-tile, N-tile); each CTA owns one ``BLOCK_M`` x ``BLOCK_N``
    output region for the expert ``sorted_expert_ids[pid_m]``.

    Steps below correspond to the ``# Step N:`` comments in the body:
      1. Validate dtypes / shapes; reject unsupported options.
      2. Permute ``w1`` to the (16, 16) MFMA-tile layout (see
         :func:`_b_preshuffle_3d`), or skip if the caller already did.
      3. Materialise ``num_valid_ids`` / ``sorted_weights`` as device
         tensors when the caller passed a Python scalar / ``None``.
      4. Launch the GEMM kernel.

    Layout contract: see the module docstring.

    Unsupported (raises ``NotImplementedError``):
      ``quant_type != per_1x32``, ``activation != Silu``,
      ``splitk not in {0, 1, None}``, non-empty ``sorted_weights``,
      ``dst_type != bfloat16``.

    Accepted-but-ignored, kept for signature compatibility with
    upstream dispatchers: ``kernelName``, ``block_m`` (kernel hardcodes
    128), ``w2``, ``use_non_temporal_load``.
    """
    # Step 1: validate inputs and reject unsupported modes.
    del quant_type, activation  # only per-1x32 MXFP4 + SwiGLU is implemented
    if splitk not in (0, 1, None):
        raise NotImplementedError(
            "invoke_gluon_mxfp4_moe_stage1: splitk in {0, 1} expected "
            f"(got splitk={splitk!r})"
        )
    if sorted_weights is not None and sorted_weights.numel() > 0:
        raise NotImplementedError(
            "invoke_gluon_mxfp4_moe_stage1: do_weight_stage1 (a non-empty "
            "sorted_weights tensor) is not implemented; pass None or a "
            "0-element placeholder"
        )
    if dst_type is not None and dst_type != torch.bfloat16:
        raise NotImplementedError(
            "invoke_gluon_mxfp4_moe_stage1: only dst_type=torch.bfloat16 is "
            f"supported, got dst_type={dst_type!r}"
        )
    del kernelName, block_m, use_non_temporal_load, w2

    assert (
        hidden_states.dtype == torch.uint8
    ), f"hidden_states must be packed fp4x2 (uint8), got {hidden_states.dtype}"
    assert w1.dtype == torch.uint8, f"w1 must be packed fp4x2 (uint8), got {w1.dtype}"
    # Step 2: shuffle w1 to the MFMA-tile layout the kernel expects.
    if not b_preshuffled:
        w1 = _b_preshuffle_3d(w1)
    if b_gdot128 and not b_preshuffled:
        raise ValueError("b_gdot128 requires a preshuffled gdot128 weight alias")
    assert w1_scale is not None and w1_scale.dtype == torch.uint8, (
        "w1_scale must be a uint8 e8m0 tensor, got "
        f"{None if w1_scale is None else w1_scale.dtype}"
    )
    assert a1_scale is not None and a1_scale.dtype == torch.uint8, (
        "a1_scale must be a uint8 e8m0 tensor, got "
        f"{None if a1_scale is None else a1_scale.dtype}"
    )
    assert out.dtype == torch.bfloat16, f"out must be bfloat16, got {out.dtype}"
    assert (
        sorted_token_ids.dtype == torch.int32
    ), f"sorted_token_ids must be int32, got {sorted_token_ids.dtype}"
    assert (
        sorted_expert_ids.dtype == torch.int32
    ), f"sorted_expert_ids must be int32, got {sorted_expert_ids.dtype}"

    assert hidden_states.dim() == 2, (
        f"hidden_states must be 2-D (M_padded, K_packed), got "
        f"shape {tuple(hidden_states.shape)}"
    )
    M_padded, D_packed = hidden_states.shape
    K = D_packed * 2
    assert (
        w1.dim() == 3
    ), f"w1 must be 3-D (E, 2*I_r, K_packed), got shape {tuple(w1.shape)}"
    E_w, two_Ir, D_packed_w = w1.shape
    assert D_packed_w == D_packed, (
        f"w1 K-packed dim ({D_packed_w}) must match hidden_states K-packed "
        f"dim ({D_packed})"
    )
    assert two_Ir % 2 == 0, f"w1.shape[1] (= 2*I_r) must be even, got {two_Ir}"
    I_r = two_Ir // 2
    N = 2 * I_r
    EM = sorted_token_ids.shape[0]
    if out.dim() == 3:
        token_num, top_k_dim, I_r_dim = out.shape
        assert top_k_dim == topk, f"out top_k mismatch: {top_k_dim} vs topk={topk}"
        assert I_r_dim == I_r, f"out I_r mismatch: {I_r_dim} vs {I_r}"
        assert out.is_contiguous(), (
            "3-D out tensor must be contiguous so .view(token_num*topk, I_r) "
            "shares memory with the original; got non-contiguous out"
        )
        out_2d = out.view(token_num * topk, I_r)
    elif out.dim() == 2:
        out_2d = out
        assert out.shape[0] >= EM, f"out 2-D first dim {out.shape[0]} < EM={EM}"
        assert (
            out.shape[1] == I_r
        ), f"out 2-D second dim {out.shape[1]} must equal I_r={I_r}"
    else:
        raise NotImplementedError(
            "out must be 2-D (EM, I_r) or 3-D (token_num, topk, I_r); "
            f"got shape {tuple(out.shape)}"
        )
    K_scale = K // 32
    assert a1_scale.dim() == 2 and a1_scale.shape[1] == K_scale, (
        f"a1_scale.shape {tuple(a1_scale.shape)} must be "
        f"(M_padded_aligned, K // 32 = {K_scale})"
    )
    assert w1_scale.shape == (E_w, N, K_scale), (
        f"w1_scale.shape {tuple(w1_scale.shape)} must be (E, N, K//32) = "
        f"({E_w}, {N}, {K_scale})"
    )
    num_tokens = M_padded
    del M_padded

    # Step 3: materialise scalar / None args as device tensors so the
    # kernel can just `gl.load` from them.
    if torch.is_tensor(num_valid_ids):
        num_valid_ids_ptr = num_valid_ids
    else:
        num_valid_ids_ptr = torch.tensor(
            [int(num_valid_ids)], dtype=torch.int32, device=hidden_states.device
        )
    if sorted_weights is None:
        sw_ptr = torch.zeros(1, dtype=torch.float32, device=hidden_states.device)
    else:
        sw_ptr = sorted_weights

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 256
    GROUP_SIZE_M = 1
    NUM_WARPS = 4
    num_pid_m = triton.cdiv(EM, BLOCK_M)
    num_pid_n = triton.cdiv(I_r, BLOCK_N)
    grid = (num_pid_m * num_pid_n,)

    stride_am = hidden_states.stride(0)
    stride_ak = hidden_states.stride(1)
    stride_be = w1.stride(0)
    stride_bn = w1.stride(1)
    stride_bk = w1.stride(2)
    stride_cm = out_2d.stride(0)
    stride_cn = out_2d.stride(1)
    stride_ase_m = a1_scale.stride(0)
    stride_ase_k = a1_scale.stride(1)
    stride_bse_e = w1_scale.stride(0)
    stride_bse_n = w1_scale.stride(1)
    stride_bse_k = w1_scale.stride(2)
    stride_se_n_pad = a1_scale.shape[1]
    K_packed_total = K // 2

    # Step 4: launch the GEMM. One CTA per (M-tile, N-tile); the per-CTA
    # expert id is ``sorted_expert_ids[pid_m]``. Each CTA produces
    # BLOCK_M rows x BLOCK_N cols of the gate-up fused intermediate.
    gluon_mxfp4_moe_stage1_kernel[grid](
        hidden_states,
        w1,
        out_2d,
        a1_scale,
        w1_scale,
        sorted_token_ids,
        sorted_expert_ids,
        num_valid_ids_ptr,
        sw_ptr,
        N,
        K,
        EM,
        num_tokens,
        topk,
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
        K_PACKED_TOTAL=K_packed_total,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        NUM_WARPS=NUM_WARPS,
        I_r=I_r,
        B_GDOT128=bool(b_gdot128),
        SWIGLU_ALPHA=float(swiglu_alpha),
        SWIGLU_LIMIT=float(swiglu_limit),
        SWIGLU_BETA=float(swiglu_beta),
        num_warps=NUM_WARPS,
    )
    return out
