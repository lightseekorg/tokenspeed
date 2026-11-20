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


"""Gluon MXFP4-weight MoE stage 1 with a fused gated activation.

For each MoE expert ``e``::

    inter_e = silu(hidden_states_e @ w1_e[:I_r, :].T)        # gate
              * (hidden_states_e @ w1_e[I_r:, :].T)          # up

where ``hidden_states_e`` is the subset of input rows routed to
expert ``e``. The kernel computes that for all experts in one launch
by walking the ``sorted_token_ids`` / ``sorted_expert_ids`` arrays
produced by the upstream routing kernel.

Kernel shape:
  BLOCK_M in {32, 64, 128}, BLOCK_N = 128, BLOCK_K = 256, with one
  accumulator group per 32-row quarter. The 32-row variant is limited to
  E4M3 activations. Interleaved W13 weights are consumed as one contiguous
  128-column MFMA tile and split into 64 gate/up outputs in the epilogue.
  K-loop pipelining: 2-slot LDS ping-pong for A data, direct VGPR loads
  for A scales, and 2-slot VGPR ping-pong for B data and B scales;
  B[k+1] is issued at the top of tile k so its VMEM latency hides behind
  tile k's MFMAs. E2M1 tiles load scales a dword per lane: an A dword holds
  both 16-row halves and both K halves of a lane's scale bytes, and a B
  dword is shared by the two warps that own its 16-row halves.

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

gdot128 banks (128 x 128 tiles with gate and up rows interleaved) and N16
banks (``n16_weights.py``: 16-row groups, 16 gate rows then the same 16 up
rows in each 32 rows) are both read in their stored row order. For N16 the
epilogue pairs accumulator column c with column c + 16 in each 32-column
group. Every output is the same dot product summed in the same K order, so
the two layouts give bitwise-equal results.

Layout contract:
  hidden_states  (M_padded, K_packed)        packed MXFP4 or MXFP8
  w1             (E, 2 * I_r, K_packed)      uint8, in the B_LAYOUT layout
  a1_scale       (M_padded_aligned, K // 32) uint8, e8m0
  w1_scale       (E, 2 * I_r, K // 32)       uint8, e8m0 (shuffled)
  out            unsorted BF16 or sorted MXFP4/MXFP8 activation rows
  out_scale      sorted CDNA4-swizzled e8m0 scales for quantized output
  sorted_token_ids    (EM,)             int32; low 24 bits = token_id
  sorted_expert_ids   (EM // BLOCK_M,)  int32; valid blocks only

Stage 1 output is post-activation at ``I_r`` columns (half of the unfused
gate||up width). ``N = 2 * I_r`` is the B-operand column count and is passed
in for indexing only; the C-side is ``I_r``.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import cdna4_async_copy, gl, gluon, triton
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.n16_weights import _n16_weight_offset
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.quantize_gluon import (
    _mxfp4_quantize_tile,
    _mxfp4_store_cdna4_scale,
    _mxfp8_quantize_tile,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.scale_layout import (
    CDNA4_SCALE_K_BLOCK,
    CDNA4_SCALE_N_BLOCK,
)

_NUM_XCDS = 8

# W13 bank layouts (``B_LAYOUT``), numbered as the warp decode's ``W_LAYOUT``:
# the (16, 16) MFMA-tile shuffle above, gdot128 128 x 128 tiles, and N16
# 16-row groups.
B_LAYOUT_MFMA16 = gl.constexpr(0)
B_LAYOUT_GDOT128 = gl.constexpr(1)
B_LAYOUT_N16 = gl.constexpr(2)


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
def _gdot128_gate_up_b_offsets(offs_bk, offs_bn, N_TILES: gl.constexpr):
    """Per-expert byte offsets of K bytes ``offs_bk`` of interleaved W13 rows
    ``offs_bn`` in a gdot128 bank (128 x 128 tiles, K-tile major)."""
    k_in = offs_bk % 128
    k_within = k_in % 16
    k_quad = (k_in // 16) % 4
    k_block = k_in // 64
    n_primary_in = offs_bn % 128
    in_primary = (
        (n_primary_in // 16)[None, :] * 2048
        + k_block[:, None] * 1024
        + k_quad[:, None] * 256
        + (n_primary_in % 16)[None, :] * 16
        + k_within[:, None]
    )
    return ((offs_bk // 128)[:, None] * N_TILES + (offs_bn // 128)[None, :]) * (
        128 * 128
    ) + in_primary


@gluon.jit
def _cdna4_scale_row_part(rows, K_PACKED_TOTAL: gl.constexpr):
    """Row part of a byte offset into CDNA4-swizzled scales: 32-row panels of
    ``2 * K_PACKED_TOTAL`` bytes that pair rows r and r + 16."""
    return (rows // 32) * (K_PACKED_TOTAL * 2) + (rows % 16) * 4 + ((rows % 32) // 16)


@gluon.jit
def _load_a_scale_vgpr(
    a_scales_ptr,
    a_scale_base_offsets,
    GROUP_IDX: gl.constexpr,
    tile_k,
    group_stride,
    A_SCALE_K_STEP: gl.constexpr,
    SCALE_LAYOUT: gl.constexpr,
    DWORD: gl.constexpr,
):
    # Direct-to-VGPR A-scale load in the same CDNA4-swizzled scale contract
    # used by decode. ``a_scale_base_offsets`` is group-0, current pid_m,
    # [GROUP_MFMA_M, BLOCK_K_SCALE] in the native MFMA scale layout. Each
    # successive 32-row group advances the row block by stride_npad * 32.
    if DWORD:
        # ``a_scale_base_offsets`` is [16, 4, 4] bytes instead: lane (r, kk)
        # loads one dword holding rows r and r + 16 for K groups kk and kk + 4.
        scale = gl.amd.cdna4.buffer_load(
            ptr=a_scales_ptr + (GROUP_IDX * group_stride + tile_k * A_SCALE_K_STEP),
            offsets=a_scale_base_offsets,
        )
        scale = gl.permute(gl.reshape(scale, [16, 4, 2, 2]), [3, 0, 2, 1])
        scale = gl.convert_layout(gl.reshape(scale, [32, 8]), SCALE_LAYOUT)
    else:
        scale = gl.amd.cdna4.buffer_load(
            ptr=a_scales_ptr,
            offsets=(
                a_scale_base_offsets
                + GROUP_IDX * group_stride
                + tile_k * A_SCALE_K_STEP
            ),
        )
    return scale


@gluon.jit
def _prefetch_a_data_lds(
    smem_a_tile,
    a_base_ptr,
    offs_a,
    a_data_k_off,
    mask=None,
    USE_ASYNC: gl.constexpr = True,
):
    # A_scale path is sorted-padded so the wave-uniform K-step
    # is enough; the A data path is per-token-gathered, so callers plumb
    # ``mask=token_mask[:, None]`` here to zero-fill padded sorted slots.
    offsets = offs_a + a_data_k_off
    if USE_ASYNC:
        cdna4_async_copy.buffer_load_to_shared(
            smem_a_tile,
            a_base_ptr,
            offsets,
            mask=mask,
        )
    else:
        values = gl.load(a_base_ptr + offsets, mask=mask, other=0)
        # The compiler's LDS dependence analysis orders this store against
        # the previous tile's cross-wave reads and the reads that follow.
        smem_a_tile.store(values)


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
def _load_b_scale_dword(
    b_scales_dword_ptr,
    b_scale_offs,
    b_scale_shift,
    k,
    b_scale_k_step: gl.constexpr,
    SCALE_LAYOUT: gl.constexpr,
):
    # ``b_scale_offs`` is [G, 2, 16, 4] dwords over 32-row group g, row half
    # h, row r and K group kk. Dword (g, r, kk) holds rows r and r + 16 for K
    # groups kk and kk + 4; both halves load it and keep their own bytes,
    # which ``b_scale_shift`` (8 * h) moves to bits 0 and 16.
    scale = gl.amd.cdna4.buffer_load(
        ptr=b_scales_dword_ptr + k * (b_scale_k_step // 4), offsets=b_scale_offs
    )
    scale = scale.to(gl.uint32, bitcast=True) >> b_scale_shift
    scale = gl.join(scale.to(gl.uint8), (scale >> 16).to(gl.uint8))
    scale = gl.permute(scale, [0, 1, 2, 4, 3])
    scale = gl.reshape(scale, [scale.shape[0] * 32, 8])
    return gl.convert_layout(scale, SCALE_LAYOUT)


@gluon.jit
def _load_b_pair_vgpr(
    b_base_ptr,
    b_scales_ptr,
    offs_b_primary,
    offs_b_secondary,
    b_scale_offs_primary,
    b_scale_offs_secondary,
    b_data_k_off,
    tile_k,
    b_scale_k_step: gl.constexpr,
    INTERLEAVED: gl.constexpr,
    b_scale_shift,
    SCALE_LAYOUT: gl.constexpr,
    SCALE_DWORD: gl.constexpr,
):
    primary = _load_b_data_vgpr(b_base_ptr, offs_b_primary, b_data_k_off)
    if INTERLEAVED:
        if SCALE_DWORD:
            primary_scale = _load_b_scale_dword(
                b_scales_ptr,
                b_scale_offs_primary,
                b_scale_shift,
                tile_k,
                b_scale_k_step,
                SCALE_LAYOUT,
            )
        else:
            primary_scale = _load_b_scale_vgpr(
                b_scales_ptr,
                b_scale_offs_primary,
                tile_k,
                b_scale_k_step,
            )
        secondary = primary
        secondary_scale = primary_scale
    else:
        secondary = _load_b_data_vgpr(b_base_ptr, offs_b_secondary, b_data_k_off)
        primary_scale = _load_b_scale_vgpr(
            b_scales_ptr,
            b_scale_offs_primary,
            tile_k,
            b_scale_k_step,
        )
        secondary_scale = _load_b_scale_vgpr(
            b_scales_ptr,
            b_scale_offs_secondary,
            tile_k,
            b_scale_k_step,
        )
    return primary, secondary, primary_scale, secondary_scale


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
def _compute_mx_group(
    cur_a,
    a_scale,
    cur_b,
    b_scale,
    acc,
    A_FORMAT: gl.constexpr,
):
    return gl.amd.cdna4.mfma_scaled(
        a=cur_a,
        a_scale=a_scale,
        a_format=A_FORMAT,
        b=cur_b,
        b_scale=b_scale,
        b_format="e2m1",
        acc=acc,
    )


@gluon.jit
def _compute_gate_up_group(
    cur_a,
    a_scale,
    cur_b_gate,
    b_scale_gate,
    cur_b_up,
    b_scale_up,
    gate_acc,
    up_acc,
    A_FORMAT: gl.constexpr,
    INTERLEAVED: gl.constexpr,
):
    gate_acc = _compute_mx_group(
        cur_a,
        a_scale,
        cur_b_gate,
        b_scale_gate,
        gate_acc,
        A_FORMAT,
    )
    if not INTERLEAVED:
        up_acc = _compute_mx_group(
            cur_a,
            a_scale,
            cur_b_up,
            b_scale_up,
            up_acc,
            A_FORMAT,
        )
    return gate_acc, up_acc


@gluon.jit
def _apply_gate_activation(
    gate,
    up,
    DO_SITU: gl.constexpr,
    ALPHA: gl.constexpr,
    LIMIT: gl.constexpr,
    BETA: gl.constexpr,
    LINEAR_BETA: gl.constexpr,
):
    if DO_SITU:
        gate = gate.to(gl.bfloat16).to(gl.float32)
        up = up.to(gl.bfloat16).to(gl.float32)
        gate = ALPHA * gl.extra.libdevice.tanh(gate / ALPHA) / (1.0 + gl.exp(-gate))
        up = LINEAR_BETA * gl.extra.libdevice.tanh(up / LINEAR_BETA)
        return gate * up
    if LIMIT > 0.0:
        gate = gl.minimum(gate, LIMIT)
        up = gl.minimum(gl.maximum(up, -LIMIT), LIMIT)
    silu = gate / (1.0 + gl.exp(-(ALPHA * gate)))
    return gl.fma(silu, up, silu * BETA)


@gluon.jit
def _apply_interleaved_gate_activation(
    acc,
    DO_SITU: gl.constexpr,
    ALPHA: gl.constexpr,
    LIMIT: gl.constexpr,
    BETA: gl.constexpr,
    LINEAR_BETA: gl.constexpr,
    OUT_BLOCK_N: gl.constexpr,
    N16_HALVES: gl.constexpr = False,
):
    block_m: gl.constexpr = acc.shape[0]
    split_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[4, 16],
        warps_per_cta=[gl.num_warps(), 1],
        order=[1, 0],
    )
    acc = gl.convert_layout(acc, split_layout)
    if N16_HALVES:
        # N16 columns come as 16 gate rows, then the same 16 up rows.
        acc = gl.permute(acc.reshape((block_m, OUT_BLOCK_N // 16, 2, 16)), [0, 1, 3, 2])
        pair_layout: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[1, 1, 4, 2],
            threads_per_warp=[4, 4, 4, 1],
            warps_per_cta=[gl.num_warps(), 1, 1, 1],
            order=[3, 2, 1, 0],
        )
        acc = gl.convert_layout(acc, pair_layout)
    gate, up = gl.split(acc.reshape((block_m, OUT_BLOCK_N, 2)))
    return _apply_gate_activation(
        gate,
        up,
        DO_SITU,
        ALPHA,
        LIMIT,
        BETA,
        LINEAR_BETA,
    )


@gluon.jit
def _store_swiglu_tile_group(
    acc_swiglu,
    c_ptr,
    c_scale_ptr,
    sorted_token_ids_ptr,
    pid_m,
    pid_n,
    EM,
    num_tokens,
    top_k,
    stride_cm,
    stride_cn,
    GROUP_IDX: gl.constexpr,
    GROUP_M: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    I_r: gl.constexpr,
    OUTPUT_K: gl.constexpr,
    OUTPUT_SORTED: gl.constexpr,
    OUTPUT_QUANTIZED: gl.constexpr,
    OUTPUT_FP8: gl.constexpr,
):
    output_layout: gl.constexpr = acc_swiglu.type.layout
    cm_layout: gl.constexpr = gl.SliceLayout(1, output_layout)
    cn_layout: gl.constexpr = gl.SliceLayout(0, output_layout)
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
    if OUTPUT_SORTED:
        dst_row_cm = offs_cm
    else:
        dst_row_cm = token_id_cm * top_k + topk_id_cm
    token_mask_cm = (offs_cm < EM) if OUTPUT_SORTED else (token_id_cm < num_tokens)
    if OUTPUT_QUANTIZED:
        if OUTPUT_FP8:
            quantized, scale_byte = _mxfp8_quantize_tile(acc_swiglu)
            quantized = quantized.reshape((GROUP_M, BLOCK_N))
            quantized_cols_per_tile: gl.constexpr = BLOCK_N
            output_cols: gl.constexpr = I_r
        else:
            quantized, scale_byte = _mxfp4_quantize_tile(acc_swiglu)
            quantized = quantized.reshape((GROUP_M, BLOCK_N // 2))
            quantized_cols_per_tile: gl.constexpr = BLOCK_N // 2
            output_cols: gl.constexpr = I_r // 2
        quantized_layout: gl.constexpr = quantized.type.layout
        quantized_m = gl.arange(0, GROUP_M, gl.SliceLayout(1, quantized_layout))
        quantized_n = gl.arange(
            0,
            quantized_cols_per_tile,
            gl.SliceLayout(0, quantized_layout),
        )
        quantized_rows = pid_m * BLOCK_M + GROUP_IDX * GROUP_M + quantized_m
        quantized_cols = pid_n * quantized_cols_per_tile + quantized_n
        quantized_ptrs = (
            c_ptr
            + quantized_rows[:, None].to(gl.int64) * stride_cm
            + quantized_cols[None, :].to(gl.int64) * stride_cn
        )
        quantized_mask = (quantized_rows[:, None] < EM) & (
            quantized_cols[None, :] < output_cols
        )
        gl.store(quantized_ptrs, quantized, mask=quantized_mask)

        scale_layout: gl.constexpr = scale_byte.type.layout
        scale_m = (
            pid_m * BLOCK_M
            + GROUP_IDX * GROUP_M
            + gl.arange(0, GROUP_M, gl.SliceLayout(1, scale_layout))
        )
        scale_k = pid_n * (BLOCK_N // 32) + gl.arange(
            0, BLOCK_N // 32, gl.SliceLayout(0, scale_layout)
        )
        _mxfp4_store_cdna4_scale(
            c_scale_ptr,
            scale_byte,
            scale_m[:, None],
            scale_k[None, :],
            1,
            (OUTPUT_K // 32) * 32,
            (scale_m[:, None] < EM) & (scale_k[None, :] < I_r // 32),
            M_SWIZZLE=32,
            K_SWIZZLE=8,
        )
        if OUTPUT_K > I_r and pid_n == gl.cdiv(I_r, BLOCK_N) - 1:
            # W2 alignment can require more than one logical output tile of
            # padding (for example, I=384 padded to K=512 with a 64-wide W13
            # tile). The final active CTA initializes every padded tile.
            num_tail_tiles: gl.constexpr = gl.cdiv(OUTPUT_K - I_r, BLOCK_N)
            for tail_tile in range(0, num_tail_tiles):
                if OUTPUT_FP8:
                    tail_quantized_cols = I_r + tail_tile * BLOCK_N + quantized_n
                    tail_output_cols: gl.constexpr = OUTPUT_K
                else:
                    tail_quantized_cols = (
                        I_r // 2 + tail_tile * (BLOCK_N // 2) + quantized_n
                    )
                    tail_output_cols: gl.constexpr = OUTPUT_K // 2
                tail_quantized_ptrs = (
                    c_ptr
                    + quantized_rows[:, None].to(gl.int64) * stride_cm
                    + tail_quantized_cols[None, :].to(gl.int64) * stride_cn
                )
                tail_quantized_mask = (quantized_rows[:, None] < EM) & (
                    tail_quantized_cols[None, :] < tail_output_cols
                )
                gl.store(
                    tail_quantized_ptrs,
                    gl.zeros_like(quantized),
                    mask=tail_quantized_mask,
                )

                tail_scale_k = (
                    I_r // 32
                    + tail_tile * (BLOCK_N // 32)
                    + gl.arange(
                        0,
                        BLOCK_N // 32,
                        gl.SliceLayout(0, scale_layout),
                    )
                )
                _mxfp4_store_cdna4_scale(
                    c_scale_ptr,
                    gl.full(
                        scale_byte.shape,
                        127,
                        gl.uint8,
                        layout=scale_byte.type.layout,
                    ),
                    scale_m[:, None],
                    tail_scale_k[None, :],
                    1,
                    (OUTPUT_K // 32) * 32,
                    (scale_m[:, None] < EM) & (tail_scale_k[None, :] < OUTPUT_K // 32),
                    M_SWIZZLE=32,
                    K_SWIZZLE=8,
                )
    else:
        c_val = acc_swiglu.to(c_ptr.type.element_ty)
        c_ptrs = (
            c_ptr
            + dst_row_cm[:, None].to(gl.int64) * stride_cm
            + offs_cn[None, :].to(gl.int64) * stride_cn
        )
        c_mask = token_mask_cm[:, None] & (offs_cn[None, :] < I_r)
        gl.store(c_ptrs, c_val, mask=c_mask)


@gluon.jit
def gluon_mxfp4_moe_stage1_e4m3_async_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    c_scale_ptr,
    a_scales_ptr,
    b_scales_ptr,
    sorted_token_ids_ptr,
    sorted_expert_ids_ptr,
    num_tokens_post_padded_ptr,
    EM,
    num_tokens,
    top_k,
    stride_be,
    stride_cm,
    stride_cn,
    stride_bse_e,
    stride_se_n_pad,
    STRIDE_AM: gl.constexpr,
    BLOCK_M: gl.constexpr,
    K_PACKED_TOTAL: gl.constexpr,
    I_r: gl.constexpr,
    OUTPUT_K: gl.constexpr,
    SWIGLU_ALPHA: gl.constexpr,
    SWIGLU_LIMIT: gl.constexpr,
    SWIGLU_BETA: gl.constexpr,
    SITU_LINEAR_BETA: gl.constexpr,
):
    """Stage 1 with four-wave async FP8 activation staging.

    Each logical K=256 tile is split into two K=128 MFMA steps. Each 32-row
    slice then moves aligned 16-byte lane transactions, the native CDNA4
    async-copy width, while preserving the GDOT128 interleaved weight and
    sorted MXFP8 output contracts.
    """

    BLOCK_N: gl.constexpr = 128
    OUTPUT_BLOCK_N: gl.constexpr = 64
    BLOCK_K_HALF: gl.constexpr = 128
    BLOCK_K_PACKED_HALF: gl.constexpr = 64
    BLOCK_K_SCALE_HALF: gl.constexpr = 4
    NUM_BUFFERS: gl.constexpr = 2
    NUM_K_TILES: gl.constexpr = K_PACKED_TOTAL // 128

    gl.static_assert(
        BLOCK_M == 32 or BLOCK_M == 64 or BLOCK_M == 128,
        "async stage1 requires BLOCK_M in {32, 64, 128}",
    )
    gl.static_assert(K_PACKED_TOTAL % 128 == 0)
    gl.static_assert(I_r % OUTPUT_BLOCK_N == 0)

    pid = gl.program_id(axis=0)
    num_pid_m = gl.cdiv(EM, BLOCK_M)
    num_pid_n: gl.constexpr = I_r // OUTPUT_BLOCK_N
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    num_tokens_post_padded = gl.load(num_tokens_post_padded_ptr)
    if pid_n >= num_pid_n or pid_m * BLOCK_M >= num_tokens_post_padded:
        return

    expert = gl.load(sorted_expert_ids_ptr + pid_m)

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
        dot_a_layout, [BLOCK_M, BLOCK_K_SCALE_HALF]
    )
    b_scale_layout: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(
        dot_b_layout, [BLOCK_N, BLOCK_K_SCALE_HALF]
    )

    # This distribution gives every lane one aligned 128-bit copy per 32-row
    # slice. Two independent half-tile buffers avoid slicing the shared
    # descriptor, which the CDNA4 direct-copy lowering cannot legalize.
    gload_a_layout: gl.constexpr = gl.BlockedLayout(
        [1, 16],
        [8, 8],
        [4, 1],
        [1, 0],
    )
    shared_a_layout: gl.constexpr = gl.SwizzledSharedLayout(16, 1, 16, order=[1, 0])
    smem_a_lo = gl.allocate_shared_memory(
        a_ptr.type.element_ty,
        [NUM_BUFFERS, BLOCK_M, BLOCK_K_HALF],
        shared_a_layout,
    )
    smem_a_hi = gl.allocate_shared_memory(
        a_ptr.type.element_ty,
        [NUM_BUFFERS, BLOCK_M, BLOCK_K_HALF],
        shared_a_layout,
    )

    m_load_layout: gl.constexpr = gl.SliceLayout(1, gload_a_layout)
    k_load_layout: gl.constexpr = gl.SliceLayout(0, gload_a_layout)
    tile_row = gl.arange(0, BLOCK_M, layout=m_load_layout)
    sorted_row = pid_m * BLOCK_M + tile_row
    packed_route = gl.load(
        sorted_token_ids_ptr + sorted_row,
        mask=sorted_row < EM,
        other=num_tokens,
    )
    token_id = packed_route & 0xFFFFFF
    token_mask = token_id < num_tokens
    a_k = gl.arange(0, BLOCK_K_HALF, layout=k_load_layout)
    # Keep the direct-copy offset within one M tile. The scalar base retains
    # the full 64-bit routed-row address while the buffer instruction gets the
    # compact 32-bit offset it requires.
    a_base = a_ptr + (pid_m * BLOCK_M).to(gl.int64) * STRIDE_AM
    a_offsets_lo = tile_row[:, None] * STRIDE_AM + a_k[None, :]
    a_offsets_hi = a_offsets_lo + BLOCK_K_HALF

    m_scale_layout: gl.constexpr = gl.SliceLayout(1, a_scale_layout)
    k_scale_layout: gl.constexpr = gl.SliceLayout(0, a_scale_layout)
    a_scale_rows = (pid_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=m_scale_layout)).to(
        gl.uint32
    )
    a_scale_m_part = (
        (a_scale_rows // 32) * (stride_se_n_pad * 32)
        + (a_scale_rows % 16) * 4
        + ((a_scale_rows % 32) // 16)
    )[:, None]
    a_scale_k = gl.arange(0, BLOCK_K_SCALE_HALF, layout=k_scale_layout).to(gl.uint32)
    a_scale_lo_offsets = a_scale_m_part + (a_scale_k % 4)[None, :] * 64
    a_scale_hi_offsets = a_scale_lo_offsets + 2

    b_k_layout: gl.constexpr = gl.SliceLayout(1, dot_b_layout)
    b_n_layout: gl.constexpr = gl.SliceLayout(0, dot_b_layout)
    b_k = gl.arange(0, BLOCK_K_PACKED_HALF, layout=b_k_layout).to(gl.uint32)
    b_n = (pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=b_n_layout)).to(gl.uint32)
    b_n_in = b_n % 128
    b_n_tiles: gl.constexpr = (2 * I_r) // 128
    b_in_lo = (
        (b_n_in // 16)[None, :] * 2048
        + (b_n_in % 16)[None, :] * 16
        + (b_k % 16)[:, None]
        + ((b_k // 16) % 4)[:, None] * 256
    )
    b_in_hi = b_in_lo + 1024
    b_tile_base = (b_n // 128)[None, :] * (128 * 128)
    b_offsets_lo = b_tile_base + b_in_lo
    b_offsets_hi = b_tile_base + b_in_hi
    B_DATA_K_STEP: gl.constexpr = b_n_tiles * (128 * 128)

    b_scale_n_layout: gl.constexpr = gl.SliceLayout(1, b_scale_layout)
    b_scale_k_layout: gl.constexpr = gl.SliceLayout(0, b_scale_layout)
    b_scale_rows = (
        pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=b_scale_n_layout)
    ).to(gl.uint32)
    b_scale_n_part = (
        (b_scale_rows // 32) * (K_PACKED_TOTAL * 2)
        + (b_scale_rows % 16) * 4
        + ((b_scale_rows % 32) // 16)
    )[:, None]
    b_scale_k = gl.arange(0, BLOCK_K_SCALE_HALF, layout=b_scale_k_layout).to(gl.uint32)
    b_scale_lo_offsets = b_scale_n_part + (b_scale_k % 4)[None, :] * 64
    b_scale_hi_offsets = b_scale_lo_offsets + 2

    b_base = b_ptr + expert.to(gl.int64) * stride_be
    b_scale_base = b_scales_ptr + expert.to(gl.int64) * stride_bse_e
    acc = gl.zeros((BLOCK_M, BLOCK_N), gl.float32, mfma_layout)

    cdna4_async_copy.buffer_load_to_shared(
        smem_a_lo.index(0),
        a_base,
        a_offsets_lo,
        mask=token_mask[:, None],
    )
    cdna4_async_copy.buffer_load_to_shared(
        smem_a_hi.index(0),
        a_base,
        a_offsets_hi,
        mask=token_mask[:, None],
    )
    cdna4_async_copy.commit_group()

    for tile_k in gl.static_range(0, NUM_K_TILES - 1):
        load_slot = tile_k % NUM_BUFFERS
        next_slot = (tile_k + 1) % NUM_BUFFERS
        next_a_offset = (tile_k + 1) * 256
        cdna4_async_copy.buffer_load_to_shared(
            smem_a_lo.index(next_slot),
            a_base,
            a_offsets_lo + next_a_offset,
            mask=token_mask[:, None],
        )
        cdna4_async_copy.buffer_load_to_shared(
            smem_a_hi.index(next_slot),
            a_base,
            a_offsets_hi + next_a_offset,
            mask=token_mask[:, None],
        )
        cdna4_async_copy.commit_group()

        b_tile_offset = tile_k * B_DATA_K_STEP
        scale_tile_offset = tile_k * 256
        b_lo = gl.amd.cdna4.buffer_load(
            ptr=b_base,
            offsets=b_offsets_lo + b_tile_offset,
        )
        a_scale_lo = gl.amd.cdna4.buffer_load(
            ptr=a_scales_ptr,
            offsets=a_scale_lo_offsets + scale_tile_offset,
        )
        b_scale_lo = gl.amd.cdna4.buffer_load(
            ptr=b_scale_base,
            offsets=b_scale_lo_offsets + scale_tile_offset,
        )
        cdna4_async_copy.wait_group(1)
        # The copy wait is wave-local; the compiler emits the CTA barrier
        # that publishes every wave's LDS writes right after the wait.
        a_lo = cdna4_async_copy.load_shared_relaxed(
            smem_a_lo.index(load_slot), dot_a_layout
        )
        acc = gl.amd.cdna4.mfma_scaled(
            a=a_lo,
            a_scale=a_scale_lo,
            a_format="e4m3",
            b=b_lo,
            b_scale=b_scale_lo,
            b_format="e2m1",
            acc=acc,
        )

        b_hi = gl.amd.cdna4.buffer_load(
            ptr=b_base,
            offsets=b_offsets_hi + b_tile_offset,
        )
        a_scale_hi = gl.amd.cdna4.buffer_load(
            ptr=a_scales_ptr,
            offsets=a_scale_hi_offsets + scale_tile_offset,
        )
        b_scale_hi = gl.amd.cdna4.buffer_load(
            ptr=b_scale_base,
            offsets=b_scale_hi_offsets + scale_tile_offset,
        )
        a_hi = cdna4_async_copy.load_shared_relaxed(
            smem_a_hi.index(load_slot), dot_a_layout
        )
        acc = gl.amd.cdna4.mfma_scaled(
            a=a_hi,
            a_scale=a_scale_hi,
            a_format="e4m3",
            b=b_hi,
            b_scale=b_scale_hi,
            b_format="e2m1",
            acc=acc,
        )

    last_tile: gl.constexpr = NUM_K_TILES - 1
    last_slot: gl.constexpr = last_tile % NUM_BUFFERS
    last_b_offset: gl.constexpr = last_tile * B_DATA_K_STEP
    last_scale_offset: gl.constexpr = last_tile * 256
    b_lo = gl.amd.cdna4.buffer_load(
        ptr=b_base,
        offsets=b_offsets_lo + last_b_offset,
    )
    a_scale_lo = gl.amd.cdna4.buffer_load(
        ptr=a_scales_ptr,
        offsets=a_scale_lo_offsets + last_scale_offset,
    )
    b_scale_lo = gl.amd.cdna4.buffer_load(
        ptr=b_scale_base,
        offsets=b_scale_lo_offsets + last_scale_offset,
    )
    cdna4_async_copy.wait_group(0)
    a_lo = cdna4_async_copy.load_shared_relaxed(
        smem_a_lo.index(last_slot), dot_a_layout
    )
    acc = gl.amd.cdna4.mfma_scaled(
        a=a_lo,
        a_scale=a_scale_lo,
        a_format="e4m3",
        b=b_lo,
        b_scale=b_scale_lo,
        b_format="e2m1",
        acc=acc,
    )
    b_hi = gl.amd.cdna4.buffer_load(
        ptr=b_base,
        offsets=b_offsets_hi + last_b_offset,
    )
    a_scale_hi = gl.amd.cdna4.buffer_load(
        ptr=a_scales_ptr,
        offsets=a_scale_hi_offsets + last_scale_offset,
    )
    b_scale_hi = gl.amd.cdna4.buffer_load(
        ptr=b_scale_base,
        offsets=b_scale_hi_offsets + last_scale_offset,
    )
    a_hi = cdna4_async_copy.load_shared_relaxed(
        smem_a_hi.index(last_slot), dot_a_layout
    )
    acc = gl.amd.cdna4.mfma_scaled(
        a=a_hi,
        a_scale=a_scale_hi,
        a_format="e4m3",
        b=b_hi,
        b_scale=b_scale_hi,
        b_format="e2m1",
        acc=acc,
    )

    activated = _apply_interleaved_gate_activation(
        acc,
        True,
        SWIGLU_ALPHA,
        SWIGLU_LIMIT,
        SWIGLU_BETA,
        SITU_LINEAR_BETA,
        OUTPUT_BLOCK_N,
    )
    _store_swiglu_tile_group(
        activated,
        c_ptr,
        c_scale_ptr,
        sorted_token_ids_ptr,
        pid_m,
        pid_n,
        EM,
        num_tokens,
        top_k,
        stride_cm,
        stride_cn,
        0,
        BLOCK_M,
        BLOCK_M,
        OUTPUT_BLOCK_N,
        I_r,
        OUTPUT_K,
        True,
        True,
        True,
    )


# The token count follows the batch and only bounds the route masks; without
# do_not_specialize, Triton compiles separate binaries for counts that are and
# are not multiples of 16.
@gluon.jit(do_not_specialize=["num_tokens"])
def gluon_mxfp4_moe_stage1_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    c_scale_ptr,
    a_scales_ptr,
    b_scales_ptr,
    sorted_token_ids_ptr,
    sorted_expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N,
    K,
    EM,
    num_tokens,
    top_k,
    stride_am,
    stride_ak,
    stride_be,
    stride_cm,
    stride_cn,
    stride_bse_e,
    stride_se_n_pad,
    K_PACKED_TOTAL: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    I_r: gl.constexpr,
    OUTPUT_K: gl.constexpr,
    B_LAYOUT: gl.constexpr,
    SWIGLU_ALPHA: gl.constexpr,
    SWIGLU_LIMIT: gl.constexpr,
    SWIGLU_BETA: gl.constexpr,
    DO_SITU: gl.constexpr,
    SITU_LINEAR_BETA: gl.constexpr,
    OUTPUT_SORTED: gl.constexpr,
    OUTPUT_QUANTIZED: gl.constexpr,
    A_FORMAT: gl.constexpr,
    OUTPUT_FP8: gl.constexpr,
    INPUT_SORTED: gl.constexpr,
    NUM_XCDS: gl.constexpr = 1,
    EARLY_A: gl.constexpr = False,
):
    """Stage 1 kernel with per-token A gather and fused SwiGLU.

    Tile config: ``BLOCK_M x 128 x 256`` (M x N x K), with ``BLOCK_M``
    in ``{16, 32, 64, 128}``; 16-row tiles use E2M1 activations. MFMA target:
    ``v_mfma_scale_f32_16x16x128_f8f6f4`` (gfx950, AMDMFMALayout
    version=4). Each CTA owns ``BLOCK_N / 2`` output columns for interleaved
    W13 (gdot128 or N16 banks) or ``BLOCK_N`` columns for the separate gate/up
    layout.

    For interleaved W13, each K tile consumes the interleaved gate/up rows in
    one ``BLOCK_N`` MFMA accumulator per 32-row quarter. The separate layout
    retains separate gate and up accumulators. The K loop alternates
    two A-data LDS slots while loading A scales and B operands into VGPRs.

    The epilogue splits interleaved accumulators into gate/up pairs, applies
    the activation, and stores the resulting output tile in ``[EM, I_r]``.
    """

    gl.static_assert(
        BLOCK_M == 32
        or BLOCK_M == 64
        or BLOCK_M == 128
        or (BLOCK_M == 16 and A_FORMAT == "e2m1"),
        "stage1 kernel requires BLOCK_M in {32, 64, 128}, or 16 for E2M1",
    )
    gl.static_assert(BLOCK_N == 128, "stage1 kernel requires BLOCK_N=128")
    gl.static_assert(BLOCK_K == 256, "stage1 kernel requires BLOCK_K=256")
    gl.static_assert(
        NUM_WARPS == max(1, BLOCK_M // 32)
        or (A_FORMAT == "e4m3" and NUM_WARPS == 8)
        or (A_FORMAT == "e2m1" and (BLOCK_M == 32 or BLOCK_M == 64) and NUM_WARPS == 4),
        "stage1 requires native E2M1 subgroups, four for BM32/BM64, or eight for E4M3",
    )
    gl.static_assert(not OUTPUT_QUANTIZED or OUTPUT_SORTED)

    SCALE_GROUP: gl.constexpr = 32
    A_DIV: gl.constexpr = 1 if A_FORMAT == "e4m3" else 2
    BLOCK_K_A: gl.constexpr = BLOCK_K // A_DIV
    BLOCK_K_B: gl.constexpr = BLOCK_K // 2
    BLOCK_K_SCALE: gl.constexpr = BLOCK_K // SCALE_GROUP
    NUM_BUFFERS: gl.constexpr = 2
    GROUP_MFMA_M: gl.constexpr = min(BLOCK_M, 32)

    pid = gl.program_id(axis=0)
    num_pid_m = gl.cdiv(EM, BLOCK_M)
    # gdot128 and N16 banks both hold gate and up rows in one stream; the
    # epilogue pairs them for each layout.
    INTERLEAVED: gl.constexpr = B_LAYOUT != B_LAYOUT_MFMA16
    OUTPUT_BLOCK_N: gl.constexpr = BLOCK_N // 2 if INTERLEAVED else BLOCK_N
    # Grid over the activated ``I_r`` columns. Interleaved W13 consumes a
    # physical BLOCK_N tile for each OUTPUT_BLOCK_N result tile; separate
    # layouts issue gate and up MFMAs for the same output tile.
    num_pid_n = gl.cdiv(I_r, OUTPUT_BLOCK_N)
    # Keep an expert's tiles on one XCD for L2 reuse; only tiles with routes are spread.
    if NUM_XCDS != 1:
        valid_pid_m = gl.cdiv(gl.load(num_tokens_post_padded_ptr), BLOCK_M)
        grid_mn = gl.minimum(valid_pid_m, num_pid_m) * num_pid_n
        if pid < grid_mn:
            pids_per_xcd = (grid_mn + NUM_XCDS - 1) // NUM_XCDS
            tall_xcds = grid_mn % NUM_XCDS
            tall_xcds = gl.where(tall_xcds == 0, NUM_XCDS, tall_xcds)
            xcd = pid % NUM_XCDS
            local_pid = pid // NUM_XCDS
            pid = gl.where(
                xcd < tall_xcds,
                xcd * pids_per_xcd + local_pid,
                tall_xcds * pids_per_xcd
                + (xcd - tall_xcds) * (pids_per_xcd - 1)
                + local_pid,
            )
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = gl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_M >= num_tokens_post_padded:
        return

    mfma_warps: gl.constexpr = (
        [2, NUM_WARPS // 2] if A_FORMAT == "e4m3" else [1, NUM_WARPS]
    )
    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 128],
        transposed=True,
        warps_per_cta=mfma_warps,
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
        [NUM_WARPS, 1],
        [1, 0],
    )
    if BLOCK_M == 16:
        shared_a_bases: gl.constexpr = [
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
        ]
    elif BLOCK_M == 32:
        if A_FORMAT == "e4m3":
            shared_a_bases: gl.constexpr = [
                [0, 1],
                [0, 2],
                [0, 4],
                [0, 8],
                [0, 16],
                [0, 32],
                [0, 64],
                [0, 128],
                [1, 0],
                [2, 0],
                [4, 0],
                [8, 0],
                [16, 0],
            ]
        else:
            shared_a_bases: gl.constexpr = [
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
            ]
    elif BLOCK_M == 64:
        if A_FORMAT == "e4m3":
            shared_a_bases: gl.constexpr = [
                [0, 1],
                [0, 2],
                [0, 4],
                [0, 8],
                [0, 16],
                [0, 32],
                [0, 64],
                [0, 128],
                [1, 0],
                [2, 0],
                [4, 0],
                [8, 0],
                [16, 0],
                [32, 0],
            ]
        else:
            shared_a_bases: gl.constexpr = [
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
            ]
    else:
        if A_FORMAT == "e4m3":
            shared_a_bases: gl.constexpr = [
                [0, 1],
                [0, 2],
                [0, 4],
                [0, 8],
                [0, 16],
                [0, 32],
                [0, 64],
                [0, 128],
                [1, 0],
                [2, 0],
                [4, 0],
                [8, 0],
                [16, 0],
                [32, 0],
                [64, 0],
            ]
        else:
            shared_a_bases: gl.constexpr = [
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
            ]
    shared_a_padding: gl.constexpr = (
        [[1024, 32]] if A_FORMAT == "e4m3" else [[4096, 128]]
    )
    shared_a: gl.constexpr = gl.PaddedSharedLayout(
        shared_a_padding,
        shared_a_bases,
        [],
        [BLOCK_M, BLOCK_K_A],
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
    if INPUT_SORTED:
        a_row = offs_token_id
    else:
        a_row = offs_token & 0xFFFFFF

    # The valid-count guard above excludes every unwritten tail block, so the
    # sort contract guarantees a valid expert for each program reaching here.
    off_experts = gl.load(sorted_expert_ids_ptr + pid_m)

    # ---- A data-load offsets (consumed by per-tile prefetch) ----------
    offs_ak = gl.arange(0, BLOCK_K_A, layout=k_layout)
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
    # E2M1 tiles share A across warps, so a lane can load the dword holding
    # both 16-row halves and both K-group halves of its scale bytes.
    A_SCALE_DWORD: gl.constexpr = A_FORMAT == "e2m1" and GROUP_MFMA_M == 32
    if A_SCALE_DWORD:
        a_dword_layout: gl.constexpr = gl.BlockedLayout(
            [1, 1, 4], [16, 4, 1], [1, 1, NUM_WARPS], [2, 0, 1]
        )
        a_dword_r = gl.arange(
            0, 16, layout=gl.SliceLayout(1, gl.SliceLayout(2, a_dword_layout))
        )
        a_dword_k = gl.arange(
            0, 4, layout=gl.SliceLayout(0, gl.SliceLayout(2, a_dword_layout))
        )
        a_dword_b = gl.arange(
            0, 4, layout=gl.SliceLayout(0, gl.SliceLayout(1, a_dword_layout))
        )
        a_scale_base_offsets = (
            ((pid_m * BLOCK_M) // 32) * (stride_se_n_pad * 32)
            + a_dword_r[:, None, None] * 4
            + a_dword_k[None, :, None] * 64
            + a_dword_b[None, None, :]
        )
    a_scale_group_stride = stride_se_n_pad * 32

    # ---- B data offsets (preshuffle-B closed form) --------------------
    # B is in the (16, 16) MFMA-tile layout described in the module
    # docstring, so the byte for logical ``(n, k_pk)`` is the closed form
    # computed below (the per-expert term is folded into ``b_base_ptr``).
    # Interleaved banks (gdot128, N16) hold gate/up rows as one contiguous
    # ``BLOCK_N`` stream; the MFMA16 layout retains separate gate and up
    # streams. The ``% N`` wraps below defend the degenerate ``I_r < BLOCK_N``
    # test path.
    offs_bk = gl.arange(0, BLOCK_K_B, layout=gl.SliceLayout(1, dot_b_layout))
    if INTERLEAVED:
        offs_bn_primary = (
            pid_n * BLOCK_N
            + gl.arange(
                0,
                BLOCK_N,
                layout=gl.SliceLayout(0, dot_b_layout),
            )
        ) % N
        if B_LAYOUT == B_LAYOUT_N16:
            offs_b_gate = _n16_weight_offset(
                offs_bn_primary[None, :],
                offs_bk[:, None],
                K_PACKED_TOTAL,
            )
        else:
            offs_b_gate = _gdot128_gate_up_b_offsets(
                offs_bk, offs_bn_primary, (2 * I_r) // 128
            )
        offs_b_up = offs_b_gate
    else:
        offs_bn_gate = (
            pid_n * BLOCK_N
            + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, dot_b_layout))
        ) % N
        offs_bn_up = (
            (pid_n + num_pid_n) * BLOCK_N
            + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, dot_b_layout))
        ) % N
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

    # ---- B_scale offsets (e8m0_shuffle_opsel_b) -----------------------
    # Same row-part decomposition as the prior preshuffle-B port; the
    # cross-tile step is the constexpr ``B_SCALE_K_STEP = 1024`` because
    # ``BLOCK_K_SCALE = 8`` covers a full ``y/8`` block per K iter. The
    # per-expert offset folds into ``b_scale_offsets_e_*`` here so the
    # K-loop only carries the K-step.
    b_n_layout_scale: gl.constexpr = gl.SliceLayout(1, b_scale_layout)
    b_k_layout_scale: gl.constexpr = gl.SliceLayout(0, b_scale_layout)
    if INTERLEAVED:
        b_rows_primary = (
            pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=b_n_layout_scale)
        ).to(gl.uint32)
        b_n_part_gate_scale = _cdna4_scale_row_part(b_rows_primary, K_PACKED_TOTAL)[
            :, None
        ]
    else:
        b_rows_gate = (
            pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=b_n_layout_scale)
        ).to(gl.uint32)
        b_rows_up = (
            (pid_n + num_pid_n) * BLOCK_N
            + gl.arange(0, BLOCK_N, layout=b_n_layout_scale)
        ).to(gl.uint32)
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
    if INTERLEAVED:
        b_k_lane_offsets = ((b_k_lanes % 4) * 64 + ((b_k_lanes % 8) // 4) * 2)[None, :]
        B_SCALE_K_STEP: gl.constexpr = 256
    else:
        b_k_lane_offsets = ((b_k_lanes % 4) * 256 + (b_k_lanes // 4))[None, :]
        B_SCALE_K_STEP: gl.constexpr = 1024
    b_scale_static_offs_gate = b_n_part_gate_scale + b_k_lane_offsets
    b_off_gate = b_scale_static_offs_gate
    if INTERLEAVED:
        b_off_up = b_off_gate
    else:
        b_scale_static_offs_up = b_n_part_up_scale + b_k_lane_offsets
        b_off_up = b_scale_static_offs_up

    b_scales_ptr_e = b_scales_ptr + off_experts.to(gl.int64) * stride_bse_e
    # Each E2M1 warp owns one 16-row half of every 32-row B-scale group, so
    # warp pairs load the same dword and each keeps its half.
    B_SCALE_DWORD: gl.constexpr = (
        (B_LAYOUT == B_LAYOUT_GDOT128 or B_LAYOUT == B_LAYOUT_N16)
        and A_FORMAT == "e2m1"
    ) and NUM_WARPS >= 2
    b_scale_shift = 0
    if B_SCALE_DWORD:
        b_dword_layout: gl.constexpr = gl.BlockedLayout(
            [1, 1, 1, 1], [1, 1, 16, 4], [NUM_WARPS // 2, 2, 1, 1], [2, 3, 1, 0]
        )
        b_dword_g = gl.arange(
            0,
            BLOCK_N // 32,
            layout=gl.SliceLayout(
                1, gl.SliceLayout(2, gl.SliceLayout(3, b_dword_layout))
            ),
        )
        b_dword_h = gl.arange(
            0,
            2,
            layout=gl.SliceLayout(
                0, gl.SliceLayout(2, gl.SliceLayout(3, b_dword_layout))
            ),
        )
        b_dword_r = gl.arange(
            0,
            16,
            layout=gl.SliceLayout(
                0, gl.SliceLayout(1, gl.SliceLayout(3, b_dword_layout))
            ),
        )
        b_dword_k = gl.arange(
            0,
            4,
            layout=gl.SliceLayout(
                0, gl.SliceLayout(1, gl.SliceLayout(2, b_dword_layout))
            ),
        )
        b_off_gate = (
            ((pid_n * BLOCK_N) // 32 + b_dword_g)[:, None, None, None]
            * (K_PACKED_TOTAL // 2)
            + b_dword_h[None, :, None, None] * 0
            + b_dword_r[None, None, :, None]
            + b_dword_k[None, None, None, :] * 16
        )
        b_off_up = b_off_gate
        b_scale_shift = (b_dword_h * 8).to(gl.uint32)[None, :, None, None]
        b_scales_ptr_e = b_scales_ptr_e.to(gl.pointer_type(gl.int32))

    # ---- LDS allocation (2 A-data slots, ping-pong) --------------------
    # A fp4 data is still gathered through LDS. A scales use direct VGPR
    # loads in the decode/CDNA4-swizzled layout, avoiding the tiny A-scale
    # async-copy path that Gluon fails to lower for prefill.
    smem_a = gl.allocate_shared_memory(
        a_ptr.type.element_ty,
        [NUM_BUFFERS, BLOCK_M, BLOCK_K_A],
        shared_a,
    )

    a_base_ptr = a_ptr
    b_base_ptr = b_ptr + off_experts.to(gl.int64) * stride_be

    offs_a_i32 = offs_a.to(gl.int32)

    a_data_k_step = BLOCK_K_A * stride_ak
    if B_LAYOUT == B_LAYOUT_N16:
        # 128 K bytes are two 1 KB chunks inside every 16-row group.
        B_DATA_K_STEP: gl.constexpr = 2048
    elif B_LAYOUT == B_LAYOUT_GDOT128:
        B_DATA_K_STEP: gl.constexpr = ((2 * I_r) // 128) * (128 * 128)
    else:
        B_DATA_K_STEP: gl.constexpr = (BLOCK_K_B // 32) * 512
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
        USE_ASYNC=A_FORMAT != "e4m3",
    )
    cdna4_async_copy.commit_group()

    cur_b_gate, cur_b_up, b_scale_gate, b_scale_up = _load_b_pair_vgpr(
        b_base_ptr,
        b_scales_ptr_e,
        offs_b_gate,
        offs_b_up,
        b_off_gate,
        b_off_up,
        0,
        0,
        B_SCALE_K_STEP,
        INTERLEAVED,
        b_scale_shift,
        b_scale_layout,
        B_SCALE_DWORD,
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
        a_scale_group_layout,
        A_SCALE_DWORD,
    )
    if BLOCK_M >= 64:
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
            a_scale_group_layout,
            A_SCALE_DWORD,
        )

    # ---- Per-quarter accumulators (gate + up) -------------------------
    gate_acc_group0 = gl.zeros(
        (GROUP_MFMA_M, BLOCK_N), dtype=gl.float32, layout=mfma_layout
    )
    up_acc_group0 = gl.zeros(
        (GROUP_MFMA_M, BLOCK_N), dtype=gl.float32, layout=mfma_layout
    )
    if BLOCK_M >= 64:
        gate_acc_group1 = gl.zeros(
            (GROUP_MFMA_M, BLOCK_N), dtype=gl.float32, layout=mfma_layout
        )
        up_acc_group1 = gl.zeros(
            (GROUP_MFMA_M, BLOCK_N), dtype=gl.float32, layout=mfma_layout
        )
    if BLOCK_M == 128:
        gate_acc_group2 = gl.zeros(
            (GROUP_MFMA_M, BLOCK_N), dtype=gl.float32, layout=mfma_layout
        )
        gate_acc_group3 = gl.zeros(
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
        next_b_gate, next_b_up, next_bsc_gate, next_bsc_up = _load_b_pair_vgpr(
            b_base_ptr,
            b_scales_ptr_e,
            offs_b_gate,
            offs_b_up,
            b_off_gate,
            b_off_up,
            nxt_b_data_off,
            nxt_b_scale_k,
            B_SCALE_K_STEP,
            INTERLEAVED,
            b_scale_shift,
            b_scale_layout,
            B_SCALE_DWORD,
        )
        if EARLY_A:
            _prefetch_a_data_lds(
                smem_a_slot1,
                a_base_ptr,
                offs_a_i32,
                (tile0_k + 1) * a_data_k_step,
                mask=token_mask[:, None],
                USE_ASYNC=A_FORMAT != "e4m3",
            )
            cdna4_async_copy.commit_group()
            nxt_a_scale_group0 = _load_a_scale_vgpr(
                a_scales_ptr,
                a_scale_base_offsets,
                GROUP0_IDX,
                tile0_k + 1,
                a_scale_group_stride,
                A_SCALE_K_STEP,
                a_scale_group_layout,
                A_SCALE_DWORD,
            )
            if BLOCK_M >= 64:
                nxt_a_scale_group1 = _load_a_scale_vgpr(
                    a_scales_ptr,
                    a_scale_base_offsets,
                    GROUP1_IDX,
                    tile0_k + 1,
                    a_scale_group_stride,
                    A_SCALE_K_STEP,
                    a_scale_group_layout,
                    A_SCALE_DWORD,
                )
        # MFMA groups 0,1 with current A[k] + current B[k].
        gate_acc_group0, up_acc_group0 = _compute_gate_up_group(
            cur_a_group0,
            a_scale_group0,
            cur_b_gate,
            b_scale_gate,
            cur_b_up,
            b_scale_up,
            gate_acc_group0,
            up_acc_group0,
            A_FORMAT,
            INTERLEAVED,
        )
        if BLOCK_M >= 64:
            gate_acc_group1, up_acc_group1 = _compute_gate_up_group(
                cur_a_group1,
                a_scale_group1,
                cur_b_gate,
                b_scale_gate,
                cur_b_up,
                b_scale_up,
                gate_acc_group1,
                up_acc_group1,
                A_FORMAT,
                INTERLEAVED,
            )

        # Prefetch A[k+1] data into LDS slot1. A-scale stays direct-to-VGPR.
        if not EARLY_A:
            _prefetch_a_data_lds(
                smem_a_slot1,
                a_base_ptr,
                offs_a_i32,
                (tile0_k + 1) * a_data_k_step,
                mask=token_mask[:, None],
                USE_ASYNC=A_FORMAT != "e4m3",
            )

        # The 128-row variant consumes the remaining two A[k] quarters while
        # the next A tile is in flight. Smaller variants have consumed their
        # complete tile above.
        if BLOCK_M == 128:
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
                a_scale_group_layout,
                A_SCALE_DWORD,
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
                a_scale_group_layout,
                A_SCALE_DWORD,
            )

        if not EARLY_A:
            cdna4_async_copy.commit_group()

        if BLOCK_M == 128:
            gate_acc_group2, up_acc_group2 = _compute_gate_up_group(
                cur_a_group2,
                a_scale_group2,
                cur_b_gate,
                b_scale_gate,
                cur_b_up,
                b_scale_up,
                gate_acc_group2,
                up_acc_group2,
                A_FORMAT,
                INTERLEAVED,
            )
            gate_acc_group3, up_acc_group3 = _compute_gate_up_group(
                cur_a_group3,
                a_scale_group3,
                cur_b_gate,
                b_scale_gate,
                cur_b_up,
                b_scale_up,
                gate_acc_group3,
                up_acc_group3,
                A_FORMAT,
                INTERLEAVED,
            )

        # Drain A[k+1] prefetch; B[k+1] should also be done by now.
        cdna4_async_copy.wait_group(0)
        cur_a_group0 = _read_a_lds_group(
            smem_a_slot1,
            GROUP0_IDX,
            dot_a_layout,
            GROUP_MFMA_M,
        )
        if EARLY_A:
            a_scale_group0 = nxt_a_scale_group0
        else:
            a_scale_group0 = _load_a_scale_vgpr(
                a_scales_ptr,
                a_scale_base_offsets,
                GROUP0_IDX,
                tile0_k + 1,
                a_scale_group_stride,
                A_SCALE_K_STEP,
                a_scale_group_layout,
                A_SCALE_DWORD,
            )
        if BLOCK_M >= 64:
            cur_a_group1 = _read_a_lds_group(
                smem_a_slot1,
                GROUP1_IDX,
                dot_a_layout,
                GROUP_MFMA_M,
            )
            if EARLY_A:
                a_scale_group1 = nxt_a_scale_group1
            else:
                a_scale_group1 = _load_a_scale_vgpr(
                    a_scales_ptr,
                    a_scale_base_offsets,
                    GROUP1_IDX,
                    tile0_k + 1,
                    a_scale_group_stride,
                    A_SCALE_K_STEP,
                    a_scale_group_layout,
                    A_SCALE_DWORD,
                )

        # Swap B: next -> current.
        cur_b_gate = next_b_gate
        b_scale_gate = next_bsc_gate
        cur_b_up = next_b_up
        b_scale_up = next_bsc_up

        # Swap A LDS slots.
        smem_a_slot0, smem_a_slot1 = smem_a_slot1, smem_a_slot0

        # ==============================================================
        # ---- Tile 1 (odd k+1, cur_slot=slot0, next_slot=slot1) -------
        # ==============================================================
        nxt_b_data_off = (tile1_k + 1) * B_DATA_K_STEP
        nxt_b_scale_k = tile1_k + 1

        next_b_gate, next_b_up, next_bsc_gate, next_bsc_up = _load_b_pair_vgpr(
            b_base_ptr,
            b_scales_ptr_e,
            offs_b_gate,
            offs_b_up,
            b_off_gate,
            b_off_up,
            nxt_b_data_off,
            nxt_b_scale_k,
            B_SCALE_K_STEP,
            INTERLEAVED,
            b_scale_shift,
            b_scale_layout,
            B_SCALE_DWORD,
        )
        if EARLY_A:
            _prefetch_a_data_lds(
                smem_a_slot1,
                a_base_ptr,
                offs_a_i32,
                (tile1_k + 1) * a_data_k_step,
                mask=token_mask[:, None],
                USE_ASYNC=A_FORMAT != "e4m3",
            )
            cdna4_async_copy.commit_group()
            nxt_a_scale_group0 = _load_a_scale_vgpr(
                a_scales_ptr,
                a_scale_base_offsets,
                GROUP0_IDX,
                tile1_k + 1,
                a_scale_group_stride,
                A_SCALE_K_STEP,
                a_scale_group_layout,
                A_SCALE_DWORD,
            )
            if BLOCK_M >= 64:
                nxt_a_scale_group1 = _load_a_scale_vgpr(
                    a_scales_ptr,
                    a_scale_base_offsets,
                    GROUP1_IDX,
                    tile1_k + 1,
                    a_scale_group_stride,
                    A_SCALE_K_STEP,
                    a_scale_group_layout,
                    A_SCALE_DWORD,
                )
        gate_acc_group0, up_acc_group0 = _compute_gate_up_group(
            cur_a_group0,
            a_scale_group0,
            cur_b_gate,
            b_scale_gate,
            cur_b_up,
            b_scale_up,
            gate_acc_group0,
            up_acc_group0,
            A_FORMAT,
            INTERLEAVED,
        )
        if BLOCK_M >= 64:
            gate_acc_group1, up_acc_group1 = _compute_gate_up_group(
                cur_a_group1,
                a_scale_group1,
                cur_b_gate,
                b_scale_gate,
                cur_b_up,
                b_scale_up,
                gate_acc_group1,
                up_acc_group1,
                A_FORMAT,
                INTERLEAVED,
            )

        if not EARLY_A:
            _prefetch_a_data_lds(
                smem_a_slot1,
                a_base_ptr,
                offs_a_i32,
                (tile1_k + 1) * a_data_k_step,
                mask=token_mask[:, None],
                USE_ASYNC=A_FORMAT != "e4m3",
            )

        if BLOCK_M == 128:
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
                a_scale_group_layout,
                A_SCALE_DWORD,
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
                a_scale_group_layout,
                A_SCALE_DWORD,
            )

        if not EARLY_A:
            cdna4_async_copy.commit_group()

        if BLOCK_M == 128:
            gate_acc_group2, up_acc_group2 = _compute_gate_up_group(
                cur_a_group2,
                a_scale_group2,
                cur_b_gate,
                b_scale_gate,
                cur_b_up,
                b_scale_up,
                gate_acc_group2,
                up_acc_group2,
                A_FORMAT,
                INTERLEAVED,
            )
            gate_acc_group3, up_acc_group3 = _compute_gate_up_group(
                cur_a_group3,
                a_scale_group3,
                cur_b_gate,
                b_scale_gate,
                cur_b_up,
                b_scale_up,
                gate_acc_group3,
                up_acc_group3,
                A_FORMAT,
                INTERLEAVED,
            )

        cdna4_async_copy.wait_group(0)
        cur_a_group0 = _read_a_lds_group(
            smem_a_slot1,
            GROUP0_IDX,
            dot_a_layout,
            GROUP_MFMA_M,
        )
        if EARLY_A:
            a_scale_group0 = nxt_a_scale_group0
        else:
            a_scale_group0 = _load_a_scale_vgpr(
                a_scales_ptr,
                a_scale_base_offsets,
                GROUP0_IDX,
                tile1_k + 1,
                a_scale_group_stride,
                A_SCALE_K_STEP,
                a_scale_group_layout,
                A_SCALE_DWORD,
            )
        if BLOCK_M >= 64:
            cur_a_group1 = _read_a_lds_group(
                smem_a_slot1,
                GROUP1_IDX,
                dot_a_layout,
                GROUP_MFMA_M,
            )
            if EARLY_A:
                a_scale_group1 = nxt_a_scale_group1
            else:
                a_scale_group1 = _load_a_scale_vgpr(
                    a_scales_ptr,
                    a_scale_base_offsets,
                    GROUP1_IDX,
                    tile1_k + 1,
                    a_scale_group_stride,
                    A_SCALE_K_STEP,
                    a_scale_group_layout,
                    A_SCALE_DWORD,
                )

        cur_b_gate = next_b_gate
        b_scale_gate = next_bsc_gate
        cur_b_up = next_b_up
        b_scale_up = next_bsc_up

        smem_a_slot0, smem_a_slot1 = smem_a_slot1, smem_a_slot0

    # ---- Peel loop (0 or 1 single-tile advance) ----------------------
    peel_base = unrolled_iters * 2
    for peel_k_offset in range(peel_iters):
        pk = peel_base + peel_k_offset
        nxt_b_data_off = (pk + 1) * B_DATA_K_STEP
        nxt_b_scale_k = pk + 1

        next_b_gate, next_b_up, next_bsc_gate, next_bsc_up = _load_b_pair_vgpr(
            b_base_ptr,
            b_scales_ptr_e,
            offs_b_gate,
            offs_b_up,
            b_off_gate,
            b_off_up,
            nxt_b_data_off,
            nxt_b_scale_k,
            B_SCALE_K_STEP,
            INTERLEAVED,
            b_scale_shift,
            b_scale_layout,
            B_SCALE_DWORD,
        )
        if EARLY_A:
            _prefetch_a_data_lds(
                smem_a_slot1,
                a_base_ptr,
                offs_a_i32,
                (pk + 1) * a_data_k_step,
                mask=token_mask[:, None],
                USE_ASYNC=A_FORMAT != "e4m3",
            )
            cdna4_async_copy.commit_group()
            nxt_a_scale_group0 = _load_a_scale_vgpr(
                a_scales_ptr,
                a_scale_base_offsets,
                GROUP0_IDX,
                pk + 1,
                a_scale_group_stride,
                A_SCALE_K_STEP,
                a_scale_group_layout,
                A_SCALE_DWORD,
            )
            if BLOCK_M >= 64:
                nxt_a_scale_group1 = _load_a_scale_vgpr(
                    a_scales_ptr,
                    a_scale_base_offsets,
                    GROUP1_IDX,
                    pk + 1,
                    a_scale_group_stride,
                    A_SCALE_K_STEP,
                    a_scale_group_layout,
                    A_SCALE_DWORD,
                )
        gate_acc_group0, up_acc_group0 = _compute_gate_up_group(
            cur_a_group0,
            a_scale_group0,
            cur_b_gate,
            b_scale_gate,
            cur_b_up,
            b_scale_up,
            gate_acc_group0,
            up_acc_group0,
            A_FORMAT,
            INTERLEAVED,
        )
        if BLOCK_M >= 64:
            gate_acc_group1, up_acc_group1 = _compute_gate_up_group(
                cur_a_group1,
                a_scale_group1,
                cur_b_gate,
                b_scale_gate,
                cur_b_up,
                b_scale_up,
                gate_acc_group1,
                up_acc_group1,
                A_FORMAT,
                INTERLEAVED,
            )

        if not EARLY_A:
            _prefetch_a_data_lds(
                smem_a_slot1,
                a_base_ptr,
                offs_a_i32,
                (pk + 1) * a_data_k_step,
                mask=token_mask[:, None],
                USE_ASYNC=A_FORMAT != "e4m3",
            )

        if BLOCK_M == 128:
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
                a_scale_group_layout,
                A_SCALE_DWORD,
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
                a_scale_group_layout,
                A_SCALE_DWORD,
            )

        if not EARLY_A:
            cdna4_async_copy.commit_group()

        if BLOCK_M == 128:
            gate_acc_group2, up_acc_group2 = _compute_gate_up_group(
                cur_a_group2,
                a_scale_group2,
                cur_b_gate,
                b_scale_gate,
                cur_b_up,
                b_scale_up,
                gate_acc_group2,
                up_acc_group2,
                A_FORMAT,
                INTERLEAVED,
            )
            gate_acc_group3, up_acc_group3 = _compute_gate_up_group(
                cur_a_group3,
                a_scale_group3,
                cur_b_gate,
                b_scale_gate,
                cur_b_up,
                b_scale_up,
                gate_acc_group3,
                up_acc_group3,
                A_FORMAT,
                INTERLEAVED,
            )

        cdna4_async_copy.wait_group(0)
        cur_a_group0 = _read_a_lds_group(
            smem_a_slot1,
            GROUP0_IDX,
            dot_a_layout,
            GROUP_MFMA_M,
        )
        if EARLY_A:
            a_scale_group0 = nxt_a_scale_group0
        else:
            a_scale_group0 = _load_a_scale_vgpr(
                a_scales_ptr,
                a_scale_base_offsets,
                GROUP0_IDX,
                pk + 1,
                a_scale_group_stride,
                A_SCALE_K_STEP,
                a_scale_group_layout,
                A_SCALE_DWORD,
            )
        if BLOCK_M >= 64:
            cur_a_group1 = _read_a_lds_group(
                smem_a_slot1,
                GROUP1_IDX,
                dot_a_layout,
                GROUP_MFMA_M,
            )
            if EARLY_A:
                a_scale_group1 = nxt_a_scale_group1
            else:
                a_scale_group1 = _load_a_scale_vgpr(
                    a_scales_ptr,
                    a_scale_base_offsets,
                    GROUP1_IDX,
                    pk + 1,
                    a_scale_group_stride,
                    A_SCALE_K_STEP,
                    a_scale_group_layout,
                    A_SCALE_DWORD,
                )

        cur_b_gate = next_b_gate
        b_scale_gate = next_bsc_gate
        cur_b_up = next_b_up
        b_scale_up = next_bsc_up

        smem_a_slot0, smem_a_slot1 = smem_a_slot1, smem_a_slot0

    # ---- 1-tile drain (last K tile, no prefetch) ---------------------
    # The active cur_a_group* values and cur_b_* hold the last tile's data.
    if BLOCK_M == 128:
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
            a_scale_group_layout,
            A_SCALE_DWORD,
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
            a_scale_group_layout,
            A_SCALE_DWORD,
        )

    gate_acc_group0, up_acc_group0 = _compute_gate_up_group(
        cur_a_group0,
        a_scale_group0,
        cur_b_gate,
        b_scale_gate,
        cur_b_up,
        b_scale_up,
        gate_acc_group0,
        up_acc_group0,
        A_FORMAT,
        INTERLEAVED,
    )
    if BLOCK_M >= 64:
        gate_acc_group1, up_acc_group1 = _compute_gate_up_group(
            cur_a_group1,
            a_scale_group1,
            cur_b_gate,
            b_scale_gate,
            cur_b_up,
            b_scale_up,
            gate_acc_group1,
            up_acc_group1,
            A_FORMAT,
            INTERLEAVED,
        )
    if BLOCK_M == 128:
        gate_acc_group2, up_acc_group2 = _compute_gate_up_group(
            cur_a_group2,
            a_scale_group2,
            cur_b_gate,
            b_scale_gate,
            cur_b_up,
            b_scale_up,
            gate_acc_group2,
            up_acc_group2,
            A_FORMAT,
            INTERLEAVED,
        )
        gate_acc_group3, up_acc_group3 = _compute_gate_up_group(
            cur_a_group3,
            a_scale_group3,
            cur_b_gate,
            b_scale_gate,
            cur_b_up,
            b_scale_up,
            gate_acc_group3,
            up_acc_group3,
            A_FORMAT,
            INTERLEAVED,
        )

    if INTERLEAVED:
        acc_swiglu_group0 = _apply_interleaved_gate_activation(
            gate_acc_group0,
            DO_SITU,
            SWIGLU_ALPHA,
            SWIGLU_LIMIT,
            SWIGLU_BETA,
            SITU_LINEAR_BETA,
            OUTPUT_BLOCK_N,
            B_LAYOUT == B_LAYOUT_N16,
        )
    else:
        acc_swiglu_group0 = _apply_gate_activation(
            gate_acc_group0,
            up_acc_group0,
            DO_SITU,
            SWIGLU_ALPHA,
            SWIGLU_LIMIT,
            SWIGLU_BETA,
            SITU_LINEAR_BETA,
        )
    if BLOCK_M >= 64:
        if INTERLEAVED:
            acc_swiglu_group1 = _apply_interleaved_gate_activation(
                gate_acc_group1,
                DO_SITU,
                SWIGLU_ALPHA,
                SWIGLU_LIMIT,
                SWIGLU_BETA,
                SITU_LINEAR_BETA,
                OUTPUT_BLOCK_N,
                B_LAYOUT == B_LAYOUT_N16,
            )
        else:
            acc_swiglu_group1 = _apply_gate_activation(
                gate_acc_group1,
                up_acc_group1,
                DO_SITU,
                SWIGLU_ALPHA,
                SWIGLU_LIMIT,
                SWIGLU_BETA,
                SITU_LINEAR_BETA,
            )
    if BLOCK_M == 128:
        if INTERLEAVED:
            acc_swiglu_group2 = _apply_interleaved_gate_activation(
                gate_acc_group2,
                DO_SITU,
                SWIGLU_ALPHA,
                SWIGLU_LIMIT,
                SWIGLU_BETA,
                SITU_LINEAR_BETA,
                OUTPUT_BLOCK_N,
                B_LAYOUT == B_LAYOUT_N16,
            )
            acc_swiglu_group3 = _apply_interleaved_gate_activation(
                gate_acc_group3,
                DO_SITU,
                SWIGLU_ALPHA,
                SWIGLU_LIMIT,
                SWIGLU_BETA,
                SITU_LINEAR_BETA,
                OUTPUT_BLOCK_N,
                B_LAYOUT == B_LAYOUT_N16,
            )
        else:
            acc_swiglu_group2 = _apply_gate_activation(
                gate_acc_group2,
                up_acc_group2,
                DO_SITU,
                SWIGLU_ALPHA,
                SWIGLU_LIMIT,
                SWIGLU_BETA,
                SITU_LINEAR_BETA,
            )
            acc_swiglu_group3 = _apply_gate_activation(
                gate_acc_group3,
                up_acc_group3,
                DO_SITU,
                SWIGLU_ALPHA,
                SWIGLU_LIMIT,
                SWIGLU_BETA,
                SITU_LINEAR_BETA,
            )

    _store_swiglu_tile_group(
        acc_swiglu_group0,
        c_ptr,
        c_scale_ptr,
        sorted_token_ids_ptr,
        pid_m,
        pid_n,
        EM,
        num_tokens,
        top_k,
        stride_cm,
        stride_cn,
        GROUP0_IDX,
        GROUP_MFMA_M,
        BLOCK_M,
        OUTPUT_BLOCK_N,
        I_r,
        OUTPUT_K,
        OUTPUT_SORTED,
        OUTPUT_QUANTIZED,
        OUTPUT_FP8,
    )
    if BLOCK_M >= 64:
        _store_swiglu_tile_group(
            acc_swiglu_group1,
            c_ptr,
            c_scale_ptr,
            sorted_token_ids_ptr,
            pid_m,
            pid_n,
            EM,
            num_tokens,
            top_k,
            stride_cm,
            stride_cn,
            GROUP1_IDX,
            GROUP_MFMA_M,
            BLOCK_M,
            OUTPUT_BLOCK_N,
            I_r,
            OUTPUT_K,
            OUTPUT_SORTED,
            OUTPUT_QUANTIZED,
            OUTPUT_FP8,
        )
    if BLOCK_M == 128:
        _store_swiglu_tile_group(
            acc_swiglu_group2,
            c_ptr,
            c_scale_ptr,
            sorted_token_ids_ptr,
            pid_m,
            pid_n,
            EM,
            num_tokens,
            top_k,
            stride_cm,
            stride_cn,
            GROUP2_IDX,
            GROUP_MFMA_M,
            BLOCK_M,
            OUTPUT_BLOCK_N,
            I_r,
            OUTPUT_K,
            OUTPUT_SORTED,
            OUTPUT_QUANTIZED,
            OUTPUT_FP8,
        )
        _store_swiglu_tile_group(
            acc_swiglu_group3,
            c_ptr,
            c_scale_ptr,
            sorted_token_ids_ptr,
            pid_m,
            pid_n,
            EM,
            num_tokens,
            top_k,
            stride_cm,
            stride_cn,
            GROUP3_IDX,
            GROUP_MFMA_M,
            BLOCK_M,
            OUTPUT_BLOCK_N,
            I_r,
            OUTPUT_K,
            OUTPUT_SORTED,
            OUTPUT_QUANTIZED,
            OUTPUT_FP8,
        )


@gluon.jit
def _load_a_scale_words(
    a_scales_ptr,
    a_scale_base_offsets,
    GROUP_IDX: gl.constexpr,
    tile_k,
    group_stride,
    A_SCALE_K_STEP: gl.constexpr,
):
    # Load the A-scale dwords of one 32-row group and K tile. Callers do so a
    # K tile ahead of the MFMAs that consume them, so those do not wait for the
    # load; ``_regroup_scale_words`` turns them into the MFMA operand.
    return gl.amd.cdna4.buffer_load(
        ptr=a_scales_ptr,
        offsets=(
            a_scale_base_offsets + GROUP_IDX * group_stride + tile_k * A_SCALE_K_STEP
        ),
    )


@gluon.jit
def _regroup_scale_words(loaded, scale_layout: gl.constexpr):
    # ``loaded`` holds one dword per lane: in the swizzle the dword at
    # (k % 4) * 64 + (row % 16) * 4 has rows m and m + 16 at scales k and
    # k + 4, the four scales a lane feeds its MFMAs for this group and tile,
    # with (row, k) at byte 2 * (k // 4) + row // 16. Regroup them into the
    # MFMA scale layout in registers.
    k_low = gl.join(loaded.to(gl.uint8), (loaded >> 8).to(gl.uint8))
    k_high = gl.join((loaded >> 16).to(gl.uint8), (loaded >> 24).to(gl.uint8))
    # [m % 16, k % 4, m // 16, k // 4] -> [m, k]
    scales = gl.join(k_low, k_high).permute((2, 0, 3, 1)).reshape((32, 8))
    return gl.convert_layout(scales, scale_layout)


@gluon.jit
def _b_scale_operand(loaded, word_shift, scale_layout: gl.constexpr):
    # B-scale MFMA operand from the dwords ``_load_b_scale_vgpr`` loaded: per
    # lane and 32-row panel, rows n and n + 16 at scales k and k + 4, with
    # (row, k) at byte 2 * (k // 4) + row % 32 // 16. Keep this wave's row half
    # and regroup.
    num_rows: gl.constexpr = loaded.shape[0]
    halves = gl.join(
        (loaded >> word_shift).to(gl.uint8),
        (loaded >> (word_shift + 16)).to(gl.uint8),
    )
    # [n, k % 4, k // 4] -> [n, k]
    scales = halves.permute((0, 2, 1)).reshape((num_rows, 8))
    return gl.convert_layout(scales, scale_layout)


@gluon.jit
def _wide_load_operands(
    b_base_ptr,
    b_scales_words_ptr,
    offs_b,
    b_scale_offs,
    a_scales_words_ptr,
    a_scale_offs,
    a_scale_group_stride,
    tile_k,
    B_DATA_K_STEP: gl.constexpr,
):
    # Both groups' activation-scale dwords, the weight-scale dwords and the
    # weights of one K tile of the wide stage-1 tile. Loads complete in issue
    # order, so the scales go first: waiting for them to regroup their bytes
    # must not wait for the weights too.
    # A 256-wide K tile's scales of one 32-row panel are 256 bytes, 64 dwords.
    SCALE_WORDS_PER_K_TILE: gl.constexpr = 64
    a_words_group0 = _load_a_scale_words(
        a_scales_words_ptr,
        a_scale_offs,
        0,
        tile_k,
        a_scale_group_stride,
        SCALE_WORDS_PER_K_TILE,
    )
    a_words_group1 = _load_a_scale_words(
        a_scales_words_ptr,
        a_scale_offs,
        1,
        tile_k,
        a_scale_group_stride,
        SCALE_WORDS_PER_K_TILE,
    )
    b_words = _load_b_scale_vgpr(
        b_scales_words_ptr, b_scale_offs, tile_k, SCALE_WORDS_PER_K_TILE
    )
    b = _load_b_data_vgpr(b_base_ptr, offs_b, tile_k * B_DATA_K_STEP)
    return b, b_words, a_words_group0, a_words_group1


@gluon.jit
def _wide_k_tile(
    acc_group0,
    acc_group1,
    slot_read,
    slot_write,
    tile_k,
    cur_b,
    cur_b_words,
    a_words_group0,
    a_words_group1,
    a_ptr,
    offs_a,
    token_mask,
    b_base_ptr,
    b_scales_words_ptr,
    offs_b,
    b_scale_offs,
    b_scale_word_shift,
    a_scales_words_ptr,
    a_scale_offs,
    a_scale_group_stride,
    B_DATA_K_STEP: gl.constexpr,
    LOAD_NEXT: gl.constexpr,
    COPY_AHEAD: gl.constexpr,
    WAIT_GROUPS: gl.constexpr,
    dot_a_layout: gl.constexpr,
    a_scale_layout: gl.constexpr,
    b_scale_layout: gl.constexpr,
):
    # One K tile of the wide stage-1 tile: issue tile k + 1's register
    # operands (LOAD_NEXT) and tile k + 2's activation copy into
    # ``slot_write`` (COPY_AHEAD), wait until tile k's copy has landed (at most
    # WAIT_GROUPS younger copies in flight), then run tile k's MFMAs from
    # ``slot_read``. Returns the accumulators and tile k + 1's operands.
    if LOAD_NEXT:
        next_b, next_b_words, next_a_words_group0, next_a_words_group1 = (
            _wide_load_operands(
                b_base_ptr,
                b_scales_words_ptr,
                offs_b,
                b_scale_offs,
                a_scales_words_ptr,
                a_scale_offs,
                a_scale_group_stride,
                tile_k + 1,
                B_DATA_K_STEP,
            )
        )
    else:
        next_b = cur_b
        next_b_words = cur_b_words
        next_a_words_group0 = a_words_group0
        next_a_words_group1 = a_words_group1
    if COPY_AHEAD:
        # The slot tile k - 1 used; the compiler orders the copy after every
        # wave's reads of that tile.
        block_k_packed: gl.constexpr = slot_write.shape[1]
        _prefetch_a_data_lds(
            slot_write,
            a_ptr,
            offs_a,
            (tile_k + 2) * block_k_packed,
            mask=token_mask,
        )
        cdna4_async_copy.commit_group()
    cdna4_async_copy.wait_group(WAIT_GROUPS)
    group_m: gl.constexpr = acc_group0.shape[0]
    b_scale = _b_scale_operand(cur_b_words, b_scale_word_shift, b_scale_layout)
    acc_group0 = _compute_mx_group(
        _read_a_lds_group(slot_read, 0, dot_a_layout, group_m),
        _regroup_scale_words(a_words_group0, a_scale_layout),
        cur_b,
        b_scale,
        acc_group0,
        "e2m1",
    )
    acc_group1 = _compute_mx_group(
        _read_a_lds_group(slot_read, 1, dot_a_layout, group_m),
        _regroup_scale_words(a_words_group1, a_scale_layout),
        cur_b,
        b_scale,
        acc_group1,
        "e2m1",
    )
    # Keep the next tile's scale regrouping out of this tile's MFMAs: loads
    # complete in issue order, so consuming tile k + 1's scale dwords here
    # would also wait for tile k + 1's weights.
    gl.amd.hint.sched_barrier()
    return (
        acc_group0,
        acc_group1,
        next_b,
        next_b_words,
        next_a_words_group0,
        next_a_words_group1,
    )


@gluon.jit
def _wide_tile_gemm(
    a_ptr,
    stride_am,
    a_scales_ptr,
    stride_se_n_pad,
    b_ptr,
    stride_be,
    b_scales_ptr,
    stride_bse_e,
    b_scale_panel_words,
    sorted_token_ids_ptr,
    EM,
    num_tokens,
    off_experts,
    pid_m,
    pid_n,
    N_TOTAL: gl.constexpr,
    NUM_K_TILES: gl.constexpr,
    INPUT_SORTED: gl.constexpr,
):
    """E2M1 x E2M1 GEMM of one 64-row x 256-column tile over sorted routes.

    Returns the FP32 accumulators of rows 0-31 and 32-63, ``[32, 256]`` each
    in the 4-wave MFMA layout. Each element accumulates
    ``v_mfma_scale_f32_16x16x128_f8f6f4`` products over 128-wide K steps in
    ascending order from zero, as the general stage-1 and stage-2 tiles do.

    Each wave owns 64 of the 256 weight columns for all 64 rows, so it reads
    the activation tile from LDS once per 256-wide K tile. Activations are
    copied two K tiles ahead through a three-slot ring; weights,
    weight-scale and activation-scale dwords load into registers one K tile
    ahead.

    Activation rows are gathered by token from ``sorted_token_ids``, or are
    already in sorted order (``INPUT_SORTED``). Weights are the GDOT128
    (128 x 128-tiled) layout of ``N_TOTAL`` columns; their scales and the
    activation scales are CDNA4-swizzled, with ``b_scale_panel_words`` and
    ``stride_se_n_pad * 8`` dwords per 32-row panel.
    """
    BLOCK_M: gl.constexpr = 64
    GROUP_M: gl.constexpr = 32
    BLOCK_N: gl.constexpr = 256
    BLOCK_K_PACKED: gl.constexpr = 128
    gl.static_assert(gl.num_warps() == 4, "the wide tile runs 4 waves")
    gl.static_assert(NUM_K_TILES >= 2, "the three-slot ring needs two K tiles")
    gl.static_assert(N_TOTAL % BLOCK_N == 0)

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
        dot_a_layout, [GROUP_M, 8]
    )
    b_scale_layout: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(
        dot_b_layout, [BLOCK_N, 8]
    )

    # ---- activation rows: gather layout and the three-slot LDS ring -------
    # Same padded layout as the 64-row E2M1 tile of the general kernel: a
    # 128-byte pad every 32 rows lets each group's ds_reads share a base.
    gload_a_layout: gl.constexpr = gl.BlockedLayout([1, 16], [8, 8], [4, 1], [1, 0])
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
        ],
        [],
        [BLOCK_M, BLOCK_K_PACKED],
    )
    smem_a = gl.allocate_shared_memory(
        a_ptr.type.element_ty, [3, BLOCK_M, BLOCK_K_PACKED], shared_a
    )
    tile_row = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, gload_a_layout))
    offs_route = pid_m * BLOCK_M + tile_row
    packed_route = gl.load(
        sorted_token_ids_ptr + offs_route, mask=offs_route < EM, other=num_tokens
    )
    token_id = packed_route & 0xFFFFFF
    token_mask = (token_id < num_tokens)[:, None]
    offs_ak = gl.arange(0, BLOCK_K_PACKED, layout=gl.SliceLayout(0, gload_a_layout))
    if INPUT_SORTED:
        # Rows in sorted order: rebase the scalar pointer per tile so the
        # 32-bit buffer offsets stay within the tile.
        a_ptr = a_ptr + (pid_m * BLOCK_M).to(gl.int64) * stride_am
        offs_a = tile_row[:, None] * stride_am + offs_ak[None, :]
    else:
        # Buffer-load offsets are 32-bit; the launcher bounds the activation
        # span.
        offs_a = (token_id[:, None].to(gl.int64) * stride_am + offs_ak[None, :]).to(
            gl.int32
        )

    # ---- activation scales: one dword per lane, 32-row group and K tile ---
    # See ``_regroup_scale_words`` for the swizzle; group 1 is the next panel.
    a_scale_word_layout: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[],
        lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2]],
        warp_bases=[[0, 0], [0, 0]],
        block_bases=[],
        shape=[16, 4],
    )
    a_scale_m = gl.arange(0, 16, layout=gl.SliceLayout(1, a_scale_word_layout))
    a_scale_k = gl.arange(0, 4, layout=gl.SliceLayout(0, a_scale_word_layout))
    a_scale_group_stride = stride_se_n_pad * 8
    # The program's first scale group goes into the base pointer in 64 bits
    # (it grows with the sorted row count); the buffer offsets keep the
    # 32-bit in-tile part.
    a_scales_words_ptr = (
        a_scales_ptr.to(gl.pointer_type(gl.uint32))
        + (pid_m * (BLOCK_M // GROUP_M)).to(gl.int64) * a_scale_group_stride
    )
    a_scale_offs = a_scale_m[:, None] + a_scale_k[None, :] * 16

    # ---- weights: (16, 16)-tiled GDOT128 closed form, 256 rows ------------
    offs_bk = gl.arange(0, BLOCK_K_PACKED, layout=gl.SliceLayout(1, dot_b_layout))
    offs_bn = pid_n * BLOCK_N + gl.arange(
        0, BLOCK_N, layout=gl.SliceLayout(0, dot_b_layout)
    )
    n_in = offs_bn % 128
    offs_b = (
        (offs_bn // 128)[None, :] * (128 * 128)
        + (n_in // 16)[None, :] * 2048
        + (offs_bk // 64)[:, None] * 1024
        + ((offs_bk // 16) % 4)[:, None] * 256
        + (n_in % 16)[None, :] * 16
        + (offs_bk % 16)[:, None]
    )
    B_DATA_K_STEP: gl.constexpr = (N_TOTAL // 128) * (128 * 128)
    b_base_ptr = b_ptr + off_experts.to(gl.int64) * stride_be

    # Weight-scale dwords: the MFMA scale layout without its k + 4 register,
    # which shares the dword (see ``_b_scale_operand``).
    b_scale_word_layout: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[64, 0], [128, 0]],
        lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2]],
        warp_bases=[[16, 0], [32, 0]],
        block_bases=[],
        shape=[BLOCK_N, 4],
    )
    b_word_rows = (
        pid_n * BLOCK_N
        + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, b_scale_word_layout))
    ).to(gl.uint32)
    b_word_k = gl.arange(0, 4, layout=gl.SliceLayout(0, b_scale_word_layout)).to(
        gl.uint32
    )
    b_scale_offs = ((b_word_rows // 32) * b_scale_panel_words + b_word_rows % 16)[
        :, None
    ] + b_word_k[None, :] * 16
    b_scale_word_shift = (((b_word_rows % 32) // 16) * 8)[:, None]
    b_scales_words_ptr = (b_scales_ptr + off_experts.to(gl.int64) * stride_bse_e).to(
        gl.pointer_type(gl.uint32)
    )

    # ---- prologue: A[0], A[1] in flight; B[0] and scales of tile 0 --------
    slot_cur = smem_a.index(0)
    slot_next = smem_a.index(1)
    slot_free = smem_a.index(2)
    _prefetch_a_data_lds(slot_cur, a_ptr, offs_a, 0, mask=token_mask)
    cdna4_async_copy.commit_group()
    _prefetch_a_data_lds(slot_next, a_ptr, offs_a, BLOCK_K_PACKED, mask=token_mask)
    cdna4_async_copy.commit_group()
    b_even, b_words_even, a_words0_even, a_words1_even = _wide_load_operands(
        b_base_ptr,
        b_scales_words_ptr,
        offs_b,
        b_scale_offs,
        a_scales_words_ptr,
        a_scale_offs,
        a_scale_group_stride,
        0,
        B_DATA_K_STEP,
    )
    acc_group0 = gl.zeros((GROUP_M, BLOCK_N), dtype=gl.float32, layout=mfma_layout)
    acc_group1 = gl.zeros((GROUP_M, BLOCK_N), dtype=gl.float32, layout=mfma_layout)

    # ---- steady state: tiles 0 .. NUM_K_TILES - 3 -------------------------
    # Unrolled twice so tile k + 1's operands load into the registers tile
    # k - 1 freed: no copies at the loop edge, so no wait for them there.
    STEADY_TILES: gl.constexpr = NUM_K_TILES - 2
    for pair in range(STEADY_TILES // 2):
        tile_k = pair * 2
        acc_group0, acc_group1, b_odd, b_words_odd, a_words0_odd, a_words1_odd = (
            _wide_k_tile(
                acc_group0,
                acc_group1,
                slot_cur,
                slot_free,
                tile_k,
                b_even,
                b_words_even,
                a_words0_even,
                a_words1_even,
                a_ptr,
                offs_a,
                token_mask,
                b_base_ptr,
                b_scales_words_ptr,
                offs_b,
                b_scale_offs,
                b_scale_word_shift,
                a_scales_words_ptr,
                a_scale_offs,
                a_scale_group_stride,
                B_DATA_K_STEP,
                True,
                True,
                2,
                dot_a_layout,
                a_scale_layout,
                b_scale_layout,
            )
        )
        slot_cur, slot_next, slot_free = slot_next, slot_free, slot_cur
        acc_group0, acc_group1, b_even, b_words_even, a_words0_even, a_words1_even = (
            _wide_k_tile(
                acc_group0,
                acc_group1,
                slot_cur,
                slot_free,
                tile_k + 1,
                b_odd,
                b_words_odd,
                a_words0_odd,
                a_words1_odd,
                a_ptr,
                offs_a,
                token_mask,
                b_base_ptr,
                b_scales_words_ptr,
                offs_b,
                b_scale_offs,
                b_scale_word_shift,
                a_scales_words_ptr,
                a_scale_offs,
                a_scale_group_stride,
                B_DATA_K_STEP,
                True,
                True,
                2,
                dot_a_layout,
                a_scale_layout,
                b_scale_layout,
            )
        )
        slot_cur, slot_next, slot_free = slot_next, slot_free, slot_cur
    if STEADY_TILES % 2 == 1:
        acc_group0, acc_group1, b_even, b_words_even, a_words0_even, a_words1_even = (
            _wide_k_tile(
                acc_group0,
                acc_group1,
                slot_cur,
                slot_free,
                STEADY_TILES - 1,
                b_even,
                b_words_even,
                a_words0_even,
                a_words1_even,
                a_ptr,
                offs_a,
                token_mask,
                b_base_ptr,
                b_scales_words_ptr,
                offs_b,
                b_scale_offs,
                b_scale_word_shift,
                a_scales_words_ptr,
                a_scale_offs,
                a_scale_group_stride,
                B_DATA_K_STEP,
                True,
                True,
                2,
                dot_a_layout,
                a_scale_layout,
                b_scale_layout,
            )
        )
        slot_cur, slot_next, slot_free = slot_next, slot_free, slot_cur

    # ---- the last two tiles: no further activation copies -----------------
    acc_group0, acc_group1, b_odd, b_words_odd, a_words0_odd, a_words1_odd = (
        _wide_k_tile(
            acc_group0,
            acc_group1,
            slot_cur,
            slot_free,
            NUM_K_TILES - 2,
            b_even,
            b_words_even,
            a_words0_even,
            a_words1_even,
            a_ptr,
            offs_a,
            token_mask,
            b_base_ptr,
            b_scales_words_ptr,
            offs_b,
            b_scale_offs,
            b_scale_word_shift,
            a_scales_words_ptr,
            a_scale_offs,
            a_scale_group_stride,
            B_DATA_K_STEP,
            True,
            False,
            1,
            dot_a_layout,
            a_scale_layout,
            b_scale_layout,
        )
    )
    acc_group0, acc_group1, _, _, _, _ = _wide_k_tile(
        acc_group0,
        acc_group1,
        slot_next,
        slot_free,
        NUM_K_TILES - 1,
        b_odd,
        b_words_odd,
        a_words0_odd,
        a_words1_odd,
        a_ptr,
        offs_a,
        token_mask,
        b_base_ptr,
        b_scales_words_ptr,
        offs_b,
        b_scale_offs,
        b_scale_word_shift,
        a_scales_words_ptr,
        a_scale_offs,
        a_scale_group_stride,
        B_DATA_K_STEP,
        False,
        False,
        0,
        dot_a_layout,
        a_scale_layout,
        b_scale_layout,
    )

    return acc_group0, acc_group1


@gluon.jit
def _split_gate_up_activation(
    acc,
    DO_SITU: gl.constexpr,
    ALPHA: gl.constexpr,
    LIMIT: gl.constexpr,
    BETA: gl.constexpr,
    LINEAR_BETA: gl.constexpr,
):
    # Interleaved W13 puts gate and up in adjacent columns, which one lane of
    # the transposed MFMA layout already holds side by side: split them in
    # registers instead of redistributing the accumulator first.
    rows: gl.constexpr = acc.shape[0]
    cols: gl.constexpr = acc.shape[1]
    gate, up = gl.split(acc.reshape((rows, cols // 2, 2)))
    return _apply_gate_activation(gate, up, DO_SITU, ALPHA, LIMIT, BETA, LINEAR_BETA)


# Token and route counts follow the batch and only bound masks; neither may
# select a binary.
def _stage1_wide_launch_metadata(grid, kernel, args):
    """Report the sorted-route count and padded W13 work of the wide tile."""
    return {
        "name": f"{kernel.name} (EM={args['EM']})",
        "flops8": 2 * args["EM"] * args["K_PACKED_TOTAL"] * 2 * 2 * args["I_r"],
    }


@gluon.jit(
    launch_metadata=_stage1_wide_launch_metadata,
    do_not_specialize=["EM", "num_tokens"],
)
def gluon_mxfp4_moe_stage1_e2m1_wide_gfx950(
    a_ptr,
    b_ptr,
    c_ptr,
    c_scale_ptr,
    a_scales_ptr,
    b_scales_ptr,
    sorted_token_ids_ptr,
    sorted_expert_ids_ptr,
    num_tokens_post_padded_ptr,
    EM,
    num_tokens,
    stride_am,
    stride_be,
    stride_cm,
    stride_bse_e,
    stride_se_n_pad,
    K_PACKED_TOTAL: gl.constexpr,
    SORT_BLOCK_M: gl.constexpr,
    I_r: gl.constexpr,
    OUTPUT_K: gl.constexpr,
    SWIGLU_ALPHA: gl.constexpr,
    SWIGLU_LIMIT: gl.constexpr,
    SWIGLU_BETA: gl.constexpr,
    DO_SITU: gl.constexpr,
    SITU_LINEAR_BETA: gl.constexpr,
):
    """E2M1 stage 1 on 64-row x 256-weight-row tiles (128 outputs), 4 waves.

    Same operands, MFMA instruction, 256-wide K tiles and per-element K order
    as ``gluon_mxfp4_moe_stage1_kernel``'s E2M1 GDOT128 path, so the sorted
    MXFP4 rows and scales it writes are bit for bit the same. The GEMM is
    ``_wide_tile_gemm`` (each wave reads the 64-row activation tile from LDS
    once per K tile, activations two K tiles ahead); the epilogue splits
    gate/up in registers and quantizes the whole 64 x 128 tile in one pass
    (one register redistribution through LDS) instead of one pass per 32-row
    group.

    ``SORT_BLOCK_M`` is the sort's block height (64 or 128), shared with stage
    2; a 128-row block is two of these tiles with the block's expert.
    """
    BLOCK_M: gl.constexpr = 64
    BLOCK_N: gl.constexpr = 256
    OUTPUT_BLOCK_N: gl.constexpr = BLOCK_N // 2
    BLOCK_K_PACKED: gl.constexpr = 128
    NUM_K_TILES: gl.constexpr = K_PACKED_TOTAL // BLOCK_K_PACKED
    gl.static_assert(gl.num_warps() == 4, "the wide stage-1 tile runs 4 waves")
    gl.static_assert(K_PACKED_TOTAL % BLOCK_K_PACKED == 0)
    gl.static_assert(NUM_K_TILES >= 2, "the three-slot ring needs two K tiles")
    gl.static_assert(I_r % OUTPUT_BLOCK_N == 0)
    gl.static_assert(SORT_BLOCK_M % BLOCK_M == 0)

    # Consecutive programs take the column tiles of one row tile; dealt
    # round-robin over the XCDs, an XCD re-reads the same weight columns of
    # each of an expert's row tiles from its L2.
    pid = gl.program_id(axis=0)
    num_pid_n: gl.constexpr = I_r // OUTPUT_BLOCK_N
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    if pid_m * BLOCK_M >= gl.load(num_tokens_post_padded_ptr):
        return
    off_experts = gl.load(sorted_expert_ids_ptr + pid_m // (SORT_BLOCK_M // BLOCK_M))

    acc_group0, acc_group1 = _wide_tile_gemm(
        a_ptr,
        stride_am,
        a_scales_ptr,
        stride_se_n_pad,
        b_ptr,
        stride_be,
        b_scales_ptr,
        stride_bse_e,
        K_PACKED_TOTAL // 2,
        sorted_token_ids_ptr,
        EM,
        num_tokens,
        off_experts,
        pid_m,
        pid_n,
        2 * I_r,
        NUM_K_TILES,
        INPUT_SORTED=False,
    )

    # ---- epilogue: one pass over the 64 x 128 output tile -----------------
    act_group0 = _split_gate_up_activation(
        acc_group0, DO_SITU, SWIGLU_ALPHA, SWIGLU_LIMIT, SWIGLU_BETA, SITU_LINEAR_BETA
    )
    act_group1 = _split_gate_up_activation(
        acc_group1, DO_SITU, SWIGLU_ALPHA, SWIGLU_LIMIT, SWIGLU_BETA, SITU_LINEAR_BETA
    )
    # [32, 128] x 2 -> [64, 128], group 1 below group 0.
    act = (
        gl.join(act_group0, act_group1)
        .permute((2, 0, 1))
        .reshape((BLOCK_M, OUTPUT_BLOCK_N))
    )
    _store_swiglu_tile_group(
        act,
        c_ptr,
        c_scale_ptr,
        sorted_token_ids_ptr,
        pid_m,
        pid_n,
        EM,
        num_tokens,
        1,
        stride_cm,
        1,
        0,
        BLOCK_M,
        BLOCK_M,
        OUTPUT_BLOCK_N,
        I_r,
        OUTPUT_K,
        True,
        True,
        False,
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
    block_m=128,
    sorted_weights=None,
    quant_type=None,
    activation=None,
    splitk=1,
    use_non_temporal_load=False,
    dst_type=None,
    b_preshuffled: bool = False,
    swiglu_alpha: float = 1.0,
    swiglu_limit: float = 0.0,
    swiglu_beta: float = 0.0,
    situ_linear_beta: float | None = None,
    output_sorted: bool = False,
    output_quantized: bool = False,
    out_scale: torch.Tensor | None = None,
    output_k: int | None = None,
    a_format: str = "e2m1",
    output_format: str = "e2m1",
    input_sorted: bool = False,
    source_tokens: int | None = None,
    *,
    b_layout: int,
    num_warps: int,
    wide_tile: bool,
):
    """Host-side launcher for Gluon MXFP4 MoE stage 1.

    ``num_warps`` explicitly selects the stage's subgroup geometry.
    ``wide_tile`` selects ``gluon_mxfp4_moe_stage1_e2m1_wide_gfx950`` (64-row x
    128-output tiles on 4 waves, three-slot activation ring, one-pass
    epilogue) for E2M1 activations and outputs in sorted MXFP4 with GDOT128
    weights over 64- or 128-row sort blocks; ``num_warps`` must then be the
    sort block's own count (``block_m // 32``). Its output is bitwise the same
    as the general kernel's; other configurations raise ``ValueError``.
    ``b_layout`` names the W13 bank layout (``B_LAYOUT_*``): the MFMA16
    shuffle, which this function applies unless ``b_preshuffled``, or a
    gdot128 or N16 bank, which must be passed preshuffled. N16 banks take
    E2M1 activations only.

    Computes, for each expert ``e``, the default SwiGLU expression::

        inter_e = silu(hidden_states_e @ w1_e[:I_r, :].T)
                  * (hidden_states_e @ w1_e[I_r:, :].T)

    When ``situ_linear_beta`` is provided, the same GEMM body instead uses the
    SiTU-v2 epilogue. Both modes run in one kernel launch over all experts.
    The grid is one CTA per (M-tile, N-tile). Separate gate/up layouts
    produce a ``BLOCK_M`` x ``BLOCK_N`` output region per CTA; interleaved
    W13 produces ``BLOCK_M`` x ``(BLOCK_N / 2)`` after its gate/up epilogue.

    ``a_format`` selects E2M1 or E4M3 block-scaled activations.
    ``output_sorted`` writes rows in routing order rather than restoring
    token/slot order. ``output_quantized`` additionally writes E2M1 or E4M3
    values to ``out`` and CDNA4-swizzled e8m0 scales to ``out_scale``;
    ``output_k`` is the physical, padding-inclusive output width.
    ``out_scale`` must reserve complete 32-row panels even for 16-row tiles.
    ``input_sorted`` consumes activation rows already ordered like
    ``sorted_token_ids``; ``source_tokens`` retains the original token bound
    used to identify padding routes. The function returns the same ``out``
    tensor supplied by the caller.

    Steps below correspond to the ``# Step N:`` comments in the body:
      1. Validate dtypes / shapes; reject unsupported options.
      2. Permute ``w1`` to the (16, 16) MFMA-tile layout (see
         :func:`_b_preshuffle_3d`), or skip if the caller already did.
      3. Materialise ``num_valid_ids`` as a device tensor when the caller
         passed a Python scalar.
      4. Launch the GEMM kernel.

    Layout contract: see the module docstring.

    Unsupported (raises ``NotImplementedError``):
      ``quant_type != per_1x32``, ``activation != Silu``,
      ``splitk not in {0, 1, None}``, non-empty ``sorted_weights``,
      ``dst_type != bfloat16``.

    Accepted-but-ignored, kept for signature compatibility with upstream
    dispatchers: ``kernelName``, ``w2``, ``use_non_temporal_load``.
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
            "invoke_gluon_mxfp4_moe_stage1: stage-1 route weighting is not "
            "implemented; pass sorted_weights=None or an empty tensor"
        )
    if dst_type is not None and dst_type != torch.bfloat16:
        raise NotImplementedError(
            "invoke_gluon_mxfp4_moe_stage1: only dst_type=torch.bfloat16 is "
            f"supported, got dst_type={dst_type!r}"
        )
    del kernelName, use_non_temporal_load, w2

    if a_format not in ("e2m1", "e4m3"):
        raise ValueError(f"stage1 a_format must be e2m1 or e4m3, got {a_format}")
    if output_format not in ("e2m1", "e4m3"):
        raise ValueError(
            f"stage1 output_format must be e2m1 or e4m3, got {output_format}"
        )
    expected_a_dtype = torch.float8_e4m3fn if a_format == "e4m3" else torch.uint8
    if hidden_states.dtype != expected_a_dtype:
        raise TypeError(
            f"{a_format} stage1 input must use {expected_a_dtype}, "
            f"got {hidden_states.dtype}"
        )
    assert w1.dtype == torch.uint8, f"w1 must be packed fp4x2 (uint8), got {w1.dtype}"
    # Step 2: shuffle w1 to the MFMA-tile layout the kernel expects.
    if not b_preshuffled:
        w1 = _b_preshuffle_3d(w1)
    if b_layout not in (
        B_LAYOUT_MFMA16.value,
        B_LAYOUT_GDOT128.value,
        B_LAYOUT_N16.value,
    ):
        raise ValueError(f"unknown stage1 W13 layout {b_layout}")
    if b_layout != B_LAYOUT_MFMA16.value and not b_preshuffled:
        raise ValueError("gdot128 and N16 W13 banks must be passed preshuffled")
    if b_layout == B_LAYOUT_N16.value and a_format != "e2m1":
        raise ValueError("N16 W13 banks take E2M1 activations only")
    assert w1_scale is not None and w1_scale.dtype == torch.uint8, (
        "w1_scale must be a uint8 e8m0 tensor, got "
        f"{None if w1_scale is None else w1_scale.dtype}"
    )
    assert a1_scale is not None and a1_scale.dtype == torch.uint8, (
        "a1_scale must be a uint8 e8m0 tensor, got "
        f"{None if a1_scale is None else a1_scale.dtype}"
    )
    if output_quantized:
        if not output_sorted:
            raise ValueError("quantized stage1 output requires sorted row order")
        expected_out_dtype = (
            torch.float8_e4m3fn if output_format == "e4m3" else torch.uint8
        )
        if out.dtype != expected_out_dtype:
            raise TypeError(
                f"{output_format} stage1 output must use {expected_out_dtype}, "
                f"got {out.dtype}"
            )
        if out_scale is None or out_scale.dtype != torch.uint8:
            raise TypeError("quantized stage1 out_scale must be a uint8 tensor")
    elif out.dtype != torch.bfloat16:
        raise TypeError(f"stage1 out must be bfloat16, got {out.dtype}")
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
    M_padded, D_stored = hidden_states.shape
    a_div = 1 if a_format == "e4m3" else 2
    K = D_stored * a_div
    assert (
        w1.dim() == 3
    ), f"w1 must be 3-D (E, 2*I_r, K_packed), got shape {tuple(w1.shape)}"
    E_w, two_Ir, D_packed_w = w1.shape
    assert (
        D_packed_w * 2 == K
    ), f"w1 packed K ({D_packed_w * 2}) must match activation K ({K})"
    assert two_Ir % 2 == 0, f"w1.shape[1] (= 2*I_r) must be even, got {two_Ir}"
    I_r = two_Ir // 2
    N = 2 * I_r
    EM = sorted_token_ids.shape[0]
    output_k = I_r if output_k is None else int(output_k)
    # A gdot128 K tile is 128 packed FP4 bytes, so its logical activation
    # width rounds up in 256-column increments and can add at most 224 columns.
    if output_k < I_r or output_k - I_r >= 256 or output_k % 32:
        raise ValueError(
            "stage1 output_k must be a multiple of 32 in "
            f"[{I_r}, {I_r + 256}), got {output_k}"
        )
    if output_quantized:
        output_div = 1 if output_format == "e4m3" else 2
        output_scale_cols = output_k // 32
        if output_scale_cols % CDNA4_SCALE_K_BLOCK:
            raise ValueError(
                "quantized stage1 output_k must be divisible by "
                f"{32 * CDNA4_SCALE_K_BLOCK} for complete CDNA4 scale panels; "
                f"got {output_k}"
            )
        if (
            out.dim() != 2
            or out.shape[0] < EM
            or out.shape[1] != output_k // output_div
        ):
            raise ValueError(
                "quantized stage1 out must have shape "
                f"(>= {EM}, {output_k // output_div}), got {tuple(out.shape)}"
            )
        assert out_scale is not None
        # Swizzled stores span a full panel even when only half its rows are used.
        output_scale_rows = triton.cdiv(EM, CDNA4_SCALE_N_BLOCK) * CDNA4_SCALE_N_BLOCK
        if (
            out_scale.dim() != 2
            or out_scale.shape[0] < output_scale_rows
            or out_scale.shape[1] != output_k // 32
        ):
            raise ValueError(
                "quantized stage1 out_scale must have shape "
                f"(>= {output_scale_rows}, {output_k // 32}) for complete CDNA4 "
                f"scale panels, got {tuple(out_scale.shape)}"
            )
        out_2d = out
    elif out.dim() == 3:
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
    out_scale_ptr = out_scale if output_quantized else a1_scale
    K_scale = K // 32
    assert a1_scale.dim() == 2 and a1_scale.shape[1] == K_scale, (
        f"a1_scale.shape {tuple(a1_scale.shape)} must be "
        f"(M_padded_aligned, K // 32 = {K_scale})"
    )
    assert w1_scale.shape == (E_w, N, K_scale), (
        f"w1_scale.shape {tuple(w1_scale.shape)} must be (E, N, K//32) = "
        f"({E_w}, {N}, {K_scale})"
    )
    num_tokens = M_padded if source_tokens is None else int(source_tokens)
    if num_tokens <= 0 or num_tokens > M_padded:
        raise ValueError(
            f"stage1 source_tokens must be in [1, {M_padded}], got {num_tokens}"
        )
    if input_sorted and M_padded < EM:
        raise ValueError(
            f"sorted stage1 input has {M_padded} rows but routing requires {EM}"
        )
    del M_padded
    # Activation rows are read with 32-bit offsets from the activation base
    # (AMD buffer loads), so the span the kernel indexes must fit: the token
    # rows of a token-order input, the EM sorted rows of a sorted one (sorted
    # input always takes the general kernel here). The general kernel reads
    # scale panels with 32-bit offsets from the scale base too.
    activation_rows = EM if input_sorted else num_tokens
    if (
        activation_rows * hidden_states.stride(0) >= 2**31
        or a1_scale.shape[0] * a1_scale.shape[1] >= 2**31
    ):
        raise ValueError(
            "stage1 activation rows and scales must span less than 2^31 bytes, "
            f"got {activation_rows} rows of stride {hidden_states.stride(0)} and "
            f"scales {tuple(a1_scale.shape)}"
        )

    # Step 3: materialise scalar / None args as device tensors so the
    # kernel can just `gl.load` from them.
    if torch.is_tensor(num_valid_ids):
        num_valid_ids_ptr = num_valid_ids
    else:
        num_valid_ids_ptr = torch.tensor(
            [int(num_valid_ids)], dtype=torch.int32, device=hidden_states.device
        )
    BLOCK_M = int(block_m)
    if BLOCK_M not in (16, 32, 64, 128) or (BLOCK_M == 16 and a_format != "e2m1"):
        raise ValueError(
            "stage1 block_m must be 32, 64 or 128, or 16 for E2M1; "
            f"got block_m={BLOCK_M}, a_format={a_format}"
        )
    BLOCK_N = 128
    BLOCK_K = 256
    GROUP_SIZE_M = 1
    NUM_WARPS = num_warps
    num_pid_m = triton.cdiv(EM, BLOCK_M)
    output_block_n = BLOCK_N // 2 if b_layout != B_LAYOUT_MFMA16.value else BLOCK_N
    num_pid_n = triton.cdiv(I_r, output_block_n)
    grid = (num_pid_m * num_pid_n,)

    # Keep the public tensor typed as FP8, but pass its bit-identical uint8 view
    # to the kernel; ``mfma_scaled(..., a_format="e4m3")`` interprets the raw
    # bytes after the synchronous HBM-to-LDS load.
    kernel_hidden_states = (
        hidden_states.view(torch.uint8) if a_format == "e4m3" else hidden_states
    )
    stride_am = kernel_hidden_states.stride(0)
    stride_ak = kernel_hidden_states.stride(1)
    stride_be = w1.stride(0)
    stride_cm = out_2d.stride(0)
    stride_cn = out_2d.stride(1)
    stride_bse_e = w1_scale.stride(0)
    stride_se_n_pad = a1_scale.shape[1]
    K_packed_total = K // 2

    use_e4m3_async = (
        a_format == "e4m3"
        and b_layout == B_LAYOUT_GDOT128.value
        and K % 256 == 0
        and I_r % 64 == 0
        and input_sorted
        and output_sorted
        and output_quantized
        and output_format == "e4m3"
        and situ_linear_beta is not None
        and stride_ak == 1
    )
    if use_e4m3_async:
        gluon_mxfp4_moe_stage1_e4m3_async_kernel[grid](
            kernel_hidden_states,
            w1,
            out_2d,
            out_scale_ptr,
            a1_scale,
            w1_scale,
            sorted_token_ids,
            sorted_expert_ids,
            num_valid_ids_ptr,
            EM,
            num_tokens,
            topk,
            stride_be,
            stride_cm,
            stride_cn,
            stride_bse_e,
            stride_se_n_pad,
            STRIDE_AM=stride_am,
            BLOCK_M=BLOCK_M,
            K_PACKED_TOTAL=K_packed_total,
            I_r=I_r,
            OUTPUT_K=output_k,
            SWIGLU_ALPHA=float(swiglu_alpha),
            SWIGLU_LIMIT=float(swiglu_limit),
            SWIGLU_BETA=float(swiglu_beta),
            SITU_LINEAR_BETA=float(situ_linear_beta),
            num_warps=4,
        )
        return out

    if wide_tile:
        if not (
            a_format == "e2m1"
            and output_format == "e2m1"
            and b_layout == B_LAYOUT_GDOT128.value
            and output_sorted
            and output_quantized
            and not input_sorted
            and BLOCK_M in (64, 128)
            and NUM_WARPS == BLOCK_M // 32
            and K % 256 == 0
            and K >= 512
            and I_r % 128 == 0
            and stride_ak == 1
            and stride_cn == 1
        ):
            raise ValueError(
                "the wide stage-1 tile serves E2M1 in, sorted MXFP4 out, GDOT128 "
                "weights, 64- or 128-row sort blocks with block_m // 32 warps, "
                "K a multiple of 256 (at least 512), I a multiple of 128 and unit "
                "column strides; got "
                f"a_format={a_format}, output_format={output_format}, "
                f"b_layout={b_layout}, output_sorted={output_sorted}, "
                f"output_quantized={output_quantized}, input_sorted={input_sorted}, "
                f"block_m={BLOCK_M}, num_warps={NUM_WARPS}, K={K}, I={I_r}, "
                f"strides ({stride_ak}, {stride_cn})"
            )
        # Every 64 rows of a sort block are one tile with the block's expert.
        gluon_mxfp4_moe_stage1_e2m1_wide_gfx950[(triton.cdiv(EM, 64) * (I_r // 128),)](
            kernel_hidden_states,
            w1,
            out_2d,
            out_scale_ptr,
            a1_scale,
            w1_scale,
            sorted_token_ids,
            sorted_expert_ids,
            num_valid_ids_ptr,
            EM,
            num_tokens,
            stride_am,
            stride_be,
            stride_cm,
            stride_bse_e,
            stride_se_n_pad,
            K_PACKED_TOTAL=K_packed_total,
            SORT_BLOCK_M=BLOCK_M,
            I_r=I_r,
            OUTPUT_K=output_k,
            SWIGLU_ALPHA=float(swiglu_alpha),
            SWIGLU_LIMIT=float(swiglu_limit),
            SWIGLU_BETA=float(swiglu_beta),
            DO_SITU=situ_linear_beta is not None,
            SITU_LINEAR_BETA=(
                1.0 if situ_linear_beta is None else float(situ_linear_beta)
            ),
            num_warps=4,
        )
        return out

    # Step 4: launch the GEMM. One CTA per (M-tile, N-tile); the per-CTA
    # expert id is ``sorted_expert_ids[pid_m]``. Interleaved W13 tiles produce
    # BLOCK_N / 2 outputs; separate gate/up layouts produce BLOCK_N outputs.
    gluon_mxfp4_moe_stage1_kernel[grid](
        kernel_hidden_states,
        w1,
        out_2d,
        out_scale_ptr,
        a1_scale,
        w1_scale,
        sorted_token_ids,
        sorted_expert_ids,
        num_valid_ids_ptr,
        N,
        K,
        EM,
        num_tokens,
        topk,
        stride_am,
        stride_ak,
        stride_be,
        stride_cm,
        stride_cn,
        stride_bse_e,
        stride_se_n_pad,
        K_PACKED_TOTAL=K_packed_total,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        NUM_WARPS=NUM_WARPS,
        I_r=I_r,
        OUTPUT_K=output_k,
        B_LAYOUT=int(b_layout),
        SWIGLU_ALPHA=float(swiglu_alpha),
        SWIGLU_LIMIT=float(swiglu_limit),
        SWIGLU_BETA=float(swiglu_beta),
        DO_SITU=situ_linear_beta is not None,
        SITU_LINEAR_BETA=(1.0 if situ_linear_beta is None else float(situ_linear_beta)),
        OUTPUT_SORTED=bool(output_sorted),
        OUTPUT_QUANTIZED=bool(output_quantized),
        A_FORMAT=a_format,
        OUTPUT_FP8=output_format == "e4m3",
        INPUT_SORTED=bool(input_sorted),
        NUM_XCDS=_NUM_XCDS,
        # Narrow tiles finish their A tile before the next one is issued, so
        # issuing it ahead of the MFMAs hides its latency. The 128-row tile
        # already overlaps it with its second half.
        EARLY_A=BLOCK_M <= 64,
        num_warps=NUM_WARPS,
        # Interleaving the MFMAs with the next K tile's loads makes the 128-row
        # E2M1 tile faster. Narrower tiles lose occupancy to the extra registers
        # and do not gain.
        **(
            {"schedule_hint": "mfma-schedule"}
            if BLOCK_M == 128 and NUM_WARPS == 4 and a_format == "e2m1"
            else {}
        ),
    )
    return out
