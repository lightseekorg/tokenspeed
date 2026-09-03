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
  tile k's MFMAs.

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
  hidden_states  (M_padded, K_packed)        packed MXFP4 or MXFP8
  w1             (E, 2 * I_r, K_packed)      uint8, (16, 16) shuffled
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

from typing import Optional  # noqa: F401  (kept for downstream type hints)

import torch
from tokenspeed_kernel_amd._triton import cdna4_async_copy, gl, gluon, triton
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.quantize_gluon import (
    _mxfp4_quantize_tile,
    _mxfp4_store_cdna4_scale,
    _mxfp8_quantize_tile,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.scale_layout import (
    CDNA4_SCALE_K_BLOCK,
)


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
        # E4M3 MFMAs share each A row across N-partitioned waves. Finish
        # reading the old tile before any wave reuses this LDS slot.
        gl.barrier()
        smem_a_tile.store(values)
        gl.barrier()


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
):
    primary = _load_b_data_vgpr(b_base_ptr, offs_b_primary, b_data_k_off)
    if INTERLEAVED:
        primary_scale = _load_b_scale_vgpr(
            b_scales_ptr,
            b_scale_offs_primary,
            tile_k,
            b_scale_k_step,
        )
        return primary, primary, primary_scale, primary_scale
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
):
    block_m: gl.constexpr = acc.shape[0]
    split_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[4, 16],
        warps_per_cta=[gl.num_warps(), 1],
        order=[1, 0],
    )
    acc = gl.convert_layout(acc, split_layout)
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
    NUM_WARPS: gl.constexpr = 4
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
        # The copy wait is wave-local. A is loaded across four waves but each
        # N-partitioned MFMA wave consumes every row, so make all LDS writes
        # visible before the cross-wave reads.
        gl.barrier()
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
        # All N-partitioned waves must finish reading the current A slot
        # before the next iteration reuses it as the async-copy destination.
        gl.barrier()

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
    gl.barrier()
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


@gluon.jit
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
    B_GDOT128: gl.constexpr,
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
):
    """Stage 1 kernel with per-token A gather and fused SwiGLU.

    Tile config: ``BLOCK_M x 128 x 256`` (M x N x K), with ``BLOCK_M``
    in ``{32, 64, 128}``; the 32-row variant is E4M3-only. MFMA target:
    ``v_mfma_scale_f32_16x16x128_f8f6f4`` (gfx950, AMDMFMALayout
    version=4). Each CTA owns ``BLOCK_N / 2`` output columns for interleaved
    GDOT128 W13 or ``BLOCK_N`` columns for separate gate/up layouts.

    For GDOT128 W13, each K tile consumes the physical interleaved gate/up
    rows in one ``BLOCK_N`` MFMA accumulator per 32-row quarter. Other
    layouts retain separate gate and up accumulators. The K loop alternates
    two A-data LDS slots while loading A scales and B operands into VGPRs.

    The epilogue splits interleaved accumulators into gate/up pairs, applies
    the activation, and stores the resulting output tile in ``[EM, I_r]``.
    """

    gl.static_assert(
        BLOCK_M == 64 or BLOCK_M == 128 or (BLOCK_M == 32 and A_FORMAT == "e4m3"),
        "stage1 kernel requires BLOCK_M in {64, 128}, or 32 for E4M3",
    )
    gl.static_assert(BLOCK_N == 128, "stage1 kernel requires BLOCK_N=128")
    gl.static_assert(BLOCK_K == 256, "stage1 kernel requires BLOCK_K=256")
    gl.static_assert(
        NUM_WARPS == BLOCK_M // 32 or (A_FORMAT == "e4m3" and NUM_WARPS == 8),
        "stage1 kernel requires the native E2M1 wave count or eight E4M3 waves",
    )
    gl.static_assert(not OUTPUT_QUANTIZED or OUTPUT_SORTED)

    SCALE_GROUP: gl.constexpr = 32
    A_DIV: gl.constexpr = 1 if A_FORMAT == "e4m3" else 2
    BLOCK_K_A: gl.constexpr = BLOCK_K // A_DIV
    BLOCK_K_B: gl.constexpr = BLOCK_K // 2
    BLOCK_K_SCALE: gl.constexpr = BLOCK_K // SCALE_GROUP
    NUM_BUFFERS: gl.constexpr = 2
    GROUP_MFMA_M: gl.constexpr = 32

    pid = gl.program_id(axis=0)
    num_pid_m = gl.cdiv(EM, BLOCK_M)
    OUTPUT_BLOCK_N: gl.constexpr = BLOCK_N // 2 if B_GDOT128 else BLOCK_N
    # Grid over the activated ``I_r`` columns. Interleaved W13 consumes a
    # physical BLOCK_N tile for each OUTPUT_BLOCK_N result tile; separate
    # layouts issue gate and up MFMAs for the same output tile.
    num_pid_n = gl.cdiv(I_r, OUTPUT_BLOCK_N)
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
    if BLOCK_M == 32:
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
            shared_a_bases: gl.constexpr = []
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
    a_scale_group_stride = stride_se_n_pad * 32

    # ---- B data offsets (preshuffle-B closed form) --------------------
    # B is in the (16, 16) MFMA-tile layout described in the module
    # docstring, so the byte for logical ``(n, k_pk)`` is the closed form
    # computed below (the per-expert term is folded into ``b_base_ptr``).
    # The GDOT128 layout physically interleaves gate/up rows and is consumed
    # as one contiguous ``BLOCK_N`` stream. Other layouts retain separate
    # gate and up streams. The ``% N`` wraps below defend the degenerate
    # ``I_r < BLOCK_N`` test path.
    offs_bk = gl.arange(0, BLOCK_K_B, layout=gl.SliceLayout(1, dot_b_layout))
    if B_GDOT128:
        n_tiles: gl.constexpr = (2 * I_r) // 128
        offs_bn_primary = (
            pid_n * BLOCK_N
            + gl.arange(
                0,
                BLOCK_N,
                layout=gl.SliceLayout(0, dot_b_layout),
            )
        ) % N
        k_in = offs_bk % 128
        k_within = k_in % 16
        k_quad = (k_in // 16) % 4
        k_block = k_in // 64
        n_primary_in = offs_bn_primary % 128
        in_primary = (
            (n_primary_in // 16)[None, :] * 2048
            + k_block[:, None] * 1024
            + k_quad[:, None] * 256
            + (n_primary_in % 16)[None, :] * 16
            + k_within[:, None]
        )
        offs_b_gate = (
            (offs_bk // 128)[:, None] * n_tiles + (offs_bn_primary // 128)[None, :]
        ) * (128 * 128) + in_primary
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
    if B_GDOT128:
        b_rows_primary = (
            pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=b_n_layout_scale)
        ).to(gl.uint32)
        b_n_part_gate_scale = (
            (b_rows_primary // 32) * (K_PACKED_TOTAL * 2)
            + (b_rows_primary % 16) * 4
            + ((b_rows_primary % 32) // 16)
        )[:, None]
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
    if B_GDOT128:
        b_k_lane_offsets = ((b_k_lanes % 4) * 64 + ((b_k_lanes % 8) // 4) * 2)[None, :]
        B_SCALE_K_STEP: gl.constexpr = 256
    else:
        b_k_lane_offsets = ((b_k_lanes % 4) * 256 + (b_k_lanes // 4))[None, :]
        B_SCALE_K_STEP: gl.constexpr = 1024
    b_scale_static_offs_gate = b_n_part_gate_scale + b_k_lane_offsets
    b_off_gate = b_scale_static_offs_gate
    if B_GDOT128:
        b_off_up = b_off_gate
    else:
        b_scale_static_offs_up = b_n_part_up_scale + b_k_lane_offsets
        b_off_up = b_scale_static_offs_up

    b_scales_ptr_e = b_scales_ptr + off_experts.to(gl.int64) * stride_bse_e

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
    if B_GDOT128:
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
        B_GDOT128,
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
            B_GDOT128,
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
            B_GDOT128,
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
                B_GDOT128,
            )

        # Prefetch A[k+1] data into LDS slot1. A-scale stays direct-to-VGPR.
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
                B_GDOT128,
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
                B_GDOT128,
            )

        # Drain A[k+1] prefetch; B[k+1] should also be done by now.
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
            tile0_k + 1,
            a_scale_group_stride,
            A_SCALE_K_STEP,
        )
        if BLOCK_M >= 64:
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
            B_GDOT128,
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
            B_GDOT128,
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
                B_GDOT128,
            )

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
                B_GDOT128,
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
                B_GDOT128,
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
        if BLOCK_M >= 64:
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
            B_GDOT128,
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
            B_GDOT128,
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
                B_GDOT128,
            )

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
                B_GDOT128,
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
                B_GDOT128,
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
        if BLOCK_M >= 64:
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
        B_GDOT128,
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
            B_GDOT128,
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
            B_GDOT128,
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
            B_GDOT128,
        )

    if B_GDOT128:
        acc_swiglu_group0 = _apply_interleaved_gate_activation(
            gate_acc_group0,
            DO_SITU,
            SWIGLU_ALPHA,
            SWIGLU_LIMIT,
            SWIGLU_BETA,
            SITU_LINEAR_BETA,
            OUTPUT_BLOCK_N,
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
        if B_GDOT128:
            acc_swiglu_group1 = _apply_interleaved_gate_activation(
                gate_acc_group1,
                DO_SITU,
                SWIGLU_ALPHA,
                SWIGLU_LIMIT,
                SWIGLU_BETA,
                SITU_LINEAR_BETA,
                OUTPUT_BLOCK_N,
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
        if B_GDOT128:
            acc_swiglu_group2 = _apply_interleaved_gate_activation(
                gate_acc_group2,
                DO_SITU,
                SWIGLU_ALPHA,
                SWIGLU_LIMIT,
                SWIGLU_BETA,
                SITU_LINEAR_BETA,
                OUTPUT_BLOCK_N,
            )
            acc_swiglu_group3 = _apply_interleaved_gate_activation(
                gate_acc_group3,
                DO_SITU,
                SWIGLU_ALPHA,
                SWIGLU_LIMIT,
                SWIGLU_BETA,
                SITU_LINEAR_BETA,
                OUTPUT_BLOCK_N,
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
    b_gdot128: bool = False,
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
):
    """Host-side launcher for Gluon MXFP4 MoE stage 1.

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
        if (
            out_scale.dim() != 2
            or out_scale.shape[0] < EM
            or out_scale.shape[1] != output_k // 32
        ):
            raise ValueError(
                "quantized stage1 out_scale must have shape "
                f"(>= {EM}, {output_k // 32}), got {tuple(out_scale.shape)}"
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

    # Step 3: materialise scalar / None args as device tensors so the
    # kernel can just `gl.load` from them.
    if torch.is_tensor(num_valid_ids):
        num_valid_ids_ptr = num_valid_ids
    else:
        num_valid_ids_ptr = torch.tensor(
            [int(num_valid_ids)], dtype=torch.int32, device=hidden_states.device
        )
    BLOCK_M = int(block_m)
    if BLOCK_M not in (32, 64, 128) or (BLOCK_M == 32 and a_format != "e4m3"):
        raise ValueError(
            "stage1 block_m must be 64 or 128, or 32 for E4M3; "
            f"got block_m={BLOCK_M}, a_format={a_format}"
        )
    BLOCK_N = 128
    BLOCK_K = 256
    GROUP_SIZE_M = 1
    NUM_WARPS = 8 if a_format == "e4m3" else BLOCK_M // 32
    num_pid_m = triton.cdiv(EM, BLOCK_M)
    output_block_n = BLOCK_N // 2 if b_gdot128 else BLOCK_N
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
        and b_gdot128
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
        B_GDOT128=bool(b_gdot128),
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
        num_warps=NUM_WARPS,
    )
    return out
