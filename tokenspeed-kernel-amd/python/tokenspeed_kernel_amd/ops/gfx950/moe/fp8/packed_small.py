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

"""Small-batch gfx950 FP8 routed MoE with optional reduction splitting.

A split factor of one pairs gate/up and fuses activation; larger factors
round partition results through FP16 before FP32 combination and activation.
All schedules share N16/K32 weights, the K256 pipeline and BF16 route outputs.
"""

from __future__ import annotations

from tokenspeed_kernel_amd._triton import gl, gluon
from tokenspeed_kernel_amd.ops.gfx950.moe.fp8.intermediate_quant import (
    quantize_intermediate_tile,
)


def _launch_metadata(grid, kernel, args):
    return {"name": kernel.name}


def packed_fp8_split_count(
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    topk: int,
    compute_units: int,
    num_experts_local: int,
    num_experts_total: int,
) -> int:
    """Estimate a small-batch split factor from output parallelism.

    Global routes bound independent output work and limit split aggressiveness.
    Static expert ownership estimates local work under balanced routing; skew
    and repeated routes can change useful work. When the global budget supports
    one partition, sparse estimated local work permits only a two-way split.
    The nonsplit producer pairs gate/up in one workgroup, whereas the split
    producer launches them independently, doubling its output tile count.
    Two workgroups per compute unit is a tuning budget, not a measured occupancy
    guarantee. Keep at least two K256 tiles per partition for the pipeline.
    """
    if num_tokens <= 0 or num_tokens > 32:
        return 1
    estimated_routes = (
        num_tokens * topk * num_experts_local + num_experts_total - 1
    ) // num_experts_total
    output_tiles = (2 * intermediate_size + 127) // 128
    global_workgroups = num_tokens * topk * output_tiles
    estimated_local_workgroups = estimated_routes * output_tiles
    target = 2 * compute_units
    budget = target // global_workgroups
    if budget == 1 and estimated_local_workgroups <= compute_units:
        budget = 2
    if budget < 2:
        return 1
    for split_count in (8, 4, 2):
        if split_count <= budget and hidden_size % (512 * split_count) == 0:
            return split_count
    return 1


@gluon.constexpr_function
def _fp8_weight_n16k32_carrier_layout():
    """Load packed FP8 bytes in physical K16 vectors before MFMA conversion."""
    return gl.DistributedLinearLayout(
        reg_bases=[[1], [2], [4], [8], [1024], [8192]],
        lane_bases=[[16], [32], [64], [128], [256], [512]],
        warp_bases=[[2048], [4096]],
        block_bases=[],
        shape=[16384],
    )


@gluon.jit
def _stage1_accumulate_k128(
    a_dot,
    gate,
    up,
    gate_acc,
    up_acc,
    gate_scale,
    up_scale,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    MFMA: gl.constexpr,
    PAIRED: gl.constexpr,
):
    dot_b: gl.constexpr = gl.DotOperandLayout(1, MFMA, k_width=16)
    gate = gate.reshape([8, 8, 16, 16]).permute([1, 3, 0, 2]).reshape([128, 128])
    if PAIRED:
        up = up.reshape([8, 8, 16, 16]).permute([1, 3, 0, 2]).reshape([128, 128])
    gate_dot = gl.convert_layout(gate, dot_b).to(gl.float8e4nv, bitcast=True)
    if PAIRED:
        up_dot = gl.convert_layout(up, dot_b).to(gl.float8e4nv, bitcast=True)
    gate_tile = gl.amd.cdna4.mfma_scaled(
        a_dot,
        None,
        "e4m3",
        gate_dot,
        None,
        "e4m3",
        gl.zeros([BLOCK_M, BLOCK_N], gl.float32, MFMA),
    )
    if PAIRED:
        up_tile = gl.amd.cdna4.mfma_scaled(
            a_dot,
            None,
            "e4m3",
            up_dot,
            None,
            "e4m3",
            gl.zeros([BLOCK_M, BLOCK_N], gl.float32, MFMA),
        )
    gate_acc = gl.fma(gate_tile, gate_scale[:, None], gate_acc)
    if PAIRED:
        up_acc = gl.fma(up_tile, up_scale[:, None], up_acc)
    return gate_acc, up_acc


@gluon.jit
def _stage1_load_scale_pair(
    x_scale_ptr,
    w_scale_ptr,
    tokens_m,
    valid_m,
    scale_base,
    pid_n,
    tile,
    SCALE_STRIDE: gl.constexpr,
    I: gl.constexpr,
    BLOCK_N: gl.constexpr,
    SCALE: gl.constexpr,
    MFMA: gl.constexpr,
    PAIRED: gl.constexpr,
):
    first = 2 * tile
    pair = gl.arange(0, 2, gl.SliceLayout(0, SCALE))
    a_pair = gl.amd.cdna3.buffer_load(
        x_scale_ptr,
        tokens_m[:, None] * SCALE_STRIDE + first + pair[None, :],
        valid_m[:, None],
        other=0.0,
    )
    gate_pair = gl.amd.cdna3.buffer_load(
        w_scale_ptr,
        scale_base + pid_n * SCALE_STRIDE + first + pair,
        cache=".cg",
    )
    gate0, gate1 = gl.split(a_pair * gate_pair[None, :])
    if PAIRED:
        up_pair = gl.amd.cdna3.buffer_load(
            w_scale_ptr,
            scale_base + (I // BLOCK_N + pid_n) * SCALE_STRIDE + first + pair,
            cache=".cg",
        )
        up0, up1 = gl.split(a_pair * up_pair[None, :])
    else:
        up0, up1 = gate0, gate1
    return (
        gl.convert_layout(gate0, gl.SliceLayout(1, MFMA)),
        gl.convert_layout(gate1, gl.SliceLayout(1, MFMA)),
        gl.convert_layout(up0, gl.SliceLayout(1, MFMA)),
        gl.convert_layout(up1, gl.SliceLayout(1, MFMA)),
    )


@gluon.jit
def _stage1_load_packed_b(
    gate_base, up_base, packed_offsets, k_group, PAIRED: gl.constexpr
):
    offset = k_group * 4096
    gate0 = gl.amd.cdna3.buffer_load(gate_base, packed_offsets + offset, cache=".cg")
    gate1 = gl.amd.cdna3.buffer_load(
        gate_base, packed_offsets + offset + 2048, cache=".cg"
    )
    if PAIRED:
        up0 = gl.amd.cdna3.buffer_load(up_base, packed_offsets + offset, cache=".cg")
        up1 = gl.amd.cdna3.buffer_load(
            up_base, packed_offsets + offset + 2048, cache=".cg"
        )
    else:
        up0, up1 = gate0, gate1
    return gate0, gate1, up0, up1


@gluon.jit
def _stage1_load_packed_b_half(
    gate_base, up_base, packed_offsets, k_group, half, PAIRED: gl.constexpr
):
    offset = k_group * 4096 + half * 2048
    gate = gl.amd.cdna3.buffer_load(gate_base, packed_offsets + offset, cache=".cg")
    if PAIRED:
        up = gl.amd.cdna3.buffer_load(up_base, packed_offsets + offset, cache=".cg")
    else:
        up = gate
    return gate, up


@gluon.jit
def _small_load_activation(
    base, row_offsets, column_offsets, tile_offset, SPLIT: gl.constexpr
):
    if SPLIT:
        return gl.amd.cdna3.buffer_load(
            base, row_offsets + column_offsets + tile_offset
        )
    # Keep the original paired path's pointer-first address expression.
    return gl.load(base + row_offsets + tile_offset + column_offsets)


@gluon.jit(
    launch_metadata=_launch_metadata,
    do_not_specialize=("num_tokens", "em", "swiglu_limit"),
)
def gluon_fp8_packed_small_stage1_gfx950(
    x_ptr,
    x_scale_ptr,
    w_ptr,
    w_scale_ptr,
    inter_ptr,
    inter_scale_ptr,
    sorted_ids_ptr,
    sorted_experts_ptr,
    num_valid_ptr,
    num_tokens,
    em,
    swiglu_limit,
    K: gl.constexpr,
    I: gl.constexpr,
    TOPK: gl.constexpr,
    SPLIT_COUNT: gl.constexpr,
    USE_RECIPROCAL: gl.constexpr,
):
    BLOCK_M: gl.constexpr = 16
    BLOCK_N: gl.constexpr = 128
    BLOCK_K: gl.constexpr = 256
    PAIRED: gl.constexpr = SPLIT_COUNT == 1
    TILES: gl.constexpr = K // (BLOCK_K * SPLIT_COUNT)
    gl.static_assert(
        SPLIT_COUNT == 1 or SPLIT_COUNT == 2 or SPLIT_COUNT == 4 or SPLIT_COUNT == 8
    )
    gl.static_assert(K % (512 * SPLIT_COUNT) == 0)
    gl.static_assert(I % 128 == 0)
    if PAIRED:
        pid = gl.program_id(0)
        pid_n = pid % (I // BLOCK_N)
        pid_m = pid // (I // BLOCK_N)
        first_tile = 0
    else:
        pid_n = gl.program_id(0)
        pid_m = gl.program_id(1)
        first_tile = gl.program_id(2) * TILES
    if pid_m * BLOCK_M >= gl.load(num_valid_ptr):
        return
    expert = gl.load(sorted_experts_ptr + pid_m)
    if expert < 0:
        return

    mfma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 128],
        transposed=True,
        warps_per_cta=[1, 4],
    )
    dot_a: gl.constexpr = gl.DotOperandLayout(0, mfma, k_width=16)
    load_a: gl.constexpr = gl.BlockedLayout([1, 16], [4, 16], [4, 1], [1, 0])
    load_b: gl.constexpr = _fp8_weight_n16k32_carrier_layout()
    shared_a: gl.constexpr = gl.SwizzledSharedLayout(16, 2, 8, [1, 0])
    a_smem = gl.allocate_shared_memory(gl.uint8, [BLOCK_M, BLOCK_K], shared_a)

    rows_a = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, gl.SliceLayout(1, load_a))
    ids_a = gl.load(sorted_ids_ptr + rows_a, rows_a < em, other=num_tokens)
    tokens_a = ids_a & 0xFFFFFF
    valid_a = tokens_a < num_tokens
    safe_tokens_a = gl.where(valid_a, tokens_a, 0)
    ka = gl.arange(0, BLOCK_K, gl.SliceLayout(0, load_a))
    packed_index = gl.arange(0, 16384, load_b)
    packed_offsets = (pid_n * 8 + packed_index // 2048) * (K * 16) + packed_index % 2048
    packed_offsets = gl.multiple_of(packed_offsets, 16)
    expert_weights = gl.multiple_of(w_ptr + expert.to(gl.int64) * (2 * I * K), 16)
    packed_gate_base = expert_weights
    packed_up_base = gl.multiple_of(packed_gate_base + I * K, 16)

    rows_m = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, gl.SliceLayout(1, mfma))
    ids_m = gl.load(sorted_ids_ptr + rows_m, rows_m < em, other=num_tokens)
    tokens_m = ids_m & 0xFFFFFF
    valid_m = tokens_m < num_tokens
    scale_layout: gl.constexpr = gl.BlockedLayout([1, 2], [16, 4], [1, 4], [0, 1])
    scale_tokens = gl.convert_layout(tokens_m, gl.SliceLayout(1, scale_layout))
    scale_valid = gl.convert_layout(valid_m, gl.SliceLayout(1, scale_layout))
    gate_acc = gl.zeros([BLOCK_M, BLOCK_N], gl.float32, mfma)
    up_acc = gl.zeros([BLOCK_M, BLOCK_N], gl.float32, mfma)

    # Keep two K128 scale groups per K256 tile.
    scale_stride: gl.constexpr = K // 128
    scale_base = expert * (2 * I // BLOCK_N) * scale_stride
    gl.static_assert(K >= 512 and K % 512 == 0)
    (
        gate_scale_cur0,
        gate_scale_cur1,
        up_scale_cur0,
        up_scale_cur1,
    ) = _stage1_load_scale_pair(
        x_scale_ptr,
        w_scale_ptr,
        scale_tokens,
        scale_valid,
        scale_base,
        pid_n,
        first_tile,
        scale_stride,
        I,
        BLOCK_N,
        scale_layout,
        mfma,
        PAIRED,
    )
    (
        gate_scale_next0,
        gate_scale_next1,
        up_scale_next0,
        up_scale_next1,
    ) = _stage1_load_scale_pair(
        x_scale_ptr,
        w_scale_ptr,
        scale_tokens,
        scale_valid,
        scale_base,
        pid_n,
        first_tile + 1,
        scale_stride,
        I,
        BLOCK_N,
        scale_layout,
        mfma,
        PAIRED,
    )
    a_rows = safe_tokens_a[:, None] * K
    a_first = _small_load_activation(
        x_ptr, a_rows, ka[None, :], first_tile * BLOCK_K, not PAIRED
    )
    gate_cur0, gate_cur1, up_cur0, up_cur1 = _stage1_load_packed_b(
        packed_gate_base, packed_up_base, packed_offsets, first_tile, PAIRED
    )
    a_smem.store(a_first)
    a_next = _small_load_activation(
        x_ptr, a_rows, ka[None, :], (first_tile + 1) * BLOCK_K, not PAIRED
    )
    gl.barrier()
    a_dot_cur0 = a_smem.slice(0, 128, 1).load(dot_a).to(gl.float8e4nv, bitcast=True)
    a_dot_cur1 = a_smem.slice(128, 128, 1).load(dot_a).to(gl.float8e4nv, bitcast=True)

    # Share the K256 pipeline, retaining the paired and single-projection
    # prefetch positions. Drain two tiles without crossing a split boundary.
    for pair in range(TILES // 2 - 1):
        for half in gl.static_range(2):
            tile = first_tile + pair * 2 + half
            if not PAIRED:
                gate_next0, gate_next1, up_next0, up_next1 = _stage1_load_packed_b(
                    packed_gate_base, packed_up_base, packed_offsets, tile + 1, PAIRED
                )
            gl.barrier()
            a_smem.store(a_next)
            a_after = _small_load_activation(
                x_ptr, a_rows, ka[None, :], (tile + 2) * BLOCK_K, not PAIRED
            )
            if PAIRED:
                (
                    gate_scale_after0,
                    gate_scale_after1,
                    up_scale_after0,
                    up_scale_after1,
                ) = _stage1_load_scale_pair(
                    x_scale_ptr,
                    w_scale_ptr,
                    scale_tokens,
                    scale_valid,
                    scale_base,
                    pid_n,
                    tile + 2,
                    scale_stride,
                    I,
                    BLOCK_N,
                    scale_layout,
                    mfma,
                    PAIRED,
                )
            gate_acc, up_acc = _stage1_accumulate_k128(
                a_dot_cur0,
                gate_cur0,
                up_cur0,
                gate_acc,
                up_acc,
                gate_scale_cur0,
                up_scale_cur0,
                BLOCK_M,
                BLOCK_N,
                mfma,
                PAIRED,
            )
            if PAIRED:
                gate_next0, up_next0 = _stage1_load_packed_b_half(
                    packed_gate_base,
                    packed_up_base,
                    packed_offsets,
                    tile + 1,
                    0,
                    PAIRED,
                )
            gate_acc, up_acc = _stage1_accumulate_k128(
                a_dot_cur1,
                gate_cur1,
                up_cur1,
                gate_acc,
                up_acc,
                gate_scale_cur1,
                up_scale_cur1,
                BLOCK_M,
                BLOCK_N,
                mfma,
                PAIRED,
            )
            if PAIRED:
                gate_next1, up_next1 = _stage1_load_packed_b_half(
                    packed_gate_base,
                    packed_up_base,
                    packed_offsets,
                    tile + 1,
                    1,
                    PAIRED,
                )
            gl.barrier()
            a_dot_cur0 = (
                a_smem.slice(0, 128, 1).load(dot_a).to(gl.float8e4nv, bitcast=True)
            )
            a_dot_cur1 = (
                a_smem.slice(128, 128, 1).load(dot_a).to(gl.float8e4nv, bitcast=True)
            )
            a_next = a_after
            gate_cur0, gate_cur1, up_cur0, up_cur1 = (
                gate_next0,
                gate_next1,
                up_next0,
                up_next1,
            )
            gate_scale_cur0, gate_scale_cur1 = gate_scale_next0, gate_scale_next1
            up_scale_cur0, up_scale_cur1 = up_scale_next0, up_scale_next1
            if not PAIRED:
                (
                    gate_scale_after0,
                    gate_scale_after1,
                    up_scale_after0,
                    up_scale_after1,
                ) = _stage1_load_scale_pair(
                    x_scale_ptr,
                    w_scale_ptr,
                    scale_tokens,
                    scale_valid,
                    scale_base,
                    pid_n,
                    tile + 2,
                    scale_stride,
                    I,
                    BLOCK_N,
                    scale_layout,
                    mfma,
                    PAIRED,
                )
            gate_scale_next0, gate_scale_next1 = gate_scale_after0, gate_scale_after1
            up_scale_next0, up_scale_next1 = up_scale_after0, up_scale_after1

    last_tile = first_tile + TILES - 1
    if not PAIRED:
        gate_next0, gate_next1, up_next0, up_next1 = _stage1_load_packed_b(
            packed_gate_base, packed_up_base, packed_offsets, last_tile, PAIRED
        )
    gl.barrier()
    a_smem.store(a_next)
    gate_acc, up_acc = _stage1_accumulate_k128(
        a_dot_cur0,
        gate_cur0,
        up_cur0,
        gate_acc,
        up_acc,
        gate_scale_cur0,
        up_scale_cur0,
        BLOCK_M,
        BLOCK_N,
        mfma,
        PAIRED,
    )
    if PAIRED:
        gate_next0, up_next0 = _stage1_load_packed_b_half(
            packed_gate_base, packed_up_base, packed_offsets, last_tile, 0, PAIRED
        )
    gate_acc, up_acc = _stage1_accumulate_k128(
        a_dot_cur1,
        gate_cur1,
        up_cur1,
        gate_acc,
        up_acc,
        gate_scale_cur1,
        up_scale_cur1,
        BLOCK_M,
        BLOCK_N,
        mfma,
        PAIRED,
    )
    if PAIRED:
        gate_next1, up_next1 = _stage1_load_packed_b_half(
            packed_gate_base, packed_up_base, packed_offsets, last_tile, 1, PAIRED
        )
    gl.barrier()
    a_dot_cur0 = a_smem.slice(0, 128, 1).load(dot_a).to(gl.float8e4nv, bitcast=True)
    a_dot_cur1 = a_smem.slice(128, 128, 1).load(dot_a).to(gl.float8e4nv, bitcast=True)
    gate_acc, up_acc = _stage1_accumulate_k128(
        a_dot_cur0,
        gate_next0,
        up_next0,
        gate_acc,
        up_acc,
        gate_scale_next0,
        up_scale_next0,
        BLOCK_M,
        BLOCK_N,
        mfma,
        PAIRED,
    )
    gate_acc, up_acc = _stage1_accumulate_k128(
        a_dot_cur1,
        gate_next1,
        up_next1,
        gate_acc,
        up_acc,
        gate_scale_next1,
        up_scale_next1,
        BLOCK_M,
        BLOCK_N,
        mfma,
        PAIRED,
    )
    if PAIRED:
        if swiglu_limit > 0.0:
            gate_acc = gl.minimum(gate_acc, swiglu_limit)
            up_acc = gl.clamp(up_acc, -swiglu_limit, swiglu_limit)
        inter = gate_acc * (1.0 / (1.0 + gl.exp(-gate_acc))) * up_acc
        out_layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [4, 1], [1, 0])
        inter = gl.convert_layout(inter, out_layout).to(gl.bfloat16)
        rows_out = pid_m * BLOCK_M + gl.arange(
            0, BLOCK_M, gl.SliceLayout(1, out_layout)
        )
        ids_out = gl.load(sorted_ids_ptr + rows_out, rows_out < em, other=num_tokens)
        tokens_out = ids_out & 0xFFFFFF
        inter_rows = tokens_out * TOPK + (ids_out >> 24)
        cols = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, gl.SliceLayout(0, out_layout))
        quantized, scales = quantize_intermediate_tile(inter, USE_RECIPROCAL)
        gl.store(
            inter_ptr + inter_rows[:, None] * I + cols[None, :],
            quantized,
            mask=tokens_out[:, None] < num_tokens,
        )
        gl.store(
            inter_scale_ptr + inter_rows * (I // 128) + pid_n,
            scales,
            mask=tokens_out < num_tokens,
        )
    else:
        # Shuffle FP32 accumulator rows before native FP32 atomic additions. The
        # separate activation launch is the global completion boundary; no flags
        # or in-kernel cross-workgroup synchronization are needed.
        out_layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [4, 1], [1, 0])
        # Round each partition through FP16 before the FP32 atomic addition,
        # even though the accumulator and scratch are both FP32.
        out_acc = gl.convert_layout(gate_acc, out_layout).to(gl.float16).to(gl.float32)
        # Reload the sorted route IDs in output ownership; avoid a separate
        # cross-wave exchange solely for metadata.
        out_sorted_rows = pid_m * 16 + gl.arange(0, 16, gl.SliceLayout(1, out_layout))
        out_ids = gl.load(
            sorted_ids_ptr + out_sorted_rows, out_sorted_rows < em, other=num_tokens
        )
        out_tokens = out_ids & 0xFFFFFF
        out_rows = out_tokens * TOPK + (out_ids >> 24)
        out_cols = pid_n * 128 + gl.arange(0, 128, gl.SliceLayout(0, out_layout))
        gl.amd.cdna4.buffer_atomic_add(
            inter_ptr,
            out_rows[:, None] * (2 * I) + out_cols[None, :],
            out_acc,
            mask=out_tokens[:, None] < num_tokens,
            sem="relaxed",
        )


@gluon.jit
def _stage2_load_packed_b(weight_base, packed_offsets, tile):
    first = tile * 4096
    return (
        gl.amd.cdna3.buffer_load(weight_base, packed_offsets + first, cache=".cg"),
        gl.amd.cdna3.buffer_load(
            weight_base, packed_offsets + first + 2048, cache=".cg"
        ),
    )


@gluon.jit
def _stage2_load_scales(
    inter_scale_ptr,
    weight_scale_ptr,
    route_rows,
    valid_rows,
    weight_scale_base,
    tile,
    K: gl.constexpr,
    SCALE_LAYOUT: gl.constexpr,
):
    pair = gl.arange(0, 2, gl.SliceLayout(0, SCALE_LAYOUT))
    a = gl.amd.cdna3.buffer_load(
        inter_scale_ptr,
        route_rows[:, None] * (K // 128) + tile * 2 + pair[None, :],
        valid_rows[:, None],
        other=0.0,
    )
    b = gl.amd.cdna3.buffer_load(
        weight_scale_ptr,
        weight_scale_base + tile * 2 + pair,
        cache=".cg",
    )
    return a, b


@gluon.jit
def _stage2_accumulate(
    a0,
    a1,
    b0,
    b1,
    a_scale,
    b_scale,
    acc,
    MFMA: gl.constexpr,
):
    dot_b: gl.constexpr = gl.DotOperandLayout(1, MFMA, k_width=16)
    b0 = b0.reshape([8, 8, 16, 16]).permute([1, 3, 0, 2]).reshape([128, 128])
    b1 = b1.reshape([8, 8, 16, 16]).permute([1, 3, 0, 2]).reshape([128, 128])
    b0 = gl.convert_layout(b0, dot_b).to(gl.float8e4nv, bitcast=True)
    b1 = gl.convert_layout(b1, dot_b).to(gl.float8e4nv, bitcast=True)
    scale0, scale1 = gl.split(a_scale * b_scale[None, :])
    scale0 = gl.convert_layout(scale0, gl.SliceLayout(1, MFMA))
    scale1 = gl.convert_layout(scale1, gl.SliceLayout(1, MFMA))
    tile0 = gl.amd.cdna4.mfma_scaled(
        a0, None, "e4m3", b0, None, "e4m3", gl.zeros([16, 128], gl.float32, MFMA)
    )
    acc = gl.fma(tile0, scale0[:, None], acc)
    tile1 = gl.amd.cdna4.mfma_scaled(
        a1, None, "e4m3", b1, None, "e4m3", gl.zeros([16, 128], gl.float32, MFMA)
    )
    return gl.fma(tile1, scale1[:, None], acc)


@gluon.jit(
    launch_metadata=_launch_metadata,
    do_not_specialize=("num_tokens", "em"),
)
def gluon_fp8_packed_small_stage2_gfx950(
    inter_ptr,
    inter_scale_ptr,
    w_ptr,
    w_scale_ptr,
    out_ptr,
    sorted_ids_ptr,
    sorted_experts_ptr,
    sorted_weights_ptr,
    num_valid_ptr,
    num_tokens,
    em,
    K: gl.constexpr,
    N: gl.constexpr,
    TOPK: gl.constexpr,
):
    BLOCK_M: gl.constexpr = 16
    BLOCK_N: gl.constexpr = 128
    BLOCK_K: gl.constexpr = 256
    pid = gl.program_id(0)
    pid_n = pid % (N // BLOCK_N)
    pid_m = pid // (N // BLOCK_N)
    if pid_m * BLOCK_M >= gl.load(num_valid_ptr):
        return
    expert = gl.load(sorted_experts_ptr + pid_m)
    if expert < 0:
        return

    mfma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 128],
        transposed=True,
        warps_per_cta=[1, 4],
    )
    dot_a: gl.constexpr = gl.DotOperandLayout(0, mfma, k_width=16)
    load_a: gl.constexpr = gl.BlockedLayout([1, 16], [4, 16], [4, 1], [1, 0])
    load_b: gl.constexpr = _fp8_weight_n16k32_carrier_layout()
    shared_a: gl.constexpr = gl.SwizzledSharedLayout(16, 2, 8, [1, 0])
    a_smem = gl.allocate_shared_memory(gl.uint8, [BLOCK_M, BLOCK_K], shared_a)

    rows_a = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, gl.SliceLayout(1, load_a))
    ids_a = gl.load(sorted_ids_ptr + rows_a, rows_a < em, other=num_tokens)
    tokens_a = ids_a & 0xFFFFFF
    slots_a = ids_a >> 24
    valid_a = tokens_a < num_tokens
    safe_tokens_a = gl.where(valid_a, tokens_a, 0)
    safe_slots_a = gl.where(valid_a, slots_a, 0)
    ka = gl.arange(0, BLOCK_K, gl.SliceLayout(0, load_a))
    packed_index = gl.arange(0, 16384, load_b)
    packed_offsets = (pid_n * 8 + packed_index // 2048) * (K * 16) + packed_index % 2048
    packed_offsets = gl.multiple_of(packed_offsets, 16)
    weight_base = gl.multiple_of(w_ptr + expert.to(gl.int64) * (N * K), 16)

    rows_m = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, gl.SliceLayout(1, mfma))
    ids_m = gl.load(sorted_ids_ptr + rows_m, rows_m < em, other=num_tokens)
    tokens_m = ids_m & 0xFFFFFF
    slots_m = ids_m >> 24
    valid_m = tokens_m < num_tokens
    inter_rows_m = tokens_m * TOPK + slots_m
    scale_layout: gl.constexpr = gl.BlockedLayout([1, 2], [16, 4], [1, 4], [0, 1])
    scale_rows = gl.convert_layout(inter_rows_m, gl.SliceLayout(1, scale_layout))
    scale_valid = gl.convert_layout(valid_m, gl.SliceLayout(1, scale_layout))
    scale_base = (expert * (N // BLOCK_N) + pid_n) * (K // 128)
    a_offsets = (
        safe_tokens_a[:, None] * (TOPK * K) + safe_slots_a[:, None] * K + ka[None, :]
    )
    a_offsets = gl.max_contiguous(gl.multiple_of(a_offsets, [1, 16]), [1, 16])
    acc = gl.zeros([BLOCK_M, BLOCK_N], gl.float32, mfma)

    # Invalid padded routes use the safe row zero; their scales/output masks
    # remain zero. An unmasked contiguous transfer retains K16 vectors.
    a_base = gl.multiple_of(inter_ptr, 16)
    a_first = gl.load(a_base + a_offsets)
    b0, b1 = _stage2_load_packed_b(weight_base, packed_offsets, 0)
    scale_a, scale_b = _stage2_load_scales(
        inter_scale_ptr,
        w_scale_ptr,
        scale_rows,
        scale_valid,
        scale_base,
        0,
        K,
        scale_layout,
    )
    a_smem.store(a_first)
    a_next = gl.load(a_base + a_offsets + BLOCK_K)
    next_scale_a, next_scale_b = _stage2_load_scales(
        inter_scale_ptr,
        w_scale_ptr,
        scale_rows,
        scale_valid,
        scale_base,
        1,
        K,
        scale_layout,
    )
    gl.barrier()
    a0 = a_smem.slice(0, 128, 1).load(dot_a).to(gl.float8e4nv, bitcast=True)
    a1 = a_smem.slice(128, 128, 1).load(dot_a).to(gl.float8e4nv, bitcast=True)

    for tile in gl.static_range(K // BLOCK_K - 1):
        next_b0, next_b1 = _stage2_load_packed_b(weight_base, packed_offsets, tile + 1)
        gl.barrier()
        a_smem.store(a_next)
        if tile + 2 < K // BLOCK_K:
            a_after = gl.load(a_base + a_offsets + (tile + 2) * BLOCK_K)
            after_scale_a, after_scale_b = _stage2_load_scales(
                inter_scale_ptr,
                w_scale_ptr,
                scale_rows,
                scale_valid,
                scale_base,
                tile + 2,
                K,
                scale_layout,
            )
        acc = _stage2_accumulate(a0, a1, b0, b1, scale_a, scale_b, acc, mfma)
        gl.barrier()
        a0 = a_smem.slice(0, 128, 1).load(dot_a).to(gl.float8e4nv, bitcast=True)
        a1 = a_smem.slice(128, 128, 1).load(dot_a).to(gl.float8e4nv, bitcast=True)
        b0, b1 = next_b0, next_b1
        scale_a, scale_b = next_scale_a, next_scale_b
        if tile + 2 < K // BLOCK_K:
            a_next = a_after
            next_scale_a, next_scale_b = after_scale_a, after_scale_b

    acc = _stage2_accumulate(a0, a1, b0, b1, scale_a, scale_b, acc, mfma)
    route_weight = gl.load(sorted_weights_ptr + rows_m, rows_m < em, other=0.0)
    value = acc * route_weight[:, None]
    out_layout: gl.constexpr = gl.BlockedLayout([1, 2], [16, 4], [1, 4], [1, 0])
    value = gl.convert_layout(value, out_layout).to(gl.bfloat16)
    out_rows = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, gl.SliceLayout(1, out_layout))
    out_ids = gl.load(sorted_ids_ptr + out_rows, out_rows < em, other=num_tokens)
    out_tokens = out_ids & 0xFFFFFF
    out_cols = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, gl.SliceLayout(0, out_layout))
    gl.amd.cdna4.buffer_atomic_add(
        out_ptr,
        out_tokens[:, None] * N + out_cols[None, :],
        value,
        out_tokens[:, None] < num_tokens,
        sem="relaxed",
    )


@gluon.jit(launch_metadata=_launch_metadata, do_not_specialize=("rows", "swiglu_limit"))
def gluon_fp8_packed_small_activation_gfx950(
    gate_up_ptr,
    inter_ptr,
    inter_scale_ptr,
    rows,
    swiglu_limit,
    I: gl.constexpr,
    BLOCK: gl.constexpr,
    USE_RECIPROCAL: gl.constexpr,
):
    offsets = gl.program_id(0) * BLOCK + gl.arange(
        0, BLOCK, gl.BlockedLayout([1], [64], [4], [0])
    )
    row, column = offsets // I, offsets % I
    gate = gl.load(gate_up_ptr + row * (2 * I) + column, row < rows, other=0.0)
    up = gl.load(gate_up_ptr + row * (2 * I) + I + column, row < rows, other=0.0)
    gate = gl.minimum(gate, swiglu_limit)
    up = gl.minimum(gl.maximum(up, -swiglu_limit), swiglu_limit)
    activated = gate / (1.0 + gl.exp(-gate)) * up
    quantized, scales = quantize_intermediate_tile(
        activated.reshape((BLOCK // 128, 128)), USE_RECIPROCAL
    )
    gl.store(inter_ptr + offsets, quantized.reshape((BLOCK,)), row < rows)
    groups = gl.program_id(0) * (BLOCK // 128) + gl.arange(
        0, BLOCK // 128, scales.type.layout
    )
    gl.store(inter_scale_ptr + groups, scales, groups < rows * (I // 128))
