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

"""gfx950 FP8 routed MoE using packed weights and public Gluon MFMA.

Two shared activation tiles alternate while packed weights stay in
registers. Each wave computes four M16 and two N16 repeats. Stage 2 rounds
weighted contributions to BF16 before atomic addition.
"""

from __future__ import annotations

from tokenspeed_kernel_amd._triton import gl, gluon, triton
from tokenspeed_kernel_amd.ops.gfx950.moe.fp8.intermediate_quant import (
    quantize_intermediate_tile,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.fp8.packed_small import (
    _fp8_weight_n16k32_carrier_layout,
    _launch_metadata,
)


@triton.constexpr_function
def _activation_layouts():
    return (
        gl.BlockedLayout([2, 16], [8, 8], [4, 1], [1, 0]),
        gl.SwizzledSharedLayout(16, 1, 8, [1, 0]),
    )


@triton.constexpr_function
def _output_layout(stage1):
    if stage1:
        return gl.BlockedLayout([4, 8], [4, 16], [4, 1], [1, 0])
    return gl.BlockedLayout([16, 2], [1, 64], [4, 1], [1, 0])


@gluon.jit
def _load_weights(
    base,
    offsets,
    k_tile,
    K: gl.constexpr,
    N: gl.constexpr,
    STAGE1: gl.constexpr,
    USE_NT: gl.constexpr,
    DOT_B: gl.constexpr,
):
    # Native buffer .cg lowers to sc0|nt; its API cannot request nt alone.
    cache: gl.constexpr = ".cg" if USE_NT else ""
    base = gl.multiple_of(base, 16)
    gate = gl.amd.cdna4.buffer_load(base, offsets + k_tile * 2048, cache=cache).to(
        gl.float8e4nv, bitcast=True
    )
    if STAGE1:
        up = gl.amd.cdna4.buffer_load(
            base + N * K, offsets + k_tile * 2048, cache=cache
        ).to(gl.float8e4nv, bitcast=True)
    else:
        up = gate
    gate = gate.reshape(8, 8, 16, 16).permute(1, 3, 0, 2).reshape(128, 128)
    up = up.reshape(8, 8, 16, 16).permute(1, 3, 0, 2).reshape(128, 128)
    gate = gl.convert_layout(gate, DOT_B, assert_trivial=True)
    up = gl.convert_layout(up, DOT_B, assert_trivial=True)
    return gate, up


@gluon.jit
def _load_scales(
    a_scale_ptr,
    b_scale_ptr,
    route_rows,
    valid,
    scale_base,
    k_tile,
    K: gl.constexpr,
    N: gl.constexpr,
    STAGE1: gl.constexpr,
    USE_NT: gl.constexpr,
    ROW_LAYOUT: gl.constexpr,
):
    # Each lane owns one row per M16 repeat, as does its transposed accumulator.
    # Keep next-tile A/B scales separate from the current scale products.
    a = ()
    for repeat in gl.static_range(4):
        a += (
            gl.amd.cdna4.buffer_load(
                a_scale_ptr,
                route_rows[repeat] * (K // 128) + k_tile,
                valid[repeat],
                other=0.0,
            ),
        )
    cache: gl.constexpr = ".cg" if USE_NT else ""
    offset = gl.full([16], k_tile, gl.int32, ROW_LAYOUT)
    b = gl.amd.cdna4.buffer_load(b_scale_ptr + scale_base, offset, cache=cache)
    if STAGE1:
        up = gl.amd.cdna4.buffer_load(
            b_scale_ptr + scale_base + (N // 128) * (K // 128), offset, cache=cache
        )
    else:
        up = b
    return a, b, up


@gluon.jit
def _combine_scales(a, b, b_up, STAGE1: gl.constexpr):
    gate = ()
    up = ()
    for repeat in gl.static_range(4):
        gate += (a[repeat] * b,)
        if STAGE1:
            up += (a[repeat] * b_up,)
        else:
            up += (gate[repeat],)
    return gate, up


@gluon.jit
def _load_initial_scales(
    a_ptr,
    b_ptr,
    rows,
    valid,
    base,
    K: gl.constexpr,
    N: gl.constexpr,
    STAGE1: gl.constexpr,
    USE_NT: gl.constexpr,
    ROW_LAYOUT: gl.constexpr,
):
    # Read the first two adjacent FP32 scales as an 8-byte
    # transfer. The four routed rows retain their MFMA lane ownership.
    pair: gl.constexpr = gl.BlockedLayout([1, 2], [64, 1], [4, 1], [1, 0])
    row: gl.constexpr = gl.SliceLayout(1, pair)
    kk = gl.arange(0, 2, gl.SliceLayout(0, pair))
    a0, a1 = (), ()
    for repeat in gl.static_range(4):
        rr = gl.convert_layout(rows[repeat], row, assert_trivial=True)
        vv = gl.convert_layout(valid[repeat], row, assert_trivial=True)
        scales = gl.amd.cdna4.buffer_load(
            a_ptr, rr[:, None] * (K // 128) + kk[None, :], vv[:, None], other=0.0
        )
        first, second = gl.split(scales)
        a0 += (gl.convert_layout(first, ROW_LAYOUT, assert_trivial=True),)
        a1 += (gl.convert_layout(second, ROW_LAYOUT, assert_trivial=True),)
    offsets = gl.full([16], 0, gl.int32, row)[:, None] + kk[None, :]
    cache: gl.constexpr = ".cg" if USE_NT else ""
    b0, b1 = gl.split(gl.amd.cdna4.buffer_load(b_ptr + base, offsets, cache=cache))
    b0 = gl.convert_layout(b0, ROW_LAYOUT, assert_trivial=True)
    b1 = gl.convert_layout(b1, ROW_LAYOUT, assert_trivial=True)
    if STAGE1:
        u0, u1 = gl.split(
            gl.amd.cdna4.buffer_load(
                b_ptr + base + (N // 128) * (K // 128), offsets, cache=cache
            )
        )
        u0 = gl.convert_layout(u0, ROW_LAYOUT, assert_trivial=True)
        u1 = gl.convert_layout(u1, ROW_LAYOUT, assert_trivial=True)
    else:
        u0, u1 = b0, b1
    return a0, b0, u0, a1, b1, u1


@gluon.jit
def _compute_tile(
    shared_current,
    shared_next,
    a0,
    a1,
    gate,
    up,
    gate_scale,
    up_scale,
    gate_accs,
    up_accs,
    MFMA: gl.constexpr,
    DOT_A: gl.constexpr,
    STAGE1: gl.constexpr,
    LAST: gl.constexpr,
):
    next_gate = ()
    next_up = ()
    for repeat in gl.static_range(4):
        # Source dependencies retain two M16 A repeats; compiler scheduling is free.
        tile = gl.amd.cdna4.mfma_scaled(
            a0, None, "e4m3", gate, None, "e4m3", gl.zeros([16, 128], gl.float32, MFMA)
        )
        next_gate += (gl.fma(tile, gate_scale[repeat][:, None], gate_accs[repeat]),)
        if STAGE1:
            tile_up = gl.amd.cdna4.mfma_scaled(
                a0,
                None,
                "e4m3",
                up,
                None,
                "e4m3",
                gl.zeros([16, 128], gl.float32, MFMA),
            )
            next_up += (gl.fma(tile_up, up_scale[repeat][:, None], up_accs[repeat]),)
        else:
            next_up += (up_accs[repeat],)

        if repeat < 2:
            following = shared_current.slice((repeat + 2) * 16, 16, 0).load(DOT_A)
            a0 = a1
            a1 = following.to(gl.float8e4nv, bitcast=True)
        elif not LAST:
            if repeat == 2:
                gl.barrier()
            following = shared_next.slice((repeat - 2) * 16, 16, 0).load(DOT_A)
            a0 = a1
            a1 = following.to(gl.float8e4nv, bitcast=True)
        else:
            a0 = a1
    return a0, a1, next_gate, next_up


@gluon.jit
def _packed_large_product(
    a_ptr,
    a_scale_ptr,
    w_ptr,
    w_scale_ptr,
    out_ptr,
    out_scale_ptr,
    sorted_ids_ptr,
    sorted_experts_ptr,
    sorted_weights_ptr,
    num_valid_ptr,
    num_tokens,
    em,
    swiglu_limit,
    K: gl.constexpr,
    N: gl.constexpr,
    TOPK: gl.constexpr,
    STAGE1: gl.constexpr,
    USE_NT: gl.constexpr,
    USE_RECIPROCAL: gl.constexpr,
):
    gl.static_assert(K >= 256 and K % 256 == 0)
    gl.static_assert(N % 128 == 0)
    pid_n = gl.program_id(0) % (N // 128)
    pid_m = gl.program_id(0) // (N // 128)
    if pid_m * 64 >= gl.load(num_valid_ptr):
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
    dot_b: gl.constexpr = gl.DotOperandLayout(1, mfma, k_width=16)
    # Each thread owns two adjacent routed rows and one K16 segment.
    load_a: gl.constexpr = _activation_layouts()[0]
    shared: gl.constexpr = _activation_layouts()[1]
    shared0 = gl.allocate_shared_memory(gl.uint8, [64, 128], shared)
    shared1 = gl.allocate_shared_memory(gl.uint8, [64, 128], shared)

    rows_a = pid_m * 64 + gl.arange(0, 64, gl.SliceLayout(1, load_a))
    ids_a = gl.load(sorted_ids_ptr + rows_a, rows_a < em, other=num_tokens)
    tokens_a = ids_a & 0xFFFFFF
    valid_a = tokens_a < num_tokens
    if STAGE1:
        source_a = tokens_a
    else:
        source_a = tokens_a * TOPK + (ids_a >> 24)
    ka = gl.arange(0, 128, gl.SliceLayout(0, load_a))
    a_offsets = source_a[:, None] * K + ka[None, :]
    a_offsets = gl.max_contiguous(gl.multiple_of(a_offsets, [1, 16]), [1, 16])

    load_b: gl.constexpr = _fp8_weight_n16k32_carrier_layout()
    packed = gl.arange(0, 16384, load_b)
    weight_base = w_ptr + expert.to(gl.int64) * (N * K * (2 if STAGE1 else 1))
    weight_offsets = (pid_n * 8 + packed // 2048) * (K * 16) + packed % 2048
    row_layout: gl.constexpr = gl.SliceLayout(1, mfma)
    source_m = ()
    valid_m = ()
    for repeat in gl.static_range(4):
        rows_m = pid_m * 64 + repeat * 16 + gl.arange(0, 16, row_layout)
        ids_m = gl.load(sorted_ids_ptr + rows_m, rows_m < em, other=num_tokens)
        tokens_m = ids_m & 0xFFFFFF
        valid_m += (tokens_m < num_tokens,)
        if STAGE1:
            source_m += (tokens_m,)
        else:
            source_m += (tokens_m * TOPK + (ids_m >> 24),)
    scale_base = (expert * (N // 128) * (2 if STAGE1 else 1) + pid_n) * (K // 128)

    zero = gl.zeros([16, 128], gl.float32, mfma)
    gate_accs = (zero, zero, zero, zero)
    up_accs = (zero, zero, zero, zero)
    gate0, up0 = _load_weights(
        weight_base, weight_offsets, 0, K, N, STAGE1, USE_NT, dot_b
    )
    sa0, sb0, sbu0, sa1, sb1, sbu1 = _load_initial_scales(
        a_scale_ptr,
        w_scale_ptr,
        source_m,
        valid_m,
        scale_base,
        K,
        N,
        STAGE1,
        USE_NT,
        row_layout,
    )
    gs0, us0 = _combine_scales(sa0, sb0, sbu0, STAGE1)
    shared0.store(gl.amd.cdna4.buffer_load(a_ptr, a_offsets, valid_a[:, None], other=0))
    pending_a = gl.amd.cdna4.buffer_load(
        a_ptr, a_offsets + 128, valid_a[:, None], other=0
    )
    gl.barrier()
    a0 = shared0.slice(0, 16, 0).load(dot_a).to(gl.float8e4nv, bitcast=True)
    a1 = shared0.slice(16, 16, 0).load(dot_a).to(gl.float8e4nv, bitcast=True)

    # Statically unroll the two halves to retain V3's alternating activation
    # buffers and the next-weight/next-activation prefetch before each MFMA.
    for pair in range(K // 256 - 1):
        for half in gl.static_range(2):
            tile = pair * 2 + half
            if half == 0:
                current_shared, next_shared = shared0, shared1
            else:
                current_shared, next_shared = shared1, shared0
            gate1, up1 = _load_weights(
                weight_base, weight_offsets, tile + 1, K, N, STAGE1, USE_NT, dot_b
            )
            next_shared.store(pending_a)
            pending_a = gl.amd.cdna4.buffer_load(
                a_ptr, a_offsets + (tile + 2) * 128, valid_a[:, None], other=0
            )
            sa2, sb2, sbu2 = _load_scales(
                a_scale_ptr,
                w_scale_ptr,
                source_m,
                valid_m,
                scale_base,
                tile + 2,
                K,
                N,
                STAGE1,
                USE_NT,
                row_layout,
            )
            a0, a1, gate_accs, up_accs = _compute_tile(
                current_shared,
                next_shared,
                a0,
                a1,
                gate0,
                up0,
                gs0,
                us0,
                gate_accs,
                up_accs,
                mfma,
                dot_a,
                STAGE1,
                False,
            )
            gs0, us0 = _combine_scales(sa1, sb1, sbu1, STAGE1)
            gate0, up0 = gate1, up1
            sa1, sb1, sbu1 = sa2, sb2, sbu2

    # Even two-tile tail: drain the pending activation without another prefetch.
    gate1, up1 = _load_weights(
        weight_base, weight_offsets, K // 128 - 1, K, N, STAGE1, USE_NT, dot_b
    )
    shared1.store(pending_a)
    gs1, us1 = _combine_scales(sa1, sb1, sbu1, STAGE1)
    a0, a1, gate_accs, up_accs = _compute_tile(
        shared0,
        shared1,
        a0,
        a1,
        gate0,
        up0,
        gs0,
        us0,
        gate_accs,
        up_accs,
        mfma,
        dot_a,
        STAGE1,
        False,
    )
    a0, a1, gate_accs, up_accs = _compute_tile(
        shared1,
        shared1,
        a0,
        a1,
        gate1,
        up1,
        gs1,
        us1,
        gate_accs,
        up_accs,
        mfma,
        dot_a,
        STAGE1,
        True,
    )

    # Apply activation or route weight in accumulator ownership, then exchange
    # the whole FP32 tile once. Joining M repeats only rearranges registers.
    values = ()
    for repeat in gl.static_range(4):
        if STAGE1:
            gate = gate_accs[repeat]
            up = up_accs[repeat]
            if swiglu_limit > 0.0:
                gate = gl.minimum(gate, swiglu_limit)
                up = gl.clamp(up, -swiglu_limit, swiglu_limit)
            value = gate * (1.0 / (1.0 + gl.extra.libdevice.exp(-gate))) * up
        else:
            rows = pid_m * 64 + repeat * 16 + gl.arange(0, 16, row_layout)
            weight = gl.load(sorted_weights_ptr + rows, rows < em, other=0.0)
            value = gate_accs[repeat] * weight[:, None]
        values += (value,)
    value = gl.join(gl.join(values[0], values[1]), gl.join(values[2], values[3]))
    value = value.permute(3, 2, 0, 1).reshape(64, 128)
    value = gl.convert_layout(value, mfma, assert_trivial=True)
    store_layout: gl.constexpr = _output_layout(STAGE1)
    value = gl.convert_layout(value, store_layout).to(gl.bfloat16)
    rows = pid_m * 64 + gl.arange(0, 64, gl.SliceLayout(1, store_layout))
    cols = pid_n * 128 + gl.arange(0, 128, gl.SliceLayout(0, store_layout))
    ids = gl.load(sorted_ids_ptr + rows, rows < em, other=num_tokens)
    token = ids & 0xFFFFFF
    valid = token < num_tokens
    if STAGE1:
        dest = token * TOPK + (ids >> 24)
        quantized, scales = quantize_intermediate_tile(value, USE_RECIPROCAL)
        gl.amd.cdna4.buffer_store(
            quantized, out_ptr, dest[:, None] * N + cols[None, :], valid[:, None]
        )
        gl.store(out_scale_ptr + dest * (N // 128) + pid_n, scales, valid)
    else:
        gl.atomic_add(
            out_ptr + token[:, None] * N + cols[None, :],
            value,
            valid[:, None],
            sem="relaxed",
        )


@gluon.jit(
    launch_metadata=_launch_metadata,
    do_not_specialize=("num_tokens", "em", "swiglu_limit"),
)
def gluon_fp8_packed_large_stage1_gfx950(
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
    USE_NT: gl.constexpr,
    USE_RECIPROCAL: gl.constexpr,
):
    _packed_large_product(
        x_ptr,
        x_scale_ptr,
        w_ptr,
        w_scale_ptr,
        inter_ptr,
        inter_scale_ptr,
        sorted_ids_ptr,
        sorted_experts_ptr,
        x_scale_ptr,
        num_valid_ptr,
        num_tokens,
        em,
        swiglu_limit,
        K,
        I,
        TOPK,
        True,
        USE_NT,
        USE_RECIPROCAL,
    )


@gluon.jit(launch_metadata=_launch_metadata, do_not_specialize=("num_tokens", "em"))
def gluon_fp8_packed_large_stage2_gfx950(
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
    USE_NT: gl.constexpr,
):
    _packed_large_product(
        inter_ptr,
        inter_scale_ptr,
        w_ptr,
        w_scale_ptr,
        out_ptr,
        inter_scale_ptr,
        sorted_ids_ptr,
        sorted_experts_ptr,
        sorted_weights_ptr,
        num_valid_ptr,
        num_tokens,
        em,
        0.0,
        K,
        N,
        TOPK,
        False,
        USE_NT,
        False,
    )
