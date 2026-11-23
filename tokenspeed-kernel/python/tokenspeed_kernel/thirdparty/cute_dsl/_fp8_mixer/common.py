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

"""Shared CuTe operations for prequantized FP8 mixers."""

import cutlass
import cutlass.cute as cute
import cutlass.utils as cute_utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass._mlir.dialects import llvm
from cutlass.cute import experimental as cute_ext
from cutlass.cute.nvgpu import CopyUniversalOp, tcgen05
from cutlass.cutlass_dsl import T

# The ring-based Up schedule consumes full K128 tiles.
_UP_K_TILE = 128


def _has_full_up_tiles(lowrank: int) -> bool:
    return lowrank % _UP_K_TILE == 0


@cute.experimental.jit
def _scale_down_value(
    value, input_scale, weight_scale, projection_scale, silu: cutlass.Constexpr
):
    if cutlass.const_expr(input_scale is not None):
        value = value * input_scale
    value = value * weight_scale
    value = value * projection_scale
    if cutlass.const_expr(silu):
        value = value * cute.arch.rcp_approx(1.0 + cute.exp(-value, fastmath=True))
    return value


@cute.experimental.jit
def _quant_scale(maximum, propagate_nan: cutlass.Constexpr):
    return cute.math.max(
        maximum / cutlass.Float32(448.0),
        cutlass.Float32(1.0 / (448.0 * 512.0)),
        propagate_nan=propagate_nan,
    )


@cute.experimental.jit
def _quant_multiplier(maximum, propagate_nan: cutlass.Constexpr):
    return cute.math.min(
        cutlass.Float32(448.0) / maximum,
        cutlass.Float32(448.0 * 512.0),
        propagate_nan=propagate_nan,
    )


@cute.experimental.jit
def _quantized_value(value, multiplier, clamp: cutlass.Constexpr):
    value = value.to(cutlass.Float32) * multiplier
    lower = cutlass.Float32(-448.0)
    if cutlass.const_expr(clamp == "ptx_lower"):
        # PTX max without .NaN selects the lower bound for NaN.
        value = cutlass.Float32(
            llvm.inline_asm(
                T.f32(),
                [value.ir_value(), lower.ir_value()],
                "max.f32 $0, $1, $2;",
                "=f,f,f",
                has_side_effects=False,
                asm_dialect=0,
            )
        )
    else:
        value = cute.math.max(value, lower)
    if cutlass.const_expr(clamp == "both"):
        value = cute.math.min(value, cutlass.Float32(448.0))
    return value


@cute.experimental.jit
def _decode_residual(codes):
    # Every E4M3 value is exactly representable in BF16 and FP32.
    words = cute.recast_tensor(codes, cutlass.Uint16)
    decoded = cute.make_rmem_tensor(cute.size(codes), cutlass.Float32)
    for i in cutlass.range_constexpr(cute.size(codes) // 2):
        result = llvm.inline_asm(
            llvm.StructType.get_literal([T.f32(), T.f32()]),
            [words[i].ir_value()],
            "{ .reg .b32 pair; .reg .b16 lo, hi;\n"
            "cvt.rn.bf16x2.e4m3x2 pair, $2;\n"
            "mov.b32 {lo, hi}, pair;\n"
            "cvt.f32.bf16 $0, lo; cvt.f32.bf16 $1, hi; }",
            "=f,=f,h",
            has_side_effects=False,
            asm_dialect=0,
        )
        decoded[2 * i] = cutlass.Float32(llvm.extractvalue(T.f32(), result, [0]))
        decoded[2 * i + 1] = cutlass.Float32(llvm.extractvalue(T.f32(), result, [1]))
    return decoded


@cute.experimental.jit
def _copy_residual_16(destination, source, size):
    llvm.inline_asm(
        None,
        [
            cutlass.Int32(destination.toint()).ir_value(),
            source.toint().ir_value(),
            size.ir_value(),
        ],
        "cp.async.ca.shared.global [$0], [$1], 16, $2;",
        "r,l,r,~{memory}",
        has_side_effects=True,
        asm_dialect=0,
    )


@cute.experimental.jit
def _commit_residual():
    llvm.inline_asm(
        None,
        [],
        "cp.async.commit_group;",
        "~{memory}",
        has_side_effects=True,
        asm_dialect=0,
    )


@cute.experimental.jit
def _wait_residual():
    llvm.inline_asm(
        None,
        [],
        "cp.async.wait_group 0;",
        "~{memory}",
        has_side_effects=True,
        asm_dialect=0,
    )


@cute.experimental.jit
def _acquire(ptr):
    return cutlass.Int64(
        llvm.inline_asm(
            T.i64(),
            [ptr.toint().ir_value()],
            "ld.global.acquire.gpu.u64 $0, [$1];",
            "=l,l,~{memory}",
            has_side_effects=True,
            asm_dialect=0,
        )
    )


@cute.experimental.jit
def _release(ptr, value):
    llvm.inline_asm(
        None,
        [ptr.toint().ir_value(), value.ir_value()],
        "st.global.release.gpu.u64 [$0], $1;",
        "l,l,~{memory}",
        has_side_effects=True,
        asm_dialect=0,
    )


@cute.experimental.jit
def _arrive_phase_release(ptr, asynchronous: cutlass.Constexpr):
    # Acquire polling observes preceding releases through the atomic RMW chain.
    instruction = "red.async" if asynchronous else "red"
    llvm.inline_asm(
        None,
        [ptr.toint().ir_value()],
        f"{instruction}.release.gpu.global.add.u64 [$0], 1;",
        "l,~{memory}",
        has_side_effects=True,
        asm_dialect=0,
    )


@cute.experimental.jit
def _grid_barrier(epochs, expected, wait_for_all, asynchronous: cutlass.Constexpr):
    # The cooperative grid is small enough for every CTA to stay resident.
    cute.arch.sync_threads()
    if cute.arch.thread_idx()[0] == 0:
        _arrive_phase_release(epochs.iterator, asynchronous)
        if wait_for_all:
            while _acquire(epochs.iterator) < expected:
                pass
    cute.arch.sync_threads()


@cute.experimental.jit
def _invalidate(barriers, count: cutlass.Constexpr):
    with cute.arch.elect_one():
        for index in cutlass.range_constexpr(count):
            address = cutlass.Int32((barriers + index).toint())
            llvm.inline_asm(
                None,
                [address.ir_value()],
                "mbarrier.inval.shared::cta.b64 [$0];",
                "r,~{memory}",
                has_side_effects=True,
                asm_dialect=0,
            )


@cute.experimental.jit
def _operand_storage(arena, first_layout, second_layout):
    dtype = cutlass.Float8E4M3FN
    first = cute.make_tensor(
        cute.make_ptr(
            dtype,
            arena.iterator.toint(),
            cute.AddressSpace.smem,
            assumed_align=1024,
            swizzle_=cute.get_swizzle_portion(first_layout),
        ),
        cute.get_nonswizzle_portion(first_layout),
    )
    second = cute.make_tensor(
        cute.make_ptr(
            dtype,
            arena.iterator.toint() + cute.cosize(first_layout),
            cute.AddressSpace.smem,
            assumed_align=1024,
            swizzle_=cute.get_swizzle_portion(second_layout),
        ),
        cute.get_nonswizzle_portion(second_layout),
    )
    return first, second


@cute.experimental.jit
def _operand_at(arena, layout, offset):
    return cute.make_tensor(
        cute.make_ptr(
            cutlass.Float8E4M3FN,
            arena.iterator.toint() + offset,
            cute.AddressSpace.smem,
            assumed_align=1024,
            swizzle_=cute.get_swizzle_portion(layout),
        ),
        cute.get_nonswizzle_portion(layout),
    )


@cute.experimental.jit
def _load_f32(ptr):
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [ptr.toint().ir_value()],
            "ld.global.cg.f32 $0, [$1];",
            "=f,l,~{memory}",
            has_side_effects=True,
            asm_dialect=0,
        )
    )


@cute.experimental.jit
def _x_scale(scales, row):
    index = cutlass.Int64(0)
    if scales.shape[0] != 1:
        index = cutlass.Int64(row)
    return _load_f32(scales.iterator + index)


@cute.experimental.jit
def _mul_hidden_float_pair(lhs0, lhs1, rhs0, rhs1):
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [lhs0.ir_value(), lhs1.ir_value(), rhs0.ir_value(), rhs1.ir_value()],
        "{ .reg .b64 a,b,c; mov.b64 a, {$2,$3}; mov.b64 b, {$4,$5}; "
        "mul.rn.f32x2 c,a,b; mov.b64 {$0,$1},c; }",
        "=f,=f,f,f,f,f",
        has_side_effects=False,
        asm_dialect=0,
    )
    return cutlass.Float32(llvm.extractvalue(T.f32(), result, [0])), cutlass.Float32(
        llvm.extractvalue(T.f32(), result, [1])
    )


@cute.experimental.jit
def _store_private_epoch(ptr, value):
    llvm.inline_asm(
        None,
        [ptr.toint().ir_value(), value.ir_value()],
        "st.global.cg.u64 [$0], $1;",
        "l,l,~{memory}",
        has_side_effects=True,
        asm_dialect=0,
    )


@cute.experimental.jit
def _issue_mma_block(mma, acc, a_desc, b_desc, block, stage):
    cute.gemm(
        mma,
        acc,
        a_desc[None, None, block, stage],
        b_desc[None, None, block, stage],
        acc,
    )


@cute.experimental.jit
def _run_cute_mma_ring(
    mma,
    acc,
    a_desc,
    b_desc,
    ready,
    empty,
    tiles: cutlass.Constexpr,
    stages: cutlass.Constexpr,
    k_unroll: cutlass.Constexpr,
    retire_empty: cutlass.Constexpr,
):
    # CuTe tracks MMA state through each loop by seeing the set calls in scope.
    mma.set(tcgen05.Field.ACCUMULATE, False)
    for tile in cutlass.range(tiles, unroll=1):
        stage = tile % stages
        cute.arch.mbarrier_wait(ready + stage, tile // stages % 2)
        if cutlass.const_expr(k_unroll is None):
            for block in cutlass.range_constexpr(cute.size(a_desc, mode=[2])):
                _issue_mma_block(mma, acc, a_desc, b_desc, block, stage)
                mma.set(tcgen05.Field.ACCUMULATE, True)
        else:
            for block in cutlass.range(cute.size(a_desc, mode=[2]), unroll=k_unroll):
                _issue_mma_block(mma, acc, a_desc, b_desc, block, stage)
                mma.set(tcgen05.Field.ACCUMULATE, True)
        if cutlass.const_expr(retire_empty):
            with cute.arch.elect_one():
                tcgen05.commit(empty + stage, None, tcgen05.CtaGroup.ONE)


@cute.experimental.jit
def _make_k_major_mma(dtype: cutlass.Constexpr, shape: cutlass.Constexpr):
    return sm100_utils.make_trivial_tiled_mma(
        dtype,
        dtype,
        tcgen05.OperandMajorMode.K,
        tcgen05.OperandMajorMode.K,
        cutlass.Float32,
        tcgen05.CtaGroup.ONE,
        shape,
    )


@cute.experimental.jit
def _make_down_operands(arena, tile: cutlass.Constexpr, stages: cutlass.Constexpr):
    m, n, _ = tile
    dtype = cutlass.Float8E4M3FN
    mma = _make_k_major_mma(dtype, (m, n))
    weight = sm100_utils.make_smem_layout_a(mma, tile, dtype, stages)
    activation = sm100_utils.make_smem_layout_b(mma, tile, dtype, stages)
    sw, sx = _operand_storage(arena, weight, activation)
    return mma, sw, sx


@cute.experimental.jit
def _make_down_fragments(thread_mma, base, sw, sx, acc_shape):
    a_desc = thread_mma.make_fragment_A(sw)
    b_desc = thread_mma.make_fragment_B(sx)
    fragment = thread_mma.make_fragment_C(acc_shape)
    accumulator = cute.make_tensor(
        cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, base), fragment.layout
    )
    return accumulator, a_desc, b_desc


@cute.experimental.jit
def _tma_matrix_view(tensor, rows, row_stride: cutlass.Constexpr, k: cutlass.Constexpr):
    return cute.make_tensor(
        tensor.iterator,
        cute.make_layout((rows, (64, k // 64), 1), stride=(row_stride, (1, 64), 0)),
    )


@cute.experimental.jit
def _reduce_split_partials(
    pieces, base, stride, row, interleaved_rows: cutlass.Constexpr
):
    columns, splits = cute.shape(pieces)
    packet = 4 if interleaved_rows or columns >= 4 else 2
    packets_per_split = 1 if interleaved_rows else columns // packet
    for rank in cutlass.range_constexpr(splits):
        for block in cutlass.range_constexpr(packets_per_split):
            outputs = ", ".join(f"${i}" for i in range(packet))
            constraints = ",".join(["=f"] * packet + ["l", "~{memory}"])
            result = llvm.inline_asm(
                llvm.StructType.get_literal([T.f32()] * packet),
                [(base + rank * stride + block * packet).toint().ir_value()],
                f"ld.global.cg.v{packet}.f32 {{{outputs}}}, [${packet}];",
                constraints,
                has_side_effects=True,
                asm_dialect=0,
            )
            if cutlass.const_expr(interleaved_rows):
                for i in cutlass.range_constexpr(columns):
                    even = cutlass.Float32(llvm.extractvalue(T.f32(), result, [2 * i]))
                    odd = cutlass.Float32(
                        llvm.extractvalue(T.f32(), result, [2 * i + 1])
                    )
                    pieces[i, rank] = cutlass.select_(row % 2 == 0, even, odd)
            else:
                for i in cutlass.range_constexpr(packet):
                    pieces[block * packet + i, rank] = cutlass.Float32(
                        llvm.extractvalue(T.f32(), result, [i])
                    )
    values = cute.make_rmem_tensor(columns, cutlass.Float32)
    values.fill(0.0)
    # Keep FP32 additions in split order.
    for rank in cutlass.range_constexpr(splits):
        for i in cutlass.range_constexpr(columns):
            values[i] = values[i] + pieces[i, rank]
    return values


@cute.experimental.jit
def _make_tmem_reader(
    accumulator,
    coords,
    original,
    tid,
    shape: cutlass.Constexpr,
    tile: cutlass.Constexpr,
):
    copy = tcgen05.make_tmem_copy(
        sm100_utils.get_tmem_load_op(
            shape,
            cute_utils.LayoutEnum.COL_MAJOR,
            cutlass.Float32,
            cutlass.Float32,
            tile,
            False,
        ),
        accumulator,
    )
    reader = copy.get_slice(tid % 128)
    dummy = cute.make_tensor(
        original.iterator, cute.make_layout(tile, stride=(1, tile[0]))
    )
    values = cute_ext.allocate(
        cutlass.Float32,
        cute.AddressSpace.rmem,
        cute_ext.make_t2r_rmem_layout(copy, cute.flat_divide(dummy, tile), tid % 128),
        alignment=32,
    )
    return accumulator, reader, values, reader.partition_D(coords)


@cute.experimental.jit
def _token_major_reader(
    mma,
    base,
    original,
    tid,
    branch: cutlass.Constexpr,
    part: cutlass.Constexpr,
    shape: cutlass.Constexpr,
    parts: cutlass.Constexpr,
    hidden: cutlass.Constexpr,
    columns: cutlass.Constexpr,
):
    rows, channels, _ = shape
    layout = cute_ext.make_tmem_layout_acc(mma, (rows, channels), acc_stage=parts)
    accumulator = cute.make_tensor(
        cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, base), layout
    )[((None, None), 0, 0, part)]
    column = branch * hidden + (tid // 128) * columns
    tile = (rows, columns)
    accumulator = cute.make_tensor(
        accumulator.iterator + accumulator.layout((0, column)),
        cute.make_layout(
            (cute.shape(accumulator, mode=[0]), columns),
            stride=(accumulator.stride[0], accumulator.stride[1]),
        ),
    )
    coords = cute.local_tile(
        cute.make_identity_tensor((rows, hidden)), tile, (0, tid // 128)
    )
    return _make_tmem_reader(accumulator, coords, original, tid, shape, tile)


@cute.experimental.jit
def _tiled_tmem_reader(
    accumulator,
    original,
    tid,
    group,
    shape: cutlass.Constexpr,
    columns: cutlass.Constexpr,
):
    m, n, _ = shape
    tile = (m, columns)
    accumulator = cute.local_tile(accumulator, tile, (0, group))
    coords = cute.local_tile(cute.make_identity_tensor((m, n)), tile, (0, group))
    return _make_tmem_reader(accumulator, coords, original, tid, shape, tile)


@cute.experimental.jit
def _prefetch_branch_residual(
    target,
    coords,
    x,
    hidden_tile,
    global_base,
    part: cutlass.Constexpr,
    branch: cutlass.Constexpr,
    rows: cutlass.Constexpr,
    hidden: cutlass.Constexpr,
    g: cutlass.Constexpr,
    h: cutlass.Constexpr,
):
    # M64 readers interleave token rows. Each contiguous packet is a channel pair.
    packet_size = 2 if rows == 64 else cute.size(target)
    for packet in cutlass.range_constexpr(cute.size(target) // packet_size):
        row, col = coords[packet * packet_size]
        global_row = global_base + cutlass.Int64(part * rows + row)
        if global_row < x.shape[0]:
            copy_bits = min(128, packet_size * 8)
            pointer = cute.make_ptr(
                cutlass.Float8E4M3FN,
                (
                    x.iterator
                    + global_row * (g * h)
                    + branch * h
                    + hidden_tile * hidden
                    + col
                ).toint(),
                cute.AddressSpace.gmem,
                assumed_align=copy_bits // 8,
            )
            source = cute.make_tensor(pointer, cute.make_layout(packet_size))
            values = cute.make_rmem_tensor(packet_size, cutlass.Float8E4M3FN)
            atom = cute.make_copy_atom(
                CopyUniversalOp(), cutlass.Float8E4M3FN, num_bits_per_copy=copy_bits
            )
            cute.copy(atom, source, values)
            for i in cutlass.range_constexpr(packet_size):
                target[packet * packet_size + i] = values[i]


@cute.experimental.jit
def _store_token_major_mix(
    out,
    sums,
    coords,
    hidden_tile,
    global_base,
    num_rows,
    part: cutlass.Constexpr,
    rows: cutlass.Constexpr,
    hidden: cutlass.Constexpr,
    g: cutlass.Constexpr,
    h: cutlass.Constexpr,
):
    packed = cute.make_rmem_tensor(cute.size(sums), out.element_type)
    for i in cutlass.range_constexpr(cute.size(sums)):
        packed[i] = (sums[i] / g).to(out.element_type)
    packet_size = 2 if rows == 64 else cute.size(sums)
    for packet in cutlass.range_constexpr(cute.size(sums) // packet_size):
        row, col = coords[packet * packet_size]
        global_row = global_base + cutlass.Int64(part * rows + row)
        if global_row < num_rows:
            pointer = cute.make_ptr(
                out.element_type,
                (out.iterator + global_row * h + hidden_tile * hidden + col).toint(),
                cute.AddressSpace.gmem,
                assumed_align=min(16, packet_size * 2),
            )
            target = cute.make_tensor(pointer, cute.make_layout(packet_size))
            output_packet = cute.make_rmem_tensor(packet_size, out.element_type)
            for i in cutlass.range_constexpr(packet_size):
                output_packet[i] = packed[packet * packet_size + i]
            cute.copy(
                cute.make_copy_atom(
                    CopyUniversalOp(),
                    out.element_type,
                    num_bits_per_copy=min(128, packet_size * 16),
                ),
                output_packet,
                target,
            )
