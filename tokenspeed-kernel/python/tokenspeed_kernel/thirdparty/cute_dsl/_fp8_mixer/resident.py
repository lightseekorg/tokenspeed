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

"""Shared resident device code for prequantized FP8 mixers."""

from typing import NamedTuple

import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass._mlir.dialects import llvm
from cutlass.cute import experimental as cute_ext
from cutlass.cute.nvgpu import CopyUniversalOp, tcgen05
from cutlass.cutlass_dsl import T
from cutlass.experimental.primitives import tcgen05_fence
from tokenspeed_kernel.thirdparty.cute_dsl._fp8_mixer.common import (
    _commit_residual,
    _copy_residual_16,
    _decode_residual,
    _invalidate,
    _load_f32,
    _make_k_major_mma,
    _make_tmem_reader,
    _operand_at,
    _prefetch_branch_residual,
    _quant_multiplier,
    _quant_scale,
    _quantized_value,
    _reduce_split_partials,
    _scale_down_value,
    _store_token_major_mix,
    _tiled_tmem_reader,
    _tma_matrix_view,
    _token_major_reader,
    _x_scale,
)
from tokenspeed_kernel.thirdparty.cute_dsl.hc_fp8 import (
    Fp8GatedResidualGemm,
    _epilogue_barrier,
)
from tokenspeed_kernel.thirdparty.cute_dsl.hc_fused import (
    _publish_activation_proxy,
    _up_sigmoid_tanh,
)


class MixerOperands(NamedTuple):
    x: cute.Tensor
    xs: cute.Tensor
    down: cute.Tensor
    ds: cute.Tensor
    up: cute.Tensor
    us: cute.Tensor
    partials: cute.Tensor
    activation: cute.Tensor
    alpha: cute.Tensor
    epochs: cute.Tensor
    out: cute.Tensor
    inject: cute.Tensor


class MixerCta(NamedTuple):
    tid: cutlass.Int32
    warp: cutlass.Int32
    pid: cutlass.Int32
    count: cutlass.Int64


class MixerShared(NamedTuple):
    arena: cute.Tensor
    down_barriers: cute.Pointer
    up_barriers: cute.Pointer
    tmem: cute.Pointer
    epoch: cute.Tensor


class DownTask(NamedTuple):
    output_tile: cutlass.Int32
    split: cutlass.Int32
    token_tile: cutlass.Int64
    entry_prefilled: cutlass.Boolean
    last: cutlass.Boolean


class UpTask(NamedTuple):
    hidden_tile: cutlass.Int32
    weights_prepared: cutlass.Boolean
    last: cutlass.Boolean


class UpMmaFragments(NamedTuple):
    a: tuple[cute.Tensor, cute.Tensor]
    b: cute.Tensor
    acc: tuple[cute.Tensor, cute.Tensor]


class QuantizationConfig(NamedTuple):
    """Compile-time layout, row reduction, and activation publication choices."""

    threads: int
    maximum_threads: int
    injection_start: int
    column_stride: int
    warp_max: str
    replicate_maximum: bool
    refined_scales: bool
    clamp: str
    activation_store: str
    release_writes: bool


@cute.experimental.jit
def _largest_magnitude_pair(previous, first, second):
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [previous.ir_value(), first.ir_value(), second.ir_value()],
            "max.abs.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            has_side_effects=False,
            asm_dialect=0,
        )
    )


@cute.experimental.jit
def _rounded_quant_scales(maximum):
    """Compute the original FP32 scales for a BF16/FP16 row maximum."""
    # Clamp the maximum at 1/512. Scale the denominator by 1/64 so the
    # reciprocal is normal even at the largest finite BF16 maximum.
    # The residual FMA corrects the starting quotient before its final round.
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [maximum.ir_value()],
        "{ .reg .f32 x, a, a_err, den, rec, m, m_err, neg_den; .reg .pred infinite;"
        "max.f32 x, $2, 0f3B000000;"
        "setp.eq.f32 infinite, x, 0f7F800000;"
        "mul.rn.f32 a, x, 0f3B124925;"
        "fma.rn.f32 a_err, a, 0fC3E00000, x;"
        "fma.rn.f32 a, a_err, 0f3B124925, a;"
        "mul.rn.f32 den, x, 0f3C800000;"
        "rcp.approx.f32 rec, den;"
        "mul.rn.f32 m, rec, 0f40E00000;"
        "neg.f32 neg_den, den;"
        "fma.rn.f32 m_err, neg_den, m, 0f40E00000;"
        "fma.rn.f32 m, m_err, rec, m;"
        "selp.f32 $0, x, a, infinite;"
        "selp.f32 $1, 0f00000000, m, infinite; }",
        "=f,=f,f",
        has_side_effects=False,
        asm_dialect=0,
    )
    return (
        cutlass.Float32(llvm.extractvalue(T.f32(), result, [0])),
        cutlass.Float32(llvm.extractvalue(T.f32(), result, [1])),
    )


@cute.experimental.jit
def _mixer_cta():
    tid = cute.arch.thread_idx()[0]
    return MixerCta(
        tid,
        cute.arch.make_warp_uniform(tid // 32),
        cute.arch.block_idx()[0],
        cutlass.Int64(cute.arch.grid_dim()[0]),
    )


@cute.experimental.jit
def _mixer_shared(
    arena_bytes: cutlass.Constexpr,
    barrier_count: cutlass.Constexpr,
    up_barrier_offset: cutlass.Constexpr,
):
    arena = cute_ext.allocate(
        cutlass.Uint8,
        cute.AddressSpace.smem,
        cute.make_layout(arena_bytes),
        alignment=1024,
    )
    bars = cute_ext.allocate(
        cutlass.Int64,
        cute.AddressSpace.smem,
        cute.make_layout(barrier_count),
        alignment=8,
    )
    up_bars = bars.iterator
    if cutlass.const_expr(up_barrier_offset):
        up_bars = up_bars + up_barrier_offset
    base = cute_ext.allocate(
        cutlass.Int32, cute.AddressSpace.smem, cute.make_layout(1), alignment=4
    )
    epoch = cute_ext.allocate(
        cutlass.Int64, cute.AddressSpace.smem, cute.make_layout(1), alignment=8
    )
    return MixerShared(arena, bars.iterator, up_bars, base.iterator, epoch)


@cute.experimental.jit
def _read_down_partials(
    mma, base, partials, tid, shape: cutlass.Constexpr, groups: cutlass.Constexpr
):
    m, n, k = shape
    fragment = mma.get_slice(0).make_fragment_C(mma.partition_shape_C((m, n)))
    accumulator = cute.make_tensor(
        cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, base), fragment.layout
    )[((None, None), 0, 0)]
    columns = n // groups
    tile = (m, columns)
    group = tid // 128
    accumulator = cute.local_tile(accumulator, tile, (0, group))
    coords = cute.local_tile(cute.make_identity_tensor((m, n)), tile, (0, group))
    accumulator, reader, values, coords = _make_tmem_reader(
        accumulator, coords, partials, tid, (m, columns, k), tile
    )
    cute_ext.partition_and_copy(reader, accumulator, values)
    cute.arch.fence_view_async_tmem_load()
    return values, coords


class ResidentDown(Fp8GatedResidualGemm):
    @cute.experimental.jit
    def _feed_down_ring(
        self, operands, full, empty, split, entry_prefilled, refill_hint
    ):
        stage_bytes = (self.m + self.n) * self.k
        first_tile = split * self.tiles
        if not entry_prefilled:
            for stage in cutlass.range_constexpr(self.stages):
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(full + stage, stage_bytes)
                self._load_down_stage(operands, full, stage, first_tile + stage)
        if cutlass.const_expr(refill_hint is not None):
            descriptor, output_tile, token_tile = refill_hint
            self._hint_refill_input(descriptor, output_tile, split, token_tile)
        for tile in cutlass.range(self.stages, self.tiles, unroll=1):
            stage = tile % self.stages
            cute.arch.mbarrier_wait(empty + stage, (tile // self.stages - 1) % 2)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(full + stage, stage_bytes)
            self._load_down_stage(operands, full, stage, first_tile + tile)


class ResidentMixer:
    """Compile-time mixer schedule with a shared resident task loop."""

    @cute.experimental.jit
    def _down_tile_coordinates(self, task, token_round):
        output_tile = task // self.splits
        input_tile = token_round
        if cutlass.const_expr(self.down_token_parts > 1):
            token_part = output_tile // self.down_tiles
            input_tile = token_round * self.down_token_parts + cutlass.Int64(token_part)
            output_tile = output_tile % self.down_tiles
        return output_tile, task % self.splits, input_tile

    @cute.experimental.jit
    def _run_mixer_rounds(
        self,
        io,
        cta,
        shared,
        maps,
        state,
        phase,
        entry_prefills_down: cutlass.Constexpr,
        prefetch_residual_before_down: cutlass.Constexpr,
        skip_last_up_join: cutlass.Constexpr,
        track_last_up_task: cutlass.Constexpr,
    ):
        """Walk resident work in split-K, quantization, and Up order.

        The plan supplies operand binding and publication. The template owns task
        assignment, row traversal and barrier retirement before scratch reuse.
        """
        global_base = cutlass.Int64(0)
        token_round = cutlass.Int64(0)
        task_stride = cutlass.Int32(cta.count)
        row_stride = cta.count * self.quant_rows
        while global_base < io.x.shape[0]:
            if cutlass.const_expr(prefetch_residual_before_down):
                self._prefetch_round_residual(io, cta, shared, maps, state, global_base)

            task = cutlass.Int32(cta.pid)
            while task < self.down_tasks:
                output_tile, split, input_tile = self._down_tile_coordinates(
                    task, token_round
                )
                entry_prefilled = cutlass.Boolean(False)
                if cutlass.const_expr(entry_prefills_down):
                    entry_prefilled = (global_base == 0) & (
                        task == cutlass.Int32(cta.pid)
                    )
                tile = DownTask(
                    output_tile,
                    split,
                    input_tile,
                    entry_prefilled,
                    task + task_stride >= self.down_tasks,
                )
                self._project_down_tile(io, cta, shared, maps, state, tile, global_base)
                if cta.warp == self.mma_warp:
                    _invalidate(shared.down_barriers, self.down_barrier_count)
                cute.arch.sync_threads()
                task = task + task_stride

            phase, row, quant = self._begin_quantization(
                io, cta, shared, maps, state, phase, global_base
            )
            while row < self.n:
                self._quantize_row(io, cta, state, quant, phase, global_base, row)
                row = row + row_stride
            activation_target = self._finish_quantization(io, cta, state, phase, quant)

            task = cutlass.Int32(cta.pid)
            while task < self.up_tasks:
                last = cutlass.Boolean(False)
                if cutlass.const_expr(track_last_up_task):
                    last = (task + task_stride >= self.up_tasks) & (
                        global_base + self.n >= io.x.shape[0]
                    )
                tile = UpTask(task, task == cutlass.Int32(cta.pid), last)
                self._project_up_tile(
                    io, shared, maps, state, tile, global_base, activation_target
                )
                reuse = cutlass.Boolean(True)
                if cutlass.const_expr(skip_last_up_join):
                    reuse = not last
                if reuse:
                    cute.arch.sync_threads()
                    if cta.warp == self.mma_warp:
                        _invalidate(shared.up_barriers, self.up_barrier_count)
                    cute.arch.sync_threads()
                task = task + task_stride

            more_rounds = global_base + self.n < io.x.shape[0]
            phase = self._finish_round(
                io, cta, shared, state, phase, activation_target, more_rounds
            )
            token_round = token_round + 1
            global_base = global_base + self.n

    @cute.experimental.jit
    def _quantize_row(self, io, cta, state, quant, phase, global_base, row):
        self._quantize_down_row(io, cta.tid, global_base, row, None, None, None)
        cute.arch.sync_threads()

    @cute.experimental.jit
    def _partials(
        self,
        partial,
        row,
        col,
        columns: cutlass.Constexpr,
        injection: cutlass.Constexpr,
    ):
        pieces = cute.make_rmem_tensor((columns, self.splits), cutlass.Float32)
        row = cutlass.Int64(row)
        stride = cutlass.Int64(partial.shape[1]) * self.p
        base = partial.iterator + row * self.p + col
        interleaved = cutlass.const_expr(self.quantization.column_stride != 1)
        if cutlass.const_expr(interleaved):
            # Select the row from each native Down store packet.
            even_row = row - row % 2
            packed_col = 2 * col
            if cutlass.const_expr(not injection):
                packed_col += 2 * (col % self.quantization.column_stride)
            base = partial.iterator + even_row * self.p + packed_col
        return _reduce_split_partials(pieces, base, stride, row, interleaved)

    @cute.experimental.jit
    def _read_quant_scales(
        self, xs, ds, row, col, columns: cutlass.Constexpr, stride: cutlass.Constexpr
    ):
        input_scale = _x_scale(xs, row)
        weight_scales = cute.make_rmem_tensor(columns, cutlass.Float32)
        for i in cutlass.range_constexpr(columns):
            weight_scales[i] = ds[col + i * stride if self.down_per_channel else 0]
        return input_scale, weight_scales

    @cute.experimental.jit
    def _quant_projection_values(
        self,
        io,
        dst,
        row,
        global_row,
        col,
        preloaded_scales,
        stride: cutlass.Constexpr,
        injection: cutlass.Constexpr,
    ):
        columns = cute.size(dst)
        if cutlass.const_expr(preloaded_scales is None):
            input_scale, weight_scales = self._read_quant_scales(
                io.xs, io.ds, global_row, col, columns, stride
            )
        else:
            input_scale, weight_scales = preloaded_scales
        values = self._partials(io.partials, row, col, columns, injection)
        for i in cutlass.range_constexpr(columns):
            value = _scale_down_value(
                values[i], input_scale, weight_scales[i], self.scale, not injection
            )
            dst[i] = value.to(dst.element_type)

    @cute.experimental.jit
    def _quant_warp_max(self, value):
        if cutlass.const_expr(self.quantization.warp_max == "shuffle"):
            return cute.arch.warp_reduction_max(value, threads_in_group=32)
        return cute.arch.warp_redux_sync(
            value, "fmax", mask_and_clamp=0xFFFFFFFF, abs=False, nan=False
        )

    @cute.experimental.jit
    def _publish_quant_max(self, rounded, shared, group, local_tid):
        maximum = cutlass.Float32(0.0)
        if cutlass.const_expr(self.quantization.warp_max == "pair"):
            for pair in cutlass.range_constexpr(self.quant_columns // 2):
                maximum = _largest_magnitude_pair(
                    maximum,
                    rounded[2 * pair].to(cutlass.Float32),
                    rounded[2 * pair + 1].to(cutlass.Float32),
                )
        else:
            for i in cutlass.range_constexpr(self.quant_columns):
                maximum = cute.math.max(
                    maximum, cute.math.abs(rounded[i].to(cutlass.Float32))
                )
        maximum = self._quant_warp_max(maximum)
        if local_tid % 32 == 0:
            shared[group, local_tid // 32] = maximum

    @cute.experimental.jit
    def _quant_row_max(self, shared, group, lane, row_warps: cutlass.Constexpr):
        maximum = cutlass.Float32(0.0)
        if lane < row_warps:
            maximum = shared[group, lane]
        return self._quant_warp_max(maximum)

    @cute.experimental.jit
    def _store_quant_alpha(self, io, row, scale, target):
        io.alpha[row] = scale

    @cute.experimental.jit
    def _store_quant_injection(self, io, row, global_row, col, preloaded_scales):
        pair = cute.make_rmem_tensor(2, io.out.element_type)
        self._quant_projection_values(
            io, pair, row, global_row, self.r + col, preloaded_scales, 1, True
        )
        for i in cutlass.range_constexpr(2):
            io.inject[global_row, col + i] = pair[i]

    @cute.experimental.jit
    def _quantize_down_row(
        self,
        io,
        tid,
        global_base,
        row_base,
        preloaded_scales,
        activation_layout,
        target,
    ):
        """Reduce Down partials and quantize the rounded SiLU result per row."""
        dtype = io.out.element_type
        rows = io.x.shape[0]
        first = cutlass.const_expr(self.quantization.injection_start)
        if cutlass.const_expr(self.p > self.r and first != 0):
            if first <= tid < first + self.quant_rows * (self.g // 2):
                lane = tid - first
                row = row_base
                if cutlass.const_expr(self.quant_rows > 1):
                    row = row_base + cutlass.Int64(lane // (self.g // 2))
                global_row = global_base + row
                col = (lane % (self.g // 2)) * 2
                if global_row < rows:
                    self._store_quant_injection(
                        io, row, global_row, col, preloaded_scales
                    )

        if tid < self.quantization.threads:
            group = cutlass.Int32(0)
            local_tid = tid
            row = row_base
            if cutlass.const_expr(self.quant_rows > 1):
                group = tid // self.quant_threads
                local_tid = tid % self.quant_threads
                row = row_base + cutlass.Int64(group)
            global_row = global_base + row
            stride = cutlass.const_expr(self.quantization.column_stride)
            col = local_tid * self.quant_columns
            if cutlass.const_expr(stride != 1):
                col = (local_tid // stride) * (2 * stride) + local_tid % stride
            rounded = cute.make_rmem_tensor(self.quant_columns, dtype)
            rounded.fill(0.0)
            if cutlass.const_expr(self.p > self.r and first == 0):
                # The fallback's first quantizer lanes also write the injection.
                if global_row < rows and local_tid < self.g // 2:
                    self._store_quant_injection(
                        io, row, global_row, 2 * local_tid, preloaded_scales
                    )
            if global_row < rows and col < self.r:
                self._quant_projection_values(
                    io, rounded, row, global_row, col, preloaded_scales, stride, False
                )

            row_warps = self.quant_threads // 32
            shared = cute_ext.allocate(
                cutlass.Float32,
                cute.AddressSpace.smem,
                cute.make_layout(
                    (self.quant_rows, row_warps + 2), stride=(row_warps + 2, 1)
                ),
                alignment=32,
            )
            max_threads = self.quantization.maximum_threads
            if cutlass.const_expr(max_threads < self.quant_threads):
                if local_tid < max_threads:
                    self._publish_quant_max(rounded, shared, group, local_tid)
                    _epilogue_barrier(max_threads, cutlass.Int32(5 + group))
            else:
                self._publish_quant_max(rounded, shared, group, local_tid)
                _epilogue_barrier(self.quant_threads, cutlass.Int32(5 + group))

            if cutlass.const_expr(not self.quantization.replicate_maximum):
                if local_tid < 32:
                    maximum = self._quant_row_max(shared, group, local_tid, row_warps)
                    if local_tid == 0:
                        if cutlass.const_expr(self.quantization.refined_scales):
                            scale, multiplier = _rounded_quant_scales(maximum)
                        else:
                            scale = _quant_scale(maximum, False)
                            multiplier = _quant_multiplier(maximum, False)
                        self._store_quant_alpha(io, row, scale, target)
                        shared[group, row_warps + 1] = multiplier
                _epilogue_barrier(self.quant_threads, cutlass.Int32(5 + group))

            if col < self.r:
                maximum = cutlass.Float32(0.0)
                if cutlass.const_expr(self.quantization.replicate_maximum):
                    maximum = self._quant_row_max(
                        shared, group, local_tid % 32, max_threads // 32
                    )
                    if cutlass.const_expr(self.quantization.refined_scales):
                        scale, multiplier = _rounded_quant_scales(maximum)
                    else:
                        multiplier = _quant_multiplier(maximum, False)
                    if cutlass.const_expr(
                        self.quantization.activation_store != "stamped"
                    ):
                        if local_tid == 0:
                            if cutlass.const_expr(self.quantization.refined_scales):
                                self._store_quant_alpha(io, row, scale, target)
                            else:
                                self._store_quant_alpha(
                                    io, row, _quant_scale(maximum, False), target
                                )
                else:
                    multiplier = shared[group, row_warps + 1]

                if cutlass.const_expr(self.quantization.activation_store == "native"):
                    for i in cutlass.range_constexpr(self.quant_columns):
                        value = _quantized_value(
                            rounded[i], multiplier, self.quantization.clamp
                        )
                        io.activation[row, col + i * stride] = value.to(
                            cutlass.Float8E4M3FN
                        )
                else:
                    self._store_packed_quantization(
                        io,
                        row,
                        col,
                        local_tid,
                        rounded,
                        multiplier,
                        maximum,
                        preloaded_scales,
                        activation_layout,
                        target,
                    )
                if cutlass.const_expr(self.quantization.release_writes):
                    _publish_activation_proxy()

    @cute.experimental.jit
    def _make_up_mma(self):
        dtype = cutlass.Float8E4M3FN
        return _make_k_major_mma(dtype, self.up_mn)

    @cute.experimental.jit
    def _up_operand_layouts(self):
        dtype = cutlass.Float8E4M3FN
        mma = self._make_up_mma()
        tiler = (*self.up_mn, self.up_k)
        al = sm100_utils.make_smem_layout_a(mma, tiler, dtype, self.up_stages)
        bl = sm100_utils.make_smem_layout_b(mma, tiler, dtype, self.up_stages)
        return mma, tiler, al, bl

    @cute.experimental.jit
    def _up_operand_storage(self, arena, al, bl):
        sw = _operand_at(arena, bl, 0)
        sa0 = _operand_at(arena, al, cute.cosize(bl))
        sa1 = _operand_at(arena, al, cute.cosize(bl) + cute.cosize(al))
        return sw, sa0, sa1

    @cute.experimental.jit
    def _up_weight_view(self, weight):
        return _tma_matrix_view(
            weight,
            (self.up_hidden, self.g, self.h // self.up_hidden),
            (self.r, self.h * self.r, self.up_hidden * self.r),
            self.r,
        )

    @cute.experimental.jit
    def _up_tma_operands(self, mma, tiler, activation, weight, hidden_tile):
        av = _tma_matrix_view(activation, activation.shape[0], self.r, self.r)
        wv = self._up_weight_view(weight)
        gx0 = cute.local_tile(av, (self.up_rows, self.up_k), (0, None, 0))
        gx1 = cute.local_tile(av, (self.up_rows, self.up_k), (1, None, 0))
        gw = cute.local_tile(wv, (self.up_channels, self.up_k), (hidden_tile, None, 0))
        if cutlass.const_expr(self.token_major_up):
            ga0, ga1, gb = gx0, gx1, gw
            ma = cute_ext.get_cta_v_map_ab(av, tiler, mma, "A")
            mb = cute_ext.get_cta_v_map_ab(wv, tiler, mma, "B")
        else:
            ga0, ga1, gb = gw, gw, gx0
            ma = cute_ext.get_cta_v_map_ab(wv, tiler, mma, "A")
            mb = cute_ext.get_cta_v_map_ab(av, tiler, mma, "B")
        return ga0, ga1, gb, ma, mb

    @cute.experimental.jit
    def _load_up_tma_operand(self, source, target, tile, slot, full, mapping):
        cute_ext.tma_load(
            source[None, None, tile],
            target[None, None, None, slot],
            full.value,
            cta_v_map=mapping,
            update_expect_tx=False,
        )

    @cute.experimental.jit
    def _up_scale_storage(self, weight_offset: cutlass.Constexpr):
        return cute_ext.allocate(
            cutlass.Float32,
            cute.AddressSpace.smem,
            cute.make_layout(weight_offset + self.up_channels),
            alignment=16,
        )

    @cute.experimental.jit
    def _up_residual_tile(
        self, row_stride: cutlass.Constexpr, alignment: cutlass.Constexpr
    ):
        return cute_ext.allocate(
            cutlass.Float8E4M3FN,
            cute.AddressSpace.smem,
            cute.make_layout((self.up_rows, self.up_channels), stride=(row_stride, 1)),
            alignment=alignment,
        )

    @cute.experimental.jit
    def _load_hidden_packet(
        self,
        tensor,
        offset,
        dtype: cutlass.Constexpr,
        width: cutlass.Constexpr,
        space: cutlass.Constexpr,
    ):
        pointer = cute.make_ptr(
            dtype,
            (tensor.iterator + offset).toint(),
            space,
            assumed_align=width * dtype.width // 8,
        )
        source = cute.make_tensor(pointer, cute.make_layout(width))
        values = cute.make_rmem_tensor(width, dtype)
        atom = cute.make_copy_atom(
            CopyUniversalOp(), dtype, num_bits_per_copy=width * dtype.width
        )
        cute.copy(atom, source, values)
        return values

    @cute.experimental.jit
    def _up_mma_fragments(self, mma, base, sa0, sa1, sw):
        tm = mma.get_slice(0)
        ad0, bd = tm.make_fragment_A(sa0), tm.make_fragment_B(sw)
        ad1 = ad0
        if cutlass.const_expr(self.up_parts == 2):
            ad1 = tm.make_fragment_A(sa1)
        layout = cute_ext.make_tmem_layout_acc(mma, self.up_mn, acc_stage=self.up_parts)
        all_acc = cute.make_tensor(
            cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, base), layout
        )
        acc0 = all_acc[None, None, None, 0]
        acc1 = acc0
        if cutlass.const_expr(self.up_parts == 2):
            acc1 = all_acc[None, None, None, 1]
        return UpMmaFragments((ad0, ad1), bd, (acc0, acc1))

    @cute.experimental.jit
    def _issue_up_mma_block(self, mma, fragments, block, slot, part: cutlass.Constexpr):
        acc = fragments.acc[part]
        cute.gemm(
            mma,
            acc,
            fragments.a[part][None, None, block, slot],
            fragments.b[None, None, block, slot],
            acc,
        )

    @cute.experimental.jit
    def _issue_up_mma_pair(self, mma, fragments, block, slot):
        for part in cutlass.range_constexpr(self.up_parts):
            self._issue_up_mma_block(mma, fragments, block, slot, part)

    @cute.experimental.jit
    def _issue_up_mma_part(self, mma, fragments, slot, part: cutlass.Constexpr):
        for block in cutlass.range_constexpr(cute.size(fragments.a[part], mode=[2])):
            self._issue_up_mma_block(mma, fragments, block, slot, part)

    @cute.experimental.jit
    def _retire_up_slot(self, empty, slot, tile, unused_empty: cutlass.Constexpr):
        notify = cutlass.Boolean(True)
        if cutlass.const_expr(unused_empty != 0):
            notify = tile < self.up_tiles - unused_empty
        if notify:
            with cute.arch.elect_one():
                tcgen05.commit(empty + slot, None, tcgen05.CtaGroup.ONE)

    @cute.experimental.jit
    def _run_up_mma_tiles(
        self,
        mma,
        fragments,
        full,
        empty,
        suffix,
        tiles: cutlass.Constexpr,
        unroll_full: cutlass.Constexpr,
    ):
        """Wait, accumulate and optionally release each Up operand slot."""
        for tile in cutlass.range(
            tiles, unroll=-1 if unroll_full else 1, unroll_full=unroll_full
        ):
            slot = tile % self.up_stages
            cute.arch.mbarrier_wait(full + slot, tile // self.up_stages % 2)
            for block in cutlass.range_constexpr(cute.size(fragments.a[0], mode=[2])):
                if cutlass.const_expr(suffix is not None and block == suffix[1]):
                    cute.arch.mbarrier_wait(suffix[0], 0)
                self._issue_up_mma_pair(mma, fragments, block, slot)
                # CuTe tracks the accumulator state across the loop here.
                mma.set(tcgen05.Field.ACCUMULATE, True)
            if cutlass.const_expr(empty is not None):
                self._retire_up_slot(empty[0], slot, tile, empty[1])

    @cute.experimental.jit
    def _run_paired_up_tail(
        self, mma, fragments, full, empty, done, unroll_full: cutlass.Constexpr
    ):
        # Start from the prefix's accumulated fragments across the JIT boundary.
        mma.set(tcgen05.Field.ACCUMULATE, self.up_tiles > self.up_tail_tiles)
        for part in cutlass.range_constexpr(2):
            for tile in cutlass.range(
                self.up_tiles - self.up_tail_tiles,
                self.up_tiles,
                unroll=-1 if unroll_full else 1,
                unroll_full=unroll_full,
            ):
                slot = tile % self.up_stages
                if cutlass.const_expr(part == 0):
                    cute.arch.mbarrier_wait(full + slot, tile // self.up_stages % 2)
                self._issue_up_mma_part(mma, fragments, slot, part)
                if cutlass.const_expr(part == 1):
                    with cute.arch.elect_one():
                        tcgen05.commit(empty + slot, None, tcgen05.CtaGroup.ONE)
            with cute.arch.elect_one():
                tcgen05.commit(done + part, None, tcgen05.CtaGroup.ONE)

    @cute.experimental.jit
    def _publish_up_done(self, done):
        for part in cutlass.range_constexpr(self.up_parts):
            with cute.arch.elect_one():
                tcgen05.commit(done + part, None, tcgen05.CtaGroup.ONE)

    @cute.experimental.jit
    def _reader(
        self,
        mma,
        base,
        original,
        tid,
        branch: cutlass.Constexpr,
        part: cutlass.Constexpr,
    ):
        return _token_major_reader(
            mma,
            base,
            original,
            tid,
            branch,
            part,
            (self.up_rows, self.up_channels, self.up_k),
            self.up_parts,
            self.up_hidden,
            self.reader_columns,
        )

    @cute.experimental.jit
    def _hidden_reader(
        self,
        mma,
        base,
        original,
        tid,
        part: cutlass.Constexpr,
    ):
        if cutlass.const_expr(part is None):
            fragment = mma.get_slice(0).make_fragment_C(
                mma.partition_shape_C((128, self.up_rows))
            )
            layout = fragment.layout
        else:
            layout = cute_ext.make_tmem_layout_acc(
                mma, self.up_mn, acc_stage=self.up_parts
            )
        accumulator = cute.make_tensor(
            cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, base), layout
        )
        if cutlass.const_expr(part is None):
            accumulator = accumulator[(None, None), 0, 0]
        else:
            accumulator = accumulator[((None, None), 0, 0, part)]
        return _tiled_tmem_reader(
            accumulator,
            original,
            tid,
            tid // 128,
            (128, self.up_rows, self.up_k),
            self.hidden_reader_columns,
        )

    @cute.experimental.jit
    def _read_hidden_up(self, mma, base, original, tid, part: cutlass.Constexpr):
        acc, reader, values, coords = self._hidden_reader(
            mma, base, original, tid, part
        )
        cute_ext.partition_and_copy(reader, acc, values)
        cute.arch.fence_view_async_tmem_load()
        return values, coords

    @cute.experimental.jit
    def _prefetch_branch_x(
        self,
        target,
        coords,
        x,
        hidden_tile,
        global_base,
        part: cutlass.Constexpr,
        branch: cutlass.Constexpr,
    ):
        _prefetch_branch_residual(
            target,
            coords,
            x,
            hidden_tile,
            global_base,
            part,
            branch,
            self.up_rows,
            self.up_hidden,
            self.g,
            self.h,
        )

    @cute.experimental.jit
    def _copy_hidden_residual(
        self,
        x,
        target,
        hidden_tile,
        global_base,
        tid,
        threads: cutlass.Constexpr,
        row_stride: cutlass.Constexpr,
    ):
        tile_bytes = self.up_rows * self.up_channels
        copy_bytes = threads * 16
        for copy_slice in cutlass.range_constexpr(
            (tile_bytes + copy_bytes - 1) // copy_bytes
        ):
            linear = tid * 16 + copy_slice * copy_bytes
            if linear < tile_bytes:
                row, channel = linear // self.up_channels, linear % self.up_channels
                global_row = global_base + cutlass.Int64(row)
                source = (
                    x.iterator
                    + global_row * (self.g * self.h)
                    + (channel // self.up_hidden) * self.h
                    + hidden_tile * self.up_hidden
                    + channel % self.up_hidden
                )
                size = cutlass.Int32(0)
                if global_row < x.shape[0]:
                    size = cutlass.Int32(16)
                offset = linear
                if cutlass.const_expr(row_stride != self.up_channels):
                    offset = row * row_stride + channel
                _copy_residual_16(target.iterator + offset, source, size)
        _commit_residual()

    @cute.experimental.jit
    def _cache_up_row_scales(
        self,
        alpha,
        xs,
        cached,
        x,
        global_base,
        tid,
        row_count: cutlass.Constexpr,
        read_alpha: cutlass.Constexpr,
    ):
        if tid < row_count:
            global_row = global_base + cutlass.Int64(tid)
            activation_scale = cutlass.Float32(0.0)
            input_scale = cutlass.Float32(0.0)
            if global_row < x.shape[0]:
                if cutlass.const_expr(read_alpha):
                    activation_scale = _load_f32(alpha.iterator + cutlass.Int64(tid))
                input_scale = _x_scale(xs, global_row)
            if cutlass.const_expr(read_alpha):
                cached[tid] = activation_scale
            cached[row_count + tid] = input_scale

    @cute.experimental.jit
    def _cache_up_weight_scales(
        self, ws, cached, hidden_tile, tid, offset: cutlass.Constexpr, prepared
    ):
        # Keep the prepared-scale check in the warps that read the scales.
        if tid < self.up_channels:
            if not prepared:
                channel = (
                    (tid // self.up_hidden) * self.h
                    + hidden_tile * self.up_hidden
                    + tid % self.up_hidden
                )
                cached[offset + tid] = ws[channel if self.up_per_channel else 0]

    @cute.experimental.jit
    def _hidden_gate_tile(self, pointer):
        return cute.make_tensor(
            cute.make_ptr(
                cutlass.Float32,
                pointer.toint(),
                cute.AddressSpace.smem,
                assumed_align=1024,
            ),
            cute.make_layout(
                (self.up_rows, self.up_channels), stride=(self.up_channels, 1)
            ),
        )

    @cute.experimental.jit
    def _mix_hidden_major_up(
        self,
        gates,
        residual_tile,
        cached,
        out,
        rows,
        tid,
        hidden_tile,
        global_base,
        threads: cutlass.Constexpr,
    ):
        col = tid % self.up_hidden
        hidden = hidden_tile * self.up_hidden + col
        scales = cute.make_rmem_tensor(self.g, cutlass.Float32)
        for branch in cutlass.range_constexpr(self.g):
            scales[branch] = cached[2 * self.up_rows + branch * self.up_hidden + col]
        rows_per_step = threads // self.up_hidden
        for row_part in cutlass.range_constexpr(self.up_rows // rows_per_step):
            row = tid // self.up_hidden + row_part * rows_per_step
            global_row = global_base + cutlass.Int64(row)
            if global_row < rows:
                activation_scale = cached[row]
                input_scale = cached[self.up_rows + row]
                total = cutlass.Float32(0.0)
                for branch in cutlass.range_constexpr(self.g):
                    channel = branch * self.up_hidden + col
                    gate = gates[row, channel]
                    factor = activation_scale * scales[branch]
                    code = residual_tile[row, channel]
                    res = code.to(cutlass.Float32) * input_scale
                    total = total + res * _up_sigmoid_tanh(gate * factor)
                out[global_row, hidden] = (total / self.g).to(out.element_type)

    @cute.experimental.jit
    def _mix_token_major_up(
        self,
        mma,
        base,
        x,
        xs,
        alpha,
        out,
        cached,
        tid,
        hidden_tile,
        global_base,
        done,
        read_done,
        first_residual,
        row_scale_source: cutlass.Constexpr,
        weight_cache_offset: cutlass.Constexpr,
        vector_weight_scales: cutlass.Constexpr,
        prefetched_branches: cutlass.Constexpr,
        wait_once: cutlass.Constexpr,
        release_after_reads: cutlass.Constexpr,
    ):
        """Reduce the branches for either resident token-major Up layout."""
        for part in cutlass.range_constexpr(self.up_parts):
            if cutlass.const_expr(prefetched_branches and part == 0):
                template, coords, _ = first_residual
            else:
                _, _, template, coords = self._reader(mma, base, x, tid, 0, part)
            sums = cute.make_rmem_tensor(cute.size(template), cutlass.Float32)
            sums.fill(0.0)
            row_a_scale = cutlass.Float32(0.0)
            row_x_scale = cutlass.Float32(0.0)
            if cutlass.const_expr(row_scale_source != "shared_element"):
                row, _ = coords[0]
                local_row = part * self.up_rows + row
                if cutlass.const_expr(row_scale_source == "global"):
                    global_row = global_base + cutlass.Int64(local_row)
                    if global_row < x.shape[0]:
                        row_a_scale = _load_f32(
                            alpha.iterator + cutlass.Int64(local_row)
                        )
                        row_x_scale = _x_scale(xs, global_row)
                else:
                    row_a_scale = cached[local_row]
                    row_x_scale = cached[self.n + local_row]
            if cutlass.const_expr(prefetched_branches and part == 0):
                _, _, x_ring = first_residual
            else:
                x_ring = cute.make_rmem_tensor(
                    (cute.size(template), 2), cutlass.Float8E4M3FN
                )
                x_ring.fill(0.0)
                self._prefetch_branch_x(
                    x_ring[None, 0], coords, x, hidden_tile, global_base, part, 0
                )
            for branch in cutlass.range_constexpr(self.g):
                if cutlass.const_expr(
                    branch + 1 < self.g
                    and (part != 0 or branch + 1 >= prefetched_branches)
                ):
                    self._prefetch_branch_x(
                        x_ring[None, (branch + 1) % 2],
                        coords,
                        x,
                        hidden_tile,
                        global_base,
                        part,
                        branch + 1,
                    )
                decoded_x = _decode_residual(x_ring[None, branch % 2])
                if cutlass.const_expr(not wait_once or branch == 0):
                    cute.arch.mbarrier_wait(done + part, 0)
                accumulator, reader, values, _ = self._reader(
                    mma, base, x, tid, branch, part
                )
                cute_ext.partition_and_copy(reader, accumulator, values)
                cute.arch.fence_view_async_tmem_load()
                if cutlass.const_expr(
                    release_after_reads
                    and part + 1 == self.up_parts
                    and branch + 1 == self.g
                ):
                    tcgen05_fence("before_thread_sync")
                    cute.arch.mbarrier_arrive(read_done)
                channel_scales = None
                if cutlass.const_expr(vector_weight_scales):
                    _, col = coords[0]
                    scale_pointer = cute.make_ptr(
                        cutlass.Float32,
                        (
                            cached.iterator
                            + weight_cache_offset
                            + branch * self.up_hidden
                            + col
                        ).toint(),
                        cute.AddressSpace.smem,
                        assumed_align=16,
                    )
                    layout = cute.make_layout(cute.size(template))
                    channel_scales = cute.make_rmem_tensor(layout, cutlass.Float32)
                    cute.copy(
                        cute.make_copy_atom(
                            CopyUniversalOp(), cutlass.Float32, num_bits_per_copy=128
                        ),
                        cute.make_tensor(scale_pointer, layout),
                        channel_scales,
                    )
                for i in cutlass.range_constexpr(cute.size(values)):
                    row, col = coords[i]
                    local_row = part * self.up_rows + row
                    global_row = global_base + cutlass.Int64(local_row)
                    if global_row < x.shape[0]:
                        a_scale, x_scale = row_a_scale, row_x_scale
                        if cutlass.const_expr(row_scale_source == "shared_element"):
                            a_scale = cached[local_row]
                            x_scale = cached[self.n + local_row]
                        if cutlass.const_expr(vector_weight_scales):
                            weight_scale = channel_scales[i]
                        else:
                            weight_scale = cached[
                                weight_cache_offset + branch * self.up_hidden + col
                            ]
                        factor = a_scale * weight_scale
                        xv = decoded_x[i] * x_scale
                        sums[i] = sums[i] + xv * _up_sigmoid_tanh(values[i] * factor)
            _store_token_major_mix(
                out,
                sums,
                coords,
                hidden_tile,
                global_base,
                x.shape[0],
                part,
                self.up_rows,
                self.up_hidden,
                self.g,
                self.h,
            )
