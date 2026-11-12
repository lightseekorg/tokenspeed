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

"""FP8-resident HC projections with TMA, tcgen05, and a fused Up epilogue."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.utils as cute_utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cuda.bindings.driver import CUstream
from cutlass._mlir.dialects import llvm
from cutlass._mlir.extras import types as T
from cutlass.cute import experimental as cute_ext
from cutlass.cute.nvgpu import tcgen05
from cutlass.experimental.primitives import tcgen05_fence
from tokenspeed_kernel.thirdparty.cute_dsl.hc_fused import (
    _epilogue_barrier,
    _up_sigmoid_tanh,
)


@cute.experimental.jit
def _convert_fp8_weight(source, target):
    if cutlass.const_expr(target.element_type == cutlass.BFloat16):
        codes = cute.recast_tensor(source, cutlass.Uint16)
        packed = cute.recast_tensor(target, cutlass.Uint32)
        for i in cutlass.range_constexpr(cute.size(packed)):
            result = llvm.inline_asm(
                T.i32(),
                [codes[i].ir_value()],
                "cvt.rn.bf16x2.e4m3x2 $0, $1;",
                "=r,h",
                has_side_effects=False,
                asm_dialect=0,
            )
            packed[i] = cutlass.Uint32(result)
    else:
        target.store(source.load().to(target.element_type))


class Fp8GatedResidualProjectionEpilogue:
    """Prefetch split partials in registers, then retain their summation order."""

    def __init__(
        self,
        *,
        lowrank: int,
        projection_rows: int,
        hc_count: int,
        splits: int,
        projection_scale: float,
        input_fp8: bool,
        activation_fp8: bool,
        weight_scale_per_channel: bool,
        use_pdl: bool,
        block_threads: int,
        early_up_signal: bool,
        parallel_inject: bool,
    ):
        self.r = lowrank
        self.p = projection_rows
        self.g = hc_count
        self.splits = splits
        self.scale = projection_scale
        self.input_fp8 = input_fp8
        self.activation_fp8 = activation_fp8
        self.per_channel = weight_scale_per_channel
        self.pdl = use_pdl
        self.threads = block_threads
        self.early_up_signal = early_up_signal
        self.parallel_inject = parallel_inject

    @cute.experimental.jit
    def __call__(
        self,
        partial,
        input_scale,
        weight_scale,
        activation,
        activation_scale,
        inject,
        stream: CUstream,
    ):
        self.kernel(
            partial, input_scale, weight_scale, activation, activation_scale, inject
        ).launch(
            grid=(partial.shape[1], 1, 1),
            block=(self.threads + (32 if self.parallel_inject else 0), 1, 1),
            stream=stream,
            use_pdl=self.pdl,
        )

    @cute.experimental.jit
    def _partials(self, partial, row, col):
        # Promote before split/row strides can exceed32bit element offsets.
        row = cutlass.Int64(row)
        pieces = cute.make_rmem_tensor((2, self.splits), cutlass.Float32)
        row_base = partial.iterator + row * self.p + col
        split_stride = cutlass.Int64(partial.shape[1]) * self.p
        for rank in cutlass.range_constexpr(self.splits):
            source = cute.make_tensor(
                cute.make_ptr(
                    cutlass.Float32,
                    (row_base + rank * split_stride).toint(),
                    cute.AddressSpace.gmem,
                    assumed_align=8,
                ),
                cute.make_layout(2),
            )
            cute.copy(
                cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(), cutlass.Float32, num_bits_per_copy=64
                ),
                source,
                pieces[None, rank],
            )
        value = cute.make_rmem_tensor(2, cutlass.Float32)
        value.fill(0.0)
        for rank in cutlass.range_constexpr(self.splits):
            for i in cutlass.range_constexpr(2):
                value[i] = value[i] + pieces[i, rank]
        return value

    @cute.experimental.kernel
    def kernel(
        self, partial, input_scale, weight_scale, activation, activation_scale, inject
    ):
        row = cutlass.Int64(cute.arch.block_idx()[0])
        tid = cute.arch.thread_idx()[0]
        col = tid * 2
        if cutlass.const_expr(self.pdl):
            cute.arch.griddepcontrol_wait()
            if cutlass.const_expr(self.early_up_signal):
                # Independent Up weights usually overlap activation quantization.
                if tid == 0:
                    cute.arch.griddepcontrol_launch_dependents()
        if cutlass.const_expr(self.parallel_inject):
            if cutlass.const_expr(self.p > self.r):
                inject_col = (tid - self.threads) * 2
                if tid >= self.threads and tid < self.threads + self.g // 2:
                    values = self._partials(partial, row, self.r + inject_col)
                    for i in cutlass.range_constexpr(2):
                        value = values[i]
                        if cutlass.const_expr(self.input_fp8):
                            value = value * input_scale[row]
                        value = (
                            value
                            * weight_scale[
                                self.r + inject_col + i if self.per_channel else 0
                            ]
                            * cutlass.Float32(self.scale)
                        )
                        inject[row, inject_col + i] = value.to(inject.element_type)
            if tid < self.threads:
                self._activation_epilogue(
                    partial,
                    input_scale,
                    weight_scale,
                    activation,
                    activation_scale,
                    inject,
                    row,
                    tid,
                    col,
                )
        else:
            if cutlass.const_expr(self.p > self.r):
                if tid < self.g // 2:
                    values = self._partials(partial, row, self.r + col)
                    for i in cutlass.range_constexpr(2):
                        value = values[i]
                        if cutlass.const_expr(self.input_fp8):
                            value = value * input_scale[row]
                        value = (
                            value
                            * weight_scale[self.r + col + i if self.per_channel else 0]
                            * cutlass.Float32(self.scale)
                        )
                        inject[row, col + i] = value.to(inject.element_type)
            self._activation_epilogue(
                partial,
                input_scale,
                weight_scale,
                activation,
                activation_scale,
                inject,
                row,
                tid,
                col,
            )

    @cute.experimental.jit
    def _activation_epilogue(
        self,
        partial,
        input_scale,
        weight_scale,
        activation,
        activation_scale,
        inject,
        row,
        tid,
        col,
    ):
        rounded = cute.make_rmem_tensor(2, inject.element_type)
        rounded.fill(0.0)
        if col < self.r:
            values = self._partials(partial, row, col)
            for i in cutlass.range_constexpr(2):
                value = values[i]
                if cutlass.const_expr(self.input_fp8):
                    value = value * input_scale[row]
                value = value * weight_scale[col + i if self.per_channel else 0]
                value = value * cutlass.Float32(self.scale)
                value = value * cute.arch.rcp_approx(
                    1.0 + cute.exp(-value, fastmath=True)
                )
                rounded[i] = value.to(inject.element_type)
        if cutlass.const_expr(self.activation_fp8):
            warp, lane = tid // 32, tid % 32
            shared = cute_ext.allocate(
                cutlass.Float32,
                cute.AddressSpace.smem,
                cute.make_layout(self.threads // 32 + 2),
                alignment=32,
            )
            maximum = cutlass.Float32(0.0)
            for i in cutlass.range_constexpr(2):
                maximum = cute.math.max(
                    maximum, cute.math.abs(rounded[i].to(cutlass.Float32))
                )
            maximum = cute.arch.warp_reduction_max(maximum, threads_in_group=32)
            if lane == 0:
                shared[warp] = maximum
            if cutlass.const_expr(self.parallel_inject):
                _epilogue_barrier(self.threads, cutlass.Int32(5))
            else:
                cute.arch.sync_threads()
            if warp == 0:
                maximum = cutlass.Float32(0.0)
                if lane < self.threads // 32:
                    maximum = shared[lane]
                maximum = cute.arch.warp_reduction_max(maximum, threads_in_group=32)
                if lane == 0:
                    # Scale and multiplier use independent FP32 RN divisions.
                    scale = cute.math.max(
                        maximum / cutlass.Float32(448.0),
                        cutlass.Float32(1.0 / (448.0 * 512.0)),
                    )
                    multiplier = cute.math.min(
                        cutlass.Float32(448.0) / maximum,
                        cutlass.Float32(448.0 * 512.0),
                    )
                    shared[self.threads // 32] = scale
                    shared[self.threads // 32 + 1] = multiplier
                    activation_scale[row] = scale
            if cutlass.const_expr(self.parallel_inject):
                _epilogue_barrier(self.threads, cutlass.Int32(5))
            else:
                cute.arch.sync_threads()
            if col < self.r:
                values = cute.make_rmem_tensor(2, cutlass.Float32)
                multiplier = shared[self.threads // 32 + 1]
                for i in cutlass.range_constexpr(2):
                    values[i] = cute.math.min(
                        cute.math.max(
                            rounded[i].to(cutlass.Float32) * multiplier,
                            cutlass.Float32(-448.0),
                        ),
                        cutlass.Float32(448.0),
                    )
                quantized = cute.make_rmem_tensor(2, activation.element_type)
                quantized.store(values.load().to(activation.element_type))
                target = cute.make_tensor(
                    cute.make_ptr(
                        activation.element_type,
                        (activation.iterator + row * self.r + col).toint(),
                        cute.AddressSpace.gmem,
                        assumed_align=2,
                    ),
                    cute.make_layout(2),
                )
                cute.copy(
                    cute.make_copy_atom(
                        cute.nvgpu.CopyUniversalOp(),
                        activation.element_type,
                        num_bits_per_copy=16,
                    ),
                    quantized,
                    target,
                )
        else:
            if col < self.r:
                target = cute.make_tensor(
                    cute.make_ptr(
                        activation.element_type,
                        (activation.iterator + row * self.r + col).toint(),
                        cute.AddressSpace.gmem,
                        assumed_align=4,
                    ),
                    cute.make_layout(2),
                )
                cute.copy(
                    cute.make_copy_atom(
                        cute.nvgpu.CopyUniversalOp(),
                        activation.element_type,
                        num_bits_per_copy=32,
                    ),
                    rounded,
                    target,
                )
        if cutlass.const_expr(self.pdl and not self.early_up_signal):
            if tid == 0:
                cute.arch.griddepcontrol_launch_dependents()


class Fp8GatedResidualGemm:
    """Load FP8 weights once per tile, retaining 8-bit global storage.

    With 16-bit activations, the epilogue warps convert only the resident
    weight tile to the activation dtype before MMA. With FP8 activations,
    TMA feeds the native FP8 MMA directly. Neither path materializes a
    dequantized global weight. Three independent barriers protect each raw,
    converted, and MMA-consumed stage.
    """

    def __init__(
        self,
        *,
        role: str,
        hc_count: int,
        hidden_size: int,
        lowrank: int,
        projection_rows: int,
        token_tile: int,
        projection_tile: int,
        k_tile: int,
        stages: int,
        split_k: int,
        use_pdl: bool,
        weights_independent: bool,
        weight_scale_per_channel: bool,
        token_major_up: bool,
        weight_scale_vectorized: bool,
        up_hidden_tile: int,
        up_token_rows: int,
        separate_operand_ready: bool,
        epilogue_threads: int,
        shared_up_scales: bool,
        interleave_tma: bool,
    ):
        if role not in ("projection", "mix"):
            raise ValueError("FP8 HC GEMM role must be projection or mix")
        if (hc_count, hidden_size, lowrank) not in ((4, 2560, 320), (4, 8192, 1024)):
            raise ValueError("FP8 HC requires validated model dimensions")
        self.separate_operand_ready = separate_operand_ready
        self.role = role
        self.hc_count = hc_count
        self.hidden_size = hidden_size
        self.lowrank = lowrank
        self.wide = hc_count * hidden_size
        self.projection_rows = projection_rows
        self.n = token_tile
        self.m = projection_tile if role == "projection" else 128
        self.k = k_tile
        self.split_k = split_k
        self.inner = self.wide if role == "projection" else lowrank
        self.tiles = (self.inner + split_k * k_tile - 1) // (split_k * k_tile)
        self.stages = min(stages, self.tiles)
        self.pdl = use_pdl
        self.weights_independent = weights_independent
        self.weight_scale_per_channel = weight_scale_per_channel
        self.token_major_up = token_major_up
        self.weight_scale_vectorized = weight_scale_vectorized
        self.up_hidden_tile = up_hidden_tile
        self.up_token_rows = up_token_rows
        self.columns = max(32, token_tile)
        self.epilogue_threads = epilogue_threads
        self.shared_up_scales = shared_up_scales
        self.interleave_tma = interleave_tma
        self.epilogue_groups = self.epilogue_threads // 128
        self.producer_warp = self.epilogue_threads // 32
        self.mma_warp = self.producer_warp + 1

    @cute.experimental.jit
    def __call__(
        self,
        x,
        weight,
        original,
        weight_scale,
        activation_scale,
        partials,
        out,
        stream: CUstream,
    ):
        if cutlass.const_expr(x.element_type == cutlass.Float8E4M3FN):
            self._native_call(
                x,
                weight,
                original,
                weight_scale,
                activation_scale,
                partials,
                out,
                stream,
            )
        else:
            if cutlass.const_expr(self.role == "mix" and self.token_major_up):
                self.up_kernel(
                    x, weight, original, weight_scale, activation_scale, out
                ).launch(
                    grid=(
                        self.hidden_size // self.up_hidden_tile,
                        cute.ceil_div(x.shape[0], self.up_token_rows),
                        1,
                    ),
                    block=(self.epilogue_threads + 64, 1, 1),
                    stream=stream,
                    use_pdl=self.pdl,
                )
            else:
                self.kernel(
                    x, weight, original, weight_scale, activation_scale, partials, out
                ).launch(
                    grid=(
                        (
                            (self.projection_rows + self.m - 1) // self.m
                            if self.role == "projection"
                            else self.hidden_size // 32
                        ),
                        cute.ceil_div(x.shape[0], self.n),
                        self.split_k,
                    ),
                    block=(self.epilogue_threads + 64, 1, 1),
                    stream=stream,
                    use_pdl=self.pdl,
                )

    @cute.experimental.jit
    def _up_prefetch(
        self,
        original,
        weight_scale,
        activation_scale,
        residual,
        scales,
        row,
        hidden,
        rows,
        branch: cutlass.Constexpr,
        slot: cutlass.Constexpr,
        native_fp8: cutlass.Constexpr,
    ):
        count = cute.size(residual, mode=[0])
        if row < rows:
            source = cute.make_tensor(
                cute.make_ptr(
                    original.element_type,
                    (
                        original.iterator
                        + row * self.wide
                        + branch * self.hidden_size
                        + hidden
                    ).toint(),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                ),
                cute.make_layout(count),
            )
            cute.copy(
                cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(),
                    original.element_type,
                    num_bits_per_copy=128,
                ),
                source,
                residual[None, slot],
            )
            if cutlass.const_expr(
                self.weight_scale_per_channel and self.weight_scale_vectorized
            ):
                source_scale = cute.make_tensor(
                    cute.make_ptr(
                        cutlass.Float32,
                        (
                            weight_scale.iterator + branch * self.hidden_size + hidden
                        ).toint(),
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    ),
                    cute.make_layout(count),
                )
                cute.copy(
                    cute.make_copy_atom(
                        cute.nvgpu.CopyUniversalOp(),
                        cutlass.Float32,
                        num_bits_per_copy=128,
                    ),
                    source_scale,
                    scales[None, slot],
                )
            else:
                for i in cutlass.range_constexpr(count):
                    scales[i, slot] = weight_scale[
                        (
                            branch * self.hidden_size + hidden + i
                            if self.weight_scale_per_channel
                            else 0
                        )
                    ]
            if cutlass.const_expr(native_fp8):
                for i in cutlass.range_constexpr(count):
                    scales[i, slot] = scales[i, slot] * activation_scale[row]

    @cute.experimental.jit
    def _up_pair_prefetch(
        self,
        original,
        residual,
        coordinates,
        token_tile,
        hidden_tile,
        rows,
        branch: cutlass.Constexpr,
    ):
        count = cute.size(residual, mode=[0])
        for packet in cutlass.range_constexpr(count // 2):
            row, col = coordinates[packet * 2]
            input_row = token_tile * self.up_token_rows + row
            hidden = hidden_tile * self.up_hidden_tile + col
            if input_row < rows:
                source = cute.make_tensor(
                    cute.make_ptr(
                        original.element_type,
                        (
                            original.iterator
                            + input_row * self.wide
                            + branch * self.hidden_size
                            + hidden
                        ).toint(),
                        cute.AddressSpace.gmem,
                        assumed_align=4,
                    ),
                    cute.make_layout(2),
                )
                values = cute.make_rmem_tensor(2, original.element_type)
                cute.copy(
                    cute.make_copy_atom(
                        cute.nvgpu.CopyUniversalOp(),
                        original.element_type,
                        num_bits_per_copy=32,
                    ),
                    source,
                    values,
                )
                for j in cutlass.range_constexpr(2):
                    residual[packet * 2 + j, branch] = values[j]

    @cute.experimental.jit
    def _up_reader(self, base, layout, original, tid, branch: cutlass.Constexpr):
        tile = (self.up_token_rows, self.up_hidden_tile // self.epilogue_groups)
        group = tid // 128 + branch * self.epilogue_groups
        accumulator = cute.make_tensor(
            cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, base), layout
        )[((None, None), 0, 0)]
        accumulator = cute.local_tile(accumulator, tile, (0, group))
        coordinates = cute.local_tile(
            cute.make_identity_tensor(
                (self.up_token_rows, self.up_hidden_tile * self.hc_count)
            ),
            tile,
            (0, group),
        )
        copy = tcgen05.make_tmem_copy(
            sm100_utils.get_tmem_load_op(
                (self.up_token_rows, self.up_hidden_tile * self.hc_count, self.k),
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
            original.iterator, cute.make_layout(tile, stride=(1, self.up_token_rows))
        )
        values = cute_ext.allocate(
            cutlass.Float32,
            cute.AddressSpace.rmem,
            cute_ext.make_t2r_rmem_layout(
                copy, cute.flat_divide(dummy, tile), tid % 128
            ),
            alignment=32,
        )
        return accumulator, reader, values, reader.partition_D(coordinates)

    @cute.experimental.kernel
    def up_kernel(self, x, weight, original, weight_scale, activation_scale, out):
        tid = cute.arch.thread_idx()[0]
        warp = cute.arch.make_warp_uniform(tid // 32)
        hidden_tile, token_tile, _ = cute.arch.block_idx()
        dtype = x.element_type
        native_fp8 = cutlass.const_expr(dtype == cutlass.Float8E4M3FN)
        mma = sm100_utils.make_trivial_tiled_mma(
            dtype,
            dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            cutlass.Float32,
            tcgen05.CtaGroup.ONE,
            (self.up_token_rows, self.up_hidden_tile * self.hc_count),
        )
        tiler = (self.up_token_rows, self.up_hidden_tile * self.hc_count, self.k)
        activation_layout = sm100_utils.make_smem_layout_a(
            mma, tiler, dtype, self.stages
        )
        raw_layout = sm100_utils.make_smem_layout_b(
            mma, tiler, cutlass.Float8E4M3FN, self.stages
        )
        weight_layout = sm100_utils.make_smem_layout_b(mma, tiler, dtype, self.stages)
        sx = cute_ext.allocate(
            dtype, cute.AddressSpace.smem, activation_layout, alignment=1024
        )
        if cutlass.const_expr(native_fp8):
            raw = cute_ext.allocate(
                cutlass.Float8E4M3FN, cute.AddressSpace.smem, raw_layout, alignment=1024
            )
            sw = raw
        else:
            sw = cute_ext.allocate(
                dtype, cute.AddressSpace.smem, weight_layout, alignment=1024
            )
            raw_base = cute.get_nonswizzle_portion(raw_layout)
            weight_base = cute.get_nonswizzle_portion(weight_layout)
            # Raw and converted tiles share the larger decoded-weight slot.
            raw = cute.make_tensor(
                cute.make_ptr(
                    cutlass.Float8E4M3FN,
                    sw.iterator.toint(),
                    cute.AddressSpace.smem,
                    assumed_align=1024,
                    swizzle_=cute.get_swizzle_portion(raw_layout),
                ),
                cute.make_layout(
                    raw_base.shape,
                    stride=(
                        raw_base.stride[0],
                        raw_base.stride[1],
                        raw_base.stride[2],
                        weight_base.stride[3] * (dtype.width // 8),
                    ),
                ),
            )
        barriers = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(3 * self.stages + 2),
            alignment=8,
        ).iterator
        full = barriers
        converted = full + self.stages
        empty = converted + self.stages
        done, read_done = empty + self.stages, empty + self.stages + 1
        base = cute_ext.allocate(
            cutlass.Int32, cute.AddressSpace.smem, cute.make_layout(1), alignment=4
        ).iterator
        if warp == self.mma_warp:
            with cute.arch.elect_one():
                for stage in cutlass.range_constexpr(self.stages):
                    cute.arch.mbarrier_init(full + stage, 1)
                    cute.arch.mbarrier_init(
                        converted + stage, self.epilogue_threads // 32
                    )
                    cute.arch.mbarrier_init(empty + stage, 1)
                cute.arch.mbarrier_init(done, 1)
                cute.arch.mbarrier_init(read_done, self.epilogue_threads)
            cute.arch.alloc_tmem(
                self.up_hidden_tile * self.hc_count, base, is_two_cta=False
            )
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        activation = cute.make_tensor(
            x.iterator,
            cute.make_layout(
                (x.shape[0], (64, self.inner // 64), 1),
                stride=(self.inner, (1, 64), 0),
            ),
        )
        weights = cute.make_tensor(
            weight.iterator,
            cute.make_layout(
                (
                    (
                        self.up_hidden_tile,
                        self.hc_count,
                        self.hidden_size // self.up_hidden_tile,
                    ),
                    (64, self.inner // 64),
                    1,
                ),
                stride=(
                    (
                        self.inner,
                        self.hidden_size * self.inner,
                        self.up_hidden_tile * self.inner,
                    ),
                    (1, 64),
                    0,
                ),
            ),
        )
        gx = cute.local_tile(
            activation, (self.up_token_rows, self.k), (token_tile, None, 0)
        )
        gw = cute.local_tile(
            weights,
            (self.up_hidden_tile * self.hc_count, self.k),
            (hidden_tile, None, 0),
        )
        map_x = cute_ext.get_cta_v_map_ab(activation, tiler, mma, "A")
        map_w = cute_ext.get_cta_v_map_ab(weights, tiler, mma, "B")
        if warp == self.producer_warp:
            for stage in cutlass.range_constexpr(self.stages):
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        full + stage,
                        self.up_hidden_tile * self.hc_count * self.k
                        + self.up_token_rows * self.k * (dtype.width // 8),
                    )
                if cutlass.const_expr(self.weights_independent):
                    cute_ext.tma_load(
                        gw[None, None, stage],
                        raw[None, None, None, stage],
                        (full + stage).value,
                        cta_v_map=map_w,
                        update_expect_tx=False,
                    )
            if cutlass.const_expr(self.pdl):
                cute.arch.griddepcontrol_wait()
                cute.arch.griddepcontrol_launch_dependents()
            for stage in cutlass.range_constexpr(self.stages):
                if cutlass.const_expr(not self.weights_independent):
                    cute_ext.tma_load(
                        gw[None, None, stage],
                        raw[None, None, None, stage],
                        (full + stage).value,
                        cta_v_map=map_w,
                        update_expect_tx=False,
                    )
                cute_ext.tma_load(
                    gx[None, None, stage],
                    sx[None, None, None, stage],
                    (full + stage).value,
                    cta_v_map=map_x,
                    update_expect_tx=False,
                )
            for tile in cutlass.range(self.stages, self.tiles, unroll=1):
                stage = tile % self.stages
                cute.arch.mbarrier_wait(empty + stage, (tile // self.stages - 1) % 2)
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        full + stage,
                        self.up_hidden_tile * self.hc_count * self.k
                        + self.up_token_rows * self.k * (dtype.width // 8),
                    )
                cute_ext.tma_load(
                    gw[None, None, tile],
                    raw[None, None, None, stage],
                    (full + stage).value,
                    cta_v_map=map_w,
                    update_expect_tx=False,
                )
                cute_ext.tma_load(
                    gx[None, None, tile],
                    sx[None, None, None, stage],
                    (full + stage).value,
                    cta_v_map=map_x,
                    update_expect_tx=False,
                )
        elif warp == self.mma_warp:
            thread_mma = mma.get_slice(0)
            a_desc = thread_mma.make_fragment_A(sx)
            b_desc = thread_mma.make_fragment_B(sw)
            fragment = thread_mma.make_fragment_C(
                mma.partition_shape_C(
                    (self.up_token_rows, self.up_hidden_tile * self.hc_count)
                )
            )
            accumulator = cute.make_tensor(
                cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, base), fragment.layout
            )
            mma.set(tcgen05.Field.ACCUMULATE, False)
            for tile in cutlass.range(self.tiles, unroll=1):
                stage = tile % self.stages
                ready = full if cutlass.const_expr(native_fp8) else converted
                cute.arch.mbarrier_wait(ready + stage, tile // self.stages % 2)
                for block in cutlass.range_constexpr(cute.size(a_desc, mode=[2])):
                    cute.gemm(
                        mma,
                        accumulator,
                        a_desc[None, None, block, stage],
                        b_desc[None, None, block, stage],
                        accumulator,
                    )
                    mma.set(tcgen05.Field.ACCUMULATE, True)
                with cute.arch.elect_one():
                    tcgen05.commit(empty + stage, None, tcgen05.CtaGroup.ONE)
            with cute.arch.elect_one():
                tcgen05.commit(done, None, tcgen05.CtaGroup.ONE)
            cute.arch.mbarrier_wait(read_done, 0)
            cute.arch.relinquish_tmem_alloc_permit()
            cute.arch.dealloc_tmem(
                cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, base),
                self.up_hidden_tile * self.hc_count,
            )
        elif tid < self.epilogue_threads:
            if cutlass.const_expr(not native_fp8):
                for tile in cutlass.range(self.tiles, unroll=1):
                    stage = tile % self.stages
                    cute.arch.mbarrier_wait(full + stage, tile // self.stages % 2)
                    source, target = (
                        raw[None, None, None, stage],
                        sw[None, None, None, stage],
                    )
                    packets = (
                        self.up_hidden_tile
                        * self.hc_count
                        * self.k
                        // (self.epilogue_threads * 8)
                    )
                    raw_values = cute.make_rmem_tensor(
                        (8, packets), cutlass.Float8E4M3FN
                    )
                    # Every reader finishes before expanded stores can
                    # overwrite another thread's compressed source bytes.
                    for offset in cutlass.range_constexpr(packets):
                        linear = (tid + offset * self.epilogue_threads) * 8
                        index = (
                            linear // self.k
                            + (linear % self.k) * self.up_hidden_tile * self.hc_count
                        )
                        src = cute.make_tensor(
                            cute.make_ptr(
                                cutlass.Float8E4M3FN,
                                (source.iterator + source.layout(index)).toint(),
                                cute.AddressSpace.smem,
                                assumed_align=8,
                                swizzle_=cute.get_swizzle_portion(raw_layout),
                            ),
                            cute.make_layout(8),
                        )
                        cute.copy(
                            cute.make_copy_atom(
                                cute.nvgpu.CopyUniversalOp(),
                                cutlass.Float8E4M3FN,
                                num_bits_per_copy=64,
                            ),
                            src,
                            raw_values[None, offset],
                        )
                    _epilogue_barrier(self.epilogue_threads, cutlass.Int32(1))
                    for offset in cutlass.range_constexpr(packets):
                        linear = (tid + offset * self.epilogue_threads) * 8
                        index = (
                            linear // self.k
                            + (linear % self.k) * self.up_hidden_tile * self.hc_count
                        )
                        dst = cute.make_tensor(
                            cute.make_ptr(
                                dtype,
                                (target.iterator + target.layout(index)).toint(),
                                cute.AddressSpace.smem,
                                assumed_align=16,
                                swizzle_=cute.get_swizzle_portion(weight_layout),
                            ),
                            cute.make_layout(8),
                        )
                        converted_values = cute.make_rmem_tensor(8, dtype)
                        _convert_fp8_weight(raw_values[None, offset], converted_values)
                        cute.copy(
                            cute.make_copy_atom(
                                cute.nvgpu.CopyUniversalOp(),
                                dtype,
                                num_bits_per_copy=128,
                            ),
                            converted_values,
                            dst,
                        )
                    cute.arch.fence_view_async_shared()
                    cute.arch.sync_warp(mask=0xFFFFFFFF)
                    if tid % 32 == 0:
                        cute.arch.mbarrier_arrive(converted + stage)
            if cutlass.const_expr(self.pdl):
                cute.arch.griddepcontrol_wait()
            thread_mma = mma.get_slice(0)
            fragment = thread_mma.make_fragment_C(
                mma.partition_shape_C(
                    (self.up_token_rows, self.up_hidden_tile * self.hc_count)
                )
            )
            _, _, template, coordinates = self._up_reader(
                base, fragment.layout, original, tid, 0
            )
            count = cute.size(template)
            ring_slots = (
                self.hc_count
                if self.up_token_rows == 64
                else (2 if self.up_hidden_tile == 64 else self.hc_count)
            )
            residual = cute.make_rmem_tensor((count, ring_slots), original.element_type)
            scales = cute.make_rmem_tensor(
                (1, 1) if self.up_token_rows == 64 else (count, ring_slots),
                cutlass.Float32,
            )
            residual.fill(0.0)
            scales.fill(1.0)
            sums = cute.make_rmem_tensor(count, cutlass.Float32)
            sums.fill(0.0)
            first_row, first_col = coordinates[0]
            global_row = token_tile * self.up_token_rows + first_row
            hidden = hidden_tile * self.up_hidden_tile + first_col
            if cutlass.const_expr(self.up_token_rows == 64):
                assert not native_fp8
                scale_cache = cute_ext.allocate(
                    cutlass.Float32,
                    cute.AddressSpace.smem,
                    cute.make_layout(256),
                    alignment=32,
                )
                if tid < 256:
                    channel = (
                        (tid // 64) * self.hidden_size + hidden_tile * 64 + tid % 64
                    )
                    scale_cache[tid] = weight_scale[
                        channel if self.weight_scale_per_channel else 0
                    ]
                _epilogue_barrier(self.epilogue_threads, cutlass.Int32(3))
                for branch in cutlass.range_constexpr(4):
                    self._up_pair_prefetch(
                        original,
                        residual,
                        coordinates,
                        token_tile,
                        hidden_tile,
                        x.shape[0],
                        branch,
                    )
            else:
                for branch in cutlass.range_constexpr(ring_slots):
                    self._up_prefetch(
                        original,
                        weight_scale,
                        activation_scale,
                        residual,
                        scales,
                        global_row,
                        hidden,
                        x.shape[0],
                        branch,
                        branch,
                        native_fp8,
                    )
            cute.arch.mbarrier_wait(done, 0)
            if ((tid % 128) // 32) * (self.up_token_rows // 4) < x.shape[
                0
            ] - token_tile * self.up_token_rows:
                for branch in cutlass.range_constexpr(self.hc_count):
                    slot = branch % ring_slots
                    accumulator, reader, gates, _ = self._up_reader(
                        base, fragment.layout, original, tid, branch
                    )
                    cute_ext.partition_and_copy(reader, accumulator, gates)
                    cute.arch.fence_view_async_tmem_load()
                    for i in cutlass.range_constexpr(count):
                        factor = cutlass.Float32(1.0)
                        if cutlass.const_expr(self.up_token_rows == 64):
                            _, col = coordinates[i]
                            factor = scale_cache[branch * 64 + col]
                        else:
                            factor = scales[i, slot]
                        sums[i] = sums[i] + residual[i, slot].to(
                            cutlass.Float32
                        ) * _up_sigmoid_tanh(gates[i] * factor)
                    if cutlass.const_expr(branch + ring_slots < self.hc_count):
                        self._up_prefetch(
                            original,
                            weight_scale,
                            activation_scale,
                            residual,
                            scales,
                            global_row,
                            hidden,
                            x.shape[0],
                            branch + ring_slots,
                            slot,
                            native_fp8,
                        )
            tcgen05_fence("before_thread_sync")
            cute.arch.mbarrier_arrive(read_done)
            if cutlass.const_expr(self.up_token_rows == 64):
                for packet in cutlass.range_constexpr(count // 2):
                    row, col = coordinates[packet * 2]
                    output_row = token_tile * self.up_token_rows + row
                    if output_row < x.shape[0]:
                        packed = cute.make_rmem_tensor(2, out.element_type)
                        for j in cutlass.range_constexpr(2):
                            packed[j] = (sums[packet * 2 + j] / self.hc_count).to(
                                out.element_type
                            )
                        target = cute.make_tensor(
                            cute.make_ptr(
                                out.element_type,
                                (
                                    out.iterator
                                    + output_row * self.hidden_size
                                    + hidden_tile * self.up_hidden_tile
                                    + col
                                ).toint(),
                                cute.AddressSpace.gmem,
                                assumed_align=4,
                            ),
                            cute.make_layout(2),
                        )
                        cute.copy(
                            cute.make_copy_atom(
                                cute.nvgpu.CopyUniversalOp(),
                                out.element_type,
                                num_bits_per_copy=32,
                            ),
                            packed,
                            target,
                        )
            else:
                if global_row < x.shape[0]:
                    packed = cute.make_rmem_tensor(count, out.element_type)
                    for i in cutlass.range_constexpr(count):
                        packed[i] = (sums[i] / self.hc_count).to(out.element_type)
                    target = cute.make_tensor(
                        cute.make_ptr(
                            out.element_type,
                            (
                                out.iterator + global_row * self.hidden_size + hidden
                            ).toint(),
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        ),
                        cute.make_layout(count),
                    )
                    cute.copy(
                        cute.make_copy_atom(
                            cute.nvgpu.CopyUniversalOp(),
                            out.element_type,
                            num_bits_per_copy=128,
                        ),
                        packed,
                        target,
                    )

    @cute.experimental.kernel
    def kernel(
        self, x, weight, original, weight_scale, activation_scale, partials, out
    ):
        tid = cute.arch.thread_idx()[0]
        warp = cute.arch.make_warp_uniform(tid // 32)
        output_tile, token_tile, split = cute.arch.block_idx()
        dtype = x.element_type
        native_fp8 = cutlass.const_expr(dtype == cutlass.Float8E4M3FN)
        mma = sm100_utils.make_trivial_tiled_mma(
            dtype,
            dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            cutlass.Float32,
            tcgen05.CtaGroup.ONE,
            (self.m, self.n),
        )
        tiler = (self.m, self.n, self.k)
        weight_layout = sm100_utils.make_smem_layout_a(mma, tiler, dtype, self.stages)
        raw_layout = sm100_utils.make_smem_layout_a(
            mma, tiler, cutlass.Float8E4M3FN, self.stages
        )
        x_layout = sm100_utils.make_smem_layout_b(mma, tiler, dtype, self.stages)
        if cutlass.const_expr(native_fp8):
            raw = cute_ext.allocate(
                cutlass.Float8E4M3FN, cute.AddressSpace.smem, raw_layout, alignment=1024
            )
            weights = raw
        else:
            weights = cute_ext.allocate(
                dtype, cute.AddressSpace.smem, weight_layout, alignment=1024
            )
            raw_base = cute.get_nonswizzle_portion(raw_layout)
            weight_base = cute.get_nonswizzle_portion(weight_layout)
            # Raw and converted tiles share the larger decoded-weight slot.
            raw = cute.make_tensor(
                cute.make_ptr(
                    cutlass.Float8E4M3FN,
                    weights.iterator.toint(),
                    cute.AddressSpace.smem,
                    assumed_align=1024,
                    swizzle_=cute.get_swizzle_portion(raw_layout),
                ),
                cute.make_layout(
                    raw_base.shape,
                    stride=(
                        raw_base.stride[0],
                        raw_base.stride[1],
                        raw_base.stride[2],
                        weight_base.stride[3] * (dtype.width // 8),
                    ),
                ),
            )
        sx = cute_ext.allocate(dtype, cute.AddressSpace.smem, x_layout, alignment=1024)
        barriers = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(3 * self.stages + 2),
            alignment=8,
        ).iterator
        full, converted, empty = (
            barriers,
            barriers + self.stages,
            barriers + 2 * self.stages,
        )
        done, read_done = barriers + 3 * self.stages, barriers + 3 * self.stages + 1
        input_ready = full
        if cutlass.const_expr(self.separate_operand_ready):
            input_ready = cute_ext.allocate(
                cutlass.Int64,
                cute.AddressSpace.smem,
                cute.make_layout(self.stages),
                alignment=8,
            ).iterator

        tmem_base = cute_ext.allocate(
            cutlass.Int32, cute.AddressSpace.smem, cute.make_layout(1), alignment=4
        ).iterator
        # TMEM readers and the branch reduction both assign adjacent M
        # coordinates to adjacent lanes.
        gate_layout = cute.make_layout((self.m, self.n), stride=(1, self.m))
        gate = cute.make_tensor(
            cute.recast_ptr(raw.iterator, dtype=cutlass.Float32), gate_layout
        )
        if cutlass.const_expr(
            self.role == "mix"
            and self.n * 4 > self.k * self.stages * (dtype.width // 8)
        ):
            gate = cute_ext.allocate(
                cutlass.Float32, cute.AddressSpace.smem, gate_layout, alignment=128
            )
        if warp == self.mma_warp:
            with cute.arch.elect_one():
                for stage in cutlass.range_constexpr(self.stages):
                    cute.arch.mbarrier_init(full + stage, 1)
                    cute.arch.mbarrier_init(
                        converted + stage,
                        (
                            1
                            if self.lowrank == 320 and self.n == 16
                            else self.epilogue_threads // 32
                        ),
                    )
                    cute.arch.mbarrier_init(empty + stage, 1)
                    if cutlass.const_expr(self.separate_operand_ready):
                        cute.arch.mbarrier_init(input_ready + stage, 1)

                cute.arch.mbarrier_init(done, 1)
                cute.arch.mbarrier_init(read_done, self.epilogue_threads)
            cute.arch.alloc_tmem(self.columns, tmem_base, is_two_cta=False)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

        x_view = cute.make_tensor(
            x.iterator,
            cute.make_layout(
                (x.shape[0], (64, self.inner // 64), 1),
                stride=(self.inner, (1, 64), 0),
            ),
        )
        if cutlass.const_expr(self.role == "projection"):
            w_view = cute.make_tensor(
                weight.iterator,
                cute.make_layout(
                    (self.projection_rows, (64, self.inner // 64), 1),
                    stride=(self.inner, (1, 64), 0),
                ),
            )
        else:
            w_view = cute.make_tensor(
                weight.iterator,
                cute.make_layout(
                    (
                        (32, self.hc_count, self.hidden_size // 32),
                        (64, self.inner // 64),
                        1,
                    ),
                    stride=(
                        (self.inner, self.hidden_size * self.inner, 32 * self.inner),
                        (1, 64),
                        0,
                    ),
                ),
            )
        gw = cute.local_tile(w_view, (self.m, self.k), (output_tile, None, 0))
        gx = cute.local_tile(x_view, (self.n, self.k), (token_tile, None, 0))
        map_w = cute_ext.get_cta_v_map_ab(w_view, tiler, mma, "A")
        map_x = cute_ext.get_cta_v_map_ab(x_view, tiler, mma, "B")

        if warp == self.producer_warp:
            # Independent weights can start before the activation producer.
            for stage in cutlass.range_constexpr(self.stages):
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        full + stage,
                        self.m * self.k
                        + (
                            0
                            if self.separate_operand_ready
                            else self.n * self.k * (dtype.width // 8)
                        ),
                    )
                if cutlass.const_expr(self.weights_independent):
                    cute_ext.tma_load(
                        gw[None, None, split * self.tiles + stage],
                        raw[None, None, None, stage],
                        (full + stage).value,
                        cta_v_map=map_w,
                        update_expect_tx=False,
                    )
            if cutlass.const_expr(self.pdl):
                cute.arch.griddepcontrol_wait()
                cute.arch.griddepcontrol_launch_dependents()
            for stage in cutlass.range_constexpr(self.stages):
                if cutlass.const_expr(not self.weights_independent):
                    cute_ext.tma_load(
                        gw[None, None, split * self.tiles + stage],
                        raw[None, None, None, stage],
                        (full + stage).value,
                        cta_v_map=map_w,
                        update_expect_tx=False,
                    )
                if cutlass.const_expr(self.separate_operand_ready):
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            input_ready + stage, self.n * self.k * (dtype.width // 8)
                        )
                cute_ext.tma_load(
                    gx[None, None, split * self.tiles + stage],
                    sx[None, None, None, stage],
                    (input_ready + stage).value,
                    cta_v_map=map_x,
                    update_expect_tx=False,
                )
            for tile in cutlass.range(self.stages, self.tiles, unroll=1):
                stage = tile % self.stages
                cute.arch.mbarrier_wait(empty + stage, (tile // self.stages - 1) % 2)
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        full + stage,
                        self.m * self.k
                        + (
                            0
                            if self.separate_operand_ready
                            else self.n * self.k * (dtype.width // 8)
                        ),
                    )
                cute_ext.tma_load(
                    gw[None, None, split * self.tiles + tile],
                    raw[None, None, None, stage],
                    (full + stage).value,
                    cta_v_map=map_w,
                    update_expect_tx=False,
                )
                if cutlass.const_expr(self.separate_operand_ready):
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            input_ready + stage, self.n * self.k * (dtype.width // 8)
                        )
                cute_ext.tma_load(
                    gx[None, None, split * self.tiles + tile],
                    sx[None, None, None, stage],
                    (input_ready + stage).value,
                    cta_v_map=map_x,
                    update_expect_tx=False,
                )
        elif warp == self.mma_warp:
            thread_mma = mma.get_slice(0)
            a_desc = thread_mma.make_fragment_A(weights)
            b_desc = thread_mma.make_fragment_B(sx)
            acc_shape = mma.partition_shape_C((self.m, self.n))
            acc_fragment = thread_mma.make_fragment_C(acc_shape)
            acc = cute.make_tensor(
                cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, tmem_base),
                acc_fragment.layout,
            )
            mma.set(tcgen05.Field.ACCUMULATE, False)
            for tile in cutlass.range(self.tiles, unroll=1):
                stage = tile % self.stages
                ready = full if cutlass.const_expr(native_fp8) else converted
                cute.arch.mbarrier_wait(ready + stage, tile // self.stages % 2)
                if cutlass.const_expr(self.separate_operand_ready):
                    cute.arch.mbarrier_wait(
                        input_ready + stage, tile // self.stages % 2
                    )

                for block in cutlass.range_constexpr(cute.size(a_desc, mode=[2])):
                    cute.gemm(
                        mma,
                        acc,
                        a_desc[None, None, block, stage],
                        b_desc[None, None, block, stage],
                        acc,
                    )
                    mma.set(tcgen05.Field.ACCUMULATE, True)
                with cute.arch.elect_one():
                    tcgen05.commit(empty + stage, None, tcgen05.CtaGroup.ONE)
            with cute.arch.elect_one():
                tcgen05.commit(done, None, tcgen05.CtaGroup.ONE)
            cute.arch.mbarrier_wait(read_done, 0)
            cute.arch.relinquish_tmem_alloc_permit()
            cute.arch.dealloc_tmem(
                cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, tmem_base),
                self.columns,
            )
        elif tid < self.epilogue_threads:
            if cutlass.const_expr(not native_fp8):
                for tile in cutlass.range(self.tiles, unroll=1):
                    stage = tile % self.stages
                    cute.arch.mbarrier_wait(full + stage, tile // self.stages % 2)
                    source = raw[None, None, None, stage]
                    target = weights[None, None, None, stage]
                    packets = self.m * self.k // (self.epilogue_threads * 8)
                    raw_values = cute.make_rmem_tensor(
                        (8, packets), cutlass.Float8E4M3FN
                    )
                    # Every reader finishes before expanded stores can
                    # overwrite another thread's compressed source bytes.
                    for offset in cutlass.range_constexpr(packets):
                        linear = (tid + offset * self.epilogue_threads) * 8
                        index = linear // self.k + (linear % self.k) * self.m
                        src = cute.make_tensor(
                            cute.make_ptr(
                                cutlass.Float8E4M3FN,
                                (source.iterator + source.layout(index)).toint(),
                                cute.AddressSpace.smem,
                                assumed_align=8,
                                swizzle_=cute.get_swizzle_portion(raw_layout),
                            ),
                            cute.make_layout(8),
                        )
                        cute.copy(
                            cute.make_copy_atom(
                                cute.nvgpu.CopyUniversalOp(),
                                cutlass.Float8E4M3FN,
                                num_bits_per_copy=64,
                            ),
                            src,
                            raw_values[None, offset],
                        )
                    _epilogue_barrier(self.epilogue_threads, cutlass.Int32(1))
                    for offset in cutlass.range_constexpr(packets):
                        linear = (tid + offset * self.epilogue_threads) * 8
                        index = linear // self.k + (linear % self.k) * self.m
                        dst = cute.make_tensor(
                            cute.make_ptr(
                                dtype,
                                (target.iterator + target.layout(index)).toint(),
                                cute.AddressSpace.smem,
                                assumed_align=16,
                                swizzle_=cute.get_swizzle_portion(weight_layout),
                            ),
                            cute.make_layout(8),
                        )
                        converted_values = cute.make_rmem_tensor(8, dtype)
                        _convert_fp8_weight(raw_values[None, offset], converted_values)
                        cute.copy(
                            cute.make_copy_atom(
                                cute.nvgpu.CopyUniversalOp(),
                                dtype,
                                num_bits_per_copy=128,
                            ),
                            converted_values,
                            dst,
                        )
                    cute.arch.fence_view_async_shared()
                    if cutlass.const_expr(self.lowrank == 320 and self.n == 16):
                        _epilogue_barrier(self.epilogue_threads, cutlass.Int32(1))
                        if tid == 0:
                            cute.arch.mbarrier_arrive(converted + stage)
                    else:
                        cute.arch.sync_warp(mask=0xFFFFFFFF)
                        if tid % 32 == 0:
                            cute.arch.mbarrier_arrive(converted + stage)
            cute.arch.mbarrier_wait(done, 0)
            thread_mma = mma.get_slice(0)
            acc_shape = mma.partition_shape_C((self.m, self.n))
            fragment = thread_mma.make_fragment_C(acc_shape)
            acc = cute.make_tensor(
                cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, tmem_base),
                fragment.layout,
            )[((None, None), 0, 0)]
            reader_n = self.n // self.epilogue_groups
            tile_shape = (self.m, reader_n)
            group = tid // 128
            local_tid = tid % 128
            acc = cute.local_tile(acc, tile_shape, (0, group))
            coordinates = cute.local_tile(
                cute.make_identity_tensor((self.m, self.n)),
                tile_shape,
                (0, group),
            )
            copy = tcgen05.make_tmem_copy(
                sm100_utils.get_tmem_load_op(
                    (self.m, reader_n, self.k),
                    cute_utils.LayoutEnum.COL_MAJOR,
                    cutlass.Float32,
                    cutlass.Float32,
                    tile_shape,
                    False,
                ),
                acc,
            )
            reader = copy.get_slice(local_tid)
            dummy = cute.make_tensor(
                out.iterator, cute.make_layout(tile_shape, stride=(1, self.m))
            )
            layout = cute_ext.make_t2r_rmem_layout(
                copy, cute.flat_divide(dummy, tile_shape), local_tid
            )
            values = cute_ext.allocate(
                cutlass.Float32, cute.AddressSpace.rmem, layout, alignment=32
            )
            coords = reader.partition_D(coordinates)
            cute_ext.partition_and_copy(reader, acc, values)
            cute.arch.fence_view_async_tmem_load()
            if cutlass.const_expr(self.role == "mix"):
                residual = cute.make_rmem_tensor_like(values, original.element_type)
                scales = cute.make_rmem_tensor_like(values)
                residual.fill(0.0)
                scales.fill(1.0)
                # Issue independent input loads before the sigmoid arithmetic.
                # Interleaving a load and store for every accumulator element
                # otherwise serializes global-memory latency.
                for index in cutlass.range_constexpr(cute.size(values)):
                    col, row = coords[index]
                    global_row = token_tile * self.n + row
                    global_col = (
                        (col // 32) * self.hidden_size + output_tile * 32 + col % 32
                    )
                    if global_row < x.shape[0]:
                        residual[index] = original[global_row, global_col]
                        scale = weight_scale[
                            global_col if self.weight_scale_per_channel else 0
                        ]
                        if cutlass.const_expr(native_fp8):
                            scale = scale * activation_scale[global_row]
                        scales[index] = scale
                for i in cutlass.range_constexpr(cute.size(values)):
                    values[i] = residual[i].to(cutlass.Float32) * _up_sigmoid_tanh(
                        values[i] * scales[i]
                    )
            for index in cutlass.range_constexpr(cute.size(values)):
                col, row = coords[index]
                global_row = token_tile * self.n + row
                if cutlass.const_expr(self.role == "projection"):
                    global_col = output_tile * self.m + col
                    if global_row < x.shape[0] and global_col < self.projection_rows:
                        partials[split, global_row, global_col] = values[index]
                else:
                    if global_row < x.shape[0]:
                        gate[col, row] = values[index]
            tcgen05_fence("before_thread_sync")
            cute.arch.mbarrier_arrive(read_done)
            if cutlass.const_expr(self.role == "mix"):
                _epilogue_barrier(self.epilogue_threads, cutlass.Int32(2))
                for offset in cutlass.range_constexpr(
                    self.n * 32 // self.epilogue_threads
                ):
                    index = tid + offset * self.epilogue_threads
                    row, col = index // 32, index % 32
                    if token_tile * self.n + row < x.shape[0]:
                        value = cutlass.Float32(0.0)
                        for branch in cutlass.range_constexpr(self.hc_count):
                            value = value + gate[branch * 32 + col, row]
                        out[token_tile * self.n + row, output_tile * 32 + col] = (
                            value / self.hc_count
                        ).to(original.element_type)

    # Native and expanded-weight operands share planning and the public entry.
    # Their kernel bodies retain the layouts and warp roles measured per dtype.
    @cute.experimental.jit
    def _native_call(
        self,
        x,
        weight,
        original,
        weight_scale,
        activation_scale,
        partials,
        out,
        stream: CUstream,
    ):
        if cutlass.const_expr(self.role == "mix" and self.token_major_up):
            self._native_up_kernel(
                x, weight, original, weight_scale, activation_scale, out
            ).launch(
                grid=(
                    self.hidden_size // self.up_hidden_tile,
                    cute.ceil_div(x.shape[0], self.up_token_rows),
                    1,
                ),
                block=(self.epilogue_threads + 64, 1, 1),
                stream=stream,
                use_pdl=self.pdl,
            )
        else:
            self._native_kernel(
                x, weight, original, weight_scale, activation_scale, partials, out
            ).launch(
                grid=(
                    (
                        (self.projection_rows + self.m - 1) // self.m
                        if self.role == "projection"
                        else self.hidden_size // 32
                    ),
                    cute.ceil_div(x.shape[0], self.n),
                    self.split_k,
                ),
                block=(self.epilogue_threads + 64, 1, 1),
                stream=stream,
                use_pdl=self.pdl,
            )

    @cute.experimental.jit
    def _native_up_prefetch(
        self,
        original,
        weight_scale,
        activation_scale,
        residual,
        scales,
        row,
        hidden,
        rows,
        branch: cutlass.Constexpr,
        slot: cutlass.Constexpr,
        native_fp8: cutlass.Constexpr,
    ):
        count = cute.size(residual, mode=[0])
        if row < rows:
            source = cute.make_tensor(
                cute.make_ptr(
                    original.element_type,
                    (
                        original.iterator
                        + row * self.wide
                        + branch * self.hidden_size
                        + hidden
                    ).toint(),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                ),
                cute.make_layout(count),
            )
            cute.copy(
                cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(),
                    original.element_type,
                    num_bits_per_copy=128,
                ),
                source,
                residual[None, slot],
            )
            if cutlass.const_expr(not self.shared_up_scales):
                if cutlass.const_expr(
                    self.weight_scale_per_channel and self.weight_scale_vectorized
                ):
                    source_scale = cute.make_tensor(
                        cute.make_ptr(
                            cutlass.Float32,
                            (
                                weight_scale.iterator
                                + branch * self.hidden_size
                                + hidden
                            ).toint(),
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        ),
                        cute.make_layout(count),
                    )
                    cute.copy(
                        cute.make_copy_atom(
                            cute.nvgpu.CopyUniversalOp(),
                            cutlass.Float32,
                            num_bits_per_copy=128,
                        ),
                        source_scale,
                        scales[None, slot],
                    )
                else:
                    for i in cutlass.range_constexpr(count):
                        scales[i, slot] = weight_scale[
                            (
                                branch * self.hidden_size + hidden + i
                                if self.weight_scale_per_channel
                                else 0
                            )
                        ]
                if cutlass.const_expr(native_fp8):
                    for i in cutlass.range_constexpr(count):
                        scales[i, slot] = scales[i, slot] * activation_scale[row]

    @cute.experimental.jit
    def _native_up_fragment_prefetch(
        self,
        original,
        weight_scale,
        activation_scale,
        residual,
        scales,
        coordinates,
        token_tile,
        hidden_tile,
        rows,
        branch: cutlass.Constexpr,
    ):
        count = cute.size(residual, mode=[0])
        for packet in cutlass.range_constexpr(count // 2):
            row, col = coordinates[packet * 2]
            input_row = token_tile * self.up_token_rows + row
            hidden = hidden_tile * 32 + col
            if input_row < rows:
                source = cute.make_tensor(
                    cute.make_ptr(
                        original.element_type,
                        (
                            original.iterator
                            + input_row * self.wide
                            + branch * self.hidden_size
                            + hidden
                        ).toint(),
                        cute.AddressSpace.gmem,
                        assumed_align=4,
                    ),
                    cute.make_layout(2),
                )
                values = cute.make_rmem_tensor(2, original.element_type)
                cute.copy(
                    cute.make_copy_atom(
                        cute.nvgpu.CopyUniversalOp(),
                        original.element_type,
                        num_bits_per_copy=32,
                    ),
                    source,
                    values,
                )
                for j in cutlass.range_constexpr(2):
                    residual[packet * 2 + j, branch] = values[j]
                    scales[packet * 2 + j, branch] = (
                        weight_scale[
                            (
                                branch * self.hidden_size + hidden + j
                                if self.weight_scale_per_channel
                                else 0
                            )
                        ]
                        * activation_scale[input_row]
                    )

    @cute.experimental.jit
    def _native_up_reader(self, base, layout, original, tid, branch: cutlass.Constexpr):
        tile = (self.up_token_rows, self.up_hidden_tile // self.epilogue_groups)
        group = tid // 128 + branch * self.epilogue_groups
        accumulator = cute.make_tensor(
            cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, base), layout
        )[((None, None), 0, 0)]
        accumulator = cute.local_tile(accumulator, tile, (0, group))
        coordinates = cute.local_tile(
            cute.make_identity_tensor(
                (self.up_token_rows, self.up_hidden_tile * self.hc_count)
            ),
            tile,
            (0, group),
        )
        copy = tcgen05.make_tmem_copy(
            sm100_utils.get_tmem_load_op(
                (self.up_token_rows, self.up_hidden_tile * self.hc_count, self.k),
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
            original.iterator, cute.make_layout(tile, stride=(1, self.up_token_rows))
        )
        values = cute_ext.allocate(
            cutlass.Float32,
            cute.AddressSpace.rmem,
            cute_ext.make_t2r_rmem_layout(
                copy, cute.flat_divide(dummy, tile), tid % 128
            ),
            alignment=32,
        )
        return accumulator, reader, values, reader.partition_D(coordinates)

    @cute.experimental.kernel
    def _native_kernel(
        self, x, weight, original, weight_scale, activation_scale, partials, out
    ):
        tid = cute.arch.thread_idx()[0]
        warp = cute.arch.make_warp_uniform(tid // 32)
        output_tile, token_tile, split = cute.arch.block_idx()
        dtype = x.element_type
        native_fp8 = cutlass.const_expr(dtype == cutlass.Float8E4M3FN)
        mma = sm100_utils.make_trivial_tiled_mma(
            dtype,
            dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            cutlass.Float32,
            tcgen05.CtaGroup.ONE,
            (self.m, self.n),
        )
        tiler = (self.m, self.n, self.k)
        weight_layout = sm100_utils.make_smem_layout_a(mma, tiler, dtype, self.stages)
        raw_layout = sm100_utils.make_smem_layout_a(
            mma, tiler, cutlass.Float8E4M3FN, self.stages
        )
        x_layout = sm100_utils.make_smem_layout_b(mma, tiler, dtype, self.stages)
        raw = cute_ext.allocate(
            cutlass.Float8E4M3FN, cute.AddressSpace.smem, raw_layout, alignment=1024
        )
        weights = raw
        if cutlass.const_expr(not native_fp8):
            weights = cute_ext.allocate(
                dtype, cute.AddressSpace.smem, weight_layout, alignment=1024
            )
        sx = cute_ext.allocate(dtype, cute.AddressSpace.smem, x_layout, alignment=1024)
        barriers = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(3 * self.stages + 2),
            alignment=8,
        ).iterator
        full, converted, empty = (
            barriers,
            barriers + self.stages,
            barriers + 2 * self.stages,
        )
        done, read_done = barriers + 3 * self.stages, barriers + 3 * self.stages + 1
        tmem_base = cute_ext.allocate(
            cutlass.Int32, cute.AddressSpace.smem, cute.make_layout(1), alignment=4
        ).iterator
        # TMEM readers and the branch reduction both assign adjacent M
        # coordinates to adjacent lanes.
        gate_layout = cute.make_layout((self.m, self.n), stride=(1, self.m))
        gate = cute.make_tensor(
            cute.recast_ptr(raw.iterator, dtype=cutlass.Float32), gate_layout
        )
        if cutlass.const_expr(self.role == "mix" and self.n * 4 > self.k * self.stages):
            gate = cute_ext.allocate(
                cutlass.Float32, cute.AddressSpace.smem, gate_layout, alignment=128
            )
        if warp == self.mma_warp:
            with cute.arch.elect_one():
                for stage in cutlass.range_constexpr(self.stages):
                    cute.arch.mbarrier_init(full + stage, 1)
                    cute.arch.mbarrier_init(converted + stage, 1)
                    cute.arch.mbarrier_init(empty + stage, 1)
                cute.arch.mbarrier_init(done, 1)
                cute.arch.mbarrier_init(read_done, self.epilogue_threads)
            cute.arch.alloc_tmem(self.columns, tmem_base, is_two_cta=False)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

        x_view = cute.make_tensor(
            x.iterator,
            cute.make_layout(
                (x.shape[0], (64, self.inner // 64), 1),
                stride=(self.inner, (1, 64), 0),
            ),
        )
        if cutlass.const_expr(self.role == "projection"):
            w_view = cute.make_tensor(
                weight.iterator,
                cute.make_layout(
                    (self.projection_rows, (64, self.inner // 64), 1),
                    stride=(self.inner, (1, 64), 0),
                ),
            )
        else:
            w_view = cute.make_tensor(
                weight.iterator,
                cute.make_layout(
                    (
                        (32, self.hc_count, self.hidden_size // 32),
                        (64, self.inner // 64),
                        1,
                    ),
                    stride=(
                        (self.inner, self.hidden_size * self.inner, 32 * self.inner),
                        (1, 64),
                        0,
                    ),
                ),
            )
        gw = cute.local_tile(w_view, (self.m, self.k), (output_tile, None, 0))
        gx = cute.local_tile(x_view, (self.n, self.k), (token_tile, None, 0))
        map_w = cute_ext.get_cta_v_map_ab(w_view, tiler, mma, "A")
        map_x = cute_ext.get_cta_v_map_ab(x_view, tiler, mma, "B")

        if warp == self.producer_warp:
            if cutlass.const_expr(self.interleave_tma and not self.pdl):
                # Feed each first-stage MMA without queuing later weight tiles first.
                for stage in cutlass.range_constexpr(self.stages):
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            full + stage,
                            self.m * self.k + self.n * self.k * (dtype.width // 8),
                        )
                    cute_ext.tma_load(
                        gw[None, None, split * self.tiles + stage],
                        raw[None, None, None, stage],
                        (full + stage).value,
                        cta_v_map=map_w,
                        update_expect_tx=False,
                    )
                    cute_ext.tma_load(
                        gx[None, None, split * self.tiles + stage],
                        sx[None, None, None, stage],
                        (full + stage).value,
                        cta_v_map=map_x,
                        update_expect_tx=False,
                    )
            else:
                # Independent weights can start before the activation producer.
                for stage in cutlass.range_constexpr(self.stages):
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            full + stage,
                            self.m * self.k + self.n * self.k * (dtype.width // 8),
                        )
                    if cutlass.const_expr(self.weights_independent):
                        cute_ext.tma_load(
                            gw[None, None, split * self.tiles + stage],
                            raw[None, None, None, stage],
                            (full + stage).value,
                            cta_v_map=map_w,
                            update_expect_tx=False,
                        )
                if cutlass.const_expr(self.pdl):
                    cute.arch.griddepcontrol_wait()
                    cute.arch.griddepcontrol_launch_dependents()
                for stage in cutlass.range_constexpr(self.stages):
                    if cutlass.const_expr(not self.weights_independent):
                        cute_ext.tma_load(
                            gw[None, None, split * self.tiles + stage],
                            raw[None, None, None, stage],
                            (full + stage).value,
                            cta_v_map=map_w,
                            update_expect_tx=False,
                        )
                    cute_ext.tma_load(
                        gx[None, None, split * self.tiles + stage],
                        sx[None, None, None, stage],
                        (full + stage).value,
                        cta_v_map=map_x,
                        update_expect_tx=False,
                    )
            for tile in cutlass.range(self.stages, self.tiles, unroll=1):
                stage = tile % self.stages
                cute.arch.mbarrier_wait(empty + stage, (tile // self.stages - 1) % 2)
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        full + stage,
                        self.m * self.k + self.n * self.k * (dtype.width // 8),
                    )
                cute_ext.tma_load(
                    gw[None, None, split * self.tiles + tile],
                    raw[None, None, None, stage],
                    (full + stage).value,
                    cta_v_map=map_w,
                    update_expect_tx=False,
                )
                cute_ext.tma_load(
                    gx[None, None, split * self.tiles + tile],
                    sx[None, None, None, stage],
                    (full + stage).value,
                    cta_v_map=map_x,
                    update_expect_tx=False,
                )
        elif warp == self.mma_warp:
            thread_mma = mma.get_slice(0)
            a_desc = thread_mma.make_fragment_A(weights)
            b_desc = thread_mma.make_fragment_B(sx)
            acc_shape = mma.partition_shape_C((self.m, self.n))
            acc_fragment = thread_mma.make_fragment_C(acc_shape)
            acc = cute.make_tensor(
                cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, tmem_base),
                acc_fragment.layout,
            )
            mma.set(tcgen05.Field.ACCUMULATE, False)
            for tile in cutlass.range(self.tiles, unroll=1):
                stage = tile % self.stages
                ready = full if cutlass.const_expr(native_fp8) else converted
                cute.arch.mbarrier_wait(ready + stage, tile // self.stages % 2)
                for block in cutlass.range_constexpr(cute.size(a_desc, mode=[2])):
                    cute.gemm(
                        mma,
                        acc,
                        a_desc[None, None, block, stage],
                        b_desc[None, None, block, stage],
                        acc,
                    )
                    mma.set(tcgen05.Field.ACCUMULATE, True)
                with cute.arch.elect_one():
                    tcgen05.commit(empty + stage, None, tcgen05.CtaGroup.ONE)
            with cute.arch.elect_one():
                tcgen05.commit(done, None, tcgen05.CtaGroup.ONE)
            cute.arch.mbarrier_wait(read_done, 0)
            cute.arch.relinquish_tmem_alloc_permit()
            cute.arch.dealloc_tmem(
                cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, tmem_base),
                self.columns,
            )
        elif tid < self.epilogue_threads:
            if cutlass.const_expr(not native_fp8):
                for tile in cutlass.range(self.tiles, unroll=1):
                    stage = tile % self.stages
                    cute.arch.mbarrier_wait(full + stage, tile // self.stages % 2)
                    source = raw[None, None, None, stage]
                    target = weights[None, None, None, stage]
                    # Each thread converts eight K-adjacent values together,
                    # retaining vector shared loads/stores across the swizzle.
                    for offset in cutlass.range(
                        self.m * self.k // (self.epilogue_threads * 8), unroll=4
                    ):
                        linear = (tid + offset * self.epilogue_threads) * 8
                        index = linear // self.k + (linear % self.k) * self.m
                        source_vector = cute.make_tensor(
                            cute.make_ptr(
                                cutlass.Float8E4M3FN,
                                (source.iterator + source.layout(index)).toint(),
                                cute.AddressSpace.smem,
                                assumed_align=8,
                                swizzle_=cute.get_swizzle_portion(raw_layout),
                            ),
                            cute.make_layout(8),
                        )
                        target_vector = cute.make_tensor(
                            cute.make_ptr(
                                dtype,
                                (target.iterator + target.layout(index)).toint(),
                                cute.AddressSpace.smem,
                                assumed_align=16,
                                swizzle_=cute.get_swizzle_portion(weight_layout),
                            ),
                            cute.make_layout(8),
                        )
                        raw_values = cute.make_rmem_tensor(8, cutlass.Float8E4M3FN)
                        converted_values = cute.make_rmem_tensor(8, dtype)
                        cute.copy(
                            cute.make_copy_atom(
                                cute.nvgpu.CopyUniversalOp(),
                                cutlass.Float8E4M3FN,
                                num_bits_per_copy=64,
                            ),
                            source_vector,
                            raw_values,
                        )
                        converted_values.store(raw_values.load().to(dtype))
                        cute.copy(
                            cute.make_copy_atom(
                                cute.nvgpu.CopyUniversalOp(),
                                dtype,
                                num_bits_per_copy=128,
                            ),
                            converted_values,
                            target_vector,
                        )
                    cute.arch.fence_view_async_shared()
                    _epilogue_barrier(self.epilogue_threads, cutlass.Int32(1))
                    if tid == 0:
                        cute.arch.mbarrier_arrive(converted + stage)
            cute.arch.mbarrier_wait(done, 0)
            thread_mma = mma.get_slice(0)
            acc_shape = mma.partition_shape_C((self.m, self.n))
            fragment = thread_mma.make_fragment_C(acc_shape)
            acc = cute.make_tensor(
                cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, tmem_base),
                fragment.layout,
            )[((None, None), 0, 0)]
            reader_n = self.n // self.epilogue_groups
            tile_shape = (self.m, reader_n)
            group = tid // 128
            local_tid = tid % 128
            acc = cute.local_tile(acc, tile_shape, (0, group))
            coordinates = cute.local_tile(
                cute.make_identity_tensor((self.m, self.n)),
                tile_shape,
                (0, group),
            )
            copy = tcgen05.make_tmem_copy(
                sm100_utils.get_tmem_load_op(
                    (self.m, reader_n, self.k),
                    cute_utils.LayoutEnum.COL_MAJOR,
                    cutlass.Float32,
                    cutlass.Float32,
                    tile_shape,
                    False,
                ),
                acc,
            )
            reader = copy.get_slice(local_tid)
            dummy = cute.make_tensor(
                out.iterator, cute.make_layout(tile_shape, stride=(1, self.m))
            )
            layout = cute_ext.make_t2r_rmem_layout(
                copy, cute.flat_divide(dummy, tile_shape), local_tid
            )
            values = cute_ext.allocate(
                cutlass.Float32, cute.AddressSpace.rmem, layout, alignment=32
            )
            coords = reader.partition_D(coordinates)
            cute_ext.partition_and_copy(reader, acc, values)
            cute.arch.fence_view_async_tmem_load()
            if cutlass.const_expr(self.role == "mix"):
                residual = cute.make_rmem_tensor_like(values, original.element_type)
                scales = cute.make_rmem_tensor_like(values)
                residual.fill(0.0)
                scales.fill(1.0)
                # Issue independent input loads before the sigmoid arithmetic.
                # Interleaving a load and store for every accumulator element
                # otherwise serializes global-memory latency.
                for index in cutlass.range_constexpr(cute.size(values)):
                    col, row = coords[index]
                    global_row = token_tile * self.n + row
                    global_col = (
                        (col // 32) * self.hidden_size + output_tile * 32 + col % 32
                    )
                    if global_row < x.shape[0]:
                        residual[index] = original[global_row, global_col]
                        scale = weight_scale[
                            global_col if self.weight_scale_per_channel else 0
                        ]
                        if cutlass.const_expr(native_fp8):
                            scale = scale * activation_scale[global_row]
                        scales[index] = scale
                for i in cutlass.range_constexpr(cute.size(values)):
                    values[i] = residual[i].to(cutlass.Float32) * _up_sigmoid_tanh(
                        values[i] * scales[i]
                    )
            for index in cutlass.range_constexpr(cute.size(values)):
                col, row = coords[index]
                global_row = token_tile * self.n + row
                if cutlass.const_expr(self.role == "projection"):
                    global_col = output_tile * self.m + col
                    if global_row < x.shape[0] and global_col < self.projection_rows:
                        partials[split, global_row, global_col] = values[index]
                else:
                    if global_row < x.shape[0]:
                        gate[col, row] = values[index]
            tcgen05_fence("before_thread_sync")
            cute.arch.mbarrier_arrive(read_done)
            if cutlass.const_expr(self.role == "mix"):
                _epilogue_barrier(self.epilogue_threads, cutlass.Int32(2))
                for offset in cutlass.range_constexpr(
                    self.n * 32 // self.epilogue_threads
                ):
                    index = tid + offset * self.epilogue_threads
                    row, col = index // 32, index % 32
                    if token_tile * self.n + row < x.shape[0]:
                        value = cutlass.Float32(0.0)
                        for branch in cutlass.range_constexpr(self.hc_count):
                            value = value + gate[branch * 32 + col, row]
                        out[token_tile * self.n + row, output_tile * 32 + col] = (
                            value / self.hc_count
                        ).to(original.element_type)

    @cute.experimental.kernel
    def _native_up_kernel(
        self, x, weight, original, weight_scale, activation_scale, out
    ):
        tid = cute.arch.thread_idx()[0]
        warp = cute.arch.make_warp_uniform(tid // 32)
        hidden_tile, token_tile, _ = cute.arch.block_idx()
        dtype = x.element_type
        native_fp8 = cutlass.const_expr(dtype == cutlass.Float8E4M3FN)
        mma = sm100_utils.make_trivial_tiled_mma(
            dtype,
            dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            cutlass.Float32,
            tcgen05.CtaGroup.ONE,
            (self.up_token_rows, self.up_hidden_tile * self.hc_count),
        )
        tiler = (self.up_token_rows, self.up_hidden_tile * self.hc_count, self.k)
        activation_layout = sm100_utils.make_smem_layout_a(
            mma, tiler, dtype, self.stages
        )
        raw_layout = sm100_utils.make_smem_layout_b(
            mma, tiler, cutlass.Float8E4M3FN, self.stages
        )
        weight_layout = sm100_utils.make_smem_layout_b(mma, tiler, dtype, self.stages)
        sx = cute_ext.allocate(
            dtype, cute.AddressSpace.smem, activation_layout, alignment=1024
        )
        raw = cute_ext.allocate(
            cutlass.Float8E4M3FN, cute.AddressSpace.smem, raw_layout, alignment=1024
        )
        sw = raw
        if cutlass.const_expr(not native_fp8):
            sw = cute_ext.allocate(
                dtype, cute.AddressSpace.smem, weight_layout, alignment=1024
            )
        barriers = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(3 * self.stages + 2),
            alignment=8,
        ).iterator
        full = barriers
        converted = full + self.stages
        empty = converted + self.stages
        done, read_done = empty + self.stages, empty + self.stages + 1
        base = cute_ext.allocate(
            cutlass.Int32, cute.AddressSpace.smem, cute.make_layout(1), alignment=4
        ).iterator
        if warp == self.mma_warp:
            with cute.arch.elect_one():
                for stage in cutlass.range_constexpr(self.stages):
                    cute.arch.mbarrier_init(full + stage, 1)
                    cute.arch.mbarrier_init(converted + stage, 1)
                    cute.arch.mbarrier_init(empty + stage, 1)
                cute.arch.mbarrier_init(done, 1)
                cute.arch.mbarrier_init(read_done, self.epilogue_threads)
            cute.arch.alloc_tmem(
                self.up_hidden_tile * self.hc_count, base, is_two_cta=False
            )
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        activation = cute.make_tensor(
            x.iterator,
            cute.make_layout(
                (x.shape[0], (64, self.inner // 64), 1),
                stride=(self.inner, (1, 64), 0),
            ),
        )
        weights = cute.make_tensor(
            weight.iterator,
            cute.make_layout(
                (
                    (
                        self.up_hidden_tile,
                        self.hc_count,
                        self.hidden_size // self.up_hidden_tile,
                    ),
                    (64, self.inner // 64),
                    1,
                ),
                stride=(
                    (
                        self.inner,
                        self.hidden_size * self.inner,
                        self.up_hidden_tile * self.inner,
                    ),
                    (1, 64),
                    0,
                ),
            ),
        )
        gx = cute.local_tile(
            activation, (self.up_token_rows, self.k), (token_tile, None, 0)
        )
        gw = cute.local_tile(
            weights,
            (self.up_hidden_tile * self.hc_count, self.k),
            (hidden_tile, None, 0),
        )
        map_x = cute_ext.get_cta_v_map_ab(activation, tiler, mma, "A")
        map_w = cute_ext.get_cta_v_map_ab(weights, tiler, mma, "B")
        if warp == self.producer_warp:
            for stage in cutlass.range_constexpr(self.stages):
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        full + stage,
                        self.up_hidden_tile * self.hc_count * self.k
                        + self.up_token_rows * self.k * (dtype.width // 8),
                    )
                if cutlass.const_expr(self.weights_independent):
                    cute_ext.tma_load(
                        gw[None, None, stage],
                        raw[None, None, None, stage],
                        (full + stage).value,
                        cta_v_map=map_w,
                        update_expect_tx=False,
                    )
            if cutlass.const_expr(self.pdl):
                cute.arch.griddepcontrol_wait()
                cute.arch.griddepcontrol_launch_dependents()
            for stage in cutlass.range_constexpr(self.stages):
                if cutlass.const_expr(not self.weights_independent):
                    cute_ext.tma_load(
                        gw[None, None, stage],
                        raw[None, None, None, stage],
                        (full + stage).value,
                        cta_v_map=map_w,
                        update_expect_tx=False,
                    )
                cute_ext.tma_load(
                    gx[None, None, stage],
                    sx[None, None, None, stage],
                    (full + stage).value,
                    cta_v_map=map_x,
                    update_expect_tx=False,
                )
            for tile in cutlass.range(self.stages, self.tiles, unroll=1):
                stage = tile % self.stages
                cute.arch.mbarrier_wait(empty + stage, (tile // self.stages - 1) % 2)
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        full + stage,
                        self.up_hidden_tile * self.hc_count * self.k
                        + self.up_token_rows * self.k * (dtype.width // 8),
                    )
                cute_ext.tma_load(
                    gw[None, None, tile],
                    raw[None, None, None, stage],
                    (full + stage).value,
                    cta_v_map=map_w,
                    update_expect_tx=False,
                )
                cute_ext.tma_load(
                    gx[None, None, tile],
                    sx[None, None, None, stage],
                    (full + stage).value,
                    cta_v_map=map_x,
                    update_expect_tx=False,
                )
        elif warp == self.mma_warp:
            thread_mma = mma.get_slice(0)
            a_desc = thread_mma.make_fragment_A(sx)
            b_desc = thread_mma.make_fragment_B(sw)
            fragment = thread_mma.make_fragment_C(
                mma.partition_shape_C(
                    (self.up_token_rows, self.up_hidden_tile * self.hc_count)
                )
            )
            accumulator = cute.make_tensor(
                cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, base), fragment.layout
            )
            mma.set(tcgen05.Field.ACCUMULATE, False)
            for tile in cutlass.range(self.tiles, unroll=1):
                stage = tile % self.stages
                ready = full if cutlass.const_expr(native_fp8) else converted
                cute.arch.mbarrier_wait(ready + stage, tile // self.stages % 2)
                for block in cutlass.range_constexpr(cute.size(a_desc, mode=[2])):
                    cute.gemm(
                        mma,
                        accumulator,
                        a_desc[None, None, block, stage],
                        b_desc[None, None, block, stage],
                        accumulator,
                    )
                    mma.set(tcgen05.Field.ACCUMULATE, True)
                with cute.arch.elect_one():
                    tcgen05.commit(empty + stage, None, tcgen05.CtaGroup.ONE)
            with cute.arch.elect_one():
                tcgen05.commit(done, None, tcgen05.CtaGroup.ONE)
            cute.arch.mbarrier_wait(read_done, 0)
            cute.arch.relinquish_tmem_alloc_permit()
            cute.arch.dealloc_tmem(
                cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, base),
                self.up_hidden_tile * self.hc_count,
            )
        elif tid < self.epilogue_threads:
            if cutlass.const_expr(not native_fp8):
                for tile in cutlass.range(self.tiles, unroll=1):
                    stage = tile % self.stages
                    cute.arch.mbarrier_wait(full + stage, tile // self.stages % 2)
                    source, target = (
                        raw[None, None, None, stage],
                        sw[None, None, None, stage],
                    )
                    for offset in cutlass.range(
                        128 * self.k // (self.epilogue_threads * 8), unroll=4
                    ):
                        linear = (tid + offset * self.epilogue_threads) * 8
                        index = linear // self.k + (linear % self.k) * 128
                        src = cute.make_tensor(
                            cute.make_ptr(
                                cutlass.Float8E4M3FN,
                                (source.iterator + source.layout(index)).toint(),
                                cute.AddressSpace.smem,
                                assumed_align=8,
                                swizzle_=cute.get_swizzle_portion(raw_layout),
                            ),
                            cute.make_layout(8),
                        )
                        dst = cute.make_tensor(
                            cute.make_ptr(
                                dtype,
                                (target.iterator + target.layout(index)).toint(),
                                cute.AddressSpace.smem,
                                assumed_align=16,
                                swizzle_=cute.get_swizzle_portion(weight_layout),
                            ),
                            cute.make_layout(8),
                        )
                        raw_values = cute.make_rmem_tensor(8, cutlass.Float8E4M3FN)
                        converted_values = cute.make_rmem_tensor(8, dtype)
                        cute.copy(
                            cute.make_copy_atom(
                                cute.nvgpu.CopyUniversalOp(),
                                cutlass.Float8E4M3FN,
                                num_bits_per_copy=64,
                            ),
                            src,
                            raw_values,
                        )
                        converted_values.store(raw_values.load().to(dtype))
                        cute.copy(
                            cute.make_copy_atom(
                                cute.nvgpu.CopyUniversalOp(),
                                dtype,
                                num_bits_per_copy=128,
                            ),
                            converted_values,
                            dst,
                        )
                    cute.arch.fence_view_async_shared()
                    _epilogue_barrier(self.epilogue_threads, cutlass.Int32(1))
                    if tid == 0:
                        cute.arch.mbarrier_arrive(converted + stage)
            if cutlass.const_expr(self.pdl):
                cute.arch.griddepcontrol_wait()
            thread_mma = mma.get_slice(0)
            fragment = thread_mma.make_fragment_C(
                mma.partition_shape_C(
                    (self.up_token_rows, self.up_hidden_tile * self.hc_count)
                )
            )
            _, _, template, coordinates = self._native_up_reader(
                base, fragment.layout, original, tid, 0
            )
            count = cute.size(template)
            ring_slots = (
                self.hc_count
                if self.shared_up_scales
                else (2 if self.up_hidden_tile == 64 else self.hc_count)
            )
            residual = cute.make_rmem_tensor((count, ring_slots), original.element_type)
            scales = cute.make_rmem_tensor(
                (1, 1) if self.shared_up_scales else (count, ring_slots),
                cutlass.Float32,
            )
            residual.fill(0.0)
            scales.fill(1.0)
            sums = cute.make_rmem_tensor(count, cutlass.Float32)
            sums.fill(0.0)
            first_row, first_col = coordinates[0]
            global_row = token_tile * self.up_token_rows + first_row
            hidden = hidden_tile * self.up_hidden_tile + first_col
            alpha = cutlass.Float32(1.0)
            if cutlass.const_expr(self.shared_up_scales):
                scale_cache = cute_ext.allocate(
                    cutlass.Float32,
                    cute.AddressSpace.smem,
                    cute.make_layout(256),
                    alignment=32,
                )
                if tid < 256:
                    channel = (
                        (tid // 64) * self.hidden_size + hidden_tile * 64 + tid % 64
                    )
                    scale_cache[tid] = weight_scale[
                        channel if self.weight_scale_per_channel else 0
                    ]
                _epilogue_barrier(self.epilogue_threads, cutlass.Int32(3))
                if cutlass.const_expr(native_fp8):
                    if global_row < x.shape[0]:
                        alpha = activation_scale[global_row]
            if cutlass.const_expr(self.up_token_rows == 64):
                for branch in cutlass.range_constexpr(self.hc_count):
                    self._native_up_fragment_prefetch(
                        original,
                        weight_scale,
                        activation_scale,
                        residual,
                        scales,
                        coordinates,
                        token_tile,
                        hidden_tile,
                        x.shape[0],
                        branch,
                    )
            else:
                for branch in cutlass.range_constexpr(ring_slots):
                    self._native_up_prefetch(
                        original,
                        weight_scale,
                        activation_scale,
                        residual,
                        scales,
                        global_row,
                        hidden,
                        x.shape[0],
                        branch,
                        branch,
                        native_fp8,
                    )
            cute.arch.mbarrier_wait(done, 0)
            if ((tid % 128) // 32) * (self.up_token_rows // 4) < x.shape[
                0
            ] - token_tile * self.up_token_rows:
                for branch in cutlass.range_constexpr(self.hc_count):
                    slot = branch % ring_slots
                    accumulator, reader, gates, _ = self._native_up_reader(
                        base, fragment.layout, original, tid, branch
                    )
                    cute_ext.partition_and_copy(reader, accumulator, gates)
                    cute.arch.fence_view_async_tmem_load()
                    for i in cutlass.range_constexpr(count):
                        factor = cutlass.Float32(1.0)
                        if cutlass.const_expr(self.shared_up_scales):
                            factor = scale_cache[branch * 64 + first_col + i] * alpha
                        else:
                            factor = scales[i, slot]
                        sums[i] = sums[i] + residual[i, slot].to(
                            cutlass.Float32
                        ) * _up_sigmoid_tanh(gates[i] * factor)
                    if cutlass.const_expr(branch + ring_slots < self.hc_count):
                        self._native_up_prefetch(
                            original,
                            weight_scale,
                            activation_scale,
                            residual,
                            scales,
                            global_row,
                            hidden,
                            x.shape[0],
                            branch + ring_slots,
                            slot,
                            native_fp8,
                        )
            tcgen05_fence("before_thread_sync")
            cute.arch.mbarrier_arrive(read_done)
            if cutlass.const_expr(self.up_token_rows == 64):
                for packet in cutlass.range_constexpr(count // 2):
                    row, col = coordinates[packet * 2]
                    output_row = token_tile * self.up_token_rows + row
                    if output_row < x.shape[0]:
                        packed = cute.make_rmem_tensor(2, out.element_type)
                        for j in cutlass.range_constexpr(2):
                            packed[j] = (sums[packet * 2 + j] / self.hc_count).to(
                                out.element_type
                            )
                        target = cute.make_tensor(
                            cute.make_ptr(
                                out.element_type,
                                (
                                    out.iterator
                                    + output_row * self.hidden_size
                                    + hidden_tile * 32
                                    + col
                                ).toint(),
                                cute.AddressSpace.gmem,
                                assumed_align=4,
                            ),
                            cute.make_layout(2),
                        )
                        cute.copy(
                            cute.make_copy_atom(
                                cute.nvgpu.CopyUniversalOp(),
                                out.element_type,
                                num_bits_per_copy=32,
                            ),
                            packed,
                            target,
                        )
            else:
                if global_row < x.shape[0]:
                    packed = cute.make_rmem_tensor(count, out.element_type)
                    for i in cutlass.range_constexpr(count):
                        packed[i] = (sums[i] / self.hc_count).to(out.element_type)
                    target = cute.make_tensor(
                        cute.make_ptr(
                            out.element_type,
                            (
                                out.iterator + global_row * self.hidden_size + hidden
                            ).toint(),
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        ),
                        cute.make_layout(count),
                    )
                    cute.copy(
                        cute.make_copy_atom(
                            cute.nvgpu.CopyUniversalOp(),
                            out.element_type,
                            num_bits_per_copy=128,
                        ),
                        packed,
                        target,
                    )


class W8GatedResidualBranchUp:
    """Keep each HC branch on identical TMEM lanes in separate column ranges."""

    def __init__(
        self,
        *,
        hidden_size: int,
        lowrank: int,
        token_tile: int,
        k_tile: int,
        stages: int,
        use_pdl: bool,
        weights_independent: bool,
        weight_scale_per_channel: bool,
    ):
        self.hidden_size = hidden_size
        self.r = lowrank
        self.n = token_tile
        self.k = k_tile
        self.stages = stages
        self.tiles = (lowrank + k_tile - 1) // k_tile
        self.pdl = use_pdl
        self.weights_independent = weights_independent
        self.per_channel = weight_scale_per_channel
        self.columns = max(32, 4 * token_tile)
        self.threads = 320
        self.readers = 256 if token_tile >= 64 else 128
        self.reader_groups = self.readers // 128

    @cute.experimental.jit
    def __call__(
        self, x, weight, original, weight_scale, activation_scale, out, stream: CUstream
    ):
        self.kernel(x, weight, original, weight_scale, activation_scale, out).launch(
            grid=(self.hidden_size // 64, cute.ceil_div(x.shape[0], self.n), 1),
            block=(self.threads, 1, 1),
            stream=stream,
            use_pdl=self.pdl,
        )

    @cute.experimental.jit
    def _accumulator(self, mma, base, branch: cutlass.Constexpr):
        fragment = mma.get_slice(0).make_fragment_C(mma.partition_shape_C((64, self.n)))
        return cute.make_tensor(
            cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, base) + branch * self.n,
            fragment.layout,
        )[((None, None), 0, 0)]

    @cute.experimental.kernel
    def kernel(self, x, weight, original, weight_scale, activation_scale, out):
        tid = cute.arch.thread_idx()[0]
        warp = cute.arch.make_warp_uniform(tid // 32)
        hidden_tile, token_tile, _ = cute.arch.block_idx()
        dtype = x.element_type
        mma = sm100_utils.make_trivial_tiled_mma(
            dtype,
            dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            cutlass.Float32,
            tcgen05.CtaGroup.ONE,
            (64, self.n),
        )
        tiler = (256, self.n, self.k)
        weight_layout = sm100_utils.make_smem_layout_a(mma, tiler, dtype, self.stages)
        x_layout = sm100_utils.make_smem_layout_b(mma, tiler, dtype, self.stages)
        sw = cute_ext.allocate(
            dtype, cute.AddressSpace.smem, weight_layout, alignment=1024
        )
        raw_layout = sm100_utils.make_smem_layout_a(
            mma, tiler, cutlass.Float8E4M3FN, self.stages
        )
        raw_base = cute.get_nonswizzle_portion(raw_layout)
        weight_base = cute.get_nonswizzle_portion(weight_layout)
        raw = cute.make_tensor(
            cute.make_ptr(
                cutlass.Float8E4M3FN,
                sw.iterator.toint(),
                cute.AddressSpace.smem,
                assumed_align=1024,
                swizzle_=cute.get_swizzle_portion(raw_layout),
            ),
            cute.make_layout(
                raw_base.shape,
                stride=(
                    raw_base.stride[0],
                    raw_base.stride[1],
                    raw_base.stride[2],
                    weight_base.stride[3] * 2,
                ),
            ),
        )
        sx = cute_ext.allocate(dtype, cute.AddressSpace.smem, x_layout, alignment=1024)
        barriers = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(3 * self.stages + 2),
            alignment=8,
        ).iterator
        full, converted, empty = (
            barriers,
            barriers + self.stages,
            barriers + self.stages * 2,
        )
        done, read_done = empty + self.stages, empty + self.stages + 1
        base = cute_ext.allocate(
            cutlass.Int32, cute.AddressSpace.smem, cute.make_layout(1), alignment=4
        ).iterator
        if warp == 9:
            with cute.arch.elect_one():
                for stage in cutlass.range_constexpr(self.stages):
                    cute.arch.mbarrier_init(full + stage, 1)
                    cute.arch.mbarrier_init(empty + stage, 1)
                    cute.arch.mbarrier_init(converted + stage, 8)
                cute.arch.mbarrier_init(done, 1)
                cute.arch.mbarrier_init(read_done, self.readers)
            cute.arch.alloc_tmem(self.columns, base, is_two_cta=False)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        weights = cute.make_tensor(
            weight.iterator,
            cute.make_layout(
                ((64, 4, self.hidden_size // 64), (64, self.r // 64), 1),
                stride=((self.r, self.hidden_size * self.r, 64 * self.r), (1, 64), 0),
            ),
        )
        activation = cute.make_tensor(
            x.iterator,
            cute.make_layout(
                (x.shape[0], (64, self.r // 64), 1), stride=(self.r, (1, 64), 0)
            ),
        )
        gw = cute.local_tile(weights, (256, self.k), (hidden_tile, None, 0))
        gx = cute.local_tile(activation, (self.n, self.k), (token_tile, None, 0))
        map_w = cute_ext.get_cta_v_map_ab(weights, tiler, mma, "A")
        map_x = cute_ext.get_cta_v_map_ab(activation, tiler, mma, "B")
        if warp == 8:
            for stage in cutlass.range_constexpr(self.stages):
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        full + stage, (256 + self.n * 2) * self.k
                    )
                if cutlass.const_expr(self.weights_independent):
                    cute_ext.tma_load(
                        gw[None, None, stage],
                        raw[None, None, None, stage],
                        (full + stage).value,
                        cta_v_map=map_w,
                        update_expect_tx=False,
                    )
            if cutlass.const_expr(self.pdl):
                cute.arch.griddepcontrol_wait()
                cute.arch.griddepcontrol_launch_dependents()
            for stage in cutlass.range_constexpr(self.stages):
                if cutlass.const_expr(not self.weights_independent):
                    cute_ext.tma_load(
                        gw[None, None, stage],
                        raw[None, None, None, stage],
                        (full + stage).value,
                        cta_v_map=map_w,
                        update_expect_tx=False,
                    )
                cute_ext.tma_load(
                    gx[None, None, stage],
                    sx[None, None, None, stage],
                    (full + stage).value,
                    cta_v_map=map_x,
                    update_expect_tx=False,
                )
            for tile in cutlass.range(self.stages, self.tiles, unroll=1):
                stage = tile % self.stages
                cute.arch.mbarrier_wait(empty + stage, (tile // self.stages - 1) % 2)
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        full + stage, (256 + self.n * 2) * self.k
                    )
                cute_ext.tma_load(
                    gw[None, None, tile],
                    raw[None, None, None, stage],
                    (full + stage).value,
                    cta_v_map=map_w,
                    update_expect_tx=False,
                )
                cute_ext.tma_load(
                    gx[None, None, tile],
                    sx[None, None, None, stage],
                    (full + stage).value,
                    cta_v_map=map_x,
                    update_expect_tx=False,
                )
        elif warp == 9:
            thread_mma = mma.get_slice(0)
            a_desc, b_desc = thread_mma.make_fragment_A(sw), thread_mma.make_fragment_B(
                sx
            )
            mma.set(tcgen05.Field.ACCUMULATE, False)
            for tile in cutlass.range(self.tiles, unroll=1):
                stage = tile % self.stages
                cute.arch.mbarrier_wait(converted + stage, tile // self.stages % 2)
                for block in cutlass.range_constexpr(cute.size(a_desc, mode=[2])):
                    for branch in cutlass.range_constexpr(4):
                        fragment = mma.get_slice(0).make_fragment_C(
                            mma.partition_shape_C((64, self.n))
                        )
                        acc = cute.make_tensor(
                            cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, base)
                            + branch * self.n,
                            fragment.layout,
                        )
                        a_branch = cute.local_tile(
                            a_desc[None, None, block, stage], (1, 1), (0, branch)
                        )
                        cute.gemm(
                            mma, acc, a_branch, b_desc[None, None, block, stage], acc
                        )
                    mma.set(tcgen05.Field.ACCUMULATE, True)
                with cute.arch.elect_one():
                    tcgen05.commit(empty + stage, None, tcgen05.CtaGroup.ONE)
            with cute.arch.elect_one():
                tcgen05.commit(done, None, tcgen05.CtaGroup.ONE)
            cute.arch.mbarrier_wait(read_done, 0)
            cute.arch.relinquish_tmem_alloc_permit()
            cute.arch.dealloc_tmem(
                cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, base), self.columns
            )
        elif tid < 256:
            for tile in cutlass.range(self.tiles, unroll=1):
                stage = tile % self.stages
                cute.arch.mbarrier_wait(full + stage, tile // self.stages % 2)
                packets: cutlass.Constexpr = 256 * self.k // (256 * 8)
                raw_values = cute.make_rmem_tensor((8, packets), cutlass.Float8E4M3FN)
                source = raw[None, None, None, stage]
                target = sw[None, None, None, stage]
                for offset in cutlass.range_constexpr(packets):
                    linear = (tid + offset * 256) * 8
                    row, k = linear // self.k, linear % self.k
                    # M=256 consists of four M64 atoms; preserve their M/K
                    # hierarchy when converting each contiguous K packet.
                    atom_m: cutlass.Constexpr = cute.size(raw_layout, mode=[0, 0])
                    atom_k: cutlass.Constexpr = cute.size(raw_layout, mode=[0, 1])
                    index = (
                        row % atom_m
                        + (k % atom_k) * atom_m
                        + (row // atom_m) * atom_m * atom_k
                        + (k // atom_k) * 256 * atom_k
                    )
                    src = cute.make_tensor(
                        cute.make_ptr(
                            cutlass.Float8E4M3FN,
                            (source.iterator + source.layout(index)).toint(),
                            cute.AddressSpace.smem,
                            assumed_align=8,
                            swizzle_=cute.get_swizzle_portion(raw_layout),
                        ),
                        cute.make_layout(8),
                    )
                    cute.copy(
                        cute.make_copy_atom(
                            cute.nvgpu.CopyUniversalOp(),
                            cutlass.Float8E4M3FN,
                            num_bits_per_copy=64,
                        ),
                        src,
                        raw_values[None, offset],
                    )
                _epilogue_barrier(256, cutlass.Int32(1))
                for offset in cutlass.range_constexpr(packets):
                    linear = (tid + offset * 256) * 8
                    row, k = linear // self.k, linear % self.k
                    decoded_m: cutlass.Constexpr = cute.size(weight_layout, mode=[0, 0])
                    decoded_k: cutlass.Constexpr = cute.size(weight_layout, mode=[0, 1])
                    index = (
                        row % decoded_m
                        + (k % decoded_k) * decoded_m
                        + (row // decoded_m) * decoded_m * decoded_k
                        + (k // decoded_k) * 256 * decoded_k
                    )
                    dst = cute.make_tensor(
                        cute.make_ptr(
                            dtype,
                            (target.iterator + target.layout(index)).toint(),
                            cute.AddressSpace.smem,
                            assumed_align=16,
                            swizzle_=cute.get_swizzle_portion(weight_layout),
                        ),
                        cute.make_layout(8),
                    )
                    values = cute.make_rmem_tensor(8, dtype)
                    _convert_fp8_weight(raw_values[None, offset], values)
                    cute.copy(
                        cute.make_copy_atom(
                            cute.nvgpu.CopyUniversalOp(), dtype, num_bits_per_copy=128
                        ),
                        values,
                        dst,
                    )
                cute.arch.fence_view_async_shared()
                cute.arch.sync_warp()
                if tid % 32 == 0:
                    cute.arch.mbarrier_arrive(converted + stage)
        if tid < self.readers:
            if cutlass.const_expr(self.pdl):
                cute.arch.griddepcontrol_wait()
            group = tid // 128
            tile_shape = (64, self.n // self.reader_groups)
            acc = cute.local_tile(
                self._accumulator(mma, base, 0), tile_shape, (0, group)
            )
            copy = tcgen05.make_tmem_copy(
                sm100_utils.get_tmem_load_op(
                    (64, self.n, self.k),
                    cute_utils.LayoutEnum.COL_MAJOR,
                    cutlass.Float32,
                    cutlass.Float32,
                    tile_shape,
                    False,
                ),
                acc,
            )
            reader = copy.get_slice(tid % 128)
            dummy = cute.make_tensor(
                original.iterator, cute.make_layout(tile_shape, stride=(1, 64))
            )
            gates = cute_ext.allocate(
                cutlass.Float32,
                cute.AddressSpace.rmem,
                cute_ext.make_t2r_rmem_layout(
                    copy, cute.flat_divide(dummy, tile_shape), tid % 128
                ),
                alignment=32,
            )
            coords = reader.partition_D(
                cute.local_tile(
                    cute.make_identity_tensor((64, self.n)), tile_shape, (0, group)
                )
            )
            count = cute.size(gates)
            ring_slots: cutlass.Constexpr = 2 if self.n >= 32 else 4
            residual = cute.make_rmem_tensor((count, ring_slots), original.element_type)
            scales = cute.make_rmem_tensor((count, ring_slots), cutlass.Float32)
            residual.fill(0.0)
            scales.fill(1.0)
            for branch in cutlass.range_constexpr(ring_slots):
                for i in cutlass.range_constexpr(count):
                    col, row = coords[i]
                    global_row = token_tile * self.n + row
                    global_col = branch * self.hidden_size + hidden_tile * 64 + col
                    if global_row < x.shape[0]:
                        residual[i, branch] = original[global_row, global_col]
                        scales[i, branch] = weight_scale[
                            global_col if self.per_channel else 0
                        ]
            sums = cute.make_rmem_tensor(count, cutlass.Float32)
            sums.fill(0.0)
            cute.arch.mbarrier_wait(done, 0)
            for branch in cutlass.range_constexpr(4):
                acc = cute.local_tile(
                    self._accumulator(mma, base, branch), tile_shape, (0, group)
                )
                cute_ext.partition_and_copy(reader, acc, gates)
                slot = branch % ring_slots
                for i in cutlass.range_constexpr(count):
                    sums[i] = sums[i] + residual[i, slot].to(
                        cutlass.Float32
                    ) * _up_sigmoid_tanh(gates[i] * scales[i, slot])
                if cutlass.const_expr(branch + ring_slots < 4):
                    for i in cutlass.range_constexpr(count):
                        col, row = coords[i]
                        global_row = token_tile * self.n + row
                        global_col = (
                            (branch + ring_slots) * self.hidden_size
                            + hidden_tile * 64
                            + col
                        )
                        if global_row < x.shape[0]:
                            residual[i, slot] = original[global_row, global_col]
                            scales[i, slot] = weight_scale[
                                global_col if self.per_channel else 0
                            ]
            cute.arch.fence_view_async_tmem_load()
            tcgen05_fence("before_thread_sync")
            cute.arch.mbarrier_arrive(read_done)
            for i in cutlass.range_constexpr(count):
                col, row = coords[i]
                global_row = token_tile * self.n + row
                if global_row < x.shape[0]:
                    out[global_row, hidden_tile * 64 + col] = (sums[i] / 4).to(
                        out.element_type
                    )


class W8GatedResidualTokenReuse(Fp8GatedResidualGemm):
    """Share each decoded weight stage across two M128 token accumulators."""

    def __init__(
        self,
        *,
        hidden_size: int,
        lowrank: int,
        use_pdl: bool,
        weights_independent: bool,
        weight_scale_per_channel: bool,
        weight_scale_vectorized: bool,
    ):
        super().__init__(
            role="mix",
            hc_count=4,
            hidden_size=hidden_size,
            lowrank=lowrank,
            projection_rows=lowrank + 4,
            token_tile=128,
            projection_tile=128,
            k_tile=128,
            stages=2,
            split_k=1,
            use_pdl=use_pdl,
            weights_independent=weights_independent,
            weight_scale_per_channel=weight_scale_per_channel,
            token_major_up=True,
            weight_scale_vectorized=weight_scale_vectorized,
            up_hidden_tile=64,
            up_token_rows=128,
            separate_operand_ready=False,
            epilogue_threads=512,
            shared_up_scales=False,
            interleave_tma=False,
        )
        self.up_hidden_tile = 64

    @cute.experimental.jit
    def __call__(
        self, x, weight, original, weight_scale, activation_scale, out, stream: CUstream
    ):
        self.kernel(x, weight, original, weight_scale, activation_scale, out).launch(
            grid=(self.hidden_size // 64, 1, 1),
            block=(self.epilogue_threads + 64, 1, 1),
            stream=stream,
            use_pdl=self.pdl,
        )

    @cute.experimental.jit
    def _up_prefetch(
        self,
        original,
        weight_scale,
        activation_scale,
        residual,
        scales,
        row,
        hidden,
        rows,
        branch: cutlass.Constexpr,
        slot: cutlass.Constexpr,
        native_fp8: cutlass.Constexpr,
    ):
        count = cute.size(residual, mode=[0])
        if row < rows:
            source = cute.make_tensor(
                cute.make_ptr(
                    original.element_type,
                    (
                        original.iterator
                        + row * self.wide
                        + branch * self.hidden_size
                        + hidden
                    ).toint(),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                ),
                cute.make_layout(count),
            )
            cute.copy(
                cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(),
                    original.element_type,
                    num_bits_per_copy=128,
                ),
                source,
                residual[None, slot],
            )
            if cutlass.const_expr(
                self.weight_scale_per_channel and self.weight_scale_vectorized
            ):
                source_scale = cute.make_tensor(
                    cute.make_ptr(
                        cutlass.Float32,
                        (
                            weight_scale.iterator + branch * self.hidden_size + hidden
                        ).toint(),
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    ),
                    cute.make_layout(count),
                )
                cute.copy(
                    cute.make_copy_atom(
                        cute.nvgpu.CopyUniversalOp(),
                        cutlass.Float32,
                        num_bits_per_copy=128,
                    ),
                    source_scale,
                    scales[None, slot],
                )
            else:
                for i in cutlass.range_constexpr(count):
                    scales[i, slot] = weight_scale[
                        (
                            branch * self.hidden_size + hidden + i
                            if self.weight_scale_per_channel
                            else 0
                        )
                    ]
            if cutlass.const_expr(native_fp8):
                for i in cutlass.range_constexpr(count):
                    scales[i, slot] = scales[i, slot] * activation_scale[row]

    @cute.experimental.jit
    def _up_reader(
        self, base, layout, original, tid, branch: cutlass.Constexpr, token_part
    ):
        tile = (128, self.up_hidden_tile // self.epilogue_groups)
        group = tid // 128 + branch * self.epilogue_groups
        accumulator = cute.make_tensor(
            cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, base) + token_part * 256,
            layout,
        )[((None, None), 0, 0)]
        accumulator = cute.local_tile(accumulator, tile, (0, group))
        coordinates = cute.local_tile(
            cute.make_identity_tensor((128, self.up_hidden_tile * self.hc_count)),
            tile,
            (0, group),
        )
        copy = tcgen05.make_tmem_copy(
            sm100_utils.get_tmem_load_op(
                (128, self.up_hidden_tile * self.hc_count, self.k),
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
            original.iterator, cute.make_layout(tile, stride=(1, 128))
        )
        values = cute_ext.allocate(
            cutlass.Float32,
            cute.AddressSpace.rmem,
            cute_ext.make_t2r_rmem_layout(
                copy, cute.flat_divide(dummy, tile), tid % 128
            ),
            alignment=32,
        )
        return accumulator, reader, values, reader.partition_D(coordinates)

    @cute.experimental.kernel
    def kernel(self, x, weight, original, weight_scale, activation_scale, out):
        tid = cute.arch.thread_idx()[0]
        warp = cute.arch.make_warp_uniform(tid // 32)
        hidden_tile, token_tile, _ = cute.arch.block_idx()
        dtype = x.element_type
        native_fp8 = cutlass.const_expr(dtype == cutlass.Float8E4M3FN)
        mma = sm100_utils.make_trivial_tiled_mma(
            dtype,
            dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            cutlass.Float32,
            tcgen05.CtaGroup.ONE,
            (128, self.up_hidden_tile * self.hc_count),
        )
        tiler = (128, self.up_hidden_tile * self.hc_count, self.k)
        activation_layout = sm100_utils.make_smem_layout_a(
            mma, tiler, dtype, self.stages
        )
        raw_layout = sm100_utils.make_smem_layout_b(
            mma, tiler, cutlass.Float8E4M3FN, self.stages
        )
        weight_layout = sm100_utils.make_smem_layout_b(mma, tiler, dtype, self.stages)
        sx = cute_ext.allocate(
            dtype, cute.AddressSpace.smem, activation_layout, alignment=1024
        )
        if cutlass.const_expr(native_fp8):
            raw = cute_ext.allocate(
                cutlass.Float8E4M3FN, cute.AddressSpace.smem, raw_layout, alignment=1024
            )
            sw = raw
        else:
            sw = cute_ext.allocate(
                dtype, cute.AddressSpace.smem, weight_layout, alignment=1024
            )
            raw_base = cute.get_nonswizzle_portion(raw_layout)
            weight_base = cute.get_nonswizzle_portion(weight_layout)
            # Raw and converted tiles share the larger decoded-weight slot.
            raw = cute.make_tensor(
                cute.make_ptr(
                    cutlass.Float8E4M3FN,
                    sw.iterator.toint(),
                    cute.AddressSpace.smem,
                    assumed_align=1024,
                    swizzle_=cute.get_swizzle_portion(raw_layout),
                ),
                cute.make_layout(
                    raw_base.shape,
                    stride=(
                        raw_base.stride[0],
                        raw_base.stride[1],
                        raw_base.stride[2],
                        weight_base.stride[3] * (dtype.width // 8),
                    ),
                ),
            )
        x1_layout = sm100_utils.make_smem_layout_a(mma, tiler, dtype, 1)
        sx1 = cute_ext.allocate(
            dtype, cute.AddressSpace.smem, x1_layout, alignment=1024
        )
        barriers = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(3 * self.stages + 4),
            alignment=8,
        ).iterator
        full = barriers
        converted = full + self.stages
        empty = converted + self.stages
        done, read_done = empty + self.stages, empty + self.stages + 1
        x1_full, x1_empty = empty + self.stages + 2, empty + self.stages + 3
        base = cute_ext.allocate(
            cutlass.Int32, cute.AddressSpace.smem, cute.make_layout(1), alignment=4
        ).iterator
        if warp == self.mma_warp:
            with cute.arch.elect_one():
                for stage in cutlass.range_constexpr(self.stages):
                    cute.arch.mbarrier_init(full + stage, 1)
                    cute.arch.mbarrier_init(
                        converted + stage, self.epilogue_threads // 32
                    )
                    cute.arch.mbarrier_init(empty + stage, 1)
                cute.arch.mbarrier_init(x1_full, 1)
                cute.arch.mbarrier_init(x1_empty, 1)
                cute.arch.mbarrier_init(done, 1)
                cute.arch.mbarrier_init(read_done, self.epilogue_threads)
            cute.arch.alloc_tmem(512, base, is_two_cta=False)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        activation = cute.make_tensor(
            x.iterator,
            cute.make_layout(
                (x.shape[0], (64, self.inner // 64), 1),
                stride=(self.inner, (1, 64), 0),
            ),
        )
        weights = cute.make_tensor(
            weight.iterator,
            cute.make_layout(
                (
                    (
                        self.up_hidden_tile,
                        self.hc_count,
                        self.hidden_size // self.up_hidden_tile,
                    ),
                    (64, self.inner // 64),
                    1,
                ),
                stride=(
                    (
                        self.inner,
                        self.hidden_size * self.inner,
                        self.up_hidden_tile * self.inner,
                    ),
                    (1, 64),
                    0,
                ),
            ),
        )
        gx = cute.local_tile(activation, (128, self.k), (token_tile, None, 0))
        gx1 = cute.local_tile(activation, (128, self.k), (1, None, 0))
        gw = cute.local_tile(
            weights,
            (self.up_hidden_tile * self.hc_count, self.k),
            (hidden_tile, None, 0),
        )
        map_x = cute_ext.get_cta_v_map_ab(activation, tiler, mma, "A")
        map_w = cute_ext.get_cta_v_map_ab(weights, tiler, mma, "B")
        if warp == self.producer_warp:
            for stage in cutlass.range_constexpr(self.stages):
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        full + stage,
                        self.up_hidden_tile * self.hc_count * self.k
                        + 128 * self.k * (dtype.width // 8),
                    )
                if cutlass.const_expr(self.weights_independent):
                    cute_ext.tma_load(
                        gw[None, None, stage],
                        raw[None, None, None, stage],
                        (full + stage).value,
                        cta_v_map=map_w,
                        update_expect_tx=False,
                    )
            if cutlass.const_expr(self.pdl):
                cute.arch.griddepcontrol_wait()
                cute.arch.griddepcontrol_launch_dependents()
            for stage in cutlass.range_constexpr(self.stages):
                if cutlass.const_expr(not self.weights_independent):
                    cute_ext.tma_load(
                        gw[None, None, stage],
                        raw[None, None, None, stage],
                        (full + stage).value,
                        cta_v_map=map_w,
                        update_expect_tx=False,
                    )
                cute_ext.tma_load(
                    gx[None, None, stage],
                    sx[None, None, None, stage],
                    (full + stage).value,
                    cta_v_map=map_x,
                    update_expect_tx=False,
                )
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(x1_full, 128 * self.k * 2)
            cute_ext.tma_load(
                gx1[None, None, 0],
                sx1[None, None, None, 0],
                x1_full.value,
                cta_v_map=map_x,
                update_expect_tx=False,
            )
            for tile in cutlass.range(1, self.tiles, unroll=1):
                # The second token tile has one slot. Its next load overlaps
                # the first token's MMA, while decoded weights have two slots.
                cute.arch.mbarrier_wait(x1_empty, (tile - 1) % 2)
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(x1_full, 128 * self.k * 2)
                cute_ext.tma_load(
                    gx1[None, None, tile],
                    sx1[None, None, None, 0],
                    x1_full.value,
                    cta_v_map=map_x,
                    update_expect_tx=False,
                )
                future = tile + self.stages - 1
                if future < self.tiles:
                    stage = future % self.stages
                    cute.arch.mbarrier_wait(
                        empty + stage, (future // self.stages - 1) % 2
                    )
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            full + stage, (256 + 128 * 2) * self.k
                        )
                    cute_ext.tma_load(
                        gw[None, None, future],
                        raw[None, None, None, stage],
                        (full + stage).value,
                        cta_v_map=map_w,
                        update_expect_tx=False,
                    )
                    cute_ext.tma_load(
                        gx[None, None, future],
                        sx[None, None, None, stage],
                        (full + stage).value,
                        cta_v_map=map_x,
                        update_expect_tx=False,
                    )
        elif warp == self.mma_warp:
            thread_mma = mma.get_slice(0)
            a_desc = thread_mma.make_fragment_A(sx)
            b_desc = thread_mma.make_fragment_B(sw)
            fragment = thread_mma.make_fragment_C(
                mma.partition_shape_C((128, self.up_hidden_tile * self.hc_count))
            )
            accumulator = cute.make_tensor(
                cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, base), fragment.layout
            )
            a1_desc = thread_mma.make_fragment_A(sx1)
            accumulator1 = cute.make_tensor(
                cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, base) + 256,
                fragment.layout,
            )
            for tile in cutlass.range(self.tiles, unroll=1):
                stage = tile % self.stages
                cute.arch.mbarrier_wait(converted + stage, tile // self.stages % 2)
                mma.set(tcgen05.Field.ACCUMULATE, tile > 0)
                for block in cutlass.range_constexpr(cute.size(a_desc, mode=[2])):
                    cute.gemm(
                        mma,
                        accumulator,
                        a_desc[None, None, block, stage],
                        b_desc[None, None, block, stage],
                        accumulator,
                    )
                    mma.set(tcgen05.Field.ACCUMULATE, True)
                cute.arch.mbarrier_wait(x1_full, tile % 2)
                mma.set(tcgen05.Field.ACCUMULATE, tile > 0)
                for block in cutlass.range_constexpr(cute.size(a1_desc, mode=[2])):
                    cute.gemm(
                        mma,
                        accumulator1,
                        a1_desc[None, None, block, 0],
                        b_desc[None, None, block, stage],
                        accumulator1,
                    )
                    mma.set(tcgen05.Field.ACCUMULATE, True)
                with cute.arch.elect_one():
                    tcgen05.commit(empty + stage, None, tcgen05.CtaGroup.ONE)
                    tcgen05.commit(x1_empty, None, tcgen05.CtaGroup.ONE)
            with cute.arch.elect_one():
                tcgen05.commit(done, None, tcgen05.CtaGroup.ONE)
            cute.arch.mbarrier_wait(read_done, 0)
            cute.arch.relinquish_tmem_alloc_permit()
            cute.arch.dealloc_tmem(
                cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, base), 512
            )
        elif tid < self.epilogue_threads:
            if cutlass.const_expr(not native_fp8):
                for tile in cutlass.range(self.tiles, unroll=1):
                    stage = tile % self.stages
                    cute.arch.mbarrier_wait(full + stage, tile // self.stages % 2)
                    source, target = (
                        raw[None, None, None, stage],
                        sw[None, None, None, stage],
                    )
                    packets = (
                        self.up_hidden_tile
                        * self.hc_count
                        * self.k
                        // (self.epilogue_threads * 8)
                    )
                    raw_values = cute.make_rmem_tensor(
                        (8, packets), cutlass.Float8E4M3FN
                    )
                    # Every reader finishes before expanded stores can
                    # overwrite another thread's compressed source bytes.
                    for offset in cutlass.range_constexpr(packets):
                        linear = (tid + offset * self.epilogue_threads) * 8
                        index = (
                            linear // self.k
                            + (linear % self.k) * self.up_hidden_tile * self.hc_count
                        )
                        src = cute.make_tensor(
                            cute.make_ptr(
                                cutlass.Float8E4M3FN,
                                (source.iterator + source.layout(index)).toint(),
                                cute.AddressSpace.smem,
                                assumed_align=8,
                                swizzle_=cute.get_swizzle_portion(raw_layout),
                            ),
                            cute.make_layout(8),
                        )
                        cute.copy(
                            cute.make_copy_atom(
                                cute.nvgpu.CopyUniversalOp(),
                                cutlass.Float8E4M3FN,
                                num_bits_per_copy=64,
                            ),
                            src,
                            raw_values[None, offset],
                        )
                    _epilogue_barrier(self.epilogue_threads, cutlass.Int32(1))
                    for offset in cutlass.range_constexpr(packets):
                        linear = (tid + offset * self.epilogue_threads) * 8
                        index = (
                            linear // self.k
                            + (linear % self.k) * self.up_hidden_tile * self.hc_count
                        )
                        dst = cute.make_tensor(
                            cute.make_ptr(
                                dtype,
                                (target.iterator + target.layout(index)).toint(),
                                cute.AddressSpace.smem,
                                assumed_align=16,
                                swizzle_=cute.get_swizzle_portion(weight_layout),
                            ),
                            cute.make_layout(8),
                        )
                        converted_values = cute.make_rmem_tensor(8, dtype)
                        _convert_fp8_weight(raw_values[None, offset], converted_values)
                        cute.copy(
                            cute.make_copy_atom(
                                cute.nvgpu.CopyUniversalOp(),
                                dtype,
                                num_bits_per_copy=128,
                            ),
                            converted_values,
                            dst,
                        )
                    cute.arch.fence_view_async_shared()
                    cute.arch.sync_warp(mask=0xFFFFFFFF)
                    if tid % 32 == 0:
                        cute.arch.mbarrier_arrive(converted + stage)
            if cutlass.const_expr(self.pdl):
                cute.arch.griddepcontrol_wait()
            for token_part in cutlass.range(2, unroll=1):
                thread_mma = mma.get_slice(0)
                fragment = thread_mma.make_fragment_C(
                    mma.partition_shape_C((128, self.up_hidden_tile * self.hc_count))
                )
                _, _, template, coordinates = self._up_reader(
                    base, fragment.layout, original, tid, 0, token_part
                )
                count = cute.size(template)
                ring_slots = 2 if self.up_hidden_tile == 64 else self.hc_count
                residual = cute.make_rmem_tensor(
                    (count, ring_slots), original.element_type
                )
                scales = cute.make_rmem_tensor((count, ring_slots), cutlass.Float32)
                residual.fill(0.0)
                scales.fill(1.0)
                sums = cute.make_rmem_tensor(count, cutlass.Float32)
                sums.fill(0.0)
                first_row, first_col = coordinates[0]
                global_row = token_part * 128 + first_row
                hidden = hidden_tile * self.up_hidden_tile + first_col
                for branch in cutlass.range_constexpr(ring_slots):
                    self._up_prefetch(
                        original,
                        weight_scale,
                        activation_scale,
                        residual,
                        scales,
                        global_row,
                        hidden,
                        x.shape[0],
                        branch,
                        branch,
                        native_fp8,
                    )
                cute.arch.mbarrier_wait(done, 0)
                if ((tid % 128) // 32) * 32 < x.shape[0] - token_part * 128:
                    for branch in cutlass.range_constexpr(self.hc_count):
                        slot = branch % ring_slots
                        accumulator, reader, gates, _ = self._up_reader(
                            base, fragment.layout, original, tid, branch, token_part
                        )
                        cute_ext.partition_and_copy(reader, accumulator, gates)
                        cute.arch.fence_view_async_tmem_load()
                        for i in cutlass.range_constexpr(count):
                            sums[i] = sums[i] + residual[i, slot].to(
                                cutlass.Float32
                            ) * _up_sigmoid_tanh(gates[i] * scales[i, slot])
                        if cutlass.const_expr(branch + ring_slots < self.hc_count):
                            self._up_prefetch(
                                original,
                                weight_scale,
                                activation_scale,
                                residual,
                                scales,
                                global_row,
                                hidden,
                                x.shape[0],
                                branch + ring_slots,
                                slot,
                                native_fp8,
                            )
                tcgen05_fence("before_thread_sync")
                if global_row < x.shape[0]:
                    packed = cute.make_rmem_tensor(count, out.element_type)
                    for i in cutlass.range_constexpr(count):
                        packed[i] = (sums[i] / self.hc_count).to(out.element_type)
                    target = cute.make_tensor(
                        cute.make_ptr(
                            out.element_type,
                            (
                                out.iterator + global_row * self.hidden_size + hidden
                            ).toint(),
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        ),
                        cute.make_layout(count),
                    )
                    cute.copy(
                        cute.make_copy_atom(
                            cute.nvgpu.CopyUniversalOp(),
                            out.element_type,
                            num_bits_per_copy=128,
                        ),
                        packed,
                        target,
                    )
            cute.arch.mbarrier_arrive(read_done)


class Fp8GatedResidualBranchUp:
    """Keep each HC branch on identical TMEM lanes in separate column ranges."""

    def __init__(
        self,
        *,
        hidden_size: int,
        lowrank: int,
        token_tile: int,
        k_tile: int,
        stages: int,
        use_pdl: bool,
        weights_independent: bool,
        weight_scale_per_channel: bool,
    ):
        self.hidden_size = hidden_size
        self.r = lowrank
        self.n = token_tile
        self.k = k_tile
        self.stages = stages
        self.tiles = (lowrank + k_tile - 1) // k_tile
        self.pdl = use_pdl
        self.weights_independent = weights_independent
        self.per_channel = weight_scale_per_channel
        self.columns = max(32, 4 * token_tile)
        self.threads = 192

    @cute.experimental.jit
    def __call__(
        self, x, weight, original, weight_scale, activation_scale, out, stream: CUstream
    ):
        self.kernel(x, weight, original, weight_scale, activation_scale, out).launch(
            grid=(self.hidden_size // 64, cute.ceil_div(x.shape[0], self.n), 1),
            block=(self.threads, 1, 1),
            stream=stream,
            use_pdl=self.pdl,
        )

    @cute.experimental.jit
    def _accumulator(self, mma, base, branch: cutlass.Constexpr):
        fragment = mma.get_slice(0).make_fragment_C(mma.partition_shape_C((64, self.n)))
        return cute.make_tensor(
            cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, base) + branch * self.n,
            fragment.layout,
        )[((None, None), 0, 0)]

    @cute.experimental.kernel
    def kernel(self, x, weight, original, weight_scale, activation_scale, out):
        tid = cute.arch.thread_idx()[0]
        warp = cute.arch.make_warp_uniform(tid // 32)
        hidden_tile, token_tile, _ = cute.arch.block_idx()
        dtype = cutlass.Float8E4M3FN
        mma = sm100_utils.make_trivial_tiled_mma(
            dtype,
            dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            cutlass.Float32,
            tcgen05.CtaGroup.ONE,
            (64, self.n),
        )
        tiler = (256, self.n, self.k)
        weight_layout = sm100_utils.make_smem_layout_a(mma, tiler, dtype, self.stages)
        x_layout = sm100_utils.make_smem_layout_b(mma, tiler, dtype, self.stages)
        sw = cute_ext.allocate(
            dtype, cute.AddressSpace.smem, weight_layout, alignment=1024
        )
        sx = cute_ext.allocate(dtype, cute.AddressSpace.smem, x_layout, alignment=1024)
        barriers = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(2 * self.stages + 2),
            alignment=8,
        ).iterator
        full, empty = barriers, barriers + self.stages
        done, read_done = empty + self.stages, empty + self.stages + 1
        base = cute_ext.allocate(
            cutlass.Int32, cute.AddressSpace.smem, cute.make_layout(1), alignment=4
        ).iterator
        if warp == 5:
            with cute.arch.elect_one():
                for stage in cutlass.range_constexpr(self.stages):
                    cute.arch.mbarrier_init(full + stage, 1)
                    cute.arch.mbarrier_init(empty + stage, 1)
                cute.arch.mbarrier_init(done, 1)
                cute.arch.mbarrier_init(read_done, 128)
            cute.arch.alloc_tmem(self.columns, base, is_two_cta=False)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        weights = cute.make_tensor(
            weight.iterator,
            cute.make_layout(
                ((64, 4, self.hidden_size // 64), (64, self.r // 64), 1),
                stride=((self.r, self.hidden_size * self.r, 64 * self.r), (1, 64), 0),
            ),
        )
        activation = cute.make_tensor(
            x.iterator,
            cute.make_layout(
                (x.shape[0], (64, self.r // 64), 1), stride=(self.r, (1, 64), 0)
            ),
        )
        gw = cute.local_tile(weights, (256, self.k), (hidden_tile, None, 0))
        gx = cute.local_tile(activation, (self.n, self.k), (token_tile, None, 0))
        map_w = cute_ext.get_cta_v_map_ab(weights, tiler, mma, "A")
        map_x = cute_ext.get_cta_v_map_ab(activation, tiler, mma, "B")
        if warp == 4:
            for stage in cutlass.range_constexpr(self.stages):
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        full + stage, (256 + self.n) * self.k
                    )
                if cutlass.const_expr(self.weights_independent):
                    cute_ext.tma_load(
                        gw[None, None, stage],
                        sw[None, None, None, stage],
                        (full + stage).value,
                        cta_v_map=map_w,
                        update_expect_tx=False,
                    )
            if cutlass.const_expr(self.pdl):
                cute.arch.griddepcontrol_wait()
                cute.arch.griddepcontrol_launch_dependents()
            for stage in cutlass.range_constexpr(self.stages):
                if cutlass.const_expr(not self.weights_independent):
                    cute_ext.tma_load(
                        gw[None, None, stage],
                        sw[None, None, None, stage],
                        (full + stage).value,
                        cta_v_map=map_w,
                        update_expect_tx=False,
                    )
                cute_ext.tma_load(
                    gx[None, None, stage],
                    sx[None, None, None, stage],
                    (full + stage).value,
                    cta_v_map=map_x,
                    update_expect_tx=False,
                )
            for tile in cutlass.range(self.stages, self.tiles, unroll=1):
                stage = tile % self.stages
                cute.arch.mbarrier_wait(empty + stage, (tile // self.stages - 1) % 2)
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        full + stage, (256 + self.n) * self.k
                    )
                cute_ext.tma_load(
                    gw[None, None, tile],
                    sw[None, None, None, stage],
                    (full + stage).value,
                    cta_v_map=map_w,
                    update_expect_tx=False,
                )
                cute_ext.tma_load(
                    gx[None, None, tile],
                    sx[None, None, None, stage],
                    (full + stage).value,
                    cta_v_map=map_x,
                    update_expect_tx=False,
                )
        elif warp == 5:
            thread_mma = mma.get_slice(0)
            a_desc, b_desc = thread_mma.make_fragment_A(sw), thread_mma.make_fragment_B(
                sx
            )
            mma.set(tcgen05.Field.ACCUMULATE, False)
            for tile in cutlass.range(self.tiles, unroll=1):
                stage = tile % self.stages
                cute.arch.mbarrier_wait(full + stage, tile // self.stages % 2)
                for block in cutlass.range_constexpr(cute.size(a_desc, mode=[2])):
                    for branch in cutlass.range_constexpr(4):
                        fragment = mma.get_slice(0).make_fragment_C(
                            mma.partition_shape_C((64, self.n))
                        )
                        acc = cute.make_tensor(
                            cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, base)
                            + branch * self.n,
                            fragment.layout,
                        )
                        a_branch = cute.local_tile(
                            a_desc[None, None, block, stage], (1, 1), (0, branch)
                        )
                        cute.gemm(
                            mma, acc, a_branch, b_desc[None, None, block, stage], acc
                        )
                    mma.set(tcgen05.Field.ACCUMULATE, True)
                with cute.arch.elect_one():
                    tcgen05.commit(empty + stage, None, tcgen05.CtaGroup.ONE)
            with cute.arch.elect_one():
                tcgen05.commit(done, None, tcgen05.CtaGroup.ONE)
            cute.arch.mbarrier_wait(read_done, 0)
            cute.arch.relinquish_tmem_alloc_permit()
            cute.arch.dealloc_tmem(
                cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, base), self.columns
            )
        else:
            if cutlass.const_expr(self.pdl):
                cute.arch.griddepcontrol_wait()
            acc = self._accumulator(mma, base, 0)
            tile_shape = (64, self.n)
            copy = tcgen05.make_tmem_copy(
                sm100_utils.get_tmem_load_op(
                    (64, self.n, self.k),
                    cute_utils.LayoutEnum.COL_MAJOR,
                    cutlass.Float32,
                    cutlass.Float32,
                    tile_shape,
                    False,
                ),
                acc,
            )
            reader = copy.get_slice(tid)
            dummy = cute.make_tensor(
                original.iterator, cute.make_layout(tile_shape, stride=(1, 64))
            )
            gates = cute_ext.allocate(
                cutlass.Float32,
                cute.AddressSpace.rmem,
                cute_ext.make_t2r_rmem_layout(
                    copy, cute.flat_divide(dummy, tile_shape), tid
                ),
                alignment=32,
            )
            coords = reader.partition_D(cute.make_identity_tensor(tile_shape))
            count = cute.size(gates)
            residual = cute.make_rmem_tensor((count, 4), original.element_type)
            scales = cute.make_rmem_tensor((count, 4), cutlass.Float32)
            residual.fill(0.0)
            scales.fill(1.0)
            for branch in cutlass.range_constexpr(4):
                for i in cutlass.range_constexpr(count):
                    col, row = coords[i]
                    global_row = token_tile * self.n + row
                    global_col = branch * self.hidden_size + hidden_tile * 64 + col
                    if global_row < x.shape[0]:
                        residual[i, branch] = original[global_row, global_col]
                        scales[i, branch] = (
                            weight_scale[global_col if self.per_channel else 0]
                            * activation_scale[global_row]
                        )
            sums = cute.make_rmem_tensor(count, cutlass.Float32)
            sums.fill(0.0)
            cute.arch.mbarrier_wait(done, 0)
            for branch in cutlass.range_constexpr(4):
                acc = self._accumulator(mma, base, branch)
                cute_ext.partition_and_copy(reader, acc, gates)
                for i in cutlass.range_constexpr(count):
                    sums[i] = sums[i] + residual[i, branch].to(
                        cutlass.Float32
                    ) * _up_sigmoid_tanh(gates[i] * scales[i, branch])
            cute.arch.fence_view_async_tmem_load()
            tcgen05_fence("before_thread_sync")
            cute.arch.mbarrier_arrive(read_done)
            for i in cutlass.range_constexpr(count):
                col, row = coords[i]
                global_row = token_tile * self.n + row
                if global_row < x.shape[0]:
                    out[global_row, hidden_tile * 64 + col] = (sums[i] / 4).to(
                        out.element_type
                    )


@cute.experimental.jit
def _pack_residual_halves(a, b):
    return cutlass.Uint32(
        llvm.inline_asm(
            T.i32(),
            [a.to(cutlass.Uint32).ir_value(), b.to(cutlass.Uint32).ir_value()],
            "prmt.b32 $0, $1, $2, 0x5410;",
            "=r,r,r",
            has_side_effects=True,
            asm_dialect=0,
        )
    )


@cute.experimental.jit
def _unpack_residual_half(word, high: cutlass.Constexpr, dtype: cutlass.Constexpr):
    if cutlass.const_expr(dtype == cutlass.BFloat16):
        if cutlass.const_expr(high):
            code = "{ .reg .b32 bits; and.b32 bits, $1, 0xffff0000; mov.b32 $0, bits; }"
        else:
            code = "{ .reg .b32 bits; shl.b32 bits, $1, 16; mov.b32 $0, bits; }"
    else:
        if cutlass.const_expr(high):
            code = "{ .reg .b16 lo, hi; mov.b32 {lo, hi}, $1; cvt.f32.f16 $0, hi; }"
        else:
            code = "{ .reg .b16 lo, hi; mov.b32 {lo, hi}, $1; cvt.f32.f16 $0, lo; }"
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [word.ir_value()],
            code,
            "=f,r",
            has_side_effects=False,
            asm_dialect=0,
        )
    )


class Fp8GatedResidualPairedHiddenUp:
    """Two M128 GEMMs cover four H64 branches with coalesced residual reads."""

    def __init__(
        self,
        *,
        hidden_size: int,
        lowrank: int,
        token_tile: int,
        k_tile: int,
        stages: int,
        use_pdl: bool,
        weights_independent: bool,
        weight_scale_per_channel: bool,
    ):
        self.h = hidden_size
        self.r = lowrank
        self.n = token_tile
        self.k = k_tile
        self.stages = stages
        self.tiles = lowrank // k_tile
        self.pdl = use_pdl
        self.independent = weights_independent
        self.per_channel = weight_scale_per_channel
        self.epi_threads = {32: 256, 64: 512, 96: 384}[token_tile]
        self.groups = self.epi_threads // 128
        self.producer = self.epi_threads // 32
        self.issuer = self.producer + 1
        self.columns = max(32, 1 << (token_tile * 2 - 1).bit_length())
        self.arena_bytes = max(
            (256 + token_tile) * k_tile * stages, 256 * token_tile * 4
        )

    @cute.experimental.jit
    def __call__(
        self, x, weight, original, weight_scale, activation_scale, out, stream: CUstream
    ):
        self.kernel(x, weight, original, weight_scale, activation_scale, out).launch(
            grid=(self.h // 64, cute.ceil_div(x.shape[0], self.n), 1),
            block=(self.epi_threads + 64, 1, 1),
            stream=stream,
            use_pdl=self.pdl,
        )

    @cute.experimental.jit
    def _accumulator(self, mma, base, pair: cutlass.Constexpr):
        fragment = mma.get_slice(0).make_fragment_C(
            mma.partition_shape_C((128, self.n))
        )
        return cute.make_tensor(
            cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, base) + pair * self.n,
            fragment.layout,
        )[((None, None), 0, 0)]

    @cute.experimental.kernel
    def kernel(self, x, weight, original, weight_scale, activation_scale, out):
        tid = cute.arch.thread_idx()[0]
        warp = cute.arch.make_warp_uniform(tid // 32)
        hidden_tile, token_tile, _ = cute.arch.block_idx()
        dtype = cutlass.Float8E4M3FN
        mma = sm100_utils.make_trivial_tiled_mma(
            dtype,
            dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            cutlass.Float32,
            tcgen05.CtaGroup.ONE,
            (128, self.n),
        )
        tiler = (256, self.n, self.k)
        wl = sm100_utils.make_smem_layout_a(mma, tiler, dtype, self.stages)
        xl = sm100_utils.make_smem_layout_b(mma, tiler, dtype, self.stages)
        arena = cute_ext.allocate(
            cutlass.Uint8,
            cute.AddressSpace.smem,
            cute.make_layout(self.arena_bytes),
            alignment=1024,
        )
        sw = cute.make_tensor(
            cute.make_ptr(
                dtype,
                arena.iterator.toint(),
                cute.AddressSpace.smem,
                assumed_align=1024,
                swizzle_=cute.get_swizzle_portion(wl),
            ),
            cute.get_nonswizzle_portion(wl),
        )
        sx = cute.make_tensor(
            cute.make_ptr(
                dtype,
                arena.iterator.toint() + cute.cosize(wl),
                cute.AddressSpace.smem,
                assumed_align=1024,
                swizzle_=cute.get_swizzle_portion(xl),
            ),
            cute.get_nonswizzle_portion(xl),
        )
        gate = cute.make_tensor(
            cute.recast_ptr(arena.iterator, dtype=cutlass.Float32),
            cute.make_layout((256, self.n), stride=(1, 256)),
        )
        scale_cache = cute_ext.allocate(
            cutlass.Float32, cute.AddressSpace.smem, cute.make_layout(256), alignment=32
        )
        alpha_cache = cute_ext.allocate(
            cutlass.Float32,
            cute.AddressSpace.smem,
            cute.make_layout(self.n),
            alignment=32,
        )
        bars = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(2 * self.stages + 2),
            alignment=8,
        ).iterator
        full, empty = bars, bars + self.stages
        done, read_done = empty + self.stages, empty + self.stages + 1
        base = cute_ext.allocate(
            cutlass.Int32, cute.AddressSpace.smem, cute.make_layout(1), alignment=4
        ).iterator
        if warp == self.issuer:
            with cute.arch.elect_one():
                for stage in cutlass.range_constexpr(self.stages):
                    cute.arch.mbarrier_init(full + stage, 1)
                    cute.arch.mbarrier_init(empty + stage, 1)
                cute.arch.mbarrier_init(done, 1)
                cute.arch.mbarrier_init(read_done, self.epi_threads)
            cute.arch.alloc_tmem(self.columns, base, is_two_cta=False)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        w = cute.make_tensor(
            weight.iterator,
            cute.make_layout(
                ((64, 4, self.h // 64), (64, self.r // 64), 1),
                stride=((self.r, self.h * self.r, 64 * self.r), (1, 64), 0),
            ),
        )
        a = cute.make_tensor(
            x.iterator,
            cute.make_layout(
                (x.shape[0], (64, self.r // 64), 1), stride=(self.r, (1, 64), 0)
            ),
        )
        gw = cute.local_tile(w, (256, self.k), (hidden_tile, None, 0))
        gx = cute.local_tile(a, (self.n, self.k), (token_tile, None, 0))
        map_w = cute_ext.get_cta_v_map_ab(w, tiler, mma, "A")
        map_x = cute_ext.get_cta_v_map_ab(a, tiler, mma, "B")
        if warp == self.producer:
            if cutlass.const_expr(not self.pdl):
                # Make stage0 usable before queuing later cold weight tiles.
                for stage in cutlass.range_constexpr(self.stages):
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            full + stage, (256 + self.n) * self.k
                        )
                    cute_ext.tma_load(
                        gw[None, None, stage],
                        sw[None, None, None, stage],
                        (full + stage).value,
                        cta_v_map=map_w,
                        update_expect_tx=False,
                    )
                    cute_ext.tma_load(
                        gx[None, None, stage],
                        sx[None, None, None, stage],
                        (full + stage).value,
                        cta_v_map=map_x,
                        update_expect_tx=False,
                    )
            else:
                for stage in cutlass.range_constexpr(self.stages):
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            full + stage, (256 + self.n) * self.k
                        )
                    if cutlass.const_expr(self.independent):
                        cute_ext.tma_load(
                            gw[None, None, stage],
                            sw[None, None, None, stage],
                            (full + stage).value,
                            cta_v_map=map_w,
                            update_expect_tx=False,
                        )
                if cutlass.const_expr(self.pdl):
                    cute.arch.griddepcontrol_wait()
                    cute.arch.griddepcontrol_launch_dependents()
                for stage in cutlass.range_constexpr(self.stages):
                    if cutlass.const_expr(not self.independent):
                        cute_ext.tma_load(
                            gw[None, None, stage],
                            sw[None, None, None, stage],
                            (full + stage).value,
                            cta_v_map=map_w,
                            update_expect_tx=False,
                        )
                    cute_ext.tma_load(
                        gx[None, None, stage],
                        sx[None, None, None, stage],
                        (full + stage).value,
                        cta_v_map=map_x,
                        update_expect_tx=False,
                    )
            for tile in cutlass.range(self.stages, self.tiles, unroll=1):
                stage = tile % self.stages
                cute.arch.mbarrier_wait(empty + stage, (tile // self.stages - 1) % 2)
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        full + stage, (256 + self.n) * self.k
                    )
                cute_ext.tma_load(
                    gw[None, None, tile],
                    sw[None, None, None, stage],
                    (full + stage).value,
                    cta_v_map=map_w,
                    update_expect_tx=False,
                )
                cute_ext.tma_load(
                    gx[None, None, tile],
                    sx[None, None, None, stage],
                    (full + stage).value,
                    cta_v_map=map_x,
                    update_expect_tx=False,
                )
        elif warp == self.issuer:
            thread_mma = mma.get_slice(0)
            ad = thread_mma.make_fragment_A(sw)
            bd = thread_mma.make_fragment_B(sx)
            mma.set(tcgen05.Field.ACCUMULATE, False)
            for tile in cutlass.range(self.tiles, unroll=1):
                stage = tile % self.stages
                cute.arch.mbarrier_wait(full + stage, tile // self.stages % 2)
                for block in cutlass.range_constexpr(cute.size(ad, mode=[2])):
                    for pair in cutlass.range_constexpr(2):
                        fragment = thread_mma.make_fragment_C(
                            mma.partition_shape_C((128, self.n))
                        )
                        acc = cute.make_tensor(
                            cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, base)
                            + pair * self.n,
                            fragment.layout,
                        )
                        a_pair = cute.local_tile(
                            ad[None, None, block, stage], (1, 1), (0, pair)
                        )
                        cute.gemm(mma, acc, a_pair, bd[None, None, block, stage], acc)
                    mma.set(tcgen05.Field.ACCUMULATE, True)
                with cute.arch.elect_one():
                    tcgen05.commit(empty + stage, None, tcgen05.CtaGroup.ONE)
            with cute.arch.elect_one():
                tcgen05.commit(done, None, tcgen05.CtaGroup.ONE)
            cute.arch.mbarrier_wait(read_done, 0)
            cute.arch.relinquish_tmem_alloc_permit()
            cute.arch.dealloc_tmem(
                cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, base), self.columns
            )
        elif tid < self.epi_threads:
            if cutlass.const_expr(self.pdl):
                cute.arch.griddepcontrol_wait()
            if tid < 256:
                channel = (tid // 64) * self.h + hidden_tile * 64 + tid % 64
                scale_cache[tid] = weight_scale[channel if self.per_channel else 0]
            if tid < self.n:
                alpha_cache[tid] = 1.0
                if token_tile * self.n + tid < x.shape[0]:
                    alpha_cache[tid] = activation_scale[token_tile * self.n + tid]
            _epilogue_barrier(self.epi_threads, cutlass.Int32(3))
            tile_shape = (128, self.n // self.groups)
            group = tid // 128
            acc = cute.local_tile(
                self._accumulator(mma, base, 0), tile_shape, (0, group)
            )
            coords = cute.local_tile(
                cute.make_identity_tensor((128, self.n)), tile_shape, (0, group)
            )
            copy = tcgen05.make_tmem_copy(
                sm100_utils.get_tmem_load_op(
                    (128, self.n, self.k),
                    cute_utils.LayoutEnum.COL_MAJOR,
                    cutlass.Float32,
                    cutlass.Float32,
                    tile_shape,
                    False,
                ),
                acc,
            )
            reader = copy.get_slice(tid % 128)
            dummy = cute.make_tensor(
                original.iterator, cute.make_layout(tile_shape, stride=(1, 128))
            )
            gates = cute_ext.allocate(
                cutlass.Float32,
                cute.AddressSpace.rmem,
                cute_ext.make_t2r_rmem_layout(
                    copy, cute.flat_divide(dummy, tile_shape), tid % 128
                ),
                alignment=32,
            )
            coordinates = reader.partition_D(coords)
            count = cute.size(gates)
            residual_words = cute.make_rmem_tensor((count // 2, 2), cutlass.Uint32)
            residual_words.fill(0)
            original_bits = cute.recast_tensor(original, cutlass.Uint16)
            for pair in cutlass.range_constexpr(2):
                for packet in cutlass.range_constexpr(count // 2):
                    col0, row0 = coordinates[packet * 2]
                    col1, row1 = coordinates[packet * 2 + 1]
                    global_row0 = token_tile * self.n + row0
                    global_row1 = token_tile * self.n + row1
                    global_col0 = (
                        (pair * 2 + col0 // 64) * self.h + hidden_tile * 64 + col0 % 64
                    )
                    global_col1 = (
                        (pair * 2 + col1 // 64) * self.h + hidden_tile * 64 + col1 % 64
                    )
                    bits0 = cutlass.Uint16(0)
                    bits1 = cutlass.Uint16(0)
                    if global_row0 < x.shape[0]:
                        bits0 = original_bits[global_row0, global_col0]
                    if global_row1 < x.shape[0]:
                        bits1 = original_bits[global_row1, global_col1]
                    residual_words[packet, pair] = _pack_residual_halves(bits0, bits1)
            cute.arch.mbarrier_wait(done, 0)
            for pair in cutlass.range_constexpr(2):
                acc_pair = cute.local_tile(
                    self._accumulator(mma, base, pair), tile_shape, (0, group)
                )
                cute_ext.partition_and_copy(reader, acc_pair, gates)
                cute.arch.fence_view_async_tmem_load()
                for i in cutlass.range_constexpr(count):
                    col, row = coordinates[i]
                    if token_tile * self.n + row < x.shape[0]:
                        factor = scale_cache[pair * 128 + col] * alpha_cache[row]
                        gate[pair * 128 + col, row] = _unpack_residual_half(
                            residual_words[i // 2, pair],
                            i % 2 != 0,
                            original.element_type,
                        ) * _up_sigmoid_tanh(gates[i] * factor)
            tcgen05_fence("before_thread_sync")
            cute.arch.mbarrier_arrive(read_done)
            _epilogue_barrier(self.epi_threads, cutlass.Int32(2))
            for offset in cutlass.range_constexpr(64 * self.n // self.epi_threads):
                index = tid + offset * self.epi_threads
                row, col = index // 64, index % 64
                if token_tile * self.n + row < x.shape[0]:
                    value = cutlass.Float32(0.0)
                    for branch in cutlass.range_constexpr(4):
                        value = value + gate[branch * 64 + col, row]
                    out[token_tile * self.n + row, hidden_tile * 64 + col] = (
                        value / 4.0
                    ).to(out.element_type)


@cute.experimental.jit
def _quantize_four_packed(word0, word1, multiplier, dtype: cutlass.Constexpr):
    if cutlass.const_expr(dtype == cutlass.BFloat16):
        instruction: cutlass.Constexpr = """{
            .reg .b16 h0,h1,h2,h3,p01,p23;
            .reg .f32 f0,f1,f2,f3;
            mov.b32 {h0,h1},$1;
            mov.b32 {h2,h3},$2;
            cvt.f32.bf16 f0,h0;
            cvt.f32.bf16 f1,h1;
            cvt.f32.bf16 f2,h2;
            cvt.f32.bf16 f3,h3;
            mul.rn.f32 f0,f0,$3;
            mul.rn.f32 f1,f1,$3;
            mul.rn.f32 f2,f2,$3;
            mul.rn.f32 f3,f3,$3;
            cvt.rn.satfinite.e4m3x2.f32 p01,f1,f0;
            cvt.rn.satfinite.e4m3x2.f32 p23,f3,f2;
            mov.b32 $0,{p01,p23};
        }"""
    else:
        instruction: cutlass.Constexpr = """{
            .reg .b16 h0,h1,h2,h3,p01,p23;
            .reg .f32 f0,f1,f2,f3;
            mov.b32 {h0,h1},$1;
            mov.b32 {h2,h3},$2;
            cvt.f32.f16 f0,h0;
            cvt.f32.f16 f1,h1;
            cvt.f32.f16 f2,h2;
            cvt.f32.f16 f3,h3;
            mul.rn.f32 f0,f0,$3;
            mul.rn.f32 f1,f1,$3;
            mul.rn.f32 f2,f2,$3;
            mul.rn.f32 f3,f3,$3;
            cvt.rn.satfinite.e4m3x2.f32 p01,f1,f0;
            cvt.rn.satfinite.e4m3x2.f32 p23,f3,f2;
            mov.b32 $0,{p01,p23};
        }"""
    return cutlass.Uint32(
        llvm.inline_asm(
            T.i32(),
            [word0.ir_value(), word1.ir_value(), multiplier.ir_value()],
            instruction,
            "=r,r,r,f",
            has_side_effects=False,
            asm_dialect=0,
        )
    )


class Fp8GatedResidualInputQuantize:
    """Keep packed BF16/FP16 inputs until the row scale is available."""

    def __init__(self, *, columns: int, block_threads: int, use_pdl: bool):
        if columns % (block_threads * 8):
            raise ValueError("quantization packets must cover complete rows")
        self.columns = columns
        self.threads = block_threads
        self.packets = columns // (block_threads * 8)
        self.pdl = use_pdl

    @cute.experimental.jit
    def __call__(self, x, codes, scales, stream: CUstream):
        self.kernel(x, codes, scales).launch(
            grid=(x.shape[0], 1, 1),
            block=(self.threads, 1, 1),
            stream=stream,
            use_pdl=self.pdl,
        )

    @cute.experimental.kernel
    def kernel(self, x, codes, scales):
        row = cute.arch.block_idx()[0]
        tid = cute.arch.thread_idx()[0]
        warp, lane = tid // 32, tid % 32
        packed = cute.make_rmem_tensor((4, self.packets), cutlass.Uint32)
        maximum_bits = cutlass.Uint32(0)
        shared = cute_ext.allocate(
            cutlass.Uint32,
            cute.AddressSpace.smem,
            cute.make_layout(self.threads // 32 + 1),
            alignment=32,
        )
        if cutlass.const_expr(self.pdl):
            cute.arch.griddepcontrol_wait()
            # Down can preload independent weights while this CTA quantizes input.
            if tid == 0:
                cute.arch.griddepcontrol_launch_dependents()
        for packet in cutlass.range_constexpr(self.packets):
            col = (tid + packet * self.threads) * 8
            source = cute.make_tensor(
                cute.make_ptr(
                    cutlass.Uint32,
                    (x.iterator + row * self.columns + col).toint(),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                ),
                cute.make_layout(4),
            )
            cute.copy(
                cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(), cutlass.Uint32, num_bits_per_copy=128
                ),
                source,
                packed[None, packet],
            )
            for word in cutlass.range_constexpr(4):
                absolute = packed[word, packet].to(cutlass.Uint32) & cutlass.Uint32(
                    0x7FFF7FFF
                )
                maximum_bits = cute.math.max(
                    maximum_bits,
                    cute.math.max(
                        absolute & cutlass.Uint32(0xFFFF),
                        absolute >> cutlass.Uint32(16),
                    ),
                )
        maximum_bits = cute.arch.warp_reduction_max(
            maximum_bits, threads_in_group=32
        ).to(cutlass.Uint32)
        if lane == 0:
            shared[warp] = maximum_bits
        cute.arch.sync_threads()
        if warp == 0:
            maximum_bits = cutlass.Uint32(0)
            if lane < self.threads // 32:
                maximum_bits = shared[lane].to(cutlass.Uint32)
            maximum_bits = cute.arch.warp_reduction_max(
                maximum_bits, threads_in_group=32
            ).to(cutlass.Uint32)
            if lane == 0:
                half = cute.make_rmem_tensor(1, x.element_type)
                cute.recast_tensor(half, cutlass.Uint16)[0] = maximum_bits.to(
                    cutlass.Uint16
                )
                maximum = half[0].to(cutlass.Float32)
                # The scale and multiplier have independent FP32 RN rounding.
                scales[row] = cute.math.max(
                    maximum / cutlass.Float32(448.0),
                    cutlass.Float32(1.0 / (448.0 * 512.0)),
                    propagate_nan=True,
                )
                multiplier = cute.math.min(
                    cutlass.Float32(448.0) / maximum,
                    cutlass.Float32(448.0 * 512.0),
                    propagate_nan=True,
                )
                cute.recast_tensor(shared, cutlass.Float32)[
                    self.threads // 32
                ] = multiplier
        cute.arch.sync_threads()
        multiplier = cute.recast_tensor(shared, cutlass.Float32)[self.threads // 32]
        for packet in cutlass.range_constexpr(self.packets):
            quantized = cute.make_rmem_tensor(2, cutlass.Uint32)
            for pair in cutlass.range_constexpr(2):
                quantized[pair] = _quantize_four_packed(
                    packed[pair * 2, packet],
                    packed[pair * 2 + 1, packet],
                    multiplier,
                    x.element_type,
                )
            col = (tid + packet * self.threads) * 8
            target = cute.make_tensor(
                cute.make_ptr(
                    cutlass.Uint32,
                    (codes.iterator + row * self.columns + col).toint(),
                    cute.AddressSpace.gmem,
                    assumed_align=8,
                ),
                cute.make_layout(2),
            )
            cute.copy(
                cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(), cutlass.Uint32, num_bits_per_copy=64
                ),
                quantized,
                target,
            )
        if cutlass.const_expr(self.pdl):
            cute.arch.sync_threads()


class W8GatedResidualIndependentUp(Fp8GatedResidualGemm):
    """Six decoded weight slots overlap with four independent input slots."""

    def __init__(
        self,
        *,
        hidden_size: int,
        lowrank: int,
        weight_stages: int,
        input_stages: int,
        use_pdl: bool,
        weights_independent: bool,
        weight_scale_per_channel: bool,
        weight_scale_vectorized: bool,
    ):
        super().__init__(
            role="mix",
            hc_count=4,
            hidden_size=hidden_size,
            lowrank=lowrank,
            projection_rows=lowrank + 4,
            token_tile=64,
            projection_tile=128,
            k_tile=64,
            stages=weight_stages,
            split_k=1,
            use_pdl=use_pdl,
            weights_independent=weights_independent,
            weight_scale_per_channel=weight_scale_per_channel,
            token_major_up=True,
            weight_scale_vectorized=weight_scale_vectorized,
            up_hidden_tile=64,
            up_token_rows=64,
            separate_operand_ready=False,
            epilogue_threads=512,
            shared_up_scales=False,
            interleave_tma=False,
        )
        self.input_stages = input_stages

    @cute.experimental.jit
    def __call__(
        self, x, weight, original, weight_scale, activation_scale, out, stream: CUstream
    ):
        self.up_kernel(x, weight, original, weight_scale, activation_scale, out).launch(
            grid=(
                self.hidden_size // self.up_hidden_tile,
                cute.ceil_div(x.shape[0], self.up_token_rows),
                1,
            ),
            block=(self.epilogue_threads + 64, 1, 1),
            stream=stream,
            use_pdl=self.pdl,
        )

    @cute.experimental.kernel
    def up_kernel(self, x, weight, original, weight_scale, activation_scale, out):
        tid = cute.arch.thread_idx()[0]
        warp = cute.arch.make_warp_uniform(tid // 32)
        hidden_tile, token_tile, _ = cute.arch.block_idx()
        dtype = x.element_type
        native_fp8 = cutlass.const_expr(dtype == cutlass.Float8E4M3FN)
        mma = sm100_utils.make_trivial_tiled_mma(
            dtype,
            dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            cutlass.Float32,
            tcgen05.CtaGroup.ONE,
            (self.up_token_rows, self.up_hidden_tile * self.hc_count),
        )
        tiler = (self.up_token_rows, self.up_hidden_tile * self.hc_count, self.k)
        activation_layout = sm100_utils.make_smem_layout_a(
            mma, tiler, dtype, self.input_stages
        )
        raw_layout = sm100_utils.make_smem_layout_b(
            mma, tiler, cutlass.Float8E4M3FN, self.stages
        )
        weight_layout = sm100_utils.make_smem_layout_b(mma, tiler, dtype, self.stages)
        sx = cute_ext.allocate(
            dtype, cute.AddressSpace.smem, activation_layout, alignment=1024
        )
        if cutlass.const_expr(native_fp8):
            raw = cute_ext.allocate(
                cutlass.Float8E4M3FN, cute.AddressSpace.smem, raw_layout, alignment=1024
            )
            sw = raw
        else:
            sw = cute_ext.allocate(
                dtype, cute.AddressSpace.smem, weight_layout, alignment=1024
            )
            raw_base = cute.get_nonswizzle_portion(raw_layout)
            weight_base = cute.get_nonswizzle_portion(weight_layout)
            raw = cute.make_tensor(
                cute.make_ptr(
                    cutlass.Float8E4M3FN,
                    sw.iterator.toint(),
                    cute.AddressSpace.smem,
                    assumed_align=1024,
                    swizzle_=cute.get_swizzle_portion(raw_layout),
                ),
                cute.make_layout(
                    raw_base.shape,
                    stride=(
                        raw_base.stride[0],
                        raw_base.stride[1],
                        raw_base.stride[2],
                        weight_base.stride[3] * (dtype.width // 8),
                    ),
                ),
            )
        barriers = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(3 * self.stages + 2 * self.input_stages + 2),
            alignment=8,
        ).iterator
        full = barriers
        converted = full + self.stages
        empty = converted + self.stages
        x_full = empty + self.stages
        x_empty = x_full + self.input_stages
        done, read_done = (x_empty + self.input_stages, x_empty + self.input_stages + 1)
        base = cute_ext.allocate(
            cutlass.Int32, cute.AddressSpace.smem, cute.make_layout(1), alignment=4
        ).iterator
        if warp == self.mma_warp:
            with cute.arch.elect_one():
                for stage in cutlass.range_constexpr(self.stages):
                    cute.arch.mbarrier_init(full + stage, 1)
                    cute.arch.mbarrier_init(
                        converted + stage, self.epilogue_threads // 32
                    )
                    cute.arch.mbarrier_init(empty + stage, 1)
                for stage in cutlass.range_constexpr(self.input_stages):
                    cute.arch.mbarrier_init(x_full + stage, 1)
                    cute.arch.mbarrier_init(x_empty + stage, 1)
                cute.arch.mbarrier_init(done, 1)
                cute.arch.mbarrier_init(read_done, self.epilogue_threads)
            cute.arch.alloc_tmem(
                self.up_hidden_tile * self.hc_count, base, is_two_cta=False
            )
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        activation = cute.make_tensor(
            x.iterator,
            cute.make_layout(
                (x.shape[0], (64, self.inner // 64), 1), stride=(self.inner, (1, 64), 0)
            ),
        )
        weights = cute.make_tensor(
            weight.iterator,
            cute.make_layout(
                (
                    (
                        self.up_hidden_tile,
                        self.hc_count,
                        self.hidden_size // self.up_hidden_tile,
                    ),
                    (64, self.inner // 64),
                    1,
                ),
                stride=(
                    (
                        self.inner,
                        self.hidden_size * self.inner,
                        self.up_hidden_tile * self.inner,
                    ),
                    (1, 64),
                    0,
                ),
            ),
        )
        gx = cute.local_tile(
            activation, (self.up_token_rows, self.k), (token_tile, None, 0)
        )
        gw = cute.local_tile(
            weights,
            (self.up_hidden_tile * self.hc_count, self.k),
            (hidden_tile, None, 0),
        )
        map_x = cute_ext.get_cta_v_map_ab(activation, tiler, mma, "A")
        map_w = cute_ext.get_cta_v_map_ab(weights, tiler, mma, "B")
        if warp == self.producer_warp:
            for stage in cutlass.range_constexpr(self.stages):
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        full + stage, self.up_hidden_tile * self.hc_count * self.k
                    )
                if cutlass.const_expr(self.weights_independent):
                    cute_ext.tma_load(
                        gw[None, None, stage],
                        raw[None, None, None, stage],
                        (full + stage).value,
                        cta_v_map=map_w,
                        update_expect_tx=False,
                    )
            if cutlass.const_expr(self.pdl):
                cute.arch.griddepcontrol_wait()
                cute.arch.griddepcontrol_launch_dependents()
            if cutlass.const_expr(not self.weights_independent):
                for stage in cutlass.range_constexpr(self.stages):
                    cute_ext.tma_load(
                        gw[None, None, stage],
                        raw[None, None, None, stage],
                        (full + stage).value,
                        cta_v_map=map_w,
                        update_expect_tx=False,
                    )
            for stage in cutlass.range_constexpr(self.input_stages):
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        x_full + stage, self.up_token_rows * self.k * 2
                    )
                cute_ext.tma_load(
                    gx[None, None, stage],
                    sx[None, None, None, stage],
                    (x_full + stage).value,
                    cta_v_map=map_x,
                    update_expect_tx=False,
                )
            for tile in cutlass.range(self.input_stages, self.tiles, unroll=1):
                input_stage = tile % self.input_stages
                cute.arch.mbarrier_wait(
                    x_empty + input_stage, (tile // self.input_stages - 1) % 2
                )
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        x_full + input_stage, self.up_token_rows * self.k * 2
                    )
                cute_ext.tma_load(
                    gx[None, None, tile],
                    sx[None, None, None, input_stage],
                    (x_full + input_stage).value,
                    cta_v_map=map_x,
                    update_expect_tx=False,
                )
                future = tile + self.stages - self.input_stages
                if future < self.tiles:
                    stage = future % self.stages
                    cute.arch.mbarrier_wait(
                        empty + stage, (future // self.stages - 1) % 2
                    )
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            full + stage, self.up_hidden_tile * self.hc_count * self.k
                        )
                    cute_ext.tma_load(
                        gw[None, None, future],
                        raw[None, None, None, stage],
                        (full + stage).value,
                        cta_v_map=map_w,
                        update_expect_tx=False,
                    )
        elif warp == self.mma_warp:
            thread_mma = mma.get_slice(0)
            a_desc = thread_mma.make_fragment_A(sx)
            b_desc = thread_mma.make_fragment_B(sw)
            fragment = thread_mma.make_fragment_C(
                mma.partition_shape_C(
                    (self.up_token_rows, self.up_hidden_tile * self.hc_count)
                )
            )
            accumulator = cute.make_tensor(
                cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, base), fragment.layout
            )
            mma.set(tcgen05.Field.ACCUMULATE, False)
            for tile in cutlass.range(self.tiles, unroll=1):
                stage = tile % self.stages
                input_stage = tile % self.input_stages
                ready = full if cutlass.const_expr(native_fp8) else converted
                cute.arch.mbarrier_wait(ready + stage, tile // self.stages % 2)
                cute.arch.mbarrier_wait(
                    x_full + input_stage, tile // self.input_stages % 2
                )
                for block in cutlass.range_constexpr(cute.size(a_desc, mode=[2])):
                    cute.gemm(
                        mma,
                        accumulator,
                        a_desc[None, None, block, input_stage],
                        b_desc[None, None, block, stage],
                        accumulator,
                    )
                    mma.set(tcgen05.Field.ACCUMULATE, True)
                with cute.arch.elect_one():
                    tcgen05.commit(empty + stage, None, tcgen05.CtaGroup.ONE)
                    tcgen05.commit(x_empty + input_stage, None, tcgen05.CtaGroup.ONE)
            with cute.arch.elect_one():
                tcgen05.commit(done, None, tcgen05.CtaGroup.ONE)
            cute.arch.mbarrier_wait(read_done, 0)
            cute.arch.relinquish_tmem_alloc_permit()
            cute.arch.dealloc_tmem(
                cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, base),
                self.up_hidden_tile * self.hc_count,
            )
        elif tid < self.epilogue_threads:
            if cutlass.const_expr(not native_fp8):
                for tile in cutlass.range(self.tiles, unroll=1):
                    stage = tile % self.stages
                    cute.arch.mbarrier_wait(full + stage, tile // self.stages % 2)
                    source, target = (
                        raw[None, None, None, stage],
                        sw[None, None, None, stage],
                    )
                    packets = (
                        self.up_hidden_tile
                        * self.hc_count
                        * self.k
                        // (self.epilogue_threads * 8)
                    )
                    raw_values = cute.make_rmem_tensor(
                        (8, packets), cutlass.Float8E4M3FN
                    )
                    for offset in cutlass.range_constexpr(packets):
                        linear = (tid + offset * self.epilogue_threads) * 8
                        index = (
                            linear // self.k
                            + linear % self.k * self.up_hidden_tile * self.hc_count
                        )
                        src = cute.make_tensor(
                            cute.make_ptr(
                                cutlass.Float8E4M3FN,
                                (source.iterator + source.layout(index)).toint(),
                                cute.AddressSpace.smem,
                                assumed_align=8,
                                swizzle_=cute.get_swizzle_portion(raw_layout),
                            ),
                            cute.make_layout(8),
                        )
                        cute.copy(
                            cute.make_copy_atom(
                                cute.nvgpu.CopyUniversalOp(),
                                cutlass.Float8E4M3FN,
                                num_bits_per_copy=64,
                            ),
                            src,
                            raw_values[None, offset],
                        )
                    _epilogue_barrier(self.epilogue_threads, cutlass.Int32(1))
                    for offset in cutlass.range_constexpr(packets):
                        linear = (tid + offset * self.epilogue_threads) * 8
                        index = (
                            linear // self.k
                            + linear % self.k * self.up_hidden_tile * self.hc_count
                        )
                        dst = cute.make_tensor(
                            cute.make_ptr(
                                dtype,
                                (target.iterator + target.layout(index)).toint(),
                                cute.AddressSpace.smem,
                                assumed_align=16,
                                swizzle_=cute.get_swizzle_portion(weight_layout),
                            ),
                            cute.make_layout(8),
                        )
                        converted_values = cute.make_rmem_tensor(8, dtype)
                        _convert_fp8_weight(raw_values[None, offset], converted_values)
                        cute.copy(
                            cute.make_copy_atom(
                                cute.nvgpu.CopyUniversalOp(),
                                dtype,
                                num_bits_per_copy=128,
                            ),
                            converted_values,
                            dst,
                        )
                    cute.arch.fence_view_async_shared()
                    cute.arch.sync_warp(mask=4294967295)
                    if tid % 32 == 0:
                        cute.arch.mbarrier_arrive(converted + stage)
            if cutlass.const_expr(self.pdl):
                cute.arch.griddepcontrol_wait()
            thread_mma = mma.get_slice(0)
            fragment = thread_mma.make_fragment_C(
                mma.partition_shape_C(
                    (self.up_token_rows, self.up_hidden_tile * self.hc_count)
                )
            )
            _, _, template, coordinates = self._up_reader(
                base, fragment.layout, original, tid, 0
            )
            count = cute.size(template)
            ring_slots = (
                self.hc_count
                if self.up_token_rows == 64
                else 2 if self.up_hidden_tile == 64 else self.hc_count
            )
            residual = cute.make_rmem_tensor((count, ring_slots), original.element_type)
            scales = cute.make_rmem_tensor(
                (1, 1) if self.up_token_rows == 64 else (count, ring_slots),
                cutlass.Float32,
            )
            residual.fill(0.0)
            scales.fill(1.0)
            sums = cute.make_rmem_tensor(count, cutlass.Float32)
            sums.fill(0.0)
            first_row, first_col = coordinates[0]
            global_row = token_tile * self.up_token_rows + first_row
            hidden = hidden_tile * self.up_hidden_tile + first_col
            if cutlass.const_expr(self.up_token_rows == 64):
                assert not native_fp8
                scale_cache = cute_ext.allocate(
                    cutlass.Float32,
                    cute.AddressSpace.smem,
                    cute.make_layout(256),
                    alignment=32,
                )
                if tid < 256:
                    channel = tid // 64 * self.hidden_size + hidden_tile * 64 + tid % 64
                    scale_cache[tid] = weight_scale[
                        channel if self.weight_scale_per_channel else 0
                    ]
                _epilogue_barrier(self.epilogue_threads, cutlass.Int32(3))
                for branch in cutlass.range_constexpr(4):
                    self._up_pair_prefetch(
                        original,
                        residual,
                        coordinates,
                        token_tile,
                        hidden_tile,
                        x.shape[0],
                        branch,
                    )
            else:
                for branch in cutlass.range_constexpr(ring_slots):
                    self._up_prefetch(
                        original,
                        weight_scale,
                        activation_scale,
                        residual,
                        scales,
                        global_row,
                        hidden,
                        x.shape[0],
                        branch,
                        branch,
                        native_fp8,
                    )
            cute.arch.mbarrier_wait(done, 0)
            if (
                tid % 128 // 32 * (self.up_token_rows // 4)
                < x.shape[0] - token_tile * self.up_token_rows
            ):
                for branch in cutlass.range_constexpr(self.hc_count):
                    slot = branch % ring_slots
                    accumulator, reader, gates, _ = self._up_reader(
                        base, fragment.layout, original, tid, branch
                    )
                    cute_ext.partition_and_copy(reader, accumulator, gates)
                    cute.arch.fence_view_async_tmem_load()
                    for i in cutlass.range_constexpr(count):
                        factor = cutlass.Float32(1.0)
                        if cutlass.const_expr(self.up_token_rows == 64):
                            _, col = coordinates[i]
                            factor = scale_cache[branch * 64 + col]
                        else:
                            factor = scales[i, slot]
                        sums[i] = sums[i] + residual[i, slot].to(
                            cutlass.Float32
                        ) * _up_sigmoid_tanh(gates[i] * factor)
                    if cutlass.const_expr(branch + ring_slots < self.hc_count):
                        self._up_prefetch(
                            original,
                            weight_scale,
                            activation_scale,
                            residual,
                            scales,
                            global_row,
                            hidden,
                            x.shape[0],
                            branch + ring_slots,
                            slot,
                            native_fp8,
                        )
            tcgen05_fence("before_thread_sync")
            cute.arch.mbarrier_arrive(read_done)
            if cutlass.const_expr(self.up_token_rows == 64):
                for packet in cutlass.range_constexpr(count // 2):
                    row, col = coordinates[packet * 2]
                    output_row = token_tile * self.up_token_rows + row
                    if output_row < x.shape[0]:
                        packed = cute.make_rmem_tensor(2, out.element_type)
                        for j in cutlass.range_constexpr(2):
                            packed[j] = (sums[packet * 2 + j] / self.hc_count).to(
                                out.element_type
                            )
                        target = cute.make_tensor(
                            cute.make_ptr(
                                out.element_type,
                                (
                                    out.iterator
                                    + output_row * self.hidden_size
                                    + hidden_tile * self.up_hidden_tile
                                    + col
                                ).toint(),
                                cute.AddressSpace.gmem,
                                assumed_align=4,
                            ),
                            cute.make_layout(2),
                        )
                        cute.copy(
                            cute.make_copy_atom(
                                cute.nvgpu.CopyUniversalOp(),
                                out.element_type,
                                num_bits_per_copy=32,
                            ),
                            packed,
                            target,
                        )
            elif global_row < x.shape[0]:
                packed = cute.make_rmem_tensor(count, out.element_type)
                for i in cutlass.range_constexpr(count):
                    packed[i] = (sums[i] / self.hc_count).to(out.element_type)
                target = cute.make_tensor(
                    cute.make_ptr(
                        out.element_type,
                        (out.iterator + global_row * self.hidden_size + hidden).toint(),
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    ),
                    cute.make_layout(count),
                )
                cute.copy(
                    cute.make_copy_atom(
                        cute.nvgpu.CopyUniversalOp(),
                        out.element_type,
                        num_bits_per_copy=128,
                    ),
                    packed,
                    target,
                )


class W8GatedResidualTmemProjection(Fp8GatedResidualGemm):
    """Keep decoded projection weights in TMEM and compressed weights in shared memory."""

    def __init__(
        self,
        *,
        role: str,
        hc_count: int,
        hidden_size: int,
        lowrank: int,
        projection_rows: int,
        token_tile: int,
        projection_tile: int,
        k_tile: int,
        stages: int,
        split_k: int,
        use_pdl: bool,
        weights_independent: bool,
        weight_scale_per_channel: bool,
        token_major_up: bool,
        weight_scale_vectorized: bool,
        up_hidden_tile: int,
        up_token_rows: int,
        separate_operand_ready: bool,
        raw_stages: int,
        input_stages: int,
        epilogue_threads: int,
        shared_up_scales: bool,
        interleave_tma: bool,
    ):
        super().__init__(
            role=role,
            hc_count=hc_count,
            hidden_size=hidden_size,
            lowrank=lowrank,
            projection_rows=projection_rows,
            token_tile=token_tile,
            projection_tile=projection_tile,
            k_tile=k_tile,
            stages=stages,
            split_k=split_k,
            use_pdl=use_pdl,
            weights_independent=weights_independent,
            weight_scale_per_channel=weight_scale_per_channel,
            token_major_up=token_major_up,
            weight_scale_vectorized=weight_scale_vectorized,
            up_hidden_tile=up_hidden_tile,
            up_token_rows=up_token_rows,
            separate_operand_ready=separate_operand_ready,
            epilogue_threads=epilogue_threads,
            shared_up_scales=shared_up_scales,
            interleave_tma=interleave_tma,
        )
        assert (
            self.role == "projection"
            and self.m == 128
            and (self.n in (16, 32, 64, 128, 256))
            and (self.k == 128)
        )
        self.raw_stages = raw_stages
        self.input_stages = input_stages
        self.conversion_threads = 128
        self.input_producer_warp = self.producer_warp + 1
        self.mma_warp = self.producer_warp + 2
        self.tmem_columns = 512
        assert self.columns + self.stages * (self.k // 2) <= self.tmem_columns

    @cute.experimental.jit
    def __call__(
        self,
        x,
        weight,
        original,
        weight_scale,
        activation_scale,
        partials,
        out,
        stream: CUstream,
    ):
        if cutlass.const_expr(self.role == "mix" and self.token_major_up):
            self.up_kernel(
                x, weight, original, weight_scale, activation_scale, out
            ).launch(
                grid=(
                    self.hidden_size // self.up_hidden_tile,
                    cute.ceil_div(x.shape[0], self.up_token_rows),
                    1,
                ),
                block=(self.epilogue_threads + 96, 1, 1),
                stream=stream,
                use_pdl=self.pdl,
            )
        else:
            self.kernel(
                x, weight, original, weight_scale, activation_scale, partials, out
            ).launch(
                grid=(
                    (
                        (self.projection_rows + self.m - 1) // self.m
                        if self.role == "projection"
                        else self.hidden_size // 32
                    ),
                    cute.ceil_div(x.shape[0], self.n),
                    self.split_k,
                ),
                block=(self.epilogue_threads + 96, 1, 1),
                stream=stream,
                use_pdl=self.pdl,
            )

    @cute.experimental.kernel
    def kernel(
        self, x, weight, original, weight_scale, activation_scale, partials, out
    ):
        tid = cute.arch.thread_idx()[0]
        warp = cute.arch.make_warp_uniform(tid // 32)
        output_tile, token_tile, split = cute.arch.block_idx()
        dtype = x.element_type
        native_fp8 = cutlass.const_expr(dtype == cutlass.Float8E4M3FN)
        mma = sm100_utils.make_trivial_tiled_mma(
            dtype,
            dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            cutlass.Float32,
            tcgen05.CtaGroup.ONE,
            (self.m, self.n),
            tcgen05.OperandSource.TMEM,
        )
        tiler = (self.m, self.n, self.k)
        weight_layout = sm100_utils.make_smem_layout_a(mma, tiler, dtype, self.stages)
        raw_layout = sm100_utils.make_smem_layout_a(
            mma, tiler, cutlass.Float8E4M3FN, self.raw_stages
        )
        x_layout = sm100_utils.make_smem_layout_b(mma, tiler, dtype, self.input_stages)
        assert not native_fp8
        raw = cute_ext.allocate(
            cutlass.Float8E4M3FN, cute.AddressSpace.smem, raw_layout, alignment=1024
        )
        sx = cute_ext.allocate(dtype, cute.AddressSpace.smem, x_layout, alignment=1024)
        barriers = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(
                2 * self.raw_stages + 2 * self.stages + 2 * self.input_stages + 2
            ),
            alignment=8,
        ).iterator
        full = barriers
        raw_empty = full + self.raw_stages
        converted = raw_empty + self.raw_stages
        empty = converted + self.stages
        input_ready = empty + self.stages
        input_empty = input_ready + self.input_stages
        done = input_empty + self.input_stages
        read_done = done + 1
        tmem_base = cute_ext.allocate(
            cutlass.Int32, cute.AddressSpace.smem, cute.make_layout(1), alignment=4
        ).iterator
        if warp == self.mma_warp:
            with cute.arch.elect_one():
                for stage in cutlass.range_constexpr(self.raw_stages):
                    cute.arch.mbarrier_init(full + stage, 1)
                    cute.arch.mbarrier_init(
                        raw_empty + stage, self.conversion_threads // 32
                    )
                for stage in cutlass.range_constexpr(self.stages):
                    cute.arch.mbarrier_init(
                        converted + stage, self.conversion_threads // 32
                    )
                    cute.arch.mbarrier_init(empty + stage, 1)
                for stage in cutlass.range_constexpr(self.input_stages):
                    cute.arch.mbarrier_init(input_ready + stage, 1)
                    cute.arch.mbarrier_init(input_empty + stage, 1)
                cute.arch.mbarrier_init(done, 1)
                cute.arch.mbarrier_init(read_done, self.epilogue_threads)
            cute.arch.alloc_tmem(self.tmem_columns, tmem_base, is_two_cta=False)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        weight_fragment = mma.make_fragment_A(
            cute.get_nonswizzle_portion(weight_layout)
        )
        weights = cute.make_tensor(
            cute.recast_ptr(
                cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, tmem_base)
                + self.columns,
                dtype=dtype,
            ),
            weight_fragment.layout,
        )
        x_view = cute.make_tensor(
            x.iterator,
            cute.make_layout(
                (x.shape[0], (64, self.inner // 64), 1), stride=(self.inner, (1, 64), 0)
            ),
        )
        if cutlass.const_expr(self.role == "projection"):
            w_view = cute.make_tensor(
                weight.iterator,
                cute.make_layout(
                    (self.projection_rows, (64, self.inner // 64), 1),
                    stride=(self.inner, (1, 64), 0),
                ),
            )
        else:
            w_view = cute.make_tensor(
                weight.iterator,
                cute.make_layout(
                    (
                        (32, self.hc_count, self.hidden_size // 32),
                        (64, self.inner // 64),
                        1,
                    ),
                    stride=(
                        (self.inner, self.hidden_size * self.inner, 32 * self.inner),
                        (1, 64),
                        0,
                    ),
                ),
            )
        gw = cute.local_tile(w_view, (self.m, self.k), (output_tile, None, 0))
        gx = cute.local_tile(x_view, (self.n, self.k), (token_tile, None, 0))
        map_w = cute_ext.get_cta_v_map_ab(w_view, tiler, mma, "A")
        map_x = cute_ext.get_cta_v_map_ab(x_view, tiler, mma, "B")
        if warp == self.producer_warp:
            if cutlass.const_expr(self.pdl and (not self.weights_independent)):
                cute.arch.griddepcontrol_wait()
            for tile in cutlass.range(self.tiles, unroll=1):
                raw_stage = tile % self.raw_stages
                if tile >= self.raw_stages:
                    cute.arch.mbarrier_wait(
                        raw_empty + raw_stage, (tile // self.raw_stages - 1) % 2
                    )
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        full + raw_stage, self.m * self.k
                    )
                cute_ext.tma_load(
                    gw[None, None, split * self.tiles + tile],
                    raw[None, None, None, raw_stage],
                    (full + raw_stage).value,
                    cta_v_map=map_w,
                    update_expect_tx=False,
                )
        elif warp == self.input_producer_warp:
            if cutlass.const_expr(self.pdl):
                cute.arch.griddepcontrol_wait()
                cute.arch.griddepcontrol_launch_dependents()
            for tile in cutlass.range(self.tiles, unroll=1):
                stage = tile % self.input_stages
                if tile >= self.input_stages:
                    cute.arch.mbarrier_wait(
                        input_empty + stage, (tile // self.input_stages - 1) % 2
                    )
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        input_ready + stage, self.n * self.k * 2
                    )
                cute_ext.tma_load(
                    gx[None, None, split * self.tiles + tile],
                    sx[None, None, None, stage],
                    (input_ready + stage).value,
                    cta_v_map=map_x,
                    update_expect_tx=False,
                )
        elif warp == self.mma_warp:
            thread_mma = mma.get_slice(0)
            a_desc = weights
            b_desc = thread_mma.make_fragment_B(sx)
            acc_shape = mma.partition_shape_C((self.m, self.n))
            acc_fragment = thread_mma.make_fragment_C(acc_shape)
            acc = cute.make_tensor(
                cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, tmem_base),
                acc_fragment.layout,
            )
            mma.set(tcgen05.Field.ACCUMULATE, False)
            for tile in cutlass.range(self.tiles, unroll=1):
                stage = tile % self.stages
                input_stage = tile % self.input_stages
                ready = full if cutlass.const_expr(native_fp8) else converted
                cute.arch.mbarrier_wait(ready + stage, tile // self.stages % 2)
                if cutlass.const_expr(self.separate_operand_ready):
                    cute.arch.mbarrier_wait(
                        input_ready + input_stage, tile // self.input_stages % 2
                    )
                tcgen05_fence("after_thread_sync")
                for block in cutlass.range_constexpr(cute.size(a_desc, mode=[2])):
                    cute.gemm(
                        mma,
                        acc,
                        a_desc[None, None, block, stage],
                        b_desc[None, None, block, input_stage],
                        acc,
                    )
                    mma.set(tcgen05.Field.ACCUMULATE, True)
                with cute.arch.elect_one():
                    tcgen05.commit(empty + stage, None, tcgen05.CtaGroup.ONE)
                    tcgen05.commit(
                        input_empty + input_stage, None, tcgen05.CtaGroup.ONE
                    )
            with cute.arch.elect_one():
                tcgen05.commit(done, None, tcgen05.CtaGroup.ONE)
            cute.arch.mbarrier_wait(read_done, 0)
            cute.arch.relinquish_tmem_alloc_permit()
            cute.arch.dealloc_tmem(
                cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, tmem_base),
                self.tmem_columns,
            )
        elif tid < self.epilogue_threads:
            if tid < self.conversion_threads:
                copy_atom = cute.make_copy_atom(
                    tcgen05.St32x32bOp(tcgen05.Repetition(8)), dtype
                )
                store_copy = tcgen05.make_tmem_copy(
                    copy_atom, weights[None, None, None, 0]
                )
                thread_copy = store_copy.get_slice(tid)
                source_tiles = thread_copy.partition_S(raw)
                target_tiles = thread_copy.partition_D(weights)
                raw_values = cute.make_rmem_tensor(
                    source_tiles[None, None, None, None, 0].shape, cutlass.Float8E4M3FN
                )
                decoded_values = cute.make_rmem_tensor(raw_values.shape, dtype)
                for tile in cutlass.range(self.tiles, unroll=1):
                    raw_stage = tile % self.raw_stages
                    stage = tile % self.stages
                    cute.arch.mbarrier_wait(
                        full + raw_stage, tile // self.raw_stages % 2
                    )
                    cute.autovec_copy(
                        source_tiles[None, None, None, None, raw_stage], raw_values
                    )
                    if tile >= self.stages:
                        cute.arch.mbarrier_wait(
                            empty + stage, (tile // self.stages - 1) % 2
                        )
                    tcgen05_fence("after_thread_sync")
                    _convert_fp8_weight(raw_values, decoded_values)
                    cute.copy(
                        store_copy,
                        decoded_values,
                        target_tiles[None, None, None, None, stage],
                    )
                    cute.arch.fence_view_async_tmem_store()
                    tcgen05_fence("before_thread_sync")
                    cute.arch.sync_warp(mask=4294967295)
                    # Keep raw storage owned until conversion and TMEM stores finish.
                    if tid % 32 == 0:
                        cute.arch.mbarrier_arrive(raw_empty + raw_stage)
                        cute.arch.mbarrier_arrive(converted + stage)
            cute.arch.mbarrier_wait(done, 0)
            thread_mma = mma.get_slice(0)
            acc_shape = mma.partition_shape_C((self.m, self.n))
            fragment = thread_mma.make_fragment_C(acc_shape)
            acc = cute.make_tensor(
                cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, tmem_base),
                fragment.layout,
            )[(None, None), 0, 0]
            reader_n = self.n // self.epilogue_groups
            tile_shape = (self.m, reader_n)
            group = tid // 128
            local_tid = tid % 128
            acc = cute.local_tile(acc, tile_shape, (0, group))
            coordinates = cute.local_tile(
                cute.make_identity_tensor((self.m, self.n)), tile_shape, (0, group)
            )
            copy = tcgen05.make_tmem_copy(
                sm100_utils.get_tmem_load_op(
                    (self.m, reader_n, self.k),
                    cute_utils.LayoutEnum.COL_MAJOR,
                    cutlass.Float32,
                    cutlass.Float32,
                    tile_shape,
                    False,
                ),
                acc,
            )
            reader = copy.get_slice(local_tid)
            dummy = cute.make_tensor(
                out.iterator, cute.make_layout(tile_shape, stride=(1, self.m))
            )
            layout = cute_ext.make_t2r_rmem_layout(
                copy, cute.flat_divide(dummy, tile_shape), local_tid
            )
            values = cute_ext.allocate(
                cutlass.Float32, cute.AddressSpace.rmem, layout, alignment=32
            )
            coords = reader.partition_D(coordinates)
            cute_ext.partition_and_copy(reader, acc, values)
            cute.arch.fence_view_async_tmem_load()
            if cutlass.const_expr(self.role == "mix"):
                residual = cute.make_rmem_tensor_like(values, original.element_type)
                scales = cute.make_rmem_tensor_like(values)
                residual.fill(0.0)
                scales.fill(1.0)
                for index in cutlass.range_constexpr(cute.size(values)):
                    col, row = coords[index]
                    global_row = token_tile * self.n + row
                    global_col = (
                        col // 32 * self.hidden_size + output_tile * 32 + col % 32
                    )
                    if global_row < x.shape[0]:
                        residual[index] = original[global_row, global_col]
                        scale = weight_scale[
                            global_col if self.weight_scale_per_channel else 0
                        ]
                        if cutlass.const_expr(native_fp8):
                            scale = scale * activation_scale[global_row]
                        scales[index] = scale
                for i in cutlass.range_constexpr(cute.size(values)):
                    values[i] = residual[i].to(cutlass.Float32) * _up_sigmoid_tanh(
                        values[i] * scales[i]
                    )
            for index in cutlass.range_constexpr(cute.size(values)):
                col, row = coords[index]
                global_row = token_tile * self.n + row
                if cutlass.const_expr(self.role == "projection"):
                    global_col = output_tile * self.m + col
                    if global_row < x.shape[0] and global_col < self.projection_rows:
                        partials[split, global_row, global_col] = values[index]
                elif global_row < x.shape[0]:
                    gate[col, row] = values[index]
            tcgen05_fence("before_thread_sync")
            cute.arch.mbarrier_arrive(read_done)
            if cutlass.const_expr(self.role == "mix"):
                _epilogue_barrier(self.epilogue_threads, cutlass.Int32(2))
                for offset in cutlass.range_constexpr(
                    self.n * 32 // self.epilogue_threads
                ):
                    index = tid + offset * self.epilogue_threads
                    row, col = (index // 32, index % 32)
                    if token_tile * self.n + row < x.shape[0]:
                        value = cutlass.Float32(0.0)
                        for branch in cutlass.range_constexpr(self.hc_count):
                            value = value + gate[branch * 32 + col, row]
                        out[token_tile * self.n + row, output_tile * 32 + col] = (
                            value / self.hc_count
                        ).to(original.element_type)


class W8GatedResidualRotatedProjection(W8GatedResidualTmemProjection):
    """Stagger K phases between CTAs while preserving each CTA reduction domain."""

    @cute.experimental.kernel
    def kernel(
        self, x, weight, original, weight_scale, activation_scale, partials, out
    ):
        tid = cute.arch.thread_idx()[0]
        warp = cute.arch.make_warp_uniform(tid // 32)
        output_tile, token_tile, split = cute.arch.block_idx()
        rotation = (output_tile * 3 + split) % self.tiles
        dtype = x.element_type
        native_fp8 = cutlass.const_expr(dtype == cutlass.Float8E4M3FN)
        mma = sm100_utils.make_trivial_tiled_mma(
            dtype,
            dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            cutlass.Float32,
            tcgen05.CtaGroup.ONE,
            (self.m, self.n),
            tcgen05.OperandSource.TMEM,
        )
        tiler = (self.m, self.n, self.k)
        weight_layout = sm100_utils.make_smem_layout_a(mma, tiler, dtype, self.stages)
        raw_layout = sm100_utils.make_smem_layout_a(
            mma, tiler, cutlass.Float8E4M3FN, self.raw_stages
        )
        x_layout = sm100_utils.make_smem_layout_b(mma, tiler, dtype, self.input_stages)
        assert not native_fp8
        raw = cute_ext.allocate(
            cutlass.Float8E4M3FN, cute.AddressSpace.smem, raw_layout, alignment=1024
        )
        sx = cute_ext.allocate(dtype, cute.AddressSpace.smem, x_layout, alignment=1024)
        barriers = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(
                2 * self.raw_stages + 2 * self.stages + 2 * self.input_stages + 2
            ),
            alignment=8,
        ).iterator
        full = barriers
        raw_empty = full + self.raw_stages
        converted = raw_empty + self.raw_stages
        empty = converted + self.stages
        input_ready = empty + self.stages
        input_empty = input_ready + self.input_stages
        done = input_empty + self.input_stages
        read_done = done + 1
        tmem_base = cute_ext.allocate(
            cutlass.Int32, cute.AddressSpace.smem, cute.make_layout(1), alignment=4
        ).iterator
        if warp == self.mma_warp:
            with cute.arch.elect_one():
                for stage in cutlass.range_constexpr(self.raw_stages):
                    cute.arch.mbarrier_init(full + stage, 1)
                    cute.arch.mbarrier_init(
                        raw_empty + stage, self.conversion_threads // 32
                    )
                for stage in cutlass.range_constexpr(self.stages):
                    cute.arch.mbarrier_init(
                        converted + stage, self.conversion_threads // 32
                    )
                    cute.arch.mbarrier_init(empty + stage, 1)
                for stage in cutlass.range_constexpr(self.input_stages):
                    cute.arch.mbarrier_init(input_ready + stage, 1)
                    cute.arch.mbarrier_init(input_empty + stage, 1)
                cute.arch.mbarrier_init(done, 1)
                cute.arch.mbarrier_init(read_done, self.epilogue_threads)
            cute.arch.alloc_tmem(self.tmem_columns, tmem_base, is_two_cta=False)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        weight_fragment = mma.make_fragment_A(
            cute.get_nonswizzle_portion(weight_layout)
        )
        weights = cute.make_tensor(
            cute.recast_ptr(
                cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, tmem_base)
                + self.columns,
                dtype=dtype,
            ),
            weight_fragment.layout,
        )
        x_view = cute.make_tensor(
            x.iterator,
            cute.make_layout(
                (x.shape[0], (64, self.inner // 64), 1), stride=(self.inner, (1, 64), 0)
            ),
        )
        if cutlass.const_expr(self.role == "projection"):
            w_view = cute.make_tensor(
                weight.iterator,
                cute.make_layout(
                    (self.projection_rows, (64, self.inner // 64), 1),
                    stride=(self.inner, (1, 64), 0),
                ),
            )
        else:
            w_view = cute.make_tensor(
                weight.iterator,
                cute.make_layout(
                    (
                        (32, self.hc_count, self.hidden_size // 32),
                        (64, self.inner // 64),
                        1,
                    ),
                    stride=(
                        (self.inner, self.hidden_size * self.inner, 32 * self.inner),
                        (1, 64),
                        0,
                    ),
                ),
            )
        gw = cute.local_tile(w_view, (self.m, self.k), (output_tile, None, 0))
        gx = cute.local_tile(x_view, (self.n, self.k), (token_tile, None, 0))
        map_w = cute_ext.get_cta_v_map_ab(w_view, tiler, mma, "A")
        map_x = cute_ext.get_cta_v_map_ab(x_view, tiler, mma, "B")
        if warp == self.producer_warp:
            if cutlass.const_expr(self.pdl and (not self.weights_independent)):
                cute.arch.griddepcontrol_wait()
            for tile in cutlass.range(self.tiles, unroll=1):
                raw_stage = tile % self.raw_stages
                if tile >= self.raw_stages:
                    cute.arch.mbarrier_wait(
                        raw_empty + raw_stage, (tile // self.raw_stages - 1) % 2
                    )
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        full + raw_stage, self.m * self.k
                    )
                cute_ext.tma_load(
                    gw[None, None, split * self.tiles + (tile + rotation) % self.tiles],
                    raw[None, None, None, raw_stage],
                    (full + raw_stage).value,
                    cta_v_map=map_w,
                    update_expect_tx=False,
                )
        elif warp == self.input_producer_warp:
            if cutlass.const_expr(self.pdl):
                cute.arch.griddepcontrol_wait()
                cute.arch.griddepcontrol_launch_dependents()
            for tile in cutlass.range(self.tiles, unroll=1):
                stage = tile % self.input_stages
                if tile >= self.input_stages:
                    cute.arch.mbarrier_wait(
                        input_empty + stage, (tile // self.input_stages - 1) % 2
                    )
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        input_ready + stage, self.n * self.k * 2
                    )
                cute_ext.tma_load(
                    gx[None, None, split * self.tiles + (tile + rotation) % self.tiles],
                    sx[None, None, None, stage],
                    (input_ready + stage).value,
                    cta_v_map=map_x,
                    update_expect_tx=False,
                )
        elif warp == self.mma_warp:
            thread_mma = mma.get_slice(0)
            a_desc = weights
            b_desc = thread_mma.make_fragment_B(sx)
            acc_shape = mma.partition_shape_C((self.m, self.n))
            acc_fragment = thread_mma.make_fragment_C(acc_shape)
            acc = cute.make_tensor(
                cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, tmem_base),
                acc_fragment.layout,
            )
            mma.set(tcgen05.Field.ACCUMULATE, False)
            for tile in cutlass.range(self.tiles, unroll=1):
                stage = tile % self.stages
                input_stage = tile % self.input_stages
                ready = full if cutlass.const_expr(native_fp8) else converted
                cute.arch.mbarrier_wait(ready + stage, tile // self.stages % 2)
                if cutlass.const_expr(self.separate_operand_ready):
                    cute.arch.mbarrier_wait(
                        input_ready + input_stage, tile // self.input_stages % 2
                    )
                tcgen05_fence("after_thread_sync")
                for block in cutlass.range_constexpr(cute.size(a_desc, mode=[2])):
                    cute.gemm(
                        mma,
                        acc,
                        a_desc[None, None, block, stage],
                        b_desc[None, None, block, input_stage],
                        acc,
                    )
                    mma.set(tcgen05.Field.ACCUMULATE, True)
                with cute.arch.elect_one():
                    tcgen05.commit(empty + stage, None, tcgen05.CtaGroup.ONE)
                    tcgen05.commit(
                        input_empty + input_stage, None, tcgen05.CtaGroup.ONE
                    )
            with cute.arch.elect_one():
                tcgen05.commit(done, None, tcgen05.CtaGroup.ONE)
            cute.arch.mbarrier_wait(read_done, 0)
            cute.arch.relinquish_tmem_alloc_permit()
            cute.arch.dealloc_tmem(
                cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, tmem_base),
                self.tmem_columns,
            )
        elif tid < self.epilogue_threads:
            if tid < self.conversion_threads:
                copy_atom = cute.make_copy_atom(
                    tcgen05.St32x32bOp(tcgen05.Repetition(8)), dtype
                )
                store_copy = tcgen05.make_tmem_copy(
                    copy_atom, weights[None, None, None, 0]
                )
                thread_copy = store_copy.get_slice(tid)
                source_tiles = thread_copy.partition_S(raw)
                target_tiles = thread_copy.partition_D(weights)
                raw_values = cute.make_rmem_tensor(
                    source_tiles[None, None, None, None, 0].shape, cutlass.Float8E4M3FN
                )
                decoded_values = cute.make_rmem_tensor(raw_values.shape, dtype)
                for tile in cutlass.range(self.tiles, unroll=1):
                    raw_stage = tile % self.raw_stages
                    stage = tile % self.stages
                    cute.arch.mbarrier_wait(
                        full + raw_stage, tile // self.raw_stages % 2
                    )
                    cute.autovec_copy(
                        source_tiles[None, None, None, None, raw_stage], raw_values
                    )
                    if tile >= self.stages:
                        cute.arch.mbarrier_wait(
                            empty + stage, (tile // self.stages - 1) % 2
                        )
                    tcgen05_fence("after_thread_sync")
                    _convert_fp8_weight(raw_values, decoded_values)
                    cute.copy(
                        store_copy,
                        decoded_values,
                        target_tiles[None, None, None, None, stage],
                    )
                    cute.arch.fence_view_async_tmem_store()
                    tcgen05_fence("before_thread_sync")
                    cute.arch.sync_warp(mask=4294967295)
                    # Finish conversion before returning raw storage to TMA.
                    if tid % 32 == 0:
                        cute.arch.mbarrier_arrive(raw_empty + raw_stage)
                        cute.arch.mbarrier_arrive(converted + stage)
            cute.arch.mbarrier_wait(done, 0)
            thread_mma = mma.get_slice(0)
            acc_shape = mma.partition_shape_C((self.m, self.n))
            fragment = thread_mma.make_fragment_C(acc_shape)
            acc = cute.make_tensor(
                cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, tmem_base),
                fragment.layout,
            )[(None, None), 0, 0]
            reader_n = self.n // self.epilogue_groups
            tile_shape = (self.m, reader_n)
            group = tid // 128
            local_tid = tid % 128
            acc = cute.local_tile(acc, tile_shape, (0, group))
            coordinates = cute.local_tile(
                cute.make_identity_tensor((self.m, self.n)), tile_shape, (0, group)
            )
            copy = tcgen05.make_tmem_copy(
                sm100_utils.get_tmem_load_op(
                    (self.m, reader_n, self.k),
                    cute_utils.LayoutEnum.COL_MAJOR,
                    cutlass.Float32,
                    cutlass.Float32,
                    tile_shape,
                    False,
                ),
                acc,
            )
            reader = copy.get_slice(local_tid)
            dummy = cute.make_tensor(
                out.iterator, cute.make_layout(tile_shape, stride=(1, self.m))
            )
            layout = cute_ext.make_t2r_rmem_layout(
                copy, cute.flat_divide(dummy, tile_shape), local_tid
            )
            values = cute_ext.allocate(
                cutlass.Float32, cute.AddressSpace.rmem, layout, alignment=32
            )
            coords = reader.partition_D(coordinates)
            cute_ext.partition_and_copy(reader, acc, values)
            cute.arch.fence_view_async_tmem_load()
            if cutlass.const_expr(self.role == "mix"):
                residual = cute.make_rmem_tensor_like(values, original.element_type)
                scales = cute.make_rmem_tensor_like(values)
                residual.fill(0.0)
                scales.fill(1.0)
                for index in cutlass.range_constexpr(cute.size(values)):
                    col, row = coords[index]
                    global_row = token_tile * self.n + row
                    global_col = (
                        col // 32 * self.hidden_size + output_tile * 32 + col % 32
                    )
                    if global_row < x.shape[0]:
                        residual[index] = original[global_row, global_col]
                        scale = weight_scale[
                            global_col if self.weight_scale_per_channel else 0
                        ]
                        if cutlass.const_expr(native_fp8):
                            scale = scale * activation_scale[global_row]
                        scales[index] = scale
                for i in cutlass.range_constexpr(cute.size(values)):
                    values[i] = residual[i].to(cutlass.Float32) * _up_sigmoid_tanh(
                        values[i] * scales[i]
                    )
            for index in cutlass.range_constexpr(cute.size(values)):
                col, row = coords[index]
                global_row = token_tile * self.n + row
                if cutlass.const_expr(self.role == "projection"):
                    global_col = output_tile * self.m + col
                    if global_row < x.shape[0] and global_col < self.projection_rows:
                        partials[split, global_row, global_col] = values[index]
                elif global_row < x.shape[0]:
                    gate[col, row] = values[index]
            tcgen05_fence("before_thread_sync")
            cute.arch.mbarrier_arrive(read_done)
            if cutlass.const_expr(self.role == "mix"):
                _epilogue_barrier(self.epilogue_threads, cutlass.Int32(2))
                for offset in cutlass.range_constexpr(
                    self.n * 32 // self.epilogue_threads
                ):
                    index = tid + offset * self.epilogue_threads
                    row, col = (index // 32, index % 32)
                    if token_tile * self.n + row < x.shape[0]:
                        value = cutlass.Float32(0.0)
                        for branch in cutlass.range_constexpr(self.hc_count):
                            value = value + gate[branch * 32 + col, row]
                        out[token_tile * self.n + row, output_tile * 32 + col] = (
                            value / self.hc_count
                        ).to(original.element_type)


class W8GatedResidualDecodedRingUp(W8GatedResidualIndependentUp):
    """Separate compressed and decoded rings remove in-place expansion barriers."""

    def __init__(
        self,
        *,
        hidden_size: int,
        lowrank: int,
        raw_stages: int,
        decoded_stages: int,
        input_stages: int,
        use_pdl: bool,
        weights_independent: bool,
        weight_scale_per_channel: bool,
        weight_scale_vectorized: bool,
        up_token_rows: int,
    ):
        Fp8GatedResidualGemm.__init__(
            self,
            role="mix",
            hc_count=4,
            hidden_size=hidden_size,
            lowrank=lowrank,
            projection_rows=lowrank + 4,
            token_tile=up_token_rows,
            projection_tile=128,
            k_tile=64,
            stages=decoded_stages,
            split_k=1,
            use_pdl=use_pdl,
            weights_independent=weights_independent,
            weight_scale_per_channel=weight_scale_per_channel,
            token_major_up=True,
            weight_scale_vectorized=weight_scale_vectorized,
            up_hidden_tile=64,
            up_token_rows=up_token_rows,
            separate_operand_ready=False,
            epilogue_threads=512,
            shared_up_scales=False,
            interleave_tma=False,
        )
        self.input_stages = input_stages
        self.raw_stages = raw_stages
        self.input_producer_warp = self.producer_warp + 1
        self.mma_warp = self.producer_warp + 2

    @cute.experimental.jit
    def __call__(
        self, x, weight, original, weight_scale, activation_scale, out, stream: CUstream
    ):
        self.up_kernel(x, weight, original, weight_scale, activation_scale, out).launch(
            grid=(
                self.hidden_size // self.up_hidden_tile,
                cute.ceil_div(x.shape[0], self.up_token_rows),
                1,
            ),
            block=(self.epilogue_threads + 96, 1, 1),
            stream=stream,
            use_pdl=self.pdl,
        )

    @cute.experimental.kernel
    def up_kernel(self, x, weight, original, weight_scale, activation_scale, out):
        tid = cute.arch.thread_idx()[0]
        warp = cute.arch.make_warp_uniform(tid // 32)
        hidden_tile, token_tile, _ = cute.arch.block_idx()
        dtype = x.element_type
        native_fp8 = cutlass.const_expr(dtype == cutlass.Float8E4M3FN)
        mma = sm100_utils.make_trivial_tiled_mma(
            dtype,
            dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            cutlass.Float32,
            tcgen05.CtaGroup.ONE,
            (self.up_token_rows, self.up_hidden_tile * self.hc_count),
        )
        tiler = (self.up_token_rows, self.up_hidden_tile * self.hc_count, self.k)
        activation_layout = sm100_utils.make_smem_layout_a(
            mma, tiler, dtype, self.input_stages
        )
        raw_layout = sm100_utils.make_smem_layout_b(
            mma, tiler, cutlass.Float8E4M3FN, self.raw_stages
        )
        weight_layout = sm100_utils.make_smem_layout_b(mma, tiler, dtype, self.stages)
        sx = cute_ext.allocate(
            dtype, cute.AddressSpace.smem, activation_layout, alignment=1024
        )
        assert not native_fp8
        raw = cute_ext.allocate(
            cutlass.Float8E4M3FN, cute.AddressSpace.smem, raw_layout, alignment=1024
        )
        sw = cute_ext.allocate(
            dtype, cute.AddressSpace.smem, weight_layout, alignment=1024
        )
        barriers = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(
                2 * self.raw_stages + 2 * self.stages + 2 * self.input_stages + 2
            ),
            alignment=8,
        ).iterator
        full = barriers
        raw_empty = full + self.raw_stages
        converted = raw_empty + self.raw_stages
        empty = converted + self.stages
        x_full = empty + self.stages
        x_empty = x_full + self.input_stages
        done, read_done = (x_empty + self.input_stages, x_empty + self.input_stages + 1)
        base = cute_ext.allocate(
            cutlass.Int32, cute.AddressSpace.smem, cute.make_layout(1), alignment=4
        ).iterator
        if warp == self.mma_warp:
            with cute.arch.elect_one():
                for stage in cutlass.range_constexpr(self.raw_stages):
                    cute.arch.mbarrier_init(full + stage, 1)
                    cute.arch.mbarrier_init(
                        raw_empty + stage, self.epilogue_threads // 32
                    )
                for stage in cutlass.range_constexpr(self.stages):
                    cute.arch.mbarrier_init(
                        converted + stage, self.epilogue_threads // 32
                    )
                    cute.arch.mbarrier_init(empty + stage, 1)
                for stage in cutlass.range_constexpr(self.input_stages):
                    cute.arch.mbarrier_init(x_full + stage, 1)
                    cute.arch.mbarrier_init(x_empty + stage, 1)
                cute.arch.mbarrier_init(done, 1)
                cute.arch.mbarrier_init(read_done, self.epilogue_threads)
            cute.arch.alloc_tmem(
                self.up_hidden_tile * self.hc_count, base, is_two_cta=False
            )
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        activation = cute.make_tensor(
            x.iterator,
            cute.make_layout(
                (x.shape[0], (64, self.inner // 64), 1), stride=(self.inner, (1, 64), 0)
            ),
        )
        weights = cute.make_tensor(
            weight.iterator,
            cute.make_layout(
                (
                    (
                        self.up_hidden_tile,
                        self.hc_count,
                        self.hidden_size // self.up_hidden_tile,
                    ),
                    (64, self.inner // 64),
                    1,
                ),
                stride=(
                    (
                        self.inner,
                        self.hidden_size * self.inner,
                        self.up_hidden_tile * self.inner,
                    ),
                    (1, 64),
                    0,
                ),
            ),
        )
        gx = cute.local_tile(
            activation, (self.up_token_rows, self.k), (token_tile, None, 0)
        )
        gw = cute.local_tile(
            weights,
            (self.up_hidden_tile * self.hc_count, self.k),
            (hidden_tile, None, 0),
        )
        map_x = cute_ext.get_cta_v_map_ab(activation, tiler, mma, "A")
        map_w = cute_ext.get_cta_v_map_ab(weights, tiler, mma, "B")
        if warp == self.producer_warp:
            if cutlass.const_expr(self.pdl and (not self.weights_independent)):
                cute.arch.griddepcontrol_wait()
            for tile in cutlass.range(self.tiles, unroll=1):
                raw_stage = tile % self.raw_stages
                if tile >= self.raw_stages:
                    cute.arch.mbarrier_wait(
                        raw_empty + raw_stage, (tile // self.raw_stages - 1) % 2
                    )
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        full + raw_stage, self.up_hidden_tile * self.hc_count * self.k
                    )
                cute_ext.tma_load(
                    gw[None, None, tile],
                    raw[None, None, None, raw_stage],
                    (full + raw_stage).value,
                    cta_v_map=map_w,
                    update_expect_tx=False,
                )
        elif warp == self.input_producer_warp:
            if cutlass.const_expr(self.pdl):
                cute.arch.griddepcontrol_wait()
                cute.arch.griddepcontrol_launch_dependents()
            for tile in cutlass.range(self.tiles, unroll=1):
                input_stage = tile % self.input_stages
                if tile >= self.input_stages:
                    cute.arch.mbarrier_wait(
                        x_empty + input_stage, (tile // self.input_stages - 1) % 2
                    )
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        x_full + input_stage, self.up_token_rows * self.k * 2
                    )
                cute_ext.tma_load(
                    gx[None, None, tile],
                    sx[None, None, None, input_stage],
                    (x_full + input_stage).value,
                    cta_v_map=map_x,
                    update_expect_tx=False,
                )
        elif warp == self.mma_warp:
            thread_mma = mma.get_slice(0)
            a_desc = thread_mma.make_fragment_A(sx)
            b_desc = thread_mma.make_fragment_B(sw)
            fragment = thread_mma.make_fragment_C(
                mma.partition_shape_C(
                    (self.up_token_rows, self.up_hidden_tile * self.hc_count)
                )
            )
            accumulator = cute.make_tensor(
                cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, base), fragment.layout
            )
            mma.set(tcgen05.Field.ACCUMULATE, False)
            for tile in cutlass.range(self.tiles, unroll=1):
                stage = tile % self.stages
                input_stage = tile % self.input_stages
                ready = full if cutlass.const_expr(native_fp8) else converted
                cute.arch.mbarrier_wait(ready + stage, tile // self.stages % 2)
                cute.arch.mbarrier_wait(
                    x_full + input_stage, tile // self.input_stages % 2
                )
                for block in cutlass.range_constexpr(cute.size(a_desc, mode=[2])):
                    cute.gemm(
                        mma,
                        accumulator,
                        a_desc[None, None, block, input_stage],
                        b_desc[None, None, block, stage],
                        accumulator,
                    )
                    mma.set(tcgen05.Field.ACCUMULATE, True)
                with cute.arch.elect_one():
                    tcgen05.commit(empty + stage, None, tcgen05.CtaGroup.ONE)
                    tcgen05.commit(x_empty + input_stage, None, tcgen05.CtaGroup.ONE)
            with cute.arch.elect_one():
                tcgen05.commit(done, None, tcgen05.CtaGroup.ONE)
            cute.arch.mbarrier_wait(read_done, 0)
            cute.arch.relinquish_tmem_alloc_permit()
            cute.arch.dealloc_tmem(
                cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, base),
                self.up_hidden_tile * self.hc_count,
            )
        elif tid < self.epilogue_threads:
            if cutlass.const_expr(not native_fp8):
                for tile in cutlass.range(self.tiles, unroll=1):
                    raw_stage = tile % self.raw_stages
                    stage = tile % self.stages
                    cute.arch.mbarrier_wait(
                        full + raw_stage, tile // self.raw_stages % 2
                    )
                    if tile >= self.stages:
                        cute.arch.mbarrier_wait(
                            empty + stage, (tile // self.stages - 1) % 2
                        )
                    source = raw[None, None, None, raw_stage]
                    target = sw[None, None, None, stage]
                    packets = (
                        self.up_hidden_tile
                        * self.hc_count
                        * self.k
                        // (self.epilogue_threads * 8)
                    )
                    for offset in cutlass.range_constexpr(packets):
                        linear = (tid + offset * self.epilogue_threads) * 8
                        index = (
                            linear // self.k
                            + linear % self.k * self.up_hidden_tile * self.hc_count
                        )
                        src = cute.make_tensor(
                            cute.make_ptr(
                                cutlass.Float8E4M3FN,
                                (source.iterator + source.layout(index)).toint(),
                                cute.AddressSpace.smem,
                                assumed_align=8,
                                swizzle_=cute.get_swizzle_portion(raw_layout),
                            ),
                            cute.make_layout(8),
                        )
                        raw_values = cute.make_rmem_tensor(8, cutlass.Float8E4M3FN)
                        cute.copy(
                            cute.make_copy_atom(
                                cute.nvgpu.CopyUniversalOp(),
                                cutlass.Float8E4M3FN,
                                num_bits_per_copy=64,
                            ),
                            src,
                            raw_values,
                        )
                        decoded = cute.make_rmem_tensor(8, dtype)
                        _convert_fp8_weight(raw_values, decoded)
                        dst = cute.make_tensor(
                            cute.make_ptr(
                                dtype,
                                (target.iterator + target.layout(index)).toint(),
                                cute.AddressSpace.smem,
                                assumed_align=16,
                                swizzle_=cute.get_swizzle_portion(weight_layout),
                            ),
                            cute.make_layout(8),
                        )
                        cute.copy(
                            cute.make_copy_atom(
                                cute.nvgpu.CopyUniversalOp(),
                                dtype,
                                num_bits_per_copy=128,
                            ),
                            decoded,
                            dst,
                        )
                    cute.arch.fence_view_async_shared()
                    cute.arch.sync_warp(mask=4294967295)
                    # Finish conversion before returning raw storage to TMA.
                    if tid % 32 == 0:
                        cute.arch.mbarrier_arrive(raw_empty + raw_stage)
                        cute.arch.mbarrier_arrive(converted + stage)
            if cutlass.const_expr(self.pdl):
                cute.arch.griddepcontrol_wait()
            thread_mma = mma.get_slice(0)
            fragment = thread_mma.make_fragment_C(
                mma.partition_shape_C(
                    (self.up_token_rows, self.up_hidden_tile * self.hc_count)
                )
            )
            _, _, template, coordinates = self._up_reader(
                base, fragment.layout, original, tid, 0
            )
            count = cute.size(template)
            ring_slots = (
                self.hc_count
                if self.up_token_rows == 64
                else 2 if self.up_hidden_tile == 64 else self.hc_count
            )
            residual = cute.make_rmem_tensor((count, ring_slots), original.element_type)
            scales = cute.make_rmem_tensor(
                (1, 1) if self.up_token_rows == 64 else (count, ring_slots),
                cutlass.Float32,
            )
            residual.fill(0.0)
            scales.fill(1.0)
            sums = cute.make_rmem_tensor(count, cutlass.Float32)
            sums.fill(0.0)
            first_row, first_col = coordinates[0]
            global_row = token_tile * self.up_token_rows + first_row
            hidden = hidden_tile * self.up_hidden_tile + first_col
            if cutlass.const_expr(self.up_token_rows == 64):
                assert not native_fp8
                scale_cache = cute_ext.allocate(
                    cutlass.Float32,
                    cute.AddressSpace.smem,
                    cute.make_layout(256),
                    alignment=32,
                )
                if tid < 256:
                    channel = tid // 64 * self.hidden_size + hidden_tile * 64 + tid % 64
                    scale_cache[tid] = weight_scale[
                        channel if self.weight_scale_per_channel else 0
                    ]
                _epilogue_barrier(self.epilogue_threads, cutlass.Int32(3))
                for branch in cutlass.range_constexpr(4):
                    self._up_pair_prefetch(
                        original,
                        residual,
                        coordinates,
                        token_tile,
                        hidden_tile,
                        x.shape[0],
                        branch,
                    )
            else:
                for branch in cutlass.range_constexpr(ring_slots):
                    self._up_prefetch(
                        original,
                        weight_scale,
                        activation_scale,
                        residual,
                        scales,
                        global_row,
                        hidden,
                        x.shape[0],
                        branch,
                        branch,
                        native_fp8,
                    )
            cute.arch.mbarrier_wait(done, 0)
            if (
                tid % 128 // 32 * (self.up_token_rows // 4)
                < x.shape[0] - token_tile * self.up_token_rows
            ):
                for branch in cutlass.range_constexpr(self.hc_count):
                    slot = branch % ring_slots
                    accumulator, reader, gates, _ = self._up_reader(
                        base, fragment.layout, original, tid, branch
                    )
                    cute_ext.partition_and_copy(reader, accumulator, gates)
                    cute.arch.fence_view_async_tmem_load()
                    for i in cutlass.range_constexpr(count):
                        factor = cutlass.Float32(1.0)
                        if cutlass.const_expr(self.up_token_rows == 64):
                            _, col = coordinates[i]
                            factor = scale_cache[branch * 64 + col]
                        else:
                            factor = scales[i, slot]
                        sums[i] = sums[i] + residual[i, slot].to(
                            cutlass.Float32
                        ) * _up_sigmoid_tanh(gates[i] * factor)
                    if cutlass.const_expr(branch + ring_slots < self.hc_count):
                        self._up_prefetch(
                            original,
                            weight_scale,
                            activation_scale,
                            residual,
                            scales,
                            global_row,
                            hidden,
                            x.shape[0],
                            branch + ring_slots,
                            slot,
                            native_fp8,
                        )
            tcgen05_fence("before_thread_sync")
            cute.arch.mbarrier_arrive(read_done)
            if cutlass.const_expr(self.up_token_rows == 64):
                for packet in cutlass.range_constexpr(count // 2):
                    row, col = coordinates[packet * 2]
                    output_row = token_tile * self.up_token_rows + row
                    if output_row < x.shape[0]:
                        packed = cute.make_rmem_tensor(2, out.element_type)
                        for j in cutlass.range_constexpr(2):
                            packed[j] = (sums[packet * 2 + j] / self.hc_count).to(
                                out.element_type
                            )
                        target = cute.make_tensor(
                            cute.make_ptr(
                                out.element_type,
                                (
                                    out.iterator
                                    + output_row * self.hidden_size
                                    + hidden_tile * self.up_hidden_tile
                                    + col
                                ).toint(),
                                cute.AddressSpace.gmem,
                                assumed_align=4,
                            ),
                            cute.make_layout(2),
                        )
                        cute.copy(
                            cute.make_copy_atom(
                                cute.nvgpu.CopyUniversalOp(),
                                out.element_type,
                                num_bits_per_copy=32,
                            ),
                            packed,
                            target,
                        )
            elif global_row < x.shape[0]:
                packed = cute.make_rmem_tensor(count, out.element_type)
                for i in cutlass.range_constexpr(count):
                    packed[i] = (sums[i] / self.hc_count).to(out.element_type)
                target = cute.make_tensor(
                    cute.make_ptr(
                        out.element_type,
                        (out.iterator + global_row * self.hidden_size + hidden).toint(),
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    ),
                    cute.make_layout(count),
                )
                cute.copy(
                    cute.make_copy_atom(
                        cute.nvgpu.CopyUniversalOp(),
                        out.element_type,
                        num_bits_per_copy=128,
                    ),
                    packed,
                    target,
                )


class W8GatedResidualDecodedTokenReuse(W8GatedResidualTokenReuse):
    """Share decoded weights across two token tiles with separate raw and decoded rings."""

    def __init__(
        self,
        *,
        hidden_size: int,
        lowrank: int,
        raw_stages: int,
        decoded_stages: int,
        input_stages: int,
        use_pdl: bool,
        weights_independent: bool,
        weight_scale_per_channel: bool,
        weight_scale_vectorized: bool,
    ):
        Fp8GatedResidualGemm.__init__(
            self,
            role="mix",
            hc_count=4,
            hidden_size=hidden_size,
            lowrank=lowrank,
            projection_rows=lowrank + 4,
            token_tile=128,
            projection_tile=128,
            k_tile=64,
            stages=decoded_stages,
            split_k=1,
            use_pdl=use_pdl,
            weights_independent=weights_independent,
            weight_scale_per_channel=weight_scale_per_channel,
            token_major_up=True,
            weight_scale_vectorized=weight_scale_vectorized,
            up_hidden_tile=64,
            up_token_rows=128,
            separate_operand_ready=False,
            epilogue_threads=512,
            shared_up_scales=False,
            interleave_tma=False,
        )
        self.input_stages = input_stages
        self.raw_stages = raw_stages
        self.input_producer_warp = self.producer_warp + 1
        self.mma_warp = self.producer_warp + 2

    @cute.experimental.jit
    def __call__(
        self, x, weight, original, weight_scale, activation_scale, out, stream: CUstream
    ):
        self.kernel(x, weight, original, weight_scale, activation_scale, out).launch(
            grid=(self.hidden_size // 64, 1, 1),
            block=(self.epilogue_threads + 96, 1, 1),
            stream=stream,
            use_pdl=self.pdl,
        )

    @cute.experimental.kernel
    def kernel(self, x, weight, original, weight_scale, activation_scale, out):
        tid = cute.arch.thread_idx()[0]
        warp = cute.arch.make_warp_uniform(tid // 32)
        hidden_tile, token_tile, _ = cute.arch.block_idx()
        dtype = x.element_type
        native_fp8 = cutlass.const_expr(dtype == cutlass.Float8E4M3FN)
        mma = sm100_utils.make_trivial_tiled_mma(
            dtype,
            dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            cutlass.Float32,
            tcgen05.CtaGroup.ONE,
            (128, self.up_hidden_tile * self.hc_count),
        )
        tiler = (128, self.up_hidden_tile * self.hc_count, self.k)
        activation_layout = sm100_utils.make_smem_layout_a(
            mma, tiler, dtype, self.input_stages
        )
        raw_layout = sm100_utils.make_smem_layout_b(
            mma, tiler, cutlass.Float8E4M3FN, self.raw_stages
        )
        weight_layout = sm100_utils.make_smem_layout_b(mma, tiler, dtype, self.stages)
        sx = cute_ext.allocate(
            dtype, cute.AddressSpace.smem, activation_layout, alignment=1024
        )
        assert not native_fp8
        raw = cute_ext.allocate(
            cutlass.Float8E4M3FN, cute.AddressSpace.smem, raw_layout, alignment=1024
        )
        sw = cute_ext.allocate(
            dtype, cute.AddressSpace.smem, weight_layout, alignment=1024
        )
        x1_layout = sm100_utils.make_smem_layout_a(mma, tiler, dtype, self.input_stages)
        sx1 = cute_ext.allocate(
            dtype, cute.AddressSpace.smem, x1_layout, alignment=1024
        )
        barriers = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(
                2 * self.raw_stages + 2 * self.stages + 4 * self.input_stages + 2
            ),
            alignment=8,
        ).iterator
        full = barriers
        raw_empty = full + self.raw_stages
        converted = raw_empty + self.raw_stages
        empty = converted + self.stages
        x0_full = empty + self.stages
        x0_empty = x0_full + self.input_stages
        x1_full = x0_empty + self.input_stages
        x1_empty = x1_full + self.input_stages
        done = x1_empty + self.input_stages
        read_done = done + 1
        base = cute_ext.allocate(
            cutlass.Int32, cute.AddressSpace.smem, cute.make_layout(1), alignment=4
        ).iterator
        if warp == self.mma_warp:
            with cute.arch.elect_one():
                for stage in cutlass.range_constexpr(self.raw_stages):
                    cute.arch.mbarrier_init(full + stage, 1)
                    cute.arch.mbarrier_init(
                        raw_empty + stage, self.epilogue_threads // 32
                    )
                for stage in cutlass.range_constexpr(self.stages):
                    cute.arch.mbarrier_init(
                        converted + stage, self.epilogue_threads // 32
                    )
                    cute.arch.mbarrier_init(empty + stage, 1)
                for stage in cutlass.range_constexpr(self.input_stages):
                    cute.arch.mbarrier_init(x0_full + stage, 1)
                    cute.arch.mbarrier_init(x0_empty + stage, 1)
                    cute.arch.mbarrier_init(x1_full + stage, 1)
                    cute.arch.mbarrier_init(x1_empty + stage, 1)
                cute.arch.mbarrier_init(done, 1)
                cute.arch.mbarrier_init(read_done, self.epilogue_threads)
            cute.arch.alloc_tmem(512, base, is_two_cta=False)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        activation = cute.make_tensor(
            x.iterator,
            cute.make_layout(
                (x.shape[0], (64, self.inner // 64), 1), stride=(self.inner, (1, 64), 0)
            ),
        )
        weights = cute.make_tensor(
            weight.iterator,
            cute.make_layout(
                (
                    (
                        self.up_hidden_tile,
                        self.hc_count,
                        self.hidden_size // self.up_hidden_tile,
                    ),
                    (64, self.inner // 64),
                    1,
                ),
                stride=(
                    (
                        self.inner,
                        self.hidden_size * self.inner,
                        self.up_hidden_tile * self.inner,
                    ),
                    (1, 64),
                    0,
                ),
            ),
        )
        gx = cute.local_tile(activation, (128, self.k), (token_tile, None, 0))
        gx1 = cute.local_tile(activation, (128, self.k), (1, None, 0))
        gw = cute.local_tile(
            weights,
            (self.up_hidden_tile * self.hc_count, self.k),
            (hidden_tile, None, 0),
        )
        map_x = cute_ext.get_cta_v_map_ab(activation, tiler, mma, "A")
        map_w = cute_ext.get_cta_v_map_ab(weights, tiler, mma, "B")
        if warp == self.producer_warp:
            if cutlass.const_expr(self.pdl and (not self.weights_independent)):
                cute.arch.griddepcontrol_wait()
            for tile in cutlass.range(self.tiles, unroll=1):
                raw_stage = tile % self.raw_stages
                if tile >= self.raw_stages:
                    cute.arch.mbarrier_wait(
                        raw_empty + raw_stage, (tile // self.raw_stages - 1) % 2
                    )
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        full + raw_stage, self.up_hidden_tile * self.hc_count * self.k
                    )
                cute_ext.tma_load(
                    gw[None, None, tile],
                    raw[None, None, None, raw_stage],
                    (full + raw_stage).value,
                    cta_v_map=map_w,
                    update_expect_tx=False,
                )
        elif warp == self.input_producer_warp:
            if cutlass.const_expr(self.pdl):
                cute.arch.griddepcontrol_wait()
                cute.arch.griddepcontrol_launch_dependents()
            for tile in cutlass.range(self.tiles, unroll=1):
                stage = tile % self.input_stages
                if tile >= self.input_stages:
                    cute.arch.mbarrier_wait(
                        x0_empty + stage, (tile // self.input_stages - 1) % 2
                    )
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        x0_full + stage, 128 * self.k * 2
                    )
                cute_ext.tma_load(
                    gx[None, None, tile],
                    sx[None, None, None, stage],
                    (x0_full + stage).value,
                    cta_v_map=map_x,
                    update_expect_tx=False,
                )
                if tile >= self.input_stages:
                    cute.arch.mbarrier_wait(
                        x1_empty + stage, (tile // self.input_stages - 1) % 2
                    )
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        x1_full + stage, 128 * self.k * 2
                    )
                cute_ext.tma_load(
                    gx1[None, None, tile],
                    sx1[None, None, None, stage],
                    (x1_full + stage).value,
                    cta_v_map=map_x,
                    update_expect_tx=False,
                )
        elif warp == self.mma_warp:
            thread_mma = mma.get_slice(0)
            a_desc = thread_mma.make_fragment_A(sx)
            b_desc = thread_mma.make_fragment_B(sw)
            fragment = thread_mma.make_fragment_C(
                mma.partition_shape_C((128, self.up_hidden_tile * self.hc_count))
            )
            accumulator = cute.make_tensor(
                cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, base), fragment.layout
            )
            a1_desc = thread_mma.make_fragment_A(sx1)
            accumulator1 = cute.make_tensor(
                cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, base) + 256,
                fragment.layout,
            )
            for tile in cutlass.range(self.tiles, unroll=1):
                stage = tile % self.stages
                input_stage = tile % self.input_stages
                input_phase = tile // self.input_stages % 2
                cute.arch.mbarrier_wait(converted + stage, tile // self.stages % 2)
                cute.arch.mbarrier_wait(x0_full + input_stage, input_phase)
                mma.set(tcgen05.Field.ACCUMULATE, tile > 0)
                for block in cutlass.range_constexpr(cute.size(a_desc, mode=[2])):
                    cute.gemm(
                        mma,
                        accumulator,
                        a_desc[None, None, block, input_stage],
                        b_desc[None, None, block, stage],
                        accumulator,
                    )
                    mma.set(tcgen05.Field.ACCUMULATE, True)
                with cute.arch.elect_one():
                    tcgen05.commit(x0_empty + input_stage, None, tcgen05.CtaGroup.ONE)
                cute.arch.mbarrier_wait(x1_full + input_stage, input_phase)
                mma.set(tcgen05.Field.ACCUMULATE, tile > 0)
                for block in cutlass.range_constexpr(cute.size(a1_desc, mode=[2])):
                    cute.gemm(
                        mma,
                        accumulator1,
                        a1_desc[None, None, block, input_stage],
                        b_desc[None, None, block, stage],
                        accumulator1,
                    )
                    mma.set(tcgen05.Field.ACCUMULATE, True)
                with cute.arch.elect_one():
                    tcgen05.commit(empty + stage, None, tcgen05.CtaGroup.ONE)
                    tcgen05.commit(x1_empty + input_stage, None, tcgen05.CtaGroup.ONE)
            with cute.arch.elect_one():
                tcgen05.commit(done, None, tcgen05.CtaGroup.ONE)
            cute.arch.mbarrier_wait(read_done, 0)
            cute.arch.relinquish_tmem_alloc_permit()
            cute.arch.dealloc_tmem(
                cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, base), 512
            )
        elif tid < self.epilogue_threads:
            if cutlass.const_expr(not native_fp8):
                for tile in cutlass.range(self.tiles, unroll=1):
                    raw_stage = tile % self.raw_stages
                    stage = tile % self.stages
                    cute.arch.mbarrier_wait(
                        full + raw_stage, tile // self.raw_stages % 2
                    )
                    if tile >= self.stages:
                        cute.arch.mbarrier_wait(
                            empty + stage, (tile // self.stages - 1) % 2
                        )
                    source = raw[None, None, None, raw_stage]
                    target = sw[None, None, None, stage]
                    packets = (
                        self.up_hidden_tile
                        * self.hc_count
                        * self.k
                        // (self.epilogue_threads * 8)
                    )
                    for offset in cutlass.range_constexpr(packets):
                        linear = (tid + offset * self.epilogue_threads) * 8
                        index = (
                            linear // self.k
                            + linear % self.k * self.up_hidden_tile * self.hc_count
                        )
                        src = cute.make_tensor(
                            cute.make_ptr(
                                cutlass.Float8E4M3FN,
                                (source.iterator + source.layout(index)).toint(),
                                cute.AddressSpace.smem,
                                assumed_align=8,
                                swizzle_=cute.get_swizzle_portion(raw_layout),
                            ),
                            cute.make_layout(8),
                        )
                        raw_values = cute.make_rmem_tensor(8, cutlass.Float8E4M3FN)
                        cute.copy(
                            cute.make_copy_atom(
                                cute.nvgpu.CopyUniversalOp(),
                                cutlass.Float8E4M3FN,
                                num_bits_per_copy=64,
                            ),
                            src,
                            raw_values,
                        )
                        decoded = cute.make_rmem_tensor(8, dtype)
                        _convert_fp8_weight(raw_values, decoded)
                        dst = cute.make_tensor(
                            cute.make_ptr(
                                dtype,
                                (target.iterator + target.layout(index)).toint(),
                                cute.AddressSpace.smem,
                                assumed_align=16,
                                swizzle_=cute.get_swizzle_portion(weight_layout),
                            ),
                            cute.make_layout(8),
                        )
                        cute.copy(
                            cute.make_copy_atom(
                                cute.nvgpu.CopyUniversalOp(),
                                dtype,
                                num_bits_per_copy=128,
                            ),
                            decoded,
                            dst,
                        )
                    cute.arch.fence_view_async_shared()
                    cute.arch.sync_warp(mask=4294967295)
                    # Finish conversion before returning raw storage to TMA.
                    if tid % 32 == 0:
                        cute.arch.mbarrier_arrive(raw_empty + raw_stage)
                        cute.arch.mbarrier_arrive(converted + stage)
            if cutlass.const_expr(self.pdl):
                cute.arch.griddepcontrol_wait()
            for token_part in cutlass.range(2, unroll=1):
                thread_mma = mma.get_slice(0)
                fragment = thread_mma.make_fragment_C(
                    mma.partition_shape_C((128, self.up_hidden_tile * self.hc_count))
                )
                _, _, template, coordinates = self._up_reader(
                    base, fragment.layout, original, tid, 0, token_part
                )
                count = cute.size(template)
                ring_slots = 2 if self.up_hidden_tile == 64 else self.hc_count
                residual = cute.make_rmem_tensor(
                    (count, ring_slots), original.element_type
                )
                scales = cute.make_rmem_tensor((count, ring_slots), cutlass.Float32)
                residual.fill(0.0)
                scales.fill(1.0)
                sums = cute.make_rmem_tensor(count, cutlass.Float32)
                sums.fill(0.0)
                first_row, first_col = coordinates[0]
                global_row = token_part * 128 + first_row
                hidden = hidden_tile * self.up_hidden_tile + first_col
                for branch in cutlass.range_constexpr(ring_slots):
                    self._up_prefetch(
                        original,
                        weight_scale,
                        activation_scale,
                        residual,
                        scales,
                        global_row,
                        hidden,
                        x.shape[0],
                        branch,
                        branch,
                        native_fp8,
                    )
                cute.arch.mbarrier_wait(done, 0)
                if tid % 128 // 32 * 32 < x.shape[0] - token_part * 128:
                    for branch in cutlass.range_constexpr(self.hc_count):
                        slot = branch % ring_slots
                        accumulator, reader, gates, _ = self._up_reader(
                            base, fragment.layout, original, tid, branch, token_part
                        )
                        cute_ext.partition_and_copy(reader, accumulator, gates)
                        cute.arch.fence_view_async_tmem_load()
                        for i in cutlass.range_constexpr(count):
                            sums[i] = sums[i] + residual[i, slot].to(
                                cutlass.Float32
                            ) * _up_sigmoid_tanh(gates[i] * scales[i, slot])
                        if cutlass.const_expr(branch + ring_slots < self.hc_count):
                            self._up_prefetch(
                                original,
                                weight_scale,
                                activation_scale,
                                residual,
                                scales,
                                global_row,
                                hidden,
                                x.shape[0],
                                branch + ring_slots,
                                slot,
                                native_fp8,
                            )
                tcgen05_fence("before_thread_sync")
                if global_row < x.shape[0]:
                    packed = cute.make_rmem_tensor(count, out.element_type)
                    for i in cutlass.range_constexpr(count):
                        packed[i] = (sums[i] / self.hc_count).to(out.element_type)
                    target = cute.make_tensor(
                        cute.make_ptr(
                            out.element_type,
                            (
                                out.iterator + global_row * self.hidden_size + hidden
                            ).toint(),
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        ),
                        cute.make_layout(count),
                    )
                    cute.copy(
                        cute.make_copy_atom(
                            cute.nvgpu.CopyUniversalOp(),
                            out.element_type,
                            num_bits_per_copy=128,
                        ),
                        packed,
                        target,
                    )
            cute.arch.mbarrier_arrive(read_done)


@cute.experimental.jit
def _up_residual_prefetch(
    original,
    residual,
    row,
    hidden,
    rows,
    branch: cutlass.Constexpr,
    slot: cutlass.Constexpr,
    wide: cutlass.Constexpr,
    hidden_size: cutlass.Constexpr,
):
    count = cute.size(residual, mode=[0])
    if row < rows:
        source = cute.make_tensor(
            cute.make_ptr(
                original.element_type,
                (
                    original.iterator + row * wide + branch * hidden_size + hidden
                ).toint(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            cute.make_layout(count),
        )
        cute.copy(
            cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                original.element_type,
                num_bits_per_copy=128,
            ),
            source,
            residual[None, slot],
        )


class W8GatedResidualCachedScaleUp(W8GatedResidualDecodedRingUp):
    """Reuse per-channel scales in shared memory without changing operand pipelines."""

    @cute.experimental.kernel
    def up_kernel(self, x, weight, original, weight_scale, activation_scale, out):
        tid = cute.arch.thread_idx()[0]
        warp = cute.arch.make_warp_uniform(tid // 32)
        hidden_tile, token_tile, _ = cute.arch.block_idx()
        dtype = x.element_type
        native_fp8 = cutlass.const_expr(dtype == cutlass.Float8E4M3FN)
        mma = sm100_utils.make_trivial_tiled_mma(
            dtype,
            dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            cutlass.Float32,
            tcgen05.CtaGroup.ONE,
            (self.up_token_rows, self.up_hidden_tile * self.hc_count),
        )
        tiler = (self.up_token_rows, self.up_hidden_tile * self.hc_count, self.k)
        activation_layout = sm100_utils.make_smem_layout_a(
            mma, tiler, dtype, self.input_stages
        )
        raw_layout = sm100_utils.make_smem_layout_b(
            mma, tiler, cutlass.Float8E4M3FN, self.raw_stages
        )
        weight_layout = sm100_utils.make_smem_layout_b(mma, tiler, dtype, self.stages)
        sx = cute_ext.allocate(
            dtype, cute.AddressSpace.smem, activation_layout, alignment=1024
        )
        assert not native_fp8
        raw = cute_ext.allocate(
            cutlass.Float8E4M3FN, cute.AddressSpace.smem, raw_layout, alignment=1024
        )
        sw = cute_ext.allocate(
            dtype, cute.AddressSpace.smem, weight_layout, alignment=1024
        )
        barriers = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(
                2 * self.raw_stages + 2 * self.stages + 2 * self.input_stages + 2
            ),
            alignment=8,
        ).iterator
        full = barriers
        raw_empty = full + self.raw_stages
        converted = raw_empty + self.raw_stages
        empty = converted + self.stages
        x_full = empty + self.stages
        x_empty = x_full + self.input_stages
        done, read_done = (x_empty + self.input_stages, x_empty + self.input_stages + 1)
        base = cute_ext.allocate(
            cutlass.Int32, cute.AddressSpace.smem, cute.make_layout(1), alignment=4
        ).iterator
        if warp == self.mma_warp:
            with cute.arch.elect_one():
                for stage in cutlass.range_constexpr(self.raw_stages):
                    cute.arch.mbarrier_init(full + stage, 1)
                    cute.arch.mbarrier_init(
                        raw_empty + stage, self.epilogue_threads // 32
                    )
                for stage in cutlass.range_constexpr(self.stages):
                    cute.arch.mbarrier_init(
                        converted + stage, self.epilogue_threads // 32
                    )
                    cute.arch.mbarrier_init(empty + stage, 1)
                for stage in cutlass.range_constexpr(self.input_stages):
                    cute.arch.mbarrier_init(x_full + stage, 1)
                    cute.arch.mbarrier_init(x_empty + stage, 1)
                cute.arch.mbarrier_init(done, 1)
                cute.arch.mbarrier_init(read_done, self.epilogue_threads)
            cute.arch.alloc_tmem(
                self.up_hidden_tile * self.hc_count, base, is_two_cta=False
            )
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        activation = cute.make_tensor(
            x.iterator,
            cute.make_layout(
                (x.shape[0], (64, self.inner // 64), 1), stride=(self.inner, (1, 64), 0)
            ),
        )
        weights = cute.make_tensor(
            weight.iterator,
            cute.make_layout(
                (
                    (
                        self.up_hidden_tile,
                        self.hc_count,
                        self.hidden_size // self.up_hidden_tile,
                    ),
                    (64, self.inner // 64),
                    1,
                ),
                stride=(
                    (
                        self.inner,
                        self.hidden_size * self.inner,
                        self.up_hidden_tile * self.inner,
                    ),
                    (1, 64),
                    0,
                ),
            ),
        )
        gx = cute.local_tile(
            activation, (self.up_token_rows, self.k), (token_tile, None, 0)
        )
        gw = cute.local_tile(
            weights,
            (self.up_hidden_tile * self.hc_count, self.k),
            (hidden_tile, None, 0),
        )
        map_x = cute_ext.get_cta_v_map_ab(activation, tiler, mma, "A")
        map_w = cute_ext.get_cta_v_map_ab(weights, tiler, mma, "B")
        if warp == self.producer_warp:
            if cutlass.const_expr(self.pdl and (not self.weights_independent)):
                cute.arch.griddepcontrol_wait()
            for tile in cutlass.range(self.tiles, unroll=1):
                raw_stage = tile % self.raw_stages
                if tile >= self.raw_stages:
                    cute.arch.mbarrier_wait(
                        raw_empty + raw_stage, (tile // self.raw_stages - 1) % 2
                    )
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        full + raw_stage, self.up_hidden_tile * self.hc_count * self.k
                    )
                cute_ext.tma_load(
                    gw[None, None, tile],
                    raw[None, None, None, raw_stage],
                    (full + raw_stage).value,
                    cta_v_map=map_w,
                    update_expect_tx=False,
                )
        elif warp == self.input_producer_warp:
            if cutlass.const_expr(self.pdl):
                cute.arch.griddepcontrol_wait()
                cute.arch.griddepcontrol_launch_dependents()
            for tile in cutlass.range(self.tiles, unroll=1):
                input_stage = tile % self.input_stages
                if tile >= self.input_stages:
                    cute.arch.mbarrier_wait(
                        x_empty + input_stage, (tile // self.input_stages - 1) % 2
                    )
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        x_full + input_stage, self.up_token_rows * self.k * 2
                    )
                cute_ext.tma_load(
                    gx[None, None, tile],
                    sx[None, None, None, input_stage],
                    (x_full + input_stage).value,
                    cta_v_map=map_x,
                    update_expect_tx=False,
                )
        elif warp == self.mma_warp:
            thread_mma = mma.get_slice(0)
            a_desc = thread_mma.make_fragment_A(sx)
            b_desc = thread_mma.make_fragment_B(sw)
            fragment = thread_mma.make_fragment_C(
                mma.partition_shape_C(
                    (self.up_token_rows, self.up_hidden_tile * self.hc_count)
                )
            )
            accumulator = cute.make_tensor(
                cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, base), fragment.layout
            )
            mma.set(tcgen05.Field.ACCUMULATE, False)
            for tile in cutlass.range(self.tiles, unroll=1):
                stage = tile % self.stages
                input_stage = tile % self.input_stages
                ready = full if cutlass.const_expr(native_fp8) else converted
                cute.arch.mbarrier_wait(ready + stage, tile // self.stages % 2)
                cute.arch.mbarrier_wait(
                    x_full + input_stage, tile // self.input_stages % 2
                )
                for block in cutlass.range_constexpr(cute.size(a_desc, mode=[2])):
                    cute.gemm(
                        mma,
                        accumulator,
                        a_desc[None, None, block, input_stage],
                        b_desc[None, None, block, stage],
                        accumulator,
                    )
                    mma.set(tcgen05.Field.ACCUMULATE, True)
                with cute.arch.elect_one():
                    tcgen05.commit(empty + stage, None, tcgen05.CtaGroup.ONE)
                    tcgen05.commit(x_empty + input_stage, None, tcgen05.CtaGroup.ONE)
            with cute.arch.elect_one():
                tcgen05.commit(done, None, tcgen05.CtaGroup.ONE)
            cute.arch.mbarrier_wait(read_done, 0)
            cute.arch.relinquish_tmem_alloc_permit()
            cute.arch.dealloc_tmem(
                cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, base),
                self.up_hidden_tile * self.hc_count,
            )
        elif tid < self.epilogue_threads:
            assert (
                self.up_hidden_tile == 64
                and self.up_token_rows == 128
                and (not native_fp8)
            )
            scale_cache = cute_ext.allocate(
                cutlass.Float32,
                cute.AddressSpace.smem,
                cute.make_layout(256),
                alignment=32,
            )
            if cutlass.const_expr(self.pdl and (not self.weights_independent)):
                cute.arch.griddepcontrol_wait()
            if tid < 256:
                channel = tid // 64 * self.hidden_size + hidden_tile * 64 + tid % 64
                scale_cache[tid] = weight_scale[
                    channel if self.weight_scale_per_channel else 0
                ]
            _epilogue_barrier(self.epilogue_threads, cutlass.Int32(3))
            if cutlass.const_expr(not native_fp8):
                for tile in cutlass.range(self.tiles, unroll=1):
                    raw_stage = tile % self.raw_stages
                    stage = tile % self.stages
                    cute.arch.mbarrier_wait(
                        full + raw_stage, tile // self.raw_stages % 2
                    )
                    if tile >= self.stages:
                        cute.arch.mbarrier_wait(
                            empty + stage, (tile // self.stages - 1) % 2
                        )
                    source = raw[None, None, None, raw_stage]
                    target = sw[None, None, None, stage]
                    packets = (
                        self.up_hidden_tile
                        * self.hc_count
                        * self.k
                        // (self.epilogue_threads * 8)
                    )
                    for offset in cutlass.range_constexpr(packets):
                        linear = (tid + offset * self.epilogue_threads) * 8
                        index = (
                            linear // self.k
                            + linear % self.k * self.up_hidden_tile * self.hc_count
                        )
                        src = cute.make_tensor(
                            cute.make_ptr(
                                cutlass.Float8E4M3FN,
                                (source.iterator + source.layout(index)).toint(),
                                cute.AddressSpace.smem,
                                assumed_align=8,
                                swizzle_=cute.get_swizzle_portion(raw_layout),
                            ),
                            cute.make_layout(8),
                        )
                        raw_values = cute.make_rmem_tensor(8, cutlass.Float8E4M3FN)
                        cute.copy(
                            cute.make_copy_atom(
                                cute.nvgpu.CopyUniversalOp(),
                                cutlass.Float8E4M3FN,
                                num_bits_per_copy=64,
                            ),
                            src,
                            raw_values,
                        )
                        decoded = cute.make_rmem_tensor(8, dtype)
                        _convert_fp8_weight(raw_values, decoded)
                        dst = cute.make_tensor(
                            cute.make_ptr(
                                dtype,
                                (target.iterator + target.layout(index)).toint(),
                                cute.AddressSpace.smem,
                                assumed_align=16,
                                swizzle_=cute.get_swizzle_portion(weight_layout),
                            ),
                            cute.make_layout(8),
                        )
                        cute.copy(
                            cute.make_copy_atom(
                                cute.nvgpu.CopyUniversalOp(),
                                dtype,
                                num_bits_per_copy=128,
                            ),
                            decoded,
                            dst,
                        )
                    cute.arch.fence_view_async_shared()
                    cute.arch.sync_warp(mask=4294967295)
                    if tid % 32 == 0:
                        cute.arch.mbarrier_arrive(raw_empty + raw_stage)
                        cute.arch.mbarrier_arrive(converted + stage)
            if cutlass.const_expr(self.pdl):
                cute.arch.griddepcontrol_wait()
            thread_mma = mma.get_slice(0)
            fragment = thread_mma.make_fragment_C(
                mma.partition_shape_C((128, self.up_hidden_tile * self.hc_count))
            )
            _, _, template, coordinates = self._up_reader(
                base, fragment.layout, original, tid, 0
            )
            count = cute.size(template)
            ring_slots = 2
            residual = cute.make_rmem_tensor((count, ring_slots), original.element_type)
            residual.fill(0.0)
            sums = cute.make_rmem_tensor(count, cutlass.Float32)
            sums.fill(0.0)
            first_row, first_col = coordinates[0]
            global_row = token_tile * 128 + first_row
            hidden = hidden_tile * self.up_hidden_tile + first_col
            for branch in cutlass.range_constexpr(ring_slots):
                _up_residual_prefetch(
                    original,
                    residual,
                    global_row,
                    hidden,
                    x.shape[0],
                    branch,
                    branch,
                    self.wide,
                    self.hidden_size,
                )
            cute.arch.mbarrier_wait(done, 0)
            if tid % 128 // 32 * 32 < x.shape[0] - token_tile * 128:
                for branch in cutlass.range_constexpr(self.hc_count):
                    slot = branch % ring_slots
                    accumulator, reader, gates, _ = self._up_reader(
                        base, fragment.layout, original, tid, branch
                    )
                    cute_ext.partition_and_copy(reader, accumulator, gates)
                    cute.arch.fence_view_async_tmem_load()
                    for i in cutlass.range_constexpr(count):
                        factor = scale_cache[branch * 64 + first_col + i]
                        sums[i] = sums[i] + residual[i, slot].to(
                            cutlass.Float32
                        ) * _up_sigmoid_tanh(gates[i] * factor)
                    if cutlass.const_expr(branch + ring_slots < self.hc_count):
                        _up_residual_prefetch(
                            original,
                            residual,
                            global_row,
                            hidden,
                            x.shape[0],
                            branch + ring_slots,
                            slot,
                            self.wide,
                            self.hidden_size,
                        )
            tcgen05_fence("before_thread_sync")
            cute.arch.mbarrier_arrive(read_done)
            if global_row < x.shape[0]:
                packed = cute.make_rmem_tensor(count, out.element_type)
                for i in cutlass.range_constexpr(count):
                    packed[i] = (sums[i] / self.hc_count).to(out.element_type)
                target = cute.make_tensor(
                    cute.make_ptr(
                        out.element_type,
                        (out.iterator + global_row * self.hidden_size + hidden).toint(),
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    ),
                    cute.make_layout(count),
                )
                cute.copy(
                    cute.make_copy_atom(
                        cute.nvgpu.CopyUniversalOp(),
                        out.element_type,
                        num_bits_per_copy=128,
                    ),
                    packed,
                    target,
                )


class W8GatedResidualCachedScaleTokenReuse(W8GatedResidualDecodedTokenReuse):
    """Reuse per-channel scales in shared memory without changing operand pipelines."""

    @cute.experimental.kernel
    def kernel(self, x, weight, original, weight_scale, activation_scale, out):
        tid = cute.arch.thread_idx()[0]
        warp = cute.arch.make_warp_uniform(tid // 32)
        hidden_tile, token_tile, _ = cute.arch.block_idx()
        dtype = x.element_type
        native_fp8 = cutlass.const_expr(dtype == cutlass.Float8E4M3FN)
        mma = sm100_utils.make_trivial_tiled_mma(
            dtype,
            dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            cutlass.Float32,
            tcgen05.CtaGroup.ONE,
            (128, self.up_hidden_tile * self.hc_count),
        )
        tiler = (128, self.up_hidden_tile * self.hc_count, self.k)
        activation_layout = sm100_utils.make_smem_layout_a(
            mma, tiler, dtype, self.input_stages
        )
        raw_layout = sm100_utils.make_smem_layout_b(
            mma, tiler, cutlass.Float8E4M3FN, self.raw_stages
        )
        weight_layout = sm100_utils.make_smem_layout_b(mma, tiler, dtype, self.stages)
        sx = cute_ext.allocate(
            dtype, cute.AddressSpace.smem, activation_layout, alignment=1024
        )
        assert not native_fp8
        raw = cute_ext.allocate(
            cutlass.Float8E4M3FN, cute.AddressSpace.smem, raw_layout, alignment=1024
        )
        sw = cute_ext.allocate(
            dtype, cute.AddressSpace.smem, weight_layout, alignment=1024
        )
        x1_layout = sm100_utils.make_smem_layout_a(mma, tiler, dtype, self.input_stages)
        sx1 = cute_ext.allocate(
            dtype, cute.AddressSpace.smem, x1_layout, alignment=1024
        )
        barriers = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(
                2 * self.raw_stages + 2 * self.stages + 4 * self.input_stages + 2
            ),
            alignment=8,
        ).iterator
        full = barriers
        raw_empty = full + self.raw_stages
        converted = raw_empty + self.raw_stages
        empty = converted + self.stages
        x0_full = empty + self.stages
        x0_empty = x0_full + self.input_stages
        x1_full = x0_empty + self.input_stages
        x1_empty = x1_full + self.input_stages
        done = x1_empty + self.input_stages
        read_done = done + 1
        base = cute_ext.allocate(
            cutlass.Int32, cute.AddressSpace.smem, cute.make_layout(1), alignment=4
        ).iterator
        if warp == self.mma_warp:
            with cute.arch.elect_one():
                for stage in cutlass.range_constexpr(self.raw_stages):
                    cute.arch.mbarrier_init(full + stage, 1)
                    cute.arch.mbarrier_init(
                        raw_empty + stage, self.epilogue_threads // 32
                    )
                for stage in cutlass.range_constexpr(self.stages):
                    cute.arch.mbarrier_init(
                        converted + stage, self.epilogue_threads // 32
                    )
                    cute.arch.mbarrier_init(empty + stage, 1)
                for stage in cutlass.range_constexpr(self.input_stages):
                    cute.arch.mbarrier_init(x0_full + stage, 1)
                    cute.arch.mbarrier_init(x0_empty + stage, 1)
                    cute.arch.mbarrier_init(x1_full + stage, 1)
                    cute.arch.mbarrier_init(x1_empty + stage, 1)
                cute.arch.mbarrier_init(done, 1)
                cute.arch.mbarrier_init(read_done, self.epilogue_threads)
            cute.arch.alloc_tmem(512, base, is_two_cta=False)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        activation = cute.make_tensor(
            x.iterator,
            cute.make_layout(
                (x.shape[0], (64, self.inner // 64), 1), stride=(self.inner, (1, 64), 0)
            ),
        )
        weights = cute.make_tensor(
            weight.iterator,
            cute.make_layout(
                (
                    (
                        self.up_hidden_tile,
                        self.hc_count,
                        self.hidden_size // self.up_hidden_tile,
                    ),
                    (64, self.inner // 64),
                    1,
                ),
                stride=(
                    (
                        self.inner,
                        self.hidden_size * self.inner,
                        self.up_hidden_tile * self.inner,
                    ),
                    (1, 64),
                    0,
                ),
            ),
        )
        gx = cute.local_tile(activation, (128, self.k), (token_tile, None, 0))
        gx1 = cute.local_tile(activation, (128, self.k), (1, None, 0))
        gw = cute.local_tile(
            weights,
            (self.up_hidden_tile * self.hc_count, self.k),
            (hidden_tile, None, 0),
        )
        map_x = cute_ext.get_cta_v_map_ab(activation, tiler, mma, "A")
        map_w = cute_ext.get_cta_v_map_ab(weights, tiler, mma, "B")
        if warp == self.producer_warp:
            if cutlass.const_expr(self.pdl and (not self.weights_independent)):
                cute.arch.griddepcontrol_wait()
            for tile in cutlass.range(self.tiles, unroll=1):
                raw_stage = tile % self.raw_stages
                if tile >= self.raw_stages:
                    cute.arch.mbarrier_wait(
                        raw_empty + raw_stage, (tile // self.raw_stages - 1) % 2
                    )
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        full + raw_stage, self.up_hidden_tile * self.hc_count * self.k
                    )
                cute_ext.tma_load(
                    gw[None, None, tile],
                    raw[None, None, None, raw_stage],
                    (full + raw_stage).value,
                    cta_v_map=map_w,
                    update_expect_tx=False,
                )
        elif warp == self.input_producer_warp:
            if cutlass.const_expr(self.pdl):
                cute.arch.griddepcontrol_wait()
                cute.arch.griddepcontrol_launch_dependents()
            for tile in cutlass.range(self.tiles, unroll=1):
                stage = tile % self.input_stages
                if tile >= self.input_stages:
                    cute.arch.mbarrier_wait(
                        x0_empty + stage, (tile // self.input_stages - 1) % 2
                    )
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        x0_full + stage, 128 * self.k * 2
                    )
                cute_ext.tma_load(
                    gx[None, None, tile],
                    sx[None, None, None, stage],
                    (x0_full + stage).value,
                    cta_v_map=map_x,
                    update_expect_tx=False,
                )
                if tile >= self.input_stages:
                    cute.arch.mbarrier_wait(
                        x1_empty + stage, (tile // self.input_stages - 1) % 2
                    )
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        x1_full + stage, 128 * self.k * 2
                    )
                cute_ext.tma_load(
                    gx1[None, None, tile],
                    sx1[None, None, None, stage],
                    (x1_full + stage).value,
                    cta_v_map=map_x,
                    update_expect_tx=False,
                )
        elif warp == self.mma_warp:
            thread_mma = mma.get_slice(0)
            a_desc = thread_mma.make_fragment_A(sx)
            b_desc = thread_mma.make_fragment_B(sw)
            fragment = thread_mma.make_fragment_C(
                mma.partition_shape_C((128, self.up_hidden_tile * self.hc_count))
            )
            accumulator = cute.make_tensor(
                cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, base), fragment.layout
            )
            a1_desc = thread_mma.make_fragment_A(sx1)
            accumulator1 = cute.make_tensor(
                cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, base) + 256,
                fragment.layout,
            )
            for tile in cutlass.range(self.tiles, unroll=1):
                stage = tile % self.stages
                input_stage = tile % self.input_stages
                input_phase = tile // self.input_stages % 2
                cute.arch.mbarrier_wait(converted + stage, tile // self.stages % 2)
                cute.arch.mbarrier_wait(x0_full + input_stage, input_phase)
                mma.set(tcgen05.Field.ACCUMULATE, tile > 0)
                for block in cutlass.range_constexpr(cute.size(a_desc, mode=[2])):
                    cute.gemm(
                        mma,
                        accumulator,
                        a_desc[None, None, block, input_stage],
                        b_desc[None, None, block, stage],
                        accumulator,
                    )
                    mma.set(tcgen05.Field.ACCUMULATE, True)
                with cute.arch.elect_one():
                    tcgen05.commit(x0_empty + input_stage, None, tcgen05.CtaGroup.ONE)
                cute.arch.mbarrier_wait(x1_full + input_stage, input_phase)
                mma.set(tcgen05.Field.ACCUMULATE, tile > 0)
                for block in cutlass.range_constexpr(cute.size(a1_desc, mode=[2])):
                    cute.gemm(
                        mma,
                        accumulator1,
                        a1_desc[None, None, block, input_stage],
                        b_desc[None, None, block, stage],
                        accumulator1,
                    )
                    mma.set(tcgen05.Field.ACCUMULATE, True)
                with cute.arch.elect_one():
                    tcgen05.commit(empty + stage, None, tcgen05.CtaGroup.ONE)
                    tcgen05.commit(x1_empty + input_stage, None, tcgen05.CtaGroup.ONE)
            with cute.arch.elect_one():
                tcgen05.commit(done, None, tcgen05.CtaGroup.ONE)
            cute.arch.mbarrier_wait(read_done, 0)
            cute.arch.relinquish_tmem_alloc_permit()
            cute.arch.dealloc_tmem(
                cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, base), 512
            )
        elif tid < self.epilogue_threads:
            assert (
                self.up_hidden_tile == 64
                and self.up_token_rows == 128
                and (not native_fp8)
            )
            scale_cache = cute_ext.allocate(
                cutlass.Float32,
                cute.AddressSpace.smem,
                cute.make_layout(256),
                alignment=32,
            )
            if cutlass.const_expr(self.pdl and (not self.weights_independent)):
                cute.arch.griddepcontrol_wait()
            if tid < 256:
                channel = tid // 64 * self.hidden_size + hidden_tile * 64 + tid % 64
                scale_cache[tid] = weight_scale[
                    channel if self.weight_scale_per_channel else 0
                ]
            _epilogue_barrier(self.epilogue_threads, cutlass.Int32(3))
            if cutlass.const_expr(not native_fp8):
                for tile in cutlass.range(self.tiles, unroll=1):
                    raw_stage = tile % self.raw_stages
                    stage = tile % self.stages
                    cute.arch.mbarrier_wait(
                        full + raw_stage, tile // self.raw_stages % 2
                    )
                    if tile >= self.stages:
                        cute.arch.mbarrier_wait(
                            empty + stage, (tile // self.stages - 1) % 2
                        )
                    source = raw[None, None, None, raw_stage]
                    target = sw[None, None, None, stage]
                    packets = (
                        self.up_hidden_tile
                        * self.hc_count
                        * self.k
                        // (self.epilogue_threads * 8)
                    )
                    for offset in cutlass.range_constexpr(packets):
                        linear = (tid + offset * self.epilogue_threads) * 8
                        index = (
                            linear // self.k
                            + linear % self.k * self.up_hidden_tile * self.hc_count
                        )
                        src = cute.make_tensor(
                            cute.make_ptr(
                                cutlass.Float8E4M3FN,
                                (source.iterator + source.layout(index)).toint(),
                                cute.AddressSpace.smem,
                                assumed_align=8,
                                swizzle_=cute.get_swizzle_portion(raw_layout),
                            ),
                            cute.make_layout(8),
                        )
                        raw_values = cute.make_rmem_tensor(8, cutlass.Float8E4M3FN)
                        cute.copy(
                            cute.make_copy_atom(
                                cute.nvgpu.CopyUniversalOp(),
                                cutlass.Float8E4M3FN,
                                num_bits_per_copy=64,
                            ),
                            src,
                            raw_values,
                        )
                        decoded = cute.make_rmem_tensor(8, dtype)
                        _convert_fp8_weight(raw_values, decoded)
                        dst = cute.make_tensor(
                            cute.make_ptr(
                                dtype,
                                (target.iterator + target.layout(index)).toint(),
                                cute.AddressSpace.smem,
                                assumed_align=16,
                                swizzle_=cute.get_swizzle_portion(weight_layout),
                            ),
                            cute.make_layout(8),
                        )
                        cute.copy(
                            cute.make_copy_atom(
                                cute.nvgpu.CopyUniversalOp(),
                                dtype,
                                num_bits_per_copy=128,
                            ),
                            decoded,
                            dst,
                        )
                    cute.arch.fence_view_async_shared()
                    cute.arch.sync_warp(mask=4294967295)
                    if tid % 32 == 0:
                        cute.arch.mbarrier_arrive(raw_empty + raw_stage)
                        cute.arch.mbarrier_arrive(converted + stage)
            if cutlass.const_expr(self.pdl):
                cute.arch.griddepcontrol_wait()
            for token_part in cutlass.range(2, unroll=1):
                thread_mma = mma.get_slice(0)
                fragment = thread_mma.make_fragment_C(
                    mma.partition_shape_C((128, self.up_hidden_tile * self.hc_count))
                )
                _, _, template, coordinates = self._up_reader(
                    base, fragment.layout, original, tid, 0, token_part
                )
                count = cute.size(template)
                ring_slots = 2
                residual = cute.make_rmem_tensor(
                    (count, ring_slots), original.element_type
                )
                residual.fill(0.0)
                sums = cute.make_rmem_tensor(count, cutlass.Float32)
                sums.fill(0.0)
                first_row, first_col = coordinates[0]
                global_row = token_part * 128 + first_row
                hidden = hidden_tile * self.up_hidden_tile + first_col
                for branch in cutlass.range_constexpr(ring_slots):
                    _up_residual_prefetch(
                        original,
                        residual,
                        global_row,
                        hidden,
                        x.shape[0],
                        branch,
                        branch,
                        self.wide,
                        self.hidden_size,
                    )
                cute.arch.mbarrier_wait(done, 0)
                if tid % 128 // 32 * 32 < x.shape[0] - token_part * 128:
                    for branch in cutlass.range_constexpr(self.hc_count):
                        slot = branch % ring_slots
                        accumulator, reader, gates, _ = self._up_reader(
                            base, fragment.layout, original, tid, branch, token_part
                        )
                        cute_ext.partition_and_copy(reader, accumulator, gates)
                        cute.arch.fence_view_async_tmem_load()
                        for i in cutlass.range_constexpr(count):
                            factor = scale_cache[branch * 64 + first_col + i]
                            sums[i] = sums[i] + residual[i, slot].to(
                                cutlass.Float32
                            ) * _up_sigmoid_tanh(gates[i] * factor)
                        if cutlass.const_expr(branch + ring_slots < self.hc_count):
                            _up_residual_prefetch(
                                original,
                                residual,
                                global_row,
                                hidden,
                                x.shape[0],
                                branch + ring_slots,
                                slot,
                                self.wide,
                                self.hidden_size,
                            )
                tcgen05_fence("before_thread_sync")
                if global_row < x.shape[0]:
                    packed = cute.make_rmem_tensor(count, out.element_type)
                    for i in cutlass.range_constexpr(count):
                        packed[i] = (sums[i] / self.hc_count).to(out.element_type)
                    target = cute.make_tensor(
                        cute.make_ptr(
                            out.element_type,
                            (
                                out.iterator + global_row * self.hidden_size + hidden
                            ).toint(),
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        ),
                        cute.make_layout(count),
                    )
                    cute.copy(
                        cute.make_copy_atom(
                            cute.nvgpu.CopyUniversalOp(),
                            out.element_type,
                            num_bits_per_copy=128,
                        ),
                        packed,
                        target,
                    )
            cute.arch.mbarrier_arrive(read_done)


class Fp8GatedResidualTokenReuseUp(Fp8GatedResidualGemm):
    """Reuse resident native-FP8 weights across two independent token tiles."""

    @cute.experimental.jit
    def __call__(
        self,
        x,
        weight,
        original,
        weight_scale,
        activation_scale,
        partials,
        out,
        stream: CUstream,
    ):
        self.up_kernel(x, weight, original, weight_scale, activation_scale, out).launch(
            grid=(
                self.hidden_size // self.up_hidden_tile,
                cute.ceil_div(x.shape[0], 256),
                1,
            ),
            block=(self.epilogue_threads + 64, 1, 1),
            stream=stream,
            use_pdl=self.pdl,
        )

    @cute.experimental.jit
    def _up_prefetch(
        self,
        original,
        weight_scale,
        activation_scale,
        residual,
        scales,
        row,
        hidden,
        rows,
        branch: cutlass.Constexpr,
        slot: cutlass.Constexpr,
        native_fp8: cutlass.Constexpr,
    ):
        count = cute.size(residual, mode=[0])
        if row < rows:
            source = cute.make_tensor(
                cute.make_ptr(
                    original.element_type,
                    (
                        original.iterator
                        + cutlass.Int64(row) * self.wide
                        + branch * self.hidden_size
                        + hidden
                    ).toint(),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                ),
                cute.make_layout(count),
            )
            cute.copy(
                cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(),
                    original.element_type,
                    num_bits_per_copy=128,
                ),
                source,
                residual[None, slot],
            )
            if cutlass.const_expr(not self.shared_up_scales):
                if cutlass.const_expr(
                    self.weight_scale_per_channel and self.weight_scale_vectorized
                ):
                    source_scale = cute.make_tensor(
                        cute.make_ptr(
                            cutlass.Float32,
                            (
                                weight_scale.iterator
                                + branch * self.hidden_size
                                + hidden
                            ).toint(),
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        ),
                        cute.make_layout(count),
                    )
                    cute.copy(
                        cute.make_copy_atom(
                            cute.nvgpu.CopyUniversalOp(),
                            cutlass.Float32,
                            num_bits_per_copy=128,
                        ),
                        source_scale,
                        scales[None, slot],
                    )
                else:
                    for i in cutlass.range_constexpr(count):
                        scales[i, slot] = weight_scale[
                            (
                                branch * self.hidden_size + hidden + i
                                if self.weight_scale_per_channel
                                else 0
                            )
                        ]
                if cutlass.const_expr(native_fp8):
                    for i in cutlass.range_constexpr(count):
                        scales[i, slot] = scales[i, slot] * activation_scale[row]

    @cute.experimental.jit
    def _up_reader(
        self,
        base,
        layout,
        original,
        tid,
        branch: cutlass.Constexpr,
        token_part: cutlass.Int32,
    ):
        tile = (self.up_token_rows, self.up_hidden_tile // self.epilogue_groups)
        group = tid // 128 + branch * self.epilogue_groups
        accumulator = cute.make_tensor(
            cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, base) + token_part * 256,
            layout,
        )[(None, None), 0, 0]
        accumulator = cute.local_tile(accumulator, tile, (0, group))
        coordinates = cute.local_tile(
            cute.make_identity_tensor(
                (self.up_token_rows, self.up_hidden_tile * self.hc_count)
            ),
            tile,
            (0, group),
        )
        copy = tcgen05.make_tmem_copy(
            sm100_utils.get_tmem_load_op(
                (self.up_token_rows, self.up_hidden_tile * self.hc_count, self.k),
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
            original.iterator, cute.make_layout(tile, stride=(1, self.up_token_rows))
        )
        values = cute_ext.allocate(
            cutlass.Float32,
            cute.AddressSpace.rmem,
            cute_ext.make_t2r_rmem_layout(
                copy, cute.flat_divide(dummy, tile), tid % 128
            ),
            alignment=32,
        )
        return (accumulator, reader, values, reader.partition_D(coordinates))

    @cute.experimental.kernel
    def up_kernel(self, x, weight, original, weight_scale, activation_scale, out):
        tid = cute.arch.thread_idx()[0]
        warp = cute.arch.make_warp_uniform(tid // 32)
        hidden_tile, token_tile, _ = cute.arch.block_idx()
        dtype = x.element_type
        native_fp8 = cutlass.const_expr(dtype == cutlass.Float8E4M3FN)
        mma = sm100_utils.make_trivial_tiled_mma(
            dtype,
            dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            cutlass.Float32,
            tcgen05.CtaGroup.ONE,
            (self.up_token_rows, self.up_hidden_tile * self.hc_count),
        )
        tiler = (self.up_token_rows, self.up_hidden_tile * self.hc_count, self.k)
        activation_layout = sm100_utils.make_smem_layout_a(
            mma, tiler, dtype, self.stages
        )
        raw_layout = sm100_utils.make_smem_layout_b(
            mma, tiler, cutlass.Float8E4M3FN, self.stages
        )
        weight_layout = sm100_utils.make_smem_layout_b(mma, tiler, dtype, self.stages)
        sx = cute_ext.allocate(
            dtype, cute.AddressSpace.smem, activation_layout, alignment=1024
        )
        sx1 = cute_ext.allocate(
            dtype, cute.AddressSpace.smem, activation_layout, alignment=1024
        )
        raw = cute_ext.allocate(
            cutlass.Float8E4M3FN, cute.AddressSpace.smem, raw_layout, alignment=1024
        )
        sw = raw
        if cutlass.const_expr(not native_fp8):
            sw = cute_ext.allocate(
                dtype, cute.AddressSpace.smem, weight_layout, alignment=1024
            )
        barriers = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(3 * self.stages + 2),
            alignment=8,
        ).iterator
        full = barriers
        converted = full + self.stages
        empty = converted + self.stages
        done, read_done = (empty + self.stages, empty + self.stages + 1)
        base = cute_ext.allocate(
            cutlass.Int32, cute.AddressSpace.smem, cute.make_layout(1), alignment=4
        ).iterator
        if warp == self.mma_warp:
            with cute.arch.elect_one():
                for stage in cutlass.range_constexpr(self.stages):
                    cute.arch.mbarrier_init(full + stage, 1)
                    cute.arch.mbarrier_init(converted + stage, 1)
                    cute.arch.mbarrier_init(empty + stage, 1)
                cute.arch.mbarrier_init(done, 1)
                cute.arch.mbarrier_init(read_done, self.epilogue_threads)
            cute.arch.alloc_tmem(512, base, is_two_cta=False)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        activation = cute.make_tensor(
            x.iterator,
            cute.make_layout(
                (x.shape[0], (64, self.inner // 64), 1), stride=(self.inner, (1, 64), 0)
            ),
        )
        weights = cute.make_tensor(
            weight.iterator,
            cute.make_layout(
                (
                    (
                        self.up_hidden_tile,
                        self.hc_count,
                        self.hidden_size // self.up_hidden_tile,
                    ),
                    (64, self.inner // 64),
                    1,
                ),
                stride=(
                    (
                        self.inner,
                        self.hidden_size * self.inner,
                        self.up_hidden_tile * self.inner,
                    ),
                    (1, 64),
                    0,
                ),
            ),
        )
        gx = cute.local_tile(
            activation, (self.up_token_rows, self.k), (token_tile * 2, None, 0)
        )
        gx1 = cute.local_tile(
            activation, (self.up_token_rows, self.k), (token_tile * 2 + 1, None, 0)
        )
        gw = cute.local_tile(
            weights,
            (self.up_hidden_tile * self.hc_count, self.k),
            (hidden_tile, None, 0),
        )
        map_x = cute_ext.get_cta_v_map_ab(activation, tiler, mma, "A")
        map_w = cute_ext.get_cta_v_map_ab(weights, tiler, mma, "B")
        if warp == self.producer_warp:
            if cutlass.const_expr(self.pdl):
                for stage in cutlass.range_constexpr(self.stages):
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            full + stage,
                            (
                                self.up_hidden_tile * self.hc_count
                                + 2 * self.up_token_rows
                            )
                            * self.k,
                        )
                    if cutlass.const_expr(self.weights_independent):
                        cute_ext.tma_load(
                            gw[None, None, stage],
                            raw[None, None, None, stage],
                            (full + stage).value,
                            cta_v_map=map_w,
                            update_expect_tx=False,
                        )
                cute.arch.griddepcontrol_wait()
                cute.arch.griddepcontrol_launch_dependents()
                for stage in cutlass.range_constexpr(self.stages):
                    if cutlass.const_expr(not self.weights_independent):
                        cute_ext.tma_load(
                            gw[None, None, stage],
                            raw[None, None, None, stage],
                            (full + stage).value,
                            cta_v_map=map_w,
                            update_expect_tx=False,
                        )
                    cute_ext.tma_load(
                        gx[None, None, stage],
                        sx[None, None, None, stage],
                        (full + stage).value,
                        cta_v_map=map_x,
                        update_expect_tx=False,
                    )
                    cute_ext.tma_load(
                        gx1[None, None, stage],
                        sx1[None, None, None, stage],
                        (full + stage).value,
                        cta_v_map=map_x,
                        update_expect_tx=False,
                    )
            else:
                for stage in cutlass.range_constexpr(self.stages):
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            full + stage,
                            (
                                self.up_hidden_tile * self.hc_count
                                + 2 * self.up_token_rows
                            )
                            * self.k,
                        )
                    cute_ext.tma_load(
                        gw[None, None, stage],
                        raw[None, None, None, stage],
                        (full + stage).value,
                        cta_v_map=map_w,
                        update_expect_tx=False,
                    )
                    cute_ext.tma_load(
                        gx[None, None, stage],
                        sx[None, None, None, stage],
                        (full + stage).value,
                        cta_v_map=map_x,
                        update_expect_tx=False,
                    )
                    cute_ext.tma_load(
                        gx1[None, None, stage],
                        sx1[None, None, None, stage],
                        (full + stage).value,
                        cta_v_map=map_x,
                        update_expect_tx=False,
                    )
            for tile in cutlass.range(self.stages, self.tiles, unroll=1):
                stage = tile % self.stages
                cute.arch.mbarrier_wait(empty + stage, (tile // self.stages - 1) % 2)
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        full + stage,
                        (self.up_hidden_tile * self.hc_count + 2 * self.up_token_rows)
                        * self.k,
                    )
                cute_ext.tma_load(
                    gw[None, None, tile],
                    raw[None, None, None, stage],
                    (full + stage).value,
                    cta_v_map=map_w,
                    update_expect_tx=False,
                )
                cute_ext.tma_load(
                    gx[None, None, tile],
                    sx[None, None, None, stage],
                    (full + stage).value,
                    cta_v_map=map_x,
                    update_expect_tx=False,
                )
                cute_ext.tma_load(
                    gx1[None, None, tile],
                    sx1[None, None, None, stage],
                    (full + stage).value,
                    cta_v_map=map_x,
                    update_expect_tx=False,
                )
        elif warp == self.mma_warp:
            thread_mma = mma.get_slice(0)
            a_desc = thread_mma.make_fragment_A(sx)
            a1_desc = thread_mma.make_fragment_A(sx1)
            b_desc = thread_mma.make_fragment_B(sw)
            fragment = thread_mma.make_fragment_C(
                mma.partition_shape_C(
                    (self.up_token_rows, self.up_hidden_tile * self.hc_count)
                )
            )
            accumulator = cute.make_tensor(
                cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, base), fragment.layout
            )
            accumulator1 = cute.make_tensor(
                cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, base) + 256,
                fragment.layout,
            )
            for tile in cutlass.range(self.tiles, unroll=1):
                stage = tile % self.stages
                cute.arch.mbarrier_wait(full + stage, tile // self.stages % 2)
                mma.set(tcgen05.Field.ACCUMULATE, tile > 0)
                for block in cutlass.range_constexpr(cute.size(a_desc, mode=[2])):
                    cute.gemm(
                        mma,
                        accumulator,
                        a_desc[None, None, block, stage],
                        b_desc[None, None, block, stage],
                        accumulator,
                    )
                    mma.set(tcgen05.Field.ACCUMULATE, True)
                mma.set(tcgen05.Field.ACCUMULATE, tile > 0)
                for block in cutlass.range_constexpr(cute.size(a1_desc, mode=[2])):
                    cute.gemm(
                        mma,
                        accumulator1,
                        a1_desc[None, None, block, stage],
                        b_desc[None, None, block, stage],
                        accumulator1,
                    )
                    mma.set(tcgen05.Field.ACCUMULATE, True)
                with cute.arch.elect_one():
                    tcgen05.commit(empty + stage, None, tcgen05.CtaGroup.ONE)
            with cute.arch.elect_one():
                tcgen05.commit(done, None, tcgen05.CtaGroup.ONE)
            cute.arch.mbarrier_wait(read_done, 0)
            cute.arch.relinquish_tmem_alloc_permit()
            cute.arch.dealloc_tmem(
                cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, base), 512
            )
        elif tid < self.epilogue_threads:
            if cutlass.const_expr(self.pdl):
                cute.arch.griddepcontrol_wait()
            scale_cache = cute_ext.allocate(
                cutlass.Float32,
                cute.AddressSpace.smem,
                cute.make_layout(256),
                alignment=32,
            )
            if tid < 256:
                channel = tid // 64 * self.hidden_size + hidden_tile * 64 + tid % 64
                scale_cache[tid] = weight_scale[
                    channel if self.weight_scale_per_channel else 0
                ]
            _epilogue_barrier(self.epilogue_threads, cutlass.Int32(3))
            for token_part in cutlass.range(2, unroll=1):
                thread_mma = mma.get_slice(0)
                fragment = thread_mma.make_fragment_C(
                    mma.partition_shape_C(
                        (self.up_token_rows, self.up_hidden_tile * self.hc_count)
                    )
                )
                _, _, template, coordinates = self._up_reader(
                    base, fragment.layout, original, tid, 0, token_part
                )
                count = cute.size(template)
                ring_slots = (
                    self.hc_count
                    if self.shared_up_scales
                    else 2 if self.up_hidden_tile == 64 else self.hc_count
                )
                residual = cute.make_rmem_tensor(
                    (count, ring_slots), original.element_type
                )
                scales = cute.make_rmem_tensor(
                    (1, 1) if self.shared_up_scales else (count, ring_slots),
                    cutlass.Float32,
                )
                residual.fill(0.0)
                scales.fill(1.0)
                sums = cute.make_rmem_tensor(count, cutlass.Float32)
                sums.fill(0.0)
                first_row, first_col = coordinates[0]
                global_row = (
                    cutlass.Int64(token_tile) * 2 + token_part
                ) * self.up_token_rows + first_row
                hidden = hidden_tile * self.up_hidden_tile + first_col
                alpha = cutlass.Float32(1.0)
                if cutlass.const_expr(self.shared_up_scales):
                    if cutlass.const_expr(native_fp8):
                        if global_row < x.shape[0]:
                            alpha = activation_scale[global_row]
                if cutlass.const_expr(self.up_token_rows == 64):
                    for branch in cutlass.range_constexpr(self.hc_count):
                        self._up_fragment_prefetch(
                            original,
                            weight_scale,
                            activation_scale,
                            residual,
                            scales,
                            coordinates,
                            token_tile,
                            hidden_tile,
                            x.shape[0],
                            branch,
                        )
                else:
                    for branch in cutlass.range_constexpr(ring_slots):
                        self._up_prefetch(
                            original,
                            weight_scale,
                            activation_scale,
                            residual,
                            scales,
                            global_row,
                            hidden,
                            x.shape[0],
                            branch,
                            branch,
                            native_fp8,
                        )
                cute.arch.mbarrier_wait(done, 0)
                if (
                    tid % 128 // 32 * (self.up_token_rows // 4)
                    < x.shape[0] - (token_tile * 2 + token_part) * self.up_token_rows
                ):
                    for branch in cutlass.range_constexpr(self.hc_count):
                        slot = branch % ring_slots
                        accumulator, reader, gates, _ = self._up_reader(
                            base, fragment.layout, original, tid, branch, token_part
                        )
                        cute_ext.partition_and_copy(reader, accumulator, gates)
                        cute.arch.fence_view_async_tmem_load()
                        for i in cutlass.range_constexpr(count):
                            factor = cutlass.Float32(1.0)
                            if cutlass.const_expr(self.shared_up_scales):
                                factor = (
                                    scale_cache[branch * 64 + first_col + i] * alpha
                                )
                            else:
                                factor = scales[i, slot]
                            sums[i] = sums[i] + residual[i, slot].to(
                                cutlass.Float32
                            ) * _up_sigmoid_tanh(gates[i] * factor)
                        if cutlass.const_expr(branch + ring_slots < self.hc_count):
                            self._up_prefetch(
                                original,
                                weight_scale,
                                activation_scale,
                                residual,
                                scales,
                                global_row,
                                hidden,
                                x.shape[0],
                                branch + ring_slots,
                                slot,
                                native_fp8,
                            )
                tcgen05_fence("before_thread_sync")
                if cutlass.const_expr(self.up_token_rows == 64):
                    for packet in cutlass.range_constexpr(count // 2):
                        row, col = coordinates[packet * 2]
                        output_row = (
                            token_tile * 2 + token_part
                        ) * self.up_token_rows + row
                        if output_row < x.shape[0]:
                            packed = cute.make_rmem_tensor(2, out.element_type)
                            for j in cutlass.range_constexpr(2):
                                packed[j] = (sums[packet * 2 + j] / self.hc_count).to(
                                    out.element_type
                                )
                            target = cute.make_tensor(
                                cute.make_ptr(
                                    out.element_type,
                                    (
                                        out.iterator
                                        + output_row * self.hidden_size
                                        + hidden_tile * 32
                                        + col
                                    ).toint(),
                                    cute.AddressSpace.gmem,
                                    assumed_align=4,
                                ),
                                cute.make_layout(2),
                            )
                            cute.copy(
                                cute.make_copy_atom(
                                    cute.nvgpu.CopyUniversalOp(),
                                    out.element_type,
                                    num_bits_per_copy=32,
                                ),
                                packed,
                                target,
                            )
                elif global_row < x.shape[0]:
                    packed = cute.make_rmem_tensor(count, out.element_type)
                    for i in cutlass.range_constexpr(count):
                        packed[i] = (sums[i] / self.hc_count).to(out.element_type)
                    target = cute.make_tensor(
                        cute.make_ptr(
                            out.element_type,
                            (
                                out.iterator
                                + cutlass.Int64(global_row) * self.hidden_size
                                + hidden
                            ).toint(),
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        ),
                        cute.make_layout(count),
                    )
                    cute.copy(
                        cute.make_copy_atom(
                            cute.nvgpu.CopyUniversalOp(),
                            out.element_type,
                            num_bits_per_copy=128,
                        ),
                        packed,
                        target,
                    )
            cute.arch.mbarrier_arrive(read_done)


class Fp8GatedResidualM64H64Up(Fp8GatedResidualGemm):
    """Native M64/N256 Up with four fragment-correct HC branches and a register fold."""

    @cute.experimental.jit
    def _up_fragment_prefetch(
        self,
        original,
        weight_scale,
        activation_scale,
        residual,
        scales,
        coordinates,
        token_tile,
        hidden_tile,
        rows,
        branch: cutlass.Constexpr,
    ):
        count = cute.size(residual, mode=[0])
        for packet in cutlass.range_constexpr(count // 2):
            row, col = coordinates[packet * 2]
            input_row = token_tile * self.up_token_rows + row
            hidden = hidden_tile * self.up_hidden_tile + col
            if input_row < rows:
                source = cute.make_tensor(
                    cute.make_ptr(
                        original.element_type,
                        (
                            original.iterator
                            + input_row * self.wide
                            + branch * self.hidden_size
                            + hidden
                        ).toint(),
                        cute.AddressSpace.gmem,
                        assumed_align=4,
                    ),
                    cute.make_layout(2),
                )
                values = cute.make_rmem_tensor(2, original.element_type)
                cute.copy(
                    cute.make_copy_atom(
                        cute.nvgpu.CopyUniversalOp(),
                        original.element_type,
                        num_bits_per_copy=32,
                    ),
                    source,
                    values,
                )
                for j in cutlass.range_constexpr(2):
                    residual[packet * 2 + j, branch] = values[j]
                    scales[packet * 2 + j, branch] = (
                        weight_scale[
                            (
                                branch * self.hidden_size + hidden + j
                                if self.weight_scale_per_channel
                                else 0
                            )
                        ]
                        * activation_scale[input_row]
                    )

    @cute.experimental.kernel
    def up_kernel(self, x, weight, original, weight_scale, activation_scale, out):
        tid = cute.arch.thread_idx()[0]
        warp = cute.arch.make_warp_uniform(tid // 32)
        hidden_tile, token_tile, _ = cute.arch.block_idx()
        token_tile = cutlass.Int64(token_tile)
        dtype = x.element_type
        native_fp8 = cutlass.const_expr(dtype == cutlass.Float8E4M3FN)
        mma = sm100_utils.make_trivial_tiled_mma(
            dtype,
            dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            cutlass.Float32,
            tcgen05.CtaGroup.ONE,
            (self.up_token_rows, self.up_hidden_tile * self.hc_count),
        )
        tiler = (self.up_token_rows, self.up_hidden_tile * self.hc_count, self.k)
        activation_layout = sm100_utils.make_smem_layout_a(
            mma, tiler, dtype, self.stages
        )
        raw_layout = sm100_utils.make_smem_layout_b(
            mma, tiler, cutlass.Float8E4M3FN, self.stages
        )
        weight_layout = sm100_utils.make_smem_layout_b(mma, tiler, dtype, self.stages)
        sx = cute_ext.allocate(
            dtype, cute.AddressSpace.smem, activation_layout, alignment=1024
        )
        raw = cute_ext.allocate(
            cutlass.Float8E4M3FN, cute.AddressSpace.smem, raw_layout, alignment=1024
        )
        sw = raw
        if cutlass.const_expr(not native_fp8):
            sw = cute_ext.allocate(
                dtype, cute.AddressSpace.smem, weight_layout, alignment=1024
            )
        barriers = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(3 * self.stages + 2),
            alignment=8,
        ).iterator
        full = barriers
        converted = full + self.stages
        empty = converted + self.stages
        done, read_done = (empty + self.stages, empty + self.stages + 1)
        base = cute_ext.allocate(
            cutlass.Int32, cute.AddressSpace.smem, cute.make_layout(1), alignment=4
        ).iterator
        if warp == self.mma_warp:
            with cute.arch.elect_one():
                for stage in cutlass.range_constexpr(self.stages):
                    cute.arch.mbarrier_init(full + stage, 1)
                    cute.arch.mbarrier_init(converted + stage, 1)
                    cute.arch.mbarrier_init(empty + stage, 1)
                cute.arch.mbarrier_init(done, 1)
                cute.arch.mbarrier_init(read_done, self.epilogue_threads)
            cute.arch.alloc_tmem(
                self.up_hidden_tile * self.hc_count, base, is_two_cta=False
            )
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        activation = cute.make_tensor(
            x.iterator,
            cute.make_layout(
                (x.shape[0], (64, self.inner // 64), 1), stride=(self.inner, (1, 64), 0)
            ),
        )
        weights = cute.make_tensor(
            weight.iterator,
            cute.make_layout(
                (
                    (
                        self.up_hidden_tile,
                        self.hc_count,
                        self.hidden_size // self.up_hidden_tile,
                    ),
                    (64, self.inner // 64),
                    1,
                ),
                stride=(
                    (
                        self.inner,
                        self.hidden_size * self.inner,
                        self.up_hidden_tile * self.inner,
                    ),
                    (1, 64),
                    0,
                ),
            ),
        )
        gx = cute.local_tile(
            activation, (self.up_token_rows, self.k), (token_tile, None, 0)
        )
        gw = cute.local_tile(
            weights,
            (self.up_hidden_tile * self.hc_count, self.k),
            (hidden_tile, None, 0),
        )
        map_x = cute_ext.get_cta_v_map_ab(activation, tiler, mma, "A")
        map_w = cute_ext.get_cta_v_map_ab(weights, tiler, mma, "B")
        if warp == self.producer_warp:
            for stage in cutlass.range_constexpr(self.stages):
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        full + stage,
                        self.up_hidden_tile * self.hc_count * self.k
                        + self.up_token_rows * self.k * (dtype.width // 8),
                    )
                if cutlass.const_expr(self.weights_independent):
                    cute_ext.tma_load(
                        gw[None, None, stage],
                        raw[None, None, None, stage],
                        (full + stage).value,
                        cta_v_map=map_w,
                        update_expect_tx=False,
                    )
            if cutlass.const_expr(self.pdl):
                cute.arch.griddepcontrol_wait()
                cute.arch.griddepcontrol_launch_dependents()
            for stage in cutlass.range_constexpr(self.stages):
                if cutlass.const_expr(not self.weights_independent):
                    cute_ext.tma_load(
                        gw[None, None, stage],
                        raw[None, None, None, stage],
                        (full + stage).value,
                        cta_v_map=map_w,
                        update_expect_tx=False,
                    )
                cute_ext.tma_load(
                    gx[None, None, stage],
                    sx[None, None, None, stage],
                    (full + stage).value,
                    cta_v_map=map_x,
                    update_expect_tx=False,
                )
            for tile in cutlass.range(self.stages, self.tiles, unroll=1):
                stage = tile % self.stages
                cute.arch.mbarrier_wait(empty + stage, (tile // self.stages - 1) % 2)
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        full + stage,
                        self.up_hidden_tile * self.hc_count * self.k
                        + self.up_token_rows * self.k * (dtype.width // 8),
                    )
                cute_ext.tma_load(
                    gw[None, None, tile],
                    raw[None, None, None, stage],
                    (full + stage).value,
                    cta_v_map=map_w,
                    update_expect_tx=False,
                )
                cute_ext.tma_load(
                    gx[None, None, tile],
                    sx[None, None, None, stage],
                    (full + stage).value,
                    cta_v_map=map_x,
                    update_expect_tx=False,
                )
        elif warp == self.mma_warp:
            thread_mma = mma.get_slice(0)
            a_desc = thread_mma.make_fragment_A(sx)
            b_desc = thread_mma.make_fragment_B(sw)
            fragment = thread_mma.make_fragment_C(
                mma.partition_shape_C(
                    (self.up_token_rows, self.up_hidden_tile * self.hc_count)
                )
            )
            accumulator = cute.make_tensor(
                cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, base), fragment.layout
            )
            mma.set(tcgen05.Field.ACCUMULATE, False)
            for tile in cutlass.range(self.tiles, unroll=1):
                stage = tile % self.stages
                ready = full if cutlass.const_expr(native_fp8) else converted
                cute.arch.mbarrier_wait(ready + stage, tile // self.stages % 2)
                for block in cutlass.range_constexpr(cute.size(a_desc, mode=[2])):
                    cute.gemm(
                        mma,
                        accumulator,
                        a_desc[None, None, block, stage],
                        b_desc[None, None, block, stage],
                        accumulator,
                    )
                    mma.set(tcgen05.Field.ACCUMULATE, True)
                with cute.arch.elect_one():
                    tcgen05.commit(empty + stage, None, tcgen05.CtaGroup.ONE)
            with cute.arch.elect_one():
                tcgen05.commit(done, None, tcgen05.CtaGroup.ONE)
            cute.arch.mbarrier_wait(read_done, 0)
            cute.arch.relinquish_tmem_alloc_permit()
            cute.arch.dealloc_tmem(
                cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, base),
                self.up_hidden_tile * self.hc_count,
            )
        elif tid < self.epilogue_threads:
            if cutlass.const_expr(not native_fp8):
                for tile in cutlass.range(self.tiles, unroll=1):
                    stage = tile % self.stages
                    cute.arch.mbarrier_wait(full + stage, tile // self.stages % 2)
                    source, target = (
                        raw[None, None, None, stage],
                        sw[None, None, None, stage],
                    )
                    for offset in cutlass.range(
                        128 * self.k // (self.epilogue_threads * 8), unroll=4
                    ):
                        linear = (tid + offset * self.epilogue_threads) * 8
                        index = linear // self.k + linear % self.k * 128
                        src = cute.make_tensor(
                            cute.make_ptr(
                                cutlass.Float8E4M3FN,
                                (source.iterator + source.layout(index)).toint(),
                                cute.AddressSpace.smem,
                                assumed_align=8,
                                swizzle_=cute.get_swizzle_portion(raw_layout),
                            ),
                            cute.make_layout(8),
                        )
                        dst = cute.make_tensor(
                            cute.make_ptr(
                                dtype,
                                (target.iterator + target.layout(index)).toint(),
                                cute.AddressSpace.smem,
                                assumed_align=16,
                                swizzle_=cute.get_swizzle_portion(weight_layout),
                            ),
                            cute.make_layout(8),
                        )
                        raw_values = cute.make_rmem_tensor(8, cutlass.Float8E4M3FN)
                        converted_values = cute.make_rmem_tensor(8, dtype)
                        cute.copy(
                            cute.make_copy_atom(
                                cute.nvgpu.CopyUniversalOp(),
                                cutlass.Float8E4M3FN,
                                num_bits_per_copy=64,
                            ),
                            src,
                            raw_values,
                        )
                        converted_values.store(raw_values.load().to(dtype))
                        cute.copy(
                            cute.make_copy_atom(
                                cute.nvgpu.CopyUniversalOp(),
                                dtype,
                                num_bits_per_copy=128,
                            ),
                            converted_values,
                            dst,
                        )
                    cute.arch.fence_view_async_shared()
                    _epilogue_barrier(self.epilogue_threads, cutlass.Int32(1))
                    if tid == 0:
                        cute.arch.mbarrier_arrive(converted + stage)
            if cutlass.const_expr(self.pdl):
                cute.arch.griddepcontrol_wait()
            thread_mma = mma.get_slice(0)
            fragment = thread_mma.make_fragment_C(
                mma.partition_shape_C(
                    (self.up_token_rows, self.up_hidden_tile * self.hc_count)
                )
            )
            _, _, template, coordinates = self._up_reader(
                base, fragment.layout, original, tid, 0
            )
            count = cute.size(template)
            ring_slots = 4
            residual = cute.make_rmem_tensor((count, ring_slots), original.element_type)
            scales = cute.make_rmem_tensor(
                (1, 1) if self.shared_up_scales else (count, ring_slots),
                cutlass.Float32,
            )
            residual.fill(0.0)
            scales.fill(1.0)
            sums = cute.make_rmem_tensor(count, cutlass.Float32)
            sums.fill(0.0)
            first_row, first_col = coordinates[0]
            global_row = token_tile * self.up_token_rows + first_row
            hidden = hidden_tile * self.up_hidden_tile + first_col
            alpha = cutlass.Float32(1.0)
            if cutlass.const_expr(self.shared_up_scales):
                scale_cache = cute_ext.allocate(
                    cutlass.Float32,
                    cute.AddressSpace.smem,
                    cute.make_layout(256),
                    alignment=32,
                )
                if tid < 256:
                    channel = tid // 64 * self.hidden_size + hidden_tile * 64 + tid % 64
                    scale_cache[tid] = weight_scale[
                        channel if self.weight_scale_per_channel else 0
                    ]
                _epilogue_barrier(self.epilogue_threads, cutlass.Int32(3))
                if cutlass.const_expr(native_fp8):
                    if global_row < x.shape[0]:
                        alpha = activation_scale[global_row]
            if cutlass.const_expr(self.up_token_rows == 64):
                for branch in cutlass.range_constexpr(self.hc_count):
                    self._up_fragment_prefetch(
                        original,
                        weight_scale,
                        activation_scale,
                        residual,
                        scales,
                        coordinates,
                        token_tile,
                        hidden_tile,
                        x.shape[0],
                        branch,
                    )
            else:
                for branch in cutlass.range_constexpr(ring_slots):
                    self._up_prefetch(
                        original,
                        weight_scale,
                        activation_scale,
                        residual,
                        scales,
                        global_row,
                        hidden,
                        x.shape[0],
                        branch,
                        branch,
                        native_fp8,
                    )
            cute.arch.mbarrier_wait(done, 0)
            if (
                tid % 128 // 32 * (self.up_token_rows // 4)
                < x.shape[0] - token_tile * self.up_token_rows
            ):
                for branch in cutlass.range_constexpr(self.hc_count):
                    slot = branch % ring_slots
                    accumulator, reader, gates, _ = self._up_reader(
                        base, fragment.layout, original, tid, branch
                    )
                    cute_ext.partition_and_copy(reader, accumulator, gates)
                    cute.arch.fence_view_async_tmem_load()
                    for i in cutlass.range_constexpr(count):
                        factor = cutlass.Float32(1.0)
                        if cutlass.const_expr(self.shared_up_scales):
                            factor = scale_cache[branch * 64 + first_col + i] * alpha
                        else:
                            factor = scales[i, slot]
                        sums[i] = sums[i] + residual[i, slot].to(
                            cutlass.Float32
                        ) * _up_sigmoid_tanh(gates[i] * factor)
                    if cutlass.const_expr(branch + ring_slots < self.hc_count):
                        self._up_prefetch(
                            original,
                            weight_scale,
                            activation_scale,
                            residual,
                            scales,
                            global_row,
                            hidden,
                            x.shape[0],
                            branch + ring_slots,
                            slot,
                            native_fp8,
                        )
            tcgen05_fence("before_thread_sync")
            cute.arch.mbarrier_arrive(read_done)
            if cutlass.const_expr(self.up_token_rows == 64):
                for packet in cutlass.range_constexpr(count // 2):
                    row, col = coordinates[packet * 2]
                    output_row = token_tile * self.up_token_rows + row
                    if output_row < x.shape[0]:
                        packed = cute.make_rmem_tensor(2, out.element_type)
                        for j in cutlass.range_constexpr(2):
                            packed[j] = (sums[packet * 2 + j] / self.hc_count).to(
                                out.element_type
                            )
                        target = cute.make_tensor(
                            cute.make_ptr(
                                out.element_type,
                                (
                                    out.iterator
                                    + output_row * self.hidden_size
                                    + hidden_tile * self.up_hidden_tile
                                    + col
                                ).toint(),
                                cute.AddressSpace.gmem,
                                assumed_align=4,
                            ),
                            cute.make_layout(2),
                        )
                        cute.copy(
                            cute.make_copy_atom(
                                cute.nvgpu.CopyUniversalOp(),
                                out.element_type,
                                num_bits_per_copy=32,
                            ),
                            packed,
                            target,
                        )
            elif global_row < x.shape[0]:
                packed = cute.make_rmem_tensor(count, out.element_type)
                for i in cutlass.range_constexpr(count):
                    packed[i] = (sums[i] / self.hc_count).to(out.element_type)
                target = cute.make_tensor(
                    cute.make_ptr(
                        out.element_type,
                        (out.iterator + global_row * self.hidden_size + hidden).toint(),
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    ),
                    cute.make_layout(count),
                )
                cute.copy(
                    cute.make_copy_atom(
                        cute.nvgpu.CopyUniversalOp(),
                        out.element_type,
                        num_bits_per_copy=128,
                    ),
                    packed,
                    target,
                )

    @cute.experimental.jit
    def __call__(
        self,
        x,
        weight,
        original,
        weight_scale,
        activation_scale,
        partials,
        out,
        stream: CUstream,
    ):
        if cutlass.const_expr(self.role == "mix" and self.token_major_up):
            self.up_kernel(
                x, weight, original, weight_scale, activation_scale, out
            ).launch(
                grid=(
                    self.hidden_size // self.up_hidden_tile,
                    cute.ceil_div(x.shape[0], self.up_token_rows),
                    1,
                ),
                block=(self.epilogue_threads + 64, 1, 1),
                stream=stream,
                use_pdl=self.pdl,
            )
        else:
            self.kernel(
                x, weight, original, weight_scale, activation_scale, partials, out
            ).launch(
                grid=(
                    (
                        (self.projection_rows + self.m - 1) // self.m
                        if self.role == "projection"
                        else self.hidden_size // 32
                    ),
                    cute.ceil_div(x.shape[0], self.n),
                    self.split_k,
                ),
                block=(self.epilogue_threads + 64, 1, 1),
                stream=stream,
                use_pdl=self.pdl,
            )


class Fp8GatedResidualM64H64CachedS5Up(Fp8GatedResidualM64H64Up):
    """Native M64/N256 Up with four fragment-correct HC branches and a register fold."""

    @cute.experimental.jit
    def _up_fragment_prefetch(
        self,
        original,
        weight_scale,
        activation_scale,
        residual,
        scales,
        coordinates,
        token_tile,
        hidden_tile,
        rows,
        branch: cutlass.Constexpr,
    ):
        count = cute.size(residual, mode=[0])
        for packet in cutlass.range_constexpr(count // 2):
            row, col = coordinates[packet * 2]
            input_row = token_tile * self.up_token_rows + row
            hidden = hidden_tile * self.up_hidden_tile + col
            if input_row < rows:
                source = cute.make_tensor(
                    cute.make_ptr(
                        original.element_type,
                        (
                            original.iterator
                            + input_row * self.wide
                            + branch * self.hidden_size
                            + hidden
                        ).toint(),
                        cute.AddressSpace.gmem,
                        assumed_align=4,
                    ),
                    cute.make_layout(2),
                )
                values = cute.make_rmem_tensor(2, original.element_type)
                cute.copy(
                    cute.make_copy_atom(
                        cute.nvgpu.CopyUniversalOp(),
                        original.element_type,
                        num_bits_per_copy=32,
                    ),
                    source,
                    values,
                )
                for j in cutlass.range_constexpr(2):
                    residual[packet * 2 + j, branch] = values[j]
                    if cutlass.const_expr(not self.shared_up_scales):
                        scales[packet * 2 + j, branch] = (
                            weight_scale[
                                (
                                    branch * self.hidden_size + hidden + j
                                    if self.weight_scale_per_channel
                                    else 0
                                )
                            ]
                            * activation_scale[input_row]
                        )

    @cute.experimental.kernel
    def up_kernel(self, x, weight, original, weight_scale, activation_scale, out):
        tid = cute.arch.thread_idx()[0]
        warp = cute.arch.make_warp_uniform(tid // 32)
        hidden_tile, token_tile, _ = cute.arch.block_idx()
        token_tile = cutlass.Int64(token_tile)
        dtype = x.element_type
        native_fp8 = cutlass.const_expr(dtype == cutlass.Float8E4M3FN)
        mma = sm100_utils.make_trivial_tiled_mma(
            dtype,
            dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            cutlass.Float32,
            tcgen05.CtaGroup.ONE,
            (self.up_token_rows, self.up_hidden_tile * self.hc_count),
        )
        tiler = (self.up_token_rows, self.up_hidden_tile * self.hc_count, self.k)
        activation_layout = sm100_utils.make_smem_layout_a(
            mma, tiler, dtype, self.stages
        )
        raw_layout = sm100_utils.make_smem_layout_b(
            mma, tiler, cutlass.Float8E4M3FN, self.stages
        )
        weight_layout = sm100_utils.make_smem_layout_b(mma, tiler, dtype, self.stages)
        sx = cute_ext.allocate(
            dtype, cute.AddressSpace.smem, activation_layout, alignment=1024
        )
        raw = cute_ext.allocate(
            cutlass.Float8E4M3FN, cute.AddressSpace.smem, raw_layout, alignment=1024
        )
        sw = raw
        if cutlass.const_expr(not native_fp8):
            sw = cute_ext.allocate(
                dtype, cute.AddressSpace.smem, weight_layout, alignment=1024
            )
        barriers = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(3 * self.stages + 2),
            alignment=8,
        ).iterator
        full = barriers
        converted = full + self.stages
        empty = converted + self.stages
        done, read_done = (empty + self.stages, empty + self.stages + 1)
        base = cute_ext.allocate(
            cutlass.Int32, cute.AddressSpace.smem, cute.make_layout(1), alignment=4
        ).iterator
        if warp == self.mma_warp:
            with cute.arch.elect_one():
                for stage in cutlass.range_constexpr(self.stages):
                    cute.arch.mbarrier_init(full + stage, 1)
                    cute.arch.mbarrier_init(converted + stage, 1)
                    cute.arch.mbarrier_init(empty + stage, 1)
                cute.arch.mbarrier_init(done, 1)
                cute.arch.mbarrier_init(read_done, self.epilogue_threads)
            cute.arch.alloc_tmem(
                self.up_hidden_tile * self.hc_count, base, is_two_cta=False
            )
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        activation = cute.make_tensor(
            x.iterator,
            cute.make_layout(
                (x.shape[0], (64, self.inner // 64), 1), stride=(self.inner, (1, 64), 0)
            ),
        )
        weights = cute.make_tensor(
            weight.iterator,
            cute.make_layout(
                (
                    (
                        self.up_hidden_tile,
                        self.hc_count,
                        self.hidden_size // self.up_hidden_tile,
                    ),
                    (64, self.inner // 64),
                    1,
                ),
                stride=(
                    (
                        self.inner,
                        self.hidden_size * self.inner,
                        self.up_hidden_tile * self.inner,
                    ),
                    (1, 64),
                    0,
                ),
            ),
        )
        gx = cute.local_tile(
            activation, (self.up_token_rows, self.k), (token_tile, None, 0)
        )
        gw = cute.local_tile(
            weights,
            (self.up_hidden_tile * self.hc_count, self.k),
            (hidden_tile, None, 0),
        )
        map_x = cute_ext.get_cta_v_map_ab(activation, tiler, mma, "A")
        map_w = cute_ext.get_cta_v_map_ab(weights, tiler, mma, "B")
        if warp == self.producer_warp:
            for stage in cutlass.range_constexpr(self.stages):
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        full + stage,
                        self.up_hidden_tile * self.hc_count * self.k
                        + self.up_token_rows * self.k * (dtype.width // 8),
                    )
                if cutlass.const_expr(self.weights_independent):
                    cute_ext.tma_load(
                        gw[None, None, stage],
                        raw[None, None, None, stage],
                        (full + stage).value,
                        cta_v_map=map_w,
                        update_expect_tx=False,
                    )
            if cutlass.const_expr(self.pdl):
                cute.arch.griddepcontrol_wait()
                cute.arch.griddepcontrol_launch_dependents()
            for stage in cutlass.range_constexpr(self.stages):
                if cutlass.const_expr(not self.weights_independent):
                    cute_ext.tma_load(
                        gw[None, None, stage],
                        raw[None, None, None, stage],
                        (full + stage).value,
                        cta_v_map=map_w,
                        update_expect_tx=False,
                    )
                cute_ext.tma_load(
                    gx[None, None, stage],
                    sx[None, None, None, stage],
                    (full + stage).value,
                    cta_v_map=map_x,
                    update_expect_tx=False,
                )
            for tile in cutlass.range(self.stages, self.tiles, unroll=1):
                stage = tile % self.stages
                cute.arch.mbarrier_wait(empty + stage, (tile // self.stages - 1) % 2)
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        full + stage,
                        self.up_hidden_tile * self.hc_count * self.k
                        + self.up_token_rows * self.k * (dtype.width // 8),
                    )
                cute_ext.tma_load(
                    gw[None, None, tile],
                    raw[None, None, None, stage],
                    (full + stage).value,
                    cta_v_map=map_w,
                    update_expect_tx=False,
                )
                cute_ext.tma_load(
                    gx[None, None, tile],
                    sx[None, None, None, stage],
                    (full + stage).value,
                    cta_v_map=map_x,
                    update_expect_tx=False,
                )
        elif warp == self.mma_warp:
            thread_mma = mma.get_slice(0)
            a_desc = thread_mma.make_fragment_A(sx)
            b_desc = thread_mma.make_fragment_B(sw)
            fragment = thread_mma.make_fragment_C(
                mma.partition_shape_C(
                    (self.up_token_rows, self.up_hidden_tile * self.hc_count)
                )
            )
            accumulator = cute.make_tensor(
                cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, base), fragment.layout
            )
            mma.set(tcgen05.Field.ACCUMULATE, False)
            for tile in cutlass.range(self.tiles, unroll=1):
                stage = tile % self.stages
                ready = full if cutlass.const_expr(native_fp8) else converted
                cute.arch.mbarrier_wait(ready + stage, tile // self.stages % 2)
                for block in cutlass.range_constexpr(cute.size(a_desc, mode=[2])):
                    cute.gemm(
                        mma,
                        accumulator,
                        a_desc[None, None, block, stage],
                        b_desc[None, None, block, stage],
                        accumulator,
                    )
                    mma.set(tcgen05.Field.ACCUMULATE, True)
                with cute.arch.elect_one():
                    tcgen05.commit(empty + stage, None, tcgen05.CtaGroup.ONE)
            with cute.arch.elect_one():
                tcgen05.commit(done, None, tcgen05.CtaGroup.ONE)
            cute.arch.mbarrier_wait(read_done, 0)
            cute.arch.relinquish_tmem_alloc_permit()
            cute.arch.dealloc_tmem(
                cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, base),
                self.up_hidden_tile * self.hc_count,
            )
        elif tid < self.epilogue_threads:
            if cutlass.const_expr(not native_fp8):
                for tile in cutlass.range(self.tiles, unroll=1):
                    stage = tile % self.stages
                    cute.arch.mbarrier_wait(full + stage, tile // self.stages % 2)
                    source, target = (
                        raw[None, None, None, stage],
                        sw[None, None, None, stage],
                    )
                    for offset in cutlass.range(
                        128 * self.k // (self.epilogue_threads * 8), unroll=4
                    ):
                        linear = (tid + offset * self.epilogue_threads) * 8
                        index = linear // self.k + linear % self.k * 128
                        src = cute.make_tensor(
                            cute.make_ptr(
                                cutlass.Float8E4M3FN,
                                (source.iterator + source.layout(index)).toint(),
                                cute.AddressSpace.smem,
                                assumed_align=8,
                                swizzle_=cute.get_swizzle_portion(raw_layout),
                            ),
                            cute.make_layout(8),
                        )
                        dst = cute.make_tensor(
                            cute.make_ptr(
                                dtype,
                                (target.iterator + target.layout(index)).toint(),
                                cute.AddressSpace.smem,
                                assumed_align=16,
                                swizzle_=cute.get_swizzle_portion(weight_layout),
                            ),
                            cute.make_layout(8),
                        )
                        raw_values = cute.make_rmem_tensor(8, cutlass.Float8E4M3FN)
                        converted_values = cute.make_rmem_tensor(8, dtype)
                        cute.copy(
                            cute.make_copy_atom(
                                cute.nvgpu.CopyUniversalOp(),
                                cutlass.Float8E4M3FN,
                                num_bits_per_copy=64,
                            ),
                            src,
                            raw_values,
                        )
                        converted_values.store(raw_values.load().to(dtype))
                        cute.copy(
                            cute.make_copy_atom(
                                cute.nvgpu.CopyUniversalOp(),
                                dtype,
                                num_bits_per_copy=128,
                            ),
                            converted_values,
                            dst,
                        )
                    cute.arch.fence_view_async_shared()
                    _epilogue_barrier(self.epilogue_threads, cutlass.Int32(1))
                    if tid == 0:
                        cute.arch.mbarrier_arrive(converted + stage)
            if cutlass.const_expr(self.pdl):
                cute.arch.griddepcontrol_wait()
            thread_mma = mma.get_slice(0)
            fragment = thread_mma.make_fragment_C(
                mma.partition_shape_C(
                    (self.up_token_rows, self.up_hidden_tile * self.hc_count)
                )
            )
            _, _, template, coordinates = self._up_reader(
                base, fragment.layout, original, tid, 0
            )
            count = cute.size(template)
            ring_slots = 4
            residual = cute.make_rmem_tensor((count, ring_slots), original.element_type)
            scales = cute.make_rmem_tensor(
                (1, 1) if self.shared_up_scales else (count, ring_slots),
                cutlass.Float32,
            )
            residual.fill(0.0)
            scales.fill(1.0)
            sums = cute.make_rmem_tensor(count, cutlass.Float32)
            sums.fill(0.0)
            first_row, first_col = coordinates[0]
            global_row = token_tile * self.up_token_rows + first_row
            hidden = hidden_tile * self.up_hidden_tile + first_col
            alpha = cutlass.Float32(1.0)
            if cutlass.const_expr(self.shared_up_scales):
                scale_cache = cute_ext.allocate(
                    cutlass.Float32,
                    cute.AddressSpace.smem,
                    cute.make_layout(256),
                    alignment=32,
                )
                if tid < 256:
                    channel = tid // 64 * self.hidden_size + hidden_tile * 64 + tid % 64
                    scale_cache[tid] = weight_scale[
                        channel if self.weight_scale_per_channel else 0
                    ]
                alpha_cache = cute_ext.allocate(
                    cutlass.Float32,
                    cute.AddressSpace.smem,
                    cute.make_layout(64),
                    alignment=32,
                )
                if tid < 64:
                    alpha_cache[tid] = 1.0
                    if token_tile * self.up_token_rows + tid < x.shape[0]:
                        alpha_cache[tid] = activation_scale[
                            token_tile * self.up_token_rows + tid
                        ]
                _epilogue_barrier(self.epilogue_threads, cutlass.Int32(3))
                if cutlass.const_expr(native_fp8):
                    if global_row < x.shape[0]:
                        alpha = activation_scale[global_row]
            if cutlass.const_expr(self.up_token_rows == 64):
                for branch in cutlass.range_constexpr(self.hc_count):
                    self._up_fragment_prefetch(
                        original,
                        weight_scale,
                        activation_scale,
                        residual,
                        scales,
                        coordinates,
                        token_tile,
                        hidden_tile,
                        x.shape[0],
                        branch,
                    )
            else:
                for branch in cutlass.range_constexpr(ring_slots):
                    self._up_prefetch(
                        original,
                        weight_scale,
                        activation_scale,
                        residual,
                        scales,
                        global_row,
                        hidden,
                        x.shape[0],
                        branch,
                        branch,
                        native_fp8,
                    )
            cute.arch.mbarrier_wait(done, 0)
            if (
                tid % 128 // 32 * (self.up_token_rows // 4)
                < x.shape[0] - token_tile * self.up_token_rows
            ):
                for branch in cutlass.range_constexpr(self.hc_count):
                    slot = branch % ring_slots
                    accumulator, reader, gates, _ = self._up_reader(
                        base, fragment.layout, original, tid, branch
                    )
                    cute_ext.partition_and_copy(reader, accumulator, gates)
                    cute.arch.fence_view_async_tmem_load()
                    for i in cutlass.range_constexpr(count):
                        factor = cutlass.Float32(1.0)
                        if cutlass.const_expr(self.shared_up_scales):
                            # M64 fragments interleave rows; each value uses its own scale coordinates.
                            factor = (
                                scale_cache[
                                    branch * self.up_hidden_tile + coordinates[i][1]
                                ]
                                * alpha_cache[coordinates[i][0]]
                            )
                        else:
                            factor = scales[i, slot]
                        sums[i] = sums[i] + residual[i, slot].to(
                            cutlass.Float32
                        ) * _up_sigmoid_tanh(gates[i] * factor)
                    if cutlass.const_expr(branch + ring_slots < self.hc_count):
                        self._up_prefetch(
                            original,
                            weight_scale,
                            activation_scale,
                            residual,
                            scales,
                            global_row,
                            hidden,
                            x.shape[0],
                            branch + ring_slots,
                            slot,
                            native_fp8,
                        )
            tcgen05_fence("before_thread_sync")
            cute.arch.mbarrier_arrive(read_done)
            if cutlass.const_expr(self.up_token_rows == 64):
                for packet in cutlass.range_constexpr(count // 2):
                    row, col = coordinates[packet * 2]
                    output_row = token_tile * self.up_token_rows + row
                    if output_row < x.shape[0]:
                        packed = cute.make_rmem_tensor(2, out.element_type)
                        for j in cutlass.range_constexpr(2):
                            packed[j] = (sums[packet * 2 + j] / self.hc_count).to(
                                out.element_type
                            )
                        target = cute.make_tensor(
                            cute.make_ptr(
                                out.element_type,
                                (
                                    out.iterator
                                    + output_row * self.hidden_size
                                    + hidden_tile * self.up_hidden_tile
                                    + col
                                ).toint(),
                                cute.AddressSpace.gmem,
                                assumed_align=4,
                            ),
                            cute.make_layout(2),
                        )
                        cute.copy(
                            cute.make_copy_atom(
                                cute.nvgpu.CopyUniversalOp(),
                                out.element_type,
                                num_bits_per_copy=32,
                            ),
                            packed,
                            target,
                        )
            elif global_row < x.shape[0]:
                packed = cute.make_rmem_tensor(count, out.element_type)
                for i in cutlass.range_constexpr(count):
                    packed[i] = (sums[i] / self.hc_count).to(out.element_type)
                target = cute.make_tensor(
                    cute.make_ptr(
                        out.element_type,
                        (out.iterator + global_row * self.hidden_size + hidden).toint(),
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    ),
                    cute.make_layout(count),
                )
                cute.copy(
                    cute.make_copy_atom(
                        cute.nvgpu.CopyUniversalOp(),
                        out.element_type,
                        num_bits_per_copy=128,
                    ),
                    packed,
                    target,
                )


class Fp8GatedResidualM64H56Up(Fp8GatedResidualM64H64CachedS5Up):
    """Native M64/H56 Up with padded readers and a bounded H16 hidden tail."""

    @cute.experimental.jit
    def __call__(
        self,
        x,
        weight,
        original,
        weight_scale,
        activation_scale,
        partials,
        out,
        stream: CUstream,
    ):
        self.up_kernel(x, weight, original, weight_scale, activation_scale, out).launch(
            grid=(
                cute.ceil_div(self.hidden_size, self.up_hidden_tile),
                cute.ceil_div(x.shape[0], self.up_token_rows),
                1,
            ),
            block=(self.epilogue_threads + 64, 1, 1),
            stream=stream,
            use_pdl=self.pdl,
        )

    @cute.experimental.jit
    def _up_reader(self, base, layout, original, tid, branch: cutlass.Constexpr):
        tile = (self.up_token_rows, 16)
        group = tid // 128
        accumulator = cute.make_tensor(
            cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, base), layout
        )[(None, None), 0, 0]
        # Padding each reader to16 columns preserves vector loads at H56.
        column = branch * self.up_hidden_tile + group * 16
        accumulator = cute.make_tensor(
            accumulator.iterator + accumulator.layout((0, column)),
            cute.make_layout(
                (cute.shape(accumulator, mode=[0]), 16),
                stride=(accumulator.stride[0], accumulator.stride[1]),
            ),
        )
        coordinates = cute.local_tile(
            cute.make_identity_tensor((self.up_token_rows, 64)), tile, (0, group)
        )
        copy = tcgen05.make_tmem_copy(
            sm100_utils.get_tmem_load_op(
                (self.up_token_rows, self.up_hidden_tile * self.hc_count, self.k),
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
            original.iterator, cute.make_layout(tile, stride=(1, self.up_token_rows))
        )
        values = cute_ext.allocate(
            cutlass.Float32,
            cute.AddressSpace.rmem,
            cute_ext.make_t2r_rmem_layout(
                copy, cute.flat_divide(dummy, tile), tid % 128
            ),
            alignment=32,
        )
        return (accumulator, reader, values, reader.partition_D(coordinates))

    @cute.experimental.jit
    def _up_fragment_prefetch(
        self,
        original,
        weight_scale,
        activation_scale,
        residual,
        scales,
        coordinates,
        token_tile,
        hidden_tile,
        rows,
        branch: cutlass.Constexpr,
    ):
        count = cute.size(residual, mode=[0])
        for packet in cutlass.range_constexpr(count // 2):
            row, col = coordinates[packet * 2]
            input_row = token_tile * self.up_token_rows + row
            hidden = hidden_tile * self.up_hidden_tile + col
            if (
                input_row < rows
                and col < self.up_hidden_tile
                and (hidden < self.hidden_size)
            ):
                source = cute.make_tensor(
                    cute.make_ptr(
                        original.element_type,
                        (
                            original.iterator
                            + input_row * self.wide
                            + branch * self.hidden_size
                            + hidden
                        ).toint(),
                        cute.AddressSpace.gmem,
                        assumed_align=4,
                    ),
                    cute.make_layout(2),
                )
                values = cute.make_rmem_tensor(2, original.element_type)
                cute.copy(
                    cute.make_copy_atom(
                        cute.nvgpu.CopyUniversalOp(),
                        original.element_type,
                        num_bits_per_copy=32,
                    ),
                    source,
                    values,
                )
                for j in cutlass.range_constexpr(2):
                    residual[packet * 2 + j, branch] = values[j]
                    if cutlass.const_expr(not self.shared_up_scales):
                        scales[packet * 2 + j, branch] = (
                            weight_scale[
                                (
                                    branch * self.hidden_size + hidden + j
                                    if self.weight_scale_per_channel
                                    else 0
                                )
                            ]
                            * activation_scale[input_row]
                        )

    @cute.experimental.kernel
    def up_kernel(self, x, weight, original, weight_scale, activation_scale, out):
        tid = cute.arch.thread_idx()[0]
        warp = cute.arch.make_warp_uniform(tid // 32)
        hidden_tile, token_tile, _ = cute.arch.block_idx()
        token_tile = cutlass.Int64(token_tile)
        dtype = x.element_type
        native_fp8 = cutlass.const_expr(dtype == cutlass.Float8E4M3FN)
        mma = sm100_utils.make_trivial_tiled_mma(
            dtype,
            dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            cutlass.Float32,
            tcgen05.CtaGroup.ONE,
            (self.up_token_rows, self.up_hidden_tile * self.hc_count),
        )
        tiler = (self.up_token_rows, self.up_hidden_tile * self.hc_count, self.k)
        activation_layout = sm100_utils.make_smem_layout_a(
            mma, tiler, dtype, self.stages
        )
        raw_layout = sm100_utils.make_smem_layout_b(
            mma, tiler, cutlass.Float8E4M3FN, self.stages
        )
        weight_layout = sm100_utils.make_smem_layout_b(mma, tiler, dtype, self.stages)
        sx = cute_ext.allocate(
            dtype, cute.AddressSpace.smem, activation_layout, alignment=1024
        )
        raw = cute_ext.allocate(
            cutlass.Float8E4M3FN, cute.AddressSpace.smem, raw_layout, alignment=1024
        )
        sw = raw
        if cutlass.const_expr(not native_fp8):
            sw = cute_ext.allocate(
                dtype, cute.AddressSpace.smem, weight_layout, alignment=1024
            )
        barriers = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(3 * self.stages + 2),
            alignment=8,
        ).iterator
        full = barriers
        converted = full + self.stages
        empty = converted + self.stages
        done, read_done = (empty + self.stages, empty + self.stages + 1)
        base = cute_ext.allocate(
            cutlass.Int32, cute.AddressSpace.smem, cute.make_layout(1), alignment=4
        ).iterator
        if warp == self.mma_warp:
            with cute.arch.elect_one():
                for stage in cutlass.range_constexpr(self.stages):
                    cute.arch.mbarrier_init(full + stage, 1)
                    cute.arch.mbarrier_init(converted + stage, 1)
                    cute.arch.mbarrier_init(empty + stage, 1)
                cute.arch.mbarrier_init(done, 1)
                cute.arch.mbarrier_init(read_done, self.epilogue_threads)
            cute.arch.alloc_tmem(256, base, is_two_cta=False)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        activation = cute.make_tensor(
            x.iterator,
            cute.make_layout(
                (x.shape[0], (64, self.inner // 64), 1), stride=(self.inner, (1, 64), 0)
            ),
        )
        weights = cute.make_tensor(
            weight.iterator,
            cute.make_layout(
                (
                    (
                        self.up_hidden_tile,
                        self.hc_count,
                        self.hidden_size // self.up_hidden_tile,
                    ),
                    (64, self.inner // 64),
                    1,
                ),
                stride=(
                    (
                        self.inner,
                        self.hidden_size * self.inner,
                        self.up_hidden_tile * self.inner,
                    ),
                    (1, 64),
                    0,
                ),
            ),
        )
        gx = cute.local_tile(
            activation, (self.up_token_rows, self.k), (token_tile, None, 0)
        )
        gw = cute.local_tile(
            weights,
            (self.up_hidden_tile * self.hc_count, self.k),
            (hidden_tile, None, 0),
        )
        map_x = cute_ext.get_cta_v_map_ab(activation, tiler, mma, "A")
        map_w = cute_ext.get_cta_v_map_ab(weights, tiler, mma, "B")
        tail_weights = cute.make_tensor(
            weight.iterator
            + self.hidden_size
            // self.up_hidden_tile
            * self.up_hidden_tile
            * self.inner,
            cute.make_layout(
                (
                    self.hidden_size % self.up_hidden_tile,
                    (64, self.inner // 64),
                    self.hc_count,
                ),
                stride=(self.inner, (1, 64), self.hidden_size * self.inner),
            ),
        )
        tail_mma = sm100_utils.make_trivial_tiled_mma(
            dtype,
            dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            cutlass.Float32,
            tcgen05.CtaGroup.ONE,
            (self.up_token_rows, self.up_hidden_tile),
        )
        tail_tiler = (self.up_token_rows, self.up_hidden_tile, self.k)
        tail_layout = sm100_utils.make_smem_layout_b(tail_mma, tail_tiler, dtype, 1)
        tail_map = cute_ext.get_cta_v_map_ab(tail_weights, tail_tiler, tail_mma, "B")
        tail_gw = cute.local_tile(
            tail_weights, (self.up_hidden_tile, self.k), (0, None, None)
        )
        if warp == self.producer_warp:
            for stage in cutlass.range_constexpr(self.stages):
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        full + stage,
                        self.up_hidden_tile * self.hc_count * self.k
                        + self.up_token_rows * self.k * (dtype.width // 8),
                    )
                if cutlass.const_expr(self.weights_independent):
                    if hidden_tile < self.hidden_size // self.up_hidden_tile:
                        cute_ext.tma_load(
                            gw[None, None, stage],
                            raw[None, None, None, stage],
                            (full + stage).value,
                            cta_v_map=map_w,
                            update_expect_tx=False,
                        )
                    else:
                        for branch in cutlass.range_constexpr(self.hc_count):
                            tail_dst = cute.make_tensor(
                                cute.recast_ptr(
                                    raw.iterator
                                    + stage
                                    * self.up_hidden_tile
                                    * self.hc_count
                                    * self.k
                                    + branch * self.up_hidden_tile * self.k,
                                    swizzle_=cute.get_swizzle_portion(tail_layout),
                                    dtype=dtype,
                                ),
                                cute.get_nonswizzle_portion(tail_layout),
                            )
                            cute_ext.tma_load(
                                tail_gw[None, None, stage, branch],
                                tail_dst[None, None, None, 0],
                                (full + stage).value,
                                cta_v_map=tail_map,
                                update_expect_tx=False,
                            )
            if cutlass.const_expr(self.pdl):
                cute.arch.griddepcontrol_wait()
                cute.arch.griddepcontrol_launch_dependents()
            for stage in cutlass.range_constexpr(self.stages):
                if cutlass.const_expr(not self.weights_independent):
                    if hidden_tile < self.hidden_size // self.up_hidden_tile:
                        cute_ext.tma_load(
                            gw[None, None, stage],
                            raw[None, None, None, stage],
                            (full + stage).value,
                            cta_v_map=map_w,
                            update_expect_tx=False,
                        )
                    else:
                        for branch in cutlass.range_constexpr(self.hc_count):
                            tail_dst = cute.make_tensor(
                                cute.recast_ptr(
                                    raw.iterator
                                    + stage
                                    * self.up_hidden_tile
                                    * self.hc_count
                                    * self.k
                                    + branch * self.up_hidden_tile * self.k,
                                    swizzle_=cute.get_swizzle_portion(tail_layout),
                                    dtype=dtype,
                                ),
                                cute.get_nonswizzle_portion(tail_layout),
                            )
                            cute_ext.tma_load(
                                tail_gw[None, None, stage, branch],
                                tail_dst[None, None, None, 0],
                                (full + stage).value,
                                cta_v_map=tail_map,
                                update_expect_tx=False,
                            )
                cute_ext.tma_load(
                    gx[None, None, stage],
                    sx[None, None, None, stage],
                    (full + stage).value,
                    cta_v_map=map_x,
                    update_expect_tx=False,
                )
            for tile in cutlass.range(self.stages, self.tiles, unroll=1):
                stage = tile % self.stages
                cute.arch.mbarrier_wait(empty + stage, (tile // self.stages - 1) % 2)
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        full + stage,
                        self.up_hidden_tile * self.hc_count * self.k
                        + self.up_token_rows * self.k * (dtype.width // 8),
                    )
                if hidden_tile < self.hidden_size // self.up_hidden_tile:
                    cute_ext.tma_load(
                        gw[None, None, tile],
                        raw[None, None, None, stage],
                        (full + stage).value,
                        cta_v_map=map_w,
                        update_expect_tx=False,
                    )
                else:
                    for branch in cutlass.range_constexpr(self.hc_count):
                        tail_dst = cute.make_tensor(
                            cute.recast_ptr(
                                raw.iterator
                                + stage * self.up_hidden_tile * self.hc_count * self.k
                                + branch * self.up_hidden_tile * self.k,
                                swizzle_=cute.get_swizzle_portion(tail_layout),
                                dtype=dtype,
                            ),
                            cute.get_nonswizzle_portion(tail_layout),
                        )
                        cute_ext.tma_load(
                            tail_gw[None, None, tile, branch],
                            tail_dst[None, None, None, 0],
                            (full + stage).value,
                            cta_v_map=tail_map,
                            update_expect_tx=False,
                        )
                cute_ext.tma_load(
                    gx[None, None, tile],
                    sx[None, None, None, stage],
                    (full + stage).value,
                    cta_v_map=map_x,
                    update_expect_tx=False,
                )
        elif warp == self.mma_warp:
            thread_mma = mma.get_slice(0)
            a_desc = thread_mma.make_fragment_A(sx)
            b_desc = thread_mma.make_fragment_B(sw)
            fragment = thread_mma.make_fragment_C(
                mma.partition_shape_C(
                    (self.up_token_rows, self.up_hidden_tile * self.hc_count)
                )
            )
            accumulator = cute.make_tensor(
                cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, base), fragment.layout
            )
            mma.set(tcgen05.Field.ACCUMULATE, False)
            for tile in cutlass.range(self.tiles, unroll=1):
                stage = tile % self.stages
                ready = full if cutlass.const_expr(native_fp8) else converted
                cute.arch.mbarrier_wait(ready + stage, tile // self.stages % 2)
                for block in cutlass.range_constexpr(cute.size(a_desc, mode=[2])):
                    cute.gemm(
                        mma,
                        accumulator,
                        a_desc[None, None, block, stage],
                        b_desc[None, None, block, stage],
                        accumulator,
                    )
                    mma.set(tcgen05.Field.ACCUMULATE, True)
                with cute.arch.elect_one():
                    tcgen05.commit(empty + stage, None, tcgen05.CtaGroup.ONE)
            with cute.arch.elect_one():
                tcgen05.commit(done, None, tcgen05.CtaGroup.ONE)
            cute.arch.mbarrier_wait(read_done, 0)
            cute.arch.relinquish_tmem_alloc_permit()
            cute.arch.dealloc_tmem(
                cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, base), 256
            )
        elif tid < self.epilogue_threads:
            if cutlass.const_expr(not native_fp8):
                for tile in cutlass.range(self.tiles, unroll=1):
                    stage = tile % self.stages
                    cute.arch.mbarrier_wait(full + stage, tile // self.stages % 2)
                    source, target = (
                        raw[None, None, None, stage],
                        sw[None, None, None, stage],
                    )
                    for offset in cutlass.range(
                        128 * self.k // (self.epilogue_threads * 8), unroll=4
                    ):
                        linear = (tid + offset * self.epilogue_threads) * 8
                        index = linear // self.k + linear % self.k * 128
                        src = cute.make_tensor(
                            cute.make_ptr(
                                cutlass.Float8E4M3FN,
                                (source.iterator + source.layout(index)).toint(),
                                cute.AddressSpace.smem,
                                assumed_align=8,
                                swizzle_=cute.get_swizzle_portion(raw_layout),
                            ),
                            cute.make_layout(8),
                        )
                        dst = cute.make_tensor(
                            cute.make_ptr(
                                dtype,
                                (target.iterator + target.layout(index)).toint(),
                                cute.AddressSpace.smem,
                                assumed_align=16,
                                swizzle_=cute.get_swizzle_portion(weight_layout),
                            ),
                            cute.make_layout(8),
                        )
                        raw_values = cute.make_rmem_tensor(8, cutlass.Float8E4M3FN)
                        converted_values = cute.make_rmem_tensor(8, dtype)
                        cute.copy(
                            cute.make_copy_atom(
                                cute.nvgpu.CopyUniversalOp(),
                                cutlass.Float8E4M3FN,
                                num_bits_per_copy=64,
                            ),
                            src,
                            raw_values,
                        )
                        converted_values.store(raw_values.load().to(dtype))
                        cute.copy(
                            cute.make_copy_atom(
                                cute.nvgpu.CopyUniversalOp(),
                                dtype,
                                num_bits_per_copy=128,
                            ),
                            converted_values,
                            dst,
                        )
                    cute.arch.fence_view_async_shared()
                    _epilogue_barrier(self.epilogue_threads, cutlass.Int32(1))
                    if tid == 0:
                        cute.arch.mbarrier_arrive(converted + stage)
            if cutlass.const_expr(self.pdl):
                cute.arch.griddepcontrol_wait()
            thread_mma = mma.get_slice(0)
            fragment = thread_mma.make_fragment_C(
                mma.partition_shape_C(
                    (self.up_token_rows, self.up_hidden_tile * self.hc_count)
                )
            )
            _, _, template, coordinates = self._up_reader(
                base, fragment.layout, original, tid, 0
            )
            count = cute.size(template)
            ring_slots = 4
            residual = cute.make_rmem_tensor((count, ring_slots), original.element_type)
            scales = cute.make_rmem_tensor(
                (1, 1) if self.shared_up_scales else (count, ring_slots),
                cutlass.Float32,
            )
            residual.fill(0.0)
            scales.fill(1.0)
            sums = cute.make_rmem_tensor(count, cutlass.Float32)
            sums.fill(0.0)
            first_row, first_col = coordinates[0]
            global_row = token_tile * self.up_token_rows + first_row
            hidden = hidden_tile * self.up_hidden_tile + first_col
            alpha = cutlass.Float32(1.0)
            if cutlass.const_expr(self.shared_up_scales):
                scale_cache = cute_ext.allocate(
                    cutlass.Float32,
                    cute.AddressSpace.smem,
                    cute.make_layout(256),
                    alignment=32,
                )
                if tid < 256:
                    scale_cache[tid] = 1.0
                    if tid < self.up_hidden_tile * self.hc_count:
                        channel = (
                            tid // self.up_hidden_tile * self.hidden_size
                            + hidden_tile * self.up_hidden_tile
                            + tid % self.up_hidden_tile
                        )
                        if (
                            hidden_tile * self.up_hidden_tile
                            + tid % self.up_hidden_tile
                            < self.hidden_size
                        ):
                            scale_cache[tid] = weight_scale[
                                channel if self.weight_scale_per_channel else 0
                            ]
                alpha_cache = cute_ext.allocate(
                    cutlass.Float32,
                    cute.AddressSpace.smem,
                    cute.make_layout(64),
                    alignment=32,
                )
                if tid < 64:
                    alpha_cache[tid] = 1.0
                    if token_tile * self.up_token_rows + tid < x.shape[0]:
                        alpha_cache[tid] = activation_scale[
                            token_tile * self.up_token_rows + tid
                        ]
                _epilogue_barrier(self.epilogue_threads, cutlass.Int32(3))
                if cutlass.const_expr(native_fp8):
                    if global_row < x.shape[0]:
                        alpha = activation_scale[global_row]
            if cutlass.const_expr(self.up_token_rows == 64):
                for branch in cutlass.range_constexpr(self.hc_count):
                    self._up_fragment_prefetch(
                        original,
                        weight_scale,
                        activation_scale,
                        residual,
                        scales,
                        coordinates,
                        token_tile,
                        hidden_tile,
                        x.shape[0],
                        branch,
                    )
            else:
                for branch in cutlass.range_constexpr(ring_slots):
                    self._up_prefetch(
                        original,
                        weight_scale,
                        activation_scale,
                        residual,
                        scales,
                        global_row,
                        hidden,
                        x.shape[0],
                        branch,
                        branch,
                        native_fp8,
                    )
            cute.arch.mbarrier_wait(done, 0)
            if (
                tid % 128 // 32 * (self.up_token_rows // 4)
                < x.shape[0] - token_tile * self.up_token_rows
            ):
                for branch in cutlass.range_constexpr(self.hc_count):
                    slot = branch % ring_slots
                    accumulator, reader, gates, _ = self._up_reader(
                        base, fragment.layout, original, tid, branch
                    )
                    cute_ext.partition_and_copy(reader, accumulator, gates)
                    cute.arch.fence_view_async_tmem_load()
                    for i in cutlass.range_constexpr(count):
                        factor = cutlass.Float32(1.0)
                        if cutlass.const_expr(self.shared_up_scales):
                            factor = (
                                scale_cache[
                                    branch * self.up_hidden_tile + coordinates[i][1]
                                ]
                                * alpha_cache[coordinates[i][0]]
                            )
                        else:
                            factor = scales[i, slot]
                        sums[i] = sums[i] + residual[i, slot].to(
                            cutlass.Float32
                        ) * _up_sigmoid_tanh(gates[i] * factor)
                    if cutlass.const_expr(branch + ring_slots < self.hc_count):
                        self._up_prefetch(
                            original,
                            weight_scale,
                            activation_scale,
                            residual,
                            scales,
                            global_row,
                            hidden,
                            x.shape[0],
                            branch + ring_slots,
                            slot,
                            native_fp8,
                        )
            tcgen05_fence("before_thread_sync")
            cute.arch.mbarrier_arrive(read_done)
            if cutlass.const_expr(self.up_token_rows == 64):
                for packet in cutlass.range_constexpr(count // 2):
                    row, col = coordinates[packet * 2]
                    output_row = token_tile * self.up_token_rows + row
                    if (
                        output_row < x.shape[0]
                        and col < self.up_hidden_tile
                        and (hidden_tile * self.up_hidden_tile + col < self.hidden_size)
                    ):
                        packed = cute.make_rmem_tensor(2, out.element_type)
                        for j in cutlass.range_constexpr(2):
                            packed[j] = (sums[packet * 2 + j] / self.hc_count).to(
                                out.element_type
                            )
                        target = cute.make_tensor(
                            cute.make_ptr(
                                out.element_type,
                                (
                                    out.iterator
                                    + output_row * self.hidden_size
                                    + hidden_tile * self.up_hidden_tile
                                    + col
                                ).toint(),
                                cute.AddressSpace.gmem,
                                assumed_align=4,
                            ),
                            cute.make_layout(2),
                        )
                        cute.copy(
                            cute.make_copy_atom(
                                cute.nvgpu.CopyUniversalOp(),
                                out.element_type,
                                num_bits_per_copy=32,
                            ),
                            packed,
                            target,
                        )
            elif global_row < x.shape[0]:
                packed = cute.make_rmem_tensor(count, out.element_type)
                for i in cutlass.range_constexpr(count):
                    packed[i] = (sums[i] / self.hc_count).to(out.element_type)
                target = cute.make_tensor(
                    cute.make_ptr(
                        out.element_type,
                        (out.iterator + global_row * self.hidden_size + hidden).toint(),
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    ),
                    cute.make_layout(count),
                )
                cute.copy(
                    cute.make_copy_atom(
                        cute.nvgpu.CopyUniversalOp(),
                        out.element_type,
                        num_bits_per_copy=128,
                    ),
                    packed,
                    target,
                )


from cutlass.experimental.primitives import Tcgen05InstrDesc


@cute.experimental.jit
def _fp8_ws_m32_issue(address, desc_a, desc_b, instruction, accumulate):
    llvm.inline_asm(
        None,
        [
            cutlass.Int32(address).ir_value(),
            desc_a.ir_value(),
            desc_b.ir_value(),
            cutlass.Int32(instruction).ir_value(),
            cutlass.Int32(accumulate).ir_value(),
        ],
        "{ .reg .pred p; setp.ne.u32 p, $4, 0; tcgen05.mma.ws.cta_group::1.kind::f8f6f4.collector::b0::discard [$0], $1, $2, $3, p; }",
        "r,l,l,r,r,~{memory}",
        has_side_effects=True,
        asm_dialect=0,
    )


class Fp8GatedResidualM32WeightStationaryUp(Fp8GatedResidualGemm):
    """Map four eight-channel stripes to the four M32 TMEM banks."""

    @cute.experimental.jit
    def __call__(
        self,
        x,
        weight,
        original,
        weight_scale,
        activation_scale,
        partials,
        out,
        stream: CUstream,
    ):
        self.up_kernel(x, weight, original, weight_scale, activation_scale, out).launch(
            grid=(
                cute.ceil_div(self.hidden_size, self.up_hidden_tile),
                cute.ceil_div(x.shape[0], self.up_token_rows),
                1,
            ),
            block=(self.epilogue_threads + 64, 1, 1),
            stream=stream,
            use_pdl=self.pdl,
        )

    @cute.experimental.jit
    def _up_reader(self, base, original, tid, branch: cutlass.Constexpr):
        stripe = self.up_hidden_tile // (128 // self.up_token_rows)
        tile = (128, stripe // self.epilogue_groups)
        group = tid // 128
        pointer = cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, base)
        accumulator = cute.make_tensor(
            pointer + branch * stripe,
            cute.make_layout((128, stripe), stride=(65536, 1)),
        )
        accumulator = cute.local_tile(accumulator, tile, (0, group))
        coordinates = cute.local_tile(
            cute.make_identity_tensor((128, stripe)), tile, (0, group)
        )
        copy = tcgen05.make_tmem_copy(
            sm100_utils.get_tmem_load_op(
                (128, stripe, self.k),
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
            original.iterator, cute.make_layout(tile, stride=(1, 128))
        )
        values = cute_ext.allocate(
            cutlass.Float32,
            cute.AddressSpace.rmem,
            cute_ext.make_t2r_rmem_layout(
                copy, cute.flat_divide(dummy, tile), tid % 128
            ),
            alignment=32,
        )
        return (accumulator, reader, values, reader.partition_D(coordinates))

    @cute.experimental.kernel
    def up_kernel(self, x, weight, original, weight_scale, activation_scale, out):
        tid = cute.arch.thread_idx()[0]
        warp = cute.arch.make_warp_uniform(tid // 32)
        hidden_tile, token_tile, _ = cute.arch.block_idx()
        token_tile = cutlass.Int64(token_tile)
        dtype = x.element_type
        assert dtype == cutlass.Float8E4M3FN and self.shared_up_scales
        assert (self.k, self.stages, self.inner) == (128, 4, 1024)
        assert self.up_token_rows == 32
        assert self.up_hidden_tile == 32
        banks = 128 // self.up_token_rows
        stripe = self.up_hidden_tile // banks
        tmem_columns = stripe * self.hc_count
        # Build descriptors from the dense M64 shape. The M32 instruction reads its lower 32 A rows.
        a_rows = 64
        mma = sm100_utils.make_trivial_tiled_mma(
            dtype,
            dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            cutlass.Float32,
            tcgen05.CtaGroup.ONE,
            (a_rows, self.up_hidden_tile * self.hc_count),
        )
        tiler = (a_rows, self.up_hidden_tile * self.hc_count, self.k)
        a_tiler = (a_rows, self.up_hidden_tile * self.hc_count, 256)
        activation_layout = sm100_utils.make_smem_layout_a(mma, a_tiler, dtype, 2)
        raw_layout = sm100_utils.make_smem_layout_b(
            mma, tiler, cutlass.Float8E4M3FN, self.stages
        )
        sx = cute_ext.allocate(
            dtype, cute.AddressSpace.smem, activation_layout, alignment=1024
        )
        raw = cute_ext.allocate(
            cutlass.Float8E4M3FN, cute.AddressSpace.smem, raw_layout, alignment=1024
        )
        barriers = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(3 * self.stages + 2),
            alignment=8,
        ).iterator
        full = barriers
        converted = full + self.stages
        empty = converted + self.stages
        done, read_done = (empty + self.stages, empty + self.stages + 1)
        base = cute_ext.allocate(
            cutlass.Int32, cute.AddressSpace.smem, cute.make_layout(1), alignment=4
        ).iterator
        if warp == self.mma_warp:
            with cute.arch.elect_one():
                for stage in cutlass.range_constexpr(self.stages):
                    cute.arch.mbarrier_init(full + stage, 1)
                    cute.arch.mbarrier_init(converted + stage, 1)
                    cute.arch.mbarrier_init(empty + stage, 1)
                cute.arch.mbarrier_init(done, 1)
                cute.arch.mbarrier_init(read_done, self.epilogue_threads)
            cute.arch.alloc_tmem(tmem_columns, base, is_two_cta=False)
            cute.arch.relinquish_tmem_alloc_permit()
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        activation = cute.make_tensor(
            x.iterator,
            cute.make_layout(
                (x.shape[0], (64, self.inner // 64), 1), stride=(self.inner, (1, 64), 0)
            ),
        )
        weights = cute.make_tensor(
            weight.iterator,
            cute.make_layout(
                (
                    (
                        stripe,
                        self.hc_count,
                        banks,
                        self.hidden_size // self.up_hidden_tile,
                    ),
                    (64, self.inner // 64),
                    1,
                ),
                stride=(
                    (
                        self.inner,
                        self.hidden_size * self.inner,
                        stripe * self.inner,
                        self.up_hidden_tile * self.inner,
                    ),
                    (1, 64),
                    0,
                ),
            ),
        )
        gx = cute.local_tile(activation, (a_rows, 256), (token_tile, None, 0))
        gw = cute.local_tile(
            weights,
            (self.up_hidden_tile * self.hc_count, self.k),
            (hidden_tile, None, 0),
        )
        map_x = cute_ext.get_cta_v_map_ab(activation, a_tiler, mma, "A")
        map_w = cute_ext.get_cta_v_map_ab(weights, tiler, mma, "B")
        if warp == self.producer_warp:
            for stage in cutlass.range_constexpr(self.stages):
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        full + stage, self.up_hidden_tile * self.hc_count * self.k
                    )
                if cutlass.const_expr(self.weights_independent):
                    cute_ext.tma_load(
                        gw[None, None, stage],
                        raw[None, None, None, stage],
                        (full + stage).value,
                        cta_v_map=map_w,
                        update_expect_tx=False,
                    )
            if cutlass.const_expr(self.pdl):
                cute.arch.griddepcontrol_wait()
                cute.arch.griddepcontrol_launch_dependents()
            for stage in cutlass.range_constexpr(self.stages):
                if cutlass.const_expr(not self.weights_independent):
                    cute_ext.tma_load(
                        gw[None, None, stage],
                        raw[None, None, None, stage],
                        (full + stage).value,
                        cta_v_map=map_w,
                        update_expect_tx=False,
                    )
            for a_stage in cutlass.range_constexpr(2):
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        converted + a_stage, a_rows * 256
                    )
                cute_ext.tma_load(
                    gx[None, None, a_stage],
                    sx[None, None, None, a_stage],
                    (converted + a_stage).value,
                    cta_v_map=map_x,
                    update_expect_tx=False,
                )
            for tile in cutlass.range(self.stages, self.tiles, unroll=1):
                stage = tile % self.stages
                cute.arch.mbarrier_wait(empty + stage, (tile // self.stages - 1) % 2)
                if tile % 2 == 1:
                    a_tile = tile // 2
                    a_stage = a_tile % 2
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            converted + a_stage, a_rows * 256
                        )
                    cute_ext.tma_load(
                        gx[None, None, a_tile],
                        sx[None, None, None, a_stage],
                        (converted + a_stage).value,
                        cta_v_map=map_x,
                        update_expect_tx=False,
                    )
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        full + stage, self.up_hidden_tile * self.hc_count * self.k
                    )
                cute_ext.tma_load(
                    gw[None, None, tile],
                    raw[None, None, None, stage],
                    (full + stage).value,
                    cta_v_map=map_w,
                    update_expect_tx=False,
                )
        elif warp == self.mma_warp:
            thread_mma = mma.get_slice(0)
            a_desc = thread_mma.make_fragment_A(sx)
            b_desc = thread_mma.make_fragment_B(raw)
            address = cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, base).toint()
            instruction = Tcgen05InstrDesc.build(
                sparse_id2=0,
                sparse_flag=0,
                saturate=0,
                sparse_format=0,
                c_dtype=cutlass.Float32,
                a_dtype=dtype,
                b_dtype=dtype,
                a_negate=0,
                b_negate=0,
                a_major=0,
                b_major=0,
                n_dim=self.up_hidden_tile * self.hc_count,
                m_dim=self.up_token_rows,
                k_dim=0,
                max_shift=0,
            )
            for tile in cutlass.range(self.tiles, unroll=1):
                stage = tile % self.stages
                if tile % 2 == 0:
                    cute.arch.mbarrier_wait(converted + tile // 2 % 2, tile // 4 % 2)
                cute.arch.mbarrier_wait(full + stage, tile // self.stages % 2)
                with cute.arch.elect_one():
                    for block in cutlass.range_constexpr(self.k // 32):
                        da = tcgen05.smem_descriptor_to_int(
                            a_desc[
                                None,
                                0,
                                block + tile % 2 * (self.k // 32),
                                tile // 2 % 2,
                            ].iterator
                        )
                        db = tcgen05.smem_descriptor_to_int(
                            b_desc[None, 0, block, stage].iterator
                        )
                        _fp8_ws_m32_issue(
                            address,
                            da,
                            db,
                            instruction,
                            cutlass.Int32(tile > 0 or block > 0),
                        )
                with cute.arch.elect_one():
                    tcgen05.commit(empty + stage, None, tcgen05.CtaGroup.ONE)
            with cute.arch.elect_one():
                tcgen05.commit(done, None, tcgen05.CtaGroup.ONE)
            cute.arch.mbarrier_wait(read_done, 0)
            cute.arch.dealloc_tmem(
                cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, base), tmem_columns
            )
        elif tid < self.epilogue_threads:
            if cutlass.const_expr(self.pdl):
                cute.arch.griddepcontrol_wait()
            _, _, template, coordinates = self._up_reader(base, original, tid, 0)
            count = cute.size(template)
            residual = cute.make_rmem_tensor(
                (count, self.hc_count), original.element_type
            )
            scales = cute.make_rmem_tensor((1, 1), cutlass.Float32)
            residual.fill(0.0)
            sums = cute.make_rmem_tensor(count, cutlass.Float32)
            sums.fill(0.0)
            data_path, first_col = coordinates[0]
            first_row = data_path % self.up_token_rows
            first_hidden = (
                data_path
                // self.up_token_rows
                * (self.up_hidden_tile // (128 // self.up_token_rows))
                + first_col
            )
            global_row = token_tile * self.up_token_rows + first_row
            hidden = hidden_tile * self.up_hidden_tile + first_hidden
            scale_cache = cute_ext.allocate(
                cutlass.Float32,
                cute.AddressSpace.smem,
                cute.make_layout(self.up_hidden_tile * self.hc_count),
                alignment=32,
            )
            alpha_cache = cute_ext.allocate(
                cutlass.Float32,
                cute.AddressSpace.smem,
                cute.make_layout(self.up_token_rows),
                alignment=32,
            )
            if tid < self.up_hidden_tile * self.hc_count:
                channel = (
                    tid // self.up_hidden_tile * self.hidden_size
                    + hidden_tile * self.up_hidden_tile
                    + tid % self.up_hidden_tile
                )
                scale_cache[tid] = weight_scale[
                    channel if self.weight_scale_per_channel else 0
                ]
            if tid < self.up_token_rows:
                alpha_cache[tid] = 1.0
                if token_tile * self.up_token_rows + tid < x.shape[0]:
                    alpha_cache[tid] = activation_scale[
                        token_tile * self.up_token_rows + tid
                    ]
            _epilogue_barrier(self.epilogue_threads, cutlass.Int32(3))
            for branch in cutlass.range_constexpr(self.hc_count):
                self._up_prefetch(
                    original,
                    weight_scale,
                    activation_scale,
                    residual,
                    scales,
                    global_row,
                    hidden,
                    x.shape[0],
                    branch,
                    branch,
                    True,
                )
            cute.arch.mbarrier_wait(done, 0)
            for branch in cutlass.range_constexpr(self.hc_count):
                accumulator, reader, gates, _ = self._up_reader(
                    base, original, tid, branch
                )
                cute_ext.partition_and_copy(reader, accumulator, gates)
                cute.arch.fence_view_async_tmem_load()
                for index in cutlass.range_constexpr(count):
                    factor = (
                        scale_cache[branch * self.up_hidden_tile + first_hidden + index]
                        * alpha_cache[first_row]
                    )
                    sums[index] = sums[index] + residual[index, branch].to(
                        cutlass.Float32
                    ) * _up_sigmoid_tanh(gates[index] * factor)
            tcgen05_fence("before_thread_sync")
            cute.arch.mbarrier_arrive(read_done)
            if global_row < x.shape[0]:
                packed = cute.make_rmem_tensor(count, out.element_type)
                for index in cutlass.range_constexpr(count):
                    packed[index] = (sums[index] / self.hc_count).to(out.element_type)
                target = cute.make_tensor(
                    cute.make_ptr(
                        out.element_type,
                        (out.iterator + global_row * self.hidden_size + hidden).toint(),
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    ),
                    cute.make_layout(count),
                )
                cute.copy(
                    cute.make_copy_atom(
                        cute.nvgpu.CopyUniversalOp(),
                        out.element_type,
                        num_bits_per_copy=128,
                    ),
                    packed,
                    target,
                )

    @cute.experimental.jit
    def _up_prefetch(
        self,
        original,
        weight_scale,
        activation_scale,
        residual,
        scales,
        row,
        hidden,
        rows,
        branch: cutlass.Constexpr,
        slot: cutlass.Constexpr,
        native_fp8: cutlass.Constexpr,
    ):
        count = cute.size(residual, mode=[0])
        if row < rows:
            source = cute.make_tensor(
                cute.make_ptr(
                    original.element_type,
                    (
                        original.iterator
                        + row * self.wide
                        + branch * self.hidden_size
                        + hidden
                    ).toint(),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                ),
                cute.make_layout(count),
            )
            cute.copy(
                cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(),
                    original.element_type,
                    num_bits_per_copy=128,
                ),
                source,
                residual[None, slot],
            )
            if cutlass.const_expr(not self.shared_up_scales):
                if cutlass.const_expr(
                    self.weight_scale_per_channel and self.weight_scale_vectorized
                ):
                    source_scale = cute.make_tensor(
                        cute.make_ptr(
                            cutlass.Float32,
                            (
                                weight_scale.iterator
                                + branch * self.hidden_size
                                + hidden
                            ).toint(),
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        ),
                        cute.make_layout(count),
                    )
                    cute.copy(
                        cute.make_copy_atom(
                            cute.nvgpu.CopyUniversalOp(),
                            cutlass.Float32,
                            num_bits_per_copy=128,
                        ),
                        source_scale,
                        scales[None, slot],
                    )
                else:
                    for i in cutlass.range_constexpr(count):
                        scales[i, slot] = weight_scale[
                            (
                                branch * self.hidden_size + hidden + i
                                if self.weight_scale_per_channel
                                else 0
                            )
                        ]
                if cutlass.const_expr(native_fp8):
                    for i in cutlass.range_constexpr(count):
                        scales[i, slot] = scales[i, slot] * activation_scale[row]


class Fp8GatedResidualM64H32IndependentAUp(Fp8GatedResidualM64H56Up):
    """Use A256/two-stage and B128/four-stage operand rings."""

    @cute.experimental.jit
    def _up_reader(self, base, layout, original, tid, branch: cutlass.Constexpr):
        tile = (self.up_token_rows, 8)
        group = tid // 128
        accumulator = cute.make_tensor(
            cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, base), layout
        )[(None, None), 0, 0]
        column = branch * self.up_hidden_tile + group * 8
        accumulator = cute.make_tensor(
            accumulator.iterator + accumulator.layout((0, column)),
            cute.make_layout(
                (cute.shape(accumulator, mode=[0]), 8),
                stride=(accumulator.stride[0], accumulator.stride[1]),
            ),
        )
        coordinates = cute.local_tile(
            cute.make_identity_tensor((self.up_token_rows, 32)), tile, (0, group)
        )
        copy = tcgen05.make_tmem_copy(
            sm100_utils.get_tmem_load_op(
                (self.up_token_rows, self.up_hidden_tile * self.hc_count, self.k),
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
            original.iterator, cute.make_layout(tile, stride=(1, self.up_token_rows))
        )
        values = cute_ext.allocate(
            cutlass.Float32,
            cute.AddressSpace.rmem,
            cute_ext.make_t2r_rmem_layout(
                copy, cute.flat_divide(dummy, tile), tid % 128
            ),
            alignment=32,
        )
        return (accumulator, reader, values, reader.partition_D(coordinates))

    @cute.experimental.kernel
    def up_kernel(self, x, weight, original, weight_scale, activation_scale, out):
        tid = cute.arch.thread_idx()[0]
        warp = cute.arch.make_warp_uniform(tid // 32)
        hidden_tile, token_tile, _ = cute.arch.block_idx()
        token_tile = cutlass.Int64(token_tile)
        dtype = x.element_type
        native_fp8 = cutlass.const_expr(dtype == cutlass.Float8E4M3FN)
        assert native_fp8 and self.shared_up_scales
        assert (self.k, self.stages, self.inner, self.up_token_rows) == (
            128,
            4,
            1024,
            64,
        )
        mma = sm100_utils.make_trivial_tiled_mma(
            dtype,
            dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            cutlass.Float32,
            tcgen05.CtaGroup.ONE,
            (self.up_token_rows, self.up_hidden_tile * self.hc_count),
        )
        tiler = (self.up_token_rows, self.up_hidden_tile * self.hc_count, self.k)
        a_tiler = (self.up_token_rows, self.up_hidden_tile * self.hc_count, 256)
        activation_layout = sm100_utils.make_smem_layout_a(mma, a_tiler, dtype, 2)
        raw_layout = sm100_utils.make_smem_layout_b(
            mma, tiler, cutlass.Float8E4M3FN, self.stages
        )
        weight_layout = sm100_utils.make_smem_layout_b(mma, tiler, dtype, self.stages)
        sx = cute_ext.allocate(
            dtype, cute.AddressSpace.smem, activation_layout, alignment=1024
        )
        raw = cute_ext.allocate(
            cutlass.Float8E4M3FN, cute.AddressSpace.smem, raw_layout, alignment=1024
        )
        sw = raw
        if cutlass.const_expr(not native_fp8):
            sw = cute_ext.allocate(
                dtype, cute.AddressSpace.smem, weight_layout, alignment=1024
            )
        barriers = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(3 * self.stages + 2),
            alignment=8,
        ).iterator
        full = barriers
        converted = full + self.stages
        empty = converted + self.stages
        done, read_done = (empty + self.stages, empty + self.stages + 1)
        base = cute_ext.allocate(
            cutlass.Int32, cute.AddressSpace.smem, cute.make_layout(1), alignment=4
        ).iterator
        if warp == self.mma_warp:
            with cute.arch.elect_one():
                for stage in cutlass.range_constexpr(self.stages):
                    cute.arch.mbarrier_init(full + stage, 1)
                    cute.arch.mbarrier_init(converted + stage, 1)
                    cute.arch.mbarrier_init(empty + stage, 1)
                cute.arch.mbarrier_init(done, 1)
                cute.arch.mbarrier_init(read_done, self.epilogue_threads)
            cute.arch.alloc_tmem(128, base, is_two_cta=False)
            cute.arch.relinquish_tmem_alloc_permit()
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        activation = cute.make_tensor(
            x.iterator,
            cute.make_layout(
                (x.shape[0], (64, self.inner // 64), 1), stride=(self.inner, (1, 64), 0)
            ),
        )
        weights = cute.make_tensor(
            weight.iterator,
            cute.make_layout(
                (
                    (
                        self.up_hidden_tile,
                        self.hc_count,
                        self.hidden_size // self.up_hidden_tile,
                    ),
                    (64, self.inner // 64),
                    1,
                ),
                stride=(
                    (
                        self.inner,
                        self.hidden_size * self.inner,
                        self.up_hidden_tile * self.inner,
                    ),
                    (1, 64),
                    0,
                ),
            ),
        )
        gx = cute.local_tile(
            activation, (self.up_token_rows, 256), (token_tile, None, 0)
        )
        gw = cute.local_tile(
            weights,
            (self.up_hidden_tile * self.hc_count, self.k),
            (hidden_tile, None, 0),
        )
        map_x = cute_ext.get_cta_v_map_ab(activation, a_tiler, mma, "A")
        map_w = cute_ext.get_cta_v_map_ab(weights, tiler, mma, "B")
        if warp == self.producer_warp:
            for stage in cutlass.range_constexpr(self.stages):
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        full + stage, self.up_hidden_tile * self.hc_count * self.k
                    )
                if cutlass.const_expr(self.weights_independent):
                    cute_ext.tma_load(
                        gw[None, None, stage],
                        raw[None, None, None, stage],
                        (full + stage).value,
                        cta_v_map=map_w,
                        update_expect_tx=False,
                    )
            if cutlass.const_expr(self.pdl):
                cute.arch.griddepcontrol_wait()
                cute.arch.griddepcontrol_launch_dependents()
            for stage in cutlass.range_constexpr(self.stages):
                if cutlass.const_expr(not self.weights_independent):
                    cute_ext.tma_load(
                        gw[None, None, stage],
                        raw[None, None, None, stage],
                        (full + stage).value,
                        cta_v_map=map_w,
                        update_expect_tx=False,
                    )
            for a_stage in cutlass.range_constexpr(2):
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        converted + a_stage, self.up_token_rows * 256
                    )
                cute_ext.tma_load(
                    gx[None, None, a_stage],
                    sx[None, None, None, a_stage],
                    (converted + a_stage).value,
                    cta_v_map=map_x,
                    update_expect_tx=False,
                )
            for tile in cutlass.range(self.stages, self.tiles, unroll=1):
                stage = tile % self.stages
                cute.arch.mbarrier_wait(empty + stage, (tile // self.stages - 1) % 2)
                # The odd B-empty commit covers both K128 uses of the old A256 slot.
                if tile % 2 == 1:
                    a_tile = tile // 2
                    a_stage = a_tile % 2
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            converted + a_stage, self.up_token_rows * 256
                        )
                    cute_ext.tma_load(
                        gx[None, None, a_tile],
                        sx[None, None, None, a_stage],
                        (converted + a_stage).value,
                        cta_v_map=map_x,
                        update_expect_tx=False,
                    )
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        full + stage, self.up_hidden_tile * self.hc_count * self.k
                    )
                cute_ext.tma_load(
                    gw[None, None, tile],
                    raw[None, None, None, stage],
                    (full + stage).value,
                    cta_v_map=map_w,
                    update_expect_tx=False,
                )
        elif warp == self.mma_warp:
            thread_mma = mma.get_slice(0)
            a_desc = thread_mma.make_fragment_A(sx)
            b_desc = thread_mma.make_fragment_B(sw)
            fragment = thread_mma.make_fragment_C(
                mma.partition_shape_C(
                    (self.up_token_rows, self.up_hidden_tile * self.hc_count)
                )
            )
            accumulator = cute.make_tensor(
                cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, base), fragment.layout
            )
            mma.set(tcgen05.Field.ACCUMULATE, False)
            for tile in cutlass.range(self.tiles, unroll=1):
                stage = tile % self.stages
                ready = full if cutlass.const_expr(native_fp8) else converted
                if tile % 2 == 0:
                    cute.arch.mbarrier_wait(converted + tile // 2 % 2, tile // 4 % 2)
                cute.arch.mbarrier_wait(ready + stage, tile // self.stages % 2)
                for block in cutlass.range_constexpr(self.k // 32):
                    cute.gemm(
                        mma,
                        accumulator,
                        a_desc[
                            None, None, block + tile % 2 * (self.k // 32), tile // 2 % 2
                        ],
                        b_desc[None, None, block, stage],
                        accumulator,
                    )
                    mma.set(tcgen05.Field.ACCUMULATE, True)
                with cute.arch.elect_one():
                    tcgen05.commit(empty + stage, None, tcgen05.CtaGroup.ONE)
            with cute.arch.elect_one():
                tcgen05.commit(done, None, tcgen05.CtaGroup.ONE)
            cute.arch.mbarrier_wait(read_done, 0)
            cute.arch.dealloc_tmem(
                cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, base), 128
            )
        elif tid < self.epilogue_threads:
            if cutlass.const_expr(not native_fp8):
                for tile in cutlass.range(self.tiles, unroll=1):
                    stage = tile % self.stages
                    cute.arch.mbarrier_wait(full + stage, tile // self.stages % 2)
                    source, target = (
                        raw[None, None, None, stage],
                        sw[None, None, None, stage],
                    )
                    for offset in cutlass.range(
                        128 * self.k // (self.epilogue_threads * 8), unroll=4
                    ):
                        linear = (tid + offset * self.epilogue_threads) * 8
                        index = linear // self.k + linear % self.k * 128
                        src = cute.make_tensor(
                            cute.make_ptr(
                                cutlass.Float8E4M3FN,
                                (source.iterator + source.layout(index)).toint(),
                                cute.AddressSpace.smem,
                                assumed_align=8,
                                swizzle_=cute.get_swizzle_portion(raw_layout),
                            ),
                            cute.make_layout(8),
                        )
                        dst = cute.make_tensor(
                            cute.make_ptr(
                                dtype,
                                (target.iterator + target.layout(index)).toint(),
                                cute.AddressSpace.smem,
                                assumed_align=16,
                                swizzle_=cute.get_swizzle_portion(weight_layout),
                            ),
                            cute.make_layout(8),
                        )
                        raw_values = cute.make_rmem_tensor(8, cutlass.Float8E4M3FN)
                        converted_values = cute.make_rmem_tensor(8, dtype)
                        cute.copy(
                            cute.make_copy_atom(
                                cute.nvgpu.CopyUniversalOp(),
                                cutlass.Float8E4M3FN,
                                num_bits_per_copy=64,
                            ),
                            src,
                            raw_values,
                        )
                        converted_values.store(raw_values.load().to(dtype))
                        cute.copy(
                            cute.make_copy_atom(
                                cute.nvgpu.CopyUniversalOp(),
                                dtype,
                                num_bits_per_copy=128,
                            ),
                            converted_values,
                            dst,
                        )
                    cute.arch.fence_view_async_shared()
                    _epilogue_barrier(self.epilogue_threads, cutlass.Int32(1))
                    if tid == 0:
                        cute.arch.mbarrier_arrive(converted + stage)
            if cutlass.const_expr(self.pdl):
                cute.arch.griddepcontrol_wait()
            thread_mma = mma.get_slice(0)
            fragment = thread_mma.make_fragment_C(
                mma.partition_shape_C(
                    (self.up_token_rows, self.up_hidden_tile * self.hc_count)
                )
            )
            _, _, template, coordinates = self._up_reader(
                base, fragment.layout, original, tid, 0
            )
            count = cute.size(template)
            ring_slots = 4
            residual = cute.make_rmem_tensor((count, ring_slots), original.element_type)
            scales = cute.make_rmem_tensor(
                (1, 1) if self.shared_up_scales else (count, ring_slots),
                cutlass.Float32,
            )
            residual.fill(0.0)
            scales.fill(1.0)
            sums = cute.make_rmem_tensor(count, cutlass.Float32)
            sums.fill(0.0)
            first_row, first_col = coordinates[0]
            global_row = token_tile * self.up_token_rows + first_row
            hidden = hidden_tile * self.up_hidden_tile + first_col
            alpha = cutlass.Float32(1.0)
            if cutlass.const_expr(self.shared_up_scales):
                scale_cache = cute_ext.allocate(
                    cutlass.Float32,
                    cute.AddressSpace.smem,
                    cute.make_layout(128),
                    alignment=32,
                )
                if tid < 128:
                    scale_cache[tid] = 1.0
                    if tid < self.up_hidden_tile * self.hc_count:
                        channel = (
                            tid // self.up_hidden_tile * self.hidden_size
                            + hidden_tile * self.up_hidden_tile
                            + tid % self.up_hidden_tile
                        )
                        if (
                            hidden_tile * self.up_hidden_tile
                            + tid % self.up_hidden_tile
                            < self.hidden_size
                        ):
                            scale_cache[tid] = weight_scale[
                                channel if self.weight_scale_per_channel else 0
                            ]
                alpha_cache = cute_ext.allocate(
                    cutlass.Float32,
                    cute.AddressSpace.smem,
                    cute.make_layout(64),
                    alignment=32,
                )
                if tid < 64:
                    alpha_cache[tid] = 1.0
                    if token_tile * self.up_token_rows + tid < x.shape[0]:
                        alpha_cache[tid] = activation_scale[
                            token_tile * self.up_token_rows + tid
                        ]
                _epilogue_barrier(self.epilogue_threads, cutlass.Int32(3))
                if cutlass.const_expr(native_fp8):
                    if global_row < x.shape[0]:
                        alpha = activation_scale[global_row]
            if cutlass.const_expr(self.up_token_rows == 64):
                for branch in cutlass.range_constexpr(self.hc_count):
                    self._up_fragment_prefetch(
                        original,
                        weight_scale,
                        activation_scale,
                        residual,
                        scales,
                        coordinates,
                        token_tile,
                        hidden_tile,
                        x.shape[0],
                        branch,
                    )
            else:
                for branch in cutlass.range_constexpr(ring_slots):
                    self._up_prefetch(
                        original,
                        weight_scale,
                        activation_scale,
                        residual,
                        scales,
                        global_row,
                        hidden,
                        x.shape[0],
                        branch,
                        branch,
                        native_fp8,
                    )
            cute.arch.mbarrier_wait(done, 0)
            if (
                tid % 128 // 32 * (self.up_token_rows // 4)
                < x.shape[0] - token_tile * self.up_token_rows
            ):
                for branch in cutlass.range_constexpr(self.hc_count):
                    slot = branch % ring_slots
                    accumulator, reader, gates, _ = self._up_reader(
                        base, fragment.layout, original, tid, branch
                    )
                    cute_ext.partition_and_copy(reader, accumulator, gates)
                    cute.arch.fence_view_async_tmem_load()
                    for i in cutlass.range_constexpr(count):
                        factor = cutlass.Float32(1.0)
                        if cutlass.const_expr(self.shared_up_scales):
                            factor = (
                                scale_cache[
                                    branch * self.up_hidden_tile + coordinates[i][1]
                                ]
                                * alpha_cache[coordinates[i][0]]
                            )
                        else:
                            factor = scales[i, slot]
                        sums[i] = sums[i] + residual[i, slot].to(
                            cutlass.Float32
                        ) * _up_sigmoid_tanh(gates[i] * factor)
                    if cutlass.const_expr(branch + ring_slots < self.hc_count):
                        self._up_prefetch(
                            original,
                            weight_scale,
                            activation_scale,
                            residual,
                            scales,
                            global_row,
                            hidden,
                            x.shape[0],
                            branch + ring_slots,
                            slot,
                            native_fp8,
                        )
            tcgen05_fence("before_thread_sync")
            cute.arch.mbarrier_arrive(read_done)
            if cutlass.const_expr(self.up_token_rows == 64):
                for packet in cutlass.range_constexpr(count // 2):
                    row, col = coordinates[packet * 2]
                    output_row = token_tile * self.up_token_rows + row
                    if (
                        output_row < x.shape[0]
                        and col < self.up_hidden_tile
                        and (hidden_tile * self.up_hidden_tile + col < self.hidden_size)
                    ):
                        packed = cute.make_rmem_tensor(2, out.element_type)
                        for j in cutlass.range_constexpr(2):
                            packed[j] = (sums[packet * 2 + j] / self.hc_count).to(
                                out.element_type
                            )
                        target = cute.make_tensor(
                            cute.make_ptr(
                                out.element_type,
                                (
                                    out.iterator
                                    + output_row * self.hidden_size
                                    + hidden_tile * self.up_hidden_tile
                                    + col
                                ).toint(),
                                cute.AddressSpace.gmem,
                                assumed_align=4,
                            ),
                            cute.make_layout(2),
                        )
                        cute.copy(
                            cute.make_copy_atom(
                                cute.nvgpu.CopyUniversalOp(),
                                out.element_type,
                                num_bits_per_copy=32,
                            ),
                            packed,
                            target,
                        )
            elif global_row < x.shape[0]:
                packed = cute.make_rmem_tensor(count, out.element_type)
                for i in cutlass.range_constexpr(count):
                    packed[i] = (sums[i] / self.hc_count).to(out.element_type)
                target = cute.make_tensor(
                    cute.make_ptr(
                        out.element_type,
                        (out.iterator + global_row * self.hidden_size + hidden).toint(),
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    ),
                    cute.make_layout(count),
                )
                cute.copy(
                    cute.make_copy_atom(
                        cute.nvgpu.CopyUniversalOp(),
                        out.element_type,
                        num_bits_per_copy=128,
                    ),
                    packed,
                    target,
                )
