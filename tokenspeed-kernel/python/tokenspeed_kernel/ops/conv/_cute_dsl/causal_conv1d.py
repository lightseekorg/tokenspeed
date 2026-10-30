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

"""SIMT causal convolution: one register window per eight-token tile."""

import math

import cuda.bindings.driver as cuda
import cutlass
from cutlass import BFloat16, Float32, Int32, Int64, Uint32, cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, vector
from cutlass.cutlass_dsl import T, dsl_user_op


@dsl_user_op
def _multiply_bf16(values, weights, *, loc=None, ip=None):
    """Round each product in BF16 using SIMT arithmetic available since Hopper."""
    count = cute.size(values.shape)
    if count == 1:
        operands = [
            llvm.bitcast(T.i16(), item[0].ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
            for item in (values, weights)
        ]
        product = llvm.inline_asm(
            T.i16(),
            operands,
            "mul.bf16 $0, $1, $2;",
            "=h,h,h",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
        result = vector.from_elements(
            ir.VectorType.get([1], BFloat16.mlir_type, loc=loc),
            [llvm.bitcast(BFloat16.mlir_type, product, loc=loc, ip=ip)],
            loc=loc,
            ip=ip,
        )
    else:
        packed_type = ir.VectorType.get([count // 2], T.i32(), loc=loc)
        a, b = [
            cute.TensorSSA(
                llvm.bitcast(
                    packed_type, item.ir_value(loc=loc, ip=ip), loc=loc, ip=ip
                ),
                count // 2,
                Uint32,
            )
            for item in (values, weights)
        ]
        products = [
            llvm.inline_asm(
                T.i32(),
                [a[i].ir_value(loc=loc, ip=ip), b[i].ir_value(loc=loc, ip=ip)],
                "mul.bf16x2 $0, $1, $2;",
                "=r,r,r",
                has_side_effects=False,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
                loc=loc,
                ip=ip,
            )
            for i in range(count // 2)
        ]
        packed = vector.from_elements(packed_type, products, loc=loc, ip=ip)
        result = llvm.bitcast(
            ir.VectorType.get([count], BFloat16.mlir_type, loc=loc),
            packed,
            loc=loc,
            ip=ip,
        )
    return cute.TensorSSA(result, count, BFloat16)


@dsl_user_op
def _add_bf16_product(acc, product, *, loc=None, ip=None):
    """Fold BF16 conversion into addition when the target supports mixed adds."""
    count = cute.size(product.shape)
    sums = [
        llvm.inline_asm(
            T.f32(),
            [
                llvm.bitcast(
                    T.i16(), product[i].ir_value(loc=loc, ip=ip), loc=loc, ip=ip
                ),
                acc[i].ir_value(loc=loc, ip=ip),
            ],
            "add.rn.f32.bf16 $0, $1, $2;",
            "=f,h,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
        for i in range(count)
    ]
    result = vector.from_elements(
        ir.VectorType.get([count], Float32.mlir_type, loc=loc), sums, loc=loc, ip=ip
    )
    return cute.TensorSSA(result, count, Float32)


@dsl_user_op
def _silu(value: Float32, *, loc=None, ip=None):
    # Match the existing convolution's exp approximation and division. These
    # scalar instructions do not require Tensor Cores or a particular GPU family.
    result = llvm.inline_asm(
        T.f32(),
        [value.ir_value(loc=loc, ip=ip)],
        "{ .reg .f32 n, e; mul.f32 n, $1, 0fBFB8AA3B; "
        "ex2.approx.f32 e, n; "
        "add.rn.f32 e, e, 0f3F800000; div.full.f32 $0, $1, e; }",
        "=f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Float32(result)


class CausalConv1dPrefill:
    """Compile one fixed model layout and tile; all request data stays dynamic."""

    def __init__(
        self,
        *,
        channels: int,
        input_stride: int,
        pointer_alignment: int,
        weight_alignment: int,
        state_strides: tuple[int, int, int],
        weight_strides: tuple[int, int],
        block_m: int,
        block_n: int,
        threads: int,
        min_blocks_per_mp: int,
        use_mixed_add: bool,
        has_bias: bool,
        silu: bool,
        use_pdl: bool,
    ):
        if threads <= 0 or threads > 1024 or block_m <= 0 or block_n <= 0:
            raise ValueError("convolution tiles and thread count must be positive")
        if block_m < 3:
            raise ValueError("the first token tile must cover all three history values")
        if min_blocks_per_mp <= 0:
            raise ValueError("minimum resident blocks must be positive")
        if block_n % threads or threads % 32:
            raise ValueError("channel tile must be a multiple of the thread count")
        vector_width = block_n // threads
        if vector_width & (vector_width - 1):
            raise ValueError("channels per thread must be a power of two")
        self.channels = channels
        self.input_stride = input_stride
        self.state_strides = state_strides
        self.weight_strides = weight_strides
        self.block_m = block_m
        self.block_n = block_n
        self.threads = threads
        self.vector = block_n // threads
        self.min_blocks_per_mp = min_blocks_per_mp
        self.use_mixed_add = use_mixed_add
        self.has_bias = has_bias
        self.silu = silu
        self.use_pdl = use_pdl
        self.input_align = math.gcd(
            pointer_alignment, 2 * self.vector, 2 * input_stride
        )
        self.output_align = math.gcd(pointer_alignment, 2 * self.vector, 2 * channels)
        self.weight_align = math.gcd(weight_alignment, 8 * self.vector)

    @cute.jit
    def __call__(
        self,
        x: cute.Pointer,
        weight: cute.Pointer,
        bias: cute.Pointer,
        state: cute.Pointer,
        slots: cute.Pointer,
        history: cute.Pointer,
        boundaries: cute.Pointer,
        batches: cute.Pointer,
        offsets: cute.Pointer,
        out: cute.Pointer,
        programs: Int32,
        stream: cuda.CUstream,
    ):
        flat = cute.make_layout(9223372036854775807)
        channel_tiles = cute.ceil_div(self.channels, self.block_n)
        # Channel tiles vary fastest so neighbouring CTAs share token rows.
        grid = (channel_tiles, programs, 1)
        self.causal_conv1d_prefill(
            cute.make_tensor(x, flat),
            cute.make_tensor(weight, flat),
            cute.make_tensor(bias, flat),
            cute.make_tensor(state, flat),
            cute.make_tensor(slots, flat),
            cute.make_tensor(history, flat),
            cute.make_tensor(boundaries, flat),
            cute.make_tensor(batches, flat),
            cute.make_tensor(offsets, flat),
            cute.make_tensor(out, flat),
        ).launch(
            grid=grid,
            block=(self.threads, 1, 1),
            stream=stream,
            use_pdl=self.use_pdl,
            min_blocks_per_mp=self.min_blocks_per_mp,
        )

    @cute.jit
    def _add(self, acc, product):
        if cutlass.const_expr(self.use_mixed_add):
            result = _add_bf16_product(acc, product)
        else:
            result = acc + product.to(Float32)
        return result

    @cute.jit
    def _load_row(
        self,
        x,
        state,
        start,
        row,
        length,
        channel,
        slot,
        history,
        first: cutlass.Constexpr,
        full: cutlass.Constexpr,
    ):
        values = cute.make_rmem_tensor((self.vector,), BFloat16)
        values.fill(0)
        if cutlass.const_expr(full and self.channels % self.block_n == 0):
            offset = cute.assume(
                (start + row) * self.input_stride + channel, divby=self.input_align // 2
            )
            address = x.iterator + offset
            source = cute.make_tensor(address, cute.make_layout(self.vector))
            cute.autovec_copy(source, values)
        elif row >= 0 and row < length:
            if cutlass.const_expr(self.channels % self.vector == 0):
                if channel < self.channels:
                    offset = cute.assume(
                        (start + row) * self.input_stride + channel,
                        divby=self.input_align // 2,
                    )
                    address = x.iterator + offset
                    source = cute.make_tensor(address, cute.make_layout(self.vector))
                    cute.autovec_copy(source, values)
            else:
                for v in cutlass.range_constexpr(self.vector):
                    if channel + v < self.channels:
                        values[v] = x[(start + row) * self.input_stride + channel + v]
        elif cutlass.const_expr(first):
            if row >= -3 and row < 0 and history:
                for v in cutlass.range_constexpr(self.vector):
                    if channel + v < self.channels:
                        values[v] = state[
                            slot * self.state_strides[0]
                            + (channel + v) * self.state_strides[1]
                            + (row + 3) * self.state_strides[2]
                        ]
        return values.load()

    @cute.jit
    def _store_row(self, out, row, channel, acc):
        result = cute.make_rmem_tensor((self.vector,), BFloat16)
        for v in cutlass.range_constexpr(self.vector):
            value = acc[v]
            if cutlass.const_expr(self.silu):
                value = _silu(value)
            result[v] = value.to(BFloat16)
        if cutlass.const_expr(self.channels % self.block_n == 0):
            offset = cute.assume(
                row * self.channels + channel, divby=self.output_align // 2
            )
            address = out.iterator + offset
            dest = cute.make_tensor(address, cute.make_layout(self.vector))
            cute.autovec_copy(result, dest)
        elif cutlass.const_expr(self.channels % self.vector == 0):
            if channel < self.channels:
                offset = cute.assume(
                    row * self.channels + channel, divby=self.output_align // 2
                )
                address = out.iterator + offset
                dest = cute.make_tensor(address, cute.make_layout(self.vector))
                cute.autovec_copy(result, dest)
        else:
            for v in cutlass.range_constexpr(self.vector):
                if channel + v < self.channels:
                    out[row * self.channels + channel + v] = result[v]

    @cute.jit
    def _body(
        self,
        x,
        weight,
        bias,
        state,
        out,
        start,
        token,
        length,
        channel,
        slot,
        history,
        first: cutlass.Constexpr,
        full: cutlass.Constexpr,
    ):
        weights = cute.make_rmem_tensor(cute.make_layout((4, self.vector)), BFloat16)
        weights.fill(0)
        initial = cute.make_rmem_tensor((self.vector,), Float32)
        initial.fill(0)
        if cutlass.const_expr(
            self.channels % self.block_n == 0 and self.weight_strides == (4, 1)
        ):
            offset = cute.assume(channel * 4, divby=self.weight_align // 2)
            source = cute.make_tensor(
                weight.iterator + offset, cute.make_layout((4, self.vector))
            )
            cute.autovec_copy(source, weights)
        else:
            for v in cutlass.range_constexpr(self.vector):
                if channel + v < self.channels:
                    for tap in cutlass.range_constexpr(4):
                        weights[tap, v] = weight[
                            (channel + v) * self.weight_strides[0]
                            + tap * self.weight_strides[1]
                        ]
        if cutlass.const_expr(self.has_bias):
            for v in cutlass.range_constexpr(self.vector):
                if (
                    cutlass.const_expr(self.channels % self.block_n == 0)
                    or channel + v < self.channels
                ):
                    initial[v] = bias[channel + v].to(Float32)

        # Exactly M+3 input rows cover all four windows in this CTA.
        window = cute.make_rmem_tensor(
            cute.make_layout((self.vector, self.block_m + 3)), BFloat16
        )
        for row in cutlass.range_constexpr(self.block_m + 3):
            window[None, row].store(
                self._load_row(
                    x,
                    state,
                    start,
                    token + row - 3,
                    length,
                    channel,
                    slot,
                    history,
                    first,
                    full,
                )
            )
        # Every output reads the immutable register window. Static expansion
        # keeps output chains independent without retaining every FP32 sum.
        for row in cutlass.range_constexpr(self.block_m):
            if cutlass.const_expr(full) or token + row < length:
                acc = initial.load()
                for tap in cutlass.range_constexpr(4):
                    value = window[None, row + tap].load()
                    w = weights[tap, None].load()
                    product = _multiply_bf16(value, w)
                    acc = self._add(acc, product)
                self._store_row(out, start + token + row, channel, acc)

        if cutlass.const_expr(first):
            # The request's last three inputs (or old history when it is
            # shorter) become its new history.
            tail = cute.make_rmem_tensor(cute.make_layout((self.vector, 3)), BFloat16)
            for row in cutlass.range_constexpr(3):
                tail[None, row].store(
                    self._load_row(
                        x,
                        state,
                        start,
                        length - 3 + row,
                        length,
                        channel,
                        slot,
                        history,
                        True,
                        False,
                    )
                )
            # Each thread owns these channels and has read its old history.
            # Later token tiles read immutable input, never this state.
            for v in cutlass.range_constexpr(self.vector):
                if channel + v < self.channels:
                    for row in cutlass.range_constexpr(3):
                        state[
                            slot * self.state_strides[0]
                            + (channel + v) * self.state_strides[1]
                            + row * self.state_strides[2]
                        ] = tail[v, row]

    @cute.kernel
    def causal_conv1d_prefill(
        self, x, weight, bias, state, slots, history, boundaries, batches, offsets, out
    ):
        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()
        channel_tile, program, _ = cute.arch.block_idx()
        channel = (
            Int64(channel_tile) * self.block_n + cute.arch.thread_idx()[0] * self.vector
        )
        request = batches[program]
        if request >= 0:
            start = Int64(boundaries[request])
            length = Int64(boundaries[request + 1]) - start
            token = Int64(offsets[program]) * self.block_m
            slot = Int64(slots[request])
            if token < length and slot >= 0:
                if token == 0:
                    self._body(
                        x,
                        weight,
                        bias,
                        state,
                        out,
                        start,
                        token,
                        length,
                        channel,
                        slot,
                        history[request] != 0,
                        True,
                        False,
                    )
                elif token + self.block_m <= length:
                    self._body(
                        x,
                        weight,
                        bias,
                        state,
                        out,
                        start,
                        token,
                        length,
                        channel,
                        slot,
                        False,
                        False,
                        True,
                    )
                else:
                    self._body(
                        x,
                        weight,
                        bias,
                        state,
                        out,
                        start,
                        token,
                        length,
                        channel,
                        slot,
                        False,
                        False,
                        False,
                    )
