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

"""Ordered BF16 route reduction with block-FP8 quantization."""

import threading

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float8E4M3FN, Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from tokenspeed_kernel.thirdparty.cute_dsl.latent_moe_tail.primitives import (
    to_cute,
    to_cute_dynamic_m,
)

_compiled = {}
_compile_lock = threading.Lock()


@dsl_user_op
def load_bf16x4(ptr: cute.Pointer, *, loc=None, ip=None):
    """Load four BF16 values as FP32 from a valid eight-byte-aligned address."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [ptr.toint(loc=loc, ip=ip).ir_value()],
        "{ .reg .b64 packed; .reg .b16 a,b,c,d; "
        "ld.global.b64 packed,[$4]; mov.b64 {a,b,c,d},packed; "
        "cvt.f32.bf16 $0,a; cvt.f32.bf16 $1,b; "
        "cvt.f32.bf16 $2,c; cvt.f32.bf16 $3,d; }",
        "=f,=f,=f,=f,l,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        Float32(llvm.extractvalue(T.f32(), result, [i], loc=loc, ip=ip))
        for i in range(4)
    )


@dsl_user_op
def divide_rn(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """Explicit round-to-nearest division for the selected dense quantizer."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "div.rn.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def store_fp8x4(
    ptr: cute.Pointer,
    a: Float32,
    b: Float32,
    c: Float32,
    d: Float32,
    *,
    loc=None,
    ip=None,
):
    """Store a,b,c,d as E4M3 bytes to an aligned contiguous 4B destination.

    Inputs already have the quantization multiplier and clamp applied. PTX first source
    occupies the high byte, so each pair passes the later value first.
    """
    llvm.inline_asm(
        None,
        [
            ptr.toint(loc=loc, ip=ip).ir_value(),
            a.ir_value(),
            b.ir_value(),
            c.ir_value(),
            d.ir_value(),
        ],
        "{ .reg .b16 lo,hi; .reg .b32 word; "
        "cvt.rn.satfinite.e4m3x2.f32 lo,$2,$1; "
        "cvt.rn.satfinite.e4m3x2.f32 hi,$4,$3; "
        "mov.b32 word,{lo,hi}; st.global.b32 [$0],word; }",
        "l,f,f,f,f,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


class CombineQuantKernel:
    def __init__(self, topk: int, column_map: bool, hidden: int):
        self.topk = topk
        self.column_map = column_map
        self.hidden = hidden
        self.quant_blocks = hidden // 128

    @cute.jit
    def __call__(
        self,
        partial: cute.Tensor,
        inverse: cute.Tensor,
        weights: cute.Tensor,
        qbuf: cute.Tensor,
        sq: cute.Tensor,
        valid_rows: cute.Tensor,
        tokens: Int32,
        blocks: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(partial, inverse, weights, qbuf, sq, valid_rows, tokens).launch(
            grid=(blocks, (self.quant_blocks + 3) // 4, 1),
            block=(160, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        partial: cute.Tensor,
        inverse: cute.Tensor,
        weights: cute.Tensor,
        qbuf: cute.Tensor,
        sq: cute.Tensor,
        valid_rows: cute.Tensor,
        tokens: Int32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        block, column_tile, _ = cute.arch.block_idx()
        lane = tid % 32
        warp = tid // 32
        kb_lane = column_tile * 4 + warp
        if tid < 128 and kb_lane < self.quant_blocks:
            for token in cutlass.range(
                block, tokens, cute.arch.grid_dim()[0], unroll=1
            ):
                valid = token < tokens
                values = cute.make_rmem_tensor((4,), Float32)
                # Exact FP32 representation of BF16(1e-10), bits 0x2edc0000.
                # Constructing a DSL BF16 literal and converting it can fold without rounding.
                maximum = Float32(1.000444171950221e-10)
                # One slot lookup feeds all four independent columns. Keep each
                # column's increasing slot-order FMA sequence, including skipped rows.
                values.fill(0.0)
                if valid:
                    for slot in cutlass.range_constexpr(self.topk):
                        if cutlass.const_expr(self.column_map):
                            row = inverse[slot * tokens + token]
                        else:
                            row = inverse[token * self.topk + slot]
                        if (
                            row >= 0
                            and row < valid_rows[0]
                            and row < partial.shape[0] // self.hidden
                        ):
                            route_weight = weights[token * self.topk + slot]
                            loaded = load_bf16x4(
                                partial.iterator
                                + row * self.hidden
                                + kb_lane * 128
                                + lane * 4
                            )
                            for i in cutlass.range_constexpr(4):
                                col = kb_lane * 128 + lane * 4 + i
                                values[i] = values[i] + route_weight * loaded[i]
                for i in cutlass.range_constexpr(4):
                    col = kb_lane * 128 + lane * 4 + i
                    acc = values[i]
                    # Preserve the native BF16 output rounding before quantization.
                    value = acc.to(BFloat16).to(Float32)
                    values[i] = value
                    absolute = value
                    if absolute < 0:
                        absolute = -absolute
                    if absolute > maximum:
                        maximum = absolute
                for shift in cutlass.range_constexpr(5):
                    other = cute.arch.shuffle_sync_bfly(maximum, offset=1 << shift)
                    if other > maximum:
                        maximum = other
                # Block quantization uses round-to-nearest divisions.
                multiplier = Float32(1)
                if maximum != Float32(0):
                    multiplier = divide_rn(Float32(448), maximum)
                scale = divide_rn(Float32(1), multiplier)
                if lane == 0:
                    sq[token + kb_lane * tokens] = scale
                for i in cutlass.range_constexpr(4):
                    col = kb_lane * 128 + lane * 4 + i
                    value = values[i] * multiplier
                    if value > Float32(448):
                        value = Float32(448)
                    if value < Float32(-448):
                        value = Float32(-448)
                    values[i] = value
                store_fp8x4(
                    qbuf.iterator + token * self.hidden + kb_lane * 128 + lane * 4,
                    values[0],
                    values[1],
                    values[2],
                    values[3],
                )


def run_combine_quant(partial, inverse, weights, valid_rows, output):
    q, scales = output
    tokens, topk = weights.shape
    hidden = partial.shape[1]
    column_map = inverse.stride(0) == 1 and inverse.stride(1) == tokens
    key = (partial.device.index, topk, column_map, hidden)
    with torch.cuda.device(partial.device):
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        with _compile_lock:
            compiled = _compiled.get(key)
            if compiled is None:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "Warm combine/quantization before CUDA graph capture"
                    )
                fake = cute.runtime.make_fake_tensor
                compiled = cute.compile(
                    CombineQuantKernel(topk, column_map, hidden),
                    fake(BFloat16, (cute.sym_int(),), (1,), assumed_align=16),
                    fake(Int32, (cute.sym_int(),), (1,), assumed_align=4),
                    fake(Float32, (cute.sym_int(),), (1,), assumed_align=4),
                    fake(Float8E4M3FN, (cute.sym_int(),), (1,), assumed_align=16),
                    fake(Float32, (cute.sym_int(),), (1,), assumed_align=4),
                    fake(Int64, (1,), (1,), assumed_align=8),
                    Int32(1),
                    Int32(1),
                    stream,
                    options="--opt-level=2",
                )
                _compiled[key] = compiled

        # Dense physical views avoid ambiguous unit strides when tokens == 1.
        # Logical token/slot and token/scale offsets are explicit in the kernel.
        def physical(tensor, alignment):
            flat = tensor.as_strided((tensor.numel(),), (1,))
            return to_cute_dynamic_m(flat, mode=0, assumed_align=alignment)

        compiled(
            physical(partial, 16),
            physical(inverse, 4),
            physical(weights, 4),
            physical(q, 16),
            physical(scales, 4),
            to_cute(valid_rows, 8),
            Int32(tokens),
            Int32(tokens),
            stream,
        )
    return output
