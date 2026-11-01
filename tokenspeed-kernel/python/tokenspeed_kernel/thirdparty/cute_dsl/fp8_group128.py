# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""CuTe DSL BF16/group128 -> E4M3 + packed UE8M0 on SM100/SM103.

Scheduling follows SGLang's per_token_group_quant flat kernel (cf04168b).
This is an independent DSL implementation; it does not import SGLang.
Eight lanes own a group; each lane handles 16 contiguous BF16 elements.
The vector arithmetic is pinned with PTX to avoid scalar FP32 expansion.
"""

import functools

import cuda.bindings.driver as cuda
import cutlass
import torch
from cutlass import cute
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import make_ptr
from cutlass.cutlass_dsl import T, dsl_user_op


@dsl_user_op
def quantize_lane(src: cute.Pointer, dst: cute.Pointer, *, loc=None, ip=None):
    """Quantize 16 BF16s per lane; return the eight-lane group's exponent.

    The input address must be 32-byte aligned. Subgroup masks allow partial
    final warps without requiring inactive groups to read or duplicate stores.
    Constants are FP32 1e-10 and FP32 1/448. SATFINITE conversion handles
    saturation and preserves NaNs without a separate BF16 clamp.
    """
    asm = """{
    .reg .b32 v<8>, a, b, t, ex, inv, scale, mask, lane, out<4>;
    .reg .b16 q<8>;
    .reg .f32 f, g;
    mov.u32 lane, %laneid;
    and.b32 lane, lane, 24;
    shl.b32 mask, 255, lane;
    ld.global.v8.b32 {v0,v1,v2,v3,v4,v5,v6,v7}, [$1];
    and.b32 a, v0, 0x7fff7fff;
    """
    for i in range(1, 8):
        asm += f"and.b32 b, v{i}, 0x7fff7fff; max.bf16x2 a, a, b;\n"
    for offset in (4, 2, 1):
        asm += f"shfl.sync.bfly.b32 b, a, {offset}, 31, mask; max.bf16x2 a, a, b;\n"
    asm += """
    shl.b32 t, a, 16; mov.b32 f, t;
    and.b32 t, a, 0xffff0000; mov.b32 g, t;
    max.f32 f, f, g;
    max.f32 f, f, 0f2edbe6ff;
    mul.f32 f, f, 0f3b124925;
    mov.b32 t, f;
    // raw_scale is positive: ceil its exponent by carrying the mantissa.
    add.u32 t, t, 0x7fffff;
    shr.u32 ex, t, 23;
    sub.u32 inv, 254, ex;
    shl.b32 inv, inv, 7;
    shl.b32 scale, inv, 16;
    or.b32 scale, scale, inv;

    """
    for i in range(8):
        asm += f"""
        mul.rn.bf16x2 v{i}, v{i}, scale;
        cvt.rn.satfinite.e4m3x2.bf16x2 q{i}, v{i};
        """
    for i in range(4):
        asm += f"mov.b32 out{i}, {{q{2 * i},q{2 * i + 1}}};\n"
    asm += "st.global.v4.b32 [$2], {out0,out1,out2,out3}; mov.b32 $0, ex; }"
    return cutlass.Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                src.toint(loc=loc, ip=ip).ir_value(),
                dst.toint(loc=loc, ip=ip).ir_value(),
            ],
            asm,
            "=r,l,l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


class QuantPack:
    """Flat eight-lane groups, with dynamic row count and scale stride."""

    def __init__(self, width: int, threads: int, use_pdl: bool):
        self.width = width
        self.threads = threads
        self.use_pdl = use_pdl

    @cute.jit
    def __call__(
        self,
        x: cute.Pointer,
        q: cute.Pointer,
        s: cute.Pointer,
        rows: cutlass.Int32,
        padded_rows: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        groups = rows * (self.width // 128)
        self.kernel(x, q, s, rows, padded_rows).launch(
            grid=(cute.ceil_div(groups * 8, self.threads), 1, 1),
            block=(self.threads, 1, 1),
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Pointer,
        q: cute.Pointer,
        s: cute.Pointer,
        rows: cutlass.Int32,
        padded_rows: cutlass.Int32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        tid = cutlass.Uint32(tid)
        bid = cutlass.Uint32(bid)
        rows = cutlass.Uint32(rows)
        padded_rows = cutlass.Uint32(padded_rows)
        group = (bid * self.threads + tid) // cutlass.Uint32(8)
        lane = tid % cutlass.Uint32(8)
        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()
        # Initialize MN padding in the same launch; no padding reads or memset.
        if bid == 0:
            scale_words = cute.make_tensor(
                cute.recast_ptr(s, dtype=cutlass.Uint32),
                cute.make_layout(
                    (padded_rows * ((self.width + 511) // 512),), stride=(1,)
                ),
            )
            for pad in cutlass.range_constexpr(3):
                if rows + pad < padded_rows:
                    for pack in range(tid, (self.width + 511) // 512, self.threads):
                        scale_words[pack * padded_rows + rows + pad] = cutlass.Uint32(0)
        # A whole eight-lane subgroup is active or inactive, including tails.
        if group < rows * (self.width // 128):
            offset = cutlass.Int64(group) * 128 + lane * 16
            exponent = quantize_lane(x + offset * 2, q + offset)
            if lane == 0:
                row = group // cutlass.Uint32(self.width // 128)
                col = group % cutlass.Uint32(self.width // 128)
                byte = cutlass.Int64(col // 4) * padded_rows * 4 + row * 4 + col % 4
                scale = cute.make_tensor(
                    s,
                    cute.make_layout(
                        (padded_rows * ((self.width + 511) // 512) * 4,), stride=(1,)
                    ),
                )
                scale[byte] = cutlass.Uint8(exponent)
                # Keep the compile-time K-tail gate separate from the row predicate.
                if cutlass.const_expr((self.width // 128) % 4 != 0):  # noqa: SIM102
                    if col == self.width // 128 - 1:
                        for tail in cutlass.range_constexpr(
                            1, 4 - (self.width // 128) % 4 + 1
                        ):
                            scale[byte + tail] = cutlass.Uint8(0)
        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_launch_dependents()


def ptr(tensor):
    return make_ptr(
        cutlass.Uint8, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
    )


@functools.cache
def compile_kernel(width, threads, use_pdl, device):
    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("warm the CuTe FP8 quantizer before CUDA Graph capture")
        p = make_ptr(cutlass.Uint8, 0, cute.AddressSpace.gmem, assumed_align=16)
        return cute.compile(
            QuantPack(width, threads, use_pdl),
            p,
            p,
            p,
            cutlass.Int32(1),
            cutlass.Int32(4),
            cuda.CUstream(0),
        )


def launch(x, q, s, *, threads, use_pdl):
    """Write preallocated codes/scales on x's current CUDA stream.

    x is contiguous BF16 with 32-byte alignment and K divisible by 128;
    q is contiguous E4M3 with 16-byte alignment. s has the packed int32
    layout [M, ceil(K/512)], strides [1, round_up(M,4)], and zero offset.
    The caller owns non-overlapping buffers. threads is a multiple of 32.
    """
    width = x.shape[-1]
    rows = x.numel() // width
    if rows == 0:
        return
    with torch.cuda.device(x.device):
        compiled = compile_kernel(width, threads, use_pdl, x.device.index)
        compiled(
            ptr(x),
            ptr(q),
            ptr(s),
            cutlass.Int32(rows),
            cutlass.Int32(s.stride(1)),
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        )
