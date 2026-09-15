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

"""Experimental lane-cooperative NVFP4 preparation; not enabled by runtime.

Communication ownership and quantization ownership are independent: each lane
polls its BF16 fragment, while 2/4/8 lanes cooperate on one 16-value scale.
The identical math runs on plain local tensors to isolate quantizer cost.
"""

import functools

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.cutlass_dsl import T, dsl_user_op
from tokenspeed_kernel.thirdparty.cute_dsl.latent_moe_tail.primitives import (
    store_lamport_sentinel_128,
    to_cute,
    to_cute_dynamic_m,
)


def _fragment_asm(values, mailbox):
    words = values // 2
    registers = ", ".join(f"${i}" for i in range(words))
    vector = "" if words == 1 else f".v{words}"
    output = registers if words == 1 else "{" + registers + "}"
    load = (
        f"ld.{'volatile.' if mailbox else ''}global{vector}.u32 {output}, [${words}];"
    )
    lines = ["{", ".reg .pred dirty, item;", "AGAIN:", load]
    if mailbox:
        lines.append("setp.eq.u32 dirty, $0, 0x80008000;")
        for i in range(1, words):
            lines += [
                f"setp.eq.u32 item, ${i}, 0x80008000;",
                "or.pred dirty, dirty, item;",
            ]
        lines.append("@dirty bra AGAIN;")
    return "\n".join([*lines, "}"])


@dsl_user_op
def load_fragment(
    address,
    *,
    values: cutlass.Constexpr[int],
    mailbox: cutlass.Constexpr[bool],
    loc,
    ip,
):
    words = values // 2
    result = llvm.inline_asm(
        T.i32() if words == 1 else llvm.StructType.get_literal([T.i32()] * words),
        [address.ir_value(loc=loc, ip=ip)],
        _fragment_asm(values, mailbox),
        ",".join(["=r"] * words + ["l"]),
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    if words == 1:
        return (cutlass.Uint32(result),)
    return tuple(
        cutlass.Uint32(llvm.extractvalue(T.i32(), result, [i], loc=loc, ip=ip))
        for i in range(words)
    )


def _quantize_asm(values):
    words, lanes = values // 2, 16 // values
    scale_arg, lane_arg = 2 + words, 3 + words
    lines = [
        "{",
        f".reg .f32 v<{values}>, a, t, s, r, ginv, sixinv;",
        ".reg .b32 u, halves, peer, bits, mask, base, shift, packed;",
        ".reg .b16 sf, half0, half1;",
        ".reg .b8 b<4>;",
        ".reg .pred zero;",
        "mov.f32 a, 0f00000000;",
        "mov.b32 {b0, b1, b2, b3}, 0;",
    ]
    for i in range(values):
        arg = 2 + i // 2
        lines += [
            (
                f"shl.b32 u, ${arg}, 16;"
                if i % 2 == 0
                else f"and.b32 u, ${arg}, 0xffff0000;"
            ),
            f"mov.b32 v{i}, u;",
            f"abs.f32 t, v{i};",
            "max.f32 a, a, t;",
        ]
    lines += [
        f"and.b32 base, ${lane_arg}, {32-lanes};",
        f"mov.b32 mask, {(1 << lanes)-1};",
        "shl.b32 mask, mask, base;",
    ]
    for distance in (1, 2, 4):
        if distance < lanes:
            lines += [
                "mov.b32 bits, a;",
                f"shfl.sync.bfly.b32 peer, bits, {distance}, 31, mask;",
                "mov.b32 t, peer;",
                "max.f32 a, a, t;",
            ]
    lines += [
        "rcp.approx.ftz.f32 sixinv, 0f40c00000;",
        "mul.rn.f32 s, a, sixinv;",
        f"mul.rn.f32 s, s, ${scale_arg};",
        "cvt.rn.satfinite.e4m3x2.f32 sf, 0f00000000, s;",
        "cvt.u32.u16 $1, sf;",
        "cvt.rn.f16x2.e4m3x2 halves, sf;",
        "mov.b32 {half0, half1}, halves;",
        "cvt.f32.f16 s, half0;",
        f"rcp.approx.ftz.f32 ginv, ${scale_arg};",
        "mul.rn.f32 r, s, ginv;",
        "rcp.approx.ftz.f32 r, r;",
        "setp.eq.f32 zero, a, 0f00000000;",
        "selp.f32 r, 0f00000000, r, zero;",
    ]
    for i in range(values):
        lines.append(f"mul.rn.f32 v{i}, v{i}, r;")
    for i in range(words):
        lines.append(f"cvt.rn.satfinite.e2m1x2.f32 b{i}, v{2*i+1}, v{2*i};")
    lines += ["mov.b32 $0, {b0, b1, b2, b3};", "}"]
    return "\n".join(lines)


@dsl_user_op
def quantize_fragment(
    fragment, scale, lane, *, values: cutlass.Constexpr[int], loc, ip
):
    args = [value.ir_value(loc=loc, ip=ip) for value in fragment]
    args += [scale.ir_value(loc=loc, ip=ip), lane.ir_value(loc=loc, ip=ip)]
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        args,
        _quantize_asm(values),
        "=r,=r," + ",".join(["r"] * len(fragment) + ["f", "r"]),
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        cutlass.Uint32(llvm.extractvalue(T.i32(), result, [i], loc=loc, ip=ip))
        for i in range(2)
    )


@dsl_user_op
def pack_scales(scale, lane, *, values: cutlass.Constexpr[int], loc, ip):
    lanes = 16 // values
    subgroup = lanes * 4
    asm = [
        "{ .reg .b32 v, t, shift, mask, base;",
        f"shr.u32 shift, $2, {lanes.bit_length()-1};",
        "and.b32 shift, shift, 3;",
        "shl.b32 shift, shift, 3;",
        "shl.b32 v, $1, shift;",
        f"and.b32 base, $2, {32-subgroup};",
        f"mov.b32 mask, {(1 << subgroup)-1};",
        "shl.b32 mask, mask, base;",
        f"shfl.sync.bfly.b32 t, v, {lanes}, 31, mask;",
        "or.b32 v, v, t;",
        f"shfl.sync.bfly.b32 t, v, {lanes*2}, 31, mask;",
        "or.b32 $0, v, t; }",
    ]
    result = llvm.inline_asm(
        T.i32(),
        [scale.ir_value(loc=loc, ip=ip), lane.ir_value(loc=loc, ip=ip)],
        "\n".join(asm),
        "=r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return cutlass.Uint32(result)


class CooperativeNvfp4Input:
    def __init__(self, hidden, values, ctas, threads, mailbox, use_pdl):
        self.hidden, self.values = hidden, values
        self.ctas, self.threads = ctas, threads
        self.mailbox, self.use_pdl = mailbox, use_pdl

    @cute.jit
    def __call__(self, source, data, scales, scale, m, stream):
        self.kernel(source, data, scales, scale, m).launch(
            grid=(self.ctas, 1, 1),
            block=(self.threads, 1, 1),
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(self, source, data, scales, scale, m: cutlass.Int32):
        cute.arch.griddepcontrol_wait()
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        stride = cutlass.Int64(self.ctas * self.threads)
        first = cutlass.Int64(block * self.threads + tid)
        count = cutlass.Int64(m) * (self.hidden // self.values)
        item = first
        while item < count:
            element = item * self.values
            address = cutlass.Int64((source.iterator + element).toint())
            fragment = load_fragment(
                address, values=self.values, mailbox=self.mailbox, loc=None, ip=None
            )
            packed, sf = quantize_fragment(
                fragment,
                cutlass.Float32(scale[0]),
                cutlass.Int32(tid % 32),
                values=self.values,
                loc=None,
                ip=None,
            )
            output_address = (data.iterator + element // 2).llvm_ptr
            if cutlass.const_expr(self.values == 8):
                pointer = cute.make_ptr(
                    cutlass.Uint32,
                    output_address,
                    cute.AddressSpace.gmem,
                    assumed_align=4,
                )
                cute.make_tensor(pointer, cute.make_layout((1,)))[0] = packed
            elif cutlass.const_expr(self.values == 4):
                pointer = cute.make_ptr(
                    cutlass.Uint16,
                    output_address,
                    cute.AddressSpace.gmem,
                    assumed_align=2,
                )
                cute.make_tensor(pointer, cute.make_layout((1,)))[0] = cutlass.Uint16(
                    packed
                )
            else:
                data[element // 2] = cutlass.Uint8(packed)
            packed_sf = pack_scales(
                sf, cutlass.Int32(tid % 32), values=self.values, loc=None, ip=None
            )
            if tid % (64 // self.values) == 0:
                scales[element // 64] = packed_sf
            item = item + stride
        cute.arch.griddepcontrol_launch_dependents()
        if cutlass.const_expr(self.mailbox):
            item = first
            while item < count:
                if tid % (8 // self.values) == 0:
                    pointer = cute.make_ptr(
                        cutlass.BFloat16,
                        (source.iterator + item * self.values).llvm_ptr,
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    )
                    store_lamport_sentinel_128(pointer, sentinel=0x80008000)
                item = item + stride


@functools.cache
def compile_kernel(hidden, values, ctas, threads, mailbox, device, use_pdl):
    with torch.cuda.device(device):
        source = make_fake_compact_tensor(
            cutlass.BFloat16, (cute.sym_int32(divisibility=8),), assumed_align=16
        )
        data = make_fake_compact_tensor(
            cutlass.Uint8, (cute.sym_int32(divisibility=4),), assumed_align=16
        )
        scales = make_fake_compact_tensor(
            cutlass.Uint32, (cute.sym_int32(),), assumed_align=4
        )
        scale = make_fake_compact_tensor(cutlass.Float32, (1,), assumed_align=4)
        return cute.compile(
            CooperativeNvfp4Input(hidden, values, ctas, threads, mailbox, use_pdl),
            source,
            data,
            scales,
            scale,
            cutlass.Int32(1),
            make_fake_stream(),
        )


def launch(
    source,
    data,
    scales,
    global_scale,
    *,
    hidden,
    m,
    values,
    ctas,
    threads,
    mailbox,
    use_pdl,
):
    """Encode caller-owned buffers with 2/4/8 values per lane.

    Args:
        source: Contiguous CUDA BF16 storage covering at least M*hidden values.
        data: Contiguous CUDA uint8 [M, hidden/2] packed output.
        scales: Contiguous CUDA uint8 [M, hidden/16] linear scale output.
        global_scale: CUDA FP32 scalar encoding multiplier of the receiver.
        hidden: Logical latent width, a positive multiple of 64.
        m: Positive live row count; no graph padding is introduced.
        values: Values per lane (2, 4, or 8), independent of producer tiling.
        ctas: Positive grid width.
        threads: Threads per CTA, a multiple of 32, at most 1024.
        mailbox: Whether source is a Lamport mailbox, consumed then rearmed.
        use_pdl: Capture-time PDL policy, kept identical for compared variants.

    Returns:
        None. Writes packed payload/scales into the supplied buffers.
    """
    if values not in (2, 4, 8) or hidden <= 0 or hidden % 64 or m <= 0:
        raise ValueError("unsupported cooperative NVFP4 geometry")
    if ctas <= 0 or threads <= 0 or threads > 1024 or threads % 32:
        raise ValueError("invalid launch geometry")
    if (
        not source.is_cuda
        or source.dtype != torch.bfloat16
        or not source.is_contiguous()
        or source.numel() < m * hidden
    ):
        raise ValueError("source must cover live contiguous BF16 rows")
    for tensor, shape in ((data, (m, hidden // 2)), (scales, (m, hidden // 16))):
        if (
            tensor.dtype != torch.uint8
            or tensor.shape != shape
            or not tensor.is_contiguous()
            or tensor.device != source.device
        ):
            raise ValueError("invalid packed output or scale storage")
    if (
        global_scale.numel() != 1
        or global_scale.dtype != torch.float32
        or global_scale.device != source.device
    ):
        raise ValueError("invalid encoding multiplier storage")
    runner = compile_kernel(
        hidden, values, ctas, threads, mailbox, source.device.index, use_pdl
    )
    runner(
        to_cute_dynamic_m(source.flatten(), mode=0, assumed_align=16),
        to_cute_dynamic_m(data.flatten(), mode=0, assumed_align=16),
        to_cute_dynamic_m(scales.view(torch.uint32).flatten(), mode=0, assumed_align=4),
        to_cute(global_scale.reshape(1), assumed_align=4),
        cutlass.Int32(m),
        cuda.CUstream(torch.cuda.current_stream(source.device).cuda_stream),
    )
