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

"""Fused BF16 mailbox/gather input preparation for NVFP4 SiTU MoE.

The numeric primitive follows FlashInfer's default (fast-math, not 4over6)
NVFP4 recipe. GEMM results have already been rounded to BF16. Communication
never reduces values: TP ranks own disjoint groups of 16 output columns.
"""

from __future__ import annotations

import functools

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.cutlass_dsl import T, dsl_user_op
from tokenspeed_kernel.platform import pdl_enabled
from tokenspeed_kernel.thirdparty.cute_dsl.latent_moe_tail.primitives import (
    fragment_is_dirty,
    load_global_u32x4,
    store_lamport_sentinel_128,
    to_cute,
    to_cute_dynamic_m,
)

PLAIN = 0
MAILBOX = 1
MULTICAST = 2
SENTINEL = 0x80008000


def _quantize_asm() -> str:
    lines = [
        "{",
        ".reg .f32 v<16>, a, t, s, r, ginv, sixinv;",
        ".reg .b32 u, halves;",
        ".reg .b16 sf, half0, half1;",
        ".reg .b8 b<8>;",
        ".reg .pred zero;",
        "mov.f32 a, 0f00000000;",
    ]
    for i in range(16):
        arg = 3 + i // 2
        lines.append(
            f"shl.b32 u, ${arg}, 16;"
            if i % 2 == 0
            else f"and.b32 u, ${arg}, 0xffff0000;"
        )
        lines.extend([f"mov.b32 v{i}, u;", f"abs.f32 t, v{i};", "max.f32 a, a, t;"])
    lines.extend(
        [
            "rcp.approx.ftz.f32 sixinv, 0f40c00000;",
            "mul.rn.f32 s, a, sixinv;",
            "mul.rn.f32 s, s, $11;",
            "cvt.rn.satfinite.e4m3x2.f32 sf, 0f00000000, s;",
            "cvt.u32.u16 $2, sf;",
            "cvt.rn.f16x2.e4m3x2 halves, sf;",
            "mov.b32 {half0, half1}, halves;",
            "cvt.f32.f16 s, half0;",
            "rcp.approx.ftz.f32 ginv, $11;",
            "mul.rn.f32 r, s, ginv;",
            "rcp.approx.ftz.f32 r, r;",
            "setp.eq.f32 zero, a, 0f00000000;",
            "selp.f32 r, 0f00000000, r, zero;",
        ]
    )
    for i in range(16):
        lines.append(f"mul.rn.f32 v{i}, v{i}, r;")
    for i in range(8):
        lines.append(f"cvt.rn.satfinite.e2m1x2.f32 b{i}, v{2*i+1}, v{2*i};")
    lines.extend(
        [
            "mov.b32 $0, {b0, b1, b2, b3};",
            "mov.b32 $1, {b4, b5, b6, b7};",
            "}",
        ]
    )
    return "\n".join(lines)


_QUANT_ASM = _quantize_asm()


@dsl_user_op
def quantize_group16(
    first, second, global_scale: cutlass.Float32, *, loc=None, ip=None
):
    """Return two packed E2M1 words and one E4M3 scale byte."""
    args = [p[i].ir_value(loc=loc, ip=ip) for p in (first, second) for i in range(4)]
    args.append(global_scale.ir_value(loc=loc, ip=ip))
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 3),
        args,
        _QUANT_ASM,
        "=r,=r,=r," + ",".join(["r"] * 8) + ",f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        cutlass.Uint32(llvm.extractvalue(T.i32(), result, [i], loc=loc, ip=ip))
        for i in range(3)
    )


@dsl_user_op
def multicast_word(address: cutlass.Int64, value: cutlass.Uint32, *, loc=None, ip=None):
    """Publish four bytes; scale shards need only four-byte alignment."""
    llvm.inline_asm(
        None,
        [address.ir_value(loc=loc, ip=ip), value.ir_value(loc=loc, ip=ip)],
        "multimem.st.relaxed.sys.global.b32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def store_packed_pair(
    address: cutlass.Int64,
    first: cutlass.Uint32,
    second: cutlass.Uint32,
    *,
    multicast: cutlass.Constexpr[bool],
    loc=None,
    ip=None,
):
    """Consecutive lanes publish consecutive eight-byte payloads."""
    asm = (
        "{ .reg .b64 packed; mov.b64 packed, {$1, $2}; "
        "multimem.st.relaxed.sys.global.b64 [$0], packed; }"
        if multicast
        else "st.global.v2.u32 [$0], {$1, $2};"
    )
    llvm.inline_asm(
        None,
        [
            address.ir_value(loc=loc, ip=ip),
            first.ir_value(loc=loc, ip=ip),
            second.ir_value(loc=loc, ip=ip),
        ],
        asm,
        "l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def pack_scale_quad(scale: cutlass.Uint32, lane: cutlass.Int32, *, loc=None, ip=None):
    """Pack four adjacent lanes' scale bytes without byte multicast stores."""
    result = llvm.inline_asm(
        T.i32(),
        [scale.ir_value(loc=loc, ip=ip), lane.ir_value(loc=loc, ip=ip)],
        "{ .reg .b32 v, t, shift, mask; and.b32 shift, $2, 3; "
        "shl.b32 shift, shift, 3; shl.b32 v, $1, shift; "
        "and.b32 shift, $2, 28; mov.b32 mask, 15; shl.b32 mask, mask, shift; "
        "shfl.sync.bfly.b32 t, v, 1, 31, mask; or.b32 v, v, t; "
        "shfl.sync.bfly.b32 t, v, 2, 31, mask; or.b32 $0, v, t; }",
        "=r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return cutlass.Uint32(result)


@dsl_user_op
def exchange_signal(
    address: cutlass.Int64, *, send: cutlass.Constexpr[bool], loc=None, ip=None
):
    """Release 0->1 or acquire 1->0 a system-scope per-CTA signal."""
    before, after, order = (0, 1, "release") if send else (1, 0, "acquire")
    llvm.inline_asm(
        None,
        [address.ir_value(loc=loc, ip=ip)],
        "{ .reg .b32 old; .reg .pred retry; AGAIN: "
        f"atom.cas.{order}.sys.global.b32 old, [$0], {before}, {after}; "
        f"setp.ne.b32 retry, old, {before}; @retry bra AGAIN; }}",
        "l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def alias_fence(*, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [],
        "fence.proxy.alias;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


class Nvfp4InputKernel:
    def __init__(
        self, hidden: int, mode: int, rank: int, world: int, ctas: int, threads: int
    ):
        self.hidden = hidden
        self.mode = mode
        self.rank = rank
        self.world = world
        self.ctas = ctas
        self.threads = threads
        self.use_pdl = pdl_enabled()
        self.input_hidden = hidden // world if mode == MULTICAST else hidden

    @cute.jit
    def __call__(
        self,
        source: cute.Tensor,
        data: cute.Tensor,
        scales: cute.Tensor,
        global_scale: cute.Tensor,
        signals: cute.Tensor,
        data_mc: cutlass.Int64,
        scale_mc: cutlass.Int64,
        m: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            source, data, scales, global_scale, signals, data_mc, scale_mc, m
        ).launch(
            grid=(self.ctas, 1, 1),
            block=(self.threads, 1, 1),
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.jit
    def barrier(self, signals, tid, block):
        if tid < self.world:
            send = cutlass.Int64(signals[tid]) + (block * self.world + self.rank) * 4
            recv = cutlass.Int64(signals[self.rank]) + (block * self.world + tid) * 4
            exchange_signal(send, send=True)
            exchange_signal(recv, send=False)
        cute.arch.sync_threads()

    @cute.kernel
    def kernel(
        self,
        source: cute.Tensor,
        data: cute.Tensor,
        scales: cute.Tensor,
        global_scale: cute.Tensor,
        signals: cute.Tensor,
        data_mc: cutlass.Int64,
        scale_mc: cutlass.Int64,
        m: cutlass.Int32,
    ):
        cute.arch.griddepcontrol_wait()
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        if cutlass.const_expr(self.mode == MULTICAST):
            self.barrier(signals, tid, block)
            alias_fence()
        thread = cutlass.Int64(block * self.threads + tid)
        stride = cutlass.Int64(self.ctas * self.threads)
        # Adjacent lanes own adjacent 16-value blocks. Every four lanes pack
        # one scale word, so payload stores remain fully coalesced.
        groups_per_row = self.input_hidden // 16
        count = cutlass.Int64(m) * groups_per_row
        item = thread
        while item < count:
            row = item // groups_per_row
            col = (item % groups_per_row) * 16
            output_col = col
            if cutlass.const_expr(self.mode == MULTICAST):
                output_col = col + self.rank * self.input_hidden
            element = row * self.input_hidden + col
            p0 = cute.make_ptr(
                cutlass.BFloat16,
                (source.iterator + element).llvm_ptr,
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            p1 = p0 + 8
            lo = load_global_u32x4(p0, volatile=self.mode == MAILBOX)
            hi = load_global_u32x4(p1, volatile=self.mode == MAILBOX)
            if cutlass.const_expr(self.mode == MAILBOX):
                while fragment_is_dirty(lo, SENTINEL):
                    lo = load_global_u32x4(p0, volatile=True)
                while fragment_is_dirty(hi, SENTINEL):
                    hi = load_global_u32x4(p1, volatile=True)
            q0, q1, sf = quantize_group16(lo, hi, cutlass.Float32(global_scale[0]))
            word = row * (self.hidden // 8) + output_col // 8
            if cutlass.const_expr(self.mode == MULTICAST):
                address = data_mc + word * 4
            else:
                address = cutlass.Int64((data.iterator + word).toint())
            store_packed_pair(address, q0, q1, multicast=self.mode == MULTICAST)
            scale_word = pack_scale_quad(sf, cutlass.Int32(tid))
            scale_index = row * (self.hidden // 64) + output_col // 64
            if tid % 4 == 0:
                if cutlass.const_expr(self.mode == MULTICAST):
                    multicast_word(scale_mc + scale_index * 4, scale_word)
                else:
                    scales[scale_index] = scale_word
            item = item + stride

        if cutlass.const_expr(self.mode == MULTICAST):
            alias_fence()
            cute.arch.sync_threads()
            self.barrier(signals, tid, block)
            alias_fence()
            # Do not release MoE early: all peer data/scales must be visible.
        else:
            cute.arch.griddepcontrol_launch_dependents()

        if cutlass.const_expr(self.mode == MAILBOX):
            item = thread
            while item < count:
                base = item * 16
                for fragment in cutlass.range_constexpr(2):
                    ptr = cute.make_ptr(
                        cutlass.BFloat16,
                        (source.iterator + base + fragment * 8).llvm_ptr,
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    )
                    store_lamport_sentinel_128(ptr, sentinel=SENTINEL)
                item = item + stride


@functools.cache
def compile_kernel(
    hidden: int,
    mode: int,
    rank: int,
    world: int,
    ctas: int,
    threads: int,
    device: int,
    use_pdl: bool,
):
    with torch.cuda.device(device):
        dynamic = cute.sym_int32(divisibility=8)
        source = make_fake_compact_tensor(
            cutlass.BFloat16, (dynamic,), assumed_align=16
        )
        data = make_fake_compact_tensor(
            cutlass.Uint32, (cute.sym_int32(divisibility=4),), assumed_align=16
        )
        scales = make_fake_compact_tensor(
            cutlass.Uint32, (cute.sym_int32(),), assumed_align=4
        )
        scale = make_fake_compact_tensor(cutlass.Float32, (1,), assumed_align=4)
        signals = make_fake_compact_tensor(cutlass.Int64, (world,), assumed_align=8)
        kernel = Nvfp4InputKernel(hidden, mode, rank, world, ctas, threads)
        kernel.use_pdl = use_pdl
        return cute.compile(
            kernel,
            source,
            data,
            scales,
            scale,
            signals,
            cutlass.Int64(0),
            cutlass.Int64(0),
            cutlass.Int32(1),
            make_fake_stream(),
        )


def launch(
    source: torch.Tensor,
    data: torch.Tensor,
    scales: torch.Tensor,
    global_scale: torch.Tensor,
    signals: torch.Tensor,
    *,
    hidden: int,
    m: int,
    mode: int,
    rank: int = 0,
    world: int = 1,
    data_mc: int = 0,
    scale_mc: int = 0,
    ctas: int = 152,
    threads: int = 128,
) -> None:
    """Quantize into caller-owned packed data and linear scales (no allocations)."""
    if mode not in (PLAIN, MAILBOX, MULTICAST) or hidden % (64 * world):
        raise ValueError("unsupported quantization mode or non-64-aligned shard")
    if (
        not source.is_cuda
        or source.dtype != torch.bfloat16
        or not source.is_contiguous()
    ):
        raise ValueError("input must be contiguous CUDA BF16")
    expected = m * (hidden // world if mode == MULTICAST else hidden)
    if source.numel() < expected or m <= 0:
        raise ValueError("input does not cover the live rows")
    if (
        data.dtype != torch.uint8
        or data.shape != (m, hidden // 2)
        or not data.is_contiguous()
    ):
        raise ValueError("packed output must be contiguous uint8 [M,H/2]")
    if (
        scales.dtype != torch.uint8
        or scales.numel() != m * hidden // 16
        or not scales.is_contiguous()
    ):
        raise ValueError("scales must be contiguous uint8 [M,H/16]")
    stream = cuda.CUstream(torch.cuda.current_stream(source.device).cuda_stream)
    runner = compile_kernel(
        hidden, mode, rank, world, ctas, threads, source.device.index, pdl_enabled()
    )
    runner(
        to_cute_dynamic_m(source.flatten(), mode=0),
        to_cute_dynamic_m(data.view(torch.uint32).flatten(), mode=0),
        to_cute_dynamic_m(scales.view(torch.uint32).flatten(), mode=0, assumed_align=4),
        to_cute(global_scale.reshape(1), assumed_align=4),
        to_cute(signals, assumed_align=8),
        cutlass.Int64(data_mc),
        cutlass.Int64(scale_mc),
        cutlass.Int32(m),
        stream,
    )
