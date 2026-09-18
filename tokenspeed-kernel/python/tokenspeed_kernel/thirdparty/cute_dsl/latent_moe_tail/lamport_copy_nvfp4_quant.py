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


"""Lamport copy fused with NVFP4 quantization using independent ready groups."""

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


def _quantize_asm():
    lines = [
        "{",
        ".reg .f32 v<8>, a, t, s, r, ginv, sixinv;",
        ".reg .b32 u, halves, peer, bits, mask, base;",
        ".reg .b16 sf, half0, half1;",
        ".reg .b8 b<4>;",
        ".reg .pred zero;",
        "mov.f32 a, 0f00000000;",
        "mov.b32 {b0, b1, b2, b3}, 0;",
    ]
    for i in range(8):
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
        "and.b32 base, $7, 30;",
        "mov.b32 mask, 3;",
        "shl.b32 mask, mask, base;",
        "mov.b32 bits, a;",
        "shfl.sync.bfly.b32 peer, bits, 1, 31, mask;",
        "mov.b32 t, peer;",
        "max.f32 a, a, t;",
        "rcp.approx.ftz.f32 sixinv, 0f40c00000;",
        "mul.rn.f32 s, a, sixinv;",
        "mul.rn.f32 s, s, $6;",
        "cvt.rn.satfinite.e4m3x2.f32 sf, 0f00000000, s;",
        "cvt.u32.u16 $1, sf;",
        "cvt.rn.f16x2.e4m3x2 halves, sf;",
        "mov.b32 {half0, half1}, halves;",
        "cvt.f32.f16 s, half0;",
        "rcp.approx.ftz.f32 ginv, $6;",
        "mul.rn.f32 r, s, ginv;",
        "rcp.approx.ftz.f32 r, r;",
        "setp.eq.f32 zero, a, 0f00000000;",
        "selp.f32 r, 0f00000000, r, zero;",
    ]
    for i in range(8):
        lines.append(f"mul.rn.f32 v{i}, v{i}, r;")
    for i in range(4):
        lines.append(f"cvt.rn.satfinite.e2m1x2.f32 b{i}, v{2*i+1}, v{2*i};")
    lines += ["mov.b32 $0, {b0, b1, b2, b3};", "}"]
    return "\n".join(lines)


@dsl_user_op
def quantize_fragment(fragment, scale, lane, *, loc, ip):
    args = [value.ir_value(loc=loc, ip=ip) for value in fragment]
    args += [scale.ir_value(loc=loc, ip=ip), lane.ir_value(loc=loc, ip=ip)]
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        args,
        _quantize_asm(),
        "=r,=r,r,r,r,r,f,r",
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
def pack_scales(scale, lane, *, loc, ip):
    result = llvm.inline_asm(
        T.i32(),
        [scale.ir_value(loc=loc, ip=ip), lane.ir_value(loc=loc, ip=ip)],
        "{ .reg .b32 v, t, shift, mask, base;\n"
        "shr.u32 shift, $2, 1;\n"
        "and.b32 shift, shift, 3;\n"
        "shl.b32 shift, shift, 3;\n"
        "shl.b32 v, $1, shift;\n"
        "and.b32 base, $2, 24;\n"
        "mov.b32 mask, 255;\n"
        "shl.b32 mask, mask, base;\n"
        "shfl.sync.bfly.b32 t, v, 2, 31, mask;\n"
        "or.b32 v, v, t;\n"
        "shfl.sync.bfly.b32 t, v, 4, 31, mask;\n"
        "or.b32 $0, v, t; }",
        "=r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return cutlass.Uint32(result)


@dsl_user_op
def _ballot(value, *, loc, ip):
    result = llvm.inline_asm(
        T.i32(),
        [value.ir_value(loc=loc, ip=ip)],
        "{ .reg .pred p; setp.ne.u32 p, $1, 0; "
        "vote.ballot.sync.b32 $0, p, 0xffffffff; }",
        "=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return cutlass.Uint32(result)


class LamportCopyNvfp4Quant:
    def __init__(self, hidden, use_pdl):
        self.hidden = hidden
        self.use_pdl = use_pdl

    @cute.jit
    def __call__(self, source, data, scales, scale, m, ctas, stream):
        self.kernel(source, data, scales, scale, m).launch(
            grid=(ctas, 1, 1),
            block=(128, 1, 1),
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(self, source, data, scales, scale, m: cutlass.Int32):
        cute.arch.griddepcontrol_wait()
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        item = cutlass.Int64(block * 128 + tid)
        pending = item < cutlass.Int64(m) * (self.hidden // 8)
        element = item * 8
        pointer = cute.make_ptr(
            cutlass.BFloat16,
            (source.iterator + element).llvm_ptr,
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        fragment = cute.make_rmem_tensor(cute.make_layout((4,)), cutlass.Uint32)
        fragment.fill(0)
        group_mask = cutlass.Uint32(255) << cutlass.Uint32((tid % 32) & 24)
        # All lanes, including padding and already-completed lanes, must vote.
        # Eight lanes cover four whole quantization blocks and one scale word.
        # Unlike a blocking per-lane load, one probe lets ready groups advance
        # while another group in the same warp still waits for a peer.
        while _ballot(cutlass.Int32(pending), loc=None, ip=None) != cutlass.Uint32(0):
            ready = pending == False
            if pending:
                fragment.store(
                    load_global_u32x4(pointer, volatile=True, loc=None, ip=None)
                )
                ready = fragment_is_dirty(fragment.load(), 0x80008000) == False
            ready_mask = _ballot(cutlass.Int32(ready), loc=None, ip=None)
            if pending & ((ready_mask & group_mask) == group_mask):
                packed, sf = quantize_fragment(
                    (fragment[0], fragment[1], fragment[2], fragment[3]),
                    cutlass.Float32(scale[0]),
                    cutlass.Int32(tid % 32),
                    loc=None,
                    ip=None,
                )
                output = cute.make_ptr(
                    cutlass.Uint32,
                    (data.iterator + element // 2).llvm_ptr,
                    cute.AddressSpace.gmem,
                    assumed_align=4,
                )
                cute.make_tensor(output, cute.make_layout((1,)))[0] = packed
                packed_sf = pack_scales(sf, cutlass.Int32(tid % 32), loc=None, ip=None)
                if tid % 8 == 0:
                    scales[element // 64] = packed_sf
                store_lamport_sentinel_128(pointer, sentinel=0x80008000)
                # Never reread a consumed fragment: its sentinel is next
                # generation's state, not a reason to wait again this round.
                pending = False
        cute.arch.griddepcontrol_launch_dependents()


@functools.cache
def compile_kernel(hidden, device, use_pdl):
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
            LamportCopyNvfp4Quant(hidden, use_pdl),
            source,
            data,
            scales,
            scale,
            cutlass.Int32(1),
            cutlass.Int32(256),
            make_fake_stream(),
        )


def launch(source, data, scales, global_scale, *, hidden, m, use_pdl):
    """Consume a BF16 mailbox and encode ready 64-value groups in one kernel.

    Args:
        source: Contiguous CUDA BF16 mailbox covering at least M*hidden values.
            Its producer must follow the existing 0x80008000 sentinel contract.
        data: Contiguous CUDA uint8 [M, hidden/2] packed FP4 output.
        scales: Contiguous CUDA uint8 [M, hidden/16] linear E4M3 scale bytes.
        global_scale: CUDA FP32 scalar encoding multiplier of this receiver.
        hidden: Positive latent width divisible by 64, keeping groups complete.
        m: Live rows, in 1..1280. Only these mailbox rows are consumed/reset.
        use_pdl: Capture-time PDL policy shared with the original producer.

    Returns:
        None. Writes payload/scales and rearms consumed mailbox fragments.
        Buffers and the two-slot symmetric-mailbox rotation remain caller-owned.
    """
    if hidden <= 0 or hidden % 64 or not 1 <= m <= 1280:
        raise ValueError("unsupported Lamport NVFP4 geometry")
    if (
        not source.is_cuda
        or source.dtype != torch.bfloat16
        or not source.is_contiguous()
        or source.numel() < m * hidden
    ):
        raise ValueError("source must cover live contiguous BF16 mailbox rows")
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
    # M changes launch size, not the compiled kernel. Prepare once before graph
    # capture, with one 128-bit fragment per lane and at least 256 CTAs.
    ctas = max(256, (m * (hidden // 8) + 127) // 128)
    runner = compile_kernel(hidden, source.device.index, use_pdl)
    runner(
        to_cute_dynamic_m(source.flatten(), mode=0, assumed_align=16),
        to_cute_dynamic_m(data.flatten(), mode=0, assumed_align=16),
        to_cute_dynamic_m(scales.view(torch.uint32).flatten(), mode=0, assumed_align=4),
        to_cute(global_scale.reshape(1), assumed_align=4),
        cutlass.Int32(m),
        cutlass.Int32(ctas),
        cuda.CUstream(torch.cuda.current_stream(source.device).cuda_stream),
    )


class LamportCopyNvfp4QuantKernel:
    """Copy a borrowed BF16 mailbox into fresh NVFP4 values and scales."""

    def __init__(self, *, hidden_dim: int, device: torch.device) -> None:
        """Compile for the mailbox's row width and device before graph capture."""
        self.hidden_dim = hidden_dim
        self.use_pdl = pdl_enabled()
        compile_kernel(hidden_dim, device.index, self.use_pdl)

    def __call__(
        self,
        symmetric_mailbox: torch.Tensor,
        scale: torch.Tensor,
        *,
        m: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize live mailbox rows and restore their empty sentinels.

        Args:
            symmetric_mailbox: Contiguous CUDA BF16 storage covering M rows of
                hidden_dim values, using the 0x80008000 sentinel contract.
            scale: This receiver's positive scalar FP32 encoding multiplier.
            m: Live rows, between 1 and 1280; only these rows are consumed.

        Returns:
            Newly allocated uint8 packed values [M, H/2] and linear E4M3
            scales [M, H/16], with scale storage padded to 16 rows.
        """
        data = torch.empty(
            (m, self.hidden_dim // 2),
            dtype=torch.uint8,
            device=symmetric_mailbox.device,
        )
        scales = torch.empty(
            ((m + 15) // 16 * 16, self.hidden_dim // 16),
            dtype=torch.uint8,
            device=symmetric_mailbox.device,
        )[:m]
        launch(
            symmetric_mailbox,
            data,
            scales,
            scale,
            hidden=self.hidden_dim,
            m=m,
            use_pdl=self.use_pdl,
        )
        return data, scales.view(torch.float8_e4m3fn)
