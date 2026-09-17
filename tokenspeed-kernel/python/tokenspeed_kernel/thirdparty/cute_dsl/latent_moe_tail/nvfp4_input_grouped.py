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


"""Fused Lamport receive and NVFP4 quantization with independent ready groups."""

import functools

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.cutlass_dsl import T, dsl_user_op
from tokenspeed_kernel.thirdparty.cute_dsl.latent_moe_tail.nvfp4_input_cooperative import (
    pack_scales,
    quantize_fragment,
)
from tokenspeed_kernel.thirdparty.cute_dsl.latent_moe_tail.primitives import (
    fragment_is_dirty,
    load_global_u32x4,
    store_lamport_sentinel_128,
    to_cute,
    to_cute_dynamic_m,
)


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


class GroupedNvfp4Input:
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
                    values=8,
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
                packed_sf = pack_scales(
                    sf, cutlass.Int32(tid % 32), values=8, loc=None, ip=None
                )
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
            GroupedNvfp4Input(hidden, use_pdl),
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
        raise ValueError("unsupported grouped NVFP4 geometry")
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
