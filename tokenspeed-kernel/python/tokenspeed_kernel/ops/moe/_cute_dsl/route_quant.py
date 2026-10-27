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

"""Single-launch expert routing and input block-FP8 quantization."""

import threading

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float8E4M3FN, Float32, Int32, Int64, utils
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op


@dsl_user_op
def divide_approx_ftz(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """Divide with PTX div.approx.ftz.f32, preserving flush-to-zero behavior."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "div.approx.ftz.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def reciprocal_approx_ftz(a: Float32, *, loc=None, ip=None) -> Float32:
    """Compute the approximate FTZ reciprocal stored as the FP32 inverse scale."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "rcp.approx.ftz.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


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


class RoutePrepare:
    """Build compact routes and quantize BF16 input in one kernel launch.

    Each CTA owns a token interval and derives disjoint per-expert row ranges
    from the route histogram. No CTA consumes another CTA's intermediate state.
    A warp quantizes each token once and copies its bytes to its local routes.
    """

    def __init__(self, tokens, experts, topk, scale_rows, expert_offset, hidden):
        self.t = tokens
        self.e = experts
        self.padded_e = ((experts + 31) // 32) * 32
        self.k = topk
        self.r = tokens * topk
        self.origin = expert_offset
        self.ld = scale_rows
        self.hidden = hidden
        self.quant_blocks = hidden // 128

    @cute.jit
    def __call__(
        self,
        inputs: cute.Tensor,
        ids: cute.Tensor,
        counts: cute.Tensor,
        cursors: cute.Tensor,
        offsets: cute.Tensor,
        packed_ids: cute.Tensor,
        scale_row: cute.Tensor,
        inverse: cute.Tensor,
        packed: cute.Tensor,
        scales: cute.Tensor,
        stream: cuda.CUstream,
    ):
        if cutlass.const_expr(self.t == 1):
            self.decode(
                inputs,
                ids,
                counts,
                cursors,
                offsets,
                packed_ids,
                scale_row,
                inverse,
                packed,
                scales,
            ).launch(grid=(1, 1, 1), block=(128, 1, 1), stream=stream)
        else:
            self.by_token_tile(
                inputs,
                ids,
                counts,
                cursors,
                offsets,
                packed_ids,
                scale_row,
                inverse,
                packed,
                scales,
            ).launch(grid=(min(self.t, 128), 1, 1), block=(256, 1, 1), stream=stream)

    @cute.jit
    def quantize_word(self, inputs: cute.Tensor, token: Int32, kb: Int32):
        lane = cute.arch.lane_idx()
        # Four adjacent BF16 columns per lane permit aligned 8-byte loads and
        # packed 4-byte FP8 stores within each 128-column quantization group.
        values = load_bf16x4(
            inputs.iterator + token * self.hidden + kb * 128 + lane * 4
        )
        amax = Float32(0)
        for i in cutlass.range_constexpr(4):
            absolute = values[i]
            if absolute < Float32(0):
                absolute = -absolute
            if absolute > amax:
                amax = absolute
        for i in cutlass.range_constexpr(5):
            other = cute.arch.shuffle_sync_bfly(amax, offset=1 << i)
            if other > amax:
                amax = other
        multiplier = Float32(1)
        if amax != Float32(0):
            multiplier = divide_approx_ftz(Float32(448), amax)
        quantized = cute.make_rmem_tensor((4,), Float8E4M3FN)
        for i in cutlass.range_constexpr(4):
            quantized[i] = Float8E4M3FN(values[i] * multiplier)
        bits = cute.recast_tensor(quantized, Int32)
        return bits[0], reciprocal_approx_ftz(multiplier)

    @cute.kernel
    def by_token_tile(
        self,
        inputs: cute.Tensor,
        ids: cute.Tensor,
        counts: cute.Tensor,
        cursors: cute.Tensor,
        offsets: cute.Tensor,
        packed_ids: cute.Tensor,
        scale_row: cute.Tensor,
        inverse: cute.Tensor,
        packed: cute.Tensor,
        scales: cute.Tensor,
    ):
        block, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        lane = cute.arch.lane_idx()
        warp = cute.arch.warp_idx()
        tile = self.t // min(self.t, 128)
        first_token = block * tile
        allocator = utils.SmemAllocator()
        # Separate warp histograms bound contention when many routes select
        # the same expert. The second half counts tokens before this CTA.
        histogram = allocator.allocate_tensor(
            Int32, cute.make_layout((16 * self.padded_e,)), byte_alignment=16
        )
        totals = allocator.allocate_tensor(
            Int32, cute.make_layout((self.padded_e,)), byte_alignment=16
        )
        prior = allocator.allocate_tensor(
            Int32, cute.make_layout((self.padded_e,)), byte_alignment=16
        )
        starts = allocator.allocate_tensor(
            Int32, cute.make_layout((self.padded_e,)), byte_alignment=16
        )
        next_row = allocator.allocate_tensor(
            Int32, cute.make_layout((self.padded_e,)), byte_alignment=16
        )
        for base in cutlass.range_constexpr(0, 16 * self.padded_e, 256):
            histogram[base + tid] = Int32(0)
        cute.arch.sync_threads()
        for base in cutlass.range(0, self.r, 256):
            route = base + tid
            if route < self.r:
                expert = ids[route // self.k, route % self.k] - self.origin
                if expert >= 0 and expert < self.e:
                    cute.arch.atomic_add(
                        histogram.iterator + warp * self.padded_e + expert,
                        Int32(1),
                        sem="relaxed",
                        scope="cta",
                    )
                    if route < first_token * self.k:
                        cute.arch.atomic_add(
                            histogram.iterator
                            + 8 * self.padded_e
                            + warp * self.padded_e
                            + expert,
                            Int32(1),
                            sem="relaxed",
                            scope="cta",
                        )
        cute.arch.sync_threads()
        for base in cutlass.range_constexpr(0, self.padded_e, 256):
            expert = base + tid
            if expert < self.padded_e:
                total = Int32(0)
                earlier = Int32(0)
                for w in cutlass.range_constexpr(8):
                    total = total + histogram[w * self.padded_e + expert]
                    earlier = (
                        earlier
                        + histogram[8 * self.padded_e + w * self.padded_e + expert]
                    )
                totals[expert] = total
                prior[expert] = earlier
        cute.arch.sync_threads()
        if warp == 0:
            carry = Int32(0)
            for chunk in cutlass.range_constexpr(self.padded_e // 32):
                expert = chunk * 32 + lane
                count = totals[expert]
                prefix = count
                for level in cutlass.range_constexpr(5):
                    previous = cute.arch.shuffle_sync_up(
                        prefix, offset=1 << level, mask_and_clamp=0
                    )
                    if lane >= (1 << level):
                        prefix = prefix + previous
                start = carry + prefix - count
                starts[expert] = start
                # Earlier token intervals own exactly prior[expert] rows.
                next_row[expert] = start + prior[expert]
                if block == 0 and expert < self.e:
                    counts[expert] = count
                    cursors[expert] = start + count
                    offsets[expert] = start.to(Int64)
                carry = carry + cute.arch.shuffle_sync(prefix, offset=31)
            if block == 0 and lane == 0:
                offsets[self.e] = carry.to(Int64)
        cute.arch.sync_threads()
        packed_words = cute.make_tensor(
            cute.recast_ptr(packed.iterator, dtype=Int32),
            cute.make_layout((self.r, self.hidden // 4), stride=(self.hidden // 4, 1)),
        )
        for local_token in cutlass.range(warp, tile, 8):
            token = first_token + local_token
            words = cute.make_rmem_tensor((self.quant_blocks,), Int32)
            inverse_scales = cute.make_rmem_tensor((self.quant_blocks,), Float32)
            for kb in cutlass.range_constexpr(self.quant_blocks):
                word, inverse_scale = self.quantize_word(inputs, token, Int32(kb))
                words[kb] = word
                inverse_scales[kb] = inverse_scale
            reserved = Int32(-1)
            scale = Int32(0)
            if lane < self.k:
                expert = ids[token, lane] - self.origin
                inverse[token, lane] = Int32(-1)
                if expert >= 0 and expert < self.e:
                    reserved = cute.arch.atomic_add(
                        next_row.iterator + expert,
                        Int32(1),
                        sem="relaxed",
                        scope="cta",
                    )
                    start = starts[expert]
                    scale = ((start + expert * 31) // 32) * 32 + reserved - start
                    packed_ids[reserved] = lane * self.t + token
                    inverse[token, lane] = reserved
                    scale_row[reserved] = scale
            for slot in cutlass.range_constexpr(self.k):
                row = cute.arch.shuffle_sync(reserved, offset=slot)
                sr = cute.arch.shuffle_sync(scale, offset=slot)
                if row >= 0:
                    for kb in cutlass.range_constexpr(self.quant_blocks):
                        packed_words[row, kb * 32 + lane] = words[kb]
                        if lane == 0:
                            scales[kb, sr] = inverse_scales[kb]

    @cute.jit
    def quantize_block(self, inputs: cute.Tensor, token: Int32, kb: Int32):
        # Each warp quantizes a 128-column block with the same reduction tree
        # and approximate division used by the expert GEMM input quantizer.
        lane = cute.arch.lane_idx()
        values = cute.make_rmem_tensor((4,), Float32)
        amax = Float32(0)
        for i in cutlass.range_constexpr(4):
            value = inputs[token, kb * 128 + lane + i * 32].to(Float32)
            values[i] = value
            absolute = value
            if absolute < Float32(0):
                absolute = -absolute
            if absolute > amax:
                amax = absolute
        for i in cutlass.range_constexpr(5):
            other = cute.arch.shuffle_sync_bfly(amax, offset=1 << i)
            if other > amax:
                amax = other
        multiplier = Float32(1)
        if amax != Float32(0):
            multiplier = divide_approx_ftz(Float32(448), amax)
        inverse_scale = reciprocal_approx_ftz(multiplier)
        quantized = cute.make_rmem_tensor((4,), Float8E4M3FN)
        for i in cutlass.range_constexpr(4):
            quantized[i] = Float8E4M3FN(values[i] * multiplier)
        return quantized, inverse_scale

    @cute.kernel
    def decode(
        self,
        inputs: cute.Tensor,
        ids: cute.Tensor,
        counts: cute.Tensor,
        cursors: cute.Tensor,
        offsets: cute.Tensor,
        packed_ids: cute.Tensor,
        scale_row: cute.Tensor,
        inverse: cute.Tensor,
        packed: cute.Tensor,
        scales: cute.Tensor,
    ):
        tid, _, _ = cute.arch.thread_idx()
        lane = cute.arch.lane_idx()
        warp = cute.arch.warp_idx()
        # Every output entry is overwritten on every replay. Prefix counts are
        # integer comparisons of the route IDs, without a global atomic or extra launch.
        for chunk in cutlass.range_constexpr((self.e + 127) // 128):
            expert = chunk * 128 + tid
            if expert < self.e:
                start = Int32(0)
                count = Int32(0)
                for slot in cutlass.range_constexpr(self.k):
                    selected = ids[0, slot] - self.origin
                    if selected >= 0 and selected < expert:
                        start = start + 1
                    if selected == expert:
                        count = count + 1
                counts[expert] = count
                cursors[expert] = start + count
                offsets[expert] = start.to(Int64)
        if tid == 0:
            valid = Int64(0)
            for slot in cutlass.range_constexpr(self.k):
                selected = ids[0, slot] - self.origin
                if selected >= 0 and selected < self.e:
                    valid = valid + 1
            offsets[self.e] = valid
        if tid < self.k:
            selected = ids[0, tid] - self.origin
            inverse[0, tid] = Int32(-1)
            if selected >= 0 and selected < self.e:
                start = Int32(0)
                earlier = Int32(0)
                for slot in cutlass.range_constexpr(self.k):
                    other = ids[0, slot] - self.origin
                    if other >= 0 and other < selected:
                        start = start + 1
                    if other == selected and slot < tid:
                        earlier = earlier + 1
                row = start + earlier
                packed_ids[row] = tid
                scale_row[row] = ((start + selected * 31) // 32) * 32 + earlier
                inverse[0, tid] = row
        # Publish the valid-row count and scale-row map to all four warps.
        cute.arch.sync_threads()
        for base in cutlass.range_constexpr(0, self.quant_blocks, 4):
            kb = base + warp
            if kb < self.quant_blocks:
                quantized, inverse_scale = self.quantize_block(inputs, Int32(0), kb)
                for row in cutlass.range_constexpr(self.k):
                    if row < offsets[self.e]:
                        if lane == 0:
                            scales[kb, scale_row[row]] = inverse_scale
                        for i in cutlass.range_constexpr(4):
                            packed[row, kb * 128 + lane + i * 32] = quantized[i]


_compiled = {}
_compile_lock = threading.Lock()


def run_route_quant(storage):
    from tokenspeed_kernel.thirdparty.cute_dsl.latent_moe_tail.primitives import to_cute

    tokens, topk = storage.ids.shape
    bucket = 1 << (tokens - 1).bit_length()
    if bucket != tokens:
        raise ValueError("The selected route bucket requires a power-of-two row count")
    key = (
        storage.input.device.index,
        bucket,
        storage.experts,
        topk,
        storage.expert_offset,
        storage.input.shape[1],
    )
    tensors = (
        storage.input,
        storage.ids,
        storage.counts,
        storage.cursors,
        storage.offsets,
        storage.source_rows,
        storage.scale_rows,
        storage.inverse,
        storage.packed,
        storage.scales,
    )
    alignments = (16, 4, 4, 4, 8, 4, 4, 4, 16, 4)
    with torch.cuda.device(storage.input.device):
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        with _compile_lock:
            compiled = _compiled.get(key)
            if compiled is None:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "Warm the route token bucket before graph capture"
                    )
                types = (
                    BFloat16,
                    Int32,
                    Int32,
                    Int32,
                    Int64,
                    Int32,
                    Int32,
                    Int32,
                    Float8E4M3FN,
                    Float32,
                )
                fake = cute.runtime.make_fake_tensor
                arguments = [
                    fake(dtype, tuple(t.shape), tuple(t.stride()), assumed_align=align)
                    for t, dtype, align in zip(tensors, types, alignments)
                ]
                compiled = cute.compile(
                    RoutePrepare(
                        bucket,
                        storage.experts,
                        topk,
                        storage.scales.shape[1],
                        storage.expert_offset,
                        storage.input.shape[1],
                    ),
                    *arguments,
                    stream,
                    options="--opt-level=2"
                )
                _compiled[key] = compiled
        compiled(*(to_cute(t, align) for t, align in zip(tensors, alignments)), stream)
