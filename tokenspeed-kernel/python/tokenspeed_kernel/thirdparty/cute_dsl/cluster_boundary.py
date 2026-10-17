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

"""Decoder-layer boundary with each row split across a 16-CTA cluster.

The CTAs exchange per-CTA variance partials through distributed shared memory
and sum them in fixed rank order. The branch may be this rank's Lamport
mailbox, polled per word and re-armed in place; a mailbox holding one partial
per rank is reduced in place too, in BF16 in rank order, the bits the one-shot
all-reduce kernel would have handed the boundary.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Int64, Uint32, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cute.typing import Float32, Int32
from cutlass.cutlass_dsl import T, dsl_user_op
from tokenspeed_kernel.thirdparty.cute_dsl.argmax import (
    elem_pointer,
    store_shared_remote,
)
from tokenspeed_kernel.thirdparty.cute_dsl.latent_moe_tail.primitives import (
    load_volatile_u32,
    store_global_u32,
)

# Sized so an 8192-wide row lands two elements per thread; the loads set the pace.
CLUSTER_CTAS = 16
CTA_THREADS = 256
_WARPS = CTA_THREADS // 32
# Two row-wide reductions per boundary, each with its own mbarrier phase.
_STAGES = 2
# Per CTA: warp partials and cluster slots per stage, then one 8-byte mbarrier each.
_SMEM_BYTES = (_STAGES * (_WARPS + CLUSTER_CTAS) * 4 + 7) // 8 * 8 + _STAGES * 8


@dsl_user_op
def rsqrt_approx(value: Float32, *, loc=None, ip=None) -> Float32:
    """``rsqrt.approx.f32``, the instruction the Triton boundary lowers to."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [value.ir_value(loc=loc, ip=ip)],
            "rsqrt.approx.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def fma_rn(a: Float32, b: Float32, c: Float32, *, loc=None, ip=None) -> Float32:
    """``fma.rn.f32``: ``a * b + c`` rounded once, as the Triton boundary pins."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [
                a.ir_value(loc=loc, ip=ip),
                b.ir_value(loc=loc, ip=ip),
                c.ir_value(loc=loc, ip=ip),
            ],
            "fma.rn.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def bf16_half_to_f32(word: Uint32, high: bool, *, loc=None, ip=None) -> Float32:
    """The low or high BF16 half of a packed word widened to FP32 by bit placement."""
    place = "and.b32 t, $1, 0xffff0000;" if high else "shl.b32 t, $1, 16;"
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [word.ir_value(loc=loc, ip=ip)],
            "{ .reg .b32 t; " + place + " mov.b32 $0, t; }",
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def bf16x2_add_rn(a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
    """``add.rn.bf16x2``: both halves summed and rounded once.

    For finite operands this is the all-reduce's ``float(a) + float(b)``
    rounded to BF16 to the bit; a NaN or Inf-cancel result differs in its
    NaN payload only.
    """
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip)],
            "add.rn.bf16x2 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def without_negative_zero(word: Uint32) -> Uint32:
    """Both BF16 halves with ``-0.0`` made ``+0.0``, as the all-reduce sanitizes its input."""
    low = word & Uint32(0xFFFF)
    high = word >> Uint32(16)
    low = cutlass.Uint32(0) if low == Uint32(0x8000) else low
    high = cutlass.Uint32(0) if high == Uint32(0x8000) else high
    return low | (high << Uint32(16))


@cute.jit
def word_is_dirty(word: Uint32, sentinel: cutlass.Constexpr[int]):
    """Whether either BF16 half of ``word`` still holds the sentinel's half."""
    half = Uint32(sentinel & 0xFFFF)
    return ((word & Uint32(0xFFFF)) == half) | ((word >> Uint32(16)) == half)


@cute.jit
def warp_sum(value: Float32) -> Float32:
    """Butterfly sum over a full warp; every lane ends with the same total."""
    for offset in (16, 8, 4, 2, 1):
        value = value + cute.arch.shuffle_sync_bfly(
            value, offset=offset, mask=-1, mask_and_clamp=31
        )
    return value


@cute.jit
def publish_to_cluster(
    value: Float32,
    slot: cute.Pointer,
    mbar_ptr: cute.Pointer,
    peers: cutlass.Constexpr[int],
    warp: Int32,
    lane: Int32,
) -> None:
    """Store ``value`` into this CTA's ``slot`` in each of ``peers`` CTAs and wait for theirs."""
    if warp == 0:
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(mbar_ptr, peers * 4)
        if lane < peers:
            store_shared_remote(value, slot, mbar_ptr, peer_cta_rank_in_cluster=lane)
    cute.arch.mbarrier_wait(mbar_ptr, phase=0)


class ClusterBoundaryKernel:
    """Post-norm, gated residual and pre-norm of ``[rows, width]``, one cluster per row.

    Grid ``(rows, 16)`` with cluster shape ``(1, 16, 1)``: the CTAs of one
    cluster own consecutive column slices of one row.

    Args:
        width: Row width, a power of two divisible by ``CLUSTER_CTAS * CTA_THREADS``.
        post_has_weight: Whether the post-norm scales its input by a weight.
        has_pre: Whether the boundary emits the next pre-norm rows.
        post_eps: Post-norm epsilon.
        post_weight_offset: Center added to the post-norm checkpoint weight.
        pre_eps: Pre-norm epsilon.
        pre_weight_offset: Center added to the pre-norm checkpoint weight.
        use_pdl: Launch with programmatic dependent launch.
        mailbox_sentinel: The word the branch mailbox is armed with, whose
            16-bit halves agree, or None for an ordinary branch tensor.
        mailbox_ranks: Partials per mailbox row, side by side; 1 reads the
            row as it stands, more sums them in BF16 in rank order.

    The call takes the boundary's tensors, then ``rows`` and the stream; an
    absent weight or pre-norm output takes any tensor of its dtype, never read.
    """

    def __init__(
        self,
        width: int,
        post_has_weight: bool,
        has_pre: bool,
        post_eps: float,
        post_weight_offset: float,
        pre_eps: float,
        pre_weight_offset: float,
        use_pdl: bool,
        mailbox_sentinel: int | None,
        mailbox_ranks: int,
    ) -> None:
        if mailbox_sentinel is not None and (
            not 0 <= mailbox_sentinel <= 0xFFFFFFFF
            or (mailbox_sentinel & 0xFFFF) != (mailbox_sentinel >> 16)
        ):
            raise ValueError("the mailbox sentinel is one word whose halves agree")
        if mailbox_ranks < 1 or (mailbox_ranks > 1 and mailbox_sentinel is None):
            raise ValueError("partials are reduced from a mailbox, one or more per row")
        self.width = width
        self.per_thread = width // (CLUSTER_CTAS * CTA_THREADS)
        self.mailbox_sentinel = mailbox_sentinel
        self.mailbox_ranks = mailbox_ranks
        self.post_has_weight = post_has_weight
        self.has_pre = has_pre
        self.post_eps = post_eps
        self.post_weight_offset = post_weight_offset
        self.pre_eps = pre_eps
        self.pre_weight_offset = pre_weight_offset
        self.use_pdl = use_pdl

    @cute.jit
    def __call__(
        self,
        stream: cute.Tensor,
        branch: cute.Tensor,
        post_weight: cute.Tensor,
        alpha: cute.Tensor,
        beta: cute.Tensor,
        pre_weight: cute.Tensor,
        out_stream: cute.Tensor,
        out_pre_fp32: cute.Tensor,
        out_pre_bf16: cute.Tensor,
        rows: Int32,
        cu_stream: cuda.CUstream,
    ):
        """Launch one cluster per row."""
        # A grid from a symbolic tensor shape is rejected at launch; rows is passed in.
        self.kernel(
            stream,
            branch,
            post_weight,
            alpha,
            beta,
            pre_weight,
            out_stream,
            out_pre_fp32,
            out_pre_bf16,
        ).launch(
            grid=[rows, CLUSTER_CTAS, 1],
            block=[CTA_THREADS, 1, 1],
            cluster=[1, CLUSTER_CTAS, 1],
            smem=_SMEM_BYTES,
            stream=cu_stream,
            use_pdl=self.use_pdl,
        )

    @cute.jit
    def _cluster_sum(
        self,
        value: Float32,
        stage: cutlass.Constexpr[int],
        warp_partials: cute.Tensor,
        cluster_slots: cute.Tensor,
        mbar_ptr: cute.Pointer,
        rank: Int32,
        warp: Int32,
        lane: Int32,
    ) -> Float32:
        """Sum ``value`` over the row: warp, CTA, then cluster, in fixed order."""
        value = warp_sum(value)
        if lane == 0:
            warp_partials[warp, stage] = value
        cute.arch.barrier()
        total = Float32(0.0)
        for w in cutlass.range_constexpr(_WARPS):
            total = total + warp_partials[w, stage]
        publish_to_cluster(
            total,
            elem_pointer(cluster_slots, (rank, stage)),
            mbar_ptr + stage,
            CLUSTER_CTAS,
            warp,
            lane,
        )
        result = Float32(0.0)
        for peer in cutlass.range_constexpr(CLUSTER_CTAS):
            result = result + cluster_slots[peer, stage]
        return result

    @cute.kernel
    def kernel(
        self,
        stream: cute.Tensor,
        branch: cute.Tensor,
        post_weight: cute.Tensor,
        alpha: cute.Tensor,
        beta: cute.Tensor,
        pre_weight: cute.Tensor,
        out_stream: cute.Tensor,
        out_pre_fp32: cute.Tensor,
        out_pre_bf16: cute.Tensor,
    ):
        """One CTA's slice of one row."""
        tidx, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        rank = cute.arch.block_idx_in_cluster()
        warp = cute.arch.warp_idx()
        lane = cute.arch.lane_idx()
        per_thread: cutlass.Constexpr = self.per_thread
        width: cutlass.Constexpr = self.width
        inv_width: cutlass.Constexpr = 1.0 / self.width

        smem = cutlass.utils.SmemAllocator()
        warp_partials = smem.allocate_tensor(
            Float32, cute.make_layout((_WARPS, _STAGES)), byte_alignment=4
        )
        cluster_slots = smem.allocate_tensor(
            Float32, cute.make_layout((CLUSTER_CTAS, _STAGES)), byte_alignment=4
        )
        mbar_ptr = smem.allocate_array(cutlass.Int64, num_elems=_STAGES)
        if tidx < _STAGES:
            cute.arch.mbarrier_init(mbar_ptr + tidx, 1)
        cute.arch.mbarrier_init_fence()
        # Peers may only store into this CTA once its barriers are initialized.
        cute.arch.cluster_arrive_relaxed()

        if const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()
            # Dependents may launch now; their own wait still covers this grid's completion.
            cute.arch.griddepcontrol_launch_dependents()

        col0 = (rank * CTA_THREADS + tidx) * per_thread
        # 2-D indexing of a symbolic-row tensor addresses only row zero in this DSL.
        row_layout = cute.make_layout((width,))
        stream_row = cute.make_tensor(stream.iterator + row * width, row_layout)
        branch_row = cute.make_tensor(branch.iterator + row * width, row_layout)
        out_row = cute.make_tensor(out_stream.iterator + row * width, row_layout)
        pre32_row = cute.make_tensor(out_pre_fp32.iterator + row * width, row_layout)
        pre16_row = cute.make_tensor(out_pre_bf16.iterator + row * width, row_layout)
        stream_v = cute.make_rmem_tensor((per_thread,), Float32)
        values_v = cute.make_rmem_tensor((per_thread,), Float32)
        alpha_v = cute.make_rmem_tensor((per_thread,), Float32)
        beta_v = cute.make_rmem_tensor((per_thread,), Float32)
        post_w = cute.make_rmem_tensor((per_thread,), Float32)
        pre_w = cute.make_rmem_tensor((per_thread,), Float32)
        if const_expr(self.mailbox_sentinel is not None):
            # Spin per word until both halves have arrived, then re-arm it. One
            # partial per rank sits side by side in the row; they sum in BF16
            # in rank order, each input stripped of -0.0 first, exactly as the
            # one-shot all-reduce kernel sums the same partials.
            ranks = self.mailbox_ranks
            word_ptr = branch.iterator + row * (width * ranks) + col0
            for w in cutlass.range_constexpr(per_thread // 2):
                acc = Uint32(0)
                for r in cutlass.range_constexpr(ranks):
                    word_at = word_ptr + (r * width + 2 * w)
                    word = load_volatile_u32(word_at)
                    while word_is_dirty(word, self.mailbox_sentinel):
                        word = load_volatile_u32(word_at)
                    store_global_u32(
                        Int64(word_at.toint()), Uint32(self.mailbox_sentinel)
                    )
                    if const_expr(ranks == 1):
                        acc = word
                    elif const_expr(r == 0):
                        acc = without_negative_zero(word)
                    else:
                        acc = bf16x2_add_rn(acc, without_negative_zero(word))
                values_v[2 * w] = bf16_half_to_f32(acc, False)
                values_v[2 * w + 1] = bf16_half_to_f32(acc, True)
        for i in cutlass.range_constexpr(per_thread):
            stream_v[i] = stream_row[col0 + i]
            if const_expr(self.mailbox_sentinel is None):
                values_v[i] = branch_row[col0 + i].to(Float32)
            alpha_v[i] = alpha[col0 + i]
            beta_v[i] = beta[col0 + i]
            if const_expr(self.post_has_weight):
                post_w[i] = (
                    (post_weight[col0 + i].to(Float32) + self.post_weight_offset)
                    .to(cutlass.BFloat16)
                    .to(Float32)
                )
            if const_expr(self.has_pre):
                pre_w[i] = (
                    (pre_weight[col0 + i].to(Float32) + self.pre_weight_offset)
                    .to(cutlass.BFloat16)
                    .to(Float32)
                )

        cute.arch.cluster_wait()

        if const_expr(self.post_has_weight):
            for i in cutlass.range_constexpr(per_thread):
                values_v[i] = values_v[i] * post_w[i]
        partial = Float32(0.0)
        for i in cutlass.range_constexpr(per_thread):
            partial = partial + values_v[i] * values_v[i]
        variance = (
            self._cluster_sum(
                partial,
                0,
                warp_partials,
                cluster_slots,
                mbar_ptr,
                rank,
                warp,
                lane,
            )
            * inv_width
        )
        scale = rsqrt_approx(variance + self.post_eps)
        for i in cutlass.range_constexpr(per_thread):
            normed = values_v[i] * scale
            # The contraction the Triton boundary pins: fma(beta, branch, alpha*stream).
            stream_v[i] = fma_rn(beta_v[i], normed, alpha_v[i] * stream_v[i])
            out_row[col0 + i] = stream_v[i]

        if const_expr(self.has_pre):
            partial = Float32(0.0)
            for i in cutlass.range_constexpr(per_thread):
                partial = partial + stream_v[i] * stream_v[i]
            variance = (
                self._cluster_sum(
                    partial,
                    1,
                    warp_partials,
                    cluster_slots,
                    mbar_ptr,
                    rank,
                    warp,
                    lane,
                )
                * inv_width
            )
            scale = rsqrt_approx(variance + self.pre_eps)
            for i in cutlass.range_constexpr(per_thread):
                pre = stream_v[i] * scale * pre_w[i]
                pre32_row[col0 + i] = pre
                pre16_row[col0 + i] = pre.to(cutlass.BFloat16)

        # Redundant with the mbarrier wait, but keeps the cluster exit ordered.
        cute.arch.cluster_arrive_relaxed()
        cute.arch.cluster_wait()
