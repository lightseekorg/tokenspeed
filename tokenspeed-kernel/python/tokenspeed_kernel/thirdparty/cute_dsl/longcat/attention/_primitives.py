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
# LIABILITY, WHETHER IN AN ACTION, CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""LongCat attention PTX helpers and shared MMA/FP8 conversion entry points."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, vector
from cutlass.cutlass_dsl import T, dsl_user_op
from tokenspeed_kernel.thirdparty.cute_dsl._mma import (
    _fp8x4_to_fp16x4,
    _mma_sync,
)


@dsl_user_op
def mma_sync(a, b, c: cute.Tensor, *, loc=None, ip=None):
    return _mma_sync(a, b, c, loc=loc, ip=ip, vector_kwargs={})


@dsl_user_op
def mma_sync_bf16_ssa(
    a0: Uint32,
    a1: Uint32,
    a2: Uint32,
    a3: Uint32,
    b0: Uint32,
    b1: Uint32,
    c0: cutlass.Float32,
    c1: cutlass.Float32,
    c2: cutlass.Float32,
    c3: cutlass.Float32,
    *,
    loc=None,
    ip=None,
):
    """``mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`` on scalar SSA
    fragments (packed bf16x2 words for A / B, fp32 for C).

    Unlike :func:`mma_sync` nothing goes through an rmem tensor, so the
    fragments and accumulators of a hand-unrolled loop stay in registers by
    construction.  Returns the four fp32 D values.
    """
    f32 = T.f32()
    out = llvm.inline_asm(
        llvm.StructType.get_literal([f32] * 4),
        [v.ir_value(loc=loc, ip=ip) for v in (a0, a1, a2, a3, b0, b1, c0, c1, c2, c3)],
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};",
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return tuple(
        cutlass.Float32(llvm.extractvalue(f32, out, [i], loc=loc, ip=ip))
        for i in range(4)
    )


@dsl_user_op
def mma_sync_f16_ssa(
    a0: Uint32,
    a1: Uint32,
    a2: Uint32,
    a3: Uint32,
    b0: Uint32,
    b1: Uint32,
    c0: cutlass.Float32,
    c1: cutlass.Float32,
    c2: cutlass.Float32,
    c3: cutlass.Float32,
    *,
    loc=None,
    ip=None,
):
    """``mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`` on scalar SSA
    fragments (packed f16x2 words for A / B, fp32 for C); the f16 twin of
    :func:`mma_sync_bf16_ssa`.  Returns the four fp32 D values.
    """
    f32 = T.f32()
    out = llvm.inline_asm(
        llvm.StructType.get_literal([f32] * 4),
        [v.ir_value(loc=loc, ip=ip) for v in (a0, a1, a2, a3, b0, b1, c0, c1, c2, c3)],
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};",
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return tuple(
        cutlass.Float32(llvm.extractvalue(f32, out, [i], loc=loc, ip=ip))
        for i in range(4)
    )


@dsl_user_op
def bf16x2_to_f16x2(x: Uint32, *, loc=None, ip=None) -> Uint32:
    """Re-encode a packed bf16x2 word as f16x2 (round to nearest), element
    order kept: the low half stays the low half."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [x.ir_value(loc=loc, ip=ip)],
            "{\n\t"
            ".reg .b16 lo, hi;\n\t"
            ".reg .f32 flo, fhi;\n\t"
            "mov.b32 {lo, hi}, $1;\n\t"
            "cvt.f32.bf16 flo, lo;\n\t"
            "cvt.f32.bf16 fhi, hi;\n\t"
            "cvt.rn.f16x2.f32 $0, fhi, flo;\n\t"
            "}\n",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def fp8x4_to_fp16x4(x: Uint32, *, loc=None, ip=None) -> cute.TensorSSA:
    return _fp8x4_to_fp16x4(x, loc=loc, ip=ip, vector_kwargs={})


def _make_l2_policy(priority: str, *, loc=None, ip=None) -> Int64:
    value = llvm.inline_asm(
        T.i64(),
        [],
        f"createpolicy.fractional.L2::{priority}.b64 $0, 1.0;",
        "=l",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Int64(value)


@dsl_user_op
def make_l2_policy_evict_first(*, loc=None, ip=None) -> Int64:
    """L2 cache policy for data read exactly once (a weight stream).

    Pass the result as ``policy`` to :func:`cpasync_bulk_g2s` /
    :func:`prefetch_bulk_l2` or as ``cache_policy`` to a TMA ``cute.copy``.
    Lines fetched under it are the first candidates for eviction, so a
    stream does not push out what a later phase or kernel has staged.
    Create it once per kernel and pass it down; it is a plain 64-bit value.
    """
    return _make_l2_policy("evict_first", loc=loc, ip=ip)


@dsl_user_op
def make_l2_policy_evict_last(*, loc=None, ip=None) -> Int64:
    """L2 cache policy for data staged ahead of its consumer.

    Lines fetched under it stay resident in preference to normal and
    evict_first lines -- the policy for a kernel prefetching its
    successor's weights.  Same usage as :func:`make_l2_policy_evict_first`.
    """
    return _make_l2_policy("evict_last", loc=loc, ip=ip)


@dsl_user_op
def make_l2_policy_evict_normal(*, loc=None, ip=None) -> Int64:
    """L2 cache policy for data staged ahead of its consumer within a step.

    Normal priority: above the evict_first streams running past it, but
    not pinned the way evict_last lines are -- a step's staged weights
    leave room for everything else the next phases fetch.  Same usage as
    :func:`make_l2_policy_evict_first`.
    """
    return _make_l2_policy("evict_normal", loc=loc, ip=ip)


@dsl_user_op
def cpasync_bulk_g2s(
    gmem_ptr: cute.Pointer,
    smem_ptr: cute.Pointer,
    mbar_ptr: cute.Pointer,
    size: int | Int32,
    policy: Int64 | None = None,
    *,
    loc=None,
    ip=None,
):
    """``cp.async.bulk`` 1D global->shared copy with mbarrier byte tracking.

    ``size`` must be a multiple of 16 and both pointers 16-byte aligned.  The
    copy posts its byte count to ``mbar_ptr`` via
    ``mbarrier::complete_tx::bytes``, so the consumer waits on an mbarrier
    armed with ``mbarrier_arrive_and_expect_tx(total_bytes)``.  ``policy``
    (from :func:`make_l2_policy_evict_first` / ``_last``) attaches an L2
    cache hint; without it the copy is unchanged.
    """
    gmem_ptr_i64 = gmem_ptr.toint(loc=loc, ip=ip).ir_value()
    smem_ptr_i32 = smem_ptr.toint(loc=loc, ip=ip).ir_value()
    mbar_ptr_i32 = mbar_ptr.toint(loc=loc, ip=ip).ir_value()
    operands = [gmem_ptr_i64, smem_ptr_i32, mbar_ptr_i32, Int32(size).ir_value()]
    asm = "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
    if policy is None:
        asm += " [$1], [$0], $3, [$2];"
        constraints = "l,r,r,r"
    else:
        operands.append(policy.ir_value())
        asm += ".L2::cache_hint [$1], [$0], $3, [$2], $4;"
        constraints = "l,r,r,r,l"
    llvm.inline_asm(
        None,
        operands,
        asm,
        constraints,
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def store_relaxed_sys_u32(
    gmem_ptr: cute.Pointer,
    value: Uint32,
    *,
    loc=None,
    ip=None,
):
    """``st.relaxed.sys.global.u32``: a plain system-scope store, for flags
    whose ordering is provided by a preceding ``fence.acq_rel.sys``."""
    llvm.inline_asm(
        None,
        [gmem_ptr.toint(loc=loc, ip=ip).ir_value(), Uint32(value).ir_value()],
        "st.relaxed.sys.global.u32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def fence_acq_rel_sys(*, loc=None, ip=None):
    """``fence.acq_rel.sys``: orders this thread's prior stores (including
    P2P stores into a peer GPU's memory) before its later system-scope
    release / gpu-scope publication."""
    llvm.inline_asm(
        None,
        [],
        "fence.acq_rel.sys;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def red_add_relaxed_gpu_s32(
    gmem_ptr: cute.Pointer,
    value: Int32,
    *,
    loc=None,
    ip=None,
):
    """``red.relaxed.gpu.global.add.s32`` -- a non-returning atomic add.

    Cheaper than ``atomicAdd`` when the old value is not needed (no
    read-back), which is the histogram-building case.  Relaxed: the caller
    orders the adds against consumers with its own release/acquire (the
    engine covers them with the per-CTA release-add on the logits ticket).
    """
    gmem_ptr_i64 = gmem_ptr.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [gmem_ptr_i64, Int32(value).ir_value()],
        "red.relaxed.gpu.global.add.s32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def cp_async_16_g2s(
    gmem_ptr: cute.Pointer,
    smem_ptr: cute.Pointer,
    *,
    loc=None,
    ip=None,
):
    """``cp.async.cg`` 16-byte global->shared copy (one per thread).

    Both pointers must be 16-byte aligned.  Unlike ``cpasync_bulk_g2s`` this
    is issued per thread through the generic path rather than the TMA unit,
    so a whole warp group can share the issue cost of one tile; completion is
    posted with :func:`cp_async_mbarrier_arrive_noinc`.
    """
    gmem_ptr_i64 = gmem_ptr.toint(loc=loc, ip=ip).ir_value()
    smem_ptr_i32 = smem_ptr.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [smem_ptr_i32, gmem_ptr_i64],
        "cp.async.cg.shared.global [$0], [$1], 16;",
        "r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def cp_async_mbarrier_arrive_noinc(
    mbar_ptr: cute.Pointer,
    *,
    loc=None,
    ip=None,
):
    """Arrive on ``mbar_ptr`` when this thread's prior ``cp.async`` complete.

    ``noinc`` does not bump the mbarrier's pending count, so the barrier must
    be initialized with an arrival count equal to the number of threads that
    execute this instruction.
    """
    mbar_ptr_i32 = mbar_ptr.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [mbar_ptr_i32],
        "cp.async.mbarrier.arrive.noinc.shared::cta.b64 [$0];",
        "r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def mapa_shared_cluster_u32(
    smem_ptr: cute.Pointer,
    peer_cta_rank_in_cluster: Int32,
    *,
    loc=None,
    ip=None,
) -> Int32:
    """Map a shared-memory pointer to the peer CTA's shared window."""
    smem_ptr_i32 = smem_ptr.toint(loc=loc, ip=ip).ir_value()
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [smem_ptr_i32, peer_cta_rank_in_cluster.ir_value(loc=loc, ip=ip)],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def ld_shared_cluster_u32x4(
    remote_smem_addr: Int32,
    *,
    loc=None,
    ip=None,
):
    """Load 16B from a peer CTA's shared memory (DSMEM pull read)."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [remote_smem_addr.ir_value(loc=loc, ip=ip)],
        "ld.shared::cluster.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    packed = vector.from_elements(
        ir.VectorType.get([4], T.i32(), loc=loc),
        [llvm.extractvalue(T.i32(), result, [i], loc=loc, ip=ip) for i in range(4)],
        loc=loc,
    )
    return cute.TensorSSA(packed, 4, Uint32)


@dsl_user_op
def st_shared_cluster_s32(
    remote_smem_addr: Int32,
    value: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Store one Int32 into a peer CTA's shared memory (DSMEM push write).

    ``remote_smem_addr`` comes from :func:`mapa_shared_cluster_u32`; the
    write becomes visible to the peer after a cluster barrier (release
    arrive / acquire wait) that both CTAs join.
    """
    llvm.inline_asm(
        None,
        [
            remote_smem_addr.ir_value(loc=loc, ip=ip),
            Int32(value).ir_value(loc=loc, ip=ip),
        ],
        "st.shared::cluster.s32 [$0], $1;",
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ld_shared_cluster_f32(
    remote_smem_addr: Int32,
    *,
    loc=None,
    ip=None,
) -> cutlass.Float32:
    """Load one Float32 from a peer CTA's shared memory (DSMEM pull read)."""
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [remote_smem_addr.ir_value(loc=loc, ip=ip)],
            "ld.shared::cluster.f32 $0, [$1];",
            "=f,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def ld_shared_cluster_f32x4(
    remote_smem_addr: Int32,
    *,
    loc=None,
    ip=None,
):
    """Load four Float32 from a peer CTA's shared memory (DSMEM pull read)."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [remote_smem_addr.ir_value(loc=loc, ip=ip)],
        "ld.shared::cluster.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    packed = vector.from_elements(
        ir.VectorType.get([4], T.f32(), loc=loc),
        [llvm.extractvalue(T.f32(), result, [i], loc=loc, ip=ip) for i in range(4)],
        loc=loc,
    )
    return cute.TensorSSA(packed, 4, cutlass.Float32)


@dsl_user_op
def exp2_f32(x, *, loc=None, ip=None):
    """Fast exp2 on a Float32 scalar."""
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [cutlass.Float32(x).ir_value(loc=loc, ip=ip)],
            "ex2.approx.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


_FP8_MAX = 448.0


@dsl_user_op
def _cvt_f32_to_e4m3_bits(
    value: cutlass.Float32,
    *,
    loc=None,
    ip=None,
) -> cutlass.Uint8:
    """Match Triton's native saturating FP32-to-E4M3 conversion."""
    packed = cutlass.Uint32(
        llvm.inline_asm(
            T.i32(),
            [cutlass.Float32(value).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .b16 fp8_pair;
                .reg .f32 zero;
                mov.f32 zero, 0f00000000;
                cvt.rn.satfinite.e4m3x2.f32 fp8_pair, zero, $1;
                cvt.u32.u16 $0, fp8_pair;
            }
            """,
            "=r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )
    return cutlass.Uint8(packed & cutlass.Uint32(0xFF))


@cute.jit
def _quantize_bits(value: cutlass.Float32, scale: cutlass.Float32):
    quantized = value / scale
    if quantized > cutlass.Float32(_FP8_MAX):
        quantized = cutlass.Float32(_FP8_MAX)
    if quantized < cutlass.Float32(-_FP8_MAX):
        quantized = cutlass.Float32(-_FP8_MAX)
    return _cvt_f32_to_e4m3_bits(quantized)


__all__ = [
    "bf16x2_to_f16x2",
    "cp_async_16_g2s",
    "fence_acq_rel_sys",
    "store_relaxed_sys_u32",
    "red_add_relaxed_gpu_s32",
    "st_shared_cluster_s32",
    "cp_async_mbarrier_arrive_noinc",
    "cpasync_bulk_g2s",
    "exp2_f32",
    "fp8x4_to_fp16x4",
    "ld_shared_cluster_f32",
    "ld_shared_cluster_f32x4",
    "ld_shared_cluster_u32x4",
    "mapa_shared_cluster_u32",
    "mma_sync",
    "mma_sync_bf16_ssa",
    "mma_sync_f16_ssa",
]


@cute.jit
def emit_topk_index(output_indices_row, pos, index, slot_table, slot_page_shift):
    """Write one selected candidate offset into ``output_indices_row[pos]``.

    ``slot_table is None`` (what every caller that does not opt in passes)
    stores the logical offset ``index`` unchanged -- the historical
    behaviour.  Otherwise the offset is mapped through a page table,
    ``slot_table[index >> shift] * (1 << shift) + index % (1 << shift)``,
    which lets a caller whose candidates are paged (page ``p`` of the logical
    row lives at physical page ``slot_table[p]``) get physical positions out
    of the collect pass instead of running a second mapping pass over the
    output.  ``slot_table`` may live in shared or global memory.

    Args:
        output_indices_row: destination row, one Int32 per selected slot.
        pos: destination position inside that row.
        index: the logical candidate offset.
        slot_table: ``None`` or an Int32 tensor indexed by logical page.
        slot_page_shift: ``log2(page_size)``, compile-time.
    """
    value = cutlass.Int32(index)
    if cutlass.const_expr(slot_table is None):
        output_indices_row[pos] = value
    else:
        page_size = cutlass.const_expr(1 << slot_page_shift)
        page = value // page_size
        output_indices_row[pos] = slot_table[page] * cutlass.Int32(page_size) + (
            value - page * page_size
        )
