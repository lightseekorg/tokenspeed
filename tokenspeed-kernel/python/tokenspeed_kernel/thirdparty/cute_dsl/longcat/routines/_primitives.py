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

"""Memory-ordering primitives for the static LongCat tile executor."""

from __future__ import annotations

import cutlass.cute as cute
from cutlass import Float32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op


@dsl_user_op
def prefetch_global_l2(
    pointer: cute.Pointer,
    *,
    loc=None,
    ip=None,
) -> Uint32:
    """Issue a best-effort L2 prefetch for one global-memory cache line."""
    address = pointer.toint(loc=loc, ip=ip)
    llvm.inline_asm(
        None,
        [address.ir_value(loc=loc, ip=ip)],
        "prefetch.global.L2 [$0];",
        "l,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    # A typed sentinel keeps this user op composable inside runtime loops.  It
    # does not represent prefetch completion; PTX prefetch remains a hint.
    return Uint32(0)


@dsl_user_op
def prefetch_bulk_l2(
    pointer: cute.Pointer,
    size: Uint32,
    policy: Int64 | None = None,
    *,
    loc=None,
    ip=None,
) -> None:
    """Issue one bulk L2 prefetch of ``size`` contiguous global bytes.

    Args:
        pointer: 16 B aligned global address of the range.
        size: Byte count, a multiple of 16.
        policy: optional L2 cache policy (``make_l2_policy_evict_first`` /
            ``_last`` in ``longcat_attention._primitives``); a kernel staging
            its successor's weights passes evict_last so the lines outlive
            its own streams.
    """
    address = pointer.toint(loc=loc, ip=ip)
    operands = [address.ir_value(loc=loc, ip=ip), size.ir_value(loc=loc, ip=ip)]
    if policy is None:
        asm, constraints = "cp.async.bulk.prefetch.L2.global [$0], $1;", "l,r,~{memory}"
    else:
        operands.append(policy.ir_value(loc=loc, ip=ip))
        asm = "cp.async.bulk.prefetch.L2.global.L2::cache_hint [$0], $1, $2;"
        constraints = "l,r,l,~{memory}"
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
def load_acquire_gpu_u32(
    pointer: cute.Pointer,
    *,
    loc=None,
    ip=None,
) -> Uint32:
    """Load a generation slot with GPU-scope acquire ordering."""
    address = pointer.toint(loc=loc, ip=ip)
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [address.ir_value(loc=loc, ip=ip)],
            "ld.global.acquire.gpu.b32 $0, [$1];",
            "=r,l,~{memory}",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def store_release_gpu_u32(
    pointer: cute.Pointer,
    value: Uint32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Publish a generation after all preceding tile stores are visible."""
    address = pointer.toint(loc=loc, ip=ip)
    llvm.inline_asm(
        None,
        [
            address.ir_value(loc=loc, ip=ip),
            value.ir_value(loc=loc, ip=ip),
        ],
        "st.global.release.gpu.b32 [$0], $1;",
        "l,r,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def store_relaxed_gpu_u32(
    pointer: cute.Pointer,
    value: Uint32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Store with GPU-scope relaxed ordering.

    Publishes only behind a ``fence.acq_rel.gpu`` in the same thread (see
    :func:`red_add_relaxed_gpu_u32`).
    """
    address = pointer.toint(loc=loc, ip=ip)
    llvm.inline_asm(
        None,
        [
            address.ir_value(loc=loc, ip=ip),
            value.ir_value(loc=loc, ip=ip),
        ],
        "st.global.relaxed.gpu.b32 [$0], $1;",
        "l,r,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def atomic_add_gpu_u32(
    pointer: cute.Pointer,
    value: Uint32,
    *,
    acq_rel: bool = True,
    clobber: bool = True,
    loc=None,
    ip=None,
) -> Uint32:
    """Atomically add, with the ordering and the clobber as separate knobs.

    ``atomic_add_acq_rel_gpu_u32`` is this with both on, and both cost: the
    GPU-scope ``acq_rel`` is a hardware fence, while ``~{memory}`` forbids
    the *compiler* from moving any memory operation across the instruction,
    so a producer warp's copies cannot overlap the add.  They are
    independent, which is why they are two arguments and not one -- a single
    "make it cheap" flag could not say which half was being paid for.

    Relaxed is enough wherever the counter is a pure ticket allocator: no
    data is published through it, and whatever the ticket points at is
    ordered by its own release -- **including the case where the ticket is
    only an index and there is no release at all**, as for the MoE tile
    counter, where the acquire the readers need comes from their phase's
    entry gate rather than from this instruction.  That is the trap: with
    ``acq_rel`` off, correctness rests on an ordering edge somewhere else
    entirely, so a caller must be able to point at it.  Never use the
    relaxed form on a counter through which data becomes visible.

    Args:
        pointer: the counter.
        value: the addend.
        acq_rel: emit ``.acq_rel.gpu`` rather than ``.relaxed.gpu``.
        clobber: declare the full-memory clobber to the compiler.

    Returns:
        The counter's value before the add.
    """
    address = pointer.toint(loc=loc, ip=ip)
    order = "acq_rel" if acq_rel else "relaxed"
    constraints = "=r,l,r,~{memory}" if clobber else "=r,l,r"
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                address.ir_value(loc=loc, ip=ip),
                value.ir_value(loc=loc, ip=ip),
            ],
            f"atom.global.{order}.gpu.add.u32 $0, [$1], $2;",
            constraints,
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def atomic_add_acq_rel_gpu_u32(
    pointer: cute.Pointer,
    value: Uint32,
    *,
    loc=None,
    ip=None,
) -> Uint32:
    """Atomically add with GPU-scope acquire-release ordering.

    The name this codebase has used everywhere; it is
    :func:`atomic_add_gpu_u32` with both knobs on.  Kept as the default
    spelling because sixty-odd call sites mean it, and delegating rather
    than repeating the asm means the two entry points cannot drift apart.
    """
    return atomic_add_gpu_u32(
        pointer, value, acq_rel=True, clobber=True, loc=loc, ip=ip
    )


@dsl_user_op
def red_add_release_gpu_u32(
    pointer: cute.Pointer,
    value: Uint32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Add with GPU-scope release ordering and no return value.

    Unlike :func:`atomic_add_acq_rel_gpu_u32` the issuing thread does not
    wait for the round trip, so a publisher can retire its ticket and move
    on; consumers still pair it with acquire loads.
    """
    address = pointer.toint(loc=loc, ip=ip)
    llvm.inline_asm(
        None,
        [
            address.ir_value(loc=loc, ip=ip),
            value.ir_value(loc=loc, ip=ip),
        ],
        "red.release.gpu.global.add.u32 [$0], $1;",
        "l,r,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def red_add_relaxed_gpu_u32(
    pointer: cute.Pointer,
    value: Uint32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Add with GPU-scope relaxed ordering and no return value.

    For a counter through which no data becomes visible: the issuing thread
    neither waits for the round trip nor fences its earlier writes.  The
    ordering rule of :func:`atomic_add_gpu_u32` applies -- the caller must
    be able to say why nothing is published through the count.

    The one exception is a word behind a ``fence.acq_rel.gpu`` in the same
    thread: fence then relaxed write is a release pattern, so several words
    can share one fence instead of each release waiting out the previous
    one's round trip.
    """
    address = pointer.toint(loc=loc, ip=ip)
    llvm.inline_asm(
        None,
        [
            address.ir_value(loc=loc, ip=ip),
            value.ir_value(loc=loc, ip=ip),
        ],
        "red.relaxed.gpu.global.add.u32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def red_add_gpu_f32(
    pointer: cute.Pointer,
    value: Float32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Fire-and-forget FP32 atomic add at GPU scope.

    Args:
        pointer: 4 B aligned global address of the accumulator element.
        value: Addend.
    """
    address = pointer.toint(loc=loc, ip=ip)
    llvm.inline_asm(
        None,
        [
            address.ir_value(loc=loc, ip=ip),
            value.ir_value(loc=loc, ip=ip),
        ],
        "red.relaxed.gpu.global.add.f32 [$0], $1;",
        "l,f,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def load_relaxed_gpu_f32(pointer, *, loc=None, ip=None) -> Float32:
    """Load one FP32 global word bypassing the non-coherent caches."""
    address = pointer.toint(loc=loc, ip=ip)
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [address.ir_value(loc=loc, ip=ip)],
            "ld.global.relaxed.gpu.f32 $0, [$1];",
            "=f,l,~{memory}",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


__all__ = [
    "load_relaxed_gpu_f32",
    "atomic_add_acq_rel_gpu_u32",
    "atomic_add_gpu_u32",
    "load_acquire_gpu_u32",
    "prefetch_global_l2",
    "red_add_gpu_f32",
    "red_add_release_gpu_u32",
    "store_release_gpu_u32",
]
