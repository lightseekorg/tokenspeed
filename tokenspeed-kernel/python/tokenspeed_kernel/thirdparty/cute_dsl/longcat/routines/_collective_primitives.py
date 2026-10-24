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

"""Packed-memory and MNNVL primitives owned by the LongCat corridor.

These primitives implement the corridor's payload stores, synchronization,
and BF16 fragment packing.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Int64, Uint32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, vector
from cutlass.cutlass_dsl import T, dsl_user_op
from tokenspeed_kernel.thirdparty.cute_dsl.latent_moe_tail.primitives import (
    VEC_BF16,
    bf16x8_to_packed_u32x4,
    packed_u32x4_to_bf16x8,
)


@dsl_user_op
def invalidate_mbarrier(pointer: cute.Pointer, *, loc=None, ip=None) -> None:
    """End a drained shared mbarrier's lifetime before reusing its storage."""
    address = pointer.toint(loc=loc, ip=ip).to(cutlass.Int32)
    llvm.inline_asm(
        None,
        [address.ir_value(loc=loc, ip=ip)],
        "mbarrier.inval.shared::cta.b64 [$0];",
        "r,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def load_global_u32x4(
    pointer: cute.Pointer,
    *,
    bypass_l1: bool = False,
    loc=None,
    ip=None,
):
    """Load 128 bits; peer payload reads bypass L1 across slot generations."""
    address = pointer.toint(loc=loc, ip=ip)
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [address.ir_value(loc=loc, ip=ip)],
        ("ld.global.cg.v4.u32" if bypass_l1 else "ld.global.v4.u32")
        + " {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l" + (",~{memory}" if bypass_l1 else ""),
        has_side_effects=bypass_l1,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    packed = vector.from_elements(
        ir.VectorType.get([4], T.i32(), loc=loc),
        [llvm.extractvalue(T.i32(), result, [i], loc=loc, ip=ip) for i in range(4)],
        loc=loc,
        ip=ip,
    )
    return cute.TensorSSA(packed, 4, Uint32)


@dsl_user_op
def store_global_u32x4(
    address: Int64,
    packed,
    *,
    loc=None,
    ip=None,
) -> None:
    """Store one packed BF16x8 fragment to a unicast global address."""
    words = [packed[i].ir_value(loc=loc, ip=ip) for i in range(4)]
    llvm.inline_asm(
        None,
        [address.ir_value(loc=loc, ip=ip), *words],
        "st.global.v4.u32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def load_acquire_sys_u32(
    pointer: cute.Pointer,
    *,
    loc=None,
    ip=None,
) -> Uint32:
    """Load an NVLS ticket with system-scope acquire ordering."""
    address = pointer.toint(loc=loc, ip=ip)
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [address.ir_value(loc=loc, ip=ip)],
            "ld.acquire.global.sys.b32 $0, [$1];",
            "=r,l,~{memory}",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def fence_proxy_async_global(*, loc=None, ip=None) -> None:
    """Order this thread's generic-proxy view of global memory before its
    async-proxy (``cp.async.bulk``) reads: after an acquire of data another
    SM wrote with ordinary stores, the bulk copy must see those stores."""
    llvm.inline_asm(
        None,
        [],
        "fence.proxy.async.global;",
        "~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


__all__ = [
    "VEC_BF16",
    "bf16x8_to_packed_u32x4",
    "load_acquire_sys_u32",
    "load_global_u32x4",
    "packed_u32x4_to_bf16x8",
    "store_global_u32x4",
]
