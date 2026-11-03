# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 MiniMax
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


"""Single-CTA TMA gather primitives shared by CuTe attention implementations."""

from cutlass import Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.experimental import primitives as prims

TMA_CACHE_EVICT_LAST = 0x14F0000000000000


def tma_cache_hint(value: int):
    """Materialize a constant L2 policy in the register required by TMA PTX."""
    return prims.inline_ptx(
        f"mov.b64 $0, 0x{value & 0xFFFFFFFFFFFFFFFF:016X};",
        write_only_types=[Int64],
    )


@dsl_user_op
def tma_gather4(
    smem_address,
    descriptor_address,
    mbar_address,
    col,
    row0,
    row1,
    row2,
    row3,
    *,
    cache_hint,
    loc=None,
    ip=None,
):
    """Issue one four-row gather with byte-count completion on a local mbarrier.

    Shared-memory destination and barrier addresses are 32-bit byte addresses;
    descriptor_address is the 64-bit tensor-map address. Coordinates are in
    tensor elements. cache_hint is an explicit L2 policy held in a register.
    The caller owns thread election and the barrier's expected byte count.
    Returns no value.
    """
    llvm.inline_asm(
        None,
        [
            Int32(smem_address).ir_value(loc=loc, ip=ip),
            Int64(descriptor_address).ir_value(loc=loc, ip=ip),
            Int32(col).ir_value(loc=loc, ip=ip),
            Int32(row0).ir_value(loc=loc, ip=ip),
            Int32(row1).ir_value(loc=loc, ip=ip),
            Int32(row2).ir_value(loc=loc, ip=ip),
            Int32(row3).ir_value(loc=loc, ip=ip),
            Int32(mbar_address).ir_value(loc=loc, ip=ip),
            Int64(cache_hint).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4."
        "mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint "
        "[$0], [$1, {$2, $3, $4, $5, $6}], [$7], $8;",
        "r,l,r,r,r,r,r,r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
