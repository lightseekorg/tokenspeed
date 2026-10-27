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

"""Shared score encoding, warp scans and asynchronous copies."""

from __future__ import annotations

import cutlass
from cutlass import cute
from cutlass._mlir.dialects import llvm as _llvm
from cutlass.cute.typing import Pointer
from cutlass.cutlass_dsl import dsl_user_op


@dsl_user_op
def _copy_u128(destination: Pointer, source: Pointer, *, loc=None, ip=None):
    _llvm.inline_asm(
        None,
        [
            destination.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            source.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
        ],
        "{ .reg .u32 address; cvt.u32.u64 address, $0; cp.async.cg.shared.global [address], [$1], 16; }",
        "l,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _copy_u128_masked(
    destination: Pointer, source: Pointer, valid: cutlass.Boolean, *, loc=None, ip=None
):
    # src-size=0 zero-fills without reading source, including past the Q tail.
    size = cutlass.Int32(valid) * 16
    _llvm.inline_asm(
        None,
        [
            destination.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            source.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            size.ir_value(loc=loc, ip=ip),
        ],
        "{ .reg .u32 address; cvt.u32.u64 address, $0; cp.async.cg.shared.global [address], [$1], 16, $2; }",
        "l,l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


class QueryGroups:
    """Resolve request-owned groups without a metadata allocation or host read.

    Uniform runs use arithmetic. Ragged runs start a group at each eight-row
    boundary and at each ownership change, splitting at the next boundary.
    Inactive ragged CTAs stop before initializing the scoring pipelines.
    """

    @cute.jit
    def _group_count(self, rows: cutlass.Int32, width: cutlass.Int32):
        if cutlass.const_expr(self.group_layout == "aligned"):
            count = rows // 8
        elif cutlass.const_expr(self.group_layout == "uniform"):
            count = (rows // width) * cute.ceil_div(width, 8)
        elif cutlass.const_expr(self.group_layout == "ragged"):
            count = rows
        return count

    @cute.jit
    def _query_group(
        self,
        requests: cute.Tensor,
        rows: cutlass.Int32,
        width: cutlass.Int32,
        group: cutlass.Int32,
    ):
        if cutlass.const_expr(self.group_layout == "aligned"):
            start = group * 8
            valid = cutlass.Int32(8)
        elif cutlass.const_expr(self.group_layout == "uniform"):
            groups_per_request = cute.ceil_div(width, 8)
            offset = (group % groups_per_request) * 8
            start = (group // groups_per_request) * width + offset
            valid = cutlass.min(cutlass.Int32(8), width - offset)
        elif cutlass.const_expr(self.group_layout == "ragged"):
            start = group
            valid = cutlass.Int32(0)
            active = start % 8 == 0
            if not active:
                active = requests[start - 1] != requests[start]
            if active:
                valid = cutlass.Int32(1)
                end = cutlass.min(rows, (start // 8 + 1) * 8)
                owner = requests[start]
                for offset in cutlass.range_constexpr(1, 8):
                    if valid == offset and start + offset < end:
                        if requests[start + offset] == owner:
                            valid += 1
        return start, valid

    @cute.jit
    def _row_complete(
        self,
        complete: cute.Tensor,
        start: cutlass.Int32,
        offset: cutlass.Int32,
        valid: cutlass.Int32,
    ):
        count = cutlass.Int32(0)
        if cutlass.const_expr(self.group_layout == "aligned"):
            count = complete[start + offset]
        elif offset < valid:
            count = complete[start + offset]
        return count

    @cute.jit
    def _copy_query(
        self,
        destination: cute.Pointer,
        source: cute.Pointer,
        offset: cutlass.Int32,
        valid: cutlass.Int32,
    ):
        if cutlass.const_expr(self.group_layout == "aligned"):
            _copy_u128(destination, source)
        else:
            _copy_u128_masked(destination, source, offset < valid)

    @cute.jit
    def _store_selected(
        self,
        selected: cute.Tensor,
        row: cutlass.Int32,
        column: cutlass.Int32,
        value,
        start: cutlass.Int32,
        valid: cutlass.Int32,
    ):
        if cutlass.const_expr(self.group_layout == "aligned"):
            selected[row, column] = value
        elif row < start + valid:
            selected[row, column] = value

    @cute.jit
    def _store_row(
        self,
        selected_row: cute.Tensor,
        column: cutlass.Int32,
        value,
        row_in_group: cutlass.Int32,
        valid_rows: cutlass.Int32,
    ):
        if cutlass.const_expr(self.group_layout == "aligned"):
            selected_row[column] = value
        elif row_in_group < valid_rows:
            selected_row[column] = value


@dsl_user_op
def _match_any(value: cutlass.Int32, *, loc=None, ip=None):
    result = _llvm.inline_asm(
        cutlass.Uint32.mlir_type,
        [value.ir_value(loc=loc, ip=ip)],
        "match.any.sync.b32 $0, $1, 0xffffffff;",
        "=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return cutlass.Uint32(result)


@dsl_user_op
def _load_global_u32(pointer: Pointer, *, loc=None, ip=None):
    value = _llvm.inline_asm(
        cutlass.Uint32.mlir_type,
        [pointer.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)],
        "ld.global.cg.u32 $0, [$1];",
        "=r,l,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return cutlass.Uint32(value)


class SelectorPrimitives:
    @cute.jit
    def _float_bits(self, value: cutlass.Float32):
        return cutlass.Uint32(_llvm.bitcast(cutlass.Uint32.mlir_type, value.ir_value()))

    @cute.jit
    def _coarse(self, bits: cutlass.Uint32):
        value = cutlass.Float32(
            _llvm.bitcast(cutlass.Float32.mlir_type, bits.ir_value())
        )
        half = cutlass.Float16(value)
        half_bits = cutlass.Uint16(
            _llvm.bitcast(cutlass.Uint16.mlir_type, half.ir_value())
        )
        return cutlass.Int32(
            ((half_bits ^ cutlass.Uint16(0xFFFF)) & cutlass.Uint16(0x7FFF))
            >> cutlass.Uint16(7)
        )

    @cute.jit
    def _warp_scan(self, value: cutlass.Int32, lane: cutlass.Int32):
        for offset in cutlass.range_constexpr(5):
            distance = cutlass.const_expr(1 << offset)
            other = cute.arch.shuffle_sync_up(
                value, distance, mask=0xFFFFFFFF, mask_and_clamp=0
            )
            if lane >= distance:
                value += other
        return value

    @cute.jit
    def _add_bucket(
        self,
        histogram: cute.Tensor,
        bucket: cutlass.Int32,
        tidx: cutlass.Int32,
        lane: cutlass.Int32,
    ):
        # All lanes participate, including filtered entries in sentinel bin 256.
        matching = _match_any(bucket)
        lower = (cutlass.Uint32(1) << cutlass.Uint32(lane)) - cutlass.Uint32(1)
        if bucket < 256 and (matching & lower) == 0:
            cute.arch.atomic_add(
                histogram.iterator + (tidx // 32) * 256 + bucket,
                cute.arch.popc(matching),
                sem="relaxed",
                scope="cta",
            )
