# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 LightSeek Foundation
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
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

"""Stream-ordered ownership metadata for batch-banked recurrent replay."""

from tokenspeed_kernel._triton import tl, triton


@triton.jit
def _owned_replay(records, row_slots, generations, valid, slot, C, S):
    live = (slot > 0) & (slot < S - 1)
    record = records + slot * 4
    code = tl.load(record + 1, live, -1)
    generation = tl.load(record, live, -1)
    in_range = (code >= 0) & (code < 2 * C)
    owned = tl.load(valid + code, in_range, False)
    owned &= tl.load(row_slots + code, in_range, -1) == slot
    owned &= tl.load(generations + code, in_range, -1) == generation
    return code, owned


@triton.jit(
    do_not_specialize=["C", "B", "S", "N"],
    do_not_specialize_on_alignment=["input_slots"],
)
def pending_metadata(
    records,
    row_slots,
    generations,
    steps,
    pages,
    valid,
    bank,
    rows,
    slots,
    reads,
    writes,
    immediate,
    current_steps,
    current_stopped,
    current_frontier,
    consumed,
    flush,
    group_indices,
    request_workspaces,
    input_slots,
    input_frontier,
    input_stopped,
    input_accepted,
    input_reads,
    input_writes,
    C,
    B,
    S,
    N,
    L: tl.constexpr,
    LB: tl.constexpr,
    OP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    if OP == 0:
        offset = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        tl.store(records + offset.to(tl.int64) * 4 + 3, -1, offset < S)
        if tl.program_id(0) == 0:
            tl.store(bank, 1 - tl.load(bank))
    elif OP == 1:
        row = tl.program_id(0)
        layer = tl.arange(0, LB)
        slot = tl.load(input_slots + row, row < B, 0).to(tl.int64)
        live = (row < B) & (slot > 0) & (slot < S - 1)
        record = records + slot * 4
        code = tl.load(record + 1)
        old_row = tl.maximum(code, 0) % C
        old_bank = tl.maximum(code, 0) // C
        group = tl.load(group_indices + layer, layer < L, 0).to(tl.int64)
        source = tl.load(input_reads + group * B + row, (layer < L) & (row < B), -1)
        destination = tl.load(
            input_writes + group * B + row, (layer < L) & (row < B), -1
        )
        old_page = tl.load(
            pages + (old_bank * L + layer.to(tl.int64)) * C + old_row, layer < L, -1
        )
        same_pages = tl.sum(((source != old_page) & (layer < L)).to(tl.int32), 0) == 0
        usable = live & (code >= 0) & same_pages & (old_bank != tl.load(bank))
        usable &= tl.load(valid + tl.maximum(code, 0))
        usable &= tl.load(generations + tl.maximum(code, 0)) == tl.load(record)
        usable &= tl.load(record + 2) == tl.load(input_frontier + slot)
        tl.store(slots + row, slot)
        tl.store(rows + row, tl.where(usable, code, -1).to(tl.int32))
        tl.store(reads + layer.to(tl.int64) * C + row, source, layer < L)
        tl.store(writes + layer.to(tl.int64) * C + row, destination, layer < L)
        if live:
            tl.store(record + 3, row)
    elif OP == 2 or OP == 3 or OP == 6:
        code = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        within = code < 2 * C
        slot = tl.load(row_slots + code, within, 0).to(tl.int64)
        record = records + slot * 4
        owned = within & tl.load(valid + code, within, False)
        owned &= tl.load(record) == tl.load(generations + code, within, -1)
        owned &= tl.load(record + 1) == code
        if OP == 2:
            member = tl.load(record + 3)
            continuing = (member >= 0) & (tl.load(rows + tl.maximum(member, 0)) == code)
            tl.store(consumed + code, owned & continuing, within)
            tl.store(flush + code, owned & ~continuing, within)
        elif OP == 3:
            selected = owned & tl.load(flush + code, within, False)
            tl.store(record + 1, -1, selected)
            tl.store(valid + code, False, selected)
        else:
            tl.store(flush + code, owned, within)
    elif OP == 4:
        row = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        slot = tl.load(slots + row, row < C, 0).to(tl.int64)
        tl.store(current_stopped + row, tl.load(input_stopped + slot), row < C)
        tl.store(current_frontier + row, tl.load(input_frontier + slot), row < C)
    elif OP == 5:
        row = tl.program_id(0)
        layer = tl.arange(0, LB)
        slot = tl.load(slots + row).to(tl.int64)
        live = (slot > 0) & (slot < S - 1)
        count = tl.minimum(4, tl.maximum(0, tl.load(input_accepted + row, row < B, 0)))
        source = tl.load(reads + layer.to(tl.int64) * C + row, layer < L, -1)
        destination = tl.load(writes + layer.to(tl.int64) * C + row, layer < L, -1)
        same_pages = (
            tl.sum(
                ((source != destination) | (destination <= 0)).to(tl.int32)
                & (layer < L).to(tl.int32),
                0,
            )
            == 0
        )
        needs_copy = tl.sum(((source != destination) & (layer < L)).to(tl.int32), 0) > 0
        defer = live & (count > 0) & ~tl.load(current_stopped + row) & same_pages
        current_bank = tl.load(bank).to(tl.int64)
        new_code = current_bank * C + row
        old_code = tl.load(rows + row).to(tl.int64)
        if old_code >= 0:
            if tl.load(consumed + old_code):
                tl.store(valid + old_code, False)
        record = records + slot * 4
        generation = tl.load(record) + 1
        if live:
            tl.store(record, generation)
            tl.store(record + 1, tl.where(defer, new_code, -1))
            tl.store(record + 2, tl.load(current_frontier + row).to(tl.int64) + count)
        tl.store(row_slots + new_code, slot)
        tl.store(generations + new_code, generation)
        tl.store(steps + new_code, count)
        tl.store(valid + new_code, defer)
        tl.store(
            pages + (current_bank * L + layer.to(tl.int64)) * C + row,
            destination,
            layer < L,
        )
        tl.store(current_steps + row, count)
        tl.store(
            immediate + row,
            live & ~defer & ((count > 0) | needs_copy) & (current_bank == 0),
        )
        tl.store(
            immediate + C + row,
            live & ~defer & ((count > 0) | needs_copy) & (current_bank == 1),
        )
    elif OP == 7:
        # Each bank compacts its selected requests directly from the slot list.
        # The list is unique, so no replay row can occupy two entries.
        selected_bank = tl.program_id(0).to(tl.int64)
        selected = request_workspaces + selected_bank * (C.to(tl.int64) + 1)
        count = tl.full((), 0, tl.int32)
        for first in range(tl.cdiv(N, BLOCK)):
            offsets = first.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
            slot = tl.load(input_slots + offsets, offsets < N, 0).to(tl.int64)
            code, owned = _owned_replay(
                records, row_slots, generations, valid, slot, C, S
            )
            keep = owned & (code // C == selected_bank)
            prefix = tl.cumsum(keep.to(tl.int32))
            tl.store(
                selected + (count + prefix - 1).to(tl.int64),
                (code % C).to(tl.int32),
                keep,
            )
            count += tl.sum(keep.to(tl.int32))
        tl.store(selected + C, count)
    elif OP == 8:
        row = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        slot = tl.load(input_slots + row, row < N, 0).to(tl.int64)
        code, owned = _owned_replay(records, row_slots, generations, valid, slot, C, S)
        tl.store(records + slot * 4 + 1, -1, owned)
        tl.store(valid + code, False, owned)
