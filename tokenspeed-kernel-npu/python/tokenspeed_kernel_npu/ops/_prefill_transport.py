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

"""Bit-preserving prefill packing and owner-cache writes."""

import torch
from tokenspeed_kernel_npu._triton import tl, triton


@triton.jit(do_not_specialize=("rows",))
def _pack_current(KV, Index, Packet, rows, HAS_INDEX: tl.constexpr):
    width: tl.constexpr = 704 if HAS_INDEX else 576
    row = tl.program_id(0).to(tl.int64) * 4 + tl.arange(0, 4)[:, None]
    lane = tl.arange(0, 1024)[None, :]
    valid = (row < rows) & (lane < 576)
    kv = tl.load(KV + row * 576 + lane, valid, other=0)
    kv_offset: tl.constexpr = 128 if HAS_INDEX else 0
    tl.store(Packet + row * width + kv_offset + lane, kv, valid)
    if HAS_INDEX:
        index_lane = tl.arange(0, 128)[None, :]
        index = tl.load(Index + row * 128 + index_lane, row < rows, other=0)
        tl.store(Packet + row * width + index_lane, index, row < rows)


@triton.jit(do_not_specialize=("rows",))
def _gather_history(KV, Index, Slots, SendKV, SendIndex, rows, HAS_INDEX: tl.constexpr):
    row = tl.program_id(0).to(tl.int64) * 4 + tl.arange(0, 4)[:, None]
    source = tl.load(Slots + row, row < rows, other=0).to(tl.int64)
    lane = tl.arange(0, 1024)[None, :]
    valid = (row < rows) & (lane < 576)
    kv = tl.load(KV + source * 576 + lane, valid, other=0)
    tl.store(SendKV + row * 576 + lane, kv, valid)
    if HAS_INDEX:
        index_lane = tl.arange(0, 128)[None, :]
        index = tl.load(Index + source * 128 + index_lane, row < rows, other=0)
        tl.store(SendIndex + row * 128 + index_lane, index, row < rows)


@triton.jit(
    do_not_specialize=(
        "rows",
        "history_rows",
        "send_rows",
        "local_rows",
        "virtual_blocks",
    )
)
def _store_current(
    Packet,
    KV,
    Index,
    CacheKV,
    CacheIndex,
    Slots,
    MappedSlots,
    Direct,
    rows,
    history_rows,
    send_rows,
    local_rows,
    virtual_blocks,
    DEGREE: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    HAS_INDEX: tl.constexpr,
):
    # Copy the dense current region in tiles. Indirect owner/partial writes
    # use scalar row addresses: Ascend otherwise lowers their channels to
    # per-element loops. Only rows that need such writes reread the packet.
    producer = tl.program_id(1).to(tl.int64)
    first = tl.program_id(0).to(tl.int64) * 4
    base, extra = rows // DEGREE, rows % DEGREE
    producer_rows = base + (producer < extra)
    packet_begin = producer * base + tl.minimum(producer, extra)
    current_begin = history_rows + producer * send_rows
    width: tl.constexpr = 704 if HAS_INDEX else 576
    kv_offset: tl.constexpr = 128 if HAS_INDEX else 0
    local_rows_tile = first + tl.arange(0, 4)[:, None]
    channels = tl.arange(0, 1024)[None, :]
    valid = (local_rows_tile < producer_rows) & (channels < 576)
    current_kv = tl.load(
        Packet + (packet_begin + local_rows_tile) * width + kv_offset + channels,
        valid,
        other=0,
    )
    tl.store(KV + (current_begin + local_rows_tile) * 576 + channels, current_kv, valid)
    if HAS_INDEX:
        index_channels = tl.arange(0, 128)[None, :]
        current_index = tl.load(
            Packet + (packet_begin + local_rows_tile) * width + index_channels,
            local_rows_tile < producer_rows,
            other=0,
        )
        tl.store(
            Index + (current_begin + local_rows_tile) * 128 + index_channels,
            current_index,
            local_rows_tile < producer_rows,
        )

    for offset in tl.static_range(4):
        local_row = first + offset
        if local_row < producer_rows:
            row = packet_begin + local_row
            slot = tl.load(Slots + row).to(tl.int64)
            block = slot // BLOCK_ROWS
            owner = (block - 1) % DEGREE
            local = ((block - 1) // DEGREE + 1) * BLOCK_ROWS + slot % BLOCK_ROWS
            valid_slot = (block > 0) & (block < virtual_blocks) & (local < local_rows)
            mapped = tl.load(MappedSlots + row).to(tl.int64)
            direct = tl.load(Direct + row)
            partial = (
                valid_slot & (direct == 0) & (mapped >= 0) & (mapped < history_rows)
            )
            owned = valid_slot & (owner == RANK)
            if partial | owned:
                kv_lane = tl.arange(0, 1024)
                kv = tl.load(
                    Packet + row * width + kv_offset + kv_lane, kv_lane < 576, other=0
                )
                if HAS_INDEX:
                    lane = tl.arange(0, 128)
                    index = tl.load(Packet + row * width + lane)
                if partial:
                    tl.store(KV + mapped * 576 + kv_lane, kv, kv_lane < 576)
                    if HAS_INDEX:
                        tl.store(Index + mapped * 128 + lane, index)
                if owned:
                    tl.store(CacheKV + local * 576 + kv_lane, kv, kv_lane < 576)
                    if HAS_INDEX:
                        tl.store(CacheIndex + local * 128 + lane, index)


def pack_current(kv, index, packet):
    rows = packet.shape[0]
    if rows:
        _pack_current[(triton.cdiv(rows, 4),)](
            kv.view(torch.int16),
            index.view(torch.int16) if index is not None else kv.view(torch.int16),
            packet.view(torch.int16),
            rows,
            HAS_INDEX=index is not None,
        )


def gather_history(kv, index, slots, send_kv, send_index):
    if slots.numel():
        _gather_history[(triton.cdiv(slots.numel(), 4),)](
            kv.view(torch.int16),
            index.view(torch.int16) if index is not None else kv.view(torch.int16),
            slots,
            send_kv.view(torch.int16),
            (
                send_index.view(torch.int16)
                if send_index is not None
                else send_kv.view(torch.int16)
            ),
            slots.numel(),
            HAS_INDEX=index is not None,
        )


def store_current(
    packet,
    kv,
    index,
    cache_kv,
    cache_index,
    slots,
    mapped_slots,
    direct,
    *,
    history_rows,
    send_rows,
    block_rows,
    virtual_blocks,
    degree,
    rank,
):
    rows = packet.shape[0]
    if rows:
        _store_current[(triton.cdiv(triton.cdiv(rows, degree), 4), degree)](
            packet.view(torch.int16),
            kv.view(torch.int16),
            index.view(torch.int16) if index is not None else kv.view(torch.int16),
            cache_kv.view(torch.int16),
            (
                cache_index.view(torch.int16)
                if cache_index is not None
                else cache_kv.view(torch.int16)
            ),
            slots,
            mapped_slots,
            direct,
            rows,
            history_rows,
            send_rows,
            cache_kv.numel() // 576,
            virtual_blocks,
            DEGREE=degree,
            RANK=rank,
            BLOCK_ROWS=block_rows,
            HAS_INDEX=index is not None,
        )
