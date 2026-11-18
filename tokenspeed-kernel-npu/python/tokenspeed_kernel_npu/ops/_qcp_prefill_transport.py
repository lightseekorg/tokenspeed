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

"""Bit-preserving QCP Prefill packing and KVP owner-cache writes."""

import torch
from tokenspeed_kernel_npu._triton import tl, triton


@triton.jit(do_not_specialize=("rows",))
def _pack_current(KV, Index, Packet, rows, HAS_INDEX: tl.constexpr):
    width: tl.constexpr = 704 if HAS_INDEX else 576
    row = tl.program_id(0).to(tl.int64) * 32 + tl.arange(0, 32)[:, None]
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
    first = tl.program_id(0).to(tl.int64) * 32
    offset = tl.arange(0, 32)
    source = tl.load(Slots + first + offset, first + offset < rows, other=0)
    source_first = tl.load(Slots + first).to(tl.int64)
    discontinuous = (first + offset < rows) & (source != source_first + offset)
    if tl.sum(discontinuous.to(tl.int32), 0) == 0:
        # A scalar base exposes page-local runs as wide contiguous transfers.
        count = tl.minimum(rows - first, 32)
        run_lane = tl.arange(0, 32768)
        run_valid = run_lane < count * 576
        run_kv = tl.load(KV + source_first * 576 + run_lane, run_valid, other=0)
        tl.store(SendKV + first * 576 + run_lane, run_kv, run_valid)
        if HAS_INDEX:
            run_index_lane = tl.arange(0, 4096)
            run_index_valid = run_index_lane < count * 128
            run_index = tl.load(
                Index + source_first * 128 + run_index_lane, run_index_valid, other=0
            )
            tl.store(
                SendIndex + first * 128 + run_index_lane, run_index, run_index_valid
            )
    else:
        # Arbitrary slot order still needs scalar row bases for contiguous DMA.
        for row_offset in range(32):
            row = first + row_offset
            if row < rows:
                source_row = tl.load(Slots + row).to(tl.int64)
                lane = tl.arange(0, 1024)
                kv = tl.load(KV + source_row * 576 + lane, lane < 576, other=0)
                tl.store(SendKV + row * 576 + lane, kv, lane < 576)
                if HAS_INDEX:
                    index_lane = tl.arange(0, 128)
                    index = tl.load(Index + source_row * 128 + index_lane)
                    tl.store(SendIndex + row * 128 + index_lane, index)


@triton.jit(
    do_not_specialize=(
        "rows",
        "history_rows",
        "current_rank_stride",
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
    current_rank_stride,
    local_rows,
    virtual_blocks,
    DEGREE: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    HAS_INDEX: tl.constexpr,
):
    # Full direct tiles share an owner and reuse the unpacked values for its
    # cache write. Irregular and partial rows retain scalar address handling.
    producer = tl.program_id(1).to(tl.int64)
    first = tl.program_id(0).to(tl.int64) * 32
    base, extra = rows // DEGREE, rows % DEGREE
    producer_rows = base + (producer < extra)
    packet_begin = producer * base + tl.minimum(producer, extra)
    current_begin = history_rows + producer * current_rank_stride
    width: tl.constexpr = 704 if HAS_INDEX else 576
    kv_offset: tl.constexpr = 128 if HAS_INDEX else 0
    if first < producer_rows:
        tile_rows = tl.arange(0, 32)
        local_rows_tile = first + tile_rows[:, None]
        channels = tl.arange(0, 1024)[None, :]
        valid = (local_rows_tile < producer_rows) & (channels < 576)
        current_kv = tl.load(
            Packet + (packet_begin + local_rows_tile) * width + kv_offset + channels,
            valid,
            other=0,
        )
        tl.store(
            KV + (current_begin + local_rows_tile) * 576 + channels, current_kv, valid
        )
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

        packet_first = packet_begin + first
        run_mask = first + tile_rows < producer_rows
        run_slots = tl.load(Slots + packet_first + tile_rows, run_mask, other=0)
        run_direct = tl.load(Direct + packet_first + tile_rows, run_mask, other=1)
        first_slot = tl.load(Slots + packet_first).to(tl.int64)
        count = tl.minimum(producer_rows - first, 32)
        first_block = first_slot // BLOCK_ROWS
        one_block = first_block == (first_slot + count - 1) // BLOCK_ROWS
        irregular = run_mask & (
            (run_slots != first_slot + tile_rows) | (run_direct == 0)
        )
        cache_first = (
            (first_block - 1) // DEGREE + 1
        ) * BLOCK_ROWS + first_slot % BLOCK_ROWS
        if (
            one_block
            & (cache_first + count <= local_rows)
            & (tl.sum(irregular.to(tl.int32), 0) == 0)
        ):
            first_owner = (first_block - 1) % DEGREE
            if (
                (first_block > 0)
                & (first_block < virtual_blocks)
                & (first_owner == RANK)
            ):
                tl.store(
                    CacheKV + cache_first * 576 + tile_rows[:, None] * 576 + channels,
                    current_kv,
                    valid,
                )
                if HAS_INDEX:
                    tl.store(
                        CacheIndex
                        + cache_first * 128
                        + tile_rows[:, None] * 128
                        + index_channels,
                        current_index,
                        local_rows_tile < producer_rows,
                    )
        else:
            for row_offset in range(32):
                local_row = first + row_offset
                if local_row < producer_rows:
                    row = packet_begin + local_row
                    slot = tl.load(Slots + row).to(tl.int64)
                    block = slot // BLOCK_ROWS
                    owner = (block - 1) % DEGREE
                    local = ((block - 1) // DEGREE + 1) * BLOCK_ROWS + slot % BLOCK_ROWS
                    valid_slot = (
                        (block > 0) & (block < virtual_blocks) & (local < local_rows)
                    )
                    mapped = tl.load(MappedSlots + row).to(tl.int64)
                    direct = tl.load(Direct + row)
                    partial = (
                        valid_slot
                        & (direct == 0)
                        & (mapped >= 0)
                        & (mapped < history_rows)
                    )
                    owned = valid_slot & (owner == RANK)
                    if partial | owned:
                        kv_lane = tl.arange(0, 1024)
                        kv = tl.load(
                            Packet + row * width + kv_offset + kv_lane,
                            kv_lane < 576,
                            other=0,
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
        _pack_current[(triton.cdiv(rows, 32),)](
            kv.view(torch.int16),
            index.view(torch.int16) if index is not None else kv.view(torch.int16),
            packet.view(torch.int16),
            rows,
            HAS_INDEX=index is not None,
        )


def gather_history(kv, index, slots, send_kv, send_index):
    if slots.numel():
        _gather_history[(triton.cdiv(slots.numel(), 32),)](
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
    current_rank_stride,
    block_rows,
    virtual_blocks,
    degree,
    rank,
):
    rows = packet.shape[0]
    if rows:
        _store_current[(triton.cdiv(triton.cdiv(rows, degree), 32), degree)](
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
            current_rank_stride,
            cache_kv.numel() // 576,
            virtual_blocks,
            DEGREE=degree,
            RANK=rank,
            BLOCK_ROWS=block_rows,
            HAS_INDEX=index is not None,
        )
