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

"""Ascend Prefill batch metadata and history/current cache transport."""

from __future__ import annotations

import torch

from tokenspeed.runtime.execution.query_shard import scatter_count


def gather_prefill_pages(
    table_cpu,
    extends,
    prefixes,
    *,
    placement,
    page_size,
    local_pages,
    degree,
    rank,
    layout,
    device,
    columns,
):
    if table_cpu is None or table_cpu.device.type != "cpu":
        raise ValueError("Prefill requires the scheduler's host page table")
    if layout.null_rows % page_size or len(extends) != table_cpu.shape[0]:
        raise ValueError("Prefill page and kernel table geometry disagree")
    per_block = 1 if placement is None else placement.block_granularity // page_size
    pages = table_cpu.tolist()
    owners = [{} for _ in range(degree)]
    for req, (prefix, extent) in enumerate(zip(prefixes, extends, strict=True)):
        if extent <= 0:
            continue
        count = (prefix + extent + page_size - 1) // page_size
        if count > min(len(pages[req]), columns):
            raise ValueError("Prefill KV length exceeds its scheduler page table")
        for page in pages[req][:count]:
            block = page // per_block
            if block <= 0:
                continue
            owner = (block - 1) % degree
            local = ((block - 1) // degree + 1) * per_block + page % per_block
            if local >= local_pages:
                raise ValueError("Prefill page lies outside its owner's cache")
            owners[owner].setdefault(page, local)
    offsets, offset = [], layout.null_rows // page_size
    for owner in owners:
        offsets.append(offset)
        offset += len(owner)
    mapping = {
        page: offsets[owner] + i
        for owner, entries in enumerate(owners)
        for i, page in enumerate(entries)
    }
    packed = torch.zeros((len(extends), columns), dtype=table_cpu.dtype)
    for req, (prefix, extent) in enumerate(zip(prefixes, extends, strict=True)):
        if extent <= 0:
            continue
        count = (prefix + extent + page_size - 1) // page_size
        packed[req, :count] = torch.tensor(
            [mapping.get(page, 0) for page in pages[req][:count]],
            dtype=packed.dtype,
        )
    source_pages = torch.tensor(tuple(owners[rank].values()), dtype=torch.int64)
    slots = (source_pages[:, None] * page_size + torch.arange(page_size)).reshape(-1)
    return (
        packed.to(device=device),
        slots.to(device=device),
        tuple(len(owner) * page_size for owner in owners),
    )


class AscendPrefill:
    """Prepare QCP/KVP metadata once per batch and reuse one transport workspace."""

    def __init__(self, *, layout, page_size, topk, runtime):
        self.layout = layout
        self.page_size = page_size
        self.topk = topk
        self.runtime = runtime
        self.degree, self.rank = runtime.degree, runtime.rank
        self.placement = None
        self.row_counts = []
        self.row_start = self.row_end = 0
        self.page_table = None
        self.query_cu = self.key_cu = None
        self.query_seq = self.scan_count = None
        self.current_direct = self.patch_slots = None
        self.has_prefix_history = False
        self.key_row_count = 0
        self.history_slots = None
        self.history_counts = ()
        self.history_rows = self.send_rows = 0

    def prepare(self, metadata, placement, *, page_table_cpu, query_partition=None):
        """Resolve existing scheduler pages and each rank's causal query span."""
        degree, rank = self.degree, self.rank
        layout, page_size, topk = self.layout, self.page_size, self.topk
        local_cache_pages = self.runtime.local_rows // page_size
        # Lengths already exist on the host. Bounded gathers also reuse the
        # scheduler's host page table; no device readback is needed.
        extends = metadata.extend_seq_lens_cpu.tolist()
        prefixes = metadata.extend_prefix_lens_cpu.tolist()
        if not (len(extends) == len(prefixes) == metadata.page_table.shape[0]):
            raise ValueError("Prefill lengths must match the request page-table rows")
        if degree < 1 or not 0 <= rank < degree or local_cache_pages <= 0:
            raise ValueError("invalid Prefill context-parallel cache geometry")
        if any(n < 0 for n in extends + prefixes):
            raise ValueError("Prefill prefix and extend lengths must be nonnegative")
        if placement is not None and placement.block_granularity % page_size:
            raise ValueError("Prefill ownership blocks must contain whole kernel pages")
        rows = sum(extends)
        counts = scatter_count(rows, degree)
        first, count = sum(counts[:rank]), counts[rank]
        if query_partition is not None:
            first, count = query_partition
            if first < 0 or count < 0 or first + count > rows:
                raise ValueError("Prefill query partition exceeds current rows")
        # The shared receive transport pads each rank's contribution to 128
        # rows, including MLA readers with 64-token kernel pages.
        send_rows = ((rows + degree * 128 - 1) // (degree * 128)) * 128
        offset, active, key_rows = 0, 0, 0
        for prefix, extent in zip(prefixes, extends, strict=True):
            if (
                count
                and offset < first + count
                and offset + extent > first
                and extent > 0
            ):
                active += 1
                key_rows += prefix + min(extent, first + count - offset)
            offset += extent
        table = metadata.page_table
        per_block = 1 if placement is None else placement.block_granularity // page_size
        history_table, history_slots, history_counts = gather_prefill_pages(
            table if table.device.type == "cpu" else page_table_cpu,
            extends,
            prefixes,
            placement=placement,
            page_size=page_size,
            local_pages=local_cache_pages,
            degree=degree,
            rank=rank,
            layout=layout,
            device=table.device,
            columns=table.shape[1],
        )
        if sum(history_counts) > layout.history_capacity:
            raise ValueError(
                "Prefill batch KV pages exceed the reserved workspace capacity"
            )
        history_pages = (layout.null_rows + sum(history_counts)) // page_size
        patch_slots = torch.empty(rows, dtype=torch.int64, device=table.device)
        if table.device.type == "cpu":
            mapped = history_table.clone()
            cumulative = metadata.cum_extend_seq_lens.long()
            prefix = metadata.extend_prefix_lens.long()
            direct = torch.zeros(rows, dtype=torch.uint8)
            base, extra = divmod(rows, degree)
            for req in range(len(extends)):

                def location(token, req=req):
                    page_start = (
                        token - (prefix[req] + token - cumulative[req]) % page_size
                    )
                    owner = torch.where(
                        page_start < (base + 1) * extra,
                        page_start // (base + 1),
                        extra + (page_start - (base + 1) * extra) // max(base, 1),
                    )
                    owner_start = owner * base + owner.clamp_max(extra)
                    owner_end = owner_start + base + (owner < extra)
                    alias = (
                        (page_start >= cumulative[req])
                        & (page_start + page_size <= cumulative[req + 1])
                        & ((page_start - owner_start) % page_size == 0)
                        & (page_start + page_size <= owner_end)
                    )
                    return owner, page_start - owner_start, alias

                page_token = (
                    cumulative[req]
                    + torch.arange(table.shape[1]) * page_size
                    - prefix[req]
                )
                owner, offset, alias = location(page_token)
                tail_page = history_pages + (owner * send_rows + offset) // page_size
                mapped[req] = torch.where(alias, tail_page, mapped[req]).int()
                token = torch.arange(cumulative[req], cumulative[req + 1])
                position = prefix[req] + token - cumulative[req]
                page = history_table[req, position // page_size].long()
                patch_slots[token] = torch.where(
                    page > 0, page * page_size + position % page_size, -1
                )
                direct[cumulative[req] : cumulative[req + 1]] = location(token)[2].to(
                    torch.uint8
                )
            ends = (
                (first + count - cumulative[:-1])
                .clamp_min(0)
                .minimum(cumulative[1:] - cumulative[:-1])
            )
            qcu = torch.cat(
                (
                    torch.zeros(1, dtype=torch.int64),
                    (cumulative[1:] - first).clamp(0, count),
                )
            )
            kcu = torch.cat(
                (torch.zeros(1, dtype=torch.int64), (prefix + ends).cumsum(0))
            )
            token = torch.arange(first, first + count)
            sequence = torch.searchsorted(cumulative[1:], token, right=True)
            scan = (prefix[sequence] + token - cumulative[sequence] + 1).clamp_max(topk)
            sequence, scan = sequence.int(), scan.int()
            active_mask = (
                (cumulative[:-1] < first + count)
                & (cumulative[1:] > first)
                & (cumulative[1:] > cumulative[:-1])
            )
            mapped = mapped[active_mask]
            sequence = (active_mask.int().cumsum(0)[sequence.long()] - 1).int()
            qcu = torch.cat(
                (
                    torch.zeros(1, dtype=torch.int64),
                    (torch.diff(qcu)[active_mask]).cumsum(0),
                )
            )
            kcu = torch.cat(
                (
                    torch.zeros(1, dtype=torch.int64),
                    (torch.diff(kcu)[active_mask]).cumsum(0),
                )
            )

        else:
            from tokenspeed_kernel.ops.attention.dsa.ascend import (
                ascend_prefill_cache_metadata,
            )

            (
                mapped,
                qcu,
                kcu,
                sequence,
                scan,
                direct,
            ) = ascend_prefill_cache_metadata(
                table,
                metadata.cum_extend_seq_lens,
                metadata.extend_prefix_lens,
                first=first,
                rows=count,
                total=rows,
                per_block=per_block,
                degree=degree,
                local_pages=local_cache_pages,
                send_rows=send_rows,
                page_size=page_size,
                topk=topk,
                active_requests=active,
                history_table=history_table,
                patch_slots=patch_slots,
                history_pages=history_pages,
            )
        self.placement = placement
        self.row_counts = counts
        self.row_start, self.row_end = first, first + count
        self.page_table = mapped
        self.query_cu, self.key_cu = qcu, kcu
        self.query_seq, self.scan_count = sequence, scan
        self.current_direct, self.patch_slots = direct, patch_slots
        self.has_prefix_history = any(prefixes)
        self.key_row_count = key_rows
        self.history_slots, self.history_counts = history_slots, history_counts
        self.history_rows = layout.null_rows + sum(history_counts)
        self.send_rows = send_rows

    def gather_prefill_history(self, kv_cache, index_cache):
        """Return this layer's gathered history in the shared workspace."""
        return self.runtime.gather_history(
            kv_cache,
            index_cache,
            total_rows=sum(self.row_counts),
            has_history=self.has_prefix_history,
            history_slots=self.history_slots,
            history_counts=self.history_counts,
        )

    def prepare_layer(self, kv_cache, index_cache, current_kv, current_index, slots):
        """Gather history, gather current projections and commit owner-local cache."""
        kv, index = self.gather_prefill_history(kv_cache, index_cache)
        packet = self.runtime.gather_current(
            current_kv, current_index, row_counts=self.row_counts
        )
        placement = self.placement
        block_rows = (
            self.page_size if placement is None else placement.block_granularity
        )
        virtual_blocks = (
            self.runtime.local_rows // self.page_size
            if placement is None
            else placement.virtual_block_count
        )
        self.runtime.store_current(
            packet,
            kv,
            index,
            kv_cache,
            index_cache,
            slots,
            self.current_direct,
            mapped_slots=self.patch_slots,
            history_rows=self.history_rows,
            send_rows=self.send_rows,
            block_rows=block_rows,
            virtual_blocks=virtual_blocks,
        )
        return kv, index
