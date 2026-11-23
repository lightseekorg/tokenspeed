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

"""Ascend QCP Prefill metadata and per-layer KV transport from KVP owners."""

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.attention.ascend_qcp_prefill_workspace import (
    AscendQCPPrefillWorkspaceLayout,
)

from tokenspeed.runtime.execution.query_shard import scatter_count


def prepare_qcp_prefill_gather_metadata(
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
    """Plan the history gather without reading KV or issuing collectives.

    Return the compact workspace page table, this rank's owner-local source
    slots, and each rank's send row count. Pages cover prefix + current tokens;
    prepare_layer_kv overwrites the current slots with this layer's projections.
    """
    if table_cpu is None or table_cpu.device.type != "cpu":
        raise ValueError("Prefill requires the scheduler's host page table")
    if layout.null_rows % page_size or len(extends) != table_cpu.shape[0]:
        raise ValueError("Prefill page and kernel table geometry disagree")
    per_block = 1 if placement is None else placement.block_granularity // page_size
    pages = table_cpu.tolist()
    # Group physical pages by KVP owner and deduplicate shared prefix pages.
    # Each entry maps the scheduler's global page to its owner's local page.
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
    # History receives directly into the attention planes, after the null rows.
    # Rank-ordered offsets match the later uneven all-gather's split sizes.
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
    # Only metadata moves to the device here; per-layer cache contents are
    # read later by runtime.gather_history using these owner-local slots.
    source_pages = torch.tensor(tuple(owners[rank].values()), dtype=torch.int64)
    slots = (source_pages[:, None] * page_size + torch.arange(page_size)).reshape(-1)
    return (
        packed.to(device=device),
        slots.to(device=device),
        tuple(len(owner) * page_size for owner in owners),
    )


class AscendQCPPrefillBackend:
    """Bind one recipe-sized Prefill workspace to a paged backend's cache pool."""

    @property
    def prefill_enabled(self):
        return self._prefill_layout is not None

    @property
    def prefill_needs_host_page_table(self):
        return self.prefill_enabled

    @property
    def prefill_kv_capacity(self):
        return self._prefill_layout.history_capacity if self.prefill_enabled else 0

    def preallocate_prefill_workspace(self, pool_options=None):
        if self._prefill is not None:
            return self._prefill.runtime.workspace_bytes
        if pool_options is None:
            return 0
        if not isinstance(pool_options, AscendQCPPrefillWorkspaceLayout):
            raise TypeError("Ascend Prefill requires a recipe workspace layout")
        if self.cache_pool is None:
            raise RuntimeError("Bind the cache before preparing Prefill")
        layer_id = min(self.cache_pool.history_group_by_layer())
        self._prefill = self._create_qcp_prefill(pool_options, layer_id)
        self._prefill_layout = pool_options
        return self._prefill.runtime.workspace_bytes


def prefill_history_group(ranks):
    """Create the cache-owner group on every participating rank before forward."""
    if len(ranks) == 1:
        return None
    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager,
    )

    return process_group_manager.get_dedicated_device_group(ranks, "prefill-history")


class AscendQCPPrefill:
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
        self.key_lengths = ()
        self.history_slots = None
        self.history_counts = ()
        self.history_rows = self.current_rank_stride = 0

    def prepare_batch_metadata(
        self, metadata, placement, *, page_table_cpu, query_partition=None
    ):
        """Prepare one batch's QCP addresses and lengths for all attention layers.

        Scheduler lengths/pages -> local query span -> compact gather addresses
        -> attention metadata and current-row patch locations. No KV is moved
        here; prepare_layer_kv supplies each layer's cache to the shared runtime.
        """
        degree, rank = self.degree, self.rank
        layout, page_size, topk = self.layout, self.page_size, self.topk
        local_cache_pages = self.runtime.local_rows // page_size
        # 1. Read the scheduler's per-request current/prefix lengths. Host
        # copies and the host page table avoid a device readback during planning.
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
        # 2. Partition the flattened current tokens across QCP producers.
        # first/count select this rank's queries; counts also sizes current KV
        # all-gather. MLA may override the query span while keeping that exchange.
        rows = sum(extends)
        counts = scatter_count(rows, degree)
        first, count = sum(counts[:rank]), counts[rank]
        if query_partition is not None:
            first, count = query_partition
            if first < 0 or count < 0 or first + count > rows:
                raise ValueError("Prefill query partition exceeds current rows")
        # Each producer's current region in the attention planes starts at
        # history_rows + rank * current_rank_stride. This 128-row-aligned
        # stride is distinct from layout.gather_send_rows (buffer capacity).
        current_rank_stride = ((rows + degree * 128 - 1) // (degree * 128)) * 128
        offset = 0
        key_lengths = []
        for prefix, extent in zip(prefixes, extends, strict=True):
            if (
                count
                and offset < first + count
                and offset + extent > first
                and extent > 0
            ):
                key_lengths.append(prefix + min(extent, first + count - offset))
            offset += extent
        # 3. Plan the batch's compact history region and owner-local reads.
        # The page set includes current cache pages; their new values arrive
        # during prepare_layer_kv. Capacity is checked before building addresses.
        table = metadata.page_table
        per_block = 1 if placement is None else placement.block_granularity // page_size
        (
            history_table,
            history_slots,
            history_counts,
        ) = prepare_qcp_prefill_gather_metadata(
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
        # 4. Build attention metadata and current-row destinations. The CPU
        # branch is the reference; native and bitwise NPU paths use the same
        # device metadata kernel in the else branch.
        if table.device.type == "cpu":
            mapped = history_table.clone()
            cumulative = metadata.cum_extend_seq_lens.long()
            prefix = metadata.extend_prefix_lens.long()
            direct = torch.zeros(rows, dtype=torch.uint8)
            base, extra = divmod(rows, degree)
            # 4a. Complete, producer-aligned pages can point directly into the
            # current region. Other pages keep their compact history address;
            # store_current patches their new rows at patch_slots.
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
                tail_page = (
                    history_pages + (owner * current_rank_stride + offset) // page_size
                )
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
            # 4b. Derive each request's Q count and causal KV extent for this
            # shard. qcu/kcu are cumulative boundaries consumed by Indexer.
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
            # SFA uses a request index and min(visible KV, topk) per local query.
            token = torch.arange(first, first + count)
            sequence = torch.searchsorted(cumulative[1:], token, right=True)
            scan = (prefix[sequence] + token - cumulative[sequence] + 1).clamp_max(topk)
            sequence, scan = sequence.int(), scan.int()
            # 4c. Keep only requests with queries on this rank. Compact their
            # table rows and Q/K boundaries together, then remap query_seq.
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
                ascend_qcp_prefill_cache_metadata,
            )

            (
                mapped,
                qcu,
                kcu,
                sequence,
                scan,
                direct,
            ) = ascend_qcp_prefill_cache_metadata(
                table,
                metadata.cum_extend_seq_lens,
                metadata.extend_prefix_lens,
                first=first,
                rows=count,
                total=rows,
                per_block=per_block,
                degree=degree,
                local_pages=local_cache_pages,
                current_rank_stride=current_rank_stride,
                page_size=page_size,
                topk=topk,
                active_requests=len(key_lengths),
                history_table=history_table,
                patch_slots=patch_slots,
                history_pages=history_pages,
            )
        # 5. Publish batch metadata, reused unchanged by successive layers.
        # Placement and row counts drive owner writes and current all-gather.
        self.placement = placement
        self.row_counts = counts
        self.row_start, self.row_end = first, first + count
        # Indexer reads page_table/query_cu/key_cu; SFA reads the same table
        # plus query_seq/scan_count. These addresses refer to workspace planes.
        self.page_table = mapped
        self.query_cu, self.key_cu = qcu, kcu
        self.query_seq, self.scan_count = sequence, scan
        # Per-layer transport uses these to gather history, place current rows
        # and patch partial pages before attention reads the shared planes.
        self.current_direct, self.patch_slots = direct, patch_slots
        self.has_prefix_history = any(prefixes)
        self.key_lengths = tuple(key_lengths)
        self.key_row_count = sum(key_lengths)
        self.history_slots, self.history_counts = history_slots, history_counts
        self.history_rows = layout.null_rows + sum(history_counts)
        self.current_rank_stride = current_rank_stride

    def gather_layer_history(self, kv_cache, index_cache):
        """Return this layer's gathered history in the shared workspace."""
        return self.runtime.gather_history(
            kv_cache,
            index_cache,
            total_rows=sum(self.row_counts),
            has_history=self.has_prefix_history,
            history_slots=self.history_slots,
            history_counts=self.history_counts,
        )

    def prepare_layer_kv(self, kv_cache, index_cache, current_kv, current_index, slots):
        """Fill shared QCP attention planes and persist this layer's current KV.

        The caller selects kv_cache/index_cache by layer_id. All layers reuse
        batch metadata and workspace addresses, but supply different cache data.
        """
        # 1. Receive owner-local history directly into the attention planes.
        kv, index = self.gather_layer_history(kv_cache, index_cache)
        # 2. Reuse the send buffer for current projections; only this exchange
        # uses the packed current receive buffer.
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
        # 3. Unpack into current regions, patch partial history pages, and write
        # each new token to its KVP owner's persistent cache before attention.
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
            current_rank_stride=self.current_rank_stride,
            block_rows=block_rows,
            virtual_blocks=virtual_blocks,
        )
        return kv, index
