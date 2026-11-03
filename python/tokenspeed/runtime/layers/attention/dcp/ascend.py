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

"""Ascend collectives shared by dense MLA and sparse DSA.

Native Prefill reuses scheduler placement and page tables.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

import torch
import torch.distributed as dist
from tokenspeed_kernel.ops.attention.dsa.ascend import ascend_dsa_cp_metadata

from tokenspeed.runtime.execution.workspace import WorkspacePool
from tokenspeed.runtime.layers.attention.dcp.placement import resolve_cache_slots


def _prefill_workspace_sizes(
    *, max_current_rows, history_capacity_rows, degree, index_width
):
    if history_capacity_rows < 1 or degree < 1 or index_width not in (0, 128):
        raise ValueError("Invalid native Prefill workspace geometry")
    send_rows = -(-max_current_rows // (degree * 128)) * 128
    history_rows = history_capacity_rows
    width = 576 + index_width
    return (
        (history_rows + degree * send_rows) * width,
        send_rows * width,
        history_rows * width,
    )


@dataclass(frozen=True)
class AscendPrefillOptions:
    """A model's explicit native Prefill choice and bounded runtime capacity."""

    max_current_rows: int
    max_kv_page_num: int
    kv_page_size: int = 128

    def __post_init__(self):
        if (
            self.max_current_rows < 1
            or self.max_kv_page_num < 1
            or self.kv_page_size < 1
            or self.max_current_rows > self.max_kv_page_num * self.kv_page_size
        ):
            raise ValueError(
                "Native Prefill needs a positive page limit, "
                "with current rows bounded by the page limit"
            )

    @property
    def history_capacity_rows(self):
        # One null page precedes the compact per-batch pages.
        return (self.max_kv_page_num + 1) * self.kv_page_size

    def required_workspace(self, *, local_rows: int, degree: int, index_width: int):
        """Transport buffers and commit slots, including pool alignment."""
        if local_rows < 1 or degree < 1 or index_width not in (0, 128):
            raise ValueError("Invalid native Prefill workspace geometry")
        return (
            2
            * sum(
                _prefill_workspace_sizes(
                    max_current_rows=self.max_current_rows,
                    history_capacity_rows=self.history_capacity_rows,
                    degree=degree,
                    index_width=index_width,
                )
            )
            + self.slot_workspace_rows * torch.int64.itemsize
        )

    @property
    def slot_workspace_rows(self):
        # WorkspacePool aligns each view to 256 bytes (32 int64 slots).
        return -(-self.max_current_rows // 32) * 32

    def allocate_workspace(self, *, local_rows, degree, index_width, device):
        """Own transport and commit views together, disjoint across streams."""
        required = self.required_workspace(
            local_rows=local_rows, degree=degree, index_width=index_width
        )
        sizes = _prefill_workspace_sizes(
            max_current_rows=self.max_current_rows,
            history_capacity_rows=self.history_capacity_rows,
            degree=degree,
            index_width=index_width,
        )
        pool = WorkspacePool(device, initial_nbytes=required)
        buffers = pool.allocate(
            *(((size,), torch.bfloat16) for size in sizes),
            ((self.slot_workspace_rows,), torch.int64),
        )
        pool.freeze()
        return pool, tuple(buffers)


class AscendPrefillCache:
    """Opt a model cache view into the native Prefill hooks it implements."""

    def __init__(self, *, prefill_options: AscendPrefillOptions | None, **kwargs):
        self.prefill_options = prefill_options
        super().__init__(**kwargs)


@dataclass(frozen=True)
class AscendPrefillGatherPlan:
    """Compact batch pages from the scheduler's existing host table, once per batch."""

    table: torch.Tensor
    send_slots: torch.Tensor
    row_counts: tuple[int, ...]

    @classmethod
    def build(
        cls,
        table_cpu,
        extends,
        prefixes,
        *,
        placement,
        page_size,
        local_pages,
        degree,
        rank,
        options,
        device,
        columns,
    ):
        if table_cpu is None or table_cpu.device.type != "cpu":
            raise ValueError(
                "Bounded native Prefill requires the scheduler's host page table"
            )
        if options.kv_page_size % page_size or len(extends) != table_cpu.shape[0]:
            raise ValueError("Prefill page budget and kernel table geometry disagree")
        total_pages = sum(
            (prefix + extent + options.kv_page_size - 1) // options.kv_page_size
            for prefix, extent in zip(prefixes, extends, strict=True)
        )
        if total_pages > options.max_kv_page_num:
            raise ValueError(
                f"Prefill batch needs {total_pages} KV pages, limit is {options.max_kv_page_num}"
            )
        per_block = 1 if placement is None else placement.block_granularity // page_size
        pages = table_cpu.tolist()
        owners = [{} for _ in range(degree)]
        for req, (prefix, extent) in enumerate(zip(prefixes, extends, strict=True)):
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
        offsets, offset = [], options.kv_page_size // page_size
        for owner in owners:
            offsets.append(offset)
            offset += len(owner)
        if offset * page_size > options.history_capacity_rows:
            raise ValueError("Compacted Prefill pages exceed the receive workspace")
        mapping = {
            page: offsets[owner] + i
            for owner, entries in enumerate(owners)
            for i, page in enumerate(entries)
        }
        packed = torch.zeros((len(extends), columns), dtype=table_cpu.dtype)
        for req, (prefix, extent) in enumerate(zip(prefixes, extends, strict=True)):
            count = (prefix + extent + page_size - 1) // page_size
            packed[req, :count] = torch.tensor(
                [mapping.get(page, 0) for page in pages[req][:count]],
                dtype=packed.dtype,
            )
        source_pages = torch.tensor(tuple(owners[rank].values()), dtype=torch.int64)
        slots = (source_pages[:, None] * page_size + torch.arange(page_size)).reshape(
            -1
        )
        return cls(
            packed.to(device=device),
            slots.to(device=device),
            tuple(len(owner) * page_size for owner in owners),
        )


@dataclass(frozen=True)
class AscendPrefillPlan:
    """Receive-page addresses, causal spans and current-row placement."""

    row_counts: list[int]
    row_start: int
    row_end: int
    page_table: torch.Tensor
    index_table: torch.Tensor
    query_cu: torch.Tensor
    key_cu: torch.Tensor
    query_seq: torch.Tensor
    scan_count: torch.Tensor
    current_direct: torch.Tensor
    cold: bool
    gather: AscendPrefillGatherPlan | None = None
    patch_slots: torch.Tensor | None = None

    def patch(
        self,
        runtime,
        layer_id,
        slots,
        *,
        slots_workspace,
        width,
        block_rows,
        virtual_blocks,
    ):
        """Commit on the compute stream using the backend-owned slots."""
        return runtime.patch(
            layer_id,
            slots,
            self.current_direct,
            slots_workspace=slots_workspace,
            mapped_slots=self.patch_slots,
            width=width,
            block_rows=block_rows,
            virtual_blocks=virtual_blocks,
        )

    @classmethod
    def build(
        cls,
        metadata,
        placement,
        *,
        degree,
        rank,
        page_size,
        topk,
        local_cache_pages,
        query_partition=None,
        page_table_cpu=None,
        options=None,
    ):
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
        counts = [rows // degree + (r < rows % degree) for r in range(degree)]
        first, count = sum(counts[:rank]), counts[rank]
        if query_partition is not None:
            first, count = query_partition
            if first < 0 or count < 0 or first + count > rows:
                raise ValueError("Prefill query partition exceeds current rows")
        # The shared receive transport pads each rank's contribution to 128
        # rows, including MLA readers with 64-token kernel pages.
        send_rows = ((rows + degree * 128 - 1) // (degree * 128)) * 128
        offset, active = 0, 0
        for extent in extends:
            active += offset < first + count and offset + extent > first and extent > 0
            offset += extent
        table = metadata.page_table
        per_block = 1 if placement is None else placement.block_granularity // page_size
        gather = None
        patch_slots = None
        history_pages = degree * local_cache_pages
        if options is not None and options.max_kv_page_num:
            gather = AscendPrefillGatherPlan.build(
                table if table.device.type == "cpu" else page_table_cpu,
                extends,
                prefixes,
                placement=placement,
                page_size=page_size,
                local_pages=local_cache_pages,
                degree=degree,
                rank=rank,
                options=options,
                device=table.device,
                columns=table.shape[1],
            )
            history_pages = options.history_capacity_rows // page_size
            patch_slots = torch.empty(rows, dtype=torch.int64, device=table.device)
        if table.device.type == "cpu":
            pages = table.long()
            block = pages // per_block
            positive = (block - 1).clamp_min(0)
            local = (positive // degree + 1) * per_block + pages % per_block
            mapped = torch.where(
                block > 0, positive % degree * local_cache_pages + local, 0
            ).int()
            if gather is not None:
                mapped = gather.table.clone()
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
                if patch_slots is not None:
                    position = prefix[req] + token - cumulative[req]
                    page = gather.table[req, position // page_size].long()
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
                history_table=None if gather is None else gather.table,
                patch_slots=patch_slots,
                history_pages=history_pages,
            )
        return cls(
            counts,
            first,
            first + count,
            mapped,
            mapped,
            qcu,
            kcu,
            sequence,
            scan,
            direct,
            not any(prefixes),
            gather,
            patch_slots,
        )


class AscendDCPContext:
    """Own group order and partial-result exchange."""

    def __init__(
        self,
        *,
        degree: int,
        rank: int,
        ranks: tuple[int, ...],
        auxiliary_namespace: str | None,
    ):
        if degree < 1 or len(ranks) != degree or len(set(ranks)) != degree:
            raise ValueError("invalid DCP group")
        if not 0 <= rank < degree:
            raise ValueError("invalid DCP rank")
        self.degree, self.rank, self.ranks = degree, rank, ranks
        self.primary_process_group = self.auxiliary_process_group = None
        self.stream_fork = None
        if degree > 1 and dist.is_initialized():
            from tokenspeed.runtime.distributed.process_group_manager import (
                process_group_manager as pg_manager,
            )

            # Import the manager used by the host, without imposing runtime
            # imports on the standalone address/reference tests.
            self.primary_process_group = pg_manager.get_device_process_group(ranks)
            if auxiliary_namespace is None:
                return
            self.auxiliary_process_group = pg_manager.get_dedicated_device_group(
                ranks, auxiliary_namespace
            )
            from tokenspeed.runtime.utils.device_stream import (
                StreamFork,
                new_device_stream,
            )

            self.stream_fork = StreamFork(new_device_stream())

    @property
    def ready(self):
        return self.degree == 1 or self.primary_process_group is not None

    @property
    def has_auxiliary(self):
        return self.auxiliary_process_group is not None

    def _process_group(self, *, auxiliary: bool):
        if not self.ready:
            raise RuntimeError("DCP process group was not configured")
        return (
            self.auxiliary_process_group
            if auxiliary and self.has_auxiliary
            else self.primary_process_group
        )

    def all_gather(self, tensor, *, auxiliary: bool):
        group = self._process_group(auxiliary=auxiliary)
        if self.degree == 1:
            return tensor
        output = torch.empty(
            (self.degree * tensor.shape[0],) + tensor.shape[1:],
            dtype=tensor.dtype,
            device=tensor.device,
        )
        dist.all_gather_into_tensor(output, tensor.contiguous(), group=group)
        return output

    def gather_query(self, query):
        """Q[T,h,D] -> Q[T,P*h,D], heads in group-rank order."""
        if self.degree == 1:
            return query
        gathered = self.all_gather(query, auxiliary=True)
        return (
            gathered.view(self.degree, *query.shape)
            .permute(1, 0, 2, 3)
            .flatten(1, 2)
            .contiguous()
        )

    def gather_cache_rows(
        self,
        cache: torch.Tensor,
        slot_groups: Sequence[torch.Tensor],
        *,
        block_granularity: int,
        virtual_block_count: int,
    ) -> Iterator[torch.Tensor]:
        """Gather the whole local cache plane, then select virtual-slot rows.

        Each owner contributes its entire physical plane, including its null
        block. The temporary result is rank-major; virtual block ``v`` is on
        owner ``(v - 1) % degree`` at local block ``(v - 1) // degree + 1``.
        Every slot group reads that same gathered plane in its original order.
        No gathered rows are retained on the context or written back to cache.
        """
        if not any(slots.numel() for slots in slot_groups):
            return
        if block_granularity <= 0 or virtual_block_count <= 1:
            raise ValueError("cache geometry must include usable blocks")
        local_rows = cache.shape[0]
        local_blocks = (virtual_block_count - 2) // self.degree + 2
        if (
            local_rows % block_granularity
            or local_rows < local_blocks * block_granularity
        ):
            raise ValueError("local cache plane does not cover its virtual blocks")
        # Communicate bytes, preserving BF16 values (including NaN payloads).
        gathered = self.all_gather(
            cache.contiguous().view(torch.uint8), auxiliary=False
        ).view(cache.dtype)
        for slots in slot_groups:
            if slots.ndim != 1 or slots.dtype not in (torch.int32, torch.int64):
                raise ValueError("cache slots must be a one-dimensional integer tensor")
            safe = slots.to(torch.int64).clamp_min(0)
            block = safe // block_granularity
            positive = (block - 1).clamp_min(0)
            owner = positive % self.degree
            local = (
                positive // self.degree + 1
            ) * block_granularity + safe % block_granularity
            valid = (slots >= block_granularity) & (block < virtual_block_count)
            index = torch.where(valid, owner * local_rows + local, 0)
            yield gathered.index_select(0, index)

    def all_to_all(self, output, input_tensor, *, auxiliary: bool):
        group = self._process_group(auxiliary=auxiliary)
        if self.degree == 1:
            output.copy_(input_tensor)
        else:
            dist.all_to_all_single(output, input_tensor.contiguous(), group=group)

    def combine(self, output, lse, *, merge: Callable):
        """Partial O[T,P*h,D], LSE[T,P*h] -> TP-local O[T,h,D]."""
        if output.ndim != 3 or output.shape[:-1] != lse.shape:
            raise ValueError("expected O[T,H,D] and LSE[T,H]")
        tokens, heads, dim = output.shape
        if heads % self.degree:
            raise ValueError("query heads must divide the DCP group")
        local_heads = heads // self.degree
        # Also sanitize the single-rank case and native merge inputs.
        output = torch.where(torch.isneginf(lse)[..., None], 0, output)
        if self.degree == 1:
            return output
        send_o = (
            output.transpose(0, 1)
            .reshape(self.degree, local_heads, tokens, dim)
            .contiguous()
        )
        send_lse = lse.float().T.reshape(self.degree, local_heads, tokens).contiguous()
        recv_o, recv_lse = torch.empty_like(send_o), torch.empty_like(send_lse)
        from tokenspeed.runtime.execution.forward_step import (
            get_is_capture_mode,
            get_is_cuda_graph_phase,
        )
        from tokenspeed.runtime.utils.device_stream import limit_stream_cores

        graph_phase = get_is_cuda_graph_phase()
        fork = self.stream_fork
        if graph_phase and self.has_auxiliary and fork is not None:
            main = fork.device_module.current_stream()
            with (
                limit_stream_cores(main, cube_num=12, vector_num=24, enable=True),
                limit_stream_cores(
                    fork.aux_stream, cube_num=12, vector_num=24, enable=True
                ),
                fork.scope(enable=True, overlap=get_is_capture_mode()) as branch,
            ):
                with branch.branch():
                    self.all_to_all(recv_lse, send_lse, auxiliary=True)
                self.all_to_all(recv_o, send_o, auxiliary=False)
        else:
            self.all_to_all(recv_o, send_o, auxiliary=False)
            self.all_to_all(recv_lse, send_lse, auxiliary=True)
        result = merge(
            recv_o.reshape(self.degree, -1, dim), recv_lse.reshape(self.degree, -1)
        )
        return result.view(local_heads, tokens, dim).transpose(0, 1).contiguous()


@dataclass(frozen=True)
class AscendDCPKernelMetadata:
    page_table: torch.Tensor
    seq_lens: torch.Tensor
    init_counts: torch.Tensor
    local_counts: torch.Tensor


@dataclass(frozen=True)
class AscendIndexerMetadata:
    q_ends: torch.Tensor
    page_table: torch.Tensor
    seq_lens: torch.Tensor
    init_counts: torch.Tensor
    local_counts: torch.Tensor
    visible_lengths: torch.Tensor | None


class AscendDCPMetadata:
    """Layer-shared native page metadata for one context-parallel group."""

    def __init__(
        self,
        *,
        degree: int,
        rank: int,
        ranks: tuple[int, ...],
        virtual_block_count: int | None,
    ) -> None:
        if degree < 1 or len(ranks) != degree or len(set(ranks)) != degree:
            raise ValueError("invalid DCP group")
        if not 0 <= rank < degree:
            raise ValueError("invalid DCP rank")
        self.degree, self.rank, self.ranks = degree, rank, ranks
        self.virtual_block_count = virtual_block_count
        self.page_table: torch.Tensor | None = None
        self.seq_lens: torch.Tensor | None = None
        self.init_counts: torch.Tensor | None = None
        self.local_counts: torch.Tensor | None = None
        self.query_ends: torch.Tensor | None = None
        self.write_slots: torch.Tensor | None = None
        self._metadata_buffers: dict[str, torch.Tensor] = {}
        self._decode_rows_capacity: int | None = None
        self._prepared_metadata_key: tuple | None = None
        self._metadata_sources: tuple[torch.Tensor, ...] = ()
        self._indexer_metadata: AscendIndexerMetadata | None = None
        self._query_group_size: int | None = None
        self._write_slot_source: tuple | None = None

    def clear_forward_metadata(self) -> None:
        """Invalidate layer-shared results before the next forward/capture."""
        self._prepared_metadata_key = None
        self._metadata_sources = ()
        self._indexer_metadata = None
        self._query_group_size = None
        self._write_slot_source = None

    def prepare_metadata(
        self,
        *,
        seq_lens,
        page_table,
        page_size,
        block_granularity,
        init_tokens,
        local_tokens,
        query_width,
        query_group_size,
        slots,
    ) -> None:
        sources = (seq_lens, page_table) + (() if slots is None else (slots,))
        key = (
            tuple(
                (t.data_ptr(), t.shape, t.stride(), t.dtype, t.device) for t in sources
            ),
            page_size,
            block_granularity,
            init_tokens,
            local_tokens,
            query_width,
            query_group_size,
        )
        if key == self._prepared_metadata_key:
            return
        self.refresh_metadata(
            seq_lens=seq_lens,
            page_table=page_table,
            page_size=page_size,
            block_granularity=block_granularity,
            init_tokens=init_tokens,
            local_tokens=local_tokens,
            query_width=query_width,
            query_group_size=query_group_size,
            slots=slots,
        )
        self._metadata_sources = sources
        self._prepared_metadata_key = key

    def prepare_write_slots(self, locations, placement):
        """Map authoritative router slots once, then share them across layers.

        MLA refreshes FIA's host lengths before capture. Its router locations
        arrive at the first Prolog, so this part runs on the model stream and
        is captured once in the graph, just as DSA's Prolog preparation is.
        """
        key = (
            locations.data_ptr(),
            locations.shape,
            locations.stride(),
            locations.dtype,
            locations.device,
            placement,
        )
        if key != self._write_slot_source:
            local, owned = resolve_cache_slots(locations, placement)
            self.write_slots = (
                (local if owned is None else torch.where(owned, local, -1))
                .to(torch.int64)
                .contiguous()
            )
            self._metadata_sources += (locations,)
            self._write_slot_source = key
        return self.write_slots

    def bind_virtual_block_count(self, virtual_block_count: int) -> None:
        """Bind the scheduler capacity published by the cache arena."""
        if virtual_block_count <= 1:
            raise ValueError("Ascend CP cache must contain usable virtual blocks")
        self.clear_forward_metadata()
        if self.virtual_block_count != virtual_block_count:
            self.virtual_block_count = virtual_block_count
            self.page_table = None
            self.seq_lens = None
            self.init_counts = None
            self.local_counts = None

    def allocate_decode_buffers(self, rows: int, columns: int, device) -> None:
        """Keep all decode metadata at stable addresses across batch sizes."""
        self.clear_forward_metadata()
        self._metadata_buffers = {
            name: torch.empty(shape, dtype=torch.int32, device=device)
            for name, shape in (
                ("query_ends", (rows,)),
                ("page_table", (rows, columns)),
                ("seq_lens", (rows,)),
                ("init_counts", (rows,)),
                ("local_counts", (rows,)),
                ("index_q_ends", (rows,)),
                ("index_page_table", (rows, columns)),
                ("index_seq_lens", (rows,)),
                ("index_init_counts", (rows,)),
            )
        }
        self._metadata_buffers["write_slots"] = torch.empty(
            rows, dtype=torch.int64, device=device
        )
        self._decode_rows_capacity = rows

    def _buffer(self, name, shape, device, dtype=torch.int32):
        target = self._metadata_buffers.get(name)
        if (
            target is None
            or target.ndim != len(shape)
            or target.shape[1:] != shape[1:]
            or target.shape[0] < shape[0]
            or target.dtype != dtype
            or target.device != device
        ):
            if self._decode_rows_capacity is not None:
                raise RuntimeError(
                    f"Ascend {name} exceeds preallocated decode geometry"
                )
            target = torch.empty(shape, dtype=dtype, device=device)
            self._metadata_buffers[name] = target
        return target[: shape[0]]

    def refresh_metadata(
        self,
        *,
        seq_lens,
        page_table,
        page_size,
        block_granularity,
        init_tokens,
        local_tokens,
        query_width,
        query_group_size,
        slots,
    ) -> None:
        self.clear_forward_metadata()
        if self.virtual_block_count is None:
            raise RuntimeError("Ascend CP cache metadata was not bound")
        rows, columns = page_table.shape[0] * query_width, page_table.shape[1]
        if query_width < 1 or query_group_size < 1 or rows % query_group_size:
            raise ValueError("DSA query rows must form complete Indexer groups")
        groups = rows // query_group_size
        device = page_table.device
        # Prefill can exceed the decode ladder without replacing captured storage.
        separate = self._decode_rows_capacity is not None and (
            rows > self._decode_rows_capacity
            or columns != self._metadata_buffers["page_table"].shape[1]
        )

        def buffer(name, shape, dtype=torch.int32):
            if separate:
                return torch.empty(shape, dtype=dtype, device=device)
            return self._buffer(name, shape, device, dtype)

        self.page_table = buffer("page_table", (rows, columns))
        self.seq_lens = buffer("seq_lens", (rows,))
        self.init_counts = buffer("init_counts", (rows,))
        self.local_counts = buffer("local_counts", (rows,))
        self.query_ends = buffer("query_ends", (rows,))
        self.write_slots = (
            None if slots is None else buffer("write_slots", slots.shape, torch.int64)
        )
        self._query_group_size = query_group_size
        self._indexer_metadata = AscendIndexerMetadata(
            buffer("index_q_ends", (groups,)),
            buffer("index_page_table", (groups, columns)),
            buffer("index_seq_lens", (groups,)),
            buffer("index_init_counts", (groups,)),
            self.local_counts,
            self.seq_lens if query_group_size > 1 else None,
        )
        index = self._indexer_metadata
        ascend_dsa_cp_metadata(
            page_table,
            seq_lens,
            page_size=page_size,
            block_granularity=block_granularity,
            virtual_block_count=self.virtual_block_count,
            degree=self.degree,
            rank=self.rank,
            initial_tokens=init_tokens,
            window_tokens=local_tokens,
            query_width=query_width,
            query_group_size=query_group_size,
            slots=slots,
            out=self.page_table,
            local_lengths=self.seq_lens,
            initial_counts=self.init_counts,
            window_counts=self.local_counts,
            query_ends=self.query_ends,
            index_table=index.page_table,
            index_lengths=index.seq_lens,
            index_initial_counts=index.init_counts,
            index_query_ends=index.q_ends,
            local_slots=self.write_slots,
        )

    def metadata(self, *, start: int, end: int) -> AscendDCPKernelMetadata:
        values = (self.page_table, self.seq_lens, self.init_counts, self.local_counts)
        if any(value is None for value in values):
            raise RuntimeError("DSA CP metadata was not refreshed")
        page_table, seq_lens, init_counts, local_counts = values
        assert page_table is not None
        assert seq_lens is not None
        assert init_counts is not None
        assert local_counts is not None
        return AscendDCPKernelMetadata(
            page_table=page_table[start:end],
            seq_lens=seq_lens[start:end],
            init_counts=init_counts[start:end],
            local_counts=local_counts[start:end],
        )

    def indexer_metadata(self, q_ends, local, *, query_group_size: int):
        """Return layouts already written by the metadata launch, without copies."""
        if (
            query_group_size != self._query_group_size
            or self._indexer_metadata is None
            or q_ends.numel() != local.seq_lens.numel()
        ):
            raise ValueError("Indexer query layout differs from prepared DSA metadata")
        return self._indexer_metadata
