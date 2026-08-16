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

"""Shared cache-group machinery for attention backends.

A group-aware backend (``uses_cache_groups = True``) receives the
scheduler's per-group block tables (``block_tables: dict[group_id,
[bs, max_pages]]``), expands them to kernel page tables, and must route
every cache read and write through the layer's own group. This
mixin holds the group-selection,
write-location, and CUDA-graph per-group buffer machinery shared by the MHA
and TRT-LLM backends; model/kernel-specific constraints (spec decode, DFLASH)
stay in the backends.

Table contract (canonical): rows are requests (padded rows carry the
zero-init dummy page 0), column tails pad with -1 and are never read past
``cache_seqlens``; SWA holes sit only at the window front and are written as
the null page 0 by the scheduler export.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import torch
from tokenspeed_kernel.ops.kvcache.triton import (
    compute_group_decode_locs,
    unpack_group_tables,
)

from tokenspeed.runtime.layers.attention.kv_cache.recipes.cache_runtime import (
    cache_debug_enabled,
)
from tokenspeed.runtime.layers.attention.page_table import expand_page_table
from tokenspeed.runtime.utils import get_colorful_logger
from tokenspeed.runtime.utils.common import ceil_div

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.attention.kv_cache.base import CachePool

logger = get_colorful_logger(__name__)


class CacheGroupsMixin:
    """Per-group table/write-loc selection + CUDA-graph buffer discipline.

    Host class requirements: ``self.device``, ``self.kernel_page_size``,
    ``self.max_num_pages``, ``self.forward_decode_metadata`` (with
    ``page_tables``/``out_cache_locs`` fields), and calling
    :meth:`_init_group_graph_buffers` from ``init_cuda_graph_state``.
    """

    cache_consumer_families = frozenset({"history"})

    # family="state" group ids (GDN/mamba state blocks); learned from the
    # pool's specs in set_cache_pool, shed from every table here.
    state_group_ids: frozenset[str] = frozenset()

    # Wrapper-owned (Inkling conv) groups: mixin skips their write-loc math and capture buffers
    engine_owned_group_ids: frozenset[str] = frozenset()

    # Value for CUDA-graph buffer column tails past this replay's table
    # width. -1 is a debug tripwire (never read past cache_seqlens by the
    # MHA kernels); backends whose kernels assume a full-width table
    # (trtllm: row stride derived from max_kv_len) override with 0, the
    # zero-init dummy page — always safe to dereference.
    table_tail_pad: int = -1

    # Replay fill pads dummy rows itself, so callers may pass UNPADDED tables (no per-step F.pad)
    tables_self_padding: bool = True

    # ------------------------------------------------------------------
    # Group selection
    # ------------------------------------------------------------------

    @staticmethod
    def _select_group_entry(layer, mapping, what: str):
        """Pick this layer's entry from a per-group dict (page tables or
        write locs): the layer's group entry, or the sole entry when the
        layer carries no/unknown group id. The sole-entry fallback supports
        ordinary single-group attention pools.
        """
        group_id = getattr(layer, "group_id", "")
        if not group_id or group_id not in mapping:
            if len(mapping) == 1:
                return next(iter(mapping.values()))
            raise KeyError(
                f"{what}: layer group_id={group_id!r} not in cache group "
                f"keys {sorted(mapping)}"
            )
        return mapping[group_id]

    def _select_page_table(self, layer, metadata):
        if metadata.page_tables is None:
            return metadata.page_table
        return self._select_group_entry(layer, metadata.page_tables, "page table")

    def _select_out_cache_loc(
        self, layer, metadata, out_cache_loc, prefer_caller=False
    ):
        # prefer_caller: draft chains own per-step locs; metadata's single loc would pin every step to one slot.
        if metadata.out_cache_locs is None or prefer_caller:
            return out_cache_loc
        return self._select_group_entry(
            layer, metadata.out_cache_locs, "cache write locations"
        )

    @staticmethod
    def _trim_kv_to_locs(out_cache_loc, k, v):
        """Slice a padded KV write down to the write-loc count.

        Prefill-graph replay pads k/v rows to the bucket while per-group
        locs cover only the real (leading) rows. Trimming beats padding the
        locs with the null page: backends that don't scrub tail rows (trtllm)
        would write garbage into page 0, breaking its stays-zero invariant.
        No-op off the padded path and for backends without grouped locations.
        """
        n = out_cache_loc.shape[0]
        if k is not None and k.shape[0] > n:
            return k[:n], v[:n]
        return k, v

    def _prewrite_metadata(self, forward_mode):
        """Metadata slot the fused prewrite writes against. Default: the
        decode slot (MHA gates prewrite to decode); backends that prewrite
        on extend too (trtllm) override to pick their extend/prefill slot.
        """
        return self.forward_decode_metadata

    def select_out_cache_loc(self, layer, out_cache_loc, forward_mode=None):
        """Per-group write locations for out-of-backend KV writers (fused
        RoPE prewrite): the write must land in the pages this layer's group
        reads, never the scheduler's single-table locations. Draft chains own
        their per-step locations, so they must keep the caller-provided tensor.
        """
        metadata = self._prewrite_metadata(forward_mode)
        if metadata is None or metadata.out_cache_locs is None:
            return out_cache_loc
        return self._select_out_cache_loc(
            layer,
            metadata,
            out_cache_loc,
            prefer_caller=self.is_draft,
        )

    def _shed_state_groups(self, tables):
        """Drop family="state" groups (GDN/mamba state blocks, consumed by the
        mamba backend): computing write locs / capture buffers over the
        hole-heavy state table writes the dummy page and trips
        TOKENSPEED_CACHE_DEBUG. Returns None when nothing is left.
        """
        if not tables:
            return None
        skip = self.state_group_ids | self.engine_owned_group_ids
        if skip:
            tables = {gid: table for gid, table in tables.items() if gid not in skip}
        return tables or None

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Empty until the pool arrives; set_cache_pool runs before any
        # metadata init, on both the eager and the CUDA-graph path.
        self.state_group_ids: frozenset[str] = frozenset()
        self.group_block_granularities: dict[str, int] = {}
        self.cache_pool: CachePool | None = None

    def set_cache_pool(self, cache_pool: CachePool) -> None:
        """Bind the pool and learn its groups in one step.

        The arena's published specs are the only source of group geometry, so
        binding is also when learning happens -- no second seeding path that
        could answer differently on the eager arm.
        """
        super().set_cache_pool(cache_pool)
        self._learn_cache_groups(cache_pool.arena.cache_group_specs)

    def _learn_cache_groups(self, cache_group_specs) -> None:
        """Record the pool's family="state" group ids (see state_group_ids)
        and per-group CacheBlock spans (heterogeneous block sizes).

        Spans are read as ``block_granularity``, the shape-agnostic name: for
        the row-geometry groups kept here it equals ``page_size`` by
        definition, and asking a snapshot-state group for ``page_size`` is a
        TypeError. State groups carry no span because they are shed before any
        page-table or write-location math (_shed_state_groups); their blocks
        hold state snapshots, not kernel pages.
        """
        self.state_group_ids = frozenset(
            str(spec.group_id) for spec in cache_group_specs if spec.family == "state"
        )
        self.group_block_granularities = {
            str(spec.group_id): spec.block_granularity
            for spec in cache_group_specs
            if spec.family != "state"
        }

    def _group_block_granularity(self, gid: str) -> int:
        return self.group_block_granularities.get(gid, self.kernel_page_size)

    def _layer_page_size(self, layer) -> int:
        """Page size of the layer's cache group (uniform when unknown)."""
        return self._group_block_granularity(getattr(layer, "group_id", ""))

    def _kernel_page_tables(self, page_tables):
        """Convert per-group page IDs to the page size consumed by the kernel.

        The mixin owns the conversion end to end: the group's page size was
        learned from the pool's specs (``_learn_cache_groups``), so no pool
        round-trip is needed. Identity when every group already matches its
        consumer page size (plain MHA).
        """
        if not page_tables:
            return page_tables
        if all(
            self._group_block_granularity(gid) == self._consumer_page_size(gid)
            for gid in page_tables
        ):
            return page_tables
        return {
            gid: expand_page_table(
                table,
                block_granularity=self._group_block_granularity(gid),
                kernel_page_size=self._consumer_page_size(gid),
                max_kernel_pages=self.max_num_pages,
            )
            for gid, table in page_tables.items()
        }

    def _consumer_page_size(self, group_id: str) -> int:
        """Page size used to view a group's cache tensor in this backend."""
        return self.kernel_page_size

    # ------------------------------------------------------------------
    # Write locations
    # ------------------------------------------------------------------

    def _compute_decode_group_out_cache_locs(
        self, page_tables, seq_lens, page_size, num_tokens_per_req=1
    ):
        """Per-group decode write locs, gathered from the group's own read
        table (M-W1). Plain decode writes one token per request at seq_len-1;
        spec verify writes num_tokens_per_req at seq_len-N..seq_len-1,
        flattened token-major per request ([bs*N], single-table verify layout).
        Positions clamp at 0 for graph-padded rows (seq_len 1 < N), which
        dereference the dummy page harmlessly. The tail page is never a hole
        (SWA holes sit only at the window front). ``page_size`` is the base
        size; groups with a heterogeneous page size divide by their own
        granularity.
        """
        n = num_tokens_per_req
        if n == 1:
            pos = (seq_lens - 1).to(torch.int64)
        else:
            steps = torch.arange(n, device=seq_lens.device, dtype=torch.int64)
            pos = (seq_lens.to(torch.int64).unsqueeze(1) - n + steps).clamp_min(0)
            pos = pos.reshape(-1)
        out = {}
        for gid, table in page_tables.items():
            ps = self._group_block_granularity(gid) if gid else page_size
            page_idx = pos // ps
            off = (pos % ps).to(torch.int32)
            if n == 1:
                pages = table.gather(1, page_idx.unsqueeze(1)).squeeze(1)
            else:
                pages = table.gather(1, page_idx.view(-1, n)).reshape(-1)
            # Mirror the graph-path kernel's clamp: -1 pads/holes route to dummy page 0.
            out[gid] = pages.clamp_min(0) * ps + off
        return out

    def _compute_extend_group_out_cache_locs(
        self, page_tables, extend_prefix_lens_cpu, extend_seq_lens_cpu, page_size
    ):
        """Per-group extend write locs: positions [prefix_len, seq_len) per
        request, flattened in q/k/v token order (cu_extend_seq_lens). Bounds
        come from the CPU mirrors — no per-request GPU sync.
        TODO(cache-perf): batch the per-request loop via repeat_interleave.
        """
        device = next(iter(page_tables.values())).device
        prefix_lens = [int(x) for x in extend_prefix_lens_cpu.tolist()]
        extend_lens = [int(x) for x in extend_seq_lens_cpu.tolist()]
        out = {gid: [] for gid in page_tables}
        for i, (start, num_new) in enumerate(zip(prefix_lens, extend_lens)):
            pos = torch.arange(start, start + num_new, dtype=torch.int64, device=device)
            for gid, table in page_tables.items():
                ps = self._group_block_granularity(gid)
                max_col = (start + num_new - 1) // ps
                if max_col >= table.shape[1]:
                    raise RuntimeError(
                        f"extend write locations out of table bounds: group "
                        f"{gid!r} table {tuple(table.shape)} req={i} "
                        f"prefix={start} new={num_new} page_size={ps} needs "
                        f"col {max_col}"
                    )
                pages = table[i].gather(0, pos // ps)
                out[gid].append(pages * ps + (pos % ps).to(torch.int32))
        return {
            gid: (
                torch.cat(chunks)
                if chunks
                else torch.empty(0, dtype=torch.int32, device=device)
            )
            for gid, chunks in out.items()
        }

    def update_draft_forward_metadata(self, frontier: torch.Tensor) -> None:
        """Re-anchor the k-row decode metadata to the committed frontier:
        seq_lens becomes ``frontier`` and the grouped write locs cover
        positions ``frontier-k..frontier-1``. Accept-dependent, so pure
        tensor ops recomputed per graph replay; the next metadata init
        resets."""
        md = self.forward_decode_metadata
        fields = {"seq_lens": frontier}
        if md.out_cache_locs is not None:
            fields["out_cache_locs"] = self._compute_decode_group_out_cache_locs(
                md.page_tables,
                frontier,
                self.kernel_page_size,
                self.spec_num_tokens,
            )
        self.forward_decode_metadata = replace(md, **fields)

    def _maybe_check_group_write_locs(self, page_tables, out_cache_locs, page_size):
        """TOKENSPEED_CACHE_DEBUG=1 (eager only, GPU sync): write pages must
        be real and inside the group's table. Not for graph-padded batches —
        dummy rows would trip the non-hole assert (see the padding contract
        in _fill_group_graph_buffers).
        """
        if not cache_debug_enabled():
            return
        for gid, locs in out_cache_locs.items():
            pages = (locs // self._group_block_granularity(gid)).to(torch.int32)
            table = page_tables[gid]
            assert (
                pages != 0
            ).all(), f"cache write location in null page 0 for group {gid!r}"
            real = table[table > 0]
            assert torch.isin(
                pages, real
            ).all(), f"cache write pages escape group {gid!r}'s table"

    # ------------------------------------------------------------------
    # CUDA-graph per-group buffers
    # ------------------------------------------------------------------

    def _init_group_graph_buffers(self, max_bs: int) -> None:
        """Reset the persistent per-group buffers; call from
        init_cuda_graph_state BEFORE any backend early return — replay reads
        the dict unconditionally for the stale-table guard.

        Attention-consumed groups get views into ONE stacked table/loc pair
        ([G, max_bs, Wmax] / [G, max_bs * spec_num_tokens]) so the
        replay-time write-loc math ALWAYS runs as a single fused triton
        launch over all groups — the per-group python chains (~4 tiny
        elementwise launches per group per step, the nsys inter-step band)
        are gone, on the spec-verify path too."""
        self.cuda_graph_page_tables: dict[str, torch.Tensor] = {}
        self.cuda_graph_out_cache_locs: dict[str, torch.Tensor] = {}
        self._cuda_graph_max_bs = max_bs
        self._group_locs_stack = None
        self._group_tables_stack = None
        self._group_source_widths = {}
        self._group_grain_ratios = ()
        self._graph_group_ids = []
        self._attention_group_count = 0
        att_gids = sorted(
            gid
            for gid in self.group_block_granularities
            if gid not in self.state_group_ids
            and gid not in self.engine_owned_group_ids
        )
        owned_gids = sorted(
            gid
            for gid in self.group_block_granularities
            if gid in self.engine_owned_group_ids
        )
        gids = att_gids + owned_gids  # attention prefix, wrapper-owned tail
        if not gids:
            return
        source_widths = {
            gid: ceil_div(
                self.max_num_pages * self.kernel_page_size,
                self._group_block_granularity(gid),
            )
            for gid in gids
        }
        consumer_page_sizes = {gid: self._consumer_page_size(gid) for gid in att_gids}
        ratios = {
            gid: (
                self._group_block_granularity(gid) // consumer_page_sizes[gid]
                if gid in att_gids
                else 1
            )
            for gid in gids
        }
        if any(
            ratio <= 0 or self._group_block_granularity(gid) % consumer_page_sizes[gid]
            for gid, ratio in ratios.items()
            if gid in att_gids
        ):
            raise ValueError(
                "cache group page sizes must be positive multiples of their "
                "consumer page sizes"
            )
        widths = {gid: source_widths[gid] * ratios[gid] for gid in gids}
        logger.debug(
            "cache graph buffers: max_num_pages=%d page_size=%d max_bs=%d "
            "source_widths=%s widths=%s",
            self.max_num_pages,
            self.kernel_page_size,
            max_bs,
            source_widths,
            widths,
        )
        wmax = max(widths.values())
        g = len(gids)
        self._graph_group_ids = gids
        self._attention_group_count = len(att_gids)
        self._group_tables_stack = torch.zeros(
            (g, max_bs, wmax), dtype=torch.int32, device=self.device
        )
        # Spec verify: graphs read [max_bs*N] loc views of the stack, so size it up front
        spec_n = max(int(getattr(self, "spec_num_tokens", 1) or 1), 1)
        self._group_locs_stack = torch.zeros(
            (len(att_gids), max_bs * spec_n), dtype=torch.int32, device=self.device
        )
        self._group_source_widths = source_widths
        self._group_grain_ratios = tuple(ratios[gid] for gid in gids)
        self._group_block_granularities_tensor = torch.tensor(
            [consumer_page_sizes[gid] for gid in att_gids],
            dtype=torch.int32,
            device=self.device,
        )
        self._unpack_metadata_device = torch.zeros(
            (g, 3), dtype=torch.int32, device=self.device
        )
        for i, gid in enumerate(gids):
            self.cuda_graph_page_tables[gid] = self._group_tables_stack[
                i, :, : widths[gid]
            ]
            if i < len(att_gids):
                self.cuda_graph_out_cache_locs[gid] = self._group_locs_stack[i]

    def _capture_group_views(self, bs: int, cache_group_ids, tokens_per_req: int = 1):
        """Capture-time (page_tables, out_cache_locs) per-group views into the
        persistent buffers initialized by :meth:`_init_group_graph_buffers`.
        Real tables only arrive at replay, which copies fresh data to these
        graph-recorded addresses.
        Verify (tokens_per_req = spec_num_tokens) keeps [bs]-row tables but
        records [bs*N] write-loc views (token-major, single-table verify layout).
        Returns (None, None) when only state groups (or none) are delivered.
        """
        if not cache_group_ids:
            return None, None
        page_tables = {}
        out_cache_locs = {}
        for gid in cache_group_ids:
            if gid in self.state_group_ids:
                # State pages ride to the mamba backend; no buffers here.
                continue
            if gid in self.engine_owned_group_ids:
                # Engine-owned (conv) group: the wrapper keeps its own capture buffers
                continue
            buf = self.cuda_graph_page_tables.get(gid)
            if buf is None:
                # Replay write locs are ALWAYS the fused triton launch over
                # the stacked buffers; a group outside the stack could never
                # get its locs filled. Every capture-visible group must be
                # known (set_cache_pool) before init_cuda_graph_state.
                raise RuntimeError(
                    f"cache group {gid!r} is not in the stacked CUDA-graph "
                    f"buffers (stack: {self._graph_group_ids}); declare every "
                    "capture-visible group's page size before graph init."
                )
            loc_buf = self.cuda_graph_out_cache_locs.get(gid)
            need = self._cuda_graph_max_bs * tokens_per_req
            if loc_buf is None or loc_buf.shape[0] < need:
                raise RuntimeError(
                    f"location stack too small for group {gid!r}: capture "
                    f"needs {need} rows, have "
                    f"{0 if loc_buf is None else loc_buf.shape[0]}; the "
                    "stack is sized max_bs * spec_num_tokens at init, so "
                    f"tokens_per_req={tokens_per_req} must not exceed "
                    f"spec_num_tokens={getattr(self, 'spec_num_tokens', 1)}."
                )
            page_tables[gid] = buf[:bs, :]
            out_cache_locs[gid] = loc_buf[: bs * tokens_per_req]
        if not page_tables:
            # Only state groups delivered: nothing for this backend.
            return None, None
        return page_tables, out_cache_locs

    def _try_packed_group_unpack(self, bs: int, block_tables) -> bool:
        """One-launch fill of the stacked graph tables from the bridge's
        packed upload. Requires the stack to cover every delivered
        non-state group and all sources to share one storage (the packed
        bridge guarantees both); returns False to take the per-group
        fallback otherwise."""
        stack = self._group_tables_stack
        if stack is None:
            return False
        gids = self._graph_group_ids
        # Fresh pinned alloc each step: a persistent pinned buffer would race with overlap scheduling
        meta = torch.empty((len(gids), 3), dtype=torch.int32, pin_memory=True)
        base_ptr = None
        actual = None
        for i, gid in enumerate(gids):
            src = block_tables.get(gid)
            if src is None or src.shape[1] == 0:
                return False
            ptr = src.untyped_storage().data_ptr()
            if base_ptr is None:
                base_ptr = ptr
                actual = src.shape[0]
            elif ptr != base_ptr or src.shape[0] != actual:
                return False
            meta[i, 0] = src.storage_offset()
            meta[i, 1] = src.shape[1]
            meta[i, 2] = self._group_grain_ratios[i]
        if base_ptr is None:
            return False
        self._unpack_metadata_device.copy_(meta, non_blocking=True)
        src0 = block_tables[gids[0]]
        packed = torch.as_strided(
            src0,
            (src0.untyped_storage().nbytes() // 4,),
            (1,),
            storage_offset=0,
        )
        unpack_group_tables(
            packed,
            self._unpack_metadata_device,
            stack,
            bs,
            actual_bs=min(actual, bs),
            tail_pad=self.table_tail_pad,
        )
        return True

    def _replay_stale_guard(self, bs: int, block_tables) -> None:
        """Fail loudly instead of replaying over stale/zero page tables.
        bs == 0 may skip: col-0 buffer entries stay valid (never -1),
        outputs are discarded, and only unit tests reach it."""
        if not self.cuda_graph_page_tables or bs <= 0:
            return
        name = type(self).__name__
        if not block_tables:
            raise RuntimeError(
                f"{name} replay: per-group CUDA-graph buffers "
                f"exist for groups "
                f"{sorted(self.cuda_graph_page_tables)} "
                f"but block_tables is missing/empty at bs={bs}; the "
                "captured graph would read stale page tables."
            )
        missing = set(self.cuda_graph_page_tables) - set(block_tables)
        if missing:
            raise RuntimeError(
                f"{name} replay: block_tables at bs="
                f"{bs} is missing captured groups {sorted(missing)} "
                f"(delivered: {sorted(block_tables)}); the captured "
                "graph would read stale page tables for those groups."
            )

    def _fill_group_graph_buffers(
        self, bs: int, block_tables, seq_lens, tokens_per_req: int = 1
    ) -> None:
        """Copy this replay's tables into the captured buffers and recompute
        the per-group write locs from the live seq_lens (tokens_per_req locs
        per request on the spec-verify path).

        Padding contract (canonical; bs is the padded bs): dummy ROWS pad
        with 0 — replayed at seq_lens=1 they dereference exactly col 0,
        the zero-init dummy page. Column tails pad with -1, never read
        past cache_seqlens.
        """
        if (
            self._group_locs_stack is None
            or self._group_locs_stack.shape[1] < bs * tokens_per_req
        ):
            raise RuntimeError(
                "replay write locations need the stacked location buffer "
                f"(bs={bs}, tokens_per_req={tokens_per_req}, stack="
                f"{None if self._group_locs_stack is None else tuple(self._group_locs_stack.shape)}); "
                "the stack is sized max_bs * spec_num_tokens at graph init "
                "and there is no python fallback."
            )
        self._packed_group_unpack_ran = self._try_packed_group_unpack(bs, block_tables)
        if not self._packed_group_unpack_ran:
            for i, gid in enumerate(self._graph_group_ids):
                src = block_tables.get(gid)
                if src is None:
                    continue
                if gid in self.engine_owned_group_ids:
                    # The wrapper fills its own scheduler-page buffer.
                    continue
                buf = self.cuda_graph_page_tables[gid]
                # Clamp: scheduler may send extra reservation columns; kernels never read past cache_seqlens
                source_cols = min(src.shape[1], self._group_source_widths[gid])
                # cols >= 1: a zero-width table would leave dummy rows' col 0 unwritten
                assert source_cols >= 1, f"table for group {gid!r}: zero-width table"
                rows = min(src.shape[0], bs)
                ratio = self._group_grain_ratios[i]
                if ratio != 1:
                    expand_page_table(
                        src[:rows, :source_cols],
                        block_granularity=self._group_block_granularity(gid),
                        kernel_page_size=self._consumer_page_size(gid),
                        max_kernel_pages=buf.shape[1],
                        out=buf[:rows],
                    )
                    if rows < bs:
                        buf[rows:bs].zero_()
                    continue
                cols = min(source_cols, buf.shape[1])
                buf[:rows, :cols].copy_(src[:rows, :cols])
                if cols < buf.shape[1]:
                    buf[:rows, cols:].fill_(self.table_tail_pad)
                if rows < bs:
                    # Dummy rows pad with 0 (the zero-init dummy page).
                    buf[rows:bs].fill_(0)

        # One fused launch writes every group's locs into the stacked buffer the graphs read
        compute_group_decode_locs(
            self._group_tables_stack[: self._attention_group_count],
            self._group_block_granularities_tensor,
            seq_lens[:bs],
            self._group_locs_stack,
            bs,
            tokens_per_req,
        )
