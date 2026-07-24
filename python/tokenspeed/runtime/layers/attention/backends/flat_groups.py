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

"""Shared flat KV-cache group machinery for attention backends.

A flat-capable backend (``uses_flat_cache_groups = True``) receives one page
table plus one logical-page base vector per cache group
(``flat_block_tables: dict[group_id, [bs, max_pages]]`` and
``flat_block_table_base_offsets: dict[group_id, [bs]]``) instead of the radix
single table, and must route every KV read AND write through the layer's own
group (M-W1). This mixin holds the group-selection, write-location, and
CUDA-graph per-group buffer machinery shared by the MHA and TRT-LLM backends;
model/kernel-specific constraints (spec decode, DFLASH) stay in the backends.

Table contract (canonical): rows are requests; column ``j`` represents
logical page ``base[row] + j``. Padded rows carry dummy page 0 and base 0,
column tails pad with -1 and are never read past ``cache_seqlens``; SWA holes
sit only at the window front and are written as the null page 0 by the
scheduler export. Full-history tables remain backward-compatible by carrying
an explicit all-zero base vector.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import replace
from types import MappingProxyType

import torch
from tokenspeed_kernel.ops.kvcache.triton import (
    flat_decode_locs,
    flat_tables_unpack,
)

from tokenspeed.runtime.configs.paged_cache_spec import PagedCacheGroupSpec
from tokenspeed.runtime.flat_cache_tables import validate_flat_table_base_offsets
from tokenspeed.runtime.utils import get_colorful_logger
from tokenspeed.runtime.utils.common import ceil_div

logger = get_colorful_logger(__name__)


class FlatCacheGroupsMixin:
    """Per-group table/write-loc selection + CUDA-graph buffer discipline.

    Host class requirements: ``self.device``, ``self.page_size``,
    ``self.max_num_pages``, ``self.forward_decode_metadata`` (with
    ``page_tables``/``block_table_base_offsets``/``out_cache_locs`` fields), and calling
    :meth:`_init_flat_graph_buffers` from ``init_cuda_graph_state``.
    """

    # family="state" group ids (GDN/mamba state pages); learned from the
    # pool's specs in init_cuda_graph_state, shed from every table here.
    flat_state_group_ids: frozenset[str] = frozenset()

    # Wrapper-owned (Inkling conv) groups: mixin skips their write-loc math and capture buffers
    flat_engine_owned_group_ids: frozenset[str] = frozenset()

    # Frozen construction-time geometry. PagedCacheGroupSpec.block_size is the
    # sole raw-token page-span authority; the backend never mirrors it into a
    # second mutable page-size mapping.
    flat_group_specs: Mapping[str, PagedCacheGroupSpec] = MappingProxyType({})

    # Groups whose scheduler tables use absolute logical columns.  They retain
    # main's one-launch stacked write-loc hot path.  Bounded/unknown layouts
    # use the paired per-request base buffers below.
    flat_absolute_group_ids: frozenset[str] = frozenset()

    # Draft decode-window lookback rows (Inkling MTP): armed by the conv
    # wrapper's configure_draft_lookback BEFORE graph init, so the lookback
    # loc stack below is sized alongside the main one.
    flat_draft_lookback: int = 0

    # Value for CUDA-graph buffer column tails past this replay's table
    # width. -1 is a debug tripwire (never read past cache_seqlens by the
    # MHA kernels); backends whose kernels assume a full-width table
    # (trtllm: row stride derived from max_kv_len) override with 0, the
    # zero-init dummy page — always safe to dereference.
    flat_table_tail_pad: int = -1

    # Replay fill pads dummy rows itself, so callers may pass UNPADDED tables (no per-step F.pad)
    flat_tables_self_padding: bool = True

    # ------------------------------------------------------------------
    # Group selection
    # ------------------------------------------------------------------

    @staticmethod
    def _select_group_entry(layer, mapping, what: str):
        """Pick this layer's entry from a flat per-group dict (page tables or
        write locs): the layer's group entry, or the sole entry when the
        layer carries no/unknown group id. TODO(radix-removal): collapses to
        `mapping[layer.group_id]` once flat is the only path.
        """
        group_id = getattr(layer, "group_id", "")
        if not group_id or group_id not in mapping:
            if len(mapping) == 1:
                return next(iter(mapping.values()))
            raise KeyError(
                f"{what}: layer group_id={group_id!r} not in flat group "
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
            layer, metadata.out_cache_locs, "flat write locs"
        )

    @staticmethod
    def _trim_kv_to_locs(out_cache_loc, k, v):
        """Slice a padded KV write down to the write-loc count.

        Prefill-graph replay pads k/v rows to the bucket while flat per-group
        locs cover only the real (leading) rows. Trimming beats padding the
        locs with the null page: backends that don't scrub tail rows (trtllm)
        would write garbage into page 0, breaking its stays-zero invariant.
        No-op off the padded path and for backends without flat locs.
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
        reads, never the scheduler's single-table locations.
        """
        metadata = self._prewrite_metadata(forward_mode)
        if metadata is None or metadata.out_cache_locs is None:
            return out_cache_loc
        return self._select_out_cache_loc(layer, metadata, out_cache_loc)

    def _shed_state_groups(self, tables):
        """Drop family="state" groups (GDN/mamba state pages, consumed by the
        mamba backend): computing write locs / capture buffers over the
        hole-heavy state table writes the dummy page and trips
        TOKENSPEED_FLAT_DEBUG. Returns None when nothing is left.
        """
        if not tables:
            return None
        skip = self.flat_state_group_ids | self.flat_engine_owned_group_ids
        if skip:
            tables = {gid: table for gid, table in tables.items() if gid not in skip}
        return tables or None

    def _shed_state_group_inputs(self, page_tables, block_table_base_offsets):
        """Validate the paired ABI, then shed state-family entries together."""
        validate_flat_table_base_offsets(
            page_tables,
            block_table_base_offsets,
            where=f"{type(self).__name__} flat metadata",
        )
        return (
            self._shed_state_groups(page_tables),
            self._shed_state_groups(block_table_base_offsets),
        )

    def _learn_flat_state_groups(self, paged_cache_group_specs) -> None:
        """Record the pool's family="state" group ids (see
        flat_state_group_ids) and per-group page sizes (heterogeneous block
        sizes); called from init_cuda_graph_state, the one place the pool's
        specs reach every backend."""
        self.flat_state_group_ids = frozenset(
            str(spec.group_id)
            for spec in paged_cache_group_specs
            if spec.family == "state"
        )
        self.flat_group_specs = MappingProxyType(
            {
                str(spec.group_id): spec
                for spec in paged_cache_group_specs
                if spec.family != "state"
            }
        )
        self.flat_absolute_group_ids = frozenset(
            str(spec.group_id)
            for spec in paged_cache_group_specs
            if spec.family != "state" and spec.table_layout == "absolute"
        )

    def _group_page_size(self, gid: str) -> int:
        if not self.flat_group_specs and not self.flat_state_group_ids:
            # Radix never publishes flat group specs, even when model layers
            # carry group labels. Once flat specs exist, however, every
            # id must resolve through that immutable geometry view.
            return self.page_size
        spec = self.flat_group_specs.get(gid)
        if spec is None:
            raise RuntimeError(
                f"flat group {gid!r} is absent from the backend cache specs: "
                f"{sorted(self.flat_group_specs)}"
            )
        block_size = spec.block_size
        if block_size is None:
            raise RuntimeError(f"flat group {gid!r} has no canonical block_size")
        return int(block_size)

    def _layer_page_size(self, layer) -> int:
        """Page size of the layer's cache group (uniform when unknown)."""
        return self._group_page_size(getattr(layer, "group_id", ""))

    # ------------------------------------------------------------------
    # Write locations
    # ------------------------------------------------------------------

    def _compute_flat_decode_out_cache_locs(
        self,
        page_tables,
        block_table_base_offsets,
        seq_lens,
        page_size,
        num_tokens_per_req=1,
    ):
        """Per-group decode write locs, gathered from the group's own read
        table (M-W1). Plain decode writes one token per request at seq_len-1;
        spec verify writes num_tokens_per_req at seq_len-N..seq_len-1,
        flattened token-major per request ([bs*N], radix verify layout).
        Positions clamp at 0 for graph-padded rows (seq_len 1 < N), which
        dereference the dummy page harmlessly. The tail page is never a hole
        (SWA holes sit only at the window front). ``page_size`` is the base
        size; groups with a heterogeneous page size divide by their own
        granularity.
        """
        n = num_tokens_per_req
        check_bounds = os.environ.get("TOKENSPEED_FLAT_DEBUG") == "1"
        if n == 1:
            pos = (seq_lens - 1).to(torch.int64)
        else:
            steps = torch.arange(n, device=seq_lens.device, dtype=torch.int64)
            pos = (seq_lens.to(torch.int64).unsqueeze(1) - n + steps).clamp_min(0)
            pos = pos.reshape(-1)
        out = {}
        for gid, table in page_tables.items():
            ps = self._group_page_size(gid) if gid else page_size
            page_idx = pos // ps
            off = (pos % ps).to(torch.int32)
            bases = block_table_base_offsets[gid].to(torch.int64)
            if n == 1:
                local_page_idx = page_idx - bases
                if check_bounds:
                    torch._assert_async(
                        (
                            (local_page_idx >= 0) & (local_page_idx < table.shape[1])
                        ).all(),
                        f"flat decode logical page is outside group {gid!r}'s table",
                    )
                pages = table.gather(1, local_page_idx.unsqueeze(1)).squeeze(1)
            else:
                local_page_idx = page_idx.view(-1, n) - bases.unsqueeze(1)
                if check_bounds:
                    torch._assert_async(
                        (
                            (local_page_idx >= 0) & (local_page_idx < table.shape[1])
                        ).all(),
                        f"flat decode logical page is outside group {gid!r}'s table",
                    )
                pages = table.gather(1, local_page_idx).reshape(-1)
            # Mirror the graph-path kernel's clamp: -1 pads/holes route to dummy page 0.
            out[gid] = pages.clamp_min(0) * ps + off
        return out

    def _compute_flat_extend_out_cache_locs(
        self,
        page_tables,
        block_table_base_offsets,
        extend_prefix_lens_cpu,
        extend_seq_lens_cpu,
        page_size,
    ):
        """Per-group extend write locs: positions [prefix_len, seq_len) per
        request, flattened in q/k/v token order (cu_extend_seq_lens). Bounds
        come from the CPU mirrors — no per-request GPU sync.
        TODO(flat-perf): batch the per-request loop via repeat_interleave.
        """
        device = next(iter(page_tables.values())).device
        check_bounds = os.environ.get("TOKENSPEED_FLAT_DEBUG") == "1"
        prefix_lens = [int(x) for x in extend_prefix_lens_cpu.tolist()]
        extend_lens = [int(x) for x in extend_seq_lens_cpu.tolist()]
        out = {gid: [] for gid in page_tables}
        for i, (start, num_new) in enumerate(zip(prefix_lens, extend_lens)):
            pos = torch.arange(start, start + num_new, dtype=torch.int64, device=device)
            for gid, table in page_tables.items():
                ps = self._group_page_size(gid) if gid else page_size
                page_idx = pos // ps
                off = (pos % ps).to(torch.int32)
                local_page_idx = page_idx - block_table_base_offsets[gid][i].to(
                    torch.int64
                )
                if check_bounds:
                    torch._assert_async(
                        (
                            (local_page_idx >= 0) & (local_page_idx < table.shape[1])
                        ).all(),
                        f"flat extend logical page is outside group {gid!r}'s table",
                    )
                pages = table[i].gather(0, local_page_idx)
                out[gid].append(pages * ps + off)
        return {
            gid: (
                torch.cat(chunks)
                if chunks
                else torch.empty(0, dtype=torch.int32, device=device)
            )
            for gid, chunks in out.items()
        }

    def flat_enter_draft_lookback(self, bs: int) -> bool:
        """Drafter hook (via the Inkling conv wrapper): swap the decode
        metadata's write locs to the lookback-window variant — N + D tokens
        per request at positions ``seq-(N+D)..seq-1`` — so the lookback
        pass's KV writes cover its extra leading rows.

        Radix metadata (no flat locs) needs no swap: the drafter's caller
        locs are live there. Returns False when lookback is disarmed, so the
        caller falls back to the plain window pass. Captured Flat metadata
        without a complete static lookback stack is a contract error: it must
        never fall through to the eager gather. The next round's metadata init
        restores the plain locs.
        """
        md = self.forward_decode_metadata
        if md is None:
            return True
        captured = getattr(self, "cuda_graph_decode_metadata", None) or {}
        captured_md = captured.get(bs)
        graph_backed = md is captured_md
        if md.out_cache_locs is None:
            if graph_backed:
                raise RuntimeError(
                    "flat CUDA graph metadata has no group-keyed write "
                    "locations; refusing draft lookback capture"
                )
            return True
        lookback = int(getattr(self, "flat_draft_lookback", 0) or 0)
        if lookback <= 0:
            return False
        spec_n = max(int(getattr(self, "spec_num_tokens", 1) or 1), 1)
        total = spec_n + lookback
        if graph_backed:
            buffers = getattr(self, "cuda_graph_flat_lookback_locs", None) or {}
            need = bs * total
            invalid = {
                gid: None if gid not in buffers else int(buffers[gid].shape[0])
                for gid in md.out_cache_locs
                if gid not in buffers or buffers[gid].shape[0] < need
            }
            if invalid:
                raise RuntimeError(
                    "flat CUDA graph metadata is missing its static lookback "
                    f"locations for batch size {bs}: need_rows={need}, "
                    f"available_rows={invalid}; refusing an eager page-table "
                    "gather during graph capture"
                )
            locs = {gid: buffers[gid][:need] for gid in md.out_cache_locs}
        else:
            locs = self._compute_flat_decode_out_cache_locs(
                md.page_tables,
                md.block_table_base_offsets,
                md.seq_lens,
                self.page_size,
                total,
            )
        if graph_backed:
            # CUDA Graph replay keys provenance by the captured metadata
            # object. Keep that object stable and only redirect its loc views
            # to the persistent lookback buffers.
            md.out_cache_locs = locs
        else:
            self.forward_decode_metadata = replace(md, out_cache_locs=locs)
        return True

    def _maybe_check_flat_write_locs(self, page_tables, out_cache_locs, page_size):
        """TOKENSPEED_FLAT_DEBUG=1 (eager only, GPU sync): write pages must
        be real and inside the group's table. Not for graph-padded batches —
        dummy rows would trip the non-hole assert (see the padding contract
        in _flat_replay_fill).
        """
        if os.environ.get("TOKENSPEED_FLAT_DEBUG") != "1":
            return
        for gid, locs in out_cache_locs.items():
            pages = (locs // self._group_page_size(gid)).to(torch.int32)
            table = page_tables[gid]
            assert (
                pages != 0
            ).all(), f"flat write loc in null page 0 for group {gid!r}"
            real = table[table > 0]
            assert torch.isin(
                pages, real
            ).all(), f"flat write pages escape group {gid!r}'s table"

    # ------------------------------------------------------------------
    # CUDA-graph per-group buffers
    # ------------------------------------------------------------------

    def _init_flat_graph_buffers(self, max_bs: int) -> None:
        """Reset the persistent per-group buffers; call from
        init_cuda_graph_state BEFORE any backend early return — replay reads
        the dict unconditionally for the stale-table guard.

        Attention-consumed groups get views into ONE stacked table/loc pair
        ([G, max_bs, Wmax] / [G, max_bs * spec_num_tokens]). Absolute-layout
        groups share one fused replay-time loc launch. Bounded groups reuse
        the same persistent stack but apply their per-request logical bases
        through base-aware per-group launches."""
        self.cuda_graph_flat_page_tables: dict[str, torch.Tensor] = {}
        self.cuda_graph_flat_block_table_base_offsets: dict[str, torch.Tensor] = {}
        self.cuda_graph_flat_out_cache_locs: dict[str, torch.Tensor] = {}
        self.cuda_graph_flat_lookback_locs: dict[str, torch.Tensor] = {}
        self._cuda_graph_max_bs = max_bs
        self._flat_locs_stack = None
        self._flat_lookback_locs_stack = None
        self._flat_tables_stack = None
        self._flat_group_widths = {}
        self._flat_stack_gids = []
        self._flat_att_group_count = 0
        self._flat_fused_group_count = 0
        self._flat_local_seq_lens = {}
        raw_att_gids = {
            gid
            for gid in self.flat_group_specs
            if gid not in self.flat_state_group_ids
            and gid not in self.flat_engine_owned_group_ids
        }
        fused_gids = sorted(raw_att_gids & set(self.flat_absolute_group_ids))
        base_aware_gids = sorted(raw_att_gids - set(fused_gids))
        att_gids = fused_gids + base_aware_gids
        owned_gids = sorted(
            gid
            for gid in self.flat_group_specs
            if gid in self.flat_engine_owned_group_ids
        )
        gids = att_gids + owned_gids  # attention prefix, wrapper-owned tail
        if not gids:
            return
        widths = {
            gid: ceil_div(
                self.max_num_pages * self.page_size, self._group_page_size(gid)
            )
            for gid in gids
        }
        logger.debug(
            "flat graph buffers: max_num_pages=%d page_size=%d max_bs=%d widths=%s",
            self.max_num_pages,
            self.page_size,
            max_bs,
            widths,
        )
        wmax = max(widths.values())
        g = len(gids)
        self._flat_stack_gids = gids
        self._flat_att_group_count = len(att_gids)
        self._flat_fused_group_count = len(fused_gids)
        self._flat_tables_stack = torch.zeros(
            (g, max_bs, wmax), dtype=torch.int32, device=self.device
        )
        # Spec verify: graphs read [max_bs*N] loc views of the stack, so size it up front
        spec_n = max(int(getattr(self, "spec_num_tokens", 1) or 1), 1)
        self._flat_locs_stack = torch.zeros(
            (len(att_gids), max_bs * spec_n), dtype=torch.int32, device=self.device
        )
        lookback = int(getattr(self, "flat_draft_lookback", 0) or 0)
        if lookback > 0:
            # Draft lookback window passes write N + D rows per request
            # (positions seq-(N+D)..seq-1); their captured kernels read this
            # second stack, refilled alongside the main one at replay.
            self._flat_lookback_locs_stack = torch.zeros(
                (len(att_gids), max_bs * (spec_n + lookback)),
                dtype=torch.int32,
                device=self.device,
            )
            for i, gid in enumerate(att_gids):
                self.cuda_graph_flat_lookback_locs[gid] = (
                    self._flat_lookback_locs_stack[i]
                )
        self._flat_group_widths = widths
        self._flat_group_ps_tensor = torch.tensor(
            [self._group_page_size(gid) for gid in att_gids],
            dtype=torch.int32,
            device=self.device,
        )
        self._flat_unpack_meta_dev = torch.zeros(
            (g, 2), dtype=torch.int32, device=self.device
        )
        for i, gid in enumerate(gids):
            self.cuda_graph_flat_page_tables[gid] = self._flat_tables_stack[
                i, :, : widths[gid]
            ]
            if i < len(att_gids):
                self.cuda_graph_flat_block_table_base_offsets[gid] = torch.zeros(
                    (max_bs,), dtype=torch.int32, device=self.device
                )
                self.cuda_graph_flat_out_cache_locs[gid] = self._flat_locs_stack[i]

        self._flat_local_seq_lens = {
            gid: torch.empty((max_bs,), dtype=torch.int32, device=self.device)
            for gid in base_aware_gids
        }

    def _flat_capture_group_views(
        self, bs: int, flat_cache_group_ids, tokens_per_req: int = 1
    ):
        """Capture-time (page_tables, bases, out_cache_locs) per-group views into the
        persistent buffers initialized by :meth:`_init_flat_graph_buffers`.
        Real tables only arrive at replay, which copies fresh data to these
        graph-recorded addresses.
        Verify (tokens_per_req = spec_num_tokens) keeps [bs]-row tables but
        records [bs*N] write-loc views (token-major, radix verify layout).
        Returns (None, None, None) when only state groups (or none) are delivered.
        """
        if not flat_cache_group_ids:
            return None, None, None
        page_tables = {}
        block_table_base_offsets = {}
        out_cache_locs = {}
        for gid in flat_cache_group_ids:
            if gid in self.flat_state_group_ids:
                # State pages ride to the mamba backend; no buffers here.
                continue
            if gid in self.flat_engine_owned_group_ids:
                # Engine-owned (conv) group: the wrapper keeps its own capture buffers
                continue
            buf = self.cuda_graph_flat_page_tables.get(gid)
            if buf is None:
                # Replay write locs are ALWAYS the fused triton launch over
                # the stacked buffers; a group outside the stack could never
                # get its locs filled. Groups must be declared (via
                # flat_group_specs) before init_cuda_graph_state.
                raise RuntimeError(
                    f"flat group {gid!r} is not in the stacked CUDA-graph "
                    f"buffers (stack: {self._flat_stack_gids}); declare every "
                    "capture-visible group's page size before graph init."
                )
            base_buf = self.cuda_graph_flat_block_table_base_offsets.get(gid)
            if base_buf is None:
                raise RuntimeError(
                    f"flat group {gid!r} is missing its stacked CUDA-graph "
                    "logical-base buffer"
                )
            loc_buf = self.cuda_graph_flat_out_cache_locs.get(gid)
            need = self._cuda_graph_max_bs * tokens_per_req
            if loc_buf is None or loc_buf.shape[0] < need:
                raise RuntimeError(
                    f"flat loc stack too small for group {gid!r}: capture "
                    f"needs {need} rows, have "
                    f"{0 if loc_buf is None else loc_buf.shape[0]}; the "
                    "stack is sized max_bs * spec_num_tokens at init, so "
                    f"tokens_per_req={tokens_per_req} must not exceed "
                    f"spec_num_tokens={getattr(self, 'spec_num_tokens', 1)}."
                )
            page_tables[gid] = buf[:bs, :]
            block_table_base_offsets[gid] = base_buf[:bs]
            out_cache_locs[gid] = loc_buf[: bs * tokens_per_req]
        if not page_tables:
            # Only state groups delivered: nothing for this backend.
            return None, None, None
        return page_tables, block_table_base_offsets, out_cache_locs

    def _flat_try_packed_unpack(self, bs: int, flat_block_tables) -> bool:
        """One-launch fill of the stacked graph tables from the bridge's
        packed upload. Requires the stack to cover every delivered
        non-state group and all sources to share one storage (the packed
        bridge guarantees both); returns False to take the per-group
        fallback otherwise."""
        stack = self._flat_tables_stack
        if stack is None:
            return False
        gids = self._flat_stack_gids
        # Fresh pinned alloc each step: a persistent pinned buffer would race with overlap scheduling
        meta = torch.empty((len(gids), 2), dtype=torch.int32, pin_memory=True)
        base_ptr = None
        actual = None
        for i, gid in enumerate(gids):
            src = flat_block_tables.get(gid)
            if src is None or src.shape[1] > stack.shape[2]:
                return False
            ptr = src.untyped_storage().data_ptr()
            if base_ptr is None:
                base_ptr = ptr
                actual = src.shape[0]
            elif ptr != base_ptr or src.shape[0] != actual:
                return False
            meta[i, 0] = src.storage_offset()
            meta[i, 1] = src.shape[1]
        if base_ptr is None:
            return False
        self._flat_unpack_meta_dev.copy_(meta, non_blocking=True)
        src0 = flat_block_tables[gids[0]]
        packed = torch.as_strided(
            src0,
            (src0.untyped_storage().nbytes() // 4,),
            (1,),
            storage_offset=0,
        )
        flat_tables_unpack(
            packed,
            self._flat_unpack_meta_dev,
            stack,
            bs,
            actual_bs=min(actual, bs),
            tail_pad=self.flat_table_tail_pad,
        )
        return True

    def _flat_replay_stale_guard(
        self, bs: int, flat_block_tables, flat_block_table_base_offsets
    ) -> frozenset[str]:
        """Fail loudly instead of replaying over stale/zero page tables.
        bs == 0 may skip: col-0 buffer entries stay valid (never -1),
        outputs are discarded, and only unit tests reach it."""
        validated_group_ids = validate_flat_table_base_offsets(
            flat_block_tables,
            flat_block_table_base_offsets,
            where=f"{type(self).__name__} replay",
        )
        if not self.cuda_graph_flat_page_tables or bs <= 0:
            return validated_group_ids
        name = type(self).__name__
        if not flat_block_tables:
            raise RuntimeError(
                f"{name} replay: flat per-group CUDA-graph buffers "
                f"exist for groups "
                f"{sorted(self.cuda_graph_flat_page_tables)} "
                f"but flat_block_tables is missing/empty at bs={bs}; the "
                "captured graph would read stale page tables."
            )
        delivered = validated_group_ids - self.flat_state_group_ids
        captured = self.cuda_graph_flat_page_tables.keys()
        if delivered != captured:
            raise RuntimeError(
                f"{name} replay: captured/delivered history groups differ at "
                f"bs={bs}: captured={sorted(captured)}, "
                f"delivered={sorted(delivered)}; the captured graph would "
                "read stale page tables."
            )
        return validated_group_ids

    def _flat_replay_fill(
        self,
        bs: int,
        flat_block_tables,
        flat_block_table_base_offsets,
        seq_lens,
        tokens_per_req: int = 1,
        *,
        validated_group_ids: frozenset[str] | None = None,
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
            self._flat_locs_stack is None
            or self._flat_locs_stack.shape[1] < bs * tokens_per_req
        ):
            raise RuntimeError(
                "flat replay write locs need the stacked loc buffer "
                f"(bs={bs}, tokens_per_req={tokens_per_req}, stack="
                f"{None if self._flat_locs_stack is None else tuple(self._flat_locs_stack.shape)}); "
                "the stack is sized max_bs * spec_num_tokens at graph init "
                "and there is no python fallback."
            )
        if validated_group_ids is None:
            validate_flat_table_base_offsets(
                flat_block_tables,
                flat_block_table_base_offsets,
                where=f"{type(self).__name__} replay fill",
            )
        self._flat_packed_unpack_ran = self._flat_try_packed_unpack(
            bs, flat_block_tables
        )
        if not self._flat_packed_unpack_ran:
            for gid, src in flat_block_tables.items():
                if gid in self.flat_state_group_ids:
                    # State group: the mamba backend consumes it directly.
                    continue
                if gid in self.flat_engine_owned_group_ids:
                    # Engine-owned (conv) group: the wrapper fills its own buffers (they alias this stack)
                    continue
                buf = self.cuda_graph_flat_page_tables[gid]
                # Clamp: scheduler may send extra reservation columns; kernels never read past cache_seqlens
                cols = min(src.shape[1], buf.shape[1])
                # cols >= 1: a zero-width table would leave dummy rows' col 0 unwritten
                assert cols >= 1, f"flat table for group {gid!r}: zero-width table"
                rows = min(src.shape[0], bs)
                buf[:rows, :cols].copy_(src[:rows, :cols])
                if cols < buf.shape[1]:
                    buf[:rows, cols:].fill_(self.flat_table_tail_pad)
                if rows < bs:
                    # Dummy rows pad with 0 (the zero-init dummy page).
                    buf[rows:bs].fill_(0)

        for gid in self.cuda_graph_flat_block_table_base_offsets:
            src = flat_block_table_base_offsets[gid]
            rows = min(src.shape[0], bs)
            self.cuda_graph_flat_block_table_base_offsets[gid][:rows].copy_(src[:rows])
            if rows < bs:
                self.cuda_graph_flat_block_table_base_offsets[gid][rows:bs].zero_()

        self._flat_replay_fill_locs(
            seq_lens[:bs], self._flat_locs_stack, bs, tokens_per_req
        )
        if self._flat_lookback_locs_stack is not None:
            # Second variant for the draft's lookback window passes: N + D
            # rows per request ending at the same frontier.
            self._flat_replay_fill_locs(
                seq_lens[:bs],
                self._flat_lookback_locs_stack,
                bs,
                tokens_per_req + self.flat_draft_lookback,
            )

    def _flat_replay_fill_locs(
        self,
        seq_lens: torch.Tensor,
        out: torch.Tensor,
        bs: int,
        tokens_per_req: int,
    ) -> None:
        """Fill stacked graph loc buffers while honoring logical table bases."""
        fused = self._flat_fused_group_count
        if fused:
            flat_decode_locs(
                self._flat_tables_stack[:fused],
                self._flat_group_ps_tensor[:fused],
                seq_lens,
                out[:fused],
                bs,
                tokens_per_req,
            )
        for i in range(fused, self._flat_att_group_count):
            gid = self._flat_stack_gids[i]
            ps = self._group_page_size(gid)
            local_seq_lens = self._flat_local_seq_lens[gid][:bs]
            torch.add(
                seq_lens,
                self.cuda_graph_flat_block_table_base_offsets[gid][:bs],
                alpha=-ps,
                out=local_seq_lens,
            )
            flat_decode_locs(
                self._flat_tables_stack[i : i + 1],
                self._flat_group_ps_tensor[i : i + 1],
                local_seq_lens,
                out[i : i + 1],
                bs,
                tokens_per_req,
            )
