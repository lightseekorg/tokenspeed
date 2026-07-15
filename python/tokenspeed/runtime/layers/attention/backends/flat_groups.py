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
table per cache group (``flat_block_tables: dict[group_id, [bs, max_pages]]``)
instead of the radix single table, and must route every KV read AND write
through the layer's own group (M-W1). This mixin holds the group-selection,
write-location, and CUDA-graph per-group buffer machinery shared by the MHA
and TRT-LLM backends; model/kernel-specific constraints (spec decode, DFLASH)
stay in the backends.

Table contract (canonical): rows are requests (padded rows carry the
zero-init dummy page 0), column tails pad with -1 and are never read past
``cache_seqlens``; SWA holes sit only at the window front and are written as
the null page 0 by the scheduler export.
"""

from __future__ import annotations

import os

import torch


class FlatCacheGroupsMixin:
    """Per-group table/write-loc selection + CUDA-graph buffer discipline.

    Host class requirements: ``self.device``, ``self.page_size``,
    ``self.max_num_pages``, ``self.forward_decode_metadata`` (with
    ``page_tables``/``out_cache_locs`` fields), and calling
    :meth:`_init_flat_graph_buffers` from ``init_cuda_graph_state``.
    """

    # family="state" group ids (GDN/mamba state pages); learned from the
    # pool's specs in init_cuda_graph_state, shed from every table here.
    flat_state_group_ids: frozenset[str] = frozenset()

    # Value for CUDA-graph buffer column tails past this replay's table
    # width. -1 is a debug tripwire (never read past cache_seqlens by the
    # MHA kernels); backends whose kernels assume a full-width table
    # (trtllm: row stride derived from max_kv_len) override with 0, the
    # zero-init dummy page — always safe to dereference.
    flat_table_tail_pad: int = -1

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
        if self.flat_state_group_ids:
            tables = {
                gid: table
                for gid, table in tables.items()
                if gid not in self.flat_state_group_ids
            }
        return tables or None

    def _learn_flat_state_groups(self, paged_cache_group_specs) -> None:
        """Record the pool's family="state" group ids (see
        flat_state_group_ids); called from init_cuda_graph_state, the one
        place the pool's specs reach every backend."""
        self.flat_state_group_ids = frozenset(
            str(spec.group_id)
            for spec in paged_cache_group_specs
            if spec.family == "state"
        )

    # ------------------------------------------------------------------
    # Write locations
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_flat_decode_out_cache_locs(
        page_tables, seq_lens, page_size, num_tokens_per_req=1
    ):
        """Per-group decode write locs, gathered from the group's own read
        table (M-W1). Plain decode writes one token per request at seq_len-1;
        spec verify writes num_tokens_per_req at seq_len-N..seq_len-1,
        flattened token-major per request ([bs*N], radix verify layout).
        Positions clamp at 0 for graph-padded rows (seq_len 1 < N), which
        dereference the dummy page harmlessly. The tail page is never a hole
        (SWA holes sit only at the window front).
        """
        n = num_tokens_per_req
        if n == 1:
            pos = (seq_lens - 1).to(torch.int64)
        else:
            steps = torch.arange(n, device=seq_lens.device, dtype=torch.int64)
            pos = (seq_lens.to(torch.int64).unsqueeze(1) - n + steps).clamp_min(0)
            pos = pos.reshape(-1)
        page_idx = pos // page_size
        off = (pos % page_size).to(torch.int32)
        out = {}
        for gid, table in page_tables.items():
            if n == 1:
                pages = table.gather(1, page_idx.unsqueeze(1)).squeeze(1)
            else:
                pages = table.gather(1, page_idx.view(-1, n)).reshape(-1)
            out[gid] = pages * page_size + off
        return out

    @staticmethod
    def _compute_flat_extend_out_cache_locs(
        page_tables, extend_prefix_lens_cpu, extend_seq_lens_cpu, page_size
    ):
        """Per-group extend write locs: positions [prefix_len, seq_len) per
        request, flattened in q/k/v token order (cu_extend_seq_lens). Bounds
        come from the CPU mirrors — no per-request GPU sync.
        TODO(flat-perf): batch the per-request loop via repeat_interleave.
        """
        device = next(iter(page_tables.values())).device
        prefix_lens = [int(x) for x in extend_prefix_lens_cpu.tolist()]
        extend_lens = [int(x) for x in extend_seq_lens_cpu.tolist()]
        out = {gid: [] for gid in page_tables}
        for i, (start, num_new) in enumerate(zip(prefix_lens, extend_lens)):
            pos = torch.arange(start, start + num_new, dtype=torch.int64, device=device)
            page_idx = pos // page_size
            off = (pos % page_size).to(torch.int32)
            for gid, table in page_tables.items():
                pages = table[i].gather(0, page_idx)
                out[gid].append(pages * page_size + off)
        return {
            gid: (
                torch.cat(chunks)
                if chunks
                else torch.empty(0, dtype=torch.int32, device=device)
            )
            for gid, chunks in out.items()
        }

    @staticmethod
    def _maybe_check_flat_write_locs(page_tables, out_cache_locs, page_size):
        """TOKENSPEED_FLAT_DEBUG=1 (eager only, GPU sync): write pages must
        be real and inside the group's table. Not for graph-padded batches —
        dummy rows would trip the non-hole assert (see the padding contract
        in _flat_replay_fill).
        """
        if os.environ.get("TOKENSPEED_FLAT_DEBUG") != "1":
            return
        for gid, locs in out_cache_locs.items():
            pages = (locs // page_size).to(torch.int32)
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
        """Reset the lazily-allocated per-group persistent buffers; call from
        init_cuda_graph_state BEFORE any backend early return — replay reads
        the dict unconditionally for the stale-table guard."""
        self.cuda_graph_flat_page_tables: dict[str, torch.Tensor] = {}
        self.cuda_graph_flat_out_cache_locs: dict[str, torch.Tensor] = {}
        self._cuda_graph_max_bs = max_bs

    def _flat_capture_group_views(
        self, bs: int, flat_cache_group_ids, tokens_per_req: int = 1
    ):
        """Capture-time (page_tables, out_cache_locs) per-group views into the
        persistent buffers, lazily allocated. Real tables only arrive at
        replay, which copy_s fresh data to these graph-recorded addresses.
        Verify (tokens_per_req = spec_num_tokens) keeps [bs]-row tables but
        records [bs*N] write-loc views (token-major, radix verify layout).
        Returns (None, None) when only state groups (or none) are delivered.
        """
        if not flat_cache_group_ids:
            return None, None
        page_tables = {}
        out_cache_locs = {}
        for gid in flat_cache_group_ids:
            if gid in self.flat_state_group_ids:
                # State pages ride to the mamba backend; no buffers here.
                continue
            buf = self.cuda_graph_flat_page_tables.get(gid)
            if buf is None:
                buf = torch.zeros(
                    (self._cuda_graph_max_bs, self.max_num_pages),
                    dtype=torch.int32,
                    device=self.device,
                )
                self.cuda_graph_flat_page_tables[gid] = buf
            loc_buf = self.cuda_graph_flat_out_cache_locs.get(gid)
            if (
                loc_buf is None
                or loc_buf.shape[0] < self._cuda_graph_max_bs * tokens_per_req
            ):
                loc_buf = torch.zeros(
                    (self._cuda_graph_max_bs * tokens_per_req,),
                    dtype=torch.int32,
                    device=self.device,
                )
                self.cuda_graph_flat_out_cache_locs[gid] = loc_buf
            page_tables[gid] = buf[:bs, :]
            out_cache_locs[gid] = loc_buf[: bs * tokens_per_req]
        if not page_tables:
            # Only state groups delivered: nothing for this backend.
            return None, None
        return page_tables, out_cache_locs

    def _flat_replay_stale_guard(self, bs: int, flat_block_tables) -> None:
        """Fail loudly instead of replaying over stale/zero page tables.
        bs == 0 may skip: col-0 buffer entries stay valid (never -1),
        outputs are discarded, and only unit tests reach it."""
        if not self.cuda_graph_flat_page_tables or bs <= 0:
            return
        name = type(self).__name__
        if not flat_block_tables:
            raise RuntimeError(
                f"{name} replay: flat per-group CUDA-graph buffers "
                f"exist for groups "
                f"{sorted(self.cuda_graph_flat_page_tables)} "
                f"but flat_block_tables is missing/empty at bs={bs}; the "
                "captured graph would read stale page tables."
            )
        missing = set(self.cuda_graph_flat_page_tables) - set(flat_block_tables)
        if missing:
            raise RuntimeError(
                f"{name} replay: flat_block_tables at bs="
                f"{bs} is missing captured groups {sorted(missing)} "
                f"(delivered: {sorted(flat_block_tables)}); the captured "
                "graph would read stale page tables for those groups."
            )

    def _flat_replay_fill(
        self, bs: int, flat_block_tables, seq_lens, tokens_per_req: int = 1
    ) -> None:
        """Copy this replay's tables into the captured buffers and recompute
        the per-group write locs from the live seq_lens (tokens_per_req locs
        per request on the spec-verify path).

        Padding contract (canonical; bs is the padded bs): dummy ROWS pad
        with 0 — replayed at seq_lens=1 they dereference exactly col 0,
        the zero-init dummy page. Column tails pad with -1, never read
        past cache_seqlens.
        """
        for gid, src in flat_block_tables.items():
            if gid in self.flat_state_group_ids:
                # State group: the mamba backend consumes it directly.
                continue
            buf = self.cuda_graph_flat_page_tables[gid]
            cols = src.shape[1]
            # cols >= 1: a zero-width table would leave dummy rows'
            # col 0 unwritten.
            assert 1 <= cols <= buf.shape[1], (
                f"flat table for group {gid!r}: {cols} cols outside"
                f" [1, {buf.shape[1]}] (CUDA-graph buffer width)"
            )
            assert src.shape[0] >= bs, (
                f"flat table for group {gid!r} has {src.shape[0]} rows"
                f" < padded bs {bs}"
            )
            buf[:bs, :cols].copy_(src[:bs, :])
            if cols < buf.shape[1]:
                buf[:bs, cols:].fill_(self.flat_table_tail_pad)

        # seq_lens is the controller-filled live buffer (current lens +
        # padding 1s), written BEFORE replay init, so [:bs] is current.
        locs = self._compute_flat_decode_out_cache_locs(
            {
                gid: self.cuda_graph_flat_page_tables[gid][:bs, :]
                for gid in flat_block_tables
                if gid not in self.flat_state_group_ids
            },
            seq_lens[:bs],
            self.page_size,
            tokens_per_req,
        )
        for gid, val in locs.items():
            self.cuda_graph_flat_out_cache_locs[gid][: bs * tokens_per_req].copy_(val)
