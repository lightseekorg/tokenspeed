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

"""Per-group table routing for the MHA-family backends.

Every backend receives the scheduler's per-group block tables
(``block_tables: dict[group_id, [bs, max_pages]]``), expands them to kernel
page tables, and must route every cache read and write through the layer's
own group. This mixin is the ROUTING layer only — which table/locations a
layer sees, state-group shedding, and the metadata-slot selection hooks the
hosts override. Its collaborators each own one concern:

* ``cache_group_geometry.CacheGroupGeometry`` — the pool-learned group
  shapes (granularities, state ids, full-history grain), learned once at
  ``set_cache_pool``;
* ``group_write_locations`` — the slot math, as pure functions;
* ``group_graph_buffers.GroupGraphBuffers`` — the stacked persistent
  CUDA-graph buffers and their capture/fill ops, composed at
  ``_init_group_graph_buffers``.

Model/kernel-specific constraints (spec decode, DFLASH) stay in the
backends.

Table contract (canonical): rows are requests (padded rows carry the
zero-init dummy page 0), column tails pad with -1 and are never read past
``cache_seqlens``; SWA holes sit only at the window front and are written as
the null page 0 by the scheduler export.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import torch

from tokenspeed.runtime.layers.attention.backends.cache_group_geometry import (
    CacheGroupGeometry,
    expand_history_table,
    learn_cache_group_geometry,
)
from tokenspeed.runtime.layers.attention.backends.group_graph_buffers import (
    GroupGraphBuffers,
)
from tokenspeed.runtime.layers.attention.backends.group_write_locations import (
    check_group_write_locs,
    decode_group_out_cache_locs,
    extend_group_out_cache_locs,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.cache_runtime import (
    cache_debug_enabled,
)
from tokenspeed.runtime.layers.attention.page_table import expand_page_table

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.attention.kv_cache.base import CachePool


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

    def bind_decode_views(self, bs: int, cache_group_ids: tuple[str, ...] = ()) -> None:
        """Pre-build the per-bs views with the capture-time group set pinned,
        so the base default capture records the exact per-group views the
        refresh repoints at (see AttentionBackend.bind_decode_views)."""
        if cache_group_ids:
            # Verify keeps [bs]-row tables plus [bs*N] location views.
            assert not (
                self.draft_block_decode and self.spec_num_tokens > 1
            ), "cache_group_ids is unsupported with DFLASH block decode"
        self._decode_views(bs, cache_group_ids=cache_group_ids)

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
        """Learn the pool's group geometry (one immutable value object; see
        CacheGroupGeometry for the state-group / span semantics) and mirror
        it into the attribute names the mixin's readers use."""
        geometry = learn_cache_group_geometry(
            cache_group_specs, default_granularity=self.kernel_page_size
        )
        self._geometry = geometry
        self.state_group_ids = geometry.state_group_ids
        self.group_block_granularities = geometry.granularities
        self._history_block_granularity = geometry.history_block_granularity

    def _expand_history_table(
        self, raw: torch.Tensor, out: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Expand a batch-ordered raw table (scheduler pages of the
        full-history grain) into this backend's kernel pages,
        ``self.max_num_pages`` wide."""
        return expand_history_table(
            raw,
            history_block_granularity=getattr(
                self, "_history_block_granularity", self.kernel_page_size
            ),
            kernel_page_size=self.kernel_page_size,
            max_kernel_pages=self.max_num_pages,
            out=out,
        )

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
        """Thin binding of :func:`decode_group_out_cache_locs` to this
        backend's learned granularities (``page_size`` is the base size for
        group-less keys)."""
        return decode_group_out_cache_locs(
            page_tables,
            seq_lens,
            lambda gid: self._group_block_granularity(gid) if gid else page_size,
            num_tokens_per_req,
        )

    def _compute_extend_group_out_cache_locs(
        self, page_tables, extend_prefix_lens_cpu, extend_seq_lens_cpu, page_size
    ):
        """Thin binding of :func:`extend_group_out_cache_locs`."""
        del page_size
        return extend_group_out_cache_locs(
            page_tables,
            extend_prefix_lens_cpu,
            extend_seq_lens_cpu,
            self._group_block_granularity,
        )

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
        """TOKENSPEED_CACHE_DEBUG=1 gate over
        :func:`check_group_write_locs` (eager only — graph-padded batches
        would trip the non-hole assert on dummy rows)."""
        del page_size
        if not cache_debug_enabled():
            return
        check_group_write_locs(
            page_tables, out_cache_locs, self._group_block_granularity
        )

    # ------------------------------------------------------------------
    # CUDA-graph per-group buffers
    # ------------------------------------------------------------------

    # The composed buffer object (see GroupGraphBuffers); None until
    # _init_group_graph_buffers runs.
    group_graph: GroupGraphBuffers | None = None

    @property
    def cuda_graph_page_tables(self) -> dict[str, torch.Tensor]:
        """Per-group graph table views ({} before graph-state init).

        A property so external adopters (Inkling's conv-table adoption,
        host backends, tests) keep reading the historical name while the
        buffers live on the composed object.
        """
        return self.group_graph.page_tables if self.group_graph is not None else {}

    @property
    def cuda_graph_out_cache_locs(self) -> dict[str, torch.Tensor]:
        """Per-group graph write-location views ({} before init)."""
        return self.group_graph.out_cache_locs if self.group_graph is not None else {}

    def _init_group_graph_buffers(self, max_bs: int) -> None:
        """Build the composed GroupGraphBuffers; call from
        init_cuda_graph_state BEFORE any backend early return — refresh reads
        the table dict unconditionally for the published-groups contract
        check. Geometry is frozen by now (set_cache_pool runs first; bare
        test fixtures that stuff the mirror attributes directly are folded
        back into a geometry here)."""
        geometry = getattr(self, "_geometry", None)
        if geometry is None or geometry.granularities != dict(
            self.group_block_granularities
        ):
            geometry = CacheGroupGeometry(
                granularities=dict(self.group_block_granularities),
                state_group_ids=frozenset(self.state_group_ids),
                history_block_granularity=getattr(
                    self, "_history_block_granularity", self.kernel_page_size
                ),
            )
        self.group_graph = GroupGraphBuffers(
            geometry,
            engine_owned_group_ids=frozenset(self.engine_owned_group_ids),
            consumer_page_size_of=self._consumer_page_size,
            max_bs=max_bs,
            max_num_pages=self.max_num_pages,
            kernel_page_size=self.kernel_page_size,
            spec_num_tokens=getattr(self, "spec_num_tokens", 1),
            device=self.device,
        )

    def _capture_group_views(self, bs: int, cache_group_ids, tokens_per_req: int = 1):
        """Capture-time per-group views (see GroupGraphBuffers.capture_views);
        the mixin's LIVE state/owned sets decide the shed (a wrapper may
        register owned groups after buffer construction)."""
        return self.group_graph.capture_views(
            bs,
            cache_group_ids,
            tokens_per_req,
            skip_group_ids=frozenset(self.state_group_ids)
            | frozenset(self.engine_owned_group_ids),
        )

    def _fill_group_graph_buffers(
        self, bs: int, block_tables, seq_lens, tokens_per_req: int = 1
    ) -> None:
        """Replay/eager fill of the captured buffers (see
        GroupGraphBuffers.fill); the host's ``table_tail_pad`` rides along."""
        self._packed_group_unpack_ran = self.group_graph.fill(
            bs,
            block_tables,
            seq_lens,
            tokens_per_req=tokens_per_req,
            tail_pad=self.table_tail_pad,
            engine_owned_group_ids=frozenset(self.engine_owned_group_ids),
        )
