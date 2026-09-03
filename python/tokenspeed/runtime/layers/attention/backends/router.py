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

"""The cache-group router: the one place scheduler blocks become kernel pages.

``CacheGroupRouter`` is the runner-facing attention backend for every model
whose attention is paged KV. It holds one ``PagedAttentionBackend`` leaf per
attention (history-family) cache group and is the only object between the
scheduler bridge and the kernels that knows about groups at all:

* it learns each group's block granularity from the pool's published specs
  (``CacheGroupGeometry``) and each leaf's kernel page size, and expands the
  bridge's raw per-group block tables into kernel page tables — the single
  logical -> physical mapping point cache-concepts.md Principle 5 asks for;
* it derives every KV write location (extend spans, the decode / verify
  window, and the draft steps the drafters publish) from those same tables,
  so slot math has exactly one home;
* it dispatches a layer's forward to the leaf of ``layer.group_id`` with
  that group's write locations.

Leaves see ``page_table`` / ``seq_lens`` / ``out_cache_loc`` and nothing
else. A single-group model is a router with one leaf; there is no
single-table special case anywhere.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from tokenspeed.runtime.execution.breakable_cuda_graph import break_point
from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
from tokenspeed.runtime.layers.attention.backends.paged import (
    PagedAttentionBackend,
)
from tokenspeed.runtime.layers.attention.backends.cache_group_geometry import (
    CacheGroupGeometry,
    learn_cache_group_geometry,
)
from tokenspeed.runtime.layers.attention.backends.group_tables import (
    GroupTableSpec,
    GroupTableStacks,
)

if TYPE_CHECKING:
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.layers.attention.kv_cache.base import CachePool
    from tokenspeed.runtime.layers.paged_attention import PagedAttention


@dataclass(frozen=True)
class DraftHistoryView:
    """The full-history table as the draft chain's write-location source.

    Attributes:
        table: ``[max_bs, stack_max_num_pages]`` contiguous kernel-page table
            (one group's table in the router's stack; address-stable for the
            graph's lifetime).
        page_size: Tokens per kernel page.
        max_tokens: Per-request token capacity of the group's own columns.
    """

    table: torch.Tensor
    page_size: int
    max_tokens: int


@dataclass
class RouterDecodeWriteLocations:
    """Per-group decode write-slot views published for one padded ``bs``.

    Pointer-stable views of the router's location stack; the captured graph
    records them through the leaves' KV writes, so the pointer guard walks
    this object (``decode_write_locations`` slot).
    """

    tokens_per_req: int
    by_group: dict[str, torch.Tensor]


class CacheGroupRouter(AttentionBackend):
    """One paged leaf per attention cache group; see the module docstring."""

    def __init__(
        self,
        leaf_factory: Callable[[str, int], PagedAttentionBackend],
        *,
        is_draft: bool,
        spec_num_tokens: int,
        device,
    ) -> None:
        """Args:
        leaf_factory: ``(group_id, block_granularity) -> leaf``; called once
            per paged group of the bound pool's view at :meth:`set_cache_pool`
            (the pool is the only source of the group set and geometry).
        is_draft: Whether this is the draft side (selects the draft-hook
            behavior; write locations are router-owned on both sides).
        spec_num_tokens: Verify width ``N`` (1 without speculation); the
            target's decode write window and the location stack's
            per-request capacity.
        device: Buffer device.
        """
        self._leaf_factory = leaf_factory
        self.leaves: dict[str, PagedAttentionBackend] = {}
        self._geometry: CacheGroupGeometry | None = None
        self.is_draft = bool(is_draft)
        self.spec_num_tokens = max(int(spec_num_tokens or 1), 1)
        self.device = device
        self.cache_pool: CachePool | None = None
        self._speculative_state_backends = []
        self._stacks: GroupTableStacks | None = None
        # Published write locations: the decode slot (graph-recorded views,
        # refreshed in place) and the extend slot (fresh per round).
        self.decode_write_locations: RouterDecodeWriteLocations | None = None
        self._decode_views: dict[tuple[int, int], RouterDecodeWriteLocations] = {}
        self._decode_request_offset = 0
        self._extend_write_locations: dict[str, torch.Tensor] | None = None

    def bind(
        self,
        geometry: CacheGroupGeometry,
        leaves: Mapping[str, PagedAttentionBackend],
    ) -> None:
        """Install the pool-derived geometry and one leaf per paged group.

        ``set_cache_pool`` is the production caller; unit fixtures bind
        hand-built leaves directly.
        """
        if not leaves:
            raise ValueError(
                f"{type(self).__name__}: the bound pool view has no paged "
                "(history-family) cache groups to route"
            )
        for gid in leaves:
            if geometry.families.get(gid, "history") != "history":
                raise ValueError(
                    f"cache group {gid!r} is family {geometry.families[gid]!r}; "
                    "the router serves paged history groups only"
                )
        self._geometry = geometry
        self.leaves = dict(sorted(leaves.items()))
        self._stacks = None
        self._decode_views = {}

    # ------------------------------------------------------------------
    # Structure
    # ------------------------------------------------------------------

    @property
    def group_ids(self) -> tuple[str, ...]:
        return tuple(self.leaves)

    def child_backends(self) -> tuple[PagedAttentionBackend, ...]:
        return tuple(self.leaves.values())

    def set_cache_pool(self, cache_pool: CachePool) -> None:
        """Bind the pool: learn the group geometry from its published specs
        and build one leaf per paged group of this view (``paged_group_ids``)."""
        self.cache_pool = cache_pool
        geometry = learn_cache_group_geometry(cache_pool.arena.cache_group_specs)
        self.bind(
            geometry,
            {
                gid: self._leaf_factory(gid, geometry.granularity_of(gid))
                for gid in cache_pool.paged_group_ids
            },
        )
        for leaf in self.leaves.values():
            leaf.set_cache_pool(cache_pool)

    def configure_runtime(self, **kwargs) -> None:
        for leaf in self.leaves.values():
            leaf.configure_runtime(**kwargs)

    def init_prefill_graph_state(self, max_num_tokens: int, max_bs: int) -> None:
        for leaf in self.leaves.values():
            leaf.init_prefill_graph_state(max_num_tokens, max_bs)

    def register_step_counter(self, step_counter) -> None:
        # The MLA leaves record the PD layerwise step inside their chunked
        # prefill themselves; everything else records at the router's forward.
        self.step_counter = step_counter
        for leaf in self.leaves.values():
            leaf.step_counter = step_counter

    def _leaf_for(self, layer: PagedAttention) -> PagedAttentionBackend:
        try:
            return self.leaves[layer.group_id]
        except KeyError:
            raise KeyError(
                f"layer {getattr(layer, 'layer_id', '?')} names cache group "
                f"{layer.group_id!r}; this router serves {self.group_ids}"
            ) from None

    def _sole_leaf(self, what: str) -> PagedAttentionBackend:
        if len(self.leaves) != 1:
            raise RuntimeError(
                f"{what} assumes a single attention cache group; this router "
                f"serves {self.group_ids}"
            )
        return next(iter(self.leaves.values()))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def geometry(self) -> CacheGroupGeometry:
        if self._geometry is None:
            raise RuntimeError(
                f"{type(self).__name__} is not bound to a cache pool yet"
            )
        return self._geometry

    def _table_specs(self) -> list[GroupTableSpec]:
        return [
            GroupTableSpec(
                group_id=gid,
                block_granularity=self.geometry.granularity_of(gid),
                kernel_page_size=leaf.kernel_page_size,
                max_num_pages=leaf.max_num_pages,
            )
            for gid, leaf in self.leaves.items()
        ]

    def init_cuda_graph_state(self, max_bs: int, **kwargs) -> None:
        """Allocate the group stacks and every leaf's persistent buffers.

        Sized by the max decode batch (never the capture ladder); runs
        unconditionally at wrapper construction so eager decode refreshes the
        same buffers a graph would.
        """
        del kwargs
        self._stacks = GroupTableStacks(
            self._table_specs(),
            max_bs=max_bs,
            max_tokens_per_req=self.spec_num_tokens,
            device=self.device,
        )
        self._decode_views = {}
        for leaf in self.leaves.values():
            leaf.init_cuda_graph_state(max_bs)

    @property
    def stacks(self) -> GroupTableStacks:
        if self._stacks is None:
            raise RuntimeError(
                "CacheGroupRouter.init_cuda_graph_state must run before any "
                "metadata call"
            )
        return self._stacks

    # ------------------------------------------------------------------
    # Write locations
    # ------------------------------------------------------------------

    @property
    def _decode_tokens_per_req(self) -> int:
        # The refresh-time window: the target's verify width, and the draft
        # step-0 window (the verify-shaped round writes the same N slots).
        # Draft steps 1+ republish their own windows through the draft hooks.
        return self.spec_num_tokens

    def _publish_decode_locations(self, bs: int, tokens_per_req: int) -> None:
        key = (bs, tokens_per_req)
        views = self._decode_views.get(key)
        if views is None:
            views = RouterDecodeWriteLocations(
                tokens_per_req=tokens_per_req,
                by_group={
                    gid: self.stacks.decode_locations(gid, bs, tokens_per_req)
                    for gid in self.leaves
                },
            )
            self._decode_views[key] = views
        self.decode_write_locations = views

    def _refresh_decode_locations(self, bs: int, seq_lens: torch.Tensor) -> None:
        n = self._decode_tokens_per_req
        self.stacks.compute_decode_locations(bs, seq_lens, n)
        self._publish_decode_locations(bs, n)

    def decode_window_locations(self) -> torch.Tensor:
        """The full-history group's current decode write window view
        (``[decode_bs * tokens_per_req]``, address-stable; a MIXED round's
        extend requests are skipped). DFLASH reads the TARGET router's verify
        window through this to copy target-aligned KV into the draft cache
        (the pools share one page-id space)."""
        published = self.decode_write_locations
        if published is None:
            raise RuntimeError("decode window requested before any decode refresh")
        gid = self.group_ids[self._draft_history_index()]
        locs = published.by_group[gid]
        if self._decode_request_offset:
            locs = locs[self._decode_request_offset * published.tokens_per_req :]
        return locs

    def extend_span_locations(self) -> torch.Tensor:
        """The full-history group's extend write span of the current round
        (``[sum(extend_seq_lens)]``, request-major)."""
        if self._extend_write_locations is None:
            raise RuntimeError("extend spans requested before init_forward_metadata")
        gid = self.group_ids[self._draft_history_index()]
        return self._extend_write_locations[gid]

    def write_locations(
        self, layer: PagedAttention, forward_mode: ForwardMode
    ) -> torch.Tensor:
        """This layer's KV write slots for the requests the forward covers.

        Decode (and the decode half of a MIXED round) returns the token-major
        ``[bs * N]`` window view over the location stack; extend returns the
        ``[sum(extend_seq_lens)]`` span computed at ``init_forward_metadata``.
        """
        self._leaf_for(layer)
        gid = layer.group_id
        if forward_mode.is_decode():
            published = self.decode_write_locations
            if published is None:
                raise RuntimeError(
                    "decode write locations requested before any decode refresh"
                )
            locs = published.by_group[gid]
            if self._decode_request_offset:
                locs = locs[self._decode_request_offset * published.tokens_per_req :]
            return locs
        if self._extend_write_locations is None:
            raise RuntimeError(
                "extend write locations requested before init_forward_metadata"
            )
        return self._extend_write_locations[gid]

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def _check_live_delivery(self, actual_bs: int, block_tables) -> None:
        """A live batch must deliver every routed group's table — the
        persistent decode buffers would otherwise serve stale pages."""
        if actual_bs <= 0:
            return
        missing = [gid for gid in self.leaves if gid not in block_tables]
        if missing:
            raise RuntimeError(
                f"{type(self).__name__}: block_tables at bs={actual_bs} is missing "
                f"cache groups {missing} (delivered: {sorted(block_tables)})"
            )

    def init_forward_metadata(
        self,
        bs: int,
        num_extends: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
        *,
        block_tables: Mapping[str, torch.Tensor],
        extend_seq_lens: torch.Tensor,
        extend_seq_lens_cpu: torch.Tensor,
        extend_prefix_lens: torch.Tensor,
        extend_prefix_lens_cpu: torch.Tensor,
        extend_with_prefix: bool,
        **kwargs,
    ) -> None:
        """Extend / mixed / idle-warmup metadata for every leaf.

        Expands each group's raw table into the stack, computes the extend
        write spans (and, for a MIXED round, the decode requests' verify
        window), then hands every leaf its ``[bs, max_num_pages]`` kernel page
        table. ``extend_with_prefix`` (some extend request continues a cached
        or chunked prefix) travels with the extend lengths: leaves size their
        paged-prefix metadata by it, so it must reach them unchanged.
        """
        del kwargs
        if not (forward_mode.is_extend_or_mixed() or forward_mode.is_idle()):
            raise RuntimeError(
                "decode metadata goes through refresh_decode_metadata; "
                f"init_forward_metadata only serves extend/mixed/idle ({forward_mode})"
            )
        live_bs = 0 if forward_mode.is_idle() else bs
        self._check_live_delivery(live_bs, block_tables)
        self.stacks.fill(bs, live_bs, block_tables)
        self._extend_write_locations = None
        self._decode_request_offset = 0
        if forward_mode.is_extend_or_mixed() and num_extends > 0:
            total = int(sum(int(x) for x in extend_seq_lens_cpu[:num_extends].tolist()))
            self._extend_write_locations = self.stacks.extend_locations(
                extend_prefix_lens[:num_extends], extend_seq_lens[:num_extends], total
            )
        if forward_mode.is_mixed():
            # The decode requests follow the extend requests; their verify
            # window is the same math as a pure decode over every request,
            # sliced.
            self._refresh_decode_locations(bs, seq_lens)
            self._decode_request_offset = num_extends
        for gid, leaf in self.leaves.items():
            leaf.init_forward_metadata(
                bs,
                num_extends,
                seq_lens,
                self.stacks.table(gid, bs),
                forward_mode,
                extend_seq_lens=extend_seq_lens,
                extend_seq_lens_cpu=extend_seq_lens_cpu,
                extend_prefix_lens=extend_prefix_lens,
                extend_prefix_lens_cpu=extend_prefix_lens_cpu,
                extend_with_prefix=extend_with_prefix,
            )
            leaf.set_request_slots(req_pool_indices[:bs])

    def refresh_decode_metadata(
        self,
        bs: int,
        actual_bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        *,
        forward_mode: ForwardMode,
        block_tables: Mapping[str, torch.Tensor],
        num_extends: int = 0,
        for_graph_replay: bool = False,
        **kwargs,
    ) -> None:
        """The single decode path: fill the stacks from this step's tables,
        derive the decode write window, refresh every leaf in place."""
        del kwargs
        if forward_mode.is_extend_or_mixed():
            raise RuntimeError(
                f"refresh_decode_metadata serves decode only ({forward_mode})"
            )
        self._check_live_delivery(actual_bs, block_tables)
        self.stacks.fill(bs, actual_bs, block_tables)
        # The window covers every request; DECODE reads skip the extend
        # requests a MIXED round's draft refresh carries (num_extends is 0 on
        # targets).
        self._decode_request_offset = num_extends
        self._refresh_decode_locations(bs, seq_lens)
        for gid, leaf in self.leaves.items():
            leaf.refresh_decode_metadata(
                bs,
                actual_bs,
                seq_lens,
                self.stacks.table(gid, bs),
                num_extends=num_extends,
                for_graph_replay=for_graph_replay,
            )
            leaf.set_request_slots(req_pool_indices[:bs])

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
        *,
        block_tables: Mapping[str, torch.Tensor],
        **kwargs,
    ) -> None:
        """Capture seeding: the idle fill over the same stacks replay
        refreshes, then each leaf's own capture hook (the leaf default is
        its idle refresh; FlashMLA installs its recorded tile schedule)."""
        del kwargs
        if not forward_mode.is_decode_or_idle():
            raise NotImplementedError(
                f"{type(self).__name__} CUDA graphs record decode only, got {forward_mode}"
            )
        self.stacks.fill(bs, 0, block_tables)
        self._decode_request_offset = 0
        self._refresh_decode_locations(bs, seq_lens)
        for gid, leaf in self.leaves.items():
            leaf.init_forward_metadata_capture_cuda_graph(
                bs, seq_lens, self.stacks.table(gid, bs)
            )
            leaf.set_request_slots(req_pool_indices[:bs])

    # ------------------------------------------------------------------
    # Draft write locations (the drafters' in-graph slot math)
    # ------------------------------------------------------------------

    def _draft_history_index(self) -> int:
        """Stack index of the full-history group the draft chain writes.

        The multi-step drafters commit their KV along the full-history
        table; a pool without one serves a single group, which then IS the
        history.
        """
        gid = self.geometry.full_history_group_id
        if gid is not None and gid in self.leaves:
            return self.stacks.index(gid)
        if len(self.leaves) == 1:
            return 0
        raise RuntimeError(
            f"{type(self).__name__}: no full-history group among {self.group_ids} "
            "to derive draft write locations from"
        )

    def draft_history_view(self) -> DraftHistoryView:
        """The address-stable full-history table for in-graph draft consumers.

        The multi-step drafters resolve their KV write slots inside the
        captured graph (inputs derive from verify's in-graph accept lengths),
        so their kernels record this table's address at capture; the router
        owns the storage and its refresh rewrites it in place each round.

        The tensor is the group's FULL contiguous table in the stack
        (``[max_bs, stack_max_num_pages]``, batch-ordered): the slot kernels
        assume row-major contiguity. Columns past the group's own
        ``max_num_pages`` are null pages by the fill contract and resolve to
        the dummy slot 0; ``max_tokens`` is the
        group's real per-request capacity, the bound block drafters clamp
        their prefix against.
        """
        i = self._draft_history_index()
        gid = self.group_ids[i]
        # Host-side spec, never ``int(self.stacks.page_sizes[i])``: the block
        # drafters call this inside the captured graph, where a device->host
        # read is a stream-capture violation.
        return DraftHistoryView(
            table=self.stacks.tables[i],
            page_size=self.stacks.group_kernel_page_size(gid),
            max_tokens=self.stacks.group_capacity_tokens(gid),
        )

    def draft_write_locations_uniform(
        self, out: torch.Tensor, cache_start: torch.Tensor, num_tokens: int
    ) -> torch.Tensor:
        """Scratch-buffer slot resolve over the full-history table (side
        writes that must not clobber the published step window)."""
        view = self.draft_history_view()
        if view.table.is_cuda:
            from tokenspeed.runtime.execution.cache_loc_kernel import (
                compute_out_cache_loc_uniform,
            )

            compute_out_cache_loc_uniform(
                out_cache_loc_ptr=out,
                uniform_input_length=num_tokens,
                cache_start=cache_start,
                page_table=view.table,
                page_size=view.page_size,
            )
            return out
        # CPU reference (unit tests): same slot rule, overflow/holes -> 0.
        bs = cache_start.shape[0]
        steps = torch.arange(num_tokens, dtype=torch.int64)
        pos = cache_start.to(torch.int64).unsqueeze(1) + steps
        max_num_pages = view.table.shape[1]
        page_idx = (pos // view.page_size).clamp_max(max_num_pages - 1)
        overflow = (pos // view.page_size) >= max_num_pages
        pages = view.table[:bs].to(torch.int64).gather(1, page_idx)
        locs = pages * view.page_size + pos % view.page_size
        locs = torch.where(overflow | (pages <= 0), torch.zeros_like(locs), locs)
        out[: bs * num_tokens].copy_(locs.reshape(-1).to(out.dtype))
        return out

    def publish_draft_step_locations(
        self, cache_start: torch.Tensor, num_tokens: int
    ) -> torch.Tensor:
        """Publish this draft step's KV write window: ``num_tokens`` slots
        per request starting at ``cache_start``, for every routed group.

        The drafters declare the window (they own the step semantics — the
        Eagle chain's one advancing slot, MTP's re-anchored k-window, the
        DFLASH block); the math and the address-stable storage live here.
        Captured-graph friendly: the same fused launch the decode refresh
        records, over the same location stack, so ``write_locations`` /
        the leaf forwards read the step's slots at recorded addresses.

        Returns the full-history group's ``[bs * num_tokens]`` view (the
        block drafters hand it to their fused prep kernels).
        """
        bs = cache_start.shape[0]
        # Window positions run cache_start .. cache_start+n-1; the decode
        # kernel derives them as end-n .. end-1, so feed end = start + n.
        self.stacks.compute_decode_locations(bs, cache_start + num_tokens, num_tokens)
        self._decode_request_offset = 0
        self._publish_decode_locations(bs, num_tokens)
        i = self._draft_history_index()
        return self.stacks.decode_locs[i, : bs * num_tokens]

    def advance_draft_forward_metadata(self, seq_lens: torch.Tensor) -> None:
        """Eagle chain step: republish the leaves' seq_lens, in-graph.

        Deliberately seq-lens-only: Eagle's step-0 rejected-tail correction
        fires this BEFORE the step-0 attention has consumed the verify-shaped
        write window, so window publication stays an explicit drafter-loop
        call (:meth:`publish_draft_step_locations`)."""
        for leaf in self.leaves.values():
            leaf.advance_draft_forward_metadata(seq_lens)

    def update_draft_forward_metadata(self, frontier: torch.Tensor) -> None:
        """Vanilla MTP re-anchor: seq_lens become the committed frontier,
        in-graph. Seq-lens-only like :meth:`advance_draft_forward_metadata`;
        the drafter publishes its k-window explicitly."""
        for leaf in self.leaves.values():
            leaf.advance_draft_forward_metadata(frontier)

    def fill_block_decode_seq_lens(self, bs: int, block_seq_lens: torch.Tensor) -> None:
        for leaf in self.leaves.values():
            leaf.fill_block_decode_seq_lens(bs, block_seq_lens)

    @contextmanager
    def override_num_extends(self, num_extends: int):
        with ExitStack() as stack:
            for leaf in self.leaves.values():
                stack.enter_context(leaf.override_num_extends(num_extends))
            yield

    def support_kv_cache_prewrite(
        self, forward_mode: ForwardMode | None = None
    ) -> bool:
        return all(
            leaf.support_kv_cache_prewrite(forward_mode)
            for leaf in self.leaves.values()
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    @break_point
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        token_to_kv_pool: CachePool,
        forward_mode: ForwardMode,
        bs: int,
        save_kv_cache: bool = True,
        record_kv_cache: bool | None = None,
        **kwargs,
    ):
        # NOTE: deliberately no ambient-ctx override here (unlike the
        # layer-id composites): a MIXED round's model code dispatches its
        # extend and decode halves through sub-contexts whose mode this
        # forward must honor; the outer ambient mode would clobber them.
        # Under a prefill-graph replay the frozen forward_mode scalar is
        # always EXTEND, which is also the live mode.
        leaf = self._leaf_for(layer)
        out_cache_loc = self.write_locations(layer, forward_mode)
        with self.record_pd_cache_step(forward_mode, save_kv_cache, record_kv_cache):
            if forward_mode.is_decode():
                return leaf.forward_decode(
                    q,
                    k,
                    v,
                    layer,
                    out_cache_loc,
                    token_to_kv_pool,
                    bs,
                    save_kv_cache=save_kv_cache,
                    **kwargs,
                )
            return leaf.forward_extend(
                q,
                k,
                v,
                layer,
                out_cache_loc,
                token_to_kv_pool,
                bs,
                save_kv_cache=save_kv_cache,
                forward_mode=forward_mode,
                **kwargs,
            )

    def forward_decode(
        self,
        q,
        k,
        v,
        layer,
        token_to_kv_pool,
        bs,
        save_kv_cache=True,
        **kwargs,
    ):
        """Composite hosts (hybrid GDN/KDA) dispatch decode directly."""
        from tokenspeed.runtime.execution.forward_batch_info import ForwardMode

        leaf = self._leaf_for(layer)
        out_cache_loc = self.write_locations(layer, ForwardMode.DECODE)
        return leaf.forward_decode(
            q,
            k,
            v,
            layer,
            out_cache_loc,
            token_to_kv_pool,
            bs,
            save_kv_cache=save_kv_cache,
            **kwargs,
        )

    def forward_extend(
        self,
        q,
        k,
        v,
        layer,
        token_to_kv_pool,
        bs,
        save_kv_cache=True,
        forward_mode=None,
        **kwargs,
    ):
        """Composite hosts dispatch extend directly."""
        from tokenspeed.runtime.execution.forward_batch_info import ForwardMode

        leaf = self._leaf_for(layer)
        out_cache_loc = self.write_locations(layer, ForwardMode.EXTEND)
        return leaf.forward_extend(
            q,
            k,
            v,
            layer,
            out_cache_loc,
            token_to_kv_pool,
            bs,
            save_kv_cache=save_kv_cache,
            forward_mode=forward_mode,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Single-group model-side surface (the MLA family's model code reads the
    # leaf's metadata and chunked prefill directly)
    # ------------------------------------------------------------------

    @property
    def chunked_prefill_metadata(self):
        return self._sole_leaf("chunked_prefill_metadata").chunked_prefill_metadata

    @property
    def forward_decode_metadata(self):
        return self._sole_leaf("forward_decode_metadata").forward_decode_metadata

    @property
    def forward_prefill_metadata(self):
        return self._sole_leaf("forward_prefill_metadata").forward_prefill_metadata

    @property
    def supports_mla_projected_value_decode(self) -> bool:
        return all(
            getattr(leaf, "supports_mla_projected_value_decode", False)
            for leaf in self.leaves.values()
        )

    @property
    def data_type(self):
        return self._sole_leaf("data_type").data_type

    @property
    def max_context_len(self) -> int:
        # Every leaf sizes from the one config.context_len; any leaf answers.
        return next(iter(self.leaves.values())).max_context_len

    def forward_extend_chunked(self, *args, **kwargs):
        return self._sole_leaf("forward_extend_chunked").forward_extend_chunked(
            *args, **kwargs
        )

    def forward_sparse_prefill(self, *args, **kwargs):
        return self._sole_leaf("forward_sparse_prefill").forward_sparse_prefill(
            *args, **kwargs
        )

    # ------------------------------------------------------------------
    # DSA KPool surface (GLM-5.3-Flash): the model drives the leaf's pooled
    # indexing runtime through the router.
    # ------------------------------------------------------------------

    def require_kpool_runtime(self):
        return self._sole_leaf("require_kpool_runtime").require_kpool_runtime()

    def kpool_prefill_page_table(self, num_requests: int):
        return self._sole_leaf("kpool_prefill_page_table").kpool_prefill_page_table(
            num_requests
        )

    def kpool_decode_page_table(self, row_start: int, num_requests: int):
        return self._sole_leaf("kpool_decode_page_table").kpool_decode_page_table(
            row_start, num_requests
        )
