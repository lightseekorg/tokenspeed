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

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypeVar

import torch

from tokenspeed.runtime.execution.breakable_cuda_graph import break_point
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
from tokenspeed.runtime.utils import get_colorful_logger

if TYPE_CHECKING:
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.layers.attention.configs.base import (
        AttnConfig,
        SoftmaxAttnConfig,
    )
    from tokenspeed.runtime.layers.attention.kv_cache.base import CachePool
    from tokenspeed.runtime.layers.paged_attention import PagedAttention
    from tokenspeed.runtime.pd.utils import StepCounter

logger = get_colorful_logger(__name__)


@dataclass(frozen=True)
class CudaGraphSupport:
    """Per-backend-class CUDA-graph capability declaration.

    Rank-uniform by construction: declarations are class attributes, resolved
    identically on every rank at startup (event-loop.md requires graph
    decisions to derive from replicated state). ``decode_graph=False``
    disables capture/replay of the whole-step decode graph — the unified
    refresh still serves eager decode, so ``init_cuda_graph_state`` and
    ``refresh_decode_metadata`` stay mandatory. ``prefill_graph=False``
    disables the breakable prefill (extend) graph. Static "never works"
    declarations only — and they are the ONLY escape: a prefill capture
    failure at runtime is fatal, so a family that cannot capture must say
    so here rather than rely on a degrade path.
    """

    decode_graph: bool = True
    prefill_graph: bool = True

    def __and__(self, other: CudaGraphSupport) -> CudaGraphSupport:
        return CudaGraphSupport(
            decode_graph=self.decode_graph and other.decode_graph,
            prefill_graph=self.prefill_graph and other.prefill_graph,
        )


def resolve_cuda_graph_support(*backends) -> CudaGraphSupport:
    """AND-compose ``cuda_graph_support`` over ``backends`` and their
    ``child_backends()`` trees, logging every backend class that lowers an
    axis.

    Args:
        backends: Root attention backends; ``None`` entries are skipped. Pass
            the target AND the draft — the decode graph records the whole
            step, drafter loop included.

    Returns:
        The composed support: an axis is False iff any backend in any tree
        declares it False.
    """
    resolved = CudaGraphSupport()
    stack = [backend for backend in backends if backend is not None]
    while stack:
        backend = stack.pop()
        declared = backend.cuda_graph_support
        if not declared.decode_graph:
            logger.info("Decode CUDA graphs disabled by %s", type(backend).__name__)
        if not declared.prefill_graph:
            logger.info("Prefill CUDA graphs disabled by %s", type(backend).__name__)
        resolved = resolved & declared
        stack.extend(backend.child_backends())
    return resolved


class SpeculativeStateBackend(Protocol):
    """Model side-state that consumes speculative verification results."""

    def commit_after_mtp_verify(
        self,
        accepted_lengths: torch.Tensor,
        *,
        num_extends: int,
    ) -> None: ...


_SpeculativeStateBackendT = TypeVar(
    "_SpeculativeStateBackendT", bound=SpeculativeStateBackend
)


class AttentionBackend(ABC):
    """The base class of attention backends"""

    # Decode-capture helpers use a real writable page for every active group
    # when the backend validates live-page geometry at metadata init (V4).
    cache_active_pages_must_be_real: bool = False
    supports_mla_projected_value_decode: bool = False
    # Cache families this backend consumes from the pool contract; wrappers
    # union their children's.
    cache_consumer_families: frozenset[str] = frozenset({"history"})
    # Replay fill pads dummy table rows itself, so the wrapper may pass
    # UNPADDED tables (no per-step F.pad). The grouped backends set True;
    # V4 and the state backends take the wrapper-padded path.
    tables_self_padding: bool = False
    # DFLASH/DSpark block drafts expand decode metadata to spec_num_tokens
    # rows per request; set from config by the backends that support it.
    draft_block_decode: bool = False
    # Bound by register_step_counter (PD layerwise transfer); None otherwise.
    step_counter: StepCounter | None = None
    # Wrapper-owned (Inkling conv) groups: shed from write-loc math and
    # capture buffers; the wrapper registers them after construction.
    engine_owned_group_ids: frozenset[str] = frozenset()
    # Value for CUDA-graph buffer column tails past this replay's table
    # width. -1 is a debug tripwire (never read past cache_seqlens by the
    # MHA kernels); backends whose kernels assume a full-width table
    # (trtllm: row stride derived from max_kv_len) override with 0, the
    # zero-init dummy page — always safe to dereference.
    table_tail_pad: int = -1
    # The composed per-group graph buffers (GroupGraphBuffers); None until
    # _init_group_graph_buffers runs (grouped backends call it from
    # init_cuda_graph_state).
    group_graph: GroupGraphBuffers | None = None
    # The shared kv-indices graph buffer some MLA-family backends allocate;
    # the runner aliases a draft's to the target's when shapes match, so the
    # name is a cross-backend protocol. None = no such buffer.
    decode_cuda_graph_kv_indices: torch.Tensor | None = None
    # Metadata attribute names exempt from the capture-time pointer-identity
    # snapshot (graph_ptr_guard): sanctioned per-step-mutable objects the
    # replayed kernels do not read through Python (e.g. FlashMLA's eager tile
    # schedule). Keep empty unless a kernel imposes such an asymmetry.
    graph_unstable_metadata_fields: frozenset[str] = frozenset()
    # Static CUDA-graph capability of this backend class; the executor
    # AND-composes it over the target+draft trees at startup
    # (resolve_cuda_graph_support) and downgrades the graph subsystems once.
    cuda_graph_support: CudaGraphSupport = CudaGraphSupport()
    # Pool-learned group geometry (set_cache_pool); the empty default serves
    # unit fixtures that never bind a pool.
    _geometry: CacheGroupGeometry = CacheGroupGeometry()

    def __init__(self, config: AttnConfig, spec: SoftmaxAttnConfig) -> None:
        # ``spec`` is the component this backend serves; hybrid sub-backends
        # built over the softmax component's plumbing receive that spec.
        self.device = config.device
        self.num_qo_heads = spec.num_attention_heads // spec.attn_tp_size
        self.num_kv_heads = max(spec.num_kv_heads // spec.attn_tp_size, 1)
        self.dtype = config.dtype
        self.head_dim = spec.head_dim
        self.is_draft = config.is_draft
        self.spec_num_tokens = config.speculative_num_draft_tokens
        self.cache_pool: CachePool | None = None
        self._speculative_state_backends: list[SpeculativeStateBackend] = []
        # True when this backend's CUDA-graph block-table (kv_indices) buffer is
        # aliased to a peer backend's (e.g. a drafter sharing the target's), so
        # the replay path skips rebuilding it — the peer already populates it.
        self._page_table_aliased = False

    def set_cache_pool(self, cache_pool: CachePool) -> None:
        """Bind the pool and learn its group geometry in one step.

        The arena's published specs are the only source of group geometry,
        so binding is also when learning happens — no second seeding path
        that could answer differently on the eager arm. ``self._geometry``
        is the frozen value object every geometry consumer reads.
        """
        self.cache_pool = cache_pool
        self._geometry = learn_cache_group_geometry(
            getattr(cache_pool.arena, "cache_group_specs", ()),
            default_granularity=getattr(self, "kernel_page_size", 1),
        )

    def _expand_history_table(
        self, raw: torch.Tensor, out: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Expand a batch-ordered raw table (scheduler pages of the
        full-history grain) into this backend's kernel pages,
        ``self.max_num_pages`` wide (see cache_group_geometry)."""
        return expand_history_table(
            raw,
            history_block_granularity=(
                self._geometry.history_block_granularity or self.kernel_page_size
            ),
            kernel_page_size=self.kernel_page_size,
            max_kernel_pages=self.max_num_pages,
            out=out,
        )

    # ------------------------------------------------------------------
    # Per-group routing (dict-of-groups metadata: MHA/TRTLLM/MSA family)
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
        # prefer_caller: draft chains own per-step locs; metadata's single
        # loc would pin every step to one slot.
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

    def _consumed_group_ids(self) -> frozenset[str]:
        """Cache groups this backend consumes, claimed positively: the
        group's pool-learned family is one this backend declared in
        ``cache_consumer_families``, and no wrapper owns the group.
        Computed live — a wrapper may register owned groups after
        construction. Pools that never published families (unit fixtures,
        pre-contract draft pools) fall back to every row-geometry group.
        """
        families = self._geometry.families
        if not families:
            return frozenset(self._geometry.granularities) - self.engine_owned_group_ids
        return frozenset(
            gid
            for gid, family in families.items()
            if family in self.cache_consumer_families
            and gid not in self.engine_owned_group_ids
        )

    def _consumed_group_tables(self, tables):
        """Keep the delivered per-group tables this backend consumes
        (:meth:`_consumed_group_ids`); the rest of the dict rides to its own
        consumers (state pages to the mamba backend, wrapper-owned conv
        groups to the wrapper) — computing write locs / capture buffers over
        a foreign hole-heavy table writes the dummy page and trips
        TOKENSPEED_CACHE_DEBUG. A table for a group the bound pool never
        published is a delivery bug and raises. Pools that published no
        families pass through unfiltered minus wrapper-owned groups (the
        pre-contract draft path selects target-pool tables the draft's own
        geometry never learned). Returns None when nothing is left.
        """
        if not tables:
            return None
        families = self._geometry.families
        if not families:
            if not self.engine_owned_group_ids:
                return tables
            kept = {
                gid: table
                for gid, table in tables.items()
                if gid not in self.engine_owned_group_ids
            }
            return kept or None
        unknown = sorted(gid for gid in tables if gid not in families)
        if unknown:
            raise RuntimeError(
                f"{type(self).__name__}: delivered tables for groups "
                f"{unknown} the bound pool never published "
                f"(published: {sorted(families)})"
            )
        consumed = self._consumed_group_ids()
        kept = {gid: table for gid, table in tables.items() if gid in consumed}
        return kept or None

    def _group_block_granularity(self, gid: str) -> int:
        return self._geometry.granularity_of(gid)

    def _layer_page_size(self, layer) -> int:
        """Page size of the layer's cache group."""
        return self._group_block_granularity(layer.group_id)

    def _kernel_page_tables(self, page_tables):
        """Convert per-group page IDs to the page size consumed by the kernel.

        The group's page size was learned from the pool's specs
        (``set_cache_pool``), so no pool round-trip is needed. Identity when
        every group already matches its consumer page size (plain MHA).
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

    def _compute_decode_group_out_cache_locs(
        self, page_tables, seq_lens, num_tokens_per_req=1
    ):
        """Thin binding of :func:`decode_group_out_cache_locs` to this
        backend's learned granularities."""
        return decode_group_out_cache_locs(
            page_tables,
            seq_lens,
            self._group_block_granularity,
            num_tokens_per_req,
        )

    def _compute_extend_group_out_cache_locs(
        self, page_tables, extend_prefix_lens_cpu, extend_seq_lens_cpu
    ):
        """Thin binding of :func:`extend_group_out_cache_locs`."""
        return extend_group_out_cache_locs(
            page_tables,
            extend_prefix_lens_cpu,
            extend_seq_lens_cpu,
            self._group_block_granularity,
        )

    def _maybe_check_group_write_locs(self, page_tables, out_cache_locs):
        """TOKENSPEED_CACHE_DEBUG=1 gate over
        :func:`check_group_write_locs` (eager only — graph-padded batches
        would trip the non-hole assert on dummy rows)."""
        if not cache_debug_enabled():
            return
        check_group_write_locs(
            page_tables, out_cache_locs, self._group_block_granularity
        )

    # ------------------------------------------------------------------
    # Per-group CUDA-graph buffers (composed GroupGraphBuffers)
    # ------------------------------------------------------------------

    @property
    def cuda_graph_page_tables(self) -> dict[str, torch.Tensor]:
        """Attention-consumed groups' graph table views ({} before init).

        Consumer-page-grain paged-KV tables only — page vocabulary stays
        with kv-cache consumers; wrapper-owned groups ride the stack tail
        under ``cuda_graph_owned_block_tables``.
        """
        return self.group_graph.page_tables if self.group_graph is not None else {}

    @property
    def cuda_graph_owned_block_tables(self) -> dict[str, torch.Tensor]:
        """Wrapper-owned groups' block-granularity stack-tail table views
        ({} before init); Inkling's conv-table adoption reads these."""
        if self.group_graph is None:
            return {}
        return self.group_graph.owned_block_tables

    @property
    def cuda_graph_out_cache_locs(self) -> dict[str, torch.Tensor]:
        """Per-group graph write-location views ({} before init)."""
        return self.group_graph.out_cache_locs if self.group_graph is not None else {}

    def _init_group_graph_buffers(self, max_bs: int) -> None:
        """Build the composed GroupGraphBuffers; call from
        init_cuda_graph_state BEFORE any backend early return — refresh reads
        the table dict unconditionally for the published-groups contract
        check. Geometry is frozen by now (set_cache_pool runs first)."""
        self.group_graph = GroupGraphBuffers(
            self._geometry,
            consumed_group_ids=self._consumed_group_ids(),
            engine_owned_group_ids=frozenset(self.engine_owned_group_ids),
            consumer_page_size_of=self._consumer_page_size,
            max_bs=max_bs,
            max_num_pages=self.max_num_pages,
            kernel_page_size=self.kernel_page_size,
            spec_num_tokens=self.spec_num_tokens,
            device=self.device,
        )

    def _capture_group_views(self, bs: int, cache_group_ids, tokens_per_req: int = 1):
        """Capture-time per-group views (see GroupGraphBuffers.capture_views);
        the LIVE consumed set decides which delivered groups get views (a
        wrapper may register owned groups after buffer construction)."""
        return self.group_graph.capture_views(
            bs,
            cache_group_ids,
            tokens_per_req,
            consumed_group_ids=self._consumed_group_ids(),
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

    def child_backends(self) -> tuple[AttentionBackend, ...]:
        """Sub-backends this backend delegates metadata and forwards to.

        Composite wrappers (hybrid linear-attention, MSA hybrid, DSA,
        Inkling) override this; leaf backends return ``()``. Drives the
        CUDA-graph support resolution and the debug pointer-identity walk
        (graph_ptr_guard), so a wrapper that grows a new child must list it
        here.
        """
        return ()

    @contextmanager
    def override_num_extends(self, num_extends: int):
        """Temporarily override the decode-metadata slice discriminator for the
        wrapped block. Used by MLA backends to flip between drafter step 0
        (slice = [num_extends:]) and step 1+ (slice = [0:]).

        Default no-op for backends that fill separate prefill/decode metadata
        at init time.
        """
        yield

    def support_kv_cache_prewrite(
        self, forward_mode: ForwardMode | None = None
    ) -> bool:
        return False

    def select_out_cache_loc(self, layer, out_cache_loc, forward_mode=None):
        """Per-group write-location hook for out-of-backend KV writers
        (fused RoPE prewrite); identity for backends without cache
        groups. ``forward_mode`` picks the
        metadata slot for backends that prewrite on extend as well."""
        return out_cache_loc

    @abstractmethod
    def init_forward_metadata(self, *args, **kwargs):
        """Construct metadata for an extend/mixed (or idle warmup) forward.

        Decode metadata goes through :meth:`refresh_decode_metadata`; a pure
        DECODE call here is a contract violation.
        """
        raise NotImplementedError()

    def init_cuda_graph_state(self, max_bs: int, **kwargs):
        """Allocate the persistent decode buffers, sized by ``max_bs``
        (= max decode bs, never the capture ladder). Backends own their
        cache-seqlens buffer and copy the live lengths in at replay time.

        Every implementation accepts ``**kwargs`` — the runner passes the
        same extras to every backend (``cache_group_specs``,
        ``cache_group_page_counts``, ``max_tokens_per_req``,
        ``overlap_schedule_depth``) and a narrower signature TypeErrors at
        boot (pinned by the signature-conformance test).
        """
        raise NotImplementedError()

    def advance_draft_forward_metadata(self, seq_lens: torch.Tensor) -> None:
        """Publish the drafter's in-graph seq_lens edits into our own buffer.

        Copies into the backend-owned ``cuda_graph_seq_lens`` (one name for
        every backend); backends with distinct draft metadata or an inner
        backend override this.
        """
        buf = getattr(self, "cuda_graph_seq_lens", None)
        if buf is None:
            return
        bs = seq_lens.shape[0]
        buf[:bs].copy_(seq_lens[:bs])

    def bind_decode_views(self, bs: int, cache_group_ids: tuple[str, ...] = ()) -> None:
        """Build/bind the pointer-stable per-bs decode views before a capture.

        ``cache_group_ids`` names the cache groups whose page tables arrive
        at replay and pins the capture-time group set (a draft may consume a
        family subset of its buffers); empty for single-table backends. The
        base default is a no-op — refresh builds views lazily — and the
        group-routing backends override it so capture records the exact
        per-group views replay refreshes.
        """

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
        cache_group_ids: tuple[str, ...] = (),
        page_table: torch.Tensor | None = None,
        **kwargs,
    ):
        """Default capture: bind the per-bs views, then run the idle refresh.

        Capture never reads live tables — ``refresh_decode_metadata`` with
        ``actual_bs=0`` and ``for_graph_replay=True`` routes every row to the
        null page over the same persistent buffers replay refreshes, against
        the runner-seeded ``seq_lens`` (filled to ``max_tokens_per_req``,
        which is >= every verify floor, so the capture-side clamp equals the
        refresh-side one). ``page_table`` is the same address-stable staged
        table replay passes; its dummy rows are zero at capture. Idempotent —
        one capture runs it several times (warmups + re-inits). Override only
        for a genuine capture-only asymmetry (docs/design/unified_path.md,
        "Capture is inherited").
        """
        if not forward_mode.is_decode_or_idle():
            raise NotImplementedError(
                f"{type(self).__name__} CUDA graphs record decode only, "
                f"got {forward_mode}"
            )
        self.bind_decode_views(bs, cache_group_ids)
        self.refresh_decode_metadata(
            bs,
            0,
            req_pool_indices,
            seq_lens,
            forward_mode=forward_mode,
            page_table=page_table,
            for_graph_replay=True,
            **kwargs,
        )

    def refresh_decode_metadata(
        self,
        bs: int,
        actual_bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        *,
        forward_mode: ForwardMode,
        page_table: torch.Tensor | None = None,
        num_extends: int = 0,
        for_graph_replay: bool = False,
        **kwargs,
    ) -> None:
        """The single decode metadata path — eager decode and graph replay.

        Refreshes the backend's persistent decode buffers in place (``copy_``)
        and points ``forward_decode_metadata`` at views over them. There is
        deliberately no fresh-allocation decode path: capture runs this
        refresh's idle arm over the same buffers (see
        ``init_forward_metadata_capture_cuda_graph``), replay refreshes them
        before ``graph.replay()``, and eager decode refreshes them before
        running the same forward code the graph recorded.

        Args:
            bs: Rows to prepare. On graph replay this is the padded capture
                batch size; eager passes ``bs == actual_bs`` (unpadded).
            actual_bs: Live-request rows. Rows in ``[actual_bs, bs)`` are
                padding: the backend must route them to the null page / dummy
                slot so they never touch a live request's cache.
                ``actual_bs == 0`` is the idle replay and the capture seeding.
            req_pool_indices: ``[>=bs]`` request-pool slots (padding rows hold
                a sentinel or slot 0 per the wrapper's padding contract).
            seq_lens: ``[>=bs]`` live cache lengths (padding rows hold 1).
            forward_mode: A decode mode; extend/mixed metadata stays on
                ``init_forward_metadata``.
            page_table: Batch-ordered table for backends outside the
                cache-group contract (and the draft's staged table).
            num_extends: Leading extend rows of a MIXED batch whose decode
                half this refresh describes; 0 for pure decode.
            for_graph_replay: True whenever a graph is in play — live replay
                AND the capture default's idle refresh. Sanctioned branches
                on it are graph-mechanics asymmetries only (FlashMLA's tile
                schedule, DFLASH block-arm seeding); see unified_path.md.
            **kwargs: Cache-contract extras — ``block_tables``,
                ``block_table_base_offsets``, ``cache_metadata``,
                ``forward_batch``, ``num_tokens``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement refresh_decode_metadata"
        )

    def fill_block_decode_seq_lens(self, bs: int, block_seq_lens: torch.Tensor) -> None:
        """DFLASH: broadcast each request's block-end length to its
        spec_num_tokens cuda-graph decode rows (uniform, non-causal).

        Called by the drafter inside the captured graph so that on every
        replay the expanded seq_lens re-derive from the live draft length
        (recomputed in-graph from the target's accept lengths). Backends
        whose kernel repeats one row per request across the block's queries
        (trtllm_mla, flashmla) override with the bs-row geometry.

        Args:
            bs: Number of draft requests.
            block_seq_lens: ``[bs]`` per-request block-end lengths
                (prefix + spec_num_tokens).
        """
        spec = self.spec_num_tokens
        self.cuda_graph_seq_lens[: bs * spec].view(bs, spec).copy_(
            block_seq_lens[:bs].clamp(spec, self.max_context_len).unsqueeze(1)
        )

    def init_prefill_graph_state(self, max_num_tokens: int, max_bs: int) -> None:
        """Allocate static buffers the breakable prefill graphs bake.

        Called once before prefill-graph capture. Default: no-op — most
        backends' extend metadata needs no graph-persistent state (attention
        stays eager at the break points); Inkling overrides to allocate its
        static conv metadata.
        """

    def update_mamba_state_after_mtp_verify(self, accepted_lengths, model) -> None:
        """Commit recurrent-state pages after MTP verification.

        Called by the runner after every spec-decode round. Default: no-op —
        only backends with Mamba/GDN state (the hybrid wrapper) override.
        """

    def configure_runtime(self, **kwargs) -> None:
        """Configure runtime state after model loading (e.g. sliding_window_size).

        Called once during ModelExecutor initialization with information that is
        not available at backend construction time.  Default: no-op.
        """
        pass

    def prepare_remote_cache_slots(self, slot_indices: list[int]) -> None:
        """Clear model-specific restore state before remote cache admission."""
        del slot_indices

    def mark_remote_cache_ready(self, slot_index: int) -> None:
        """Arm model-specific hydration after a remote cache transfer succeeds."""
        del slot_index

    def register_step_counter(self, step_counter: StepCounter):
        self.step_counter = step_counter

    def register_speculative_state_backend(
        self, backend: SpeculativeStateBackend
    ) -> None:
        """Register a model side-state consumer of MTP verification results.

        Args:
            backend: Side backend implementing ``commit_after_mtp_verify``.

        Returns:
            None.
        """

        # Some composite backends predate this registry and intentionally do
        # not call ``AttentionBackend.__init__``.  Initialize lazily so model
        # side-state remains usable through those wrappers as well.
        backends = getattr(self, "_speculative_state_backends", None)
        if backends is None:
            backends = []
            self._speculative_state_backends = backends
        if backend not in backends:
            backends.append(backend)

    def find_speculative_state_backend(
        self, backend_type: type[_SpeculativeStateBackendT]
    ) -> _SpeculativeStateBackendT | None:
        """Return the registered speculative side backend of ``backend_type``.

        Args:
            backend_type: Concrete side-backend type to locate.

        Returns:
            The first matching backend, or ``None`` when it is not registered.
        """

        return next(
            (
                backend
                for backend in getattr(self, "_speculative_state_backends", ())
                if isinstance(backend, backend_type)
            ),
            None,
        )

    def commit_speculative_state_after_verify(
        self,
        accepted_lengths: torch.Tensor,
        *,
        num_extends: int,
    ) -> None:
        """Publish MTP accept/reject results to registered model side-state.

        Args:
            accepted_lengths: Per-request accepted lengths from the sampler.
            num_extends: Number of leading extend requests in a mixed batch.

        Returns:
            None.
        """

        for backend in getattr(self, "_speculative_state_backends", ()):
            backend.commit_after_mtp_verify(
                accepted_lengths,
                num_extends=num_extends,
            )

    @contextmanager
    def record_pd_cache_step(
        self,
        forward_mode: ForwardMode,
        save_kv_cache: bool,
        record_kv_cache: bool | None,
    ):
        """Anchor the PD layerwise cache-step record to the wrapped KV write.

        Records the ``StepCounter`` step before the attention call when the KV
        was pre-written (``save_kv_cache=False``) and after it otherwise, so a
        layerwise cache transfer always observes a fully written layer. See
        ``forward`` for the ``record_kv_cache`` override contract. No-op when no
        step counter is registered. Backends that own the record (e.g. the
        hybrid wrapper, which counts once per model layer across full-attn +
        mamba children) reuse this to avoid duplicating the gate logic.
        """
        if record_kv_cache is None:
            record_cache = not forward_mode.is_decode() and not forward_mode.is_idle()
        else:
            record_cache = record_kv_cache
        record_cache = record_cache and self.step_counter is not None

        if record_cache and not save_kv_cache:
            self.step_counter.record_cache()
        yield
        if record_cache and save_kv_cache:
            self.step_counter.record_cache()

    @break_point
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool: CachePool,
        forward_mode: ForwardMode,
        bs: int,
        save_kv_cache: bool = True,
        record_kv_cache: bool | None = None,
        **kwargs,
    ):
        """Run forward on an attention layer with explicit scheduler metadata.

        ``record_kv_cache`` overrides the PD layerwise cache-step recording:
        ``None`` keeps the default (record on the EXTEND-side path), an explicit
        bool forces it so a DECODE-dispatched draft catch-up can still record.
        """
        with self.record_pd_cache_step(forward_mode, save_kv_cache, record_kv_cache):
            if forward_mode.is_decode():
                ret = self.forward_decode(
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
            else:
                ret = self.forward_extend(
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
        return ret

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool: CachePool,
        bs: int,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        """Run a forward for decode."""
        raise NotImplementedError()

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool: CachePool,
        bs: int,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        """Run a forward for extend."""
        raise NotImplementedError()
