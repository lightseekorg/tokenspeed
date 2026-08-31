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

import inspect
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypeVar

import torch

from tokenspeed.runtime.execution.breakable_cuda_graph import break_point
from tokenspeed.runtime.utils import get_colorful_logger

if TYPE_CHECKING:
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.layers.attention.configs.base import BaseAttnConfig
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
    declarations only; runtime capture failures keep their own degrade paths
    (``PrefillGraph._capture_unanimous``).
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


def init_backend_cuda_graph_state(
    backend: "AttentionBackend",
    max_bs: int,
    **extras,
) -> None:
    """Call ``backend.init_cuda_graph_state`` with only the kwargs its
    signature accepts (VAR_KEYWORD accepts all of them).

    Signature-probe instead of try/except TypeError: cache_group_specs
    is load-bearing for the state shed, so a TypeError raised from inside the
    backend's body must propagate rather than silently retry without specs.

    Shared by the cuda-graph wrapper and by composite backends (hybrid) that
    forward to user-selectable sub-backends with possibly narrow signatures.
    """
    params = inspect.signature(backend.init_cuda_graph_state).parameters
    if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        extras = {k: v for k, v in extras.items() if k in params}
    backend.init_cuda_graph_state(max_bs, **extras)


class AttentionBackend(ABC):
    """The base class of attention backends"""

    # Capture helpers use a real writable page for every active group when
    # the backend rejects the reserved null page for live sequence metadata.
    cache_active_pages_must_be_real: bool = False
    supports_mla_projected_value_decode: bool = False
    # Backend-owned cuda-graph cache-seqlens buffer the decode metadata views.
    draft_seq_lens_attr: str = "cuda_graph_seq_lens"
    # Metadata attribute names exempt from the capture-time pointer-identity
    # snapshot (graph_ptr_guard): sanctioned per-step-mutable objects the
    # replayed kernels do not read through Python (e.g. FlashMLA's eager tile
    # schedule). Keep empty unless a kernel imposes such an asymmetry.
    graph_unstable_metadata_fields: frozenset[str] = frozenset()
    # Static CUDA-graph capability of this backend class; the executor
    # AND-composes it over the target+draft trees at startup
    # (resolve_cuda_graph_support) and downgrades the graph subsystems once.
    cuda_graph_support: CudaGraphSupport = CudaGraphSupport()

    def __init__(self, config: BaseAttnConfig) -> None:
        self.device = config.device
        self.num_qo_heads = config.num_attention_heads // config.attn_tp_size
        self.num_kv_heads = max(config.num_kv_heads // config.attn_tp_size, 1)
        self.dtype = config.dtype
        self.head_dim = config.head_dim
        self.is_draft = config.is_draft
        self.spec_num_tokens = config.speculative_num_draft_tokens
        self.cache_pool: CachePool | None = None
        self._speculative_state_backends: list[SpeculativeStateBackend] = []
        # True when this backend's CUDA-graph block-table (kv_indices) buffer is
        # aliased to a peer backend's (e.g. a drafter sharing the target's), so
        # the replay path skips rebuilding it — the peer already populates it.
        self._page_table_aliased = False

    def set_cache_pool(self, cache_pool: CachePool) -> None:
        self.cache_pool = cache_pool

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

    @property
    def sinks_dtype(self) -> torch.dtype:
        return torch.bfloat16

    @abstractmethod
    def init_forward_metadata(self, *args, **kwargs):
        """Construct metadata for an extend/mixed (or idle warmup) forward.

        Decode metadata goes through :meth:`refresh_decode_metadata`; a pure
        DECODE call here is a contract violation.
        """
        raise NotImplementedError()

    def init_cuda_graph_state(self, max_bs: int):
        """Init the global shared states for cuda graph. Backends own their
        cache-seqlens buffer and copy the live lengths in at replay time."""
        raise NotImplementedError()

    def advance_draft_forward_metadata(self, seq_lens: torch.Tensor) -> None:
        """Publish the drafter's in-graph seq_lens edits into our own buffer.

        Copies into ``draft_seq_lens_attr``; backends with distinct draft
        metadata or an inner backend override this.
        """
        buf = getattr(self, self.draft_seq_lens_attr, None)
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
        cache-group mixins override it so capture records the exact
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
        deliberately no fresh-allocation decode path: capture allocates and
        seeds the buffers, replay refreshes them before ``graph.replay()``,
        and eager decode refreshes them before running the same forward code
        the graph recorded.

        Args:
            bs: Rows to prepare. On graph replay this is the padded capture
                batch size; eager passes ``bs == actual_bs`` (unpadded).
            actual_bs: Live-request rows. Rows in ``[actual_bs, bs)`` are
                padding: the backend must route them to the null page / dummy
                slot so they never touch a live request's cache.
                ``actual_bs == 0`` is the idle replay.
            req_pool_indices: ``[>=bs]`` request-pool slots (padding rows hold
                a sentinel or slot 0 per the wrapper's padding contract).
            seq_lens: ``[>=bs]`` live cache lengths (padding rows hold 1).
            forward_mode: A decode mode; extend/mixed metadata stays on
                ``init_forward_metadata``.
            page_table: Batch-ordered table for backends outside the
                cache-group contract (and the draft's staged table).
            num_extends: Leading extend rows of a MIXED batch whose decode
                half this refresh describes; 0 for pure decode.
            for_graph_replay: True only under graph replay. The only
                sanctioned use is a kernel-imposed asymmetry (e.g. FlashMLA
                must swap in a fresh tile-schedule object per eager step
                because the kernel freezes the schedule on first use, while
                the captured schedule-build re-runs inside the graph).
            **kwargs: Cache-contract extras — ``block_tables``,
                ``block_table_base_offsets``, ``cache_metadata``,
                ``forward_batch``, ``num_tokens``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement refresh_decode_metadata"
        )

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
        record_cache = record_cache and getattr(self, "step_counter", None) is not None

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
