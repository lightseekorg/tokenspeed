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

"""The runner-facing attention backend contract.

An ``AttentionBackend`` is what ``ForwardStepRunner`` / ``ModelExecutor`` /
``PrefillGraph`` talk to: it receives the scheduler bridge's per-group
``block_tables`` (raw scheduler blocks, batch-ordered) on every metadata
call and answers the model's ``forward``. Three kinds of node implement it:

* ``CacheGroupRouter`` — the paged-KV composite that maps blocks to kernel
  pages once and fans out to ``PagedAttentionBackend`` leaves (``paged.py``);
* composites that wrap a router next to a state consumer (hybrid GDN/KDA,
  Inkling's conv columns, MSA's sparse layers);
* block consumers that read raw tables of their own state groups (Mamba /
  KDA state paging, DeepSeek-V4's bespoke multi-group backend).

``block_tables`` is always a complete mapping — the runner synthesizes
placeholder tables for capture, idle and warmup — so no implementation
carries a "no tables" arm. Padding is the consumer's job: requests in
``[actual_bs, bs)`` are dummies it must route to the null page itself.
"""

from __future__ import annotations

from abc import ABC
from collections.abc import Mapping
from contextlib import contextmanager
from typing import TYPE_CHECKING, Protocol, TypeVar

import torch

from tokenspeed.runtime.execution.breakable_cuda_graph import break_point
from tokenspeed.runtime.layers.attention.backends.support import (  # noqa: F401
    CudaGraphSupport,
    resolve_cuda_graph_support,
)

if TYPE_CHECKING:
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.layers.attention.configs.base import (
        AttnConfig,
        SoftmaxAttnConfig,
    )
    from tokenspeed.runtime.layers.attention.kv_cache.base import CachePool
    from tokenspeed.runtime.layers.paged_attention import PagedAttention
    from tokenspeed.runtime.pd.utils import StepCounter


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
    """The runner-facing contract; see the module docstring."""

    # Cache families this node consumes from the pool contract (startup
    # validation: every published family must have a consumer); composites
    # union their children's.
    cache_consumer_families: frozenset[str] = frozenset({"history"})
    supports_mla_projected_value_decode: bool = False
    # Bound by register_step_counter (PD layerwise transfer); None otherwise.
    step_counter: StepCounter | None = None
    # Static CUDA-graph capability of this class; the executor AND-composes
    # it over the target+draft trees (resolve_cuda_graph_support).
    cuda_graph_support: CudaGraphSupport = CudaGraphSupport()

    def __init__(self, config: AttnConfig, spec: SoftmaxAttnConfig) -> None:
        self.device = config.device
        self.dtype = config.dtype
        self.is_draft = bool(config.is_draft)
        self.spec_num_tokens = max(int(config.speculative_num_draft_tokens or 1), 1)
        self.num_qo_heads = spec.num_attention_heads // spec.attn_tp_size
        self.num_kv_heads = max(spec.num_kv_heads // spec.attn_tp_size, 1)
        self.head_dim = spec.head_dim
        self.cache_pool: CachePool | None = None
        self._speculative_state_backends: list[SpeculativeStateBackend] = []

    # ------------------------------------------------------------------
    # Structure
    # ------------------------------------------------------------------

    def set_cache_pool(self, cache_pool: CachePool) -> None:
        """Bind the pool whose buffers this node's kernels read."""
        self.cache_pool = cache_pool

    def child_backends(self) -> tuple[AttentionBackend, ...]:
        """Sub-backends this node delegates to (drives the CUDA-graph
        support resolution and the pointer-identity walk); leaves return
        ``()``."""
        return ()

    def configure_runtime(self, **kwargs) -> None:
        """Post-load configuration hook (information unavailable at
        construction, e.g. sliding window sizes). Default: no-op."""

    def init_prefill_graph_state(self, max_num_tokens: int, max_bs: int) -> None:
        """Allocate static buffers the breakable prefill graphs bake.
        Default: no-op — attention stays eager at the break points."""

    # ------------------------------------------------------------------
    # Metadata (docs/design/unified_path.md)
    # ------------------------------------------------------------------

    def init_cuda_graph_state(self, max_bs: int, **kwargs) -> None:
        """Allocate the persistent decode buffers, sized by ``max_bs`` (the
        max decode bs, never the capture ladder). Runs unconditionally at
        wrapper construction, ``enforce_eager`` included.

        Args:
            max_bs: Persistent-buffer row capacity.
            **kwargs: Runner extras every node accepts
                (``cache_group_specs``, ``cache_group_page_counts``,
                ``max_tokens_per_req``, ``overlap_schedule_depth``); a
                narrower signature TypeErrors at boot.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement init_cuda_graph_state"
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
        """Build metadata for an extend / mixed (or idle warmup) forward.

        Decode metadata goes through :meth:`refresh_decode_metadata`; a pure
        DECODE call here is a contract violation.

        Args:
            bs: Requests in the batch (extend requests first, then decode
                requests).
            num_extends: Leading extend requests.
            req_pool_indices: ``[>= bs]`` request-pool slots.
            seq_lens: ``[>= bs]`` total cache lengths after this step.
            forward_mode: EXTEND, MIXED or IDLE.
            block_tables: ``group_id -> [>= bs, cols]`` int32 raw scheduler
                tables for every published group (placeholders on warmup).
            extend_*: ``[>= num_extends]`` per-request new-token / prefix
                lengths and their pinned host mirrors (empty on idle warmup).
            extend_with_prefix: Whether any extend row continues a cached or
                chunked prefix (some ``extend_prefix_lens`` entry is non-zero).
            **kwargs: Model-side extras (positions, capture mode, ...) a
                node may ignore.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement init_forward_metadata"
        )

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
        """The single decode metadata path — eager decode and graph replay.

        Refreshes the persistent decode buffers in place; there is no
        fresh-allocation decode arm anywhere (capture runs the idle arm of
        this refresh, replay refreshes before ``graph.replay()``, eager
        refreshes before the same forward the graph recorded).

        Args:
            bs: Requests to prepare (the padded capture batch under replay);
                eager passes ``bs == actual_bs``.
            actual_bs: Live requests; ``[actual_bs, bs)`` are padding the node
                routes to the null page. ``0`` is the idle replay / capture
                seeding.
            req_pool_indices: ``[>= bs]`` request-pool slots.
            seq_lens: ``[>= bs]`` live cache lengths (padding requests hold 1).
            forward_mode: A decode mode.
            block_tables: ``group_id -> [>= actual_bs, cols]`` raw scheduler
                tables for every published group (placeholders when idle).
            num_extends: Leading extend requests of a MIXED round whose decode
                half this refresh describes; 0 for pure decode.
            for_graph_replay: A graph is in play (live replay or capture
                seeding). Branch on it only for graph-mechanics asymmetries.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement refresh_decode_metadata"
        )

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
        """Default capture: the idle refresh (``actual_bs=0``,
        ``for_graph_replay=True``) over the same persistent buffers replay
        refreshes, against the runner's placeholder tables and the seeded
        ``seq_lens``. Idempotent. Override only for a kernel-imposed
        capture asymmetry (unified_path.md, "Capture is inherited")."""
        if not forward_mode.is_decode_or_idle():
            raise NotImplementedError(
                f"{type(self).__name__} CUDA graphs record decode only, got {forward_mode}"
            )
        self.refresh_decode_metadata(
            bs,
            0,
            req_pool_indices,
            seq_lens,
            forward_mode=forward_mode,
            block_tables=block_tables,
            for_graph_replay=True,
            **kwargs,
        )

    def advance_draft_forward_metadata(self, seq_lens: torch.Tensor) -> None:
        """Publish a drafter's in-graph seq_lens edits (Eagle chains); nodes
        without per-request decode lengths ignore it."""

    def update_draft_forward_metadata(self, frontier: torch.Tensor) -> None:
        """Vanilla MTP re-anchors the draft requests to the committed frontier."""

    def fill_block_decode_seq_lens(self, bs: int, block_seq_lens: torch.Tensor) -> None:
        """DFLASH: broadcast block-end lengths to each request's materialized
        decode entries."""

    @contextmanager
    def override_num_extends(self, num_extends: int):
        """Temporarily override the decode-row slice discriminator (MLA
        family). Default no-op."""
        yield

    def update_mamba_state_after_mtp_verify(self, accepted_lengths) -> None:
        """Commit recurrent-state pages after MTP verification; only nodes
        with Mamba/GDN state override."""

    def support_kv_cache_prewrite(
        self, forward_mode: ForwardMode | None = None
    ) -> bool:
        return False

    # ------------------------------------------------------------------
    # Write locations
    # ------------------------------------------------------------------

    def draft_write_locations_uniform(
        self, out: torch.Tensor, cache_start: torch.Tensor, num_tokens: int
    ) -> torch.Tensor:
        """Resolve ``num_tokens`` KV write slots per request into a
        caller-owned scratch buffer (in-graph safe: fixed table address).

        For slots a FORWARD will consume, use
        :meth:`publish_draft_step_locations` instead — this variant serves
        side writes that must not clobber the published step window (e.g.
        DFLASH copying target-row KV into the draft cache).
        """
        raise NotImplementedError(
            f"{type(self).__name__} owns no draft write locations"
        )

    def publish_draft_step_locations(
        self, cache_start: torch.Tensor, num_tokens: int
    ) -> torch.Tensor:
        """Publish a draft step's write window (``num_tokens`` slots per
        request starting at ``cache_start``) so the next forward's
        ``write_locations`` serve it. The drafters declare the window; the
        math and the address-stable storage live in the backend. In-graph
        safe."""
        raise NotImplementedError(
            f"{type(self).__name__} owns no draft write locations"
        )

    def write_locations(
        self, layer: PagedAttention, forward_mode: ForwardMode
    ) -> torch.Tensor:
        """This layer's KV write slots for the requests the forward covers —
        the one accessor for writers outside the backend (fused RoPE
        prewrite, model-side MLA cache writes)."""
        raise NotImplementedError(
            f"{type(self).__name__} owns no paged write locations"
        )

    # ------------------------------------------------------------------
    # PD / speculative side state
    # ------------------------------------------------------------------

    def prepare_remote_cache_slots(self, slot_indices: list[int]) -> None:
        """Clear model-specific restore state before remote cache admission."""
        del slot_indices

    def mark_remote_cache_ready(self, slot_index: int) -> None:
        """Arm model-specific hydration after a remote cache transfer succeeds."""
        del slot_index

    def register_step_counter(self, step_counter: StepCounter) -> None:
        self.step_counter = step_counter

    def register_speculative_state_backend(
        self, backend: SpeculativeStateBackend
    ) -> None:
        """Register a model side-state consumer of MTP verification results."""
        backends = getattr(self, "_speculative_state_backends", None)
        if backends is None:
            backends = []
            self._speculative_state_backends = backends
        if backend not in backends:
            backends.append(backend)

    def find_speculative_state_backend(
        self, backend_type: type[_SpeculativeStateBackendT]
    ) -> _SpeculativeStateBackendT | None:
        """The registered side backend of ``backend_type``, or None."""
        return next(
            (
                backend
                for backend in getattr(self, "_speculative_state_backends", ())
                if isinstance(backend, backend_type)
            ),
            None,
        )

    def commit_speculative_state_after_verify(
        self, accepted_lengths: torch.Tensor, *, num_extends: int
    ) -> None:
        """Publish MTP accept/reject results to registered model side-state."""
        for backend in getattr(self, "_speculative_state_backends", ()):
            backend.commit_after_mtp_verify(accepted_lengths, num_extends=num_extends)

    @contextmanager
    def record_pd_cache_step(
        self,
        forward_mode: ForwardMode,
        save_kv_cache: bool,
        record_kv_cache: bool | None,
    ):
        """Anchor the PD layerwise cache-step record to the wrapped KV write:
        before the attention call when the KV was pre-written
        (``save_kv_cache=False``), after it otherwise. No-op without a step
        counter."""
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
        """Run one attention layer; KV write locations come from
        :meth:`write_locations` (there is no caller-supplied location
        vector). ``record_kv_cache`` overrides the PD layerwise recording
        (None: record on the extend-side path)."""
        out_cache_loc = self.write_locations(layer, forward_mode)
        with self.record_pd_cache_step(forward_mode, save_kv_cache, record_kv_cache):
            if forward_mode.is_decode():
                return self.forward_decode(
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
            return self.forward_extend(
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
        raise NotImplementedError()
