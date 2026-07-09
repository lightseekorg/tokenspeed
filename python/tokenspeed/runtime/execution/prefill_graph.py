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

"""Breakable CUDA graphs for prefill (extend) forwards.

:class:`PrefillGraph` holds one breakable graph per padded token bucket
(captured from a dummy bs=1 extend batch). The embedding lookup stays OUTSIDE
the captured region: graphs start from a static input-embeds buffer, filled at
replay by an eager ``embed_tokens`` gather (text) or by precomputed merged
embeddings (multimodal, via the model's ``multimodal_input_embeds`` seam).
Constructed after the decode
:class:`~tokenspeed.runtime.execution.cuda_graph_wrapper.CudaGraphWrapper`,
borrowing its capture stream so all graphs share one mempool. At serving time
the executor's target-forward dispatch is a flat
three-way -- decode & captured replays the decode graph (one level up, since
it captures the whole step), prefill & captured replays here (:meth:`can_run`
/ :meth:`replay`), everything else runs the eager model forward.

Unlike decode (whole forward captured, keyed by batch size), the captured
region here is purely token-shaped compute keyed by total token count:
attention runs as an eager break (see
:mod:`tokenspeed.runtime.execution.breakable_cuda_graph`), so one graph per
bucket serves any batch size at that token count, and a replayed forward is
finished with the model's eager logits tail.
"""

from __future__ import annotations

import bisect
from contextlib import contextmanager
from typing import TYPE_CHECKING, NamedTuple

import torch

from tokenspeed.runtime.execution import cuda_graph_wrapper as _cuda_graph_wrapper
from tokenspeed.runtime.execution.breakable_cuda_graph import (
    BreakableCapture,
    active_forward,
)
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
)
from tokenspeed.runtime.layers.logits_processor import LogitsMetadata
from tokenspeed.runtime.utils import get_colorful_logger
from tokenspeed.runtime.utils.common import maybe_inference_mode

logger = get_colorful_logger(__name__)

if TYPE_CHECKING:
    from tokenspeed.runtime.execution.cuda_graph_wrapper import CudaGraphWrapper
    from tokenspeed.runtime.execution.input_buffer import InputBuffers
    from tokenspeed.runtime.execution.model_executor import ModelExecutorConfig
    from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend


# Smallest prefill bucket; below this, denser rungs would only add capture time.
PREFILL_BUCKET_FLOOR: int = 16

# Relative rung spacing (largest pow2 <= size/8), bounding the padded tail at ~12.5%.
PREFILL_BUCKET_STEP_DIVISOR: int = 8

# Absolute rung-spacing cap, bounding the worst case at the top of the ladder.
PREFILL_BUCKET_MAX_STEP: int = 512


def get_prefill_token_buckets(config: ModelExecutorConfig) -> list[int]:
    """Padded token-count buckets to capture for the breakable prefill graph.

    Unlike decode (keyed by batch size), the breakable prefill graph captures
    pure token-shaped compute, so it is keyed by total token count. A live extend
    forward is padded up to the smallest bucket >= its token count; forwards above
    the largest bucket run eager.

    Returns an empty list (graph disabled) when ``disable_prefill_graph`` is set or
    ``prefill_graph_max_tokens <= 0``. The largest bucket is clamped to the
    chunked-prefill size: the scheduler's per-forward token budget
    (``max_scheduled_tokens`` = chunked-prefill size) covers extends AND any fused
    decode rows -- with mixed batching, decodes are scheduled first and each
    decrements the budget, and the prefill chunk is sized to what remains
    (scheduler ``newForwardOperation``/``push_op``) -- so no forward, mixed or
    pure, ever exceeds the chunk. No headroom above it is needed.

    The default ladder bounds RELATIVE padding waste: a forward pads its graphed
    compute to the next bucket, so what matters is the gap as a fraction of the
    size -- a flat stride is needlessly coarse for short prompts and needlessly
    dense at the top. Each bucket's step is the largest power of two <= size/8
    (padded tail at most ~12.5% anywhere on the ladder), floored at 16 tokens and
    capped at 512 so the absolute worst case stays bounded at the top end. Dense
    ladders are cheap: all captures share one stream + mempool, so graph memory
    is ~the largest bucket's peak regardless of bucket count (see
    ``BreakableCapture``); the remaining cost is ~0.5s of startup capture per
    bucket.

    ``prefill_graph_capture_sizes`` overrides the ladder with an explicit list
    (mirroring decode's ``cudagraph_capture_sizes``) -- e.g. a short list for
    faster startup on dev boots; sizes are clamped to the largest bucket.

    Args:
        config: The model-executor config carrying ``disable_prefill_graph``,
            ``prefill_graph_max_tokens``, ``prefill_graph_capture_sizes`` and
            ``chunked_prefill_size``.

    Returns:
        Sorted ascending list of token-bucket sizes (possibly empty).
    """
    max_tokens = int(config.prefill_graph_max_tokens or 0)
    if config.disable_prefill_graph or max_tokens <= 0:
        return []
    chunk = int(config.chunked_prefill_size or 0)
    if chunk > 0:
        max_tokens = min(max_tokens, chunk)
    explicit = config.prefill_graph_capture_sizes
    if explicit:
        buckets = {int(b) for b in explicit if 0 < int(b) <= max_tokens}
        buckets.add(max_tokens)
        return sorted(buckets)
    buckets = []
    size = min(PREFILL_BUCKET_FLOOR, max_tokens)
    while size < max_tokens:
        buckets.append(size)
        size += _prefill_bucket_step(size)
    buckets.append(max_tokens)
    return sorted(set(buckets))


def _prefill_bucket_step(size: int) -> int:
    """Distance from bucket ``size`` to the next rung.

    The largest power of two <= ``size / PREFILL_BUCKET_STEP_DIVISOR`` (so the
    padded tail stays within ~1/8 of the real token count), clamped between
    ``PREFILL_BUCKET_FLOOR`` and ``PREFILL_BUCKET_MAX_STEP``.
    """
    relative = size // PREFILL_BUCKET_STEP_DIVISOR
    if relative <= PREFILL_BUCKET_FLOOR:
        return PREFILL_BUCKET_FLOOR
    largest_pow2 = 1 << (relative.bit_length() - 1)
    return min(largest_pow2, PREFILL_BUCKET_MAX_STEP)


class CapturedForward(NamedTuple):
    """A bucket's captured inner-forward outputs (stable pool addresses)."""

    # Final hidden states with shape [bucket, hidden]; padded tail is garbage.
    hidden_states: torch.Tensor

    # Aux hidden states for drafting, each [bucket, hidden]; None when mode is NULL.
    aux_hidden_states: list[torch.Tensor] | None

    def sliced(self, num_tokens: int) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        """The leading real-token rows, in the (hidden, aux) shape callers expect."""
        hidden = self.hidden_states[:num_tokens]
        if self.aux_hidden_states is None:
            return hidden, None
        return hidden, [a[:num_tokens] for a in self.aux_hidden_states]


class PrefillGraph:
    """The breakable prefill (extend) CUDA graphs.

    A pure graph object -- :meth:`can_run` / :meth:`replay` -- holding no
    reference to any other component. Constructed AFTER the decode
    ``CudaGraphWrapper`` and captures in ``__init__`` like it: the decode
    wrapper is used transiently for its capture stream and dummy paged-cache
    tables, not kept. (The executor's target-forward dispatch therefore mode-
    checks before touching this object -- decode capture runs that dispatch
    while this object does not exist yet.) The dispatch checks :meth:`can_run`
    and calls :meth:`replay`; the eager path stays a direct
    ``model_runner.forward`` call at that call site. Capture failure degrades
    to eager -- world-agreed, so DP/TP ranks stay in lockstep.

    Args:
        model_runner: The target ModelRunner. Supplies the loaded model
            (multimodal wrappers are unwrapped internally: the graph wraps the
            nested ``language_model``'s text transformer, image prefills run
            eager) and ``is_generation`` (embedding models run eager).
        attn_backend: Backend whose extend metadata the dummy capture batch sets.
        token_to_kv_pool: KV pool the dummy batch points at (reserved dummy slot).
        input_buffers: The shared static input buffers the graphs read from.
        config: Model-executor config (buckets, DP/world topology, device).
        req_to_page: Request page table; row 0 backs the dummy capture request.
        drafter: If present, aux-hidden capture (EAGLE3/MTP) is baked into the
            captured graphs.
    """

    def __init__(
        self,
        model_runner,
        attn_backend: AttentionBackend,
        token_to_kv_pool,
        input_buffers: InputBuffers,
        config: ModelExecutorConfig,
        req_to_page: torch.Tensor | None,
        drafter=None,
        decode_wrapper: CudaGraphWrapper | None = None,
        num_warmup: int = 3,
    ) -> None:
        model = model_runner.model if model_runner is not None else None
        # Multimodal seam: models whose multimodal path is embeds-only expose
        # multimodal_input_embeds; others (e.g. deepstack) replay text only.
        self._multimodal_input_embeds = getattr(model, "multimodal_input_embeds", None)
        self.text_model = (
            model.language_model if hasattr(model, "language_model") else model
        )
        self.inner_model = getattr(self.text_model, "model", None)
        # Embedding runs eagerly OUTSIDE the graphs (see capture); the graphs
        # read a static input-embeds buffer instead of gathering from input_ids.
        self._embed_tokens = getattr(self.inner_model, "embed_tokens", None)
        self._input_embeds_buf: torch.Tensor | None = None
        self.attn_backend = attn_backend
        self.token_to_kv_pool = token_to_kv_pool
        self.input_buffers = input_buffers
        self.config = config
        self.req_to_page = req_to_page
        self.drafter = drafter
        self.num_warmup = num_warmup
        self.dp_size = config.data_parallel_size

        self.capture_buckets = get_prefill_token_buckets(config)
        self.disable = (
            config.enforce_eager
            or config.disable_prefill_graph
            or not self.capture_buckets
            or self.inner_model is None
            or self._embed_tokens is None
            or model_runner is None
            or not model_runner.is_generation
            # DP replay decisions must come from replicated state, and a
            # forward's multimodal-ness is rank-local: one rank running its mm
            # prefill eager while text-only peers replay desyncs the EP
            # collectives. Until the DP metadata gather carries a multimodal
            # flag, keep the graph off for multimodal models under DP.
            or (config.data_parallel_size > 1 and model_runner.is_multimodal)
        )

        self._ctx: ForwardContext | None = None
        self._pool = None
        self._engaged_logged: set[str] = set()
        # Aux-capture mode baked into the graphs; mismatched live forwards run eager.
        self._captured_hidden_mode = None
        # One captured graph + bucket-sized output per padded token bucket.
        self._captures: dict[int, BreakableCapture] = {}
        self._outputs: dict[int, CapturedForward] = {}

        if not self.disable:
            self.capture(decode_wrapper)

    # ------------------------------------------------------------------
    # Graph capture
    # ------------------------------------------------------------------

    def capture(self, decode_wrapper: CudaGraphWrapper | None = None) -> None:
        """Capture one breakable graph per token bucket (no-op when disabled).

        Called from ``__init__``; ``decode_wrapper`` supplies the shared
        capture stream and dummy paged-cache block tables (used here only,
        not stored). Sharing
        decode's stream + mempool puts all graphs in one pool-reuse domain
        (pool blocks are stream-keyed), so graph memory stays ~the largest
        bucket's peak instead of growing per bucket.

        Runs under inference mode like serving forwards (in-place updates on
        inference-mode model state buffers are only legal there). OOM
        propagates -- the buckets genuinely didn't fit, don't silently boot in
        a slower eager mode. Any other failure means the dummy-batch machinery
        doesn't cover this model family yet: degrade to eager prefill instead
        of crashing the server, and agree on that across the world (a MIN
        all-reduce over the success flag) -- replay force-sets
        ``global_num_tokens`` on every rank, so one eager rank among replaying
        peers diverges the token counts and deadlocks the next collective.
        """
        if self.disable:
            return
        weight = self._embed_tokens.weight
        self._input_embeds_buf = torch.zeros(
            max(self.capture_buckets),
            weight.shape[1],
            dtype=weight.dtype,
            device=weight.device,
        )
        self._pool = _cuda_graph_wrapper.global_graph_memory_pool
        captured_ok = True
        try:
            with maybe_inference_mode():
                self._capture_all_buckets(decode_wrapper)
        except torch.cuda.OutOfMemoryError:
            raise
        except (NotImplementedError, AttributeError, KeyError, RuntimeError) as exc:
            logger.warning(
                "Prefill graph capture failed (%s: %s); falling back to eager "
                "prefill. This model family may need dedicated dummy-batch support.",
                type(exc).__name__,
                exc,
            )
            captured_ok = False
        if not self._capture_unanimous(captured_ok):
            self.disable = True

    def _capture_all_buckets(self, decode_wrapper: CudaGraphWrapper | None) -> None:
        for bucket in sorted(self.capture_buckets, reverse=True):
            self._ctx = self._make_dummy_batch(bucket, decode_wrapper)
            self._land_input_embeds(
                self._embed_tokens(self.input_buffers.input_ids_buf[:bucket]), bucket
            )
            self._captured_hidden_mode = self._ctx.capture_hidden_mode
            # Breaks record the ambient dummy ctx; it is rebound live at replay.
            try:
                with active_forward(self._ctx):
                    self._capture_bucket(bucket, decode_wrapper)
            finally:
                self._ctx = None
        if self.config.global_rank == 0:
            sample = next(iter(self._captures.values()), None)
            logger.info(
                "prefill breakable graph: captured buckets %s (segments=%d, eager "
                "attention breaks)",
                sorted(self._captures),
                sample.num_segments if sample is not None else 0,
            )

    def _capture_bucket(
        self, bucket: int, decode_wrapper: CudaGraphWrapper | None
    ) -> None:
        """Warm up and capture the breakable graph for ``bucket`` from the buffers."""
        for _ in range(self.num_warmup):
            self._run_inner(bucket)
        torch.cuda.synchronize()
        stream = decode_wrapper.stream if decode_wrapper is not None else None
        cap = BreakableCapture(pool=self._pool, stream=stream)
        with cap:
            self._outputs[bucket] = CapturedForward(*self._run_inner(bucket))
        if self._pool is None:
            self._pool = cap.pool  # share the pool across all subsequent buckets
        cap.replay()  # capture records kernels without executing; smoke-test replay
        self._captures[bucket] = cap

    def _run_inner(self, num_tokens: int):
        """Run the inner model over the leading ``num_tokens`` of the static buffers.

        ``num_tokens`` is the padded bucket size; the padded tail [real:bucket] is
        already scrubbed to safe values (embeds=0, positions=0,
        out_cache_loc=dummy_kv_slot) by :meth:`_land_input_embeds` and
        ``InputBuffers.fill_input_buffers``. The embedding is NOT part of the
        graph: the inner model starts from the static input-embeds buffer, so a
        replay can take precomputed (e.g. merged multimodal) embeddings.
        """
        ib = self.input_buffers
        if self.config.model_is_mrope:
            positions = ib.mrope_positions_buf[:, :num_tokens]
        else:
            positions = ib.positions_buf[:num_tokens]
        return self.inner_model(
            ib.input_ids_buf[:num_tokens],
            positions,
            self._ctx,
            ib.out_cache_loc_buf[:num_tokens],
            input_embeds=self._input_embeds_buf[:num_tokens],
        )

    def _land_input_embeds(self, embeds: torch.Tensor, bucket: int) -> None:
        """Copy ``embeds`` into the static buffer's leading rows, zero the tail.

        The zeroed padded tail keeps the graphed compute over garbage-free rows
        (RMSNorm of zeros is zeros; the tail is discarded by the output slice).
        """
        num_tokens = embeds.shape[0]
        self._input_embeds_buf[:num_tokens].copy_(embeds)
        if num_tokens < bucket:
            self._input_embeds_buf[num_tokens:bucket].zero_()

    def _make_dummy_batch(
        self, num_tokens: int, decode_wrapper: CudaGraphWrapper | None
    ) -> ForwardContext:
        """Populate the static buffers + attention metadata for a dummy bs=1 extend
        forward of ``num_tokens`` tokens, and return its ForwardContext.

        The prefill analogue of decode's ``_init_capture_metadata``. KV writes
        go to the reserved dummy slot and the page table points at page 0, so
        the forward runs (producing discarded garbage) without touching real
        cache state. Backends with extra paged caches (DeepSeek-V4 DSA: SWA +
        compressor + indexer state) also need per-cache block tables, or their
        extend metadata comes up incomplete and the eager attention break
        aborts the capture -- reuse the decode wrapper's dummy-table builder
        (all zeros, the safe page 0) for those.
        """
        ib = self.input_buffers
        ib.input_ids_buf[:num_tokens].fill_(1)
        ib.out_cache_loc_buf[:num_tokens].fill_(ib.dummy_kv_slot)
        ib.positions_buf[:num_tokens].copy_(
            torch.arange(num_tokens, device=self.config.device)
        )
        ib.req_pool_indices_buf[:1].zero_()
        ib.seq_lens_buf[:1].fill_(num_tokens)
        ib.extend_seq_lens_buf[:1].fill_(num_tokens)
        ib.extend_seq_lens_cpu[:1].fill_(num_tokens)
        ib.extend_prefix_lens_buf[:1].zero_()
        ib.extend_prefix_lens_cpu[:1].zero_()
        self.req_to_page[0].zero_()  # dummy request's pages -> page 0 (valid memory)

        ctx = ForwardContext(
            attn_backend=self.attn_backend,
            token_to_kv_pool=self.token_to_kv_pool,
            req_to_page=self.req_to_page,
            bs=1,
            num_extends=1,
            input_num_tokens=num_tokens,
            forward_mode=ForwardMode.EXTEND,
            capture_hidden_mode=(
                CaptureHiddenMode.FULL
                if self.drafter is not None
                else CaptureHiddenMode.NULL
            ),
        )
        if self.dp_size > 1:
            ctx.global_num_tokens = [num_tokens] * self.config.world_size
            ctx.global_bs = [1] * self.config.world_size
        extra_metadata_kwargs: dict = {}
        if (
            getattr(self.attn_backend, "uses_paged_cache_groups", False)
            and decode_wrapper is not None
        ):
            tables = decode_wrapper._capture_paged_cache_block_tables(
                1, self.token_to_kv_pool
            )
            if tables is not None:
                extra_metadata_kwargs["paged_cache_block_tables"] = tables
            extra_metadata_kwargs["num_tokens"] = num_tokens
            extra_metadata_kwargs["positions"] = ib.positions_buf[:num_tokens]
        self.attn_backend.init_forward_metadata(
            bs=1,
            num_extends=1,
            req_pool_indices=ib.req_pool_indices_buf[:1],
            seq_lens=ib.seq_lens_buf[:1],
            req_to_page=self.req_to_page,
            forward_mode=ForwardMode.EXTEND,
            extend_seq_lens=ib.extend_seq_lens_buf[:1],
            extend_seq_lens_cpu=ib.extend_seq_lens_cpu[:1],
            extend_prefix_lens=ib.extend_prefix_lens_buf[:1],
            extend_prefix_lens_cpu=ib.extend_prefix_lens_cpu[:1],
            **extra_metadata_kwargs,
        )
        return ctx

    def _capture_unanimous(self, captured_ok: bool) -> bool:
        """MIN-reduce capture success across the world (see ``capture``)."""
        if self.config.world_group is None or self.config.world_size <= 1:
            return captured_ok
        from tokenspeed.runtime.distributed.process_group_manager import (
            process_group_manager as pg_manager,
        )

        cpu_group = pg_manager.get_process_group("gloo", self.config.world_group)
        flag = torch.tensor([1 if captured_ok else 0], dtype=torch.int32)
        torch.distributed.all_reduce(
            flag, op=torch.distributed.ReduceOp.MIN, group=cpu_group
        )
        unanimous = bool(flag.item())
        if not unanimous and captured_ok:
            logger.warning(
                "Prefill graph: a peer rank failed capture; falling back to "
                "eager prefill on all ranks to keep DP/TP token counts in lockstep."
            )
        return unanimous

    # ------------------------------------------------------------------
    # Replay dispatch
    # ------------------------------------------------------------------

    def can_run(self, ctx: ForwardContext, multimodal_context=None) -> bool:
        """Whether this forward replays a captured graph (mirrors decode's can_run).

        A forward carrying multimodal inputs replays only when the model
        exposes the embeds-only ``multimodal_input_embeds`` seam; models with
        extra per-layer inputs (deepstack) run eager.
        """
        if multimodal_context is not None and self._multimodal_input_embeds is None:
            return False
        return self._replay_bucket(ctx) is not None

    def replay(
        self,
        ctx: ForwardContext,
        input_ids: torch.Tensor,
        multimodal_context=None,
    ):
        """Replay the captured graph for ``ctx`` (caller checked :meth:`can_run`).

        The embedding runs eagerly here, outside the graph: a plain text
        prefill gathers ``embed_tokens(input_ids)`` into the static buffer; a
        multimodal prefill builds the merged text+vision embeddings via the
        model's ``multimodal_input_embeds`` seam (vision encoder included)
        instead -- both replay the same graphs. Then the inner stack replays
        over the padded bucket and the model's eager logits tail finishes on
        the real-token rows.
        """
        bucket = self._replay_bucket(ctx)
        assert bucket is not None, "replay() called without can_run()"
        self._log_engaged_once(bucket, ctx, multimodal_context is not None)
        num_tokens = ctx.input_num_tokens
        input_embeds = None
        if multimodal_context is not None:
            input_embeds = self._multimodal_input_embeds(
                input_ids, ctx, multimodal_context
            )
        self._land_input_embeds(
            input_embeds if input_embeds is not None else self._embed_tokens(input_ids),
            bucket,
        )
        with self._padded_to(ctx, bucket):
            self._captures[bucket].replay()
        hidden_states, aux_hidden_states = self._outputs[bucket].sliced(num_tokens)
        # The eager logits tail of BaseCausalLM.forward, on the replayed hidden states.
        logits_metadata = LogitsMetadata.from_forward_context(ctx)
        return self.text_model.logits_processor(
            input_ids,
            hidden_states,
            self.text_model.lm_head,
            logits_metadata,
            aux_hidden_states,
        )

    def _replay_bucket(self, ctx: ForwardContext) -> int | None:
        """The captured bucket this forward replays, or ``None`` to run eager.

        Pure-extend AND mixed extend+decode batches are eligible: the attention
        break reads the LIVE ambient ctx and dispatches the prefill/decode
        split itself, while the captured token-shaped compute is uniform over
        all rows (pure decode is the decode graph's job). Two ctx fields are
        baked into the captured segments rather than rebound at replay -- the
        draft first-step row narrowing (keyed on ``accept_lengths``) and the
        ``capture_hidden_mode`` aux-hidden capture -- so a live forward carrying
        different values falls back to eager rather than silently dropping the
        reduce / mismatching aux. Prefix caching (cache hits and chunked-prefill
        chunks 2+) IS eligible: the prefix affects only the ragged attention,
        which runs entirely inside the eager break, and it adds zero new tokens,
        so the padded bucket -- hence the baked EP all-to-all shape under DP --
        is identical on prefix and non-prefix ranks.
        """
        if self.disable or ctx.forward_mode is None:
            return None
        if ctx.num_extends <= 0:
            return None
        if not (ctx.forward_mode.is_extend() or ctx.forward_mode.is_mixed()):
            return None
        if ctx.accept_lengths is not None:
            return None
        if ctx.capture_hidden_mode != self._captured_hidden_mode:
            return None
        bucket = self._select_bucket(ctx)
        if bucket is None or bucket not in self._captures:
            return None
        return bucket

    def _select_bucket(self, ctx: ForwardContext) -> int | None:
        """The padded bucket for this forward, or ``None`` to run eager.

        Under data parallelism the MoE expert-parallel all-to-all is a collective
        across ALL ranks, sized from a replicated per-rank token list. The captured
        graph bakes a uniform ``[bucket]*world_size`` layout, so every rank must
        replay the SAME bucket or the collective desyncs (NCCL deadlock). Decide
        purely from replicated global state -- the all-extend flag and the global
        max token count -- so all ranks reach the identical decision/bucket with no
        extra sync (mirrors the decode graph). Idle ranks run a DECODE forward, so
        ``all_extend`` is False whenever any rank is idle and the graph stays off
        (e.g. warmup), correctly falling back to eager.
        """
        if self.dp_size <= 1 or ctx.global_num_tokens is None:
            return self._padded_bucket(ctx.input_num_tokens)
        if not ctx.all_extend:
            return None
        return self._padded_bucket(max(ctx.global_num_tokens))

    def _padded_bucket(self, num_tokens: int) -> int | None:
        """Smallest bucket >= ``num_tokens``, or ``None`` if over the largest.

        With ``--disable-cuda-graph-padding``, only an exact bucket match
        replays (mirroring the decode wrapper's no-padding semantics).
        """
        idx = bisect.bisect_left(self.capture_buckets, num_tokens)
        if idx == len(self.capture_buckets):
            return None
        bucket = self.capture_buckets[idx]
        if self.config.disable_cuda_graph_padding and bucket != num_tokens:
            return None
        return bucket

    @contextmanager
    def _padded_to(self, ctx: ForwardContext, bucket: int):
        """Publish ``ctx`` as the ambient live context, pinned to the padded bucket.

        The graph replays over ``bucket`` (padded) tokens; attention metadata stays
        at the real count (set upstream), so the eager attention break only touches
        real tokens and the padded rows produce discarded garbage. Pin
        ``input_num_tokens`` to the bucket and, under DP, ``global_num_tokens`` /
        ``global_bs`` to the captured uniform layout so any live read during the
        break matches the baked EP shapes. The break reads ``forward_mode`` / ``bs``
        / ``num_extends`` LIVE off this same (ambient) ctx -- which we do NOT pin --
        so models split prefill vs decode and dispatch the per-mode backend
        correctly with no side channel.
        """
        saved = (ctx.input_num_tokens, ctx.global_num_tokens, ctx.global_bs)
        ctx.input_num_tokens = bucket
        if self.dp_size > 1 and ctx.global_num_tokens is not None:
            ctx.global_num_tokens = [bucket] * self.config.world_size
            ctx.global_bs = [1] * self.config.world_size
        try:
            with active_forward(ctx):
                yield
        finally:
            ctx.input_num_tokens, ctx.global_num_tokens, ctx.global_bs = saved

    def _log_engaged_once(
        self, bucket: int, ctx: ForwardContext, is_multimodal: bool
    ) -> None:
        kind = "multimodal" if is_multimodal else "text"
        if kind in self._engaged_logged:
            return
        self._engaged_logged.add(kind)
        logger.info(
            "prefill breakable graph ENGAGED (%s): bucket=%d dp=%s mode=%s "
            "(mixed prefill+decode batches supported)",
            kind,
            bucket,
            # The replay mode actually taken (mirrors _select_bucket), a DP-debug anchor.
            self.dp_size > 1 and ctx.global_num_tokens is not None,
            ctx.forward_mode,
        )
