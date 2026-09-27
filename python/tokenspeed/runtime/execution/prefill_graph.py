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

:class:`PrefillGraph` owns captures keyed by ``(token_bucket, exact_bs)``.
``exact_bs=None`` denotes ordinary segments with eager attention breaks.
Dummy batches balance tokens across the selected nonempty request count.
The embedding lookup stays OUTSIDE
the captured region: graphs start from a static input-embeds buffer, filled at
replay by an eager ``embed_tokens`` gather (text) or by precomputed merged
embeddings (multimodal, via the model's ``multimodal_input_embeds`` seam).
Capture borrows the decode
:class:`~tokenspeed.runtime.execution.forward_step.ForwardStepRunner`'s
stream; buckets share one private mempool, deliberately not the decode graphs'
pool (see :meth:`capture`). At serving time
the executor's target-forward dispatch is a simple
three-way -- decode & captured replays the decode graph (one level up, since
it captures the whole step), prefill & captured replays here (:meth:`can_run`
/ :meth:`replay`), everything else runs the eager model forward.

Ordinary captures keep attention at eager breaks (see
:mod:`tokenspeed.runtime.execution.breakable_cuda_graph`) and reuse the
token-shaped segments across batch sizes. Inline captures include compatible
KDA at a fitting request capacity; full attention keeps its breaks. Mixed batches,
request counts beyond captured capacity, layerwise transfer and DP retain
the ordinary route, subject to its admission rules. Both routes finish with
the model's eager logits tail.

A :class:`NarrowingPrefillModel` drops its row count once, at a fixed layer,
by a completion-dependent amount (DeepSeek-V4.1's CED decoder runs on each
prompt's last window). One token-shaped graph cannot express that, so such a
model is captured as two graph families around an eager narrowing: encoder
graphs per token bucket (``encoder_forward``) and decoder graphs per
decoder-row bucket (``decoder_forward`` from a fixed-row static state). The
decoder graphs depend only on their row count, so they are shared by every
token bucket; a forward whose narrowed rows exceed the largest decoder bucket
runs its decoder stage eager after the encoder replay. Under attention DP the
narrowed row count is rank-local, so the split graph is disabled there.
"""

from __future__ import annotations

import bisect
from contextlib import AbstractContextManager, contextmanager
from types import SimpleNamespace
from typing import TYPE_CHECKING, NamedTuple, Protocol, runtime_checkable

import torch
import tqdm

from tokenspeed.runtime.execution.breakable_cuda_graph import (
    BreakableCapture,
    active_forward,
)
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.cudagraph_memory import (
    CapturedLadder,
    probe_positions,
)
from tokenspeed.runtime.execution.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
)
from tokenspeed.runtime.execution.memory_delta import MemoryDeltaObserver
from tokenspeed.runtime.layers.attention.backends.cache_metadata import (
    CacheBatchMetadata,
)
from tokenspeed.runtime.layers.logits_processor import LogitsMetadata
from tokenspeed.runtime.utils import get_colorful_logger
from tokenspeed.runtime.utils.common import (
    get_available_gpu_memory,
    maybe_inference_mode,
)

logger = get_colorful_logger(__name__)

if TYPE_CHECKING:
    from tokenspeed.runtime.execution.forward_step import ForwardStepRunner
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

    Both ordinary and inline captures use this total-token capacity ladder;
    inline captures additionally select a fitting request capacity. A live extend
    forward is padded up to a bucket fitting its tokens and any dummy scan slots.
    Forwards above the largest bucket run eager.

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
    dense at the top. Each bucket's step is the largest power of two <= size/8,
    floored at 16 tokens and capped at 512 so the absolute worst case stays
    bounded at the top end. The ~12.5% tail that step implies holds only where
    size/8 exceeds the floor: below ~128 tokens the floor dominates and the
    relative waste grows sharply (17 -> 32 pads 88%, 1 -> 16 pads 1500%),
    which is where a graphed forward is least likely to pay for itself.
    Captures share a stream and mempool to reuse scratch, but graph objects,
    outputs and stable metadata are retained per configuration. Total memory
    is not determined by the largest bucket alone; denser ladders also add
    startup capture work.

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


def dummy_batch_size(num_tokens: int, context_len: int) -> int:
    """Minimum request count for a fabricated extend of ``num_tokens`` tokens.

    Each request holds at most ``context_len`` tokens, and takes one parent
    block of a capture-time arena.
    """
    return -(-num_tokens // max(1, int(context_len)))


def resolve_prefill_capture_batch_sizes(
    config: ModelExecutorConfig, token_bucket: int
) -> list[int]:
    """Resolve defaults, validate limits and select BS candidates for this bucket.

    Each request must have at least one token and fit the model context.
    Unset configuration retains the minimum count required by the bucket.
    Counts outside a particular bucket's range are skipped, not padded with
    empty sequences. Invalid configured counts fail before capture. The result
    is sorted and deduplicated; backend support is checked separately.
    """
    minimum = dummy_batch_size(token_bucket, config.context_len)
    maximum = config.max_num_seqs // config.data_parallel_size
    sizes = config.prefill_graph_capture_batch_sizes
    if sizes is None:
        sizes = [minimum]
    if any(size <= 0 or size > maximum for size in sizes):
        raise ValueError(
            "prefill graph capture batch sizes must be positive and no larger "
            "than max_num_seqs / data_parallel_size"
        )
    return sorted({size for size in sizes if minimum <= size <= token_bucket})


@runtime_checkable
class NarrowedRowState(Protocol):
    """Row-shaped activations a :class:`NarrowingPrefillModel` carries between
    its stages. The graph owner treats it opaquely except for these two."""

    @property
    def rows(self) -> int: ...

    def land_into(self, dst: NarrowedRowState) -> None:
        """Copy into the leading rows of the fixed-row ``dst``; zero its tail."""
        ...


@runtime_checkable
class NarrowingPrefillModel(Protocol):
    """An inner model whose prefill row count drops once, at a fixed layer.

    ``forward`` must equal ``finish_forward(*decoder_forward(narrowing_forward(
    encoder_forward(...))))``. The encoder stage is token-shaped and is captured
    per token bucket; the narrowing stage runs eager on the real rows and may
    receive the encoder's padded output; the decoder stage computes every row
    of the state it is given (a padded static one under the graph) and is
    captured per decoder-row bucket; the finish stage gathers the sampled rows
    and publishes any per-forward row report on ``ctx``. Stages read
    per-forward quantities from ``ctx`` and its attention backend, never from
    loose arguments a captured break would freeze.
    """

    # The most decoder rows one request can contribute (V4.1: its last window).
    max_decoder_rows_per_request: int

    def encoder_forward(
        self, input_ids: torch.Tensor, positions: torch.Tensor, ctx, **model_kwargs
    ) -> NarrowedRowState: ...

    def narrowing_forward(self, state: NarrowedRowState, ctx) -> NarrowedRowState: ...

    def decoder_forward(
        self, state: NarrowedRowState, ctx
    ) -> tuple[torch.Tensor, list[torch.Tensor]]: ...

    def finish_forward(
        self, hidden: torch.Tensor, captured: list[torch.Tensor], ctx
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]: ...

    def decoder_rows(self, ctx) -> int:
        """Rows ``narrowing_forward`` yields for the forward ``ctx`` describes."""
        ...

    def allocate_decoder_state(self, rows: int) -> NarrowedRowState:
        """A zero ``rows``-row state laid out like ``narrowing_forward``'s output."""
        ...


def text_and_inner_model(
    model: torch.nn.Module | None,
) -> tuple[torch.nn.Module | None, torch.nn.Module | None]:
    """The text model under a causal-LM wrapper, and the stack under that."""
    text_model = model.language_model if hasattr(model, "language_model") else model
    return text_model, getattr(text_model, "model", None)


def narrowing_prefill_model(model) -> NarrowingPrefillModel | None:
    """The narrowing model a prefill capture would drive, or ``None``.

    The protocol is implemented by the inner text model, not the causal-LM
    wrapper the runner holds, so the wrapper is unwrapped first.
    """
    _, inner = text_and_inner_model(model)
    return inner if isinstance(inner, NarrowingPrefillModel) else None


def get_decoder_row_buckets(
    token_buckets: list[int], rows_per_request: int, max_bs: int
) -> list[int]:
    """Decoder-row capacities to capture for a :class:`NarrowingPrefillModel`.

    Narrowing never adds rows, so the decoder never sees more than the token
    bucket; per request it keeps at most ``rows_per_request``, so it never
    sees more than ``rows_per_request * max_bs`` either. The ladder is the
    token ladder clipped to that cap (same relative padding bound, same
    ``--prefill-graph-capture-sizes`` override), with the cap itself as the
    top rung. Every decoder bucket is captured once and shared by all token
    buckets.

    Args:
        token_buckets: The captured token buckets (``get_prefill_token_buckets``).
        rows_per_request: The model's ``max_decoder_rows_per_request``.
        max_bs: Rank-local request capacity.

    Returns:
        Sorted ascending decoder-row buckets; empty iff ``token_buckets`` is.
    """
    if not token_buckets:
        return []
    top = min(max(token_buckets), max(1, rows_per_request) * max(1, max_bs))
    buckets = {bucket for bucket in token_buckets if bucket <= top}
    buckets.add(top)
    return sorted(buckets)


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


class CapturedEncoder(NamedTuple):
    """A token bucket's captured encoder stage and its pool-pinned output state."""

    capture: BreakableCapture
    state: NarrowedRowState


class CapturedDecoder(NamedTuple):
    """A decoder-row bucket's captured decoder stage.

    ``statics`` is the fixed-row input state the graph reads (landed into
    before each replay); ``output`` its pool-pinned normalized rows and taps.
    """

    capture: BreakableCapture
    statics: NarrowedRowState
    output: CapturedForward


class PrefillGraph:
    """The breakable prefill (extend) CUDA graphs.

    A pure graph object -- :meth:`can_run` / :meth:`replay` -- holding no
    reference to any other component. The executor calls :meth:`capture` once
    kernel tuning has run, passing the decode wrapper transiently for its
    capture stream; it is not kept. The dispatch
    checks :meth:`can_run` and calls :meth:`replay`; the eager path stays a
    direct ``model_runner.forward`` call at that call site. Capture failure
    fails the boot (see :meth:`capture`).

    Args:
        model_runner: The target ModelRunner. Supplies the loaded model
            (multimodal wrappers are unwrapped internally: the graph wraps the
            nested ``language_model``'s text transformer, image prefills run
            eager) and ``is_generation`` (embedding models run eager).
        attn_backend: Backend whose extend metadata the dummy capture batch sets.
        token_to_kv_pool: KV pool the dummy batch points at (reserved dummy slot).
        input_buffers: The shared static input buffers the graphs read from.
        config: Model-executor config (buckets, DP/world topology, device).
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
        drafter=None,
        num_warmup: int = 3,
        graph_supported: bool = True,
    ) -> None:
        model = model_runner.model if model_runner is not None else None
        # Multimodal seam: models whose multimodal path is embeds-only expose
        # multimodal_input_embeds; others (e.g. deepstack) replay text only.
        self._multimodal_input_embeds = getattr(model, "multimodal_input_embeds", None)
        self.text_model, self.inner_model = text_and_inner_model(model)
        # Embedding runs eagerly OUTSIDE the graphs (see capture); the graphs
        # read a static input-embeds buffer instead of gathering from input_ids.
        self._embed_tokens = getattr(self.inner_model, "embed_tokens", None)
        self._input_embeds_buf: torch.Tensor | None = None
        self.attn_backend = attn_backend
        self.token_to_kv_pool = token_to_kv_pool
        self.input_buffers = input_buffers
        self.config = config
        self.drafter = drafter
        self.num_warmup = num_warmup
        self.dp_size = config.data_parallel_size

        # A narrowing model is captured as encoder + decoder graph families
        # around its eager narrowing stage (module docstring); None means the
        # whole inner forward is one token-shaped capture.
        self._narrowing: NarrowingPrefillModel | None = narrowing_prefill_model(model)

        self.capture_buckets = get_prefill_token_buckets(config)
        self.disable = (
            config.enforce_eager
            or config.disable_prefill_graph
            # Backend-declared restriction (cuda_graph_support), resolved by
            # ModelExecutor over the backend tree at startup.
            or not graph_supported
            or not self.capture_buckets
            or self.inner_model is None
            or self._embed_tokens is None
            or model_runner is None
            or not model_runner.is_generation
            # The graph's embedding seam takes input_ids alone; an embedding
            # that reads request-token history stays eager.
            or model_runner.model_config.requires_request_token_history
            # DP replay decisions must come from replicated state, and a
            # forward's multimodal-ness is rank-local: one rank running its mm
            # prefill eager while text-only peers replay desyncs the EP
            # collectives. Until the DP metadata gather carries a multimodal
            # flag, keep the graph off for multimodal models under DP.
            or (config.data_parallel_size > 1 and model_runner.is_multimodal)
            # The narrowed row count is rank-local (which prompts complete on
            # this rank), so the decoder bucket -- and the collective shapes
            # its graph bakes -- would differ across ranks; the stages also
            # size their collectives from their own rows, which the DP
            # gather does not carry. Until narrowed counts are exchanged,
            # the split graph stays off under attention DP.
            or (config.data_parallel_size > 1 and self._narrowing is not None)
        )
        if (
            self._narrowing is not None
            and config.data_parallel_size > 1
            and not config.enforce_eager
            and not config.disable_prefill_graph
        ):
            logger.info(
                "Prefill CUDA graphs disabled: the narrowing prefill model's "
                "decoder rows are rank-local under attention DP"
            )

        self.decoder_buckets: list[int] = (
            get_decoder_row_buckets(
                self.capture_buckets,
                self._narrowing.max_decoder_rows_per_request,
                int(config.max_num_seqs) // max(int(config.data_parallel_size), 1),
            )
            if self._narrowing is not None and not self.disable
            else []
        )

        self._ctx: ForwardContext | None = None
        self._pool = None
        self._engaged_logged: set[str] = set()
        # Aux-capture mode baked into the graphs; mismatched live forwards run eager.
        self._captured_hidden_mode = None
        # One owner for ordinary and attention-containing captures. None is the
        # batch-independent attention-break variant; integers require exact BS.
        # Checkpoint counts are refreshed metadata, never another capture key.
        self._captures: dict[
            tuple[int, int | None], tuple[BreakableCapture, CapturedForward]
        ] = {}
        # The narrowing model's two families: encoders by token bucket,
        # decoders by decoder-row bucket (shared across token buckets).
        self._encoders: dict[int, CapturedEncoder] = {}
        self._decoders: dict[int, CapturedDecoder] = {}

    # ------------------------------------------------------------------
    # Graph capture
    # ------------------------------------------------------------------

    def capture(
        self,
        decode_wrapper: ForwardStepRunner | None = None,
        *,
        entries: int | None,
        observer: MemoryDeltaObserver,
    ) -> None:
        """Capture ordinary and compatible inline variants per token bucket.

        No-op when disabled. ``entries`` samples each ladder's buckets at the
        probe's positions, for a caller that sizes memory from ``observer``,
        measured around each capture; ``None`` captures every bucket.

        ``decode_wrapper`` supplies the shared capture stream (used here only,
        not stored). Buckets share
        one PRIVATE mempool (first capture
        allocates it) to reuse scratch; graph objects and metadata still cost
        memory per variant. Never use the decode graphs' pool: eager ops cache
        raw pointers to
        buffers they lazily allocated inside a decode capture (flashinfer's
        trtllm-gen MoE runner), and a prefill capture reusing those freed
        blocks means every replay rewrites them, corrupting the next eager
        call (IMA; A/B-proven on qwen3.5 MTP).

        Runs under inference mode like serving forwards (in-place updates on
        inference-mode model state buffers are only legal there). There is no
        handler here: every failure kills the boot, OOM included (the graph
        pool did not fit next to weights + KV cache -- free headroom, lower
        ``--prefill-graph-max-tokens``, or set it to 0). A model family that
        cannot capture has to say so up front in ``ModelExecutor``'s
        ``disable_prefill_graph`` condition, because degrading here silently
        served eager prefill to a whole model family while CI stayed green.
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
        # Seam: backends alloc static buffers or refuse capture; kept
        # outside inference mode (in-place refresh). Base default: no-op.
        self._captures.clear()
        self._encoders.clear()
        self._decoders.clear()
        self.attn_backend.init_prefill_graph_state(
            max_num_tokens=max(self.capture_buckets),
            max_bs=int(self.config.max_num_seqs)
            // max(int(self.config.data_parallel_size), 1),
        )
        with maybe_inference_mode():
            self._capture_all_buckets(decode_wrapper, entries, observer)
            if self._narrowing is not None:
                self._capture_decoders(decode_wrapper, entries, observer)

    def _capture_all_buckets(
        self,
        decode_wrapper: ForwardStepRunner | None,
        entries: int | None,
        observer: MemoryDeltaObserver,
    ) -> None:
        rank = self.config.global_rank
        # Off the plan: a bucket it omits is one nothing here can capture.
        series = "prefill" if self._narrowing is None else "prefill:encoder"
        ladder = self.capture_ladders(entries)[series]
        sampled = {ladder.widths[i] for i in ladder.sampled}
        buckets: list[int] = []
        inline_counts: dict[int, list[int]] = {}
        for bucket, bs in self.capture_plan[series]:
            if bucket not in sampled:
                continue
            if bs is None:
                buckets.append(bucket)
            else:
                inline_counts.setdefault(bucket, []).append(bs)
        capture_range = tqdm.tqdm(buckets) if rank == 0 else buckets
        for bucket in capture_range:
            if rank == 0:
                avail_mem = get_available_gpu_memory(
                    self.config.device, self.config.gpu_id, empty_cache=False
                )
                capture_range.set_description(
                    f"Capturing prefill buckets ({bucket=} {avail_mem=:.2f} GB)"
                )
            minimum_bs = dummy_batch_size(bucket, self.config.context_len)
            self._ctx = self.make_dummy_batch(bucket, minimum_bs)
            self._land_input_embeds(
                self._embed_tokens(self.input_buffers.input_ids_buf[:bucket]), bucket
            )
            self._captured_hidden_mode = self._ctx.capture_hidden_mode
            # Breaks record the ambient dummy ctx; it is rebound live at replay.
            try:
                with active_forward(self._ctx):
                    if self.dp_size == 1:
                        self.attn_backend.prepare_prefill_metadata(
                            bucket, minimum_bs, self._ctx.forward_mode, capture=False
                        )
                    if self._narrowing is not None:
                        self._encoders[bucket] = self._capture_encoder(
                            bucket, decode_wrapper, observer.measure("prefill:encoder")
                        )
                    else:
                        self._captures[bucket, None] = self._capture_bucket(
                            bucket, decode_wrapper, observer.measure("prefill")
                        )
                for bs in inline_counts.get(bucket, ()):
                    self._ctx = self.make_dummy_batch(bucket, bs)
                    with active_forward(self._ctx):
                        if not self.attn_backend.prepare_prefill_metadata(
                            bucket, bs, self._ctx.forward_mode, capture=True
                        ):
                            raise RuntimeError(
                                f"{type(self.attn_backend).__name__} admitted "
                                f"({bucket}, {bs}) and then refused to prepare it"
                            )
                        self._captures[bucket, bs] = self._capture_bucket(
                            bucket, decode_wrapper, observer.measure("prefill")
                        )
            finally:
                self._ctx = None
        if self.config.global_rank == 0:
            if self._narrowing is not None:
                sample = next(iter(self._encoders.values()), None)
                logger.info(
                    "prefill breakable graph: captured encoder buckets "
                    f"{sorted(self._encoders)!s} (segments="
                    f"{(sample.capture.num_segments if sample is not None else 0):d}"
                    ", eager attention breaks; the narrowing stage runs eager)",
                )
                return
            ordinary = {
                bucket: value
                for (bucket, bs), value in self._captures.items()
                if bs is None
            }
            sample = next(iter(ordinary.values()), None)
            logger.info(
                f"prefill breakable graph: captured buckets {sorted(ordinary)!s} "
                f"(segments={(sample[0].num_segments if sample is not None else 0):d}, "
                "eager attention breaks)",
            )
            variants = {
                key: value
                for key, value in self._captures.items()
                if key[1] is not None
            }
            if variants:
                logger.info(
                    "prefill inline attention: captured (tokens, requests) "
                    f"{sorted(variants)!s} "
                    "with fixed checkpoint slots "
                    f"(segments={next(iter(variants.values()))[0].num_segments:d}, "
                    "ordinary captures retained for fallback)",
                )

    def _capture_decoders(
        self,
        decode_wrapper: ForwardStepRunner | None,
        entries: int | None,
        observer: MemoryDeltaObserver,
    ) -> None:
        """Capture the narrowing model's decoder stage per decoder-row bucket.

        Each bucket gets a static input state of exactly that many rows and a
        dummy batch whose decoder view keeps every row (requests of at most
        ``max_decoder_rows_per_request`` tokens, all completing), so the
        captured breaks see the full row count. The encoder and narrowing
        stages run eagerly on that batch first: the decoder stage consumes
        per-forward backend state its preceding layers produce (V4.1's reuse
        layers read the index source's selection), and their output fills the
        statics for warmup and capture. The graphs share the encoder pool; the
        statics live outside it, so no capture can alias them.
        """
        rank = self.config.global_rank
        per_request = max(1, int(self._narrowing.max_decoder_rows_per_request))
        # Off the plan, for the same reason the bucket ladder is.
        ladder = self.capture_ladders(entries)["prefill:decoder"]
        buckets = [ladder.widths[i] for i in ladder.sampled]
        capture_range = tqdm.tqdm(buckets) if rank == 0 else buckets
        for rows in capture_range:
            if rank == 0:
                avail_mem = get_available_gpu_memory(
                    self.config.device, self.config.gpu_id, empty_cache=False
                )
                capture_range.set_description(
                    f"Capturing prefill decoder buckets ({rows=} {avail_mem=:.2f} GB)"
                )
            bs = -(-rows // per_request)
            self._ctx = self.make_dummy_batch(rows, bs)
            self._land_input_embeds(
                self._embed_tokens(self.input_buffers.input_ids_buf[:rows]), rows
            )
            statics = self._narrowing.allocate_decoder_state(rows)
            try:
                with active_forward(self._ctx):
                    if self.dp_size == 1:
                        self.attn_backend.prepare_prefill_metadata(
                            rows, bs, self._ctx.forward_mode, capture=False
                        )
                    kept = self._narrowing.decoder_rows(self._ctx)
                    if kept != rows:
                        raise RuntimeError(
                            f"prefill decoder capture for {rows} rows narrowed its "
                            f"dummy batch to {kept} rows"
                        )
                    encoded = self._run_encoder(rows)

                    def rearm() -> None:
                        self._narrowing.narrowing_forward(encoded, self._ctx).land_into(
                            statics
                        )

                    self._decoders[rows] = self._capture_decoder(
                        statics,
                        rearm,
                        decode_wrapper,
                        observer.measure("prefill:decoder"),
                    )
            finally:
                self._ctx = None
        if rank == 0:
            sample = next(iter(self._decoders.values()), None)
            logger.info(
                "prefill breakable graph: captured decoder buckets "
                f"{sorted(self._decoders)!s} (segments="
                f"{(sample.capture.num_segments if sample is not None else 0):d}"
                ", shared across token buckets)",
            )

    @property
    def capture_plan(self) -> dict[str, list[tuple[int, int | None]]]:
        """Every graph a full capture records, per ladder, widest first.

        The capture loops iterate this rather than deciding admission as they
        go, so what a capture records can be read without running one. An
        inline variant is entered only when the backend admits its shape, the
        same answer its ``prepare_prefill_metadata`` gives; asked as an extend
        because ``make_dummy_batch`` fabricates nothing else.
        The CUDA-graph memory projection counts the same plan, so it cannot
        price a graph the capture does not record.
        """
        if self.disable:
            return {}
        buckets = sorted(self.capture_buckets, reverse=True)
        # Resolved for every bucket: the counts are validated even where unused.
        counts = {
            bucket: resolve_prefill_capture_batch_sizes(self.config, bucket)
            for bucket in buckets
        }
        # A narrowing model's attention stays at its breaks: no inline
        # variants.
        if self._narrowing is not None:
            return {
                "prefill:encoder": [(bucket, None) for bucket in buckets],
                "prefill:decoder": [
                    (rows, None) for rows in sorted(self.decoder_buckets, reverse=True)
                ],
            }
        plan: list[tuple[int, int | None]] = []
        for bucket in buckets:
            # Retain the ordinary graph for mixed/different-count batches.
            plan.append((bucket, None))
            # DP admission must be rank-uniform; keep its existing route.
            if self.dp_size > 1:
                continue
            plan.extend(
                (bucket, bs)
                for bs in counts[bucket]
                if self.attn_backend.admits_prefill_graph(
                    bucket, bs, ForwardMode.EXTEND
                )
            )
        return {"prefill": plan}

    def capture_ladders(self, entries: int | None) -> dict[str, CapturedLadder]:
        """Each ladder's entry widths off the plan, and the positions ``entries`` samples.

        A bucket's inline variants are entries at the bucket's width, sampled
        with it.
        """
        ladders = {}
        for series, graphs in self.capture_plan.items():
            buckets = [bucket for bucket, bs in graphs if bs is None]
            sampled = {buckets[i] for i in probe_positions(len(buckets), entries)}
            ladders[series] = CapturedLadder(
                [bucket for bucket, _ in graphs],
                [i for i, (bucket, _) in enumerate(graphs) if bucket in sampled],
            )
        return ladders

    def release_graphs(self) -> None:
        """Drop the captured buckets and the private pool they share.

        The captures recorded the bound cache pool's buffers, so a caller that
        rebinds releases here first; the next capture allocates a fresh pool.
        """
        if self.disable:
            return
        self._captures.clear()
        self._encoders.clear()
        self._decoders.clear()
        self._pool = None

    def _capture_bucket(
        self,
        bucket: int,
        decode_wrapper: ForwardStepRunner | None,
        observer: AbstractContextManager[None],
    ) -> tuple[BreakableCapture, CapturedForward]:
        """Warm up and capture the breakable graph for ``bucket`` from the buffers.

        ``observer`` wraps the capture alone: the warmups above it are eager
        forwards, and what they keep is left to the utilization headroom.
        """
        for _ in range(self.num_warmup):
            self._run_inner(bucket)
        torch.cuda.synchronize()
        stream = decode_wrapper.stream if decode_wrapper is not None else None
        cap = BreakableCapture(pool=self._pool, stream=stream)
        with observer, cap:
            output = CapturedForward(*self._run_inner(bucket))
        if self._pool is None:
            self._pool = cap.pool  # share the pool across all subsequent buckets
        cap.replay()  # capture records kernels without executing; smoke-test replay
        return cap, output

    def _capture_encoder(
        self,
        bucket: int,
        decode_wrapper: ForwardStepRunner | None,
        observer: AbstractContextManager[None],
    ) -> CapturedEncoder:
        """Warm up and capture the encoder stage for ``bucket`` from the buffers."""
        for _ in range(self.num_warmup):
            self._run_encoder(bucket)
        torch.cuda.synchronize()
        stream = decode_wrapper.stream if decode_wrapper is not None else None
        cap = BreakableCapture(pool=self._pool, stream=stream)
        with observer, cap:
            state = self._run_encoder(bucket)
        if self._pool is None:
            self._pool = cap.pool
        cap.replay()
        return CapturedEncoder(cap, state)

    def _capture_decoder(
        self,
        statics: NarrowedRowState,
        rearm,
        decode_wrapper: ForwardStepRunner | None,
        observer: AbstractContextManager[None],
    ) -> CapturedDecoder:
        """Warm up and capture the decoder stage over the static state ``statics``.

        ``rearm`` runs the narrowing stage into ``statics`` and is called before
        every decoder run here, as serving does before every decoder replay:
        the decoder consumes per-forward backend state its predecessor
        produces and later layers overwrite (V4.1's index selection chain).
        """
        for _ in range(self.num_warmup):
            rearm()
            self._narrowing.decoder_forward(statics, self._ctx)
        torch.cuda.synchronize()
        rearm()
        stream = decode_wrapper.stream if decode_wrapper is not None else None
        cap = BreakableCapture(pool=self._pool, stream=stream)
        with observer, cap:
            output = CapturedForward(
                *self._narrowing.decoder_forward(statics, self._ctx)
            )
        if self._pool is None:
            self._pool = cap.pool
        rearm()
        cap.replay()
        return CapturedDecoder(cap, statics, output)

    def _inner_inputs(self, num_tokens: int):
        """The static-buffer inputs of a captured forward over ``num_tokens`` rows."""
        ib = self.input_buffers
        if self.config.model_is_mrope:
            positions = ib.mrope_positions_buf[:, :num_tokens]
        else:
            positions = ib.positions_buf[:num_tokens]
        input_ids = ib.input_ids_buf[:num_tokens]
        model_kwargs = {
            "input_embeds": self._input_embeds_buf[:num_tokens],
            **ib.ngram_model_kwargs(num_tokens),
        }
        prepare_kwargs = getattr(self.text_model, "prepare_model_kwargs", None)
        if prepare_kwargs is not None:
            model_kwargs = prepare_kwargs(self._ctx, input_ids, model_kwargs)
        return input_ids, positions, model_kwargs

    def _run_encoder(self, num_tokens: int) -> NarrowedRowState:
        """The narrowing model's encoder stage over the static buffers (see
        :meth:`_run_inner` for the padding contract it inherits)."""
        input_ids, positions, model_kwargs = self._inner_inputs(num_tokens)
        return self._narrowing.encoder_forward(
            input_ids, positions, self._ctx, **model_kwargs
        )

    def _run_inner(self, num_tokens: int):
        """Run the inner model over the leading ``num_tokens`` of the static buffers.

        ``num_tokens`` is the padded bucket size; the padded tail [real:bucket] is
        already scrubbed to safe values (embeds=0, positions=0) by
        :meth:`_land_input_embeds` and ``InputBuffers.fill_input_buffers``;
        the backends' extend write spans cover only the real tokens, so
        padded rows never write KV. The embedding is NOT part of the
        graph: the inner model starts from the static input-embeds buffer, so a
        replay can take precomputed (e.g. merged multimodal) embeddings.
        """
        input_ids, positions, model_kwargs = self._inner_inputs(num_tokens)
        return self.inner_model(
            input_ids,
            positions,
            self._ctx,
            **model_kwargs,
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

    def _dummy_group_tables(self, bs: int) -> dict[str, "torch.Tensor"]:
        """Build the capture batch's group tables: one row per fabricated
        request, one width per group, row ``i`` holding block ``i + 1``.

        Width is ``ceil(physical_context_len / grain)`` for every group,
        whatever its retention. The physical extent is the quantity every
        backend sizes its own per-request tables from
        (``BaseAttnConfig.context_len`` is the model context plus
        ``spec_context_pad``), so a row sized from it covers any column a
        consumer can derive: rows travel in the group's raw scheduler grain,
        and kernel-page expansion happens at the one conversion point (the
        router's table stacks; V4's bespoke metadata build). The width is
        a property of the group's published
        spec, not of the kernel that reads it, which is why no per-backend
        knob is needed.

        Deliberately NOT ``compute_max_logical_pages_for_capture``: that
        helper answers the decode question, where a row describes live cache
        history, so a sliding group is bounded by its window. Capture derives
        a write column per position of the extend it fabricates
        (``extend_out_cache_locs``, ``(prefix + new - 1) // grain``)
        with no window bound, so a window-sized row underflows.
        Trying the helper here made Inkling capture die with "extend write
        locations out of table bounds" -- its ``sliding_attention_0`` row was
        6 columns against the 63 the bucket needed.

        Blocks are distinct per row because a state group takes one working
        block per request; two rows sharing one clobber each other. Note the
        runtime check for that (``_gather_state_block_indices``) is gated on
        ``TOKENSPEED_CACHE_DEBUG``, so a regression here would be silent --
        ``test_each_capture_row_gets_its_own_block`` is the guard.

        An empty dict (a pool publishing no groups: unit fixtures, warmup
        before binding) skips the cache-metadata kwargs downstream.
        """
        # ALL groups, state included: hybrid wrappers forward the dict to the
        # mamba child, which requires its state group; KV children keep only
        # the families they declared (_consumed_group_tables).
        out = {}
        extent = max(1, int(self.config.physical_context_len))
        # Built on the host: make_dummy_batch's only use of these is
        # ``.cpu().numpy()`` for the contract packer, so a device tensor here
        # would be allocated and copied straight back off per group per bucket.
        first_block = torch.arange(1, bs + 1, dtype=torch.int32)
        for spec in self.token_to_kv_pool.arena.cache_group_specs:
            cols = -(-extent // int(spec.block_granularity))
            # Never the reserved block 0: attention runs eager inside the
            # break, so capture really does write KV, and block 0 must stay
            # zero for the padding and table holes that resolve into it.
            # The upper bound is the group's own block count, enforced by the
            # contract packer: block_tables_from_forward_op rejects anything
            # past group_page_counts[gid] - 1, and capture failure is fatal, so
            # a violation is a loud dead boot rather than a bad table. It is
            # never reached in practice, but not because bs is bounded by the
            # pool -- the bucket ladder does not clamp against max_bs, and an
            # oversized bs dies earlier on the max_bs-sized request buffers
            # make_dummy_batch writes before it gets here.
            out[str(spec.group_id)] = first_block[:, None].expand(bs, cols).contiguous()
        return out

    def make_dummy_batch(self, num_tokens: int, bs: int) -> ForwardContext:
        """Populate the static buffers + attention metadata for a dummy extend
        forward of ``num_tokens`` tokens over ``bs`` positive-length requests,
        and return its ForwardContext.

        The tokens are balanced across the requested count, with the longer
        requests first. No request may exceed the model context length: every
        per-request structure (page-table rows, DSA indexer tables) is sized
        for ``physical_context_len``, and a longer fabricated request indexes
        past them. The request count must also fit the input buffers. A real
        forward carries more than ``context_len`` tokens only as
        a multi-request batch, never as one sequence.

        The prefill analogue of decode's ``_init_capture_metadata``. KV writes
        go to the reserved dummy slot; per-group table widths come from
        :meth:`_dummy_group_tables`. Backends with extra cache groups
        (DeepSeek-V4 DSA: SWA + compressor + indexer state) need every group
        table, or their extend metadata is incomplete.
        """
        ib = self.input_buffers
        # Logical context_len, deliberately NOT physical_context_len: the
        # fabricated positions run 0..max_req_tokens-1 and must stay inside
        # the rope tables; per-request structures are sized for the (larger)
        # physical extent, so this remains in bounds.
        max_req_tokens = max(1, int(self.config.context_len))
        if not (
            1 <= bs <= min(num_tokens, ib.seq_lens_buf.numel())
            and num_tokens <= bs * max_req_tokens
        ):
            raise ValueError(
                "prefill capture requests do not fit token/context capacity"
            )
        length, remainder = divmod(num_tokens, bs)
        seq_lens = [length + (row < remainder) for row in range(bs)]
        seq_lens_cpu = torch.tensor(seq_lens, dtype=ib.seq_lens_buf.dtype)
        seq_lens_gpu = seq_lens_cpu.to(self.config.device)
        ib.input_ids_buf[:num_tokens].fill_(1)
        ib.positions_buf[:num_tokens].copy_(
            torch.cat([torch.arange(l, device=self.config.device) for l in seq_lens])
        )
        ib.req_pool_indices_buf[:bs].copy_(
            torch.arange(bs, dtype=ib.req_pool_indices_buf.dtype)
        )
        ib.seq_lens_buf[:bs].copy_(seq_lens_gpu)
        ib.extend_seq_lens_buf[:bs].copy_(seq_lens_gpu)
        ib.extend_seq_lens_cpu[:bs].copy_(seq_lens_cpu)
        ib.extend_prefix_lens_buf[:bs].zero_()
        ib.extend_prefix_lens_cpu[:bs].zero_()
        ib.extend_replay_lens_cpu[:bs].zero_()
        ib.extend_prompt_lens_cpu[:bs].copy_(seq_lens_cpu)
        ib.input_lengths_buf[:bs].copy_(seq_lens_gpu)
        ib.prepare_request_token_history_inputs(
            batch_size=bs, num_extends=bs, decode_width=1
        )

        ctx = ForwardContext(
            attn_backend=self.attn_backend,
            token_to_kv_pool=self.token_to_kv_pool,
            bs=bs,
            num_extends=bs,
            input_num_tokens=num_tokens,
            forward_mode=ForwardMode.EXTEND,
            capture_hidden_mode=(
                CaptureHiddenMode.FULL
                if self.drafter is not None
                else CaptureHiddenMode.NULL
            ),
            gather_ids=torch.cumsum(seq_lens_gpu.to(torch.int64), dim=0) - 1,
        )
        if self.dp_size > 1:
            ctx.global_num_tokens = [num_tokens] * self.config.world_size
            ctx.global_bs = [bs] * self.config.world_size
        # Every backend gets the same kwargs; V4 reads num_tokens/positions
        # for its packed rows, the others absorb them via **kwargs.
        extra_metadata_kwargs: dict = {
            "num_tokens": num_tokens,
            "positions": ib.positions_buf[:num_tokens],
        }
        group_tables = self._dummy_group_tables(bs)
        if group_tables:
            # Route the dummy tables through the same bridge packer live
            # batches use: one packed device storage, contract order and
            # bounds validated (the router's packed unpack and V4's
            # packed-storage checks both key on that layout).
            arrays = {
                group_id: table.cpu().numpy()
                for group_id, table in group_tables.items()
            }
            dummy_forward_op = SimpleNamespace(block_tables_arrays=lambda: arrays)
            cache_metadata = CacheBatchMetadata.from_forward_op(
                dummy_forward_op,
                device=self.config.device,
                contract=self.token_to_kv_pool.arena.runtime_contract,
                num_requests=bs,
            )
            group_tables = dict(
                cache_metadata.tables(active_forward_op=dummy_forward_op)
            )
            extra_metadata_kwargs["block_tables"] = group_tables
            extra_metadata_kwargs["block_tables_cpu"] = dict(
                cache_metadata.tables_cpu(active_forward_op=dummy_forward_op)
            )
        self.attn_backend.init_forward_metadata(
            bs=bs,
            num_extends=bs,
            req_pool_indices=ib.req_pool_indices_buf[:bs],
            seq_lens=ib.seq_lens_buf[:bs],
            forward_mode=ForwardMode.EXTEND,
            extend_seq_lens=ib.extend_seq_lens_buf[:bs],
            extend_seq_lens_cpu=ib.extend_seq_lens_cpu[:bs],
            extend_prefix_lens=ib.extend_prefix_lens_buf[:bs],
            extend_prefix_lens_cpu=ib.extend_prefix_lens_cpu[:bs],
            extend_replay_lens_cpu=ib.extend_replay_lens_cpu[:bs],
            extend_prompt_lens_cpu=ib.extend_prompt_lens_cpu[:bs],
            extend_with_prefix=False,
            **extra_metadata_kwargs,
        )
        return ctx

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
        # Re-pad tail rows: they hold the previous forward's residue, which captured kernels consume.
        if num_tokens < bucket:
            ib = self.input_buffers
            ib.input_ids_buf[num_tokens:bucket].fill_(1)
            if self.config.model_is_mrope:
                ib.mrope_positions_buf[:, num_tokens:bucket].zero_()
            else:
                ib.positions_buf[num_tokens:bucket].zero_()
        if self._narrowing is not None:
            hidden_states, aux_hidden_states = self._replay_narrowed(
                bucket, ctx, num_tokens
            )
        else:
            cap, output = self._captures[bucket, None]
            if self.dp_size == 1 and self.attn_backend.step_counter is None:
                capture_bs = self._merged_capture_bs(bucket, ctx)
                ready = self.attn_backend.prepare_prefill_metadata(
                    bucket,
                    capture_bs if capture_bs is not None else ctx.bs,
                    ctx.forward_mode,
                    capture=False,
                )
                if ready and capture_bs is not None:
                    cap, output = self._captures[bucket, capture_bs]
            with self._padded_to(ctx, bucket):
                cap.replay(valid_rows=num_tokens)
            hidden_states, aux_hidden_states = output.sliced(num_tokens)
        # The eager logits tail of BaseCausalLM.forward, on the replayed hidden states.
        logits_metadata = LogitsMetadata.from_forward_context(ctx)
        return self.text_model.logits_processor(
            input_ids,
            hidden_states,
            self.text_model.lm_head,
            logits_metadata,
            aux_hidden_states,
        )

    def _replay_narrowed(
        self, bucket: int, ctx: ForwardContext, num_tokens: int
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        """Encoder replay, eager narrowing, decoder replay (or eager decoder).

        The whole sequence runs under the padded ambient context so the eager
        breaks and the narrowing stage see the live forward; the narrowing and
        decoder stages size their own collectives from their row counts. The
        decoder graph replays with the narrowed row count as its valid rows,
        so its breaks scrub the static state's padded tail; a row count above
        the largest decoder bucket runs the decoder stage eager.
        """
        model = self._narrowing
        encoder = self._encoders[bucket]
        if self.dp_size == 1 and self.attn_backend.step_counter is None:
            self.attn_backend.prepare_prefill_metadata(
                bucket, ctx.bs, ctx.forward_mode, capture=False
            )
        rows = model.decoder_rows(ctx)
        decoder_bucket = self._decoder_bucket(rows)
        route = "decoder graph" if decoder_bucket is not None else "decoder eager"
        if route not in self._engaged_logged:
            self._engaged_logged.add(route)
            logger.info(
                f"prefill breakable graph {route!s} ENGAGED: bucket={bucket:d} "
                f"rows={rows:d} decoder_bucket={decoder_bucket!s}"
            )
        with self._padded_to(ctx, bucket):
            encoder.capture.replay(valid_rows=num_tokens)
            narrowed = model.narrowing_forward(encoder.state, ctx)
            if narrowed.rows != rows:
                raise RuntimeError(
                    f"prefill narrowing yielded {narrowed.rows} rows; the model "
                    f"reported {rows} for this forward"
                )
            if decoder_bucket is None:
                hidden, captured = model.decoder_forward(narrowed, ctx)
            else:
                decoder = self._decoders[decoder_bucket]
                narrowed.land_into(decoder.statics)
                decoder.capture.replay(valid_rows=rows)
                hidden, captured = decoder.output.sliced(rows)
        return model.finish_forward(hidden, list(captured), ctx)

    def _decoder_bucket(self, rows: int) -> int | None:
        """Smallest decoder bucket >= ``rows``, or ``None`` to run the decoder eager."""
        idx = bisect.bisect_left(self.decoder_buckets, rows)
        if idx == len(self.decoder_buckets):
            return None
        return self.decoder_buckets[idx]

    def _merged_capture_bs(self, bucket: int, ctx: ForwardContext) -> int | None:
        """Smallest captured request capacity, including native dummy-token room.

        Padding belongs only to KDA execution metadata. The live context, MLA
        breaks, logits and scheduler continue to see the real request count.
        """
        if (
            self.dp_size != 1
            or self.attn_backend.step_counter is not None
            or not ctx.forward_mode.is_extend()
        ):
            return None
        return min(
            (
                bs
                for tokens, bs in self._captures
                if tokens == bucket
                and bs is not None
                and bs >= ctx.bs
                and ctx.input_num_tokens + bs - ctx.bs <= bucket
            ),
            default=None,
        )

    def _replay_bucket(self, ctx: ForwardContext) -> int | None:
        """The captured bucket this forward replays, or ``None`` to run eager.

        Pure-extend AND mixed extend+decode batches are eligible for the ordinary
        capture: attention breaks read the LIVE context and dispatch the split.
        Replay may select an inline KDA variant for a compatible pure extend
        after this bucket check; pure decode is the decode graph's job.
        Two ctx fields are
        baked into the captured segments rather than rebound at replay -- the
        draft first-step row narrowing (``draft_narrowing``) and the
        ``capture_hidden_mode`` aux-hidden capture -- so a live forward carrying
        different values falls back to eager rather than silently dropping the
        reduce / mismatching aux. Prefix caching (cache hits and chunked-prefill
        chunks 2+) IS eligible: the prefix changes attention metadata, not the
        input-token count. Ordinary breaks read live metadata; inline KDA uses
        refreshed fixed-address metadata. Under DP only the ordinary route is
        used, and the replicated token counts determine its EP all-to-all shape.
        """
        if self.disable or ctx.forward_mode is None:
            return None
        if ctx.num_extends <= 0:
            return None
        if not (ctx.forward_mode.is_extend() or ctx.forward_mode.is_mixed()):
            return None
        if ctx.draft_narrowing is not None:
            return None
        if ctx.capture_hidden_mode != self._captured_hidden_mode:
            return None
        bucket = self._select_bucket(ctx)
        if bucket is None or not self._has_bucket(bucket):
            return None
        # A full token bucket may have no room for padded native scan slots.
        # Reuse the next existing token bucket; never capture while serving.
        for candidate in sorted(
            {
                tokens
                for tokens, bs in self._captures
                if bs is not None and tokens >= bucket
            }
        ):
            if self._merged_capture_bs(candidate, ctx) is not None:
                return candidate
        return bucket

    def _has_bucket(self, bucket: int) -> bool:
        """Whether ``bucket`` has a captured ordinary graph (or encoder graph)."""
        if self._narrowing is not None:
            return bucket in self._encoders
        return (bucket, None) in self._captures

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

        ``--disable-cuda-graph-padding`` deliberately does NOT apply here: the
        bucket ladder IS the padding scheme (real token counts almost never
        equal a bucket, so honoring the flag reduced the prefill graph to
        exact matches -- effectively off for ragged traffic). The flag keeps
        its decode-wrapper meaning, where padding trades wasted compute.
        """
        idx = bisect.bisect_left(self.capture_buckets, num_tokens)
        if idx == len(self.capture_buckets):
            return None
        return self.capture_buckets[idx]

    @contextmanager
    def _padded_to(self, ctx: ForwardContext, bucket: int):
        """Publish ``ctx`` as the ambient live context, pinned to the padded bucket.

        The graph replays over ``bucket`` tokens. Ordinary attention breaks read
        live metadata, then clear padding in their output handoffs. Inline KDA
        reads refreshed fixed-address capacity metadata and clears output padding
        inside the graph. Pin
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
            # The replay mode actually taken (mirrors _select_bucket), a DP-debug anchor.
            f"prefill breakable graph ENGAGED ({kind!s}): bucket={bucket:d} dp="
            f"{self.dp_size > 1 and ctx.global_num_tokens is not None!s} mode="
            f"{ctx.forward_mode!s} "
            "(mixed prefill+decode batches supported)",
        )
