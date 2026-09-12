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
Capture borrows the decode
:class:`~tokenspeed.runtime.execution.forward_step.ForwardStepRunner`'s
stream; buckets share one private mempool, deliberately not the decode graphs'
pool (see :meth:`capture`). At serving time
the executor's target-forward dispatch is a simple
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
from types import SimpleNamespace
from typing import TYPE_CHECKING, NamedTuple

import torch
import tqdm

from tokenspeed.runtime.execution.breakable_cuda_graph import (
    BreakableCapture,
    active_forward,
)
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
)
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
    dense at the top. Each bucket's step is the largest power of two <= size/8,
    floored at 16 tokens and capped at 512 so the absolute worst case stays
    bounded at the top end. The ~12.5% tail that step implies holds only where
    size/8 exceeds the floor: below ~128 tokens the floor dominates and the
    relative waste grows sharply (17 -> 32 pads 88%, 1 -> 16 pads 1500%),
    which is where a graphed forward is least likely to pay for itself. Dense
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
    """A bucket's captured inner-forward outputs at stable shared addresses."""

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
        self.drafter = drafter
        self.num_warmup = num_warmup
        self.dp_size = config.data_parallel_size

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
        # One captured graph per padded token bucket.
        self._captures: dict[int, BreakableCapture] = {}
        self._outputs: dict[int, CapturedForward] = {}
        self._hidden_output_buf: torch.Tensor | None = None
        self._aux_output_bufs: list[torch.Tensor] | None = None

    # ------------------------------------------------------------------
    # Graph capture
    # ------------------------------------------------------------------

    def capture(self, decode_wrapper: ForwardStepRunner | None = None) -> None:
        """Capture one breakable graph per token bucket (no-op when disabled).

        ``decode_wrapper`` supplies the shared capture stream (used here only,
        not stored). Buckets share one PRIVATE mempool (first capture allocates
        it), so reclaimable graph intermediates stay ~the largest bucket's
        peak. Retained hidden and aux outputs instead alias buffers sized for
        the largest bucket; retaining each graph-private output would make
        their memory grow with the sum of all buckets. The private pool is
        never the decode graphs' pool: eager ops cache raw pointers to
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
        self.attn_backend.init_prefill_graph_state(
            max_num_tokens=max(self.capture_buckets),
            max_bs=int(self.config.max_num_seqs)
            // max(int(self.config.data_parallel_size), 1),
        )
        with maybe_inference_mode():
            self._capture_all_buckets(decode_wrapper)

    def _capture_all_buckets(self, decode_wrapper: ForwardStepRunner | None) -> None:
        rank = self.config.global_rank
        buckets = sorted(self.capture_buckets, reverse=True)
        capture_range = tqdm.tqdm(buckets) if rank == 0 else buckets
        for bucket in capture_range:
            if rank == 0:
                avail_mem = get_available_gpu_memory(
                    self.config.device, self.config.gpu_id, empty_cache=False
                )
                capture_range.set_description(
                    f"Capturing prefill buckets ({bucket=} {avail_mem=:.2f} GB)"
                )
            self._ctx = self.make_dummy_batch(bucket)
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
        self, bucket: int, decode_wrapper: ForwardStepRunner | None
    ) -> None:
        """Warm up and capture the breakable graph for ``bucket`` from the buffers."""
        sample_outputs = None
        for _ in range(self.num_warmup):
            sample_outputs = self._run_inner(bucket)
        if self._hidden_output_buf is None:
            if sample_outputs is None:
                sample_outputs = self._run_inner(bucket)
            self._allocate_output_buffers(CapturedForward(*sample_outputs))
        # Only the first bucket needs it; holding it pins a bucket of rows.
        sample_outputs = None
        torch.cuda.synchronize()
        stream = decode_wrapper.stream if decode_wrapper is not None else None
        cap = BreakableCapture(pool=self._pool, stream=stream)
        with cap:
            outputs = CapturedForward(*self._run_inner(bucket))
            self._outputs[bucket] = self._store_outputs(bucket, outputs)
        if self._pool is None:
            self._pool = cap.pool  # share the pool across all subsequent buckets
        cap.replay()  # capture records kernels without executing; smoke-test replay
        self._captures[bucket] = cap

    def _allocate_output_buffers(self, outputs: CapturedForward) -> None:
        max_bucket = max(self.capture_buckets)
        self._hidden_output_buf = outputs.hidden_states.new_empty(
            max_bucket, outputs.hidden_states.shape[1]
        )
        if outputs.aux_hidden_states is not None:
            self._aux_output_bufs = [
                aux.new_empty(max_bucket, aux.shape[1])
                for aux in outputs.aux_hidden_states
            ]

    def _store_outputs(self, bucket: int, outputs: CapturedForward) -> CapturedForward:
        assert self._hidden_output_buf is not None
        hidden = self._hidden_output_buf[:bucket]
        hidden.copy_(outputs.hidden_states)
        if outputs.aux_hidden_states is None:
            return CapturedForward(hidden, None)
        assert self._aux_output_bufs is not None
        aux_views = []
        for destination, output in zip(
            self._aux_output_bufs, outputs.aux_hidden_states, strict=True
        ):
            view = destination[:bucket]
            view.copy_(output)
            aux_views.append(view)
        # Safe alias: replays serialize and callers consume outputs immediately.
        return CapturedForward(hidden, aux_views)

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
        ib = self.input_buffers
        if self.config.model_is_mrope:
            positions = ib.mrope_positions_buf[:, :num_tokens]
        else:
            positions = ib.positions_buf[:num_tokens]
        return self.inner_model(
            ib.input_ids_buf[:num_tokens],
            positions,
            self._ctx,
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

    def make_dummy_batch(self, num_tokens: int) -> ForwardContext:
        """Populate the static buffers + attention metadata for a dummy extend
        forward of ``num_tokens`` tokens, and return its ForwardContext.

        The tokens are split across ``ceil(num_tokens / context_len)`` dummy
        requests so no single request exceeds the model context length: every
        per-request structure (page-table rows, DSA indexer tables) is sized
        for ``physical_context_len``, and a longer fabricated request indexes
        past them. It does not bound the request-indexed buffers, which are
        sized ``max_num_seqs // dp``: a bucket above ``context_len * max_bs``
        overflows them and kills the boot (pre-existing; ``_autotune`` clamps
        its token count for exactly that reason, the bucket ladder does not). A real forward carries more than ``context_len`` tokens only as
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
        bs = max(1, -(-num_tokens // max_req_tokens))
        seq_lens = [max_req_tokens] * (bs - 1) + [
            num_tokens - max_req_tokens * (bs - 1)
        ]
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
        with self._padded_to(ctx, bucket):
            self._captures[bucket].replay(valid_rows=num_tokens)
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
        draft first-step row narrowing (``draft_narrowing``) and the
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
        if ctx.draft_narrowing is not None:
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

        The graph replays over ``bucket`` (padded) tokens; attention metadata stays
        at the real count (set upstream), so the eager attention break only touches
        real tokens; eager-break handoffs clear the padded rows before the
        following graph segment consumes them. Pin
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
