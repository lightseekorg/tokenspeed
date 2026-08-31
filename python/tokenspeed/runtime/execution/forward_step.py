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

import bisect
import gc
import queue
from collections.abc import Callable
from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
import tqdm

from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
)
from tokenspeed.runtime.execution.graph_ptr_guard import (
    graph_debug_enabled,
    snapshot_graph_metadata,
    verify_graph_metadata,
)
from tokenspeed.runtime.layers.attention.backends.base import (
    init_backend_cuda_graph_state,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    compute_max_logical_pages_for_capture,
)
from tokenspeed.runtime.sampling.backends.base import CUDA_GRAPH_VARIANT_DEFAULT
from tokenspeed.runtime.sampling.sampling_batch_info import SamplingBatchInfo
from tokenspeed.runtime.utils import (
    get_available_gpu_memory,
    get_colorful_logger,
)
from tokenspeed.runtime.utils.nvtx import nvtx_range

if TYPE_CHECKING:
    from tokenspeed.runtime.execution.drafter.base import BaseDrafter
    from tokenspeed.runtime.execution.input_buffer import InputBuffers
    from tokenspeed.runtime.execution.model_executor import ModelExecutorConfig
    from tokenspeed.runtime.execution.runtime_states import RuntimeStates
    from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
    from tokenspeed.runtime.layers.attention.kv_cache.base import CachePool
    from tokenspeed.runtime.sampling.backends.base import SamplingBackend

logger = get_colorful_logger(__name__)


_is_capture_mode = False
_is_cuda_graph_phase = False


def get_is_capture_mode() -> bool:
    return _is_capture_mode


def get_is_cuda_graph_phase() -> bool:
    return _is_cuda_graph_phase


def _should_update_mamba_state_after_mtp_verify(
    drafter, attn_backend, forward_mode: ForwardMode
) -> bool:
    return (
        drafter is not None
        and forward_mode.is_decode()
        and hasattr(attn_backend, "update_mamba_state_after_mtp_verify")
    )


@contextmanager
def freeze_gc(enable_cudagraph_gc: bool):
    """
    Optimize garbage collection during CUDA graph capture.
    Clean up, then freeze all remaining objects from being included
    in future collections if GC is disabled during capture.
    """
    gc.collect()
    should_freeze = not enable_cudagraph_gc
    if should_freeze:
        gc.freeze()
    try:
        yield
    finally:
        if should_freeze:
            gc.unfreeze()
            gc.collect()


def get_batch_sizes_to_capture(config: ModelExecutorConfig):
    capture_bs = config.cudagraph_capture_sizes
    max_bs = config.max_num_seqs // max(config.data_parallel_size, 1)

    if capture_bs is None:
        if config.disable_cuda_graph_padding:
            capture_bs = list(range(1, 33)) + [64, 96, 128, 160]
        else:
            capture_bs = [1, 2, 4] + [i * 8 for i in range(1, 21)]

    if max(capture_bs) > max_bs:
        capture_bs = list(sorted(set(capture_bs + [max_bs - 1] + [max_bs])))

    effective_max = min(config.max_cudagraph_capture_size, max_bs)
    capture_bs = [bs for bs in capture_bs if 0 < bs <= effective_max]
    return capture_bs


global_graph_memory_pool = None


class DeepEPCudaGraphRunnerAdapter:
    """Manages DeepEP dispatch mode consistency across CUDA graph capture/replay.

    During capture the forward pass (including DeepEP low-latency RDMA
    dispatch/combine) is recorded. On replay the Python wrapper code
    that normally sets dispatch mode and manages the RDMA workspace
    never re-executes. This adapter restores both before each replay.

    Follows the same CUDA graph replay contract as the upstream DeepEP runner.
    """

    def __init__(self):
        self._active = False

    @staticmethod
    def _get_buffer_cls():
        try:
            from tokenspeed_kernel.ops.communication.deep_ep import (
                DeepEPBuffer,
            )

            return DeepEPBuffer
        except ImportError:
            return None

    def capture(self):
        """Call before ``torch.cuda.graph()`` capture."""
        cls = self._get_buffer_cls()
        if cls is None or cls._buffer is None:
            return
        self._active = True
        cls.set_dispatch_mode_as_low_latency()

    def replay(self):
        """Call before every ``graph.replay()``; restores dispatch mode
        and resets RDMA workspace so stale sync state doesn't corrupt
        the combine kernel across replays."""
        if not self._active:
            return
        cls = self._get_buffer_cls()
        if cls is None or cls._buffer is None:
            return
        cls.set_dispatch_mode_as_low_latency()
        cls.clean_buffer()


class ForwardStepRunner:
    """Owns one forward step end to end: metadata prep, then execution.

    Every decode prepares its metadata through the single refresh path
    (``refresh_decode_metadata``, see docs/design/unified_path.md); extend and
    mixed batches construct theirs via ``init_forward_metadata``. Execution is
    then either replaying a captured CUDA graph (decode at a captured batch
    size) or running the same forward Python the graph recorded.

    Callers always use the same interface::

        output_tokens, output_lengths, output_logprobs = runner(
            bs, ctx, sampling_info, page_table,
            extend_with_prefix=..., extend_prefix_lens=...,
        )
    """

    def __init__(
        self,
        forward_func: Callable,
        attn_backend: AttentionBackend,
        token_to_kv_pool: CachePool,
        input_buffers: InputBuffers,
        config: ModelExecutorConfig,
        draft_attn_backend: AttentionBackend | None = None,
        draft_token_to_kv_pool: CachePool | None = None,
        drafter: BaseDrafter | None = None,
        capturable_grammar=None,
        eager_grammar_buffers=None,
        sampling_backend: SamplingBackend | None = None,
        runtime_states: RuntimeStates | None = None,
        page_table: torch.Tensor | None = None,
        decode_graph_supported: bool = True,
    ):
        self.config = config
        # The executor's address-stable staged full-history table — the same
        # object every live __call__ passes as page_table. Capture hands it
        # to the backends so the default capture's idle refresh runs the very
        # arm replay runs (its dummy rows are zero at capture time).
        self.page_table = page_table
        self.attn_backend = attn_backend
        self.draft_attn_backend = draft_attn_backend
        self.draft_token_to_kv_pool = draft_token_to_kv_pool
        self.token_to_kv_pool = token_to_kv_pool
        self.drafter = drafter
        self.sampling_backend = sampling_backend
        self.input_buffers = input_buffers
        self.capturable_grammar = capturable_grammar
        self.eager_grammar_buffers = eager_grammar_buffers
        self.runtime_states = runtime_states
        self.disable_padding = config.disable_cuda_graph_padding
        self.enable_cudagraph_gc = config.enable_cudagraph_gc
        self.device = config.device
        self.device_module = torch.get_device_module(self.device)
        self.gpu_id = config.gpu_id
        self.global_rank = config.global_rank
        # Physical extent: capture tables must cover the spec-verify overshoot
        # of a finished request lingering one overlap step.
        self.context_len = config.physical_context_len
        self.vocab_size = config.vocab_size
        self.grammar_backend = config.grammar_backend
        self.capture_bs = get_batch_sizes_to_capture(config)
        # Two distinct maxima: graphs exist only for the capture ladder
        # (bounded by max_cudagraph_capture_size), but the persistent decode
        # buffers cover every decode batch this rank can serve, so a decode
        # above the ladder runs the same refresh path with no graph — the
        # ladder is a performance subset, never a capacity limit.
        self.max_capture_bs = max(self.capture_bs)
        self.max_decode_bs = max(
            config.max_num_seqs // max(config.data_parallel_size, 1),
            self.max_capture_bs,
        )
        self.max_tokens_per_req = (
            config.spec_num_tokens if config.spec_algo is not None else 1
        )
        self.overlap_schedule_depth = config.overlap_schedule_depth
        self.dp_size = config.data_parallel_size
        self.world_size = config.world_size
        # User intent (enforce_eager) OR a backend-declared restriction
        # (resolve_cuda_graph_support in ModelExecutor); the unified refresh
        # still serves eager decode either way.
        self.disable = config.enforce_eager or not decode_graph_supported
        # Backends alias their cache_seqlens buffer. Draft backend aliases
        # the drafter-owned draft_seq_lens to keep InputBuffers read-only.
        init_backend_cuda_graph_state(
            attn_backend,
            self.max_decode_bs,
            cache_group_specs=tuple(token_to_kv_pool.arena.cache_group_specs),
            cache_group_page_counts=(token_to_kv_pool.arena.cache_group_page_counts),
            max_tokens_per_req=self.max_tokens_per_req,
            overlap_schedule_depth=self.overlap_schedule_depth,
        )
        if draft_attn_backend is not None:
            init_backend_cuda_graph_state(
                draft_attn_backend,
                self.max_decode_bs,
                cache_group_specs=tuple(draft_token_to_kv_pool.arena.cache_group_specs),
                cache_group_page_counts=(
                    draft_token_to_kv_pool.arena.cache_group_page_counts
                ),
                max_tokens_per_req=self.max_tokens_per_req,
                overlap_schedule_depth=self.overlap_schedule_depth,
            )
            # Drafter reads the TARGET's group tables (_draft_group_tables):
            # its page-size view must match the target's hetero geometry.
            target_ps = getattr(attn_backend, "group_block_granularities", None)
            draft_ps = getattr(draft_attn_backend, "group_block_granularities", None)
            if target_ps and draft_ps is not None:
                for gid in self._draft_cache_group_ids():
                    if gid in target_ps:
                        draft_ps[gid] = target_ps[gid]

            # Target and draft resolve the same batch-ordered full-history
            # table (the target from cache metadata, the draft from the
            # published draft page table), so both backends would compute
            # identical block_kv_indices. When the backing buffer shapes/dtypes
            # also line up, point the draft backend at the target's buffer and
            # skip its gather+copy in the replay path: the target's metadata
            # prep runs first and populates the shared buffer (see
            # refresh_decode_metadata).
            target_kv = getattr(attn_backend, "decode_cuda_graph_kv_indices", None)
            draft_kv = getattr(draft_attn_backend, "decode_cuda_graph_kv_indices", None)
            if (
                target_kv is not None
                and draft_kv is not None
                and target_kv.shape == draft_kv.shape
                and target_kv.dtype == draft_kv.dtype
            ):
                draft_attn_backend.decode_cuda_graph_kv_indices = target_kv
                draft_attn_backend._page_table_aliased = True

        self.graphs: dict[tuple[str, int], object] = {}
        self.output_buffers: dict[tuple[str, int], tuple] = {}
        # TOKENSPEED_GRAPH_DEBUG=1: capture-time tensor-identity snapshots,
        # re-verified before every replay (graph_ptr_guard). Off by default —
        # replays then pay a single bool check.
        self._graph_debug = graph_debug_enabled()
        self._metadata_snapshots: dict[tuple[str, int], dict[str, dict]] = {}

        self._forward_func: Callable | None = forward_func
        self.deepep_adapter = DeepEPCudaGraphRunnerAdapter()

    # ------------------------------------------------------------------
    # Graph capture
    # ------------------------------------------------------------------

    def capture(self):
        """
        Capture CUDA graphs for all configured batch sizes.

        Args:
            forward_func: ModelExecutor.forward_step(bs, ctx, sampling_info).
        """
        rank = self.global_rank
        with freeze_gc(self.enable_cudagraph_gc):
            self.stream = self.device_module.Stream()
            # Capture backend-declared sampler variants explicitly.
            capture_items = [
                (variant, bs)
                for variant in self._cuda_graph_capture_variants()
                for bs in sorted(self.capture_bs, reverse=True)
            ]
            capture_range = tqdm.tqdm(capture_items) if rank == 0 else capture_items
            if rank == 0:
                logger.info("Capturing batches: %s", self.capture_bs)
            for variant, bs in capture_range:
                if rank == 0:
                    avail_mem = get_available_gpu_memory(
                        self.device, self.gpu_id, empty_cache=False
                    )
                    variant_desc = (
                        ""
                        if variant == CUDA_GRAPH_VARIANT_DEFAULT
                        else f" variant={variant}"
                    )
                    capture_range.set_description(
                        f"Capturing batches ({bs=}{variant_desc} {avail_mem=:.2f} GB)"
                    )
                graph, output_buffers = self._capture_one(bs, variant=variant)
                self.graphs[(variant, bs)] = graph
                self.output_buffers[(variant, bs)] = output_buffers

    def _cuda_graph_capture_variants(self) -> tuple[str, ...]:
        if self.sampling_backend is None:
            return (CUDA_GRAPH_VARIANT_DEFAULT,)
        variants = self.sampling_backend.cuda_graph_capture_variants(
            self.max_tokens_per_req
        )
        if not variants:
            return (CUDA_GRAPH_VARIANT_DEFAULT,)
        deduped = tuple(dict.fromkeys((CUDA_GRAPH_VARIANT_DEFAULT, *variants)))
        return deduped

    def _prepare_sampling_capture(self, bs: int, variant: str) -> None:
        if self.sampling_backend is None:
            return
        self.sampling_backend.prepare_capture_variant(
            bs=bs,
            num_tokens_per_req=self.max_tokens_per_req,
            variant=variant,
        )

    def _cuda_graph_replay_variant(self) -> str:
        if self.sampling_backend is None:
            return CUDA_GRAPH_VARIANT_DEFAULT
        return self.sampling_backend.cuda_graph_replay_variant(self.max_tokens_per_req)

    def _cuda_graph_key(self, bs: int) -> tuple[str, int]:
        variant = self._cuda_graph_replay_variant()
        key = (variant, bs)
        if key in self.graphs:
            return key
        if variant != CUDA_GRAPH_VARIANT_DEFAULT:
            captured_variants = sorted(
                graph_variant
                for graph_variant, graph_bs in self.graphs
                if graph_bs == bs
            )
            raise RuntimeError(
                "Sampling backend requested CUDA graph variant "
                f"{variant!r} for batch size {bs}, but it was not captured. "
                f"Captured variants for this batch size: {captured_variants}."
            )
        return (CUDA_GRAPH_VARIANT_DEFAULT, bs)

    def _has_cuda_graph_for_bs(self, bs: int) -> bool:
        return (CUDA_GRAPH_VARIANT_DEFAULT, bs) in self.graphs

    def _verify_graph_metadata(self, graph_key: tuple[str, int]) -> None:
        """Assert the refresh rebound the tensors the captured graph reads."""
        snapshots = self._metadata_snapshots.get(graph_key)
        if snapshots is None:
            return
        context = f"variant={graph_key[0]!r}, bs={graph_key[1]}"
        verify_graph_metadata(
            self.attn_backend, snapshots["target"], context=f"target, {context}"
        )
        if self.draft_attn_backend is not None and "draft" in snapshots:
            verify_graph_metadata(
                self.draft_attn_backend, snapshots["draft"], context=f"draft, {context}"
            )

    def _capture_one(self, bs: int, variant: str = CUDA_GRAPH_VARIANT_DEFAULT):
        graph_cls = (
            self.device_module.NPUGraph
            if self.device == "npu"
            else self.device_module.CUDAGraph
        )
        graph = graph_cls()

        capture_forward_mode = ForwardMode.DECODE
        ctx = ForwardContext(
            attn_backend=self.attn_backend,
            token_to_kv_pool=self.token_to_kv_pool,
            bs=bs,
            num_extends=0,
            input_num_tokens=bs * self.max_tokens_per_req,
            forward_mode=capture_forward_mode,
            # A decode graph is only ever replayed when every DP rank is
            # decoding or idle (see _can_use_graph), so capture must record that
            # same answer. Leaving the default False would let capture-time code
            # take a different path than replay -- for DeepEP MoE that means
            # recording the normal-mode legs, whose host-side receive counts
            # cannot be captured at all.
            all_decode_or_idle=True,
            capture_hidden_mode=(
                CaptureHiddenMode.FULL
                if self.drafter is not None
                else CaptureHiddenMode.NULL
            ),
        )

        # For DP mode, global_num_tokens must be set so that the MoE
        # all-gather comm layers know token counts for all DP ranks.
        # During capture, use uniform dummy counts across ranks.
        if self.dp_size > 1:
            ctx.global_num_tokens = [bs * self.max_tokens_per_req] * self.world_size
            # global_bs must ALSO be set at capture. The draft first step's
            # collective sizing (reported via report_collective_sizing) reads
            # global_bs; if left None at capture it records a single-rank
            # layout (fallback branch in comm_manager), but at replay global_bs
            # is the live per-rank batch list -> multi-rank layout. The mismatch
            # makes the captured (frozen-offset) gather read uninitialized
            # symm-mem -> NaN draft logits -> accept_rate 0. Set the matching
            # uniform dummy.
            ctx.global_bs = [bs] * self.world_size

        # The sampler has a single pool-indexed route: greedy requests are
        # served by top_k=1 in the pool buffers, so capture and every replay
        # record the same top_k_top_p sampling path.
        ibd = self.input_buffers
        sampling_info = SamplingBatchInfo(
            req_pool_indices=ibd.req_pool_indices_buf[:bs],
            valid_cache_lengths=(
                self.runtime_states.valid_cache_lengths
                if self.runtime_states is not None
                else None
            ),
            vocab_size=self.vocab_size,
            device=self.device,
        )

        from tokenspeed.runtime.grammar.capturable_grammar import (
            bind_grammar_mask_buf,
        )

        # Bind whichever grammar buffer is active so the captured sampler
        # records the apply_vocab_mask call. At replay, runtime fills the
        # bound buffer in place (hostfunc for capturable, sync H2D for
        # eager) — the captured graph reads from the same memory.
        bind_grammar_mask_buf(
            sampling_info,
            self.eager_grammar_buffers,
            bs,
            spec=self.drafter is not None,
            capturable=self.capturable_grammar,
            grammar_backend=self.grammar_backend,
        )

        def run_once():
            # Dummy add_batch keeps the grammar queue 1:1 with replays —
            # fetch_batch pops once per forward, so warmup + capture
            # would otherwise raise queue.Empty.
            if self.capturable_grammar is not None:
                self.capturable_grammar.add_batch(
                    grammars=[None] * bs, bs=bs, has_candidates=False
                )
            return self._forward_func(bs=bs, ctx=ctx, sampling_info=sampling_info)

        global _is_cuda_graph_phase
        _is_cuda_graph_phase = True

        # Warm the same primary stream used by capture. Capture-only auxiliary
        # branches use this graph phase to warm their own streams serially.
        with self.device_module.stream(self.stream):
            for _ in range(4):
                self.device_module.synchronize()
                dist.barrier()
                self._prepare_sampling_capture(bs=bs, variant=variant)
                # Keep warmup seq_lens >= q_len_per_req so no query row gets an
                # empty causal span; a stale seq_len of 1 overflows to non-finite KV.
                self.input_buffers.seq_lens_buf[:bs].fill_(self.max_tokens_per_req)
                self._init_capture_metadata(bs)
                run_once()
            # Order the reset below after the last warmup's stateful kernels.
            self.device_module.synchronize()

        # Clear any per-pool state that warm-up dirtied at pool row 0,
        # so the graph captures reads against a clean baseline.
        if self.sampling_backend is not None:
            self.sampling_backend.reset_capture_state()

        self.device_module.synchronize()
        dist.barrier()

        # Warmups can switch a backend back to eager metadata objects. Restore
        # the graph-backed metadata immediately before capture so replay-time
        # metadata refreshes update the same tensors recorded by the graph.
        self._init_capture_metadata(bs)

        # Fill sampler buffers OUTSIDE the capture so RNG ops aren't recorded.
        self._prepare_sampling_capture(bs=bs, variant=variant)
        # Warmup forwards can mutate aliased metadata buffers, so refresh
        # them again immediately before graph capture records the final views.
        self._init_capture_metadata(bs)

        self.deepep_adapter.capture()

        global _is_capture_mode
        _is_capture_mode = True
        global global_graph_memory_pool
        graph_kwargs = {"auto_dispatch_capture": True} if self.device == "npu" else {}
        with self.device_module.graph(
            graph,
            pool=global_graph_memory_pool,
            stream=self.stream,
            **graph_kwargs,
        ):
            out = run_once()

        self.device_module.synchronize()
        dist.barrier()
        _is_capture_mode = False
        _is_cuda_graph_phase = False

        # Graph capture records the hostfunc launches without invoking
        # them, so the dummy run_once pushed stays queued — drain it, and
        # reset prev_batch/current_batch so the first real replay's build
        # doesn't advance the matcher from a stale warmup entry.
        if self.capturable_grammar is not None:
            while True:
                try:
                    self.capturable_grammar.queue.get_nowait()
                except queue.Empty:
                    break
            self.capturable_grammar.reset_state()

        global_graph_memory_pool = graph.pool()

        if self._graph_debug:
            # The slots bound by the last _init_capture_metadata are exactly
            # what the graph recorded; snapshot their tensor identities so
            # every replay can assert the refresh still writes them.
            snapshots = {"target": snapshot_graph_metadata(self.attn_backend)}
            if self.draft_attn_backend is not None:
                snapshots["draft"] = snapshot_graph_metadata(self.draft_attn_backend)
            self._metadata_snapshots[(variant, bs)] = snapshots

        return graph, out

    def prewarm_comm_states(self, batch_sizes: tuple[int, ...] = (1,)) -> None:
        """Initialize lazy comm state with capture-style dummy forwards."""
        if self._forward_func is None:
            return

        global _is_cuda_graph_phase
        old_cuda_graph_phase = _is_cuda_graph_phase
        _is_cuda_graph_phase = True
        try:
            for bs in batch_sizes:
                ctx = ForwardContext(
                    attn_backend=self.attn_backend,
                    token_to_kv_pool=self.token_to_kv_pool,
                    bs=bs,
                    num_extends=0,
                    input_num_tokens=bs * self.max_tokens_per_req,
                    forward_mode=ForwardMode.DECODE,
                    # Match _capture_one: the lazy state this warms up (DeepEP
                    # buffers among it) must be the state capture will record.
                    all_decode_or_idle=True,
                    capture_hidden_mode=(
                        CaptureHiddenMode.FULL
                        if self.drafter is not None
                        else CaptureHiddenMode.NULL
                    ),
                )
                if self.dp_size > 1:
                    ctx.global_num_tokens = [
                        bs * self.max_tokens_per_req
                    ] * self.world_size
                    ctx.global_bs = [bs] * self.world_size

                sampling_info = SamplingBatchInfo(
                    req_pool_indices=self.input_buffers.req_pool_indices_buf[:bs],
                    valid_cache_lengths=(
                        self.runtime_states.valid_cache_lengths
                        if self.runtime_states is not None
                        else None
                    ),
                    vocab_size=self.vocab_size,
                    device=self.device,
                )

                from tokenspeed.runtime.grammar.capturable_grammar import (
                    bind_grammar_mask_buf,
                )

                bind_grammar_mask_buf(
                    sampling_info,
                    self.eager_grammar_buffers,
                    bs,
                    spec=self.drafter is not None,
                    capturable=self.capturable_grammar,
                    grammar_backend=self.grammar_backend,
                )

                self.device_module.synchronize()
                dist.barrier()
                self._prepare_sampling_capture(
                    bs=bs,
                    variant=CUDA_GRAPH_VARIANT_DEFAULT,
                )
                self.input_buffers.seq_lens_buf[:bs].fill_(self.max_tokens_per_req)
                self._init_capture_metadata(bs)
                self._forward_func(bs=bs, ctx=ctx, sampling_info=sampling_info)
                self.device_module.synchronize()
                dist.barrier()

                if self.sampling_backend is not None:
                    self.sampling_backend.reset_capture_state()
        finally:
            _is_cuda_graph_phase = old_cuda_graph_phase

    def _capture_group_block_tables(self, bs: int, pool) -> dict | None:
        specs = tuple(pool.arena.cache_group_specs)
        if not specs:
            return None
        out = {}
        for spec in specs:
            max_pages = compute_max_logical_pages_for_capture(
                spec,
                max_context_len=(
                    self.max_tokens_per_req * self.max_decode_bs
                    if self.context_len <= 0
                    else self.context_len
                ),
                max_tokens_per_req=self.max_tokens_per_req,
                overlap_schedule_depth=self.overlap_schedule_depth,
            )
            out[str(spec.group_id)] = torch.zeros(
                (bs, max_pages),
                dtype=torch.int32,
                device=self.device,
            )
        return out

    def _cache_group_ids(self, pool) -> tuple[str, ...]:
        """Group ids for per-group CUDA-graph capture: real tables only
        arrive at replay, so capture needs just the ids to allocate its
        persistent per-group buffers."""
        return tuple(str(spec.group_id) for spec in pool.arena.cache_group_specs)

    def _draft_cache_group_ids(self) -> tuple[str, ...]:
        """Per-group page tables consumed by the drafter.

        DFLASH block decode owns an independent draft page table and must stay
        on that single-table path. Other draft heads share the target's
        page-id space (EAGLE writes its own pool tensors at the same indices),
        so each draft group consumes the target tables for the cache families
        declared by its backend.

        A draft pool that publishes its own specs (Inkling MTP: mixed full/SWA
        depths) names exactly the groups its layers carry. Older draft paths
        without a separate pool select the same families from the target pool.
        """
        if self.draft_attn_backend is None:
            return ()
        if getattr(self.draft_attn_backend, "draft_block_decode", False):
            return ()
        families = frozenset(
            getattr(
                self.draft_attn_backend,
                "cache_consumer_families",
                ("history",),
            )
        )
        draft_arena = getattr(self.draft_token_to_kv_pool, "arena", None)
        draft_specs = getattr(draft_arena, "cache_group_specs", ())
        if draft_specs:
            return tuple(
                str(spec.group_id) for spec in draft_specs if spec.family in families
            )
        return tuple(
            str(spec.group_id)
            for spec in self.token_to_kv_pool.arena.cache_group_specs
            if spec.family in families
        )

    def _draft_group_tables(self, block_tables):
        """Subset of the target's per-group tables the drafter consumes."""
        gids = self._draft_cache_group_ids()
        if not gids or not block_tables:
            return None
        subset = {gid: block_tables[gid] for gid in gids if gid in block_tables}
        return subset or None

    def _init_capture_metadata(self, bs: int):
        capture_kwargs = {}
        if getattr(self.attn_backend, "cache_active_pages_must_be_real", False):
            # Capture-time placeholder tables, only for backends that
            # validate live-page geometry at capture (V4); the others bind
            # persistent buffers from cache_group_ids alone.
            group_block_tables = self._capture_group_block_tables(
                bs,
                self.token_to_kv_pool,
            )
            if group_block_tables is not None:
                capture_kwargs["block_tables"] = group_block_tables
                if self.drafter is not None:
                    capture_kwargs["num_tokens"] = bs * self.max_tokens_per_req
        cache_group_ids = self._cache_group_ids(self.token_to_kv_pool)
        if cache_group_ids:
            capture_kwargs["cache_group_ids"] = cache_group_ids
        if self.page_table is not None:
            capture_kwargs["page_table"] = self.page_table
        self.attn_backend.init_forward_metadata_capture_cuda_graph(
            bs,
            self.input_buffers.req_pool_indices_buf[:bs],
            self.input_buffers.seq_lens_buf[:bs],
            ForwardMode.DECODE,
            **capture_kwargs,
        )
        if self.draft_attn_backend is not None:
            draft_kwargs = {}
            if self.draft_token_to_kv_pool is not None and getattr(
                self.draft_attn_backend, "cache_active_pages_must_be_real", False
            ):
                draft_group_block_tables = self._capture_group_block_tables(
                    bs,
                    self.draft_token_to_kv_pool,
                )
                if draft_group_block_tables is not None:
                    draft_kwargs["block_tables"] = draft_group_block_tables
                    draft_kwargs["num_tokens"] = bs * self.max_tokens_per_req
            draft_group_ids = self._draft_cache_group_ids()
            if draft_group_ids:
                draft_kwargs["cache_group_ids"] = draft_group_ids
            if self.drafter is not None:
                # Same staged table the draft's live refresh reads
                # (_prepare_decode_metadata passes drafter.cache_view.table).
                draft_kwargs["page_table"] = self.drafter.cache_view.table
            # Drafter mutates seq_lens_buf in place per step; backends alias.
            self.draft_attn_backend.init_forward_metadata_capture_cuda_graph(
                bs,
                self.input_buffers.req_pool_indices_buf[:bs],
                self.input_buffers.seq_lens_buf[:bs],
                ForwardMode.DECODE,
                **draft_kwargs,
            )

    def _idle_block_tables(self, padded_bs: int) -> dict | None:
        """Minimal per-group tables for the bs==0 idle replay: all rows are
        dummy rows, so one column of page-0 entries per group is valid.
        None when the pool publishes no groups."""
        specs = tuple(self.token_to_kv_pool.arena.cache_group_specs)
        if not specs:
            return None
        table = torch.zeros((padded_bs, 1), dtype=torch.int32, device=self.device)
        return {str(spec.group_id): table for spec in specs}

    @staticmethod
    def _pad_block_tables_to_padded_bs(
        block_tables: dict,
        *,
        actual_bs: int,
        padded_bs: int,
        pad_value: int = -1,
    ) -> dict:
        """Pad each table with dummy rows up to ``padded_bs``.

        Group-table captures pass page 0; backends using a sentinel tail keep
        -1. See the MHA replay guard for the padding contract.
        """
        if actual_bs < 0 or padded_bs < actual_bs:
            raise RuntimeError(
                "Grouped cache table batch contract requires "
                f"0 <= actual_bs <= padded_bs, got {actual_bs=} {padded_bs=}"
            )
        out = {}
        for key, table in block_tables.items():
            if not isinstance(table, torch.Tensor):
                out[key] = table
                continue
            if table.ndim < 1:
                raise RuntimeError(
                    f"Grouped cache table {key!r} must have a batch dimension"
                )
            rows = int(table.shape[0])
            if rows not in {actual_bs, padded_bs}:
                raise RuntimeError(
                    f"Grouped cache table {key!r} has {rows} rows; expected "
                    f"actual_bs={actual_bs} or padded_bs={padded_bs}"
                )
            if rows == padded_bs:
                out[key] = table
                continue
            out[key] = torch.nn.functional.pad(
                table,
                (0, 0, 0, padded_bs - rows),
                value=pad_value,
            )
        return out

    @staticmethod
    def _pad_offsets_to_padded_bs(
        base_offsets: dict,
        *,
        actual_bs: int,
        padded_bs: int,
    ) -> dict:
        if padded_bs <= actual_bs:
            return base_offsets
        out = {}
        for key, off in base_offsets.items():
            if not isinstance(off, torch.Tensor):
                out[key] = off
                continue
            rows = int(off.shape[0])
            if rows == padded_bs:
                out[key] = off
                continue
            # Base 0: padded rows have no real request; the paired padded
            # table row is invalid (-1).
            out[key] = torch.nn.functional.pad(
                off,
                (0, padded_bs - rows),
                value=0,
            )
        return out

    def _prepare_decode_metadata(
        self,
        padded_bs: int,
        actual_bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        page_table: torch.Tensor,
        forward_mode: ForwardMode,
        *,
        use_graph: bool,
        **kwargs,
    ):
        """The single decode metadata path — graph replay AND eager decode.

        Assembles per-group tables once (row padding is a no-op when
        ``padded_bs == actual_bs``, i.e. every eager call), then refreshes the
        target backend's persistent decode buffers and the draft's. The target
        must refresh first: a draft whose kv-indices buffer is aliased to the
        target's (``_page_table_aliased``) reads what the target's refresh
        populated.
        """
        block_table_base_offsets = kwargs.pop("block_table_base_offsets", None)
        block_tables = kwargs.pop("block_tables", None)
        group_placeholder_tables = kwargs.pop("group_placeholder_tables", None)

        # The bs==0 idle replay carries no live tables: synthesize minimal
        # valid ones so the backends' stale-table guards hold. Eager idle
        # bypasses the wrapper entirely (see execute_idle_forward).
        if actual_bs == 0 and use_graph:
            if group_placeholder_tables is None and getattr(
                self.attn_backend, "cache_active_pages_must_be_real", False
            ):
                # A backend that validates live-page geometry (V4) needs
                # full-width capture placeholders on idle rather than the
                # 1-column zero tables below.
                group_placeholder_tables = self._capture_group_block_tables(
                    padded_bs,
                    self.token_to_kv_pool,
                )
            if not block_tables:
                block_tables = self._idle_block_tables(padded_bs)
        if group_placeholder_tables is not None:
            table_bs = next(
                (
                    int(table.shape[0])
                    for table in group_placeholder_tables.values()
                    if isinstance(table, torch.Tensor)
                ),
                int(req_pool_indices.shape[0]),
            )
            group_placeholder_tables = self._pad_block_tables_to_padded_bs(
                group_placeholder_tables,
                actual_bs=table_bs,
                padded_bs=padded_bs,
            )
            if block_table_base_offsets is not None:
                block_table_base_offsets = self._pad_offsets_to_padded_bs(
                    block_table_base_offsets,
                    actual_bs=actual_bs,
                    padded_bs=padded_bs,
                )
            kwargs["block_tables"] = group_placeholder_tables
            if block_table_base_offsets is not None:
                kwargs["block_table_base_offsets"] = block_table_base_offsets
        padded_block_tables = None
        if block_tables is not None and (
            not getattr(self.attn_backend, "tables_self_padding", False)
            or (
                self.draft_attn_backend is not None
                and not getattr(
                    self.draft_attn_backend,
                    "tables_self_padding",
                    False,
                )
            )
        ):
            # Normalize once so target and draft consume the same authoritative
            # replay rows. In particular, an actual batch smaller than the
            # selected graph batch must not reach the draft as an unpadded map.
            padded_block_tables = self._pad_block_tables_to_padded_bs(
                block_tables,
                actual_bs=actual_bs,
                padded_bs=padded_bs,
                pad_value=0,
            )
        if block_tables is not None:
            if getattr(self.attn_backend, "tables_self_padding", False):
                # Backend pads dummy rows itself; F.pad would reallocate tables and break storage sharing.
                kwargs["block_tables"] = block_tables
            else:
                kwargs["block_tables"] = padded_block_tables
        if not use_graph and block_table_base_offsets is not None:
            # Live replay reads offsets through cache_metadata; only the eager
            # path and the padded placeholder branch above feed them directly.
            kwargs["block_table_base_offsets"] = block_table_base_offsets
        # Pure decode: ctx.input_num_tokens == bs * max_tokens_per_req, so
        # this is the same value both paths passed before unification
        # (backends that key off their own buffers ignore it).
        kwargs["num_tokens"] = padded_bs * self.max_tokens_per_req
        self.attn_backend.refresh_decode_metadata(
            padded_bs,
            actual_bs,
            req_pool_indices,
            seq_lens,
            forward_mode=forward_mode,
            page_table=page_table,
            for_graph_replay=use_graph,
            **kwargs,
        )
        if self.draft_attn_backend is not None:
            # Seed the drafter-owned buffer for the round: models and the
            # DFLASH drafter read it directly, and the refresh below snapshots
            # it into the backend's own seq_lens buffer. Drafters republish
            # their in-loop edits each step through the same refresh (block
            # drafters) or advance_draft_forward_metadata.
            draft_seq_lens = self.drafter.draft_seq_lens_buf[:padded_bs]
            draft_seq_lens.copy_(seq_lens[:padded_bs])
            draft_attn_kwargs = {}
            if group_placeholder_tables is not None:
                # Idle replay: the padded placeholder pair travels together.
                draft_attn_kwargs["block_tables"] = group_placeholder_tables
                if block_table_base_offsets is not None:
                    draft_attn_kwargs["block_table_base_offsets"] = (
                        block_table_base_offsets
                    )
            elif not use_graph and block_table_base_offsets is not None:
                # Live replay reads offsets through cache_metadata; only
                # eager feeds them directly.
                draft_attn_kwargs["block_table_base_offsets"] = block_table_base_offsets
            draft_attn_kwargs["num_tokens"] = padded_bs * self.max_tokens_per_req
            draft_table_source = (
                block_tables
                if getattr(
                    self.draft_attn_backend,
                    "tables_self_padding",
                    False,
                )
                else padded_block_tables
            )
            draft_group_tables = self._draft_group_tables(draft_table_source)
            if draft_group_tables is not None:
                draft_attn_kwargs["block_tables"] = draft_group_tables
            self.draft_attn_backend.refresh_decode_metadata(
                padded_bs,
                actual_bs,
                req_pool_indices,
                draft_seq_lens,
                forward_mode=ForwardMode.DECODE,
                page_table=self.drafter.cache_view.table,
                for_graph_replay=use_graph,
                **draft_attn_kwargs,
            )

    @nvtx_range("attn_meta_prep", color="orange")
    def _init_forward_metadata(
        self,
        padded_bs: int,
        num_extends: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        page_table: torch.Tensor,
        forward_mode: ForwardMode,
        **kwargs,
    ):
        """Extend/mixed path — construct metadata for the upcoming forward.

        Pure decode goes through ``_prepare_decode_metadata``; this arm keeps
        the dynamic-shape prefill construction.
        """
        self.attn_backend.init_forward_metadata(
            bs=padded_bs,
            num_extends=num_extends,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            page_table=page_table,
            forward_mode=forward_mode,
            **kwargs,
        )
        if self.draft_attn_backend is not None:
            # One draft metadata contract for every backend family, the same
            # two steps as the pure-decode path: the prefill init reads the
            # accepted-prefix seq_lens view, then the unified refresh prepares
            # the draft's step-1+ decode metadata. Drafters republish their
            # in-loop seq_lens edits through advance_draft_forward_metadata /
            # update_draft_forward_metadata. The copy below seeds the
            # drafter-owned buffer for the round (models and the DFLASH
            # drafter read ctx.draft_seq_lens_buf directly).
            draft_seq_lens = self.drafter.draft_seq_lens_buf[:padded_bs]
            draft_seq_lens.copy_(seq_lens[:padded_bs])
            draft_extend_kwargs = dict(kwargs)
            # Drafter consumes its own groups' tables (see _draft_group_tables).
            draft_group_tables = self._draft_group_tables(kwargs.get("block_tables"))
            if draft_extend_kwargs.get("block_tables") is not None:
                draft_extend_kwargs["block_tables"] = draft_group_tables
            # cache_metadata describes the target pool and exposes all target
            # cache groups. A draft consumes the wrapper-selected group subset
            # above plus its staged batch-ordered page table (the same page
            # ids a contract draft would resolve from the metadata), never
            # the target metadata itself — exactly like the decode path.
            draft_extend_kwargs.pop("cache_metadata", None)
            draft_extend_kwargs.pop("forward_batch", None)
            self.draft_attn_backend.init_forward_metadata(
                bs=padded_bs,
                num_extends=num_extends,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                page_table=self.drafter.cache_view.table,
                forward_mode=forward_mode,
                **draft_extend_kwargs,
            )
            # Step two: the draft's step-1+ decode metadata for this round,
            # via the same refresh the pure-decode path uses. Plain 1-token
            # rows — deliberately NOT the packed verify width: V4's
            # packed-decode arm would clobber forward_prefill_metadata with a
            # DECODE-mode object and break the draft's first-step prefill.
            # Refresh writes decode-slot metadata only, so rounds that run no
            # decode steps (vanilla MTP extend depths, DFLASH block decode)
            # simply leave it unread or overwrite it per step.
            draft_refresh_kwargs = {}
            if draft_group_tables is not None:
                draft_refresh_kwargs["block_tables"] = draft_group_tables
            if kwargs.get("block_table_base_offsets") is not None:
                draft_refresh_kwargs["block_table_base_offsets"] = kwargs[
                    "block_table_base_offsets"
                ]
            self.draft_attn_backend.refresh_decode_metadata(
                padded_bs,
                padded_bs,
                req_pool_indices,
                draft_seq_lens,
                forward_mode=ForwardMode.DECODE,
                num_extends=num_extends,
                page_table=self.drafter.cache_view.table,
                **draft_refresh_kwargs,
            )

    def _global_graph_bs(self, ctx: ForwardContext) -> int | None:
        if self.dp_size <= 1 or ctx.global_num_tokens is None:
            return None
        max_num_tokens = max(ctx.global_num_tokens)
        return (max_num_tokens + self.max_tokens_per_req - 1) // self.max_tokens_per_req

    def _can_use_graph(self, bs: int, ctx: ForwardContext) -> bool:
        if self.disable:
            return False
        if not ctx.forward_mode.is_decode():
            return False
        if self.dp_size > 1:
            if not ctx.all_decode_or_idle:
                return False
            global_bs = self._global_graph_bs(ctx)
            if global_bs is None or global_bs == 0:
                return False
            if self.disable_padding:
                return self._has_cuda_graph_for_bs(global_bs)
            return global_bs <= self.max_capture_bs
        if self.disable_padding:
            return self._has_cuda_graph_for_bs(bs)
        return bs <= self.max_capture_bs

    def can_run(self, bs: int, ctx: ForwardContext) -> bool:
        return self._can_use_graph(bs, ctx)

    def padded_bs(self, bs: int, ctx: ForwardContext) -> int:
        return self._padded_bs(bs, ctx)

    def _padded_bs(self, bs: int, ctx: ForwardContext) -> int:
        graph_bs = self._global_graph_bs(ctx)
        target_bs = graph_bs if graph_bs is not None else bs
        index = bisect.bisect_left(self.capture_bs, target_bs)
        return self.capture_bs[index]

    def _pad_graph_req_pool_indices(
        self, active_req_pool_indices: torch.Tensor, padded_bs: int
    ) -> torch.Tensor:
        pad = padded_bs - active_req_pool_indices.shape[0]
        if pad <= 0:
            return active_req_pool_indices
        if self.config.spec_algo in ("DFLASH", "DSPARK"):
            # Route padding rows to the sentinel req-pool slot
            # (max_req_pool_size), not slot 0. The DFLASH draft derives each
            # row's block seq_len from valid_cache_lengths[req_pool], so
            # padding rows pointing at slot 0 would grow unbounded with
            # request 0's context and hang the draft block-decode kernel.
            # The sentinel row stays zero-init (length 0, dummy page 0).
            sentinel = int(self.config.max_req_pool_size)
            return torch.cat(
                [
                    active_req_pool_indices,
                    active_req_pool_indices.new_full((pad,), sentinel),
                ]
            )
        return torch.cat(
            [active_req_pool_indices, active_req_pool_indices.new_zeros(pad)]
        )

    def _set_graph_state_write_indices(
        self, active_req_pool_indices: torch.Tensor, padded_bs: int
    ) -> None:
        state_indices = self.input_buffers.state_write_req_pool_indices_buf[:padded_bs]
        active_bs = active_req_pool_indices.shape[0]
        if active_bs > 0:
            state_indices[:active_bs].copy_(active_req_pool_indices)
        if active_bs < padded_bs:
            state_indices[active_bs:padded_bs].fill_(int(self.config.max_req_pool_size))

    def __call__(
        self,
        bs: int,
        ctx: ForwardContext,
        sampling_info: SamplingBatchInfo,
        page_table: torch.Tensor,
        extend_with_prefix: bool = False,
        extend_prefix_lens: torch.Tensor | None = None,
        extend_prefix_lens_cpu: torch.Tensor | None = None,
        extend_seq_lens: torch.Tensor | None = None,
        extend_seq_lens_cpu: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        out_cache_loc: torch.Tensor | None = None,
        block_tables: dict | None = None,
        block_table_base_offsets: dict | None = None,
        cache_metadata=None,
        forward_batch=None,
    ):
        """
        Unified forward entry point.

        Every decode prepares its metadata through the same
        ``_prepare_decode_metadata`` refresh; a captured graph is replayed when
        one exists for the (padded) batch size, otherwise the same forward
        code the graph recorded runs eagerly. Extend/mixed batches keep the
        eager ``init_forward_metadata`` construction path. The caller does not
        need to know which path was taken.
        """
        use_graph = self._can_use_graph(bs, ctx)
        padded_bs = self._padded_bs(bs, ctx) if use_graph else bs
        active_req_pool_indices = self.input_buffers.req_pool_indices_buf[:bs]

        if use_graph and padded_bs != bs:
            ctx.bs = padded_bs
            pad = padded_bs - bs
            seq_lens = torch.nn.functional.pad(
                self.input_buffers.seq_lens_buf[:bs], (0, pad), value=1
            )
            req_pool_indices = self._pad_graph_req_pool_indices(
                active_req_pool_indices, padded_bs
            )
            self.input_buffers.seq_lens_buf[:padded_bs].copy_(seq_lens)
            self.input_buffers.req_pool_indices_buf[:padded_bs].copy_(req_pool_indices)
        else:
            seq_lens = self.input_buffers.seq_lens_buf[:padded_bs]
            req_pool_indices = self.input_buffers.req_pool_indices_buf[:padded_bs]

        if use_graph:
            self._set_graph_state_write_indices(active_req_pool_indices, padded_bs)

        # V4 (bespoke multi-group slot mapping) and KDA state paging still
        # build metadata from the cache contract object; the MLA family now
        # consumes only the per-group block_tables like every other backend.
        # Thread it only when present so non-contract paths remain unchanged.
        cache_kwargs = (
            {
                "cache_metadata": cache_metadata,
                "forward_batch": forward_batch,
            }
            if cache_metadata is not None
            else {}
        )

        # Eager stale-table guard, all modes: with >1 published group the
        # single-table fallback would serve first-group pages to every layer.
        # Replay relies on the backends' own _replay_stale_guard instead.
        # Idle/bs==0 forwards carry no requests.
        if (
            not use_graph
            and bs > 0
            and not ctx.forward_mode.is_idle()
            and not block_tables
            and len(self.token_to_kv_pool.arena.cache_group_specs) > 1
        ):
            raise RuntimeError(
                "ForwardStepRunner eager forward: pool publishes "
                f"{len(self.token_to_kv_pool.arena.cache_group_specs)} "
                "cache groups and the backend consumes group tables, "
                f"but block_tables is missing/empty at bs={bs} "
                f"({ctx.forward_mode.name}); the single-table fallback "
                "would use one group's pages for all layers."
            )

        # _can_use_graph already requires a decode mode, so this branches on
        # the mode alone: decode → unified refresh; extend/mixed → construct.
        if ctx.forward_mode.is_decode():
            self._prepare_decode_metadata(
                padded_bs,
                bs,
                req_pool_indices,
                seq_lens,
                page_table=page_table,
                forward_mode=ctx.forward_mode,
                use_graph=use_graph,
                block_table_base_offsets=block_table_base_offsets,
                block_tables=block_tables,
                **cache_kwargs,
            )
        else:
            # Extend/mixed (and the never-in-practice eager idle): dynamic
            # shapes, fresh construction.
            self._init_forward_metadata(
                padded_bs,
                ctx.num_extends,
                req_pool_indices,
                seq_lens,
                page_table=page_table,
                forward_mode=ctx.forward_mode,
                extend_with_prefix=extend_with_prefix,
                extend_prefix_lens=extend_prefix_lens,
                extend_prefix_lens_cpu=extend_prefix_lens_cpu,
                extend_seq_lens=extend_seq_lens,
                extend_seq_lens_cpu=extend_seq_lens_cpu,
                positions=positions,
                out_cache_loc=out_cache_loc,
                global_num_tokens=ctx.global_num_tokens,
                all_decode_or_idle=ctx.all_decode_or_idle,
                capture_hidden_mode=ctx.capture_hidden_mode,
                num_tokens=ctx.input_num_tokens,
                block_tables=block_tables,
                block_table_base_offsets=block_table_base_offsets,
                **cache_kwargs,
            )

        if use_graph:
            # Runtime prepare() is called by ModelExecutor with per-request rids
            # BEFORE self.forward_step — we don't refill here to avoid clobbering
            # the per-request generators with the capture-stub generator.
            self.deepep_adapter.replay()

            graph_key = self._cuda_graph_key(padded_bs)
            graph = self.graphs[graph_key]
            if self._graph_debug:
                self._verify_graph_metadata(graph_key)
            if self.device == "npu":
                graph.update(
                    cpu_update_input=[
                        {"actual_seq_lengths_kv": seq_lens.to("cpu").tolist()}
                    ]
                )
            with nvtx_range("graph_replay", color="red"):
                graph.replay()

            (
                output_tokens,
                output_lengths,
                output_logprobs,
            ) = self.output_buffers[graph_key]

            result = (
                output_tokens[: bs * self.max_tokens_per_req],
                output_lengths[:bs],
                (
                    output_logprobs[: bs * self.max_tokens_per_req]
                    if output_logprobs is not None
                    else None
                ),
            )
        else:
            result = self._forward_func(bs=bs, ctx=ctx, sampling_info=sampling_info)

        if use_graph and padded_bs != bs:
            ctx.bs = bs

        # Update mamba/GDN state after speculative verify
        if _should_update_mamba_state_after_mtp_verify(
            self.drafter, self.attn_backend, ctx.forward_mode
        ):
            accept_lengths = result[1]
            self.attn_backend.update_mamba_state_after_mtp_verify(accept_lengths, None)
        if self.drafter is not None and (
            ctx.forward_mode.is_decode() or ctx.forward_mode.is_mixed()
        ):
            self.attn_backend.commit_speculative_state_after_verify(
                result[1],
                num_extends=ctx.num_extends,
            )

        return result
