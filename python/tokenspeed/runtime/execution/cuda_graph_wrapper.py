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
    from tokenspeed.runtime.layers.attention.kv_cache.base import BaseTokenToKVPool
    from tokenspeed.runtime.sampling.backends.base import SamplingBackend

logger = get_colorful_logger(__name__)

_is_capture_mode = False


def get_is_capture_mode() -> bool:
    return _is_capture_mode


def compute_max_logical_pages_for_capture(
    spec,
    *,
    max_context_len: int,
    max_tokens_per_req: int = 1,
) -> int:
    raw_per_page = max(1, int(spec.rows_per_page) * int(spec.entry_stride_tokens))
    if str(getattr(spec, "retention", "")) == "sliding_window":
        window = int(getattr(spec, "sliding_window_tokens", 0) or 0)
        live_tokens = max(1, window - 1 + max(1, int(max_tokens_per_req)))
        if int(max_context_len) > 0:
            live_tokens = min(live_tokens, int(max_context_len))
        return max(1, (live_tokens + raw_per_page - 1) // raw_per_page + 1)
    return max(1, (max(1, int(max_context_len)) + raw_per_page - 1) // raw_per_page)


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
    capture_bs = [bs for bs in capture_bs if bs <= effective_max]
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
            from tokenspeed.runtime.layers.moe.dispatcher.deep_ep import (
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


class CudaGraphWrapper:
    """
    Wraps a forward_func and transparently dispatches to either a captured
    CUDA graph (decode, supported batch size) or the eager path (prefill /
    unsupported batch size).

    Callers always use the same interface::

        output_tokens, output_lengths, output_logprobs = runner(
            bs, ctx, sampling_info, req_to_page,
            extend_with_prefix=..., extend_prefix_lens=...,
        )

    Internally the wrapper owns both paths and calls init_forward_metadata
    with use_cuda_graph=True/False to select the appropriate backend buffers.
    """

    def __init__(
        self,
        forward_func: Callable,
        attn_backend: AttentionBackend,
        token_to_kv_pool: BaseTokenToKVPool,
        input_buffers: InputBuffers,
        config: ModelExecutorConfig,
        draft_attn_backend: AttentionBackend | None = None,
        draft_token_to_kv_pool: BaseTokenToKVPool | None = None,
        drafter: BaseDrafter | None = None,
        capturable_grammar=None,
        eager_grammar_buffers=None,
        sampling_backend: SamplingBackend | None = None,
        runtime_states: RuntimeStates | None = None,
    ):
        self.config = config
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
        self.enable_torch_compile = getattr(config, "enable_torch_compile", False)
        self.disable_padding = config.disable_cuda_graph_padding
        self.enable_cudagraph_gc = getattr(config, "enable_cudagraph_gc", True)
        self.device = config.device
        self.gpu_id = config.gpu_id
        self.global_rank = config.global_rank
        self.context_len = config.context_len
        self.vocab_size = config.vocab_size
        self.grammar_backend = config.grammar_backend
        self.capture_bs = get_batch_sizes_to_capture(config)
        self.max_bs = max(self.capture_bs)
        self.max_tokens_per_req = (
            config.spec_num_tokens if config.spec_algo is not None else 1
        )
        self.dp_size = config.data_parallel_size
        self.world_size = config.world_size
        # Backends alias their cache_seqlens buffer. Draft backend aliases
        # the drafter-owned draft_seq_lens to keep InputBuffers read-only.
        paged_cache_group_specs = tuple(
            getattr(token_to_kv_pool, "paged_cache_group_specs", ()) or ()
        )
        try:
            attn_backend.init_cuda_graph_state(
                self.max_bs,
                self.input_buffers.seq_lens_buf,
                paged_cache_group_specs=paged_cache_group_specs,
                max_tokens_per_req=self.max_tokens_per_req,
            )
        except TypeError:
            attn_backend.init_cuda_graph_state(
                self.max_bs,
                self.input_buffers.seq_lens_buf,
            )
        if draft_attn_backend is not None:
            draft_paged_cache_group_specs = tuple(
                getattr(draft_token_to_kv_pool, "paged_cache_group_specs", ()) or ()
            )
            try:
                draft_attn_backend.init_cuda_graph_state(
                    self.max_bs,
                    self.drafter.draft_seq_lens_buf,
                    paged_cache_group_specs=draft_paged_cache_group_specs,
                    max_tokens_per_req=self.max_tokens_per_req,
                )
            except TypeError:
                draft_attn_backend.init_cuda_graph_state(
                    self.max_bs, self.drafter.draft_seq_lens_buf
                )

        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.output_buffers: dict[int, tuple] = {}

        self._forward_func: Callable | None = forward_func
        self.disable = config.enforce_eager
        self.deepep_adapter = DeepEPCudaGraphRunnerAdapter()
        if not self.disable:
            self.capture()

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
            self.stream = torch.cuda.Stream()
            capture_range = tqdm.tqdm(self.capture_bs) if rank == 0 else self.capture_bs
            if rank == 0:
                logger.info("Capturing batches: %s", self.capture_bs)
            for bs in capture_range:
                if rank == 0:
                    avail_mem = get_available_gpu_memory(
                        self.device, self.gpu_id, empty_cache=False
                    )
                    capture_range.set_description(
                        f"Capturing batches ({bs=} {avail_mem=:.2f} GB)"
                    )
                graph, output_buffers = self._capture_one(bs)
                self.graphs[bs] = graph
                self.output_buffers[bs] = output_buffers

    def _capture_one(self, bs: int):
        graph = torch.cuda.CUDAGraph()

        capture_forward_mode = (
            ForwardMode.TARGET_VERIFY
            if self.drafter is not None
            else ForwardMode.DECODE
        )
        ctx = ForwardContext(
            attn_backend=self.attn_backend,
            token_to_kv_pool=self.token_to_kv_pool,
            bs=bs,
            num_extends=0,
            input_num_tokens=bs * self.max_tokens_per_req,
            forward_mode=capture_forward_mode,
            capture_hidden_mode=(
                CaptureHiddenMode.FULL
                if self.drafter is not None
                else CaptureHiddenMode.NULL
            ),
            keep_full_logits=True,
        )

        # For DP mode, global_num_tokens must be set so that the MoE
        # all-gather comm layers know token counts for all DP ranks.
        # During capture, use uniform dummy counts across ranks.
        if self.dp_size > 1:
            ctx.global_num_tokens = [bs * self.max_tokens_per_req] * self.world_size

        # Capture with is_all_greedy=False so the graph records the full
        # top_k_top_p_sampling path (greedy-only requests are served by the
        # same path with top_k=1 in the buffer, which effectively argmaxes).
        # is_all_greedy=True at capture would freeze the graph into
        # argmax and bypass per-request seeding at replay.
        ibd = self.input_buffers
        sampling_info = SamplingBatchInfo(
            req_pool_indices=ibd.req_pool_indices_buf[:bs],
            valid_cache_lengths=(
                self.runtime_states.valid_cache_lengths
                if self.runtime_states is not None
                else None
            ),
            is_all_greedy=False,
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

        self._init_capture_metadata(bs)

        def run_once():
            # Dummy add_batch keeps the grammar queue 1:1 with replays —
            # fetch_batch pops once per forward, so warmup + capture
            # would otherwise raise queue.Empty.
            if self.capturable_grammar is not None:
                self.capturable_grammar.add_batch(
                    grammars=[None] * bs, bs=bs, has_candidates=False
                )
            return self._forward_func(bs=bs, ctx=ctx, sampling_info=sampling_info)

        # Warm up before capture.
        for _ in range(4):
            torch.cuda.synchronize()
            dist.barrier()
            if self.sampling_backend is not None:
                self.sampling_backend.prepare_capture(
                    bs=bs, num_tokens_per_req=self.max_tokens_per_req
                )
            run_once()

        # Clear any per-pool state that warm-up dirtied at pool row 0,
        # so the graph captures reads against a clean baseline.
        if self.sampling_backend is not None:
            self.sampling_backend.reset_capture_state()

        torch.cuda.synchronize()
        dist.barrier()

        # Fill sampler buffers OUTSIDE the capture so RNG ops aren't recorded.
        if self.sampling_backend is not None:
            self.sampling_backend.prepare_capture(
                bs=bs, num_tokens_per_req=self.max_tokens_per_req
            )

        self.deepep_adapter.capture()

        global _is_capture_mode
        _is_capture_mode = True
        global global_graph_memory_pool
        with torch.cuda.graph(graph, pool=global_graph_memory_pool, stream=self.stream):
            out = run_once()

        torch.cuda.synchronize()
        dist.barrier()
        _is_capture_mode = False

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
        return graph, out

    def _capture_paged_cache_block_tables(self, bs: int, pool) -> dict | None:
        specs = tuple(getattr(pool, "paged_cache_group_specs", ()) or ())
        if not specs:
            return None
        out = {}
        for spec in specs:
            max_pages = compute_max_logical_pages_for_capture(
                spec,
                max_context_len=(
                    self.max_tokens_per_req * self.max_bs
                    if self.context_len <= 0
                    else self.context_len
                ),
                max_tokens_per_req=self.max_tokens_per_req,
            )
            out[str(spec.group_id)] = torch.zeros(
                (bs, max_pages),
                dtype=torch.int32,
                device=self.device,
            )
        return out

    def _init_capture_metadata(self, bs: int):
        capture_kwargs = {}
        if self.input_buffers.has_mamba:
            capture_kwargs["mamba_pool_indices"] = (
                self.input_buffers.mamba_pool_indices_buf[:bs]
            )
        paged_cache_block_tables = self._capture_paged_cache_block_tables(
            bs,
            self.token_to_kv_pool,
        )
        if paged_cache_block_tables is not None and getattr(
            self.attn_backend,
            "uses_paged_cache_groups",
            False,
        ):
            capture_kwargs["paged_cache_block_tables"] = paged_cache_block_tables
        self.attn_backend.init_forward_metadata_capture_cuda_graph(
            bs,
            bs * self.max_tokens_per_req,
            self.input_buffers.req_pool_indices_buf[:bs],
            self.input_buffers.seq_lens_buf[:bs],
            ForwardMode.DECODE if self.drafter is None else ForwardMode.TARGET_VERIFY,
            **capture_kwargs,
        )
        if self.draft_attn_backend is not None:
            draft_kwargs = {}
            if self.draft_token_to_kv_pool is not None:
                draft_paged_cache_block_tables = self._capture_paged_cache_block_tables(
                    bs,
                    self.draft_token_to_kv_pool,
                )
                if draft_paged_cache_block_tables is not None and getattr(
                    self.draft_attn_backend,
                    "uses_paged_cache_groups",
                    False,
                ):
                    draft_kwargs["paged_cache_block_tables"] = (
                        draft_paged_cache_block_tables
                    )
            # Drafter mutates seq_lens_buf in place per step; backends alias.
            self.draft_attn_backend.init_forward_metadata_capture_cuda_graph(
                bs,
                bs * self.max_tokens_per_req,
                self.input_buffers.req_pool_indices_buf[:bs],
                self.input_buffers.seq_lens_buf[:bs],
                ForwardMode.DRAFT_EXTEND,
                **draft_kwargs,
            )

    @staticmethod
    def _pad_block_tables_to_padded_bs(
        block_tables: dict,
        *,
        actual_bs: int,
        padded_bs: int,
    ) -> dict:
        if padded_bs <= actual_bs:
            return block_tables
        out = {}
        for key, table in block_tables.items():
            if not isinstance(table, torch.Tensor):
                out[key] = table
                continue
            rows = int(table.shape[0])
            if rows == padded_bs:
                out[key] = table
                continue
            out[key] = torch.nn.functional.pad(
                table,
                (0, 0, 0, padded_bs - rows),
                value=0,
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
            # Padded rows have no real request — base 0 keeps absolute
            # indexing aligned to column 0, matching dummy-page row padding.
            out[key] = torch.nn.functional.pad(
                off,
                (0, padded_bs - rows),
                value=0,
            )
        return out

    def _init_replay_metadata(
        self,
        padded_bs: int,
        actual_bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        req_to_page: torch.Tensor,
        forward_mode: ForwardMode,
        **kwargs,
    ):
        """Graph-replay path — update persistent cuda-graph buffers in place."""
        paged_cache_block_tables = kwargs.pop("paged_cache_block_tables", None)
        paged_cache_block_table_base_offsets = kwargs.pop(
            "paged_cache_block_table_base_offsets", None
        )
        if paged_cache_block_tables is not None and getattr(
            self.attn_backend,
            "uses_paged_cache_groups",
            False,
        ):
            table_bs = next(
                (
                    int(table.shape[0])
                    for table in paged_cache_block_tables.values()
                    if isinstance(table, torch.Tensor)
                ),
                int(req_pool_indices.shape[0]),
            )
            paged_cache_block_tables = self._pad_block_tables_to_padded_bs(
                paged_cache_block_tables,
                actual_bs=table_bs,
                padded_bs=padded_bs,
            )
            kwargs["paged_cache_block_tables"] = paged_cache_block_tables
            if paged_cache_block_table_base_offsets:
                paged_cache_block_table_base_offsets = self._pad_offsets_to_padded_bs(
                    paged_cache_block_table_base_offsets,
                    actual_bs=actual_bs,
                    padded_bs=padded_bs,
                )
                kwargs["paged_cache_block_table_base_offsets"] = (
                    paged_cache_block_table_base_offsets
                )
        if getattr(self.attn_backend, "uses_padded_decode_token_mask", False):
            kwargs["actual_bs"] = actual_bs
        self.attn_backend.init_forward_metadata_replay_cuda_graph(
            padded_bs,
            req_pool_indices,
            seq_lens,
            req_to_page=req_to_page,
            forward_mode=forward_mode,
            **kwargs,
        )
        if self.draft_attn_backend is not None:
            # DRAFT_EXTEND covers step 0 + N-1 decode steps (drafter syncs per step).
            self.draft_attn_backend.init_forward_metadata_replay_cuda_graph(
                padded_bs,
                req_pool_indices,
                seq_lens,
                req_to_page=self.drafter.req_to_page,
                forward_mode=ForwardMode.DRAFT_EXTEND,
                **kwargs,
            )

    @nvtx_range("attn_meta_prep", color="orange")
    def _init_forward_metadata(
        self,
        padded_bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        req_to_page: torch.Tensor,
        forward_mode: ForwardMode,
        **kwargs,
    ):
        """Eager path — allocate/refresh metadata for the upcoming forward."""
        self.attn_backend.init_forward_metadata(
            padded_bs,
            padded_bs * self.max_tokens_per_req,
            req_pool_indices,
            seq_lens,
            req_to_page=req_to_page,
            forward_mode=forward_mode,
            **kwargs,
        )
        if self.draft_attn_backend is not None:
            if forward_mode.is_extend():
                # Initial prefill: draft step 0 uses EXTEND (regular prefill)
                # kernel with the caller's prefix kwargs. Step 0 and the
                # subsequent decode steps have structurally different
                # metadata, so populate each slot with its own init call.
                #
                # Note: drops the pre-refactor subtraction of
                # `extend_prefix_lens` from `seq_lens` for the draft init.
                self.draft_attn_backend.init_forward_metadata(
                    padded_bs,
                    padded_bs * self.max_tokens_per_req,
                    req_pool_indices,
                    seq_lens,
                    req_to_page=self.drafter.req_to_page,
                    forward_mode=forward_mode,
                    **kwargs,
                )
                self.draft_attn_backend.init_forward_metadata(
                    padded_bs,
                    padded_bs,
                    req_pool_indices,
                    seq_lens,
                    req_to_page=self.drafter.req_to_page,
                    forward_mode=ForwardMode.DECODE,
                )
            else:
                # TARGET_VERIFY / DECODE caller: DRAFT_EXTEND is the "prepare
                # full drafter pass" signal. The backend populates whichever
                # slots it needs for step 0 + N-1 decode steps — a single
                # slot (trtllm_mla) or both prefill and decode slots
                # (trtllm_mha compound init).
                self.draft_attn_backend.init_forward_metadata(
                    padded_bs,
                    padded_bs * self.max_tokens_per_req,
                    req_pool_indices,
                    seq_lens,
                    req_to_page=self.drafter.req_to_page,
                    forward_mode=ForwardMode.DRAFT_EXTEND,
                )

    def _global_graph_bs(self, ctx: ForwardContext) -> int | None:
        if self.dp_size <= 1 or ctx.global_num_tokens is None:
            return None
        max_num_tokens = max(ctx.global_num_tokens)
        return (max_num_tokens + self.max_tokens_per_req - 1) // self.max_tokens_per_req

    def _can_use_graph(self, bs: int, ctx: ForwardContext) -> bool:
        if self.disable:
            return False
        if not (ctx.forward_mode.is_decode() or ctx.forward_mode.is_target_verify()):
            return False
        if self.dp_size > 1:
            if not ctx.all_decode_or_idle:
                return False
            global_bs = self._global_graph_bs(ctx)
            if global_bs is None or global_bs == 0:
                return False
            if self.disable_padding:
                return global_bs in self.graphs
            return global_bs <= self.max_bs
        if self.disable_padding:
            return bs in self.graphs
        return bs <= self.max_bs

    def can_run(self, bs: int, ctx: ForwardContext) -> bool:
        return self._can_use_graph(bs, ctx)

    def padded_bs(self, bs: int, ctx: ForwardContext) -> int:
        return self._padded_bs(bs, ctx)

    def _padded_bs(self, bs: int, ctx: ForwardContext) -> int:
        graph_bs = self._global_graph_bs(ctx)
        target_bs = graph_bs if graph_bs is not None else bs
        index = bisect.bisect_left(self.capture_bs, target_bs)
        return self.capture_bs[index]

    def __call__(
        self,
        bs: int,
        ctx: ForwardContext,
        sampling_info: SamplingBatchInfo,
        req_to_page: torch.Tensor,
        extend_with_prefix: bool = False,
        extend_prefix_lens: torch.Tensor | None = None,
        extend_prefix_lens_cpu: torch.Tensor | None = None,
        extend_seq_lens: torch.Tensor | None = None,
        extend_seq_lens_cpu: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        out_cache_loc: torch.Tensor | None = None,
        mamba_pool_indices: torch.Tensor | None = None,
        mamba_cow_src_indices: torch.Tensor | None = None,
        mamba_branching_seqlens: torch.Tensor | None = None,
        mamba_track_pool_indices: torch.Tensor | None = None,
        spec_info=None,
        paged_cache_block_tables: dict | None = None,
        paged_cache_block_table_base_offsets: dict | None = None,
    ):
        """
        Unified forward entry point.

        Dispatches to the captured CUDA graph when possible; falls back to the
        eager forward_func otherwise.  The caller does not need to know which
        path was taken.
        """
        use_graph = self._can_use_graph(bs, ctx)
        padded_bs = self._padded_bs(bs, ctx) if use_graph else bs

        if use_graph and padded_bs != bs:
            ctx.bs = padded_bs
            pad = padded_bs - bs
            seq_lens = torch.nn.functional.pad(
                self.input_buffers.seq_lens_buf[:bs], (0, pad), value=1
            )
            req_pool_indices = torch.nn.functional.pad(
                self.input_buffers.req_pool_indices_buf[:bs], (0, pad), value=0
            )
            if mamba_pool_indices is not None:
                mamba_pool_indices = torch.nn.functional.pad(
                    mamba_pool_indices, (0, pad), value=0
                )
            if mamba_cow_src_indices is not None:
                mamba_cow_src_indices = torch.nn.functional.pad(
                    mamba_cow_src_indices, (0, pad), value=-1
                )
            if mamba_branching_seqlens is not None:
                mamba_branching_seqlens = torch.nn.functional.pad(
                    mamba_branching_seqlens, (0, pad), value=-1
                )
            if mamba_track_pool_indices is not None:
                mamba_track_pool_indices = torch.nn.functional.pad(
                    mamba_track_pool_indices, (0, pad), value=-1
                )
        else:
            seq_lens = self.input_buffers.seq_lens_buf[:padded_bs]
            req_pool_indices = self.input_buffers.req_pool_indices_buf[:padded_bs]

        mamba_kwargs = {}
        if mamba_pool_indices is not None:
            mamba_kwargs["mamba_pool_indices"] = mamba_pool_indices
        if mamba_cow_src_indices is not None:
            mamba_kwargs["mamba_cow_src_indices"] = mamba_cow_src_indices
        if mamba_branching_seqlens is not None:
            mamba_kwargs["mamba_branching_seqlens"] = mamba_branching_seqlens
        if mamba_track_pool_indices is not None:
            mamba_kwargs["mamba_track_pool_indices"] = mamba_track_pool_indices
        if getattr(self.config, "enable_mamba", False):
            mamba_kwargs["mamba_cache_chunk_size"] = self.config.mamba_cache_chunk_size

        if use_graph:
            if (
                bs == 0
                and paged_cache_block_tables is None
                and getattr(self.attn_backend, "uses_paged_cache_groups", False)
            ):
                paged_cache_block_tables = self._capture_paged_cache_block_tables(
                    padded_bs,
                    self.token_to_kv_pool,
                )
            self._init_replay_metadata(
                padded_bs,
                bs,
                req_pool_indices,
                seq_lens,
                req_to_page=req_to_page,
                forward_mode=ctx.forward_mode,
                num_padding=padded_bs - bs if padded_bs != bs else 0,
                paged_cache_block_tables=paged_cache_block_tables,
                paged_cache_block_table_base_offsets=(
                    paged_cache_block_table_base_offsets
                ),
                **mamba_kwargs,
            )

            # Runtime prepare() is called by ModelExecutor with per-request rids
            # BEFORE self.forward_step — we don't refill here to avoid clobbering
            # the per-request generators with the capture-stub generator.
            self.deepep_adapter.replay()

            with nvtx_range("graph_replay", color="red"):
                self.graphs[padded_bs].replay()

            output_tokens, output_lengths, output_logprobs = self.output_buffers[
                padded_bs
            ]

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
            self._init_forward_metadata(
                padded_bs,
                req_pool_indices,
                seq_lens,
                req_to_page=req_to_page,
                forward_mode=ctx.forward_mode,
                extend_with_prefix=extend_with_prefix,
                extend_prefix_lens=extend_prefix_lens,
                extend_prefix_lens_cpu=extend_prefix_lens_cpu,
                extend_seq_lens=extend_seq_lens,
                extend_seq_lens_cpu=extend_seq_lens_cpu,
                num_extends=ctx.num_extends,
                positions=positions,
                out_cache_loc=out_cache_loc,
                global_num_tokens=ctx.global_num_tokens,
                all_decode_or_idle=ctx.all_decode_or_idle,
                capture_hidden_mode=ctx.capture_hidden_mode,
                padded_static_len=ctx.padded_static_len,
                spec_info=spec_info,
                paged_cache_block_tables=(
                    paged_cache_block_tables
                    if getattr(self.attn_backend, "uses_paged_cache_groups", False)
                    else None
                ),
                paged_cache_block_table_base_offsets=(
                    paged_cache_block_table_base_offsets
                    if getattr(self.attn_backend, "uses_paged_cache_groups", False)
                    else None
                ),
                **mamba_kwargs,
            )

            result = self._forward_func(bs=bs, ctx=ctx, sampling_info=sampling_info)

        # Update mamba/GDN state after speculative verify
        if (
            self.drafter is not None
            and ctx.forward_mode.is_target_verify()
            and hasattr(self.attn_backend, "update_mamba_state_after_mtp_verify")
        ):
            accept_lengths = result[1]
            self.attn_backend.update_mamba_state_after_mtp_verify(accept_lengths, None)

        return result
