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

from tokenspeed.runtime.configs.flat_memory_plan import (
    validate_flat_graph_owner_allocation,
)
from tokenspeed.runtime.configs.paged_cache_spec import (
    compute_max_logical_pages_for_capture,
)
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
)
from tokenspeed.runtime.flat_cache_tables import (
    CacheTableBinding,
    CacheTableSource,
    FlatCacheTableOwnerView,
    require_flat_cache_generation,
    resolve_cache_table_source,
)
from tokenspeed.runtime.layers.attention.backends.base import (
    init_backend_cuda_graph_state,
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
    from tokenspeed.runtime.layers.attention.kv_cache.base import BaseTokenToKVPool
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
        target_cache_table_binding: CacheTableBinding | None = None,
        draft_cache_table_binding: CacheTableBinding | None = None,
        flat_generation_pool=None,
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
        self.overlap_schedule_depth = config.overlap_schedule_depth
        self.dp_size = config.data_parallel_size
        self.world_size = config.world_size
        if target_cache_table_binding is None:
            raise ValueError("target cache table binding must be resolved at startup")
        if (draft_attn_backend is None) != (draft_cache_table_binding is None):
            raise ValueError(
                "draft backend and cache table binding must be provided together"
            )
        self._target_uses_grouped_metadata = bool(attn_backend.uses_paged_cache_groups)
        self._draft_uses_grouped_metadata = bool(
            draft_attn_backend is not None
            and draft_attn_backend.uses_paged_cache_groups
        )
        self._target_uses_flat_source = target_cache_table_binding.kind == "flat"
        self._draft_uses_flat_source = bool(
            draft_cache_table_binding is not None
            and draft_cache_table_binding.kind == "flat"
        )
        self._flat_generation_pool = flat_generation_pool
        if self._target_uses_flat_source and self._draft_uses_flat_source:
            target_arena = getattr(token_to_kv_pool, "device_cache_arena", None)
            draft_arena = getattr(draft_token_to_kv_pool, "device_cache_arena", None)
            if (target_arena is None) != (draft_arena is None) or (
                target_arena is not None and target_arena is not draft_arena
            ):
                raise RuntimeError(
                    "target/draft flat cache pools must share one device arena"
                )
        self._target_uses_paged_groups = bool(
            self._target_uses_grouped_metadata and not self._target_uses_flat_source
        )
        self._draft_uses_paged_groups = bool(
            self._draft_uses_grouped_metadata and not self._draft_uses_flat_source
        )
        self._target_flat_inputs = FlatCacheTableOwnerView(
            target_cache_table_binding.group_ids
        )
        self._draft_flat_inputs = FlatCacheTableOwnerView(
            draft_cache_table_binding.group_ids
            if draft_cache_table_binding is not None
            else ()
        )
        self._target_flat_group_ids = self._target_flat_inputs.group_ids
        self._draft_flat_group_ids = self._draft_flat_inputs.group_ids
        self._target_accepts_bound_source = bool(
            getattr(attn_backend, "accepts_bound_cache_table_source", False)
        )
        self._draft_accepts_bound_source = bool(
            draft_attn_backend is not None
            and getattr(draft_attn_backend, "accepts_bound_cache_table_source", False)
        )
        self._flat_graph_consumers_self_pad = bool(
            getattr(attn_backend, "flat_tables_self_padding", False)
            and (
                not self._draft_uses_flat_source
                or getattr(draft_attn_backend, "flat_tables_self_padding", False)
            )
        )
        self._draft_metadata_sequence = getattr(
            draft_attn_backend,
            "init_speculative_draft_metadata",
            None,
        )
        if self._draft_metadata_sequence is not None and not callable(
            self._draft_metadata_sequence
        ):
            raise TypeError(
                "draft backend init_speculative_draft_metadata must be callable"
            )
        target_flat_graph_plan_kwargs = CudaGraphWrapper._flat_graph_plan_init_kwargs(
            token_to_kv_pool,
            owner="target",
            max_bs=self.max_bs,
        )
        # Backends alias their cache_seqlens buffer. Draft backend aliases
        # the drafter-owned draft_seq_lens to keep InputBuffers read-only.
        init_backend_cuda_graph_state(
            attn_backend,
            self.max_bs,
            self.input_buffers.seq_lens_buf,
            paged_cache_group_specs=tuple(token_to_kv_pool.paged_cache_group_specs),
            max_tokens_per_req=self.max_tokens_per_req,
            overlap_schedule_depth=self.overlap_schedule_depth,
            cache_table_source_kind=target_cache_table_binding.kind,
            **target_flat_graph_plan_kwargs,
        )
        if draft_attn_backend is not None:
            draft_flat_graph_plan_kwargs = (
                CudaGraphWrapper._flat_graph_plan_init_kwargs(
                    draft_token_to_kv_pool,
                    owner="draft",
                    max_bs=self.max_bs,
                )
            )
            init_backend_cuda_graph_state(
                draft_attn_backend,
                self.max_bs,
                self.drafter.draft_seq_lens_buf,
                paged_cache_group_specs=tuple(
                    draft_token_to_kv_pool.paged_cache_group_specs
                ),
                max_tokens_per_req=self.max_tokens_per_req,
                overlap_schedule_depth=self.overlap_schedule_depth,
                cache_table_source_kind=draft_cache_table_binding.kind,
                **draft_flat_graph_plan_kwargs,
            )
            # Legacy drafter backends are constructed with the target's
            # req_to_page, and the replay path hands both backends the same
            # req_pool_indices. Their block-table gather is
            # req_to_page[req_pool_indices] (see
            # _create_block_kv_indices; it does not depend on seq_lens), so both
            # backends would compute identical block_kv_indices. When the backing
            # buffer shapes/dtypes also line up, point the draft backend at the
            # target's buffer and skip its gather+copy in the replay path: the
            # target's metadata prep runs first and populates the shared buffer
            # (see init_forward_metadata_replay_cuda_graph). Group-keyed flat
            # caches are deliberately excluded: target/draft consume distinct owner-local
            # subsets of the scheduler union and have no canonical req_to_page.
            target_kv = getattr(attn_backend, "decode_cuda_graph_kv_indices", None)
            draft_kv = getattr(draft_attn_backend, "decode_cuda_graph_kv_indices", None)
            target_group_keyed = target_cache_table_binding.group_keyed_cache_locs
            draft_group_keyed = draft_cache_table_binding.group_keyed_cache_locs
            if (
                not target_group_keyed
                and not draft_group_keyed
                and target_kv is not None
                and draft_kv is not None
                and target_kv.shape == draft_kv.shape
                and target_kv.dtype == draft_kv.dtype
            ):
                draft_attn_backend.decode_cuda_graph_kv_indices = target_kv
                draft_attn_backend._block_table_aliased = True

        CudaGraphWrapper._validate_flat_graph_metadata_accounting(self)

        self.graph_variants = (
            sampling_backend.cuda_graph_capture_variants(self.max_tokens_per_req)
            if sampling_backend is not None
            else (CUDA_GRAPH_VARIANT_DEFAULT,)
        )
        self.graphs: dict[tuple[str, int], torch.cuda.CUDAGraph] = {}
        self.output_buffers: dict[tuple[str, int], tuple] = {}
        # Cold acceptance evidence.  These booleans flip only on the first
        # successful replay of each route, so the steady-state hot path pays a
        # predictable boolean branch rather than a counter/hash update.
        self._decode_graph_replayed = False
        self._idle_graph_replayed = False

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
            # Capture backend-declared sampler variants explicitly.
            capture_items = [
                (variant, bs)
                for variant in self._cuda_graph_capture_variants()
                for bs in self.capture_bs
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

    def _capture_one(self, bs: int, variant: str = CUDA_GRAPH_VARIANT_DEFAULT):
        graph = torch.cuda.CUDAGraph()

        capture_forward_mode = ForwardMode.DECODE
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

        # Warm up before capture.
        for _ in range(4):
            torch.cuda.synchronize()
            dist.barrier()
            self._prepare_sampling_capture(bs=bs, variant=variant)
            # Keep warmup seq_lens >= q_len_per_req so no query row gets an
            # empty causal span; a stale seq_len of 1 overflows to non-finite KV.
            self.input_buffers.seq_lens_buf[:bs].fill_(self.max_tokens_per_req)
            self._init_capture_metadata(bs)
            run_once()

        # Clear any per-pool state that warm-up dirtied at pool row 0,
        # so the graph captures reads against a clean baseline.
        if self.sampling_backend is not None:
            self.sampling_backend.reset_capture_state()

        torch.cuda.synchronize()
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
        with torch.cuda.graph(graph, pool=global_graph_memory_pool, stream=self.stream):
            out = run_once()

        torch.cuda.synchronize()
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

        return graph, out

    def _capture_paged_cache_block_tables(self, bs: int, pool) -> dict | None:
        specs = tuple(pool.paged_cache_group_specs)
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
                overlap_schedule_depth=self.overlap_schedule_depth,
            )
            out[str(spec.group_id)] = torch.zeros(
                (bs, max_pages),
                dtype=torch.int32,
                device=self.device,
            )
        return out

    @staticmethod
    def _flat_graph_plan_init_kwargs(pool, *, owner: str, max_bs: int) -> dict:
        """Return the graph-table shape contract validated by the owning pool."""
        if pool is None:
            return {}
        plan = getattr(pool, "flat_memory_plan", None)
        if plan is None:
            return {}
        pool_owner = getattr(pool, "cache_owner", None)
        if pool_owner != owner:
            raise RuntimeError(
                "flat CUDA graph pool owner mismatch: "
                f"expected={owner!r}, actual={pool_owner!r}"
            )
        runtime_metadata = plan.runtime_metadata
        canonical_cols = runtime_metadata.graph_capture_cols_by_group(owner)
        planned_rows = int(runtime_metadata.graph_batch_rows)
        # Eager-only plans deliberately budget zero graph rows even though the
        # wrapper still initializes generic backend state with a nominal max_bs.
        if planned_rows not in (0, int(max_bs)):
            raise RuntimeError(
                f"flat {owner} CUDA graph rows disagree with executor config: "
                f"wrapper={max_bs}, plan={planned_rows}"
            )
        return {
            "flat_capture_cols_by_group": canonical_cols,
            "flat_graph_batch_rows": planned_rows,
        }

    @staticmethod
    def _actual_flat_graph_metadata_bytes(backend, pool, *, owner: str) -> int:
        """Validate and total the backend's actual persistent table/base tensors."""
        plan = getattr(pool, "flat_memory_plan", None)
        if plan is None:
            raise RuntimeError(f"flat {owner} backend is missing its memory plan")
        runtime_metadata = plan.runtime_metadata
        canonical_cols = runtime_metadata.graph_capture_cols_by_group(owner)
        rows = int(runtime_metadata.graph_batch_rows)
        tables = getattr(backend, "_cuda_graph_paged_cache_block_tables", None)
        bases = getattr(backend, "_cuda_graph_paged_cache_base_offsets", None)
        if not isinstance(tables, dict) or not isinstance(bases, dict):
            raise RuntimeError(
                f"flat {owner} backend did not publish graph table/base tensors"
            )
        return validate_flat_graph_owner_allocation(
            owner=owner,
            capture_cols_by_group=canonical_cols,
            batch_rows=rows,
            table_shapes={gid: tensor.shape for gid, tensor in tables.items()},
            base_shapes={gid: tensor.shape for gid, tensor in bases.items()},
            table_nbytes={
                gid: int(tensor.numel()) * int(tensor.element_size())
                for gid, tensor in tables.items()
            },
            base_nbytes={
                gid: int(tensor.numel()) * int(tensor.element_size())
                for gid, tensor in bases.items()
            },
        )

    def _validate_flat_graph_metadata_accounting(self) -> None:
        """Cross-check target+draft actual tensors against one shared plan."""
        target_pool = self.token_to_kv_pool
        target_plan = getattr(self.token_to_kv_pool, "flat_memory_plan", None)
        draft_pool = getattr(self, "draft_token_to_kv_pool", None)
        draft_plan = getattr(draft_pool, "flat_memory_plan", None)
        if target_plan is None:
            if draft_plan is not None:
                raise RuntimeError("draft flat graph plan exists without a target plan")
            return
        runtime_metadata = target_plan.runtime_metadata

        target_actual_bytes = CudaGraphWrapper._actual_flat_graph_metadata_bytes(
            self.attn_backend,
            target_pool,
            owner="target",
        )
        draft_cols = runtime_metadata.graph_capture_cols_by_group("draft")
        if draft_cols:
            if self.draft_attn_backend is None or draft_plan is None:
                raise RuntimeError(
                    "flat plan budgets draft graph metadata but the "
                    "draft backend/pool is missing"
                )
            draft_actual_bytes = CudaGraphWrapper._actual_flat_graph_metadata_bytes(
                self.draft_attn_backend,
                draft_pool,
                owner="draft",
            )
        else:
            if draft_plan is not None:
                raise RuntimeError("draft flat pool exists without draft graph groups")
            draft_actual_bytes = 0
        runtime_metadata.validate_graph_metadata_allocation(
            target_actual_bytes=target_actual_bytes,
            draft_actual_bytes=draft_actual_bytes,
        )

    @staticmethod
    def _set_flat_source_kwargs(
        kwargs: dict,
        source: CacheTableSource | None,
        *,
        accepts_bound_source: bool,
    ) -> None:
        """Pass one already-bound flat source through the backend's ABI."""

        if source is None:
            return
        if accepts_bound_source:
            kwargs["cache_table_source"] = source
            return
        kwargs["flat_block_tables"] = source.tables
        kwargs["flat_block_table_base_offsets"] = source.base_offsets

    @staticmethod
    def _flat_pool_generation(pool) -> int | None:
        """Return the generation fence for a pool with an explicit flat plan."""

        if getattr(pool, "flat_memory_plan", None) is None:
            return None
        return require_flat_cache_generation(
            getattr(pool, "arena_generation", None),
            where="flat cache plan",
        )

    @staticmethod
    def _resolve_flat_source_input(
        source: CacheTableSource | None,
        tables: dict | None,
        base_offsets: dict | None,
        *,
        where: str,
    ) -> CacheTableSource | None:
        """Normalize the typed ABI, retaining map pairs only as a legacy seam."""

        if source is not None:
            if tables is not None or base_offsets is not None:
                raise RuntimeError(
                    f"{where}: typed flat source cannot be combined with table/base maps"
                )
            if not isinstance(source, CacheTableSource) or source.kind != "flat":
                raise TypeError(f"{where}: expected a flat CacheTableSource")
            return source
        if tables is None and base_offsets is None:
            return None
        return resolve_cache_table_source(
            paged_tables=None,
            paged_base_offsets=None,
            flat_tables=tables,
            flat_base_offsets=base_offsets,
        )

    def _init_capture_metadata(self, bs: int):
        capture_kwargs = {}
        if self.input_buffers.has_mamba:
            capture_kwargs["mamba_pool_indices"] = (
                self.input_buffers.mamba_pool_indices_buf[:bs]
            )
        if self._target_uses_paged_groups:
            paged_cache_block_tables = self._capture_paged_cache_block_tables(
                bs,
                self.token_to_kv_pool,
            )
            if paged_cache_block_tables is not None:
                capture_kwargs["paged_cache_block_tables"] = paged_cache_block_tables
        # Packed speculative decode is a token-shape contract, independent of
        # whether grouped cache metadata comes from radix or flat tables.
        if self._target_uses_grouped_metadata and self.drafter is not None:
            capture_kwargs["num_tokens"] = bs * self.max_tokens_per_req
        flat_cache_group_ids = self._target_flat_group_ids
        if flat_cache_group_ids:
            capture_kwargs["flat_cache_group_ids"] = flat_cache_group_ids
        self.attn_backend.init_forward_metadata_capture_cuda_graph(
            bs,
            self.input_buffers.req_pool_indices_buf[:bs],
            self.input_buffers.seq_lens_buf[:bs],
            ForwardMode.DECODE,
            **capture_kwargs,
        )
        if self.draft_attn_backend is not None:
            draft_kwargs = {}
            if self._draft_uses_paged_groups:
                draft_paged_cache_block_tables = self._capture_paged_cache_block_tables(
                    bs,
                    self.draft_token_to_kv_pool,
                )
                if draft_paged_cache_block_tables is not None:
                    draft_kwargs["paged_cache_block_tables"] = (
                        draft_paged_cache_block_tables
                    )
            if self._draft_uses_grouped_metadata:
                draft_kwargs["num_tokens"] = bs * self.max_tokens_per_req
            draft_flat_ids = self._draft_flat_group_ids
            if draft_flat_ids:
                draft_kwargs["flat_cache_group_ids"] = draft_flat_ids
            # Drafter mutates seq_lens_buf in place per step; backends alias.
            self.draft_attn_backend.init_forward_metadata_capture_cuda_graph(
                bs,
                self.input_buffers.req_pool_indices_buf[:bs],
                self.input_buffers.seq_lens_buf[:bs],
                ForwardMode.DECODE,
                **draft_kwargs,
            )

    def _idle_flat_block_tables(self, padded_bs: int) -> dict | None:
        """Minimal per-group tables for the bs==0 idle replay: all rows are
        dummy rows, so one column of page-0 entries per group is valid.
        None when the pool publishes no groups."""
        specs = tuple(
            getattr(
                self.token_to_kv_pool,
                "scheduler_group_specs",
                self.token_to_kv_pool.paged_cache_group_specs,
            )
        )
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
        """Pad each table with dummy ROWS up to padded_bs. Flat passes
        pad_value=0, radix/grouped backends keep -1 — see the padding contract at the MHA
        backend's replay guard (backends/mha.py).
        """
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
                value=pad_value,
            )
        return out

    def _idle_flat_block_table_base_offsets(
        self, padded_bs: int, flat_block_tables: dict
    ) -> dict:
        """Pair idle dummy table rows with explicit logical base zero."""
        return {
            gid: torch.zeros(
                (padded_bs,),
                dtype=torch.int32,
                device=self.device,
            )
            for gid in flat_block_tables
        }

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
        flat_cache_table_source = kwargs.pop("flat_cache_table_source", None)
        if paged_cache_block_tables is not None and (
            self._target_uses_paged_groups or self._draft_uses_paged_groups
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
            if paged_cache_block_table_base_offsets is not None:
                paged_cache_block_table_base_offsets = self._pad_offsets_to_padded_bs(
                    paged_cache_block_table_base_offsets,
                    actual_bs=actual_bs,
                    padded_bs=padded_bs,
                )
            if self._target_uses_paged_groups:
                kwargs["paged_cache_block_tables"] = paged_cache_block_tables
                if paged_cache_block_table_base_offsets is not None:
                    kwargs["paged_cache_block_table_base_offsets"] = (
                        paged_cache_block_table_base_offsets
                    )
        scheduler_flat_source = None
        if self._target_uses_flat_source:
            if flat_cache_table_source is None or not flat_cache_table_source.tables:
                raise RuntimeError(
                    "CUDA graph replay requires staged flat table/base maps"
                )
            flat_staging = getattr(self.input_buffers, "flat_block_table_staging", None)
            if self._flat_graph_consumers_self_pad:
                # Preserve one packed upload. Consumers fill graph dummy rows
                # while unpacking into their persistent exact-width buffers.
                scheduler_flat_source = flat_cache_table_source
            else:
                if flat_staging is not None:
                    raise RuntimeError(
                        "planned flat table staging requires graph consumers "
                        "with flat_tables_self_padding"
                    )
                flat_block_tables = flat_cache_table_source.tables
                flat_block_table_base_offsets = flat_cache_table_source.base_offsets
                flat_table_bs = next(
                    (
                        int(table.shape[0])
                        for table in flat_block_tables.values()
                        if isinstance(table, torch.Tensor)
                    ),
                    int(req_pool_indices.shape[0]),
                )
                scheduler_flat_block_tables = self._pad_block_tables_to_padded_bs(
                    flat_block_tables,
                    actual_bs=flat_table_bs,
                    padded_bs=padded_bs,
                    pad_value=0,
                )
                scheduler_flat_block_table_base_offsets = (
                    self._pad_offsets_to_padded_bs(
                        flat_block_table_base_offsets,
                        actual_bs=actual_bs,
                        padded_bs=padded_bs,
                    )
                )
                scheduler_flat_source = CacheTableSource(
                    kind="flat",
                    tables=scheduler_flat_block_tables,
                    base_offsets=scheduler_flat_block_table_base_offsets,
                    generation=flat_cache_table_source.generation,
                )
            target_source = self._target_flat_inputs.bind(
                scheduler_flat_source,
                owner="target",
            )
            self._set_flat_source_kwargs(
                kwargs,
                target_source,
                accepts_bound_source=self._target_accepts_bound_source,
            )
        if self.attn_backend.uses_padded_decode_token_mask:
            kwargs["actual_bs"] = actual_bs
        if (
            self._target_uses_grouped_metadata
            and getattr(self, "drafter", None) is not None
        ):
            kwargs["num_tokens"] = padded_bs * self.max_tokens_per_req
        self.attn_backend.init_forward_metadata_replay_cuda_graph(
            padded_bs,
            req_pool_indices,
            seq_lens,
            req_to_page=req_to_page,
            forward_mode=forward_mode,
            **kwargs,
        )
        if self.draft_attn_backend is not None:
            draft_attn_kwargs = {}
            if self._draft_uses_paged_groups and paged_cache_block_tables is not None:
                draft_attn_kwargs["paged_cache_block_tables"] = paged_cache_block_tables
                if paged_cache_block_table_base_offsets is not None:
                    draft_attn_kwargs["paged_cache_block_table_base_offsets"] = (
                        paged_cache_block_table_base_offsets
                    )
            if getattr(self.draft_attn_backend, "uses_padded_decode_token_mask", False):
                draft_attn_kwargs["actual_bs"] = actual_bs
            if self._draft_uses_flat_source:
                draft_source = self._draft_flat_inputs.bind(
                    scheduler_flat_source,
                    owner="draft",
                )
                self._set_flat_source_kwargs(
                    draft_attn_kwargs,
                    draft_source,
                    accepts_bound_source=self._draft_accepts_bound_source,
                )
            draft_forward_mode = ForwardMode.DECODE
            if self._draft_uses_grouped_metadata:
                draft_attn_kwargs["num_tokens"] = padded_bs * self.max_tokens_per_req
            draft_seq_lens = self.drafter.draft_seq_lens_buf[:padded_bs]
            draft_seq_lens.copy_(seq_lens[:padded_bs])
            self.draft_attn_backend.init_forward_metadata_replay_cuda_graph(
                padded_bs,
                req_pool_indices,
                draft_seq_lens,
                req_to_page=self.drafter.req_to_page,
                forward_mode=draft_forward_mode,
                **draft_attn_kwargs,
            )

    @nvtx_range("attn_meta_prep", color="orange")
    def _init_forward_metadata(
        self,
        padded_bs: int,
        num_extends: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        req_to_page: torch.Tensor,
        forward_mode: ForwardMode,
        **kwargs,
    ):
        """Eager path — allocate/refresh metadata for the upcoming forward."""
        paged_tables = kwargs.pop("paged_cache_block_tables", None)
        paged_bases = kwargs.pop("paged_cache_block_table_base_offsets", None)
        scheduler_flat_source = kwargs.pop("flat_cache_table_source", None)
        if (
            self._target_uses_grouped_metadata
            and self.drafter is not None
            and forward_mode.is_decode()
        ):
            kwargs.setdefault("num_tokens", padded_bs * self.max_tokens_per_req)

        if (
            self._target_uses_flat_source
            and scheduler_flat_source is not None
            and scheduler_flat_source.tables
        ):
            target_source = self._target_flat_inputs.bind(
                scheduler_flat_source,
                owner="target",
            )
            self._set_flat_source_kwargs(
                kwargs,
                target_source,
                accepts_bound_source=self._target_accepts_bound_source,
            )
        elif self._target_uses_paged_groups and paged_tables is not None:
            kwargs["paged_cache_block_tables"] = paged_tables
            if paged_bases is not None:
                kwargs["paged_cache_block_table_base_offsets"] = paged_bases
        self.attn_backend.init_forward_metadata(
            bs=padded_bs,
            num_extends=num_extends,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            req_to_page=req_to_page,
            forward_mode=forward_mode,
            **kwargs,
        )
        if self.draft_attn_backend is not None:
            for key in (
                "cache_table_source",
                "flat_block_tables",
                "flat_block_table_base_offsets",
                "paged_cache_block_tables",
                "paged_cache_block_table_base_offsets",
            ):
                kwargs.pop(key, None)
            draft_kwargs = {}
            draft_source = None
            if (
                self._draft_uses_flat_source
                and scheduler_flat_source is not None
                and scheduler_flat_source.tables
            ):
                draft_source = self._draft_flat_inputs.bind(
                    scheduler_flat_source,
                    owner="draft",
                )
                self._set_flat_source_kwargs(
                    draft_kwargs,
                    draft_source,
                    accepts_bound_source=self._draft_accepts_bound_source,
                )
            elif self._draft_uses_paged_groups and paged_tables is not None:
                draft_kwargs["paged_cache_block_tables"] = paged_tables
                if paged_bases is not None:
                    draft_kwargs["paged_cache_block_table_base_offsets"] = paged_bases

            # The drafter mutates draft_seq_lens_buf between MTP draft steps;
            # decode metadata must alias that buffer.
            draft_seq_lens = self.drafter.draft_seq_lens_buf[:padded_bs]
            draft_seq_lens.copy_(seq_lens[:padded_bs])
            draft_extend_kwargs = kwargs
            if draft_source is not None:
                self._set_flat_source_kwargs(
                    draft_extend_kwargs,
                    draft_source,
                    accepts_bound_source=self._draft_accepts_bound_source,
                )
            elif self._draft_uses_paged_groups and paged_tables is not None:
                draft_extend_kwargs["paged_cache_block_tables"] = paged_tables
                if paged_bases is not None:
                    draft_extend_kwargs["paged_cache_block_table_base_offsets"] = (
                        paged_bases
                    )
            if (
                not forward_mode.is_extend_or_mixed()
                and self._draft_uses_grouped_metadata
            ):
                draft_kwargs["num_tokens"] = padded_bs * self.max_tokens_per_req

            if not forward_mode.is_extend_or_mixed():
                from tokenspeed.runtime.execution.drafter.dflash import DFlash

                if isinstance(self.drafter, DFlash):
                    return

            if self._draft_metadata_sequence is not None:
                self._draft_metadata_sequence(
                    bs=padded_bs,
                    num_extends=num_extends,
                    req_pool_indices=req_pool_indices,
                    target_seq_lens=seq_lens,
                    draft_seq_lens=draft_seq_lens,
                    req_to_page=self.drafter.req_to_page,
                    forward_mode=forward_mode,
                    extend_kwargs=draft_extend_kwargs,
                    decode_kwargs=draft_kwargs,
                )
                return
            if forward_mode.is_extend_or_mixed():
                # Legacy draft backends fill prefill and subsequent decode
                # metadata in one EXTEND/MIXED initialization.
                self.draft_attn_backend.init_forward_metadata(
                    bs=padded_bs,
                    num_extends=num_extends,
                    req_pool_indices=req_pool_indices,
                    seq_lens=draft_seq_lens,
                    req_to_page=self.drafter.req_to_page,
                    forward_mode=forward_mode,
                    **draft_extend_kwargs,
                )
            else:
                draft_forward_mode = ForwardMode.DECODE
                if self._draft_uses_grouped_metadata:
                    draft_kwargs["num_tokens"] = padded_bs * self.max_tokens_per_req
                self.draft_attn_backend.init_forward_metadata(
                    bs=padded_bs,
                    num_extends=0,
                    req_pool_indices=req_pool_indices,
                    seq_lens=draft_seq_lens,
                    req_to_page=self.drafter.req_to_page,
                    forward_mode=draft_forward_mode,
                    **draft_kwargs,
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
            return global_bs <= self.max_bs
        if self.disable_padding:
            return self._has_cuda_graph_for_bs(bs)
        return bs <= self.max_bs

    def can_run(self, bs: int, ctx: ForwardContext) -> bool:
        return self._can_use_graph(bs, ctx)

    @property
    def uses_flat_table_source(self) -> bool:
        """The target backend's cache-table ABI bound during initialization."""

        return self._target_uses_flat_source

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
        if self.config.spec_algo == "DFLASH":
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
        paged_cache_block_tables: dict | None = None,
        paged_cache_block_table_base_offsets: dict | None = None,
        flat_cache_table_source: CacheTableSource | None = None,
        flat_block_tables: dict | None = None,
        flat_block_table_base_offsets: dict | None = None,
    ):
        """
        Unified forward entry point.

        Dispatches to the captured CUDA graph when possible; falls back to the
        eager forward_func otherwise.  The caller does not need to know which
        path was taken.
        """
        flat_cache_table_source = self._resolve_flat_source_input(
            flat_cache_table_source,
            flat_block_tables,
            flat_block_table_base_offsets,
            where="forward entry",
        )
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
            if mamba_pool_indices is not None:
                # Pad with -1 (PAD_SLOT_ID), NOT 0. Mamba slot 0 is a real
                # allocatable slot, so padding with 0 aliases a live request's
                # mamba state and corrupts it. -1 is the kernel-skipped pad id.
                mamba_pool_indices = torch.nn.functional.pad(
                    mamba_pool_indices, (0, pad), value=-1
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

        if use_graph:
            self._set_graph_state_write_indices(active_req_pool_indices, padded_bs)

        mamba_kwargs = {}
        if mamba_pool_indices is not None:
            mamba_kwargs["mamba_pool_indices"] = mamba_pool_indices
        if mamba_cow_src_indices is not None:
            mamba_kwargs["mamba_cow_src_indices"] = mamba_cow_src_indices
        if mamba_branching_seqlens is not None:
            mamba_kwargs["mamba_branching_seqlens"] = mamba_branching_seqlens
        if mamba_track_pool_indices is not None:
            mamba_kwargs["mamba_track_pool_indices"] = mamba_track_pool_indices

        if use_graph:
            if (
                bs == 0
                and paged_cache_block_tables is None
                and self._target_uses_paged_groups
            ):
                paged_cache_block_tables = self._capture_paged_cache_block_tables(
                    padded_bs,
                    self.token_to_kv_pool,
                )
            # The backend's stale-table guard also covers the bs==0 idle
            # replay: synthesize minimal valid tables for it.
            if (
                bs == 0
                and (
                    flat_cache_table_source is None
                    or not flat_cache_table_source.tables
                )
                and self._target_uses_flat_source
            ):
                flat_staging = getattr(
                    self.input_buffers, "flat_block_table_staging", None
                )
                if flat_staging is not None:
                    flat_cache_table_source = flat_staging.stage_idle(
                        padded_rows=padded_bs,
                        cache_generation=self._flat_pool_generation(
                            self.token_to_kv_pool
                        ),
                    )
                else:
                    flat_block_tables = self._idle_flat_block_tables(padded_bs)
                    flat_block_table_base_offsets = (
                        self._idle_flat_block_table_base_offsets(
                            padded_bs, flat_block_tables
                        )
                    )
                    flat_cache_table_source = resolve_cache_table_source(
                        paged_tables=None,
                        paged_base_offsets=None,
                        flat_tables=flat_block_tables,
                        flat_base_offsets=flat_block_table_base_offsets,
                        flat_generation=self._flat_pool_generation(
                            self.token_to_kv_pool
                        ),
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
                flat_cache_table_source=flat_cache_table_source,
                **mamba_kwargs,
            )

            # Runtime prepare() is called by ModelExecutor with per-request rids
            # BEFORE self.forward_step — we don't refill here to avoid clobbering
            # the per-request generators with the capture-stub generator.
            self.deepep_adapter.replay()

            graph_key = self._cuda_graph_key(padded_bs)
            with nvtx_range("graph_replay", color="red"):
                self.graphs[graph_key].replay()
            if bs == 0:
                if not self._idle_graph_replayed:
                    self._idle_graph_replayed = True
            elif not self._decode_graph_replayed:
                self._decode_graph_replayed = True

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
            # Eager parity with the replay stale-table guard: with >1 group
            # the single-table fallback would serve first-group pages to
            # every layer. Idle/bs==0 forwards carry no requests (exempt);
            # a single published group falls back to the single table.
            if (
                bs > 0
                and not ctx.forward_mode.is_idle()
                and (
                    flat_cache_table_source is None
                    or not flat_cache_table_source.tables
                )
                and self._target_uses_flat_source
                and len(self.token_to_kv_pool.paged_cache_group_specs) > 1
            ):
                raise RuntimeError(
                    "CudaGraphWrapper eager forward: pool publishes "
                    f"{len(self.token_to_kv_pool.paged_cache_group_specs)} "
                    "flat cache groups and the backend consumes flat tables, "
                    f"but flat_block_tables is missing/empty at bs={bs} "
                    f"({ctx.forward_mode.name}); the single-table fallback "
                    "would use one group's pages for all layers."
                )
            metadata_num_tokens = (
                {"num_tokens": ctx.input_num_tokens}
                if self._target_uses_paged_groups
                else {}
            )
            self._init_forward_metadata(
                padded_bs,
                ctx.num_extends,
                req_pool_indices,
                seq_lens,
                req_to_page=req_to_page,
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
                **metadata_num_tokens,
                paged_cache_block_tables=(
                    paged_cache_block_tables if self._target_uses_paged_groups else None
                ),
                paged_cache_block_table_base_offsets=(
                    paged_cache_block_table_base_offsets
                    if self._target_uses_paged_groups
                    else None
                ),
                flat_cache_table_source=(
                    flat_cache_table_source if self._target_uses_flat_source else None
                ),
                **mamba_kwargs,
            )

            result = self._forward_func(bs=bs, ctx=ctx, sampling_info=sampling_info)

        if use_graph and padded_bs != bs:
            ctx.bs = bs

        # Update mamba/GDN state after speculative verify
        if _should_update_mamba_state_after_mtp_verify(
            self.drafter, self.attn_backend, ctx.forward_mode
        ):
            accept_lengths = result[1]
            self.attn_backend.update_mamba_state_after_mtp_verify(accept_lengths, None)

        return result

    def runtime_evidence(self) -> dict[str, object]:
        """Return cold, JSON-safe proof of actual decode/idle graph use."""

        return {
            "captured_keys": [
                {"variant": str(variant), "batch_size": int(batch_size)}
                for variant, batch_size in sorted(self.graphs)
            ],
            "decode_replayed": bool(self._decode_graph_replayed),
            "idle_replayed": bool(self._idle_graph_replayed),
        }
