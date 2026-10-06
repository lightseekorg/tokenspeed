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

import gc
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import tokenspeed_kernel
import torch
from tokenspeed_kernel.ops.metadata import advance_accepted_frontier
from tokenspeed_kernel.ops.tuning import (
    autotune,
    autotune_cache_path,
    load_autotune_cache,
    save_autotune_cache,
    set_autotune_max_num_tokens,
    set_autotune_process_group,
)
from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.configs.model_config import ModelConfig
from tokenspeed.runtime.configs.utils import get_rope_parameters
from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)
from tokenspeed.runtime.engine.scheduler_utils import engram_context_len
from tokenspeed.runtime.execution.breakable_cuda_graph import active_forward
from tokenspeed.runtime.execution.context import ForwardContext, InputLogprobRows
from tokenspeed.runtime.execution.drafter import get_drafter_impl
from tokenspeed.runtime.execution.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
)
from tokenspeed.runtime.execution.forward_step import ForwardStepRunner
from tokenspeed.runtime.execution.forward_thread import ForwardThread
from tokenspeed.runtime.execution.input_buffer import InputBuffers
from tokenspeed.runtime.execution.memory_delta import MemoryDeltaObserver
from tokenspeed.runtime.execution.model_runner import ModelRunner
from tokenspeed.runtime.execution.multimodal_runtime import MultimodalRuntime
from tokenspeed.runtime.execution.nan_guard import NanGuard
from tokenspeed.runtime.execution.output_layout import ForwardOutputLayout
from tokenspeed.runtime.execution.prefill_graph import (
    PrefillGraph,
    dummy_batch_size,
    narrowing_prefill_model,
)
from tokenspeed.runtime.execution.query_shard import QueryShardPlan
from tokenspeed.runtime.execution.runtime_states import RuntimeStates
from tokenspeed.runtime.execution.tree_spec import TreeSpec, TreeSpecConfig
from tokenspeed.runtime.execution.types import (
    DpForwardMetadata,
    InputLogprobPlan,
    ModelExecutionResult,
    NGramInputs,
    RequestHistorySeeds,
)
from tokenspeed.runtime.execution.workspace import workspace_pool
from tokenspeed.runtime.grammar.capturable_grammar import (
    create_grammar_runtime,
    setup_grammar_step,
)
from tokenspeed.runtime.layers.attention.backends.base import (
    resolve_cuda_graph_support,
)
from tokenspeed.runtime.layers.attention.backends.cache_metadata import (
    CacheBatchMetadata,
)
from tokenspeed.runtime.layers.attention.backends.paged.tree_verify import (
    TreeVerifyInputs,
)
from tokenspeed.runtime.layers.attention.backends.support import resolve_tree_support
from tokenspeed.runtime.layers.attention.configs.base import is_block_drafter
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    validate_scheduler_config,
)
from tokenspeed.runtime.layers.attention.kv_cache.virtual_blocks import (
    local_pages_by_group,
)
from tokenspeed.runtime.layers.logits_processor import LogitsProcessorOutput
from tokenspeed.runtime.layers.paged_attention import (
    bind_cache_groups,
    check_block_drafter_storage,
)
from tokenspeed.runtime.sampling.backends.base import SamplingBackend
from tokenspeed.runtime.sampling.dp_sampling_config import (
    DpSamplingRuntimeLimits,
    setup_dp_sampling,
)
from tokenspeed.runtime.sampling.sampling_batch_info import SamplingBatchInfo
from tokenspeed.runtime.sampling.tree_verify import TreeVerifyBatch
from tokenspeed.runtime.utils import get_colorful_logger, is_pin_memory_available
from tokenspeed.runtime.utils.common import maybe_inference_mode
from tokenspeed.runtime.utils.env import envs
from tokenspeed.runtime.utils.hf_transformers_utils import get_context_length
from tokenspeed.runtime.utils.nvtx import nvtx_range
from tokenspeed.runtime.utils.server_args import ServerArgs

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
    from tokenspeed.runtime.layers.attention.kv_cache.base import CachePool
    from tokenspeed.runtime.sampling.sampling_params import SamplingParams

logger = get_colorful_logger(__name__)

LOG_MM_TIMING = envs.TOKENSPEED_LOG_MM_TIMING.get()
LOG_SPEC_ACCEPT_LENGTHS = envs.TOKENSPEED_LOG_SPEC_ACCEPT_LENGTHS.get()


def _sampling_info_for_requests(
    sampling_info: SamplingBatchInfo,
    requests: slice,
    *,
    mask_width: int | None,
    prefill: bool,
) -> SamplingBatchInfo:
    """Select original request parameters and their token-indexed masks."""
    info = sampling_info[requests]
    if mask_width is not None and sampling_info.vocab_mask is not None:
        mask = sampling_info.vocab_mask[
            requests.start * mask_width : requests.stop * mask_width
        ]
        info.vocab_mask = mask[::mask_width].contiguous() if prefill else mask
    return info


PREFILL_GRAPH_DEFAULT_MAX_TOKENS = 2048


def _resolve_prefill_graph_max_tokens(server_args) -> int:
    """Largest prefill-graph bucket: explicit value, or min(2048, chunk, kv budget).

    Returns 0 (graph off) when the MoE all-to-all backend is DeepEP: an
    extend-shaped forward takes DeepEP's normal dispatch, whose per-expert
    receive counts come back to the host, and a host sync cannot be captured.
    """
    if server_args.all2all_backend == "deepep":
        return 0
    if server_args.prefill_graph_max_tokens is not None:
        return int(server_args.prefill_graph_max_tokens)
    cap = PREFILL_GRAPH_DEFAULT_MAX_TOKENS
    if server_args.chunked_prefill_size:
        cap = min(cap, int(server_args.chunked_prefill_size))
    if server_args.max_total_tokens:
        cap = min(cap, int(server_args.max_total_tokens))
    return cap


def _cache_arena_attr(pool, name: str, default):
    """Read one arena attribute off a cache view, tolerating fakes.

    Every production pool is a view onto an arena; test doubles need not be.
    """
    return getattr(getattr(pool, "arena", None), name, default)


def _autotune_cache_key(
    server_args: ServerArgs,
    model_config: ModelConfig,
) -> dict[str, object] | None:
    """Rank-independent identity for tactics that may safely share a cache.

    Returns None under deterministic numerics, which keep heuristic tactics
    and so must not read a cache that tuned runs wrote.
    """
    if server_args.numerics != "auto":
        return None
    mapping = server_args.mapping
    return {
        "model": model_config.model_path,
        "role": server_args.disaggregation_mode,
        "revision": model_config.revision,
        "architectures": getattr(model_config.hf_config, "architectures", None),
        "quantization": model_config.quantization,
        "dtype": server_args.dtype,
        "moe_backend": str(server_args.moe_backend),
        "attention_backend": str(server_args.attention_backend),
        "attention": (mapping.attn.tp_size, mapping.attn.dp_size),
        "dense": (mapping.dense.tp_size, mapping.dense.dp_size),
        "moe": (mapping.moe.tp_size, mapping.moe.ep_size, mapping.moe.dp_size),
        "linear_attention_tp": mapping.linear_attn.tp_size,
        # Pipeline stages can reuse shape-keyed tactics from a full-model run.
        "speculative_algorithm": server_args.speculative_algorithm,
        "speculative_num_draft_tokens": server_args.speculative_num_draft_tokens,
    }


def select_dspark_context_producer(
    *,
    spec_algo: str | None,
    pp_size: int,
    draft_model: torch.nn.Module | None,
    draft_token_to_kv_pool,
):
    """Return the stage's DSpark context producer, or None when nothing is produced.

    A DSpark draft reads target taps that live on several pipeline stages, so
    each stage projects its own during the target forward and the final stage
    writes the draft context; off the pipeline the drafter keeps projecting
    and writing context itself. An MTP (NextN) draft consumes only the last
    stage's captured hidden states, so no stage produces anything for it: the
    executor then captures FULL hidden states for its drafter. EAGLE3 is
    refused here as well as in ``ServerArgs``: its aux taps live on several
    stages and nothing carries them through the stage boundary.

    Args:
        spec_algo: The speculative algorithm, or None without speculation.
        pp_size: Pipeline stage count; a single stage never produces.
        draft_model: The loaded draft model, or None when this stage builds
            none (``pipeline_stage_builds_draft``).
        draft_token_to_kv_pool: The draft cache pool this stage owns, or None.

    Returns:
        A ``DSparkContextProducer`` for a pipeline DSpark draft, else None.

    Raises:
        ValueError: EAGLE3 on the pipeline.
        TypeError: A block drafter (DFLASH/DSPARK) on the pipeline whose model
            cannot produce context across stages; it would draft from one
            stage's taps alone.
    """
    if spec_algo is None or pp_size <= 1:
        return None
    if spec_algo == "EAGLE3":
        raise ValueError(
            "EAGLE3 cannot run on a pipeline: its aux taps live on several "
            "stages and nothing carries them through the stage boundary."
        )
    from tokenspeed.runtime.execution.dspark_context import (
        DSparkContextModel,
        DSparkContextProducer,
    )

    if isinstance(draft_model, DSparkContextModel):
        return DSparkContextProducer(draft_model, draft_token_to_kv_pool)
    if is_block_drafter(spec_algo, is_draft=True):
        raise TypeError(
            f"{type(draft_model).__name__} cannot produce DSpark context across "
            "pipeline stages."
        )
    return None


@dataclass
class ModelExecutorConfig:
    """
    Scalar configuration for ModelExecutor.
    Contains only primitive values — no heavy objects.
    Created once via from_server_args() and injected into ModelExecutor.
    """

    # Rank-local graph-padding req-pool index. The C++ scheduler owns real rows
    # 1..max_batch_size and row 0 is reserved, so this must sit after the
    # scheduler-owned range.
    max_req_pool_size: int
    output_length: int
    enforce_eager: bool
    prefix_granularity: int
    max_num_seqs: int
    chunked_prefill_size: int
    vocab_size: int
    # Logical context limit (user semantics: input validation, max_new_tokens
    # folding, stop checks all key off this).
    context_len: int
    # Physical KV extent: context_len + ServerArgs.spec_context_pad. Spec
    # verify on the overlap scheduler commits up to that pad past context_len
    # for a request that already finished (see _SPEC_OVERSHOOT_SPANS in
    # server_args.py); every buffer/table sized per request must use this.
    physical_context_len: int
    device: str
    gpu_id: int
    global_rank: int
    cudagraph_capture_sizes: list[int] | None
    disable_cuda_graph_padding: bool
    # Children per draft node per step; above 1 the draft is a tree (tree_spec.py).
    spec_topk: int
    max_cudagraph_capture_size: int
    model_is_mrope: bool
    autotune_cache_key: dict[str, object] | None
    # The prefill role of a disaggregated deployment computes prompts only:
    # it never runs a decode/verify step of its own, so the decode graph is
    # never captured there while the prefill graph keeps its ordinary gating.
    prefill_only: bool
    # The mirror image: an attention layout that serves decode rows only
    # (head TP under attention DP), so startup never runs an extend-shaped
    # dummy forward and tunes on a decode-shaped one instead.
    decode_only_attention: bool
    # Explicit None selects the minimum request count for each token bucket.
    prefill_graph_capture_batch_sizes: list[int] | None
    # Prompt-logprob gather: how many prompt rows go through the LM head at
    # once (``--input-logprob-chunk-tokens``). Bounds the transient
    # ``[rows, vocab]`` logits; log-softmax is row-local so the value never
    # changes a result.
    input_logprob_chunk_tokens: int
    # Draft-prob rejection sampling: the drafter records its per-step
    # proposal distributions in RuntimeStates.draft_probs and verify accepts
    # with coin * q(x) < p(x) (see --enable-speculative-sampling). Selects
    # the verify rule, so it is explicit.
    enable_speculative_sampling: bool
    # Query context parallelism (mapping.attn.qcp_size / qcp_rank): an extend
    # forward's rows are split over the attention TP group and this executor
    # computes shard ``query_shard_rank``; size 1 means every rank computes
    # every row.
    query_shard_size: int
    query_shard_rank: int
    enable_nan_detection: bool = False
    disable_autotune: bool = False
    enable_cudagraph_gc: bool = False

    # ====== DP =========
    data_parallel_size: int = 1
    world_size: int = 1
    world_group: list[int] | None = None

    # ====== PP (prefill chunk pipeline) =========
    pp_size: int = 1
    pp_rank: int = 0
    pp_group: tuple[int, ...] | None = None

    # ====== SPEC =========
    spec_algo: str | None = None
    spec_num_steps: int | None = None
    # Verify window width: spec_num_steps + 1 for a chain, the tree's node budget otherwise.
    spec_num_tokens: int | None = None
    # Recorded draft probabilities above this value mark a slot with no
    # proposal (always reject); only read under enable_speculative_sampling.
    spec_reject_draft_prob_threshold: float = 2.0
    overlap_schedule_depth: int = 0
    dp_sampling: bool = False
    dp_sampling_min_bs: int | None = None

    # ====== GRAMMAR =========
    # "none" disables all grammar handling; otherwise the backend name
    # (currently only "xgrammar" is implemented).
    grammar_backend: str = "xgrammar"
    # Force the synchronous eager grammar fallback even on CUDA. For
    # parity-testing the captured-grammar path.
    disable_capturable_grammar: bool = False

    # ====== PREFILL CUDA GRAPH (breakable) =========
    disable_prefill_graph: bool = False
    # Opt-in: > 0 enables the prefill graph and caps the largest token bucket.
    prefill_graph_max_tokens: int = 0
    # Explicit bucket list overriding the ladder (see get_prefill_token_buckets).
    prefill_graph_capture_sizes: list[int] | None = None

    @staticmethod
    def from_server_args(
        server_args: ServerArgs,
        model_config: ModelConfig,
        max_req_pool_size: int,
        gpu_id: int,
        global_rank: int,
        prefix_granularity: int,
        overlap_schedule_depth: int = 0,
    ) -> ModelExecutorConfig:
        output_length = (
            server_args.speculative_num_draft_tokens
            if server_args.speculative_algorithm
            else 1
        )
        rope_parameters = get_rope_parameters(model_config.hf_text_config)
        model_is_mrope = bool(rope_parameters and "mrope_section" in rope_parameters)

        # Spec verify commits positions up to physical_context_len - 1 for a
        # finished request lingering one overlap step. Rope cos/sin tables are
        # precomputed for the model's derived context length, so positions in
        # the pad read past them when context_len is set flush against the
        # model limit. The values only feed a dead request's garbage KV, but
        # the read itself is out of table bounds — warn so the operator can
        # lower --max-model-len by the pad.
        physical_context_len = model_config.context_len + server_args.spec_context_pad
        derived_context_len = get_context_length(model_config.hf_text_config)
        if physical_context_len > derived_context_len:
            logger.warning(
                f"physical context extent {physical_context_len!s} (context_len "
                f"{model_config.context_len!s} + spec overshoot "
                f"pad {server_args.spec_context_pad!s}) exceeds the model's derived "
                f"context length {derived_context_len!s}; "
                "positions in the pad index past the precomputed rope tables. "
                "Lower --max-model-len by at least "
                f"{physical_context_len - derived_context_len!s} to stay in bounds.",
            )

        # User intent only; backend-imposed graph restrictions are declared on
        # the backend classes (cuda_graph_support) and resolved in
        # ModelExecutor.__init__ once the backend instances exist.
        disable_prefill_graph = bool(server_args.disable_prefill_graph)

        return ModelExecutorConfig(
            max_req_pool_size=max_req_pool_size,
            output_length=output_length,
            enforce_eager=server_args.enforce_eager,
            prefix_granularity=prefix_granularity,
            max_num_seqs=server_args.max_num_seqs,
            chunked_prefill_size=server_args.chunked_prefill_size,
            vocab_size=model_config.vocab_size,
            context_len=model_config.context_len,
            physical_context_len=(
                model_config.context_len + server_args.spec_context_pad
            ),
            device=server_args.device,
            gpu_id=gpu_id,
            global_rank=global_rank,
            cudagraph_capture_sizes=server_args.cudagraph_capture_sizes,
            disable_cuda_graph_padding=server_args.disable_cuda_graph_padding,
            disable_autotune=server_args.disable_autotune,
            autotune_cache_key=_autotune_cache_key(server_args, model_config),
            enable_cudagraph_gc=server_args.enable_cudagraph_gc,
            max_cudagraph_capture_size=server_args.max_cudagraph_capture_size,
            disable_prefill_graph=disable_prefill_graph,
            prefill_graph_max_tokens=_resolve_prefill_graph_max_tokens(server_args),
            prefill_graph_capture_sizes=server_args.prefill_graph_capture_sizes,
            prefill_graph_capture_batch_sizes=server_args.prefill_graph_capture_batch_sizes,
            model_is_mrope=model_is_mrope,
            prefill_only=server_args.disaggregation_mode == "prefill",
            input_logprob_chunk_tokens=server_args.input_logprob_chunk_tokens,
            decode_only_attention=server_args.mapping.attn.head_tp_serves_decode_only,
            query_shard_size=server_args.mapping.attn.qcp_size,
            query_shard_rank=server_args.mapping.attn.qcp_rank,
            data_parallel_size=server_args.mapping.attn.dp_size,
            world_size=server_args.mapping.world_size,
            world_group=server_args.mapping.world_group,
            pp_size=server_args.mapping.pp_size,
            pp_rank=(server_args.mapping.pp_rank if server_args.mapping.has_pp else 0),
            pp_group=(
                server_args.mapping.pp_group if server_args.mapping.has_pp else None
            ),
            spec_algo=server_args.speculative_algorithm,
            spec_num_steps=server_args.speculative_num_steps,
            spec_num_tokens=server_args.speculative_num_draft_tokens,
            enable_speculative_sampling=server_args.enable_speculative_sampling,
            spec_reject_draft_prob_threshold=server_args.spec_reject_draft_prob_threshold,
            spec_topk=(
                server_args.speculative_eagle_topk
                if server_args.speculative_algorithm
                else 1
            ),
            overlap_schedule_depth=overlap_schedule_depth,
            dp_sampling=server_args.dp_sampling,
            dp_sampling_min_bs=server_args.dp_sampling_min_bs,
            enable_nan_detection=server_args.enable_nan_detection,
            grammar_backend=server_args.grammar_backend,
            disable_capturable_grammar=server_args.disable_capturable_grammar,
        )


class ModelExecutor:
    """
    Orchestrates model forward execution.
    """

    def __init__(
        self,
        config: ModelExecutorConfig,
        model_runner: ModelRunner,
        attn_backend: AttentionBackend,
        token_to_kv_pool: CachePool,
        sampling_backend: SamplingBackend,
        draft_model_runner: ModelRunner | None = None,
        draft_attn_backend: AttentionBackend | None = None,
        draft_token_to_kv_pool: CachePool | None = None,
    ):
        self.device = config.device
        self.config = config
        self.model_runner = model_runner
        self.sampling_backend = sampling_backend
        self.attn_backend = attn_backend
        self.token_to_kv_pool = token_to_kv_pool
        # Every pool runs on the shared cache arena and publishes a runtime
        # contract; the per-group tables travel as CacheBatchMetadata. Fail
        # fast here rather than at the first forward or, worse, a CUDA-graph
        # capture-path assert: an uncovered contract family means a backend
        # that never reads that group's tables.
        validate_scheduler_config(
            attn_backend=attn_backend,
            kv_pool=token_to_kv_pool,
        )
        self._cache_runtime_contract = token_to_kv_pool.arena.runtime_contract
        self.draft_attn_backend = draft_attn_backend
        self.draft_token_to_kv_pool = draft_token_to_kv_pool
        self._draft_model_runner = draft_model_runner
        self._draft_final_step_counter = None
        self._pp_wire_logged = False

        max_bs = config.max_num_seqs // max(config.data_parallel_size, 1)

        spec_num_tokens = config.spec_num_tokens if config.spec_algo is not None else 1
        self.input_buffers = InputBuffers(
            max_bs=max_bs,
            max_num_tokens=max(
                config.chunked_prefill_size, max_bs * config.output_length
            ),
            state_write_padding_pool_index=config.max_req_pool_size,
            device=self.device,
        )
        # Group-keyed zeroing requests carry scheduler (virtual) block IDs; this
        # rank's position in the DCP group selects the pages it owns.
        self._cache_dcp_rank = model_runner.mapping.attn.dcp_rank
        ngram_context = engram_context_len(model_runner.model_config.hf_text_config)
        if ngram_context and (config.pp_size != 1 or config.overlap_schedule_depth > 1):
            raise NotImplementedError(
                "Engram input history requires PP=1 and in-flight depth <= 1"
            )
        self.input_buffers.init_ngram_buffers(ngram_context)
        self.runtime_states = RuntimeStates(
            req_pool_size=config.max_req_pool_size,
            vocab_size=config.vocab_size,
            device=self.device,
            output_length=config.output_length,
        )
        self.runtime_states.init_ngram_state(ngram_context)
        self.runtime_states.init_request_token_history(
            config.physical_context_len
            if model_runner.model_config.requires_request_token_history
            else 0
        )
        if config.enable_speculative_sampling:
            if config.spec_algo is None:
                raise ValueError(
                    "enable_speculative_sampling needs a speculative drafter to "
                    "record proposal distributions for"
                )
            self.runtime_states.init_draft_probs(
                spec_num_tokens=spec_num_tokens,
                reject_threshold=config.spec_reject_draft_prob_threshold,
            )
        # Sized like InputBuffers.max_bs so the padded graph-bucket bs fits.
        self.nan_guard = NanGuard.create(
            config.enable_nan_detection,
            max_bs,
            self.device,
        )
        self.dspark_context_producer = select_dspark_context_producer(
            spec_algo=config.spec_algo,
            pp_size=config.pp_size,
            draft_model=(
                draft_model_runner.model if draft_model_runner is not None else None
            ),
            draft_token_to_kv_pool=draft_token_to_kv_pool,
        )
        if self.config.spec_algo is not None and self._pp_is_last_stage:
            # Model-to-model wiring (shared embed/head, eagle3 capture ids)
            # already happened in create_model_runner, right after both
            # models loaded. Here only the drafter instance is built and
            # wired to the target.
            DrafterImpl = get_drafter_impl(config.spec_algo, draft_model_runner.model)
            self.drafter = DrafterImpl(
                spec_num_tokens=config.spec_num_tokens,
                spec_num_steps=config.spec_num_steps,
                draft_model_runner=draft_model_runner,
                runtime_states=self.runtime_states,
                input_buffers=self.input_buffers,
                attn_backend=draft_attn_backend,
                token_to_kv_pool=draft_token_to_kv_pool,
                vocab_size=config.vocab_size,
            )
            self.drafter.wire_target(self.model_runner.model)
            # Draft-prob sampling reads the request's temperature / top-k /
            # seed from the verifier's pool buffers: one owner of per-request
            # sampling state.
            self.drafter.bind_sampling_backend(self.sampling_backend)
            MultimodalRuntime.wire_drafter(
                self.input_buffers, self.model_runner.model_config
            )
        else:
            self.drafter = None

        self.tree_spec: TreeSpec | None = None
        if config.spec_topk > 1:
            self._init_tree_spec(max_bs)

        self.grammar_runtime = create_grammar_runtime(
            grammar_backend=config.grammar_backend,
            disable_capturable=config.disable_capturable_grammar,
            is_nvidia=current_platform().is_nvidia,
            max_bs=max_bs,
            vocab_size=config.vocab_size,
            max_tokens_per_req=spec_num_tokens,
            device=self.device,
        )

        self._configure_for_pools()

        # Backend-declared CUDA-graph support, AND-composed over the target
        # and draft trees (the decode graph records the whole step, drafter
        # loop included). Startup-time and class-attribute-driven, so every
        # DP rank resolves the same answer.
        graph_support = resolve_cuda_graph_support(attn_backend, draft_attn_backend)

        self.dp_sampling_runtime_config = setup_dp_sampling(
            model=self.model_runner.model,
            sampling_backend=self.sampling_backend,
            requested=self.config.dp_sampling,
            drafter_available=self.drafter is not None,
            limits=DpSamplingRuntimeLimits(
                runtime_vocab_size=self.config.vocab_size,
                max_num_seqs=config.max_num_seqs,
                data_parallel_size=config.data_parallel_size,
                num_tokens_per_req=spec_num_tokens,
                configured_min_bs=self.config.dp_sampling_min_bs,
                device=self.device,
            ),
        )
        self._last_dp_sampling_route_log: (
            tuple[str, int, bool, int, int, bool, int] | None
        ) = None

        self._active_multimodal_context = None
        self._active_positions_override = None

        self._graph_support = graph_support
        self._build_graph_owners()

        # Encoder graphs are installed before KV-cache sizing and retained by
        # the model runner; preserve the executor-level handle for callers.
        self.encoder_graph_wrappers = getattr(
            self.model_runner, "encoder_graph_wrappers", {}
        )

        self.device_module = torch.get_device_module(self.device)
        # Two streams, named once. `default_stream` is the forward thread's
        # own: page zeroing runs here, and the cache ops take it by name for
        # their fences and start events. `execution_stream` carries the model
        # launches and the runtime-state writes. Dependencies between them are
        # placed by the consumer: each forward waits on the default stream in
        # its prologue; zeroing and write-back wait on the execution stream
        # themselves.
        self.default_stream = self.device_module.default_stream(self.device)
        self.execution_stream = self.device_module.Stream()
        # The data plane: every CUDA-touching operation after startup is
        # submitted here and runs in FIFO order on one thread. The event loop
        # (control plane) never waits on the GPU along the per-round path —
        # dispatch submits and moves on, only commit joins — so a round stays
        # microseconds and its cross-rank gloo collectives always find every
        # rank promptly regardless of GPU depth. (A few low-rate paths do
        # block on purpose: the DP idle forward, the PD receive and landing,
        # the post-wake KV repair, RL weight updates. See DeviceHandle.)
        self.forward_thread = ForwardThread(self.device)
        # Throttles the mm_timing line inside execute_forward_op; the
        # per-round batch lines have their own counter on the control plane.
        self.log_step = 0
        self._prev_decode_bs: int = 0
        self._sentinel_neg1 = torch.tensor(-1, device=self.device, dtype=torch.int64)
        self.mm_runtime = MultimodalRuntime(
            model_is_mrope=config.model_is_mrope,
            input_buffers=self.input_buffers,
            device=self.device,
        )

        logger.info("ModelExecutor initialized")

    def _init_tree_spec(self, max_bs: int) -> None:
        """Arm draft-tree speculation: shared tree state, verify leaves, drafter."""
        if (
            self.runtime_states.has_request_token_history
            or self.runtime_states.ngram_accepted_tokens is not None
        ):
            raise NotImplementedError(
                "draft trees write verify-window tokens in node order; a target that "
                "reads request token or n-gram history needs them along the accepted path"
            )
        if not self.sampling_backend.supports_tree_verify:
            raise NotImplementedError(
                f"{type(self.sampling_backend).__name__} cannot verify draft trees; "
                "use --sampling-backend greedy or triton"
            )
        if self.draft_attn_backend is self.attn_backend:
            raise NotImplementedError(
                "draft trees bind verify and lanes on distinct target and draft backends"
            )
        resolve_tree_support(self.attn_backend, self.draft_attn_backend)
        config = self.config
        self.tree_spec = TreeSpec(
            TreeSpecConfig(
                topk=config.spec_topk,
                num_steps=config.spec_num_steps,
                num_nodes=config.spec_num_tokens,
            ),
            max_bs=max_bs,
            device=torch.device(self.device),
        )
        self.runtime_states.init_draft_trees(config.spec_num_tokens)
        self.attn_backend.bind_tree_verify(
            TreeVerifyInputs(
                self.tree_spec.mask_buf,
                config.spec_num_tokens,
                parent=self.tree_spec.parent_buf,
            )
        )
        self.drafter.bind_tree(self.tree_spec)

    def _compact_accepted_tree(
        self, bs: int, logits_output: LogitsProcessorOutput
    ) -> None:
        """Pack each request's accepted path to the front of its verify window:
        target hidden rows, target KV, and the window positions back to ``vc + i``."""
        tree = self.tree_spec
        path = self.sampling_backend.accepted_path(bs, tree.num_nodes)
        self.attn_backend.compact_verify_window(path)
        tree.compact_rows(
            path,
            logits_output.hidden_states,
            self.input_buffers.positions_buf[: bs * tree.num_nodes],
        )

    def _configure_for_pools(self) -> None:
        """Publish the bound pools to the backends and the model's layers."""
        self.attn_backend.configure_runtime(
            cache_group_specs=tuple(self.token_to_kv_pool.arena.cache_group_specs),
            cache_group_page_counts=_cache_arena_attr(
                self.token_to_kv_pool, "cache_group_page_counts", None
            ),
        )
        if self.draft_attn_backend is not None:
            self.draft_attn_backend.configure_runtime(
                cache_group_specs=tuple(
                    _cache_arena_attr(
                        self.draft_token_to_kv_pool, "cache_group_specs", ()
                    )
                ),
                cache_group_page_counts=_cache_arena_attr(
                    self.draft_token_to_kv_pool, "cache_group_page_counts", None
                ),
            )

        # Storage is the plan's decision: stamp each attention layer's cache
        # group from the pool and check the group retains what the layer's
        # mask can see. A block drafter additionally borrows the target's
        # full-history group -- it writes at the target's cache locations.
        bind_cache_groups(self.model_runner.model, self.token_to_kv_pool)
        draft_runner = self._draft_model_runner
        if draft_runner is not None and self.draft_token_to_kv_pool is not None:
            bind_cache_groups(draft_runner.model, self.draft_token_to_kv_pool)
            if is_block_drafter(self.config.spec_algo, is_draft=True):
                check_block_drafter_storage(draft_runner.model, self.token_to_kv_pool)

    def release_graphs(self) -> None:
        """Drop the captured graphs and unfreeze the workspace they pinned.

        A caller that measured a capture releases here before it allocates the
        replacement arena: a captured graph's private pool is not returned by
        empty_cache, so it would still hold memory the new arena and the
        serving capture need. The graphs sit in reference cycles, so the
        collection is what actually drops them; without it empty_cache returns
        0.34 GB less on Qwen3-8B.
        """
        self.forward_step.release_graphs()
        self.prefill_graph.release_graphs()
        if self.drafter is not None:
            self.drafter.release_prefill_graph()
        workspace_pool(self.device).unfreeze()
        gc.collect()

    def set_cache_pool(
        self,
        token_to_kv_pool: CachePool,
        draft_token_to_kv_pool: CachePool | None,
    ) -> None:
        """Take a replacement cache pool the backends are already bound to.

        A construction-time operation, after release_graphs: the backends took
        the pool when it was built, and this republishes what the executor
        holds itself and rebuilds the graph owners for the new arena.
        """
        self.token_to_kv_pool = token_to_kv_pool
        # A rebind publishes a second pool; it gets the same fail-fast check.
        validate_scheduler_config(
            attn_backend=self.attn_backend,
            kv_pool=token_to_kv_pool,
        )
        self._cache_runtime_contract = token_to_kv_pool.arena.runtime_contract
        self.draft_token_to_kv_pool = draft_token_to_kv_pool
        if self.drafter is not None:
            self.drafter.set_cache_pool(draft_token_to_kv_pool)
        if self.dspark_context_producer is not None:
            self.dspark_context_producer.set_cache_pool(draft_token_to_kv_pool)

        self._configure_for_pools()
        self._build_graph_owners()

    def _build_graph_owners(self) -> None:
        """Build the decode and prefill graph owners for the bound pools.

        A graph owner is built for one pool: a rebind releases its graphs and
        replaces it rather than re-pointing it at a new arena.
        """
        self.forward_step = ForwardStepRunner(
            forward_func=self._forward_step,
            attn_backend=self.attn_backend,
            token_to_kv_pool=self.token_to_kv_pool,
            input_buffers=self.input_buffers,
            config=self.config,
            drafter=self.drafter,
            draft_attn_backend=self.draft_attn_backend,
            draft_token_to_kv_pool=self.draft_token_to_kv_pool,
            capturable_grammar=self.capturable_grammar,
            eager_grammar_buffers=self.eager_grammar_buffers,
            sampling_backend=self.sampling_backend,
            runtime_states=self.runtime_states,
            decode_graph_supported=self._graph_support.decode_graph,
        )
        # Eager warmup can be DP-asymmetric; prewarm RSAG under uniform dummy inputs.
        # The prefill role never decodes: a DECODE-shaped dummy would need the
        # verify scratch it does not allocate, and its ranks initialize lazy
        # collectives together on their first prefill round instead.
        if self.config.enforce_eager and not self.config.prefill_only:
            logger.info("Prewarming Triton RSAG communication states")
            self.forward_step.warmup_decode_path(batch_sizes=(1,), graph_phase=True)
            logger.info("Finished prewarming Triton RSAG communication states")

        # Prompt (input) logprobs need the LM head to score every prompt row:
        # one activation row per input token, which a model that narrows its
        # prefill rows (NarrowingPrefillModel) does not keep. The last pipeline
        # stage scores them and the commit path broadcasts the result to the
        # other stages with the sampled tokens. Decided here, once, so the
        # ingress refuses such requests instead of the data plane finding out.
        self.supports_prompt_logprobs: bool = (
            narrowing_prefill_model(self.model_runner.model) is None
        )

        # Breakable prefill (extend) CUDA graphs, the extend-mode analogue of
        # the decode wrapper above; borrows the decode capture stream so all
        # graphs share one mempool-reuse domain.
        self.prefill_graph = PrefillGraph(
            model_runner=self.model_runner,
            attn_backend=self.attn_backend,
            token_to_kv_pool=self.token_to_kv_pool,
            input_buffers=self.input_buffers,
            config=self.config,
            drafter=self.drafter,
            graph_supported=self._graph_support.prefill_graph,
        )

    def capture_graphs(
        self,
        *,
        entries: int | None,
        observer: MemoryDeltaObserver,
    ) -> None:
        """Pin the workspace, then capture the graphs.

        A step of its own, so the caller decides when the graph owners start
        recording the pools' buffers. Construction has already read the pools
        (validation, configure_runtime, bind_cache_groups and the runners'
        init_cuda_graph_state), so a rebind between the two re-publishes them
        through ``set_cache_pool``. Tuning is a separate step, run once per boot:
        a captured graph keeps the tactic chosen when it was captured, and
        set_autotune_max_num_tokens must be called once per process.
        ``entries`` samples each ladder at the probe's positions; ``None``
        captures every graph.
        """
        workspace_pool(self.device).freeze()

        if not self.forward_step.disable:
            self.forward_step.capture(entries=entries, observer=observer)
        if not self.prefill_graph.disable:
            self.prefill_graph.capture(
                self.forward_step, entries=entries, observer=observer
            )
            # Only a drafter that declared the ladder gets a window to fill.
            if self.captures_drafter_prefill_graph:
                self.drafter.capture_prefill_graph(
                    self.forward_step.stream, observer.measure("prefill:drafter")
                )

    @property
    def captures_drafter_prefill_graph(self) -> bool:
        """Whether this boot records a drafter prefill graph beside the ladders.

        One property because the capture and the projection's entry count have
        to reach the same verdict: declaring a window the capture never fills,
        or filling one that was never declared, both kill the boot inside
        ``_estimate_series``.
        """
        return (
            not self.prefill_graph.disable
            and self.drafter is not None
            and self.drafter.captures_prefill_graph
        )

    def autotune(self) -> None:
        """Tune missing prefill/decode configs and persist the shared cache."""

        per_rank_max_batch = max(
            1,
            int(self.config.max_num_seqs)
            // max(int(self.config.data_parallel_size), 1),
        )
        if self.model_runner is None:
            return
        ib = self.input_buffers
        # The one traversal that discovers each operator is shaped like the
        # forwards this engine runs: an extend over the prefill budget, or,
        # for an attention layout that serves decode rows only (head TP), a
        # decode step at the largest batch -- the shape capture records later.
        decode_only = self.config.decode_only_attention
        if decode_only:
            num_tokens = (
                self.forward_step.max_decode_bs * self.forward_step.max_tokens_per_req
            )
        else:
            num_tokens = min(
                ib.input_ids_buf.numel(),
                int(self.config.context_len) * per_rank_max_batch,
            )
            if self.config.chunked_prefill_size > 0:
                num_tokens = min(num_tokens, int(self.config.chunked_prefill_size))
        set_autotune_max_num_tokens(num_tokens)
        cpu_group = None
        if self.config.world_size > 1:
            cpu_group = pg_manager.get_process_group("gloo", self.config.world_group)
        owner_rank = (
            self.config.world_group[0]
            if self.config.world_group
            else self.config.global_rank
        )

        cache_path = (
            autotune_cache_path(self.config.autotune_cache_key)
            if self.config.autotune_cache_key is not None
            else None
        )
        load_autotune_cache(cache_path, cpu_group, owner_rank)

        if self.config.pp_size > 1:
            # These dummy forwards do not perform pipeline stage transfers.
            logger.info("Kernel tuning skipped under pipeline parallelism")
            return
        if self.config.disable_autotune:
            logger.info("Kernel tuning disabled (--disable-autotune)")
            return

        # One traversal discovers each operator. Its dispatch exposes the
        # tunable branches; FI enumerates native buckets independently of the
        # graph ladder. Prefill stays inside the actual buffer/request limits.
        logger.info(
            f"Kernel startup tuning with {num_tokens} "
            f"{'decode' if decode_only else 'prefill'} tokens"
        )

        tic = time.time()
        set_autotune_process_group(cpu_group)
        # Capture later refreshes the metadata created here in place.
        with torch.no_grad(), autotune(
            tune_mode=True, tuning_buckets=None, round_up=None
        ):
            # Dummy forwards must not borrow live Engram history or masks.
            ib.fill_dummy_decode_buffers(
                batch_size=ib.max_bs, total_tokens=ib.max_num_tokens
            )
            if decode_only:
                # No extend forward exists on this layout; the decode step
                # reaches the target and, through the speculative forward,
                # the draft.
                if self.drafter is not None:
                    self._autotune_draft_experts(num_tokens)
                self.forward_step.warmup_decode_path(
                    batch_sizes=(self.forward_step.max_decode_bs,), graph_phase=False
                )
            else:
                bs = dummy_batch_size(num_tokens, self.config.context_len)
                ctx = self.prefill_graph.make_dummy_batch(num_tokens, bs)
                # The model's rows: the span, or this rank's query shard of
                # it, as on a real extend (_run_target_forward).
                rows = (
                    slice(0, num_tokens)
                    if ctx.query_shard is None
                    else ctx.query_shard.local_slice
                )
                positions = (
                    ib.mrope_positions_buf[:, rows]
                    if self.config.model_is_mrope
                    else ib.positions_buf[rows]
                )
                with active_forward(ctx):
                    self.model_runner.forward(
                        ctx=ctx,
                        input_ids=ib.input_ids_buf[rows],
                        positions=positions,
                        **self._model_input_kwargs(num_tokens, ctx.bs, rows),
                    )
                if self.drafter is not None:
                    self._autotune_draft_experts(num_tokens)
                if self.drafter is not None and not self.config.prefill_only:
                    # Prefill-only roles do not allocate decode/verify scratch.
                    # The draft model is reached through the shared speculative
                    # forward, not model_runner.forward above. One request
                    # exposes its operators; FI still owns their bucket
                    # enumeration.
                    self.forward_step.warmup_decode_path(
                        batch_sizes=(1,), graph_phase=False
                    )
        set_autotune_process_group(None)

        torch.get_device_module(self.device).synchronize()
        save_autotune_cache(cache_path, cpu_group, owner_rank)

        logger.info(f"Kernel startup tuning finished in {time.time() - tic:.1f}s")

    def _autotune_draft_experts(self, num_tokens: int) -> None:
        """Tune the draft model's routed-expert kernels inside the tuning window.

        The dummy prefill drives the target model only. A draft with its own
        expert geometry (DSpark's 128-expert MoE) is keyed separately by the
        tuner and would otherwise fall back to untuned tactics on every draft
        step. One apply per distinct expert geometry at the prefill token
        count lets the tuner enumerate every smaller bucket, exactly as the
        target's prefill does for the target experts.
        """
        from tokenspeed.runtime.layers.moe.expert import MoELayer

        tuned: set[tuple] = set()
        for layer in self.drafter.draft_model_runner.model.modules():
            if not isinstance(layer, MoELayer):
                continue
            # All-to-all and MegaMoE plans own collectives sized by the real
            # batch geometry; they have no FlashInfer tactics to tune either.
            if (
                layer.plan["a2a_backend"] == "deepep"
                or layer.plan["solution"] == "mega_moe"
            ):
                continue
            key = (
                layer.plan["apply_kernel_name"],
                layer.hidden_size,
                layer.intermediate_size,
                layer.num_experts,
                layer.top_k,
            )
            if key in tuned:
                continue
            tuned.add(key)
            hidden_states = torch.zeros(
                num_tokens,
                layer.hidden_size,
                dtype=layer.input_dtype,
                device=self.device,
            )
            # Distinct expert ids per token: kernels may reject repeats.
            scores = torch.rand(num_tokens, layer.num_experts, device=self.device)
            topk_weights, topk_ids = torch.topk(scores, layer.top_k, dim=-1)
            topk_weights = topk_weights / topk_weights.sum(-1, keepdim=True)
            if layer.supports_precomputed_topk:
                tokenspeed_kernel.moe_apply(
                    layer.plan,
                    hidden_states,
                    layer,
                    None,
                    topk_weights=topk_weights,
                    topk_ids=topk_ids.to(torch.int32),
                    num_tokens_global=num_tokens,
                )
            else:
                tokenspeed_kernel.moe_apply(
                    layer.plan,
                    hidden_states,
                    layer,
                    scores.to(torch.float32),
                    num_tokens_global=num_tokens,
                )
            logger.info(
                f"Kernel tuning covered draft experts {layer.prefix!s} "
                f"({layer.plan['apply_kernel_name']!s}, {layer.num_experts:d} experts)"
            )

    @property
    def capturable_grammar(self):
        """Captured-graph grammar handle, or None on the eager-fallback path.

        Used by ``_forward_step`` to fence the side-stream grammar fill
        against the captured forward — those calls only make sense for
        the captured flavor of grammar runtime.
        """
        from tokenspeed.runtime.grammar.capturable_grammar import (
            CapturableGrammarExecutor,
        )

        return (
            self.grammar_runtime
            if isinstance(self.grammar_runtime, CapturableGrammarExecutor)
            else None
        )

    @property
    def eager_grammar_buffers(self):
        """Eager-fallback grammar buffer handle, or None on the captured path."""
        from tokenspeed.runtime.grammar.capturable_grammar import (
            EagerGrammarBuffers,
        )

        return (
            self.grammar_runtime
            if isinstance(self.grammar_runtime, EagerGrammarBuffers)
            else None
        )

    def _pp_recv_stage_state(self, num_tokens: int):
        """Receive the upstream stage's boundary bundle (mid-pipeline ranks).

        Geometry comes from the model's wire spec — both sides derive it from
        config + token count, so no metadata crosses the wire. Runs on the
        current (execution) stream; NCCL P2P ops on one communicator match in
        issue order against the upstream sends.
        """
        from tokenspeed.runtime.distributed.comm_ops import pp_recv
        from tokenspeed.runtime.distributed.pp_stage import PPStageState

        spec = self.model_runner.model.model.pp_stage_state_spec(
            num_tokens, torch.device(self.device)
        )
        if not self._pp_wire_logged:
            self._pp_wire_logged = True
            logger.info(
                f"PP stage {self.config.pp_rank:d} recv wire: "
                f"{[(tuple(shape), str(dtype)) for _, shape, dtype in spec]!s}",
            )
        tensors = [
            pp_recv(
                shape,
                dtype,
                torch.device(self.device),
                self.config.pp_rank - 1,
                self.config.pp_group,
            )
            for _, shape, dtype in spec
        ]
        return PPStageState.from_tensors(tensors, [name for name, _, _ in spec])

    def _pp_send_stage_state(self, state) -> None:
        """Send this stage's boundary bundle downstream (non-last ranks)."""
        from tokenspeed.runtime.distributed.comm_ops import pp_send

        tensors = state.tensors()
        if not self._pp_wire_logged:
            self._pp_wire_logged = True
            logger.info(
                f"PP stage {self.config.pp_rank:d} send wire: "
                f"{[(tuple(t.shape), str(t.dtype)) for t in tensors]!s}",
            )
        for tensor in tensors:
            pp_send(tensor, self.config.pp_rank + 1, self.config.pp_group)

    @property
    def draft_model_runner(self) -> ModelRunner | None:
        """The speculative draft's runner, or None without speculation.

        Present on every pipeline stage that loaded draft weights, including
        stages whose ``drafter`` is None (the draft only proposes on the last
        stage). Live weight updates read it to refresh the draft in place.
        """
        return self._draft_model_runner

    @property
    def _pp_is_last_stage(self) -> bool:
        return self.config.pp_rank == self.config.pp_size - 1

    @property
    def _pp_is_first_stage(self) -> bool:
        return self.config.pp_rank == 0

    @nvtx_range("target_forward", color="red")
    def _run_target_forward(self, ctx: ForwardContext):
        # The model's rows: the whole packed span, or this rank's shard of it
        # under query context parallelism. Every buffer below is the full span
        # on every rank; the model sees the slice.
        rows = (
            slice(0, ctx.input_num_tokens)
            if ctx.query_shard is None
            else ctx.query_shard.local_slice
        )
        positions = self._active_positions_override
        if positions is None:
            if self.config.model_is_mrope:
                positions = self.input_buffers.mrope_positions_buf[:, rows]
            else:
                positions = self.input_buffers.positions_buf[rows]
        elif ctx.query_shard is not None:
            positions = positions[..., rows]
        input_ids = self.input_buffers.input_ids_buf[rows]
        model_kwargs = self._model_input_kwargs(ctx.input_num_tokens, ctx.bs, rows)
        # PP mid-pipeline: receive the upstream boundary state and thread it
        # through the model's pp_inbound channel. Pipeline parallelism forces
        # eager (ServerArgs.resolve_disaggregation), so neither graph path
        # below can be active alongside PP.
        if self.config.pp_size > 1 and not self._pp_is_first_stage:
            # The boundary bundle carries the rows this stage computes.
            pp_inbound = self._pp_recv_stage_state(rows.stop - rows.start)
            output = self.model_runner.forward(
                ctx,
                input_ids,
                positions,
                pp_inbound=pp_inbound,
                **model_kwargs,
            )
            return output
        # Prefill-graph replay when captured for this forward (the decode graph
        # replays one level up: it captures the whole _forward_step).
        mode = ctx.forward_mode
        if (
            mode is not None
            and (mode.is_extend() or mode.is_mixed())
            and self.prefill_graph.can_run(ctx, self._active_multimodal_context)
        ):
            if ctx.query_shard is not None:
                raise RuntimeError(
                    "a sharded extend cannot replay a prefill graph; query context "
                    "parallelism requires --disable-prefill-graph"
                )
            return self.prefill_graph.replay(
                ctx,
                input_ids,
                self._active_multimodal_context,
            )
        if (
            mode is not None
            and mode.is_extend()
            and self.config.data_parallel_size == 1
        ):
            # The same execution metadata as replay, sized to eager's physical
            # input extent. This neither pads requests nor captures new graphs.
            self.attn_backend.prepare_prefill_metadata(
                ctx.input_num_tokens, ctx.bs, mode, capture=False
            )
        return self.model_runner.forward(
            ctx,
            input_ids,
            positions,
            multimodal_context=self._active_multimodal_context,
            **model_kwargs,
        )

    def _model_input_kwargs(
        self, num_tokens: int, bs: int, rows: slice
    ) -> dict[str, object]:
        """Model inputs beyond ids and positions, as views of persistent buffers.

        Every forward call site passes these, so eager, captured and replayed
        forwards read the same storage. ``rows`` is the slice of the packed
        span the model computes (the whole span, or a query shard): per-row
        inputs are sliced by it, per-request inputs keep the whole batch with
        the slice's start as their row offset.
        """
        # The n-gram history views are per row: hand the model its rows.
        kwargs: dict[str, object] = {
            name: view[rows]
            for name, view in self.input_buffers.ngram_model_kwargs(num_tokens).items()
        }
        if self.runtime_states.has_request_token_history:
            ib = self.input_buffers
            kwargs["request_token_history"] = (
                self.runtime_states.request_token_history_view(
                    req_pool_indices=ib.req_pool_indices_buf[:bs],
                    input_start_offsets=ib.input_start_offsets_buf[: bs + 1],
                    active_request_mask=ib.active_request_mask_buf[:bs],
                    row_offset=rows.start,
                )
            )
        return kwargs

    def _apply_force_single_token_verify(
        self,
        accept_lengths: torch.Tensor,
        row_offset: int,
        row_count: int,
        decode_input_ids: list[int] | None,
    ) -> torch.Tensor:
        if decode_input_ids is None or row_count <= 0:
            return accept_lengths
        force_mask = self.input_buffers.force_single_token_verify_buf[
            row_offset : row_offset + row_count
        ]
        return torch.where(force_mask, torch.ones_like(accept_lengths), accept_lengths)

    def _decode_candidates(self, ctx: ForwardContext) -> torch.Tensor | None:
        """This step's decode-row candidate window: ``[num_decodes, N]``.

        Decode rows' input ids sit at the tail of ``input_ids_buf`` (prefill
        tokens first), N tokens per request: column 0 the last verified
        token, columns 1.. the draft candidates. Non-speculative serving is
        the N == 1 case — a one-column window with nothing to accept, which
        verify() resolves to exactly one sampled token (equivalence pinned
        by test_decode_verify_n1_equivalence.py). A persistent-buffer view,
        so the captured sampler reads live ids on every replay.
        """
        num_decodes = ctx.bs - ctx.num_extends
        if num_decodes == 0:
            return None
        n = self.config.output_length
        num_prefill_tokens = ctx.input_num_tokens - num_decodes * n
        return self.input_buffers.input_ids_buf[
            num_prefill_tokens : ctx.input_num_tokens
        ].reshape(num_decodes, n)

    @nvtx_range("sampling", color="yellow")
    def _run_sampling(
        self,
        logits_output: LogitsProcessorOutput,
        sampling_info: SamplingBatchInfo,
        ctx: ForwardContext,
        candidates: torch.Tensor | None = None,
    ):
        """One sampling rule for every batch: prefill rows sample, decode
        rows verify.

        Non-speculative decode is verify's N == 1 case: the one-column
        candidate window accepts nothing and resolves to exactly one sampled
        token through the same pool kernels sample() uses (equivalence
        pinned by test_decode_verify_n1_equivalence.py)."""
        num_extends = ctx.num_extends
        num_decodes = ctx.bs - num_extends
        layout = ctx.output_layout
        num_prefill_outputs = layout.num_prefill_outputs

        if num_decodes == 0 and num_prefill_outputs == num_extends:
            return self.sampling_backend.sample(logits_output, sampling_info)
        if num_extends == 0:
            output_tokens, accept_lengths = self.sampling_backend.verify(
                logits_output,
                sampling_info,
                candidates,
                tree=(
                    None
                    if self.tree_spec is None
                    else TreeVerifyBatch(
                        parents=self.tree_spec.parent_buf[:num_decodes],
                        depths=self.tree_spec.depth_buf[:num_decodes],
                    )
                ),
            )
            accept_lengths = self._apply_force_single_token_verify(
                accept_lengths, 0, num_decodes, ctx.decode_input_ids
            )
            return output_tokens, accept_lengths

        # Parameters remain request-indexed; logits may omit open prefills.
        prefill = layout.prefill_slice
        decode_requests = layout.decode_request_slice
        decode_outputs = layout.decode_output_slice
        mask_width = (
            sampling_info.vocab_mask.shape[0] // ctx.bs
            if sampling_info.vocab_mask is not None
            else None
        )
        logits = logits_output.next_token_logits
        token_parts, length_parts, logprob_parts = [], [], []

        if num_prefill_outputs:
            prefill_out = LogitsProcessorOutput(next_token_logits=logits[prefill])
            tokens, lengths = self.sampling_backend.sample(
                prefill_out,
                _sampling_info_for_requests(
                    sampling_info, prefill, mask_width=mask_width, prefill=True
                ),
            )
            # verify writes the same backend buffers; snapshot the prefix.
            token_parts.append(tokens.clone() if num_decodes else tokens)
            length_parts.append(lengths.clone() if num_decodes else lengths)
            if prefill_out.next_token_logprobs is not None:
                logprob_parts.append(
                    prefill_out.next_token_logprobs.clone()
                    if num_decodes
                    else prefill_out.next_token_logprobs
                )
        # Empty prefills contribute request lengths, not token storage.
        if num_prefill_outputs < num_extends:
            length_parts.append(
                torch.zeros(
                    num_extends - num_prefill_outputs,
                    dtype=torch.int32,
                    device=logits.device,
                )
            )
        if num_decodes:
            decode_out = LogitsProcessorOutput(next_token_logits=logits[decode_outputs])
            tokens, lengths = self.sampling_backend.verify(
                decode_out,
                _sampling_info_for_requests(
                    sampling_info, decode_requests, mask_width=mask_width, prefill=False
                ),
                candidates,
                tree=None,
            )
            lengths = self._apply_force_single_token_verify(
                lengths, num_extends, num_decodes, ctx.decode_input_ids
            )
            token_parts.append(tokens)
            length_parts.append(lengths)
            if decode_out.next_token_logprobs is not None:
                logprob_parts.append(decode_out.next_token_logprobs)
        if logprob_parts:
            logits_output.next_token_logprobs = (
                logprob_parts[0]
                if len(logprob_parts) == 1
                else torch.cat(logprob_parts)
            )
        if not token_parts:
            token_parts.append(torch.empty(0, dtype=torch.int32, device=logits.device))
        tokens = token_parts[0] if len(token_parts) == 1 else torch.cat(token_parts)
        lengths = length_parts[0] if len(length_parts) == 1 else torch.cat(length_parts)
        return tokens, lengths

    def _log_dp_sampling_route(self, bs: int, ctx: ForwardContext) -> None:
        runtime = self.dp_sampling_runtime_config
        if (
            self.config.global_rank != 0
            or not runtime.enabled
            or runtime.min_bs is None
            or runtime.topology is None
            or ctx.forward_mode is None
            or not ctx.forward_mode.is_decode()
        ):
            return

        use_graph = self.forward_step.can_run(bs=bs, ctx=ctx)
        effective_bs = self.forward_step.padded_bs(bs=bs, ctx=ctx) if use_graph else bs
        tp_size = runtime.topology.tp_size
        bucket_bs = ((effective_bs + tp_size - 1) // tp_size) * tp_size
        dp_sampling = effective_bs >= runtime.min_bs
        route_key = (
            ctx.forward_mode.name,
            bs,
            use_graph,
            effective_bs,
            bucket_bs,
            dp_sampling,
            runtime.min_bs,
        )
        if route_key == self._last_dp_sampling_route_log:
            return
        self._last_dp_sampling_route_log = route_key
        logger.debug(
            f"Batch-DP route: forward_mode={ctx.forward_mode.name.lower()!s} bs={bs:d} "
            f"effective_bs={effective_bs:d} "
            f"use_graph={use_graph!s} bucket_bs={bucket_bs:d} dp_sampling="
            f"{dp_sampling!s} min_bs={runtime.min_bs:d}",
        )

    @maybe_inference_mode()
    def _forward_step(
        self,
        bs: int,
        ctx: ForwardContext,
        sampling_info: SamplingBatchInfo,
    ):
        # Fork grammar onto its side stream so fill + H2D overlap with
        # attention/MoE. Rejoined at wait_bitmask() before apply_mask.
        if self.capturable_grammar is not None:
            n = self.capturable_grammar.max_tokens_per_req
            slice_ = None
            if n > 1 and ctx.output_layout.num_decodes:
                # Verify candidates: the decode rows' tokens at the tail of
                # the live input buffer, after every prefill token.
                count = ctx.output_layout.num_decodes * n
                slice_ = self.input_buffers.input_ids_buf[
                    ctx.input_num_tokens - count : ctx.input_num_tokens
                ]
            self.capturable_grammar.schedule_fill(
                input_ids_buf_slice=slice_, candidate_start=ctx.num_extends
            )

        ctx.dspark_context_producer = self.dspark_context_producer
        if self.drafter is not None:
            self.drafter.prepare_target_forward(ctx)

        logits_output = self._run_target_forward(ctx)

        if self.config.pp_size > 1 and not self._pp_is_last_stage:
            # Mid-pipeline stage: the model returned the boundary bundle, not
            # logits. Ship it downstream and return placeholder outputs — the
            # commit path recognizes a PP placeholder and emits no tokens.
            self._pp_send_stage_state(logits_output)
            output_tokens = torch.zeros(bs, dtype=torch.int32, device=self.device)
            accept_lengths = torch.ones(bs, dtype=torch.int32, device=self.device)
            return output_tokens, accept_lengths, None, None

        # Flag NaN per request and sanitize in place, before any sampling kernel.
        self.nan_guard.audit_logits(logits_output, ctx)
        if logits_output.input_token_logprobs is not None:
            # The prompt rows ship their logprobs as values, so audit those.
            self.nan_guard.audit_input_logprobs(
                logits_output.input_token_logprobs,
                ctx.input_logprob_rows.slots,
                ctx.num_extends,
            )

        candidates = self._decode_candidates(ctx)

        if self.capturable_grammar is not None:
            self.capturable_grammar.wait_bitmask()

        output_tokens, accept_lengths = self._run_sampling(
            logits_output, sampling_info, ctx, candidates
        )

        # Backstop: flag any request whose sampled id falls outside [0, vocab)
        # so the output processor can terminate it. Covers sampler/verify kernel
        # corruption and DP-sharded steps that audit_logits cannot attribute.
        self.nan_guard.merge_oov(output_tokens, ctx, self.runtime_states.vocab_size)

        # Fork sampler-output D2H onto the grammar side stream so the
        # next step's build hostfunc can advance the matcher.
        if self.capturable_grammar is not None:
            self.capturable_grammar.schedule_post_sampler(output_tokens, accept_lengths)

        if self.tree_spec is not None and ctx.num_extends == 0:
            self._compact_accepted_tree(ctx.bs, logits_output)

        if self.drafter is not None:
            next_round_input_ids = self.drafter.run(
                base_ctx=ctx,
                logits_output=logits_output,
                output_tokens=output_tokens,
                accept_lengths=accept_lengths,
            )
            # _update_runtime_state skips future_input_map when drafter is
            # active — drafter writes the next-round inputs directly.
            indices = self.input_buffers.state_write_req_pool_indices_buf[: ctx.bs]
            for requests in (
                ctx.output_layout.prefill_slice,
                ctx.output_layout.decode_request_slice,
            ):
                if requests.start == requests.stop:
                    continue
                self.runtime_states.future_input_map[indices[requests]] = (
                    next_round_input_ids[requests].to(torch.int32)
                )
                if self.tree_spec is not None:
                    self.runtime_states.future_parent_map[indices[requests]] = (
                        self.tree_spec.draft_parent_buf[requests]
                    )
            self._record_draft_final_cache_step(ctx.num_extends)

        output_logprobs = logits_output.next_token_logprobs
        return (
            output_tokens,
            accept_lengths,
            output_logprobs,
            logits_output.input_token_logprobs,
        )

    @nvtx_range("update_runtime_state", color="orange")
    def _update_runtime_state(
        self,
        req_pool_indices: torch.Tensor,
        output_tokens: torch.Tensor,
        accept_lengths: torch.Tensor,
        input_lengths: torch.Tensor,
        num_extends: int,
        *,
        output_layout: ForwardOutputLayout,
    ):
        """Advance accepted inputs and cache lengths together on execution_stream.

        Serving calls this after eager execution or graph replay. All writes
        are tensor-only, including padding masks, so recording this update has
        the same semantics. Callers must pass the state-write pool indices.
        """
        if self.drafter is None:
            prefill = output_layout.prefill_slice
            if output_layout.num_prefill_outputs:
                self.runtime_states.future_input_map[req_pool_indices[prefill], :1] = (
                    output_tokens[prefill, None].to(torch.int32)
                )
            if output_layout.num_decodes:
                requests = output_layout.decode_request_slice
                self.runtime_states.future_input_map[
                    req_pool_indices[requests], : output_layout.decode_width
                ] = (
                    output_tokens[output_layout.decode_output_slice]
                    .view(output_layout.num_decodes, output_layout.decode_width)
                    .to(torch.int32)
                )

        ib = self.input_buffers
        tail = self.runtime_states.ngram_accepted_tokens
        advance_accepted_frontier(
            req_pool_indices,
            input_lengths,
            accept_lengths,
            self.runtime_states.valid_cache_lengths,
            num_extends,
            ib.state_write_padding_pool_index,
            ngram_tail=tail,
            ngram_previous_tokens=ib.ngram_previous_tokens_buf,
            ngram_token_mask=ib.ngram_token_mask_buf,
            input_ids=ib.input_ids_buf if tail is not None else None,
        )

    def _build_sampling_info(self, bs: int) -> SamplingBatchInfo:
        return SamplingBatchInfo(
            req_pool_indices=self.input_buffers.req_pool_indices_buf[:bs],
            valid_cache_lengths=self.runtime_states.valid_cache_lengths,
            draft_probs=self.runtime_states.draft_probs,
            vocab_size=self.runtime_states.vocab_size,
            device=self.device,
        )

    def execute_idle_forward(self, dp_metadata: DpForwardMetadata):
        """Run a zero-token forward so this rank participates in NCCL collectives.

        Called by the EventLoop when this DP rank has no work but other
        ranks do. The MoE all-to-all is a collective that requires ALL
        ranks to participate.
        """
        graph_forward_mode = ForwardMode.DECODE
        ctx = ForwardContext(
            attn_backend=self.attn_backend,
            token_to_kv_pool=self.token_to_kv_pool,
            bs=0,
            num_extends=0,
            output_layout=ForwardOutputLayout(0, 0, 0, 1),
            input_num_tokens=0,
            forward_mode=graph_forward_mode,
            global_num_tokens=dp_metadata.global_num_tokens,
            global_bs=dp_metadata.global_batch_size,
            all_decode_or_idle=dp_metadata.all_decode_or_idle,
        )
        sampling_info = SamplingBatchInfo(
            req_pool_indices=self.input_buffers.req_pool_indices_buf[:0],
            valid_cache_lengths=self.runtime_states.valid_cache_lengths,
            vocab_size=self.runtime_states.vocab_size,
            device=self.device,
        )
        if self.forward_step.can_run(bs=0, ctx=ctx):
            padded_bs = self.forward_step.padded_bs(bs=0, ctx=ctx)
            self.input_buffers.fill_dummy_decode_buffers(
                batch_size=padded_bs,
                total_tokens=padded_bs * self.config.output_length,
            )
            # Captured hostfunc pops one entry per replay; push a dummy
            # for this idle replay, same as run_once.
            if self.capturable_grammar is not None:
                self.capturable_grammar.add_batch(
                    grammars=[None] * padded_bs,
                    bs=padded_bs,
                    has_candidates=False,
                    output_layout=ForwardOutputLayout(
                        0, 0, padded_bs, self.config.output_length
                    ),
                )
            # IDLE doesn't produce tokens, so no sampler/drafter call here —
            # only the model forward, which still participates in collectives.
            # The draft router's idle refresh zeroes its history stack rows,
            # so the captured drafter steps' KV writes land on the dummy
            # page (#955's aliasing hazard is handled at the table source).
            ib = self.input_buffers
            with nvtx_range("forward_step idle", color="blue"):
                self.forward_step(
                    bs=0,
                    ctx=ctx,
                    sampling_info=sampling_info,
                    extend_with_prefix=False,
                    extend_prefix_lens=ib.extend_prefix_lens_buf[:0],
                    extend_prefix_lens_cpu=ib.extend_prefix_lens_cpu[:0],
                    extend_seq_lens=ib.extend_seq_lens_buf[:0],
                    extend_seq_lens_cpu=ib.extend_seq_lens_cpu[:0],
                    extend_replay_lens_cpu=ib.extend_replay_lens_cpu[:0],
                    extend_prompt_lens_cpu=ib.extend_prompt_lens_cpu[:0],
                    # No request, so no group tables on either side.
                    block_tables_cpu={},
                )
            return

        # Run model forward with IDLE mode — skips attention but still
        # participates in MLP NCCL collectives (dense all-gather, MoE).
        ctx.forward_mode = ForwardMode.IDLE
        empty = torch.zeros(0, dtype=torch.int32, device=self.device)
        self.model_runner.forward(
            ctx,
            input_ids=empty,
            positions=empty,
            **self._model_input_kwargs(0, 0, slice(0, 0)),
        )

        # If a drafter is active, its model also has MoE layers that issue
        # NCCL collectives. Idle ranks must match those collectives: the
        # drafter lists the draft forwards the active ranks run per round,
        # each as the per-rank token counts sizing its collectives
        # (idle_forward_global_num_tokens); every step runs the IDLE forward
        # over an empty window with its own spec_step_idx.
        if self.drafter is not None:
            # A draft model that reads request-token history takes the view
            # on every forward; the idle rank hands it an empty one, as the
            # target's idle forward above does.
            draft_kwargs: dict[str, object] = {}
            if (
                self.drafter.draft_model_runner.model_config.requires_request_token_history
            ):
                ib = self.input_buffers
                draft_kwargs["request_token_history"] = (
                    self.runtime_states.draft_request_token_history_view(
                        req_pool_indices=ib.req_pool_indices_buf[:0],
                        input_start_offsets=ib.input_start_offsets_buf[:1],
                        active_request_mask=ib.active_request_mask_buf[:0],
                        committed_lengths=self.runtime_states.valid_cache_lengths,
                        row_offset=0,
                    )
                )
            step_global_num_tokens = self.drafter.idle_forward_global_num_tokens(
                dp_metadata.global_num_tokens, dp_metadata.global_batch_size
            )
            for step_idx, draft_global_num_tokens in enumerate(step_global_num_tokens):
                draft_ctx = ForwardContext(
                    attn_backend=self.drafter.attn_backend,
                    token_to_kv_pool=self.drafter.token_to_kv_pool,
                    bs=0,
                    num_extends=0,
                    output_layout=ForwardOutputLayout(0, 0, 0, 1),
                    input_num_tokens=0,
                    forward_mode=ForwardMode.IDLE,
                    global_num_tokens=draft_global_num_tokens,
                    global_bs=dp_metadata.global_batch_size,
                    all_decode_or_idle=dp_metadata.all_decode_or_idle,
                )
                self.drafter.draft_model_runner.forward(
                    draft_ctx,
                    input_ids=empty,
                    positions=empty,
                    spec_step_idx=step_idx,
                    **draft_kwargs,
                )

    def zero_cache_pages(self, pages: Mapping[str, Sequence[int]] | Sequence[int]):
        """Clear newly owned pages and return a CUDA completion event when needed.

        Runs on ``default_stream``, ordered behind the forwards in flight on
        ``execution_stream``: the pages' previous owner may still be writing
        them. The plan's later work on the default stream (the load-backs'
        start event, the remote prefill's fence) inherits the order; the next
        forward's prologue waits on the default stream and so runs after the
        zeroing.
        """
        if not pages:
            return None
        self.default_stream.wait_stream(self.execution_stream)

        if isinstance(pages, Mapping):
            # Group-keyed requests carry scheduler (virtual) IDs; pools and the
            # arena only ever see this rank's local pages.
            pages = local_pages_by_group(
                pages,
                contract=self._cache_runtime_contract,
                rank=self._cache_dcp_rank,
            )

        def sanitize(pool, pool_pages) -> bool:
            zero_new_blocks = getattr(pool, "zero_new_blocks", None)
            zero_pages = getattr(pool, "zero_pages", None)
            if isinstance(pool_pages, Mapping) and callable(zero_new_blocks):
                zero_new_blocks(pool_pages)
                return True
            if callable(zero_pages):
                page_ids = (
                    sorted(
                        {
                            int(page_id)
                            for group_pages in pool_pages.values()
                            for page_id in group_pages
                        }
                    )
                    if isinstance(pool_pages, Mapping)
                    else pool_pages
                )
                zero_pages(page_ids)
                return True
            if getattr(pool, "requires_page_zeroing", False):
                raise RuntimeError(
                    "scheduler emitted pages to zero but an active KV "
                    "pool does not implement physical-page sanitization"
                )
            return False

        with nvtx_range("zero_cache_pages", color="purple"):
            sanitized = sanitize(self.token_to_kv_pool, pages)
            draft_pool = self.draft_token_to_kv_pool
            if draft_pool is not None and getattr(
                draft_pool,
                "requires_page_zeroing",
                False,
            ):
                draft_pages = pages
                if isinstance(pages, Mapping):
                    draft_group_ids = {
                        str(spec.group_id)
                        for spec in _cache_arena_attr(
                            draft_pool, "cache_group_specs", ()
                        )
                    }
                    draft_pages = {
                        group_id: page_ids
                        for group_id, page_ids in pages.items()
                        if group_id in draft_group_ids
                    }
                if draft_pages:
                    sanitized = sanitize(draft_pool, draft_pages) or sanitized
        if not sanitized:
            return None
        if torch.device(self.device).type not in {"cuda", "npu"}:
            return None
        done = self.device_module.Event()
        done.record(self.default_stream)
        return done

    @nvtx_range("reset_valid_cache_length", color="orange")
    def _reset_valid_cache_length(self, forward_op) -> None:
        """Rewind the prefill rows' valid cache lengths before a forward.

        A forward's prologue, not a caller step: ``execute_forward_op`` runs
        it on the forward thread so the state writes land on the execution
        stream ahead of the model launches.
        """
        num_extends = forward_op.num_extends()
        if num_extends == 0:
            return
        self._write_valid_cache_lengths(
            forward_op.request_pool_indices[:num_extends],
            forward_op.extend_prefix_lens,
        )

    @nvtx_range("reset_remote_prefill_cache_lengths", color="orange")
    def reset_remote_prefill_cache_lengths(self, forward_op) -> None:
        """Seed rows whose prompt was computed on another node.

        A PD decode destination never executes the prompt locally, so no
        forward of its own can establish these lengths — they come from the
        complete remotely-computed prompt instead, before the first local
        decode. The cache-transfer path additionally selects the transferred
        recurrent-state snapshot block from the resulting sequence length.
        """
        num_extends = forward_op.num_extends()
        if num_extends <= 0:
            return
        self._write_valid_cache_lengths(
            forward_op.request_pool_indices[:num_extends],
            forward_op.prefill_lengths[:num_extends],
        )

    def _write_valid_cache_lengths(self, pool_indices, lengths) -> None:
        """Publish per-row valid cache lengths on the execution stream."""
        self.execution_stream.wait_stream(self.default_stream)
        with self.device_module.stream(self.execution_stream):
            rows = torch.tensor(
                pool_indices,
                dtype=torch.int64,
                device="cpu",
                pin_memory=True,
            ).to(self.device, non_blocking=True)
            values = torch.tensor(
                lengths,
                dtype=torch.int32,
                device="cpu",
                pin_memory=True,
            ).to(self.device, non_blocking=True)
            self.runtime_states.reset_states(rows, values)

    def execute_forward_op(
        self,
        forward_op,
        sampling_params_list: list[SamplingParams],
        dp_metadata: DpForwardMetadata | None = None,
        grammar_inputs=None,
        multimodal_context=None,
        capture_next_input_ids: bool = False,
        *,
        ngram_inputs: NGramInputs | None,
        request_history_seeds: RequestHistorySeeds | None,
        input_logprob_plan: InputLogprobPlan | None,
    ) -> ModelExecutionResult:
        self._reset_valid_cache_length(forward_op)
        self.log_step += 1
        num_extends = forward_op.num_extends()
        total_tokens = sum(forward_op.input_lengths)
        self._active_multimodal_context = multimodal_context
        self._active_positions_override = None
        timing_enabled = LOG_MM_TIMING
        timing_start = time.perf_counter() if timing_enabled else 0.0
        input_fill_ms = 0.0
        mrope_ms = 0.0
        sampling_prep_ms = 0.0
        forward_step_ms = 0.0
        output_d2h_ms = 0.0
        graph_capable = False
        graph_padded_bs = 0

        with nvtx_range("pre_fill_setup", color="orange"):
            # Behind the default-stream work the plan enqueued ahead of this
            # forward: the page zeroing, the retraction write-back's fence,
            # the multimodal features. The runtime-state reads below need no
            # cross-stream wait -- their writers ran on execution_stream too.
            self.execution_stream.wait_stream(self.default_stream)
        with self.device_module.stream(self.execution_stream):
            bs = len(forward_op.request_ids)
            # Outside the graph: in-graph sites only OR into the flag buffer.
            self.nan_guard.reset(bs)
            cache_metadata = None
            block_tables = {}
            block_tables_cpu = {}
            if bs > 0:
                # Validate and pack the per-group tables once for this batch.
                cache_metadata = CacheBatchMetadata.from_forward_op(
                    forward_op,
                    device=self.device,
                    contract=self._cache_runtime_contract,
                    num_requests=bs,
                )
                block_tables = dict(cache_metadata.tables(active_forward_op=forward_op))
                block_tables_cpu = dict(
                    cache_metadata.tables_cpu(active_forward_op=forward_op)
                )
            decode_input_ids = self.input_buffers.fill_input_buffers(
                forward_op=forward_op,
                runtime_states=self.runtime_states,
                total_tokens=total_tokens,
                ngram_inputs=ngram_inputs,
            )
            if self.tree_spec is not None and num_extends == 0 and bs > 0:
                self.tree_spec.load_step(
                    bs,
                    self.input_buffers.req_pool_indices_buf[:bs],
                    self.runtime_states.future_parent_map,
                )
                self.tree_spec.depth_positions(
                    bs, self.input_buffers.positions_buf[:total_tokens]
                )
            if request_history_seeds is not None:
                self.runtime_states.seed_request_token_history(request_history_seeds)
            if self.drafter is not None and hasattr(
                self.drafter, "prepare_request_state"
            ):
                self.drafter.prepare_request_state(
                    forward_op.request_ids,
                    forward_op.request_pool_indices,
                    num_extends,
                )
            if timing_enabled:
                input_fill_done = time.perf_counter()
                input_fill_ms = (input_fill_done - timing_start) * 1000.0
            mrope_start = time.perf_counter() if timing_enabled else 0.0
            self._active_positions_override = self.mm_runtime.build_positions_override(
                forward_op=forward_op,
                multimodal_context=multimodal_context,
                total_tokens=total_tokens,
            )
            if timing_enabled:
                mrope_ms = (time.perf_counter() - mrope_start) * 1000.0

            forward_mode = ForwardMode.from_num_extends(num_extends, bs)

            if num_extends <= 0:
                self._prev_decode_bs = bs

            grammar_completion = None

            input_token_logprobs = None
            if total_tokens == 0:
                # Fully prefix-cached prefill: no tokens to process.
                output_tokens = torch.zeros(0, dtype=torch.int32, device=self.device)
                output_lengths = torch.zeros(bs, dtype=torch.int32, device=self.device)
                output_logprobs = None
                if input_logprob_plan is not None:
                    raise RuntimeError(
                        "prompt logprobs planned for a forward without input rows"
                    )
            else:
                gather_ids = None
                if num_extends > 0:
                    num_decodes = bs - num_extends
                    if self.drafter is not None and num_decodes > 0:
                        # MIXED + spec: prefill rows pruned to last token,
                        # decode block kept full at verify width.
                        num_decode_tokens = num_decodes * self.config.spec_num_tokens
                        num_prefill_tokens = total_tokens - num_decode_tokens
                        gather_ids = torch.empty(
                            num_extends + num_decode_tokens,
                            dtype=torch.int64,
                            device=self.device,
                        )
                        gather_ids[:num_extends] = (
                            torch.cumsum(
                                self.input_buffers.input_lengths_buf[:num_extends],
                                dim=0,
                            )
                            - 1
                        )
                        gather_ids[num_extends:] = torch.arange(
                            num_prefill_tokens,
                            total_tokens,
                            device=self.device,
                            dtype=torch.int64,
                        )
                    else:
                        # EXTEND, MIXED non-spec, or EXTEND + spec: last token
                        # per request via cumsum.
                        gather_ids = (
                            torch.cumsum(
                                self.input_buffers.input_lengths_buf[:bs], dim=0
                            )
                            - 1
                        )

                output_layout = ForwardOutputLayout(
                    num_extends=num_extends,
                    num_prefill_outputs=num_extends,
                    num_decodes=bs - num_extends,
                    decode_width=self.config.output_length,
                )
                if num_extends and self.attn_backend.skips_incomplete_prefill_outputs:
                    output_layout = ForwardOutputLayout.from_prefill(
                        prefix_lengths=forward_op.extend_prefix_lens,
                        input_lengths=forward_op.input_lengths[:num_extends],
                        prompt_lengths=forward_op.prefill_lengths[:num_extends],
                        num_decodes=bs - num_extends,
                        decode_width=self.config.output_length,
                    )
                query_shard = None
                if self.config.query_shard_size > 1:
                    # The shard splits the packed extend span; host integers
                    # from the same lengths gather_ids come from. The prefill
                    # role never carries decode rows, so num_extends == bs.
                    if num_extends != bs:
                        raise RuntimeError(
                            "query context parallelism shards pure extend "
                            f"forwards; got {bs - num_extends} decode requests"
                        )
                    query_shard = QueryShardPlan.from_forward(
                        total_tokens=total_tokens,
                        input_lengths=forward_op.input_lengths[:bs],
                        size=self.config.query_shard_size,
                        rank=self.config.query_shard_rank,
                    )
                ctx = ForwardContext(
                    attn_backend=self.attn_backend,
                    token_to_kv_pool=self.token_to_kv_pool,
                    bs=bs,
                    num_extends=num_extends,
                    input_num_tokens=total_tokens,
                    forward_mode=forward_mode,
                    capture_hidden_mode=(
                        CaptureHiddenMode.FULL
                        if self.drafter is not None
                        and self.dspark_context_producer is None
                        else CaptureHiddenMode.NULL
                    ),
                    gather_ids=gather_ids,
                    input_logprob_rows=self._input_logprob_rows(
                        input_logprob_plan, num_extends, total_tokens, query_shard
                    ),
                    decode_input_ids=decode_input_ids,
                    output_layout=output_layout,
                    query_shard=query_shard,
                )
                if self.config.data_parallel_size > 1:
                    if dp_metadata is None:
                        raise RuntimeError(
                            "DP forward metadata must be gathered on CPU by "
                            "the event loop before model execution."
                        )
                    ctx.global_num_tokens = dp_metadata.global_num_tokens
                    ctx.global_bs = dp_metadata.global_batch_size
                    ctx.all_decode_or_idle = dp_metadata.all_decode_or_idle
                    ctx.all_extend = dp_metadata.all_extend
                with nvtx_range("sampling_prep", color="yellow"):
                    sampling_start = time.perf_counter() if timing_enabled else 0.0
                    sampling_info = self._build_sampling_info(bs)
                    grammar_completion = setup_grammar_step(
                        sampling_info=sampling_info,
                        bs=bs,
                        is_spec_decode=self.drafter is not None and num_extends < bs,
                        spec_num_tokens=self.config.spec_num_tokens or 1,
                        grammar_inputs=grammar_inputs,
                        grammar_runtime=self.grammar_runtime,
                        input_ids_buf=self.input_buffers.input_ids_buf[:total_tokens],
                        grammar_backend=self.config.grammar_backend,
                        output_layout=output_layout,
                    )
                    extend_with_prefix = num_extends > 0 and any(
                        forward_op.extend_prefix_lens
                    )
                    # Flip detection + per-slot scalar scatter + backend-owned
                    # RNG state refill. Runs OUTSIDE the CUDA graph. Generators
                    # are now backend-internal (pool-indexed, seeded on flip
                    # from sp.seed), so the event loop no longer threads them
                    # through.
                    self.sampling_backend.prepare_step(
                        request_ids=forward_op.request_ids,
                        request_pool_indices=forward_op.request_pool_indices,
                        sampling_params_list=sampling_params_list,
                        num_tokens_per_req=self.config.output_length,
                    )
                    if timing_enabled:
                        sampling_prep_ms = (
                            time.perf_counter() - sampling_start
                        ) * 1000.0

                with nvtx_range(
                    f"forward_step ext={num_extends} dec={bs - num_extends}",
                    color="blue",
                ):
                    self._log_dp_sampling_route(bs, ctx)
                    forward_step_start = 0.0
                    if timing_enabled:
                        graph_capable = self.forward_step.can_run(bs, ctx)
                        graph_padded_bs = (
                            self.forward_step.padded_bs(bs, ctx)
                            if graph_capable
                            else bs
                        )
                        forward_step_start = time.perf_counter()
                    (
                        output_tokens,
                        output_lengths,
                        output_logprobs,
                        input_token_logprobs,
                    ) = self.forward_step(
                        bs=bs,
                        ctx=ctx,
                        sampling_info=sampling_info,
                        extend_with_prefix=extend_with_prefix,
                        extend_prefix_lens=self.input_buffers.extend_prefix_lens_buf[
                            :num_extends
                        ],
                        extend_prefix_lens_cpu=self.input_buffers.extend_prefix_lens_cpu[
                            :num_extends
                        ],
                        extend_seq_lens=self.input_buffers.extend_seq_lens_buf[
                            :num_extends
                        ],
                        extend_seq_lens_cpu=self.input_buffers.extend_seq_lens_cpu[
                            :num_extends
                        ],
                        extend_replay_lens_cpu=self.input_buffers.extend_replay_lens_cpu[
                            :num_extends
                        ],
                        extend_prompt_lens_cpu=self.input_buffers.extend_prompt_lens_cpu[
                            :num_extends
                        ],
                        block_tables=block_tables,
                        block_tables_cpu=block_tables_cpu,
                    )
                    if timing_enabled:
                        forward_step_ms = (
                            time.perf_counter() - forward_step_start
                        ) * 1000.0

                # Update runtime state on execution_stream (NOT in the CUDA graph).
                self._update_runtime_state(
                    req_pool_indices=self.input_buffers.state_write_req_pool_indices_buf[
                        :bs
                    ],
                    output_tokens=output_tokens,
                    accept_lengths=output_lengths,
                    input_lengths=self.input_buffers.input_lengths_buf[:bs],
                    num_extends=num_extends,
                    output_layout=ctx.output_layout,
                )
            with nvtx_range("output_d2h", color="green"):
                output_d2h_start = time.perf_counter() if timing_enabled else 0.0
                next_input_ids = None
                spec_candidate_tokens = None
                if (
                    capture_next_input_ids
                    and self.drafter is not None
                    and num_extends > 0
                ):
                    next_input_ids = self.runtime_states.future_input_map.index_select(
                        0, self.input_buffers.req_pool_indices_buf[:num_extends]
                    ).to("cpu", non_blocking=True)

                # The candidate-vs-target compare reads the window as a chain.
                if (
                    LOG_SPEC_ACCEPT_LENGTHS
                    and self.config.spec_algo is not None
                    and num_extends == 0
                    and self.tree_spec is None
                ):
                    spec_candidate_tokens = self.input_buffers.input_ids_buf[
                        : bs * self.config.spec_num_tokens
                    ].to("cpu", non_blocking=True)

                # Defensive clamp into the valid vocab range (kept from the
                # pre-pack path). An out-of-range token id -- e.g. a stale/corrupt
                # value surfaced by the intermittent spec-decode decode-state race
                # -- would otherwise reach the detokenizer, whose HF
                # tokenizer.decode raises a fatal OverflowError on ids outside
                # [0, vocab) and tears down the whole server process tree.
                # It must run on-GPU *before* the non_blocking D2H: clamping the
                # CPU result afterwards would race the in-flight copy. In-place
                # (clamp_) so output_tokens keeps aliasing _output_pack_buf and
                # the get_packed_output_d2h data_ptr fast-path still fires -- and
                # in-place on the forward's inference tensors is only legal inside
                # inference mode, so re-enter it (maybe_inference_mode mirrors the
                # forward and reduces to no_grad when inference mode is disabled,
                # where output_tokens isn't an inference tensor anyway).
                vocab_size = self.runtime_states.vocab_size
                with maybe_inference_mode():
                    output_tokens.clamp_(0, vocab_size - 1)

                packed = self.sampling_backend.get_packed_output_d2h(
                    output_tokens, output_lengths
                )
                if packed is not None:
                    output_tokens, output_lengths = packed
                else:
                    output_tokens = output_tokens.to("cpu", non_blocking=True)
                    output_lengths = output_lengths.to("cpu", non_blocking=True)

                if output_logprobs is not None:
                    output_logprobs = output_logprobs.to("cpu", non_blocking=True)
                if input_token_logprobs is not None:
                    input_token_logprobs = input_token_logprobs.to(
                        "cpu", non_blocking=True
                    )

                output_nan_flags = self.nan_guard.flags_cpu

                copy_event = self.device_module.Event()
                copy_event.record()
                if timing_enabled:
                    output_d2h_ms = (time.perf_counter() - output_d2h_start) * 1000.0

            if timing_enabled and (
                num_extends > 0 or self.log_step < 64 or self.log_step % 100 == 0
            ):
                has_mm, mm_count, mm_delta_count = MultimodalRuntime.timing_counts(
                    multimodal_context
                )
                logger.info(
                    "mm_timing forward_execute_ms total="
                    f"{(time.perf_counter() - timing_start) * 1000.0:.3f} input_fill="
                    f"{input_fill_ms:.3f} "
                    f"mrope={mrope_ms:.3f} sampling={sampling_prep_ms:.3f} "
                    f"forward_step={forward_step_ms:.3f} output_d2h={output_d2h_ms:.3f}"
                    " "
                    f"mode={forward_mode.name!s} bs={bs!s} total_tokens="
                    f"{total_tokens!s} graph={graph_capable!s} padded_bs="
                    f"{graph_padded_bs!s} "
                    f"has_mm={has_mm!s} mm_count={mm_count!s} mm_delta_count="
                    f"{mm_delta_count!s}",
                )

        return ModelExecutionResult(
            output_tokens=output_tokens,
            output_lengths=output_lengths,
            output_logprobs=output_logprobs,
            copy_event=copy_event,
            grammar_completion=grammar_completion,
            next_input_ids=next_input_ids,
            output_nan_flags=output_nan_flags,
            spec_candidate_tokens=spec_candidate_tokens,
            input_token_logprobs=input_token_logprobs,
            # The plan rides along whether or not this rank scored the rows: a
            # pipeline stage without logits adopts the last stage's logprobs on
            # the commit path and pairs them with its own (mirrored) plan.
            input_logprob_plan=input_logprob_plan,
        )

    def _input_logprob_rows(
        self,
        plan: InputLogprobPlan | None,
        num_extends: int,
        total_tokens: int,
        query_shard: QueryShardPlan | None,
    ) -> InputLogprobRows | None:
        """Expand the plan into device rows, targets and slots for the logits processor.

        The per-slot triples become the flat row index (an ``arange`` per
        slot) and the slot of every row on the host, staged pinned and copied
        non-blocking like the other per-forward inputs: the forward thread
        never synchronizes on its per-round path. Each row's target is the next
        prompt token, read from the scheduler's shifted input ids that
        ``fill_input_buffers`` landed for this prefill (they cover the chunk
        boundary). A target outside the vocabulary flags its request through
        the NaN guard, which terminates it; the clamp only keeps the gather
        from faulting on a flagged row.

        Under a query shard every rank stages the whole plan's targets and
        slots (the shifted ids are the whole span on every rank; every rank
        scores every row once the planned activations are gathered, and the
        target audit flags the same requests everywhere) and keeps as its
        ``rows`` the ones inside its shard, re-based to it: the plan's rows
        are sorted batch-global rows, so each rank's are one contiguous run
        and the per-rank counts are host arithmetic over the shard boundaries
        (``QueryShardPlan.rows_per_rank``).
        """
        if plan is None:
            return None
        if max(map(sum, zip(plan.row_starts, plan.counts))) > total_tokens:
            raise RuntimeError("input logprob plan names rows past the forward's input")
        counts = torch.tensor(plan.counts, dtype=torch.int64)
        slots_cpu = torch.repeat_interleave(torch.arange(len(plan.counts)), counts)
        first_row = torch.tensor(plan.row_starts, dtype=torch.int64) - (
            torch.cumsum(counts, dim=0) - counts
        )
        rows_cpu = torch.arange(plan.num_rows, dtype=torch.int64) + first_row[slots_cpu]
        staged = torch.stack((rows_cpu, slots_cpu))
        if is_pin_memory_available():
            staged = staged.pin_memory()
        rows, slots = staged.to(self.device, non_blocking=True)
        targets = self.input_buffers.shifted_prefill_ids_buf[rows].to(torch.int64)
        self.nan_guard.audit_input_logprob_targets(
            targets, slots, num_extends, self.runtime_states.vocab_size
        )
        targets.clamp_(0, self.runtime_states.vocab_size - 1)
        rows_per_rank = None
        num_input_rows = total_tokens
        if query_shard is not None and query_shard.size > 1:
            rows_per_rank = query_shard.rows_per_rank(rows_cpu)
            local = query_shard.local_rows_run(rows_per_rank)
            rows = rows[local] - query_shard.local_start
            num_input_rows = query_shard.local_rows
        return InputLogprobRows(
            rows=rows,
            targets=targets,
            slots=slots,
            num_input_rows=num_input_rows,
            chunk_tokens=self.config.input_logprob_chunk_tokens,
            rows_per_rank=rows_per_rank,
        )

    def write_remote_spec_candidate_ids(
        self, req_pool_idx: int, candidate_ids: list[int]
    ) -> None:
        # Remote spec candidates are CPU materialized; enqueue the H2D copy and
        # future_input_map update on execution_stream. The next forward's input
        # prep already waits on execution_stream before reading runtime state.
        with self.device_module.stream(self.execution_stream):
            self.runtime_states.write_remote_spec_candidate_ids(
                req_pool_idx, candidate_ids
            )

    @property
    def draft_field_writer(self):
        """Whoever writes this rank's draft cache fields, or None.

        The context producer when configured (pipeline DSpark), else the
        drafter. Its ``supports_pd_layerwise_finalization`` says whether the
        rank can finalize layerwise CachePD writes with speculation on; this
        is the one place that choice is made.
        """
        if self.dspark_context_producer is not None:
            return self.dspark_context_producer
        return self.drafter

    def register_draft_final_step_counter(self, step_counter) -> None:
        """Publish one CachePD step after a supported drafter's complete run."""
        writer = self.draft_field_writer
        if writer is None or not writer.supports_pd_layerwise_finalization:
            raise RuntimeError(
                "the speculative drafter cannot finalize layerwise CachePD writes"
            )
        if self.draft_attn_backend is None:
            raise RuntimeError("draft-final CachePD readiness requires a draft backend")
        if self.draft_attn_backend is self.attn_backend:
            raise RuntimeError(
                "draft-final CachePD readiness requires distinct target and draft backends"
            )
        self._draft_final_step_counter = step_counter

    def _record_draft_final_cache_step(self, num_extends: int) -> None:
        step_counter = self._draft_final_step_counter
        if step_counter is not None and num_extends > 0:
            step_counter.record_cache()

    def prepare_remote_cache_slots(self, req_pool_indices: list[int]) -> None:
        """Clear backend restore state before publishing RDMA destinations."""
        slots = [int(slot) for slot in req_pool_indices]
        with self.device_module.stream(self.execution_stream):
            self.attn_backend.prepare_remote_cache_slots(slots)

    def mark_remote_cache_ready(self, req_pool_idx: int) -> None:
        """Arm backend first-decode hydration after remote transfer success."""
        with self.device_module.stream(self.execution_stream):
            self.attn_backend.mark_remote_cache_ready(int(req_pool_idx))
