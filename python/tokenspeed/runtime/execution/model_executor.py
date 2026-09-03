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

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from tokenspeed_kernel.ops.tuning import (
    autotune,
    set_autotune_max_num_tokens,
    set_autotune_process_group,
)
from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.configs.model_config import AttentionArch, ModelConfig
from tokenspeed.runtime.configs.utils import get_rope_parameters
from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)
from tokenspeed.runtime.execution.breakable_cuda_graph import active_forward
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.cuda_graph_wrapper import CudaGraphWrapper
from tokenspeed.runtime.execution.draft_page_staging import DraftPageStaging
from tokenspeed.runtime.execution.drafter import get_drafter_impl
from tokenspeed.runtime.execution.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
)
from tokenspeed.runtime.execution.forward_thread import ForwardThread
from tokenspeed.runtime.execution.input_buffer import InputBuffers
from tokenspeed.runtime.execution.model_runner import ModelRunner
from tokenspeed.runtime.execution.multimodal_runtime import MultimodalRuntime
from tokenspeed.runtime.execution.nan_guard import NanGuard
from tokenspeed.runtime.execution.prefill_graph import PrefillGraph
from tokenspeed.runtime.execution.runtime_states import RuntimeStates
from tokenspeed.runtime.execution.types import (
    DpForwardMetadata,
    ModelExecutionResult,
)
from tokenspeed.runtime.execution.workspace import workspace_pool
from tokenspeed.runtime.grammar.capturable_grammar import (
    create_grammar_runtime,
    setup_grammar_step,
)
from tokenspeed.runtime.layers.attention.backends.cache_metadata import (
    CacheBatchMetadata,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    validate_scheduler_config,
)
from tokenspeed.runtime.layers.logits_processor import LogitsProcessorOutput
from tokenspeed.runtime.layers.paged_attention import (
    validate_cache_group_ids,
)
from tokenspeed.runtime.sampling.backends.base import SamplingBackend
from tokenspeed.runtime.sampling.dp_sampling_config import (
    DpSamplingRuntimeLimits,
    setup_dp_sampling,
)
from tokenspeed.runtime.sampling.sampling_batch_info import SamplingBatchInfo
from tokenspeed.runtime.utils import get_colorful_logger, set_random_seed
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


def _draft_idle_global_num_tokens_for_step(
    step_idx: int,
    global_num_tokens: list[int],
    global_bs: list[int] | None,
) -> list[int]:
    if step_idx == 0 or global_bs is None:
        return global_num_tokens
    return global_bs


PREFILL_GRAPH_DEFAULT_MAX_TOKENS = 2048


def _resolve_prefill_graph_max_tokens(server_args) -> int:
    """Largest prefill-graph bucket: explicit value, or min(2048, chunk, kv budget).

    Returns 0 (graph off) when the MoE all-to-all backend is DeepEP: an
    extend-shaped forward takes DeepEP's normal dispatch, whose per-expert
    receive counts come back to the host, and a host sync cannot be captured.
    """
    if server_args.all2all_backend not in (None, "none"):
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
    max_cudagraph_capture_size: int
    model_is_mrope: bool
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
    # spec_num_tokens == spec_num_steps + 1 for now (without Tree Attention)
    spec_num_tokens: int | None = None
    overlap_schedule_depth: int = 0
    dp_sampling: bool = False
    dp_sampling_min_bs: int | None = None
    use_v4_mtp_paged_metadata: bool = False

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
                "physical context extent %s (context_len %s + spec overshoot "
                "pad %s) exceeds the model's derived context length %s; "
                "positions in the pad index past the precomputed rope tables. "
                "Lower --max-model-len by at least %s to stay in bounds.",
                physical_context_len,
                model_config.context_len,
                server_args.spec_context_pad,
                derived_context_len,
                physical_context_len - derived_context_len,
            )

        # Most DSA sparse indexers read the attention backend's
        # ``chunked_prefill_metadata`` from inside the captured prefill segment,
        # but the prefill graph rebinds only the live ForwardContext at replay --
        # the backend metadata object stays frozen at capture-time (dummy) values.
        # GLM-5.3-Flash handles padded replay explicitly.
        # Qwen4-Exp's PLE/QSA modules likewise own token-indexed side-state writes;
        # replay pads token rows to a bucket while their cache metadata remains
        # real-token shaped. Keep those prefills eager so padding can never
        # advance n-gram, short-conv, or compressed-key state.
        text_config = model_config.hf_text_config
        qwen4_exp_has_side_state = getattr(text_config, "model_type", None) == (
            "qwen4_exp_text"
        ) and bool(
            getattr(text_config, "ple_layer_ids", None)
            or getattr(text_config, "indexer_n_heads", None) is not None
        )
        disable_prefill_graph = (
            bool(server_args.disable_prefill_graph)
            or (
                model_config.attention_arch == AttentionArch.DSA
                and getattr(model_config.hf_config, "model_type", None) != "glm53_flash"
            )
            or qwen4_exp_has_side_state
        )

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
            enable_cudagraph_gc=server_args.enable_cudagraph_gc,
            max_cudagraph_capture_size=server_args.max_cudagraph_capture_size,
            disable_prefill_graph=disable_prefill_graph,
            prefill_graph_max_tokens=_resolve_prefill_graph_max_tokens(server_args),
            prefill_graph_capture_sizes=server_args.prefill_graph_capture_sizes,
            model_is_mrope=model_is_mrope,
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
            overlap_schedule_depth=overlap_schedule_depth,
            dp_sampling=server_args.dp_sampling,
            dp_sampling_min_bs=server_args.dp_sampling_min_bs,
            enable_nan_detection=server_args.enable_nan_detection,
            use_v4_mtp_paged_metadata=model_config.use_v4_mtp_paged_metadata,
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
        # capture-path assert: a missing contract means the model family has
        # no cache recipe yet, and an uncovered family means a backend that
        # never reads that group's tables.
        validate_scheduler_config(
            attn_backend=attn_backend,
            kv_pool=token_to_kv_pool,
        )
        self._cache_runtime_contract = token_to_kv_pool.arena.runtime_contract

        # The batch-ordered full-history table backs out_cache_loc and the
        # draft page table. First contract group with family=history and
        # retention=full_history.
        self._full_history_group_id = next(
            (
                str(spec.group_id)
                for spec in self._cache_runtime_contract.group_specs
                if spec.family == "history" and spec.retention == "full_history"
            ),
            None,
        )
        self.draft_attn_backend = draft_attn_backend
        self.draft_token_to_kv_pool = draft_token_to_kv_pool
        self._draft_final_step_counter = None

        # fill_input_buffers indexes the scheduler table in its own table pages; the drafter indexes draft_page_table in its backend's kernel pages.
        # The grain comes off the draft view's arena (the plan's P), never from
        # the view itself -- a recipe may pin a P the config never saw.
        self._block_granularity = int(
            _cache_arena_attr(draft_token_to_kv_pool, "prefix_granularity", 0)
            or config.prefix_granularity
        )
        draft_kernel_page_size = getattr(draft_attn_backend, "kernel_page_size", None)
        if draft_attn_backend is not None and draft_kernel_page_size is None:
            raise RuntimeError("draft attention backend must expose kernel_page_size")
        # Without a draft backend the staging path degenerates to the
        # scheduler table grain (identity expansion).
        self._draft_kernel_page_size = int(
            draft_kernel_page_size or self._block_granularity
        )
        if self._block_granularity % self._draft_kernel_page_size:
            raise ValueError(
                f"prefix granularity {self._block_granularity} is not a multiple "
                f"of the draft kernel page size {self._draft_kernel_page_size}"
            )
        # DraftPageStaging.publish expands the target full-history table into the
        # draft backend's kernel pages once, and every draft backend reads that
        # staged table as-is (identity). No backend re-expands with a logical
        # size, so there is no double-expansion to guard against here.

        # physical_context_len already covers the spec-verify overshoot of a
        # finished request lingering one overlap step, including the lingering
        # step's next draft block (see _SPEC_OVERSHOOT_SPANS in server_args.py).
        # A write past this width would go out of bounds and hang the attention
        # kernel; the output processor's physical-extent tripwire raises first.
        max_num_pages_per_req = (
            config.physical_context_len + self._draft_kernel_page_size - 1
        ) // self._draft_kernel_page_size

        max_bs = config.max_num_seqs // max(config.data_parallel_size, 1)

        # Address-stable staging of the full-history table for in-graph draft
        # consumers; also the zero/dummy placeholder for idle/warmup forwards
        # before the cache contract binds. Single writer; unit is the draft
        # kernel page; publish scrubs [bs, padded_bs).
        self._draft_staging = DraftPageStaging(
            max_bs=max_bs,
            max_pages_per_req=max_num_pages_per_req,
            block_granularity=self._block_granularity,
            draft_kernel_page_size=self._draft_kernel_page_size,
            full_history_group_id=self._full_history_group_id,
            enabled=not getattr(
                attn_backend, "cache_group_tables_replace_draft_page_table", False
            ),
            device=self.device,
        )
        self.draft_page_table = self._draft_staging.table
        spec_num_tokens = config.spec_num_tokens if config.spec_algo is not None else 1
        self.input_buffers = InputBuffers(
            max_bs=max_bs,
            max_num_tokens=config.chunked_prefill_size,
            # Indexes the scheduler's full-history table: scheduler-table page ids.
            page_size=self._block_granularity,
            # The cache arena reserves parent 0 as the null page, so slot 0
            # is the dummy slot padded tokens write into.
            dummy_kv_slot=0,
            state_write_padding_pool_index=config.max_req_pool_size,
            device=self.device,
        )
        self.runtime_states = RuntimeStates(
            req_pool_size=config.max_req_pool_size,
            vocab_size=config.vocab_size,
            device=self.device,
            output_length=config.output_length,
        )
        # Sized like InputBuffers.max_bs so the padded graph-bucket bs fits.
        self.nan_guard = NanGuard.create(
            config.enable_nan_detection,
            max_bs,
            self.device,
        )
        if self.config.spec_algo is not None:
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
                cache_view=self._draft_staging.view,
                attn_backend=draft_attn_backend,
                token_to_kv_pool=draft_token_to_kv_pool,
                vocab_size=config.vocab_size,
            )
            self.drafter.wire_target(self.model_runner.model)
            MultimodalRuntime.wire_drafter(
                self.input_buffers, self.model_runner.model_config
            )
        else:
            self.drafter = None

        self.grammar_runtime = create_grammar_runtime(
            grammar_backend=config.grammar_backend,
            disable_capturable=config.disable_capturable_grammar,
            is_nvidia=current_platform().is_nvidia,
            max_bs=max_bs,
            vocab_size=config.vocab_size,
            max_tokens_per_req=spec_num_tokens,
            device=self.device,
        )

        attn_backend.configure_runtime(
            sliding_window_size=model_runner.sliding_window_size,
            cache_group_specs=tuple(token_to_kv_pool.arena.cache_group_specs),
            cache_group_page_counts=_cache_arena_attr(
                token_to_kv_pool, "cache_group_page_counts", None
            ),
        )
        if draft_attn_backend is not None:
            draft_attn_backend.configure_runtime(
                sliding_window_size=model_runner.sliding_window_size,
                cache_group_specs=tuple(
                    _cache_arena_attr(draft_token_to_kv_pool, "cache_group_specs", ())
                ),
                cache_group_page_counts=_cache_arena_attr(
                    draft_token_to_kv_pool, "cache_group_page_counts", None
                ),
            )

        validate_cache_group_ids(
            model_runner.model,
            token_to_kv_pool.arena.cache_group_specs,
        )
        if draft_model_runner is not None and draft_token_to_kv_pool is not None:
            validate_cache_group_ids(
                draft_model_runner.model,
                draft_token_to_kv_pool.arena.cache_group_specs,
            )

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

        self.forward_step = CudaGraphWrapper(
            forward_func=self._forward_step,
            attn_backend=attn_backend,
            token_to_kv_pool=token_to_kv_pool,
            input_buffers=self.input_buffers,
            config=config,
            drafter=self.drafter,
            draft_attn_backend=draft_attn_backend,
            draft_token_to_kv_pool=draft_token_to_kv_pool,
            capturable_grammar=self.capturable_grammar,
            eager_grammar_buffers=self.eager_grammar_buffers,
            sampling_backend=self.sampling_backend,
            runtime_states=self.runtime_states,
        )
        # Eager warmup can be DP-asymmetric; prewarm RSAG under uniform dummy inputs.
        if config.enforce_eager:
            logger.info("Prewarming Triton RSAG communication states")
            self.forward_step.prewarm_comm_states(batch_sizes=(1,))
            logger.info("Finished prewarming Triton RSAG communication states")

        # Breakable prefill (extend) CUDA graphs, the extend-mode analogue of
        # the decode wrapper above; borrows the decode capture stream so all
        # graphs share one mempool-reuse domain.
        self.prefill_graph = PrefillGraph(
            model_runner=self.model_runner,
            attn_backend=attn_backend,
            token_to_kv_pool=token_to_kv_pool,
            input_buffers=self.input_buffers,
            config=config,
            page_table=self.draft_page_table,
            drafter=self.drafter,
        )

        self._autotune()

        workspace_pool(self.device).freeze()

        if not self.forward_step.disable:
            self.forward_step.capture()
        if not self.prefill_graph.disable:
            self.prefill_graph.capture(self.forward_step)

        # Encoder graphs are installed before KV-cache sizing and retained by
        # the model runner; preserve the executor-level handle for callers.
        self.encoder_graph_wrappers = getattr(
            self.model_runner, "encoder_graph_wrappers", {}
        )

        self.device_module = torch.get_device_module(self.device)
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

        set_random_seed(48)

        logger.info("ModelExecutor initialized")

    def _autotune(self) -> None:
        """Profile tunable kernels over one dummy prefill before graph capture.

        The dummy batch is capped by both the chunked-prefill token budget and
        rank-local request capacity. ``make_dummy_batch`` splits tokens into
        requests of at most ``context_len``, while request-indexed buffers
        contain only ``max_num_seqs // data_parallel_size`` rows. Keeping the
        token count within their product prevents autotuning from constructing
        a batch that cannot fit those buffers.

        The tuner enumerates every smaller shape bucket from this pass, so a
        separate decode-sized pass is unnecessary. This must run before graph
        capture because a captured graph retains the tactic selected during
        capture. On distributed boots, per-tactic timings are averaged across
        ranks so every rank selects the same tactic.
        """
        per_rank_max_batch = max(
            1,
            int(self.config.max_num_seqs)
            // max(int(self.config.data_parallel_size), 1),
        )
        num_tokens = min(
            int(self.config.chunked_prefill_size),
            int(self.config.context_len) * per_rank_max_batch,
        )
        if num_tokens <= 0 or self.model_runner is None:
            return
        if self.config.pp_size > 1:
            # The tuning forward drives the model directly (no stage recv/send
            # threading), which a mid-pipeline stage cannot run. Fall back to
            # heuristic tactics on every rank so the world-averaged tactic
            # collective is skipped consistently.
            set_autotune_max_num_tokens(num_tokens)
            logger.info(
                "Kernel tuning skipped under pipeline parallelism; tunable "
                "kernels use heuristic tactics"
            )
            return

        # The bucket mapper keys serving-time tactic lookups, so it must match
        # any pre-swept table loaded earlier even when tuning itself is off.
        set_autotune_max_num_tokens(num_tokens)
        if self.config.disable_autotune:
            logger.info(
                "Kernel tuning disabled (--disable-autotune); tunable kernels "
                "use heuristic tactics"
            )
            return

        cpu_group = None
        if self.config.world_size > 1:
            cpu_group = pg_manager.get_process_group("gloo", self.config.world_group)

        logger.info(f"Kernel tuning with a dummy prefill of {num_tokens} tokens")
        ib = self.input_buffers
        tic = time.time()
        set_autotune_process_group(cpu_group)
        with autotune(), maybe_inference_mode():
            ctx = self.prefill_graph.make_dummy_batch(num_tokens)
            positions = (
                ib.mrope_positions_buf[:, :num_tokens]
                if self.config.model_is_mrope
                else ib.positions_buf[:num_tokens]
            )
            with active_forward(ctx):
                self.model_runner.forward(
                    ctx=ctx,
                    input_ids=ib.input_ids_buf[:num_tokens],
                    positions=positions,
                    out_cache_loc=ib.out_cache_loc_buf[:num_tokens],
                )
        set_autotune_process_group(None)
        torch.get_device_module(self.device).synchronize()
        dist.barrier()
        logger.info(f"Kernel tuning finished in {time.time() - tic:.1f}s")

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

    def _publish_draft_page_table(self, forward_op, block_tables) -> None:
        """Stage the full-history table for the draft (see DraftPageStaging).

        The upcoming replay may read up to the widest captured batch; without
        the wrapper's padded_bs at hand, scrub through the table end.
        """
        if self.drafter is None:
            return
        bs = len(forward_op.request_pool_indices)
        self._draft_staging.publish(
            block_tables, bs=bs, padded_bs=self.draft_page_table.shape[0]
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
        if not getattr(self, "_pp_wire_logged", False):
            self._pp_wire_logged = True
            logger.info(
                "PP stage %d recv wire: %s",
                self.config.pp_rank,
                [(tuple(shape), str(dtype)) for _, shape, dtype in spec],
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
        if not getattr(self, "_pp_wire_logged", False):
            self._pp_wire_logged = True
            logger.info(
                "PP stage %d send wire: %s",
                self.config.pp_rank,
                [(tuple(t.shape), str(t.dtype)) for t in tensors],
            )
        for tensor in tensors:
            pp_send(tensor, self.config.pp_rank + 1, self.config.pp_group)

    @property
    def _pp_is_last_stage(self) -> bool:
        return self.config.pp_rank == self.config.pp_size - 1

    @property
    def _pp_is_first_stage(self) -> bool:
        return self.config.pp_rank == 0

    @nvtx_range("target_forward", color="red")
    def _run_target_forward(self, bs: int, ctx: ForwardContext, req_pool_indices):
        positions = self._active_positions_override
        if positions is None:
            if self.config.model_is_mrope:
                positions = self.input_buffers.mrope_positions_buf[
                    :, : ctx.input_num_tokens
                ]
            else:
                positions = self.input_buffers.positions_buf[: ctx.input_num_tokens]
        # PP mid-pipeline: receive the upstream boundary state and thread it
        # through the model's pp_inbound channel. The P role forces eager, so
        # neither graph path below can be active alongside PP.
        if self.config.pp_size > 1 and not self._pp_is_first_stage:
            pp_inbound = self._pp_recv_stage_state(ctx.input_num_tokens)
            output = self.model_runner.forward(
                ctx,
                self.input_buffers.input_ids_buf[: ctx.input_num_tokens],
                positions,
                self.input_buffers.out_cache_loc_buf[: ctx.input_num_tokens],
                req_pool_indices=req_pool_indices,
                seq_lens=self.input_buffers.seq_lens_buf[:bs],
                extend_prefix_lens=self.input_buffers.extend_prefix_lens_buf[
                    : ctx.num_extends
                ],
                pp_inbound=pp_inbound,
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
            return self.prefill_graph.replay(
                ctx,
                self.input_buffers.input_ids_buf[: ctx.input_num_tokens],
                self._active_multimodal_context,
            )
        return self.model_runner.forward(
            ctx,
            self.input_buffers.input_ids_buf[: ctx.input_num_tokens],
            positions,
            self.input_buffers.out_cache_loc_buf[: ctx.input_num_tokens],
            req_pool_indices=req_pool_indices,
            seq_lens=self.input_buffers.seq_lens_buf[:bs],
            extend_prefix_lens=self.input_buffers.extend_prefix_lens_buf[
                : ctx.num_extends
            ],
            multimodal_context=self._active_multimodal_context,
        )

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

    @nvtx_range("sampling", color="yellow")
    def _run_sampling(
        self,
        logits_output: LogitsProcessorOutput,
        sampling_info: SamplingBatchInfo,
        ctx: ForwardContext,
        candidates: torch.Tensor | None = None,
    ):
        if self.drafter is None:
            return self.sampling_backend.sample(logits_output, sampling_info)

        num_extends = ctx.num_extends
        num_decodes = ctx.bs - num_extends

        if num_decodes == 0:
            return self.sampling_backend.sample(logits_output, sampling_info)

        if num_extends == 0:
            output_tokens, accept_lengths = self.sampling_backend.verify(
                logits_output, sampling_info, candidates
            )
            accept_lengths = self._apply_force_single_token_verify(
                accept_lengths, 0, num_decodes, ctx.decode_input_ids
            )
            return output_tokens, accept_lengths

        logits = logits_output.next_token_logits
        prefill_out = LogitsProcessorOutput(next_token_logits=logits[:num_extends])
        prefill_tokens, prefill_accept = self.sampling_backend.sample(
            prefill_out, sampling_info[:num_extends]
        )
        decode_out = LogitsProcessorOutput(next_token_logits=logits[num_extends:])
        decode_tokens, decode_accept = self.sampling_backend.verify(
            decode_out, sampling_info[num_extends:], candidates
        )
        decode_accept = self._apply_force_single_token_verify(
            decode_accept, num_extends, num_decodes, ctx.decode_input_ids
        )
        if (
            prefill_out.next_token_logprobs is not None
            and decode_out.next_token_logprobs is not None
        ):
            logits_output.next_token_logprobs = torch.cat(
                [prefill_out.next_token_logprobs, decode_out.next_token_logprobs]
            )
        return (
            torch.cat([prefill_tokens, decode_tokens]),
            torch.cat([prefill_accept, decode_accept]),
        )

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
            "Batch-DP route: forward_mode=%s bs=%d effective_bs=%d "
            "use_graph=%s bucket_bs=%d dp_sampling=%s min_bs=%d",
            ctx.forward_mode.name.lower(),
            bs,
            effective_bs,
            use_graph,
            bucket_bs,
            dp_sampling,
            runtime.min_bs,
        )

    @maybe_inference_mode()
    def _forward_step(
        self,
        bs: int,
        ctx: ForwardContext,
        sampling_info: SamplingBatchInfo,
    ):
        req_pool_indices = self.input_buffers.req_pool_indices_buf[:bs]

        # Fork grammar onto its side stream so fill + H2D overlap with
        # attention/MoE. Rejoined at wait_bitmask() before apply_mask.
        if self.capturable_grammar is not None:
            n = self.capturable_grammar.max_tokens_per_req
            is_spec_verify = n > 1 and ctx.forward_mode.is_decode()
            slice_ = (
                self.input_buffers.input_ids_buf[: bs * n] if is_spec_verify else None
            )
            self.capturable_grammar.schedule_fill(input_ids_buf_slice=slice_)

        if (
            self.drafter is not None
            and getattr(self.drafter, "_incremental_proj_enabled", False)
            and ctx.num_extends == 0
        ):
            self.drafter._prepare_incremental_proj(
                ctx.input_num_tokens,
                self.input_buffers.positions_buf[: ctx.input_num_tokens],
                self.input_buffers.out_cache_loc_buf[: ctx.input_num_tokens],
            )

        logits_output = self._run_target_forward(bs, ctx, req_pool_indices)

        if self.config.pp_size > 1 and not self._pp_is_last_stage:
            # Mid-pipeline stage: the model returned the boundary bundle, not
            # logits. Ship it downstream and return placeholder outputs — the
            # commit path recognizes a PP placeholder and emits no tokens.
            self._pp_send_stage_state(logits_output)
            output_tokens = torch.zeros(bs, dtype=torch.int32, device=self.device)
            accept_lengths = torch.ones(bs, dtype=torch.int32, device=self.device)
            return output_tokens, accept_lengths, None

        if self.drafter is not None and getattr(
            self.drafter, "_incremental_proj_enabled", False
        ):
            self.drafter.target_language_model.model._dflash_incr_active = False

        # Flag NaN per request and sanitize in place, before any sampling kernel.
        self.nan_guard.audit_logits(logits_output, ctx)

        candidates = (
            self.drafter.get_candidates(ctx)
            if self.config.spec_algo is not None
            else None
        )

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

        if self.drafter is not None:
            next_round_input_ids = self.drafter.run(
                base_ctx=ctx,
                logits_output=logits_output,
                output_tokens=output_tokens,
                accept_lengths=accept_lengths,
            )
            # _update_runtime_state skips future_input_map when drafter is
            # active — drafter writes the next-round inputs directly.
            self.runtime_states.future_input_map[
                self.input_buffers.state_write_req_pool_indices_buf[: ctx.bs]
            ] = next_round_input_ids.to(torch.int32)
            self._record_draft_final_cache_step(ctx.num_extends)

        output_logprobs = logits_output.next_token_logprobs
        return output_tokens, accept_lengths, output_logprobs

    @nvtx_range("update_runtime_state", color="orange")
    def _update_runtime_state(
        self,
        req_pool_indices: torch.Tensor,
        output_tokens: torch.Tensor,
        accept_lengths: torch.Tensor,
        input_lengths: torch.Tensor,
        num_extends: int,
    ):
        """Write output tokens to future_input_map and update cache lengths.

        Must NOT be captured in CUDA graph — these writes are read by the
        next iteration's batch prep on the default stream, so they need
        explicit stream synchronization (see execute_forward_op).
        """
        if self.drafter is None:
            # Without drafter, store output tokens for next round.
            # With drafter, _forward_step already wrote the drafter's
            # next-round input (verified + draft tokens) to future_input_map.
            tokens_per_req = self.config.output_length if num_extends == 0 else 1
            next_round_input_ids = output_tokens.to(torch.int32).reshape(
                -1, tokens_per_req
            )
            self.runtime_states.future_input_map[req_pool_indices, :tokens_per_req] = (
                next_round_input_ids
            )

        bs = req_pool_indices.shape[0]
        if num_extends == 0:
            deltas = accept_lengths
        elif num_extends == bs:
            deltas = input_lengths
        else:
            deltas = torch.cat(
                [input_lengths[:num_extends], accept_lengths[num_extends:]]
            )
        self.runtime_states.update_valid_cache_length(req_pool_indices, deltas)

    def _build_sampling_info(
        self,
        bs: int,
        sampling_params_list: list[SamplingParams],
    ) -> SamplingBatchInfo:
        return SamplingBatchInfo(
            req_pool_indices=self.input_buffers.req_pool_indices_buf[:bs],
            valid_cache_lengths=self.runtime_states.valid_cache_lengths,
            is_all_greedy=all(p.top_k <= 1 for p in sampling_params_list),
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
            input_num_tokens=0,
            forward_mode=graph_forward_mode,
            global_num_tokens=dp_metadata.global_num_tokens,
            global_bs=dp_metadata.global_batch_size,
            all_decode_or_idle=dp_metadata.all_decode_or_idle,
        )
        sampling_info = SamplingBatchInfo(
            req_pool_indices=self.input_buffers.req_pool_indices_buf[:0],
            valid_cache_lengths=self.runtime_states.valid_cache_lengths,
            is_all_greedy=True,
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
                    grammars=[None] * padded_bs, bs=padded_bs, has_candidates=False
                )
            # IDLE doesn't produce tokens, so no sampler/drafter call here —
            # only the model forward, which still participates in collectives.
            # A rank that previously served a larger batch still has real page
            # ids in the padded_bs rows the captured drafter steps read; their
            # draft KV writes would alias live requests' pages (#955).
            self._draft_staging.publish(None, bs=0, padded_bs=padded_bs)
            with nvtx_range("forward_step idle", color="blue"):
                self.forward_step(
                    bs=0,
                    ctx=ctx,
                    sampling_info=sampling_info,
                    page_table=self.draft_page_table,
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
            out_cache_loc=empty,
        )

        # If a drafter is active, its model also has MoE layers that issue
        # NCCL collectives. Idle ranks must match those collectives:
        # 1 first-step forward + (spec_num_steps - 1) multi-step decode forwards.
        if self.drafter is not None:
            # DFLASH is a block drafter (idle_forward_steps=1); EAGLE3/MTP
            # default to spec_num_steps. Mirror the active rank's per-step
            # collective sizing either way.
            idle_forward_steps = getattr(
                self.drafter, "idle_forward_steps", self.drafter.spec_num_steps
            )
            for step_idx in range(idle_forward_steps or 0):
                # Mirror active rank's catch-up step: when all non-idle ranks
                # are decoding, step 0 sizes collectives from bs/global_bs.
                draft_global_num_tokens = _draft_idle_global_num_tokens_for_step(
                    step_idx,
                    dp_metadata.global_num_tokens,
                    dp_metadata.global_batch_size,
                )
                draft_ctx = ForwardContext(
                    attn_backend=self.drafter.attn_backend,
                    token_to_kv_pool=self.drafter.token_to_kv_pool,
                    bs=0,
                    num_extends=0,
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
                    out_cache_loc=empty,
                    spec_step_idx=step_idx,
                )

    def zero_cache_pages(self, pages):
        """Clear newly owned pages and return a CUDA completion event when needed."""
        if not pages:
            return None

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
            draft_pool = getattr(self, "draft_token_to_kv_pool", None)
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
        done.record(self.device_module.current_stream(self.device))
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
        self.execution_stream.wait_stream(self.device_module.current_stream())
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
            # Wait for previous iteration's runtime state updates
            # (future_input_map, valid_cache_lengths) on execution_stream to
            # complete before reading them.
            self.device_module.current_stream().wait_stream(self.execution_stream)
            self.execution_stream.wait_stream(self.device_module.current_stream())
        with self.device_module.stream(self.execution_stream):
            bs = len(forward_op.request_ids)
            # Outside the graph: in-graph sites only OR into the flag buffer.
            self.nan_guard.reset(bs)
            cache_metadata = None
            block_tables = {}
            if bs > 0:
                # Validate and pack the per-group tables once for this batch.
                cache_metadata = CacheBatchMetadata.from_forward_op(
                    forward_op,
                    device=self.device,
                    contract=self._cache_runtime_contract,
                    num_requests=bs,
                )
                block_tables = dict(cache_metadata.tables(active_forward_op=forward_op))
            # out_cache_loc reads the batch-ordered full-history table (row i ==
            # batch position i). Without a full-history group the zeroed draft
            # table stands in (out_cache_loc then lands on the dummy page 0;
            # such pools address their KV through their own per-group tables).
            page_table = (
                block_tables.get(self._full_history_group_id)
                if self._full_history_group_id is not None
                else None
            )
            if page_table is None:
                page_table = self.draft_page_table
            # Drafts read their pages from the batch-ordered draft page table.
            self._publish_draft_page_table(forward_op, block_tables)
            decode_input_ids = self.input_buffers.fill_input_buffers(
                forward_op=forward_op,
                runtime_states=self.runtime_states,
                total_tokens=total_tokens,
                out_loc_table=page_table,
            )
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

            if total_tokens == 0:
                # Fully prefix-cached prefill: no tokens to process.
                output_tokens = torch.zeros(0, dtype=torch.int32, device=self.device)
                output_lengths = torch.zeros(bs, dtype=torch.int32, device=self.device)
                output_logprobs = None
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
                        else CaptureHiddenMode.NULL
                    ),
                    gather_ids=gather_ids,
                    decode_input_ids=decode_input_ids,
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
                    sampling_info = self._build_sampling_info(bs, sampling_params_list)
                    grammar_completion = setup_grammar_step(
                        sampling_info=sampling_info,
                        bs=bs,
                        is_spec_decode=self.drafter is not None and num_extends < bs,
                        spec_num_tokens=self.config.spec_num_tokens or 1,
                        grammar_inputs=grammar_inputs,
                        grammar_runtime=self.grammar_runtime,
                        input_ids_buf=self.input_buffers.input_ids_buf,
                        grammar_backend=self.config.grammar_backend,
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
                    output_tokens, output_lengths, output_logprobs = self.forward_step(
                        bs=bs,
                        ctx=ctx,
                        sampling_info=sampling_info,
                        page_table=self.draft_page_table,
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
                        block_tables=block_tables,
                        cache_metadata=cache_metadata,
                        forward_batch=(
                            forward_op if cache_metadata is not None else None
                        ),
                    )
                    if timing_enabled:
                        forward_step_ms = (
                            time.perf_counter() - forward_step_start
                        ) * 1000.0

                # Update runtime state on execution_stream (NOT in the CUDA graph).
                self._update_runtime_state(
                    req_pool_indices=self.input_buffers.req_pool_indices_buf[:bs],
                    output_tokens=output_tokens,
                    accept_lengths=output_lengths,
                    input_lengths=self.input_buffers.input_lengths_buf[:bs],
                    num_extends=num_extends,
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

                if (
                    LOG_SPEC_ACCEPT_LENGTHS
                    and self.config.spec_num_steps
                    and num_extends == 0
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
                    "mm_timing forward_execute_ms total=%.3f input_fill=%.3f "
                    "mrope=%.3f sampling=%.3f forward_step=%.3f output_d2h=%.3f "
                    "mode=%s bs=%s total_tokens=%s graph=%s padded_bs=%s "
                    "has_mm=%s mm_count=%s mm_delta_count=%s",
                    (time.perf_counter() - timing_start) * 1000.0,
                    input_fill_ms,
                    mrope_ms,
                    sampling_prep_ms,
                    forward_step_ms,
                    output_d2h_ms,
                    forward_mode.name,
                    bs,
                    total_tokens,
                    graph_capable,
                    graph_padded_bs,
                    has_mm,
                    mm_count,
                    mm_delta_count,
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

    def register_draft_final_step_counter(self, step_counter) -> None:
        """Publish one CachePD step after a supported drafter's complete run."""
        if self.drafter is None or not getattr(
            self.drafter, "supports_pd_layerwise_finalization", False
        ):
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
