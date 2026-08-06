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
from tokenspeed.runtime.execution.drafter.deepseek_v4_dspark import (
    DeepseekV4DSpark,
)
from tokenspeed.runtime.execution.drafter.dflash import DFlash
from tokenspeed.runtime.execution.drafter.dspark import DSpark
from tokenspeed.runtime.execution.drafter.eagle import Eagle
from tokenspeed.runtime.execution.drafter.mtp import Mtp
from tokenspeed.runtime.execution.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
)
from tokenspeed.runtime.execution.input_buffer import InputBuffers
from tokenspeed.runtime.execution.model_runner import ModelRunner
from tokenspeed.runtime.execution.multimodal_runtime import MultimodalRuntime
from tokenspeed.runtime.execution.nan_guard import NanGuard
from tokenspeed.runtime.execution.prefill_graph import PrefillGraph
from tokenspeed.runtime.execution.runtime_states import RuntimeStates
from tokenspeed.runtime.execution.types import ModelExecutionResult
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
from tokenspeed.runtime.layers.attention.page_table import expand_page_table
from tokenspeed.runtime.layers.logits_processor import LogitsProcessorOutput
from tokenspeed.runtime.layers.paged_attention import (
    validate_paged_cache_group_ids,
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
from tokenspeed.runtime.utils.nvtx import nvtx_range
from tokenspeed.runtime.utils.server_args import ServerArgs

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
    from tokenspeed.runtime.layers.attention.kv_cache.base import CachePool
    from tokenspeed.runtime.sampling.sampling_params import SamplingParams

logger = get_colorful_logger(__name__)

LOG_MM_TIMING = envs.TOKENSPEED_LOG_MM_TIMING.get()
LOG_SPEC_ACCEPT_LENGTHS = envs.TOKENSPEED_LOG_SPEC_ACCEPT_LENGTHS.get()


def _get_drafter_impl(spec_algo: str, model: torch.nn.Module):
    from tokenspeed.runtime.models.inkling_nextn import (
        InklingForConditionalGenerationNextN,
    )

    DRAFTER_MAPPING = {
        "EAGLE3": Eagle,
        "MTP": Eagle,
        "DFLASH": DFlash,
        "DSPARK": DSpark,
    }

    # "MTP" covers two algorithms:
    # (1) Eagle-like MTP (e.g. DeepSeek) stays on Eagle in eagle.py;
    # (2) Vanilla MTP (e.g. Inkling) with multi-layer weights stays on Mtp in mtp.py.
    if spec_algo == "DSPARK":
        from tokenspeed.runtime.models.deepseek_v4_dspark import (
            DeepseekV4ForCausalLMDSpark,
        )

        if isinstance(model, DeepseekV4ForCausalLMDSpark):
            return DeepseekV4DSpark
    if spec_algo == "MTP" and isinstance(model, InklingForConditionalGenerationNextN):
        return Mtp
    else:
        return DRAFTER_MAPPING[spec_algo]


def _eagle_aux_layer_ids(hf_config) -> list[int] | None:
    """Return EAGLE3 capture ids from a draft config, including K3 text config.

    K3 wraps the language configuration in ``text_config``.  Draft exports may
    place ``eagle_config`` either on that text config or on the top-level
    wrapper, so inspect both without falling back to the target's defaults.
    """
    candidates = [hf_config]
    text_config = (
        hf_config.get("text_config")
        if isinstance(hf_config, dict)
        else getattr(hf_config, "text_config", None)
    )
    if text_config is not None:
        candidates.append(text_config)

    for config in candidates:
        if isinstance(config, dict):
            eagle_config = config.get("eagle_config")
            direct_ids = config.get("eagle_aux_hidden_state_layer_ids")
        else:
            eagle_config = getattr(config, "eagle_config", None)
            direct_ids = getattr(config, "eagle_aux_hidden_state_layer_ids", None)
        if isinstance(eagle_config, dict):
            ids = eagle_config.get("eagle_aux_hidden_state_layer_ids")
        elif eagle_config is not None:
            ids = getattr(eagle_config, "eagle_aux_hidden_state_layer_ids", None)
        else:
            ids = direct_ids
        if ids:
            return list(ids)
    return None


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
    logical_page_size: int
    max_num_seqs: int
    chunked_prefill_size: int
    vocab_size: int
    context_len: int
    device: str
    gpu_id: int
    global_rank: int
    num_total_pages: int
    decode_log_interval: int
    cudagraph_capture_sizes: list[int] | None
    disable_cuda_graph_padding: bool
    max_cudagraph_capture_size: int
    model_is_mrope: bool
    enable_nan_detection: bool = False

    # ====== DP =========
    data_parallel_size: int = 1
    world_size: int = 1
    world_group: list[int] | None = None

    # ====== SPEC =========
    spec_algo: str | None = None
    spec_num_steps: int | None = None
    # spec_num_tokens == spec_num_steps + 1 for now (without Tree Attention)
    spec_num_tokens: int | None = None
    # Explicit EAGLE3 capture ids; overrides the draft checkpoint's ids.
    eagle3_layers_to_capture: list[int] | None = None
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
        num_total_pages: int,
        logical_page_size: int,
        overlap_schedule_depth: int = 0,
    ) -> ModelExecutorConfig:
        output_length = (
            server_args.speculative_num_draft_tokens
            if server_args.speculative_algorithm
            else 1
        )
        rope_parameters = get_rope_parameters(model_config.hf_text_config)
        model_is_mrope = bool(rope_parameters and "mrope_section" in rope_parameters)

        # DSA's sparse indexer reads the attention backend's
        # ``chunked_prefill_metadata`` from inside the captured prefill segment,
        # but the prefill graph rebinds only the live ForwardContext at replay --
        # the backend metadata object stays frozen at capture-time (dummy) values.
        # So the two are fundamentally incompatible; force eager prefill for DSA.
        disable_prefill_graph = bool(server_args.disable_prefill_graph) or (
            model_config.attention_arch == AttentionArch.DSA
        )

        return ModelExecutorConfig(
            max_req_pool_size=max_req_pool_size,
            output_length=output_length,
            enforce_eager=server_args.enforce_eager,
            logical_page_size=logical_page_size,
            max_num_seqs=server_args.max_num_seqs,
            chunked_prefill_size=server_args.chunked_prefill_size,
            vocab_size=model_config.vocab_size,
            context_len=model_config.context_len,
            device=server_args.device,
            gpu_id=gpu_id,
            global_rank=global_rank,
            num_total_pages=num_total_pages,
            decode_log_interval=server_args.decode_log_interval,
            cudagraph_capture_sizes=server_args.cudagraph_capture_sizes,
            disable_cuda_graph_padding=server_args.disable_cuda_graph_padding,
            max_cudagraph_capture_size=server_args.max_cudagraph_capture_size,
            disable_prefill_graph=disable_prefill_graph,
            prefill_graph_max_tokens=_resolve_prefill_graph_max_tokens(server_args),
            prefill_graph_capture_sizes=server_args.prefill_graph_capture_sizes,
            model_is_mrope=model_is_mrope,
            data_parallel_size=server_args.mapping.attn.dp_size,
            world_size=server_args.mapping.world_size,
            world_group=server_args.mapping.world_group,
            spec_algo=server_args.speculative_algorithm,
            spec_num_steps=server_args.speculative_num_steps,
            spec_num_tokens=server_args.speculative_num_draft_tokens,
            eagle3_layers_to_capture=server_args.eagle3_layers_to_capture,
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
        # contract; the per-group tables travel as CacheBatchMetadata. Fail fast
        # here rather than at the first forward: a missing contract means the
        # model family has no cache recipe yet (add one in
        # kv_cache.recipes.setup.prepare_cache_setup, see the 'mha' / 'msa' recipes for
        # the pattern).
        self._cache_runtime_contract = getattr(
            token_to_kv_pool, "runtime_contract", None
        )
        if self._cache_runtime_contract is None:
            raise RuntimeError(
                f"KV pool {type(token_to_kv_pool).__name__} publishes no "
                "PagedCacheRuntimeContract. Every pool must be built from a "
                "cache recipe (kv_cache.recipes.setup.prepare_cache_setup)."
            )
        # The batch-ordered full-history table backs out_cache_loc and the
        # draft page table. First contract group with family=history and
        # retention=full_history -- the same selection CacheBatchMetadata
        # exposes as ``first_full_attention_group_id``.
        self._full_history_group_id = next(
            (
                str(spec.group_id)
                for spec in self._cache_runtime_contract.group_specs
                if getattr(spec, "family", "history") == "history"
                and getattr(spec, "retention", None) == "full_history"
            ),
            None,
        )
        self.draft_attn_backend = draft_attn_backend
        self.draft_token_to_kv_pool = draft_token_to_kv_pool

        # Must precede CUDA-graph capture: unsupported cache combinations
        # (e.g. spec on a backend without cache_group_spec_capable) would otherwise
        # die on a capture-path assert instead of this actionable error.
        validate_scheduler_config(
            paged_cache_groups=self._cache_runtime_contract.group_specs,
            attn_backend=attn_backend,
            kv_pool=token_to_kv_pool,
            speculative_algorithm=config.spec_algo,
        )

        # fill_input_buffers indexes the scheduler table in logical pages; the drafter indexes draft_page_table in its backend's kernel pages.
        self._logical_page_size = int(
            getattr(draft_token_to_kv_pool, "page_size", 0) or config.logical_page_size
        )
        self._draft_page_size = int(
            getattr(draft_attn_backend, "page_size", 0) or self._logical_page_size
        )
        if self._logical_page_size % self._draft_page_size:
            raise ValueError(
                f"logical page size {self._logical_page_size} is not a multiple "
                f"of the draft kernel page size {self._draft_page_size}"
            )
        self._draft_page_ratio = self._logical_page_size // self._draft_page_size

        if config.spec_algo is not None:
            # The DFLASH overlap scheduler reserves a fresh draft block per
            # decode step a request stays scheduled, including the few steps it
            # lingers between finishing and eviction, so peak page count runs
            # ~1 page past context_len + spec_num_tokens. Without headroom
            # page_table overflows and the next draft block's page write goes
            # out of bounds, hanging the attention kernel. Pad generously; a few
            # int32 columns per request. Non-DFLASH algorithms do not need this.
            draft_block_reservation_slack = (
                config.spec_num_tokens * 64
                if config.spec_algo in ("DFLASH", "DSPARK")
                else 0
            )
            max_num_pages_per_req = (
                config.context_len
                + config.spec_num_tokens
                + draft_block_reservation_slack
                + self._draft_page_size
                - 1
            ) // self._draft_page_size
        else:
            max_num_pages_per_req = (
                config.context_len + self._draft_page_size
            ) // self._draft_page_size

        max_bs = config.max_num_seqs // max(config.data_parallel_size, 1)

        # Batch-ordered page table (row i == batch position i). Populated per
        # forward from the target's full-history table so drafts read their
        # pages without a req-pool round-trip; also the zero/dummy placeholder
        # for idle/warmup forwards before the cache contract binds. Graph-
        # stable: the drafter's captured segments record this buffer's address.
        self.draft_page_table = torch.zeros(
            (max_bs, max_num_pages_per_req),
            dtype=torch.int32,
            device=self.device,
        )
        spec_num_tokens = config.spec_num_tokens if config.spec_algo is not None else 1
        self.input_buffers = InputBuffers(
            max_bs=max_bs,
            max_num_tokens=config.chunked_prefill_size,
            # Indexes the scheduler's full-history table: logical page ids.
            page_size=self._logical_page_size,
            # token_to_kv_pool allocates size+page_size slots; index `size` is
            # the reserved dummy slot (see MHATokenToKVPool._create_buffers).
            dummy_kv_slot=0,
            state_write_padding_pool_index=config.max_req_pool_size,
            device=self.device,
        )
        self.runtime_states = RuntimeStates(
            req_pool_size=config.max_req_pool_size,
            context_len=config.context_len,
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
            DrafterImpl = _get_drafter_impl(config.spec_algo, draft_model_runner.model)
            self.drafter = DrafterImpl(
                spec_num_tokens=config.spec_num_tokens,
                spec_num_steps=config.spec_num_steps,
                draft_model_runner=draft_model_runner,
                page_size=self._draft_page_size,
                runtime_states=self.runtime_states,
                input_buffers=self.input_buffers,
                page_table=self.draft_page_table,
                attn_backend=draft_attn_backend,
                token_to_kv_pool=draft_token_to_kv_pool,
                vocab_size=config.vocab_size,
            )
            if hasattr(self.drafter, "bind_target_model"):
                self.drafter.bind_target_model(self.model_runner.model)
            # The V4 checkpoint-local DSpark path shares the target's embed and
            # LM head. Generic DSpark and DFlash ship their own draft weights.
            if config.spec_algo in ("EAGLE3", "MTP") or isinstance(
                self.drafter, DeepseekV4DSpark
            ):
                embed, head = self.model_runner.model.get_embed_and_head()
                draft_model_runner.model.set_embed_and_head(embed, head)
            MultimodalRuntime.wire_drafter(
                self.drafter, self.model_runner.model_config.hf_config
            )
            if config.spec_algo in ("EAGLE3",) and hasattr(
                self.model_runner.model, "set_eagle3_layers_to_capture"
            ):
                # capture the layers the draft was trained on, not the default
                aux_layer_ids = config.eagle3_layers_to_capture or _eagle_aux_layer_ids(
                    draft_model_runner.model_config.hf_config
                )
                self.model_runner.model.set_eagle3_layers_to_capture(aux_layer_ids)
            if config.spec_algo in ("DFLASH", "DSPARK") and not isinstance(
                self.drafter, DeepseekV4DSpark
            ):
                if not hasattr(self.model_runner.model, "set_dflash_layers_to_capture"):
                    raise ValueError(
                        "DFLASH requires the target model to support "
                        "set_dflash_layers_to_capture."
                    )
                incr_callback = None
                incr_slot_bufs = None
                if getattr(self.drafter, "_incremental_proj_enabled", False):
                    incr_callback = self.drafter._on_capture_slot_ready
                    incr_slot_bufs = self.drafter._incr_slot_bufs
                self.model_runner.model.set_dflash_layers_to_capture(
                    self.drafter.target_layer_ids,
                    incremental_callback=incr_callback,
                    slot_bufs=incr_slot_bufs,
                )
            if isinstance(self.drafter, DeepseekV4DSpark):
                if not hasattr(self.model_runner.model, "set_dspark_layers_to_capture"):
                    raise ValueError(
                        "DSPARK requires the target model to support "
                        "set_dspark_layers_to_capture."
                    )
                self.model_runner.model.set_dspark_layers_to_capture(
                    self.drafter.target_layer_ids
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
            paged_cache_group_specs=tuple(token_to_kv_pool.paged_cache_group_specs),
            paged_cache_group_page_counts=getattr(
                token_to_kv_pool,
                "paged_cache_group_page_counts",
                None,
            ),
        )
        if draft_attn_backend is not None:
            draft_attn_backend.configure_runtime(
                sliding_window_size=model_runner.sliding_window_size,
                paged_cache_group_specs=tuple(
                    getattr(draft_token_to_kv_pool, "paged_cache_group_specs", ())
                ),
                paged_cache_group_page_counts=getattr(
                    draft_token_to_kv_pool,
                    "paged_cache_group_page_counts",
                    None,
                ),
            )

        validate_paged_cache_group_ids(
            model_runner.model,
            token_to_kv_pool.paged_cache_group_specs,
        )
        if draft_model_runner is not None and draft_token_to_kv_pool is not None:
            validate_paged_cache_group_ids(
                draft_model_runner.model,
                draft_token_to_kv_pool.paged_cache_group_specs,
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

        if not self.forward_step.disable:
            self.forward_step.capture()
        if not self.prefill_graph.disable:
            self.prefill_graph.capture(self.forward_step)

        # Encoder graphs are installed before KV-cache sizing and retained by
        # the model runner; preserve the executor-level handle for callers.
        self.encoder_graph_wrappers = getattr(
            self.model_runner, "encoder_graph_wrappers", {}
        )

        self.execution_stream = torch.cuda.Stream()
        self.log_step = 0
        self._seen_prefill_ids: set[str] = set()
        self._prev_decode_bs: int = 0
        self._sentinel_neg1 = torch.tensor(-1, device=self.device, dtype=torch.int64)
        self.mm_runtime = MultimodalRuntime(
            model_is_mrope=config.model_is_mrope,
            input_buffers=self.input_buffers,
            device=self.device,
        )
        # Decode stats — accumulated from synced results (no GPU sync needed)
        self.num_generated_tokens = 0
        self.num_decode_steps = 0
        self.last_decode_stats_tic = time.time()

        set_random_seed(48)

        logger.info("ModelExecutor initialized")

    def _autotune(self) -> None:
        """Profile tunable kernels over one dummy prefill, before graph capture.

        Runs a single extend forward at the largest token count a forward can
        carry; the tuner enumerates every smaller shape bucket from it, so no
        decode-sized pass is needed. Must precede capture: a captured graph
        records the tactic chosen while it was recorded, so tuning afterwards
        cannot change a replay. On distributed boots, per-tactic timings are
        averaged over the world so every rank picks the same tactic.
        """
        num_tokens = int(self.config.chunked_prefill_size)
        if num_tokens <= 0 or self.model_runner is None:
            return

        cpu_group = None
        if self.config.world_size > 1:
            cpu_group = pg_manager.get_process_group("gloo", self.config.world_group)

        logger.info(f"Kernel tuning with a dummy prefill of {num_tokens} tokens")
        ib = self.input_buffers
        tic = time.time()
        set_autotune_max_num_tokens(num_tokens)
        set_autotune_process_group(cpu_group)
        with autotune(), maybe_inference_mode():
            ctx = self.prefill_graph.make_dummy_batch(num_tokens, self.forward_step)
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
        torch.cuda.synchronize()
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
        """Publish the full-history table into the batch-ordered draft table.

        Every draft reads its pages from ``draft_page_table`` (the drafter
        passes no cache metadata). The table is batch-ordered -- row ``i`` is
        batch position ``i`` -- so the draft indexes it directly by batch row,
        no page_table round-trip. Ids arrive in the allocator's logical pages
        and are published in draft-page units (see ``_draft_page_ratio``).
        """
        if (
            getattr(
                getattr(self, "attn_backend", None),
                "cache_group_tables_replace_draft_page_table",
                False,
            )
            or self.drafter is None
            or not block_tables
            or self._full_history_group_id is None
        ):
            return
        table = block_tables.get(self._full_history_group_id)
        if table is None:
            return
        bs = len(forward_op.request_pool_indices)
        width = table.shape[1]
        max_width = self.draft_page_table.shape[1]
        rows = self.draft_page_table[:bs]
        if self._draft_page_ratio > 1:
            # -1 pads clamp into logical page 0, itself reserved as the null page.
            expand_page_table(
                table,
                logical_page_size=self._logical_page_size,
                kernel_page_size=self._draft_page_size,
                max_kernel_pages=max_width,
                out=rows,
            )
            return
        # -1 column pads -> dummy page 0 (negative locs otherwise).
        rows[:, :width].copy_(table)
        rows[:, :width].clamp_min_(0)
        if width < max_width:
            rows[:, width:].zero_()
        # Graph replay reads padded_bs rows; stale ids past bs alias another request's pages.
        self.draft_page_table[bs:].zero_()

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

    def _cap_accept_to_context_len(
        self,
        accept_lengths: torch.Tensor,
        decode_req_pool_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Clamp spec-verify accept so committed length never exceeds
        ``context_len``.

        ``page_table`` is sized for ``context_len + spec_num_tokens`` pages. A
        request at the context limit whose ``max_new_tokens`` termination lags a
        step can accept past ``context_len``, so its next draft block needs a
        page beyond ``page_table``'s width — an out-of-bounds access that hangs
        the attention kernel. Clamping to the remaining budget keeps the table in
        range; the request is still removed a step later. Deterministic in
        ``valid_cache_lengths`` / ``accept_lengths``, so no cross-rank divergence.
        """
        if accept_lengths.numel() == 0:
            return accept_lengths
        committed = self.runtime_states.valid_cache_lengths.index_select(
            0, decode_req_pool_indices
        ).to(accept_lengths.dtype)
        remaining = (self.config.context_len - committed).clamp_(min=0)
        # In-place: the drafter reads this same buffer to size its next block.
        accept_lengths.copy_(torch.minimum(accept_lengths, remaining))
        return accept_lengths

    def _clamp_committed_to_context_len(
        self,
        output_lengths: torch.Tensor,
        num_extends: int,
        bs: int,
    ) -> torch.Tensor:
        """Return ``output_lengths`` with decode rows clamped so committed KV
        length never exceeds ``context_len`` (post-forward, outside the CUDA
        graph).

        The clamp must reach BOTH ``_update_runtime_state``
        (``valid_cache_lengths``) and the ``ModelExecutionResult`` that drives
        scheduler page reservation, so they stay in lock-step. Hence a FRESH
        tensor, not the persistent ``_accept_length_buf``: the verify path also
        mirrors accept counts into ``_output_pack_buf``, and an in-place clamp
        would leave that mirror (read by the packed-D2H fast path) uncapped,
        reserving a draft block past ``page_table``'s width and hanging the
        kernel. A fresh tensor forces the safe two-D2H fallback.

        Only decode rows ``[num_extends:bs]`` carry an accept delta; prefill rows
        pass through. Deterministic, so no cross-rank divergence.
        """
        if bs <= num_extends:
            return output_lengths
        decode_rpi = self.input_buffers.req_pool_indices_buf[num_extends:bs]
        committed = self.runtime_states.valid_cache_lengths.index_select(
            0, decode_rpi
        ).to(output_lengths.dtype)
        remaining = (self.config.context_len - committed).clamp_(min=0)
        capped_decode = torch.minimum(output_lengths[num_extends:bs], remaining)
        if num_extends == 0:
            return capped_decode
        return torch.cat([output_lengths[:num_extends], capped_decode])

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
            if self.config.spec_algo in ("DFLASH", "DSPARK"):
                accept_lengths = self._cap_accept_to_context_len(
                    accept_lengths, sampling_info.req_pool_indices[:num_decodes]
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
        if self.config.spec_algo in ("DFLASH", "DSPARK"):
            decode_accept = self._cap_accept_to_context_len(
                decode_accept, sampling_info.req_pool_indices[num_extends:]
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

    def accumulate_decode_stats(self, results: ModelExecutionResult, bs: int):
        """Accumulate decode stats from already-synced results. No GPU sync."""
        accept_lengths = results.output_lengths
        self.num_generated_tokens += int(accept_lengths.sum().item())
        self.num_decode_steps += bs
        if (
            LOG_SPEC_ACCEPT_LENGTHS
            and self.config.global_rank == 0
            and self.config.spec_num_steps
        ):
            accepted_widths = [int(value) for value in accept_lengths.tolist()]
            accepted_draft_tokens = [max(0, value - 1) for value in accepted_widths]
            logger.info(
                "Spec verify step. accept_lengths=%s, accepted_draft_tokens=%s",
                accepted_widths,
                accepted_draft_tokens,
            )
            candidates = getattr(results, "spec_candidate_tokens", None)
            if candidates is not None:
                verify_width = int(self.config.spec_num_tokens)
                candidate_rows = candidates.view(bs, verify_width)
                target_rows = results.output_tokens.view(bs, verify_width)
                # Candidate column j+1 is verified by the target token sampled
                # from column j. The final target column is the bonus token.
                draft_rows = candidate_rows[:, 1:]
                target_draft_rows = target_rows[:, :-1]
                logger.info(
                    "Spec token compare. anchor=%s, draft=%s, target=%s, match=%s",
                    candidate_rows[:, 0].tolist(),
                    draft_rows.tolist(),
                    target_draft_rows.tolist(),
                    draft_rows.eq(target_draft_rows).tolist(),
                )

    def execute_forward_op_with_log(
        self,
        forward_op,
        sampling_params_list: list[SamplingParams],
        num_active_pages: int = 0,
        num_cached_pages: int = 0,
        num_queue_reqs: int = 0,
        dp_global_num_tokens=None,
        dp_global_bs=None,
        dp_all_decode_or_idle: bool = False,
        dp_all_extend: bool = False,
        grammar_inputs=None,
        multimodal_context=None,
        capture_next_input_ids: bool = False,
    ) -> ModelExecutionResult:
        self.log_step += 1

        num_extends = forward_op.num_extends()
        bs = len(forward_op.request_ids)
        is_decode = num_extends <= 0

        if not is_decode and self.config.global_rank == 0:
            mode = "Prefill" if num_extends == bs else "Mix"
            total_tokens = sum(forward_op.input_lengths)
            cached_tokens = sum(
                pl
                for rid, pl in zip(
                    forward_op.request_ids[:num_extends],
                    forward_op.extend_prefix_lens,
                )
                if rid not in self._seen_prefill_ids
            )
            if len(self._seen_prefill_ids) > 100_000:
                self._seen_prefill_ids.clear()  # log-dedup only; bound the growth
            self._seen_prefill_ids.update(forward_op.request_ids[:num_extends])
            logger.info(
                "%s batch. #new-seq: %s, #new-token: %s, #cached-token: %s, "
                "#running-req: %s, #queue-req: %s",
                mode,
                num_extends,
                total_tokens,
                cached_tokens,
                bs,
                num_queue_reqs,
            )

        result = self.execute_forward_op(
            forward_op,
            sampling_params_list,
            dp_global_num_tokens,
            dp_global_bs,
            dp_all_decode_or_idle,
            dp_all_extend,
            grammar_inputs=grammar_inputs,
            multimodal_context=multimodal_context,
            capture_next_input_ids=capture_next_input_ids,
        )

        if is_decode and (
            self.config.global_rank == 0
            and self.log_step % self.config.decode_log_interval == 0
        ):
            now = time.time()
            gap = now - self.last_decode_stats_tic
            gen_throughput = self.num_generated_tokens / gap if gap > 0 else 0
            avg_accept = (
                self.num_generated_tokens / self.num_decode_steps
                if self.num_decode_steps > 0
                else 0
            )
            accept_rate = (
                (avg_accept - 1) / self.config.spec_num_steps
                if self.config.spec_num_steps
                else 0
            )
            num_total_pages = self.config.num_total_pages
            page_ratio = (
                num_active_pages / num_total_pages if num_total_pages > 0 else 0
            )
            if self.config.spec_num_steps:
                logger.info(
                    "Decode batch. #running-req: %s, "
                    "#pages(active/cached/total): %s/%s/%s, "
                    "page ratio: %.2f, gen throughput (token/s): %.2f, "
                    "avg_accept_len: %.2f, accept_rate: %.2f, #queue-req: %s",
                    bs,
                    num_active_pages,
                    num_cached_pages,
                    num_total_pages,
                    page_ratio,
                    gen_throughput,
                    avg_accept,
                    accept_rate,
                    num_queue_reqs,
                )
            else:
                logger.info(
                    "Decode batch. #running-req: %s, "
                    "#pages(active/cached/total): %s/%s/%s, "
                    "page ratio: %.2f, gen throughput (token/s): %.2f, "
                    "#queue-req: %s",
                    bs,
                    num_active_pages,
                    num_cached_pages,
                    num_total_pages,
                    page_ratio,
                    gen_throughput,
                    num_queue_reqs,
                )
            self.token_to_kv_pool.maybe_log_paged_cache_group_pages()
            self.num_generated_tokens = 0
            self.num_decode_steps = 0
            self.last_decode_stats_tic = now

        return result

    def execute_idle_forward(
        self,
        global_num_tokens: list[int],
        global_bs: list[int],
        all_decode_or_idle: bool,
    ):
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
            global_num_tokens=global_num_tokens,
            global_bs=global_bs,
            all_decode_or_idle=all_decode_or_idle,
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
                    global_num_tokens,
                    global_bs,
                )
                draft_ctx = ForwardContext(
                    attn_backend=self.drafter.attn_backend,
                    token_to_kv_pool=self.drafter.token_to_kv_pool,
                    bs=0,
                    num_extends=0,
                    input_num_tokens=0,
                    forward_mode=ForwardMode.IDLE,
                    global_num_tokens=draft_global_num_tokens,
                    global_bs=global_bs,
                    all_decode_or_idle=all_decode_or_idle,
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
            zero_new_pages = getattr(pool, "zero_new_pages", None)
            zero_pages = getattr(pool, "zero_pages", None)
            if isinstance(pool_pages, Mapping) and callable(zero_new_pages):
                zero_new_pages(pool_pages)
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
            if getattr(pool, "paged_cache_requires_page_zeroing", False):
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
                "paged_cache_requires_page_zeroing",
                False,
            ):
                draft_pages = pages
                if isinstance(pages, Mapping):
                    draft_group_ids = {
                        str(spec.group_id)
                        for spec in draft_pool.paged_cache_group_specs
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
        if torch.device(self.device).type != "cuda":
            return None
        done = torch.cuda.Event()
        done.record(torch.cuda.current_stream(self.device))
        return done

    @nvtx_range("reset_valid_cache_length", color="orange")
    def reset_valid_cache_length(self, forward_op) -> None:
        num_extends = forward_op.num_extends()
        if num_extends == 0:
            return

        self.execution_stream.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(self.execution_stream):
            extend_request_pool_indices = torch.tensor(
                forward_op.request_pool_indices[:num_extends],
                dtype=torch.int64,
                device="cpu",
                pin_memory=True,
            ).to(self.device, non_blocking=True)

            extend_prefix_lens = torch.tensor(
                forward_op.extend_prefix_lens,
                dtype=torch.int32,
                device="cpu",
                pin_memory=True,
            ).to(self.device, non_blocking=True)

            self.runtime_states.reset_states(
                extend_request_pool_indices, extend_prefix_lens
            )

    def execute_forward_op(
        self,
        forward_op,
        sampling_params_list: list[SamplingParams],
        dp_global_num_tokens=None,
        dp_global_bs=None,
        dp_all_decode_or_idle: bool = False,
        dp_all_extend: bool = False,
        grammar_inputs=None,
        multimodal_context=None,
        capture_next_input_ids: bool = False,
    ) -> ModelExecutionResult:
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
            torch.cuda.current_stream().wait_stream(self.execution_stream)
            self.execution_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.execution_stream):
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
                page_table=page_table,
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
                    if dp_global_num_tokens is None:
                        raise RuntimeError(
                            "DP forward metadata must be gathered on CPU by "
                            "the event loop before model execution."
                        )
                    ctx.global_num_tokens = dp_global_num_tokens
                    ctx.global_bs = dp_global_bs
                    ctx.all_decode_or_idle = dp_all_decode_or_idle
                    ctx.all_extend = dp_all_extend
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

                if self.config.spec_algo in ("DFLASH", "DSPARK"):
                    # Clamp the committed-length delta so no request grows past
                    # context_len. Done here (outside the graph) so it reaches
                    # both _update_runtime_state and the scheduler page
                    # reservation; see _clamp_committed_to_context_len.
                    output_lengths = self._clamp_committed_to_context_len(
                        output_lengths, num_extends, bs
                    )

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

                copy_event = torch.cuda.Event()
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
        with torch.cuda.stream(self.execution_stream):
            self.runtime_states.write_remote_spec_candidate_ids(
                req_pool_idx, candidate_ids
            )
