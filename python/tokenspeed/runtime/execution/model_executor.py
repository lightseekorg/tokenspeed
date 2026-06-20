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
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.configs.model_config import ModelConfig
from tokenspeed.runtime.configs.utils import get_rope_parameters
from tokenspeed.runtime.engine.scheduler_utils import (
    paged_cache_block_table_base_offsets_from_forward_op,
    paged_cache_block_tables_from_forward_op,
)
from tokenspeed.runtime.execution.cache_loc_kernel import update_block_table
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.cuda_graph_wrapper import CudaGraphWrapper
from tokenspeed.runtime.execution.drafter.eagle import Eagle
from tokenspeed.runtime.execution.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
)
from tokenspeed.runtime.execution.input_buffer import InputBuffers
from tokenspeed.runtime.execution.model_runner import ModelRunner
from tokenspeed.runtime.execution.nan_guard import NanGuard
from tokenspeed.runtime.execution.runtime_states import RuntimeStates
from tokenspeed.runtime.execution.types import ModelExecutionResult
from tokenspeed.runtime.grammar.capturable_grammar import setup_grammar_step
from tokenspeed.runtime.layers.logits_processor import LogitsProcessorOutput
from tokenspeed.runtime.models.llama_eagle3 import LlamaForCausalLMEagle3
from tokenspeed.runtime.models.qwen3_5_nextn import Qwen3_5ForConditionalGenerationNextN
from tokenspeed.runtime.sampling.backends.base import SamplingBackend
from tokenspeed.runtime.sampling.dp_sampling_config import (
    DpSamplingRuntimeLimits,
    DpSamplingTopology,
    resolve_dp_sampling_runtime,
    resolve_dp_sampling_support,
)
from tokenspeed.runtime.sampling.sampling_batch_info import SamplingBatchInfo
from tokenspeed.runtime.utils import get_colorful_logger, set_random_seed
from tokenspeed.runtime.utils.common import maybe_inference_mode
from tokenspeed.runtime.utils.env import envs
from tokenspeed.runtime.utils.nvtx import nvtx_range
from tokenspeed.runtime.utils.server_args import ServerArgs

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
    from tokenspeed.runtime.layers.attention.kv_cache.base import BaseTokenToKVPool
    from tokenspeed.runtime.sampling.sampling_params import SamplingParams

logger = get_colorful_logger(__name__)

_DRAFTER_MAPPING = {"EAGLE3": Eagle, "MTP": Eagle}


def _draft_idle_global_num_tokens_for_step(
    step_idx: int,
    global_num_tokens: list[int],
    global_bs: list[int] | None,
) -> list[int]:
    if step_idx == 0 or global_bs is None:
        return global_num_tokens
    return global_bs


@dataclass
class ModelExecutorConfig:
    """
    Scalar configuration for ModelExecutor.
    Contains only primitive values — no heavy objects.
    Created once via from_server_args() and injected into ModelExecutor.
    """

    max_req_pool_size: int
    output_length: int
    enforce_eager: bool
    block_size: int
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
    dp_sampling: bool = False
    dp_sampling_min_bs: int | None = None
    use_target_verify_forward_mode: bool = False

    # ====== GRAMMAR =========
    # "none" disables all grammar handling; otherwise the backend name
    # (currently only "xgrammar" is implemented).
    grammar_backend: str = "xgrammar"
    # Force the synchronous eager grammar fallback even on CUDA. For
    # parity-testing the captured-grammar path.
    disable_capturable_grammar: bool = False
    mamba_cache_chunk_size: int = 64

    @staticmethod
    def from_server_args(
        server_args: ServerArgs,
        model_config: ModelConfig,
        max_req_pool_size: int,
        gpu_id: int,
        global_rank: int,
        num_total_pages: int,
    ) -> ModelExecutorConfig:
        output_length = (
            server_args.speculative_num_draft_tokens
            if server_args.speculative_algorithm
            else 1
        )
        rope_parameters = get_rope_parameters(model_config.hf_text_config)
        model_is_mrope = bool(rope_parameters and "mrope_section" in rope_parameters)

        return ModelExecutorConfig(
            max_req_pool_size=max_req_pool_size,
            output_length=output_length,
            enforce_eager=server_args.enforce_eager,
            block_size=server_args.block_size,
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
            model_is_mrope=model_is_mrope,
            data_parallel_size=server_args.mapping.attn.dp_size,
            world_size=server_args.mapping.world_size,
            world_group=server_args.mapping.world_group,
            spec_algo=server_args.speculative_algorithm,
            spec_num_steps=server_args.speculative_num_steps,
            spec_num_tokens=server_args.speculative_num_draft_tokens,
            dp_sampling=server_args.dp_sampling,
            dp_sampling_min_bs=server_args.dp_sampling_min_bs,
            enable_nan_detection=server_args.enable_nan_detection,
            use_target_verify_forward_mode=model_config.use_target_verify_forward_mode,
            grammar_backend=server_args.grammar_backend,
            disable_capturable_grammar=server_args.disable_capturable_grammar,
            mamba_cache_chunk_size=server_args.mamba_cache_chunk_size,
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
        token_to_kv_pool: BaseTokenToKVPool,
        sampling_backend: SamplingBackend,
        draft_model_runner: ModelRunner | None = None,
        draft_attn_backend: AttentionBackend | None = None,
        draft_token_to_kv_pool: BaseTokenToKVPool | None = None,
        mamba_pool: object | None = None,
    ):
        self.device = config.device
        self.config = config
        self.model_runner = model_runner
        self.sampling_backend = sampling_backend
        self.attn_backend = attn_backend
        self.token_to_kv_pool = token_to_kv_pool
        self.draft_attn_backend = draft_attn_backend
        self.draft_token_to_kv_pool = draft_token_to_kv_pool
        self._layerwise_mamba_cow_done = None

        if config.spec_algo is not None:
            max_num_pages_per_req = (
                config.context_len + config.spec_num_tokens + config.block_size - 1
            ) // config.block_size
        else:
            max_num_pages_per_req = (
                config.context_len + config.block_size
            ) // config.block_size

        self.req_to_page = torch.zeros(
            (config.max_req_pool_size + 1, max_num_pages_per_req),
            dtype=torch.int32,
            device=self.device,
        )
        spec_num_tokens = config.spec_num_tokens if config.spec_algo is not None else 1
        self.input_buffers = InputBuffers(
            max_bs=config.max_num_seqs // max(config.data_parallel_size, 1),
            max_num_tokens=config.chunked_prefill_size,
            page_size=config.block_size,
            # token_to_kv_pool allocates size+page_size slots; index `size` is
            # the reserved dummy slot (see MHATokenToKVPool._create_buffers).
            dummy_kv_slot=0,
            device=self.device,
            has_mamba=(mamba_pool is not None),
        )
        self.runtime_states = RuntimeStates(
            req_pool_size=config.max_req_pool_size,
            context_len=config.context_len,
            vocab_size=config.vocab_size,
            device=self.device,
            output_length=config.output_length,
            mamba_pool=mamba_pool,
        )
        # Sized like InputBuffers.max_bs so the padded graph-bucket bs fits.
        self.nan_guard = NanGuard.create(
            config.enable_nan_detection,
            config.max_num_seqs // max(config.data_parallel_size, 1),
            self.device,
        )
        if self.config.spec_algo is not None:
            DrafterImpl = _DRAFTER_MAPPING[config.spec_algo]
            self.drafter = DrafterImpl(
                spec_num_tokens=config.spec_num_tokens,
                spec_num_steps=config.spec_num_steps,
                draft_model_runner=draft_model_runner,
                page_size=config.block_size,
                runtime_states=self.runtime_states,
                input_buffers=self.input_buffers,
                req_to_page=self.req_to_page,
                attn_backend=draft_attn_backend,
                token_to_kv_pool=draft_token_to_kv_pool,
                vocab_size=config.vocab_size,
            )
            embed, head = self.model_runner.model.get_embed_and_head()
            draft_model_runner.model.set_embed_and_head(embed, head)
            target_hf = self.model_runner.model_config.hf_config
            mm_pad_substitute_id = getattr(
                target_hf, "image_token_id", None
            ) or getattr(target_hf, "media_placeholder_token_id", None)
            if mm_pad_substitute_id is not None:
                self.drafter.set_mm_pad_substitute_id(mm_pad_substitute_id)
            if config.spec_algo in ("EAGLE3",) and hasattr(
                self.model_runner.model, "set_eagle3_layers_to_capture"
            ):
                self.model_runner.model.set_eagle3_layers_to_capture()
        else:
            self.drafter = None

        # Single grammar handle: CapturableGrammarExecutor on CUDA (uses
        # cudaLaunchHostFunc on a side stream so the xgrammar fill +
        # H2D overlap with the forward, and is also CUDA-graph-capturable),
        # EagerGrammarBuffers on non-CUDA (synchronous fallback).
        # ``disable_capturable_grammar`` forces the eager path on CUDA too
        # for parity-testing.
        self.grammar_runtime = None
        if config.grammar_backend != "none":
            from tokenspeed.runtime.grammar.capturable_grammar import (
                CapturableGrammarExecutor,
                EagerGrammarBuffers,
            )

            use_captured = (
                current_platform().is_nvidia and not config.disable_capturable_grammar
            )
            if use_captured:
                self.grammar_runtime = CapturableGrammarExecutor(
                    max_bs=config.max_num_seqs,
                    vocab_size=config.vocab_size,
                    max_tokens_per_req=spec_num_tokens,
                    device=self.device,
                )
            else:
                self.grammar_runtime = EagerGrammarBuffers(
                    max_bs=config.max_num_seqs // max(config.data_parallel_size, 1),
                    vocab_size=config.vocab_size,
                    max_tokens_per_req=spec_num_tokens,
                    device=self.device,
                )

        attn_backend.configure_runtime(
            sliding_window_size=model_runner.sliding_window_size,
            req_to_page=self.req_to_page,
        )
        if draft_attn_backend is not None:
            draft_attn_backend.configure_runtime(
                sliding_window_size=model_runner.sliding_window_size,
                req_to_page=self.req_to_page,
            )

        processor = self.model_runner.model.logits_processor
        dp_topology = DpSamplingTopology(
            tp_rank=processor.tp_rank,
            tp_size=processor.tp_size,
            tp_group=processor.tp_group,
            skip_all_gather=processor.skip_all_gather,
            tie_word_embeddings=bool(
                getattr(processor.config, "tie_word_embeddings", False)
            ),
        )
        dp_support = resolve_dp_sampling_support(
            requested=self.config.dp_sampling,
            drafter_available=self.drafter is not None,
            backend_supports_verify=bool(
                getattr(self.sampling_backend, "_SUPPORTS_DP_VERIFY", False)
            ),
            topology=dp_topology,
        )

        lm_head_rows = 0
        if dp_support.enabled:
            lm_head_weight = self.model_runner.model.lm_head.weight
            if lm_head_weight.ndim < 1:
                raise RuntimeError(
                    "dp_sampling LM head weight must be at least 1D, got "
                    f"{lm_head_weight.ndim}D"
                )
            lm_head_rows = int(lm_head_weight.shape[0])
        dp_runtime_config = resolve_dp_sampling_runtime(
            support=dp_support,
            lm_head_rows=lm_head_rows,
            topology=dp_topology,
            limits=DpSamplingRuntimeLimits(
                runtime_vocab_size=self.config.vocab_size,
                max_num_seqs=config.max_num_seqs,
                data_parallel_size=config.data_parallel_size,
                num_tokens_per_req=spec_num_tokens,
                configured_min_bs=self.config.dp_sampling_min_bs,
                device=self.device,
            ),
        )
        self.dp_sampling_runtime_config = dp_runtime_config
        self._last_dp_sampling_route_log: (
            tuple[str, int, bool, int, int, bool, int] | None
        ) = None
        if dp_runtime_config.enabled:
            self.sampling_backend.configure_dp_sampling(dp_runtime_config)
            processor.configure_dp_logits_layout(dp_runtime_config)
        logger.info(
            "Batch-DP spec-verify: requested=%s, infra_supports=%s, enabled=%s "
            "min_bs=%s (drafter=%s, backend_supports_dp=%s, "
            "tp_size=%s, tp_group=%s)",
            dp_support.requested,
            dp_support.infra_supports,
            dp_support.enabled,
            dp_runtime_config.min_bs,
            dp_support.drafter_available,
            dp_support.backend_supports_verify,
            dp_support.tp_size,
            dp_support.tp_group_set,
        )

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

        # CUDA graph warmup runs the drafter with dummy data whose
        # uninitialized KV cache produces NaN → argmax sentinel -1.
        # Clear the residue so pool slots reused by real requests start clean.
        self.runtime_states.future_input_map.zero_()

        # Encoder CUDA graph: install the model-built wrapper by overriding
        # ``image_encoder``. Vision-encoder analogue of ``forward_step``'s
        # ``CudaGraphWrapper``.
        self.encoder_graph_wrapper = None
        _mm_model = self.model_runner.model
        if (
            hasattr(_mm_model, "make_encoder_cudagraph_wrapper")
            and getattr(_mm_model, "is_multimodal_active", True)
            and envs.TOKENSPEED_MM_ENABLE_ENCODER_CUDA_GRAPH.get()
            and self.model_runner.server_args.mm_attention_backend != "flashinfer_cudnn"
        ):
            self.encoder_graph_wrapper = _mm_model.make_encoder_cudagraph_wrapper(
                _mm_model.mapping
            )
            _mm_model.image_encoder = self.encoder_graph_wrapper

        self.execution_stream = torch.cuda.Stream()
        self.log_step = 0
        self._seen_prefill_ids: set[str] = set()
        self._prev_decode_bs: int = 0
        self._sentinel_neg1 = torch.tensor(-1, device=self.device, dtype=torch.int64)
        # Decode stats — accumulated from synced results (no GPU sync needed)
        self.num_generated_tokens = 0
        self.num_decode_steps = 0
        self.last_decode_stats_tic = time.time()

        set_random_seed(48)

        logger.info("ModelExecutor initialized")

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
            or not (ctx.forward_mode.is_decode() or ctx.forward_mode.is_target_verify())
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
            is_spec_verify = n > 1 and (
                ctx.forward_mode.is_decode() or ctx.forward_mode.is_target_verify()
            )
            slice_ = (
                self.input_buffers.input_ids_buf[: bs * n] if is_spec_verify else None
            )
            self.capturable_grammar.schedule_fill(input_ids_buf_slice=slice_)

        logits_output = self._run_target_forward(bs, ctx, req_pool_indices)

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

        # Single choke point for every (sample/verify) x (greedy/flashinfer)
        # path: NaN logits (e.g. DP dummy/warmup batches over uninitialized KV)
        # make sampling argmax emit the -1 sentinel. Left unclamped, that -1 poisons
        # both consumers below -- it flows into output_ids -> detokenizer (the
        # HF Rust tokenizer raises OverflowError on a negative id and kills the
        # engine) and, via the drafter, back into future_input_map as the next
        # round's input_ids (negative embedding index -> garbage / CUDA illegal
        # access). Clamp in place (this runs inside the CUDA graph) so grammar,
        # drafter, and the return value all observe clean ids. Mirror of the
        # draft-side draft_ids.clamp_(min=0) guard.
        # Record which requests produced bad ids before the clamp erases them.
        self.nan_guard.merge_oov(output_tokens, ctx, self.runtime_states.vocab_size)
        output_tokens.clamp_(min=0)

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
                self.input_buffers.req_pool_indices_buf[: ctx.bs]
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
        self.num_generated_tokens += int(results.output_lengths.sum().item())
        self.num_decode_steps += bs

    @staticmethod
    @torch.compile(dynamic=True)
    def _compute_mtp_snapshot_indices(
        valid_cache_lengths: torch.Tensor,
        req_pool_indices: torch.Tensor,
        accept_lengths: torch.Tensor,
        output_indices: torch.Tensor,
        track_indices: torch.Tensor,
        sentinel: torch.Tensor,
        page_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fused elementwise pipeline computing snapshot src/dst for MTP.

        All operations are batched and fused by torch.compile into a single
        Triton kernel (plus the two gathers), eliminating the ~14 individual
        elementwise kernel launches of the eager implementation.
        """
        new_cl = valid_cache_lengths[req_pool_indices]
        old_cl = new_cl - accept_lengths.to(new_cl.dtype)
        first_boundary = ((old_cl // page_size) + 1) * page_size

        step_raw = first_boundary - old_cl - 1
        max_col = output_indices.shape[1] - 1
        step = step_raw.clamp(min=0, max=max_col).to(torch.int64)

        bs = req_pool_indices.shape[0]
        req_range = torch.arange(bs, device=req_pool_indices.device)
        src_raw = output_indices[req_range, step].to(torch.int64)
        dst_raw = track_indices.to(torch.int64)

        invalid = (
            (first_boundary > new_cl)
            | (dst_raw < 0)
            | (src_raw < 0)
            | (src_raw == dst_raw)
            | (step_raw < 0)
        )
        src = torch.where(invalid, sentinel, src_raw)
        dst = torch.where(invalid, sentinel, dst_raw)
        return src, dst

    def _snapshot_mamba_checkpoints(
        self,
        accept_lengths: torch.Tensor,
        bs: int,
        num_extends: int,
    ) -> None:
        """Snapshot mamba states to checkpoint slots at page boundaries.

        Called after ``_update_runtime_state`` on the execution stream so
        ``valid_cache_lengths`` already reflects the accepted tokens.

        Non-MTP (accept_length == 1):
            The working slot holds the up-to-date state for the new
            cache_length.  Pass the kernel page_size so it copies only
            when the new length is page-aligned.

        MTP (accept_length > 1):
            cache_length may jump over a page boundary.  The intermediate
            state lives in ``mamba_output_indices[req, step]``.  Boundary
            detection and source-slot selection are done entirely on GPU
            with -1 sentinels so the snapshot kernel skips invalid entries
            via its bounds check — no GPU-to-CPU sync, preserving
            overlap-schedule pipelining.
        """
        if self.runtime_states.mamba_pool is None or num_extends > 0:
            return
        if not self.input_buffers.has_mamba:
            return

        req_pool_indices = self.input_buffers.req_pool_indices_buf[:bs]
        track_indices = self.input_buffers.mamba_track_pool_indices_buf[:bs]
        page_size = self.config.block_size
        dev = req_pool_indices.device
        sentinel = self._sentinel_neg1

        if self.drafter is not None:
            # -- MTP path: find the output slot at the crossed boundary --
            backend = getattr(
                self.attn_backend, "linear_attn_backend", self.attn_backend
            )
            fm = getattr(backend, "forward_metadata", None)
            if fm is None:
                return
            output_indices = fm.mamba_output_indices
            if output_indices is None:
                return

            src, dst = self._compute_mtp_snapshot_indices(
                self.runtime_states.valid_cache_lengths,
                req_pool_indices,
                accept_lengths[:bs].to(device=dev),
                output_indices,
                track_indices,
                sentinel,
                page_size,
            )

            self.runtime_states.snapshot_mamba_checkpoints(
                src,
                dst,
                cache_lengths=None,
                page_size=0,
                num_valid=bs,
            )
        else:
            # -- Non-MTP path: working slot IS the up-to-date state --
            src_raw = self.input_buffers.mamba_pool_indices_buf[:bs].to(
                device=dev, dtype=torch.int64
            )
            dst_raw = track_indices.to(device=dev, dtype=torch.int64)

            invalid = (src_raw < 0) | (dst_raw < 0) | (src_raw == dst_raw)
            src = torch.where(invalid, sentinel, src_raw)
            dst = torch.where(invalid, sentinel, dst_raw)

            cache_lengths = self.runtime_states.valid_cache_lengths[req_pool_indices]
            self.runtime_states.snapshot_mamba_checkpoints(
                src,
                dst,
                cache_lengths=cache_lengths,
                page_size=page_size,
                num_valid=bs,
            )

    def flush_mamba_draft_to_working_on_retract(self) -> None:
        """Copy accepted draft mamba state -> working slot for all previous-batch requests.

        Called from event_loop when retract WriteBackOps are detected.
        Uses the previous decode iteration's input_buffers (still valid since
        no new forward has overwritten them).
        Runs on execution_stream to respect ordering with previous forward writes.
        """
        bs = self._prev_decode_bs
        if bs <= 0:
            return

        backend = getattr(self.attn_backend, "linear_attn_backend", self.attn_backend)
        pool = getattr(backend, "pool", None)
        if pool is None:
            return

        sentinel = self._sentinel_neg1

        with torch.cuda.stream(self.execution_stream):
            req_pool_indices = self.input_buffers.req_pool_indices_buf[:bs]
            working = self.input_buffers.mamba_pool_indices_buf[:bs]

            src_raw = pool.current_input_indices[req_pool_indices.clamp(0).long()].to(
                dtype=torch.int64
            )
            dst_raw = working.to(dtype=torch.int64)

            invalid = (src_raw < 0) | (dst_raw < 0) | (src_raw == dst_raw)
            src = torch.where(invalid, sentinel, src_raw)
            dst = torch.where(invalid, sentinel, dst_raw)

            self.runtime_states.snapshot_mamba_checkpoints(
                src,
                dst,
                cache_lengths=None,
                page_size=0,
                num_valid=bs,
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
        graph_forward_mode = ForwardMode.decode_or_target_verify(
            has_drafter=self.drafter is not None,
            use_target_verify=self.config.use_target_verify_forward_mode,
        )
        ctx = ForwardContext(
            attn_backend=self.attn_backend,
            token_to_kv_pool=self.token_to_kv_pool,
            req_to_page=self.req_to_page,
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
                    req_to_page=self.req_to_page,
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
            for step_idx in range(self.drafter.spec_num_steps):
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
                    req_to_page=self.drafter.req_to_page,
                    bs=0,
                    num_extends=0,
                    input_num_tokens=0,
                    forward_mode=ForwardMode.IDLE,
                    global_num_tokens=draft_global_num_tokens,
                    global_bs=global_bs,
                    all_decode_or_idle=all_decode_or_idle,
                    # Mirror the active-rank broaden in eagle.py: Llama Eagle3
                    # and Qwen3.5 NextN narrow for any non-idle catch-up, so the
                    # idle peer must size collectives the same way.
                    draft_first_step_reduce=(
                        step_idx == 0
                        and (
                            all_decode_or_idle
                            or isinstance(
                                self.drafter.draft_model_runner.model,
                                (
                                    LlamaForCausalLMEagle3,
                                    Qwen3_5ForConditionalGenerationNextN,
                                ),
                            )
                        )
                    ),
                )
                self.drafter.draft_model_runner.forward(
                    draft_ctx,
                    input_ids=empty,
                    positions=empty,
                    out_cache_loc=empty,
                    spec_step_idx=step_idx,
                )

    def update_block_table(self, forward_op) -> ModelExecutionResult:
        # Update page tables on the default stream before switching to execution stream.
        # HostTodevice segment begins
        with nvtx_range("update_block_table", color="cyan"):
            update_block_table(
                forward_op=forward_op,
                device=self.device,
                req_to_page=self.req_to_page,
            )

    def reset_remote_prefill_mamba_inputs(self, forward_op) -> None:
        if self.runtime_states.mamba_pool is None:
            return
        if not hasattr(self.attn_backend, "reset_current_inputs"):
            return

        num_extends = forward_op.num_extends()
        if num_extends <= 0:
            return

        mamba_indices = list(getattr(forward_op, "mamba_pool_indices", []))
        if not mamba_indices:
            return

        req_pool_indices = list(forward_op.request_pool_indices[:num_extends])
        pairs = [
            (int(req_pool_idx), int(mamba_idx))
            for req_pool_idx, mamba_idx in zip(
                req_pool_indices, mamba_indices[:num_extends]
            )
            if int(mamba_idx) >= 0
        ]
        if not pairs:
            return

        req_pool_tensor = torch.tensor(
            [req_pool_idx for req_pool_idx, _ in pairs],
            dtype=torch.int64,
            device="cpu",
            pin_memory=True,
        ).to(self.device, non_blocking=True)
        mamba_tensor = torch.tensor(
            [mamba_idx for _, mamba_idx in pairs],
            dtype=torch.int64,
            device="cpu",
            pin_memory=True,
        ).to(self.device, non_blocking=True)
        self.attn_backend.reset_current_inputs(req_pool_tensor, mamba_tensor)

    @nvtx_range("reset_valid_cache_length", color="orange")
    def reset_valid_cache_length(self, forward_op) -> None:

        num_extends = forward_op.num_extends()
        is_prefill = num_extends > 0

        # Retraction recovery: scheduler pushes -1 per decode op, overriding to
        # a real length only on ScheduleDecodeFromRetractedEvent.
        has_retract = not is_prefill and any(
            x != -1 for x in forward_op.hist_token_lens
        )

        # Pure decode without retraction has nothing to do — skip the
        # cross-stream wait + stream-context entry entirely.
        if not is_prefill and not has_retract:
            return

        if has_retract:
            hist_token_lens_tensor = torch.tensor(
                forward_op.hist_token_lens,
                dtype=torch.int32,
                device="cpu",
                pin_memory=True,
            )
            all_pool_indices = torch.tensor(
                forward_op.request_pool_indices,
                dtype=torch.int64,
                device="cpu",
                pin_memory=True,
            )
        else:
            hist_token_lens_tensor = None
            all_pool_indices = None

        self.execution_stream.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(self.execution_stream):
            if is_prefill:
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

            elif hist_token_lens_tensor is not None:
                # Apply retraction recovery: override valid_cache_lengths with hist_token_lens
                # where the scheduler has specified a non-(-1) value, so that out_cache_loc
                # and position IDs are computed against the retracted KV length.
                pool_idx_dev = all_pool_indices.to(self.device, non_blocking=True)
                hist_dev = hist_token_lens_tensor.to(self.device, non_blocking=True)

                mask_1d = hist_dev != -1
                vcl = self.runtime_states.valid_cache_lengths[pool_idx_dev]

                self.runtime_states.valid_cache_lengths[pool_idx_dev] = torch.where(
                    mask_1d, hist_dev, vcl
                )

    def set_layerwise_mamba_cow_done(
        self, cow_by_src: dict[int, list[int]] | None
    ) -> None:
        self._layerwise_mamba_cow_done = (
            {
                int(src): {int(dst) for dst in dsts}
                for src, dsts in cow_by_src.items()
                if dsts
            }
            if cow_by_src
            else None
        )

    def _skip_completed_layerwise_mamba_cow(
        self, forward_op, bs: int
    ) -> torch.Tensor | None:
        cow_done = self._layerwise_mamba_cow_done
        self._layerwise_mamba_cow_done = None
        if not cow_done or not getattr(self.input_buffers, "has_mamba", False):
            return None
        cow_src_indices = getattr(forward_op, "mamba_cow_src_indices", None)
        working_indices = getattr(forward_op, "mamba_pool_indices", None)
        if cow_src_indices is None or working_indices is None:
            return None

        cow_src_indices = list(cow_src_indices)[:bs]
        working_indices = list(working_indices)[:bs]
        skipped_mask = [False] * bs
        changed = False
        for i, (cow_src, working) in enumerate(zip(cow_src_indices, working_indices)):
            cow_src = int(cow_src)
            working = int(working)
            if working in cow_done.get(cow_src, set()):
                cow_src_indices[i] = -1
                skipped_mask[i] = True
                changed = True
        if not changed:
            return None

        self.input_buffers._mamba_cow_src_indices_cpu[:bs].copy_(
            torch.as_tensor(cow_src_indices, dtype=torch.int32)
        )
        cow_src_buf = self.input_buffers.mamba_cow_src_indices_buf
        cow_src_buf[:bs].copy_(
            self.input_buffers._mamba_cow_src_indices_cpu[:bs], non_blocking=True
        )
        return torch.tensor(skipped_mask, dtype=torch.bool, device=cow_src_buf.device)

    @staticmethod
    def _mamba_retract_reset_mask(
        mamba_cow_src: torch.Tensor,
        bs: int,
        skipped_layerwise_cow_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        reset_mask = mamba_cow_src[:bs] >= 0
        if skipped_layerwise_cow_mask is not None:
            reset_mask = reset_mask | skipped_layerwise_cow_mask[:bs].to(
                device=reset_mask.device, dtype=torch.bool
            )
        return reset_mask

    def execute_forward_op(
        self,
        forward_op,
        sampling_params_list: list[SamplingParams],
        dp_global_num_tokens=None,
        dp_global_bs=None,
        dp_all_decode_or_idle: bool = False,
        grammar_inputs=None,
        multimodal_context=None,
        capture_next_input_ids: bool = False,
    ) -> ModelExecutionResult:
        num_extends = forward_op.num_extends()
        total_tokens = sum(forward_op.input_lengths)
        self._active_multimodal_context = multimodal_context
        self._active_positions_override = None

        with nvtx_range("pre_fill_setup", color="orange"):
            has_retract = num_extends <= 0 and any(
                x != -1 for x in getattr(forward_op, "hist_token_lens", [])
            )

            # Wait for previous iteration's runtime state updates
            # (future_input_map, valid_cache_lengths) on execution_stream to
            # complete before reading them.
            torch.cuda.current_stream().wait_stream(self.execution_stream)
            self.execution_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.execution_stream):
            bs = len(forward_op.request_ids)
            # Outside the graph: in-graph sites only OR into the flag buffer.
            self.nan_guard.reset(bs)
            decode_input_ids = self.input_buffers.fill_input_buffers(
                forward_op=forward_op,
                runtime_states=self.runtime_states,
                req_to_page=self.req_to_page,
                total_tokens=total_tokens,
            )
            skipped_layerwise_cow_mask = self._skip_completed_layerwise_mamba_cow(
                forward_op, bs
            )
            self._active_positions_override = self._build_mrope_positions_override(
                forward_op=forward_op,
                multimodal_context=multimodal_context,
                total_tokens=total_tokens,
            )

            forward_mode = ForwardMode.from_num_extends(
                num_extends,
                bs,
                has_drafter=self.drafter is not None,
                use_target_verify=self.config.use_target_verify_forward_mode,
            )

            if num_extends <= 0:
                self._prev_decode_bs = bs

            if self.runtime_states.mamba_pool is not None and (
                num_extends > 0 or has_retract
            ):
                mamba_pool_indices = self.input_buffers.mamba_pool_indices_buf[:bs]
                mamba_cow_src = self.input_buffers.mamba_cow_src_indices_buf[:bs]
                self.runtime_states.copy_mamba_states(
                    mamba_pool_indices, mamba_cow_src, bs
                )
                if num_extends > 0:
                    self.runtime_states.zero_mamba_states(
                        mamba_pool_indices,
                        mamba_cow_src,
                        self.input_buffers.extend_prefix_lens_buf[:num_extends],
                        num_extends,
                    )
                    if hasattr(self.attn_backend, "reset_current_inputs"):
                        self.attn_backend.reset_current_inputs(
                            self.input_buffers.req_pool_indices_buf[:num_extends],
                            mamba_pool_indices[:num_extends],
                        )
                elif has_retract:
                    if hasattr(self.attn_backend, "reset_current_inputs"):
                        retract_mask = self._mamba_retract_reset_mask(
                            mamba_cow_src,
                            bs,
                            skipped_layerwise_cow_mask,
                        )
                        self.attn_backend.reset_current_inputs(
                            self.input_buffers.req_pool_indices_buf[:bs][retract_mask],
                            mamba_pool_indices[:bs][retract_mask],
                        )

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
                    req_to_page=self.req_to_page,
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
                with nvtx_range("sampling_prep", color="yellow"):
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

                with nvtx_range(
                    f"forward_step ext={num_extends} dec={bs - num_extends}",
                    color="blue",
                ):
                    mamba_kwargs = (
                        {
                            "mamba_pool_indices": self.input_buffers.mamba_pool_indices_buf[
                                :bs
                            ],
                            "mamba_cow_src_indices": self.input_buffers.mamba_cow_src_indices_buf[
                                :bs
                            ],
                            "mamba_branching_seqlens": self.input_buffers.mamba_branching_seqlens_buf[
                                :bs
                            ],
                            "mamba_track_pool_indices": self.input_buffers.mamba_track_pool_indices_buf[
                                :bs
                            ],
                        }
                        if self.input_buffers.has_mamba
                        else {}
                    )
                    paged_cache_block_tables = paged_cache_block_tables_from_forward_op(
                        forward_op,
                        device=self.device,
                        num_reqs=bs,
                    )
                    (
                        paged_cache_block_table_base_offsets,
                        _paged_cache_block_table_base_offset_max,
                    ) = paged_cache_block_table_base_offsets_from_forward_op(
                        forward_op,
                        device=self.device,
                        num_reqs=bs,
                    )
                    self._log_dp_sampling_route(bs, ctx)
                    output_tokens, output_lengths, output_logprobs = self.forward_step(
                        bs=bs,
                        ctx=ctx,
                        sampling_info=sampling_info,
                        req_to_page=self.req_to_page,
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
                        paged_cache_block_tables=paged_cache_block_tables,
                        paged_cache_block_table_base_offsets=(
                            paged_cache_block_table_base_offsets
                        ),
                        **mamba_kwargs,
                    )

                # Update runtime state on execution_stream (NOT in the CUDA graph).
                self._update_runtime_state(
                    req_pool_indices=self.input_buffers.req_pool_indices_buf[:bs],
                    output_tokens=output_tokens,
                    accept_lengths=output_lengths,
                    input_lengths=self.input_buffers.input_lengths_buf[:bs],
                    num_extends=num_extends,
                )
                self._snapshot_mamba_checkpoints(
                    output_lengths,
                    bs,
                    num_extends,
                )

            with nvtx_range("output_d2h", color="green"):
                next_input_ids = None
                if (
                    capture_next_input_ids
                    and self.drafter is not None
                    and num_extends > 0
                ):
                    next_input_ids = self.runtime_states.future_input_map.index_select(
                        0, self.input_buffers.req_pool_indices_buf[:num_extends]
                    ).to("cpu", non_blocking=True)

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

        return ModelExecutionResult(
            output_tokens=output_tokens,
            output_lengths=output_lengths,
            output_logprobs=output_logprobs,
            copy_event=copy_event,
            grammar_completion=grammar_completion,
            next_input_ids=next_input_ids,
            output_nan_flags=output_nan_flags,
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

    def _expand_mrope_from_input(self, mm_input, seq_len: int) -> torch.Tensor:
        # Cache delta expansion for retracted/chunked requests.
        if mm_input.mrope_position_delta_repeated_cache is None:
            mm_input.mrope_position_delta_repeated_cache = (
                (mm_input.mrope_position_delta - 1).flatten().unsqueeze(0).repeat(3, 1)
            )
        return mm_input.mrope_position_delta_repeated_cache + seq_len

    def _build_mrope_positions_override(
        self,
        forward_op,
        multimodal_context,
        total_tokens: int,
    ) -> torch.Tensor | None:
        if not self.config.model_is_mrope or total_tokens == 0:
            return None

        is_prefill = forward_op.num_extends() > 0
        base_positions = self.input_buffers.positions_buf[:total_tokens]
        pos_chunks = torch.split(base_positions, list(forward_op.input_lengths), dim=0)
        mrope_chunks = []
        mm_inputs = (
            multimodal_context.mm_inputs
            if multimodal_context is not None and multimodal_context.has_inputs()
            else []
        )

        for batch_idx, base_chunk in enumerate(pos_chunks):
            mm_input = mm_inputs[batch_idx] if batch_idx < len(mm_inputs) else None
            if mm_input is None or mm_input.mrope_positions is None:
                mrope_chunks.append(base_chunk.unsqueeze(0).expand(3, -1))
                continue

            if is_prefill and batch_idx < len(forward_op.extend_prefix_lens):
                start = int(forward_op.extend_prefix_lens[batch_idx])
                end = start + int(forward_op.input_lengths[batch_idx])
                positions = mm_input.mrope_positions[:, start:end]
                if positions.numel() != 0:
                    mrope_chunks.append(
                        positions.to(device=self.device, dtype=torch.int64)
                    )
                    continue
                if base_chunk.numel() == 1:
                    seq_len = int(base_chunk[-1].item()) + 1
                    mrope_chunks.append(
                        self._expand_mrope_from_input(mm_input, seq_len).to(
                            device=self.device, dtype=torch.int64
                        )
                    )
                    continue

            delta = mm_input.mrope_position_delta
            if delta is None:
                delta = torch.zeros(1, dtype=torch.int64)
            delta = delta.flatten()[0].to(device=self.device, dtype=torch.int64)
            # Decode positions need (mrope_delta - 1) + seq_len. positions_buf
            # already stores the per-token zero-based position (seq_len - 1 for
            # decode), so this is the same value without a GPU-to-CPU sync.
            mrope_chunks.append((base_chunk + delta).unsqueeze(0).expand(3, -1))

        mrope_positions = torch.cat(mrope_chunks, dim=1).contiguous()
        self.input_buffers.mrope_positions_buf[:, :total_tokens].copy_(mrope_positions)
        return self.input_buffers.mrope_positions_buf[:, :total_tokens]
