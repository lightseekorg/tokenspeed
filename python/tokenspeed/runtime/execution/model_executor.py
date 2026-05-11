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
from tokenspeed.runtime.execution.runtime_states import RuntimeStates
from tokenspeed.runtime.execution.types import ModelExecutionResult
from tokenspeed.runtime.grammar.capturable_grammar import setup_grammar_step
from tokenspeed.runtime.sampling.backends.base import SamplingBackend
from tokenspeed.runtime.sampling.sampling_batch_info import SamplingBatchInfo
from tokenspeed.runtime.utils import get_colorful_logger, set_random_seed
from tokenspeed.runtime.utils.common import maybe_inference_mode
from tokenspeed.runtime.utils.nvtx import nvtx_range
from tokenspeed.runtime.utils.server_args import ServerArgs

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
    from tokenspeed.runtime.layers.attention.kv_cache.base import BaseTokenToKVPool
    from tokenspeed.runtime.sampling.sampling_params import SamplingParams

logger = get_colorful_logger(__name__)

_DRAFTER_MAPPING = {"EAGLE3": Eagle, "MTP": Eagle}


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

    # ====== DP =========
    data_parallel_size: int = 1
    world_size: int = 1
    world_group: list[int] | None = None

    # ====== SPEC =========
    spec_algo: str | None = None
    spec_num_steps: int | None = None
    # spec_num_tokens == spec_num_steps + 1 for now (without Tree Attention)
    spec_num_tokens: int | None = None

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
            data_parallel_size=server_args.mapping.attn.dp_size,
            world_size=server_args.mapping.world_size,
            world_group=server_args.mapping.world_group,
            spec_algo=server_args.speculative_algorithm,
            spec_num_steps=server_args.speculative_num_steps,
            spec_num_tokens=server_args.speculative_num_draft_tokens,
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

        self.execution_stream = torch.cuda.Stream()
        self.log_step = 0
        self._seen_prefill_ids: set[str] = set()
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
        return self.model_runner.forward(
            ctx,
            self.input_buffers.input_ids_buf[: ctx.input_num_tokens],
            self.input_buffers.positions_buf[: ctx.input_num_tokens],
            self.input_buffers.out_cache_loc_buf[: ctx.input_num_tokens],
            self.input_buffers.input_lengths_buf[:bs],
            req_pool_indices=req_pool_indices,
            seq_lens=self.input_buffers.seq_lens_buf[:bs],
            extend_prefix_lens=self.input_buffers.extend_prefix_lens_buf[
                : ctx.num_extends
            ],
        )

    @nvtx_range("sampling", color="yellow")
    def _run_sampling(
        self,
        logits_output,
        sampling_info: SamplingBatchInfo,
        ctx: ForwardContext,
        candidates,
    ):
        if self.drafter is not None and (
            ctx.forward_mode == ForwardMode.DECODE
            or ctx.forward_mode == ForwardMode.TARGET_VERIFY
        ):
            return self.sampling_backend.verify(
                logits_output, sampling_info, candidates
            )
        return self.sampling_backend.sample(logits_output, sampling_info)

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
                ctx.forward_mode.is_target_verify() or ctx.forward_mode.is_decode()
            )
            slice_ = (
                self.input_buffers.input_ids_buf[: bs * n] if is_spec_verify else None
            )
            self.capturable_grammar.schedule_fill(input_ids_buf_slice=slice_)

        logits_output = self._run_target_forward(bs, ctx, req_pool_indices)
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
        is_extend: bool,
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
            tokens_per_req = self.config.output_length if not is_extend else 1
            next_round_input_ids = output_tokens.to(torch.int32).reshape(
                -1, tokens_per_req
            )
            self.runtime_states.future_input_map[req_pool_indices, :tokens_per_req] = (
                next_round_input_ids
            )
        if is_extend:
            self.runtime_states.update_valid_cache_length(
                req_pool_indices, input_lengths
            )
        else:
            self.runtime_states.update_valid_cache_length(
                req_pool_indices, accept_lengths
            )

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

    def snapshot_mamba_checkpoints_for_op(self, forward_op) -> None:
        """Snapshot completed decode working states into checkpoint slots."""
        if self.runtime_states.mamba_pool is None or forward_op.num_extends() > 0:
            return
        if not getattr(forward_op, "mamba_pool_indices", None):
            return
        bs = len(forward_op.request_ids)
        mamba_pool_indices = torch.tensor(
            forward_op.mamba_pool_indices,
            dtype=torch.int32,
            device=self.device,
        )
        mamba_checkpoint_indices = torch.tensor(
            forward_op.mamba_track_pool_indices,
            dtype=torch.int32,
            device=self.device,
        )
        req_pool_indices = torch.tensor(
            forward_op.request_pool_indices,
            dtype=torch.int64,
            device=self.device,
        )
        cache_lengths = self.runtime_states.valid_cache_lengths[req_pool_indices]
        self.runtime_states.snapshot_mamba_checkpoints(
            mamba_pool_indices,
            mamba_checkpoint_indices,
            cache_lengths,
            self.config.block_size,
            bs,
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
        graph_forward_mode = (
            ForwardMode.TARGET_VERIFY
            if self.drafter is not None
            else ForwardMode.DECODE
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
            input_lengths=empty,
        )

        # If a drafter is active, its model also has MoE layers that issue
        # NCCL collectives. Idle ranks must match those collectives:
        # 1 first-step forward + (spec_num_steps - 1) multi-step decode forwards.
        if self.drafter is not None:
            draft_ctx = ForwardContext(
                attn_backend=self.drafter.attn_backend,
                token_to_kv_pool=self.drafter.token_to_kv_pool,
                req_to_page=self.drafter.req_to_page,
                bs=0,
                num_extends=0,
                input_num_tokens=0,
                forward_mode=ForwardMode.IDLE,
                global_num_tokens=global_num_tokens,
                global_bs=global_bs,
                all_decode_or_idle=all_decode_or_idle,
            )
            for _ in range(self.drafter.spec_num_steps):
                self.drafter.draft_model_runner.forward(
                    draft_ctx,
                    input_ids=empty,
                    positions=empty,
                    out_cache_loc=empty,
                    input_lengths=empty,
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

    def execute_forward_op(
        self,
        forward_op,
        sampling_params_list: list[SamplingParams],
        dp_global_num_tokens=None,
        dp_global_bs=None,
        dp_all_decode_or_idle: bool = False,
        grammar_inputs=None,
    ) -> ModelExecutionResult:

        with nvtx_range("pre_fill_setup", color="orange"):
            num_extends = forward_op.num_extends()
            total_tokens = sum(forward_op.input_lengths)

            # Wait for previous iteration's runtime state updates
            # (future_input_map, valid_cache_lengths) on execution_stream to
            # complete before reading them.
            torch.cuda.current_stream().wait_stream(self.execution_stream)
            self.execution_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.execution_stream):
            self.input_buffers.fill_input_buffers(
                forward_op=forward_op,
                runtime_states=self.runtime_states,
                req_to_page=self.req_to_page,
                total_tokens=total_tokens,
            )

            if num_extends > 0:
                forward_mode = ForwardMode.EXTEND
            elif self.drafter is not None:
                forward_mode = ForwardMode.TARGET_VERIFY
            else:
                forward_mode = ForwardMode.DECODE

            bs = len(forward_op.request_ids)

            if self.runtime_states.mamba_pool is not None and num_extends > 0:
                mamba_pool_indices = self.input_buffers.mamba_pool_indices_buf[:bs]
                mamba_cow_src = self.input_buffers.mamba_cow_src_indices_buf[:bs]
                self.runtime_states.copy_mamba_states(
                    mamba_pool_indices, mamba_cow_src, bs
                )
                self.runtime_states.zero_mamba_states(
                    mamba_pool_indices,
                    mamba_cow_src,
                    self.input_buffers.extend_prefix_lens_buf[:bs],
                    bs,
                )

            grammar_completion = None

            if total_tokens == 0:
                # Fully prefix-cached prefill: no tokens to process.
                output_tokens = torch.zeros(0, dtype=torch.int32, device=self.device)
                output_lengths = torch.zeros(bs, dtype=torch.int32, device=self.device)
                output_logprobs = None
            else:
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
                    padded_static_len=-1,
                    keep_full_logits=forward_mode.is_decode_or_idle()
                    or forward_mode.is_target_verify(),
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
                        is_spec_decode=self.drafter is not None and num_extends <= 0,
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
                    is_extend=num_extends > 0,
                )

            with nvtx_range("output_d2h", color="green"):
                output_tokens = output_tokens.to("cpu", non_blocking=True)
                output_lengths = output_lengths.to("cpu", non_blocking=True)

                if output_logprobs is not None:
                    output_logprobs = output_logprobs.to("cpu", non_blocking=True)

                copy_event = torch.cuda.Event()
                copy_event.record()

        return ModelExecutionResult(
            output_tokens=output_tokens,
            output_lengths=output_lengths,
            output_logprobs=output_logprobs,
            copy_event=copy_event,
            grammar_completion=grammar_completion,
        )
