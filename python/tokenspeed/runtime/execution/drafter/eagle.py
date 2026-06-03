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

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from tokenspeed_kernel.ops.sampling.cute_dsl import argmax as cute_argmax
from typing_extensions import override

from tokenspeed.runtime.execution.cache_loc_kernel import (
    compute_out_cache_loc_uniform,
)
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.drafter.base import BaseDrafter
from tokenspeed.runtime.execution.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
)
from tokenspeed.runtime.models.base import BaseCausalLM
from tokenspeed.runtime.multimodal.inputs import maybe_substitute_mm_pad
from tokenspeed.runtime.utils import get_colorful_logger
from tokenspeed.runtime.utils.nvtx import nvtx_range

logger = get_colorful_logger(__name__)

if TYPE_CHECKING:
    from tokenspeed.runtime.execution.input_buffer import InputBuffers
    from tokenspeed.runtime.execution.model_runner import ModelRunner
    from tokenspeed.runtime.execution.runtime_states import RuntimeStates
    from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
    from tokenspeed.runtime.layers.attention.kv_cache.base import BaseTokenToKVPool
    from tokenspeed.runtime.layers.logits_processor import LogitsProcessorOutput


def _advance_draft_forward_metadata_if_supported(attn_backend, seq_lens) -> None:
    advance = getattr(attn_backend, "advance_draft_forward_metadata", None)
    if advance is not None:
        advance(seq_lens)


@dataclass
class EagleDraftInput:
    input_num_tokens: int
    num_extends: int
    forward_mode: ForwardMode
    base_model_output: torch.Tensor  # [bs]
    accept_lengths: torch.Tensor  # [bs]
    base_out_hidden_states: torch.Tensor
    global_num_tokens: list[int] | None = None
    global_bs: list[int] | None = None
    all_decode_or_idle: bool = False


class Eagle(BaseDrafter):
    """
    Draft model runner that implements the Eagle/Eagle3 algorithm.
    """

    def __init__(
        self,
        spec_num_tokens: int,
        spec_num_steps: int,
        page_size: int,
        draft_model_runner: ModelRunner,
        req_to_page: torch.Tensor,
        attn_backend: AttentionBackend | None = None,
        token_to_kv_pool: BaseTokenToKVPool | None = None,
        runtime_states: RuntimeStates | None = None,
        input_buffers: InputBuffers | None = None,
        vocab_size: int | None = None,
    ) -> None:

        super().__init__(
            spec_num_tokens,
            spec_num_steps,
            draft_model_runner,
            runtime_states=runtime_states,
            input_buffers=input_buffers,
            page_size=page_size,
            req_to_page=req_to_page,
            attn_backend=attn_backend,
            token_to_kv_pool=token_to_kv_pool,
            vocab_size=vocab_size,
        )

        self.device = draft_model_runner.device
        hot_token_ids = draft_model_runner.model.get_hot_token_id()

        if hot_token_ids is not None:
            self.hot_token_ids = hot_token_ids.to(self.device)
        else:
            self.hot_token_ids = None

        # For constructing fallback global_num_tokens during CUDA graph capture.
        self.dp_size = draft_model_runner.mapping.attn.dp_size
        self.world_size = draft_model_runner.mapping.world_size

        # Drafter-owned alias source for the draft attn backend; advanced in
        # place during multi-step decode.
        self.draft_seq_lens_buf = torch.zeros_like(self.input_buffers.seq_lens_buf)

        # Persistent output buffer for the draft step's compute_out_cache_loc.
        self.draft_out_cache_loc_buf = torch.empty(
            (self.input_buffers.max_bs * (spec_num_steps - 1),),
            dtype=torch.int32,
            device=self.device,
        )

        # Per-request input length is always 1 in multi-step decode (one token per request).
        self.draft_input_lengths_buf = torch.ones(
            (self.input_buffers.max_bs,),
            dtype=torch.int32,
            device=self.device,
        )

        # Precomputed `arange(max_bs) * spec_num_tokens - 1`
        # gather_ids = gather_ids_offsets + accept_lengths
        self.padded_gather_ids_offsets_buf = (
            torch.arange(
                self.input_buffers.max_bs, dtype=torch.int64, device=self.device
            )
            * spec_num_tokens
            - 1
        )

        # VLM placeholder id plumbed by ModelExecutor; None for text-only targets.
        self.mm_pad_substitute_id: int | None = None

    def set_mm_pad_substitute_id(self, token_id: int) -> None:
        self.mm_pad_substitute_id = token_id

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _map_hot(self, ids: torch.Tensor) -> torch.Tensor:
        """Map token ids through hot_token_ids if available, otherwise return as-is."""
        return self.hot_token_ids[ids] if self.hot_token_ids is not None else ids

    def _get_first_step_input(
        self,
        draft_input: EagleDraftInput,
        bs: int,
        input_num_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (input_ids, unpadded_input_lengths, gather_ids) for the first draft step.

        The first-step input shape matches the base model's: ragged
        ``[prefill_part || decode_part]`` under MIXED, full prefill chunks
        under EXTEND, ``base_model_output`` directly under DECODE.
        """
        num_extends = draft_input.num_extends
        num_decodes = bs - num_extends
        if num_extends > 0:
            num_decode_tokens = num_decodes * self.spec_num_tokens
            num_prefill_tokens = input_num_tokens - num_decode_tokens

            input_ids = self.input_buffers.shifted_prefill_ids_buf[:input_num_tokens]
            unpadded_input_lengths = self.input_buffers.input_lengths_buf[:bs]
            if num_decodes > 0:
                input_ids[num_prefill_tokens:].copy_(
                    draft_input.base_model_output[num_extends:]
                )
                unpadded_input_lengths[num_extends:].copy_(
                    draft_input.accept_lengths[num_extends:]
                )

            last_indices = unpadded_input_lengths[:num_extends].cumsum(0) - 1
            last_input_ids = input_ids[last_indices]
            input_ids[last_indices] = torch.where(
                last_input_ids == -1,
                draft_input.base_model_output[:num_extends],
                last_input_ids,
            )

            gather_ids = last_indices
            if num_decodes > 0:
                gather_ids = torch.cat(
                    [
                        gather_ids,
                        self.padded_gather_ids_offsets_buf[:num_decodes]
                        + draft_input.accept_lengths[num_extends:]
                        + num_prefill_tokens,
                    ]
                )
        else:
            input_ids = draft_input.base_model_output
            unpadded_input_lengths = draft_input.accept_lengths
            gather_ids = (
                self.padded_gather_ids_offsets_buf[:bs] + draft_input.accept_lengths
            )

        return input_ids, unpadded_input_lengths, gather_ids

    @nvtx_range("draft_first_step", color="purple")
    def _run_first_step(
        self,
        bs: int,
        draft_input: EagleDraftInput,
    ) -> LogitsProcessorOutput:

        buffers = self.input_buffers
        forward_mode = draft_input.forward_mode

        input_ids, unpadded_input_lengths, gather_ids = self._get_first_step_input(
            draft_input, bs, draft_input.input_num_tokens
        )
        input_ids = maybe_substitute_mm_pad(input_ids, self.mm_pad_substitute_id)
        draft_first_step_reduce = forward_mode.is_decode()

        # TODO: remove the isinstance/flag gate together with pre_attention_trim
        # once Qwen NextN and DeepSeek V3 NextN also pre-slice q.
        draft_model = self.draft_model_runner.model
        if (
            draft_first_step_reduce
            and self.attn_backend.support_kv_cache_prewrite
            and isinstance(draft_model, BaseCausalLM)
            and draft_model.pre_attention_trim
        ):
            correction = (self.spec_num_tokens - draft_input.accept_lengths).to(
                self.draft_seq_lens_buf.dtype
            )
            self.draft_seq_lens_buf[:bs].sub_(correction)

        draft_first_mode = (
            ForwardMode.DRAFT_EXTEND
            if forward_mode.is_target_verify()
            else forward_mode
        )

        ctx = ForwardContext(
            attn_backend=self.attn_backend,
            token_to_kv_pool=self.token_to_kv_pool,
            req_to_page=self.req_to_page,
            bs=bs,
            num_extends=draft_input.num_extends,
            input_num_tokens=draft_input.input_num_tokens,
            forward_mode=draft_first_mode,
            capture_hidden_mode=CaptureHiddenMode.LAST,
            gather_ids=gather_ids,
            global_num_tokens=draft_input.global_num_tokens,
            global_bs=draft_input.global_bs,
            all_decode_or_idle=draft_input.all_decode_or_idle,
            draft_first_step_reduce=draft_first_step_reduce,
        )

        return self.draft_model_runner.forward(
            ctx=ctx,
            input_ids=input_ids,
            positions=buffers.positions_buf[: draft_input.input_num_tokens],
            out_cache_loc=buffers.out_cache_loc_buf[: draft_input.input_num_tokens],
            input_lengths=unpadded_input_lengths,  # Used in logits processor
            captured_hidden_states=draft_input.base_out_hidden_states,
            spec_step_idx=0,
        )

    @nvtx_range("draft_multi_step", color="purple")
    def _run_multi_step_decode(
        self,
        bs: int,
        draft_ids: torch.Tensor,
        next_tokens: torch.Tensor,
        logits_output: LogitsProcessorOutput,
        draft_input: EagleDraftInput,
    ) -> None:
        num_extends = draft_input.num_extends
        num_decodes = bs - num_extends
        req_pool_indices = self.input_buffers.req_pool_indices_buf[:bs]
        cache_start = self.input_buffers.seq_lens_buf[:bs].clone()
        # Step 1's write position uses vc+accept_length after target verify so
        # rotary/cache metadata stay on the accepted prefix, not rejected tail.
        if num_decodes > 0:
            cache_start[num_extends:] = (
                self.runtime_states.valid_cache_lengths.index_select(
                    0, req_pool_indices[num_extends:]
                )
                + draft_input.accept_lengths[num_extends:]
            )

        # Write cache slots for steps 1..N-1.
        cache_locs = self.draft_out_cache_loc_buf[: bs * (self.spec_num_steps - 1)]
        compute_out_cache_loc_uniform(
            out_cache_loc_ptr=cache_locs,
            req_pool_indices=req_pool_indices,
            uniform_input_length=self.spec_num_steps - 1,
            cache_start=cache_start,
            req_to_pages=self.req_to_page,
            page_size=self.page_size,
        )
        cache_locs = cache_locs.view(bs, self.spec_num_steps - 1)
        # +1 is the kernel's read-inclusive convention; advanced per iter.
        draft_seq_lens = self.draft_seq_lens_buf[:bs]
        torch.add(cache_start, 1, out=draft_seq_lens)

        input_lengths = self.draft_input_lengths_buf[:bs]
        positions = cache_start.clone()

        for i in range(1, self.spec_num_steps):
            # make a ctx every time model runner forward
            # Multi-step decode is pure DECODE mode: one token per request.
            # global_num_tokens must reflect each rank's batch size, not the
            # target model's total tokens (which may be bs * spec_num_tokens).
            global_num_tokens = draft_input.global_num_tokens

            if self.dp_size > 1:
                if draft_input.global_bs is not None:
                    global_num_tokens = draft_input.global_bs
                else:
                    # CUDA graph capture path: uniform batch size across ranks.
                    global_num_tokens = [bs] * self.world_size

            ctx = ForwardContext(
                bs=bs,
                num_extends=0,
                attn_backend=self.attn_backend,
                token_to_kv_pool=self.token_to_kv_pool,
                req_to_page=self.req_to_page,
                input_num_tokens=bs,
                forward_mode=ForwardMode.DECODE,
                capture_hidden_mode=CaptureHiddenMode.LAST,
                global_num_tokens=global_num_tokens,
                global_bs=draft_input.global_bs,
                all_decode_or_idle=draft_input.all_decode_or_idle,
            )

            out_cache_loc = cache_locs[:, i - 1].contiguous()
            # Keep attention metadata on the accepted prefix; rejected verify
            # tail slots may still contain stale draft KV.
            _advance_draft_forward_metadata_if_supported(
                ctx.attn_backend, draft_seq_lens
            )

            with nvtx_range("draft_forward", color="red"):
                logits_output = self.draft_model_runner.forward(
                    ctx=ctx,
                    input_ids=self._map_hot(draft_ids),
                    positions=positions,
                    out_cache_loc=out_cache_loc,
                    input_lengths=input_lengths,
                    captured_hidden_states=logits_output.hidden_states,
                    spec_step_idx=i,
                )

            with nvtx_range("draft_sample", color="yellow"):
                draft_ids = cute_argmax(logits_output.next_token_logits)
                draft_ids.clamp_(min=0)
                # Column 0 holds last_verified_ids; drafter writes step `i` into column `i + 1`.
                next_tokens[:, i + 1] = self._map_hot(draft_ids)
                if i + 1 < self.spec_num_steps:
                    positions.add_(1)
                    draft_seq_lens.add_(1)

    # ------------------------------------------------------------------
    # Public entry point (type-based dispatch from ModelExecutor)
    # ------------------------------------------------------------------

    @override
    def get_candidates(
        self,
        base_ctx: ForwardContext,
    ) -> torch.Tensor | None:
        num_extends = base_ctx.num_extends
        num_decodes = base_ctx.bs - num_extends
        if num_decodes == 0:
            return None

        num_decode_tokens = num_decodes * self.spec_num_tokens
        num_prefill_tokens = base_ctx.input_num_tokens - num_decode_tokens
        return self.input_buffers.input_ids_buf[
            num_prefill_tokens : base_ctx.input_num_tokens
        ].reshape(num_decodes, self.spec_num_tokens)

    @override
    def draft(
        self,
        draft_input: EagleDraftInput,
    ) -> torch.Tensor:

        bs = draft_input.accept_lengths.shape[0]

        # Layout: column 0 holds the last verified id (the base model's accepted token);
        # columns 1..spec_num_steps hold the drafter's speculative tokens.
        next_tokens = torch.empty(
            (bs, self.spec_num_steps + 1),
            dtype=torch.int32,
            device=self.device,
        )

        # Last verified id per request → next_tokens[:, 0].
        num_extends = draft_input.num_extends
        num_decodes = bs - num_extends
        if num_extends > 0:
            next_tokens[:num_extends, 0] = draft_input.base_model_output[:num_extends]
        if num_decodes > 0:
            indices = (
                self.padded_gather_ids_offsets_buf[:num_decodes]
                + draft_input.accept_lengths[num_extends:]
            )
            if num_extends > 0:
                indices.add_(num_extends)
            torch.index_select(
                draft_input.base_model_output,
                0,
                indices,
                out=next_tokens[num_extends:, 0],
            )

        # Seed the draft attn backend's aliased seq_lens for the first step.
        self.draft_seq_lens_buf[:bs].copy_(self.input_buffers.seq_lens_buf[:bs])

        # First draft step. LogitsProcessor prunes `[num_prefill_tokens + num_decodes * spec_num_tokens, ...]`
        # down to `[bs, ...]`, so logits/hidden_states arrive here already aligned to one row per request.
        logits_output = self._run_first_step(bs, draft_input)

        draft_ids = cute_argmax(logits_output.next_token_logits)
        draft_ids.clamp_(min=0)
        next_tokens[:, 1] = self._map_hot(draft_ids)

        if self.spec_num_steps <= 1:
            return next_tokens

        if self.input_buffers.all_extends_mid_chunk and self.dp_size == 1:
            # Skip multi-step when the whole batch is mid-chunk EXTEND: no
            # request transitions to target_verify after this forward, so
            # any speculative tokens we draft would be discarded.
            #
            # In DP we still run, because peer ranks may have completing
            # extends or decodes; diverging here would desync the drafter's
            # dense-TP / MoE-EP collectives (NCCL hang or RSAG mismatch).
            return next_tokens

        # Draft step 2+ (multi-step decode).
        # Multi-step decode operates on full bs; drop the [num_extends:]
        # slice that step 0 may have set up for MIXED target. No-op on
        # backends that fill separate prefill/decode metadata at init
        # time.
        with self.attn_backend.override_num_extends(0):
            self._run_multi_step_decode(
                bs, draft_ids, next_tokens, logits_output, draft_input
            )
        return next_tokens

    @override
    @nvtx_range("drafter", color="purple")
    def run(
        self,
        base_ctx: ForwardContext,
        logits_output: LogitsProcessorOutput,
        output_tokens: torch.Tensor,
        accept_lengths: torch.Tensor,
    ) -> torch.Tensor:

        draft_input = EagleDraftInput(
            input_num_tokens=base_ctx.input_num_tokens,
            num_extends=base_ctx.num_extends,
            forward_mode=base_ctx.forward_mode,
            base_model_output=output_tokens,
            accept_lengths=accept_lengths,
            base_out_hidden_states=logits_output.hidden_states,
            global_num_tokens=base_ctx.global_num_tokens,
            global_bs=base_ctx.global_bs,
            all_decode_or_idle=base_ctx.all_decode_or_idle,
        )

        # next_tokens layout: column 0 = last verified id, columns 1.. = drafter tokens.
        return self.draft(draft_input)
