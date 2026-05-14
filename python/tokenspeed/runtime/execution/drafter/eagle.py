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


@dataclass
class EagleDraftInput:
    input_num_tokens: int
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

        # Precomputed `arange(max_bs) * spec_num_tokens - 1`, sliced and passed
        # via ForwardContext for the padded-static-len last-token selection in
        # LogitsProcessor.
        self._last_index_offsets_buf = (
            torch.arange(
                self.input_buffers.max_bs, dtype=torch.int64, device=self.device
            )
            * spec_num_tokens
            - 1
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _map_hot(self, ids: torch.Tensor) -> torch.Tensor:
        """Map token ids through hot_token_ids if available, otherwise return as-is."""
        return self.hot_token_ids[ids] if self.hot_token_ids is not None else ids

    def _get_first_step_input(
        self,
        forward_mode: ForwardMode,
        draft_input: EagleDraftInput,
        bs: int,
        input_num_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (input_ids, unpadded_input_lengths) for the first draft step."""
        # Only pure EXTEND (prefill) uses shifted_prefill_ids. TARGET_VERIFY
        # and DRAFT_EXTEND use base_model_output + accept_lengths.
        if forward_mode == ForwardMode.EXTEND:

            input_ids = self.input_buffers.shifted_prefill_ids_buf[
                :input_num_tokens
            ].clone()

            unpadded_input_lengths = self.input_buffers.input_lengths_buf[:bs]
            req_boundaries = unpadded_input_lengths.cumsum(0) - 1  # [bs]
            boundary_ids = input_ids[req_boundaries]
            needs_fill = boundary_ids == -1  # [bs]
            input_ids[req_boundaries] = torch.where(
                needs_fill, draft_input.base_model_output[:bs], boundary_ids
            )
        else:

            input_ids = draft_input.base_model_output
            unpadded_input_lengths = draft_input.accept_lengths

        return input_ids, unpadded_input_lengths

    @nvtx_range("draft_first_step", color="purple")
    def _run_first_step(
        self,
        bs: int,
        draft_input: EagleDraftInput,
    ) -> LogitsProcessorOutput:

        buffers = self.input_buffers
        forward_mode = draft_input.forward_mode

        input_ids, unpadded_input_lengths = self._get_first_step_input(
            forward_mode, draft_input, bs, draft_input.input_num_tokens
        )

        # The drafter's first step uses DRAFT_EXTEND when base model did
        # TARGET_VERIFY, or EXTEND when base model did EXTEND (prefill).
        if forward_mode.is_target_verify():
            draft_first_mode = ForwardMode.DRAFT_EXTEND
        else:
            draft_first_mode = forward_mode

        is_decode_like = draft_first_mode.is_draft_extend()

        # make a ctx every time model runner forward
        first_step_ctx = ForwardContext(
            attn_backend=self.attn_backend,
            token_to_kv_pool=self.token_to_kv_pool,
            req_to_page=self.req_to_page,
            bs=bs,
            num_extends=0,
            input_num_tokens=draft_input.input_num_tokens,
            forward_mode=draft_first_mode,
            capture_hidden_mode=CaptureHiddenMode.LAST,
            padded_static_len=self.spec_num_tokens if is_decode_like else -1,
            last_index_offsets=(
                self._last_index_offsets_buf[:bs] if is_decode_like else None
            ),
            keep_full_logits=False,
            global_num_tokens=draft_input.global_num_tokens,
            global_bs=draft_input.global_bs,
            all_decode_or_idle=draft_input.all_decode_or_idle,
        )

        return self.draft_model_runner.forward(
            ctx=first_step_ctx,
            input_ids=input_ids,
            positions=buffers.positions_buf[: draft_input.input_num_tokens],
            out_cache_loc=buffers.out_cache_loc_buf[: draft_input.input_num_tokens],
            input_lengths=unpadded_input_lengths,  # Used in logits processor
            captured_hidden_states=draft_input.base_out_hidden_states,
        )

    @nvtx_range("draft_multi_step", color="purple")
    def _run_multi_step_decode(
        self,
        bs: int,
        draft_ids: torch.Tensor,
        draft_tokens: torch.Tensor,
        logits_output: LogitsProcessorOutput,
        draft_input: EagleDraftInput,
    ) -> None:

        req_pool_indices = self.input_buffers.req_pool_indices_buf[:bs]
        # Step 1's new write position. TARGET_VERIFY uses vc+accept_length
        # (vc+N would shift rotary past the rejected tail); EXTEND uses
        # vc+input_lengths (= seq_lens_buf).
        if draft_input.forward_mode.is_target_verify():
            cache_start = (
                self.runtime_states.valid_cache_lengths.index_select(
                    0, req_pool_indices
                )
                + draft_input.accept_lengths
            )
        else:
            cache_start = self.input_buffers.seq_lens_buf[:bs].clone()

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
        cache_locs_trans = cache_locs.view(bs, self.spec_num_steps - 1).t().contiguous()

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
                keep_full_logits=True,
                global_num_tokens=global_num_tokens,
                global_bs=draft_input.global_bs,
                all_decode_or_idle=draft_input.all_decode_or_idle,
            )

            with nvtx_range("draft_forward", color="red"):
                logits_output = self.draft_model_runner.forward(
                    ctx=ctx,
                    input_ids=self._map_hot(draft_ids),
                    positions=positions,
                    out_cache_loc=cache_locs_trans[i - 1],
                    input_lengths=input_lengths,
                    captured_hidden_states=logits_output.hidden_states,
                )

            with nvtx_range("draft_sample", color="yellow"):
                draft_ids = torch.argmax(logits_output.next_token_logits, dim=-1)
                draft_tokens[:, i] = self._map_hot(draft_ids)
                if i + 1 < self.spec_num_steps:
                    positions.add_(1)
                    draft_seq_lens.add_(1)

    def _get_last_verified_ids(
        self, bs: int, forward_mode: ForwardMode, draft_input: EagleDraftInput
    ) -> torch.Tensor:

        if forward_mode == ForwardMode.EXTEND:
            # Last verified id is simply the base output for each request
            return draft_input.base_model_output[:bs]
        else:
            # Pick the last accepted token per request from the flattened base output
            req_offsets = torch.arange(bs, device=self.device) * self.spec_num_tokens
            indices = req_offsets + draft_input.accept_lengths - 1
            return draft_input.base_model_output[indices]

    # ------------------------------------------------------------------
    # Public entry point (type-based dispatch from ModelExecutor)
    # ------------------------------------------------------------------

    @override
    def get_candidates(
        self,
        base_ctx: ForwardContext,
    ) -> torch.Tensor | None:

        if not (
            base_ctx.forward_mode.is_decode()
            or base_ctx.forward_mode.is_target_verify()
        ):
            return None

        return self.input_buffers.input_ids_buf[: base_ctx.input_num_tokens].reshape(
            base_ctx.bs, self.spec_num_tokens
        )

    @override
    def draft(
        self,
        draft_input: EagleDraftInput,
    ) -> torch.Tensor:

        bs = draft_input.accept_lengths.shape[0]

        draft_tokens = torch.empty(
            (bs, self.spec_num_steps),
            dtype=torch.int32,
            device=self.device,
        )

        # Seed the draft attn backend's aliased seq_lens for the first step.
        self.draft_seq_lens_buf[:bs].copy_(self.input_buffers.seq_lens_buf[:bs])

        # First draft step. LogitsProcessor prunes `[bs * spec_num_tokens, ...]`
        # down to `[bs, ...]` via padded_static_len, so logits/hidden_states
        # arrive here already aligned to one row per request.
        logits_output = self._run_first_step(bs, draft_input)

        draft_ids = torch.argmax(logits_output.next_token_logits, dim=-1)
        draft_tokens[:, 0] = self._map_hot(draft_ids)

        # Draft step 2+ (multi-step decode).
        if self.spec_num_steps > 1:
            # Skip multi-step when the whole batch is mid-chunk EXTEND: no
            # request transitions to target_verify after this forward, so
            # any speculative tokens we draft would be discarded.
            #
            # In DP we still run, because peer ranks may have completing
            # extends or decodes; diverging here would desync the drafter's
            # dense-TP / MoE-EP collectives (NCCL hang or RSAG mismatch).
            skip = self.dp_size == 1 and self.input_buffers.all_extends_mid_chunk
            if not skip:
                self._run_multi_step_decode(
                    bs, draft_ids, draft_tokens, logits_output, draft_input
                )

        return draft_tokens

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
            forward_mode=base_ctx.forward_mode,
            base_model_output=output_tokens,
            accept_lengths=accept_lengths,
            base_out_hidden_states=logits_output.hidden_states,
            global_num_tokens=base_ctx.global_num_tokens,
            global_bs=base_ctx.global_bs,
            all_decode_or_idle=base_ctx.all_decode_or_idle,
        )

        draft_tokens = self.draft(draft_input)

        last_verified_ids = self._get_last_verified_ids(
            base_ctx.bs, base_ctx.forward_mode, draft_input
        )

        return torch.cat([last_verified_ids.unsqueeze(1), draft_tokens], dim=1)
