# Copyright (c) 2023 DeepSeek
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

from typing import TYPE_CHECKING

import torch

from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.drafter.base import BaseDrafter
from tokenspeed.runtime.execution.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
)
from tokenspeed.runtime.models.deepseek_v4_dspark_ops.heads import (
    sample_dspark_block_greedy,
)
from tokenspeed.runtime.utils.nvtx import nvtx_range

if TYPE_CHECKING:
    from tokenspeed.runtime.execution.input_buffer import InputBuffers
    from tokenspeed.runtime.execution.model_runner import ModelRunner
    from tokenspeed.runtime.execution.runtime_states import RuntimeStates
    from tokenspeed.runtime.layers.logits_processor import LogitsProcessorOutput


def _dspark_decode_position_plan(
    old_context_lengths: torch.Tensor,
    accepted: torch.Tensor,
    block_offsets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map target verification rows to official DSpark absolute positions."""

    interim_positions = old_context_lengths.unsqueeze(1) + block_offsets.unsqueeze(0)
    interim_valid = block_offsets.unsqueeze(0) < (accepted.unsqueeze(1) - 1)
    main_positions = old_context_lengths + accepted - 1
    next_context_lengths = main_positions + 1
    return (
        interim_positions,
        interim_valid,
        main_positions,
        next_context_lengths,
    )


def _dspark_prefill_position_plan(
    target_positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep target absolute positions and return the next context length."""

    if target_positions.ndim != 2 or target_positions.shape[1] == 0:
        raise ValueError(
            "DSpark prefill positions must be a non-empty [batch, tokens] tensor."
        )
    return target_positions, target_positions[:, -1] + 1


class DeepseekV4DSpark(BaseDrafter):
    """DeepSeek V4 DSpark block drafter with request-persistent context windows."""

    # The V4 checkpoint-local DSpark path shares the target's embed and LM
    # head. Generic DSpark and DFlash ship their own draft weights.
    shares_target_embed_head = True

    def __init__(
        self,
        spec_num_tokens: int,
        spec_num_steps: int,
        draft_model_runner: ModelRunner | None = None,
        page_staging=None,
        attn_backend=None,
        token_to_kv_pool=None,
        runtime_states: RuntimeStates | None = None,
        input_buffers: InputBuffers | None = None,
        vocab_size: int | None = None,
    ) -> None:
        super().__init__(
            spec_num_tokens=spec_num_tokens,
            spec_num_steps=spec_num_steps,
            draft_model_runner=draft_model_runner,
            runtime_states=runtime_states,
            input_buffers=input_buffers,
            page_staging=page_staging,
            attn_backend=attn_backend,
            token_to_kv_pool=token_to_kv_pool,
            vocab_size=vocab_size,
        )
        if draft_model_runner is None or input_buffers is None:
            raise ValueError("DSPARK requires a draft model runner and input buffers.")
        self._validate_tp_only_mapping(draft_model_runner.mapping)

        self.device = torch.device(draft_model_runner.device)
        self.draft_model = draft_model_runner.model
        self.model = self.draft_model.model
        self.block_size = int(self.model.block_size)
        if int(spec_num_tokens) != self.block_size + 1:
            raise ValueError(
                "DSPARK verify width must equal checkpoint block_size + 1; "
                f"got verify_width={spec_num_tokens}, block_size={self.block_size}."
            )
        if int(spec_num_steps) != self.block_size:
            raise ValueError(
                "DSPARK speculative steps must equal checkpoint block_size; "
                f"got steps={spec_num_steps}, block_size={self.block_size}."
            )
        self.target_layer_ids = list(self.model.target_layer_ids)
        self.hidden_width = len(self.target_layer_ids) * int(self.model.hidden_size)
        self.idle_forward_steps = 1
        self._init_buffers()

    @staticmethod
    def _validate_tp_only_mapping(mapping) -> None:
        dp_size = int(mapping.attn.dp_size)
        cp_size = int(mapping.attn.cp_size)
        if dp_size != 1 or cp_size != 1:
            raise ValueError(
                "Week-0 DSPARK supports tensor parallelism only; "
                f"got attention dp_size={dp_size}, cp_size={cp_size}."
            )

    def _init_buffers(self) -> None:
        max_bs = int(self.input_buffers.max_bs)
        first_padding_slot = int(self.input_buffers.state_write_padding_pool_index)
        self.first_padding_slot = first_padding_slot
        self.padding_slots = torch.arange(
            first_padding_slot,
            first_padding_slot + max_bs,
            dtype=torch.int64,
            device=self.device,
        )
        self.slot_indices_buf = self.padding_slots.clone()
        num_window_slots = first_padding_slot + max_bs
        self.kv_windows = torch.zeros(
            (
                num_window_slots,
                int(self.model.num_stages),
                int(self.model.window_size),
                int(self.model.attention_params["head_dim"]),
            ),
            dtype=torch.bfloat16,
            device=self.device,
        )
        self.context_lengths = torch.zeros(
            (num_window_slots,), dtype=torch.int64, device=self.device
        )
        self._request_by_pool_slot: list[object | None] = [None] * first_padding_slot

        self.next_tokens_buf = torch.empty(
            (max_bs, self.spec_num_tokens),
            dtype=torch.int32,
            device=self.device,
        )
        self.draft_tokens_buf = torch.empty(
            (max_bs, self.block_size),
            dtype=torch.int32,
            device=self.device,
        )
        self.block_offsets = torch.arange(
            self.block_size, dtype=torch.int64, device=self.device
        )
        self.decode_row_offsets = (
            torch.arange(max_bs, dtype=torch.int64, device=self.device)
            * self.spec_num_tokens
        )

        tp_size = int(self.draft_model.mapping.attn.tp_size)
        self.gathered_values = torch.empty(
            (tp_size, max_bs), dtype=torch.float32, device=self.device
        )
        self.gathered_ids = torch.empty(
            (tp_size, max_bs), dtype=torch.int64, device=self.device
        )

    def wire_target(self, target_model) -> None:
        self.target_model = target_model
        self.lm_head = target_model.lm_head
        self.tp_group = target_model.logits_processor.tp_group
        if not hasattr(target_model, "set_dspark_layers_to_capture"):
            raise ValueError(
                "DSPARK requires the target model to support "
                "set_dspark_layers_to_capture."
            )
        target_model.set_dspark_layers_to_capture(self.target_layer_ids)

    def prepare_request_state(
        self,
        request_ids: list[object],
        request_pool_indices: list[int],
        num_extends: int,
    ) -> None:
        """Refresh request-to-window slots outside CUDA Graph replay."""

        if len(request_ids) != len(request_pool_indices):
            raise ValueError("DSPARK request IDs and pool indices must align.")
        if len(set(request_pool_indices)) != len(request_pool_indices):
            raise ValueError("DSPARK request pool indices must be unique per batch.")
        changed_slots = []
        for row, (request_id, pool_slot) in enumerate(
            zip(request_ids, request_pool_indices)
        ):
            pool_slot = int(pool_slot)
            if pool_slot < 0 or pool_slot >= self.first_padding_slot:
                raise ValueError(
                    "DSPARK request pool index is outside the persistent state domain: "
                    f"{pool_slot}."
                )
            starts_new_prefill = (
                row < num_extends
                and int(self.input_buffers.extend_prefix_lens_cpu[row]) == 0
            )
            if (
                self._request_by_pool_slot[pool_slot] != request_id
                or starts_new_prefill
            ):
                self._request_by_pool_slot[pool_slot] = request_id
                changed_slots.append(pool_slot)

        if changed_slots:
            changed = torch.tensor(changed_slots, dtype=torch.int64, device=self.device)
            self.kv_windows.index_fill_(0, changed, 0)
            self.context_lengths.index_fill_(0, changed, 0)

        # CUDA Graph padding rows execute the same captured draft path as live
        # requests. Reset their persistent state before every replay so an
        # inactive row cannot accumulate positions or retain window contents.
        self.kv_windows.index_fill_(0, self.padding_slots, 0)
        self.context_lengths.index_fill_(0, self.padding_slots, 0)

        active_bs = len(request_pool_indices)
        self.slot_indices_buf[:active_bs].copy_(
            self.input_buffers.req_pool_indices_buf[:active_bs]
        )
        if active_bs < self.input_buffers.max_bs:
            self.slot_indices_buf[active_bs:].copy_(self.padding_slots[active_bs:])

    @staticmethod
    def _bonus_tokens_from_output(
        output_tokens: torch.Tensor,
        accept_lengths: torch.Tensor,
        num_extends: int,
        verify_width: int,
        out: torch.Tensor,
    ) -> torch.Tensor:
        if num_extends > 0:
            out[:num_extends].copy_(output_tokens[:num_extends])
        num_decodes = accept_lengths.shape[0] - num_extends
        if num_decodes > 0:
            accepted = (
                accept_lengths[num_extends:].to(torch.int64).clamp(1, verify_width)
            )
            offsets = (
                torch.arange(
                    num_decodes,
                    dtype=torch.int64,
                    device=output_tokens.device,
                )
                * verify_width
                + num_extends
            )
            out[num_extends:].copy_(output_tokens[offsets + accepted - 1])
        return out

    def _seed_prefill_windows(
        self,
        hidden_states: torch.Tensor,
        num_extends: int,
    ) -> int:
        if num_extends < 0:
            raise ValueError(f"DSPARK num_extends must be non-negative: {num_extends}.")
        if num_extends == 0:
            return 0

        # fill_input_buffers derives this host mirror from the same scheduler
        # lengths as input_lengths_buf. Reading the CUDA buffer row-by-row here
        # serialized every chunk behind the target stream.
        lengths_cpu = self.input_buffers.extend_seq_lens_cpu
        if (
            not isinstance(lengths_cpu, torch.Tensor)
            or lengths_cpu.device.type != "cpu"
            or lengths_cpu.dtype != torch.int32
            or lengths_cpu.ndim != 1
            or lengths_cpu.numel() < num_extends
        ):
            raise RuntimeError(
                "DSPARK prefill window seeding requires a complete int32 CPU "
                "extend-length mirror."
            )
        chunk_lengths = [int(length) for length in lengths_cpu[:num_extends].tolist()]
        if any(length < 0 for length in chunk_lengths):
            raise RuntimeError("DSPARK prefill chunk lengths must be non-negative.")
        total_prefill_tokens = sum(chunk_lengths)
        if total_prefill_tokens > hidden_states.shape[0]:
            raise RuntimeError(
                "DSPARK prefill chunk lengths exceed captured hidden-state rows: "
                f"{total_prefill_tokens} > {hidden_states.shape[0]}."
            )

        offset = 0
        for row, chunk_len in enumerate(chunk_lengths):
            if chunk_len <= 0:
                continue
            chunk_end = offset + chunk_len
            keep = min(int(self.model.window_size), chunk_len)
            kept_hidden = hidden_states[chunk_end - keep : chunk_end].unsqueeze(0)
            positions, next_context_lengths = _dspark_prefill_position_plan(
                self.input_buffers.positions_buf[
                    chunk_end - keep : chunk_end
                ].unsqueeze(0)
            )
            slot = self.slot_indices_buf[row : row + 1]
            valid = torch.ones_like(positions, dtype=torch.bool)
            self.model.write_context_windows_batched(
                kept_hidden,
                positions,
                slot,
                valid,
                self.kv_windows,
                self.first_padding_slot,
            )
            self.context_lengths[slot] = next_context_lengths
            offset = chunk_end
        return offset

    def _draft_decode_rows(
        self,
        base_ctx: ForwardContext,
        hidden_states: torch.Tensor,
        output_tokens: torch.Tensor,
        accept_lengths: torch.Tensor,
        next_tokens: torch.Tensor,
        prefill_tokens: int,
    ) -> None:
        num_extends = base_ctx.num_extends
        num_decodes = base_ctx.bs - num_extends
        if num_decodes <= 0:
            return

        decode_hidden = hidden_states[prefill_tokens:].reshape(
            num_decodes,
            self.spec_num_tokens,
            self.hidden_width,
        )
        accepted = (
            accept_lengths[num_extends:].to(torch.int64).clamp(1, self.spec_num_tokens)
        )
        accepted_index = accepted - 1
        rows = torch.arange(num_decodes, dtype=torch.int64, device=self.device)
        main_hidden = decode_hidden[rows, accepted_index]

        bonus = next_tokens[num_extends:, 0]
        slots = self.slot_indices_buf[num_extends : num_extends + num_decodes]
        old_context_lengths = self.context_lengths.index_select(0, slots)
        (
            interim_positions,
            interim_valid,
            main_positions,
            next_context_lengths,
        ) = _dspark_decode_position_plan(
            old_context_lengths,
            accepted,
            self.block_offsets,
        )
        self.model.write_context_windows_batched(
            decode_hidden[:, : self.block_size],
            interim_positions,
            slots,
            interim_valid,
            self.kv_windows,
            self.first_padding_slot,
        )
        self.context_lengths.index_copy_(0, slots, next_context_lengths)

        draft_ctx = ForwardContext(
            attn_backend=base_ctx.attn_backend,
            token_to_kv_pool=base_ctx.token_to_kv_pool,
            bs=num_decodes,
            num_extends=0,
            input_num_tokens=num_decodes * self.block_size,
            forward_mode=ForwardMode.DECODE,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            all_decode_or_idle=base_ctx.all_decode_or_idle,
        )
        draft_hidden = self.model.forward_backbone(
            main_hidden,
            bonus,
            main_positions,
            self.kv_windows,
            slots,
            draft_ctx,
        )
        local_logits = self.model.local_base_logits(draft_hidden, self.lm_head)
        sample_dspark_block_greedy(
            local_logits,
            bonus,
            self.model.markov_head,
            self.lm_head,
            self.tp_group,
            self.gathered_values,
            self.gathered_ids,
            self.draft_tokens_buf[:num_decodes],
        )
        next_tokens[num_extends:, 1:].copy_(self.draft_tokens_buf[:num_decodes])

    @nvtx_range("drafter:dspark", color="purple")
    def run(
        self,
        base_ctx: ForwardContext,
        logits_output: LogitsProcessorOutput,
        output_tokens: torch.Tensor,
        accept_lengths: torch.Tensor,
    ) -> torch.Tensor:
        if not hasattr(self, "target_model"):
            raise RuntimeError("DSPARK drafter is not bound to the target model.")
        hidden_states = logits_output.hidden_states
        if hidden_states is None:
            raise RuntimeError("DSPARK requires target hidden-state captures.")
        if hidden_states.ndim != 2 or hidden_states.shape[1] != self.hidden_width:
            raise RuntimeError(
                "DSPARK target hidden-state shape mismatch: "
                f"expected [tokens, {self.hidden_width}], got "
                f"{tuple(hidden_states.shape)}."
            )

        next_tokens = self.next_tokens_buf[: base_ctx.bs]
        self._bonus_tokens_from_output(
            output_tokens,
            accept_lengths,
            base_ctx.num_extends,
            self.spec_num_tokens,
            next_tokens[:, 0],
        )
        next_tokens[:, 1:].copy_(next_tokens[:, :1])
        prefill_tokens = self._seed_prefill_windows(
            hidden_states,
            base_ctx.num_extends,
        )
        self._draft_decode_rows(
            base_ctx,
            hidden_states,
            output_tokens,
            accept_lengths,
            next_tokens,
            prefill_tokens,
        )
        next_tokens.clamp_(0, int(self.vocab_size) - 1)
        return next_tokens

    def draft(self, *args, **kwargs) -> torch.Tensor | None:
        raise RuntimeError("DSPARK drafts through run() with target captures.")
