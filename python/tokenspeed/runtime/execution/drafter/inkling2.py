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

"""Native recurrent proposals over the common fixed-block draft round."""

import torch
from tokenspeed_kernel.ops.metadata import gather_last_accepted_rows

# A stateful NVIDIA collective outside the registry; without a state the gather path runs.
from tokenspeed_kernel.ops.sampling.cute_dsl import (
    DistArgmaxState,
    distributed_argmax,
    supports_dist_argmax_shape,
)
from torch import nn

from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.drafter.dflash import DFlash
from tokenspeed.runtime.execution.forward_step import get_is_cuda_graph_phase
from tokenspeed.runtime.layers.logits_processor import _dist_argmax_vetoed

# The distributed argmax reads rows whose width is a multiple of this.
_STEP_LOGITS_ALIGNMENT = 32


class Inkling2DSpark(DFlash):
    """Per-layer context mixtures and recurrent proposals, without slot state.

    The draft samples after the anchor, so a verify window of W tokens takes W
    draft queries, independently of the decoder depth. Every layer mixes every
    target tap, so a verify round's context write starts at the last tap: the
    taps are packed into a drafter buffer and each layer's context K/V is
    written on a side stream beside the target's LM head and sampling. The flag
    recording that launch is the one the round skips the inline write and joins by.
    """

    spec_algorithm = "DSPARK"
    sample_from_anchor = False
    uses_shared_context_projection = False

    def _init_native_buffers(self) -> None:
        context_length = self.draft_model_runner.model_config.context_len
        padding = self.draft_model_runner.server_args.spec_context_pad
        limit = self.model.config.max_position_embeddings
        # The learned draft bound includes finished-request speculative work.
        if context_length + padding > limit:
            raise ValueError(
                f"Native draft context {context_length} plus speculative padding "
                f"{padding} exceeds checkpoint limit {limit}. "
                f"Set --max-model-len <= {limit - padding}."
            )
        super()._init_native_buffers()
        self._proposal_features_buf = torch.empty(
            (self.input_buffers.max_bs, self.hidden_size),
            dtype=torch.bfloat16,
            device=self.device,
        )
        # Verify rounds write context K/V on a side stream from the target's last tap on.
        on_cuda = self.device.type == "cuda"
        self._context_stream = torch.cuda.Stream(self.device) if on_cuda else None
        self._context_fork_event = torch.cuda.Event() if on_cuda else None
        self._context_join_event = torch.cuda.Event() if on_cuda else None
        self._context_taps_buf = torch.empty(
            (
                self.input_buffers.max_bs * self.spec_num_tokens,
                self.model.context_in_features,
            ),
            dtype=self.model.context_dtype,
            device=self.device,
        )
        self._context_taps: list[torch.Tensor] = []
        self._context_positions: torch.Tensor | None = None
        self._context_cache_locs: torch.Tensor | None = None
        # Whether this round's context K/V is in flight on the side stream.
        self._context_written = False
        # Step logits, sized before capture by the first proposal.
        self._step_logits_buf: torch.Tensor | None = None

    def wire_target(self, target_model: nn.Module) -> None:
        if (
            self.model.embed_tokens is None
            or self.model.lm_head is not target_model.lm_head
        ):
            raise RuntimeError(
                "Native draft resources must be attached during model setup"
            )
        self.target_model = self.target_language_model = target_model
        self.embed_tokens = self.model.embed_tokens
        self.lm_head = target_model.lm_head
        self.logits_processor = target_model.logits_processor

    def _first_layer_embeds(self, flat_ids: torch.Tensor) -> torch.Tensor:
        return self.model.forward_embed(flat_ids)

    def prepare_target_forward(self, ctx: ForwardContext) -> None:
        """Take the target's taps for a verify round whose context K/V can be
        written on the side stream: decode rows only, no external context
        producer, and not a graph warmup (side branches are recorded only
        under capture)."""
        super().prepare_target_forward(ctx)
        self._context_taps = []
        self._context_written = False
        capturing = (
            torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
        )
        if (
            self._context_stream is None
            or ctx.dspark_context_producer is not None
            or ctx.num_extends != 0
            or ctx.input_num_tokens > self._context_taps_buf.shape[0]
            or (get_is_cuda_graph_phase() and not capturing)
        ):
            return
        rows = ctx.input_num_tokens
        self._context_positions = self.input_buffers.positions_buf[:rows]
        self._context_cache_locs = ctx.attn_backend.decode_window_locations()[:rows]
        ctx.target_capture_sink = self

    def on_target_capture(self, capture_idx: int, hidden: torch.Tensor) -> None:
        """Keep the taps; with the last one, pack them (in the order the
        logits processor concatenates them) and write the context K/V on the
        side stream."""
        self._context_taps.append(hidden)
        if capture_idx != self.model.num_context_features - 1:
            return
        rows = hidden.shape[0]
        packed = self._context_taps_buf[:rows]
        torch.cat(self._context_taps, dim=-1, out=packed)
        self._context_taps = []
        self._context_fork_event.record()
        with torch.cuda.stream(self._context_stream):
            self._context_stream.wait_event(self._context_fork_event)
            super()._write_native_cache(
                packed,
                self._context_positions,
                self._context_cache_locs,
                decode_only=True,
            )
            self._context_join_event.record()
        self._context_written = True

    def _write_native_cache(
        self,
        target_hidden: torch.Tensor,
        target_positions: torch.Tensor,
        target_cache_locs: torch.Tensor,
        decode_only: bool = False,
    ) -> None:
        if self._context_written:
            # Already in flight on the side stream; _draft_native joins it.
            return
        super()._write_native_cache(
            target_hidden, target_positions, target_cache_locs, decode_only
        )

    def _draft_native(
        self,
        current_tokens: torch.Tensor,
        prepared: bool = False,
    ) -> torch.Tensor:
        if self._context_written:
            torch.cuda.current_stream().wait_event(self._context_join_event)
            self._context_written = False
        return super()._draft_native(current_tokens, prepared)

    def _accepted_target_rows(
        self, base_ctx: ForwardContext, accept_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bs = base_ctx.bs
        if base_ctx.num_extends == 0:
            # Verify-only fast path: request r's rows start at r * spec_num_tokens.
            takes = (
                accept_lengths[:bs]
                .to(torch.int64)
                .clamp(min=0, max=self.spec_num_tokens)
            )
            last_row = (self.decode_offsets_buf[:bs] + takes).clamp(
                min=0, max=base_ctx.input_num_tokens - 1
            )
            return takes, last_row
        return super()._accepted_target_rows(base_ctx, accept_lengths)

    def _prepare_proposal_features(
        self, ctx: ForwardContext, hidden: torch.Tensor, accept_lengths: torch.Tensor
    ) -> None:
        # Every round has fresh target rows, so batch scratch replaces a per-request feature table.
        if ctx.num_extends == 0:
            # Verify-only: one launch picks the rows _accepted_target_rows names.
            last = gather_last_accepted_rows(
                hidden,
                accept_lengths[: ctx.bs],
                rows_per_request=self.spec_num_tokens,
            )
        else:
            _, last_rows = self._accepted_target_rows(ctx, accept_lengths)
            last = hidden.index_select(0, last_rows)
        features = self.model.compute_f_rnn(last)
        self._proposal_features_buf[: ctx.bs].copy_(features)

    def _step_logits(self, shard: int) -> torch.Tensor:
        """The step-logits rows: the local vocabulary padded to the
        distributed argmax's column alignment."""
        if self._step_logits_buf is None:
            width = -(-shard // _STEP_LOGITS_ALIGNMENT) * _STEP_LOGITS_ALIGNMENT
            self._step_logits_buf = torch.empty(
                (self.input_buffers.max_bs, width),
                dtype=torch.bfloat16,
                device=self.device,
            )
        return self._step_logits_buf

    def _probe_dist_argmax_state(
        self, dtype: torch.dtype, device: torch.device
    ) -> DistArgmaxState | None:
        """Ask for a state over the step-logits rows, which ``_step_logits``
        sizes first: one row per request, over the padded shards."""
        tp_size = int(self.logits_processor.tp_size)
        width = self._step_logits_buf.shape[1]
        if (
            _dist_argmax_vetoed()
            or not 2 <= tp_size <= 32
            or not supports_dist_argmax_shape(width, dtype, tp_size)
        ):
            return None
        return self.logits_processor.acquire_dist_argmax_state(
            self.lm_head,
            max_M=self.input_buffers.max_bs,
            skip_ping_pong=False,
            dtype=dtype,
        )

    def _sample_block(
        self,
        draft_hidden: torch.Tensor,
        block_ids: torch.Tensor,
        next_tokens: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.model.proposal_rows(draft_hidden)
        base = self.model.compute_base_logits(hidden)
        count = self.model.markov_head.vocab_count
        shard = base.shape[-1]
        logits_out = self._step_logits(shard)[: hidden.shape[0]]
        # The probe reads the step-logits width; the gather fallback reuses its verdict.
        state = self._ensure_dist_argmax_state(logits_out.dtype, logits_out.device)

        def sample(logits: torch.Tensor, step: int) -> torch.Tensor:
            if state is not None:
                # Indices over the -inf-padded shards; the head maps them back to vocabulary ids.
                return distributed_argmax(state, logits)[1]
            return self._greedy_argmax_vocab_parallel(
                hidden[:, step],
                out=None,
                bias_fn=None,
                base_logits=logits[:, :count].contiguous(),
            )

        proposals = self.model.markov_head.sample_block(
            base,
            first_prev_tokens=block_ids[:, 0].clamp(0, self.vocab_size - 1),
            hidden_states=hidden,
            f_rnn=self._proposal_features_buf[: hidden.shape[0]],
            logits_out=logits_out,
            sampler=sample,
            shard_padding=None if state is None else (logits_out.shape[1], shard),
        )
        next_tokens[:, 0].copy_(block_ids[:, 0])
        next_tokens[:, 1:].copy_(proposals)
        return next_tokens
