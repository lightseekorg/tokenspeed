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

"""N-gram (prompt-lookup) speculative drafter.

Draft tokens are proposed by matching the suffix of each request's running
token history against earlier positions in the same history (KMP-style
longest prefix-as-suffix search, capped by ``max_ngram``). Tokens that
follow the matched window become the speculative draft for the next round.

This drafter is CPU-only: no draft model, no draft KV cache, no draft
attention backend. The chain greedy / chain stochastic verify kernels on
the target side are unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from typing_extensions import override

from tokenspeed.runtime.execution.drafter.base import BaseDrafter
from tokenspeed.runtime.execution.drafter.ngram_lookup import propose_batch_into
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.utils.nvtx import nvtx_range

if TYPE_CHECKING:
    from tokenspeed.runtime.execution.context import ForwardContext
    from tokenspeed.runtime.execution.input_buffer import InputBuffers
    from tokenspeed.runtime.execution.runtime_states import RuntimeStates
    from tokenspeed.runtime.layers.logits_processor import LogitsProcessorOutput


class NgramDrafter(BaseDrafter):
    """Prompt-lookup speculative drafter (no draft model).

    Maintains a CPU-side per-request token history keyed by
    request-pool index, mirroring the slot semantics used elsewhere in
    the executor. On each ``run()`` it appends the freshly accepted
    tokens to the matching slot, runs a KMP-based suffix-ngram lookup
    per request, and stages the proposed ``[last_verified, d1, ...,
    d_K]`` row for the next round's verify input.
    """

    def __init__(
        self,
        spec_num_tokens: int,
        spec_num_steps: int,
        runtime_states: RuntimeStates,
        input_buffers: InputBuffers,
        max_context_len: int,
        vocab_size: int | None = None,
        min_ngram: int = 1,
        max_ngram: int = 3,
    ) -> None:
        super().__init__(
            spec_num_tokens=spec_num_tokens,
            spec_num_steps=spec_num_steps,
            draft_model_runner=None,
            runtime_states=runtime_states,
            input_buffers=input_buffers,
            page_size=None,
            req_to_page=None,
            attn_backend=None,
            token_to_kv_pool=None,
            vocab_size=vocab_size,
        )

        if min_ngram < 1:
            raise ValueError(f"min_ngram must be >= 1, got {min_ngram}")
        if max_ngram < min_ngram:
            raise ValueError(
                f"max_ngram ({max_ngram}) must be >= min_ngram ({min_ngram})"
            )

        self.min_ngram = int(min_ngram)
        self.max_ngram = int(max_ngram)
        self.max_context_len = int(max_context_len)
        self.device = runtime_states.device

        pool_capacity = runtime_states.valid_cache_lengths.shape[0]
        self.history = np.zeros(
            (pool_capacity, self.max_context_len), dtype=np.int32
        )
        self.history_len = np.zeros((pool_capacity,), dtype=np.int32)

        # Staging buffers for batched H2D of the next-round inputs.
        self._next_input_np = np.zeros(
            (input_buffers.max_bs, spec_num_tokens), dtype=np.int32
        )
        self._next_input_pinned = torch.empty(
            (input_buffers.max_bs, spec_num_tokens),
            dtype=torch.int32,
            pin_memory=(self.device == "cuda"),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset_slot(self, pool_idx: int) -> None:
        self.history_len[pool_idx] = 0

    def _append_to_slot(self, pool_idx: int, tokens: np.ndarray) -> None:
        if tokens.size == 0:
            return
        cur = int(self.history_len[pool_idx])
        new_total = cur + tokens.size
        cap = self.max_context_len
        if new_total <= cap:
            self.history[pool_idx, cur:new_total] = tokens
            self.history_len[pool_idx] = new_total
        else:
            # Slide the window: keep the most recent ``cap`` tokens.
            combined = np.concatenate(
                [self.history[pool_idx, :cur], tokens.astype(np.int32, copy=False)]
            )
            tail = combined[-cap:]
            self.history[pool_idx, : tail.size] = tail
            self.history_len[pool_idx] = tail.size

    # ------------------------------------------------------------------
    # BaseDrafter contract
    # ------------------------------------------------------------------

    @override
    def get_candidates(
        self,
        base_ctx: ForwardContext,
    ) -> torch.Tensor | None:
        # Identical layout to EAGLE: verify reads the [bs, spec_num_tokens]
        # window that was written into input_ids_buf by fill_input_buffers.
        if not (
            base_ctx.forward_mode.is_decode()
            or base_ctx.forward_mode.is_target_verify()
        ):
            return None
        return self.input_buffers.input_ids_buf[: base_ctx.input_num_tokens].reshape(
            base_ctx.bs, self.spec_num_tokens
        )

    @override
    def draft(self, *_args, **_kwargs) -> torch.Tensor:
        # Drafting in this proposer is part of ``run()`` (history update +
        # ngram lookup are tightly coupled). Keep the abstract method
        # satisfied without exposing a separately-callable surface.
        raise NotImplementedError(
            "NgramDrafter does not expose a standalone draft(); use run()."
        )

    @override
    @nvtx_range("ngram_drafter", color="purple")
    def run(
        self,
        base_ctx: ForwardContext,
        logits_output: LogitsProcessorOutput,
        output_tokens: torch.Tensor,
        accept_lengths: torch.Tensor,
    ) -> torch.Tensor:
        del logits_output  # unused; ngram drafter ignores hidden states.

        bs = base_ctx.bs
        # The drafter intentionally runs outside the CUDA-graph capture
        # path (executor forces enforce_eager when NGRAM is active), so
        # these D2H syncs are acceptable.
        pool_indices = (
            self.input_buffers.req_pool_indices_buf[:bs].to("cpu").numpy()
        )

        self._update_history(base_ctx, output_tokens, accept_lengths, pool_indices)
        self._propose(bs, pool_indices)

        staging = self._next_input_pinned[:bs]
        staging.copy_(torch.from_numpy(self._next_input_np[:bs]))
        return staging.to(self.device, non_blocking=True)

    # ------------------------------------------------------------------
    # History bookkeeping
    # ------------------------------------------------------------------

    def _update_history(
        self,
        base_ctx: ForwardContext,
        output_tokens: torch.Tensor,
        accept_lengths: torch.Tensor,
        pool_indices: np.ndarray,
    ) -> None:
        bs = base_ctx.bs

        if base_ctx.forward_mode == ForwardMode.EXTEND:
            num_extends = base_ctx.num_extends
            total = base_ctx.input_num_tokens
            input_ids = self.input_buffers.input_ids_buf[:total].to("cpu").numpy()
            input_lengths = (
                self.input_buffers.input_lengths_buf[:bs].to("cpu").numpy()
            )
            # extend_prefix_lens is only populated for prefill rows (first
            # ``num_extends`` entries) per the C++ scheduler's
            # FlatForwardOperation. A zero entry marks the first chunk of
            # a fresh prompt; reset the slot before appending.
            if num_extends > 0:
                extend_prefix_lens = (
                    self.input_buffers.extend_prefix_lens_buf[:num_extends]
                    .to("cpu")
                    .numpy()
                )
            else:
                extend_prefix_lens = np.empty((0,), dtype=np.int32)
            sampled = output_tokens.to("cpu").numpy().reshape(-1)
            append_sampled = not self.input_buffers.all_extends_mid_chunk

            offset = 0
            for i in range(bs):
                pool_idx = int(pool_indices[i])
                length = int(input_lengths[i])

                is_prefill_row = i < num_extends
                if is_prefill_row and int(extend_prefix_lens[i]) == 0:
                    self._reset_slot(pool_idx)

                self._append_to_slot(pool_idx, input_ids[offset : offset + length])
                if append_sampled and i < sampled.size:
                    self._append_to_slot(pool_idx, sampled[i : i + 1])
                offset += length
            return

        # TARGET_VERIFY: output_tokens is laid out as (bs * spec_num_tokens,)
        # and accept_lengths tells us how many of those columns were
        # accepted per request (1..N).
        verified = (
            output_tokens.to("cpu").numpy().reshape(bs, self.spec_num_tokens)
        )
        accepted_n = accept_lengths.to("cpu").numpy().astype(np.int32)
        for i in range(bs):
            pool_idx = int(pool_indices[i])
            n = int(accepted_n[i])
            if n <= 0:
                continue
            self._append_to_slot(pool_idx, verified[i, :n])

    # ------------------------------------------------------------------
    # Proposal
    # ------------------------------------------------------------------

    def _propose(self, bs: int, pool_indices: np.ndarray) -> None:
        propose_batch_into(
            history=self.history,
            history_len=self.history_len,
            pool_indices=pool_indices[:bs],
            out=self._next_input_np[:bs],
            min_ngram=self.min_ngram,
            max_ngram=self.max_ngram,
            spec_num_steps=self.spec_num_steps,
        )
