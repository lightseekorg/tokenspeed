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

"""DFlash2 proposal path on top of TokenSpeed's native DFlash block runtime."""

from __future__ import annotations

import torch

from tokenspeed.runtime.execution.drafter.dflash import DFlash
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.logits_processor import LogitsMetadata, LogitsProcessor
from tokenspeed.runtime.utils.nvtx import nvtx_range


def _walk_greedy_path(
    candidate_ids: torch.Tensor,
    scores: torch.Tensor,
    anchor_token_ids: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """Greedily walk a fixed DFlash2 lattice without host-side tensor reads."""
    batch_size, num_steps, top_k = candidate_ids.shape
    out[:, 0].copy_(anchor_token_ids)
    previous = torch.zeros(batch_size, dtype=torch.int64, device=candidate_ids.device)
    for step in range(num_steps):
        transitions = torch.gather(
            scores[:, step],
            1,
            previous[:, None, None].expand(-1, 1, top_k),
        ).squeeze(1)
        previous = torch.argmax(transitions, dim=-1)
        token = torch.gather(candidate_ids[:, step], 1, previous[:, None]).squeeze(1)
        out[:, step + 1].copy_(token)
    return out


class DFlash2(DFlash):
    """DFlash block runtime with the DFlash2 top-k transition selector."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.candidate_selector = getattr(self.model, "candidate_selector", None)
        if self.candidate_selector is None:
            raise ValueError(
                "DFlash2 requires a draft model with candidate_selector weights."
            )
        if self.draft_query_width != self.spec_num_tokens:
            raise ValueError("DFlash2 requires the anchor-plus-mask DFlash layout.")

        config = self.model.config
        nested = getattr(config, "dflash_config", {}) or {}
        self.selector_top_k = int(nested.get("selector_top_k"))
        self.output_multiplier = float(nested.get("output_multiplier", 1.0))
        self.final_logit_softcapping = float(
            nested.get("final_logit_softcapping") or 0.0
        )
        self.candidate_logits_processor: LogitsProcessor | None = None

    def wire_target(self, target_model) -> None:
        super().wire_target(target_model)
        self.candidate_logits_processor = LogitsProcessor(
            self.model.config,
            logit_scale=self.output_multiplier,
            tp_rank=self.logits_processor.tp_rank,
            tp_size=self.logits_processor.tp_size,
            tp_group=self.logits_processor.tp_group,
        )
        self.candidate_logits_processor.final_logit_softcapping = (
            self.final_logit_softcapping if self.final_logit_softcapping > 0 else None
        )

    def _compute_candidates(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.candidate_logits_processor is None:
            raise RuntimeError("DFlash2 must be wired to the target before drafting.")
        metadata = LogitsMetadata(forward_mode=ForwardMode.DECODE)
        logits = self.candidate_logits_processor._get_logits(
            hidden_states, self.lm_head, metadata
        )
        unary_logits, candidate_ids = torch.topk(
            logits, self.selector_top_k, dim=-1, sorted=False
        )
        # Match DFlash2's selector contract: unary scores are accumulated in
        # FP32 even when the vocabulary-parallel head emits BF16 logits.
        return candidate_ids, unary_logits.float()

    @nvtx_range("dflash2_sample_block", color="purple")
    def _sample_block(
        self,
        draft_hidden: torch.Tensor,
        block_ids: torch.Tensor,
        next_tokens: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = draft_hidden[:, 1:, :]
        batch_size, num_steps, _ = hidden_states.shape
        candidate_ids, unary_logits = self._compute_candidates(
            hidden_states.reshape(-1, self.hidden_size)
        )
        candidate_ids = candidate_ids.view(batch_size, num_steps, self.selector_top_k)
        unary_logits = unary_logits.view_as(candidate_ids)
        anchor_token_ids = (
            block_ids[:, 0]
            .to(torch.int64)
            .clamp(0, int(self.model.config.vocab_size) - 1)
        )
        scores = self.candidate_selector(
            candidate_ids,
            unary_logits,
            hidden_states,
            anchor_token_ids,
        )
        _walk_greedy_path(candidate_ids, scores, anchor_token_ids, next_tokens)
        next_tokens.clamp_(min=0, max=int(self.model.config.vocab_size) - 1)
        return next_tokens
