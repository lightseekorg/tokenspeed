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
from tokenspeed_kernel.ops.sampling.triton import dflash2_greedy_path

from tokenspeed.runtime.distributed.comm_ops import all_gather_into_tensor
from tokenspeed.runtime.execution.drafter.dflash import DFlash
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.logits_processor import (
    LogitsMetadata,
    LogitsProcessor,
    fused_softcap_generic,
)
from tokenspeed.runtime.utils import get_colorful_logger
from tokenspeed.runtime.utils.nvtx import nvtx_range

logger = get_colorful_logger(__name__)


def _greedy_path_torch(
    candidate_ids: torch.Tensor,
    scores: torch.Tensor,
    anchor_token_ids: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """Torch reference for the walk; CUDA batches take the Triton kernel."""
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


def _walk_greedy_path(
    candidate_ids: torch.Tensor,
    scores: torch.Tensor,
    anchor_token_ids: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """Greedily walk a fixed DFlash2 lattice without host-side tensor reads."""
    if scores.is_cuda:
        return dflash2_greedy_path(candidate_ids, scores, anchor_token_ids, out)
    return _greedy_path_torch(candidate_ids, scores, anchor_token_ids, out)


class DFlash2(DFlash):
    """DFlash block runtime with the DFlash2 top-k transition selector."""

    #: Set from the wired head's shard geometry; see _init_distributed_topk.
    _distributed_topk_enabled = False

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
        self._distributed_topk_enabled = False
        self._candidate_gather_buffers: tuple[torch.Tensor, torch.Tensor] | None = None

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
        self._init_distributed_topk()

    def _init_distributed_topk(self) -> None:
        """Decide once whether top-k can skip the whole-vocabulary gather.

        Only ``selector_top_k`` candidates per row survive, so all-gathering
        every rank's whole logits row moves three orders of magnitude more
        bytes than the answer needs. Taking each shard's local top-k first is
        exact only when a shard's local index maps to a global token id by a
        constant offset, which is what these checks establish.
        """
        processor = self.candidate_logits_processor
        head = self.lm_head
        shard = getattr(head, "shard_indices", None)
        self._distributed_topk_enabled = False
        if (
            int(processor.tp_size) <= 1
            or shard is None
            or not hasattr(head, "weight")
            or processor.skip_all_gather
            or processor.dp_sampling_enabled
        ):
            return
        num_org = int(shard.num_org_elements)
        if int(self.model.config.vocab_size) > 2**24:
            logger.info(
                "DFlash2 distributed top-k disabled: a token id in a "
                "%d-token vocabulary is not exact in the fp32 the packed "
                "all-gather carries it in.",
                int(self.model.config.vocab_size),
            )
            return
        if (
            int(shard.num_added_elements) != 0
            or int(shard.num_org_elements_padded) != num_org
            or num_org * int(processor.tp_size) != int(self.model.config.vocab_size)
            or int(shard.org_vocab_start_index) != num_org * int(processor.tp_rank)
        ):
            logger.info(
                "DFlash2 distributed top-k disabled: this %d-way vocabulary shard "
                "is padded or carries added tokens, so a shard-local index is not "
                "a global token id.",
                int(processor.tp_size),
            )
            return
        self._distributed_topk_enabled = True

    def _ensure_candidate_gather_buffers(
        self, rows: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Resident staging and landing pad for one packed all-gather.

        Values and ids ride the same fp32 rows, so a step pays one collective
        rather than two. ``_init_distributed_topk`` checks the vocabulary is
        small enough for an id to be exact in fp32.
        """
        capacity = int(self.input_buffers.max_bs) * max(self.spec_num_tokens - 1, 1)
        tp_size = int(self.candidate_logits_processor.tp_size)
        width = 2 * self.selector_top_k
        buffers = self._candidate_gather_buffers
        if buffers is None or buffers[0].device != device:
            buffers = (
                torch.empty((capacity, width), dtype=torch.float32, device=device),
                torch.empty(
                    (tp_size * capacity, width), dtype=torch.float32, device=device
                ),
            )
            self._candidate_gather_buffers = buffers
        return buffers[0][:rows], buffers[1][: rows * tp_size]

    def _distributed_topk_candidates(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Top-k over a vocab-parallel head without gathering whole rows."""
        processor = self.candidate_logits_processor
        shard = self.lm_head.shard_indices
        weight = self.lm_head.weight
        top_k = self.selector_top_k

        logits = torch.matmul(
            hidden_states.to(weight.dtype),
            weight[: int(shard.num_org_elements)].T,
        )
        if processor.logit_scale is not None:
            logits.mul_(processor.logit_scale)
        values, ids = torch.topk(logits, top_k, dim=-1, sorted=False)

        tp_size = int(processor.tp_size)
        rows = int(values.shape[0])
        staged, gathered = self._ensure_candidate_gather_buffers(rows, values.device)
        staged[:, :top_k].copy_(values)
        staged[:, top_k:].copy_(ids)
        staged[:, top_k:].add_(float(shard.org_vocab_start_index))
        all_gather_into_tensor(gathered, staged, processor.tp_group)

        # Rank-major to row-major, so one row's candidates from every rank sit
        # side by side for the final selection, values and ids each on a plane.
        planes = (
            gathered.view(tp_size, rows, 2, top_k)
            .permute(1, 2, 0, 3)
            .reshape(rows, 2, tp_size * top_k)
        )
        # Sorted, so the surviving candidate order does not depend on which
        # rank happened to contribute a value.
        unary_logits, lanes = torch.topk(planes[:, 0], top_k, dim=-1, sorted=True)
        # Everything downstream indexes codebooks with these.
        candidate_ids = torch.gather(planes[:, 1], 1, lanes).to(torch.int64)
        if processor.final_logit_softcapping:
            # Monotone in the logit, so it changes the scores the selector sees
            # but never which candidates got here.
            fused_softcap_generic(unary_logits, processor.final_logit_softcapping)
        return candidate_ids, unary_logits

    def _compute_candidates(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.candidate_logits_processor is None:
            raise RuntimeError("DFlash2 must be wired to the target before drafting.")
        if self._distributed_topk_enabled:
            return self._distributed_topk_candidates(hidden_states)
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
