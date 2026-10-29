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

"""DFlash2 proposal path on top of TokenSpeed's native DFlash block runtime.

The candidate selector scores, for every draft step, ``K`` candidate tokens
against each of the previous step's ``K`` candidates; the walk then picks one
candidate per step. Greedy requests take the best edge (lowest index on ties).
Under ``--enable-speculative-sampling`` a sampling request instead draws each
step's candidate from ``q = softmax(scores / T)`` over the ``K`` edges out of
the previously drawn one, and the walk records ``q`` per step in
``RuntimeStates.future_draft_q`` (a sparse q: ``K`` entries per step, not the
vocabulary) so the next round's verify can run rejection sampling against it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tokenspeed_kernel.ops.sampling.triton import dflash2_sample_path
from typing_extensions import override

from tokenspeed.runtime.execution.drafter.dflash import DFlash
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.logits_processor import LogitsMetadata, LogitsProcessor
from tokenspeed.runtime.layers.vocab_parallel_topk import VocabParallelTopK
from tokenspeed.runtime.sampling.backends.base import SpeculativeSamplingPools
from tokenspeed.runtime.utils.nvtx import nvtx_range

if TYPE_CHECKING:
    from tokenspeed.runtime.sampling.backends.base import SamplingBackend
    from tokenspeed.runtime.sampling.tree_verify import DraftTreeQ


class DFlash2(DFlash):
    """DFlash block runtime with the DFlash2 top-k transition selector."""

    #: Built in wire_target from the wired head's shard geometry.
    candidate_topk: VocabParallelTopK | None = None
    # The walk samples each step from the selector's q and records it.
    supports_speculative_sampling = True

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
        self.candidate_topk: VocabParallelTopK | None = None

        # The walk's per-request sampling parameters. Until the flag binds the
        # verifier's pools (bind_sampling_backend), every row reads row 0 of
        # one greedy pool (top_k 1), so the walk is the plain best-edge walk.
        device = self.device
        self._walk_pools = SpeculativeSamplingPools(
            temperature=torch.ones((1,), dtype=torch.float32, device=device),
            top_k=torch.ones((1,), dtype=torch.int32, device=device),
            top_p=torch.ones((1,), dtype=torch.float32, device=device),
            seed=torch.zeros((1,), dtype=torch.int64, device=device),
        )
        # Where the walk records q for verify; None while drafting greedily.
        self._draft_q: DraftTreeQ | None = None
        max_bs = self.input_buffers.max_bs
        num_steps = self.spec_num_tokens - 1
        top_k = self.selector_top_k
        # Row -> pool index the walk reads its parameters at (all 0 while
        # greedy), and the per-request draft noise offset base.
        self._walk_pool_indices = torch.zeros(
            (max_bs,), dtype=torch.int32, device=device
        )
        self._walk_offsets = torch.zeros((max_bs,), dtype=torch.int64, device=device)
        # Each row's frontier (valid cache length), gathered without a
        # temporary before it is scaled into _walk_offsets.
        self._walk_frontier = torch.zeros((max_bs,), dtype=torch.int32, device=device)
        # The walk's chain in the draft-q table layout (published to
        # _draft_q): each step's q and candidate ids, and each node's lane and
        # q slot (-1 on every node when the request proposes nothing).
        self._walk_q = torch.zeros(
            (max_bs, num_steps, top_k), dtype=torch.float32, device=device
        )
        self._walk_q_ids = torch.zeros(
            (max_bs, num_steps, top_k), dtype=torch.int32, device=device
        )
        self._walk_pos = torch.zeros(
            (max_bs, num_steps + 1), dtype=torch.int32, device=device
        )
        self._walk_slot = torch.zeros(
            (max_bs, num_steps + 1), dtype=torch.int32, device=device
        )

    @override
    def bind_sampling_backend(self, sampling_backend: SamplingBackend) -> None:
        """Arm the sampled walk under --enable-speculative-sampling.

        The walk reads the verifier's per-request temperature / top-k / top-p
        / seed and records each step's q over the selector's candidates in
        ``RuntimeStates.future_draft_q`` (allocated here). No dense per-vocab
        ``draft_probs``: the selector only reads its top-K head, so the draft
        head keeps its shard-local top-K and needs no full-vocab logits.

        Args:
            sampling_backend: The executor's verifier, owner of the
                pool-indexed sampling scalars.
        """
        super().bind_sampling_backend(sampling_backend)
        if not sampling_backend.config.enable_speculative_sampling:
            return
        self._walk_pools = sampling_backend.speculative_sampling_pools()
        self.runtime_states.init_draft_tree_q(
            num_nodes=self.spec_num_tokens,
            num_slots=self.spec_num_tokens - 1,
            num_candidates=self.selector_top_k,
        )
        self._draft_q = self.runtime_states.future_draft_q

    def wire_target(self, target_model) -> None:
        super().wire_target(target_model)
        # The target's LM-head layout, so a vocab-sharded head under attention
        # DP (--lm-head-tp-size) transposes its logits back the same way.
        self.candidate_logits_processor = LogitsProcessor(
            self.model.config,
            skip_all_gather=self.logits_processor.skip_all_gather,
            logit_scale=self.output_multiplier,
            tp_rank=self.logits_processor.tp_rank,
            tp_size=self.logits_processor.tp_size,
            tp_group=self.logits_processor.tp_group,
            dp_lm_head_tp=self.logits_processor.dp_lm_head_tp,
        )
        self.candidate_logits_processor.final_logit_softcapping = (
            self.final_logit_softcapping if self.final_logit_softcapping > 0 else None
        )
        processor = self.candidate_logits_processor
        self.candidate_topk = VocabParallelTopK(
            self.lm_head,
            tp_size=processor.tp_size,
            tp_rank=processor.tp_rank,
            tp_group=processor.tp_group,
            vocab_size=self.model.config.vocab_size,
            top_k=self.selector_top_k,
            max_rows=self.input_buffers.max_bs * max(self.spec_num_tokens - 1, 1),
            logit_scale=processor.logit_scale,
            softcapping=processor.final_logit_softcapping,
            skip_all_gather=processor.skip_all_gather,
            dp_sampling_enabled=processor.dp_sampling_enabled,
        )

    def _compute_candidates(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.candidate_logits_processor is None:
            raise RuntimeError("DFlash2 must be wired to the target before drafting.")
        if self.candidate_topk.enabled:
            return self.candidate_topk(hidden_states)
        # Replicated draft rows: no query shard.
        metadata = LogitsMetadata(forward_mode=ForwardMode.DECODE, query_shard=None)
        logits = self.candidate_logits_processor._get_logits(
            hidden_states, self.lm_head, metadata, require_full_vocab=False
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
        self._walk(candidate_ids, scores, anchor_token_ids, next_tokens)
        next_tokens.clamp_(min=0, max=int(self.model.config.vocab_size) - 1)
        return next_tokens

    def _walk(
        self,
        candidate_ids: torch.Tensor,
        scores: torch.Tensor,
        anchor_token_ids: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        """Walk the lattice into ``out`` and, when sampling, record its q.

        One kernel for every row: greedy rows (and every row while the flag
        is off) take the best edge; sampling rows draw from q, with noise
        keyed by the request's seed, this round's position and the drawn
        token id, so every TP rank draws the same token from the same
        replicated scores. Tensor-only, so it records into the draft graph.
        """
        bs = candidate_ids.shape[0]
        draft_q = self._draft_q
        if draft_q is not None:
            pool_indices = self.input_buffers.req_pool_indices_buf[:bs]
            self._walk_pool_indices[:bs].copy_(pool_indices)
            # Draft noise offset base valid_cache_length * N, the convention
            # the step drafters' DraftProposalSampler uses; the kernel adds
            # DRAFT_SAMPLE_OFFSET_SALT + step. The frontier grows every round,
            # so no two rounds of a request share a draw.
            frontier = self._walk_frontier[:bs]
            torch.index_select(
                self.runtime_states.valid_cache_lengths, 0, pool_indices, out=frontier
            )
            torch.mul(frontier, self.spec_num_tokens, out=self._walk_offsets[:bs])
        pools = self._walk_pools
        dflash2_sample_path(
            candidate_ids,
            scores,
            anchor_token_ids,
            self._walk_pool_indices[:bs],
            pools.temperature,
            pools.top_k,
            pools.top_p,
            pools.seed,
            self._walk_offsets[:bs],
            out,
            self._walk_q[:bs],
            self._walk_q_ids[:bs],
            self._walk_pos[:bs],
            self._walk_slot[:bs],
        )
        if draft_q is not None:
            draft_q.publish_chain(
                self.input_buffers.state_write_req_pool_indices_buf[:bs],
                self._walk_q[:bs],
                self._walk_q_ids[:bs],
                self._walk_pos[:bs],
                self._walk_slot[:bs],
            )
