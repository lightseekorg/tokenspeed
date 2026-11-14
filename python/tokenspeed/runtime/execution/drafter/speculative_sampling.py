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

"""Sampled draft proposals for draft-prob rejection sampling.

Under ``--enable-speculative-sampling`` a chain drafter no longer proposes the
argmax of its logits: each step's token is drawn from the drafter's own
distribution ``q = softmax(logits / T)`` at the request's temperature, and
``q`` is recorded per request and step so the next round's verify can run
the standard accept test ``coin * q(x) < p(x)`` with residual
``norm(relu(p - q))``. The output law is the target's ``p`` as long as the
proposal really follows the recorded ``q``; the temperature only aligns ``q``
with ``p`` for a higher acceptance rate (top-k / top-p / penalties are the
verifier's business and stay out of ``q``).

Greedy requests (``top_k == 1``) keep the canonical lowest-index argmax and
record a one-hot ``q``, so their verify stays exactly greedy. Everything here
is tensor-only (``torch.where`` over rows, pool-indexed gathers), so the
captured decode graph records one path for every request mix.

The proposal is drawn with the in-tree Triton Gumbel-max kernel keyed per
request by the verifier's seed pool and a salted position, so the draft
stream is run- and batch-invariant like the per-slot verify coins.

When ``draft_probs`` holds each rank's vocabulary columns only (the target's
rows stay sharded and the verifier reads q per shard), the draft head hands
over its shard instead of gathered logits (``propose_vocab_shard``): each
rank reduces its shard to a few partials, the ranks gather those, and every
rank draws the token the whole row would draw (the Gumbel noise is keyed by
global token id) and records q over its own columns. A hot-token head (Eagle3
``d2t``) samples its gathered logits as with whole rows, because its columns
do not line up with the target's shards, and each rank records only the columns
of its tokens that fall in the rank's shard.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tokenspeed_kernel.ops.sampling import argmax as sampling_argmax
from tokenspeed_kernel.ops.sampling import (
    vocab_shard_draft_record,
    vocab_shard_draft_stats,
    vocab_shard_draft_stats_width,
)
from tokenspeed_kernel.ops.sampling.triton import (
    DRAFT_SAMPLE_OFFSET_SALT,
    gumbel_sample_from_pools,
    gumbel_scratch_shape,
)
from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.layers.sampled_vocab_shard import gather_topk_pairs

if current_platform().is_nvidia:
    from tokenspeed_kernel.ops.sampling.flashinfer import softmax
else:

    def softmax(logits: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
        """``softmax(logits / temperature)`` in fp32, off NVIDIA (no FlashInfer)."""
        return torch.softmax(torch.div(logits.float(), temperature), dim=-1)


if TYPE_CHECKING:
    from tokenspeed.runtime.execution.input_buffer import InputBuffers
    from tokenspeed.runtime.execution.runtime_states import RuntimeStates
    from tokenspeed.runtime.layers.sampled_vocab_shard import SampledVocabShard
    from tokenspeed.runtime.sampling.backends.base import SpeculativeSamplingPools


class DraftProposalSampler:
    """Sample one draft step from ``q`` and record ``q`` for verify."""

    def __init__(
        self,
        *,
        pools: SpeculativeSamplingPools,
        runtime_states: RuntimeStates,
        input_buffers: InputBuffers,
        spec_num_tokens: int,
        vocab_map: torch.Tensor | None,
        device: torch.device | str,
    ) -> None:
        """
        Args:
            pools: The verifier's pool-indexed temperature / top-k / seed.
            runtime_states: Owner of ``draft_probs`` (must be allocated; over
                whole-vocabulary rows or this rank's vocabulary columns, from
                ``draft_probs_vocab_start``) and ``valid_cache_lengths``.
            input_buffers: The executor's batch-ordered pool index buffers.
            spec_num_tokens: Verify chain width N; steps ``0..N-2`` record.
            vocab_map: ``[V_draft]`` full-vocab id of each draft logit column
                (Eagle3 hot tokens), or None when the draft vocab is the
                target's. Bound once: the recorded rows scatter through it,
                keeping only this rank's columns of a shard table.
            device: Where the scratch lives.
        """
        if runtime_states.draft_probs is None:
            raise RuntimeError(
                "DraftProposalSampler needs RuntimeStates.draft_probs; the executor "
                "allocates it under enable_speculative_sampling"
            )
        self._pools = pools
        self._draft_probs = runtime_states.draft_probs
        self._sentinel = runtime_states.draft_probs_sentinel
        self._valid_cache_lengths = runtime_states.valid_cache_lengths
        self._req_pool_indices_buf = input_buffers.req_pool_indices_buf
        self._state_write_req_pool_indices_buf = (
            input_buffers.state_write_req_pool_indices_buf
        )
        self._spec_num_tokens = spec_num_tokens
        max_bs = input_buffers.max_bs
        pool_rows = self._valid_cache_lengths.shape[0]
        vocab_size = self._draft_probs.shape[2]
        # Per-slot Philox offsets for this step, refreshed in place.
        self._offsets_pool = torch.zeros((pool_rows,), dtype=torch.int64, device=device)
        self._pool_indices_i32 = torch.empty(
            (max_bs,), dtype=torch.int32, device=device
        )
        self._gumbel_out = torch.empty((max_bs,), dtype=torch.int32, device=device)
        # Shard scratch when the draft head samples this rank's columns of q, else the whole-row scratch; the other stays None.
        self._shard_ranks: int | None = None
        self._shard_stats: torch.Tensor | None = None
        self._shard_landing: torch.Tensor | None = None
        self._gumbel_local_ids: torch.Tensor | None = None
        self._gumbel_local_scores: torch.Tensor | None = None
        # Hot-token heads scatter q into a full-vocab row, or this rank's columns of one; unmapped columns stay zero from allocation.
        self._vocab_map: torch.Tensor | None = None
        self._vocab_select: torch.Tensor | None = None
        self._full_q: torch.Tensor | None = None
        shard_table = vocab_size != runtime_states.vocab_size
        if shard_table and runtime_states.vocab_size % vocab_size:
            raise ValueError(
                f"draft_probs over {vocab_size} of {runtime_states.vocab_size} "
                "columns must be one of equal vocabulary shards"
            )
        if shard_table and vocab_map is None:
            self._shard_ranks = runtime_states.vocab_size // vocab_size
            width = vocab_shard_draft_stats_width(vocab_size)
            self._shard_stats = torch.empty(
                (max_bs, width), dtype=torch.float32, device=device
            )
            self._shard_landing = torch.empty(
                (self._shard_ranks * max_bs * width,),
                dtype=torch.float32,
                device=device,
            )
        else:
            # Sized by the draft head's width: the Gumbel draw runs over the draft
            # logits, which a hot-token head keeps narrower than the full vocab.
            draft_vocab_size = vocab_size if vocab_map is None else vocab_map.shape[0]
            scratch_shape = gumbel_scratch_shape(max_bs, draft_vocab_size)
            self._gumbel_local_ids = torch.empty(
                scratch_shape, dtype=torch.int32, device=device
            )
            self._gumbel_local_scores = torch.empty(
                scratch_shape, dtype=torch.float32, device=device
            )
            if vocab_map is not None:
                if (
                    vocab_map.ndim != 1
                    or vocab_map.shape[0] > runtime_states.vocab_size
                ):
                    raise ValueError(
                        f"vocab_map must be a [V_draft <= {runtime_states.vocab_size}] "
                        f"vector, got {tuple(vocab_map.shape)}"
                    )
                self._vocab_map = vocab_map.to(device=device, dtype=torch.int64)
                if shard_table:
                    # This rank's columns: the draft columns whose tokens fall in its shard.
                    start = runtime_states.draft_probs_vocab_start
                    mine = (self._vocab_map >= start) & (
                        self._vocab_map < start + vocab_size
                    )
                    self._vocab_select = mine.nonzero().view(-1)
                    self._vocab_map = self._vocab_map[mine] - start
                self._full_q = torch.zeros(
                    (max_bs, vocab_size), dtype=torch.float32, device=device
                )

    def _check_step(self, step: int) -> None:
        """Refuse a draft step without a verify column."""
        if step < 0 or step >= self._spec_num_tokens - 1:
            raise ValueError(
                f"draft step {step} has no verify column in a chain of "
                f"{self._spec_num_tokens} tokens"
            )

    def _refresh_offsets(self, step: int) -> None:
        """This step's Philox offset per slot: its round's position and the salted step."""
        torch.add(
            self._valid_cache_lengths.to(torch.int64) * self._spec_num_tokens,
            DRAFT_SAMPLE_OFFSET_SALT + step,
            out=self._offsets_pool,
        )

    def propose(self, logits: torch.Tensor, *, step: int) -> torch.Tensor:
        """Sample this step's draft tokens and record their distribution.

        Args:
            logits: ``[bs, V_draft]`` draft logits, one row per request in
                batch order (padding rows included; ``bs`` is the padded
                graph batch under replay).
            step: Draft step index; the token lands in verify candidate
                column ``step + 1`` and ``q`` in ``draft_probs[:, step]``.

        Returns:
            ``[bs]`` int32 draft-vocab token ids.
        """
        self._check_step(step)
        bs = logits.shape[0]
        if self._shard_ranks is not None:
            # A forward without logits rows (a cache-only chunk) hands over no shard and proposes nothing.
            if bs == 0:
                return self._gumbel_out[:0]
            raise RuntimeError(
                "draft_probs hold this rank's vocabulary columns: propose from "
                "the draft head's shard (propose_vocab_shard)"
            )
        pool_indices = self._req_pool_indices_buf[:bs]
        pool_indices_i32 = self._pool_indices_i32[:bs]
        pool_indices_i32.copy_(pool_indices)
        temperature = self._pools.temperature.index_select(0, pool_indices)
        greedy_rows = (self._pools.top_k.index_select(0, pool_indices) == 1).view(-1, 1)

        # q at the request's temperature, fp32 like the verifier's target probs.
        q = softmax(logits, temperature=temperature.view(-1, 1))
        # A row whose logits give no finite distribution (all NaN, or an
        # overflow) proposes nothing: it is recorded as the sentinel below.
        no_proposal = ~torch.isfinite(q).all(dim=1, keepdim=True)
        # The argmax kernel marks an all-NaN row with -1; clamp it to a real
        # column so the one-hot scatter below stays in bounds.
        canonical = sampling_argmax(logits).to(torch.int64).clamp_min_(0).view(-1, 1)
        # Greedy rows: one-hot at the canonical argmax, so verify stays exact.
        # masked_fill_, not a multiply by 0: NaN * 0 is NaN.
        q.masked_fill_(greedy_rows, 0.0)
        q.scatter_add_(1, canonical, greedy_rows.to(q.dtype))
        # Verify reads the sentinel as "no proposal": it rejects the token
        # and samples from the full target, as for an unrecorded slot. A NaN
        # q would also fail coin * q < p, but then poison the residual
        # relu(p - q) the replacement token is drawn from.
        q.masked_fill_(no_proposal, self._sentinel)

        # Gumbel-max over logits / T draws exactly Categorical(q); the noise is
        # keyed by the request's seed and this (round, step) offset.
        self._refresh_offsets(step)
        sampled = gumbel_sample_from_pools(
            logits,
            pool_indices_i32,
            self._pools.temperature,
            self._pools.seed,
            self._offsets_pool,
            self._gumbel_local_ids[:bs],
            self._gumbel_local_scores[:bs],
            self._gumbel_out[:bs],
        )
        # An all-NaN row has no maximum; the kernel resolves it to the first
        # masked column (vocab_size). Keep the id a real column so the hot-token
        # map and the verify gather stay in bounds; its sentinel q rejects it.
        sampled.clamp_(0, logits.shape[1] - 1)
        tokens = torch.where(
            greedy_rows.view(-1), canonical.view(-1).to(sampled.dtype), sampled
        )

        if self._vocab_map is not None:
            assert self._full_q is not None
            full = self._full_q[:bs]
            if self._vocab_select is not None:
                q = q.index_select(1, self._vocab_select)
            full.index_copy_(1, self._vocab_map, q)
            q = full
        # Padding rows resolve to the reserved last slot, which verify never
        # gathers.
        self._draft_probs[:, step, :].index_copy_(
            0, self._state_write_req_pool_indices_buf[:bs], q
        )
        return tokens

    def propose_vocab_shard(
        self, shard: SampledVocabShard, *, step: int
    ) -> torch.Tensor:
        """``propose`` from this rank's shard of the draft head's logits.

        Every rank draws the token ``propose`` would draw from the gathered
        rows (the same Gumbel noise, keyed by global token id) and records q
        over its own vocabulary columns: ``softmax(logits / T)``, one-hot for
        greedy rows, the sentinel for rows without a finite softmax. Only a
        few partials per row cross the ranks.

        Args:
            shard: The step's draft head shard, one row per request in batch
                order (padding rows included).
            step: Draft step index; the token lands in verify candidate
                column ``step + 1`` and ``q`` in ``draft_probs[:, step]``.

        Returns:
            ``[bs]`` int32 token ids.
        """
        self._check_step(step)
        if self._shard_ranks is None or shard.tp_size != self._shard_ranks:
            raise RuntimeError(
                f"draft_probs hold whole rows, a hot-token head's columns or another "
                f"shard count than the head's {shard.tp_size}: propose from gathered "
                "logits instead"
            )
        logits = shard.logits
        bs = logits.shape[0]
        slots = self._req_pool_indices_buf[:bs]
        stats = self._shard_stats[:bs]
        self._refresh_offsets(step)
        vocab_shard_draft_stats(
            logits,
            shard.vocab_start,
            slots,
            self._pools.temperature,
            self._pools.seed,
            self._offsets_pool,
            stats,
            logit_scale=shard.logit_scale,
            softcap=shard.softcap,
        )
        landing = self._shard_landing[: shard.tp_size * stats.numel()]
        gathered = gather_topk_pairs(
            stats,
            landing.view(shard.tp_size * bs, stats.shape[1]),
            shard.tp_size,
            shard.tp_group,
        )
        tokens = self._gumbel_out[:bs]
        vocab_shard_draft_record(
            logits,
            gathered,
            shard.vocab_start,
            slots,
            self._state_write_req_pool_indices_buf[:bs],
            self._pools.temperature,
            self._pools.top_k,
            self._draft_probs[:, step],
            tokens,
            logit_scale=shard.logit_scale,
            softcap=shard.softcap,
            sentinel=self._sentinel,
        )
        return tokens
