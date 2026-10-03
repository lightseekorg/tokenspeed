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
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tokenspeed_kernel.ops.sampling import argmax as sampling_argmax
from tokenspeed_kernel.ops.sampling.flashinfer import softmax
from tokenspeed_kernel.ops.sampling.triton import gumbel_sample_from_pools

if TYPE_CHECKING:
    from tokenspeed.runtime.execution.input_buffer import InputBuffers
    from tokenspeed.runtime.execution.runtime_states import RuntimeStates
    from tokenspeed.runtime.sampling.backends.base import SpeculativeSamplingPools

# The Gumbel kernel's two-stage path reduces the vocab in blocks of this many
# tokens (tokenspeed_kernel.ops.sampling.triton.gumbel._GUMBEL_BLOCK_SIZE).
_GUMBEL_BLOCK_SIZE = 1024

# Philox offset salt for the draft proposal stream. sample() keys the
# request's stream by (seed, valid_cache_length); the draft stream keys by
# (seed, SALT + valid_cache_length * N + step), far above any context length,
# so the two never share a draw and no two (round, step) pairs of one request
# do either (valid_cache_length grows every round).
DRAFT_SAMPLE_OFFSET_SALT = 1 << 40


class DraftProposalSampler:
    """Sample one draft step from ``q`` and record ``q`` for verify."""

    def __init__(
        self,
        *,
        pools: SpeculativeSamplingPools,
        runtime_states: RuntimeStates,
        input_buffers: InputBuffers,
        spec_num_tokens: int,
        device: torch.device | str,
    ) -> None:
        if runtime_states.draft_probs is None:
            raise RuntimeError(
                "DraftProposalSampler needs RuntimeStates.draft_probs; the executor "
                "allocates it under enable_speculative_sampling"
            )
        self._pools = pools
        self._draft_probs = runtime_states.draft_probs
        self._valid_cache_lengths = runtime_states.valid_cache_lengths
        self._req_pool_indices_buf = input_buffers.req_pool_indices_buf
        self._state_write_req_pool_indices_buf = (
            input_buffers.state_write_req_pool_indices_buf
        )
        self._spec_num_tokens = spec_num_tokens
        max_bs = input_buffers.max_bs
        pool_rows = self._valid_cache_lengths.shape[0]
        vocab_size = self._draft_probs.shape[2]
        gumbel_blocks = (vocab_size + _GUMBEL_BLOCK_SIZE - 1) // _GUMBEL_BLOCK_SIZE
        # Per-slot Philox offsets for this step, refreshed in place.
        self._offsets_pool = torch.zeros((pool_rows,), dtype=torch.int64, device=device)
        self._pool_indices_i32 = torch.empty((max_bs,), dtype=torch.int32, device=device)
        self._gumbel_out = torch.empty((max_bs,), dtype=torch.int32, device=device)
        self._gumbel_local_ids = torch.empty(
            (max_bs, gumbel_blocks), dtype=torch.int32, device=device
        )
        self._gumbel_local_scores = torch.empty(
            (max_bs, gumbel_blocks), dtype=torch.float32, device=device
        )

    def propose(
        self,
        logits: torch.Tensor,
        *,
        step: int,
        bs: int,
        vocab_map: torch.Tensor | None,
    ) -> torch.Tensor:
        """Sample this step's draft tokens and record their distribution.

        Args:
            logits: ``[bs, V_draft]`` draft logits, one row per request in
                batch order (padding rows included).
            step: Draft step index; the token lands in verify candidate
                column ``step + 1`` and ``q`` in ``draft_probs[:, step]``.
            bs: Rows in ``logits`` (the padded graph batch under replay).
            vocab_map: ``[V_draft]`` full-vocab id of each draft logit column
                (Eagle3 hot tokens), or None when the draft vocab is the
                target's.

        Returns:
            ``[bs]`` int32 draft-vocab token ids.
        """
        if step < 0 or step >= self._spec_num_tokens - 1:
            raise ValueError(
                f"draft step {step} has no verify column in a chain of "
                f"{self._spec_num_tokens} tokens"
            )
        if logits.shape[0] != bs:
            raise ValueError(f"expected {bs} draft logit rows, got {logits.shape[0]}")
        pool_indices = self._req_pool_indices_buf[:bs]
        pool_indices_i32 = self._pool_indices_i32[:bs]
        pool_indices_i32.copy_(pool_indices)
        temperature = self._pools.temperature.index_select(0, pool_indices)
        greedy_rows = (self._pools.top_k.index_select(0, pool_indices) == 1).view(-1, 1)

        # q at the request's temperature, fp32 like the verifier's target probs.
        q = softmax(logits, temperature=temperature.view(-1, 1))
        canonical = sampling_argmax(logits).to(torch.int64).view(-1, 1)
        # Greedy rows: one-hot at the canonical argmax, so verify stays exact.
        q.mul_((~greedy_rows).to(q.dtype))
        q.scatter_add_(1, canonical, greedy_rows.to(q.dtype))

        # Gumbel-max over logits / T draws exactly Categorical(q); the noise is
        # keyed by the request's seed and this (round, step) offset.
        torch.add(
            self._valid_cache_lengths.to(torch.int64) * self._spec_num_tokens,
            DRAFT_SAMPLE_OFFSET_SALT + step,
            out=self._offsets_pool,
        )
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
        tokens = torch.where(
            greedy_rows.view(-1), canonical.view(-1).to(sampled.dtype), sampled
        )

        if vocab_map is not None:
            full = torch.zeros(
                (bs, self._draft_probs.shape[2]), dtype=q.dtype, device=q.device
            )
            full.index_copy_(1, vocab_map.to(torch.int64), q)
            q = full
        # Padding rows resolve to the reserved last slot, which verify never
        # gathers.
        self._draft_probs[:, step, :].index_copy_(
            0, self._state_write_req_pool_indices_buf[:bs], q
        )
        return tokens
