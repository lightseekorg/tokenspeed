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
from tokenspeed_kernel.ops.sampling.cuda import chain_speculative_sampling_target_only
from tokenspeed_kernel.ops.sampling.flashinfer import (
    min_p_sampling_from_probs,
    softmax,
    top_k_renorm_prob,
    top_p_renorm_prob,
)
from tokenspeed_kernel.ops.sampling.triton import gather_and_expand_scalars
from tokenspeed_kernel.torch_compile import get_compiler_backend

from tokenspeed.runtime.sampling.backends.base import (
    SPECULATIVE_ACCEPT_THRESHOLD_ACC,
    SPECULATIVE_ACCEPT_THRESHOLD_SINGLE,
    SamplingBackendConfig,
)
from tokenspeed.runtime.sampling.backends.flashinfer import FlashInferSamplingBackend
from tokenspeed.runtime.sampling.registry import register_backend
from tokenspeed.runtime.sampling.utils import nan_guard_logits
from tokenspeed.runtime.utils.nvtx import nvtx_range

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.logits_processor import LogitsProcessorOutput
    from tokenspeed.runtime.sampling.sampling_batch_info import SamplingBatchInfo
    from tokenspeed.runtime.sampling.sampling_params import SamplingParams


class FlashInferFullSamplingBackend(FlashInferSamplingBackend):
    """Superset of `flashinfer` adding min_p, frequency/presence/repetition
    penalties, and per-token logit_bias, for both single-step sampling and
    multi-step spec-decode verification.

    Stochastic path runs the 4-kernel sequence softmax(temperature) →
    top_k_renorm → top_p_renorm → min_p_sampling, unconditionally (requests
    with min_p == 0 are a no-op through min_p_sampling_from_probs) so the
    captured CUDA graph matches the runtime flow.

    Layout:
      * Per-pool-idx token counts (int32[max_req_pool_size, vocab]) —
        accumulated after each sample/verify. Zeroed when a pool slot is
        re-assigned to a new rid (see `on_pool_assignment`).
      * Per-pool-idx logit bias (bf16[max_req_pool_size, vocab]) — zero by
        default, scattered from SamplingParams.logit_bias on pool
        assignment. Added to logits per step.
      * Per-batch-row bf16 penalty scalars flowing through SamplingBatchInfo.

    sample() / verify() apply (in order, BEFORE temperature/softmax):
      1. repetition (multiplicative): logits = where(count>0,
         where(logits>0, logits/rep, logits*rep), logits)
      2. frequency + presence (additive):
         logits -= freq_pen * count + pres_pen * (count>0)
      3. logit_bias (additive): logits += logit_bias[req_pool_idx]

    Post-sample/verify, accumulate accepted tokens into counts.

    Out of scope in this iteration: min_new_tokens EOS mask, grammar vocab
    mask. Both remain silently-ignored no-ops.
    """

    def __init__(self, config: SamplingBackendConfig) -> None:

        super().__init__(config)

        if config.max_req_pool_size <= 0 or config.vocab_size <= 0:

            raise ValueError(
                "FlashInferFullSamplingBackend requires max_req_pool_size > 0 and "
                f"vocab_size > 0; got max_req_pool_size={config.max_req_pool_size}, "
                f"vocab_size={config.vocab_size}"
            )

        # Valid pool indices run 0..max_req_pool_size inclusive.
        pool_rows = config.max_req_pool_size + 1

        self._counts = torch.zeros(
            (pool_rows, config.vocab_size),
            dtype=torch.int32,
            device=config.device,
        )

        # bf16 is enough precision for typical client-supplied bias values
        # (OpenAI caps |logit_bias| at 100).
        self._logit_bias = torch.zeros(
            (pool_rows, config.vocab_size),
            dtype=torch.bfloat16,
            device=config.device,
        )

        # Per-request penalty scalars + min_p. rep_pen starts at 1.0
        # (multiplicative identity); others at 0.0 (additive identity).
        self._min_p_pool = torch.zeros(
            (pool_rows,), dtype=torch.float32, device=config.device
        )
        self._freq_pen_pool = torch.zeros(
            (pool_rows,), dtype=torch.bfloat16, device=config.device
        )
        self._pres_pen_pool = torch.zeros(
            (pool_rows,), dtype=torch.bfloat16, device=config.device
        )
        self._rep_pen_pool = torch.full(
            (pool_rows,), 1.0, dtype=torch.bfloat16, device=config.device
        )

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def _reset_slot(self, pool_idx: int, sp: SamplingParams) -> None:

        # Scatter scalars inherited from the parent backend (temperature, top_k,
        # top_p, seed).
        super()._reset_slot(pool_idx, sp)

        # Penalty + min_p scalars.
        self._min_p_pool[pool_idx].fill_(float(sp.min_p))
        self._freq_pen_pool[pool_idx].fill_(float(sp.frequency_penalty))
        self._pres_pen_pool[pool_idx].fill_(float(sp.presence_penalty))
        self._rep_pen_pool[pool_idx].fill_(float(sp.repetition_penalty))

        # Zero the slot's count row (history from the previous occupant is
        # no longer applicable).
        self._counts[pool_idx].fill_(0)

        # Zero + scatter logit_bias for the new rid. Zeroing the whole row
        # first rather than diffing because the previous occupant's bias
        # keys are unknown here.
        self._logit_bias[pool_idx].fill_(0.0)

        bias_map = getattr(sp, "logit_bias", None) if sp is not None else None
        if bias_map:
            vocab = self._logit_bias.shape[1]
            raw_ids = [int(tid) for tid in bias_map.keys()]
            assert all(0 <= tid < vocab for tid in raw_ids), (
                f"logit_bias contains out-of-vocab token id(s); "
                f"vocab_size={vocab}, offending={[t for t in raw_ids if not 0 <= t < vocab]}"
            )
            token_ids = torch.tensor(
                raw_ids,
                device=self._logit_bias.device,
                dtype=torch.long,
            )
            bias_values = torch.tensor(
                list(bias_map.values()),
                device=self._logit_bias.device,
                dtype=torch.bfloat16,
            )
            self._logit_bias[pool_idx, token_ids] = bias_values

    def reset_capture_state(self) -> None:

        # Warm-up iterations route all pool indices to row 0, which
        # accumulates sampled tokens into _counts[0]. Zero it so the graph
        # captures reads against a clean baseline. _logit_bias[0] is only
        # written in on_pool_assignment, so it stays zero across warm-up.
        self._counts[0].fill_(0)

    # ------------------------------------------------------------------
    # Penalty + bias application (shared by sample and verify)
    # ------------------------------------------------------------------

    @nvtx_range("sampling:penalties", color="yellow")
    @torch.compile(dynamic=True, backend=get_compiler_backend())
    def _apply_penalties_and_bias(
        self,
        logits: torch.Tensor,
        sampling_info: SamplingBatchInfo,
        num_tokens_per_req: int = 1,
    ) -> torch.Tensor:
        """logits is [bs * num_tokens_per_req, V]. Penalty scalars are gathered
        from the pool-indexed buffers. num_tokens_per_req > 1 is the spec-decode
        verify() path where per-request scalars are repeat_interleave'd to
        align with flat logits.
        """

        pool_idx = sampling_info.req_pool_indices.long()

        if num_tokens_per_req > 1:

            pool_idx = torch.repeat_interleave(pool_idx, num_tokens_per_req, dim=0)

        counts = self._counts.index_select(0, pool_idx)  # [bs*N, V]
        active = counts > 0
        counts_f = counts.to(logits.dtype)
        active_f = active.to(logits.dtype)

        # Gather per-request penalty scalars from the pool. [bs*N] → [bs*N, 1]
        # for broadcast against [bs*N, V] logits.
        rep = (
            self._rep_pen_pool.index_select(0, pool_idx).to(logits.dtype).unsqueeze(-1)
        )
        freq = (
            self._freq_pen_pool.index_select(0, pool_idx).to(logits.dtype).unsqueeze(-1)
        )
        presence = (
            self._pres_pen_pool.index_select(0, pool_idx).to(logits.dtype).unsqueeze(-1)
        )

        # 1. Repetition (multiplicative). scales is 1.0 where count==0, else
        # rep_pen. Apply as logits/scales where logits>0, logits*scales else.
        scales = torch.where(active, rep.expand_as(logits), torch.ones_like(logits))
        logits = torch.where(logits > 0, logits / scales, logits * scales)

        # 2. Frequency + presence (additive). Fused into a single subtract.
        logits = logits - freq * counts_f - presence * active_f

        # 3. Per-token logit_bias (additive). Rows without a logit_bias are
        # all-zero, so the add is a no-op for them.
        logits = logits + self._logit_bias.index_select(0, pool_idx)

        return logits

    @nvtx_range("sampling:accum_counts", color="yellow")
    @torch.compile(dynamic=True, backend=get_compiler_backend())
    def _accumulate_counts(
        self,
        pool_idx: torch.Tensor,
        tokens: torch.Tensor,
        weights: torch.Tensor,
    ) -> None:
        """Graph-safe in-place scatter: counts[pool_idx, tokens] += weights.
        weights is int32; 0 masks invalid rows, 1 accumulates."""

        self._counts.index_put_(
            (pool_idx.long(), tokens.long()),
            weights.to(torch.int32),
            accumulate=True,
        )

    # ------------------------------------------------------------------
    # Sample / verify
    # ------------------------------------------------------------------

    @nvtx_range("sampling:sample", color="yellow")
    def sample(
        self,
        logits_output: LogitsProcessorOutput,
        sampling_info: SamplingBatchInfo,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        logits = nan_guard_logits(
            logits_output.next_token_logits, self.config.enable_nan_detection
        )

        # Grammar bitmask apply — captured inside the CUDA graph. Buffer is
        # pre-bound by bind_grammar_mask_buf; non-grammar rows stay all-ones.
        # Applied before raw_logprobs capture so constrained logprobs reflect
        # the grammar-masked distribution.
        if sampling_info.vocab_mask is not None:
            sampling_info.apply_vocab_mask(
                logits=logits, vocab_mask=sampling_info.vocab_mask
            )

        # Raw-distribution logprobs (pre-penalty, pre-temperature) when the
        # server flag is on. Gather is done after we know the sampled id.
        raw_logprobs = (
            torch.log_softmax(logits, dim=-1)
            if self.config.enable_output_logprobs
            else None
        )

        logits = self._apply_penalties_and_bias(logits, sampling_info)

        temperatures, top_ks, top_ps, min_ps, seeds, offsets = (
            gather_and_expand_scalars(
                sampling_info.req_pool_indices,
                temperature=self._temperature_pool,
                top_k=self._top_k_pool,
                top_p=self._top_p_pool,
                min_p=self._min_p_pool,
                seed=self._seed_pool,
                offsets=sampling_info.valid_cache_lengths,
            )
        )

        probs = softmax(logits, temperature=temperatures.view(-1, 1))
        probs = top_k_renorm_prob(probs, top_ks)
        probs = top_p_renorm_prob(probs, top_ps, is_deterministic=True)

        batch_next_token_ids = min_p_sampling_from_probs(
            probs,
            min_ps,
            seed=seeds,
            offset=offsets,
        )

        sampled = batch_next_token_ids.to(torch.int32)

        # TP-rank sync BEFORE _accumulate_counts so per-rank counts stay aligned.
        self.maybe_broadcast(sampled)

        if raw_logprobs is not None:

            logits_output.next_token_logprobs = raw_logprobs.gather(
                -1, sampled.long().unsqueeze(-1)
            ).squeeze(-1)

        # Accumulate sampled tokens into counts (greedy path accumulates too
        # so mixed later batches see the correct history).
        self._accumulate_counts(
            sampling_info.req_pool_indices,
            sampled,
            torch.ones_like(sampled, dtype=torch.int32),
        )

        bs = logits.shape[0]

        return sampled, self._ones_buf[:bs]

    @nvtx_range("sampling:verify", color="yellow")
    def verify(
        self,
        logits_output: LogitsProcessorOutput,
        sampling_info: SamplingBatchInfo,
        candidates: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        bs = candidates.shape[0]
        num_tokens_per_req = candidates.shape[1]

        predict = self._predict_buf[: bs * num_tokens_per_req]
        accept_index = (
            self._accept_index_buf[: bs * num_tokens_per_req]
            .view(bs, num_tokens_per_req)
            .fill_(-1)
        )
        accept_length = self._accept_length_buf[:bs]

        # Per-draft-position grammar bitmask: buffer shape
        # [bs * num_tokens_per_req, V/32] matches the flat target logits.
        # Applied before raw_logprobs capture so constrained logprobs reflect
        # the grammar-masked distribution.
        if sampling_info.vocab_mask is not None:
            sampling_info.apply_vocab_mask(
                logits=logits_output.next_token_logits,
                vocab_mask=sampling_info.vocab_mask,
            )

        # Raw (pre-penalty) logprobs captured before penalty application to
        # match sample()'s semantics.
        raw_logprobs = (
            torch.log_softmax(logits_output.next_token_logits, dim=-1)
            if self.config.enable_output_logprobs
            else None
        )

        logits = self._apply_penalties_and_bias(
            logits_output.next_token_logits,
            sampling_info,
            num_tokens_per_req=num_tokens_per_req,
        )

        temperatures, top_ks, top_ps, min_ps, _, _ = gather_and_expand_scalars(
            sampling_info.req_pool_indices,
            temperature=self._temperature_pool,
            top_k=self._top_k_pool,
            top_p=self._top_p_pool,
            min_p=self._min_p_pool,
            n=num_tokens_per_req,
        )

        target_probs = softmax(logits, temperature=temperatures.view(-1, 1))
        target_probs = top_k_renorm_prob(target_probs, top_ks)
        target_probs = top_p_renorm_prob(target_probs, top_ps, is_deterministic=True)

        # min_p renorm open-coded: zero probs below `min_p * max_prob` per
        # row, then renormalize. The chain-speculative-sampling kernel has no
        # min_p knob, and flashinfer exposes no `min_p_renorm_prob`.
        max_probs = target_probs.max(dim=-1, keepdim=True).values

        target_probs = torch.where(
            target_probs >= min_ps.view(-1, 1) * max_probs,
            target_probs,
            torch.zeros_like(target_probs),
        )

        target_probs.div_(target_probs.sum(dim=-1, keepdim=True))

        target_probs = target_probs.reshape(bs, num_tokens_per_req, -1)

        draft_probs = torch.zeros(
            target_probs.shape, dtype=torch.float32, device=target_probs.device
        )

        coins = self._coins_buf[:bs, :num_tokens_per_req]
        coins_for_final_sampling = self._final_coins_buf[:bs]

        chain_speculative_sampling_target_only(
            predicts=predict,
            accept_index=accept_index,
            accept_token_num=accept_length,
            candidates=candidates.to(torch.int32),
            uniform_samples=coins,
            uniform_samples_for_final_sampling=coins_for_final_sampling,
            target_probs=target_probs,
            draft_probs=draft_probs,
            threshold_single=SPECULATIVE_ACCEPT_THRESHOLD_SINGLE,
            threshold_acc=SPECULATIVE_ACCEPT_THRESHOLD_ACC,
            deterministic=True,
        )

        accept_length += 1

        # TP-rank sync BEFORE _accumulate_counts so per-rank counts stay aligned.
        self.maybe_broadcast(predict, accept_index, accept_length)

        # Accumulate accepted tokens into counts. accept_index is [bs, N]
        # with -1 in unused slots; clamp to a safe index and mask with a
        # weight of 0 so invalid slots are no-ops.
        valid = accept_index >= 0  # [bs, N]
        safe_positions = accept_index.clamp(min=0).long()  # [bs, N]
        accepted_tokens = predict.long().gather(0, safe_positions.view(-1))

        pool_idx_expanded = (
            sampling_info.req_pool_indices.long()
            .unsqueeze(-1)
            .expand(-1, num_tokens_per_req)
            .reshape(-1)
        )

        self._accumulate_counts(
            pool_idx_expanded,
            accepted_tokens,
            valid.reshape(-1).to(torch.int32),
        )

        if raw_logprobs is not None:

            logits_output.next_token_logprobs = raw_logprobs.gather(
                -1, predict.long().unsqueeze(-1)
            ).squeeze(-1)

        return predict, accept_length


register_backend("flashinfer_full", FlashInferFullSamplingBackend)
