# SPDX-License-Identifier: MIT AND Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 LightSeek Foundation
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
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

# TokenSpeed keeps pool-owned counts and logit-bias state.

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tokenspeed_kernel.ops.sampling.triton import (
    accumulate_counts_inplace,
    apply_penalties_logit_bias_inplace,
    gumbel_sample_from_pools,
    gumbel_sample_from_pools_compact,
    gumbel_sample_from_pools_generic,
    gumbel_sample_min_p_from_pools,
    gumbel_sample_min_p_from_pools_parallel,
    gumbel_sample_top_k_top_p_from_pools,
    gumbel_sample_top_k_top_p_qrita_from_pools,
    gumbel_sample_top_p_parallel_from_pools,
    verify_chain_target_sampled,
)

from tokenspeed.runtime.sampling.backends.base import (
    CUDA_GRAPH_VARIANT_DEFAULT,
    SamplingBackend,
    SamplingBackendConfig,
)
from tokenspeed.runtime.sampling.backends.triton import (
    _COMPACT_GUMBEL_BLOCK_SIZE,
    _COMPACT_GUMBEL_VOCAB_MAX,
    _SAMPLE_ROUTE_GUMBEL_NO_FILTER,
    _SAMPLE_ROUTE_GUMBEL_TOP_K,
    _SAMPLE_ROUTE_GUMBEL_TOP_K_TOP_P,
    _SAMPLE_ROUTE_GUMBEL_TOP_P,
    _TOP_K_TOP_P_PAD,
    _TOP_K_TOP_P_SMALL_BLOCK_SIZE,
    _TOP_P_PARALLEL_SAMPLE_ATTEMPTS,
    _TOP_P_PARALLEL_SAMPLE_BLOCK_SIZE,
    _TOP_P_PARALLEL_VERIFY_ATTEMPTS,
    TritonSamplingBackend,
)
from tokenspeed.runtime.sampling.registry import register_backend
from tokenspeed.runtime.sampling.utils import nan_guard_logits
from tokenspeed.runtime.utils.nvtx import nvtx_range
from tokenspeed.runtime.utils.pdl import pdl_enabled

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.logits_processor import LogitsProcessorOutput
    from tokenspeed.runtime.sampling.sampling_batch_info import SamplingBatchInfo
    from tokenspeed.runtime.sampling.sampling_params import SamplingParams


CUDA_GRAPH_VARIANT_TRITON_FULL_MIN_P = "triton_full_min_p"
CUDA_GRAPH_VARIANT_TRITON_FULL_TOP_K_TOP_P_MIN_P = "triton_full_top_k_top_p_min_p"


class TritonFullSamplingBackend(TritonSamplingBackend):
    """Full sampling backend with TokenSpeed-owned state and Triton kernels."""

    def __init__(self, config: SamplingBackendConfig) -> None:
        super().__init__(config)
        if config.max_req_pool_size <= 0 or config.vocab_size <= 0:
            raise ValueError(
                "TritonFullSamplingBackend requires max_req_pool_size > 0 and "
                f"vocab_size > 0; got max_req_pool_size={config.max_req_pool_size}, "
                f"vocab_size={config.vocab_size}"
            )

        pool_rows = config.max_req_pool_size + 1
        self._counts = torch.zeros(
            (pool_rows, config.vocab_size),
            dtype=torch.int32,
            device=config.device,
        )
        self._logit_bias = torch.zeros(
            (pool_rows, config.vocab_size),
            dtype=torch.bfloat16,
            device=config.device,
        )
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
        self._min_p_row_max = torch.empty(
            (config.max_bs * config.max_draft_tokens_per_req,),
            dtype=torch.float32,
            device=config.device,
        )
        self._full_has_min_p = True

    def prepare_step(
        self,
        request_ids: list[str],
        request_pool_indices: list[int],
        sampling_params_list: list[SamplingParams],
        num_tokens_per_req: int = 1,
    ) -> None:
        super().prepare_step(
            request_ids=request_ids,
            request_pool_indices=request_pool_indices,
            sampling_params_list=sampling_params_list,
            num_tokens_per_req=num_tokens_per_req,
        )
        self._full_has_min_p = any(float(sp.min_p) > 0.0 for sp in sampling_params_list)

    def prepare_capture(self, bs: int, num_tokens_per_req: int = 1) -> None:
        self._full_has_min_p = True
        super().prepare_capture(bs=bs, num_tokens_per_req=num_tokens_per_req)

    def prepare_capture_variant(
        self,
        bs: int,
        num_tokens_per_req: int,
        variant: str,
    ) -> None:
        if variant == CUDA_GRAPH_VARIANT_TRITON_FULL_MIN_P:
            self._full_has_min_p = True
            self._sample_route = _SAMPLE_ROUTE_GUMBEL_NO_FILTER
            SamplingBackend.prepare_capture(
                self,
                bs=bs,
                num_tokens_per_req=num_tokens_per_req,
            )
            return
        if variant == CUDA_GRAPH_VARIANT_TRITON_FULL_TOP_K_TOP_P_MIN_P:
            self._full_has_min_p = True
            self._sample_route = _SAMPLE_ROUTE_GUMBEL_TOP_K_TOP_P
            self._top_k_top_p_pad = _TOP_K_TOP_P_PAD
            SamplingBackend.prepare_capture(
                self,
                bs=bs,
                num_tokens_per_req=num_tokens_per_req,
            )
            return
        self._full_has_min_p = variant == CUDA_GRAPH_VARIANT_DEFAULT
        super().prepare_capture_variant(
            bs=bs,
            num_tokens_per_req=num_tokens_per_req,
            variant=variant,
        )

    def cuda_graph_capture_variants(self, num_tokens_per_req: int) -> tuple[str, ...]:
        return (
            *super().cuda_graph_capture_variants(num_tokens_per_req),
            CUDA_GRAPH_VARIANT_TRITON_FULL_MIN_P,
            CUDA_GRAPH_VARIANT_TRITON_FULL_TOP_K_TOP_P_MIN_P,
        )

    def cuda_graph_replay_variant(self, num_tokens_per_req: int) -> str:
        if self._full_has_min_p:
            if self._sample_route == _SAMPLE_ROUTE_GUMBEL_NO_FILTER:
                return CUDA_GRAPH_VARIANT_TRITON_FULL_MIN_P
            if self._sample_route in (
                _SAMPLE_ROUTE_GUMBEL_TOP_K,
                _SAMPLE_ROUTE_GUMBEL_TOP_K_TOP_P,
            ):
                return CUDA_GRAPH_VARIANT_TRITON_FULL_TOP_K_TOP_P_MIN_P
            return CUDA_GRAPH_VARIANT_DEFAULT
        return super().cuda_graph_replay_variant(num_tokens_per_req)

    def _reset_slot(self, pool_idx: int, sp: SamplingParams) -> None:
        super()._reset_slot(pool_idx, sp)

        self._min_p_pool[pool_idx].fill_(float(sp.min_p))
        self._freq_pen_pool[pool_idx].fill_(float(sp.frequency_penalty))
        self._pres_pen_pool[pool_idx].fill_(float(sp.presence_penalty))
        self._rep_pen_pool[pool_idx].fill_(float(sp.repetition_penalty))

        self._counts[pool_idx].fill_(0)
        self._logit_bias[pool_idx].fill_(0.0)

        bias_map = getattr(sp, "logit_bias", None) if sp is not None else None
        if bias_map:
            vocab = self._logit_bias.shape[1]
            raw_ids = [int(tid) for tid in bias_map.keys()]
            assert all(0 <= tid < vocab for tid in raw_ids), (
                f"logit_bias contains out-of-vocab token id(s); "
                f"vocab_size={vocab}, offending="
                f"{[t for t in raw_ids if not 0 <= t < vocab]}"
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
        self._counts[0].fill_(0)

    @nvtx_range("sampling:penalties", color="yellow")
    def _apply_penalties_and_bias(
        self,
        logits: torch.Tensor,
        req_pool_indices: torch.Tensor,
        num_tokens_per_req: int = 1,
    ) -> torch.Tensor:
        return apply_penalties_logit_bias_inplace(
            logits,
            req_pool_indices,
            self._counts,
            self._logit_bias,
            self._freq_pen_pool,
            self._pres_pen_pool,
            self._rep_pen_pool,
            num_tokens_per_req=num_tokens_per_req,
        )

    @nvtx_range("sampling:accum_counts", color="yellow")
    def _accumulate_counts(
        self,
        pool_idx: torch.Tensor,
        tokens: torch.Tensor,
        weights: torch.Tensor,
    ) -> None:
        accumulate_counts_inplace(self._counts, pool_idx, tokens, weights)

    def _gumbel_sample_full_logits(
        self,
        logits: torch.Tensor,
        req_pool_indices: torch.Tensor,
        offsets_pool: torch.Tensor,
        out: torch.Tensor,
        *,
        num_tokens_per_req: int = 1,
    ) -> torch.Tensor:
        rows = logits.shape[0]
        if (
            self._full_has_min_p
            and self._sample_route == _SAMPLE_ROUTE_GUMBEL_NO_FILTER
        ):
            if logits.shape[1] > _COMPACT_GUMBEL_VOCAB_MAX:
                local_ids = (
                    self._gumbel_local_ids
                    if num_tokens_per_req == 1
                    else self._gumbel_verify_local_ids
                )
                local_scores = (
                    self._gumbel_local_scores
                    if num_tokens_per_req == 1
                    else self._gumbel_verify_local_scores
                )
                return gumbel_sample_min_p_from_pools_parallel(
                    logits,
                    req_pool_indices,
                    self._temperature_pool,
                    self._min_p_pool,
                    self._seed_pool,
                    offsets_pool,
                    local_ids[:rows],
                    local_scores[:rows],
                    self._min_p_row_max[:rows],
                    out[:rows],
                    num_tokens_per_req=num_tokens_per_req,
                )
            return gumbel_sample_min_p_from_pools(
                logits,
                req_pool_indices,
                self._temperature_pool,
                self._min_p_pool,
                self._seed_pool,
                offsets_pool,
                out[:rows],
                num_tokens_per_req=num_tokens_per_req,
            )

        if self._sample_route in (
            _SAMPLE_ROUTE_GUMBEL_TOP_K,
            _SAMPLE_ROUTE_GUMBEL_TOP_K_TOP_P,
        ):
            if (
                not self._full_has_min_p
                and num_tokens_per_req > 1
                and self._use_qrita_verify_top_k_route(rows, logits.shape[1])
            ):
                return gumbel_sample_top_k_top_p_qrita_from_pools(
                    logits,
                    req_pool_indices,
                    self._temperature_pool,
                    self._top_k_pool,
                    self._top_p_pool,
                    self._seed_pool,
                    offsets_pool,
                    self._qrita_verify_buffer,
                    self._qrita_percentile_to_std_table,
                    out[:rows],
                    num_tokens_per_req=num_tokens_per_req,
                    num_programs=min(self._qrita_verify_num_programs, rows),
                )
            candidate_ids = (
                self._topk_candidate_ids
                if num_tokens_per_req == 1
                else self._topk_verify_candidate_ids
            )
            candidate_logits = (
                self._topk_candidate_logits
                if num_tokens_per_req == 1
                else self._topk_verify_candidate_logits
            )
            return gumbel_sample_top_k_top_p_from_pools(
                logits,
                req_pool_indices,
                self._temperature_pool,
                self._top_k_pool,
                self._top_p_pool,
                self._seed_pool,
                offsets_pool,
                candidate_ids[:rows],
                candidate_logits[:rows],
                out[:rows],
                min_p_pool=self._min_p_pool if self._full_has_min_p else None,
                block_size=_TOP_K_TOP_P_SMALL_BLOCK_SIZE,
                top_k_pad=self._top_k_top_p_pad,
                num_tokens_per_req=num_tokens_per_req,
            )

        if not self._full_has_min_p:
            if self._sample_route == _SAMPLE_ROUTE_GUMBEL_NO_FILTER:
                if logits.shape[1] <= _COMPACT_GUMBEL_VOCAB_MAX:
                    return gumbel_sample_from_pools_compact(
                        logits,
                        req_pool_indices,
                        self._temperature_pool,
                        self._seed_pool,
                        offsets_pool,
                        out[:rows],
                        block_size=_COMPACT_GUMBEL_BLOCK_SIZE,
                        num_tokens_per_req=num_tokens_per_req,
                    )
                local_ids = (
                    self._gumbel_local_ids
                    if num_tokens_per_req == 1
                    else self._gumbel_verify_local_ids
                )
                local_scores = (
                    self._gumbel_local_scores
                    if num_tokens_per_req == 1
                    else self._gumbel_verify_local_scores
                )
                return gumbel_sample_from_pools(
                    logits,
                    req_pool_indices,
                    self._temperature_pool,
                    self._seed_pool,
                    offsets_pool,
                    local_ids[:rows],
                    local_scores[:rows],
                    out[:rows],
                    num_tokens_per_req=num_tokens_per_req,
                )
            if self._sample_route == _SAMPLE_ROUTE_GUMBEL_TOP_P:
                return gumbel_sample_top_p_parallel_from_pools(
                    logits,
                    req_pool_indices,
                    self._temperature_pool,
                    self._top_p_pool,
                    self._seed_pool,
                    offsets_pool,
                    self._top_p_local_max[:rows],
                    self._top_p_local_sum[:rows],
                    self._top_p_local_argmax[:rows],
                    self._top_p_local_scores[:rows],
                    self._top_p_local_logits[:rows],
                    self._top_p_local_ids[:rows],
                    self._top_p_row_max[:rows],
                    self._top_p_row_total[:rows],
                    self._top_p_row_argmax[:rows],
                    self._top_p_row_candidate_logits[:rows],
                    self._top_p_row_candidate_ids[:rows],
                    self._top_p_accepted[:rows],
                    out[:rows],
                    block_size=_TOP_P_PARALLEL_SAMPLE_BLOCK_SIZE,
                    num_attempts=(
                        _TOP_P_PARALLEL_SAMPLE_ATTEMPTS
                        if num_tokens_per_req == 1
                        else _TOP_P_PARALLEL_VERIFY_ATTEMPTS
                    ),
                    num_tokens_per_req=num_tokens_per_req,
                )

        return gumbel_sample_from_pools_generic(
            logits,
            req_pool_indices,
            self._temperature_pool,
            self._top_k_pool,
            self._top_p_pool,
            self._seed_pool,
            offsets_pool,
            out[:rows],
            min_p_pool=self._min_p_pool,
            num_tokens_per_req=num_tokens_per_req,
        )

    @nvtx_range("sampling:sample", color="yellow")
    def sample(
        self,
        logits_output: LogitsProcessorOutput,
        sampling_info: SamplingBatchInfo,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = nan_guard_logits(
            logits_output.next_token_logits, self.config.enable_nan_detection
        ).float()

        if sampling_info.vocab_mask is not None:
            sampling_info.apply_vocab_mask(
                logits=logits, vocab_mask=sampling_info.vocab_mask
            )

        logits_for_logprobs = (
            logits.clone() if self.config.enable_output_logprobs else None
        )

        req_pool_indices = self._req_pool_indices_for_kernels(
            sampling_info.req_pool_indices, logits.shape[0]
        )
        logits = self._apply_penalties_and_bias(logits, req_pool_indices)
        offsets_pool = (
            sampling_info.valid_cache_lengths
            if sampling_info.valid_cache_lengths is not None
            else self._zero_offsets_pool
        )
        sampled = self._gumbel_sample_full_logits(
            logits,
            req_pool_indices,
            offsets_pool,
            self._gumbel_out[: logits.shape[0]],
        ).to(torch.int32)

        self.maybe_broadcast(sampled)

        if logits_for_logprobs is not None:
            self._write_logprob_outputs(
                logits_output,
                logits_for_logprobs,
                sampled,
            )

        self._accumulate_counts(
            req_pool_indices,
            sampled,
            torch.ones_like(sampled, dtype=torch.int32),
        )

        return sampled, self._ones_buf[: logits.shape[0]]

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

        logits = nan_guard_logits(
            logits_output.next_token_logits, self.config.enable_nan_detection
        ).float()

        if sampling_info.vocab_mask is not None:
            sampling_info.apply_vocab_mask(
                logits=logits,
                vocab_mask=sampling_info.vocab_mask,
            )

        logits_for_logprobs = (
            logits.clone() if self.config.enable_output_logprobs else None
        )

        req_pool_indices = self._req_pool_indices_for_kernels(
            sampling_info.req_pool_indices, bs
        )
        logits = self._apply_penalties_and_bias(
            logits,
            req_pool_indices,
            num_tokens_per_req=num_tokens_per_req,
        )

        offsets_pool = (
            sampling_info.valid_cache_lengths
            if sampling_info.valid_cache_lengths is not None
            else self._zero_offsets_pool
        )
        target_sampled = self._gumbel_sample_full_logits(
            logits,
            req_pool_indices,
            offsets_pool,
            self._gumbel_verify_out[: bs * num_tokens_per_req],
            num_tokens_per_req=num_tokens_per_req,
        )
        verify_chain_target_sampled(
            predicts=predict,
            accept_index=accept_index,
            accept_token_num=accept_length,
            candidates=candidates.to(torch.int32),
            target_sampled=target_sampled,
            enable_pdl=pdl_enabled(),
        )

        accept_length += 1

        self.maybe_broadcast(predict, accept_index, accept_length)

        valid = accept_index >= 0
        safe_positions = accept_index.clamp(min=0).long()
        accepted_tokens = predict.long().gather(0, safe_positions.view(-1))

        pool_idx_expanded = (
            req_pool_indices.unsqueeze(-1).expand(-1, num_tokens_per_req).reshape(-1)
        )

        self._accumulate_counts(
            pool_idx_expanded,
            accepted_tokens,
            valid.reshape(-1).to(torch.int32),
        )

        if logits_for_logprobs is not None:
            self._write_logprob_outputs(
                logits_output,
                logits_for_logprobs,
                predict,
            )

        return predict, accept_length


register_backend("triton_full", TritonFullSamplingBackend)
