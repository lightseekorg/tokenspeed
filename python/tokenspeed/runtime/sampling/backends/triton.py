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

# TokenSpeed keeps its request-pool state and backend boundary.

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tokenspeed_kernel.ops.sampling.cute_dsl import argmax as cute_argmax
from tokenspeed_kernel.ops.sampling.triton import (
    _QRITA_PERCENTILE_TO_STD_TABLE,
    gumbel_sample_from_pools,
    gumbel_sample_from_pools_compact,
    gumbel_sample_from_pools_generic,
    gumbel_sample_top_k_top_p_from_pools,
    gumbel_sample_top_k_top_p_qrita_from_pools,
    gumbel_sample_top_p_parallel_from_pools,
    selected_token_logprobs,
    verify_chain_target_sampled,
)

from tokenspeed.runtime.sampling.backends.base import (
    CUDA_GRAPH_VARIANT_DEFAULT,
    CUDA_GRAPH_VARIANT_GREEDY,
    SamplingBackend,
    SamplingBackendConfig,
)
from tokenspeed.runtime.sampling.registry import register_backend
from tokenspeed.runtime.sampling.sampling_params import _SAMPLING_EPS, _TOP_K_DISABLED
from tokenspeed.runtime.sampling.utils import nan_guard_logits
from tokenspeed.runtime.utils.nvtx import nvtx_range
from tokenspeed.runtime.utils.pdl import pdl_enabled

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.logits_processor import LogitsProcessorOutput
    from tokenspeed.runtime.sampling.sampling_batch_info import SamplingBatchInfo
    from tokenspeed.runtime.sampling.sampling_params import SamplingParams


_GUMBEL_BLOCK_SIZE = 1024
_COMPACT_GUMBEL_BLOCK_SIZE = 4096
_COMPACT_GUMBEL_VOCAB_MAX = 32768
_TOP_K_TOP_P_SMALL_BLOCK_SIZE = 1024
_TOP_K_TOP_P_TUNED_VOCAB_MAX = 32768
_TOP_K_TOP_P_PAD = 128
_TOP_P_PARALLEL_SAMPLE_BLOCK_SIZE = 1024
_TOP_P_PARALLEL_SAMPLE_ATTEMPTS = 3
_TOP_P_PARALLEL_VERIFY_ATTEMPTS = 4
_TOP_P_PARALLEL_MAX_ATTEMPTS = max(
    _TOP_P_PARALLEL_SAMPLE_ATTEMPTS, _TOP_P_PARALLEL_VERIFY_ATTEMPTS
)
_QRITA_VERIFY_MIN_ROWS = 128

_SAMPLE_ROUTE_GUMBEL_GENERIC = 0
_SAMPLE_ROUTE_GUMBEL_NO_FILTER = 1
_SAMPLE_ROUTE_GUMBEL_TOP_K = 2
_SAMPLE_ROUTE_GUMBEL_TOP_K_TOP_P = 3
_SAMPLE_ROUTE_GUMBEL_TOP_P = 4
CUDA_GRAPH_VARIANT_TRITON_NO_FILTER = "triton_no_filter"
CUDA_GRAPH_VARIANT_TRITON_TOP_K = "triton_top_k"
CUDA_GRAPH_VARIANT_TRITON_TOP_K_TOP_P = "triton_top_k_top_p"
CUDA_GRAPH_VARIANT_TRITON_TOP_P = "triton_top_p"
CUDA_GRAPH_VARIANT_TRITON_VERIFY_NO_FILTER = "triton_verify_no_filter"

_CUDA_GRAPH_VARIANT_SAMPLE_ROUTES = {
    CUDA_GRAPH_VARIANT_TRITON_NO_FILTER: _SAMPLE_ROUTE_GUMBEL_NO_FILTER,
    CUDA_GRAPH_VARIANT_TRITON_TOP_K: _SAMPLE_ROUTE_GUMBEL_TOP_K,
    CUDA_GRAPH_VARIANT_TRITON_TOP_K_TOP_P: _SAMPLE_ROUTE_GUMBEL_TOP_K_TOP_P,
    CUDA_GRAPH_VARIANT_TRITON_TOP_P: _SAMPLE_ROUTE_GUMBEL_TOP_P,
    CUDA_GRAPH_VARIANT_TRITON_VERIFY_NO_FILTER: _SAMPLE_ROUTE_GUMBEL_NO_FILTER,
}


class TritonSamplingBackend(SamplingBackend):
    """TokenSpeed pool-state backend using Triton Gumbel-Max kernels."""

    _HAS_POOL_STATE = True
    _SUPPORTS_GREEDY_GRAPH_VARIANT = True

    def __init__(self, config: SamplingBackendConfig) -> None:
        super().__init__(config)
        self._init_triton_pool_state(config)
        self._init_triton_buffers(config)
        self._sample_route = _SAMPLE_ROUTE_GUMBEL_GENERIC
        self._top_k_top_p_pad = _TOP_K_TOP_P_PAD

    def _init_triton_pool_state(self, config: SamplingBackendConfig) -> None:
        pool_rows = config.max_req_pool_size + 1
        self._temperature_pool = torch.ones(
            (pool_rows,), dtype=torch.float32, device=config.device
        )
        self._top_k_pool = torch.ones(
            (pool_rows,), dtype=torch.int32, device=config.device
        )
        self._top_p_pool = torch.ones(
            (pool_rows,), dtype=torch.float32, device=config.device
        )
        self._seed_pool = torch.zeros(
            (pool_rows,), dtype=torch.int64, device=config.device
        )

        self._ones_buf = torch.ones(
            (config.max_bs,), dtype=torch.int32, device=config.device
        )
        self._predict_buf = torch.zeros(
            (config.max_bs * config.max_draft_tokens_per_req,),
            dtype=torch.int32,
            device=config.device,
        )
        # Flat layout so [:bs * n].view(bs, n) is contiguous for any bs/n
        # (required by maybe_broadcast / NCCL).
        self._accept_index_buf = torch.zeros(
            (config.max_bs * config.max_draft_tokens_per_req,),
            dtype=torch.int32,
            device=config.device,
        )
        self._accept_length_buf = torch.zeros(
            (config.max_bs,), dtype=torch.int32, device=config.device
        )

    def _init_triton_buffers(self, config: SamplingBackendConfig) -> None:
        pool_rows = config.max_req_pool_size + 1
        self._zero_offsets_pool = torch.zeros(
            (pool_rows,), dtype=torch.int64, device=config.device
        )

        vocab_size = max(int(config.vocab_size), 1)
        gumbel_blocks = (vocab_size + _GUMBEL_BLOCK_SIZE - 1) // _GUMBEL_BLOCK_SIZE
        self._gumbel_local_ids = torch.empty(
            (config.max_bs, gumbel_blocks),
            dtype=torch.int32,
            device=config.device,
        )
        self._gumbel_local_scores = torch.empty(
            (config.max_bs, gumbel_blocks),
            dtype=torch.float32,
            device=config.device,
        )
        self._gumbel_out = torch.empty(
            (config.max_bs,), dtype=torch.int32, device=config.device
        )
        self._req_pool_indices_i32 = torch.empty(
            (config.max_bs,), dtype=torch.int32, device=config.device
        )
        self._gumbel_verify_out = torch.empty(
            (config.max_bs * config.max_draft_tokens_per_req,),
            dtype=torch.int32,
            device=config.device,
        )
        self._gumbel_verify_local_ids = torch.empty(
            (config.max_bs * config.max_draft_tokens_per_req, gumbel_blocks),
            dtype=torch.int32,
            device=config.device,
        )
        self._gumbel_verify_local_scores = torch.empty(
            (config.max_bs * config.max_draft_tokens_per_req, gumbel_blocks),
            dtype=torch.float32,
            device=config.device,
        )

        topk_blocks = (vocab_size + _TOP_K_TOP_P_SMALL_BLOCK_SIZE - 1) // (
            _TOP_K_TOP_P_SMALL_BLOCK_SIZE
        )
        topk_candidates = topk_blocks * _TOP_K_TOP_P_PAD
        self._topk_candidate_ids = torch.empty(
            (config.max_bs, topk_candidates),
            dtype=torch.int32,
            device=config.device,
        )
        self._topk_candidate_logits = torch.empty(
            (config.max_bs, topk_candidates),
            dtype=torch.float32,
            device=config.device,
        )
        self._topk_verify_candidate_ids = torch.empty(
            (config.max_bs * config.max_draft_tokens_per_req, topk_candidates),
            dtype=torch.int32,
            device=config.device,
        )
        self._topk_verify_candidate_logits = torch.empty(
            (config.max_bs * config.max_draft_tokens_per_req, topk_candidates),
            dtype=torch.float32,
            device=config.device,
        )
        max_verify_rows = max(config.max_bs * config.max_draft_tokens_per_req, 1)
        num_sms = torch.cuda.get_device_properties(config.device).multi_processor_count
        self._qrita_verify_num_programs = min(num_sms, max_verify_rows)
        self._qrita_verify_buffer = torch.empty(
            (self._qrita_verify_num_programs, vocab_size),
            dtype=torch.float32,
            device=config.device,
        )
        self._qrita_percentile_to_std_table = torch.tensor(
            _QRITA_PERCENTILE_TO_STD_TABLE,
            dtype=torch.float32,
            device=config.device,
        )

        top_p_blocks = (vocab_size + _TOP_P_PARALLEL_SAMPLE_BLOCK_SIZE - 1) // (
            _TOP_P_PARALLEL_SAMPLE_BLOCK_SIZE
        )
        top_p_rows = max(config.max_bs * config.max_draft_tokens_per_req, 1)
        self._top_p_local_max = torch.empty(
            (top_p_rows, top_p_blocks), dtype=torch.float32, device=config.device
        )
        self._top_p_local_sum = torch.empty(
            (top_p_rows, top_p_blocks), dtype=torch.float32, device=config.device
        )
        self._top_p_local_argmax = torch.empty(
            (top_p_rows, top_p_blocks), dtype=torch.int32, device=config.device
        )
        self._top_p_local_scores = torch.empty(
            (top_p_rows, top_p_blocks, _TOP_P_PARALLEL_MAX_ATTEMPTS),
            dtype=torch.float32,
            device=config.device,
        )
        self._top_p_local_logits = torch.empty(
            (top_p_rows, top_p_blocks, _TOP_P_PARALLEL_MAX_ATTEMPTS),
            dtype=torch.float32,
            device=config.device,
        )
        self._top_p_local_ids = torch.empty(
            (top_p_rows, top_p_blocks, _TOP_P_PARALLEL_MAX_ATTEMPTS),
            dtype=torch.int32,
            device=config.device,
        )
        self._top_p_row_max = torch.empty(
            (top_p_rows,), dtype=torch.float32, device=config.device
        )
        self._top_p_row_total = torch.empty(
            (top_p_rows,), dtype=torch.float32, device=config.device
        )
        self._top_p_row_argmax = torch.empty(
            (top_p_rows,), dtype=torch.int32, device=config.device
        )
        self._top_p_row_candidate_logits = torch.empty(
            (top_p_rows, _TOP_P_PARALLEL_MAX_ATTEMPTS),
            dtype=torch.float32,
            device=config.device,
        )
        self._top_p_row_candidate_ids = torch.empty(
            (top_p_rows, _TOP_P_PARALLEL_MAX_ATTEMPTS),
            dtype=torch.int32,
            device=config.device,
        )
        self._top_p_accepted = torch.empty(
            (top_p_rows,), dtype=torch.int32, device=config.device
        )
        self._selected_logprob_out = torch.empty(
            (top_p_rows,), dtype=torch.float32, device=config.device
        )

    def _req_pool_indices_for_kernels(
        self, req_pool_indices: torch.Tensor, rows: int
    ) -> torch.Tensor:
        req_pool_indices = req_pool_indices[:rows]
        if req_pool_indices.dtype == torch.int32:
            return req_pool_indices
        if req_pool_indices.dtype != torch.int64:
            raise ValueError(
                "Triton sampling requires int32/int64 req_pool_indices, "
                f"got {req_pool_indices.dtype}"
            )
        out = self._req_pool_indices_i32[:rows]
        out.copy_(req_pool_indices, non_blocking=True)
        return out

    def _write_logprob_outputs(
        self,
        logits_output: LogitsProcessorOutput,
        logits: torch.Tensor,
        sampled: torch.Tensor,
    ) -> None:
        if not self.config.enable_output_logprobs:
            return

        rows = logits.shape[0]
        selected_out = self._selected_logprob_out[:rows]
        logits_output.next_token_logprobs = selected_token_logprobs(
            logits, sampled, selected_out
        )

    @staticmethod
    def _select_sample_route(
        sampling_params_list: list[SamplingParams],
    ) -> int:
        if len(sampling_params_list) == 0:
            return _SAMPLE_ROUTE_GUMBEL_GENERIC

        top_ks = [int(sp.top_k) for sp in sampling_params_list]
        top_ps = [float(sp.top_p) for sp in sampling_params_list]
        all_top_p_one = all(abs(p - 1.0) <= _SAMPLING_EPS for p in top_ps)
        all_top_k_disabled = all(k == _TOP_K_DISABLED for k in top_ks)
        all_top_k_finite = all(k != _TOP_K_DISABLED for k in top_ks)

        if all_top_k_disabled and all_top_p_one:
            return _SAMPLE_ROUTE_GUMBEL_NO_FILTER
        if all_top_k_disabled:
            return _SAMPLE_ROUTE_GUMBEL_TOP_P
        if all_top_k_finite:
            if all_top_p_one:
                return _SAMPLE_ROUTE_GUMBEL_TOP_K
            return _SAMPLE_ROUTE_GUMBEL_TOP_K_TOP_P
        return _SAMPLE_ROUTE_GUMBEL_GENERIC

    def _reset_slot(self, pool_idx: int, sp: SamplingParams) -> None:
        self._temperature_pool[pool_idx].fill_(float(sp.temperature))
        self._top_k_pool[pool_idx].fill_(int(sp.top_k))
        self._top_p_pool[pool_idx].fill_(float(sp.top_p))
        self._seed_pool[pool_idx].fill_(int(sp.seed))

    def prepare_step(
        self,
        request_ids: list[str],
        request_pool_indices: list[int],
        sampling_params_list: list[SamplingParams],
        num_tokens_per_req: int = 1,
    ) -> None:
        SamplingBackend.prepare_step(
            self,
            request_ids=request_ids,
            request_pool_indices=request_pool_indices,
            sampling_params_list=sampling_params_list,
            num_tokens_per_req=num_tokens_per_req,
        )
        self._sample_route = self._select_sample_route(sampling_params_list)
        self._top_k_top_p_pad = self._select_top_k_top_p_pad(sampling_params_list)

    @staticmethod
    def _select_top_k_top_p_pad(sampling_params_list: list[SamplingParams]) -> int:
        finite_top_ks = [
            int(sp.top_k)
            for sp in sampling_params_list
            if int(sp.top_k) != _TOP_K_DISABLED
        ]
        if finite_top_ks and max(finite_top_ks) <= 64:
            return 64
        return _TOP_K_TOP_P_PAD

    def _use_qrita_verify_top_k_route(self, rows: int, vocab_size: int) -> bool:
        return (
            self._sample_route
            in (_SAMPLE_ROUTE_GUMBEL_TOP_K, _SAMPLE_ROUTE_GUMBEL_TOP_K_TOP_P)
            and rows >= _QRITA_VERIFY_MIN_ROWS
            and vocab_size >= _TOP_K_TOP_P_TUNED_VOCAB_MAX
            and (
                vocab_size > _TOP_K_TOP_P_TUNED_VOCAB_MAX or self._top_k_top_p_pad > 64
            )
        )

    def prepare_capture(self, bs: int, num_tokens_per_req: int = 1) -> None:
        self._sample_route = _SAMPLE_ROUTE_GUMBEL_GENERIC
        self._top_k_top_p_pad = _TOP_K_TOP_P_PAD
        SamplingBackend.prepare_capture(
            self, bs=bs, num_tokens_per_req=num_tokens_per_req
        )

    def cuda_graph_capture_variants(self, num_tokens_per_req: int) -> tuple[str, ...]:
        variants = (
            *super().cuda_graph_capture_variants(num_tokens_per_req),
            CUDA_GRAPH_VARIANT_TRITON_NO_FILTER,
            CUDA_GRAPH_VARIANT_TRITON_TOP_P,
            CUDA_GRAPH_VARIANT_TRITON_TOP_K,
            CUDA_GRAPH_VARIANT_TRITON_TOP_K_TOP_P,
        )
        if num_tokens_per_req <= 1:
            return variants
        return (*variants, CUDA_GRAPH_VARIANT_TRITON_VERIFY_NO_FILTER)

    def prepare_capture_variant(
        self,
        bs: int,
        num_tokens_per_req: int,
        variant: str,
    ) -> None:
        sample_route = _CUDA_GRAPH_VARIANT_SAMPLE_ROUTES.get(variant)
        if sample_route is not None:
            self._sample_route = sample_route
            if sample_route in (
                _SAMPLE_ROUTE_GUMBEL_TOP_K,
                _SAMPLE_ROUTE_GUMBEL_TOP_K_TOP_P,
            ):
                self._top_k_top_p_pad = _TOP_K_TOP_P_PAD
            SamplingBackend.prepare_capture(
                self,
                bs=bs,
                num_tokens_per_req=num_tokens_per_req,
            )
            return
        if variant == CUDA_GRAPH_VARIANT_GREEDY:
            SamplingBackend.prepare_capture_variant(
                self,
                bs=bs,
                num_tokens_per_req=num_tokens_per_req,
                variant=variant,
            )
            return
        if variant == CUDA_GRAPH_VARIANT_DEFAULT:
            self.prepare_capture(bs=bs, num_tokens_per_req=num_tokens_per_req)
            return
        raise ValueError(f"Unsupported CUDA graph variant: {variant}")

    def cuda_graph_replay_variant(self, num_tokens_per_req: int) -> str:
        base_variant = super().cuda_graph_replay_variant(num_tokens_per_req)
        if base_variant == CUDA_GRAPH_VARIANT_GREEDY:
            return base_variant
        if self._sample_route == _SAMPLE_ROUTE_GUMBEL_NO_FILTER:
            if num_tokens_per_req > 1:
                return CUDA_GRAPH_VARIANT_TRITON_VERIFY_NO_FILTER
            return CUDA_GRAPH_VARIANT_TRITON_NO_FILTER
        if self._sample_route == _SAMPLE_ROUTE_GUMBEL_TOP_K:
            return CUDA_GRAPH_VARIANT_TRITON_TOP_K
        if self._sample_route == _SAMPLE_ROUTE_GUMBEL_TOP_K_TOP_P:
            return CUDA_GRAPH_VARIANT_TRITON_TOP_K_TOP_P
        if self._sample_route == _SAMPLE_ROUTE_GUMBEL_TOP_P:
            return CUDA_GRAPH_VARIANT_TRITON_TOP_P
        return CUDA_GRAPH_VARIANT_DEFAULT

    def sample(
        self,
        logits_output: LogitsProcessorOutput,
        sampling_info: SamplingBatchInfo,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = nan_guard_logits(
            logits_output.next_token_logits, self.config.enable_nan_detection
        )

        if sampling_info.vocab_mask is not None:
            sampling_info.apply_vocab_mask(
                logits=logits, vocab_mask=sampling_info.vocab_mask
            )

        if sampling_info.is_all_greedy:
            batch_next_token_ids = cute_argmax(logits)
        else:
            offsets_pool = (
                sampling_info.valid_cache_lengths
                if sampling_info.valid_cache_lengths is not None
                else self._zero_offsets_pool
            )
            bs = logits.shape[0]
            req_pool_indices = self._req_pool_indices_for_kernels(
                sampling_info.req_pool_indices, bs
            )
            if self._sample_route == _SAMPLE_ROUTE_GUMBEL_NO_FILTER:
                if logits.shape[1] <= _COMPACT_GUMBEL_VOCAB_MAX:
                    batch_next_token_ids = gumbel_sample_from_pools_compact(
                        logits,
                        req_pool_indices,
                        self._temperature_pool,
                        self._seed_pool,
                        offsets_pool,
                        self._gumbel_out[:bs],
                        block_size=_COMPACT_GUMBEL_BLOCK_SIZE,
                    )
                else:
                    batch_next_token_ids = gumbel_sample_from_pools(
                        logits,
                        req_pool_indices,
                        self._temperature_pool,
                        self._seed_pool,
                        offsets_pool,
                        self._gumbel_local_ids[:bs],
                        self._gumbel_local_scores[:bs],
                        self._gumbel_out[:bs],
                    )
            elif self._sample_route in (
                _SAMPLE_ROUTE_GUMBEL_TOP_K,
                _SAMPLE_ROUTE_GUMBEL_TOP_K_TOP_P,
            ):
                batch_next_token_ids = gumbel_sample_top_k_top_p_from_pools(
                    logits,
                    req_pool_indices,
                    self._temperature_pool,
                    self._top_k_pool,
                    self._top_p_pool,
                    self._seed_pool,
                    offsets_pool,
                    self._topk_candidate_ids[:bs],
                    self._topk_candidate_logits[:bs],
                    self._gumbel_out[:bs],
                    block_size=_TOP_K_TOP_P_SMALL_BLOCK_SIZE,
                    top_k_pad=self._top_k_top_p_pad,
                )
            elif self._sample_route == _SAMPLE_ROUTE_GUMBEL_TOP_P:
                batch_next_token_ids = gumbel_sample_top_p_parallel_from_pools(
                    logits,
                    req_pool_indices,
                    self._temperature_pool,
                    self._top_p_pool,
                    self._seed_pool,
                    offsets_pool,
                    self._top_p_local_max[:bs],
                    self._top_p_local_sum[:bs],
                    self._top_p_local_argmax[:bs],
                    self._top_p_local_scores[:bs],
                    self._top_p_local_logits[:bs],
                    self._top_p_local_ids[:bs],
                    self._top_p_row_max[:bs],
                    self._top_p_row_total[:bs],
                    self._top_p_row_argmax[:bs],
                    self._top_p_row_candidate_logits[:bs],
                    self._top_p_row_candidate_ids[:bs],
                    self._top_p_accepted[:bs],
                    self._gumbel_out[:bs],
                    block_size=_TOP_P_PARALLEL_SAMPLE_BLOCK_SIZE,
                    num_attempts=_TOP_P_PARALLEL_SAMPLE_ATTEMPTS,
                )
            else:
                batch_next_token_ids = gumbel_sample_from_pools_generic(
                    logits,
                    req_pool_indices,
                    self._temperature_pool,
                    self._top_k_pool,
                    self._top_p_pool,
                    self._seed_pool,
                    offsets_pool,
                    self._gumbel_out[:bs],
                )

        sampled = batch_next_token_ids.to(torch.int32)
        self.maybe_broadcast(sampled)

        self._write_logprob_outputs(logits_output, logits, sampled)

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
        )

        if sampling_info.vocab_mask is not None:
            sampling_info.apply_vocab_mask(
                logits=logits,
                vocab_mask=sampling_info.vocab_mask,
            )

        if sampling_info.is_all_greedy:
            target_sampled = cute_argmax(logits)
            verify_chain_target_sampled(
                predicts=predict,
                accept_index=accept_index,
                accept_token_num=accept_length,
                candidates=candidates,
                target_sampled=target_sampled,
                enable_pdl=pdl_enabled(),
            )
        else:
            offsets_pool = (
                sampling_info.valid_cache_lengths
                if sampling_info.valid_cache_lengths is not None
                else self._zero_offsets_pool
            )
            req_pool_indices = self._req_pool_indices_for_kernels(
                sampling_info.req_pool_indices, bs
            )
            if self._sample_route == _SAMPLE_ROUTE_GUMBEL_NO_FILTER:
                if logits.shape[1] <= _COMPACT_GUMBEL_VOCAB_MAX:
                    target_sampled = gumbel_sample_from_pools_compact(
                        logits,
                        req_pool_indices,
                        self._temperature_pool,
                        self._seed_pool,
                        offsets_pool,
                        self._gumbel_verify_out[: bs * num_tokens_per_req],
                        block_size=_COMPACT_GUMBEL_BLOCK_SIZE,
                        num_tokens_per_req=num_tokens_per_req,
                    )
                else:
                    target_sampled = gumbel_sample_from_pools(
                        logits,
                        req_pool_indices,
                        self._temperature_pool,
                        self._seed_pool,
                        offsets_pool,
                        self._gumbel_verify_local_ids[: bs * num_tokens_per_req],
                        self._gumbel_verify_local_scores[: bs * num_tokens_per_req],
                        self._gumbel_verify_out[: bs * num_tokens_per_req],
                        num_tokens_per_req=num_tokens_per_req,
                    )
            elif self._sample_route in (
                _SAMPLE_ROUTE_GUMBEL_TOP_K,
                _SAMPLE_ROUTE_GUMBEL_TOP_K_TOP_P,
            ):
                rows = bs * num_tokens_per_req
                if self._use_qrita_verify_top_k_route(rows, logits.shape[1]):
                    target_sampled = gumbel_sample_top_k_top_p_qrita_from_pools(
                        logits,
                        req_pool_indices,
                        self._temperature_pool,
                        self._top_k_pool,
                        self._top_p_pool,
                        self._seed_pool,
                        offsets_pool,
                        self._qrita_verify_buffer,
                        self._qrita_percentile_to_std_table,
                        self._gumbel_verify_out[:rows],
                        num_tokens_per_req=num_tokens_per_req,
                        num_programs=min(self._qrita_verify_num_programs, rows),
                    )
                else:
                    target_sampled = gumbel_sample_top_k_top_p_from_pools(
                        logits,
                        req_pool_indices,
                        self._temperature_pool,
                        self._top_k_pool,
                        self._top_p_pool,
                        self._seed_pool,
                        offsets_pool,
                        self._topk_verify_candidate_ids[:rows],
                        self._topk_verify_candidate_logits[:rows],
                        self._gumbel_verify_out[:rows],
                        block_size=_TOP_K_TOP_P_SMALL_BLOCK_SIZE,
                        top_k_pad=self._top_k_top_p_pad,
                        num_tokens_per_req=num_tokens_per_req,
                    )
            elif self._sample_route == _SAMPLE_ROUTE_GUMBEL_TOP_P:
                rows = bs * num_tokens_per_req
                target_sampled = gumbel_sample_top_p_parallel_from_pools(
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
                    self._gumbel_verify_out[:rows],
                    block_size=_TOP_P_PARALLEL_SAMPLE_BLOCK_SIZE,
                    num_attempts=_TOP_P_PARALLEL_VERIFY_ATTEMPTS,
                    num_tokens_per_req=num_tokens_per_req,
                )
            else:
                target_sampled = gumbel_sample_from_pools_generic(
                    logits,
                    req_pool_indices,
                    self._temperature_pool,
                    self._top_k_pool,
                    self._top_p_pool,
                    self._seed_pool,
                    offsets_pool,
                    self._gumbel_verify_out[: bs * num_tokens_per_req],
                    num_tokens_per_req=num_tokens_per_req,
                )
            verify_chain_target_sampled(
                predicts=predict,
                accept_index=accept_index,
                accept_token_num=accept_length,
                candidates=candidates,
                target_sampled=target_sampled,
                enable_pdl=pdl_enabled(),
            )

        accept_length += 1

        # Rank 0 remains the source of truth for attention-TP agreement.
        self.maybe_broadcast(predict, accept_index, accept_length)

        if self.config.enable_output_logprobs:
            self._write_logprob_outputs(
                logits_output,
                logits,
                predict,
            )

        return predict, accept_length


register_backend("triton", TritonSamplingBackend)
