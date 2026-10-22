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

"""Device-neutral state and lifecycle for full-featured samplers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from tokenspeed.runtime.sampling.backends.base import SamplingBackendConfig
from tokenspeed.runtime.sampling.backends.pooled import PooledSamplingBackend
from tokenspeed.runtime.utils.nvtx import nvtx_range

if TYPE_CHECKING:
    from tokenspeed.runtime.sampling.sampling_batch_info import SamplingBatchInfo
    from tokenspeed.runtime.sampling.sampling_params import SamplingParams


class FullSamplingBackendBase(PooledSamplingBackend):
    """Per-request penalties, token counts, and logit bias.

    Concrete backends own probability filtering and chain verification. This
    base owns only state whose semantics are identical across devices.
    """

    def __init__(self, config: SamplingBackendConfig) -> None:
        super().__init__(config)
        if config.max_req_pool_size <= 0 or config.vocab_size <= 0:
            raise ValueError(
                f"{type(self).__name__} requires max_req_pool_size > 0 and "
                f"vocab_size > 0; got max_req_pool_size={config.max_req_pool_size}, "
                f"vocab_size={config.vocab_size}"
            )

        pool_rows = config.max_req_pool_size + 1
        self._counts = torch.zeros(
            (pool_rows, config.vocab_size), dtype=torch.int32, device=config.device
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
        self._rep_pen_pool = torch.ones(
            (pool_rows,), dtype=torch.bfloat16, device=config.device
        )

    def _reset_slot(self, pool_idx: int, sp: SamplingParams) -> None:
        super()._reset_slot(pool_idx, sp)
        self._min_p_pool[pool_idx].fill_(float(sp.min_p))
        self._freq_pen_pool[pool_idx].fill_(float(sp.frequency_penalty))
        self._pres_pen_pool[pool_idx].fill_(float(sp.presence_penalty))
        self._rep_pen_pool[pool_idx].fill_(float(sp.repetition_penalty))
        self._counts[pool_idx].fill_(0)
        self._logit_bias[pool_idx].fill_(0.0)

        bias_map = getattr(sp, "logit_bias", None)
        if not bias_map:
            return
        vocab = self._logit_bias.shape[1]
        raw_ids = [int(token_id) for token_id in bias_map]
        assert all(0 <= token_id < vocab for token_id in raw_ids), (
            "logit_bias contains out-of-vocab token id(s); "
            f"vocab_size={vocab}, "
            f"offending={[token_id for token_id in raw_ids if not 0 <= token_id < vocab]}"
        )
        token_ids = torch.tensor(
            raw_ids, device=self._logit_bias.device, dtype=torch.long
        )
        bias_values = torch.tensor(
            list(bias_map.values()),
            device=self._logit_bias.device,
            dtype=self._logit_bias.dtype,
        )
        self._logit_bias[pool_idx, token_ids] = bias_values

    def reset_capture_state(self) -> None:
        self._counts[0].fill_(0)

    @nvtx_range("sampling:penalties", color="yellow")
    def _apply_penalties_and_bias_for_indices(
        self,
        logits: torch.Tensor,
        pool_indices: torch.Tensor,
        *,
        num_tokens_per_req: int = 1,
    ) -> torch.Tensor:
        if num_tokens_per_req > 1:
            pool_indices = torch.repeat_interleave(
                pool_indices, num_tokens_per_req, dim=0
            )

        counts = self._counts.index_select(0, pool_indices)
        active = counts > 0
        counts_f = counts.to(logits.dtype)
        active_f = active.to(logits.dtype)
        rep = (
            self._rep_pen_pool.index_select(0, pool_indices)
            .to(logits.dtype)
            .unsqueeze(-1)
        )
        freq = (
            self._freq_pen_pool.index_select(0, pool_indices)
            .to(logits.dtype)
            .unsqueeze(-1)
        )
        presence = (
            self._pres_pen_pool.index_select(0, pool_indices)
            .to(logits.dtype)
            .unsqueeze(-1)
        )
        scales = torch.where(active, rep.expand_as(logits), torch.ones_like(logits))
        logits = torch.where(logits > 0, logits / scales, logits * scales)
        logits = logits - freq * counts_f - presence * active_f
        return logits + self._logit_bias.index_select(0, pool_indices)

    def _apply_penalties_and_bias(
        self,
        logits: torch.Tensor,
        sampling_info: SamplingBatchInfo,
        num_tokens_per_req: int = 1,
    ) -> torch.Tensor:
        return self._apply_penalties_and_bias_for_indices(
            logits,
            sampling_info.req_pool_indices,
            num_tokens_per_req=num_tokens_per_req,
        )

    @nvtx_range("sampling:accum_counts", color="yellow")
    def _accumulate_counts(
        self,
        pool_indices: torch.Tensor,
        tokens: torch.Tensor,
        weights: torch.Tensor,
    ) -> None:
        self._counts.index_put_(
            (pool_indices, tokens.long()),
            weights.to(torch.int32),
            accumulate=True,
        )
