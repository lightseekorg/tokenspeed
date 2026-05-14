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

import dataclasses
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)


if TYPE_CHECKING:
    from tokenspeed.runtime.engine.schedule_batch import ScheduleBatch


@dataclasses.dataclass
class SamplingBatchInfo:
    # Basic batched sampling params. Disaggregated decode populates these via
    # from_schedule_batch. The standard hot path leaves them None; sampling
    # backends gather params from their own pool-indexed buffers.
    temperatures: torch.Tensor | None = None
    top_ps: torch.Tensor | None = None
    top_ks: torch.Tensor | None = None
    min_ps: torch.Tensor | None = None

    # Whether all requests use greedy sampling
    is_all_greedy: bool = False

    # Masking tensors for grammar-guided structured outputs
    vocab_size: int = 0
    grammars: list | None = None
    vocab_mask: torch.Tensor | None = None
    # Backend-specific in-place fn ``(logits, vocab_mask) -> None``,
    # bound by ``capturable_grammar.bind_grammar_mask_buf`` so the
    # captured sampler can apply the bitmask without branching on
    # backend.
    apply_vocab_mask: Callable[[torch.Tensor, torch.Tensor], None] | None = None

    # An event used for overlap schedule
    sampling_info_done: threading.Event | None = None

    # int64[bs] — req_pool_idx per batch row. Sampling backends gather
    # their pool-indexed scalar buffers (temperature / top_k / top_p /
    # seeds / penalties / logit_bias / counts) against this index.
    req_pool_indices: torch.Tensor | None = None

    # int32[pool_rows] — RuntimeStates.valid_cache_lengths, read-only
    # reference. Sampling backends derive the per-request Philox offset
    # from `valid_cache_lengths.index_select(0, req_pool_indices)`;
    # carrying the reference rather than the gathered view keeps the
    # index_select inside the captured graph.
    valid_cache_lengths: torch.Tensor | None = None

    # Device
    device: str = "cuda"

    @classmethod
    def from_schedule_batch(
        cls, batch: ScheduleBatch, vocab_size: int
    ) -> SamplingBatchInfo:
        reqs = batch.reqs
        device = batch.device
        temperatures = (
            torch.tensor(
                [r.sampling_params.temperature for r in reqs],
                dtype=torch.float,
            )
            .view(-1, 1)
            .to(device, non_blocking=True)
        )
        top_ps = torch.tensor(
            [r.sampling_params.top_p for r in reqs], dtype=torch.float
        ).to(device, non_blocking=True)
        top_ks = torch.tensor(
            [r.sampling_params.top_k for r in reqs], dtype=torch.int32
        ).to(device, non_blocking=True)
        min_ps = torch.tensor(
            [r.sampling_params.min_p for r in reqs], dtype=torch.float
        ).to(device, non_blocking=True)

        ret = cls(
            temperatures=temperatures,
            top_ps=top_ps,
            top_ks=top_ks,
            min_ps=min_ps,
            is_all_greedy=all(r.sampling_params.top_k <= 1 for r in reqs),
            vocab_size=vocab_size,
            device=device,
        )
        return ret
