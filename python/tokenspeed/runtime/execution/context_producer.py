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

"""Context production shared by prefill pipeline stages.

The producer owns only immutable projection weights and the destination cache
view. Accumulated rows travel in the forward's PP state, so queued prefill
chunks cannot overwrite a different chunk's projection buffer.
"""

from __future__ import annotations

import torch

from tokenspeed.runtime.models.context_projection import TargetContextProjector


class PrefillTargetContextProducer:
    """Project stage-local target taps and write context before last-stage drafting."""

    supports_pd_layerwise_finalization = True

    def __init__(self, model: TargetContextProjector, token_to_kv_pool) -> None:
        self.model = model
        self.token_to_kv_pool = token_to_kv_pool
        if bool(model.mapping.is_last_pp_rank) != (token_to_kv_pool is not None):
            raise ValueError(
                "Only the final pipeline stage may own the draft context cache"
            )

    def begin_stage(
        self, hidden: torch.Tensor, inbound: torch.Tensor | None
    ) -> torch.Tensor:
        """Return this chunk's accumulator, continuing its upstream partial sum."""
        expected = (hidden.shape[0], self.model.hidden_size)
        if inbound is None:
            if not self.model.mapping.is_first_pp_rank:
                raise ValueError(
                    "A downstream context producer is missing its projection sum"
                )
            return torch.zeros(expected, dtype=torch.float32, device=hidden.device)
        if tuple(inbound.shape) != expected or inbound.dtype != torch.float32:
            raise ValueError(
                f"Invalid pipeline context sum: expected {expected} float32"
            )
        return inbound

    def add_capture(
        self, projected: torch.Tensor, capture_idx: int, hidden: torch.Tensor
    ) -> None:
        """Add one owned tap; float32 accumulation avoids rounding between taps."""
        if hidden.shape[0] != projected.shape[0]:
            raise ValueError("Target tap and pipeline context token counts differ")
        projected.add_(self.model.project_target_tap(capture_idx, hidden))

    def write_context(
        self,
        projected: torch.Tensor,
        positions: torch.Tensor,
        cache_locs: torch.Tensor,
    ) -> None:
        """Finalize normalization and write every draft layer's prompt context KV."""
        if self.token_to_kv_pool is None:
            raise RuntimeError(
                "Only the final pipeline stage can write draft context KV"
            )
        if (
            projected.shape[0] != positions.numel()
            or positions.numel() != cache_locs.numel()
        ):
            raise ValueError(
                "Context projection, positions, and cache locations must agree"
            )
        self.model.write_context_kv(
            self.model.finalize_target_projection(projected),
            positions,
            cache_locs,
            self.token_to_kv_pool,
        )
