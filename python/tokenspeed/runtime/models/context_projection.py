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

"""Shared target-tap projection arithmetic for block drafters.

A concatenated linear projection is the sum of its per-tap projections.
Tap normalization precedes each projection; output normalization follows the
complete sum. The accumulator has one draft-hidden row per target token.
"""

from __future__ import annotations

from typing import Protocol

import torch
from torch import nn
from torch.nn import functional as F


def project_context_tap(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    norm: nn.Module | None,
) -> torch.Tensor:
    """Project one tap after its optional independent normalization.

    Args:
        hidden: Complete target hidden rows, [tokens, target_hidden].
        weight: This tap's projection columns, [draft_hidden, target_hidden].
        norm: The tap's normalization module, or None.

    Returns:
        Projected rows of shape [tokens, draft_hidden].
    """
    if norm is not None:
        hidden = norm(hidden)
    return F.linear(hidden.to(weight.dtype), weight)


class ContextStageMapping(Protocol):
    """Pipeline ownership needed by context production."""

    is_first_pp_rank: bool
    is_last_pp_rank: bool


class TargetContextProjector(Protocol):
    """Weight-owning portion shared by local and pipeline context production."""

    hidden_size: int
    mapping: ContextStageMapping

    def project_target_tap(
        self, capture_idx: int, hidden: torch.Tensor
    ) -> torch.Tensor:
        """Return one positional tap's contribution before output normalization."""

    def finalize_target_projection(self, projected: torch.Tensor) -> torch.Tensor:
        """Normalize the complete sum in the draft's activation dtype."""

    def write_context_kv(
        self,
        ctx_hidden: torch.Tensor,
        positions: torch.Tensor,
        cache_locs: torch.Tensor,
        token_to_kv_pool,
    ) -> None:
        """Materialize normalized context rows in the draft's native cache layout."""


def context_tap_owner_layer(
    layer_id: int, target_num_layers: int, aux_hidden_stream: str
) -> int:
    """Return the layer whose stage can produce a tap without extra weights.

    Prefix taps belong to their completed layer. An AttnRes tap uses the next
    layer's attention-mixing parameters, so it belongs to that consumer;
    the final tap uses the last stage's output-mixing parameters.
    """
    if aux_hidden_stream == "prefix":
        return layer_id
    if aux_hidden_stream == "attn_res":
        return min(layer_id + 1, target_num_layers - 1)
    raise ValueError(f"Unknown target hidden stream {aux_hidden_stream!r}")
