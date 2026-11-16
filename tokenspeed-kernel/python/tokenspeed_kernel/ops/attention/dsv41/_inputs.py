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

"""Index inputs and outputs that a kernel prepares or resolves in its launch.

Projected index inputs have their preprocessing fused into their consumer;
selection outputs are resolved by their producer in the same launch.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class IndexQueryInput:
    """Pre-RoPE BF16 queries and the inputs needed to pack them for scoring.

    values is [tokens, heads, 128], positions is integer [tokens], and
    cos_sin_cache is FP32 [max_position, 64]. The separately supplied head
    weights are unscaled BF16; weight_scale is applied before BF16 rounding.
    Slicing preserves the position mapping. This record owns no cached state.
    """

    values: torch.Tensor
    positions: torch.Tensor
    cos_sin_cache: torch.Tensor
    weight_scale: float

    def __getitem__(self, rows) -> IndexQueryInput:
        return IndexQueryInput(
            self.values[rows],
            self.positions[rows],
            self.cos_sin_cache,
            self.weight_scale,
        )


@dataclass(frozen=True)
class IndexKeyInput:
    """Projected BF16 [tokens,128] keys awaiting normalization and cache storage.

    norm_weight is BF16 [128], cos_sin_cache is FP32 [max_position, rotary_dim],
    and norm_eps is positive. The cache writer supplies positions and slots;
    normalization, RoPE, FP4 packing and the cache write share one launch.
    """

    values: torch.Tensor
    norm_weight: torch.Tensor
    cos_sin_cache: torch.Tensor
    norm_eps: float


@dataclass(frozen=True)
class SelectionSlots:
    """Caller-owned physical slots for index_topk's selected rows.

    out is contiguous int32 [tokens, topk]. The other fields are exactly what
    global_slots takes for the same rows: integer positions and requests
    [tokens] (query causality and table row), the owner's integer
    [requests, pages] LCM block table, raw tokens per global row (ratio) and
    the field's page bound. out ends up equal to global_slots of the selected
    row IDs. Slicing selects query rows. This record owns no cached state.
    """

    out: torch.Tensor
    positions: torch.Tensor
    requests: torch.Tensor
    table: torch.Tensor
    ratio: int
    pages: int

    def __getitem__(self, rows) -> SelectionSlots:
        # A basic slice keeps out a view; advanced indexing would copy it and
        # the selected slots would never reach the caller's tensor.
        if not isinstance(rows, slice):
            raise TypeError(
                "SelectionSlots supports only slice indexing, so out stays a view"
            )
        return SelectionSlots(
            self.out[rows],
            self.positions[rows],
            self.requests[rows],
            self.table,
            self.ratio,
            self.pages,
        )
