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

"""Cache-format-independent placement of logical token slots."""

from dataclasses import dataclass

import torch
from tokenspeed_kernel.ops.kvcache.triton_cache_placement import virtual_slots_to_local


@dataclass(frozen=True)
class CachePlacement:
    """Ownership geometry for one sharded cache group, independent of its writer."""

    block_granularity: int
    virtual_block_count: int
    group: tuple[int, ...]
    rank: int

    def __post_init__(self) -> None:
        if self.block_granularity <= 0 or self.virtual_block_count <= 0:
            raise ValueError("Cache placement geometry must be positive")
        if not self.group or not 0 <= self.rank < len(self.group):
            raise ValueError("Cache placement rank must index its nonempty group")


def resolve_cache_slots(
    loc: torch.Tensor, placement: CachePlacement | None
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return local slots and an ownership mask; None placement preserves slots.

    The mask applies to both source loads and destination stores. Foreign rows
    resolve to safe dummy addresses but must never be written. This function
    knows neither the cache format nor the attention kernel's page-table layout.
    """
    if placement is None:
        return loc, None
    return virtual_slots_to_local(
        loc,
        rows_per_page=placement.block_granularity,
        virtual_block_count=placement.virtual_block_count,
        degree=len(placement.group),
        rank=placement.rank,
    )
