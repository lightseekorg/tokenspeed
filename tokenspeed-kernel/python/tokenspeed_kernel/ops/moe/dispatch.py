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

"""Expert placement inputs for routing kernels.

With redundant experts a logical expert has several physical replicas, each
on some rank. Routing then emits physical ids: every rank runs the same
routing over the same tokens, so the replica must be a pure function of the
token row and route rank -- ``replicas[logical, (row + rank) % count]`` --
for exactly one rank to own each (token, expert) pair. The same kernel can
count how many routes land on every physical expert, which is what the
placement algorithm consumes.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ExpertDispatch:
    """One MoE layer's logical-to-physical routing tables.

    Attributes:
        replicas: ``[num_logical, X]`` int32 physical ids of every logical
            expert's replicas, ``-1`` padded; contiguous.
        num_replicas: ``[num_logical]`` int32 valid entries per row, all >= 1.
        load: ``[num_physical]`` int32 counters the routing kernel increments
            once per route, or None to skip counting.
    """

    replicas: torch.Tensor
    num_replicas: torch.Tensor
    load: torch.Tensor | None

    def __post_init__(self) -> None:
        if self.replicas.ndim != 2 or self.replicas.dtype != torch.int32:
            raise ValueError("replicas must be an int32 [num_logical, X] table")
        if (
            self.num_replicas.shape != (self.replicas.shape[0],)
            or self.num_replicas.dtype != torch.int32
        ):
            raise ValueError("num_replicas must be int32 [num_logical]")
        if not (self.replicas.is_contiguous() and self.num_replicas.is_contiguous()):
            raise ValueError("dispatch tables must be contiguous")
        if self.load is not None and (
            self.load.ndim != 1
            or self.load.dtype != torch.int32
            or not self.load.is_contiguous()
        ):
            raise ValueError("load must be a contiguous int32 [num_physical] counter")

    @property
    def max_replicas(self) -> int:
        return self.replicas.shape[1]


def dispatch_topk_ids_reference(
    topk_ids: torch.Tensor, dispatch: ExpertDispatch
) -> torch.Tensor:
    """Tensor-op reference of the kernel's replica choice (and load counting)."""
    rows = torch.arange(topk_ids.shape[0], device=topk_ids.device)[:, None]
    ranks = torch.arange(topk_ids.shape[1], device=topk_ids.device)[None, :]
    logical = topk_ids.long()
    count = dispatch.num_replicas.long()[logical]
    physical = dispatch.replicas.long()[logical, (rows + ranks) % count]
    if dispatch.load is not None:
        dispatch.load.scatter_add_(
            0,
            physical.reshape(-1),
            torch.ones_like(physical.reshape(-1), dtype=torch.int32),
        )
    return physical.to(topk_ids.dtype)
