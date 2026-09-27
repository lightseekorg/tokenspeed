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

"""Explicit planning bounds for capacity-based CuTeDSL KDA prefill."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class KdaPrefillCapacity:
    """Separate packed token storage from per-request planning capacity.

    Args:
        token_capacity: Number of rows physically present in Q/K/V.
        num_sequences: Number of real sequence slots in the captured graph.

    Each sequence may use up to token_capacity rows, but their combined live
    lengths must fit token_capacity. Compact native planning bounds total
    chunks by ceil(token_capacity / chunk_size) + num_sequences - 1.
    The compatibility path reserves capacity independently per sequence;
    live GPU boundaries remain authoritative in either case.
    """

    token_capacity: int
    num_sequences: int

    def __post_init__(self):
        if self.token_capacity <= 0 or self.num_sequences <= 0:
            raise ValueError("KDA planning capacities must be positive")

    def boundaries_cpu(self):
        """Return host int64 planning boundaries, not live token boundaries."""
        return (
            torch.arange(self.num_sequences + 1, dtype=torch.int64)
            * self.token_capacity
        )

    def validate(self, live_boundaries, tokens):
        """Validate admission from the existing CPU mirror without a D2H."""
        if tokens != self.token_capacity:
            raise ValueError("KDA input extent differs from planning token capacity")
        if live_boundaries.numel() != self.num_sequences + 1:
            raise ValueError("KDA planning sequence count differs from live boundaries")
        lengths = live_boundaries[1:] - live_boundaries[:-1]
        if live_boundaries[0].item() != 0 or (lengths <= 0).any().item():
            raise ValueError("KDA capacity replay requires positive packed sequences")
        if live_boundaries[-1].item() > self.token_capacity:
            raise ValueError("KDA live tokens exceed planning capacity")
