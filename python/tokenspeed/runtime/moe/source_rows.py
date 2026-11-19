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

"""Live rows contributed by each expert-parallel source after attention scatter."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import accumulate

import torch

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.query_shard import scatter_count


class MoESourceRows:
    """One fixed-address device vector shared by the model's MoE layers."""

    def __init__(self, mapping: Mapping, *, device, max_rows_per_rank: int):
        self.mapping = mapping
        self.max_rows_per_rank = max_rows_per_rank
        self.counts: tuple[int, ...] | None = None
        self.source_token_counts = torch.zeros(
            len(mapping.moe.tp_ep_group), dtype=torch.int32, device=device
        )

    def upload_scattered(self, padded_tokens: int, global_num_tokens: Sequence[int]):
        """Intersect each captured TP shard with its DP group's live prefix."""
        attn = self.mapping.attn
        if len(global_num_tokens) != self.mapping.world_size:
            raise ValueError("expected one scheduler token count per global rank")
        lengths = scatter_count(padded_tokens, attn.tp_size)
        offsets = tuple(accumulate(lengths, initial=0))
        counts = []
        for rank in self.mapping.moe.tp_ep_group:
            dp, tp = divmod(attn.scatter_index(rank), attn.tp_size)
            live = global_num_tokens[dp * attn.tp_size]
            if not 0 <= live <= padded_tokens:
                raise ValueError(
                    f"{live} live rows do not fit {padded_tokens} padded rows"
                )
            counts.append(min(lengths[tp], max(0, live - offsets[tp])))
        self.publish(tuple(counts))

    def publish(self, counts: tuple[int, ...]) -> None:
        if len(counts) != self.source_token_counts.numel() or any(
            count < 0 or count > self.max_rows_per_rank for count in counts
        ):
            raise ValueError(
                f"expected one source count in [0,{self.max_rows_per_rank}] per EP rank"
            )
        # A later step can prepare its counts while the previous DMA is in flight.
        staging = torch.tensor(
            counts,
            dtype=torch.int32,
            device="cpu",
            pin_memory=self.source_token_counts.device.type != "cpu",
        )
        self.source_token_counts.copy_(staging, non_blocking=True)
        self.counts = counts
