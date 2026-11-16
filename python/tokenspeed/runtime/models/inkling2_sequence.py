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

"""Sequence-parallel prefill of Inkling2's residual stream.

A prefill-sized forward splits its tokens between the ranks of the attention
TP group, which is also the MoE EP group. Each sublayer's partial output (the
attention output projection, the MoE route fold) is reduce-scattered by rows
instead of all-reduced; every rank convolves and depth-mixes only its share
of the tokens, and stores its share of each mix through multicast memory, so
the next projection, which needs every token, reads the full mix. The
communication moves as many bytes as the all-reduce did, and the per-token
work of the residual stream runs once per token instead of once per rank.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
from tokenspeed_kernel.ops.communication.triton import (
    MulticastRows,
    TritonRSAGState,
    create_state,
    row_reduce_scatter,
)

from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.layers.attention_residual import ResidualShare

# Shares start on ShortConv prefill tiles (inkling_ring_sconv's token tile),
# so neighbouring ranks never compute the same output tile.
_SHARE_ALIGN = 16


@dataclass(frozen=True)
class SequenceShare:
    """This rank's share of one sequence-parallel forward.

    ``rows`` are the rows this rank reduces: its share, the convolution
    window's rows before it, and each request's last ring rows, which every
    rank writes into its own copy of the convolution state.
    """

    first: int
    count: int
    rows: torch.Tensor
    residual: ResidualShare
    attention_partials: TritonRSAGState
    mlp_partials: TritonRSAGState

    @property
    def token_range(self) -> tuple[int, int]:
        return self.first, self.count

    def reduce(self, partial: torch.Tensor, partials: TritonRSAGState) -> torch.Tensor:
        """The group's sum of ``partial`` (``[T, H]`` or ``[T, 1, H]``) on
        this rank's rows; other rows are unspecified. A partial its producer
        wrote into ``partials``' buffer is read in place."""
        rows = partial.reshape(partial.shape[0], -1)
        if rows.data_ptr() != partials.comm_buff.data_ptr():
            partials.comm_buff[: rows.shape[0]].copy_(rows)
        return row_reduce_scatter(partials, self.rows, torch.empty_like(rows))


class Inkling2SequenceParallel:
    """Buffers of the sequence-parallel prefill over one TP group.

    Creation rendezvouses the group (every rank builds it, eagerly and
    outside graph capture) and raises when the group has no multicast memory.
    """

    # Smaller forwards keep the all-reduces, whose fixed cost is lower (B300
    # TP4 prefill: 45.2 against 44.0 ms at 2K tokens, 57.7 against 59.9 at 3K).
    min_tokens = 2304

    def __init__(
        self,
        group: dist.ProcessGroup,
        *,
        max_tokens: int,
        hidden: int,
        conv_window: int,
        device: torch.device,
    ) -> None:
        self.rank = dist.get_rank(group)
        self.world = dist.get_world_size(group)
        self.max_tokens = max_tokens
        self.conv_window = conv_window
        self.attention_partials = create_state(
            group, self.rank, max_tokens, hidden, device
        )
        self.mlp_partials = create_state(group, self.rank, max_tokens, hidden, device)
        self.attention_mixes = MulticastRows(group, 3 * max_tokens * hidden, device)
        self.mlp_mixes = MulticastRows(group, max_tokens * hidden, device)
        self.gather = MulticastRows(group, max_tokens * hidden, device)

    def serves(self, ctx: ForwardContext, tokens: int) -> bool:
        """Whether a forward of ``tokens`` rows splits: extend forwards of at
        least ``min_tokens`` rows that fit the buffers."""
        return (
            ctx.forward_mode.is_extend_or_mixed()
            and self.min_tokens <= tokens <= self.max_tokens
        )

    def share(self, ctx: ForwardContext, tokens: int) -> SequenceShare:
        """This rank's share of the forward's ``tokens`` rows."""
        size = -(-tokens // (self.world * _SHARE_ALIGN)) * _SHARE_ALIGN
        first = min(self.rank * size, tokens)
        count = min(size, tokens - first)
        backend = ctx.attn_backend
        ring = backend.conv_pool.layer_state_wd(0).shape[1]
        ends = backend.conv_prefill_metadata.query_start_loc[1:]
        device = ends.device
        own = torch.arange(
            max(first - (self.conv_window - 1), 0),
            first + count,
            dtype=torch.int32,
            device=device,
        )
        tails = ends[:, None] - torch.arange(
            ring, 0, -1, dtype=torch.int32, device=device
        )
        tails = tails.clamp(min=0, max=tokens - 1).flatten()
        return SequenceShare(
            first=first,
            count=count,
            rows=torch.cat((own, tails)),
            residual=ResidualShare(
                first=first,
                count=count,
                attention=self.attention_mixes,
                mlp=self.mlp_mixes,
                gather=self.gather,
            ),
            attention_partials=self.attention_partials,
            mlp_partials=self.mlp_partials,
        )
