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

"""DCP attention collectives; all probability arithmetic uses natural-log LSE."""

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.attention.dsv4._triton.dcp import (
    dcp_apply_sink,
    dcp_weight_for_reduce_scatter,
)

from tokenspeed.runtime.distributed.comm_ops import (
    all_gather,
    reduce_scatter,
)


def gather_query_heads(query: torch.Tensor, group: tuple[int, ...]) -> torch.Tensor:
    """Gather only actual TP query heads after QNorm/RoPE; padding stays local."""
    if len(group) == 1:
        return query
    tokens, heads, dim = query.shape
    # The 2-D inner-dimension collective uses the existing low-latency
    # backend where supported, with its topology/dtype/NCCL fallbacks. Its
    # symmetric buffer is sized for the prefill token budget although decode
    # only ever gathers max_decode_bs * spec_tokens rows; a per-collective
    # capacity needs the symmetric buffers managed in one place first.
    gathered = all_gather(
        query.reshape(tokens, heads * dim).contiguous(), group, dim=-1
    )
    return gathered.reshape(tokens, heads * len(group), dim)


def combine_attention_partials(
    local_output: torch.Tensor,
    local_lse: torch.Tensor,
    *,
    group: tuple[int, ...],
    rank: int,
    sink: torch.Tensor,
) -> torch.Tensor:
    """Gather LSE and reduce-scatter weighted partials to the TP head owner.

    Args:
        local_output: CUDA local context output [tokens, gathered_heads, head_dim].
        local_lse: Natural-log FP32 LSE [tokens, gathered_heads].
        group: Consecutive DCP subgroup of attention TP.
        rank: This process's position in group.
        sink: Original TP-local sink logits.

    Returns:
        Original-dtype output [tokens, TP-local heads, head_dim], with the
        sink applied once after combining every context shard.
    """
    degree = len(group)
    if not 0 <= rank < degree or local_output.shape[1] % degree:
        raise ValueError("DCP combine topology does not partition query heads")
    if local_lse.shape != local_output.shape[:-1]:
        raise ValueError("DCP combine output and LSE shapes disagree")
    heads = local_output.shape[1] // degree
    if sink.numel() < heads:
        raise ValueError("DCP sink must cover the TP-local heads")
    gathered_lse = all_gather(local_lse.float().unsqueeze(0).contiguous(), group, dim=0)
    weighted, lse = dcp_weight_for_reduce_scatter(local_output, gathered_lse, rank)
    output = reduce_scatter(weighted, group).movedim(0, 1)
    return dcp_apply_sink(output, lse, sink, dtype=local_output.dtype)
