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

"""PyTorch references for ``tokenspeed_kernel.ops.attention.msa`` (MiniMax
sparse attention): block selection and attention over the selected blocks."""

from __future__ import annotations

import math

import torch
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures


@register_kernel(
    "attention",
    "msa_select_blocks",
    name="torch_msa_select_blocks",
    solution="torch",
    signatures=format_signatures("query", "dense", {torch.bfloat16, torch.float32}),
    priority=Priority.REFERENCE,
)
def torch_msa_select_blocks(
    query: torch.Tensor,
    keys: torch.Tensor,
    query_position: int,
    *,
    block_size: int,
    topk: int,
) -> torch.Tensor:
    """Top-k key blocks of one query head by maximum scaled logit; the block
    holding the query always wins.

    Args:
        query: ``[D]``.
        keys: ``[positions, D]``, of which ``[0, query_position]`` are visible.

    Returns:
        Indices of the selected blocks, best first.
    """
    scores = query.float() @ keys[: query_position + 1].float().T
    scores *= query.shape[-1] ** -0.5
    num_blocks = math.ceil((query_position + 1) / block_size)
    scores = torch.nn.functional.pad(
        scores, (0, num_blocks * block_size - scores.numel()), value=-torch.inf
    )
    block_scores = scores.view(num_blocks, block_size).amax(dim=-1)
    block_scores[query_position // block_size] = torch.inf
    return block_scores.topk(min(topk, num_blocks)).indices


@register_kernel(
    "attention",
    "msa_block_attention",
    name="torch_msa_block_attention",
    solution="torch",
    signatures=format_signatures("query", "dense", {torch.bfloat16, torch.float32}),
    priority=Priority.REFERENCE,
)
def torch_msa_block_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    selected_blocks: torch.Tensor,
    block_table: torch.Tensor,
    query_position: int,
    *,
    block_size: int,
) -> torch.Tensor:
    """Causal attention of one query head over its selected key blocks.

    Block ``b`` reads page ``block_table[b]`` (head 0) of the caches; keys
    after ``query_position`` are masked.

    Returns:
        FP32 ``[Dv]``.
    """
    blocks = selected_blocks.long()
    keys = torch.cat([key_cache[block_table[b].long(), 0] for b in blocks], dim=0)
    values = torch.cat([value_cache[block_table[b].long(), 0] for b in blocks], dim=0)
    key_positions = (
        blocks[:, None] * block_size
        + torch.arange(block_size, device=query.device)[None]
    ).flatten()
    visible = key_positions <= query_position
    probabilities = torch.softmax(
        query.float() @ keys[visible].float().T * (query.shape[-1] ** -0.5), dim=-1
    )
    return probabilities @ values[visible].float()
