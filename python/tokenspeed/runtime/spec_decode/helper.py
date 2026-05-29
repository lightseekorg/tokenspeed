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

"""Speculative decoding helpers."""

from __future__ import annotations

import torch

from tokenspeed.runtime.execution.context import ForwardContext


def apply_draft_active_row_slice(
    tensor: torch.Tensor, ctx: ForwardContext
) -> torch.Tensor:
    """Drop dead-position rows when drafter requested the slice; no-op otherwise.

    Called inside attention modules just before ``o_proj`` so ``o_proj`` only
    runs on live rows. Idle ranks have ``gather_ids=None`` and pass through.
    """
    if not ctx.draft_active_row_slice or ctx.gather_ids is None:
        return tensor
    return tensor.index_select(0, ctx.gather_ids)


def apply_draft_active_row_slice_post_attn(
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    ctx: ForwardContext,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Finalize the active-row slice after the layer's self-attention.

    Active ranks: attn already sliced its output via ``apply_draft_active_row_slice``;
    here we gather residual to match and update ``input_num_tokens``.

    Idle ranks (``gather_ids is None``): tensors are empty already, but we
    still switch ``global_num_tokens`` to ``global_bs`` so downstream MoE
    collectives see cross-rank-consistent scatter sizes (active ranks switch
    here too — DP ranks must agree).

    Both: clear ``gather_ids`` / ``draft_active_row_slice`` so downstream
    layernorms, MLP, and final-norm don't double-slice.
    """
    if not ctx.draft_active_row_slice:
        return hidden_states, residual
    gather_ids = ctx.gather_ids
    if gather_ids is not None:
        assert hidden_states.size(0) == gather_ids.size(0), (
            "attention module must call apply_draft_active_row_slice before "
            "apply_draft_active_row_slice_post_attn"
        )
        if residual is not None and residual.size(0) != gather_ids.size(0):
            residual = residual.index_select(0, gather_ids)
        ctx.input_num_tokens = gather_ids.size(0)
    if ctx.global_bs is not None:
        ctx.global_num_tokens = ctx.global_bs
    ctx.gather_ids = None
    ctx.draft_active_row_slice = False
    return hidden_states, residual
