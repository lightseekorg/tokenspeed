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

"""Speculative decoding helpers for the draft head's first-step active-row slice.

The draft head emits a hidden state for every input token but downstream only
consumes the per-request last token (lm_head + next draft step). These helpers
drop the dead-position rows so MLP / norm / collective ops only touch live
rows.

Usage in a single layer:

    attn_output = self.attn(...)
    attn_output = apply_draft_active_row_slice_pre_oproj(attn_output, ctx)    # before o_proj
    output = self.o_proj(attn_output)
    ...
    hidden_states, residual, ctx = apply_draft_active_row_slice_post_attn(
        hidden_states, residual, ctx,
    )                                                                # once per layer
"""

from __future__ import annotations

import torch

from tokenspeed.runtime.execution.context import ForwardContext


def apply_draft_active_row_slice_pre_oproj(
    tensor: torch.Tensor, ctx: ForwardContext
) -> torch.Tensor:
    """Gather ``tensor`` to one row per request; no-op if not requested or idle."""
    if not ctx.draft_active_row_slice or ctx.gather_ids is None:
        return tensor
    return tensor.index_select(0, ctx.gather_ids)


def _post_slice_global_num_tokens(
    ctx: ForwardContext, gather_ids: torch.Tensor | None
) -> list[int] | None:
    """Compute ``global_num_tokens`` after the active-row slice.

    Three paths:
    - ``global_bs`` set (production event-loop): source of truth.
    - ``global_bs`` unset + active rank (cuda_graph_wrapper capture path):
      broadcast local bs — capture assumes uniform batch across ranks.
    - ``global_bs`` unset + idle rank: can't infer from this rank alone;
      clear so downstream collective sizing degrades to the non-DP path
      instead of using stale capture totals.
    """
    if ctx.global_bs is not None:
        return ctx.global_bs
    if ctx.global_num_tokens is None:
        return None
    if gather_ids is not None:
        return [gather_ids.size(0)] * len(ctx.global_num_tokens)
    return None


def apply_draft_active_row_slice_post_attn(
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    ctx: ForwardContext,
) -> tuple[torch.Tensor, torch.Tensor | None, ForwardContext]:
    """Finalize the active-row slice after the layer's self-attention.

    Active rank: gather ``residual`` to match the already-sliced
    ``attn_output`` and update ``ctx.input_num_tokens``.

    All ranks (active + idle): switch ``ctx.global_num_tokens`` to the
    post-slice scatter sizes (see ``_post_slice_global_num_tokens``) so
    cross-rank MoE / RSAG see consistent values, then clear ``gather_ids``
    and ``draft_active_row_slice``.

    The returned ctx is the same object — mutated in place; callers must
    rebind (``hidden_states, residual, ctx = apply_..._post_attn(...)``)
    so the ctx mutation is visible at the call site rather than hidden.
    """
    if not ctx.draft_active_row_slice:
        return hidden_states, residual, ctx

    gather_ids = ctx.gather_ids
    if gather_ids is not None:
        assert hidden_states.size(0) == gather_ids.size(0), (
            "attn module must call apply_draft_active_row_slice_pre_oproj before this"
        )
        if residual is not None and residual.size(0) != gather_ids.size(0):
            residual = residual.index_select(0, gather_ids)
        ctx.input_num_tokens = gather_ids.size(0)

    ctx.global_num_tokens = _post_slice_global_num_tokens(ctx, gather_ids)
    ctx.gather_ids = None
    ctx.draft_active_row_slice = False

    return hidden_states, residual, ctx
