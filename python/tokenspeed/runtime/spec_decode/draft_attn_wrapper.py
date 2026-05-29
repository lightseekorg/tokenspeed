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

"""Draft attention wrapper.

Wraps PagedAttention on the draft model's last layer to drop dead-row compute
downstream of attention: gathers attention output to one row per request
before o_proj / MLP / norm / collectives.

- Instance forward serves single-call attn modules (Llama Eagle3, Qwen MTP).
- Static ``pre_oproj`` serves multi-call attn (DeepSeek MLA) which slices
  once on its merged output buffer.
- Static ``post_attn`` is layer-level finalize: gathers ``residual`` to
  match the sliced ``attn_output`` and switches scatter sizes downstream.
"""

from __future__ import annotations

import torch
from torch import nn

from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.paged_attention import PagedAttention


class DraftSliceAttnWrapper(nn.Module):

    def __init__(self, inner: PagedAttention):
        super().__init__()
        self.inner = inner

    def __getattr__(self, name: str):
        # Forward attribute access to inner so consumers (fused KV write,
        # layer-id readers) see a drop-in PagedAttention replacement.
        try:
            return super().__getattr__(name)
        except AttributeError:
            inner = self._modules.get("inner")
            if inner is None:
                raise
            return getattr(inner, name)

    @staticmethod
    def _active(ctx: ForwardContext) -> bool:
        return ctx.draft_active_row_slice and ctx.gather_ids is not None

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
        save_kv_cache: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        active = self._active(ctx)

        # Decode catch-up with prewritten KV: slice Q to bs rows and dispatch
        # to the decode kernel directly — the standard kernel would otherwise
        # run on the dead rows we are about to drop.
        if active and not save_kv_cache and ctx.forward_mode.is_decode():
            q = q.index_select(0, ctx.gather_ids)
            return ctx.attn_backend.forward(
                q,
                None,
                None,
                self.inner,
                out_cache_loc,
                ctx.token_to_kv_pool,
                ForwardMode.DECODE,
                ctx.bs,
                save_kv_cache=False,
                **kwargs,
            )

        attn_output = self.inner(
            q,
            k,
            v,
            ctx=ctx,
            out_cache_loc=out_cache_loc,
            save_kv_cache=save_kv_cache,
            **kwargs,
        )
        if active:
            attn_output = attn_output.index_select(0, ctx.gather_ids)
        return attn_output

    @staticmethod
    def pre_oproj(tensor: torch.Tensor, ctx: ForwardContext) -> torch.Tensor:
        """Gather to one row per request; no-op if not active."""
        if not DraftSliceAttnWrapper._active(ctx):
            return tensor
        return tensor.index_select(0, ctx.gather_ids)

    @staticmethod
    def post_attn(
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        ctx: ForwardContext,
    ) -> tuple[torch.Tensor, torch.Tensor | None, ForwardContext]:
        """Layer-level finalize after self-attention.

        Gathers ``residual`` to match the already-sliced ``attn_output``,
        switches ``ctx.input_num_tokens`` / ``global_num_tokens`` to the
        post-slice scatter sizes (active + idle ranks must agree on these
        for downstream MoE / RSAG), then clears the slice flags.

        Returned ctx is the same object — callers must rebind so the
        mutation is visible at the call site.
        """
        if not ctx.draft_active_row_slice:
            return hidden_states, residual, ctx

        gather_ids = ctx.gather_ids
        if gather_ids is not None:
            assert hidden_states.size(0) == gather_ids.size(
                0
            ), "attn must apply pre_oproj before this"
            if residual is not None and residual.size(0) != gather_ids.size(0):
                residual = residual.index_select(0, gather_ids)
            ctx.input_num_tokens = gather_ids.size(0)

        ctx.global_num_tokens = DraftSliceAttnWrapper._post_slice_global_num_tokens(
            ctx,
            gather_ids,
        )
        ctx.gather_ids = None
        ctx.draft_active_row_slice = False

        return hidden_states, residual, ctx

    @staticmethod
    def _post_slice_global_num_tokens(
        ctx: ForwardContext,
        gather_ids: torch.Tensor | None,
    ) -> list[int] | None:
        # global_bs set: source of truth; unset + active: broadcast local bs
        # (cuda graph capture path); unset + idle: clear so downstream
        # collective sizing falls back to the non-DP path instead of using
        # stale capture totals.
        if ctx.global_bs is not None:
            return ctx.global_bs
        if ctx.global_num_tokens is None:
            return None
        if gather_ids is not None:
            return [gather_ids.size(0)] * len(ctx.global_num_tokens)
        return None
