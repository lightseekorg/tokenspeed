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

"""QSA indexing layout over its own raw tables and full-KV address views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from tokenspeed_kernel.ops.attention.qsa.triton import (
    qwen4_exp_qsa_prepare_metadata,
)

from tokenspeed.runtime.layers.attention.backends.specific.qwen4_exp import (
    qwen4_exp_backend,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import FULL_ATTENTION

if TYPE_CHECKING:
    from tokenspeed.runtime.execution.context import ForwardContext


def decode_query_lengths(
    ctx: ForwardContext, total_tokens: int, *, force_uniform: bool
) -> int | None:
    """Return the uniform decode query width, or None for a ragged extend.

    Args:
        ctx: The live model forward context.
        total_tokens: Query rows, after any draft step-0 narrowing.
        force_uniform: Whether a narrowed draft requires uniform rows even
            when its original context still names an extend forward.

    Returns:
        Query rows per request for uniform forwards, otherwise None.
    """
    if not ctx.bs or (
        not force_uniform
        and (ctx.forward_mode is None or not ctx.forward_mode.is_decode())
    ):
        return None
    if total_tokens % ctx.bs:
        raise RuntimeError("Qwen4-Exp QSA decode rows must be divisible by batch size")
    return total_tokens // ctx.bs


@dataclass(frozen=True)
class QSALayout:
    """Layer-invariant cache geometry for one QSA model forward."""

    seq_lens: torch.Tensor
    logical_positions: torch.Tensor
    request_indices: torch.Tensor
    qsa_locs: torch.Tensor
    recent_locs: torch.Tensor
    complete_blocks: torch.Tensor
    qsa_page_table: torch.Tensor
    full_page_table: torch.Tensor
    full_kernel_page_size: int
    reset_draft_tags: torch.Tensor | None


def qsa_forward_layout(
    ctx: ForwardContext,
    total_tokens: int,
    *,
    compressed_token_page_size: int,
    recent_page_size: int,
    compress_ratio: int,
    reset_draft_tags: torch.Tensor | None,
) -> QSALayout:
    """Build or reuse the QSA row layout shared by every local QSA layer."""

    cached = ctx.attn_backend.sparse_topk.qsa_metadata
    if cached is not None:
        if not isinstance(cached, QSALayout):
            raise RuntimeError("invalid QSA per-forward metadata memo")
        if (
            cached.logical_positions.shape[0] != total_tokens
            or cached.seq_lens.shape[0] < ctx.bs
        ):
            raise RuntimeError("stale QSA per-forward metadata memo")
        if (
            reset_draft_tags is not None
            and cached.reset_draft_tags is not reset_draft_tags
        ):
            reset_draft_tags.fill_(torch.iinfo(torch.int64).min)
        return cached

    backend = qwen4_exp_backend(ctx.attn_backend).indexer_backend
    if backend is None:
        raise RuntimeError("QSA requires an indexer backend")
    metadata = backend.metadata_for(ctx.forward_mode)
    query_lengths = decode_query_lengths(
        ctx,
        total_tokens,
        force_uniform=False,
    )
    if query_lengths is None:
        if metadata.extend_seq_lens is None:
            raise RuntimeError("QSA extend metadata requires query lengths")
        query_lengths = metadata.extend_seq_lens[: ctx.bs]
    full = backend.full_attn_backend.group_view(FULL_ATTENTION, ctx.bs)
    qsa_page_table = metadata.qsa_block_table
    recent_page_table = metadata.recent_block_table
    seq_lens = metadata.seq_lens[: ctx.bs]
    logical, requests, qsa_locs, recent_locs, complete_blocks = (
        qwen4_exp_qsa_prepare_metadata(
            seq_lens,
            query_lengths,
            total_tokens,
            qsa_page_table,
            compressed_token_page_size,
            recent_page_table,
            recent_page_size,
            compress_ratio,
            draft_logical_positions=reset_draft_tags,
        )
    )
    layout = QSALayout(
        seq_lens=seq_lens,
        logical_positions=logical,
        request_indices=requests,
        qsa_locs=qsa_locs,
        recent_locs=recent_locs,
        complete_blocks=complete_blocks,
        qsa_page_table=qsa_page_table,
        full_page_table=full.page_table,
        full_kernel_page_size=full.kernel_page_size,
        reset_draft_tags=reset_draft_tags,
    )
    ctx.attn_backend.sparse_topk.qsa_metadata = layout
    return layout
