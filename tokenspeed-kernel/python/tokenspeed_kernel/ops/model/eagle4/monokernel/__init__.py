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

"""The EAGLE-4 drafter's decode monokernel: its draft layer and final norm
as one launch per chain draft step (TP1).

See ``docs/design/monokernel.md`` for the design and ``README.md`` in this
package for the op. The runtime packs the layer once after weight loading
(:func:`prepare_eagle4_draft_layer`) and runs it per draft step
(:func:`eagle4_draft_layer_decode`). The prepare function returns None when
the platform has no monokernel for the layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from tokenspeed_kernel.selection import NoKernelFoundError, select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

__all__ = [
    "FusedDecodeLayer",
    "eagle4_draft_layer_decode",
    "prepare_eagle4_draft_layer",
]


@dataclass(frozen=True)
class FusedDecodeLayer:
    """The draft layer packed for its monokernel.

    Attributes:
        decode: The selected kernel. :func:`eagle4_draft_layer_decode` calls
            it.
        weights: The kernel's packed weights (opaque to the caller).
        scratch_pairs: int64 mailbox pairs the kernel needs at the packed
            ``max_rows``.
    """

    decode: Any
    weights: Any
    scratch_pairs: int


def _select(mode: str, rows: int, solution: str | None):
    signature = format_signature(h=dense_tensor_format(torch.bfloat16))
    return select_kernel(
        "monokernel", mode, signature, traits={"rows": rows}, solution=solution
    )


def prepare_eagle4_draft_layer(
    *, tensors: dict[str, torch.Tensor], max_rows: int, solution: str | None = None
) -> FusedDecodeLayer | None:
    """Pack the EAGLE-4 draft layer (and its final norm) for its monokernel.

    Args:
        tensors: BF16 weights by role: ``input_norm``, ``hidden_norm``,
            ``post_norm``, ``final_norm`` ``[hidden]``; ``qkv`` (q, k, v
            rows of ``norm(embeds) || norm(hidden)``); ``o_proj``;
            ``gate_up`` (gate rows, then up rows); ``down``.
        max_rows: The most rows (requests) of one draft step. The scratch is
            sized for it; every launch of 1 to ``max_rows`` rows uses it.
        solution: Optional registered solution (such as ``"gluon"``) to
            force; its kernels must serve ``max_rows`` here, else this
            raises ``NoKernelFoundError``.

    Returns:
        The packed layer, or None when no monokernel serves ``max_rows``
        here.

    Raises:
        ValueError: A tensor's dtype or shape is not the kernel's.
    """
    try:
        prepare = _select("eagle4_draft_layer_prepare", max_rows, solution)
        decode = _select("eagle4_draft_layer_decode", max_rows, solution)
    except NoKernelFoundError:
        if solution is not None:
            raise
        return None
    weights, scratch_pairs = prepare(tensors=tensors, max_rows=max_rows)
    return FusedDecodeLayer(decode=decode, weights=weights, scratch_pairs=scratch_pairs)


def eagle4_draft_layer_decode(
    layer: FusedDecodeLayer,
    *,
    embeds: torch.Tensor,
    hidden: torch.Tensor,
    normed: torch.Tensor,
    positions: torch.Tensor,
    seq_len: torch.Tensor,
    slots: torch.Tensor,
    page_table: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cos_sin: torch.Tensor,
    scratch: torch.Tensor,
    step: torch.Tensor,
    layer_index: int,
    compile_only: bool,
) -> None:
    """One chain draft step of the packed layer and the final norm: one row
    per request.

    Args:
        layer: From :func:`prepare_eagle4_draft_layer`.
        embeds: BF16 ``[rows, hidden]``: the draft tokens' embeddings.
        hidden: BF16 ``[rows, hidden]``: the hidden state the step drafts
            from, also the layer's residual.
        normed: BF16 ``[rows, hidden]``: the final norm's output (aliases no
            input).
        positions: int32 ``[rows]``: the rows' positions (RoPE).
        seq_len: int32 ``[rows]``: each request's length, the row's own
            token included.
        slots: int32 or int64 ``[rows]``: the rows' KV cache slots.
        page_table: int32 ``[>= rows, pages]``: each request's page table
            (64 tokens a page, absolute positions).
        k_cache, v_cache: The draft layer's BF16 ``[slots, heads, 128]``
            caches.
        cos_sin: FP32 ``[positions, 128]`` rotary table, cos then sin.
        scratch: int64 mailbox scratch of at least ``layer.scratch_pairs``,
            zero before its first launch. Nothing clears it.
        step: int32 ``[1]`` counter that changes once per draft forward on
            device (CUDA graph safe).
        layer_index: The launch's layer slot. Launches of one ``step`` need
            distinct slots.
        compile_only: Only compile the launch for these operands.
    """
    layer.decode(
        weights=layer.weights,
        embeds=embeds,
        hidden=hidden,
        normed=normed,
        positions=positions,
        seq_len=seq_len,
        slots=slots,
        page_table=page_table,
        k_cache=k_cache,
        v_cache=v_cache,
        cos_sin=cos_sin,
        scratch=scratch,
        step=step,
        layer_index=layer_index,
        compile_only=compile_only,
    )


import tokenspeed_kernel.ops.model.eagle4.monokernel.gluon  # noqa: E402,F401
