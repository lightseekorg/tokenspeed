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

"""PyTorch references for ``tokenspeed_kernel.ops.kvcache``."""

from __future__ import annotations

import torch
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures


@register_kernel(
    "kvcache",
    "virtual_slots_to_local",
    name="torch_virtual_slots_to_local",
    solution="torch",
    signatures=format_signatures("slots", "dense", {torch.int32, torch.int64}),
    priority=Priority.REFERENCE,
)
def torch_virtual_slots_to_local(
    slots: torch.Tensor,
    *,
    rows_per_page: int,
    virtual_block_count: int,
    degree: int,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Translate virtual cache slots to this owner's local slots.

    Virtual block ``b >= 1`` belongs to rank ``(b - 1) % degree`` and lands in
    local block ``(b - 1) // degree + 1``; block 0, negative slots, slots past
    ``virtual_block_count`` blocks and other ranks' blocks are not owned.

    Returns:
        ``(local, owned)``: local slots, 0 where not owned, and the mask.
    """
    block = slots.clamp_min(0) // rows_per_page
    owned = (
        (slots >= rows_per_page)
        & (slots < virtual_block_count * rows_per_page)
        & ((block - 1) % degree == rank)
    )
    local = ((block - 1) // degree + 1) * rows_per_page + slots.clamp_min(
        0
    ) % rows_per_page
    return torch.where(owned, local, torch.zeros_like(local)), owned


@register_kernel(
    "kvcache",
    "store_sf_interleaved",
    name="torch_store_sf_interleaved",
    solution="torch",
    signatures=format_signatures("sf_in", "dense", {torch.uint8}),
    priority=Priority.REFERENCE,
)
def torch_store_sf_interleaved(
    sf_in: torch.Tensor, sf_out: torch.Tensor, loc: torch.Tensor, page_size: int
) -> None:
    """Scatter each token's per-head 4-byte MXFP8 scale word into its page.

    Token slot ``s`` lands in page ``s // page_size`` at word ``(off % 32) * 4
    + off // 32`` (``off = s % page_size``) of each head's 128-word block;
    ``sf_out`` is updated in place.

    Args:
        sf_in: ``[tokens, heads, 4]`` scale bytes (any 1-byte dtype).
        sf_out: Paged scale storage, ``heads * 128`` words per page.
        loc: ``[tokens]`` cache slots.
    """
    heads = sf_in.shape[1]
    out_words = sf_out.view(torch.uint8).reshape(-1, 4).view(torch.int32).reshape(-1)
    in_words = (
        sf_in.view(torch.uint8)
        .reshape(sf_in.shape[0], heads, 4)
        .contiguous()
        .view(torch.int32)
        .reshape(sf_in.shape[0], heads)
    )
    page_stride = heads * 128
    for token in range(sf_in.shape[0]):
        page, off = divmod(int(loc[token]), page_size)
        pos = (off % 32) * 4 + (off // 32)
        for head in range(heads):
            out_words[page * page_stride + head * 128 + pos] = in_words[token, head]
