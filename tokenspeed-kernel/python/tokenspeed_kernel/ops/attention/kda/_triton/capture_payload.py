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

"""Stage a layer's three replay payloads in one launch."""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton


@triton.jit(
    do_not_specialize=["rows", "n_a", "n_b", "n_c", "sa", "sb", "sc", "da", "db", "dc"]
)
def _capture_payload_kernel(
    src_a,
    dst_a,
    src_b,
    dst_b,
    src_c,
    dst_c,
    rows,
    n_a,
    n_b,
    n_c,
    sa,
    sb,
    sc,
    da,
    db,
    dc,
    BLOCK: tl.constexpr,
):
    which = tl.program_id(1)
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if which == 0:
        live = off < rows * n_a
        r, c = off // n_a, off % n_a
        v = tl.load(src_a + r * sa + c, mask=live, other=0)
        tl.store(dst_a + r * da + c, v, mask=live)
    elif which == 1:
        live = off < rows * n_b
        r, c = off // n_b, off % n_b
        v = tl.load(src_b + r * sb + c, mask=live, other=0)
        tl.store(dst_b + r * db + c, v, mask=live)
    else:
        live = off < rows * n_c
        r, c = off // n_c, off % n_c
        v = tl.load(src_c + r * sc + c, mask=live, other=0)
        tl.store(dst_c + r * dc + c, v, mask=live)


def capture_replay_payload(
    sources: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    destinations: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    rows: int,
) -> None:
    """Copy the leading ``rows`` of three projections into their payload rows.

    A speculative verify stages each layer's projections so the commit can
    replay them, which as three ``copy_`` calls costs three launches per layer
    on every decode step. The three are independent, so one launch with a
    payload-selecting grid dimension does the same work.

    Args:
        sources: Projection outputs; row ``i`` of each is copied. Rows may be a
            slice of a wider buffer, so only the inner dimension must be dense.
        destinations: Payload rows, each ``[>=rows, >=source width]``; a wider
            row keeps its trailing columns.
        rows: Live rows, ``batch_size * draft_tokens``.
    """
    widths = []
    for src, dst in zip(sources, destinations):
        if src.dim() != 2 or dst.dim() != 2:
            raise ValueError("payload capture takes 2-D sources and destinations")
        if src.dtype != dst.dtype:
            raise ValueError("payload capture cannot convert dtypes")
        if src.stride(1) != 1 or dst.stride(1) != 1:
            raise ValueError("payload capture needs a dense inner dimension")
        if src.shape[0] < rows:
            raise ValueError(f"source holds {src.shape[0]} rows, need {rows}")
        if dst.shape[0] < rows or dst.shape[1] < src.shape[1]:
            raise ValueError(
                f"payload row {tuple(dst.shape)} cannot hold "
                f"({rows}, {src.shape[1]})"
            )
        widths.append(src.shape[1])
    if rows <= 0:
        return

    block = 1024
    grid = (triton.cdiv(rows * max(widths), block), 3)
    _capture_payload_kernel[grid](
        sources[0],
        destinations[0],
        sources[1],
        destinations[1],
        sources[2],
        destinations[2],
        rows,
        widths[0],
        widths[1],
        widths[2],
        sources[0].stride(0),
        sources[1].stride(0),
        sources[2].stride(0),
        destinations[0].stride(0),
        destinations[1].stride(0),
        destinations[2].stride(0),
        BLOCK=block,
    )
