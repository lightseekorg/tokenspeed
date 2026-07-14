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

"""Length-aware (ragged) DSA decode indexer top-k via the CUDA persistent kernel.

Selects the per-row top-k over a dense indexer-logits matrix with the native CUDA
persistent-radix ``persistent_topk`` kernel: each row's valid context length is
read from ``lengths`` so padded columns beyond a request's context are never
scanned, and tie handling is delegated to the kernel. This is the ragged path
behind the DSA decode top-k selection, so callers pre-mask nothing.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.attention.cuda.deepseek_v4 import (
    has_persistent_topk,
    persistent_topk,
)

__all__ = [
    "has_ragged_decode_topk",
    "ragged_decode_topk",
]


def has_ragged_decode_topk() -> bool:
    """Whether the length-aware CUDA persistent-radix top-k path is importable."""
    return has_persistent_topk()


def ragged_decode_topk(
    logits: torch.Tensor,
    out: torch.Tensor,
    topk: int,
    *,
    lengths: torch.Tensor,
    workspace: torch.Tensor,
    max_seq_len: int | None = None,
    q_len_per_req: int = 1,
) -> None:
    """Select per-row top-``topk`` local offsets over ragged (length-aware) logits.

    Args:
        logits: Indexer logits ``[num_rows, num_cols]``; padded columns beyond
            each row's valid length are skipped in-kernel (no pre-masking).
        out: int32 output buffer ``[num_rows, topk]`` written in place with the
            selected column indices.
        topk: Number of candidates to select per row.
        lengths: Per-row valid context length, cast to int32 and flattened.
        workspace: Scratch uint8 buffer for the persistent kernel.
        max_seq_len: Max scanned columns; defaults to ``logits.shape[1]``.
        q_len_per_req: Query rows per request (speculative decode); 1 for decode.

    Raises:
        RuntimeError: if the CUDA persistent kernel is unavailable.
    """
    if not has_ragged_decode_topk():
        raise RuntimeError("length-aware DSA decode top-k is unavailable.")
    persistent_topk(
        logits.contiguous(),
        lengths.to(torch.int32).contiguous().reshape(-1),
        out,
        workspace,
        int(topk),
        int(max_seq_len or logits.shape[1]),
        int(q_len_per_req),
    )
