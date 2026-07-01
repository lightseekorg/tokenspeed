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

"""Deterministic DSA decode indexer top-k via flashinfer.

The trtllm ``indexer_topk_decode`` kernel breaks ties (equal logits competing for
the last selected slot) non-deterministically: repeated runs select *different*
index sets, which makes long-context greedy decode irreproducible and breaks
eager-vs-CUDA-graph parity. flashinfer's radix top-k exposes a stable,
index-ordered tie-break plus a graph-safe path, so the selection is identical
across eager, repeated runs, and CUDA-graph replay -- with zero accuracy loss
(it still selects the mathematically-correct top-k set, only the tie-break and
output order become deterministic).
"""

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.attention.cuda.deepseek_v4 import (
    has_persistent_topk,
    persistent_topk,
)
from tokenspeed_kernel.platform import current_platform

platform = current_platform()

top_k = None
TopKTieBreak = None

if platform.is_nvidia:
    try:
        from flashinfer import TopKTieBreak, top_k
    except ImportError:
        pass


def has_deterministic_decode_topk() -> bool:
    """Whether the flashinfer deterministic top-k fallback is importable."""
    return top_k is not None and TopKTieBreak is not None


def has_ragged_decode_topk() -> bool:
    """Whether the length-aware CUDA top-k path is importable."""
    return has_persistent_topk()


def deterministic_decode_topk(
    logits: torch.Tensor,
    out: torch.Tensor,
    topk: int,
    *,
    lengths: torch.Tensor | None = None,
    workspace: torch.Tensor | None = None,
    max_seq_len: int | None = None,
) -> None:
    """Select per-row top-``topk`` local offsets deterministically.

    When ``lengths`` and ``workspace`` are provided, use the ragged CUDA top-k
    path so padded logits beyond each row's valid context are not scanned. This
    path delegates tie handling to the persistent top-k kernel. Otherwise,
    ``logits`` rows must already be pre-masked with ``-inf`` beyond each
    request's valid length, and the flashinfer fallback uses a stable
    ``tie_break=SMALL`` plus ``deterministic`` + ``dsa_graph_safe``.
    """
    if lengths is not None or workspace is not None:
        if lengths is None or workspace is None or not has_ragged_decode_topk():
            raise RuntimeError("length-aware DSA decode top-k is unavailable.")
        persistent_topk(
            logits.contiguous(),
            lengths.to(torch.int32).contiguous().reshape(-1),
            out,
            workspace,
            int(topk),
            int(max_seq_len or logits.shape[1]),
        )
        return

    if top_k is None or TopKTieBreak is None:
        raise RuntimeError("flashinfer deterministic top_k is unavailable.")
    _values, indices = top_k(
        logits.contiguous(),
        int(topk),
        deterministic=True,
        tie_break=TopKTieBreak.SMALL,
        dsa_graph_safe=True,
    )
    out.copy_(indices.to(torch.int32))
