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
    """Whether the flashinfer deterministic top-k path is importable."""
    return top_k is not None and TopKTieBreak is not None


def deterministic_decode_topk(
    logits: torch.Tensor,
    out: torch.Tensor,
    topk: int,
) -> None:
    """Select per-row top-``topk`` local offsets deterministically.

    ``logits`` rows are pre-masked with ``-inf`` beyond each request's valid
    length (so a global per-row top-``topk`` yields the in-sequence candidates).
    Writes int32 local offsets into ``out`` in-place. Uses a stable
    ``tie_break=SMALL`` (smallest index wins ties) and ``deterministic`` +
    ``dsa_graph_safe`` so eager and CUDA-graph replay agree.
    """
    if not has_deterministic_decode_topk():
        raise RuntimeError("flashinfer deterministic top_k is unavailable.")
    _values, indices = top_k(
        logits.contiguous(),
        int(topk),
        deterministic=True,
        tie_break=TopKTieBreak.SMALL,
        dsa_graph_safe=True,
    )
    out.copy_(indices.to(torch.int32))
