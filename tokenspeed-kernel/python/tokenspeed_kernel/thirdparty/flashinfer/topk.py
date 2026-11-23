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

"""Optional exact-membership TopK for the SM120 Full indexer."""

from __future__ import annotations

import inspect

import torch


def has_exact_topk() -> bool:
    """Check the installed public API for tie-breaking and graph-safe selection."""
    try:
        from flashinfer import top_k
    except ImportError:
        return False
    parameters = inspect.signature(top_k).parameters
    return "tie_break" in parameters and "dsa_graph_safe" in parameters


def exact_topk(scores: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Select largest scores with smaller-ID boundary ties, without output ordering.

    The caller restores its canonical ID order. Graph-safe mode admits CUB on
    SM120, where FilteredTopK's shared-memory requirement does not fit.
    """
    from flashinfer import top_k

    return top_k(
        scores, k, sorted=False, deterministic=False, tie_break=1, dsa_graph_safe=True
    )
