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

"""Static CUDA-graph capability declarations shared by every backend node."""

from __future__ import annotations

from dataclasses import dataclass

from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)


@dataclass(frozen=True)
class CudaGraphSupport:
    """Per-backend-class CUDA-graph capability declaration.

    Rank-uniform by construction: declarations are class attributes, resolved
    identically on every rank at startup (event-loop.md requires graph
    decisions to derive from replicated state). ``decode_graph=False``
    disables capture/replay of the whole-step decode graph — the unified
    refresh still serves eager decode, so ``init_cuda_graph_state`` and
    ``refresh_decode_metadata`` stay mandatory. ``prefill_graph=False``
    disables the breakable prefill (extend) graph. Static "never works"
    declarations only — and they are the ONLY escape: a prefill capture
    failure at runtime is fatal, so a family that cannot capture must say
    so here rather than rely on a degrade path.
    """

    decode_graph: bool = True
    prefill_graph: bool = True

    def __and__(self, other: CudaGraphSupport) -> CudaGraphSupport:
        return CudaGraphSupport(
            decode_graph=self.decode_graph and other.decode_graph,
            prefill_graph=self.prefill_graph and other.prefill_graph,
        )


def resolve_cuda_graph_support(*backends) -> CudaGraphSupport:
    """AND-compose ``cuda_graph_support`` over ``backends`` and their
    ``child_backends()`` trees, logging every backend class that lowers an
    axis.

    Args:
        backends: Root attention backends; ``None`` entries are skipped. Pass
            the target AND the draft — the decode graph records the whole
            step, drafter loop included.

    Returns:
        The composed support: an axis is False iff any backend in any tree
        declares it False.
    """
    resolved = CudaGraphSupport()
    stack = [backend for backend in backends if backend is not None]
    while stack:
        backend = stack.pop()
        declared = backend.cuda_graph_support
        if not declared.decode_graph:
            logger.info("Decode CUDA graphs disabled by %s", type(backend).__name__)
        if not declared.prefill_graph:
            logger.info("Prefill CUDA graphs disabled by %s", type(backend).__name__)
        resolved = resolved & declared
        stack.extend(backend.child_backends())
    return resolved
