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

"""EAGLE-Harness device-chained rounds (EAGLE-4): CUDA graphs instantiated
for device launch, and the kernel inside each draft depth's graph that
tail-launches the round's next graph; see ``csrc/eagle_harness_chain.cu``.

Graph and exec handles are the raw ``cudaGraph_t`` / ``cudaGraphExec_t``
pointers as integers (``torch.cuda.CUDAGraph(keep_graph=True)``'s
``raw_cuda_graph()``), streams the raw ``cudaStream_t`` (``cuda_stream``)."""

import functools
from pathlib import Path

import torch

__all__ = [
    "eagle_harness_chain_destroy",
    "eagle_harness_chain_instantiate",
    "eagle_harness_chain_load",
    "eagle_harness_chain_next",
]


@functools.cache
def _module():
    import tvm_ffi

    so_path = (
        Path(__file__).parent
        / "objs"
        / "eagle_harness_chain"
        / "eagle_harness_chain.so"
    )
    if not so_path.exists():
        raise RuntimeError(
            f"tokenspeed_kernel eagle_harness_chain library not found at {so_path}. "
            "Run: pip install -e tokenspeed_kernel/python/"
        )
    return tvm_ffi.load_module(str(so_path))


def eagle_harness_chain_load() -> None:
    """Load the kernel library (no launch), before a graph capture uses it."""
    _module()


def eagle_harness_chain_instantiate(graph: int, stream: int) -> int:
    """Instantiate the captured ``graph`` for device launch and upload it on
    ``stream``; returns the exec handle (``eagle_harness_chain_destroy``
    frees it). The graph may hold kernel, memcpy (device or pinned memory)
    and memset nodes, fork/join edges included; no event, host, memory
    allocation or conditional nodes, and no dynamic parallelism beyond this
    module's own launch (CUDA's device graph requirements)."""
    return _module().eagle_harness_chain_instantiate(graph, stream)


def eagle_harness_chain_destroy(exec_handle: int) -> None:
    """Free an exec handle no queued or running chain still launches."""
    _module().eagle_harness_chain_destroy(exec_handle)


def eagle_harness_chain_next(
    execs: torch.Tensor,
    go: torch.Tensor,
    sampled: torch.Tensor,
    depth: int,
    last_depth: int,
) -> None:
    """Enqueue, inside depth ``depth``'s graph (``1 <= depth <= last_depth``)
    and after its decision, the kernel that tail-launches the round's next
    graph: depth ``depth + 1`` while ``depth < last_depth`` and ``go[depth +
    1]`` (int32 ``[>= last_depth + 1]``) is set, else the finalize graph
    ``sampled[0]`` (bool) selects.

    ``execs`` (int64 ``[last_depth + 3]``) holds the padded batch's exec
    handles: depth ``d`` at column ``d`` (from 2), then the finalize graphs,
    not sampled and sampled. The tail launch runs once the launching graph
    has completed; a zero handle where the round goes traps."""
    _module().eagle_harness_chain_next(execs, go, sampled, depth, last_depth)
