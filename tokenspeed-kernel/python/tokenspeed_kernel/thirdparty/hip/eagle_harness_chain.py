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

"""EAGLE-Harness chained rounds (EAGLE-4), HIP build; see
``csrc/eagle_harness_chain.hip``. Each op has the contract of the CUDA
build's ``tokenspeed_kernel.thirdparty.cuda.eagle_harness_chain`` op of the
same name.

HIP has no device-side graph launch: the kernel inside each depth's graph
chooses the round's next graph on the GPU, as the CUDA build's does, and posts
it to host memory, where a native launcher thread (per device, not holding
the GIL) launches it on the stream that launched the round's depth 1. Graph
and exec handles are the raw ``hipGraph_t`` / ``hipGraphExec_t`` pointers as
integers, streams the raw ``hipStream_t`` (``cuda_stream``)."""

import functools
from pathlib import Path

import torch

__all__ = [
    "eagle_harness_chain_begin",
    "eagle_harness_chain_destroy",
    "eagle_harness_chain_instantiate",
    "eagle_harness_chain_join",
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
            "Run: pip install -e tokenspeed-kernel/python/ (ROCm backend)"
        )
    return tvm_ffi.load_module(str(so_path))


def eagle_harness_chain_load() -> None:
    """HIP build of the CUDA op of the same name; also starts the current
    device's launcher thread and allocates its mailbox (no allocation may run
    inside a graph capture)."""
    _module().eagle_harness_chain_init(torch.cuda.current_device())


def eagle_harness_chain_instantiate(graph: int, stream: int) -> int:
    """HIP build of the CUDA op of the same name: an ordinary instantiation
    (the launcher thread launches it from the host), uploaded on ``stream``."""
    return _module().eagle_harness_chain_instantiate(graph, stream)


def eagle_harness_chain_destroy(exec_handle: int) -> None:
    """HIP build of the CUDA op of the same name."""
    _module().eagle_harness_chain_destroy(exec_handle)


def eagle_harness_chain_next(
    execs: torch.Tensor,
    go: torch.Tensor,
    sampled: torch.Tensor,
    depth: int,
    last_depth: int,
) -> None:
    """HIP build of the CUDA op of the same name: the kernel posts the chosen
    exec to the launcher thread, which queues it behind the posting graph on
    the round's stream (``eagle_harness_chain_begin``), once the call that
    launched the posting graph has returned. A zero handle where the round
    goes traps."""
    _module().eagle_harness_chain_next(execs, go, sampled, depth, last_depth)


def eagle_harness_chain_begin(stream: int) -> None:
    """HIP build of the CUDA op of the same name: hand the round to the
    launcher thread, which launches its posted graphs on ``stream``; raises if
    the previous round was not joined, or failed."""
    _module().eagle_harness_chain_begin(stream)


def eagle_harness_chain_join(stream: int) -> None:
    """HIP build of the CUDA op of the same name: block, without the GIL,
    until the launcher thread has queued the round's finalize on ``stream``
    (the last decision lands while the depth before it still runs); raises if
    a launch failed or no graph was posted for 60 s."""
    _module().eagle_harness_chain_join(stream)
