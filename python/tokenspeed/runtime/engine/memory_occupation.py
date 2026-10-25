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

"""GPU memory a sleeping engine releases (``release_memory_occupation``)
and restores on wake, one :class:`MemoryRegion` per tag.

The Rust control plane drains the engine and orders the regions (the KV
cache first, NCCL last; resume in reverse); the data plane frees and
restores each one. Each tag is released and restored independently, so the
RL loop can resume weights (and NCCL), push fresh weights, then resume the KV
cache:

- ``weights``: model parameters (restored from their CPU backup);
- ``kv_cache``: the cache arena (discarded; zeroed on resume);
- ``cuda_graph``: the decode CUDA graph memory pool (discarded; every replay
  rewrites it before reading);
- ``nccl``: NCCL communicator buffers (suspended and resumed in place); only
  on an engine whose NCCL can suspend a communicator.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryRegion:
    """How one releasable tag frees and restores its GPU memory.

    Attributes:
        release: Frees the memory; runs once the scheduler has drained.
        resume: Restores the memory and whatever state must accompany it
            (zeroed KV, transfer-engine registration) before scheduling
            resumes.
    """

    release: Callable[[], None]
    resume: Callable[[], None]


def memory_regions(
    device, adapter, *, nccl_suspendable: bool
) -> dict[str, MemoryRegion]:
    """The releasable GPU-memory regions of an engine, keyed by tag.

    The memory-saver ``adapter`` unmaps and re-maps the allocator regions;
    the device handle does the data-plane work that must surround them.
    Which pools and peers that work touches is the handle's business; this
    side only says when. ``nccl_suspendable`` says whether the process's
    NCCL can suspend communicators; without it there is no ``nccl`` region.
    """

    def release_kv_cache() -> None:
        # Wait for in-flight copies and deregister the arena from the PD
        # transfer engine before its memory goes away.
        device.prepare_kv_release()
        adapter.pause(tag="kv_cache")

    def resume_kv_cache() -> None:
        adapter.resume(tag="kv_cache")
        # Re-mapped memory holds garbage: zero it, then register it with
        # the PD transfer engine again.
        device.finish_kv_resume()

    def allocator_region(tag: str) -> MemoryRegion:
        def release() -> None:
            device.synchronize()
            adapter.pause(tag=tag)

        return MemoryRegion(release=release, resume=lambda: adapter.resume(tag=tag))

    regions = {
        "kv_cache": MemoryRegion(release=release_kv_cache, resume=resume_kv_cache),
        "weights": allocator_region("weights"),
        "cuda_graph": allocator_region("cuda_graph"),
    }
    if nccl_suspendable:
        regions["nccl"] = MemoryRegion(
            release=device.suspend_nccl, resume=device.resume_nccl
        )
    return regions
