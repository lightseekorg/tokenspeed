# MIT License
#
# Copyright (c) 2026 LightSeek Foundation <contact@lightseek.org>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Return freed device memory to the driver without outrunning its release.

MoE weight preprocessing replaces each layer's checkpoint weights and returns
the freed blocks with ``torch.cuda.empty_cache()``, so the KV pool later sizes
itself from free memory. The amdgpu driver releases freed VRAM asynchronously,
and on some MI355X GPUs much more slowly than one layer frees it. The freed but
not yet released memory then piles up across layers until the device is full,
and the driver evicts the process's buffers and queues while its kernels run;
model loading has crashed there with an instruction fetch at address 0.
``release_cached_memory`` therefore waits after each release until the driver
has caught up.
"""

import time
from pathlib import Path

import torch

# How much freed VRAM the driver may still be holding when a release returns.
RELEASE_BACKLOG_SLACK_BYTES = 8 << 30
_RELEASE_POLL_SECONDS = 0.01
_RELEASE_TIMEOUT_SECONDS = 60.0

# Per device index: the lowest observed driver-accounted VRAM use outside the
# caching allocator (runtime, communication buffers and the like). Anything
# above it is memory the allocator returned that the driver has not released.
_non_allocator_vram: dict[int, int] = {}


def driver_vram_used(device: int) -> int | None:
    """Return the device's VRAM use as the driver accounts it.

    Unlike ``torch.cuda.mem_get_info``, this keeps counting freed buffers until
    the driver has released them.

    Args:
        device: CUDA (HIP) device index.

    Returns:
        Bytes in use, or ``None`` when the driver's sysfs counter is not
        readable (for example a container without ``/sys/bus/pci``).
    """
    props = torch.cuda.get_device_properties(device)
    path = Path(
        f"/sys/bus/pci/devices/{props.pci_domain_id:04x}:{props.pci_bus_id:02x}:"
        f"{props.pci_device_id:02x}.0/mem_info_vram_used"
    )
    try:
        return int(path.read_text())
    except (OSError, ValueError):
        return None


def release_cached_memory(device: torch.device) -> None:
    """Return the caching allocator's free blocks and wait for the driver to release them.

    Waits until the freed-but-unreleased backlog is at most
    ``RELEASE_BACKLOG_SLACK_BYTES``. Returns without waiting when the driver
    counter is unavailable. A wait that exceeds the timeout adopts the current
    use as the new baseline, so a lasting change in memory outside the caching
    allocator cannot stall later releases.

    Args:
        device: CUDA (HIP) device whose cached blocks to release.
    """
    index = torch.device(device).index
    if index is None:
        index = torch.cuda.current_device()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(index)
    reserved = torch.cuda.memory_reserved(index)
    deadline = time.monotonic() + _RELEASE_TIMEOUT_SECONDS
    while True:
        used = driver_vram_used(index)
        if used is None:
            return
        outside = used - reserved
        baseline = min(_non_allocator_vram.get(index, outside), outside)
        _non_allocator_vram[index] = baseline
        if outside - baseline <= RELEASE_BACKLOG_SLACK_BYTES:
            return
        if time.monotonic() > deadline:
            _non_allocator_vram[index] = outside
            return
        time.sleep(_RELEASE_POLL_SECONDS)
