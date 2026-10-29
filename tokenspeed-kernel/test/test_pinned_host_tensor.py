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

"""Host buffers pinned at exactly their size (``empty_pinned_host_tensor``)."""

from __future__ import annotations

import ctypes
import gc

import pytest
import torch
from tokenspeed_kernel.platform import ArchVersion, PlatformInfo, current_platform

# Not a power of two: PyTorch's pinned allocator would round it up to 16384.
NBYTES = 3 * 4096 + 5

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a GPU to page-lock Host memory"
)


def _platform(vendor: str, cls: type[PlatformInfo] = PlatformInfo) -> PlatformInfo:
    """A platform of ``vendor`` without a device."""
    return cls(
        vendor=vendor,
        arch_version=ArchVersion(0, 0),
        device_name="no GPU",
        device_count=0,
        total_memory=0,
        memory_bandwidth=0.0,
        sm_count=0,
        max_threads_per_sm=0,
        max_shared_memory_per_sm=0,
    )


class _PinnedAllocations:
    """The Host memory a platform page-locked, in place of a GPU runtime."""

    def __init__(self) -> None:
        # Address to the memory of every allocation not yet freed.
        self.live: dict[int, ctypes.Array] = {}
        # Each free's address and the last byte of its memory at that point.
        self.freed: list[tuple[int, int]] = []

    def alloc(self, nbytes: int):
        memory = ctypes.create_string_buffer(nbytes)
        address = ctypes.addressof(memory)
        self.live[address] = memory

        def free() -> None:
            self.freed.append((address, self.live.pop(address).raw[-1]))

        return address, free


@pytest.fixture(params=["nvidia", "amd"])
def gpu_free(request) -> tuple[PlatformInfo, _PinnedAllocations]:
    """A platform of each GPU vendor whose pinned allocations are recorded."""
    allocations = _PinnedAllocations()

    class GpuFreePlatform(PlatformInfo):
        def _alloc_pinned_host_memory(self, nbytes):
            return allocations.alloc(nbytes)

    return _platform(request.param, GpuFreePlatform), allocations


def test_a_buffer_is_allocated_at_exactly_its_size(gpu_free):
    platform, allocations = gpu_free

    buffer = platform.empty_pinned_host_tensor(NBYTES)

    assert buffer.dtype == torch.uint8 and buffer.device.type == "cpu"
    assert buffer.numel() == NBYTES
    assert {address: len(memory) for address, memory in allocations.live.items()} == {
        buffer.data_ptr(): NBYTES
    }
    assert platform.pinned_host_bytes(NBYTES) == NBYTES


def test_the_last_view_frees_the_buffer_once(gpu_free):
    platform, allocations = gpu_free
    buffer = platform.empty_pinned_host_tensor(NBYTES)
    address = buffer.data_ptr()
    buffer[-1] = 0x5A
    tail = buffer[-16:]
    as_numpy = buffer.numpy()

    del buffer, tail
    gc.collect()
    # A view keeps the memory alive.
    assert list(allocations.live) == [address]
    assert allocations.freed == []

    del as_numpy
    gc.collect()
    # Freed once, and the buffer's bytes were the allocation's.
    assert allocations.live == {}
    assert allocations.freed == [(address, 0x5A)]


def test_an_empty_buffer_allocates_nothing(gpu_free):
    platform, allocations = gpu_free

    buffer = platform.empty_pinned_host_tensor(0)

    assert buffer.numel() == 0
    assert allocations.live == {}


def test_a_negative_size_is_refused(gpu_free):
    platform, _ = gpu_free

    with pytest.raises(ValueError, match="nbytes"):
        platform.empty_pinned_host_tensor(-1)


@pytest.mark.parametrize("vendor", ["nvidia", "amd"])
def test_a_gpu_buffer_counts_at_its_size(vendor):
    platform = _platform(vendor)

    assert platform.pinned_host_bytes(0) == 0
    assert platform.pinned_host_bytes(2**37 + 1) == 2**37 + 1
    assert platform.largest_pinned_host_fit(-1) == 0
    assert platform.largest_pinned_host_fit(2**38 - 1) == 2**38 - 1


def test_an_npu_buffer_counts_at_the_power_of_two_torch_npu_may_round_to():
    platform = _platform("ascend")

    assert platform.pinned_host_bytes(0) == 0
    assert platform.pinned_host_bytes(2**37) == 2**37
    assert platform.pinned_host_bytes(2**37 + 1) == 2**38
    assert platform.largest_pinned_host_fit(-1) == 0
    assert platform.largest_pinned_host_fit(0) == 0
    assert platform.largest_pinned_host_fit(2**38 - 1) == 2**37
    assert platform.largest_pinned_host_fit(2**38) == 2**38
    for room in (1, 3, 2**37 - 1, 228_260 * 10**6):
        fit = platform.largest_pinned_host_fit(room)
        assert platform.pinned_host_bytes(fit) <= room
        assert platform.pinned_host_bytes(fit + 1) > room


def _active_pinned_allocator_bytes() -> int:
    """The bytes PyTorch's caching pinned allocator has handed out."""
    gc.collect()
    torch.cuda.synchronize()
    return torch.cuda.host_memory_stats().get("active_bytes.current", 0)


@requires_gpu
def test_the_buffer_takes_asynchronous_copies_both_ways():
    torch.cuda.init()
    active = _active_pinned_allocator_bytes()

    host = current_platform().empty_pinned_host_tensor(NBYTES)

    # Page-locked, and not by PyTorch's pinned allocator.
    assert host.is_pinned()
    assert _active_pinned_allocator_bytes() == active
    source = torch.randint(0, 256, (NBYTES,), dtype=torch.uint8, device="cuda")
    destination = torch.zeros_like(source)

    host.copy_(source, non_blocking=True)
    torch.cuda.synchronize()
    assert torch.equal(host, source.cpu())

    destination.copy_(host, non_blocking=True)
    torch.cuda.synchronize()
    assert torch.equal(destination, source)


@requires_gpu
def test_a_released_buffer_is_freed_once_by_the_gpu_runtime():
    # Each free's address and the error it raised, if any: a finalizer's
    # error would otherwise only be printed.
    frees: list[tuple[int, BaseException | None]] = []

    class RecordingPlatform(PlatformInfo):
        def _alloc_pinned_host_memory(self, nbytes):
            address, free = super()._alloc_pinned_host_memory(nbytes)

            def recorded_free() -> None:
                try:
                    free()
                except BaseException as error:
                    frees.append((address, error))
                    raise
                frees.append((address, None))

            return address, recorded_free

    # The detected platform's fields, with the recording subclass.
    platform = RecordingPlatform(**vars(current_platform()))
    for round_index in range(3):
        buffer = platform.empty_pinned_host_tensor(NBYTES)
        address = buffer.data_ptr()
        assert buffer.is_pinned()

        del buffer
        gc.collect()

        assert frees[round_index:] == [(address, None)]
