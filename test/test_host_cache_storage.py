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

from __future__ import annotations

import ctypes
import dataclasses
import gc
import weakref

import pytest
from tokenspeed_kernel.platform import ArchVersion, Platform, PlatformInfo

from tokenspeed.runtime.cache.host_storage import HostCacheStorage
from tokenspeed.runtime.cache.transfer.layout import (
    CacheField,
    CacheGroupLayout,
    CacheTransferLayout,
)


@pytest.fixture(autouse=True)
def pinned(monkeypatch) -> dict[int, int]:
    """The page-locked Host allocations not yet freed, address to byte count.

    The platform allocates ordinary Host memory and records it here instead
    of calling the GPU runtime, which these tests do not need.
    """
    sizes: dict[int, int] = {}
    # Keeps each allocation's memory until it is freed.
    memories: dict[int, ctypes.Array] = {}

    class GpuFreePlatform(PlatformInfo):
        def _alloc_pinned_host_memory(self, nbytes):
            memory = ctypes.create_string_buffer(nbytes)
            address = ctypes.addressof(memory)
            sizes[address], memories[address] = nbytes, memory

            def free():
                del sizes[address], memories[address]

            return address, free

    platform = GpuFreePlatform(
        vendor="nvidia",
        arch_version=ArchVersion(10, 0),
        device_name="no GPU",
        device_count=0,
        total_memory=0,
        memory_bandwidth=0.0,
        sm_count=0,
        max_threads_per_sm=0,
        max_shared_memory_per_sm=0,
    )
    monkeypatch.setattr(Platform, "_instance", platform)
    return sizes


def _layout():
    field = lambda name, size: CacheField(name, 0, 0, size, size)
    return CacheTransferLayout(
        num_lcm_blocks=2,
        groups=(
            CacheGroupLayout("full", 4, (field("k", 32),)),
            CacheGroupLayout("state", 1, (field("state", 80),)),
        ),
        buffers=(object(),),
        consumers=(("k",), ("state",)),
    )


def test_block_offsets_follow_group_packing_without_overlap():
    layout = _layout()
    storage = HostCacheStorage(layout, num_host_lcm_blocks=2)

    assert storage.host_block_offset(0, 1) == 0
    assert storage.host_block_offset(0, 4) == 3 * storage.host_cache_block_bytes[0]
    assert storage.host_block_offset(0, 5) == storage.host_lcm_block_bytes
    assert storage.host_block_offset(1, 1) == 0
    assert storage.host_block_offset(1, 2) == storage.host_lcm_block_bytes


@pytest.mark.parametrize("group_index,block_id", [(0, 0), (0, 9), (1, 3)])
def test_null_and_out_of_range_blocks_are_rejected(group_index, block_id):
    layout = _layout()
    storage = HostCacheStorage(layout, num_host_lcm_blocks=2)

    with pytest.raises(IndexError):
        storage.host_block_offset(group_index, block_id)


def test_host_fields_are_packed_without_padding():
    field = lambda name, stride, payload: CacheField(name, 0, 0, stride, payload)
    layout = CacheTransferLayout(
        num_lcm_blocks=10,
        groups=(
            CacheGroupLayout(
                "full",
                16,
                (
                    field("full.k", 96, 40),
                    field("full.v", 96, 40),
                ),
            ),
            CacheGroupLayout(
                "state",
                1,
                (
                    field("state.ssm", 2048, 1000),
                    field("state.conv", 1024, 200),
                ),
            ),
        ),
        buffers=(object(),),
        consumers=(("full.k", "full.v"), ("state.ssm", "state.conv")),
    )

    storage = HostCacheStorage(layout, num_host_lcm_blocks=1)

    assert storage.host_cache_block_bytes == (80, 1200)
    assert storage.host_field_offsets == ((0, 40), (0, 1000))
    assert storage.host_lcm_block_bytes == 1280


def test_host_storage_does_not_retain_device_buffers():
    class DeviceBuffer:
        pass

    buffer = DeviceBuffer()
    reference = weakref.ref(buffer)
    layout = dataclasses.replace(_layout(), buffers=(buffer,))
    storage = HostCacheStorage(layout, num_host_lcm_blocks=2)
    del layout, buffer
    gc.collect()
    assert reference() is None
    assert storage.host_block_range(0, 1) == (0, 32)
    assert len(storage.host_buffer) == 256


def test_host_buffer_is_pinned_at_exactly_its_size(pinned):
    # Three 128-byte LCM blocks pin 384 bytes, not the 512 PyTorch's pinned
    # allocator would round them up to.
    storage = HostCacheStorage(_layout(), num_host_lcm_blocks=3)

    assert storage.host_buffer_bytes == 384
    assert storage.host_buffer.numel() == 384
    assert pinned == {storage.host_buffer.data_ptr(): 384}


def test_releasing_the_storage_frees_its_buffer(pinned):
    storage = HostCacheStorage(_layout(), num_host_lcm_blocks=3)
    other = HostCacheStorage(_layout(), num_host_lcm_blocks=1)

    del storage
    gc.collect()

    assert pinned == {other.host_buffer.data_ptr(): 128}
