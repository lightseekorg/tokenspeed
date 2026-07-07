from multiprocessing import shared_memory

import pytest
import torch

from tokenspeed.runtime.multimodal.shm_transport import ShmTensorHandle


def test_copy_to_pinned_retains_shm_until_release():
    source = torch.tensor([1.0, 2.0], dtype=torch.float32)
    handle = ShmTensorHandle.publish(source)
    handle.attach()

    try:
        copied = handle.copy_to_pinned()
        still_open = shared_memory.SharedMemory(name=handle.shm_name)
        still_open.close()

        assert copied.is_pinned()
        assert torch.equal(copied, source)
    finally:
        handle.release()

    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=handle.shm_name)


def test_consume_still_releases_shm():
    source = torch.tensor([3.0, 4.0], dtype=torch.float32)
    handle = ShmTensorHandle.publish(source)
    handle.attach()

    try:
        copied = handle.consume()

        assert copied.is_pinned()
        assert torch.equal(copied, source)
    finally:
        handle.release()

    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=handle.shm_name)


def test_copy_into_reshapes_byte_exactly_and_releases_shm():
    source = torch.arange(12, dtype=torch.int32).reshape(3, 4)
    handle = ShmTensorHandle.publish(source)
    handle.attach()
    destination = torch.empty(source.numel(), dtype=source.dtype)

    handle.copy_into(destination)

    assert torch.equal(destination.reshape(source.shape), source)
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=handle.shm_name)


def test_copy_into_rejects_dtype_conversion_and_releases_shm():
    source = torch.arange(4, dtype=torch.int32)
    handle = ShmTensorHandle.publish(source)
    handle.attach()

    with pytest.raises(ValueError, match="dtypes differ"):
        handle.copy_into(torch.empty(4, dtype=torch.float32))

    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=handle.shm_name)
