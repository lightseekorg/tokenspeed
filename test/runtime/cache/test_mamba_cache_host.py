from __future__ import annotations

import torch

from tokenspeed.runtime.cache.mamba_cache_host import MambaPoolHost
from tokenspeed.runtime.cache.transfer.mamba_pool import MambaCachePool
from tokenspeed.runtime.layers.attention.backends.hybrid_linear_attn import (
    SimpleMambaPool,
)


def _new_mamba_pool(size: int = 4):
    return SimpleMambaPool(
        size=size,
        num_mamba_layers=2,
        conv_state_shape=(3,),
        temporal_state_shape=(2, 2),
        conv_dtype=torch.float32,
        ssm_dtype=torch.float32,
        mamba_layer_ids=[4, 7],
        device="cpu",
        page_size=1,
    )


def test_mamba_pool_host_alloc_free_tracks_slots():
    device_pool = _new_mamba_pool(size=3)
    host_pool = MambaPoolHost(
        device_pool, host_size_slots=2, pin_memory=False, register_host=False
    )

    first = host_pool.alloc(1)
    second = host_pool.alloc(1)
    assert first.tolist() == [0]
    assert second.tolist() == [1]
    assert host_pool.alloc(1) is None

    host_pool.free(first)
    assert host_pool.available_size() == 1
    assert host_pool.alloc(1).tolist() == [0]


class _SpyPlatform:
    def __init__(self):
        self.registered = []

    def register_host_tensor_for_gpu_access(self, tensor):
        self.registered.append(tensor)


def test_mamba_pool_host_registers_unpinned_cpu_buffers(monkeypatch):
    import tokenspeed.runtime.cache.mamba_cache_host as mamba_cache_host

    device_pool = _new_mamba_pool(size=2)
    platform = _SpyPlatform()
    monkeypatch.setattr(mamba_cache_host, "current_platform", lambda: platform)

    host_pool = MambaPoolHost(
        device_pool, host_size_slots=2, pin_memory=True, register_host=True
    )

    assert platform.registered == [host_pool.conv_buffer, host_pool.ssm_buffer]
    assert not host_pool.conv_buffer.is_pinned()
    assert not host_pool.ssm_buffer.is_pinned()


def test_mamba_pool_host_kernel_writeback_reuses_pointer_tables(monkeypatch):
    import tokenspeed.runtime.cache.mamba_cache_host as mamba_cache_host

    class PtrPlatform(_SpyPlatform):
        def device_visible_data_ptr(self, tensor):
            return tensor.data_ptr()

    calls = []

    def fake_transfer_kv_all_layer_mla(**kwargs):
        calls.append((kwargs["src_layers"], kwargs["dst_layers"]))

    monkeypatch.setattr(mamba_cache_host, "current_platform", lambda: PtrPlatform())
    monkeypatch.setattr(
        mamba_cache_host,
        "transfer_kv_all_layer_mla",
        fake_transfer_kv_all_layer_mla,
    )

    device_pool = _new_mamba_pool(size=4)
    host_pool = MambaPoolHost(
        device_pool, host_size_slots=4, pin_memory=False, register_host=False
    )
    device_indices = torch.tensor([0, 2], dtype=torch.int64)
    host_indices = torch.tensor([1, 3], dtype=torch.int64)

    for _ in range(2):
        host_pool.backup_from_device_all_layer(
            device_pool, host_indices, device_indices, io_backend="kernel"
        )

    ptrs = host_pool._kernel_ptr_tables
    assert ptrs is not None
    assert (
        calls
        == [
            (ptrs["device_conv"], ptrs["host_conv"]),
            (ptrs["device_ssm"], ptrs["host_ssm"]),
        ]
        * 2
    )


def test_mamba_pool_host_direct_roundtrip_is_layerwise():
    device_pool = _new_mamba_pool(size=4)
    host_pool = MambaPoolHost(
        device_pool, host_size_slots=4, pin_memory=False, register_host=False
    )
    device_indices = torch.tensor([0, 2], dtype=torch.int64)
    host_indices = torch.tensor([1, 3], dtype=torch.int64)

    device_pool.conv_state[:, device_indices] = torch.arange(
        device_pool.conv_state[:, device_indices].numel(), dtype=torch.float32
    ).reshape_as(device_pool.conv_state[:, device_indices])
    device_pool.ssm_state[:, device_indices] = (
        torch.arange(
            device_pool.ssm_state[:, device_indices].numel(), dtype=torch.float32
        ).reshape_as(device_pool.ssm_state[:, device_indices])
        + 1000
    )
    expected_conv = device_pool.conv_state[:, device_indices].clone()
    expected_ssm = device_pool.ssm_state[:, device_indices].clone()

    host_pool.backup_from_device_all_layer(
        device_pool,
        host_indices=host_indices,
        device_indices=device_indices,
        io_backend="direct",
    )
    device_pool.conv_state.zero_()
    device_pool.ssm_state.zero_()

    host_pool.load_to_device_per_layer(
        device_pool,
        host_indices=host_indices,
        device_indices=device_indices,
        layer_idx=0,
        io_backend="direct",
    )
    assert torch.equal(device_pool.conv_state[0, device_indices], expected_conv[0])
    assert torch.equal(device_pool.ssm_state[0, device_indices], expected_ssm[0])
    assert torch.equal(
        device_pool.conv_state[1, device_indices], torch.zeros_like(expected_conv[1])
    )
    assert torch.equal(
        device_pool.ssm_state[1, device_indices], torch.zeros_like(expected_ssm[1])
    )

    host_pool.load_to_device_per_layer(
        device_pool,
        host_indices=host_indices,
        device_indices=device_indices,
        layer_idx=1,
        io_backend="direct",
    )
    assert torch.equal(device_pool.conv_state[:, device_indices], expected_conv)
    assert torch.equal(device_pool.ssm_state[:, device_indices], expected_ssm)


def test_mamba_cache_pool_registers_layer_counter_and_delegates():
    device_pool = _new_mamba_pool(size=2)
    host_pool = MambaPoolHost(
        device_pool, host_size_slots=2, pin_memory=False, register_host=False
    )
    cache_pool = MambaCachePool(device_pool, host_pool, io_backend="direct")

    assert device_pool.layer_transfer_counter is cache_pool.get_layer_done_counter()
    assert cache_pool.kind.value == "mamba"
    assert cache_pool.page_size() == 1
    assert cache_pool.num_layers() == 2
    assert cache_pool.local_layer_idx(7) == 1


class SpyCounter:
    def __init__(self):
        self.waited = []

    def wait_until(self, layer_idx: int):
        self.waited.append(layer_idx)


def test_simple_mamba_pool_waits_for_local_layer_before_returning_params():
    device_pool = _new_mamba_pool(size=2)
