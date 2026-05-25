from __future__ import annotations

import threading
from functools import wraps
from typing import Optional

import torch
from tokenspeed_kernel.ops.kvcache.cuda import (
    transfer_kv_all_layer_mla,
    transfer_kv_direct,
    transfer_kv_per_layer_mla,
)
from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.layers.attention.backends.hybrid_linear_attn import (
    SimpleMambaPool,
)
from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)
MAMBA_KVSTORE_LOADBACK_BLOCK_QUOTA = 16
MAMBA_KVSTORE_WRITEBACK_BLOCK_QUOTA = 16


def synchronized(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        with self.lock:
            return func(self, *args, **kwargs)

    return wrapper


class MambaPoolHost:
    """Pinned host mirror for SimpleMambaPool conv_state and ssm_state."""

    def __init__(
        self,
        device_pool: SimpleMambaPool,
        host_size_slots: int,
        layout: str = "layer_first",
        pin_memory: bool = True,
        device: str = "cpu",
        register_host: bool = True,
    ):
        if layout != "layer_first":
            raise ValueError("MambaPoolHost v1 only supports layer_first layout")
        if host_size_slots <= 0:
            raise ValueError("host_size_slots must be positive")
        self.device_pool = device_pool
        self.layout = layout
        self.device = device
        self.size = int(host_size_slots)
        self.page_size = 1
        self.num_layers = int(device_pool.conv_state.shape[0])
        self.conv_shape = tuple(device_pool.conv_state.shape[2:])
        self.ssm_shape = tuple(device_pool.ssm_state.shape[2:])
        self.conv_dtype = device_pool.conv_state.dtype
        self.ssm_dtype = device_pool.ssm_state.dtype
        self.conv_item_size = device_pool.conv_state[0, 0].nbytes
        self.ssm_item_size = device_pool.ssm_state[0, 0].nbytes
        self.size_per_slot = self.num_layers * (
            self.conv_item_size + self.ssm_item_size
        )

        # cudaHostRegister pins ordinary host memory for GPU-side access.
        # Avoid allocating an already pinned tensor when we will register it,
        # because some CUDA stacks reject double registration.
        use_pin_memory = bool(pin_memory and device == "cpu" and not register_host)
        self.conv_buffer = torch.empty(
            (self.num_layers, self.size, *self.conv_shape),
            dtype=self.conv_dtype,
            device=device,
            pin_memory=use_pin_memory,
        )
        self.ssm_buffer = torch.empty(
            (self.num_layers, self.size, *self.ssm_shape),
            dtype=self.ssm_dtype,
            device=device,
            pin_memory=use_pin_memory,
        )
        if register_host:
            platform = current_platform()
            platform.register_host_tensor_for_gpu_access(self.conv_buffer)
            platform.register_host_tensor_for_gpu_access(self.ssm_buffer)

        self.conv_data_refs = [self.conv_buffer[i] for i in range(self.num_layers)]
        self.ssm_data_refs = [self.ssm_buffer[i] for i in range(self.num_layers)]
        # Keep CUDA all-layer kernel pointer tables alive across async launches.
        self._kernel_ptr_tables: dict[str, torch.Tensor] | None = None

        self.lock = threading.RLock()
        self.clear()
        logger.info(
            "[mamba_l2] alloc host buffer pool_type=%s size_slots=%s "
            "size_per_slot_mb=%.2f num_mamba_layers=%s layout=%s pin_memory=%s "
            "total_gb=%.2f",
            type(self).__name__,
            self.size,
            self.size_per_slot / 1e6,
            self.num_layers,
            self.layout,
            use_pin_memory,
            self.size * self.size_per_slot / 1e9,
        )

    @synchronized
    def clear(self) -> None:
        self.free_slots = torch.arange(self.size, dtype=torch.int64)

    def available_size(self) -> int:
        return len(self.free_slots)

    @synchronized
    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        if need_size <= 0:
            return torch.empty((0,), dtype=torch.int64)
        if need_size > self.available_size():
            logger.warning(
                "[mamba_l2] alloc FAILED n=%s remain=%s (will trigger eviction)",
                need_size,
                self.available_size(),
            )
            return None
        selected = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        logger.debug(
            "[mamba_l2] alloc n=%s remain=%s", need_size, self.available_size()
        )
        return selected

    @synchronized
    def free(self, indices: torch.Tensor) -> int:
        indices = indices.to(dtype=torch.int64, device="cpu")
        self.free_slots = torch.cat([self.free_slots, indices])
        logger.debug(
            "[mamba_l2] free n=%s deferred=%s remain=%s",
            len(indices),
            False,
            self.available_size(),
        )
        return len(indices)

    def backup_from_device_all_layer(
        self,
        device_pool: SimpleMambaPool,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        io_backend: str,
        block_quota: Optional[int] = None,
    ) -> None:
        if block_quota is None:
            block_quota = MAMBA_KVSTORE_WRITEBACK_BLOCK_QUOTA
        if io_backend == "kernel":
            ptrs = self._ensure_kernel_ptr_tables(device_pool)
            transfer_kv_all_layer_mla(
                src_layers=ptrs["device_conv"],
                dst_layers=ptrs["host_conv"],
                src_indices=device_indices,
                dst_indices=host_indices,
                item_size=self.conv_item_size,
                num_layers=self.num_layers,
                block_quota=block_quota,
            )
            transfer_kv_all_layer_mla(
                src_layers=ptrs["device_ssm"],
                dst_layers=ptrs["host_ssm"],
                src_indices=device_indices,
                dst_indices=host_indices,
                item_size=self.ssm_item_size,
                num_layers=self.num_layers,
                block_quota=block_quota,
            )
        elif io_backend == "direct":
            transfer_kv_direct(
                src_layers=self._layer_refs(device_pool.conv_state)
                + self._layer_refs(device_pool.ssm_state),
                dst_layers=self.conv_data_refs + self.ssm_data_refs,
                src_indices=device_indices,
                dst_indices=host_indices,
                page_size=self.page_size,
            )
        else:
            raise ValueError(f"Unsupported IO backend: {io_backend}")

    def load_to_device_per_layer(
        self,
        device_pool: SimpleMambaPool,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        layer_idx: int,
        io_backend: str = "kernel",
    ) -> None:
        if not 0 <= layer_idx < self.num_layers:
            raise IndexError(f"layer_idx out of range: {layer_idx}")
        if io_backend == "kernel":
            transfer_kv_per_layer_mla(
                src=self.conv_buffer[layer_idx],
                dst=device_pool.conv_state[layer_idx],
                src_indices=host_indices,
                dst_indices=device_indices,
                item_size=self.conv_item_size,
                block_quota=MAMBA_KVSTORE_LOADBACK_BLOCK_QUOTA,
            )
            transfer_kv_per_layer_mla(
                src=self.ssm_buffer[layer_idx],
                dst=device_pool.ssm_state[layer_idx],
                src_indices=host_indices,
                dst_indices=device_indices,
                item_size=self.ssm_item_size,
                block_quota=MAMBA_KVSTORE_LOADBACK_BLOCK_QUOTA,
            )
        elif io_backend == "direct":
            transfer_kv_direct(
                src_layers=[self.conv_buffer[layer_idx], self.ssm_buffer[layer_idx]],
                dst_layers=[
                    device_pool.conv_state[layer_idx],
                    device_pool.ssm_state[layer_idx],
                ],
                src_indices=host_indices,
                dst_indices=device_indices,
                page_size=self.page_size,
            )
        else:
            raise ValueError(f"Unsupported IO backend: {io_backend}")

    def get_hybrid_pool_buffer(self) -> list[torch.Tensor]:
        return [self.conv_buffer, self.ssm_buffer]

    @staticmethod
    def _layer_refs(buffer: torch.Tensor) -> list[torch.Tensor]:
        return [buffer[i] for i in range(buffer.shape[0])]

    def _ensure_kernel_ptr_tables(
        self, device_pool: SimpleMambaPool
    ) -> dict[str, torch.Tensor]:
        if self._kernel_ptr_tables is None:
            self._kernel_ptr_tables = {
                "device_conv": self._data_ptrs(
                    device_pool.conv_state, device_pool.device
                ),
                "host_conv": self._data_ptrs(self.conv_buffer, device_pool.device),
                "device_ssm": self._data_ptrs(
                    device_pool.ssm_state, device_pool.device
                ),
                "host_ssm": self._data_ptrs(self.ssm_buffer, device_pool.device),
            }
        return self._kernel_ptr_tables

    @staticmethod
    def _data_ptrs(buffer: torch.Tensor, device) -> torch.Tensor:
        platform = current_platform()
        return torch.tensor(
            [
                platform.device_visible_data_ptr(buffer[i])
                for i in range(buffer.shape[0])
            ],
            dtype=torch.uint64,
            device=device,
        )
