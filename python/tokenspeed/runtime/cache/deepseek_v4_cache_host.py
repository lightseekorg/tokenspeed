from __future__ import annotations

import math
import threading
from functools import wraps
from typing import Optional

import psutil
import torch
from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.configs.deepseek_v4_cache_spec import (
    V4_INDEXER_COMPRESSOR_STATE_GROUP_ID,
    V4_SWA_KV_GROUP_ID,
    v4_compressed_kv_group_id,
    v4_compressor_state_group_id,
)
from tokenspeed.runtime.layers.attention.kv_cache.deepseek_v4 import (
    DeepseekV4TokenToKVPool,
    _deepseek_v4_cache_group_page_bytes,
)
from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)


def synchronized(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        with self.lock:
            return func(self, *args, **kwargs)

    return wrapper


def _allocate_host_group_pages(
    *,
    device_counts: dict[str, int],
    page_bytes: dict[str, int],
    host_ratio: float,
    host_size_gb: int,
) -> dict[str, int]:
    active_groups = [
        group_id
        for group_id, bytes_per_page in page_bytes.items()
        if bytes_per_page > 0 and int(device_counts.get(group_id, 0)) > 0
    ]
    if not active_groups:
        return {}

    if host_size_gb <= 0:
        return {
            group_id: max(2, int(math.ceil(device_counts[group_id] * host_ratio)))
            for group_id in active_groups
        }

    budget_bytes = int(host_size_gb * 1e9)
    min_pages_per_group = 2  # page 0 is the dummy page; keep one usable page.
    min_bytes = sum(
        min_pages_per_group * page_bytes[group_id] for group_id in active_groups
    )
    if budget_bytes < min_bytes:
        raise ValueError(
            "DeepSeek V4 KVStore host budget is too small to allocate one usable "
            "page plus the dummy page per group: "
            f"budget={budget_bytes} bytes minimum={min_bytes} bytes"
        )

    counts = {group_id: min_pages_per_group for group_id in active_groups}
    remaining_budget = budget_bytes - min_bytes
    desired_bytes = {
        group_id: int(device_counts[group_id]) * int(page_bytes[group_id])
        for group_id in active_groups
    }
    total_desired = max(sum(desired_bytes.values()), 1)
    for group_id in active_groups:
        extra_budget = remaining_budget * desired_bytes[group_id] // total_desired
        counts[group_id] += extra_budget // int(page_bytes[group_id])
    return counts


class DeepseekV4TokenToKVPoolHost:
    """Registered host mirror for DeepSeek V4 paged-cache groups."""

    def __init__(
        self,
        device_pool: DeepseekV4TokenToKVPool,
        host_to_device_ratio: float,
        host_size_gb: int,
        layout: str = "layer_first",
        device: str = "cpu",
        register_host: bool = True,
    ) -> None:
        if layout != "layer_first":
            raise ValueError("DeepSeek V4 KVStore host pool only supports layer_first")
        if host_to_device_ratio <= 0 and host_size_gb <= 0:
            raise ValueError("host_to_device_ratio must be positive")

        self.device_pool = device_pool
        self.layout = layout
        self.device = device
        self.layer_num = int(device_pool.layer_num)
        self.lock = threading.RLock()

        self.paged_cache_group_page_bytes = _deepseek_v4_cache_group_page_bytes(
            device_pool.layout,
            device_pool.paged_cache_group_specs,
            self.layer_num,
        )
        self.paged_cache_group_page_counts = _allocate_host_group_pages(
            device_counts=device_pool.paged_cache_group_page_counts,
            page_bytes=self.paged_cache_group_page_bytes,
            host_ratio=float(host_to_device_ratio),
            host_size_gb=int(host_size_gb),
        )
        self._shadow_page_num = self._compute_shadow_page_num(device_pool)
        self.total_bytes = int(
            sum(
                self.paged_cache_group_page_counts[group_id]
                * self.paged_cache_group_page_bytes[group_id]
                for group_id in self.paged_cache_group_page_counts
            )
        )
        self._check_host_memory(self.total_bytes)

        def alloc_like_group(
            source: torch.Tensor | None,
            group_id: str,
        ) -> torch.Tensor | None:
            if source is None:
                return None
            pages = int(self.paged_cache_group_page_counts[group_id])
            use_pin_memory = bool(
                device == "cpu" and not register_host and torch.cuda.is_available()
            )
            tensor = torch.empty(
                (pages, *source.shape[1:]),
                dtype=source.dtype,
                device=device,
                pin_memory=use_pin_memory,
            )
            if register_host:
                current_platform().register_host_tensor_for_gpu_access(tensor)
            return tensor

        self.swa_kv_buffer = [
            alloc_like_group(buf, V4_SWA_KV_GROUP_ID)
            for buf in device_pool.swa_kv_buffer
        ]
        self.compressed_kv_buffer: list[torch.Tensor | None] = []
        self.compressor_state_buffer: list[torch.Tensor | None] = []
        self.indexer_kv_buffer: list[torch.Tensor | None] = []
        self.indexer_state_buffer: list[torch.Tensor | None] = []

        for layer_id, ratio in enumerate(device_pool.layout.layer_ratio):
            compressed_group_id = v4_compressed_kv_group_id(ratio)
            state_group_id = v4_compressor_state_group_id(ratio)
            self.compressed_kv_buffer.append(
                alloc_like_group(
                    device_pool.compressed_kv_buffer[layer_id],
                    compressed_group_id,
                )
                if ratio > 1
                else None
            )
            self.compressor_state_buffer.append(
                alloc_like_group(
                    device_pool.compressor_state_buffer[layer_id],
                    state_group_id,
                )
                if ratio > 1
                else None
            )
            self.indexer_kv_buffer.append(
                alloc_like_group(
                    device_pool.indexer_kv_buffer[layer_id],
                    compressed_group_id,
                )
                if ratio == 4
                else None
            )
            self.indexer_state_buffer.append(
                alloc_like_group(
                    device_pool.indexer_state_buffer[layer_id],
                    V4_INDEXER_COMPRESSOR_STATE_GROUP_ID,
                )
                if ratio == 4
                else None
            )

        self.clear()
        logger.info(
            "Allocating %.2f GB host memory for DeepSeek V4 KVStore. "
            "group_pages=%s group_page_bytes=%s shadow_page_num=%s layout=%s",
            self.total_bytes / 1e9,
            self.paged_cache_group_page_counts,
            self.paged_cache_group_page_bytes,
            self._shadow_page_num,
            self.layout,
        )

    def _compute_shadow_page_num(self, device_pool: DeepseekV4TokenToKVPool) -> int:
        """Token-page capacity used only for scheduler HostResource pinning."""
        usable_token_pages_by_history_group: list[int] = []
        token_page_size = max(1, int(device_pool.page_size))
        for spec in device_pool.paged_cache_group_specs:
            if getattr(spec, "family", "history") != "history":
                continue
            group_pages = int(self.paged_cache_group_page_counts.get(spec.group_id, 0))
            usable_group_pages = max(0, group_pages - 1)  # page 0 is the dummy page.
            raw_tokens = (
                usable_group_pages
                * int(spec.rows_per_page)
                * int(spec.entry_stride_tokens)
            )
            usable_token_pages_by_history_group.append(
                int(math.ceil(raw_tokens / token_page_size))
            )
        if not usable_token_pages_by_history_group:
            return 0
        usable_token_pages = min(usable_token_pages_by_history_group)
        return usable_token_pages + 1 if usable_token_pages > 0 else 0

    @staticmethod
    def _check_host_memory(requested_bytes: int) -> None:
        host_mem = psutil.virtual_memory()
        ten_gb = 10 * (1024**3)
        available_bytes = host_mem.available - ten_gb
        if available_bytes <= 0:
            available_bytes = host_mem.available
        if requested_bytes > available_bytes:
            raise ValueError(
                "Not enough host memory available for DeepSeek V4 KVStore. "
                f"Requesting {requested_bytes / 1e9:.2f} GB but only have "
                f"{available_bytes / 1e9:.2f} GB free. Please reduce kvstore size."
            )

    @property
    def page_num(self) -> int:
        return self._shadow_page_num

    @synchronized
    def clear(self) -> None:
        self.free_pages_by_group = {
            group_id: torch.arange(1, count, dtype=torch.int64)
            for group_id, count in self.paged_cache_group_page_counts.items()
        }

    def available_size(self, group_id: str | None = None) -> int:
        if group_id is None:
            return sum(len(pages) for pages in self.free_pages_by_group.values())
        return len(self.free_pages_by_group[str(group_id)])

    @synchronized
    def alloc(self, group_id: str, need_pages: int) -> Optional[torch.Tensor]:
        group_id = str(group_id)
        if need_pages <= 0:
            return torch.empty((0,), dtype=torch.int64)
        free_pages = self.free_pages_by_group[group_id]
        if need_pages > len(free_pages):
            logger.warning(
                "[deepseek_v4_l2] alloc FAILED group=%s n=%s remain=%s",
                group_id,
                need_pages,
                len(free_pages),
            )
            return None
        selected = free_pages[:need_pages]
        self.free_pages_by_group[group_id] = free_pages[need_pages:]
        return selected

    @synchronized
    def free(self, group_id: str, pages: torch.Tensor) -> int:
        group_id = str(group_id)
        max_pages = int(self.paged_cache_group_page_counts[group_id])
        page_ids = {
            int(page)
            for page in pages.to(dtype=torch.int64, device="cpu").reshape(-1).tolist()
            if 0 < int(page) < max_pages
        }
        if not page_ids:
            return 0
        existing_ids = set(
            int(page) for page in self.free_pages_by_group[group_id].tolist()
        )
        new_pages = [page for page in sorted(page_ids) if page not in existing_ids]
        if not new_pages:
            return 0
        pages = torch.tensor(new_pages, dtype=torch.int64)
        self.free_pages_by_group[group_id] = torch.cat(
            [self.free_pages_by_group[group_id], pages]
        )
        return len(new_pages)
