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

"""Tier-2 CPU LoRA adapter cache with async disk prefetch."""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor

import torch

from tokenspeed.runtime.lora.adapter_io import AdapterWeights, load_adapter_weights
from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)


class LoraCpuCache:
    def __init__(
        self,
        *,
        capacity: int,
        is_gpu_resident: Callable[[str], bool],
    ) -> None:
        self.capacity = capacity
        self.is_gpu_resident = is_gpu_resident
        self.cache: dict[str, AdapterWeights] = {}
        self.lru: OrderedDict[str, None] = OrderedDict()
        self.adapter_paths: dict[str, str] = {}
        self.loader_executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="lora-loader"
        )
        self.lock = threading.Lock()
        self.pending_loads: dict[str, Future] = {}

    def set_path(self, name: str, adapter_path: str) -> None:
        self.adapter_paths[name] = adapter_path

    def remove(self, name: str) -> None:
        self.evict(name)
        self.adapter_paths.pop(name, None)
        with self.lock:
            self.pending_loads.pop(name, None)

    def prefetch(self, name: str) -> None:
        """Best-effort async warm of the CPU pool for *name*."""
        with self.lock:
            if name in self.cache:
                self.lru.move_to_end(name)
                return
            if name in self.pending_loads:
                return
            adapter_path = self.adapter_paths.get(name)
            if adapter_path is None:
                return
            fut = self.loader_executor.submit(
                self._async_load_weights, name, adapter_path
            )
            self.pending_loads[name] = fut

    def ensure(
        self,
        name: str,
        weights: AdapterWeights | None = None,
    ) -> None:
        """Synchronously ensure *name* is CPU-resident."""
        with self.lock:
            if name in self.cache:
                self.lru.move_to_end(name)
                return
            pending = self.pending_loads.get(name)

        if pending is not None:
            pending.result()
            with self.lock:
                if name in self.cache:
                    self.lru.move_to_end(name)
                    return

        if weights is None:
            adapter_path = self.adapter_paths.get(name)
            if adapter_path is None:
                raise KeyError(f"Adapter '{name}' has no recorded disk path.")
            weights = load_adapter_weights(adapter_path)

        with self.lock:
            if name in self.cache:
                self.lru.move_to_end(name)
                return
            self._install_locked(name, weights)

    def evict(self, name: str) -> None:
        with self.lock:
            self._evict_locked(name)

    def _async_load_weights(self, name: str, adapter_path: str) -> None:
        try:
            weights = load_adapter_weights(adapter_path)
        except Exception:
            logger.exception("Async LoRA load failed for '%s'", name)
            with self.lock:
                self.pending_loads.pop(name, None)
            return
        with self.lock:
            try:
                if name not in self.cache:
                    self._install_locked(name, weights)
            finally:
                self.pending_loads.pop(name, None)

    def _install_locked(self, name: str, weights: AdapterWeights) -> None:
        while len(self.cache) >= self.capacity:
            evicted = False
            # Prefer evicting non-GPU-resident entries first: they cost a disk
            # read to bring back, while GPU-resident ones cost nothing until
            # their GPU slot is also evicted.
            for stage in ("non_gpu", "gpu_resident"):
                for candidate in list(self.lru.keys()):
                    if candidate == name:
                        continue
                    is_gpu = self.is_gpu_resident(candidate)
                    if stage == "non_gpu" and is_gpu:
                        continue
                    self._evict_locked(candidate)
                    evicted = True
                    break
                if evicted:
                    break
            if not evicted:
                raise RuntimeError(
                    f"CPU LoRA pool is full ({len(self.cache)}/{self.capacity}) "
                    "and no evictable entry was found. "
                    f"cpu_lru={list(self.lru.keys())}. "
                    "Increase max_loras_cpu."
                )
        self.cache[name] = self._pin_weights(weights)
        self.lru[name] = None

    def _evict_locked(self, name: str) -> None:
        if name in self.cache:
            del self.cache[name]
            self.lru.pop(name, None)
            logger.debug(
                "Evicted '%s' from CPU pool (now %d/%d)",
                name,
                len(self.cache),
                self.capacity,
            )

    def _pin_weights(self, weights: AdapterWeights) -> AdapterWeights:
        return {
            layer_id: {
                module: (
                    self._pin_tensor(lora_A),
                    self._pin_tensor(lora_B),
                )
                for module, (lora_A, lora_B) in modules.items()
            }
            for layer_id, modules in weights.items()
        }

    @staticmethod
    def _pin_tensor(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.device.type != "cpu" or tensor.is_pinned():
            return tensor
        try:
            return tensor.pin_memory()
        except RuntimeError:
            return tensor
