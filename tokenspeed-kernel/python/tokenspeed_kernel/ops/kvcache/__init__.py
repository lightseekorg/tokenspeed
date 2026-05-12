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

"""Unified KV cache transfer entrypoints.

Public functions ``transfer_kv_per_layer`` and ``transfer_kv_all_layer``
auto-select between the DMA backend (``cuMemcpyBatchAsync`` via
``ops/kvcache/dma.py``, NVIDIA + CUDA 12.8+; copies run on GPU copy
engines, no SM contention) and the Triton backend
(``ops/kvcache/triton.py``, SM-using). The engine queries
``loadback_indices_device`` to learn where to place index tensors for
the active backend. Override with
``TOKENSPEED_KVCACHEIO_BACKEND={auto,dma,triton}``.
"""

from __future__ import annotations

import os
from functools import lru_cache

import torch
from tokenspeed_kernel.ops.kvcache import dma as _dma
from tokenspeed_kernel.ops.kvcache import triton as _triton
from tokenspeed_kernel.platform import current_platform

_OVERRIDE_ENV = "TOKENSPEED_KVCACHEIO_BACKEND"


@lru_cache(maxsize=1)
def select_backend() -> str:
    """Return ``'dma'`` or ``'triton'`` for the current platform/env."""
    override = os.environ.get(_OVERRIDE_ENV, "auto").lower()
    if override in ("dma", "triton"):
        return override
    if current_platform().is_nvidia and _dma.is_available():
        return "dma"
    return "triton"


def loadback_indices_device(default_device: torch.device) -> torch.device:
    """Where the engine should keep loadback index tensors."""
    if select_backend() == "dma":
        return torch.device("cpu")
    return default_device


def transfer_kv_per_layer(
    src_k: torch.Tensor,
    dst_k: torch.Tensor,
    src_v: torch.Tensor,
    dst_v: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    item_size: int,
    page_size: int = 1,
) -> None:
    """One-layer KV transfer; routes to DMA or Triton based on backend.

    ``page_size`` lets the DMA path coalesce contiguous slots into one
    descriptor per page (k descriptors instead of k*page_size). Triton
    ignores it — its kernel is already per-slot.
    """
    if select_backend() == "dma":
        _dma.transfer_kv_per_layer(
            src_k, dst_k, src_v, dst_v, src_indices, dst_indices, item_size, page_size
        )
        return
    _triton.transfer_kv_per_layer(
        src_k, dst_k, src_v, dst_v, src_indices, dst_indices, item_size
    )


_ptr_tensor_cache: dict = {}


def _layers_to_ptr_tensor(layers: list, device: torch.device) -> torch.Tensor:
    # Pool-owned layer lists live for the process lifetime, so id() is stable.
    key = (id(layers), device)
    cached = _ptr_tensor_cache.get(key)
    if cached is not None:
        return cached
    ptrs = torch.tensor(
        [t.data_ptr() for t in layers], dtype=torch.uint64, device=device
    )
    _ptr_tensor_cache[key] = ptrs
    return ptrs


def transfer_kv_all_layer(
    src_k_layers: list,
    dst_k_layers: list,
    src_v_layers: list,
    dst_v_layers: list,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    item_size: int,
    num_layers: int,
    page_size: int = 1,
) -> None:
    """All-layer KV transfer; routes to DMA or Triton based on backend.

    ``src_*_layers`` / ``dst_*_layers`` are lists of per-layer Tensors
    (e.g. the pool's ``k_buffer`` / ``k_data_refs``). ``page_size``
    lets the DMA path coalesce contiguous slots into one descriptor per
    page; pass ``self.page_size`` from the engine when indices are
    page-aligned (the usual case for loadback / writeback).
    """
    if select_backend() == "dma":
        _dma.transfer_kv_all_layer(
            src_k_layers,
            dst_k_layers,
            src_v_layers,
            dst_v_layers,
            src_indices,
            dst_indices,
            item_size,
            num_layers,
            page_size,
        )
        return
    gpu = src_indices.device
    _triton.transfer_kv_all_layer(
        _layers_to_ptr_tensor(src_k_layers, gpu),
        _layers_to_ptr_tensor(dst_k_layers, gpu),
        _layers_to_ptr_tensor(src_v_layers, gpu),
        _layers_to_ptr_tensor(dst_v_layers, gpu),
        src_indices,
        dst_indices,
        item_size,
        num_layers,
    )


__all__ = [
    "select_backend",
    "loadback_indices_device",
    "transfer_kv_per_layer",
    "transfer_kv_all_layer",
]
