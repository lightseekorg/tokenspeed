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

"""DMA solution for KV cache transfers via ``cuMemcpyBatchAsync``.

Same per-layer / all-layer function signatures as the Triton solution in
``triton.py`` so the dispatcher in ``__init__.py`` can swap backends
transparently. Copies run on the GPU's copy engines, leaving SMs / L2
free for concurrent compute. This module never calls ``.cpu()`` /
``.detach()`` / ``.cuda()`` on its inputs — the engine is responsible
for placing indices on host (a misplaced input raises rather than
syncing silently).
"""

from __future__ import annotations

import torch
from tokenspeed_kernel.thirdparty.cuda.dma_kvcacheio import (
    batch_async_load_kv_pages,
    is_available,
)

__all__ = ["is_available", "transfer_kv_per_layer", "transfer_kv_all_layer"]


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
    """One-layer KV transfer via ``cuMemcpyBatchAsync``.

    With ``page_size > 1`` indices must be page-aligned and we coalesce
    contiguous slots into one descriptor per page (e.g. 100 page descriptors
    instead of 3200 slot descriptors for a 100-page transfer).
    """
    if src_indices.numel() == 0:
        return
    assert (
        not src_indices.is_cuda and not dst_indices.is_cuda
    ), "DMA backend requires CPU index tensors"
    src_off, dst_off, block_bytes = _coalesce_to_pages(
        src_indices, dst_indices, item_size, page_size
    )
    batch_async_load_kv_pages(
        host_k_layers=[src_k],
        host_v_layers=[src_v],
        device_k_layers=[dst_k],
        device_v_layers=[dst_v],
        src_page_token_offsets=src_off,
        dst_page_token_offsets=dst_off,
        page_bytes=block_bytes,
        token_stride_bytes=item_size,
        stream=torch.cuda.current_stream(),
    )


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
    """All-layer KV transfer via ``cuMemcpyBatchAsync``.

    See :func:`transfer_kv_per_layer` for the ``page_size`` semantics.
    """
    if src_indices.numel() == 0:
        return
    assert (
        not src_indices.is_cuda and not dst_indices.is_cuda
    ), "DMA backend requires CPU index tensors"
    assert (
        len(src_k_layers)
        == len(dst_k_layers)
        == len(src_v_layers)
        == len(dst_v_layers)
        == num_layers
    )
    src_off, dst_off, block_bytes = _coalesce_to_pages(
        src_indices, dst_indices, item_size, page_size
    )
    batch_async_load_kv_pages(
        host_k_layers=src_k_layers,
        host_v_layers=src_v_layers,
        device_k_layers=dst_k_layers,
        device_v_layers=dst_v_layers,
        src_page_token_offsets=src_off,
        dst_page_token_offsets=dst_off,
        page_bytes=block_bytes,
        token_stride_bytes=item_size,
        stream=torch.cuda.current_stream(),
    )


def _coalesce_to_pages(
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    item_size: int,
    page_size: int,
):
    """Stride down to one descriptor per page when indices are page-aligned."""
    if page_size <= 1:
        return src_indices, dst_indices, item_size
    return (
        src_indices.view(-1, page_size)[:, 0],
        dst_indices.view(-1, page_size)[:, 0],
        page_size * item_size,
    )
