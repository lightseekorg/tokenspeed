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

"""Vendor-neutral byte transfer boundary for compact Host cache storage."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Literal

import torch

from tokenspeed_kernel.ops.kvcache.cuda import (
    transfer_cache_segments as _transfer_cache_segments_kernel,
)


_mapped_host_kernel_available: bool | None = None


def _kernel_is_unavailable(error: Exception) -> bool:
    message = str(error).lower()
    return isinstance(error, AttributeError) or any(
        marker in message
        for marker in (
            "kvcacheio library not found",
            "kernel implementation not found",
            "pinned but not device-mapped",
            "is not available",
            "has no attribute 'transfer_cache_segments'",
        )
    )


def _validate_segments(
    device_buffers: Sequence[torch.Tensor],
    host_backing: torch.Tensor,
    segments: Sequence[tuple[int, int, int, int]],
) -> None:
    if (
        host_backing.device.type != "cpu"
        or host_backing.dtype != torch.uint8
        or not host_backing.is_contiguous()
        or not host_backing.is_pinned()
    ):
        raise ValueError("host_backing must be contiguous pinned CPU uint8")
    for buffer in device_buffers:
        if buffer.device.type == "cpu" or not buffer.is_contiguous():
            raise ValueError("device cache buffers must be contiguous device tensors")
    for buffer_index, device_offset, host_offset, num_bytes in segments:
        if not 0 <= buffer_index < len(device_buffers):
            raise IndexError(f"unknown device buffer {buffer_index}")
        if device_offset < 0 or host_offset < 0 or num_bytes <= 0:
            raise ValueError("cache transfer offsets must be non-negative and non-empty")
        device_bytes = device_buffers[buffer_index].numel() * device_buffers[
            buffer_index
        ].element_size()
        if device_offset + num_bytes > device_bytes:
            raise IndexError("cache transfer segment lies outside its device buffer")
        if host_offset + num_bytes > host_backing.numel():
            raise IndexError("cache transfer segment lies outside Host backing")


def _transfer_dma(
    direction: Literal["d2h", "h2d"],
    device_buffers: Sequence[torch.Tensor],
    host_backing: torch.Tensor,
    segments: Sequence[tuple[int, int, int, int]],
) -> None:
    byte_buffers = tuple(buffer.view(torch.uint8).reshape(-1) for buffer in device_buffers)
    for buffer_index, device_offset, host_offset, num_bytes in segments:
        device_slice = byte_buffers[buffer_index][
            device_offset : device_offset + num_bytes
        ]
        host_slice = host_backing[host_offset : host_offset + num_bytes]
        destination, source = (
            (host_slice, device_slice)
            if direction == "d2h"
            else (device_slice, host_slice)
        )
        destination.copy_(source, non_blocking=True)


def transfer_cache_segments(
    direction: Literal["d2h", "h2d"],
    device_buffers: Sequence[torch.Tensor],
    host_backing: torch.Tensor,
    segments: Sequence[tuple[int, int, int, int]],
    stream,
    *,
    backend: Literal["auto", "kernel", "dma"] = "auto",
) -> None:
    """Copy byte segments between cache buffers and compact pinned Host memory.

    Args:
        direction: ``"d2h"`` for snapshot/store or ``"h2d"`` for load/recover.
        device_buffers: Device tensors referenced by segment buffer indices.
        host_backing: Contiguous pinned uint8 Host allocation.
        segments: ``(buffer_index, device_offset, host_offset, num_bytes)`` rows.
        stream: Device stream that orders the asynchronous copies.
        backend: Prefer one mapped-Host kernel launch or use asynchronous DMA.

    Returns:
        None. Completion is observed by recording an event on ``stream``.
    """

    if direction not in ("d2h", "h2d"):
        raise ValueError(f"unknown cache transfer direction {direction!r}")
    if backend not in ("auto", "kernel", "dma"):
        raise ValueError(f"unknown cache transfer backend {backend!r}")
    _validate_segments(device_buffers, host_backing, segments)
    if not segments:
        return

    global _mapped_host_kernel_available
    with torch.cuda.stream(stream):
        if backend != "dma" and _mapped_host_kernel_available is not False:
            descriptors = torch.tensor(segments, dtype=torch.int64, device="cpu")
            try:
                _transfer_cache_segments_kernel(
                    list(device_buffers),
                    host_backing,
                    descriptors,
                    0 if direction == "d2h" else 1,
                )
                _mapped_host_kernel_available = True
                return
            except (AttributeError, RuntimeError) as error:
                if backend == "kernel" or not _kernel_is_unavailable(error):
                    raise
                _mapped_host_kernel_available = False
                warnings.warn(
                    "Mapped Host cache transfer is unavailable; falling back to DMA",
                    RuntimeWarning,
                    stacklevel=2,
                )
        if backend == "kernel":
            raise RuntimeError("mapped Host cache transfer kernel is unavailable")
        _transfer_dma(
            direction,
            device_buffers,
            host_backing,
            segments,
        )
