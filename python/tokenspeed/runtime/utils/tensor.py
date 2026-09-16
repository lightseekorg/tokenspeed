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

"""Pack small host tensors into a single device transfer."""

from __future__ import annotations

import torch


def upload_packed(
    parts: tuple[torch.Tensor, ...], device: torch.device | str
) -> tuple[torch.Tensor, ...]:
    """Upload CPU metadata once and return views in the input order.

    Args:
        parts: Dense CPU tensors, including scalars, empty tensors and
            noncontiguous views. Their values, shapes and dtypes are preserved;
            their strides are not. An empty tuple produces an empty tuple.
        device: Destination for the shared storage. CUDA transfers use pinned
            host memory and are asynchronous on the current stream.

    Returns:
        Contiguous typed views over one shared allocation, independent of the
        inputs. Byte offsets are aligned to each view's element size.

    A fresh host buffer per call keeps overlapping forwards independent. As in
    InputBuffers._bulk_pinned, PyTorch's pinned allocator fences reuse until
    the asynchronous copy finishes; staging must never be a persistent buffer.
    """
    if not parts:
        return ()
    if any(part.device.type != "cpu" for part in parts):
        raise ValueError("upload_packed requires CPU tensors")
    device = torch.device(device)
    offsets = []
    total_bytes = 0
    for part in parts:
        alignment = part.element_size()
        total_bytes = (total_bytes + alignment - 1) // alignment * alignment
        offsets.append(total_bytes)
        total_bytes += part.numel() * alignment
    host = torch.empty(total_bytes, dtype=torch.uint8, pin_memory=device.type == "cuda")
    for part, offset in zip(parts, offsets):
        data = part.reshape(-1).view(torch.uint8)
        host[offset : offset + data.numel()].copy_(data)
    uploaded = host.to(device=device, non_blocking=True)
    return tuple(
        uploaded[offset : offset + part.numel() * part.element_size()]
        .view(part.dtype)
        .view(part.shape)
        for part, offset in zip(parts, offsets)
    )
