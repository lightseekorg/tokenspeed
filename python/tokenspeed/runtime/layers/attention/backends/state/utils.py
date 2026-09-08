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

"""Shared row geometry for recurrent and auxiliary state copies."""

import torch


def row_stride_i32(tensor: torch.Tensor) -> int:
    """Return the physical state-row stride in int32 words for batched copies.

    Args:
        tensor: State rows with a nonempty leading dimension. Individual row
            payloads must be contiguous; padding between rows is allowed.

    Returns:
        The stride between rows in four-byte units, including padding.

    Raises:
        RuntimeError: A row payload is noncontiguous or the row stride is not
            aligned to four bytes.
    """
    if tensor[0].numel() and not tensor[0].is_contiguous():
        raise RuntimeError("batched verify state copy requires contiguous row payloads")
    stride_bytes = tensor.stride(0) * tensor.element_size()
    if stride_bytes % 4:
        raise RuntimeError("state row stride must be 4-byte aligned")
    return stride_bytes // 4
