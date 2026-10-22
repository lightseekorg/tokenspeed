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

"""Device copy helpers."""

from __future__ import annotations

import torch


def zero_rows_after(tensor: torch.Tensor, live_rows: torch.Tensor) -> None:
    """Zero rows ``[live_rows, tensor.shape[0])`` of ``tensor`` in place.

    The count is read on the device, so the call never synchronizes and a
    captured CUDA graph replays it against whatever count the tensor holds at
    replay. Rows before the count are not read or written.

    Args:
        tensor: Rows along dim 0. Its trailing dims must form one strided
            dimension (a 2-D tensor of any strides, or contiguous trailing
            dims); otherwise a ``RuntimeError`` is raised rather than zeroing
            a copy.
        live_rows: One-element int32 or int64 tensor on ``tensor``'s device
            holding the number of leading rows to keep.

    Returns:
        None.
    """
    if tensor.device.type == "cpu":
        if live_rows.numel() != 1:
            raise ValueError("live_rows must be a one-element tensor")
        tensor[int(live_rows.reshape(())) :].zero_()
        return
    from tokenspeed_kernel.ops.copy.triton import triton_zero_rows_after

    triton_zero_rows_after(tensor, live_rows)
