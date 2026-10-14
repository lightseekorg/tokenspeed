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

"""Arrival-counter contract shared by the gfx950 split-K GEMMs.

Each split of a tile writes its FP32 partial, then bumps the tile's int32
counter; the last split to arrive reduces the partials and resets the counter
to zero. Callers therefore pass counters that are zero before the launch, and
get them back zeroed, so one buffer serves any number of ordered launches
(``tokenspeed_kernel.workspace.zeroed_state`` provides them).
"""

from __future__ import annotations

import torch


def check_splitk_counters(
    counters: torch.Tensor, num_counters: int, device: torch.device, kernel: str
) -> None:
    """Reject counters that cannot hold ``num_counters`` tile arrivals.

    Args:
        counters: Caller-provided zeroed int32 counters.
        num_counters: Counters the launch needs; 0 when it does not split K.
        device: Device of the launch.
        kernel: Kernel name for error messages.
    """
    if num_counters == 0:
        return
    if (
        counters.dtype != torch.int32
        or counters.device != device
        or not counters.is_contiguous()
        or counters.numel() < num_counters
    ):
        raise ValueError(
            f"{kernel} needs {num_counters} contiguous int32 counters on {device}, "
            f"got {counters.dtype} {tuple(counters.shape)} on {counters.device}"
        )
