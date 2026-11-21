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
"""Compute-unit and XCD counts of a gfx950 device.

gfx950 parts differ in size: MI350X and MI355X have 8 XCDs and 256 CUs, MI350P
has 4 XCDs and 128 CUs (https://instinct.docs.amd.com/latest/gpu-arch/mi350.html).
Every XCD has 32 CUs. Kernels size grids and XCD swizzles from these values
instead of fixed constants. Call them at launch time, not at import time, so
that importing a kernel module does not initialize the GPU runtime.
"""

from __future__ import annotations

import functools

import torch

_CUS_PER_XCD = 32


@functools.cache
def _num_cus(index: int) -> int:
    return torch.cuda.get_device_properties(index).multi_processor_count


def num_cus(device: torch.device | str | int) -> int:
    """Return the number of compute units of ``device``.

    A device without an index refers to the current device.
    """
    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise ValueError(f"expected a GPU device, got {resolved}")
    index = (
        resolved.index if resolved.index is not None else torch.cuda.current_device()
    )
    return _num_cus(index)


def num_xcds(device: torch.device | str | int) -> int:
    """Return the number of XCD chiplets of ``device`` (32 CUs per XCD)."""
    return max(1, num_cus(device) // _CUS_PER_XCD)
