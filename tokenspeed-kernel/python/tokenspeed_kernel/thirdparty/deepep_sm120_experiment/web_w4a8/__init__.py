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

import torch


def pack_weight_scales(sf):
    """Prepare group32 UE8M0 bytes as MN-major int32 without changing weights."""
    experts, rows, groups = sf.shape
    if groups % 4:
        raise ValueError("Weight K must be a multiple of 128")
    words = sf.reshape(experts, rows, groups // 4, 4).to(torch.int32)
    shifts = torch.arange(4, device=sf.device, dtype=torch.int32) * 8
    words = (words << shifts).sum(-1).to(torch.int32)
    base = torch.empty(
        (experts, groups // 4, (rows + 3) // 4 * 4),
        dtype=torch.int32,
        device=sf.device,
    )
    base[:, :, :rows].copy_(words.transpose(1, 2))
    return base.transpose(1, 2)[:, :rows, :]
