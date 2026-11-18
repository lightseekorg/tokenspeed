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
from __future__ import annotations

import pytest
import torch
from utils import is_cdna4

if not is_cdna4():
    pytest.skip(
        "AMD CDNA4 is required for the gfx950 device geometry test",
        allow_module_level=True,
    )

from tokenspeed_kernel_amd._device import num_cus, num_xcds  # noqa: E402


@pytest.mark.parametrize("device_index", range(torch.cuda.device_count()))
def test_geometry_matches_device_properties(device_index: int) -> None:
    device = torch.device("cuda", device_index)
    properties = torch.cuda.get_device_properties(device)
    assert num_cus(device) == properties.multi_processor_count
    assert num_xcds(device) * 32 == num_cus(device)


def test_device_without_index_is_the_current_device() -> None:
    current = torch.device("cuda", torch.cuda.current_device())
    assert num_cus(torch.device("cuda")) == num_cus(current)
    assert num_xcds(torch.device("cuda")) == num_xcds(current)


def test_cpu_device_is_rejected() -> None:
    with pytest.raises(ValueError, match="GPU device"):
        num_cus(torch.device("cpu"))
