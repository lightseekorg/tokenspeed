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
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.sm_reservation import (
    max_active_clusters,
    reserve_sm_clusters,
    reserved_sm_clusters,
)

pytestmark = pytest.mark.skipif(
    not current_platform().is_nvidia, reason="SM reservations need an NVIDIA GPU"
)


@pytest.fixture
def device():
    device = torch.device("cuda", torch.cuda.current_device())
    yield device
    reserve_sm_clusters(device, 0)


def test_reserved_clusters_leave_matching_slots(device):
    import cutlass.utils as utils

    hardware = utils.HardwareInfo(device.index)
    free = {size: hardware.get_max_active_clusters(size) for size in (1, 2, 4, 8, 16)}
    assert {size: max_active_clusters(device, size) for size in free} == free
    reserve_sm_clusters(device, 3)
    assert reserved_sm_clusters(device) == 3
    # Each reserved 8-CTA cluster takes 8 single-CTA slots, 4 pair slots,
    # 2 quad slots, and one slot of 8 or more.
    assert max_active_clusters(device, 1) == free[1] - 24
    assert max_active_clusters(device, 2) == free[2] - 12
    assert max_active_clusters(device, 4) == free[4] - 6
    assert max_active_clusters(device, 8) == free[8] - 3
    assert max_active_clusters(device, 16) == free[16] - 3


def test_invalid_reservations_are_rejected(device):
    with pytest.raises(ValueError):
        reserve_sm_clusters(device, -1)
    with pytest.raises(ValueError):
        reserve_sm_clusters(torch.device("cpu"), 1)
    reserve_sm_clusters(device, 1 << 10)
    with pytest.raises(ValueError):
        max_active_clusters(device, 1)
