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

from types import SimpleNamespace
from unittest.mock import Mock

import torch
from tokenspeed_kernel.platform import CapabilityRequirement, Platform, current_platform


def test_registration_defers_physical_properties_until_after_rank_binding(monkeypatch):
    import tokenspeed_kernel.platform as p

    selected = None
    lookups = []
    npu = Mock()
    npu.device_count.return_value = 16
    npu.current_device.side_effect = lambda: selected

    def properties(device):
        assert device == 7
        lookups.append(device)
        return SimpleNamespace(name="rank-7", total_memory=1024, cube_core_num=32)

    npu.get_device_properties.side_effect = properties
    monkeypatch.setattr(torch, "npu", npu, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(p, "_npu_is_available", lambda: True)
    monkeypatch.setattr(Platform, "_instance", None)
    platform = current_platform()
    assert platform.is_npu and not platform.is_hopper
    assert CapabilityRequirement(vendors=frozenset({"ascend"})).satisfied_by(platform)
    npu.current_device.assert_not_called()
    npu.get_device_properties.assert_not_called()

    selected = 7
    assert (platform.device_name, platform.total_memory, platform.sm_count) == (
        "rank-7",
        1024,
        32,
    )
    assert lookups == [7]
