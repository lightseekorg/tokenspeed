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

"""Producer-step counters use the requested accelerator and publish progress."""

import pytest
import torch

from tokenspeed.runtime.pd.utils import StepCounter


@pytest.mark.parametrize("device", ["cuda", "npu"])
def test_counter_publishes_progress_on_selected_device(device):
    if device == "npu":
        pytest.importorskip("torch_npu")
    module = torch.get_device_module(device)
    if not module.is_available():
        pytest.skip(f"{device} is unavailable")
    index = module.current_device()
    counter = StepCounter(device, index)
    assert counter.d_ready_cache_step.device == torch.device(device, index)
    assert counter.d_ready_cache_step.item() == 0
    counter.record_cache()
    module.synchronize()
    assert counter.h_ready_cache_step.item() == 1
    counter.d_ready_cache_step.fill_(counter.COUNT_NUM_MAX - 1)
    counter.record_cache()
    module.synchronize()
    assert counter.h_ready_cache_step.item() == 0
