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

import pytest
import torch

pytest.importorskip("torch_npu")
from tokenspeed.runtime.pd.utils import StepCounter


@pytest.mark.parametrize("device", ["npu", "npu:0"])
def test_step_counter_records_on_selected_npu(device):
    torch.npu.set_device(0)
    counter = StepCounter(device, 0)
    assert counter.d_ready_cache_step.device == torch.device("npu:0")
    assert counter.d_ready_aux_step.device == torch.device("npu:0")
    assert counter.query_ready_cache_step() == 0
    assert counter.current_step() == (0, 0)
    counter.record_cache()
    counter.record_cache()
    torch.npu.synchronize()
    assert counter.query_ready_cache_step() == 2
    counter.advance_step(2, 1)
    assert counter.current_step() == (2, 1)
