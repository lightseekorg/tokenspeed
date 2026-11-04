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

"""Ascend initialization sets event defaults without changing explicit options."""

import math

import pytest
import torch

pytest.importorskip("torch_npu")

from tokenspeed_kernel.platform import _npu_is_available


@pytest.fixture
def npu_events():
    if not torch.npu.is_available():
        pytest.skip("requires NPU")
    original_defaults = torch.npu.Event.__new__.__defaults__
    try:
        assert _npu_is_available()
        yield
    finally:
        torch.npu.Event.__new__.__defaults__ = original_defaults


def _record_pair(start):
    end = torch.npu.Event(enable_timing=True)
    stream = torch.npu.Stream()
    start.record(stream)
    end.record(stream)
    end.synchronize()
    return end


def test_npu_event_default_supports_timing(npu_events):
    start = torch.npu.Event()
    elapsed = start.elapsed_time(_record_pair(start))
    assert start.recorded_time() > 0
    assert math.isfinite(elapsed) and elapsed >= 0


@pytest.mark.parametrize("timing", [False, True])
@pytest.mark.parametrize("positional", [False, True])
def test_npu_event_preserves_explicit_timing(npu_events, timing, positional):
    start = (
        torch.npu.Event(timing, True, False)
        if positional
        else torch.npu.Event(enable_timing=timing, blocking=True, interprocess=False)
    )
    end = _record_pair(start)
    if timing:
        assert start.recorded_time() > 0
        elapsed = start.elapsed_time(end)
        assert math.isfinite(elapsed) and elapsed >= 0
    else:
        assert start.recorded_time() == 0
