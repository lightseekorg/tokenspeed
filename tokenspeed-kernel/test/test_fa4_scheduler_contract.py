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

import inspect

import pytest

tile_scheduler = pytest.importorskip("flash_attn.cute.tile_scheduler")

from tokenspeed_kernel.ops.attention.cute_dsl.rel_mha import fmha_bias_helper


def test_rel_mha_uses_current_fa4_scheduler_state_api() -> None:
    assert fmha_bias_helper.SchedulerState is tile_scheduler.SchedulerState
    assert (
        fmha_bias_helper.DynamicPersistentVarlenScheduler
        is tile_scheduler.DynamicPersistentVarlenScheduler
    )

    for scheduler in (
        tile_scheduler.SingleTileLPTScheduler,
        tile_scheduler.DynamicPersistentVarlenScheduler,
    ):
        parameters = inspect.signature(scheduler.create).parameters
        assert "ctx" in parameters
        assert "clc" not in parameters
