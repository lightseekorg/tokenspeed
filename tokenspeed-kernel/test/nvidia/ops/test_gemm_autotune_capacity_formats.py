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

"""Startup autotune capacity of the non-FP8 GEMM formats; runs post-submit."""

import pytest
from gemm_capacity_helpers import (
    assert_rows_reuse_tuned_configs,
    isolated_tuner,
    requires_sm100_gemms,
    round_trip_persisted_configs,
    tune_startup,
)

pytestmark = requires_sm100_gemms


@pytest.fixture
def tuner(monkeypatch, caplog):
    with isolated_tuner(monkeypatch, caplog) as isolated:
        yield isolated


@pytest.mark.parametrize(
    "kind,op",
    [
        ("mxfp8", "mxfp8_gemm"),
        ("nvfp4", "fp4_gemm"),
        ("w4a16", "bf16_fp4_cute_dsl_sm100_gemm"),
        ("swiglu", "nvfp4_gemm_swiglu_nvfp4_quant"),
    ],
)
def test_gemm_capacity(tuner, tmp_path, caplog, kind, op):
    tune_startup(tuner, kind, op)
    # MXFP8 persists compound CuTe DSL tactics tuned with a cold L2, unlike FP8.
    if kind == "mxfp8":
        round_trip_persisted_configs(tuner, kind, op, tmp_path)
    assert_rows_reuse_tuned_configs(tuner, kind, caplog)
