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

"""Real GEMM calls verify shared startup coverage and cache reuse."""

import pytest
from gemm_capacity_helpers import (
    assert_rows_reuse_tuned_configs,
    isolated_tuner,
    profiled_rows,
    requires_sm100_gemms,
    round_trip_persisted_configs,
    run_gemm,
    tune_startup,
)
from tokenspeed_kernel.ops.tuning import autotune

# The other GEMM formats run post-submit from test_gemm_autotune_capacity_formats.py.
pytestmark = requires_sm100_gemms


@pytest.fixture
def tuner(monkeypatch, caplog):
    with isolated_tuner(monkeypatch, caplog) as isolated:
        yield isolated


def test_gemm_capacity(tuner, tmp_path, caplog):
    tune_startup(tuner, "fp8", "fp8_gemm")
    round_trip_persisted_configs(tuner, "fp8", "fp8_gemm", tmp_path)
    assert_rows_reuse_tuned_configs(tuner, "fp8", caplog)


def test_tuning_controls_and_cleanup(tuner):
    from flashinfer.autotuner import autotune as native_autotune

    with autotune(tune_mode=False, tuning_buckets=None, round_up=None):
        run_gemm("fp8", 1)
    assert not tuner.profiling_cache

    with pytest.raises(RuntimeError, match="leave startup"), autotune(
        tune_mode=True, tuning_buckets=(1, 4), round_up=True
    ):
        run_gemm("fp8", 1)
        raise RuntimeError("leave startup")
    assert profiled_rows(tuner, "fp8_gemm") == {1, 4}
    assert not tuner.is_tuning_mode

    tuner.clear_cache()
    with native_autotune(tune_mode=True):
        run_gemm("fp8", 1)
    assert profiled_rows(tuner, "fp8_gemm") == {1}
