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

"""CPU checks of real Ascend GMM registration and initialization-time selection."""

import pytest
from tokenspeed_kernel import selection
from tokenspeed_kernel.ops.gemm.ascend_w4a4 import plan_grouped_w4a4
from tokenspeed_kernel.platform import ArchVersion, PlatformInfo
from tokenspeed_kernel.selection import NoKernelFoundError, kernel_override
from tokenspeed_kernel_npu.ops import dsv41_w4a4_native


@pytest.fixture(autouse=True)
def ascend_platform(monkeypatch):
    # Only the unavailable device and installed native library are simulated.
    platform = PlatformInfo(
        vendor="ascend",
        arch_version=ArchVersion(9, 1),
        device_name="Ascend910B",
        device_count=1,
        total_memory=64 * 1024**3,
        memory_bandwidth=0,
        sm_count=24,
        max_threads_per_sm=0,
        max_shared_memory_per_sm=192 * 1024,
    )
    monkeypatch.setattr(selection, "current_platform", lambda: platform)
    monkeypatch.delenv("TOKENSPEED_KERNEL_OVERRIDE_GEMM_GROUPED_W4A4", raising=False)


@pytest.mark.parametrize("shape", [(5120, 4608), (2304, 5120)])
def test_auto_selects_native_implementation(monkeypatch, shape):
    monkeypatch.setattr(dsv41_w4a4_native, "available", lambda: True)
    assert (
        plan_grouped_w4a4(*shape, solution=None).kernel.name == "ascend_c_grouped_w4a4"
    )


@pytest.mark.parametrize("solution", [None, "ascend_c"])
@pytest.mark.parametrize("use_override", [False, True])
def test_missing_native_library_fails_without_fallback(
    monkeypatch, solution, use_override
):
    def missing_library():
        raise RuntimeError("Required Ascend library dsv41_w4a4 is unavailable")

    monkeypatch.setattr(dsv41_w4a4_native, "available", missing_library)
    if use_override:
        monkeypatch.setenv(
            "TOKENSPEED_KERNEL_OVERRIDE_GEMM_GROUPED_W4A4", "ascend_c_grouped_w4a4"
        )
    with pytest.raises(RuntimeError, match="Required Ascend library"):
        plan_grouped_w4a4(5120, 4608, solution=solution)


def test_unknown_solution_fails(monkeypatch):
    monkeypatch.setattr(dsv41_w4a4_native, "available", lambda: True)
    with pytest.raises(NoKernelFoundError):
        plan_grouped_w4a4(5120, 4608, solution="not-a-solution")


def test_override_cannot_bypass_geometry_checks(monkeypatch):
    monkeypatch.setattr(dsv41_w4a4_native, "available", lambda: True)
    with kernel_override("gemm", "grouped_w4a4", "ascend_c_grouped_w4a4"):
        with pytest.raises(NoKernelFoundError):
            plan_grouped_w4a4(5120, 5120, solution=None)


def test_plan_stays_fixed_when_environment_changes(monkeypatch):
    monkeypatch.setattr(dsv41_w4a4_native, "available", lambda: True)
    plan = plan_grouped_w4a4(5120, 4608, solution="ascend_c")
    monkeypatch.setenv(
        "TOKENSPEED_KERNEL_OVERRIDE_GEMM_GROUPED_W4A4", "cann_grouped_w4a4"
    )
    later = plan_grouped_w4a4(5120, 4608, solution=None)
    assert plan.kernel.name == "ascend_c_grouped_w4a4"
    assert later.kernel.name == "cann_grouped_w4a4"


def test_native_solution_rejects_non_ascend(monkeypatch, h100_platform):
    monkeypatch.setattr(selection, "current_platform", lambda: h100_platform)
    monkeypatch.setattr(dsv41_w4a4_native, "available", lambda: True)
    with pytest.raises(NoKernelFoundError):
        plan_grouped_w4a4(5120, 4608, solution="ascend_c")


@pytest.mark.parametrize("use_override", [False, True])
def test_explicit_cann_does_not_load_native_library(monkeypatch, use_override):
    def unavailable_runtime():
        raise OSError("Native runtime is unavailable")

    monkeypatch.setattr(dsv41_w4a4_native, "available", unavailable_runtime)
    if use_override:
        monkeypatch.setenv(
            "TOKENSPEED_KERNEL_OVERRIDE_GEMM_GROUPED_W4A4", "cann_grouped_w4a4"
        )
    plan = plan_grouped_w4a4(5120, 4608, solution=None if use_override else "cann")
    assert plan.kernel.name == "cann_grouped_w4a4"


@pytest.mark.parametrize("shape", [(128, 64), (5120, 5120), (2304, 4608)])
def test_unsupported_geometry_does_not_load_native_library(monkeypatch, shape):
    def unavailable_runtime():
        raise OSError("Native runtime is unavailable")

    monkeypatch.setattr(dsv41_w4a4_native, "available", unavailable_runtime)
    assert plan_grouped_w4a4(*shape, solution=None).kernel.name == "cann_grouped_w4a4"
    with pytest.raises(NoKernelFoundError):
        plan_grouped_w4a4(*shape, solution="ascend_c")
