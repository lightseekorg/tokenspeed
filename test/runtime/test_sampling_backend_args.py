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

"""``--sampling-backend ascend_full`` needs flash-npu-kernel's `flash_ops` operators.

The backend runs kernels this repository neither ships nor installs, so on a
machine without them the launch is refused while its arguments are validated.
"""

import os
import sys
from unittest import mock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

from tokenspeed.runtime.utils.server_args import (
    ServerArgs,
    _ascend_full_sampling_dependency_error,
)

register_cuda_ci(est_time=10, suite="runtime-1gpu")


def _flash_ops_installed() -> bool:
    try:
        import flash_ops.npu_sample  # noqa: F401
    except ImportError:
        return False
    return True


def test_ascend_full_names_the_package_to_install_when_its_operators_are_absent():
    if _flash_ops_installed():
        pytest.skip("flash_ops is installed, so ascend_full is runnable here")

    assert _ascend_full_sampling_dependency_error() is not None

    with pytest.raises(ValueError, match="flash-npu-kernel") as refused:
        ServerArgs(model="x", sampling_backend="ascend_full")
    assert "--sampling-backend greedy" in str(refused.value)


def test_backend_without_those_operators_starts_regardless():
    args = ServerArgs(model="x", sampling_backend="greedy")

    assert args.sampling_backend == "greedy"


def _as_npu_platform():
    """Report the running platform as an Ascend NPU and NVIDIA as absent."""

    class _NpuPlatform:
        is_nvidia = False
        is_npu = True

    return mock.patch(
        "tokenspeed.runtime.utils.server_args.current_platform",
        return_value=_NpuPlatform(),
    )


def test_npu_default_falls_back_to_greedy_without_those_operators():
    """The NPU default must be runnable from a plain source checkout."""
    if _flash_ops_installed():
        pytest.skip("flash_ops is installed, so ascend_full stays the default")

    with _as_npu_platform():
        args = ServerArgs(model="x")

    assert args.sampling_backend == "greedy"


def test_npu_default_stays_ascend_full_when_the_operators_are_importable():
    """When the package is present, the sampling-respecting default is kept."""
    if not _flash_ops_installed():
        pytest.skip("flash_ops is missing, so the fallback is the expected default")

    with _as_npu_platform():
        args = ServerArgs(model="x")

    assert args.sampling_backend == "ascend_full"
