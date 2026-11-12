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

"""Committed TunableOp BLAS files and their lookup; runs without a GPU."""

from __future__ import annotations

import glob
import os
from unittest.mock import patch

import pytest
import torch

hipblaslt_amd = pytest.importorskip("tokenspeed_kernel_amd.ops.gemm.hipblaslt")
from tokenspeed_kernel.ops.gemm import hipblaslt  # noqa: E402

# The keys TunableOp stamps on ROCm; a file missing one is rejected at load.
_STAMP_KEYS = {
    "PT_VERSION",
    "HIP_VERSION",
    "HIPBLASLT_VERSION",
    "ROCBLAS_VERSION",
    "GCN_ARCH_NAME",
}


def test_file_is_named_after_device():
    assert hipblaslt_amd.hipblaslt_tuned_file("AMD Instinct MI355X") == os.path.join(
        hipblaslt_amd.TUNED_CONFIG_DIR, "device_name=AMD_Instinct_MI355X.csv"
    )


def _committed_files() -> list[str]:
    return sorted(glob.glob(os.path.join(hipblaslt_amd.TUNED_CONFIG_DIR, "*.csv")))


def test_mi355x_file_is_committed():
    assert os.path.exists(hipblaslt_amd.hipblaslt_tuned_file("AMD Instinct MI355X"))


@pytest.mark.parametrize("path", _committed_files(), ids=os.path.basename)
def test_committed_file_is_wellformed(path):
    name = os.path.basename(path)
    assert name.startswith("device_name=") and " " not in name
    assert set(hipblaslt_amd.hipblaslt_file_validators(path)) == _STAMP_KEYS
    entries = 0
    with open(path) as f:
        for line in f:
            fields = line.rstrip("\n").split(",")
            assert len(fields) in (3, 4), line
            if fields[0] == "Validator":
                continue
            op, params, solution, ms = fields
            assert op.startswith("GemmTunableOp_"), line
            assert solution.startswith(("Gemm_Hipblaslt_", "Gemm_Rocblas_")), line
            assert float(ms) > 0
            entries += 1
    assert entries > 0


def test_gpu_without_file_is_skipped():
    with (
        patch.object(torch.version, "hip", "7.2"),
        patch.object(torch.cuda, "get_device_name", return_value="No Such GPU"),
    ):
        # Returns before TunableOp or the GPU are touched.
        assert not hipblaslt_amd.load_hipblaslt_tuned_solutions()


def test_default_load_is_noop_off_rocm():
    hipblaslt._load_default.cache_clear()
    try:
        with (
            patch.object(torch.version, "hip", None),
            patch.object(hipblaslt_amd, "load_hipblaslt_tuned_solutions") as load,
        ):
            assert not hipblaslt.load_hipblaslt_tuned_solutions()
            assert not hipblaslt.load_hipblaslt_tuned_solutions()
        load.assert_not_called()
    finally:
        hipblaslt._load_default.cache_clear()


def test_default_load_runs_once_per_process():
    hipblaslt._load_default.cache_clear()
    try:
        with (
            patch.object(torch.version, "hip", "7.2"),
            patch.object(
                hipblaslt_amd, "load_hipblaslt_tuned_solutions", return_value=False
            ) as load,
        ):
            hipblaslt.load_hipblaslt_tuned_solutions()
            hipblaslt.load_hipblaslt_tuned_solutions()
        load.assert_called_once_with()
    finally:
        hipblaslt._load_default.cache_clear()
