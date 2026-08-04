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

"""load_flashinfer_tuning_cache never fails startup.

Every failure mode -- missing file, corrupt JSON, a table swept on a different
GPU model -- must come back as ``False`` (with a warning) so the startup
autotune window tunes those shapes instead of engine startup aborting over a
stale table.
"""

from __future__ import annotations

import json
from importlib.util import find_spec

import pytest
from tokenspeed_kernel.ops.other.tuning.flashinfer import (
    flashinfer_tuning_cache_filename,
    load_flashinfer_tuning_cache,
    set_autotune_process_group,
)

requires_flashinfer = pytest.mark.skipif(
    find_spec("flashinfer") is None, reason="requires flashinfer"
)


def test_flashinfer_tuning_cache_filename_includes_cudnn() -> None:
    assert flashinfer_tuning_cache_filename(
        "kimi-k3",
        8,
        1,
        "NVIDIA B300 SXM6 AC",
        "0.6.16",
        92400,
    ) == (
        "kimi-k3,ep=8,tp=1,device_name=NVIDIA_B300_SXM6_AC,"
        "flashinfer=0.6.16,cudnn=92400.json"
    )


@requires_flashinfer
def test_missing_file_returns_false(tmp_path) -> None:
    assert load_flashinfer_tuning_cache(str(tmp_path / "absent.json")) is False


@requires_flashinfer
def test_corrupt_file_returns_false(tmp_path) -> None:
    path = tmp_path / "corrupt.json"
    path.write_text("{ not json")
    assert load_flashinfer_tuning_cache(str(path)) is False


@requires_flashinfer
def test_packaged_lookup_miss_returns_false() -> None:
    import torch
    from tokenspeed_kernel.ops.other.tuning.flashinfer import (
        load_packaged_flashinfer_tuning_cache,
    )

    if not torch.cuda.is_available():
        pytest.skip("device-name lookup requires CUDA")
    # No table ships for this made-up model; the miss must be a quiet False
    # (INFO log), leaving the startup autotune window to tune these shapes.
    assert (
        load_packaged_flashinfer_tuning_cache("no-such-model-unit-test", 999, 1)
        is False
    )


@requires_flashinfer
def test_set_autotune_process_group_sets_and_clears() -> None:
    import flashinfer.autotuner as fi

    sentinel = object()
    set_autotune_process_group(sentinel)
    try:
        assert fi.get_autotune_process_group() is sentinel
    finally:
        set_autotune_process_group(None)
    assert fi.get_autotune_process_group() is None


def test_set_autotune_process_group_tolerates_missing_backend() -> None:
    # Like autotune(), a no-op without flashinfer installed.
    set_autotune_process_group(None)


@requires_flashinfer
def test_mismatched_gpu_metadata_returns_false(tmp_path) -> None:
    # A definite metadata conflict (wrong GPU model) must reject the whole
    # table -- this is the guard that keeps a B300-swept table off other SKUs.
    path = tmp_path / "wrong_gpu.json"
    path.write_text(
        json.dumps(
            {
                "_metadata": {
                    "flashinfer_version": "0.0.1",
                    "cuda_version": "0.0",
                    "cublas_version": "0",
                    "cudnn_version": "0",
                    "cudnn_frontend_version": "0",
                    "gpu": "NVIDIA UnitTest GPU That Does Not Exist",
                },
            }
        )
    )
    assert load_flashinfer_tuning_cache(str(path)) is False
