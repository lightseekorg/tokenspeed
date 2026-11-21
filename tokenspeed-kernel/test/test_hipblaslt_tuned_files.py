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

"""Committed TunableOp BLAS files, their lookup and the online-tuning switch; no GPU."""

from __future__ import annotations

import contextlib
import glob
import os
from unittest.mock import patch

import pytest
import torch
from tokenspeed_kernel.ops.gemm import hipblaslt
from tokenspeed_kernel.ops.gemm._hipblaslt import tuning

# The keys TunableOp stamps on ROCm; a file missing one is rejected at load.
_STAMP_KEYS = {
    "PT_VERSION",
    "HIP_VERSION",
    "HIPBLASLT_VERSION",
    "ROCBLAS_VERSION",
    "GCN_ARCH_NAME",
}


def test_file_is_named_after_arch():
    assert tuning.hipblaslt_tuned_file("gfx950") == os.path.join(
        tuning.TUNED_CONFIG_DIR, "arch=gfx950.csv"
    )


class _Props:
    def __init__(self, gcn_arch_name: str):
        self.gcnArchName = gcn_arch_name


@pytest.mark.parametrize("device_name", ["AMD Instinct MI355X", "AMD Radeon Graphics"])
def test_default_file_ignores_device_name(device_name):
    # Without libdrm's amdgpu.ids (as in some containers) an MI355X reports
    # itself as "AMD Radeon Graphics"; the file is still found by its arch.
    with (
        patch.object(torch.cuda, "get_device_name", return_value=device_name),
        patch.object(
            torch.cuda,
            "get_device_properties",
            return_value=_Props("gfx950:sramecc+:xnack-"),
        ),
    ):
        assert tuning.hipblaslt_tuned_file() == tuning.hipblaslt_tuned_file("gfx950")


def _committed_files() -> list[str]:
    return sorted(glob.glob(os.path.join(tuning.TUNED_CONFIG_DIR, "*.csv")))


def test_gfx950_file_is_committed():
    assert os.path.exists(tuning.hipblaslt_tuned_file("gfx950"))


@pytest.mark.parametrize("path", _committed_files(), ids=os.path.basename)
def test_committed_file_is_wellformed(path):
    name = os.path.basename(path)
    assert name.startswith("arch=gfx") and ":" not in name
    validators = tuning.hipblaslt_file_validators(path)
    assert set(validators) == _STAMP_KEYS
    # The file name's arch is the one TunableOp checks at load.
    arch = name.removeprefix("arch=").removesuffix(".csv")
    assert validators["GCN_ARCH_NAME"].split(":")[0] == arch
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
        patch.object(
            torch.cuda, "get_device_properties", return_value=_Props("gfx000")
        ),
    ):
        # Returns before TunableOp or the GPU are touched.
        assert not tuning.load_hipblaslt_tuned_solutions()


def test_default_load_is_noop_off_rocm():
    hipblaslt._load_default.cache_clear()
    try:
        with (
            patch.object(torch.version, "hip", None),
            patch.object(tuning, "load_hipblaslt_tuned_solutions") as load,
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
                tuning, "load_hipblaslt_tuned_solutions", return_value=False
            ) as load,
        ):
            hipblaslt.load_hipblaslt_tuned_solutions()
            hipblaslt.load_hipblaslt_tuned_solutions()
        load.assert_called_once_with()
    finally:
        hipblaslt._load_default.cache_clear()


class _FakeTunable:
    """The ``torch.cuda.tunable`` calls online tuning makes, with their state."""

    def __init__(self, *, enabled: bool, results: int):
        self.enabled = enabled
        self.tuning = False
        self.filename = "committed.csv" if enabled else ""
        self.rotating_mb = 0
        self.results = [("op", "params", "solution", 1.0)] * results
        self.calls = []

    def is_enabled(self):
        return self.enabled

    def enable(self, value):
        self.calls.append(("enable", value))
        self.enabled = value

    def tuning_enable(self, value):
        self.calls.append(("tuning_enable", value))
        self.tuning = value

    def record_untuned_enable(self, value):
        self.calls.append(("record_untuned_enable", value))

    def set_filename(self, name, insert_device_ordinal=False):
        self.calls.append(("set_filename", name))
        self.filename = name

    def get_rotating_buffer_size(self):
        return self.rotating_mb

    def set_rotating_buffer_size(self, mb):
        self.rotating_mb = mb

    def get_results(self):
        return self.results


@contextlib.contextmanager
def _patched_tunable(fake):
    with (
        patch.object(torch.cuda, "tunable", fake),
        patch.object(torch.cuda, "synchronize"),
    ):
        yield


def test_online_tuning_extends_loaded_file_in_memory():
    fake = _FakeTunable(enabled=True, results=30)
    with _patched_tunable(fake):
        with tuning.hipblaslt_online_tuning(cold_bytes=512 * 2**20):
            assert fake.tuning and fake.enabled
            # New results go nowhere on disk; the committed file stays as is.
            assert fake.filename == os.devnull
            assert fake.rotating_mb == 512
            fake.results = fake.results + [("op", "new", "solution", 1.0)]
    assert not fake.tuning and fake.enabled  # replay continues
    assert fake.filename == os.devnull
    assert fake.rotating_mb == 0
    # The loaded file's TunableOp state is kept, not reset.
    assert ("enable", True) not in fake.calls


def test_online_tuning_without_file_avoids_default_file():
    fake = _FakeTunable(enabled=False, results=0)
    with _patched_tunable(fake):
        with tuning.hipblaslt_online_tuning():
            assert fake.tuning and fake.enabled
    # The file name is set before TunableOp's first lookup reads it.
    assert fake.calls.index(("set_filename", os.devnull)) < fake.calls.index(
        ("enable", True)
    )
    assert not fake.tuning and fake.enabled


def test_online_tuning_stops_on_error():
    fake = _FakeTunable(enabled=True, results=30)
    with _patched_tunable(fake):
        with pytest.raises(RuntimeError):
            with tuning.hipblaslt_online_tuning():
                raise RuntimeError("warmup failed")
    assert not fake.tuning and fake.rotating_mb == 0


@pytest.mark.parametrize(
    "hip, env, expected",
    [("7.2", None, False), ("7.2", "0", False), ("7.2", "1", True), (None, "1", False)],
)
def test_autotune_flag(monkeypatch, hip, env, expected):
    if env is None:
        monkeypatch.delenv(hipblaslt.AUTOTUNE_ENV, raising=False)
    else:
        monkeypatch.setenv(hipblaslt.AUTOTUNE_ENV, env)
    with patch.object(torch.version, "hip", hip):
        assert hipblaslt.hipblaslt_autotune_enabled() is expected


def test_online_tuning_is_noop_off_rocm():
    with (
        patch.object(torch.version, "hip", None),
        patch.object(tuning, "hipblaslt_online_tuning") as online,
    ):
        with hipblaslt.hipblaslt_online_tuning():
            pass
    online.assert_not_called()
