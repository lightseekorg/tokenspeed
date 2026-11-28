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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import logging
import subprocess
import sys
import types
from dataclasses import dataclass

from tokenspeed.runtime.utils import common


@dataclass(frozen=True)
class _FakeMemoryInfo:
    total: int


class _FakeNVMLError(Exception):
    pass


class _FakeNVMLErrorNotSupported(_FakeNVMLError):
    pass


class _FakePynvmlModule(types.ModuleType):
    NVMLError = _FakeNVMLError
    NVMLError_NotSupported = _FakeNVMLErrorNotSupported
    NVML_DEVICE_MIG_DISABLE = 0
    NVML_DEVICE_MIG_ENABLE = 1

    def __init__(
        self,
        totals_bytes: list[int],
        *,
        fail_init: bool = False,
        mig_modes: list[int | None] | None = None,
    ) -> None:
        super().__init__("pynvml")
        self._totals_bytes = totals_bytes
        self._fail_init = fail_init
        self._mig_modes = (
            [self.NVML_DEVICE_MIG_DISABLE] * len(totals_bytes)
            if mig_modes is None
            else mig_modes
        )
        self.initialized = False
        self.shutdown_count = 0

    def nvmlInit(self) -> None:
        if self._fail_init:
            raise self.NVMLError("NVML unavailable")
        self.initialized = True

    def nvmlShutdown(self) -> None:
        self.initialized = False
        self.shutdown_count += 1

    def nvmlDeviceGetCount(self) -> int:
        return len(self._totals_bytes)

    def nvmlDeviceGetHandleByIndex(self, index: int) -> int:
        return index

    def nvmlDeviceGetMigMode(self, handle: int) -> tuple[int, int]:
        mode = self._mig_modes[handle]
        if mode is None:
            raise self.NVMLError_NotSupported("MIG not supported")
        return mode, mode

    def nvmlDeviceGetMemoryInfo(self, handle: int) -> _FakeMemoryInfo:
        return _FakeMemoryInfo(total=self._totals_bytes[handle])


def test_get_nvgpu_memory_capacity_uses_nvml_without_nvidia_smi(monkeypatch):
    # Include fractional MiB bytes to verify integer-MiB floor parity with nvidia-smi,
    # and include a GPU where nvmlDeviceGetMigMode raises NotSupported.
    fake_nvml = _FakePynvmlModule(
        [
            286720 * 1024 * 1024 + 512,
            278528 * 1024 * 1024 + 4096,
            286720 * 1024 * 1024,
        ],
        mig_modes=[None, _FakePynvmlModule.NVML_DEVICE_MIG_DISABLE, None],
    )
    monkeypatch.setitem(sys.modules, "pynvml", fake_nvml)

    def _forbid_subprocess(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi not found")

    monkeypatch.setattr(subprocess, "run", _forbid_subprocess)

    capacity_mib = common.get_nvgpu_memory_capacity()
    assert capacity_mib == 278528.0
    assert not fake_nvml.initialized
    assert fake_nvml.shutdown_count == 1


def test_get_nvgpu_memory_capacity_falls_back_to_nvidia_smi_on_nvml_error(
    monkeypatch, caplog
):
    fake_nvml = _FakePynvmlModule([], fail_init=True)
    monkeypatch.setitem(sys.modules, "pynvml", fake_nvml)

    def _fake_nvidia_smi(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout="81920\n40960\n",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", _fake_nvidia_smi)

    with caplog.at_level(logging.WARNING):
        assert common.get_nvgpu_memory_capacity() == 40960.0
    assert any("NVML" in record.message for record in caplog.records)


def test_get_nvgpu_memory_capacity_falls_back_when_mig_enabled(monkeypatch):
    fake_nvml = _FakePynvmlModule(
        [81920 * 1024 * 1024],
        mig_modes=[_FakePynvmlModule.NVML_DEVICE_MIG_ENABLE],
    )
    monkeypatch.setitem(sys.modules, "pynvml", fake_nvml)

    def _fake_nvidia_smi(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout="10240\n",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", _fake_nvidia_smi)

    assert common.get_nvgpu_memory_capacity() == 10240.0
    assert not fake_nvml.initialized
    assert fake_nvml.shutdown_count == 1


def test_get_nvgpu_memory_capacity_falls_back_when_pynvml_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", None)

    def _fake_nvidia_smi(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout="49152\n",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", _fake_nvidia_smi)

    assert common.get_nvgpu_memory_capacity() == 49152.0
