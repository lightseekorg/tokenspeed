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

from __future__ import annotations

from types import SimpleNamespace

import pytest
import tokenspeed_kernel.platform as platform_module
import torch
from tokenspeed_kernel.platform import ArchVersion


def _mock_rocm_device(monkeypatch: pytest.MonkeyPatch, arch: str) -> None:
    props = SimpleNamespace(
        gcnArchName=f"{arch}:sramecc+:xnack-",
        name=f"AMD {arch}",
        total_memory=288 * (1024**3),
        multi_processor_count=384,
        max_threads_per_multi_processor=2048,
        max_shared_memory_per_block=65536,
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _device: props)


def test_detect_rocm_platform_rejects_gfx942(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_rocm_device(monkeypatch, "gfx942")

    with pytest.raises(
        RuntimeError,
        match="Detected unsupported AMD GPU architecture 'gfx942'",
    ):
        platform_module._detect_rocm_platform()


def test_detect_rocm_platform_accepts_gfx950(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_rocm_device(monkeypatch, "gfx950")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)
    monkeypatch.setattr(
        platform_module,
        "_get_rocm_runtime_features",
        lambda: frozenset(),
    )
    monkeypatch.setattr(platform_module, "_detect_rocm_interconnect", lambda: None)

    detected = platform_module._detect_rocm_platform()

    assert detected.vendor == "amd"
    assert detected.arch_version == ArchVersion(9, 5)
    assert detected.generation_name == "CDNA4"
    assert detected.max_shared_memory_per_sm == 160 * 1024
    assert detected.sm_features == frozenset(
        {
            "tensor_core:f16",
            "tensor_core:f8",
            "tensor_core:f4",
            "memory:async_copy",
        }
    )
