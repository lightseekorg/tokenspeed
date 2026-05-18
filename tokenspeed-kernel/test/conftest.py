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

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from tokenspeed_kernel.platform import (
    ArchVersion,
    InterconnectInfo,
    PlatformInfo,
)
from tokenspeed_kernel.registry import KernelRegistry
from tokenspeed_kernel.selection import (
    _global_overrides,
    _oracles,
    clear_config_overrides,
)
from utils import make_sample_specs


@pytest.fixture
def h100_platform() -> PlatformInfo:
    return PlatformInfo(
        vendor="nvidia",
        arch_version=ArchVersion(9, 0),
        device_name="NVIDIA H100",
        device_count=8,
        total_memory=80 * (1024**3),
        memory_bandwidth=3350.0,
        sm_count=132,
        max_threads_per_sm=2048,
        max_shared_memory_per_sm=232448,
        sm_features=frozenset(
            {
                "tensor_core:f16",
                "tensor_core:int8",
                "tensor_core:f8",
                "memory:async_copy",
                "memory:tma",
                "compute:cluster",
            }
        ),
        runtime_features=frozenset({"runtime:cuda_graph"}),
        interconnect=InterconnectInfo(topology="nvlink_full"),
    )


@pytest.fixture
def a100_platform() -> PlatformInfo:
    return PlatformInfo(
        vendor="nvidia",
        arch_version=ArchVersion(8, 0),
        device_name="NVIDIA A100",
        device_count=8,
        total_memory=80 * (1024**3),
        memory_bandwidth=2039.0,
        sm_count=108,
        max_threads_per_sm=2048,
        max_shared_memory_per_sm=167936,
        sm_features=frozenset(
            {
                "tensor_core:f16",
                "tensor_core:int8",
                "memory:async_copy",
            }
        ),
        runtime_features=frozenset({"runtime:cuda_graph"}),
        interconnect=InterconnectInfo(topology="nvlink_full"),
    )


@pytest.fixture
def mi300_platform() -> PlatformInfo:
    return PlatformInfo(
        vendor="amd",
        arch_version=ArchVersion(9, 4),
        device_name="AMD Instinct MI300X",
        device_count=8,
        total_memory=192 * (1024**3),
        memory_bandwidth=5300.0,
        sm_count=304,
        max_threads_per_sm=2048,
        max_shared_memory_per_sm=65536,
        sm_features=frozenset(
            {
                "tensor_core:f16",
                "tensor_core:f8",
            }
        ),
        runtime_features=frozenset(),
        interconnect=InterconnectInfo(topology="pcie"),
    )


@pytest.fixture
def mi350_platform() -> PlatformInfo:
    return PlatformInfo(
        vendor="amd",
        arch_version=ArchVersion(9, 5),
        device_name="AMD Instinct MI350X/MI355X",
        device_count=8,
        total_memory=288 * (1024**3),
        memory_bandwidth=8000.0,
        sm_count=384,
        max_threads_per_sm=2048,
        max_shared_memory_per_sm=65536,
        sm_features=frozenset(
            {
                "tensor_core:f16",
                "tensor_core:f8",
                "tensor_core:f4",
            }
        ),
        runtime_features=frozenset(),
        interconnect=InterconnectInfo(topology="pcie"),
    )


@pytest.fixture
def b200_platform() -> PlatformInfo:
    return PlatformInfo(
        vendor="nvidia",
        arch_version=ArchVersion(10, 0),
        device_name="NVIDIA B200",
        device_count=8,
        total_memory=192 * (1024**3),
        memory_bandwidth=8000.0,
        sm_count=160,
        max_threads_per_sm=2048,
        max_shared_memory_per_sm=262144,
        sm_features=frozenset(
            {
                "tensor_core:f16",
                "tensor_core:int8",
                "tensor_core:f8",
                "tensor_core:f4",
                "memory:async_copy",
                "memory:tma",
                "compute:cluster",
            }
        ),
        runtime_features=frozenset({"runtime:cuda_graph"}),
        interconnect=InterconnectInfo(topology="nvlink_full"),
    )


@pytest.fixture
def fresh_registry():
    KernelRegistry.reset()
    clear_config_overrides()
    _oracles.clear()
    _global_overrides.clear()
    yield
    KernelRegistry.reset()
    clear_config_overrides()
    _oracles.clear()
    _global_overrides.clear()


@pytest.fixture
def sample_specs():
    return make_sample_specs()


@pytest.fixture
def device() -> str:
    return "cuda"
