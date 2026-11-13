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
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from kernel_test_selection import selection_from_env
from tokenspeed_kernel.platform import (
    ArchVersion,
    InterconnectInfo,
    PlatformInfo,
)
from tokenspeed_kernel.registry import KernelRegistry
from tokenspeed_kernel.selection import (
    _global_overrides,
    _logged_overrides,
    _oracles,
)
from utils import make_sample_specs

# On pull requests CI lists the changed paths; run only the test files they name.
_SELECTION = selection_from_env(Path(__file__).resolve().parents[2])
# Whether the selection deselected every collected test of this run.
_SELECTION_EMPTIED_RUN = pytest.StashKey[bool]()


def pytest_report_header(config: pytest.Config) -> str | None:
    if _SELECTION is None:
        return None
    if _SELECTION.run_all:
        return "kernel test selection: every test (a change names no family)"
    families = ", ".join(sorted(_SELECTION.families)) or "none"
    return f"kernel test selection: families {families}"


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if _SELECTION is None:
        return
    selected = {}
    kept, deselected = [], []
    for item in items:
        if item.path not in selected:
            selected[item.path] = _SELECTION.selects(item.path)
        (kept if selected[item.path] else deselected).append(item)
    config.stash[_SELECTION_EMPTIED_RUN] = bool(deselected) and not kept
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = kept


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    # A change that names none of this run's test files leaves nothing to run;
    # a run that collected nothing by itself still fails.
    if exitstatus == pytest.ExitCode.NO_TESTS_COLLECTED and session.config.stash.get(
        _SELECTION_EMPTIED_RUN, False
    ):
        session.exitstatus = pytest.ExitCode.OK


@pytest.fixture
def set_matmul_precision() -> Iterator[Callable[[str], None]]:
    # PyTorch stores this setting outside __dict__, so monkeypatch cannot undo it.
    previous = torch.backends.cuda.matmul.fp32_precision

    def set_precision(precision: str) -> None:
        torch.backends.cuda.matmul.fp32_precision = precision

    try:
        yield set_precision
    finally:
        torch.backends.cuda.matmul.fp32_precision = previous


@pytest.fixture
def require() -> Callable[[str, str, str, torch.dtype, str], None]:
    def _require(
        family: str,
        mode: str,
        solution: str,
        dtype: torch.dtype,
        dtype_role: str,
    ) -> None:
        from tokenspeed_kernel.platform import current_platform

        specs = [
            spec
            for spec in KernelRegistry.get().get_for_operator(
                family,
                mode,
                platform=current_platform(),
                solution=solution,
            )
            if spec.format_signatures_for_storage_dtype(dtype, dtype_role)
        ]
        if not specs:
            pytest.skip(f"{family}.{mode} solution {solution!r} is not registered")

    return _require


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
def mi450_platform() -> PlatformInfo:
    return PlatformInfo(
        vendor="amd",
        arch_version=ArchVersion(12, 5),
        device_name="AMD Instinct MI455X",
        device_count=8,
        total_memory=432 * (1024**3),
        memory_bandwidth=23300.0,
        sm_count=256,
        max_threads_per_sm=2048,
        max_shared_memory_per_sm=320 * 1024,
        sm_features=frozenset(
            {
                "tensor_core:f16",
                "tensor_core:f8",
                "tensor_core:f4",
                "memory:async_copy",
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
def b300_platform() -> PlatformInfo:
    return PlatformInfo(
        vendor="nvidia",
        arch_version=ArchVersion(10, 3),
        device_name="NVIDIA B300",
        device_count=8,
        total_memory=288 * (1024**3),
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
def fresh_registry(monkeypatch: pytest.MonkeyPatch):
    # Isolate tests with an empty singleton while preserving the built-in
    # registry for later integration tests in the same pytest process.
    monkeypatch.setattr(KernelRegistry, "_instance", None)
    _oracles.clear()
    _global_overrides.clear()
    _logged_overrides.clear()
    yield
    KernelRegistry.reset()
    _oracles.clear()
    _global_overrides.clear()
    _logged_overrides.clear()


@pytest.fixture
def sample_specs():
    return make_sample_specs()


@pytest.fixture
def device() -> str:
    return "cuda"
