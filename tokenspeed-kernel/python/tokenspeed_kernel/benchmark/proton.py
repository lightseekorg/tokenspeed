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

import json
import math
import statistics
import tempfile
from collections import defaultdict
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Self

from tokenspeed_kernel._triton import proton as _proton
from tokenspeed_kernel.benchmark.profiler import ProfilePhase

__all__ = ["ProtonProfiler", "aggregate_proton_profiles"]

_FLOP_METRICS = ("flops", "flops4", "flops8", "flops16", "flops32", "flops64")
_METADATA_KERNELS = {"<metric>", "__proton_launch_metadata"}

InvocationProfile = dict[str, Any]
ProfileAggregator = Callable[[Sequence[InvocationProfile]], dict[str, Any]]


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile / 100.0
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _summary(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.fmean(values)),
        "p50": float(_percentile(values, 50.0)),
        "p90": float(_percentile(values, 90.0)),
    }


def aggregate_proton_profiles(
    invocations: Sequence[InvocationProfile],
) -> dict[str, Any]:
    """Summarize per-invocation Proton records by runtime kernel symbol."""

    grouped: dict[tuple[tuple[str, ...], str, int], dict[str, list[float]]] = {}
    for invocation in invocations:
        for kernel in invocation["kernels"]:
            key = (
                tuple(kernel["scope_path"]),
                kernel["name"],
                kernel["occurrence"],
            )
            measurements = grouped.setdefault(key, defaultdict(list))
            measurements["device_time_us"].append(kernel["device_time_us"])
            measurements["launches"].append(float(kernel.get("launches", 1)))
            duration_us = kernel["device_time_us"]
            for metric in _FLOP_METRICS:
                if metric in kernel:
                    measurements[f"t{metric}"].append(
                        kernel[metric] / duration_us / 1_000_000.0
                    )

    kernels = []
    for (scope_path, name, occurrence), measurements in grouped.items():
        kernel: dict[str, Any] = {
            "scope_path": list(scope_path),
            "name": name,
            "occurrence": occurrence,
            "samples": len(measurements["device_time_us"]),
            "device_time_us": _summary(measurements["device_time_us"]),
        }
        launch_counts = {int(value) for value in measurements["launches"]}
        if launch_counts != {1}:
            kernel["launches_per_invocation"] = (
                next(iter(launch_counts))
                if len(launch_counts) == 1
                else sorted(launch_counts)
            )
        for metric, values in measurements.items():
            if metric not in ("device_time_us", "launches"):
                kernel[metric] = _summary(values)
        kernels.append(kernel)

    return {
        "provider": "proton",
        "execution_mode": "graph_replay",
        "invocations": len(invocations),
        "kernels": kernels,
    }


class ProtonProfiler:
    """Collect eager launch metadata and measured graph-replay timings."""

    def __init__(
        self,
        *,
        backend: str | None = "roctracer",
        aggregate: ProfileAggregator = aggregate_proton_profiles,
        proton_module: Any = _proton,
        synchronize: Callable[[], None] | None = None,
    ) -> None:
        self.backend = backend
        self._aggregate = aggregate
        self._proton = proton_module
        self._synchronize = synchronize or self._synchronize_device
        self._temporary_directory: tempfile.TemporaryDirectory[str] | None = None
        self._metadata_output: Path | None = None
        self._replay_output: Path | None = None
        self._metadata_session: int | None = None
        self._replay_session: int | None = None
        self._scope_counter = 0
        self._scopes: dict[str, tuple[str, ProfilePhase, int | None]] = {}
        self.profiles: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _synchronize_device() -> None:
        import torch

        torch.cuda.synchronize()

    def __enter__(self) -> Self:
        if self._proton is None:
            raise RuntimeError("Proton is not available")

        self.profiles = {}
        self._scopes = {}
        self._scope_counter = 0
        self._temporary_directory = tempfile.TemporaryDirectory(
            prefix="tokenspeed-benchmark-proton-"
        )
        directory = Path(self._temporary_directory.name)
        self._metadata_output = directory / "metadata"
        self._replay_output = directory / "replay"
        try:
            self._metadata_session = self._proton.start(
                str(self._metadata_output),
                data="trace",
                backend=self.backend,
                hook="triton",
            )
            if self._metadata_session is None:
                raise RuntimeError("Proton did not start a profiling session")
            self._proton.deactivate(self._metadata_session)
            self._replay_session = self._proton.start(
                str(self._replay_output),
                data="tree",
                backend=self.backend,
            )
            if self._replay_session is None:
                raise RuntimeError("Proton did not start a replay profiling session")
            self._proton.deactivate(self._replay_session)
        except BaseException:
            self._temporary_directory.cleanup()
            self._temporary_directory = None
            self._metadata_output = None
            self._replay_output = None
            raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
    ) -> None:
        try:
            if self._metadata_session is not None and self._metadata_output is not None:
                self._proton.finalize(self._metadata_session, "chrome_trace")
            if self._replay_session is not None and self._replay_output is not None:
                self._proton.finalize(self._replay_session, "hatchet")
            if self._metadata_output is not None and self._replay_output is not None:
                self.profiles = self._read_profiles(
                    Path(f"{self._metadata_output}.chrome_trace"),
                    Path(f"{self._replay_output}.hatchet"),
                )
        finally:
            self._metadata_session = None
            self._replay_session = None
            self._metadata_output = None
            self._replay_output = None
            if self._temporary_directory is not None:
                self._temporary_directory.cleanup()
                self._temporary_directory = None

    @contextmanager
    def profile(
        self,
        case_id: str,
        phase: ProfilePhase,
        invocation_index: int | None,
    ):
        if self._metadata_session is None or self._replay_session is None:
            raise RuntimeError("ProtonProfiler must be entered before profiling")

        if phase == "measurement":
            self._proton.activate(self._replay_session)
            try:
                yield
            finally:
                self._proton.deactivate(self._replay_session)
            return

        marker = f"__tokenspeed_benchmark_{phase}_{self._scope_counter}"
        self._scope_counter += 1
        self._scopes[marker] = (case_id, phase, invocation_index)
        if phase == "eager_metadata":
            self._proton.activate(self._metadata_session)
        try:
            with self._proton.scope(marker):
                try:
                    yield
                finally:
                    if phase == "eager_metadata":
                        self._synchronize()
        finally:
            if phase == "eager_metadata":
                self._proton.deactivate(self._metadata_session)

    def _read_profiles(
        self, metadata_path: Path, replay_path: Path
    ) -> dict[str, dict[str, Any]]:
        metadata_by_case = self._read_metadata(metadata_path)
        replays_by_case = self._read_replays(replay_path)

        profiles = {}
        for case_id, replay_invocations in replays_by_case.items():
            metadata = metadata_by_case.get(case_id, {})
            for name, matching in metadata.items():
                replay_kernels = [
                    next(
                        (
                            kernel
                            for kernel in invocation["kernels"]
                            if kernel["name"] == name
                        ),
                        None,
                    )
                    for invocation in replay_invocations
                ]
                if not all(
                    kernel is not None and kernel["launches"] == len(matching)
                    for kernel in replay_kernels
                ):
                    continue
                for metric in _FLOP_METRICS:
                    if all(metric in item for item in matching):
                        total = sum(item[metric] for item in matching)
                        for kernel in replay_kernels:
                            kernel[metric] = total
            profiles[case_id] = self._aggregate(replay_invocations)
        return profiles

    def _read_metadata(
        self, path: Path
    ) -> dict[str, dict[str, list[dict[str, float]]]]:
        events = json.loads(path.read_text(encoding="utf-8"))["traceEvents"]
        by_case: dict[str, dict[str, list[dict[str, float]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for event in events:
            if event.get("cat") != "kernel" or event.get("name") in _METADATA_KERNELS:
                continue
            call_stack = event["args"]["call_stack"]
            marker = next(
                (scope for scope in call_stack if scope in self._scopes), None
            )
            if marker is None:
                continue
            case_id, phase, _ = self._scopes[marker]
            if phase != "eager_metadata":
                continue
            metrics = event["args"].get("metrics", {})
            by_case[case_id][event["name"]].append(
                {
                    metric: float(metrics[metric])
                    for metric in _FLOP_METRICS
                    if metric in metrics
                }
            )
        return by_case

    def _read_replays(self, path: Path) -> dict[str, list[InvocationProfile]]:
        root = json.loads(path.read_text(encoding="utf-8"))[0]
        by_case: dict[str, list[InvocationProfile]] = defaultdict(list)
        for marker_node in self._marker_nodes(root):
            marker = marker_node["frame"]["name"]
            case_id, phase, invocation_index = self._scopes[marker]
            if phase != "graph_replay":
                continue

            kernels: dict[str, dict[str, Any]] = {}
            for node in self._device_nodes(marker_node):
                name = node["frame"]["name"]
                metrics = node["metrics"]
                kernel = kernels.setdefault(
                    name,
                    {
                        "scope_path": [],
                        "name": name,
                        "occurrence": 0,
                        "device_time_us": 0.0,
                        "launches": 0,
                    },
                )
                kernel["device_time_us"] += float(metrics["time (ns)"]) / 1000.0
                kernel["launches"] += int(metrics.get("count", 1))
            by_case[case_id].append(
                {
                    "phase": phase,
                    "invocation_index": invocation_index,
                    "kernels": list(kernels.values()),
                }
            )

        for invocations in by_case.values():
            invocations.sort(key=lambda item: item["invocation_index"])
        return by_case

    def _marker_nodes(self, node: dict[str, Any]):
        name = node["frame"]["name"]
        if name in self._scopes:
            yield node
        for child in node.get("children", []):
            yield from self._marker_nodes(child)

    @staticmethod
    def _device_nodes(node: dict[str, Any]):
        for child in node.get("children", []):
            if "device_type" in child.get("metrics", {}):
                yield child
            yield from ProtonProfiler._device_nodes(child)
