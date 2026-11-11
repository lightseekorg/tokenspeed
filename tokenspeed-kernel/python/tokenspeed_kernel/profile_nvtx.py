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

"""Native NVTX identities and cost metadata; Nsight owns GPU timing."""

from __future__ import annotations

import json
import os
import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from functools import cache
from importlib.metadata import PackageNotFoundError, version
from itertools import count
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

import torch
from tokenspeed_kernel.op_profile import CostRegistry, ProfileSession
from tokenspeed_kernel.profile_utilization import HardwareProfile

if TYPE_CHECKING:
    from nvtx import Domain

_scope_ids = count(1)
PROFILE_METADATA_COPY_LABEL = "Profiler integer metadata D2H"


def _next_scope_id() -> int:
    scope_id = next(_scope_ids)
    if not 0 <= scope_id < (1 << 63):
        raise OverflowError("Native NVTX scope_id must be in [0, 2**63 - 1]")
    return scope_id


class NvtxUnavailableError(RuntimeError):
    """Eager profiling cannot emit native NVTX ranges in this process."""


@cache
def _nvtx_domain() -> Domain:
    try:
        import nvtx
    except ModuleNotFoundError as error:
        if error.name != "nvtx":
            raise
        raise NvtxUnavailableError(
            "Eager native profiling requires NVIDIA nvtx==0.2.17"
        ) from error
    try:
        installed = version("nvtx")
    except PackageNotFoundError:
        installed = None
    if installed != "0.2.17":
        raise NvtxUnavailableError(
            f"Eager native profiling requires NVIDIA nvtx==0.2.17 (found {installed})"
        )
    domain = nvtx.get_domain("TokenSpeed") if nvtx.enabled() else None
    if not isinstance(domain, nvtx.Domain):
        raise NvtxUnavailableError(
            "TokenSpeed NVTX domain is inactive; run under Nsight Systems "
            "2026.5+ with --trace=cuda,nvtx and without NVTX_DISABLE"
        )
    return domain


@dataclass(frozen=True)
class _RecordedScope:
    trace_label: str
    nvtx_label: str
    nvtx_id: int


@dataclass(frozen=True)
class _MetadataCopyScope(_RecordedScope):
    bytes: int


@dataclass
class _IntegerSnapshot:
    target: dict[str, tuple[int, ...] | None]
    name: str
    host: torch.Tensor


class NvtxTimeline:
    """Record native NVTX identities and costs without measuring GPU time.

    Nsight timestamps ranges and CUDA activities. After capture, profile_nsys
    joins its GPU projection to these identities and calculates utilization.

    Args:
        session: An unused ProfileSession with backend='nvtx'.
        costs: Explicit implementation cost rules; missing costs remain N/A.
        hardware: Per-device peaks with matching compute types and source.
        stream: The execution stream, also used for explicit metadata snapshots.
        max_scopes: Maximum number of op and layer calls together.
        output: Cost metadata accompanying nsys-rep; GPU metrics are pending.
        max_metadata_bytes: Pinned-host capacity for explicitly requested integer
            tensor snapshots, allocated before collection. Exhaustion raises.
        eager_step: Serving step identity and config hash. Its existing
            result event gates CPU-side finalization instead of a stream wait.
    """

    def __init__(
        self,
        session: ProfileSession,
        *,
        costs: CostRegistry,
        hardware: HardwareProfile,
        stream: torch.cuda.Stream,
        max_scopes: int,
        output: Path,
        max_metadata_bytes: int = 1 << 20,
        eager_step: dict | None = None,
    ) -> None:
        if session.backend != "nvtx":
            raise ValueError("Native timeline requires backend='nvtx'")
        if max_scopes < 1:
            raise ValueError("max_scopes must be positive")
        if type(max_metadata_bytes) is not int or max_metadata_bytes < 1:
            raise ValueError("max_metadata_bytes must be a positive integer")
        self.session = session
        self.costs = costs
        self.hardware = (
            hardware.validate_cuda_device(stream.device)
            if hardware.schema_version == 2
            else hardware
        )
        self.stream = stream
        self.max_scopes = max_scopes
        self.output = output
        self.max_metadata_bytes = max_metadata_bytes
        self.eager_step = eager_step
        self._device_name: str | None = None
        self._stream_id: int | None = None
        self._deferred = False
        self._domain: Domain | None = None
        self._thread: int | None = None
        self._scopes: list[_RecordedScope] = []
        self._metadata_copy_scopes: list[_MetadataCopyScope] = []
        self._snapshots: list[_IntegerSnapshot] = []
        self._unavailable_snapshots: list[tuple[int, str, str]] = []
        self._metadata_buffer: torch.Tensor | None = None
        self._metadata_used = 0
        self._entered = False
        self._finished = False
        self._observer_context = None

    def _check_context(self) -> None:
        if threading.get_native_id() != self._thread:
            raise RuntimeError("Native timeline must stay on its execution thread")
        if torch.cuda.current_stream() != self.stream:
            raise RuntimeError("Native timeline requires its declared current stream")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Native timeline supports eager execution, not CUDA graphs"
            )

    def __enter__(self) -> NvtxTimeline:
        if self._entered:
            raise RuntimeError("Nested or reused native timeline is unsupported")
        if self.session.ops or self.session.layers:
            raise ValueError("Native timeline requires an unused ProfileSession")
        self._entered = True
        self._thread = threading.get_native_id()
        self._check_context()
        self._domain = _nvtx_domain()
        try:
            self._device_name = torch.cuda.get_device_name(self.stream.device)
            self._stream_id = self.stream.cuda_stream
            self._metadata_buffer = torch.empty(
                self.max_metadata_bytes,
                dtype=torch.uint8,
                device="cpu",
                pin_memory=True,
            )
            self._observer_context = self.session.observe(self)
            self._observer_context.__enter__()
        except BaseException:
            self._metadata_buffer = None
            self._observer_context = None
            self._domain = None
            raise
        return self

    def snapshot_integer_selection(self, producer, target, inputs, name):
        from tokenspeed_kernel.op_profile import _describe

        source = producer()
        if (
            source.dtype not in (torch.int32, torch.int64)
            or not source.is_contiguous()
            or not 0 < source.numel() <= 4096
        ):
            raise ValueError("Integer selection must be bounded and contiguous")
        _describe(source, name, inputs)
        self.snapshot_integer_values(source.view(-1), target, name)

    def snapshot_integer_values(
        self, source: torch.Tensor, target: dict[str, tuple[int, ...] | None], name: str
    ) -> None:
        """Queue a validated integer vector before the op window; never wait here."""
        self._check_context()
        if source.device.type == "cpu":
            target[name] = tuple(source.tolist())
            return
        if self._finished or self._metadata_buffer is None:
            raise RuntimeError("Native timeline is not collecting")
        if source.device != self.stream.device:
            if self.eager_step is not None:
                self._unavailable_snapshots.append(
                    (id(target), name, "integer_argument_on_other_device")
                )
            return  # Do not copy unordered metadata from another device.
        start = (self._metadata_used + 7) // 8 * 8
        end = start + source.numel() * source.element_size()
        if end > self.max_metadata_bytes:
            if self.eager_step is None:
                raise RuntimeError("Native timeline max_metadata_bytes exhausted")
            self._unavailable_snapshots.append(
                (id(target), name, "eager_metadata_byte_capacity")
            )
            return
        host = self._metadata_buffer[start:end].view(source.dtype)
        if self.eager_step is None or start == end:
            host.copy_(source, non_blocking=True)
        else:
            domain = self._domain
            if domain is None:
                raise RuntimeError(
                    "Eager metadata copy requires its active NVTX domain"
                )
            scope_id = _next_scope_id()
            domain.push_range(message=PROFILE_METADATA_COPY_LABEL, payload=scope_id)
            try:
                host.copy_(source, non_blocking=True)
            finally:
                domain.pop_range()
            self._metadata_copy_scopes.append(
                _MetadataCopyScope(
                    f"profiler.metadata_copy/{scope_id}",
                    PROFILE_METADATA_COPY_LABEL,
                    scope_id,
                    end - start,
                )
            )
        source.record_stream(self.stream)
        self._metadata_used = end
        self._snapshots.append(_IntegerSnapshot(target, name, host))

    @contextmanager
    def record_func(self, label: str, *, display_name: str) -> Iterator[None]:
        """Record one scope without synchronizing or calculating costs."""
        self._check_context()
        domain = self._domain
        if self._finished or domain is None:
            raise RuntimeError("Native timeline is not collecting")
        if len(self._scopes) >= self.max_scopes:
            raise RuntimeError("Native timeline max_scopes exhausted")
        # A native numeric payload joins to the full identity in the sidecar;
        # repeated calls need no extra text on the visible timeline.
        scope_id = _next_scope_id()
        domain.push_range(message=display_name, payload=scope_id)
        error = None
        try:
            self._scopes.append(_RecordedScope(label, display_name, scope_id))
            yield
        except BaseException as exc:
            error = exc
            raise
        finally:
            try:
                if threading.get_native_id() != self._thread:
                    raise RuntimeError(
                        "Native NVTX range must close on its execution thread"
                    )
                domain.pop_range()
            except BaseException as close_error:
                if error is None:
                    raise
                error.add_note(f"Native NVTX range close failed: {close_error}")
        self._check_context()

    def finish(self) -> None:
        """Resolve optional metadata snapshots and save costs exactly once."""
        if self._finished:
            return
        if self.eager_step is not None:
            raise RuntimeError("Serving eager metadata must await the result event")
        if (
            not self._entered
            or ProfileSession.current() is self.session
            or self._domain is None
            or not self.session.observed_by(self)
        ):
            raise RuntimeError(
                "Finish outside iteration windows while capture is active"
            )
        self._check_context()
        if self._snapshots:
            self.stream.synchronize()
        self._write_report()

    def verify_result_stream(self) -> None:
        """Require the existing serving result event to follow the queued copies."""
        if not self._deferred or self._finished:
            raise RuntimeError("Eager timeline did not defer metadata finalization")
        self._check_context()

    def finish_after_results(self) -> None:
        """Resolve pinned data after the result owner's existing synchronization."""
        if not self._deferred or self._finished:
            raise RuntimeError("Eager timeline is not awaiting ready results")
        try:
            self._write_report()
        finally:
            self._snapshots.clear()
            self._metadata_buffer = None
            self._deferred = False

    def _write_report(self) -> None:
        if not self._scopes:
            raise RuntimeError(
                "No op/layer scopes were recorded; native timeline is empty"
            )
        for snapshot in self._snapshots:
            snapshot.target[snapshot.name] = tuple(snapshot.host.tolist())
        self._snapshots.clear()
        report = self.session.report(costs=self.costs)
        hardware = self.hardware
        if hardware.schema_version == 1 and hardware.selection is None:
            hardware = hardware.validate_device({"name": self._device_name})
        report["hardware"] = asdict(hardware)
        report["device_analysis"] = {
            "timing_kind": "pending_nsight",
            "nvtx_payload_type": "int64",
            "process_id": os.getpid(),
            "thread_id": self._thread,
            "device": str(self.stream.device),
            "device_name": self._device_name,
            "stream": self._stream_id,
            "metadata_snapshot_bytes": self._metadata_used,
            "scopes": [asdict(scope) for scope in self._scopes],
        }
        if self.eager_step is not None:
            targets = {id(op.tensor_values): op.trace_label for op in self.session.ops}
            report["device_analysis"]["metadata_copy_scopes"] = [
                asdict(scope) for scope in self._metadata_copy_scopes
            ]
            report["device_analysis"]["unavailable_integer_metadata"] = [
                {"trace_label": targets[key], "name": name, "reason": reason}
                for key, name, reason in self._unavailable_snapshots
            ]
            report["eager_step"] = {
                **self.eager_step,
                "schema": "tokenspeed.profile_eager_step/v1",
            }
        self.output.parent.mkdir(parents=True, exist_ok=True)
        mode = "x" if self.eager_step is not None else "w"
        with self.output.open(mode) as output:
            output.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
        self._finished = True

    def __exit__(self, exc_type, exc, traceback) -> None:
        try:
            if self.eager_step is None:
                self.finish()
            elif exc is None:
                self._check_context()
                self._deferred = True
        except BaseException as error:
            if exc is None:
                raise
            exc.add_note(f"Native timeline finalization failed: {error}")
        finally:
            if self._observer_context is not None:
                self._observer_context.__exit__(exc_type, exc, traceback)
                self._observer_context = None
            if not self._deferred:
                self._snapshots.clear()
                self._metadata_buffer = None
            self._domain = None
