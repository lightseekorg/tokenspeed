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

"""Optional CUDA event timing and native NVTX submission before capture ends."""

from __future__ import annotations

import ctypes
import json
import subprocess
import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import torch
from tokenspeed_kernel.op_profile import CostRegistry, ProfileSession
from tokenspeed_kernel.profile_utilization import (
    HardwareProfile,
    ScopeTiming,
    analyze_utilization,
)


def build_nvtx_bridge(*, include_dir: Path, output: Path, compiler: str) -> Path:
    """Build the optional C bridge before collection; return its absolute path.

    include_dir contains nvtx3/ from the supported Nsight Systems installation.
    No CUDA compilation, package installation or network access is performed.
    """
    if not (include_dir / "nvtx3/nvToolsExtPayload.h").is_file():
        raise ValueError(f"NVTX payload headers not found in {include_dir}")
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            compiler,
            "-shared",
            "-fPIC",
            "-O2",
            "-I",
            str(include_dir),
            str(Path(__file__).with_name("_nvtx_profile.c")),
            "-ldl",
            "-o",
            str(output),
        ],
        check=True,
    )
    return output


class _Nvtx:
    def __init__(self, library: Path) -> None:
        self.library = ctypes.CDLL(str(library.resolve()))
        self.library.ts_nvtx_open.restype = ctypes.c_void_p
        self.library.ts_nvtx_timestamp.restype = ctypes.c_int64
        self.library.ts_nvtx_submit.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_char_p,
        ]
        self.library.ts_nvtx_submit.restype = ctypes.c_int
        self.library.ts_nvtx_close.argtypes = [ctypes.c_void_p]
        self.library.ts_nvtx_close.restype = None
        self.handle = self.library.ts_nvtx_open()
        if not self.handle:
            raise RuntimeError(
                "Run under Nsight Systems 2026.5+ with --trace=cuda,nvtx"
            )

    def timestamp(self) -> int:
        return self.library.ts_nvtx_timestamp()

    def submit(self, timing: ScopeTiming, label: str) -> None:
        if (
            self.library.ts_nvtx_submit(
                self.handle,
                timing.start_timestamp,
                timing.end_timestamp,
                label.encode("utf-8"),
            )
            != 0
        ):
            raise ValueError(
                "Invalid NVTX time range or label exceeds 4095 UTF-8 bytes"
            )

    def close(self) -> None:
        self.library.ts_nvtx_close(self.handle)
        self.handle = None


@dataclass
class _Pending:
    label: str
    start: int
    end: int
    begin_event: torch.cuda.Event
    end_event: torch.cuda.Event


class NvtxTimeline:
    """Finalize an eager session into the collecting process's native NVTX rows.

    All work must run on stream, or join stream before the op returns. The
    caller must keep Nsight capture active through context exit. This measures
    CUDA event windows, including stream waits and launch gaps, not kernel sums.

    Args:
        session: An unused ProfileSession with backend='nvtx'.
        costs: Explicit implementation cost rules; missing costs remain N/A.
        hardware: Per-device peaks with matching compute types and source.
        stream: The CUDA stream that owns or joins all scoped GPU work.
        max_scopes: Capacity for op and layer calls together. Events are prepared
            before collection; exhaustion raises instead of allocating mid-scope.
        library: Prebuilt optional NVTX bridge.
        output: Metadata, assumptions and metrics JSON accompanying nsys-rep.
    """

    def __init__(
        self,
        session: ProfileSession,
        *,
        costs: CostRegistry,
        hardware: HardwareProfile,
        stream: torch.cuda.Stream,
        max_scopes: int,
        library: Path,
        output: Path,
    ) -> None:
        if session.backend != "nvtx":
            raise ValueError("Native timeline requires backend='nvtx'")
        if max_scopes < 1:
            raise ValueError("max_scopes must be positive")
        self.session = session
        self.costs = costs
        self.hardware = hardware
        self.stream = stream
        self.max_scopes = max_scopes
        self.library = library
        self.output = output
        self._nvtx: _Nvtx | None = None
        self._thread: int | None = None
        self._events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self._pending: list[_Pending] = []
        self._count = 0
        self._entered = False
        self._finished = False

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
        if self._entered or self.session._timeline is not None or self.session._active:
            raise RuntimeError("Nested or reused native timeline is unsupported")
        if self.session.ops or self.session.layers:
            raise ValueError("Native timeline requires an unused ProfileSession")
        self._entered = True
        self._thread = threading.get_native_id()
        self._check_context()
        self._nvtx = _Nvtx(self.library)
        try:
            for _ in range(self.max_scopes):
                begin = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                begin.record(self.stream)
                end.record(self.stream)
                self._events.append((begin, end))
            self.stream.synchronize()
        except BaseException:
            self._nvtx.close()
            raise
        self.session._timeline = self
        return self

    @contextmanager
    def record_func(self, label: str) -> Iterator[None]:
        """Record one scope without synchronizing or calculating costs."""
        self._check_context()
        if self._finished or self._nvtx is None:
            raise RuntimeError("Native timeline is not collecting")
        if self._count == self.max_scopes:
            raise RuntimeError("Native timeline max_scopes exhausted")
        begin, end = self._events[self._count]
        self._count += 1
        start = self._nvtx.timestamp()
        begin.record(self.stream)
        error = None
        try:
            yield
        except BaseException as exc:
            error = exc
            raise
        finally:
            try:
                end.record(self.stream)
                finish = self._nvtx.timestamp()
                self._pending.append(_Pending(label, start, finish, begin, end))
            except BaseException as timing_error:
                if error is None:
                    raise
                error.add_note(f"Native scope timing failed: {timing_error}")
        self._check_context()

    def finish(self) -> None:
        """Wait once, calculate all metrics and submit native ranges exactly once."""
        if self._finished:
            return
        if (
            not self._entered
            or self._nvtx is None
            or self.session._active
            or self.session._timeline is not self
        ):
            raise RuntimeError(
                "Finish outside iteration windows while capture is active"
            )
        self._check_context()
        if not self._pending:
            raise RuntimeError(
                "No op/layer scopes were recorded; native timeline is empty"
            )
        self.stream.synchronize()
        ops = {op.trace_label: op for op in self.session.ops}
        timings = []
        for pending in self._pending:
            reason = None
            if pending.label in ops:
                op = ops[pending.label]
                devices = {
                    tensor.device
                    for tensor in [*op.inputs.values(), *op.outputs.values()]
                    if tensor.device.startswith("cuda:")
                }
                if devices != {str(self.stream.device)}:
                    reason = "op tensor metadata does not establish work on the declared CUDA device"
            timings.append(
                ScopeTiming(
                    pending.label,
                    pending.start,
                    pending.end,
                    pending.begin_event.elapsed_time(pending.end_event) * 1e6,
                    reason,
                )
            )
        report = self.session.report(costs=self.costs)
        rows = analyze_utilization(report, timings, self.hardware)
        by_label = {row["trace_label"]: row for row in rows}
        if any(len(row["nvtx_label"].encode("utf-8")) >= 4096 for row in rows):
            raise ValueError("NVTX label exceeds 4095 UTF-8 bytes")
        report["hardware"] = asdict(self.hardware)
        report["device_analysis"] = {
            "timing_kind": "cuda_event_window",
            "timing_includes": ["stream waits", "host launch gaps", "event overhead"],
            "execution_contract": "single stream or work joined before scope exit",
            "native_timestamp_unit": "nvtx_tool_clock",
            "thread_id": self._thread,
            "device": str(self.stream.device),
            "device_name": torch.cuda.get_device_name(self.stream.device),
            "stream": self.stream.cuda_stream,
            "scopes": rows,
        }
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        # Validate and save first, so a file error cannot duplicate emitted ranges
        # when finish is retried. Parent-first ordering preserves NVTX nesting.
        for timing in sorted(
            timings, key=lambda t: (t.start_timestamp, -t.end_timestamp)
        ):
            self._nvtx.submit(timing, by_label[timing.trace_label]["nvtx_label"])
        self._finished = True

    def __exit__(self, exc_type, exc, traceback) -> None:
        try:
            self.finish()
        except BaseException as error:
            if exc is None:
                raise
            exc.add_note(f"Native timeline finalization failed: {error}")
        finally:
            self.session._timeline = None
            if self._nvtx is not None:
                self._nvtx.close()
