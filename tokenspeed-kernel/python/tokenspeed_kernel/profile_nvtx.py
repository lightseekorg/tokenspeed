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

import ctypes
import json
import os
import subprocess
import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from itertools import count
from pathlib import Path
from typing import Iterator

import torch
from tokenspeed_kernel.op_profile import CostRegistry, ProfileSession
from tokenspeed_kernel.profile_utilization import HardwareProfile

_scope_ids = count(1)


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
        self.library.ts_nvtx_push.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_uint64,
        ]
        self.library.ts_nvtx_push.restype = ctypes.c_int
        self.library.ts_nvtx_pop.argtypes = [ctypes.c_void_p]
        self.library.ts_nvtx_pop.restype = ctypes.c_int
        self.library.ts_nvtx_close.argtypes = [ctypes.c_void_p]
        self.library.ts_nvtx_close.restype = None
        self.handle = self.library.ts_nvtx_open()
        if not self.handle:
            raise RuntimeError(
                "Run under Nsight Systems 2026.5+ with --trace=cuda,nvtx"
            )

    def push(self, label: str, scope_id: int) -> None:
        if self.library.ts_nvtx_push(self.handle, label.encode("utf-8"), scope_id) < 0:
            raise RuntimeError("Could not start native NVTX range")

    def pop(self) -> None:
        if self.library.ts_nvtx_pop(self.handle) < 0:
            raise RuntimeError("Could not end native NVTX range")

    def close(self) -> None:
        self.library.ts_nvtx_close(self.handle)
        self.handle = None


@dataclass(frozen=True)
class _RecordedScope:
    trace_label: str
    nvtx_label: str
    nvtx_id: int


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
        library: Prebuilt optional NVTX bridge.
        output: Cost metadata accompanying nsys-rep; GPU metrics are pending.
        max_metadata_bytes: Pinned-host capacity for explicitly requested integer
            tensor snapshots, allocated before collection. Exhaustion raises.
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
        max_metadata_bytes: int = 1 << 20,
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
        self.library = library
        self.output = output
        self.max_metadata_bytes = max_metadata_bytes
        self._nvtx: _Nvtx | None = None
        self._thread: int | None = None
        self._scopes: list[_RecordedScope] = []
        self._snapshots: list[_IntegerSnapshot] = []
        self._metadata_buffer: torch.Tensor | None = None
        self._metadata_used = 0
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
            self._metadata_buffer = torch.empty(
                self.max_metadata_bytes,
                dtype=torch.uint8,
                device="cpu",
                pin_memory=True,
            )
        except BaseException:
            self._nvtx.close()
            raise
        self.session._timeline = self
        return self

    def _snapshot_integer_values(
        self, source: torch.Tensor, target: dict[str, tuple[int, ...] | None], name: str
    ) -> None:
        """Queue a validated integer vector before the op window; never wait here."""
        self._check_context()
        if self._finished or self._metadata_buffer is None:
            raise RuntimeError("Native timeline is not collecting")
        if source.device != self.stream.device:
            return  # Do not copy unordered metadata from another device.
        start = (self._metadata_used + 7) // 8 * 8
        end = start + source.numel() * source.element_size()
        if end > self.max_metadata_bytes:
            raise RuntimeError("Native timeline max_metadata_bytes exhausted")
        host = self._metadata_buffer[start:end].view(source.dtype)
        host.copy_(source, non_blocking=True)
        source.record_stream(self.stream)
        self._metadata_used = end
        self._snapshots.append(_IntegerSnapshot(target, name, host))

    @contextmanager
    def record_func(self, label: str, *, display_name: str) -> Iterator[None]:
        """Record one scope without synchronizing or calculating costs."""
        self._check_context()
        if self._finished or self._nvtx is None:
            raise RuntimeError("Native timeline is not collecting")
        if len(self._scopes) >= self.max_scopes:
            raise RuntimeError("Native timeline max_scopes exhausted")
        # A native numeric payload joins to the full identity in the sidecar;
        # repeated calls need no extra text on the visible timeline.
        scope_id = next(_scope_ids)
        self._nvtx.push(display_name, scope_id)
        self._scopes.append(_RecordedScope(label, display_name, scope_id))
        error = None
        try:
            yield
        except BaseException as exc:
            error = exc
            raise
        finally:
            try:
                self._nvtx.pop()
            except BaseException as close_error:
                if error is None:
                    raise
                error.add_note(f"Native NVTX range close failed: {close_error}")
        self._check_context()

    def finish(self) -> None:
        """Resolve optional metadata snapshots and save costs exactly once."""
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
        if not self._scopes:
            raise RuntimeError(
                "No op/layer scopes were recorded; native timeline is empty"
            )
        if self._snapshots:
            self.stream.synchronize()
        for snapshot in self._snapshots:
            snapshot.target[snapshot.name] = tuple(snapshot.host.tolist())
        self._snapshots.clear()
        report = self.session.report(costs=self.costs)
        hardware = self.hardware
        if hardware.schema_version == 1 and hardware.selection is None:
            hardware = hardware.validate_device(
                {"name": torch.cuda.get_device_name(self.stream.device)}
            )
        report["hardware"] = asdict(hardware)
        report["device_analysis"] = {
            "timing_kind": "pending_nsight",
            "process_id": os.getpid(),
            "thread_id": self._thread,
            "device": str(self.stream.device),
            "device_name": torch.cuda.get_device_name(self.stream.device),
            "stream": self.stream.cuda_stream,
            "metadata_snapshot_bytes": self._metadata_used,
            "scopes": [asdict(scope) for scope in self._scopes],
        }
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
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
            self._snapshots.clear()
            self._metadata_buffer = None
            if self._nvtx is not None:
                self._nvtx.close()
