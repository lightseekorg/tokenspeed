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

import atexit
import json
import os
import threading
import time
import warnings
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tokenspeed_kernel._triton import proton

_HAS_PROTON = proton is not None

__all__ = [
    "CapturedShape",
    "ProfilingConfig",
    "ProfilingState",
    "ShapeCapture",
    "bootstrap_profiling_from_env",
    "kernel_scope",
    "profile_config_from_env",
    "profiling",
    "proton_available",
    "shape_capture",
    "start_shape_capture",
    "start_profiling",
    "stop_shape_capture",
    "stop_profiling",
]

# Enable profiling bootstrap at import time.
# Truthy values: 1/true/yes/on (case-insensitive).
_ENV_PROFILE = "TOKENSPEED_KERNEL_PROFILE"
# Proton output prefix/path.
# Default: "profile".
_ENV_PROFILE_OUTPUT = "TOKENSPEED_KERNEL_PROFILE_OUTPUT"
# Profiling data mode.
# Supported: "tree" or "trace". Default: "tree".
_ENV_PROFILE_DATA = "TOKENSPEED_KERNEL_PROFILE_DATA"
# Activity backend override.
# Supported: "cupti" or "roctracer".
_ENV_PROFILE_BACKEND = "TOKENSPEED_KERNEL_PROFILE_BACKEND"
# Profiling mode override.
# Supported: "pcsampling" or "periodic_flushing".
_ENV_PROFILE_MODE = "TOKENSPEED_KERNEL_PROFILE_MODE"
# Launch hook override.
# Typical value: "triton".
_ENV_PROFILE_HOOK = "TOKENSPEED_KERNEL_PROFILE_HOOK"
# Finalized report format.
# Supported: "hatchet", "hatchet_msgpack", "chrome_trace".
_ENV_PROFILE_OUTPUT_FORMAT = "TOKENSPEED_KERNEL_PROFILE_OUTPUT_FORMAT"
# Enable shape capture bootstrap at import time.
# Truthy values: 1/true/yes/on (case-insensitive).
_ENV_CAPTURE_SHAPES = "TOKENSPEED_KERNEL_CAPTURE_SHAPES"
# Shape capture output JSON path.
# Default: "shapes.json".
_ENV_CAPTURE_SHAPES_OUTPUT = "TOKENSPEED_KERNEL_CAPTURE_SHAPES_OUTPUT"


@dataclass
class ProfilingConfig:
    output: str = "profile"
    data: str = "tree"
    backend: str | None = None
    mode: str | None = None
    hook: str | None = "triton"
    output_format: str = ""


class ProfilingState:
    _instance: "ProfilingState | None" = None

    def __init__(self) -> None:
        self.enabled: bool = False
        self._session: int | None = None
        self._config: ProfilingConfig | None = None

    @classmethod
    def get(cls) -> "ProfilingState":
        if cls._instance is None:
            cls._instance = ProfilingState()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None

    @property
    def active(self) -> bool:
        return self.enabled and self._session is not None


@dataclass
class CapturedShape:
    family: str
    mode: str
    kernel_name: str
    dtype: str
    shape_params: dict[str, Any]
    timestamp_ns: int


class ShapeCapture:
    _instance: "ShapeCapture | None" = None

    def __init__(self) -> None:
        self.enabled: bool = False
        self._records: list[CapturedShape] = []
        self._lock = threading.Lock()

    @classmethod
    def get(cls) -> "ShapeCapture":
        if cls._instance is None:
            cls._instance = ShapeCapture()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None

    def record(
        self,
        family: str,
        mode: str,
        kernel_name: str,
        dtype: Any,
        shape_params: dict[str, Any],
    ) -> None:
        if not self.enabled:
            return
        entry = CapturedShape(
            family=family,
            mode=mode,
            kernel_name=kernel_name,
            dtype=str(dtype),
            shape_params=dict(shape_params),
            timestamp_ns=time.time_ns(),
        )
        with self._lock:
            self._records.append(entry)

    def dump(self, path: str | Path) -> None:
        with self._lock:
            payload = [asdict(record) for record in self._records]
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str)
        )

    def clear(self) -> None:
        with self._lock:
            self._records.clear()


class _NoopScope:
    def __enter__(self) -> "_NoopScope":
        return self

    def __exit__(self, *args: object) -> None:
        _ = args


_NOOP_SCOPE = _NoopScope()
_BOOTSTRAPPED = False


def _is_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def proton_available() -> bool:
    """Report whether the Proton profiler can be used in this process.

    Returns:
        True if the vendored Triton distribution provides
        ``tokenspeed_triton.profiler``; False otherwise (profiling calls
        become no-ops with a warning).
    """
    return _HAS_PROTON


def profile_config_from_env(output: str | None = None) -> ProfilingConfig:
    """Build a :class:`ProfilingConfig` from ``TOKENSPEED_KERNEL_PROFILE_*``.

    Args:
        output: Optional Proton output prefix/path. When provided it takes
            precedence over ``TOKENSPEED_KERNEL_PROFILE_OUTPUT``; callers that
            profile multiple processes (e.g. one scheduler per rank) use this
            to give each process a distinct output file.

    Returns:
        A :class:`ProfilingConfig` with ``data``/``backend``/``mode``/``hook``/
        ``output_format`` sourced from the environment.
    """
    return ProfilingConfig(
        output=output or os.environ.get(_ENV_PROFILE_OUTPUT, "profile"),
        data=os.environ.get(_ENV_PROFILE_DATA, "tree"),
        backend=os.environ.get(_ENV_PROFILE_BACKEND),
        mode=os.environ.get(_ENV_PROFILE_MODE),
        hook=os.environ.get(_ENV_PROFILE_HOOK, "triton"),
        output_format=os.environ.get(_ENV_PROFILE_OUTPUT_FORMAT, ""),
    )


def start_profiling(config: ProfilingConfig | None = None) -> int | None:
    state = ProfilingState.get()
    if state.active:
        return state._session

    if not _HAS_PROTON:
        warnings.warn("Proton not installed; profiling disabled", stacklevel=2)
        return None

    config = config or ProfilingConfig()
    session = proton.start(
        config.output,
        data=config.data,
        backend=config.backend,
        mode=config.mode,
        hook=config.hook,
    )
    state._config = config
    state._session = session
    state.enabled = session is not None
    return session


def stop_profiling() -> None:
    state = ProfilingState.get()
    if not state.active or not _HAS_PROTON:
        state._session = None
        state._config = None
        state.enabled = False
        return

    output_format = state._config.output_format if state._config is not None else ""
    proton.finalize(state._session, output_format)
    state._session = None
    state._config = None
    state.enabled = False


def start_shape_capture() -> None:
    ShapeCapture.get().enabled = True


def stop_shape_capture(output_path: str | Path = "shapes.json") -> None:
    capture = ShapeCapture.get()
    if not capture.enabled:
        return
    capture.dump(output_path)
    capture.enabled = False
    capture.clear()


@contextmanager
def profiling(config: ProfilingConfig | None = None):
    session = start_profiling(config)
    try:
        yield session
    finally:
        stop_profiling()


@contextmanager
def shape_capture(output_path: str | Path = "shapes.json"):
    start_shape_capture()
    try:
        yield
    finally:
        stop_shape_capture(output_path)


def kernel_scope(
    family: str,
    mode: str,
    dtype: Any,
    *,
    kernel_name: str = "",
    **metrics: object,
):
    state = ProfilingState.get()
    if not state.active:
        return _NOOP_SCOPE

    name = f"{family}.{mode}[{kernel_name}]" if kernel_name else f"{family}.{mode}"
    scope_metrics = {"dtype": str(dtype)}
    scope_metrics.update(metrics)
    return proton.scope(name, metrics=scope_metrics)


def _atexit_stop_profiling() -> None:
    try:
        stop_profiling()
    except Exception as exc:
        warnings.warn(f"Failed to finalize profiling at exit: {exc}", stacklevel=1)


def _atexit_stop_shape_capture() -> None:
    output_path = os.environ.get(_ENV_CAPTURE_SHAPES_OUTPUT, "shapes.json")
    try:
        stop_shape_capture(output_path)
    except Exception as exc:
        warnings.warn(
            f"Failed to finalize shape capture at exit: {exc}",
            stacklevel=1,
        )


def bootstrap_profiling_from_env() -> None:
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return
    _BOOTSTRAPPED = True

    if _is_truthy(os.environ.get(_ENV_PROFILE)):
        start_profiling(profile_config_from_env())
    if _is_truthy(os.environ.get(_ENV_CAPTURE_SHAPES)):
        start_shape_capture()

    atexit.register(_atexit_stop_profiling)
    atexit.register(_atexit_stop_shape_capture)
