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

"""Opt-in host-wall startup spans; never synchronize devices or ranks."""

from __future__ import annotations

import itertools
import json
import os
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from tokenspeed_kernel.compile_monitor import compile_stats

from tokenspeed.runtime.utils import get_colorful_logger
from tokenspeed.runtime.utils.env import envs

logger = get_colorful_logger(__name__)


@dataclass(frozen=True)
class _Span:
    id: int
    rank: int | None
    role: str | None


_current: ContextVar[_Span | None] = ContextVar("startup_span", default=None)
_ids = itertools.count()


def _emit(record: dict) -> None:
    # Diagnostics must not replace a model-loading exception or fail startup.
    try:
        logger.info(f"startup_timing {json.dumps(record, separators=(',', ':'))}")
    except Exception:
        pass


@contextmanager
def startup_phase(name: str, *, rank: int | None = None, role: str | None = None):
    """Log paired, nested spans with inherited rank/role metadata.

    Durations are inclusive host wall time, not device execution time. Wall
    timestamps permit approximate cross-process alignment on synchronized
    hosts. Triton deltas count observed compilations, not cache lookups.
    """
    if not envs.TOKENSPEED_STARTUP_TIMING.get():
        yield
        return

    parent = _current.get()
    span = _Span(
        next(_ids),
        rank if rank is not None else parent.rank if parent else None,
        role if role is not None else parent.role if parent else None,
    )
    token = _current.set(span)
    record = {
        "phase": name,
        "pid": os.getpid(),
        "rank": span.rank,
        "role": span.role,
        "span_id": span.id,
        "parent_id": parent.id if parent else None,
    }
    before = compile_stats()
    started = time.perf_counter()
    _emit({**record, "event": "start", "wall_time_ns": time.time_ns()})
    status = "ok"
    error_type = None
    try:
        yield
    except BaseException as exc:
        status = "error"
        error_type = type(exc).__name__
        raise
    finally:
        elapsed = time.perf_counter() - started
        _current.reset(token)
        after = compile_stats()
        _emit(
            {
                **record,
                "event": "end",
                "wall_time_ns": time.time_ns(),
                "duration_s": elapsed,
                "status": status,
                "error_type": error_type,
                "triton_compiles": (
                    after.startup_compiles - before.startup_compiles
                    if before is not None and after is not None
                    else None
                ),
                "triton_compile_s": (
                    after.startup_seconds - before.startup_seconds
                    if before is not None and after is not None
                    else None
                ),
            }
        )
