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

"""Optional CPU service markers. NVTX supplies its own independent clock.

The disabled/JSON-only path imports no Torch/NVTX. Enabled markers resolve
Torch's synchronous NVTX binding before callbacks; no CUDA device work,
synchronization or stream callback belongs here. Generation is the control plane's
existing admission number; retraction epochs remain unknown until the scheduler
exports them. The maps below retain observation metadata, never request state.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import TextIO

# Observation loss remains explicit instead of growing metadata without bound.
_MAX_DISPATCHES = 4096
_MAX_READY_OUTPUTS = 16384


def prepare_nvtx_marker() -> Callable[[str], None]:
    """Resolve the public synchronous marker only for explicit NVTX collection.

    The trace's first start marker calls the binding before request warmup; a
    missing CUDA-build NVTX binding raises there rather than losing markers.
    """
    try:
        from torch.cuda.nvtx import mark
    except ImportError as error:
        raise RuntimeError("Service NVTX requires torch.cuda.nvtx.mark") from error
    if not callable(mark):
        raise RuntimeError("Service NVTX marker binding is not callable")
    return mark


class ServiceTrace:
    def __init__(self, output: TextIO, *, run_id: str, nvtx_enabled: bool):
        self._output = output
        self.run_id = run_id
        self._lock = threading.Lock()
        self._dispatches: dict[int, dict] = {}
        self._ready: dict[tuple[str, int | None, int], dict] = {}
        if type(nvtx_enabled) is not bool:
            raise TypeError("nvtx_enabled must be an explicit boolean")
        self._mark = prepare_nvtx_marker() if nvtx_enabled else None
        self._record("start", nvtx=self._mark is not None)

    @classmethod
    def from_environment(cls) -> ServiceTrace | None:
        path = os.environ.get("TOKENSPEED_SERVICE_TRACE")
        if path is None:
            return None
        if "TOKENSPEED_SERVICE_NVTX_LIBRARY" in os.environ:
            raise ValueError(
                "TOKENSPEED_SERVICE_NVTX_LIBRARY was removed; the paired profiler command selects torch.cuda.nvtx.mark"
            )
        enabled = os.environ.get("TOKENSPEED_SERVICE_NVTX", "0")
        if enabled not in {"0", "1"}:
            raise ValueError("TOKENSPEED_SERVICE_NVTX must be 0 or 1")
        output = Path(path.format(pid=os.getpid()))
        output.parent.mkdir(parents=True, exist_ok=True)
        return cls(
            output.open("a", encoding="utf-8", buffering=1),
            run_id=os.environ.get("TOKENSPEED_PROFILE_RUN_ID", str(uuid.uuid4())),
            nvtx_enabled=enabled == "1",
        )

    def _record(self, boundary: str, **fields) -> None:
        event = {
            "schema": "tokenspeed.service/v1",
            "boundary": boundary,
            "run_id": self.run_id,
            "pid": os.getpid(),
            "observed_ns": time.perf_counter_ns(),
            **fields,
        }
        text = json.dumps(event, separators=(",", ":"), sort_keys=True)
        # NVTX timestamps the actual callback on this thread. No elapsed time
        # from the client/reference is ever encoded as a timeline duration.
        if self._mark is not None:
            self._mark(text)
        self._output.write(text + "\n")

    def dispatch(self, forward_id: int, identity) -> None:
        with self._lock:
            graph = (
                {
                    "run_id": identity.run_id,
                    "rank": identity.rank,
                    "step_id": identity.step_id,
                }
                if identity is not None
                else None
            )
            if graph is not None and graph["run_id"] != self.run_id:
                raise ValueError("service and graph trace run_id differ")
            if forward_id in self._dispatches:
                raise ValueError("duplicate forward identity")
            if len(self._dispatches) >= _MAX_DISPATCHES:
                self._record(
                    "observer_drop", forward_id=forward_id, reason="dispatch_capacity"
                )
                return
            self._dispatches[forward_id] = {"forward_id": forward_id, "graph": graph}
            self._record("dispatch", **self._dispatches[forward_id])

    def commit(
        self,
        forward_id: int,
        rows: list[tuple],
        outputs: list[tuple],
        *,
        owns_frontend: bool,
        observer_delay_ns: int = 0,
    ) -> None:
        callback_at = time.perf_counter_ns()
        with self._lock:
            # Remove once the actual commit succeeded, not when device copies
            # became ready or the old profiler's forward count advanced.
            dispatch = self._dispatches.pop(forward_id, None)
            commits = {}
            for rid, admission, cumulative, token_ids, finished in rows:
                fields = {
                    "forward_id": forward_id,
                    "graph": dispatch["graph"] if dispatch is not None else None,
                    "rid": rid,
                    "generation": admission,
                    "retraction_epoch": None,
                    "completion_tokens": cumulative,
                    "output_ids": token_ids,
                    "finish_reason": finished,
                }
                commits[(rid, cumulative)] = fields
                self._record(
                    "commit",
                    observer_delay_ns=observer_delay_ns
                    + time.perf_counter_ns()
                    - callback_at,
                    **fields,
                )
            for rid, cumulative, token_ids, finished in outputs:
                key = (rid, cumulative)
                source = commits.get(key)
                fields = {
                    **(
                        source
                        or {
                            "forward_id": None,
                            "graph": None,
                            "rid": rid,
                            "generation": None,
                            "retraction_epoch": None,
                        }
                    ),
                    "completion_tokens": cumulative,
                    "output_ids": token_ids,
                    "finish_reason": finished,
                }
                # Only the actual frontend owner will later call emit. Other
                # ranks still record readiness, but retain no send join state.
                if owns_frontend:
                    ready_key = (rid, fields["generation"], cumulative)
                    if len(self._ready) < _MAX_READY_OUTPUTS:
                        self._ready[ready_key] = fields
                    else:
                        self._record("observer_drop", rid=rid, reason="ready_capacity")
                self._record(
                    "output_ready",
                    observer_delay_ns=observer_delay_ns
                    + time.perf_counter_ns()
                    - callback_at,
                    **fields,
                )

    def emit(self, outputs: list[tuple], *, observer_delay_ns: int = 0) -> None:
        callback_at = time.perf_counter_ns()
        with self._lock:
            for rid, cumulative, token_ids, finished in outputs:
                candidates = [
                    key for key in self._ready if key[0] == rid and key[2] == cumulative
                ]
                # The wire output has no generation field. A unique retained
                # authoritative generation is required; ambiguity stays unknown.
                source = (
                    self._ready.pop(candidates[0]) if len(candidates) == 1 else None
                )
                if len(candidates) > 1:
                    for key in candidates:
                        self._ready.pop(key)
                self._record(
                    "frontend_send_returned",
                    observer_delay_ns=observer_delay_ns
                    + time.perf_counter_ns()
                    - callback_at,
                    **{
                        **(
                            source
                            or {
                                "forward_id": None,
                                "graph": None,
                                "rid": rid,
                                "generation": None,
                                "retraction_epoch": None,
                            }
                        ),
                        "completion_tokens": cumulative,
                        "output_ids": token_ids,
                        "finish_reason": finished,
                    },
                )

    def release(self, rids: list[str]) -> None:
        """Drop pending joins for ended/cancelled requests after the send hook."""
        with self._lock:
            ended = set(rids)
            for key in list(self._ready):
                if key[0] in ended:
                    self._ready.pop(key)
