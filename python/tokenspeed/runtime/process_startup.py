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

"""Helpers for supervising multiprocessing workers during startup."""

from __future__ import annotations

import time
from collections.abc import Sequence
from multiprocessing.connection import Connection, wait
from multiprocessing.process import BaseProcess
from typing import Any


def wait_for_process_startup(
    readers: Sequence[Connection],
    processes: Sequence[BaseProcess],
    *,
    description: str,
    timeout_seconds: float,
) -> list[dict[str, Any]]:
    """Wait for all workers to report ready or fail on the first bad worker."""
    if not readers or len(readers) != len(processes):
        raise ValueError("readers and processes must have the same non-zero length")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    deadline = time.monotonic() + timeout_seconds
    pending = set(range(len(processes)))
    results: list[dict[str, Any] | None] = [None] * len(processes)

    while pending:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"Timed out after {timeout_seconds:g}s waiting for {description} startup"
            )

        waitables = [readers[index] for index in pending]
        waitables.extend(process.sentinel for process in processes)
        ready = set(wait(waitables, timeout=remaining))
        if not ready:
            raise TimeoutError(
                f"Timed out after {timeout_seconds:g}s waiting for {description} startup"
            )

        # Consume readiness messages before checking sentinels. A worker can
        # send its payload and exit closely enough for both handles to fire.
        for index in list(pending):
            reader = readers[index]
            if reader not in ready:
                continue
            try:
                message = reader.recv()
            except EOFError as exc:
                processes[index].join(timeout=1)
                raise RuntimeError(
                    f"{description} {index} exited before reporting readiness "
                    f"(exit code {processes[index].exitcode})"
                ) from exc
            if not isinstance(message, dict) or message.get("status") != "ready":
                detail = message.get("error") if isinstance(message, dict) else message
                raise RuntimeError(
                    f"{description} {index} failed during startup:\n{detail}"
                )
            results[index] = message
            pending.remove(index)

        # Any exit observed before this wait iteration completes is a startup
        # failure, even if the child managed to report ready immediately first.
        exited = [
            index
            for index, process in enumerate(processes)
            if process.sentinel in ready
        ]
        # Prefer an unready process when several sentinels fire together; it is
        # more likely to be the failure that prevented startup from completing.
        for index in sorted(exited, key=lambda item: item not in pending):
            process = processes[index]
            process.join(timeout=1)
            state = (
                "before reporting readiness"
                if index in pending
                else "while peer processes were still starting"
            )
            raise RuntimeError(
                f"{description} {index} exited {state} "
                f"(exit code {process.exitcode})"
            )

    return [result for result in results if result is not None]
