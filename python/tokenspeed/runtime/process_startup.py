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

from collections.abc import Sequence
from multiprocessing.connection import Connection, wait
from multiprocessing.process import BaseProcess
from typing import Any


def _failure_detail(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("error") or message)
    return repr(message)


def wait_for_process_startup(
    readers: Sequence[Connection],
    processes: Sequence[BaseProcess],
    *,
    description: str,
) -> list[dict[str, Any]]:
    """Wait for all workers to report ready, failing on the first bad worker.

    Every worker is monitored through both its readiness pipe and process
    sentinel. This avoids rank-ordered ``recv`` calls hiding a later rank's
    failure while an earlier rank remains blocked in distributed rendezvous.

    Args:
        readers: One receive-only readiness pipe per worker.
        processes: The corresponding started multiprocessing workers.
        description: Human-readable worker name used in errors.

    Returns:
        Readiness payloads in the same order as ``processes``.

    Raises:
        ValueError: If the reader/process counts differ or are empty.
        RuntimeError: If a worker reports failure, closes its pipe, or exits
            before every worker has become ready.
    """
    if not readers or len(readers) != len(processes):
        raise ValueError("readers and processes must have the same non-zero length")

    pending = set(range(len(processes)))
    results: list[dict[str, Any] | None] = [None] * len(processes)

    while pending:
        waitables = [readers[index] for index in pending]
        waitables.extend(process.sentinel for process in processes)
        ready = set(wait(waitables))

        # Consume pipe payloads first. A worker can send its failure and exit
        # close enough together for both the connection and sentinel to fire.
        for index in list(pending):
            reader = readers[index]
            sentinel_ready = processes[index].sentinel in ready
            if reader not in ready and not (sentinel_ready and reader.poll()):
                continue
            try:
                message = reader.recv()
            except EOFError as exc:
                processes[index].join(timeout=0)
                raise RuntimeError(
                    f"{description} {index} exited before reporting readiness "
                    f"(exit code {processes[index].exitcode})"
                ) from exc

            if not isinstance(message, dict) or message.get("status") != "ready":
                raise RuntimeError(
                    f"{description} {index} failed during startup:\n"
                    f"{_failure_detail(message)}"
                )
            results[index] = message
            pending.remove(index)

        # A process that exits while any peer is still starting is always a
        # startup failure, even if it managed to report ready just beforehand.
        for index, process in enumerate(processes):
            if process.sentinel not in ready:
                continue
            process.join(timeout=0)
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
