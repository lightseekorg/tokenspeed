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

"""Periodic rank-attributed stacks that survive a test runner killing workers."""

import faulthandler
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def worker_diagnostics(
    directory: Path, rank: int, *, dump_after_seconds: float
) -> Iterator[Callable[[str], None]]:
    """Record phases and dump stacks every interval without terminating the worker.

    Use only in a dedicated worker process: faulthandler owns one process-wide
    timer. The caller retains the directory on failure for artifact collection.
    Phase labels mark host progress, not completion of asynchronous GPU work.
    Snapshots are periodic; they do not identify whether a worker is stalled.
    """
    with (directory / f"rank-{rank}.log").open("w") as output:
        print(f"rank={rank} pid={os.getpid()}", file=output, flush=True)

        def phase(name: str) -> None:
            print(f"phase={name}", file=output, flush=True)

        faulthandler.dump_traceback_later(
            dump_after_seconds, repeat=True, file=output, exit=False
        )
        try:
            yield phase
        finally:
            faulthandler.cancel_dump_traceback_later()
