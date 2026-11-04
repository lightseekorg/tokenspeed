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

"""Failure records that outlive the rank that wrote them.

A kernel that stops its process on a device-side failure (a decode
monokernel's fail-stop) cannot report through the host: once it traps, its
queue is gone and the process aborts. Before trapping, it copies its record
into a file-backed host page. The rank names that file with ``rank_record_path``,
under a directory owned by its launcher (the rank's parent process). When the
ranks are gone, the launcher logs every record that holds a failure and
removes the directory (``report_failure_records``); it discards a directory
an earlier process with its pid left before it starts the ranks.

A record is int32 words in host order; their meaning is the writing kernel's
(see its README).
"""

from __future__ import annotations

import logging
import os
import shutil
from array import array

_ROOT = "/dev/shm"


def _record_dir(launcher_pid: int) -> str:
    return os.path.join(_ROOT, f"tokenspeed-{launcher_pid}-failure-records")


def rank_record_path(kind: str, rank: int) -> str:
    """The file a rank's ``kind`` kernel mirrors its failure record into.

    Args:
        kind: The writing kernel family, such as ``"decoder-layer"``.
        rank: The rank's global rank.
    """
    directory = _record_dir(os.getppid())
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, f"{kind}-rank{rank}.bin")


def discard_failure_records(launcher_pid: int) -> None:
    """Remove records an earlier launcher with this pid left (it was killed
    before it could report them), so they are not reported as this run's."""
    shutil.rmtree(_record_dir(launcher_pid), ignore_errors=True)


def report_failure_records(launcher_pid: int, logger: logging.Logger) -> None:
    """Log the failure records the ranks of ``launcher_pid`` left, then remove
    them. A record of all zeros holds no failure and is not logged."""
    directory = _record_dir(launcher_pid)
    if not os.path.isdir(directory):
        return
    for name in sorted(os.listdir(directory)):
        words = array("i")
        with open(os.path.join(directory, name), "rb") as handle:
            data = handle.read()
        words.frombytes(data[: len(data) // 4 * 4])
        # Records sit at the start of their page; the rest stays zero.
        last = max((i for i, word in enumerate(words) if word), default=-1)
        if last >= 0:
            logger.error(f"failure record {name}: {list(words[: last + 1])}")
    discard_failure_records(launcher_pid)
