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

"""A launcher reports the failure records its ranks left, then removes them."""

import logging
import multiprocessing as mp
import os
import struct

from tokenspeed.runtime.utils.failure_records import (
    rank_record_path,
    report_failure_records,
)


def _rank(rank: int, words: list[int]) -> None:
    # What a kernel's host mirror leaves: a page whose record leads it.
    with open(rank_record_path("monokernel", rank), "wb") as handle:
        handle.write(struct.pack(f"<{len(words)}i", *words).ljust(4096, b"\0"))


def test_launcher_logs_only_records_that_hold_a_failure(caplog):
    for rank, words in ((0, [0, 0, 0]), (3, [4, 17, -1, 9, 0, 0])):
        child = mp.get_context("spawn").Process(target=_rank, args=(rank, words))
        child.start()
        child.join()
        assert child.exitcode == 0
    logger = logging.getLogger("launcher")
    with caplog.at_level(logging.ERROR, logger="launcher"):
        report_failure_records(os.getpid(), logger)
    assert [record.getMessage() for record in caplog.records] == [
        "failure record monokernel-rank3.bin: [4, 17, -1, 9]"
    ]
    # Removed with the records: a second report finds nothing.
    caplog.clear()
    report_failure_records(os.getpid(), logger)
    assert caplog.records == []
