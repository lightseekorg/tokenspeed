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

"""Marking intentional host synchronization on the data plane.

The forward thread must not wait for the device on the per-round path: a
``.cpu()``, ``.item()``, ``bool(tensor)`` or pageable copy waits for the
stream, and the stream holds the step in flight, so the next step's launch
slips behind it and overlap scheduling degrades to depth 0
(``docs/design/event-loop.md``). ``TOKENSPEED_DATA_PLANE_SYNC_DEBUG`` arms
torch's sync-debug mode to report or reject every such wait once serving
starts. The few waits that are deliberate -- validation that only runs on
prefill rounds, whose step is long enough to absorb it -- are wrapped in
``allow_host_sync`` so they are explicit in the source and pass the check.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch


@contextmanager
def allow_host_sync(reason: str) -> Iterator[None]:
    """Permit device synchronization inside the block.

    Args:
        reason: Why this wait is acceptable on the data plane, for the reader
            and for ``grep``; it is not interpreted.

    The sync-debug mode is process-wide, so the block briefly disables it for
    every thread; the control plane never performs a flagged operation, and
    the data plane runs one round at a time, so nothing else is masked.
    """
    del reason
    if not torch.cuda.is_available():
        yield
        return
    previous = torch.cuda.get_sync_debug_mode()
    if previous == 0:
        yield
        return
    torch.cuda.set_sync_debug_mode(0)
    try:
        yield
    finally:
        torch.cuda.set_sync_debug_mode(previous)
