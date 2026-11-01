# MIT License
#
# Copyright (c) 2026 LightSeek Foundation <contact@lightseek.org>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""CPU-only contract for releasing cached memory without outrunning the driver."""

import pytest
import torch

_memory = pytest.importorskip("tokenspeed_kernel_amd._memory")

GIB = 1 << 30
RESERVED = 200 * GIB
OUTSIDE = GIB  # Driver-accounted use outside the caching allocator.


def _release(monkeypatch, driver_used: list[int | None]) -> int:
    """Run one release against a scripted driver counter; return the polls made."""
    readings = iter(driver_used)
    polls = []

    def read(device: int) -> int | None:
        polls.append(device)
        return next(readings)

    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: RESERVED)
    monkeypatch.setattr(_memory, "driver_vram_used", read)
    monkeypatch.setattr(_memory.time, "sleep", lambda seconds: None)
    _memory.release_cached_memory(torch.device("cuda", 3))
    assert set(polls) <= {3}
    return len(polls)


def test_release_waits_until_driver_backlog_drains(monkeypatch) -> None:
    monkeypatch.setattr(_memory, "_non_allocator_vram", {})
    # The first release learns the baseline and has nothing to wait for.
    assert _release(monkeypatch, [RESERVED + OUTSIDE]) == 1
    # Freed memory the driver still holds: 20 GiB, 12 GiB, then within slack.
    backlog = [20 * GIB, 12 * GIB, _memory.RELEASE_BACKLOG_SLACK_BYTES]
    assert _release(monkeypatch, [RESERVED + OUTSIDE + b for b in backlog]) == 3


def test_release_without_driver_counter_does_not_wait(monkeypatch) -> None:
    monkeypatch.setattr(_memory, "_non_allocator_vram", {})
    assert _release(monkeypatch, [None]) == 1
