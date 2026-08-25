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

"""CPU-only tests for EventLoop L3 admit-path registration.

These bind ``_submit_scheduler_requests`` / ``_register_l3_storage_hits`` onto
a fake loop so the merge-time wiring (pause flush, EPD drain, and the normal
admit path all go through the helper) can be checked without a model or GPU.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

# CPU-only tests scheduled in runtime-1gpu because they import the full runtime.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed.runtime.engine.event_loop import EventLoop  # noqa: E402


class _L3Store:
    def __init__(self, exists_flags: list[bool]) -> None:
        self.exists_flags = exists_flags
        self.pages = None

    def exists(self, pages):
        self.pages = list(pages)
        return list(self.exists_flags)

    def present_keys(self, group_ids, content_hashes, page_offsets, exists):
        hits = [
            (group_id, content_hash, page_offset)
            for group_id, content_hash, page_offset, present in zip(
                group_ids, content_hashes, page_offsets, exists
            )
            if present
        ]
        if not hits:
            return [], [], []
        groups, hashes, offsets = zip(*hits)
        return list(groups), list(hashes), list(offsets)


class _Scheduler:
    def __init__(self) -> None:
        self.submitted: list[list] = []
        self.registered = None

    def submit_requests(self, specs) -> None:
        self.submitted.append(list(specs))

    def prefix_hashes_for_tokens(self, tokens):
        return [f"h{len(tokens)}"]

    def expand_prefix_keys(self, hashes):
        return [0] * len(hashes), list(hashes), [0] * len(hashes)

    def register_storage_keys(self, groups, hashes, offsets) -> None:
        self.registered = (list(groups), list(hashes), list(offsets))


class _Loop:
    """Only the EventLoop methods under test plus the state they read."""

    _submit_scheduler_requests = EventLoop._submit_scheduler_requests
    _register_l3_storage_hits = EventLoop._register_l3_storage_hits

    def __init__(self, l3=None) -> None:
        self.l2_cache_executor = (
            SimpleNamespace(l3_store=l3) if l3 is not None else None
        )
        self.scheduler = _Scheduler()
        self.attn_tp_size = 1
        self.attn_tp_cpu_group = None


def _spec(rid: str, tokens: list[int]):
    return SimpleNamespace(request_id=rid, tokens=tokens)


def test_submit_without_l3_still_admits() -> None:
    loop = _Loop(l3=None)
    spec = _spec("r0", [1, 2, 3, 4])

    loop._submit_scheduler_requests([spec])

    assert loop.scheduler.submitted == [[spec]]
    assert loop.scheduler.registered is None


def test_submit_registers_only_keys_l3_reports_present() -> None:
    store = _L3Store(exists_flags=[True])
    loop = _Loop(l3=store)
    spec = _spec("r0", [1, 2, 3, 4])

    loop._submit_scheduler_requests([spec])

    assert loop.scheduler.submitted == [[spec]]
    assert store.pages == [(0, 0, "h4", 0)]
    assert loop.scheduler.registered == ([0], ["h4"], [0])


def test_submit_skips_register_when_l3_misses() -> None:
    store = _L3Store(exists_flags=[False])
    loop = _Loop(l3=store)
    spec = _spec("r0", [1, 2, 3, 4])

    loop._submit_scheduler_requests([spec])

    assert loop.scheduler.submitted == [[spec]]
    assert loop.scheduler.registered is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
