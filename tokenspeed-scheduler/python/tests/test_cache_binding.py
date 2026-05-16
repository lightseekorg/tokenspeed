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

"""Tests for CacheEvent Python bindings."""

from tokenspeed_scheduler import Cache, ExecutionEvent


def test_cache_transfer_kinds_are_bound():
    assert Cache.TRANSFER_KIND_KV == 0
    assert Cache.TRANSFER_KIND_MAMBA == 1


def test_cache_event_fields_are_bound():
    write_back = Cache.WriteBackDoneEvent()
    write_back.success = True
    assert write_back.success is True

    prefetch_done = Cache.PrefetchDoneEvent()
    prefetch_done.request_id = "req-1"
    assert prefetch_done.request_id == "req-1"


def test_execution_event_accepts_cache_events():
    execution_event = ExecutionEvent()

    write_back = Cache.WriteBackDoneEvent()
    write_back.success = True
    assert execution_event.add_event(write_back) is execution_event

    prefetch_done = Cache.PrefetchDoneEvent()
    prefetch_done.request_id = "req-0"
    assert execution_event.add_event(prefetch_done) is execution_event
