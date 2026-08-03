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

"""Tests for cache-related Python bindings."""

import tokenspeed_scheduler as ts
from tokenspeed_scheduler import Cache, ExecutionEvent


def test_removed_storage_cache_api_is_not_exported():
    assert not hasattr(ts, "PrefixCacheAdjunctSpec")

    config = ts.SchedulerConfig()
    assert not hasattr(config, "prefix_cache_adjunct")
    assert not hasattr(config, "prefetch_threshold")

    request = ts.RequestSpec()
    assert not hasattr(request, "rolling_hashes")
    assert not hasattr(request, "storage_hit_pages")

    assert not hasattr(Cache, "PrefetchDoneEvent")
    assert not hasattr(Cache, "PrefetchOp")
    assert not hasattr(Cache, "BackUpOp")
    assert not hasattr(Cache, "CacheKind")
    assert not hasattr(Cache.WriteBackOp, "is_retract")
    assert not hasattr(Cache.WriteBackOp, "src_pages_by_kind")
    assert not hasattr(Cache.LoadBackOp, "src_pages_by_kind")
    assert not hasattr(ts.Forward.Batch, "hist_token_lens")
    assert not hasattr(ts.Scheduler, "get_request_paged_cache_page_ids")


def test_cache_event_fields_are_bound():
    write_back = Cache.WriteBackDoneEvent()
    write_back.success = True
    assert write_back.success is True

    load_back = Cache.LoadBackDoneEvent()
    load_back.success = True
    assert load_back.success is True


def test_execution_event_accepts_cache_events():
    execution_event = ExecutionEvent()

    write_back = Cache.WriteBackDoneEvent()
    write_back.success = True
    assert execution_event.add_event(write_back) is execution_event

    load_back = Cache.LoadBackDoneEvent()
    assert execution_event.add_event(load_back) is execution_event
