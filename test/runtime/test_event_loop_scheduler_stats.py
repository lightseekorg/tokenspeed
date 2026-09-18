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

"""Runtime page gauges against the native scheduler's LCM block API."""

import os
import sys
from types import SimpleNamespace

import pytest
import tokenspeed_scheduler as ts

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.engine.event_loop import EventLoop


@pytest.mark.parametrize(
    ("packing", "active_parents", "cached_parents"), [(1, 3, 2), (2, 2, 1)]
)
def test_scheduler_stats_partition_lcm_blocks(packing, active_parents, cached_parents):
    config = ts.SchedulerConfig()
    config.prefix_granularity = 2
    config.num_device_pages = 17  # Null parent 0 plus 16 usable LCM parents.
    config.num_host_pages = 0
    config.max_scheduled_tokens = 16
    config.max_batch_size = 8
    config.decode_input_tokens = 1
    config.overlap_schedule_depth = 0
    config.disable_l2_cache = True
    config.disable_prefix_cache = False
    config.cache_groups = [
        ts.CacheGroupConfig(
            group_id="history",
            block_granularity=2,
            total_pages=16 * packing + 1,
            retention=ts.CacheRetention.FullHistory,
            sliding_window_tokens=None,
            family=ts.CacheGroupFamily.History,
            cache_blocks_per_lcm_block=packing,
            transfer_policy=ts.CacheTransferPolicy.Unspecified,
        )
    ]
    scheduler = ts.Scheduler(config)
    loop = SimpleNamespace(
        scheduler=scheduler,
        _scheduler_cache_geometry=SimpleNamespace(num_usable_pages=16),
    )
    assert EventLoop._get_scheduler_stats(loop) == {
        "num_active_pages": 0,
        "num_cached_pages": 0,
        "num_queue_reqs": 0,
    }

    request = ts.RequestSpec()
    request.request_id = "r"
    request.tokens = [1, 2, 3, 4]
    scheduler.submit_requests([request])
    assert EventLoop._get_scheduler_stats(loop) == {
        "num_active_pages": 0,
        "num_cached_pages": 0,
        "num_queue_reqs": 1,
    }

    scheduler.next_execution_plan()
    assert EventLoop._get_scheduler_stats(loop) == {
        "num_active_pages": active_parents,
        "num_cached_pages": 0,
        "num_queue_reqs": 0,
    }

    result = ts.ForwardEvent.ExtendResult()
    result.request_id = "r"
    result.tokens = [5]
    scheduler.advance(ts.ExecutionEvent().add_event(result))
    finish = ts.ForwardEvent.Finish()
    finish.request_id = "r"
    scheduler.advance(ts.ExecutionEvent().add_event(finish))
    scheduler.next_execution_plan()
    assert EventLoop._get_scheduler_stats(loop) == {
        "num_active_pages": 0,
        "num_cached_pages": cached_parents,
        "num_queue_reqs": 0,
    }

    assert scheduler.clear_cache()
    assert EventLoop._get_scheduler_stats(loop) == {
        "num_active_pages": 0,
        "num_cached_pages": 0,
        "num_queue_reqs": 0,
    }


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
