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


def storage_ops(plan):
    """The Host-only commands in a scheduler's unified cache plan."""
    from tokenspeed_scheduler import Cache

    types = (
        Cache.StorageQueryOp,
        Cache.StorageGetOp,
        Cache.StoragePutOp,
        Cache.StorageCancelOp,
    )
    return [operation for operation in plan.cache if isinstance(operation, types)]


def storage_scheduler():
    """A small real scheduler with explicit Host-storage credits."""
    import tokenspeed_scheduler as ts

    cfg = ts.SchedulerConfig()
    cfg.role = ts.SchedulerConfig.Role.Fused
    cfg.prefix_granularity = 2
    cfg.num_device_pages = 32
    cfg.num_host_pages = 8
    cfg.num_snapshot_pages = 1
    cfg.max_scheduled_tokens = 8
    cfg.max_batch_size = 4
    cfg.disable_l2_cache = False
    cfg.enable_l3_storage = True
    cfg.storage_config = ts.StorageConfig(
        8,
        8,
        4,
        4,
        32,
        1,
        1000,
        1000,
        ts.StoragePressure(8, 65536, 1000),
        ts.StoragePressure(4, 32768, 500),
    )
    cfg.cache_groups = [ts.CacheGroupConfig("full", 2, 32, ts.AttnKind.Full)]
    scheduler = ts.Scheduler(cfg)
    tick = ts.ExecutionEvent()
    tick.add_event(ts.Cache.StorageTick(1, True, True, ts.StoragePressure(0, 0, 0)))
    scheduler.advance(tick)
    request = ts.RequestSpec()
    request.request_id = "waiting"
    request.tokens = [1, 2, 3, 4]
    scheduler.submit_requests([request])
    return scheduler
