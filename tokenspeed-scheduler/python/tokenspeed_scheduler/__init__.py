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

"""Python API for the TokenSpeed scheduler."""

import tokenspeed_scheduler.tokenspeed_scheduler_ext as _ext
from tokenspeed_scheduler.tokenspeed_scheduler_ext import (  # Core; Execution plan; Events
    DisaggregationMode,
    ExecutionEvent,
    ExecutionPlan,
    PagedCacheGroupAllocator,
    PagedCacheGroupConfig,
    PagedCacheGroupTable,
    PagedCacheRetention,
    PrefixCacheAdjunctSpec,
    RequestSpec,
    Scheduler,
    SchedulerConfig,
    SchedulerStats,
)

PD = _ext.PD
Cache = _ext.Cache
Forward = _ext.Forward
ForwardEvent = _ext.ForwardEvent
KVEvent = _ext.KVEvent


def _flat_forward_op_repr(self):
    return (
        f"FlatForwardOp("
        f"request_ids={list(self.request_ids)}, "
        f"request_pool_indices={list(self.request_pool_indices)}, "
        f"input_lengths={list(self.input_lengths)}, "
        f"occupied_pages={[list(p) for p in self.occupied_pages]}, "
        f"begins={list(self.begins)}, "
        f"sizes={list(self.sizes)}, "
        f"input_ids={list(self.input_ids)}, "
        f"shifted_input_ids={list(self.shifted_input_ids)}, "
        f"extend_prefix_lens={list(self.extend_prefix_lens)}, "
        f"num_extends={self.num_extends()}"
        f")"
    )


Forward.FlatForwardOp.__repr__ = _flat_forward_op_repr

__all__ = [
    # Core
    "Scheduler",
    "SchedulerConfig",
    "SchedulerStats",
    "DisaggregationMode",
    "RequestSpec",
    "PagedCacheRetention",
    "PagedCacheGroupConfig",
    "PagedCacheGroupAllocator",
    "PagedCacheGroupTable",
    "PrefixCacheAdjunctSpec",
    # Execution plan & operations
    "ExecutionPlan",
    "Forward",
    "PD",
    "Cache",
    "KVEvent",
    # Events
    "ExecutionEvent",
    "ForwardEvent",
]
