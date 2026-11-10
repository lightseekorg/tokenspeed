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

"""Python API for the TokenSpeed scheduler.

``Forward.Batch.input_ids`` and ``shifted_input_ids`` are one-dimensional,
read-only NumPy int32 views. Repeated reads share the batch's token storage;
each view keeps its batch alive independently of the execution plan. Both
arrays are empty on decode-only batches. Use ``.tolist()`` for Python integers,
or ``np.array(view, dtype=np.int32, copy=True)`` for writable storage (including
when passing the data to PyTorch). The explicit dtype also accepts token lists
from older scheduler releases without changing the result's dtype.

``Forward.Batch.block_table_windows()`` gives each group's read-only int32
row windows and its absolute column count. Each window starts at
``max(pages - window_width, 0)``. The ``(released, pages)`` pairs from
``block_table_spans()`` mark the released prefix (0) and the start of
row padding (-1). ``block_tables_arrays()`` materializes the absolute rows
on demand and returns read-only views of the batch's cached tables.
All three views keep the batch alive.
"""

import tokenspeed_scheduler.tokenspeed_scheduler_ext as _ext
from tokenspeed_scheduler.tokenspeed_scheduler_ext import (  # Core; Execution plan; Events
    AbortReason,
    AttnKind,
    CacheGroupConfig,
    CacheTransferPolicy,
    CapacityModel,
    ExecutionEvent,
    ExecutionPlan,
    PrefillComputeConfig,
    RequestSpec,
    Scheduler,
    SchedulerAbort,
    SchedulerConfig,
)

StorageStats = _ext.StorageStats
HostCacheStats = _ext.HostCacheStats
StorageConfig = _ext.StorageConfig
StoragePressure = _ext.StoragePressure
StorageKind = _ext.StorageKind

PD = _ext.PD
Cache = _ext.Cache
Forward = _ext.Forward
ForwardEvent = _ext.ForwardEvent
KVEvent = _ext.KVEvent


def _forward_batch_repr(self):
    return (
        f"ForwardBatch("
        f"request_ids={list(self.request_ids)}, "
        f"request_pool_indices={list(self.request_pool_indices)}, "
        f"input_lengths={list(self.input_lengths)}, "
        f"input_ids={self.input_ids.tolist()}, "
        f"shifted_input_ids={self.shifted_input_ids.tolist()}, "
        f"extend_prefix_lens={list(self.extend_prefix_lens)}, "
        f"extend_replay_lens={list(self.extend_replay_lens)}, "
        f"num_extends={self.num_extends()}"
        f")"
    )


Forward.Batch.__repr__ = _forward_batch_repr

__all__ = [
    # Core
    "Scheduler",
    "SchedulerConfig",
    "PrefillComputeConfig",
    "StorageConfig",
    "StoragePressure",
    "StorageStats",
    "HostCacheStats",
    "StorageKind",
    "RequestSpec",
    "AttnKind",
    "CacheGroupConfig",
    "CacheTransferPolicy",
    "CapacityModel",
    # Execution plan & operations
    "ExecutionPlan",
    "SchedulerAbort",
    "AbortReason",
    "Forward",
    "PD",
    "Cache",
    "KVEvent",
    # Events
    "ExecutionEvent",
    "ForwardEvent",
]
