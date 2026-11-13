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

"""Record deterministic public scheduler outputs for offline revision comparison.

Run with --module-dir pointing to an isolated native extension build and
--output selecting its JSON trace. Compare parsed traces from two revisions.
No production alternate scheduling path is introduced.
"""

import argparse
import enum
import json
import random
import sys
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--module-dir", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
sys.path.insert(0, args.module_dir)
import tokenspeed_scheduler_ext as ts  # noqa: E402


def canonical(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, enum.Enum):
        return {"enum": type(value).__qualname__, "name": value.name}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [canonical(item) for item in value]
    if isinstance(value, dict):
        return {str(key): canonical(item) for key, item in sorted(value.items())}
    fields = {"type": type(value).__qualname__}
    for name in sorted(dir(value)):
        if name.startswith("_"):
            continue
        item = getattr(value, name)
        if not callable(item):
            fields[name] = canonical(item)
    return fields


def config(case):
    cfg = ts.SchedulerConfig()
    cfg.prefix_granularity = 8
    cfg.num_device_pages = 97
    cfg.num_host_pages = 129 if case in (2, 3) else 1
    cfg.num_snapshot_pages = 97
    cfg.max_retracted_requests = 8
    cfg.max_scheduled_tokens = 32
    cfg.max_batch_size = 4
    cfg.decode_input_tokens = 1
    cfg.disable_l2_cache = case not in (2, 3)
    cfg.enable_l3_storage = False
    cfg.disable_prefix_cache = case == 5
    cfg.enable_kv_cache_events = case != 0
    cfg.enable_mixed_prefill_decode = True
    cfg.enable_short_prefill_first = True
    cfg.prefix_replay_tokens = 1
    cfg.overlap_schedule_depth = 0
    cfg.checkpoint_retention_requests = 2
    cfg.debug_force_retraction_interval = 5 if case in (3, 4) else 0
    cfg.role = ts.SchedulerConfig.Role.Fused
    cfg.host_write_policy = (
        ts.SchedulerConfig.HostWritePolicy.WriteBack
        if case == 3
        else ts.SchedulerConfig.HostWritePolicy.WriteThrough
    )
    packed = case in (1, 3, 4)
    groups = [
        ts.CacheGroupConfig(
            group_id="full",
            block_granularity=4 if packed else 8,
            total_pages=193 if packed else 97,
            attn_kind=ts.AttnKind.Full,
        )
    ]
    if packed:
        groups[0].cache_blocks_per_lcm_block = 2
        groups[0].shard_count = 2
    if case in (2, 3):
        groups.append(
            ts.CacheGroupConfig(
                group_id="state",
                block_granularity=0,
                total_pages=97,
                attn_kind=ts.AttnKind.State,
            )
        )
    if case in (3, 4):
        window = ts.CacheGroupConfig(
            group_id="window",
            block_granularity=4,
            total_pages=193,
            attn_kind=ts.AttnKind.SlidingWindow,
            sliding_window_tokens=12,
        )
        window.cache_blocks_per_lcm_block = 2
        window.shard_count = 2
        window.replayable = case == 4
        groups.append(window)
    cfg.cache_groups = groups
    cfg.host_group_packing = [
        2 if group.group_id != "state" and packed else 1 for group in groups
    ]
    return cfg


def send(scheduler, events):
    batch = ts.ExecutionEvent()
    for event in events:
        batch.add_event(event)
    scheduler.advance(batch)


def result(rid, tokens):
    event = ts.ForwardEvent.ExtendResult()
    event.request_id = rid
    event.tokens = tokens
    return event


records = []
for case in range(6):
    scheduler = ts.Scheduler(config(case))
    random_source = random.Random(391 + case)
    live = {}
    history = []
    pending = []
    serial = 0
    for round_id in range(160):
        due = [item for item in pending if item[0] <= round_id]
        pending = [item for item in pending if item[0] > round_id]
        events = []
        for _, kind, operation in sorted(due):
            event = {
                "write": ts.Cache.WriteBackDoneEvent,
                "load": ts.Cache.LoadBackDoneEvent,
                "snapshot": ts.Cache.SnapshotDoneEvent,
                "restore": ts.Cache.RestoreDoneEvent,
            }[kind]()
            event.op_id = operation
            events.append(event)
        if events:
            send(scheduler, events)
        submitted = []
        if round_id < 96 and len(live) < 9:
            for _ in range(2 if round_id % 3 == 0 else 1):
                length = random_source.choice([9, 23, 40, 63, 81])
                if history and serial % 3 == 0:
                    tokens = history[random_source.randrange(len(history))] + [
                        90000 + serial
                    ]
                else:
                    tokens = list(range(1, 17))[:length]
                    tokens += list(
                        range(
                            1000 + serial * 100,
                            1000 + serial * 100 + max(length - 16, 0),
                        )
                    )
                spec = ts.RequestSpec()
                spec.request_id = f"r{serial}"
                spec.tokens = tokens
                spec.max_new_tokens = 3
                submitted.append(spec)
                live[spec.request_id] = {"tokens": tokens, "generated": 0}
                serial += 1
            scheduler.submit_requests(submitted)
        aborted = None
        if round_id % 11 == 7 and live:
            aborted = sorted(live)[0]
            event = ts.ForwardEvent.Abort()
            event.request_id = aborted
            send(scheduler, [event])
            del live[aborted]
        plan = scheduler.next_execution_plan()
        for abort in plan.aborts:
            live.pop(abort.request_id, None)
        events = []
        for batch in plan.forward:
            extends = len(batch.extend_prefix_lens)
            for row, rid in enumerate(batch.request_ids):
                state = live[rid]
                ready = row >= extends or batch.extend_prefix_lens[
                    row
                ] + batch.input_lengths[row] >= len(state["tokens"])
                produced = []
                if ready:
                    produced = [700000 + int(rid[1:]) * 10 + state["generated"]]
                    state["generated"] += 1
                events.append(result(rid, produced))
                if state["generated"] == 3:
                    finish = ts.ForwardEvent.Finish()
                    finish.request_id = rid
                    events.append(finish)
                    history.append(state["tokens"])
                    del live[rid]
        acknowledgements = set()
        for operation in plan.cache:
            for cls, kind in [
                (ts.Cache.WriteBackOp, "write"),
                (ts.Cache.LoadBackOp, "load"),
                (ts.Cache.SnapshotOp, "snapshot"),
                (ts.Cache.RestoreOp, "restore"),
            ]:
                if isinstance(operation, cls):
                    for operation_id in operation.op_ids:
                        acknowledgements.add((kind, operation_id))
        pending.extend(
            (round_id + 2, kind, op) for kind, op in sorted(acknowledgements)
        )
        if events:
            send(scheduler, events)
        records.append(
            {
                "case": case,
                "round": round_id,
                "submitted": canonical(submitted),
                "abort": aborted,
                "plan": canonical(plan),
                "events": canonical(scheduler.drain_kv_events()),
                "stats": canonical(scheduler.host_cache_stats()),
                "counts": [
                    scheduler.waiting_size(),
                    scheduler.active_lcm_blocks(),
                    scheduler.empty_lcm_blocks(),
                    scheduler.retracted_size(),
                    scheduler.host_pool_cached_blocks(),
                    scheduler.host_pool_free_blocks(),
                    scheduler.snapshot_pool_free_blocks(),
                    scheduler.tail_entries(),
                ],
            }
        )
    del scheduler
Path(args.output).write_text(
    json.dumps(records, sort_keys=True, separators=(",", ":")) + "\n"
)
print(f"Wrote {len(records)} plan/lifecycle snapshots to {args.output}")
