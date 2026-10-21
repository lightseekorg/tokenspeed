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

"""Real 2x2 TP/PP Gloo collectives, runtime hooks and mirrored schedulers."""

import json
import threading
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest
import tokenspeed_scheduler as ts
import torch.distributed as dist
import torch.multiprocessing as mp

from tokenspeed.runtime.engine.cache_hooks import CacheOpHooks
from tokenspeed.runtime.execution.forward_thread import ForwardThread


def config(*, snapshot, overlap, host, batch_size=4):
    c = ts.SchedulerConfig()
    c.prefix_granularity = 8
    c.max_scheduled_tokens = 1024
    c.max_batch_size = batch_size
    c.num_device_pages = 128
    c.num_host_pages = 128 if host else 0
    c.disable_l2_cache = not host
    c.num_snapshot_pages = 128 if snapshot else 1
    c.max_retracted_requests = 4 if snapshot else 0
    c.decode_input_tokens = 8
    c.overlap_schedule_depth = overlap
    c.enable_l3_storage = False
    c.disable_prefix_cache = False
    c.cache_groups = [
        ts.CacheGroupConfig(
            group_id="kv",
            block_granularity=8,
            total_pages=128,
            attn_kind=ts.AttnKind.Full,
        ),
        ts.CacheGroupConfig(
            group_id="state",
            block_granularity=0,
            total_pages=128,
            attn_kind=ts.AttnKind.State,
        ),
    ]
    return c


def scheduler(*, snapshot=False, overlap=0, host=False, deferred=True, batch_size=4):
    c = config(snapshot=snapshot, overlap=overlap, host=host, batch_size=batch_size)
    return ts.Scheduler(c, ["state"]) if deferred else ts.Scheduler(c)


def send(s, *events):
    e = ts.ExecutionEvent()
    for item in events:
        e.add_event(item)
    s.advance(e)


def result(s, rid, tokens):
    e = ts.ForwardEvent.ExtendResult()
    e.request_id, e.tokens = rid, tokens
    send(s, e)


def control(s, rid, kind):
    e = kind()
    e.request_id = rid
    send(s, e)


def submit(s, rid, tokens):
    spec = ts.RequestSpec()
    spec.request_id, spec.tokens, spec.max_new_tokens = rid, tokens, 100
    s.submit_requests([spec])


def forwards(plan):
    return [b for b in plan.forward if b.request_ids]


def row(batch, rid):
    i = list(batch.request_ids).index(rid)
    return list(batch.block_tables["state"][i])


def ack_writebacks(s, plan):
    for op in plan.cache:
        if isinstance(op, ts.Cache.WriteBackOp):
            for op_id in op.op_ids:
                event = ts.Cache.WriteBackDoneEvent()
                event.op_id = int(op_id)
                send(s, event)


def started(s, rid="r"):
    submit(s, rid, list(range(125)))
    p = s.next_execution_plan()
    ack_writebacks(s, p)
    assert len(forwards(p)) == 1
    result(s, rid, [1000])
    p = s.next_execution_plan()
    ack_writebacks(s, p)
    b = forwards(p)[0]
    assert not p.state_prepares
    assert b.num_extends() == 0
    result(s, rid, [1001, 1002, 1003, 1004, 1005])
    return int(b.request_pool_indices[0]), row(b, rid)[1]


class CpuDevice:
    """Only missing GPU execution/events are substituted; FIFO is real TS."""

    def __init__(self, *, success):
        self._success = success
        self._ready = threading.Event()
        self._thread = ForwardThread("cpu")
        self._submissions = []
        self._completed = []
        self._lock = threading.Lock()

    def execute(self, plan):
        for op in plan.state_prepares:
            # Captured op is immutable; the scheduler holds its resources.
            def prepare(op=op):
                with self._lock:
                    self._completed.append(op)

            self._submissions.append(self._thread.submit(prepare))

    def drain_submissions(self):
        for future in self._submissions:
            future.result(timeout=10)
        self._submissions.clear()

    def complete(self):
        self._ready.set()

    def poll_cache_results(self):
        if not self._ready.is_set():
            return []
        with self._lock:
            completed, self._completed = self._completed, []
        return [
            ts.Cache.StatePrepareDoneEvent(op.op_id, self._success) for op in completed
        ]

    def consume_l3_backup_poll_failure(self):
        return False

    def shutdown(self):
        self._thread.shutdown()


def worker(rank, init, out_dir, success):
    dist.init_process_group(
        "gloo", init_method=init, rank=rank, world_size=4, timeout=timedelta(seconds=40)
    )
    tp = [dist.new_group([0, 1]), dist.new_group([2, 3])]
    pp = [dist.new_group([0, 2]), dist.new_group([1, 3])]
    s = scheduler()
    started(s)
    control(s, "r", ts.ForwardEvent.Finish)
    plan = s.next_execution_plan()
    assert len(plan.state_prepares) == 1
    # Rank 3 can fail after its writer drains; every rank must agree on
    # failure, not take the result from rank 0.
    device = CpuDevice(success=success or rank != 3)
    hooks = CacheOpHooks(
        device,
        speculative_algorithm="DUMMY",
        attn_tp_rank=rank % 2,
        attn_tp_size=2,
        attn_tp_cpu_group=tp[rank // 2],
        pp_size=2,
        pp_cpu_group=pp[rank % 2],
        global_rank=rank,
    )
    hooks.count_plan_ops(plan)
    device.execute(plan)
    device.drain_submissions()
    # Stagger readiness through both replica groups. Empty intersections
    # still participate in the second group, so this cannot deadlock.
    checks = []
    for step in range(4):
        if rank == step:
            device.complete()
        events = hooks.poll_ready_events()
        if step < 3:
            assert events == []
            assert s.active_lcm_blocks() > 0
        else:
            assert len(events) == 1 and events[0].success == success
            send(s, *events)
            assert s.active_lcm_blocks() == 0
        checks.append(
            {
                "step": step,
                "events": len(events),
                "active_blocks": s.active_lcm_blocks(),
            }
        )
    # Ticket was consumed once. No unmatched gather on the idle round.
    assert hooks.poll_ready_events() == []
    Path(out_dir, f"rank-{rank}-success-{success}.json").write_text(
        json.dumps(checks, indent=2)
    )
    device.shutdown()
    dist.destroy_process_group()


@pytest.mark.parametrize("success", [True, False])
def test_real_gloo_all_rank_readiness_and_failure(tmp_path, success):
    init = f"file://{tmp_path}/rendezvous"
    mp.start_processes(
        worker,
        args=(init, str(tmp_path), success),
        nprocs=4,
        join=True,
        start_method="fork",
    )
    assert len(list(tmp_path.glob("rank-*.json"))) == 4
