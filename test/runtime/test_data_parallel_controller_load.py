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

"""Tests for DP routing from scheduler-published load snapshots."""

from __future__ import annotations

from types import SimpleNamespace

from tokenspeed.runtime.engine.data_parallel_controller import (
    DataParallelController,
    DPBudget,
    LoadBalanceMethod,
)
from tokenspeed.runtime.engine.io_struct import GetLoadReqOutput, LoadSnapshot
from tokenspeed.runtime.engine.load_snapshot import LoadSnapshotStore


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class RecordingWorker:
    def __init__(self) -> None:
        self.sent = []

    def send_pyobj(self, value) -> None:
        self.sent.append(value)


def snapshot(dp_rank: int, **overrides) -> LoadSnapshot:
    fields = dict(
        epoch="boot-a",
        sequence=1,
        dp_rank=dp_rank,
        num_running_reqs=0,
        num_waiting_reqs=0,
        num_active_pages=0,
        num_used_pages=0,
        max_total_pages=100,
        valid_for_ms=1_000,
    )
    fields.update(overrides)
    return LoadSnapshot(**fields)


def make_controller(
    *, clock: FakeClock | None = None, dp_size: int = 2
) -> DataParallelController:
    controller = DataParallelController.__new__(DataParallelController)
    controller.server_args = SimpleNamespace(disaggregation_mode="null")
    controller.workers = [RecordingWorker() for _ in range(dp_size)]
    controller.round_robin_counter = 0
    controller.load_balance_method = LoadBalanceMethod.SHORTEST_QUEUE
    controller.dp_budget = DPBudget(controller.load_balance_method)
    controller.load_snapshot_store = LoadSnapshotStore(
        dp_size=dp_size, clock=clock or FakeClock()
    )
    controller.dispatching = controller.budget_scheduler
    controller.init_dispatcher()
    return controller


def test_snapshot_is_consumed_locally_and_never_broadcast():
    controller = make_controller()

    controller._request_dispatcher(snapshot(0))

    assert controller.load_snapshot_store.fresh_snapshots() == [snapshot(0)]
    assert [worker.sent for worker in controller.workers] == [[], []]


def test_complete_fresh_snapshots_route_from_shortest_queue_budget():
    controller = make_controller()
    request = object()
    controller.handle_load_snapshot(snapshot(0, num_running_reqs=0))
    controller.handle_load_snapshot(snapshot(1, num_running_reqs=2))

    controller.budget_scheduler(request)

    assert controller.workers[0].sent == [request]
    assert controller.workers[1].sent == []


def test_partial_snapshot_set_clears_old_budget_and_falls_back_to_round_robin():
    controller = make_controller()
    request = object()
    controller.dp_budget.update_budget(
        [
            GetLoadReqOutput(dp_rank=0, num_reqs=2),
            GetLoadReqOutput(dp_rank=1, num_reqs=0),
        ]
    )

    controller.handle_load_snapshot(snapshot(0))
    controller.budget_scheduler(request)

    assert controller.workers[0].sent == [request]
    assert controller.workers[1].sent == []


def test_expired_snapshot_set_clears_budget_and_falls_back_to_round_robin():
    clock = FakeClock()
    controller = make_controller(clock=clock)
    request = object()
    controller.handle_load_snapshot(snapshot(0, num_running_reqs=2))
    controller.handle_load_snapshot(snapshot(1, num_running_reqs=0))
    clock.advance(1.001)

    controller.budget_scheduler(request)

    assert controller.workers[0].sent == [request]
    assert controller.workers[1].sent == []


def test_duplicate_and_out_of_order_snapshots_do_not_replace_current_budget():
    controller = make_controller()
    first_request = object()
    second_request = object()
    controller.handle_load_snapshot(snapshot(0, sequence=2, num_running_reqs=0))
    controller.handle_load_snapshot(snapshot(1, sequence=2, num_running_reqs=2))

    controller.handle_load_snapshot(snapshot(0, sequence=2, num_running_reqs=100))
    controller.handle_load_snapshot(snapshot(0, sequence=1, num_running_reqs=100))
    controller.budget_scheduler(first_request)
    controller.budget_scheduler(second_request)

    assert controller.workers[0].sent == [first_request, second_request]
    assert controller.workers[1].sent == []


def test_fresh_budget_is_not_rebuilt_on_every_dispatch():
    controller = make_controller()
    requests = [object(), object(), object()]
    controller.handle_load_snapshot(snapshot(0, num_running_reqs=2))
    controller.handle_load_snapshot(snapshot(1, num_running_reqs=0))

    for request in requests:
        controller.budget_scheduler(request)

    assert controller.workers[0].sent == [requests[2]]
    assert controller.workers[1].sent == requests[:2]
