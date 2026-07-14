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

"""Tests for PD Python bindings."""

from tokenspeed_scheduler import (
    PD,
    ExecutionEvent,
    RequestSpec,
    Scheduler,
    SchedulerConfig,
)


def make_scheduler() -> Scheduler:
    cfg = SchedulerConfig()
    cfg.block_size = 16
    cfg.max_scheduled_tokens = 32
    cfg.max_batch_size = 4
    cfg.num_device_pages = 1024
    return Scheduler(cfg)


def make_spec(request_id: str, tokens: list[int]) -> RequestSpec:
    spec = RequestSpec()
    spec.request_id = request_id
    spec.tokens = tokens
    return spec


def test_pd_event_fields_are_bound():
    """PD event objects require request_id constructor arg and expose it as read-only."""
    event = PD.BootstrappedEvent("req-0")

    assert event.request_id == "req-0"


def test_execution_event_accepts_pd_events():
    """ExecutionEvent.add_event accepts PD events and returns self for chaining."""
    execution_event = ExecutionEvent()
    event = PD.SucceededEvent("req-0")

    assert execution_event.add_event(event) is execution_event


def test_execution_plan_exposes_forward():
    scheduler = make_scheduler()
    scheduler.submit_requests([make_spec("r0", [1, 2, 3, 4])])

    plan = scheduler.next_execution_plan()

    assert len(plan.forward) == 1
    assert plan.forward[0].request_ids == ["r0"]
