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

"""Unit tests for the pause/resume controller (CPU-only, no scheduler/GPU)."""

from __future__ import annotations

from tokenspeed.runtime.engine.io_struct import (
    IsSchedulerPausedReqInput,
    IsSchedulerPausedReqOutput,
    PauseSchedulerReqInput,
    PauseSchedulerReqOutput,
    ResumeSchedulerReqInput,
    ResumeSchedulerReqOutput,
)
from tokenspeed.runtime.engine.pause import (
    PauseController,
    PauseState,
    scheduler_drained,
)


class _Sender:
    def __init__(self):
        self.items = []

    def send_pyobj(self, x):
        self.items.append(x)


class _FakeScheduler:
    """Implements just the four size accessors the drain check reads."""

    def __init__(self, waiting=0, decoding=0, prefilling=0, retract=0):
        self._w, self._d, self._p, self._r = waiting, decoding, prefilling, retract

    def waiting_size(self):
        return self._w

    def decoding_size(self):
        return self._d

    def prefilling_size(self):
        return self._p

    def retract_count(self):
        return self._r


def _controller():
    sender = _Sender()
    return PauseController(sender), sender


# --- scheduler_drained -------------------------------------------------------


def test_scheduler_drained_predicate():
    assert scheduler_drained(_FakeScheduler())
    assert not scheduler_drained(_FakeScheduler(waiting=1))
    assert not scheduler_drained(_FakeScheduler(decoding=1))
    assert not scheduler_drained(_FakeScheduler(prefilling=1))
    assert not scheduler_drained(_FakeScheduler(retract=1))


# --- keep mode ---------------------------------------------------------------


def test_keep_mode_freezes_and_replies_immediately():
    ctrl, sender = _controller()
    ctrl.handle_pause(PauseSchedulerReqInput(mode="keep"))

    assert ctrl.state == PauseState.PAUSED_ALL
    assert ctrl.is_paused
    assert ctrl.admit_blocked
    assert ctrl.forward_blocked  # keep stops all scheduling
    # Reply is sent right away — nothing to drain.
    assert len(sender.items) == 1
    assert isinstance(sender.items[0], PauseSchedulerReqOutput)
    assert sender.items[0].success
    # keep freezes in place: no abort-all, no grammar-queue cancel.
    assert not ctrl.consume_abort_all()
    assert not ctrl.consume_cancel_grammar()


# --- abort mode --------------------------------------------------------------


def test_abort_mode_defers_reply_until_drained():
    ctrl, sender = _controller()
    ctrl.handle_pause(PauseSchedulerReqInput(mode="abort"))

    assert ctrl.state == PauseState.PAUSED_NEW
    assert ctrl.admit_blocked
    assert not ctrl.forward_blocked  # running requests keep stepping to drain
    # No reply yet — deferred until the scheduler drains.
    assert sender.items == []
    # abort requests an all-cancel exactly once.
    assert ctrl.consume_abort_all()
    assert not ctrl.consume_abort_all()
    # abort also cancels grammar-queued (still-compiling) requests, once.
    assert ctrl.consume_cancel_grammar()
    assert not ctrl.consume_cancel_grammar()

    # Still busy → no reply.
    ctrl.maybe_finish_drain(_FakeScheduler(decoding=2))
    assert sender.items == []

    # Drained → reply now fires.
    ctrl.maybe_finish_drain(_FakeScheduler())
    assert len(sender.items) == 1
    assert isinstance(sender.items[0], PauseSchedulerReqOutput)
    assert sender.items[0].success

    # Idempotent: a second drain check does not double-reply.
    ctrl.maybe_finish_drain(_FakeScheduler())
    assert len(sender.items) == 1


def test_second_pause_while_draining_is_rejected():
    # A second abort/wait pause arriving before the first drains must be
    # rejected: otherwise it would overwrite the deferred reply and strand the
    # first caller forever on its ZMQ await.
    ctrl, sender = _controller()
    ctrl.handle_pause(PauseSchedulerReqInput(mode="abort"))
    ctrl.consume_abort_all()  # first pause armed its abort-all
    ctrl.consume_cancel_grammar()  # ...and its grammar-queue cancel
    assert sender.items == []  # first reply still deferred

    # Second pause (any mode) is rejected with a failure, leaving the first
    # pause's pending reply intact.
    for mode in ("abort", "wait", "keep"):
        sender.items.clear()
        ctrl.handle_pause(PauseSchedulerReqInput(mode=mode))
        assert len(sender.items) == 1
        assert isinstance(sender.items[0], PauseSchedulerReqOutput)
        assert not sender.items[0].success
        # Rejected pause does not re-arm abort-all / grammar cancel.
        assert not ctrl.consume_abort_all()
        assert not ctrl.consume_cancel_grammar()

    # The original pause still resolves normally once the scheduler drains.
    sender.items.clear()
    ctrl.maybe_finish_drain(_FakeScheduler())
    assert len(sender.items) == 1
    assert sender.items[0].success


# --- wait mode ---------------------------------------------------------------


def test_wait_mode_defers_reply_without_abort_all():
    ctrl, sender = _controller()
    ctrl.handle_pause(PauseSchedulerReqInput(mode="wait"))

    assert ctrl.state == PauseState.PAUSED_NEW
    assert not ctrl.forward_blocked
    assert sender.items == []
    # wait lets running requests finish naturally — no abort-all — but still
    # cancels grammar-queued pre-pause requests (they can't finish while paused).
    assert not ctrl.consume_abort_all()
    assert ctrl.consume_cancel_grammar()

    ctrl.maybe_finish_drain(_FakeScheduler())
    assert len(sender.items) == 1
    assert sender.items[0].success


# --- invalid mode ------------------------------------------------------------


def test_invalid_mode_replies_failure_and_stays_unpaused():
    ctrl, sender = _controller()
    ctrl.handle_pause(PauseSchedulerReqInput(mode="bogus"))
    assert ctrl.state == PauseState.UNPAUSED
    assert len(sender.items) == 1
    assert isinstance(sender.items[0], PauseSchedulerReqOutput)
    assert not sender.items[0].success


# --- resume ------------------------------------------------------------------


def test_resume_rejects_inflight_pause_then_acks():
    # A resume arriving while a wait/abort pause is still awaiting its drain
    # reply must resolve that pause (separate communicators — resume can't wake
    # the pause caller), then ack the resume. The pause reply must be a FAILURE:
    # the scheduler had not drained, so acking success would let a weight-swap
    # caller proceed while pre-pause requests are still in flight.
    ctrl, sender = _controller()
    ctrl.handle_pause(PauseSchedulerReqInput(mode="wait"))
    ctrl.handle_resume(ResumeSchedulerReqInput())

    assert ctrl.state == PauseState.UNPAUSED
    assert not ctrl.is_paused
    assert not ctrl.admit_blocked
    # Pending pause reply resolved (as failure) first, then the resume ack.
    assert len(sender.items) == 2
    assert isinstance(sender.items[0], PauseSchedulerReqOutput)
    assert not sender.items[0].success
    assert (
        isinstance(sender.items[1], ResumeSchedulerReqOutput)
        and sender.items[1].success
    )

    # The reply is consumed: a later drain check emits nothing more.
    ctrl.maybe_finish_drain(_FakeScheduler())
    assert len(sender.items) == 2


def test_resume_without_inflight_pause_only_acks():
    ctrl, sender = _controller()
    ctrl.handle_resume(ResumeSchedulerReqInput())
    assert ctrl.state == PauseState.UNPAUSED
    assert len(sender.items) == 1
    assert isinstance(sender.items[0], ResumeSchedulerReqOutput)


# --- is_paused ---------------------------------------------------------------


def test_is_paused_reflects_state():
    ctrl, sender = _controller()
    ctrl.handle_is_paused(IsSchedulerPausedReqInput())
    assert isinstance(sender.items[-1], IsSchedulerPausedReqOutput)
    assert sender.items[-1].is_paused is False

    ctrl.handle_pause(PauseSchedulerReqInput(mode="keep"))
    ctrl.handle_is_paused(IsSchedulerPausedReqInput())
    assert sender.items[-1].is_paused is True


# --- spec buffering ----------------------------------------------------------


def test_spec_buffering_is_fifo():
    ctrl, _ = _controller()
    ctrl.buffer_specs(["a", "b"])
    ctrl.buffer_specs(["c"])
    assert ctrl.buffered_specs == ["a", "b", "c"]
    assert ctrl.take_buffered_specs() == ["a", "b", "c"]
    # Drained after take.
    assert ctrl.buffered_specs == []
    assert ctrl.take_buffered_specs() == []
