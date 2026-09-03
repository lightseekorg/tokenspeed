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

"""CPU-only tests for PauseHooks, the event loop side of pause/resume.

The PauseController state machine is covered in test_pause_controller.py;
these exercise PauseHooks against a fake event loop, so no model, CUDA
context, or scheduler is created.
"""

from __future__ import annotations

import logging
import os
import sys
from types import SimpleNamespace

import pytest

# CPU-only tests scheduled in runtime-1gpu because they import the full runtime.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed.runtime.engine.event_loop import EventLoop  # noqa: E402
from tokenspeed.runtime.engine.io_struct import NullSender  # noqa: E402
from tokenspeed.runtime.engine.pause import (  # noqa: E402
    PauseController,
    PauseHooks,
    PauseState,
)


class _State:
    def __init__(self, *, finished: bool = False) -> None:
        self.finished = finished
        self.finish_calls: list[tuple[str, bool]] = []

    def set_finish_with_abort(self, message: str, notify_client: bool = False) -> None:
        self.finish_calls.append((message, notify_client))


class _OutputProcessor:
    def __init__(self, rid_to_state: dict) -> None:
        self.rid_to_state = rid_to_state
        self.aborted: list[tuple[str, bool]] = []
        self.reaped: list[str] = []

    def mark_abort(self, request_id: str, notify_client: bool = False) -> None:
        self.aborted.append((request_id, notify_client))

    def reap_finished_orphan(self, request_id: str, _state) -> None:
        self.reaped.append(request_id)


class _GrammarManager:
    def __init__(self, queued_states=()) -> None:
        self.grammar_queue = [(None, state, None) for state in queued_states]
        self.aborted: list[str] = []

    def mark_abort(self, rid: str) -> None:
        self.aborted.append(rid)


class _Scheduler:
    """Drained live-state accessors + the submit sink the hooks act on."""

    def __init__(self) -> None:
        self.submitted: list[list] = []

    def submit_requests(self, specs: list) -> None:
        self.submitted.append(list(specs))

    def waiting_size(self) -> int:
        return 0

    def decoding_size(self) -> int:
        return 0

    def prefilling_size(self) -> int:
        return 0


class _FakeLoop:
    """Only the loop state PauseHooks reads."""

    # The real abort marker so the notify_client contract stays honest.
    _request_abort_or_mark = EventLoop._request_abort_or_mark

    def __init__(self, rid_to_state: dict | None = None) -> None:
        self.output_processor = _OutputProcessor(rid_to_state or {})
        self.scheduler = _Scheduler()
        self.has_dp = False

    def _submit_scheduler_requests(self, specs: list) -> None:
        self.scheduler.submit_requests(specs)


def _hooks(
    rid_to_state: dict | None = None, device=None
) -> tuple[PauseHooks, _FakeLoop]:
    loop = _FakeLoop(rid_to_state)
    hooks = PauseHooks(loop, PauseController(NullSender()), device)
    return hooks, loop


def _spec(rid: str):
    return SimpleNamespace(request_id=rid)


# --- apply_transitions --------------------------------------------------------


def test_abort_pause_edge_cancels_inflight_and_grammar_queue() -> None:
    hooks, loop = _hooks(rid_to_state={"r0": _State(), "r1": _State()})
    hooks._pause.request_drain(
        abort_inflight=True, on_drained=lambda: None, on_cancelled=lambda: None
    )
    queued = _State()
    grammar_manager = _GrammarManager(queued_states=[queued])

    hooks.apply_transitions(grammar_manager)

    # In-flight requests aborted with a terminating finish for the passive client.
    assert loop.output_processor.aborted == [("r0", True), ("r1", True)]
    assert grammar_manager.aborted == ["r0", "r1"]
    # Still-compiling requests finished so they are published, not admitted.
    assert queued.finish_calls == [("Aborted by pause", True)]

    # Edges are one-shot: a second pass in the same pause does nothing.
    hooks.apply_transitions(grammar_manager)
    assert loop.output_processor.aborted == [("r0", True), ("r1", True)]
    assert queued.finish_calls == [("Aborted by pause", True)]


def test_wait_pause_edge_cancels_grammar_queue_but_not_inflight() -> None:
    hooks, loop = _hooks(rid_to_state={"r0": _State()})
    hooks._pause.request_drain(
        abort_inflight=False, on_drained=lambda: None, on_cancelled=lambda: None
    )
    queued = _State()
    grammar_manager = _GrammarManager(queued_states=[queued])

    hooks.apply_transitions(grammar_manager)

    assert loop.output_processor.aborted == []
    assert queued.finish_calls == [("Aborted by pause", True)]


def test_resume_flushes_live_buffered_specs_and_reaps_finished() -> None:
    live_state = _State()
    done_state = _State(finished=True)
    hooks, loop = _hooks(rid_to_state={"live": live_state, "done": done_state})
    hooks._pause.state = PauseState.PAUSED_NEW
    live, done, gone = _spec("live"), _spec("done"), _spec("gone")
    hooks._pause.buffer_specs([live, done, gone])

    # Still paused: nothing flushed.
    hooks.apply_transitions(_GrammarManager())
    assert loop.scheduler.submitted == []

    # Resumed: live specs re-admitted; aborted-while-paused ones reaped in
    # place; already-reaped ones dropped silently.
    hooks._pause.state = PauseState.UNPAUSED
    hooks.apply_transitions(_GrammarManager())
    assert loop.scheduler.submitted == [[live]]
    assert loop.output_processor.reaped == ["done"]
    assert hooks._pause.buffered_specs == []


# --- withhold_admissions ------------------------------------------------------


def test_admission_gate_passes_through_when_unpaused() -> None:
    hooks, _ = _hooks()
    specs = [_spec("r0")]

    assert not hooks.withhold_admissions(specs, True)
    assert hooks._pause.buffered_specs == []


def test_admission_gate_buffers_specs_while_paused() -> None:
    hooks, _ = _hooks()
    hooks._pause.state = PauseState.PAUSED_NEW
    specs = [_spec("r0"), _spec("r1")]

    assert hooks.withhold_admissions(specs, True)
    assert hooks._pause.buffered_specs == specs


def test_admission_gate_warns_when_pause_coalesced_with_requests(
    caplog: pytest.LogCaptureFixture,
) -> None:
    hooks, _ = _hooks()
    hooks._pause.state = PauseState.PAUSED_NEW

    # pause_blocked_before=False: the pause flipped in this very recv batch,
    # so the FIFO-order caveat must be surfaced (see TODO(pause-fifo)).
    with caplog.at_level(logging.WARNING):
        assert hooks.withhold_admissions([_spec("r0")], False)
    assert "TODO(pause-fifo)" in caplog.text


# --- paused_idle_step ---------------------------------------------------------


def test_paused_idle_step_leaves_the_drain_check_to_the_loop_tail() -> None:
    # The event loop resolves the drain at its tail, after the round's results
    # have advanced the scheduler; paused_idle_step must not consume it early.
    hooks, _ = _hooks()
    hooks._pause.request_drain(
        abort_inflight=False,
        on_drained=lambda: pytest.fail("drain must resolve at the loop tail"),
        on_cancelled=lambda: None,
    )

    hooks.paused_idle_step()

    assert hooks._pause.is_drain_pending


# --- memory release/wake data plane -------------------------------------------


def test_reset_caches_for_release_uses_scheduler_clear_when_present() -> None:
    hooks, loop = _hooks()
    # No clear_l1_cache on the scheduler: nothing to invalidate, release allowed.
    assert hooks.reset_caches_for_release()

    loop.scheduler.clear_l1_cache = lambda: False
    assert not hooks.reset_caches_for_release()
    loop.scheduler.clear_l1_cache = lambda: True
    assert hooks.reset_caches_for_release()


def test_kv_repair_after_wake_delegates_to_the_injected_device() -> None:
    cleared: list[str] = []
    hooks, _ = _hooks(
        device=SimpleNamespace(run_kv_repair=lambda: cleared.append("repair"))
    )

    hooks.kv_repair_after_wake()

    # Which pools get walked is the data plane's business; this side only
    # says when, and cannot reach a pool to walk it itself.
    assert cleared == ["repair"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
