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
from tokenspeed.runtime.engine.io_struct import (  # noqa: E402
    NullSender,
    PauseSchedulerReqInput,
    ReleaseMemoryOccupationReqInput,
    ResumeSchedulerReqInput,
)
from tokenspeed.runtime.engine.memory_occupation import (  # noqa: E402
    MemoryOccupationController,
    MemoryRegion,
)
from tokenspeed.runtime.engine.pause import (  # noqa: E402
    PauseController,
    PauseHooks,
    PauseState,
)
from tokenspeed.runtime.pd.base.bootstrap import BootstrapInfo  # noqa: E402


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
        # PD lifecycle counts the release gate reads.
        self.bootstrapping = 0
        self.pd_transfers = 0

    def submit_requests(self, specs: list) -> None:
        self.submitted.append(list(specs))

    def bootstrapping_size(self) -> int:
        return self.bootstrapping

    def pd_transfer_size(self) -> int:
        return self.pd_transfers

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
        self.kv_transfer = None

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


class _Transfer:
    def __init__(self) -> None:
        self.registered: list[tuple[str, BootstrapInfo]] = []

    def register(self, request_id: str, bootstrap: BootstrapInfo) -> None:
        self.registered.append((request_id, bootstrap))


@pytest.mark.parametrize("state", [PauseState.PAUSED_NEW, PauseState.PAUSED_ALL])
def test_buffered_pd_handshake_starts_only_when_live_request_is_admitted(state):
    hooks, loop = _hooks(
        rid_to_state={"live": _State(), "aborted": _State(finished=True)}
    )
    transfer = _Transfer()
    loop.kv_transfer = transfer
    hooks._pause.state = state
    specs = [_spec("live"), _spec("aborted"), _spec("gone")]
    infos = [BootstrapInfo("127.0.0.1", 1234, i) for i in range(3)]
    for spec, info in zip(specs, infos, strict=True):
        hooks.register_pd_admission(spec.request_id, info)
    assert hooks.withhold_admissions(specs, True)
    hooks.apply_transitions(_GrammarManager())
    assert transfer.registered == []

    hooks._pause.handle_resume(ResumeSchedulerReqInput())
    hooks.apply_transitions(_GrammarManager())
    assert transfer.registered == [("live", infos[0])]
    assert loop.scheduler.submitted == [[specs[0]]]
    assert hooks._pause.buffered_pd_bootstraps == {}
    hooks.apply_transitions(_GrammarManager())
    assert transfer.registered == [("live", infos[0])]


def test_unpaused_pd_handshake_starts_immediately():
    hooks, loop = _hooks()
    transfer = _Transfer()
    loop.kv_transfer = transfer
    info = BootstrapInfo("127.0.0.1", 1234, 42)
    hooks.register_pd_admission("live", info)
    assert transfer.registered == [("live", info)]


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


class _Device:
    """Records the data-plane work the memory regions hand to the handle."""

    def __init__(self, trace: list[str]) -> None:
        self._trace = trace

    def synchronize(self) -> None:
        self._trace.append("synchronize")

    def prepare_kv_release(self) -> None:
        self._trace.append("prepare_kv_release")

    def finish_kv_resume(self) -> None:
        self._trace.append("finish_kv_resume")

    def suspend_nccl(self) -> None:
        self._trace.append("suspend_nccl")

    def resume_nccl(self) -> None:
        self._trace.append("resume_nccl")


class _Saver:
    def __init__(self, trace: list[str]) -> None:
        self._trace = trace

    def pause(self, tag: str) -> None:
        self._trace.append(f"pause:{tag}")

    def resume(self, tag: str) -> None:
        self._trace.append(f"resume:{tag}")


def test_kv_region_quiesces_before_unmapping_and_repairs_after_remapping() -> None:
    trace: list[str] = []
    hooks, _ = _hooks(device=_Device(trace))
    regions = hooks.memory_regions(_Saver(trace), nccl_suspendable=True)

    regions["kv_cache"].release()
    regions["kv_cache"].resume()

    # Which pools and peers get walked is the data plane's business; this
    # side only orders the handle's work around the unmap and the re-map.
    assert trace == [
        "prepare_kv_release",
        "pause:kv_cache",
        "resume:kv_cache",
        "finish_kv_resume",
    ]


def test_allocator_regions_synchronize_before_unmapping_and_nccl_suspends() -> None:
    trace: list[str] = []
    hooks, _ = _hooks(device=_Device(trace))
    regions = hooks.memory_regions(_Saver(trace), nccl_suspendable=True)

    for tag in ("weights", "cuda_graph", "nccl"):
        regions[tag].release()
    for tag in ("nccl", "cuda_graph", "weights"):
        regions[tag].resume()

    assert trace == [
        "synchronize",
        "pause:weights",
        "synchronize",
        "pause:cuda_graph",
        "suspend_nccl",
        "resume_nccl",
        "resume:cuda_graph",
        "resume:weights",
    ]


def test_no_nccl_region_without_an_nccl_that_can_suspend() -> None:
    trace: list[str] = []
    hooks, _ = _hooks(device=_Device(trace))

    regions = hooks.memory_regions(_Saver(trace), nccl_suspendable=False)

    assert sorted(regions) == ["cuda_graph", "kv_cache", "weights"]


class _TransferPeer:
    def __init__(self) -> None:
        self.inflight = False

    def has_inflight_transfers(self) -> bool:
        return self.inflight


def test_transfers_idle_needs_no_bootstrap_no_pinned_pages_and_an_idle_peer() -> None:
    hooks, loop = _hooks()
    # Outside PD: no peer, nothing bootstrapping or pinned.
    assert hooks.transfers_idle_for_release()

    peer = _TransferPeer()
    loop.kv_transfer = peer
    assert hooks.transfers_idle_for_release()

    loop.scheduler.bootstrapping = 1
    assert not hooks.transfers_idle_for_release()
    loop.scheduler.bootstrapping = 0
    loop.scheduler.pd_transfers = 2
    assert not hooks.transfers_idle_for_release()
    loop.scheduler.pd_transfers = 0
    # The request is gone, but its sender/receiver is still registered.
    peer.inflight = True
    assert not hooks.transfers_idle_for_release()


@pytest.mark.parametrize("mode", ["wait", "keep"])
@pytest.mark.parametrize("pd", [False, True])
def test_release_with_buffered_specs_refuses_only_pd_without_changing_pause(mode, pd):
    class Sender:
        def __init__(self):
            self.items = []

        def send_pyobj(self, reply):
            self.items.append(reply)

    sender = Sender()
    pause = PauseController(sender)
    loop = _FakeLoop()
    if pd:
        loop.kv_transfer = _TransferPeer()
        loop.kv_transfer.inflight = True
    hooks = PauseHooks(loop, pause, None)
    calls = []
    memory = MemoryOccupationController(
        send_func=sender,
        pause_controller=pause,
        regions={
            "weights": MemoryRegion(
                release=lambda: calls.append("release"), resume=lambda: None
            )
        },
        enabled=True,
        reset_caches_fn=lambda: True,
        transfers_idle_fn=hooks.transfers_idle_for_release,
        release_refusal_fn=hooks.memory_release_refusal,
        weights_busy_fn=lambda: False,
    )
    pause.handle_pause(PauseSchedulerReqInput(mode=mode))
    pause.maybe_finish_drain(loop.scheduler)
    pause.buffer_specs([_spec("buffered")])
    prior_state = pause.state

    memory.handle_release(ReleaseMemoryOccupationReqInput(tags=["weights"]))

    if pd:
        assert not sender.items[-1].success
        assert "buffered" in sender.items[-1].message
        assert pause.state == prior_state and not pause.is_drain_pending
        assert not memory.is_releasing_or_released and calls == []
        # Follow the refusal's recovery instruction: resume, admit and drain
        # the buffered request before asking for a release again.
        pause.handle_resume(ResumeSchedulerReqInput())
        assert len(pause.take_buffered_specs()) == 1
        loop.kv_transfer.inflight = False
        memory.handle_release(ReleaseMemoryOccupationReqInput(tags=["weights"]))
    pause.maybe_finish_drain(loop.scheduler)
    assert sender.items[-1].success and calls == ["release"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
