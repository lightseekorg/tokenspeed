"""CPU-only runtime shutdown regression tests.

These tests deliberately bypass the heavyweight constructors.  They exercise
the real loop/watchdog methods with fakes, so no model, CUDA context, ZMQ
endpoint, or Mooncake native transfer is created.
"""

from __future__ import annotations

import os
import signal
import sys
import threading
from types import SimpleNamespace

import pytest

# CPU-only tests scheduled in runtime-1gpu because they import the full runtime.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed.runtime.engine import event_loop as event_loop_module  # noqa: E402
from tokenspeed.runtime.engine.event_loop import EventLoop  # noqa: E402


class _PauseHarness:
    forward_blocked = False

    def __init__(self, trace: list[str]) -> None:
        self._trace = trace

    def maybe_finish_drain(self, _scheduler) -> None:
        self._trace.append("pause_finish")


class _SchedulerHarness:
    def __init__(self, trace: list[str]) -> None:
        self._trace = trace

    def next_execution_plan(self):
        self._trace.append("next_plan")
        return SimpleNamespace(pages_to_zero=(), cache=())


class _DeviceHarness:
    def __init__(self, trace: list[str]) -> None:
        self._trace = trace

    def execute(self, execution_plan, planned):
        # The harness plans no device work and no batch; trace anything that
        # does appear rather than fail on a missing attr.
        if execution_plan.pages_to_zero or execution_plan.cache or planned:
            self._trace.append("execute")
        return None


class _EventLoopHarness:
    """Only the state read by ``EventLoop.event_loop``."""

    def __init__(self, *, pre_set: bool) -> None:
        self.trace: list[str] = []
        self.shutdown_event = threading.Event()
        if pre_set:
            self.shutdown_event.set()
        self._pause = _PauseHarness(self.trace)
        self.scheduler = _SchedulerHarness(self.trace)
        self._device = _DeviceHarness(self.trace)
        self.output_processor = SimpleNamespace(rid_to_state={})
        self.has_dp = False
        self.kv_transfer = None
        self.in_flight_depth = 0
        self._epd_hooks = SimpleNamespace(
            drain_ready_embeddings=lambda: self.trace.append("drain_epd")
        )
        self._cache_hooks = SimpleNamespace(
            poll_ready_events=lambda: (self.trace.append("poll_cache"), [])[1],
            count_plan_ops=lambda _plan: self.trace.append("count_cache"),
        )
        self._pd_hooks = SimpleNamespace(
            poll_transfer_events=lambda: (self.trace.append("poll_pd"), [])[1]
        )
        self.load_reporter = SimpleNamespace(
            observe=lambda _stats, _running: self.trace.append("observe_load"),
            sample_and_observe=lambda _running: self.trace.append("sample_load"),
            close=lambda: self.trace.append("close_load"),
        )

    def _shutdown_complete(self) -> bool:
        return EventLoop._shutdown_complete(self)

    def _process_new_requests(self) -> None:
        self.trace.append("process_requests")
        # Exercise the important case where SIGTERM arrives during an
        # iteration: finish this scheduler step, then stop at the next head.
        self.shutdown_event.set()

    def _publish_scheduler_kv_events(self) -> None:
        self.trace.append("publish_kv")

    def _get_forward_op(self, _execution_plan):
        self.trace.append("get_forward")
        return None

    def _get_scheduler_stats(self):
        self.trace.append("stats")
        return object()

    def _num_running(self) -> int:
        return 0

    def _record_scheduler_iteration_metrics(
        self, _stats, _num_iter_tokens: int
    ) -> None:
        self.trace.append("metrics")


def test_event_loop_returns_without_work_when_shutdown_is_pre_set() -> None:
    loop = _EventLoopHarness(pre_set=True)

    EventLoop.event_loop(loop)

    assert loop.trace == []


def test_event_loop_finishes_current_iteration_then_observes_shutdown() -> None:
    loop = _EventLoopHarness(pre_set=False)

    EventLoop.event_loop(loop)

    assert loop.trace == [
        "process_requests",
        "drain_epd",
        "poll_cache",
        "next_plan",
        # No "zero_pages": the round's plan plans no page, so the loop
        # submits no zeroing work to the forward thread.
        "count_cache",
        "get_forward",
        "stats",
        "observe_load",
        "metrics",
        "poll_pd",
        "publish_kv",
        "pause_finish",
    ]


def test_abort_uses_the_output_marker() -> None:
    calls = []
    output = SimpleNamespace(
        mark_abort=lambda request_id, **kwargs: calls.append((request_id, kwargs))
    )
    loop = SimpleNamespace(
        output_processor=output,
    )

    EventLoop._request_abort_or_mark(
        loop,
        "request-0",
        "cancelled",
        notify_client=True,
    )

    assert calls == [("request-0", {"notify_client": True})]


def test_run_event_loop_sigterm_sets_event_and_finally_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace: list[str] = []
    signal_calls: list[tuple[int, object]] = []
    installed_handler: dict[str, object] = {}
    parent_signals: list[int] = []

    previous_handler = object()

    def fake_signal(signum: int, handler: object) -> object:
        signal_calls.append((signum, handler))
        if len(signal_calls) == 1:
            installed_handler["value"] = handler
        return previous_handler

    class _ParentProcess:
        def send_signal(self, signum: int) -> None:
            parent_signals.append(signum)

    parent_process = _ParentProcess()

    class _Process:
        def parent(self):
            return parent_process

    class _FakeEventLoop:
        def __init__(
            self,
            _server_args,
            _port_args,
            _gpu_id,
            _attn_tp_rank,
            _dp_rank,
            _global_rank,
            shutdown_event,
        ) -> None:
            trace.append("construct")
            self.shutdown_event = shutdown_event
            self.max_total_num_tokens = 1024
            self.max_single_request_tokens = 768
            self.max_model_len = 4096
            self.max_req_input_len = 512
            self.multimodal_encoder_dtype = None
            self.model_config = SimpleNamespace(context_len=4096)
            self.has_dp = False
            self.use_overlap_schedule = False

        def event_loop(self) -> None:
            trace.append("loop_enter")
            assert not self.shutdown_event.is_set()
            handler = installed_handler["value"]
            assert callable(handler)
            handler(signal.SIGTERM, None)
            assert self.shutdown_event.is_set()
            trace.append("loop_return")

        def close(self) -> None:
            trace.append("close")

    class _PipeWriter:
        def __init__(self) -> None:
            self.messages: list[object] = []

        def send(self, message: object) -> None:
            self.messages.append(message)

    mapping = SimpleNamespace(
        rank=0,
        nprocs_per_node=1,
        attn=SimpleNamespace(tp_rank=0, dp_rank=0),
    )
    server_args = SimpleNamespace(
        mapping=mapping,
        base_gpu_id=0,
        disaggregation_mode="decode",
        max_num_seqs=8,
        chunked_prefill_size=128,
    )
    pipe_writer = _PipeWriter()

    monkeypatch.setattr(event_loop_module, "EventLoop", _FakeEventLoop)
    monkeypatch.setattr(event_loop_module.psutil, "Process", _Process)
    monkeypatch.setattr(
        event_loop_module.setproctitle, "setproctitle", lambda _title: None
    )
    monkeypatch.setattr(event_loop_module.faulthandler, "enable", lambda: None)
    monkeypatch.setattr(event_loop_module, "register_usr_signal", lambda: None)
    monkeypatch.setattr(
        event_loop_module, "configure_logger", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        event_loop_module.signal, "getsignal", lambda _sig: previous_handler
    )
    monkeypatch.setattr(event_loop_module.signal, "signal", fake_signal)

    event_loop_module.run_event_loop(server_args, object(), pipe_writer)

    assert trace == ["construct", "loop_enter", "loop_return", "close"]
    assert parent_signals == []
    assert len(pipe_writer.messages) == 1
    assert pipe_writer.messages[0]["status"] == "ready"
    assert signal_calls[0][0] == signal.SIGTERM
    assert callable(signal_calls[0][1])
    assert signal_calls[1] == (signal.SIGTERM, previous_handler)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
