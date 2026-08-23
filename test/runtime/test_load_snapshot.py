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

"""Tests for the scheduler-published load snapshot cache contract."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest
import zmq

from tokenspeed.runtime.engine import load_snapshot as load_snapshot_module
from tokenspeed.runtime.engine.io_struct import LoadSnapshot, MsgpackDecoder
from tokenspeed.runtime.engine.load_snapshot import LoadSnapshotStore

VALID_FOR_SECONDS = 1.0


class FakeClock:
    """A controllable monotonic clock for cache expiry behavior."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class RecordingContext:
    """Record publisher lifecycle calls and the threads that made them."""

    def __init__(self, socket) -> None:
        self.socket_value = socket
        self.calls = []

    def socket(self, socket_type):
        self.calls.append(("socket", threading.get_ident(), socket_type))
        return self.socket_value

    def term(self) -> None:
        self.calls.append(("term", threading.get_ident()))


class RecordingSocket:
    """A minimal PUSH socket double that decodes successfully sent frames."""

    def __init__(
        self,
        failures: int = 0,
        failure_gate: threading.Event | None = None,
        connect_error: Exception | None = None,
        close_error: Exception | None = None,
    ) -> None:
        self.failures = failures
        self.failure_gate = failure_gate
        self.connect_error = connect_error
        self.close_error = close_error
        self.calls = []
        self.decoded_snapshots = []
        self.send_attempted = threading.Event()
        self._decoder = MsgpackDecoder(LoadSnapshot)

    def setsockopt(self, option, value) -> None:
        self.calls.append(("setsockopt", threading.get_ident(), option, value))

    def connect(self, endpoint: str) -> None:
        self.calls.append(("connect", threading.get_ident(), endpoint))
        if self.connect_error is not None:
            raise self.connect_error

    def send(self, frame, flags=0) -> None:
        self.calls.append(("send", threading.get_ident(), flags))
        self.send_attempted.set()
        if self.failures:
            self.failures -= 1
            if self.failure_gate is not None:
                self.failure_gate.wait(timeout=2.0)
            raise zmq.Again()
        self.decoded_snapshots.append(self._decoder.decode(frame))

    def close(self, linger=None) -> None:
        self.calls.append(("close", threading.get_ident(), linger))
        if self.close_error is not None:
            raise self.close_error


def wait_until(predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition was not met before timeout")


def observed_values(**overrides) -> dict:
    values = dict(
        num_running_reqs=1,
        num_waiting_reqs=2,
        num_active_pages=3,
        num_used_pages=4,
        max_total_pages=10,
    )
    values.update(overrides)
    return values


def observed_tuple(**overrides) -> tuple[int, int, int, int, int]:
    values = observed_values(**overrides)
    return (
        values["num_running_reqs"],
        values["num_waiting_reqs"],
        values["num_active_pages"],
        values["num_used_pages"],
        values["max_total_pages"],
    )


def start_publisher(
    socket: RecordingSocket,
    *,
    heartbeat_interval: float = 60.0,
    factory_gate: threading.Event | None = None,
):
    context = RecordingContext(socket)

    def socket_factory(endpoint):
        if factory_gate is not None:
            factory_gate.wait(timeout=2.0)
        return context, socket

    publisher = load_snapshot_module.LoadSnapshotPublisher(
        "tcp://load-snapshots",
        dp_rank=2,
        heartbeat_interval=heartbeat_interval,
        socket_factory=socket_factory,
    )
    return publisher, context


def make_snapshot(**overrides) -> LoadSnapshot:
    fields = dict(
        epoch="boot-a",
        sequence=1,
        dp_rank=0,
        num_running_reqs=2,
        num_waiting_reqs=3,
        num_active_pages=4,
        num_used_pages=5,
        max_total_pages=10,
        valid_for_ms=int(VALID_FOR_SECONDS * 1_000),
    )
    fields.update(overrides)
    return LoadSnapshot(**fields)


def test_changed_values_replace_one_slot_without_waiting_for_interval():
    """A blocked publisher sends only the newest changed scheduler sample."""
    factory_gate = threading.Event()
    socket = RecordingSocket()
    publisher, _ = start_publisher(socket, factory_gate=factory_gate)
    try:
        publisher.observe(observed_tuple(num_running_reqs=1))
        publisher.observe(observed_tuple(num_running_reqs=2))
        factory_gate.set()

        wait_until(lambda: len(socket.decoded_snapshots) == 1)

        assert socket.decoded_snapshots[0].num_running_reqs == 2
        assert socket.decoded_snapshots[0].sequence == 1
    finally:
        factory_gate.set()
        publisher.close()


def test_publisher_observe_accepts_one_scalar_tuple():
    """The scheduler boundary accepts one cheap, immutable scalar value tuple."""
    socket = RecordingSocket()
    publisher, _ = start_publisher(socket)
    try:
        publisher.observe(observed_tuple())
        wait_until(lambda: socket.decoded_snapshots)

        assert socket.decoded_snapshots[0].num_running_reqs == 1
    finally:
        publisher.close()


def test_publisher_observe_does_not_encode_or_touch_zmq(monkeypatch):
    """Scheduler observation performs only guarded scalar mailbox work."""
    factory_gate = threading.Event()
    socket = RecordingSocket()
    encode_threads = []
    real_encode = load_snapshot_module.MsgpackEncoder.encode

    def record_encode(self, snapshot):
        encode_threads.append(threading.get_ident())
        return real_encode(self, snapshot)

    monkeypatch.setattr(load_snapshot_module.MsgpackEncoder, "encode", record_encode)
    publisher, context = start_publisher(socket, factory_gate=factory_gate)
    caller_thread = threading.get_ident()
    try:
        publisher.observe(observed_tuple())
        assert encode_threads == []
        assert socket.calls == []

        factory_gate.set()
        wait_until(lambda: socket.decoded_snapshots)

        assert publisher.socket_owner_thread != caller_thread
        assert set(encode_threads) == {publisher.socket_owner_thread}
        assert {call[1] for call in socket.calls if call[0] != "close"} == {
            publisher.socket_owner_thread
        }
        assert context.calls == []
    finally:
        factory_gate.set()
        publisher.close()


def test_unchanged_values_only_emit_a_heartbeat():
    """An unchanged observation is quiet until the one-second heartbeat."""
    socket = RecordingSocket()
    publisher, _ = start_publisher(socket, heartbeat_interval=0.01)
    try:
        publisher.observe(observed_tuple())
        wait_until(lambda: len(socket.decoded_snapshots) == 1)

        publisher.observe(observed_tuple())
        time.sleep(0.05)
        assert len(socket.decoded_snapshots) == 1

        wait_until(lambda: len(socket.decoded_snapshots) == 2, timeout=1.5)
        first, heartbeat = socket.decoded_snapshots
        assert (first.sequence, heartbeat.sequence) == (1, 2)
        assert heartbeat.valid_for_ms == 3_000
    finally:
        publisher.close()


def test_publisher_retry_is_replaced_by_newest_changed_values():
    """A nonblocking send failure cannot preserve stale data over a new sample."""
    failure_gate = threading.Event()
    socket = RecordingSocket(failures=1, failure_gate=failure_gate)
    publisher, _ = start_publisher(socket)
    try:
        publisher.observe(observed_tuple(num_running_reqs=1))
        assert socket.send_attempted.wait(timeout=1.0)
        publisher.observe(observed_tuple(num_running_reqs=7))
        failure_gate.set()

        wait_until(lambda: socket.decoded_snapshots)

        assert socket.decoded_snapshots[0].num_running_reqs == 7
        assert socket.decoded_snapshots[0].sequence == 2
    finally:
        failure_gate.set()
        publisher.close()


def test_publisher_default_socket_has_bounded_thread_owned_lifecycle(monkeypatch):
    """The publisher thread creates, configures, uses, and closes its own socket."""
    caller_thread = threading.get_ident()
    socket = RecordingSocket()
    context = RecordingContext(socket)
    context_created_on = []

    def make_context():
        context_created_on.append(threading.get_ident())
        return context

    monkeypatch.setattr(load_snapshot_module.zmq, "Context", make_context)
    publisher = load_snapshot_module.LoadSnapshotPublisher(
        "tcp://load-snapshots", dp_rank=0, heartbeat_interval=60.0
    )
    publisher.observe(observed_tuple())
    wait_until(lambda: socket.decoded_snapshots)
    publisher.close()

    owner = publisher.socket_owner_thread
    assert owner != caller_thread
    assert context_created_on == [owner]
    assert ("socket", owner, zmq.PUSH) in context.calls
    assert ("setsockopt", owner, zmq.SNDHWM, 1) in socket.calls
    assert ("setsockopt", owner, zmq.CONFLATE, 1) in socket.calls
    assert ("connect", owner, "tcp://load-snapshots") in socket.calls
    assert ("send", owner, zmq.NOBLOCK) in socket.calls
    assert ("close", owner, 0) in socket.calls
    assert ("term", owner) in context.calls


def test_publisher_setup_failure_cleans_partial_resources_on_owner_thread(
    monkeypatch,
):
    """A failed socket setup still closes and terminates every acquired resource."""
    caller_thread = threading.get_ident()
    socket = RecordingSocket(
        connect_error=RuntimeError("connect failed"),
        close_error=RuntimeError("close failed"),
    )
    context = RecordingContext(socket)
    context_created_on = []

    def make_context():
        context_created_on.append(threading.get_ident())
        return context

    monkeypatch.setattr(load_snapshot_module.zmq, "Context", make_context)
    publisher = load_snapshot_module.LoadSnapshotPublisher(
        "tcp://load-snapshots", dp_rank=0, heartbeat_interval=60.0
    )
    wait_until(lambda: any(call[0] == "connect" for call in socket.calls))

    close_started = time.monotonic()
    publisher.close()
    close_elapsed = time.monotonic() - close_started

    owner = publisher.socket_owner_thread
    assert owner != caller_thread
    assert close_elapsed < 0.5
    assert context_created_on == [owner]
    assert ("socket", owner, zmq.PUSH) in context.calls
    assert ("connect", owner, "tcp://load-snapshots") in socket.calls
    assert ("close", owner, 0) in socket.calls
    assert ("term", owner) in context.calls


def test_event_loop_observation_projects_the_sampled_scheduler_values():
    """The shared helper maps the existing sample without another query."""
    pytest.importorskip("tokenspeed_scheduler")
    from tokenspeed.runtime.engine.event_loop import EventLoop

    observed = []
    loop = EventLoop.__new__(EventLoop)
    loop.load_snapshot_publisher = SimpleNamespace(
        observe=lambda values: observed.append(values)
    )
    loop.output_processor = SimpleNamespace(rid_to_state={"a": object(), "b": object()})
    loop._scheduler_cache_geometry = SimpleNamespace(num_usable_pages=20)

    loop._observe_load_snapshot(
        {"num_queue_reqs": 3, "num_active_pages": 4, "num_cached_pages": 5}
    )

    assert observed == [
        observed_tuple(
            num_running_reqs=2,
            num_waiting_reqs=3,
            num_active_pages=4,
            num_used_pages=5,
            max_total_pages=20,
        )
    ]


@pytest.mark.parametrize(
    ("attn_tp_rank", "zmq_msgpack", "should_publish"),
    [(0, False, True), (1, False, False), (0, True, False)],
)
def test_event_loop_publisher_is_gated_to_standard_serving_tp0(
    monkeypatch, attn_tp_rank, zmq_msgpack, should_publish
):
    """Non-TP0 and direct-ZMQ engines never open the internal PUSH socket."""
    pytest.importorskip("tokenspeed_scheduler")
    from tokenspeed.runtime.engine import event_loop as event_loop_module

    created = []

    class FakePublisher:
        def __init__(self, endpoint, dp_rank, heartbeat_interval):
            created.append((endpoint, dp_rank, heartbeat_interval))

        def observe(self, values):
            raise AssertionError(values)

        def close(self):
            raise AssertionError("unused fake publisher was closed")

    monkeypatch.setattr(event_loop_module, "LoadSnapshotPublisher", FakePublisher)
    loop = event_loop_module.EventLoop.__new__(event_loop_module.EventLoop)
    loop.attn_tp_rank = attn_tp_rank
    loop.dp_rank = 4
    loop.server_args = SimpleNamespace(
        zmq_msgpack=zmq_msgpack, load_watch_interval=0.25
    )
    loop.port_args = SimpleNamespace(metrics_ipc_name="tcp://metrics")

    loop._init_load_snapshot_publisher()

    assert created == ([("tcp://metrics", 4, 0.25)] if should_publish else [])
    if not should_publish:
        loop.load_snapshot_publisher.observe(observed_tuple())
        loop.load_snapshot_publisher.close()


def test_load_snapshot_is_immutable_and_positional():
    """Snapshots cannot be mutated after their wire values are assigned."""
    snapshot = LoadSnapshot("boot-a", 1, 0, 2, 3, 4, 5, 10, 1_000)

    with pytest.raises(AttributeError):
        snapshot.sequence = 2


def test_store_rejects_duplicate_and_retired_epoch_without_refreshing_age():
    """Duplicate and retired messages cannot revive stale rank state."""
    clock = FakeClock()
    store = LoadSnapshotStore(dp_size=1, clock=clock)

    assert store.accept(make_snapshot(epoch="a", sequence=1))
    clock.advance(0.1)
    assert not store.accept(make_snapshot(epoch="a", sequence=1))
    assert store.accept(make_snapshot(epoch="b", sequence=1))
    assert not store.accept(make_snapshot(epoch="a", sequence=2))


def test_store_rejects_non_increasing_sequence_in_current_epoch():
    """Only strictly newer versions can replace a rank's current epoch."""
    store = LoadSnapshotStore(dp_size=1, clock=FakeClock())

    assert store.accept(make_snapshot(sequence=2))
    assert not store.accept(make_snapshot(sequence=2))
    assert not store.accept(make_snapshot(sequence=1))
    assert store.accept(make_snapshot(sequence=3))


@pytest.mark.parametrize(
    "overrides",
    [
        {"dp_rank": -1},
        {"dp_rank": 2},
        {"sequence": -1},
        {"num_running_reqs": -1},
        {"num_waiting_reqs": -1},
        {"num_active_pages": -1},
        {"num_used_pages": -1},
        {"max_total_pages": 0},
        {"num_active_pages": 11},
        {"num_used_pages": 11},
        {"valid_for_ms": -1},
    ],
)
def test_store_rejects_invalid_snapshot_values(overrides):
    """Malformed values never enter the replica cache."""
    store = LoadSnapshotStore(dp_size=2, clock=FakeClock())

    assert not store.accept(make_snapshot(**overrides))
    assert store.fresh_snapshots() == []


def test_store_projects_only_a_complete_fresh_rank_set():
    """Public loads stay unavailable until every rank has a live snapshot."""
    clock = FakeClock()
    store = LoadSnapshotStore(dp_size=2, clock=clock)

    assert store.project_loads() == []
    assert store.accept(make_snapshot(dp_rank=0))
    assert store.project_loads() == []
    assert store.accept(make_snapshot(dp_rank=1))
    loads = store.project_loads()
    assert [load.dp_rank for load in loads] == [0, 1]
    assert [
        (load.num_reqs, load.num_waiting_reqs, load.num_pages) for load in loads
    ] == [
        (5, 3, 5),
        (5, 3, 5),
    ]

    clock.advance(VALID_FOR_SECONDS + 0.001)

    assert not store.is_complete_fresh()
    assert store.fresh_snapshots() == []
    assert store.project_loads() == []


def test_store_projection_checks_completeness_at_one_clock_read():
    """A rank expiring during projection cannot leak a partial public result."""
    reads = iter((0.0, 0.5, 0.999, 1.001))
    store = LoadSnapshotStore(dp_size=2, clock=lambda: next(reads))

    assert store.accept(make_snapshot(dp_rank=0))
    assert store.accept(make_snapshot(dp_rank=1))

    assert [load.dp_rank for load in store.project_loads()] == [0, 1]


def test_store_uses_local_receipt_time_and_does_not_refresh_rejection_age():
    """Only accepted messages extend a snapshot's locally measured lifetime."""
    clock = FakeClock()
    store = LoadSnapshotStore(dp_size=1, clock=clock)

    assert store.accept(make_snapshot())
    clock.advance(0.9)
    assert not store.accept(make_snapshot())
    clock.advance(0.101)

    assert store.fresh_snapshots() == []


def test_store_rejects_non_positive_dp_size():
    """A store must have at least one rank to form a complete replica."""
    with pytest.raises(ValueError, match="dp_size"):
        LoadSnapshotStore(dp_size=0)
