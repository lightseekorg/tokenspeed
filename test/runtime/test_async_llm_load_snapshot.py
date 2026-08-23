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

"""Tests for AsyncLLM's scheduler-published load replica."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
import zmq

from tokenspeed.runtime.engine import core_client as core_client_module
from tokenspeed.runtime.engine.io_struct import GetLoadReqOutput, LoadSnapshot
from tokenspeed.runtime.engine.load_snapshot import LoadSnapshotStore
from tokenspeed.runtime.engine.scheduler_control_client import SchedulerControlClient


class _RecordingSocket:
    def __init__(self) -> None:
        self.calls = []

    def setsockopt(self, option, value) -> None:
        self.calls.append(("setsockopt", option, value))

    def bind(self, endpoint) -> None:
        self.calls.append(("bind", endpoint))


class _RecordingContext:
    def __init__(self) -> None:
        self.metrics_socket = _RecordingSocket()

    def socket(self, socket_type):
        assert socket_type == zmq.PULL
        return self.metrics_socket


class _QueueReceiver:
    def __init__(self, values) -> None:
        self.values = list(values)

    async def recv_pyobj(self):
        if not self.values:
            raise asyncio.CancelledError
        return self.values.pop(0)


class _RecordingSender:
    def __init__(self) -> None:
        self.sent = []

    def send_pyobj(self, value) -> None:
        self.sent.append(value)


def _snapshot(**overrides) -> LoadSnapshot:
    fields = dict(
        epoch="epoch-a",
        sequence=1,
        dp_rank=0,
        num_running_reqs=2,
        num_waiting_reqs=3,
        num_active_pages=4,
        num_used_pages=5,
        max_total_pages=10,
        valid_for_ms=1_000,
    )
    fields.update(overrides)
    return LoadSnapshot(**fields)


def test_engine_core_metrics_pull_keeps_only_the_newest_snapshot(monkeypatch):
    context = _RecordingContext()
    monkeypatch.setattr(core_client_module.zmq.asyncio, "Context", lambda _: context)
    monkeypatch.setattr(
        core_client_module,
        "get_zmq_socket",
        lambda *_: _RecordingSocket(),
    )

    core_client_module.EngineCoreClient(
        SimpleNamespace(
            tokenizer_ipc_name="tcp://tokenizer",
            scheduler_input_ipc_name="tcp://scheduler",
            metrics_ipc_name="tcp://metrics",
        )
    )

    assert context.metrics_socket.calls == [
        ("setsockopt", zmq.RCVHWM, 1),
        ("bind", "tcp://metrics"),
    ]


@pytest.mark.asyncio
async def test_load_snapshot_loop_caches_and_forwards_only_accepted_snapshots():
    accepted = _snapshot()
    duplicate = _snapshot()
    out_of_order = _snapshot(sequence=0)
    invalid = _snapshot(sequence=2, num_used_pages=11)
    sender = _RecordingSender()
    llm = SimpleNamespace()
    llm.load_snapshot_store = LoadSnapshotStore(dp_size=1)
    llm.engine_core_client = SimpleNamespace(
        recv_load_snapshot=_QueueReceiver([accepted, duplicate, out_of_order, invalid]),
        send_to_scheduler=sender,
    )
    llm.server_args = SimpleNamespace(
        mapping=SimpleNamespace(attn=SimpleNamespace(has_dp=True)),
        load_balance_method="shortest_queue",
    )

    with pytest.raises(asyncio.CancelledError):
        await SchedulerControlClient.load_snapshot_loop(llm)

    assert llm.load_snapshot_store.project_loads() == [
        GetLoadReqOutput(dp_rank=0, num_reqs=5, num_waiting_reqs=3, num_pages=5)
    ]
    assert sender.sent == [accepted]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("has_dp", "load_balance_method"),
    [(False, "shortest_queue"), (True, "round_robin")],
)
async def test_load_snapshot_loop_does_not_forward_without_internal_dp_balancing(
    has_dp, load_balance_method
):
    accepted = _snapshot()
    sender = _RecordingSender()
    llm = SimpleNamespace(
        load_snapshot_store=LoadSnapshotStore(dp_size=1),
        engine_core_client=SimpleNamespace(
            recv_load_snapshot=_QueueReceiver([accepted]),
            send_to_scheduler=sender,
        ),
        server_args=SimpleNamespace(
            mapping=SimpleNamespace(attn=SimpleNamespace(has_dp=has_dp)),
            load_balance_method=load_balance_method,
        ),
    )

    with pytest.raises(asyncio.CancelledError):
        await SchedulerControlClient.load_snapshot_loop(llm)

    assert llm.load_snapshot_store.project_loads()[0].num_reqs == 5
    assert sender.sent == []


@pytest.mark.asyncio
async def test_get_load_starts_receivers_and_projects_only_fresh_complete_cache():
    calls = []
    now = [0.0]
    llm = SimpleNamespace(
        load_snapshot_store=LoadSnapshotStore(dp_size=1, clock=lambda: now[0]),
        auto_create_handle_loop=lambda: calls.append("started"),
        engine_core_client=SimpleNamespace(
            send_to_scheduler=SimpleNamespace(
                send_pyobj=lambda _: pytest.fail("get_load must not poll the scheduler")
            )
        ),
    )

    assert await SchedulerControlClient.get_load(llm) == []
    llm.load_snapshot_store.accept(_snapshot(valid_for_ms=1))

    assert await SchedulerControlClient.get_load(llm) == [
        GetLoadReqOutput(dp_rank=0, num_reqs=5, num_waiting_reqs=3, num_pages=5)
    ]
    now[0] = 0.002
    assert await SchedulerControlClient.get_load(llm) == []
    assert calls == ["started", "started", "started"]
