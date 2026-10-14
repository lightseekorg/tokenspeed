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

"""PD KV transfer metrics: the collector itself and its transfer call sites."""

from __future__ import annotations

import os
import sys
import threading
import time
from types import SimpleNamespace

import pytest
from prometheus_client import CollectorRegistry

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

# CPU-only; scheduled in runtime-1gpu because it imports the full runtime.
register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed.runtime.metrics.collector import KVTransferMetrics  # noqa: E402

LABELS = {"model_name": "m", "app_key": "k"}


def _make(registry: CollectorRegistry, *, enabled: bool = True) -> KVTransferMetrics:
    return KVTransferMetrics(LABELS, enabled=enabled, registry=registry)


def test_disabled_records_nothing() -> None:
    registry = CollectorRegistry()
    metrics = _make(registry, enabled=False)

    metrics.observe_kv_transfer_latency(0.5)
    metrics.record_kv_transfer_timeout("bootstrapping")
    metrics.record_kv_transfer_failure()

    assert (
        registry.get_sample_value("tokenspeed:kv_transfer_latency_seconds_count")
        is None
    )
    assert registry.get_sample_value("tokenspeed:kv_transfer_timeouts_total") is None
    assert registry.get_sample_value("tokenspeed:kv_transfer_failures_total") is None


def test_latency_histogram_records_count_sum_and_buckets() -> None:
    registry = CollectorRegistry()
    metrics = _make(registry)

    metrics.observe_kv_transfer_latency(0.003)
    metrics.observe_kv_transfer_latency(0.5)

    assert (
        registry.get_sample_value(
            "tokenspeed:kv_transfer_latency_seconds_count", LABELS
        )
        == 2.0
    )
    assert registry.get_sample_value(
        "tokenspeed:kv_transfer_latency_seconds_sum", LABELS
    ) == pytest.approx(0.503)
    # The 3ms sample lands in the 5ms bucket, the 0.5s one does not.
    assert (
        registry.get_sample_value(
            "tokenspeed:kv_transfer_latency_seconds_bucket", {**LABELS, "le": "0.005"}
        )
        == 1.0
    )
    assert (
        registry.get_sample_value(
            "tokenspeed:kv_transfer_latency_seconds_bucket", {**LABELS, "le": "+Inf"}
        )
        == 2.0
    )


def test_timeouts_counted_by_stage() -> None:
    registry = CollectorRegistry()
    metrics = _make(registry)

    metrics.record_kv_transfer_timeout("bootstrapping")
    metrics.record_kv_transfer_timeout("bootstrapping")
    metrics.record_kv_transfer_timeout("waiting_for_input")

    assert (
        registry.get_sample_value(
            "tokenspeed:kv_transfer_timeouts_total",
            {**LABELS, "stage": "bootstrapping"},
        )
        == 2.0
    )
    assert (
        registry.get_sample_value(
            "tokenspeed:kv_transfer_timeouts_total",
            {**LABELS, "stage": "waiting_for_input"},
        )
        == 1.0
    )


def test_failures_counted() -> None:
    registry = CollectorRegistry()
    metrics = _make(registry)

    metrics.record_kv_transfer_failure()
    metrics.record_kv_transfer_failure()

    assert (
        registry.get_sample_value("tokenspeed:kv_transfer_failures_total", LABELS)
        == 2.0
    )


def test_prefill_abort_room_records_failure() -> None:
    from tokenspeed.runtime.pd.base.status import TransferPoll
    from tokenspeed.runtime.pd.mooncake.prefill import MooncakeKVManagerPrefill

    registry = CollectorRegistry()
    manager = object.__new__(MooncakeKVManagerPrefill)
    manager.kv_transfer_metrics = _make(registry)
    manager.request_status = {}
    manager.failure_records = {}
    manager.failure_lock = threading.Lock()
    manager.transfer_infos = {}

    manager.abort_room(9, "mooncake transfer ret=-1")

    assert (
        registry.get_sample_value("tokenspeed:kv_transfer_failures_total", LABELS)
        == 1.0
    )
    assert manager.request_status[9] == TransferPoll.Failed


def test_prefill_abort_room_counts_once_per_room() -> None:
    from tokenspeed.runtime.pd.base.status import TransferPoll
    from tokenspeed.runtime.pd.mooncake.prefill import MooncakeKVManagerPrefill

    registry = CollectorRegistry()
    manager = object.__new__(MooncakeKVManagerPrefill)
    manager.kv_transfer_metrics = _make(registry)
    manager.request_status = {}
    manager.failure_records = {}
    manager.failure_lock = threading.Lock()
    manager.transfer_infos = {}

    # Later chunks of the same room funnel back into abort_room after the
    # first abort popped transfer_infos; only the first may count.
    manager.abort_room(9, "first failure")
    manager.abort_room(9, "later chunk of the same room")

    # A room already recorded (e.g. by a sender-side timeout) is not also a
    # failure — timeout and failure accounting stay mutually exclusive.
    manager.failure_records[10] = "sender bootstrap timed out"
    manager.abort_room(10, "abort after timeout")

    assert (
        registry.get_sample_value("tokenspeed:kv_transfer_failures_total", LABELS)
        == 1.0
    )
    assert manager.request_status[9] == TransferPoll.Failed
    assert manager.request_status[10] == TransferPoll.Failed


def test_decode_node_failure_records_one_failure_per_affected_room() -> None:
    from tokenspeed.runtime.pd.base.status import TransferPoll
    from tokenspeed.runtime.pd.mooncake.decode import MooncakeKVManagerDecode

    registry = CollectorRegistry()
    manager = object.__new__(MooncakeKVManagerDecode)
    manager.kv_transfer_metrics = _make(registry)
    manager.connection_lock = threading.Lock()
    manager.connection_pool = {}
    manager.prefill_parallel_info = {}
    manager.addr_to_rooms_tracker = {"prefill-0": {9, 10, 11, 12, 13}}
    manager.request_status = {
        9: TransferPoll.Bootstrapped,
        10: TransferPoll.Success,
        11: TransferPoll.Transferring,
        12: TransferPoll.Failed,
        13: TransferPoll.WaitingForInput,
    }
    manager.failure_records = {13: "receiver waiting timed out"}
    manager.failure_lock = threading.Lock()

    manager._handle_node_failure("prefill-0")

    # The successful room is not a failure, nor are the already-failed room
    # (counted on the prefill side) and the already-timed-out room (counted
    # as a timeout); only rooms 9 and 11 count.
    assert (
        registry.get_sample_value("tokenspeed:kv_transfer_failures_total", LABELS)
        == 2.0
    )
    assert manager.request_status[9] == TransferPoll.Failed
    assert manager.request_status[12] == TransferPoll.Failed
    assert manager.request_status[10] == TransferPoll.Success
    assert manager.request_status[11] == TransferPoll.Failed


def test_sender_bootstrap_timeout_records_timeout() -> None:
    from tokenspeed.runtime.pd.base.status import TransferPoll
    from tokenspeed.runtime.pd.mooncake.sender import MooncakeKVSender

    registry = CollectorRegistry()
    sender = object.__new__(MooncakeKVSender)
    sender.kv_mgr = SimpleNamespace(
        check_status=lambda _room: TransferPoll.Bootstrapping,
        bootstrap_time_out=0.0,
        record_failure=lambda _room, _reason: None,
        kv_transfer_metrics=_make(registry),
    )
    sender.bootstrap_room = 9
    sender.init_time = time.time() - 1.0
    sender.conclude_state = None

    assert sender.poll() == TransferPoll.Failed
    assert (
        registry.get_sample_value(
            "tokenspeed:kv_transfer_timeouts_total",
            {**LABELS, "stage": "bootstrapping"},
        )
        == 1.0
    )


def test_receiver_waiting_timeout_records_timeout() -> None:
    from tokenspeed.runtime.pd.base.status import TransferPoll
    from tokenspeed.runtime.pd.mooncake.receiver import MooncakeKVReceiver

    registry = CollectorRegistry()
    receiver = object.__new__(MooncakeKVReceiver)
    receiver.kv_mgr = SimpleNamespace(
        check_status=lambda _room: TransferPoll.WaitingForInput,
        waiting_timeout=0.0,
        record_failure=lambda _room, _reason: None,
        kv_transfer_metrics=_make(registry),
    )
    receiver.bootstrap_room = 9
    receiver.init_time = time.time() - 1.0
    receiver.conclude_state = None

    assert receiver.poll() == TransferPoll.Failed
    assert (
        registry.get_sample_value(
            "tokenspeed:kv_transfer_timeouts_total",
            {**LABELS, "stage": "waiting_for_input"},
        )
        == 1.0
    )
