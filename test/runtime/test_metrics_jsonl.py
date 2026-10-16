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

"""Raw metric transport, file ownership, and bounded failure behavior."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import tempfile
import time
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")


@pytest.fixture
def endpoint():
    # Unix socket names have a short platform-dependent length limit.
    with tempfile.TemporaryDirectory(prefix="metric-", dir="/tmp") as directory:
        yield f"ipc://{directory}/events.sock"


def _produce(endpoint: str, producer: str) -> None:
    from tokenspeed.runtime.metrics.jsonl import JsonlMetricSender, MetricEvent

    sender = JsonlMetricSender(endpoint, queue_size=128)
    try:
        for index in range(25):
            labels = {"producer": producer}
            sender.emit(
                MetricEvent(index, "tokenspeed:test", "observe", index / 10, labels, 2)
            )
            labels["producer"] = "mutated"
    finally:
        sender.close(timeout=5)
    assert sender.dropped_events == 0
    assert sender.error_count == 0


def _records(directory: Path) -> list[dict]:
    records = []
    for path in sorted(directory.glob("metrics-*.jsonl")):
        content = path.read_bytes()
        assert not content or content.endswith(b"\n")
        records.extend(json.loads(line) for line in content.splitlines())
    return records


def _wait_until(predicate) -> None:
    deadline = time.monotonic() + 5
    while not predicate():
        assert time.monotonic() < deadline, "metric operation did not finish"
        time.sleep(0.01)


def test_two_spawned_producers_share_one_ordered_stream(tmp_path, endpoint):
    from tokenspeed.runtime.metrics.jsonl import JsonlMetricWriter

    writer = JsonlMetricWriter(
        str(tmp_path),
        endpoint,
        max_bytes=1_000_000,
        backup_count=2,
        flush_interval=0.01,
    )
    processes = [
        mp.get_context("spawn").Process(target=_produce, args=(endpoint, producer))
        for producer in ("frontend", "rank0")
    ]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=10)
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
        writer.close(timeout=5)

    records = _records(tmp_path)
    assert len(records) == 50
    assert {record["stream_id"] for record in records} == {writer.stream_id}
    assert [record["sequence"] for record in records] == list(range(50))
    assert {record["schema_version"] for record in records} == {1}
    for producer in ("frontend", "rank0"):
        observations = [r for r in records if r["labels"]["producer"] == producer]
        assert [r["timestamp_ns"] for r in observations] == list(range(25))
        assert [r["value"] for r in observations] == [index / 10 for index in range(25)]
        assert {r["metric"] for r in observations} == {"tokenspeed:test"}
        assert {r["operation"] for r in observations} == {"observe"}
        assert {r["weight"] for r in observations} == {2}
    assert writer.written_events == 50
    assert writer.error_count == 0


@pytest.mark.parametrize("backup_count, expected_sequences", [(0, [5]), (2, [3, 4, 5])])
def test_rotation_keeps_complete_records_and_ignores_unrelated_files(
    tmp_path, endpoint, backup_count, expected_sequences
):
    from tokenspeed.runtime.metrics.jsonl import (
        JsonlMetricSender,
        JsonlMetricWriter,
        MetricEvent,
    )

    other_stream = tmp_path / "metrics-other-stream-000001.jsonl"
    other_stream.write_text("another instance owns this\n")
    writer = JsonlMetricWriter(
        str(tmp_path),
        endpoint,
        max_bytes=100,
        backup_count=backup_count,
        flush_interval=0.01,
    )
    sender = JsonlMetricSender(endpoint, queue_size=32)
    try:
        for index in range(6):
            sender.emit(
                MetricEvent(index, "large-record", "inc", 1.0, {"label": "x" * 200})
            )
        sender.close(timeout=5)
    finally:
        sender.close(timeout=5)
        writer.close(timeout=5)

    assert other_stream.read_text() == "another instance owns this\n"
    paths = sorted(tmp_path.glob(f"metrics-{writer.stream_id}-*.jsonl"))
    assert len(paths) == backup_count + 1
    records = []
    for path in paths:
        content = path.read_bytes()
        assert content.endswith(b"\n")
        assert len(content.splitlines()) == 1
        records.append(json.loads(content))
    assert [record["sequence"] for record in records] == expected_sequences
    assert writer.written_events == 6


def test_unavailable_writer_drops_without_blocking_producer(endpoint, caplog):
    from tokenspeed.runtime.metrics.jsonl import JsonlMetricSender, MetricEvent

    sender = JsonlMetricSender(endpoint, queue_size=2)
    event = MetricEvent(1, "tokenspeed:test", "inc", 1.0, {})
    try:
        started = time.monotonic()
        for _ in range(5000):
            sender.emit(event)
        assert time.monotonic() - started < 1.0
        assert sender.dropped_events > 0
    finally:
        started = time.monotonic()
        sender.close(timeout=0.2)
        assert time.monotonic() - started < 1.0
    _wait_until(lambda: sender.dropped_events == 5000)
    assert "drop" in caplog.text.lower()


def test_sender_freezes_labels_before_writer_starts(tmp_path, endpoint):
    from tokenspeed.runtime.metrics.jsonl import (
        JsonlMetricSender,
        JsonlMetricWriter,
        MetricEvent,
    )

    sender = JsonlMetricSender(endpoint, queue_size=4)
    labels = {"model_name": "original"}
    sender.emit(MetricEvent(1, "tokenspeed:test", "set", 3.0, labels))
    labels["model_name"] = "changed"
    writer = JsonlMetricWriter(
        str(tmp_path),
        endpoint,
        max_bytes=1_000_000,
        backup_count=2,
        flush_interval=0.01,
    )
    try:
        sender.close(timeout=5)
    finally:
        writer.close(timeout=5)
    assert _records(tmp_path)[0]["labels"] == {"model_name": "original"}


def test_writer_disk_failure_does_not_escape_or_block_sender(
    tmp_path, endpoint, caplog
):
    from tokenspeed.runtime.metrics.jsonl import (
        JsonlMetricSender,
        JsonlMetricWriter,
        MetricEvent,
    )

    directory = tmp_path / "not-a-directory"
    directory.write_text("existing file")
    writer = JsonlMetricWriter(
        str(directory), endpoint, max_bytes=1024, backup_count=2, flush_interval=0.01
    )
    sender = JsonlMetricSender(endpoint, queue_size=2)
    try:
        _wait_until(lambda: writer.error_count > 0)
        started = time.monotonic()
        for _ in range(1000):
            sender.emit(MetricEvent(1, "tokenspeed:test", "inc", 1.0, {}))
        assert time.monotonic() - started < 1.0
    finally:
        sender.close(timeout=0.2)
        writer.close(timeout=0.2)
    assert writer.written_events == 0
    assert "writer" in caplog.text.lower()
    assert directory.read_text() == "existing file"


def test_disk_write_failure_stops_only_writer(tmp_path, endpoint, monkeypatch, caplog):
    from tokenspeed.runtime.metrics.jsonl import (
        JsonlMetricSender,
        JsonlMetricWriter,
        MetricEvent,
    )

    open_file = Path.open

    class FullDiskFile:
        def __init__(self, file):
            self.file = file

        def write(self, data):
            raise OSError("disk full")

        def flush(self):
            self.file.flush()

        def close(self):
            self.file.close()

    def open_on_full_disk(path, *args, **kwargs):
        file = open_file(path, *args, **kwargs)
        return FullDiskFile(file) if path.suffix == ".jsonl" else file

    monkeypatch.setattr(Path, "open", open_on_full_disk)
    writer = JsonlMetricWriter(
        str(tmp_path), endpoint, max_bytes=1024, backup_count=2, flush_interval=0.01
    )
    sender = JsonlMetricSender(endpoint, queue_size=2)
    try:
        assert writer.error_count == 0
        sender.emit(MetricEvent(1, "tokenspeed:test", "inc", 1.0, {}))
        _wait_until(lambda: writer.error_count == 1)
        started = time.monotonic()
        for _ in range(1000):
            sender.emit(MetricEvent(1, "tokenspeed:test", "inc", 1.0, {}))
        assert time.monotonic() - started < 1.0
        assert sender.dropped_events > 0
    finally:
        sender.close(timeout=0.2)
        writer.close(timeout=0.2)
    assert writer.written_events == 0
    assert "disk full" in caplog.text


def test_flush_makes_data_visible_without_close(tmp_path, endpoint, monkeypatch):
    from tokenspeed.runtime.metrics.jsonl import (
        JsonlMetricSender,
        JsonlMetricWriter,
        MetricEvent,
    )

    writer = JsonlMetricWriter(
        str(tmp_path), endpoint, max_bytes=1024, backup_count=2, flush_interval=0.01
    )
    sender = JsonlMetricSender(endpoint, queue_size=4)
    try:
        sender.emit(MetricEvent(1, "tokenspeed:test", "inc", 1.0, {}))
        _wait_until(lambda: len(_records(tmp_path)) == 1)
        assert _records(tmp_path)[0]["value"] == 1.0
    finally:
        sender.close(timeout=1)
        writer.close(timeout=1)


def test_closed_sender_counts_late_events_as_dropped(endpoint):
    from tokenspeed.runtime.metrics.jsonl import JsonlMetricSender, MetricEvent

    sender = JsonlMetricSender(endpoint, queue_size=2)
    sender.close(timeout=1)
    sender.emit(MetricEvent(1, "tokenspeed:test", "inc", 1.0, {}))
    assert sender.dropped_events == 1


def test_restart_retains_only_configured_backups_across_streams(tmp_path, endpoint):
    from tokenspeed.runtime.metrics.jsonl import (
        JsonlMetricSender,
        JsonlMetricWriter,
        MetricEvent,
    )

    writer = JsonlMetricWriter(
        str(tmp_path), endpoint, max_bytes=100, backup_count=2, flush_interval=0.01
    )
    sender = JsonlMetricSender(endpoint, queue_size=32)
    for index in range(6):
        sender.emit(MetricEvent(index, "tokenspeed:test", "inc", 1.0, {}))
    sender.close(timeout=5)
    writer.close(timeout=5)
    old_paths = sorted(tmp_path.glob("metrics-*.jsonl"))
    assert len(old_paths) == 3

    restarted = JsonlMetricWriter(
        str(tmp_path), endpoint, max_bytes=100, backup_count=2, flush_interval=0.01
    )
    try:
        assert restarted.stream_id != writer.stream_id
        assert set(tmp_path.glob("metrics-*.jsonl")) == {
            *old_paths[-2:],
            tmp_path / f"metrics-{restarted.stream_id}-000000.jsonl",
        }
        sender = JsonlMetricSender(endpoint, queue_size=32)
        for index in range(2):
            sender.emit(MetricEvent(index, "tokenspeed:test", "inc", 1.0, {}))
        sender.close(timeout=5)
    finally:
        restarted.close(timeout=5)

    records = _records(tmp_path)
    assert len(list(tmp_path.glob("metrics-*.jsonl"))) == 3
    assert {(row["stream_id"], row["sequence"]) for row in records} == {
        (writer.stream_id, 5),
        (restarted.stream_id, 0),
        (restarted.stream_id, 1),
    }


def test_directory_lock_rejects_second_writer_and_releases_on_close(tmp_path, endpoint):
    from tokenspeed.runtime.metrics.jsonl import (
        JsonlMetricSender,
        JsonlMetricWriter,
        MetricEvent,
    )

    writer = JsonlMetricWriter(
        str(tmp_path), endpoint, max_bytes=10000, backup_count=2, flush_interval=0.01
    )
    sender = JsonlMetricSender(endpoint, queue_size=32)
    rejected = None
    try:
        sender.emit(MetricEvent(0, "tokenspeed:test", "inc", 1.0, {}))
        _wait_until(lambda: writer.written_events == 1)
        paths = set(tmp_path.glob("metrics-*.jsonl"))
        rejected = JsonlMetricWriter(
            str(tmp_path),
            endpoint.replace("events.sock", "second.sock"),
            max_bytes=100,
            backup_count=0,
            flush_interval=0.01,
        )
        assert rejected.error_count == 1
        assert set(tmp_path.glob("metrics-*.jsonl")) == paths
        sender.emit(MetricEvent(1, "tokenspeed:test", "inc", 1.0, {}))
        sender.close(timeout=5)
    finally:
        sender.close(timeout=5)
        if rejected is not None:
            rejected.close(timeout=5)
        writer.close(timeout=5)

    assert [row["timestamp_ns"] for row in _records(tmp_path)] == [0, 1]
    lock_inode = (tmp_path / ".metrics.lock").stat().st_ino
    restarted = JsonlMetricWriter(
        str(tmp_path), endpoint, max_bytes=10000, backup_count=2, flush_interval=0.01
    )
    restarted.close(timeout=5)
    assert restarted.error_count == 0
    assert (tmp_path / ".metrics.lock").stat().st_ino == lock_inode


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf")])
def test_invalid_observation_drops_only_that_event(
    tmp_path, endpoint, invalid_value, caplog
):
    from tokenspeed.runtime.metrics.jsonl import (
        JsonlMetricSender,
        JsonlMetricWriter,
        MetricEvent,
    )

    writer = JsonlMetricWriter(
        str(tmp_path), endpoint, max_bytes=10000, backup_count=2, flush_interval=0.01
    )
    sender = JsonlMetricSender(endpoint, queue_size=32)
    try:
        for index, value in enumerate((1.0, invalid_value, 2.0)):
            sender.emit(MetricEvent(index, "tokenspeed:test", "set", value, {}))
        sender.close(timeout=5)
    finally:
        sender.close(timeout=5)
        writer.close(timeout=5)

    assert [row["value"] for row in _records(tmp_path)] == [1.0, 2.0]
    assert sender.error_count == 1
    assert sender.dropped_events == 1
    assert writer.error_count == 0
    assert "encod" in caplog.text.lower()


@pytest.mark.parametrize(
    "update",
    [
        {"unknown": 1},
        {"directory": ""},
        {"directory": 3},
        {"max_bytes": 0},
        {"max_bytes": True},
        {"backup_count": -1},
        {"queue_size": 1.5},
        {"flush_interval": float("inf")},
        {"flush_interval": False},
        {"flush_interval": 10**1000},
    ],
)
def test_jsonl_plugin_rejects_invalid_configuration(tmp_path, update):
    from tokenspeed.runtime.metrics import jsonl

    config = {
        "directory": str(tmp_path),
        "max_bytes": 1024,
        "backup_count": 2,
        "queue_size": 100,
        "flush_interval": 0.1,
        **update,
    }
    with pytest.raises(ValueError):
        jsonl.validate_jsonl_config(config)


def test_jsonl_plugin_requires_explicit_configuration():
    from tokenspeed.runtime.metrics import jsonl

    with pytest.raises(ValueError):
        jsonl.validate_jsonl_config({})


def test_jsonl_plugin_owns_and_cleans_its_ipc_directory(tmp_path):
    from tokenspeed.runtime.metrics import jsonl

    config = jsonl.validate_jsonl_config(
        {
            "directory": str(tmp_path),
            "max_bytes": 1024,
            "backup_count": 2,
            "queue_size": 100,
            "flush_interval": 0.01,
        }
    )
    session = jsonl.start_jsonl_node(config)
    producer_config = session.producer_config
    ipc_directory = Path(producer_config["endpoint"][len("ipc://") :]).parent
    sender = jsonl.create_jsonl_sink(producer_config)
    try:
        sender.emit(jsonl.MetricEvent(1, "plugin", "inc", 3.0, {}))
    finally:
        sender.close(timeout=2)
        session.close(timeout=2)
    assert _records(tmp_path)[0]["value"] == 3.0
    assert not ipc_directory.exists()
    session.close(timeout=1)


def test_jsonl_node_cleans_ipc_directory_when_construction_raises(
    tmp_path, monkeypatch
):
    from tokenspeed.runtime.metrics import jsonl

    temporary_directory = tempfile.TemporaryDirectory
    directories = []

    def record_directory(*args, **kwargs):
        directory = temporary_directory(*args, **kwargs)
        directories.append(Path(directory.name))
        return directory

    monkeypatch.setattr(jsonl.tempfile, "TemporaryDirectory", record_directory)
    with pytest.raises(ValueError):
        jsonl.start_jsonl_node(
            {
                "directory": str(tmp_path),
                "max_bytes": -1,
                "backup_count": 2,
                "queue_size": 100,
                "flush_interval": 0.01,
            }
        )
    assert len(directories) == 1
    assert not directories[0].exists()
