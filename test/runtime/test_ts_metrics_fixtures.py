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

"""The Python half of ``ts-metrics``' fixtures, and the metrics relay.

``tokenspeed-engine/crates/ts-metrics/tests/fixtures/`` records what the
Python engine's metrics export and the message the relay reads; the Rust
crate must produce both. The fixture tests fail when the Python metrics
change without the fixtures, and so the Rust port, following them.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import threading
from pathlib import Path

import msgspec
import pytest
import zmq

# CI registration (AST-parsed, runtime no-op).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

from tokenspeed.runtime.metrics.events import MetricEvent  # noqa: E402
from tokenspeed.runtime.metrics.relay import decode_event, relay_events  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

_FIXTURE_DIR = (
    Path(__file__).resolve().parents[2]
    / "tokenspeed-engine"
    / "crates"
    / "ts-metrics"
    / "tests"
    / "fixtures"
)


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "ts_metrics_fixture_generator", _FIXTURE_DIR / "generate.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_GENERATOR = _load_generator()
_EXPECTED = _GENERATOR.build_fixtures()


@pytest.mark.parametrize("file_name", sorted(_EXPECTED))
def test_fixture_matches_python_metrics(file_name: str) -> None:
    on_disk = (_FIXTURE_DIR / file_name).read_text()
    assert on_disk == _EXPECTED[file_name], (
        f"{file_name} no longer matches the Python metrics. If they changed on "
        "purpose, run tokenspeed-engine/crates/ts-metrics/tests/fixtures/"
        "generate.py and update ts-metrics to match."
    )


def test_the_relay_decodes_a_host_message() -> None:
    frame = bytes.fromhex((_FIXTURE_DIR / "relay_event.hex").read_text().strip())
    assert decode_event(frame) == MetricEvent(
        timestamp_ns=1_700_000_000_123_456_789,
        metric="tokenspeed:kv_transfer_timeouts_total",
        operation="inc",
        value=1.0,
        labels={"model_name": "m", "app_key": "", "stage": "bootstrapping"},
        weight=1,
    )


class _CollectingSink:
    def __init__(self, expected: int) -> None:
        self.events: list[MetricEvent] = []
        self.done = threading.Event()
        self._expected = expected

    def emit(self, event: MetricEvent) -> None:
        self.events.append(event)
        if len(self.events) == self._expected:
            self.done.set()

    def close(self, timeout: float) -> None:
        pass


def test_the_relay_hands_every_event_to_the_sink(tmp_path) -> None:
    endpoint = f"ipc://{tmp_path}/relay"
    context = zmq.Context()
    pull = context.socket(zmq.PULL)
    pull.bind(endpoint)
    push = context.socket(zmq.PUSH)
    push.connect(endpoint)
    sink = _CollectingSink(expected=3)
    relay = threading.Thread(
        target=relay_events, args=(pull, sink, sink.done.is_set), daemon=True
    )
    relay.start()
    for value in (1.0, 2.5, 4.0):
        push.send(
            msgspec.msgpack.encode(
                [1, "tokenspeed:num_requests_running", "set", value, {"a": "b"}, 1]
            )
        )
    relay.join(timeout=10)
    push.close(linger=0)
    pull.close(linger=0)
    context.term()
    assert not relay.is_alive()
    assert [event.value for event in sink.events] == [1.0, 2.5, 4.0]
    assert all(event.operation == "set" for event in sink.events)
