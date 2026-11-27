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

"""The node's metrics relay: output plugins for the Rust engine hosts.

A Rust host (``tokenspeed-engine/crates/ts-host``) produces the engine's
metrics in its control thread, which holds no interpreter, so the output
plugins (``MetricEventSink``) cannot run there. The headless launcher starts
one relay process per node instead: the hosts send it each ``MetricEvent``
over a local socket (``ts-metrics``), and the relay hands the events to the
sinks the selected reporters create. A slow or failing plugin then stalls
neither a rank nor its interpreter.

The relay owns the sinks only; the launcher keeps the node sessions
(``metrics_service.start``) and the Prometheus endpoint, which reads the
hosts' multiprocess files directly. A standalone interpreter loads only the
runtime entry points that registered the selected outputs in the launcher.
"""

from __future__ import annotations

import atexit
import multiprocessing as mp
import os
import signal
import subprocess
import sys
import tempfile
import threading
import uuid
from collections.abc import Callable
from multiprocessing.connection import Connection

import msgspec
import zmq

from tokenspeed.runtime.metrics.events import (
    MetricConfig,
    MetricEvent,
    MetricEventSink,
)
from tokenspeed.runtime.metrics.service import metrics_service

# How long the relay waits for a message before checking for a stop.
_POLL_MS = 100


def _reporter_entry_points(names: dict[str, MetricConfig]) -> list[str]:
    """Select successful launcher registrations, preserving override order."""
    from tokenspeed.runtime.plugins import list_plugins

    return [
        plugin.entry_point
        for plugin in list_plugins()
        if plugin.entry_point is not None
        and any(
            kind == "metric reporter" and names.keys() & registered
            for kind, registered in plugin.registrations
        )
    ]


def decode_event(frame: bytes) -> MetricEvent:
    """One host message: ``[timestamp_ns, metric, operation, value, labels,
    weight]`` as a msgpack array."""
    timestamp_ns, metric, operation, value, labels, weight = msgspec.msgpack.decode(
        frame
    )
    return MetricEvent(
        timestamp_ns=timestamp_ns,
        metric=metric,
        operation=operation,
        value=float(value),
        labels=labels,
        weight=weight,
    )


def relay_events(
    socket: zmq.Socket, sink: MetricEventSink, stopping: Callable[[], bool]
) -> None:
    """Hand every event ``socket`` receives to ``sink`` until ``stopping``
    returns true."""
    while not stopping():
        if not socket.poll(_POLL_MS):
            continue
        # Drain what arrived before checking for a stop again.
        while True:
            try:
                frame = socket.recv(zmq.NOBLOCK)
            except zmq.Again:
                break
            sink.emit(decode_event(frame))


def _relay_main(
    endpoint: str,
    producer_configs: dict[str, MetricConfig],
    ready: Connection,
    entry_points: list[str],
) -> None:
    """Create selected sinks, acknowledge the bound socket, then relay events."""
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda _signum, _frame: stop.set())
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    metrics_service.connect(producer_configs, reporter_entry_points=entry_points)
    sink = metrics_service.get_event_sink()
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.bind(endpoint)
    ready.send(True)
    ready.close()
    try:
        if sink is not None:
            relay_events(socket, sink, stop.is_set)
    finally:
        socket.close(linger=0)
        context.term()
        metrics_service.close_sinks(timeout=2.0)


class MetricsRelay:
    """The launcher's handle on a running relay.

    Args:
        producer_configs: The selected output plugins' producer
            configurations (``PortArgs.metrics_producer_configs``).
    """

    def __init__(self, producer_configs: dict[str, MetricConfig]) -> None:
        self.endpoint = (
            f"ipc://{tempfile.gettempdir()}/ts-metrics-relay-"
            f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
        )
        entry_points = _reporter_entry_points(producer_configs)
        reader, writer = mp.Pipe()
        try:
            # A fresh module entry avoids multiprocessing.spawn re-importing
            # the launcher's engine/model stack before entering the relay.
            self._process = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    "from tokenspeed.runtime.metrics.relay import main; main()",
                    str(writer.fileno()),
                ],
                pass_fds=(writer.fileno(),),
            )
        except BaseException:
            reader.close()
            raise
        finally:
            writer.close()
        try:
            reader.send((self.endpoint, producer_configs, entry_points))
            reader.recv()
        except (EOFError, BrokenPipeError, ConnectionResetError):
            self.close(timeout=1.0)
            raise RuntimeError(
                "the metrics relay failed to start; see its error above"
            ) from None
        except BaseException:
            self.close(timeout=1.0)
            raise
        finally:
            reader.close()
        atexit.register(self.close, timeout=2.0)

    def close(self, timeout: float) -> None:
        """Stop the relay; it closes its sinks within ``timeout``."""
        atexit.unregister(self.close)
        if self._process.poll() is None:
            self._process.terminate()
        try:
            self._process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=1.0)


def main() -> None:
    """Start only the launcher's selected metric plugins in a fresh interpreter."""
    with Connection(int(sys.argv[1])) as ready:
        endpoint, configs, entry_points = ready.recv()
        _relay_main(endpoint, configs, ready, entry_points)


if __name__ == "__main__":
    main()
