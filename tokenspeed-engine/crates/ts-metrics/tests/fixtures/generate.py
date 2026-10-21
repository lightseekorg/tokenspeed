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

"""What the Python engine's metrics export, for ``ts-metrics`` to match.

``exposition.txt`` is the Prometheus exposition of one scenario of
``EngineMetrics`` and ``KVTransferMetrics`` updates, written in
multiprocess mode and read back by ``MultiProcessCollector``, its lines
sorted. ``relay_event.hex`` is one ``MetricEvent`` as the metrics relay
(``metrics/relay.py``) receives it. ``ts-metrics``' tests run the same
scenario and must produce both; ``test/runtime/test_ts_metrics_fixtures.py``
checks the files still match Python. Rewrite them after an intentional
change:

    python tokenspeed-engine/crates/ts-metrics/tests/fixtures/generate.py

``generate.py --render DIR`` prints the sorted exposition of the
multiprocess files in ``DIR``; it needs only ``prometheus_client``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

FIXTURE_DIR = Path(__file__).resolve().parent

#: One relay message: ``[timestamp_ns, metric, operation, value, labels,
#: weight]``.
RELAY_EVENT = [
    1_700_000_000_123_456_789,
    "tokenspeed:kv_transfer_timeouts_total",
    "inc",
    1.0,
    {"model_name": "m", "app_key": "", "stage": "bootstrapping"},
    1,
]


def render(directory: str) -> str:
    """The sorted exposition of the multiprocess files in ``directory``."""
    from prometheus_client import CollectorRegistry, generate_latest, multiprocess

    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry, path=directory)
    lines = generate_latest(registry).decode().splitlines()
    return "\n".join(sorted(lines)) + "\n"


def _run_scenario() -> None:
    """The updates ``ts-metrics``' test repeats, into this process's
    multiprocess files (``PROMETHEUS_MULTIPROC_DIR`` is set by the caller
    before ``prometheus_client`` is imported)."""
    from tokenspeed_kernel.compile_monitor import CompileStats

    from tokenspeed.runtime.metrics.collector import EngineMetrics, KVTransferMetrics

    engine = EngineMetrics(
        labels={"model_name": "m", "app_key": "", "dp_rank": "0"},
        enabled=True,
        reporters=["prometheus"],
    )
    engine.record_scheduler_iteration(
        running=3,
        waiting=2,
        num_active_pages=10,
        num_total_pages=40,
        num_iteration_tokens=17,
    )
    engine.record_scheduler_iteration(
        running=1,
        waiting=0,
        num_active_pages=4,
        num_total_pages=40,
        num_iteration_tokens=0,
    )
    engine.record_spec_decode_step(
        num_decode_slots=4, accepted_draft_tokens=5, draft_width=3
    )
    engine.record_jit_compiles(
        CompileStats(
            startup_compiles=9,
            startup_seconds=4.0,
            serving_compiles=2,
            serving_seconds=0.5,
        )
    )
    engine.record_nan_abort()
    engine.record_capacity_abort()
    engine.record_preemptions(2)
    engine.observe_request_queue_time(0.4)
    engine.observe_request_queue_time(9000.0)

    transfer = KVTransferMetrics(
        labels={"model_name": "m", "app_key": ""},
        enabled=True,
        reporters=["prometheus"],
    )
    transfer.observe_kv_transfer_latency(0.003)
    transfer.record_kv_transfer_timeout("bootstrapping")
    transfer.record_kv_transfer_timeout("waiting_for_input")
    transfer.record_kv_transfer_failure()


def build_fixtures() -> dict[str, str]:
    """File name -> expected content."""
    import msgspec

    with tempfile.TemporaryDirectory() as directory:
        # prometheus_client picks its multiprocess values at import, so the
        # scenario runs in a fresh interpreter.
        subprocess.run(
            [sys.executable, __file__, "--scenario"],
            env={**os.environ, "PROMETHEUS_MULTIPROC_DIR": directory},
            check=True,
        )
        exposition = render(directory)
    return {
        "exposition.txt": exposition,
        "relay_event.hex": msgspec.msgpack.encode(RELAY_EVENT).hex() + "\n",
    }


def main() -> None:
    if sys.argv[1:] == ["--scenario"]:
        _run_scenario()
    elif len(sys.argv) == 3 and sys.argv[1] == "--render":
        sys.stdout.write(render(sys.argv[2]))
    else:
        for name, content in build_fixtures().items():
            (FIXTURE_DIR / name).write_text(content)


if __name__ == "__main__":
    main()
