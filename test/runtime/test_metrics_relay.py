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

"""Exercise relay isolation, delivery and shutdown with real local processes."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, suite="runtime-1gpu")


@pytest.fixture
def relay_environment(tmp_path):
    """Install real reporter plugins; isolate GPU discovery and the model catalog."""
    (tmp_path / "recording_output.py").write_text(textwrap.dedent(r"""
        import json
        import os
        import sys
        import time
        from pathlib import Path

        from tokenspeed.runtime.metrics.registry import register_metric_reporter

        def record(kind, **values):
            with open(os.environ["RELAY_RECORD"], "a") as stream:
                stream.write(json.dumps({"kind": kind, "pid": os.getpid(), **values}) + "\n")

        class Sink:
            def __init__(self, config, implementation):
                self.config = config
                self.implementation = implementation
                record("sink", implementation=implementation, heavy=[
                    name for name in sys.modules
                    if name == "tokenspeed_kernel" or name.startswith("tokenspeed_kernel.")
                    or name.startswith("tokenspeed.runtime.engine")
                    or name == "unrelated_model"
                ])
                if config.get("fail"):
                    raise RuntimeError("fixture sink startup failed")

            def emit(self, event):
                record("event", metric=event.metric, value=event.value, weight=event.weight,
                       labels=event.labels, implementation=self.implementation)
                Path(self.config["received"]).touch()

            def close(self, timeout):
                record("close", timeout=timeout)
                if self.config.get("hang_close"):
                    time.sleep(30)

        def register_output(implementation, override):
            # A nested factory remains supported: the registration is replayed,
            # rather than pickling Python callbacks into the child.
            def create_sink(config):
                return Sink(config, implementation)
            register_metric_reporter("recorded", validate=lambda config: config,
                                     create_sink=create_sink, start_node=None,
                                     override=override)

        def register():
            register_output("base", False)

        def override():
            register_output("override", True)

        def broken():
            register_metric_reporter("rolled_back", validate=lambda config: config,
                                     create_sink=lambda config: None, start_node=None,
                                     override=False)
            raise RuntimeError("fixture registration failure")
    """))
    (tmp_path / "unrelated_model.py").write_text(textwrap.dedent(r"""
        from recording_output import record
        record("model_import")
        def register():
            record("model_register")
    """))
    dist = tmp_path / "relay_fixture-1.0.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: relay-fixture\nVersion: 1.0\n"
    )
    (dist / "entry_points.txt").write_text(textwrap.dedent(r"""
        [tokenspeed.plugins]
        a_output = recording_output:register
        b_broken = recording_output:broken
        c_model = unrelated_model:register
        z_override = recording_output:override
    """))
    failed_distribution = tmp_path / "failed_output-1.0.dist-info"
    failed_distribution.mkdir()
    (failed_distribution / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: failed-output\nVersion: 1.0\n"
    )
    (failed_distribution / "entry_points.txt").write_text(
        "[tokenspeed.plugins]\na_output = recording_output:broken\n"
    )
    launcher = tmp_path / "launcher.py"
    launcher.write_text(textwrap.dedent(r"""
        import json
        import os
        import sys
        import time
        import types
        from pathlib import Path

        # The parent is an inference launcher. Only GPU kernel discovery and
        # platform support are unavailable in this CPU lifecycle test.
        kernel = types.ModuleType("tokenspeed_kernel")
        kernel.__path__ = []
        platform = types.ModuleType("tokenspeed_kernel.platform")
        platform.current_platform = object()
        kernel_plugins = types.ModuleType("tokenspeed_kernel.plugins")
        kernel_plugins.discover_plugins = lambda: None
        sys.modules.update({"tokenspeed_kernel": kernel,
                            "tokenspeed_kernel.platform": platform,
                            "tokenspeed_kernel.plugins": kernel_plugins})

        # Load the real registry/profile modules without the unrelated model
        # configuration catalog, whose Transformers version belongs to the GPU image.
        import tokenspeed
        configs = types.ModuleType("tokenspeed.runtime.configs")
        configs.__path__ = [str(Path(base) / "runtime" / "configs") for base in tokenspeed.__path__]
        sys.modules[configs.__name__] = configs

        from recording_output import record
        record("launcher_import")
        from tokenspeed.runtime import plugins
        from tokenspeed.runtime.metrics.registry import get_metric_reporter
        from tokenspeed.runtime.metrics.relay import MetricsRelay
        import msgspec
        import zmq

        if __name__ == "__main__":
            plugins.ensure_loaded()
            try:
                get_metric_reporter("rolled_back")
            except ValueError:
                pass
            else:
                raise AssertionError("failed plugin registration survived rollback")
            received = Path(os.environ["RELAY_RECEIVED"])
            mode = sys.argv[1]
            relay = MetricsRelay({"recorded": {"received": str(received),
                                              "fail": mode == "fail",
                                              "hang_close": mode == "kill"}})
            try:
                context = zmq.Context()
                with context.socket(zmq.PUSH) as push:
                    push.setsockopt(zmq.SNDTIMEO, 5000)
                    push.setsockopt(zmq.LINGER, 0)
                    push.connect(relay.endpoint)
                    push.send(msgspec.msgpack.encode([1, "ready-event", "observe", 0.5,
                                                     {"rank": "0"}, 3]))
                    deadline = time.monotonic() + 10
                    while not received.exists():
                        if time.monotonic() >= deadline:
                            raise AssertionError("relay did not deliver the first event")
                        time.sleep(0.01)
                context.term()
            finally:
                if mode != "exit":
                    relay.close(timeout=0.1 if mode == "kill" else 2.0)
                    relay.close(timeout=0.1)
                    try:
                        os.kill(relay._process.pid, 0)
                    except ProcessLookupError:
                        pass
                    else:
                        raise AssertionError("closed relay is still alive")
    """))
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(tmp_path),
            env.get("PYTHONPATH", str(Path(__file__).resolve().parents[2] / "python")),
        ]
    )
    env["RELAY_RECORD"] = str(tmp_path / "record.jsonl")
    env["RELAY_RECEIVED"] = str(tmp_path / "received")
    env["TOKENSPEED_DISABLE_PLUGINS"] = "z_override"
    return launcher, env


def run_launcher(relay_environment, mode):
    launcher, env = relay_environment
    record = Path(env["RELAY_RECORD"])
    record.touch()
    try:
        result = subprocess.run(
            [sys.executable, str(launcher), mode],
            env=env,
            capture_output=True,
            text=True,
            timeout=45,
        )
    finally:
        events = [json.loads(line) for line in record.read_text().splitlines()]
        child_pids = {event["pid"] for event in events if event["kind"] == "sink"}
        survivors = []
        for pid in child_pids:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                continue
            survivors.append(pid)
            os.kill(pid, signal.SIGKILL)
    assert not survivors, f"relay processes survived launcher shutdown: {survivors}"
    return result, events


@pytest.mark.parametrize("override", [False, True])
def test_metrics_relay_delivers_without_reimporting_launcher_or_models(
    relay_environment, override
):
    if override:
        relay_environment[1]["TOKENSPEED_DISABLE_PLUGINS"] = ""
    result, events = run_launcher(relay_environment, "normal")
    assert result.returncode == 0, result.stderr
    assert len([event for event in events if event["kind"] == "launcher_import"]) == 1
    assert len([event for event in events if event["kind"] == "model_import"]) == 1
    assert len([event for event in events if event["kind"] == "model_register"]) == 1
    (sink,) = [event for event in events if event["kind"] == "sink"]
    assert sink["heavy"] == []
    (event,) = [event for event in events if event["kind"] == "event"]
    assert event == {
        "kind": "event",
        "pid": sink["pid"],
        "metric": "ready-event",
        "value": 0.5,
        "weight": 3,
        "labels": {"rank": "0"},
        "implementation": "override" if override else "base",
    }
    assert len([event for event in events if event["kind"] == "close"]) == 1


def test_metrics_relay_propagates_startup_failure_and_reaps_child(
    relay_environment,
):
    result, events = run_launcher(relay_environment, "fail")
    assert result.returncode != 0
    assert "the metrics relay failed to start" in result.stderr
    assert "fixture sink startup failed" in result.stderr
    assert not [event for event in events if event["kind"] == "event"]


def test_metrics_relay_closes_sinks_on_launcher_exit(relay_environment):
    result, events = run_launcher(relay_environment, "exit")
    assert result.returncode == 0, result.stderr
    assert len([event for event in events if event["kind"] == "close"]) == 1


def test_metrics_relay_kills_child_when_sink_exceeds_shutdown_budget(
    relay_environment,
):
    result, events = run_launcher(relay_environment, "kill")
    assert result.returncode == 0, result.stderr
    assert len([event for event in events if event["kind"] == "event"]) == 1
