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

"""Launcher cleanup after real worker failures and termination signals.

The headless launcher (``launch_scheduler_headless``) owns its ranks'
process tree. These tests replace only the ranks it starts with real child
processes and check that a rank's failure signal, repeated signals, and a
normal exit each leave nothing running behind it.
"""

import contextlib
import json
import multiprocessing as mp
import os
import signal
import sys
import threading

import psutil
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=20, suite="runtime-1gpu")

from tokenspeed.runtime.utils.process import (  # noqa: E402
    register_child_failure_signals,
    start_process_with_ready_pipe,
)

_TIMEOUT_S = 60


@pytest.fixture(scope="module", autouse=True)
def _spawn_start_method():
    """Run this module's children on production's spawn start method.

    Several children import the engine stack inside the child; a forked child
    cannot re-initialize a parent's CUDA state on that import path, which is
    why production sets spawn in ``_set_envs_and_config``. These tests bypass
    that entry point, so set the same method here: every process, Event,
    Barrier and nested worker created in this module then shares one context.
    Restore the previous method so other modules are unaffected.
    """
    previous = mp.get_start_method(allow_none=True)
    mp.set_start_method("spawn", force=True)
    try:
        yield
    finally:
        mp.set_start_method(previous, force=True)


def _wait_for_termination(ready, writer):
    writer.send({"status": "ready"})
    ready.set()
    threading.Event().wait(_TIMEOUT_S * 10)


def _report_and_exit(writer):
    writer.send({"status": "ready"})


def _fail_after_ready(signals, peer_ready, writer):
    """A rank that reports ready and, once its peer has too, fails the way
    ``ts-host`` does: it signals its launcher (SIGUSR1; SIGQUIT from its
    process-tree cleanup)."""
    writer.send({"status": "ready"})
    peer_ready.wait(_TIMEOUT_S)
    for sig in signals:
        os.kill(os.getppid(), sig)
    threading.Event().wait(_TIMEOUT_S * 10)


def _run_launcher(workers, server_args, path, writer):
    """Run the headless launcher with its ranks replaced by ``workers``
    (``(target, args)`` each), with ``path`` first on ``sys.path``, and
    report how it ended and what of them it left running."""
    # Spawned processes inherit this path, so they load what it holds.
    sys.path[:0] = path
    from tokenspeed.runtime.entrypoints import engine as engine_module
    from tokenspeed.runtime.utils.server_args import ServerArgs

    started = []

    def start_without_model(server_args, port_args, metrics_relay, parent_death_signal):
        # Replace GPU/model loading only: the launcher's supervision, signal
        # handlers and process-tree shutdown are the real implementation.
        for target, args in workers:
            started.append(
                start_process_with_ready_pipe(target, args, contextlib.nullcontext())
            )
        return [proc for proc, _ in started], [reader for _, reader in started]

    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(engine_module, "_start_ranks", start_without_model)
            status = 0
            try:
                engine_module.launch_scheduler_headless(
                    ServerArgs(
                        model="unused",
                        device="cpu",
                        world_size=1,
                        nprocs_per_node=1,
                        **server_args,
                    )
                )
            except SystemExit as exc:
                status = exc.code
            except RuntimeError:
                # A failure the launcher sees before it has read every
                # rank's readiness unwinds it this way; its process exits
                # with 1.
                status = 1
        for proc, _ in started:
            proc.join(_TIMEOUT_S)
        writer.send(
            {
                "status": status,
                "alive": [proc.is_alive() for proc, _ in started],
                "exitcodes": [proc.exitcode for proc, _ in started],
            }
        )
    finally:
        for proc, reader in started:
            if proc.is_alive():
                proc.kill()
            proc.join(_TIMEOUT_S)
            reader.close()
        writer.close()


def _launch(workers, server_args=None, path=()):
    """The launcher's report, from a child process that runs it."""
    proc, reader = start_process_with_ready_pipe(
        _run_launcher,
        (workers, server_args or {}, list(path)),
        contextlib.nullcontext(),
    )
    try:
        # The child imports the engine stack and waits on its workers with
        # their own _TIMEOUT_S budgets.
        assert reader.poll(_TIMEOUT_S * 3), "the launcher did not finish"
        result = reader.recv()
        proc.join(_TIMEOUT_S)
        assert proc.exitcode == 0
        return result
    finally:
        _stop_supervisor(proc)
        reader.close()


def _stop_supervisor(proc):
    if proc.is_alive():
        # Test-only fallback must stop descendants if the regression hangs.
        try:
            for child in psutil.Process(proc.pid).children(recursive=True):
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
        except psutil.NoSuchProcess:
            pass
        proc.kill()
    proc.join(_TIMEOUT_S)


@pytest.mark.parametrize("sig", [signal.SIGUSR1, signal.SIGQUIT])
def test_a_rank_failure_signal_takes_every_rank_down(sig):
    peer_ready = mp.Event()
    result = _launch(
        [
            (_fail_after_ready, ((sig,), peer_ready)),
            (_wait_for_termination, (peer_ready,)),
        ]
    )
    assert result["status"] == 1
    assert result["alive"] == [False, False]


def test_repeated_failure_signals_cannot_interrupt_cleanup():
    # Both kinds, twice: the first unwinds the launcher; none of the others
    # may cut its cleanup short.
    signals = (signal.SIGUSR1, signal.SIGQUIT, signal.SIGUSR1, signal.SIGQUIT)
    peer_ready = mp.Event()
    result = _launch(
        [
            (_fail_after_ready, (signals, peer_ready)),
            (_wait_for_termination, (peer_ready,)),
        ]
    )
    assert result["status"] == 1
    assert result["alive"] == [False, False]


def test_a_launcher_installed_after_a_failure_receives_its_own():
    # A launcher started after a failed one in the same process must receive
    # its own first failure, even after the earlier one latched the shutdown.
    exitcodes = []
    for _ in range(2):
        register_child_failure_signals()
        try:
            os.kill(os.getpid(), signal.SIGUSR1)
        except SystemExit as exc:
            exitcodes.append(exc.code)
    assert exitcodes == [1, 1]


# A metric reporter plugin whose sink records that it was closed.
_RELAY_PROBE = """
import pathlib

from tokenspeed.runtime.metrics.registry import register_metric_reporter


class _Sink:
    def __init__(self, path):
        self._path = pathlib.Path(path)

    def emit(self, event):
        pass

    def close(self, timeout):
        self._path.write_text("closed")


def register():
    register_metric_reporter(
        "relay-probe",
        validate=lambda config: config,
        create_sink=lambda config: _Sink(config["path"]),
        start_node=None,
        override=False,
    )
"""


def _install_relay_probe(directory):
    """Install the probe plugin as a distribution on ``directory``."""
    (directory / "relay_probe.py").write_text(_RELAY_PROBE)
    dist = directory / "relay_probe-0.0.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: relay-probe\nVersion: 0.0\n"
    )
    (dist / "entry_points.txt").write_text(
        "[tokenspeed.plugins]\nrelay-probe = relay_probe:register\n"
    )


def test_the_launcher_lets_the_metrics_relay_flush(tmp_path):
    # The relay closes its sinks on a stop; killed with the process tree, it
    # would drop what they still buffer.
    _install_relay_probe(tmp_path)
    marker = tmp_path / "closed"
    config = tmp_path / "reporters.json"
    config.write_text(json.dumps({"relay-probe": {"path": str(marker)}}))
    result = _launch(
        [(_report_and_exit, ())],
        {
            "enable_metrics": True,
            "metrics_reporters": ["relay-probe"],
            "metrics_reporter_config": str(config),
        },
        path=[str(tmp_path)],
    )
    assert result["status"] == 0
    assert marker.read_text() == "closed"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
