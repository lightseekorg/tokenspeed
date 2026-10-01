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

"""Stopping a non-zero-rank node of a multi-node deployment (CPU-only).

Each test runs ``_launch_subprocesses`` as node rank 1 in a subprocess, with
stand-in schedulers in place of ``run_event_loop``, signals the node and
checks that none of its schedulers is left running.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import time

import psutil
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

# CPU-only; scheduled in runtime-1gpu because the node imports the full runtime.
register_cuda_ci(est_time=120, suite="runtime-1gpu")

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="parent-death signal is Linux-only"
)

# Node rank 1 with two stand-in schedulers. Each one reports its parent-death
# signal, starts a helper process of its own, reports ready and sleeps. On
# SIGTERM a "cooperative" scheduler leaves a marker file and exits 0, like
# run_event_loop's graceful stop; a "stuck" one ignores SIGTERM. With
# IN_EVENT_LOOP=1 the node launches inside asyncio.run, as the gRPC server
# does under `tokenspeed serve`.
NODE = """
import asyncio, ctypes, json, multiprocessing, os, pathlib, signal, subprocess, sys
import time
from types import SimpleNamespace as NS

PR_GET_PDEATHSIG = 2


def emit(**fields):
    print(json.dumps(fields), flush=True)


def stopped_gracefully(signum, frame):
    pathlib.Path(os.environ["MARKER_DIR"], str(os.getpid())).touch()
    sys.exit(0)


def scheduler(server_args, port_args, writer):
    pdeathsig = ctypes.c_int()
    ctypes.CDLL(None).prctl(PR_GET_PDEATHSIG, ctypes.byref(pdeathsig), 0, 0, 0)
    if os.environ["STAND_IN"] == "stuck":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    else:
        signal.signal(signal.SIGTERM, stopped_gracefully)
    subprocess.Popen(["sleep", "300"])
    emit(scheduler=os.getpid(), pdeathsig=pdeathsig.value)
    writer.send({"status": "ready"})
    time.sleep(300)


if __name__ == "__main__":
    from tokenspeed.runtime.entrypoints import engine

    multiprocessing.set_start_method("spawn", force=True)
    engine.configure_logger = engine._set_envs_and_config = lambda args: None
    engine.prepare_model_and_tokenizer = lambda *names: names
    engine.run_event_loop = scheduler
    engine.launch_dummy_health_check_server = lambda *args: emit(waiting=True)
    mapping = NS(attn=NS(has_dp=False), nprocs_per_node=2)
    args = NS(mapping=mapping, node_rank=1, model="", tokenizer="", host="", port=0)
    args.enable_memory_saver = args.enable_metrics = False
    if os.environ.get("IN_EVENT_LOOP") == "1":
        async def launch():
            engine._launch_subprocesses(args, NS())
        asyncio.run(launch())
    else:
        engine._launch_subprocesses(args, NS())
    sigterm_default = signal.getsignal(signal.SIGTERM) is signal.SIG_DFL
    emit(returned=True, sigterm_default=sigterm_default)
"""


@contextlib.contextmanager
def node_rank_1(tmp_path, stand_in, **env):
    """Start the node; yield it with its stand-in schedulers' reports and the
    node's first report after they are ready. Kill what is left on exit."""
    script = tmp_path / "node.py"
    script.write_text(NODE)
    env = {**os.environ, "STAND_IN": stand_in, "MARKER_DIR": str(tmp_path), **env}
    node = subprocess.Popen(
        [sys.executable, str(script)], stdout=subprocess.PIPE, text=True, env=env
    )
    tree = []
    try:
        reports = []
        for line in node.stdout:
            reports.append(json.loads(line))
            if "scheduler" not in reports[-1]:
                break
        *schedulers, report = reports
        assert len(schedulers) == 2, reports
        tree = psutil.Process(node.pid).children(recursive=True)
        yield node, schedulers, report, tree
    finally:
        for proc in [node, *tree]:
            with contextlib.suppress(psutil.NoSuchProcess):
                proc.kill()
        node.wait()


def left_running(procs, timeout):
    """Processes still running after ``timeout`` s; an unreaped zombie is gone."""
    _, alive = psutil.wait_procs(procs, timeout=timeout)
    return [p for p in alive if p.status() != psutil.STATUS_ZOMBIE]


@pytest.mark.parametrize(
    "name, env",
    [
        ("SIGTERM", {}),
        ("SIGINT", {}),
        ("SIGINT", {"IN_EVENT_LOOP": "1"}),
        ("SIGUSR1", {}),
    ],
    ids=["SIGTERM", "SIGINT", "SIGINT-in-event-loop", "SIGUSR1"],
)
def test_signal_stops_schedulers_then_node(tmp_path, name, env):
    signum = signal.Signals[name]
    with node_rank_1(tmp_path, "cooperative", **env) as started:
        node, schedulers, report, tree = started
        assert report == {"waiting": True}
        # SIGUSR1 is what a failing scheduler sends its node.
        node.send_signal(signum)
        # The node still ends by the signal it received.
        assert node.wait(timeout=60) == -signum
        assert left_running(tree, timeout=10) == []
        # Schedulers were asked to stop (SIGTERM) before anything was killed.
        assert sorted(p.name for p in tmp_path.iterdir() if p.name.isdigit()) == (
            sorted(str(s["scheduler"]) for s in schedulers)
        )


def test_stuck_scheduler_is_killed_after_timeout(tmp_path):
    env = {"TOKENSPEED_NONZERO_RANK_SHUTDOWN_TIMEOUT": "2"}
    with node_rank_1(tmp_path, "stuck", **env) as (node, _, report, tree):
        assert report == {"waiting": True}
        start = time.monotonic()
        node.send_signal(signal.SIGTERM)
        assert node.wait(timeout=60) == -signal.SIGTERM
        assert 2 <= time.monotonic() - start < 30
        assert left_running(tree, timeout=10) == []


def test_schedulers_die_with_killed_node(tmp_path):
    with node_rank_1(tmp_path, "stuck") as (node, schedulers, _, _):
        assert [s["pdeathsig"] for s in schedulers] == [signal.SIGKILL] * 2
        procs = [psutil.Process(s["scheduler"]) for s in schedulers]
        node.kill()
        node.wait(timeout=60)
        assert left_running(procs, timeout=10) == []


def test_non_blocking_node_is_unchanged(tmp_path):
    # As used through the Python ``Engine`` API: the caller owns the schedulers.
    env = {"TOKENSPEED_BLOCK_NONZERO_RANK_CHILDREN": "0"}
    with node_rank_1(tmp_path, "stuck", **env) as (_, schedulers, report, _):
        assert report == {"returned": True, "sigterm_default": True}
        assert [s["pdeathsig"] for s in schedulers] == [0, 0]


def test_parent_death_signal_when_parent_is_already_gone():
    # The process was reparented before it could ask for the signal.
    code = (
        "import os\n"
        "from tokenspeed.runtime.utils.process import run_with_parent_death_signal\n"
        "run_with_parent_death_signal(os.getpid(), print, 'target ran')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == -signal.SIGKILL
    assert "target ran" not in result.stdout


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
