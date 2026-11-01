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

Each test runs ``launch_scheduler_headless`` as node rank 1 in a subprocess,
with a stand-in ``ts-host`` binary in place of the real one, signals the node
and checks that none of its schedulers is left running. Only ``host_binary``
and the node's service setup are substituted: the ranks start through the
real ``start_rust_host``, so the parent-death signal they report is the one
the launcher asked the kernel for.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import stat
import subprocess
import sys
import textwrap
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

# Node rank 1 with two stand-in schedulers: the FAKE_HOST binary, started by
# the real start_rust_host. The node reports once it supervises them.
NODE = """
import functools, json, os, sys
from types import SimpleNamespace as NS


def emit(**fields):
    print(json.dumps(fields), flush=True)


if __name__ == "__main__":
    from tokenspeed.runtime.entrypoints import engine

    engine.configure_logger = engine._set_envs_and_config = lambda args: None
    engine.prepare_model_and_tokenizer = lambda *names: names
    engine.host_binary = lambda: os.environ["FAKE_HOST"]
    engine.launch_metrics_server = lambda args: None
    engine.rust_host_refusals = lambda args: []
    engine._start_kv_bootstrap_server = lambda args: None
    engine.metrics_service = NS(
        prepare=lambda args: None,
        start=lambda args, port_args: None,
        close=lambda timeout: None,
    )
    engine.PortArgs = NS(init_new=lambda args: NS(metrics_producer_configs=None))
    engine._SCHEDULER_EXIT_GRACE_S = float(os.environ.get("EXIT_GRACE_S", "30"))
    supervise = engine._supervise_schedulers

    def supervise_and_report(*args, **kwargs):
        emit(waiting=True)
        return supervise(*args, **kwargs)

    engine._supervise_schedulers = supervise_and_report
    mapping = NS(
        attn=NS(has_dp=False, tp_rank=0, has_kvp=False),
        moe=NS(has_ep=False),
        nprocs_per_node=2,
        nnodes=2,
        has_pp=False,
    )
    args = NS(
        mapping=mapping,
        node_rank=1,
        model="",
        tokenizer="",
        disaggregation_mode="null",
        kv_events_config=None,
        dist_init_addr="127.0.0.1:25000",
        # The ranks get the args pickled: a builtin's partial survives that.
        zmq_handshake_endpoint=functools.partial(str, "tcp://127.0.0.1:25001"),
    )
    args.enable_memory_saver = args.enable_metrics = False
    args.emulate_rank_zero = False
    engine.launch_scheduler_headless(args)
"""

# A stand-in `ts-host`: takes the host command line, reports its parent-death
# signal, starts a helper process of its own, reports ready on the control
# connection and sleeps. On SIGTERM a "cooperative" stand-in leaves a marker
# file and exits 0, like a rank's graceful stop; a "stuck" one ignores
# SIGTERM. The report is printed before the ready message so it always
# reaches the node's stdout before the node's own first report.
FAKE_HOST = """
import ctypes, json, os, pathlib, signal, subprocess, sys, time
from multiprocessing.connection import Connection

PR_GET_PDEATHSIG = 2

args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
connection = Connection(int(args["--control-fd"]))

pdeathsig = ctypes.c_int()
ctypes.CDLL(None).prctl(PR_GET_PDEATHSIG, ctypes.byref(pdeathsig), 0, 0, 0)


def stopped_gracefully(signum, frame):
    pathlib.Path(os.environ["MARKER_DIR"], str(os.getpid())).touch()
    sys.exit(0)


server_args, port_args, relay = connection.recv()
if os.environ["STAND_IN"] == "stuck":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
else:
    signal.signal(signal.SIGTERM, stopped_gracefully)
subprocess.Popen(["sleep", "300"])
print(json.dumps({"scheduler": os.getpid(), "pdeathsig": pdeathsig.value}), flush=True)
connection.send({"status": "ready"})
time.sleep(300)
"""


@contextlib.contextmanager
def node_rank_1(tmp_path, stand_in, **env):
    """Start the node; yield it with its stand-in schedulers' reports and the
    node's first report after they are ready. Kill what is left on exit."""
    script = tmp_path / "node.py"
    script.write_text(NODE)
    fake_host = tmp_path / "ts-host"
    fake_host.write_text(f"#!{sys.executable}\n" + textwrap.dedent(FAKE_HOST))
    fake_host.chmod(fake_host.stat().st_mode | stat.S_IXUSR)
    env = {
        **os.environ,
        "STAND_IN": stand_in,
        "MARKER_DIR": str(tmp_path),
        "FAKE_HOST": str(fake_host),
        **env,
    }
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


@pytest.mark.parametrize("name", ["SIGTERM", "SIGINT", "SIGUSR1"])
def test_signal_stops_schedulers_then_node(tmp_path, name):
    signum = signal.Signals[name]
    with node_rank_1(tmp_path, "cooperative") as started:
        node, schedulers, report, tree = started
        assert report == {"waiting": True}
        # SIGUSR1 is what a failing scheduler sends its node: it fails.
        node.send_signal(signum)
        assert node.wait(timeout=60) == (1 if signum == signal.SIGUSR1 else 0)
        assert left_running(tree, timeout=10) == []
        # Schedulers were asked to stop (SIGTERM) before anything was killed.
        assert sorted(p.name for p in tmp_path.iterdir() if p.name.isdigit()) == (
            sorted(str(s["scheduler"]) for s in schedulers)
        )


def test_stuck_scheduler_is_killed_after_the_grace_period(tmp_path):
    with node_rank_1(tmp_path, "stuck", EXIT_GRACE_S="2") as (node, _, report, tree):
        assert report == {"waiting": True}
        start = time.monotonic()
        node.send_signal(signal.SIGTERM)
        assert node.wait(timeout=60) == 1
        assert 2 <= time.monotonic() - start < 30
        assert left_running(tree, timeout=10) == []


def test_schedulers_die_with_killed_node(tmp_path):
    with node_rank_1(tmp_path, "stuck") as (node, schedulers, _, _):
        assert [s["pdeathsig"] for s in schedulers] == [signal.SIGKILL] * 2
        procs = [psutil.Process(s["scheduler"]) for s in schedulers]
        node.kill()
        node.wait(timeout=60)
        assert left_running(procs, timeout=10) == []


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
