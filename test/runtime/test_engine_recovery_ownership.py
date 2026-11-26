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

"""A failed launch must release only the ranks it owns before replacement."""

import contextlib
import subprocess
import sys
from pathlib import Path

import psutil
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, suite="runtime-1gpu")

from tokenspeed.runtime.engine.rust_host import start_rust_host
from tokenspeed.runtime.entrypoints import engine
from tokenspeed.runtime.utils.server_args import PortArgs, ServerArgs


def _host(path):
    path.write_text(f"#!{sys.executable}\n" + """
import sys,time
from multiprocessing.connection import Connection
args=dict(zip(sys.argv[1::2],sys.argv[2::2]))
connection=Connection(int(args['--control-fd']))
connection.recv()
connection.send({'status':'ready'})
time.sleep(60)
""")
    path.chmod(0o755)
    return str(path)


@pytest.mark.parametrize("encode", [False, True])
def test_partial_launch_closes_prior_rank_and_pipe(tmp_path, monkeypatch, encode):
    binary = _host(tmp_path / "host")
    monkeypatch.setattr(engine, "host_binary", lambda: binary)
    owned = []

    def start(binary, args, *rest):
        # Real child and control pipe; only GPU execution is substituted.
        if args.mapping.rank == 1:
            binary = str(tmp_path / "missing-host")
        proc, pipe = start_rust_host(binary, args, *rest)
        owned.append((proc, pipe))
        return proc, pipe

    monkeypatch.setattr(engine, "start_rust_host", start)
    args = ServerArgs(
        model="unused",
        device="cpu",
        world_size=2,
        nprocs_per_node=2,
        attn_tp_size=2,
        disaggregation_mode="encode" if encode else "null",
    )
    healthy = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        launch = engine._start_encode_ranks if encode else engine._start_ranks
        with pytest.raises(FileNotFoundError):
            launch(args, PortArgs.init_new(args), None, True)
        assert len(owned) == 1
        assert not owned[0][0].is_alive()
        assert owned[0][1].closed
        assert healthy.poll() is None
    finally:
        healthy.terminate()
        healthy.wait(timeout=5)
        for proc, pipe in owned:
            proc.kill()
            proc.join(timeout=5)
            pipe.close()


def test_failed_spawn_does_not_leak_descriptors(tmp_path):
    args = ServerArgs(model="unused", device="cpu")
    args.mapping.rank = 0
    ports = PortArgs.init_new(args)
    before = len(list(Path("/proc/self/fd").iterdir()))
    for _ in range(8):
        with pytest.raises(FileNotFoundError):
            start_rust_host(
                str(tmp_path / "missing"),
                args,
                ports,
                contextlib.nullcontext(),
                None,
                True,
            )
    assert len(list(Path("/proc/self/fd").iterdir())) == before


def test_failed_bootstrap_serialization_leaves_no_child(tmp_path):
    args = ServerArgs(model="unused", device="cpu")
    args.mapping.rank = 0
    ports = PortArgs.init_new(args)
    children = {p.pid for p in psutil.Process().children()}
    before = len(list(Path("/proc/self/fd").iterdir()))
    with pytest.raises((AttributeError, TypeError)):
        start_rust_host(
            _host(tmp_path / "host"),
            args,
            ports,
            contextlib.nullcontext(),
            lambda: None,
            True,
        )
    assert {p.pid for p in psutil.Process().children()} == children
    assert len(list(Path("/proc/self/fd").iterdir())) == before
