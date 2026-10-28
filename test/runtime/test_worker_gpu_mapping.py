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

"""Worker startup preserves explicit physical GPU spacing."""

import multiprocessing as mp
import signal
from unittest.mock import patch

import pytest

from tokenspeed.runtime.utils.server_args import PortArgs, ServerArgs


def _observe_worker(rank, base, result):
    import tokenspeed.runtime.engine.event_loop as worker

    args = ServerArgs(
        model="unused",
        world_size=16,
        nprocs_per_node=2,
        nnodes=8,
        attn_tp_size=1,
        data_parallel_size=16,
        moe_tp_size=1,
        ep_size=16,
        base_gpu_id=base,
        gpu_id_step=2,
    )
    args.mapping.rank = rank
    ports = PortArgs("", "", 0, "", "", "", None)

    def unavailable_model(server_args, port_args, gpu_id, *unused):
        raise SystemExit(gpu_id)

    reader, writer = mp.Pipe(duplex=False)
    try:
        with patch.object(worker, "EventLoop", unavailable_model):
            try:
                worker.run_event_loop(args, ports, writer)
            except SystemExit as stopped:
                result.send(stopped.code)
    finally:
        reader.close()
        writer.close()
        result.close()


@pytest.mark.parametrize("base", [0, 1])
@pytest.mark.parametrize("rank", [0, 1, 14, 15])
def test_worker_build_receives_interleaved_gpu_id(base, rank):
    context = mp.get_context("spawn")
    reader, writer = context.Pipe(duplex=False)
    child = context.Process(target=_observe_worker, args=(rank, base, writer))
    failures = []
    previous = signal.getsignal(signal.SIGUSR1)
    signal.signal(signal.SIGUSR1, lambda *_: failures.append(True))
    try:
        child.start()
        writer.close()
        child.join(45)
        assert not child.is_alive(), "worker startup did not reach model construction"
        assert child.exitcode == 0 and not failures
        assert reader.poll(1), "worker did not report its selected GPU"
        assert reader.recv() == base + (rank % 2) * 2
    finally:
        if child.is_alive():
            child.terminate()
            child.join(5)
        reader.close()
        writer.close()
        signal.signal(signal.SIGUSR1, previous)
