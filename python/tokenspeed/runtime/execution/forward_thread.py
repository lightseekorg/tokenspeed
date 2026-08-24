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

"""The per-rank forward thread: the engine's single data plane.

The event loop is the CONTROL plane: ZMQ input, gloo collectives, the C++
scheduler, and commit post-processing. It must never block on the GPU —
a control-plane round is microseconds, so the cross-rank collectives that
keep the redundant schedulers aligned (request broadcast, DP sync) always
find every rank promptly, no matter how deep the GPUs are in queued work.

Everything that touches CUDA is therefore submitted here and runs on ONE
thread per rank, in submission order:

- model forwards (eager launches, graph replays, idle forwards),
- pipeline-stage NCCL recv/send,
- KV page zeroing.

One thread, one ordering: NCCL collectives on a shared communicator are
issued in the same order on every rank because every rank enqueues the same
work sequence (the redundant schedulers plan identically), and a stage's
launch-queue backpressure or a blocking stage recv stalls only this thread,
never the control plane. This is the FluentLLM ForwardThread design point,
generalized to every mode: depth 0/1/pp_size only changes how long a result
future may stay unresolved, not where work runs.

Why a thread and not a process: the forward work reads the executor's device
buffers, CUDA graphs, and NCCL communicators in place. A process would need
its own CUDA context and communicator clones plus an IPC copy of every
batch's metadata — the cost and failure modes of a second engine. The GIL is
not a bottleneck here: the thread spends its time inside CUDA launches and
NCCL waits, which release the GIL.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from concurrent.futures import Future
from typing import Any

import torch


class ForwardThread:
    """Single consumer thread executing submitted GPU work in FIFO order.

    ``submit`` returns a ``concurrent.futures.Future`` resolved with the
    callable's return value (or its exception). The callable runs with the
    thread's CUDA device set; stream discipline stays inside the executor
    (``execute_forward_op`` manages ``execution_stream`` itself).
    """

    def __init__(self, device) -> None:
        # Resolve the CUDA index on the CALLER's thread: an index-less
        # "cuda" means the caller's current device (set_device(local_rank)
        # ran during distributed init), which a fresh thread would not
        # inherit — torch defaults new threads to device 0.
        self._cuda_index: int | None = None
        resolved = torch.device(device)
        if torch.cuda.is_available() and resolved.type == "cuda":
            self._cuda_index = (
                resolved.index
                if resolved.index is not None
                else torch.cuda.current_device()
            )
        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        self._thread = threading.Thread(
            target=self._run, name="tokenspeed::forward", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        if self._cuda_index is not None:
            torch.cuda.set_device(self._cuda_index)
        while True:
            item = self._queue.get()
            if item is None:
                return
            fn, future = item
            if not future.set_running_or_notify_cancel():
                continue
            try:
                future.set_result(fn())
            except BaseException as exc:  # noqa: BLE001 — relayed to the waiter
                future.set_exception(exc)

    def submit(self, fn: Callable[[], Any]) -> Future:
        """Enqueue ``fn`` for FIFO execution; resolve the returned future."""
        future: Future = Future()
        self._queue.put((fn, future))
        return future

    def run(self, fn: Callable[[], Any]) -> Any:
        """Submit ``fn`` and block until it completes (startup/teardown use)."""
        return self.submit(fn).result()

    def shutdown(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=30)
