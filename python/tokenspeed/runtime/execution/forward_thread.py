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
scheduler, and commit post-processing. Dispatching a round never waits on
the GPU — it submits and moves on, and only commit joins — so a round is
microseconds and the cross-rank collectives that keep the redundant
schedulers aligned (request broadcast, DP sync) always find every rank
promptly, no matter how deep the GPUs are in queued work. (``DeviceHandle``
does have blocking ``run_*`` operations, but only on rounds that dispatch
nothing — the DP idle forward — or off the round path entirely.)

Everything that touches CUDA is therefore submitted here and runs on ONE
thread per rank, in submission order:

- model forwards (eager launches, graph replays, idle forwards),
- pipeline-stage NCCL recv/send,
- KV page zeroing, and KV buffer repair after a memory-saver wake,
- the PD receive trigger and the writes a completed remote prefill lands,
- in-place weight updates from an RL trainer.

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

The capture contract
--------------------

Sharing an address space is a convenience, not a licence: the two planes
exchange information ONLY through the two channels below, so that the day
this becomes a process the interface is already the whole interface.

1. Outbound is the closure, and only the closure. The control plane hands
   work over as ``submit(fn)`` / ``run(fn)`` arguments captured in ``fn``.
   Passing something by parking it on a shared object — a field on the
   executor, on an attention backend, on a request state — is not a channel;
   it is a race waiting for a deep enough queue.

2. Captured means frozen. Submission is fire-and-forget: once ``fn`` is
   queued, the control plane must not mutate what it captured — no attribute
   rebinding, no in-place edit, no releasing a resource the object holds.
   Capture CPU-plain values, or a snapshot the control plane will only read
   afterwards. Bind at capture time (``functools.partial``, or a default
   argument) rather than closing over a variable the caller will rebind — a
   bare lambda reads it at execution time, possibly a round later.

3. Inbound is the future, and only the future. Results reach the control
   plane through ``PendingExecution.result()``; nothing else crosses back.

4. Device-side state is not reachable from the control plane at all. The
   execution stream, ``runtime_states``, ``input_buffers``, and every
   attention backend's forward metadata live behind ``DeviceHandle``
   (``device.py``), which is what the event loop holds instead of the
   executor and which hands back no device object at all. Rules 1 and 2 are
   enforced by that shape rather than by anyone remembering them: what the
   loop cannot see, it cannot pass implicitly or mutate afterwards.

5. Two registered exceptions — and new ones belong in this list, not in a
   comment at the call site:

   - Grammar matchers. ``GrammarStepInputs.grammars`` holds the control
     plane's live matcher objects, and ownership is split by path. Under
     capturable grammar the side-stream hostfunc advances them and the
     commit path deliberately does not; under eager grammar the data plane
     reads them during the fill and the commit path advances them, so the
     loop drains its in-flight queue before dispatching a grammar batch
     (``EventLoop._dispatch_depends_on_pending_commit`` is the registry of
     those drain rules). Copying a matcher per round is not free, which is
     why this stays an exception rather than becoming rule 2.

   - Multimodal items. The gather's snapshot is shallow on purpose: the
     ``MultimodalDataItem`` objects stay shared, and the data plane WRITES
     through them mid-forward — the embedder fills ``item.encoded`` and
     drops the raw ``feature``/``feature_shm`` it no longer needs, and EPD
     publishes ``encoded`` via the handle's embedding slot. The discipline
     is ordering, not freezing: mutations happen on the FIFO (inside the
     forward, or a submitted closure), the control plane only reads them at
     commit time or later, and the release of the shared SHM handles rides
     the FIFO too (``run_multimodal_work`` with ``wait=False``) so it lands
     behind any forward still reading them. What the shallow copy does
     freeze is the OUTER struct — ``mrope_positions`` and its siblings,
     which the control plane rebinds every round.
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
        """Enqueue ``fn`` for FIFO execution; resolve the returned future.

        ``fn``'s captures are the whole interface (see the capture contract
        above): everything it needs must be bound at capture time, and the
        caller must not mutate any of it afterwards.

        Args:
            fn: Zero-argument callable to run on the data plane.

        Returns:
            A future resolved with ``fn``'s return value, or its exception.
        """
        future: Future = Future()
        self._queue.put((fn, future))
        return future

    def run(self, fn: Callable[[], Any]) -> Any:
        """Submit ``fn`` and block until it completes.

        For the rare submissions whose caller genuinely needs the outcome
        before it can continue — startup and teardown, and the low-rate PD
        events that must see their device write land. The per-round forward
        path uses ``submit`` and waits only at commit.

        Args:
            fn: Zero-argument callable to run on the data plane.

        Returns:
            ``fn``'s return value; its exception is re-raised here.
        """
        return self.submit(fn).result()

    def shutdown(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=30)
