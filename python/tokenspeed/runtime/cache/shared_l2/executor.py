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

"""The Host cache executor of a rank whose Host L2 is a shared L2 pool.

The scheduler's Host tier is a proxy of the pool: a load-back names proxy
Host blocks, each standing for a pool page. A load copies those pages by RDMA
layer by layer, like a Host L2 load: once every provider delivered a layer,
the loader sets the layer's ready word in host memory, and the forward's
stream waits on that word before the layer reads its cache. Write-backs never
occur: the rank's pool client writes newly published blocks.

This executor is the forward thread's part of a load: it hands the load to
the engine host, whose loader thread reads the pages (``shared_l2/load.rs``
of ``ts-control``), and puts the load's waits on the streams. The host
acknowledges a finished load to the scheduler itself, from its loader
(``PythonDataPlane::poll_cache_events`` in ``ts-host``): no acknowledgment
returns through this executor.

The loader makes no CUDA call. While the GPU waits on a ready word, the
forward thread may be inside a CUDA call that holds the driver until the GPU
drains (loading a kernel on first use does); a loader that then needed the
driver could never release the GPU.
"""

from __future__ import annotations

import ctypes
import functools
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from tokenspeed_engine.scheduler import Cache

from tokenspeed.runtime.cache.l2.layerwise_load import LayerwiseLoadTracker
from tokenspeed.runtime.cache.shared_l2.client import SharedL2PoolView
from tokenspeed.runtime.cache.shared_l2.layout import SharedL2Layout
from tokenspeed.runtime.execution.forward_step import get_is_capture_mode
from tokenspeed.runtime.utils import get_device_module

device_module = get_device_module()

# cuda.h values.
_RDMA_WRITES_ORDERING = 118  # CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING
_ORDERING_OWNER = 100  # CU_GPU_DIRECT_RDMA_WRITES_ORDERING_OWNER
_HOST_ALLOC_FLAGS = 0x3  # CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP
_WAIT_GEQ = 0x0  # CU_STREAM_WAIT_VALUE_GEQ: (int32_t)(*addr - value) >= 0
_WRITE_FENCED = 0x0  # CU_STREAM_WRITE_VALUE_DEFAULT

_WORD_MASK = 0xFFFFFFFF


class _ThreadContext(threading.local):
    """Whether the calling thread has made the device's context current."""

    def __init__(self):
        self.current = False


class HostSignals:
    """A rank's signal words, the start and ready words of its loads: 32-bit
    words in host memory the GPU maps. Streams wait on or write them with
    CUDA stream memory operations, the loader with plain CPU accesses."""

    def __init__(self, device_index: int, num_words: int):
        cuda = ctypes.CDLL("libcuda.so.1")
        cuda.cuMemHostAlloc.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
            ctypes.c_uint,
        ]
        cuda.cuMemHostGetDevicePointer_v2.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_void_p,
            ctypes.c_uint,
        ]
        cuda.cuStreamWaitValue32_v2.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_uint32,
            ctypes.c_uint,
        ]
        cuda.cuStreamWriteValue32_v2.argtypes = cuda.cuStreamWaitValue32_v2.argtypes
        self._cuda = cuda
        device = ctypes.c_int()
        self._check(cuda.cuDeviceGet(ctypes.byref(device), device_index))
        ordering = ctypes.c_int()
        self._check(
            cuda.cuDeviceGetAttribute(
                ctypes.byref(ordering), _RDMA_WRITES_ORDERING, device
            )
        )
        # A layer is released as soon as its RDMA reads complete, without
        # cuFlushGPUDirectRDMAWrites (a CUDA call the loader may not make).
        if ordering.value < _ORDERING_OWNER:
            raise RuntimeError(
                "shared L2 needs GPUDirect RDMA writes ordered for their GPU"
            )
        self._context = ctypes.c_void_p()
        self._check(cuda.cuDevicePrimaryCtxRetain(ctypes.byref(self._context), device))
        self._local = _ThreadContext()
        self._make_current()
        self._host = ctypes.c_void_p()
        self._check(
            cuda.cuMemHostAlloc(
                ctypes.byref(self._host), 4 * num_words, _HOST_ALLOC_FLAGS
            )
        )
        #: Where the words are in host memory, for the loader.
        self.address = int(self._host.value)
        self.words = np.ctypeslib.as_array(
            (ctypes.c_uint32 * num_words).from_address(self.address)
        )
        self.words[:] = 0
        device_address = ctypes.c_uint64()
        self._check(
            cuda.cuMemHostGetDevicePointer_v2(
                ctypes.byref(device_address), self._host, 0
            )
        )
        self._device_address = device_address.value
        # Not every device and driver configuration runs stream memory
        # operations. One write and one wait of the first word, on the kinds
        # of stream a load uses, so that a rank that cannot run them fails at
        # start and not on its first load.
        waiter = torch.cuda.Stream(device_index)
        try:
            self.write(torch.cuda.default_stream(device_index), 0, 1)
            self.wait(waiter, 0, 1)
        except RuntimeError as exc:
            raise RuntimeError("shared L2 needs CUDA stream memory operations") from exc
        waiter.synchronize()
        self.words[0] = 0

    def wait(self, stream: torch.cuda.Stream, word: int, value: int) -> None:
        """Hold ``stream`` until ``word`` reaches ``value``."""
        self._make_current()
        self._check(
            self._cuda.cuStreamWaitValue32_v2(
                stream.cuda_stream, self._device_address + 4 * word, value, _WAIT_GEQ
            )
        )

    def write(self, stream: torch.cuda.Stream, word: int, value: int) -> None:
        """Set ``word`` to ``value`` once ``stream``'s prior work is done."""
        self._make_current()
        self._check(
            self._cuda.cuStreamWriteValue32_v2(
                stream.cuda_stream,
                self._device_address + 4 * word,
                value,
                _WRITE_FENCED,
            )
        )

    def close(self) -> None:
        self._make_current()
        self._check(self._cuda.cuMemFreeHost(self._host))

    def _make_current(self) -> None:
        if not self._local.current:
            self._check(self._cuda.cuCtxSetCurrent(self._context))
            self._local.current = True

    @staticmethod
    def _check(result: int) -> None:
        if result != 0:
            raise RuntimeError(f"CUDA driver call failed with error {result}")


@dataclass(frozen=True, slots=True)
class _ReadyWords:
    """One load's ready words for one tracker's layers, from
    ``first_ready_word`` on."""

    signals: Any
    first_ready_word: int
    seq: int


def _wait_ready_word(words: _ReadyWords, layer_index: int) -> None:
    # A stream memory operation, not a tokenspeed-kernel op: the word lives in
    # host memory the loader writes without CUDA.
    words.signals.wait(
        device_module.current_stream(),
        words.first_ready_word + layer_index,
        words.seq,
    )


class SharedL2LoadExecutor:
    """Hands the loads of proxy Host blocks from the shared L2 pool into
    Device blocks to the host's loader."""

    def __init__(
        self,
        layout: SharedL2Layout,
        shared_l2_handle: Any,
        *,
        signals: Callable[[int], Any],
        trackers: Sequence[tuple[LayerwiseLoadTracker, int]],
        pool: SharedL2PoolView,
    ):
        """
        Args:
            layout: Where the rank's Device blocks are.
            shared_l2_handle: The host's shared L2 handle (``SharedL2Handle``),
                which runs the loads and the rest of the rank's pool client.
                ``submit_load(op_ids=, rows=, start_word=, seq=)`` queues a
                load of ``(group, proxy Host block, Device block)`` rows
                without waiting; ``close()`` stops the client's threads and
                the transfer engine and returns whether every thread ended.
            signals: Builds the signal words for a word count
                (``HostSignals`` on the rank's GPU).
            trackers: Each cache pool view's layer-wise load tracker,
                registered with the view, and its layer count; their layers,
                in order, are the flat layers of the layout's consumers.
            pool: The pool the rank joined (``client.join_pool``).
        """
        self.layout = layout.transfer
        self._shared_l2_handle = shared_l2_handle
        self._trackers = list(trackers)
        num_layers = sum(count for _, count in self._trackers)
        if num_layers != len(layout.transfer.consumers):
            raise RuntimeError("shared L2 load trackers and cache consumers diverged")
        self._num_layers = num_layers
        num_sets = len(self._trackers[0][0].event_sets)
        # Per tracker event set: a start word, then a ready word per layer.
        self._signals = signals(num_sets * (num_layers + 1))
        # Carries, per load, a wait for the last layer's ready word and the
        # event recorded behind it.
        self._completion_stream = device_module.Stream()
        self._seq = 0
        # What the device build reads off a Host cache executor. The pool
        # takes the place of local Host bytes; the proxy tier holds at most
        # what the pool pins, and twice the pool leaves room for its slot
        # fragmentation.
        self.host_storage = None
        self.block_owners = None
        self.num_host_pages = 2 * pool.num_lcm_blocks + 1
        self.host_group_packing = layout.host_packing
        self.num_snapshot_pages = 1
        self.max_retracted_requests = 0

    @property
    def signal_words(self) -> tuple[int, int]:
        """Where the loads' signal words are in host memory, and how many
        there are: what the host's loader reads and sets."""
        return self._signals.address, len(self._signals.words)

    def submit_write_backs(
        self, cache_ops: Sequence, *, prerequisite_stream, fence_stream
    ) -> None:
        """A shared L2 rank has no write-backs or snapshot stores."""
        if any(isinstance(op, Cache.WriteBackOp) for op in cache_ops):
            raise TypeError("a shared L2 rank writes the pool, not a Host L2")

    def submit_load_backs(self, cache_ops: Sequence, *, prerequisite_stream) -> None:
        """Start the plan's loads; runs after the plan's page zeroing.

        The device handle calls it on every plan: it also points the
        layer-wise load trackers at this plan's load, or at none.

        Args:
            cache_ops: The round's cache ops; the ``Cache.LoadBackOp``
                batches are read here.
            prerequisite_stream: The stream the plan's page zeroing ran on;
                the RDMA reads start once it reaches this point.
        """
        op_ids: list[int] = []
        rows: list[tuple[int, int, int]] = []
        for operation in cache_ops:
            if isinstance(operation, Cache.LoadBackOp):
                for op_id, groups, sources, destinations in zip(
                    operation.op_ids,
                    operation.group_ids,
                    operation.src_pages,
                    operation.dst_pages,
                    strict=True,
                ):
                    op_ids.append(op_id)
                    rows.extend(zip(groups, sources, destinations, strict=True))
            elif not isinstance(operation, Cache.WriteBackOp):
                raise TypeError(
                    f"a shared L2 rank does not run {type(operation).__name__}"
                )
        if not op_ids:
            for tracker, _ in self._trackers:
                tracker.set_consumers(-1)
            return
        if get_is_capture_mode():
            raise RuntimeError("shared L2 loads must run outside CUDA Graph capture")
        index = None
        for tracker, _ in self._trackers:
            current = tracker.begin_load()
            if index is not None and current != index:
                raise RuntimeError("target and draft load trackers diverged")
            index = current
        # The words' previous load is over by now: the loader runs the loads
        # in the order they are handed over, and that one finished long
        # before its forward did.
        self._seq = (self._seq + 1) & _WORD_MASK
        start_word = index * (self._num_layers + 1)
        # Handed over before any wait goes on a stream: a load the loader
        # refuses (a proxy block that stands for no page) leaves no stream
        # waiting on a word nobody sets. The loader resolves each proxy block
        # to its pool page now, as the block may stand for another page once
        # its ticket is released, and reads nothing before the start word.
        self._shared_l2_handle.submit_load(
            op_ids=op_ids, rows=rows, start_word=start_word, seq=self._seq
        )
        self._signals.write(prerequisite_stream, start_word, self._seq)
        # The whole load as one event, for what waits on no single layer (a
        # graph replay, the next use of these words): recorded behind a wait
        # for the last layer's word, on this thread, so the loader still makes
        # no CUDA call. An event never recorded would pass at once.
        self._signals.wait(
            self._completion_stream, start_word + self._num_layers, self._seq
        )
        finish = device_module.Event()
        finish.record(self._completion_stream)
        first_ready_word = start_word + 1
        for tracker, count in self._trackers:
            events = tracker.event_sets[index]
            events.layer_ready_flags = _ReadyWords(
                self._signals, first_ready_word, self._seq
            )
            events.wait_layer_ready = _wait_ready_word
            events.set_completion(finish)
            first_ready_word += count
            tracker.set_consumers(index)

    def shutdown(self) -> None:
        # The client's threads end first: a load in flight sets ready words,
        # and the pool's writes read the Device cache.
        threads_ended = self._shared_l2_handle.close()
        if not threads_ended:
            # A thread still runs: it may set a word later, and a stream may
            # wait on a word that nobody sets any more. The streams and the
            # words stay as they are, for the process's exit to take.
            return
        device_module.synchronize()
        self._signals.close()


def build_shared_l2_load_executor(
    layout: SharedL2Layout,
    shared_l2_handle: Any,
    *,
    pool: SharedL2PoolView,
    device_pool: Any,
    draft_pool: Any | None,
    gpu_id: int,
) -> SharedL2LoadExecutor:
    """Arm the cache pools' layer-wise load trackers, as the Host cache
    executor does, and build the executor over them.

    Args:
        layout: The rank's shared L2 layout.
        shared_l2_handle: The host's shared L2 handle, which runs the loads.
        pool: The pool the rank joined.
        device_pool: The target cache pool.
        draft_pool: The draft pool sharing the arena, or None.
        gpu_id: Local device index of the rank's GPU, the one its RDMA
            devices are named for.
    """
    trackers = []
    for cache_pool, num_layers in zip(
        (device_pool, draft_pool), layout.view_layers, strict=False
    ):
        tracker = LayerwiseLoadTracker(num_layers)
        cache_pool.register_layerwise_load_tracker(tracker)
        trackers.append((tracker, num_layers))
    return SharedL2LoadExecutor(
        layout,
        shared_l2_handle,
        signals=functools.partial(HostSignals, gpu_id),
        trackers=trackers,
        pool=pool,
    )
