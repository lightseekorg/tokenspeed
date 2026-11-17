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

The loader makes no CUDA call. While the GPU waits on a ready word, the
forward thread may be inside a CUDA call that holds the driver until the GPU
drains (loading a kernel on first use does); a loader that then needed the
driver could never release the GPU.
"""

from __future__ import annotations

import ctypes
import functools
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from tokenspeed_engine.scheduler import Cache

from tokenspeed.runtime.cache.l2.layerwise_load import LayerwiseLoadTracker
from tokenspeed.runtime.cache.shared_l2.client import SharedL2Client, SharedL2PoolView
from tokenspeed.runtime.cache.shared_l2.layout import SharedL2Layout
from tokenspeed.runtime.execution.forward_step import get_is_capture_mode
from tokenspeed.runtime.utils import get_colorful_logger, get_device_module

logger = get_colorful_logger(__name__)
device_module = get_device_module()

# cuda.h values.
_RDMA_WRITES_ORDERING = 118  # CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING
_ORDERING_OWNER = 100  # CU_GPU_DIRECT_RDMA_WRITES_ORDERING_OWNER
_HOST_ALLOC_FLAGS = 0x3  # CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP
_WAIT_GEQ = 0x0  # CU_STREAM_WAIT_VALUE_GEQ: (int32_t)(*addr - value) >= 0
_WRITE_FENCED = 0x0  # CU_STREAM_WRITE_VALUE_DEFAULT

_WORD_MASK = 0xFFFFFFFF
_START_POLL_S = 20e-6


def _reached(word: int, seq: int) -> bool:
    """Whether a word holds load ``seq``'s value or a later load's."""
    return (word - seq) & _WORD_MASK < 0x80000000


class _ThreadContext(threading.local):
    """Whether the calling thread has made the device's context current."""

    def __init__(self):
        self.current = False


class HostSignals:
    """32-bit words in host memory the GPU maps: streams wait on or write them
    with CUDA stream memory operations, the loader with plain CPU accesses."""

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
        self.words = np.ctypeslib.as_array(
            (ctypes.c_uint32 * num_words).from_address(self._host.value)
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
    """One load's ready words for one tracker's layers."""

    signals: Any
    first_word: int
    seq: int


def _wait_ready_word(words: _ReadyWords, layer_index: int) -> None:
    # A stream memory operation, not a tokenspeed-kernel op: the word lives in
    # host memory the loader writes without CUDA.
    words.signals.wait(
        device_module.current_stream(), words.first_word + layer_index, words.seq
    )


class SharedL2LoadExecutor:
    """Loads proxy Host blocks from the shared L2 pool into Device blocks."""

    def __init__(
        self,
        layout: SharedL2Layout,
        engine: Any,
        *,
        signals: Callable[[int], Any],
        trackers: Sequence[tuple[LayerwiseLoadTracker, int]],
        proxy_page: Callable[[int, int], tuple[int, int, int]],
        load_pool: Callable[[int], SharedL2PoolView | None],
        load_workers: int,
        num_lcm_blocks: int,
        close: Callable[[], None],
    ):
        """
        Args:
            layout: Where the rank's Device blocks and the pool's pages are.
            engine: Reads byte ranges: ``read(session, local, remote,
                lengths)``, raising when a copy fails (``RdmaEngine``).
            signals: Builds the start and ready words for a word count
                (``HostSignals`` on the rank's GPU).
            trackers: Each cache pool view's layer-wise load tracker,
                registered with the view, and its layer count; their layers,
                in order, are the flat layers of the layout's consumers.
            proxy_page: ``(group, proxy Host block)`` to the ``(ticket, bucket,
                page)`` it stands for (``SharedL2Client.proxy_page``).
            load_pool: The pool a load planned from a ticket reads, or None
                once the load may not start any more
                (``SharedL2Client.load_pool``).
            load_workers: Threads reading one load's providers in parallel.
            num_lcm_blocks: LCM blocks of the pool the rank joined at start.
            close: Closes the rank's pool client, once the loads have
                stopped: its writes share the engine with them
                (``SharedL2Client.close``).
        """
        self.layout = layout.transfer
        self._layout = layout
        self._engine = engine
        self._trackers = list(trackers)
        consumers = layout.transfer.consumers
        num_layers = sum(count for _, count in self._trackers)
        if num_layers != len(consumers):
            raise RuntimeError("shared L2 load trackers and cache consumers diverged")
        layer_of = {
            field_id: layer
            for layer, consumer in enumerate(consumers)
            for field_id in consumer
        }
        # Per group, the layer each of its fields feeds, in pool page order.
        self._field_layers = tuple(
            tuple(layer_of[field.field_id] for field in group.fields)
            for group in layout.transfer.groups
        )
        self._num_layers = num_layers
        self._proxy_page = proxy_page
        self._load_pool = load_pool
        self._close = close
        num_sets = len(self._trackers[0][0].event_sets)
        # Per tracker event set: a start word, then a ready word per layer.
        self._signals = signals(num_sets * (num_layers + 1))
        # Carries, per load, a wait for the last layer's ready word and the
        # event recorded behind it.
        self._completion_stream = device_module.Stream()
        self._seq = 0
        self._jobs: list[Future | None] = [None] * num_sets
        # Loads run one after another, each reading its providers in parallel.
        self._loader = ThreadPoolExecutor(1, thread_name_prefix="shared-l2-load")
        self._readers = ThreadPoolExecutor(
            load_workers, thread_name_prefix="shared-l2-read"
        )
        self._done: list[int] = []
        self._done_lock = threading.Lock()
        self._failure: BaseException | None = None
        self._closed = False
        # What the device build reads off a Host cache executor. The pool
        # takes the place of local Host bytes; the proxy tier holds at most
        # what the pool pins, and twice the pool leaves room for its slot
        # fragmentation.
        self.host_storage = None
        self.block_owners = None
        self.num_host_pages = 2 * num_lcm_blocks + 1
        self.host_group_packing = layout.host_packing
        self.num_snapshot_pages = 1
        self.max_retracted_requests = 0

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
        rows: list[tuple[int, int, int, int]] = []
        tickets: set[int] = set()
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
                    for group, source, destination in zip(
                        groups, sources, destinations, strict=True
                    ):
                        # Resolved now: the proxy block may stand for another
                        # page once its ticket is released.
                        ticket, bucket, page = self._proxy_page(group, source)
                        tickets.add(ticket)
                        rows.append((group, bucket, page, destination))
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
        # The words' previous load finished long before its forward did.
        previous = self._jobs[index]
        if previous is not None:
            previous.result()
        self._seq = (self._seq + 1) & _WORD_MASK
        base = index * (self._num_layers + 1)
        self._signals.write(prerequisite_stream, base, self._seq)
        # The whole load as one event, for what waits on no single layer (a
        # graph replay, the next use of these words): recorded behind a wait
        # for the last layer's word, on this thread, so the loader still makes
        # no CUDA call. An event never recorded would pass at once.
        self._signals.wait(self._completion_stream, base + self._num_layers, self._seq)
        finish = device_module.Event()
        finish.record(self._completion_stream)
        word = base + 1
        for tracker, count in self._trackers:
            events = tracker.event_sets[index]
            events.layer_ready_flags = _ReadyWords(self._signals, word, self._seq)
            events.wait_layer_ready = _wait_ready_word
            events.set_completion(finish)
            word += count
            tracker.set_consumers(index)
        self._jobs[index] = self._loader.submit(
            self._load, op_ids, rows, tickets, base, self._seq
        )

    def _layer_reads(
        self, pool: SharedL2PoolView, rows: Sequence[tuple[int, int, int, int]]
    ) -> list[dict[str, tuple[list, list, list]]]:
        """Per layer, per provider session: local and remote addresses and
        lengths of every field a load copies."""
        by_target: dict[tuple[int, str], tuple[list[int], list[int]]] = {}
        for group, bucket, page, block in rows:
            session, address = self._layout.page_address(pool, group, bucket, page)
            remotes, blocks = by_target.setdefault((group, session), ([], []))
            remotes.append(address)
            blocks.append(block)
        layers: list[dict[str, tuple[list, list, list]]] = [
            {} for _ in range(self._num_layers)
        ]
        for (group, session), (remotes, blocks) in by_target.items():
            remote = np.asarray(remotes, dtype=np.int64)
            block = np.asarray(blocks, dtype=np.int64)
            for (address, stride, size), offset, layer in zip(
                self._layout.device_fields(group),
                self._layout.field_offsets[group],
                self._field_layers[group],
                strict=True,
            ):
                local_list, remote_list, lengths = layers[layer].setdefault(
                    session, ([], [], [])
                )
                local_list.extend((address + block * stride).tolist())
                remote_list.extend((remote + offset).tolist())
                lengths.extend([size] * len(blocks))
        return layers

    def _load(self, op_ids, rows, tickets, base: int, seq: int) -> None:
        words = self._signals.words
        try:
            # The start word lands once the plan's page zeroing is done.
            while not _reached(int(words[base]), seq):
                if self._closed:
                    raise RuntimeError("the executor shut down before the load")
                time.sleep(_START_POLL_S)
            # Past the lease's start bound the owner may have dropped the pins
            # of a rank it has not heard: the pages could then hold another
            # prompt's bytes. The tickets of one plan are of one pool.
            pools = {self._load_pool(ticket) for ticket in tickets}
            if None in pools or len(pools) > 1:
                raise RuntimeError(
                    "the rank's lease on the pool pages ran out before the load"
                )
            # A load without a row names no ticket and reads nothing.
            pool = pools.pop() if pools else None
            for layer, reads in enumerate(self._layer_reads(pool, rows)):
                pending = [
                    self._readers.submit(self._engine.read, session, *read)
                    for session, read in reads.items()
                ]
                for future in pending:
                    future.result()
                words[base + 1 + layer] = seq
        except BaseException as exc:
            # A failed load stops the engine; the forward waiting on it is
            # released so the stop is not a hang. The failure is recorded
            # before the release: the commit of that forward polls once the
            # forward has landed, finds it and returns no result.
            logger.exception("shared L2 load failed")
            self._failure = exc
            words[base + 1 : base + 1 + self._num_layers] = seq
            return
        with self._done_lock:
            self._done.extend(op_ids)

    def poll_results(self) -> list:
        """The completed loads' ACKs; a failed load raises here."""
        if self._failure is not None:
            raise RuntimeError("shared L2 load failed") from self._failure
        with self._done_lock:
            done, self._done = self._done, []
        return [Cache.LoadBackDoneEvent(op_id) for op_id in done]

    def shutdown(self) -> None:
        self._closed = True
        self._loader.shutdown(wait=True)
        self._readers.shutdown(wait=True)
        device_module.synchronize()
        self._signals.close()
        self._close()


def build_shared_l2_load_executor(
    client: SharedL2Client,
    layout: SharedL2Layout,
    engine: Any,
    *,
    device_pool: Any,
    draft_pool: Any | None,
    gpu_id: int,
) -> SharedL2LoadExecutor:
    """Arm the cache pools' layer-wise load trackers, as the Host cache
    executor does, and build the executor over them.

    Args:
        client: The rank's shared L2 client, joined to the pool.
        layout: The rank's shared L2 layout.
        engine: The rank's RDMA engine, which the client's transfer writes
            with.
        device_pool: The target cache pool.
        draft_pool: The draft pool sharing the arena, or None.
        gpu_id: Local device index of the rank's GPU, the one its RDMA
            devices are named for.
    """
    trackers = []
    for pool, num_layers in zip(
        (device_pool, draft_pool), layout.view_layers, strict=False
    ):
        tracker = LayerwiseLoadTracker(num_layers)
        pool.register_layerwise_load_tracker(tracker)
        trackers.append((tracker, num_layers))
    return SharedL2LoadExecutor(
        layout,
        engine,
        signals=functools.partial(HostSignals, gpu_id),
        trackers=trackers,
        proxy_page=client.proxy_page,
        load_pool=client.load_pool,
        load_workers=client.config.load_workers,
        num_lcm_blocks=client.pool.num_lcm_blocks,
        close=client.close,
    )
