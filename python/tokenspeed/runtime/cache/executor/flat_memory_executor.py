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

"""Flat host-tier executor (M15 Phase D2): drives FlatWriteBack/FlatLoadBack
page-id pairs against the byte-blind :class:`FlatHostMirror`, replacing the
radix ``MemoryExecutor`` when serving with a flat-built scheduler ext and the
kvstore enabled. Unlike the radix host executor it ACKS loadbacks: the flat
C++ scheduler pins source host pages and destination device blocks until a
``Cache.LoadBackDoneEvent`` retires the op.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Sequence

import psutil
import torch
from tokenspeed_scheduler import Cache

from tokenspeed.runtime.cache.executor.host_executor import (
    _Ack,
    _cache_stream_priorities,
    _new_cache_stream,
    _ordered_unique,
)
from tokenspeed.runtime.cache.flat_host_mirror import (
    FlatHostMirror,
    flat_bytes_per_host_page,
)
from tokenspeed.runtime.cache.kvstore_controller import LayerDoneCounter
from tokenspeed.runtime.cache.transfer.types import CacheKind
from tokenspeed.runtime.execution.cuda_graph_wrapper import get_is_capture_mode
from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)

_HOST_MEM_HEADROOM_BYTES = 10 * (1024**3)


def flat_num_host_pages(
    *,
    bytes_per_host_page: int,
    device_pool_size: int,
    page_size: int,
    host_ratio: float,
    host_size_gb: float,
) -> int:
    """Host page budget from the kvstore sizing knobs (same knobs the radix
    ``HostKVCache`` resolves, kv_cache_host.py:91-102, budget arithmetic only):

    - ``host_size_gb > 0``: explicit byte budget, floor to whole mirror pages
      (never exceeds the requested bytes):
      ``host_size_gb * 1e9 // bytes_per_host_page``.
    - otherwise ratio sizing, mirroring the radix token->page align-up:
      ``int(device_pool_size * host_ratio) // page_size + 1``.
    """
    if bytes_per_host_page <= 0:
        raise ValueError(f"bytes_per_host_page must be > 0, got {bytes_per_host_page}")
    if page_size <= 0:
        raise ValueError(f"page_size must be > 0, got {page_size}")
    if host_size_gb > 0:
        num_pages = int(host_size_gb * 1e9 // bytes_per_host_page)
    else:
        num_pages = int(device_pool_size * host_ratio) // page_size + 1
    if num_pages <= 0:
        raise ValueError(
            "flat host tier resolved to zero host pages "
            f"(host_size_gb={host_size_gb}, host_ratio={host_ratio}, "
            f"bytes_per_host_page={bytes_per_host_page}); increase the "
            "kvstore size."
        )
    return num_pages


class FlatMemoryExecutor:
    """Slim replacement for ``MemoryExecutor`` under the flat host tier.

    Exposes the exact surface ``EventLoop`` drives: ``submit_plan`` /
    ``poll_results`` / ``get_producer_index`` / ``set_consumer`` (plus the
    ``host_exec.pools`` attribute walk in ``_setup_layerwise_loadback``).
    No host pool, no storage executor, no mamba: the flat scheduler config
    validation already rejects those setups.
    """

    # EventLoop keys per-op inflight accounting off this: flat loadbacks are
    # acknowledged with one LoadBackDoneEvent per op.
    emits_loadback_acks = True

    def __init__(self, device_pool, *, host_ratio: float, host_size_gb: float):
        self.page_size = int(device_pool.page_size)
        self.layer_num = len(device_pool.k_buffer)

        bytes_per_host_page = flat_bytes_per_host_page(device_pool)
        num_host_pages = flat_num_host_pages(
            bytes_per_host_page=bytes_per_host_page,
            device_pool_size=int(device_pool.size),
            page_size=self.page_size,
            host_ratio=host_ratio,
            host_size_gb=host_size_gb,
        )
        requested_bytes = num_host_pages * bytes_per_host_page
        available_bytes = psutil.virtual_memory().available - _HOST_MEM_HEADROOM_BYTES
        if requested_bytes > available_bytes:
            raise ValueError(
                f"Not enough host memory for the flat host tier. Requesting "
                f"{requested_bytes / 1e9:.2f} GB but only have "
                f"{available_bytes / 1e9:.2f} GB free. Please reduce the "
                f"size of the KVStore."
            )
        logger.info(
            "Allocating %.2f GB pinned host memory for the flat host tier "
            "(num_host_pages=%s bytes_per_host_page=%s host_size_gb=%r "
            "host_ratio=%r device_pool.size=%r)",
            requested_bytes / 1e9,
            num_host_pages,
            bytes_per_host_page,
            host_size_gb,
            host_ratio,
            device_pool.size,
        )
        self.mirror = FlatHostMirror(device_pool, num_host_pages)
        self.num_host_pages = num_host_pages

        # Layerwise loadback fencing: register the counter where the radix
        # KVCachePool would, so pool.get_key_buffer/get_value_buffer gate on
        # the same wait_until(layer_id) machinery.
        self._counter = LayerDoneCounter(self.layer_num)
        device_pool.register_layer_transfer_counter(self._counter)
        # _start_loading maps layer -> mirror V-tensor event and relies on
        # load_events[-1] (LayerLoadingEvent.finish_event, reuse fence in
        # update_producer) covering EVERY copy: it pins the last layer to
        # the op's last per-tensor event, which without state slabs is the
        # last layer's V event only if that V mirror is the last KV tensor
        # pair. Holds for both layouts (legacy: identity; slab: last layer
        # is the last occurrence of its group). A state last layer has no
        # KV mirror at all -- its copies (state slabs trail every KV
        # tensor) are covered by the events[-1] pin in _start_loading.
        assert (
            self.mirror.state_tensor_indices_of_layer(self.layer_num - 1) is not None
            or self.mirror.tensor_index_of_layer(self.layer_num - 1)
            == self.mirror.num_k_tensors - 1
        ), "flat host tier: last layer's V mirror is not the last KV tensor pair"

        write_priority, load_priority = _cache_stream_priorities()
        self.write_stream = _new_cache_stream(write_priority)
        self.load_stream = _new_cache_stream(load_priority)

        # (device_page, host_page) pairs staged between submit() and flush().
        self._pending_write_pairs: list[tuple[int, int]] = []
        self._pending_write_op_ids: list[int] = []
        self._pending_load_pairs: list[tuple[int, int]] = []
        self._pending_load_op_ids: list[int] = []
        self.ack_write_queue: list[_Ack] = []
        self.ack_load_queue: list[_Ack] = []
        # Ops whose page lists were empty on the wire (C++ dedups transfers
        # across ops of one batched operation) and no batch event covers them.
        self._immediate_write_op_ids: list[int] = []
        self._immediate_load_op_ids: list[int] = []

        self._producer_map: OrderedDict[int, int] = OrderedDict()
        self._producer_map_limit = 1024

        # Surface for EventLoop._setup_layerwise_loadback, which walks
        # memory_executor.host_exec.pools to enumerate fencing kinds.
        self.host_exec = self
        self.pools = {CacheKind.KV: self.mirror}

    # ------------------------------------------------------------------
    # Submission (wire shape: batched Flat{WriteBack,LoadBack}Operation)
    # ------------------------------------------------------------------

    def submit_plan(self, plan) -> None:
        if plan.cache:
            logger.debug("[cache_op] flat submit_plan: %s cache ops", len(plan.cache))
        for op in plan.cache:
            self.submit(op)
        self.flush()

    def submit(self, op) -> None:
        if isinstance(op, Cache.WriteBackOp):
            self.submit_writeback(op.op_ids, op.src_pages, op.dst_pages)
        elif isinstance(op, Cache.LoadBackOp):
            self.submit_loadback(op.op_ids, op.src_pages, op.dst_pages)
        else:
            raise ValueError(
                f"flat host tier: unsupported cache op kind {type(op).__name__}"
            )

    def _submit(
        self,
        op_ids: Sequence[int],
        src_pages: Sequence[Sequence[int]],
        dst_pages: Sequence[Sequence[int]],
        *,
        pending_op_ids: list[int],
        pending_pairs: list[tuple[int, int]],
        src_is_device: bool,
    ) -> None:
        """Stage copies as (device_page, host_page) pairs; fail loud on a
        ragged wire payload instead of silently dropping trailing ops."""
        assert len(op_ids) == len(src_pages) == len(dst_pages), (
            f"flat host tier: ragged cache-op payload (op_ids={len(op_ids)}, "
            f"src_pages={len(src_pages)}, dst_pages={len(dst_pages)})"
        )
        for op_id, src, dst in zip(op_ids, src_pages, dst_pages):
            assert len(src) == len(dst), (
                f"flat host tier: op {op_id} src/dst page lists differ "
                f"({len(src)} vs {len(dst)})"
            )
            pending_op_ids.append(int(op_id))
            device_pages, host_pages = (src, dst) if src_is_device else (dst, src)
            pending_pairs.extend(
                (int(d), int(h)) for d, h in zip(device_pages, host_pages)
            )

    def submit_writeback(
        self,
        op_ids: Sequence[int],
        src_pages: Sequence[Sequence[int]],
        dst_pages: Sequence[Sequence[int]],
    ) -> None:
        """Stage device->host copies: src=device pages, dst=host pages."""
        self._submit(
            op_ids,
            src_pages,
            dst_pages,
            pending_op_ids=self._pending_write_op_ids,
            pending_pairs=self._pending_write_pairs,
            src_is_device=True,
        )

    def submit_loadback(
        self,
        op_ids: Sequence[int],
        src_pages: Sequence[Sequence[int]],
        dst_pages: Sequence[Sequence[int]],
    ) -> None:
        """Stage host->device copies: src=host pages, dst=device pages."""
        self._submit(
            op_ids,
            src_pages,
            dst_pages,
            pending_op_ids=self._pending_load_op_ids,
            pending_pairs=self._pending_load_pairs,
            src_is_device=False,
        )

    def flush(self) -> None:
        self._start_loading()
        self._start_writing()

    def _start_writing(self) -> None:
        if not self._pending_write_op_ids:
            return
        op_ids = _ordered_unique(self._pending_write_op_ids)
        pairs = self._pending_write_pairs
        self._pending_write_op_ids = []
        self._pending_write_pairs = []
        if not pairs:
            self._immediate_write_op_ids.extend(op_ids)
            return
        # Order the D2H copies after already-enqueued default-stream work
        # (same fence the radix _start_writing places).
        start_event = torch.cuda.Event()
        start_event.record()
        start_event.wait(self.write_stream)
        self.mirror.store_pages(pairs, self.write_stream)
        finish_event = torch.cuda.Event()
        finish_event.record(self.write_stream)
        self.ack_write_queue.append(_Ack(finish_event, op_ids))

    def _start_loading(self) -> None:
        if not self._pending_load_op_ids:
            return
        assert (
            not get_is_capture_mode()
        ), "cache loadback must run in eager admission iter"
        op_ids = _ordered_unique(self._pending_load_op_ids)
        pairs = self._pending_load_pairs
        self._pending_load_op_ids = []
        self._pending_load_pairs = []
        if not pairs:
            self._immediate_load_op_ids.extend(op_ids)
            return

        producer_id = self._counter.update_producer()
        producer_event = self._counter.events[producer_id]
        producer_event.start_event.record()
        producer_event.start_event.wait(self.load_stream)

        events = self.mirror.load_pages_with_events(pairs, self.load_stream)
        # Layer fence: layer L is readable once its V mirror copy lands; the
        # load stream is serial (all K copies precede all V copies), so the
        # V-tensor event also covers L's K copy. Paired slab layers share the
        # slab event -- correct by design. State layers instead fence on
        # their ssm event: conv precedes ssm in tensor_pairs order (and both
        # follow every KV tensor), so on the serial stream the ssm event
        # covers the conv copy and the layer's KV copies -- the same
        # K-before-V reasoning as above.
        num_k = self.mirror.num_k_tensors
        for layer_id in range(self.layer_num):
            state_indices = self.mirror.state_tensor_indices_of_layer(layer_id)
            if state_indices is not None:
                producer_event.load_events[layer_id] = events[state_indices[1]]
            else:
                producer_event.load_events[layer_id] = events[
                    num_k + self.mirror.tensor_index_of_layer(layer_id)
                ]
        # finish_event (== load_events[-1]) is the producer-slot reuse fence
        # in update_producer and must cover EVERY copy of the op; state
        # tensors copy after all KV tensors, so pin the last layer to the
        # op's last per-tensor event. A no-op without state slabs: events[-1]
        # is then the last layer's V event (ctor assert).
        producer_event.load_events[self.layer_num - 1] = events[-1]
        # events[-1] is also the reassigned finish_event, so the ack covers
        # every copy.
        self.ack_load_queue.append(_Ack(events[-1], op_ids))
        for op_id in op_ids:
            self._producer_map[op_id] = producer_id
        while len(self._producer_map) > self._producer_map_limit:
            self._producer_map.popitem(last=False)

    # ------------------------------------------------------------------
    # Ack draining
    # ------------------------------------------------------------------

    def poll_results(self) -> list:
        results: list = []
        for op_id in self._immediate_write_op_ids:
            results.append(self._write_done(op_id))
        self._immediate_write_op_ids.clear()
        for op_id in self._immediate_load_op_ids:
            results.append(self._load_done(op_id))
        self._immediate_load_op_ids.clear()

        remaining_writes = []
        for ack in self.ack_write_queue:
            if ack.finish_event.query():
                results.extend(self._write_done(op_id) for op_id in ack.op_ids)
            else:
                remaining_writes.append(ack)
        self.ack_write_queue[:] = remaining_writes

        remaining_loads = []
        for ack in self.ack_load_queue:
            if ack.finish_event.query():
                results.extend(self._load_done(op_id) for op_id in ack.op_ids)
            else:
                remaining_loads.append(ack)
        self.ack_load_queue[:] = remaining_loads

        if results:
            for r in results:
                logger.debug(
                    "[cache_op] flat done op_id=%s success=%s type=%s",
                    r.op_id,
                    r.success,
                    type(r).__name__,
                )
        return results

    @staticmethod
    def _write_done(op_id: int):
        evt = Cache.WriteBackDoneEvent()
        evt.op_id = op_id
        evt.success = True
        return evt

    @staticmethod
    def _load_done(op_id: int):
        evt = Cache.LoadBackDoneEvent()
        evt.op_id = op_id
        evt.success = True
        return evt

    # ------------------------------------------------------------------
    # Layerwise loadback fencing (EventLoop._setup_layerwise_loadback)
    # ------------------------------------------------------------------

    def get_producer_index(
        self, kind_or_op_id: CacheKind | str | int, op_id: int | None = None
    ) -> int | None:
        if op_id is None:
            op_id = int(kind_or_op_id)
        return self._producer_map.pop(int(op_id), None)

    def set_consumer(
        self,
        kind_or_producer_index: CacheKind | str | int | Iterable[int],
        producer_index: int | Iterable[int] | None = None,
    ) -> None:
        if producer_index is None:
            producer_index = kind_or_producer_index
        self._counter.set_consumer(producer_index)

    # ------------------------------------------------------------------
    # MemoryExecutor surface stubs
    # ------------------------------------------------------------------

    def set_mamba_layerwise_cow(self, cow_dst_pages_by_src) -> None:
        assert not cow_dst_pages_by_src, (
            "flat host tier has no mamba L2: the flat scheduler config "
            "validation rejects state-only pools"
        )

    def query_l3_pages(self, hashes: list[str]) -> int:
        # No L3 storage tier under the flat build (EventLoop refuses a
        # storage backend up front); report zero hits.
        return 0

    def shutdown(self) -> None:
        self.write_stream.synchronize()
        self.load_stream.synchronize()

    def reset(self) -> None:
        self.write_stream.synchronize()
        self.load_stream.synchronize()
        self._pending_write_pairs.clear()
        self._pending_write_op_ids.clear()
        self._pending_load_pairs.clear()
        self._pending_load_op_ids.clear()
        self.ack_write_queue.clear()
        self.ack_load_queue.clear()
        self._immediate_write_op_ids.clear()
        self._immediate_load_op_ids.clear()
        self._producer_map.clear()
        self._counter.reset()
