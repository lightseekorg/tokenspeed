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

"""Host-side executor for cache writeback and loadback operations."""

from __future__ import annotations

from collections import OrderedDict
from typing import Iterable, NamedTuple

import torch
from tokenspeed_scheduler import Cache

from tokenspeed.runtime.cache.transfer.kv_pool import KVCachePool
from tokenspeed.runtime.cache.transfer.pool import CachePool
from tokenspeed.runtime.cache.transfer.types import (
    PAGED_CACHE_KIND,
    CacheKind,
    Location,
    PagedCacheTransferUnit,
    TransferUnit,
)
from tokenspeed.runtime.execution.cuda_graph_wrapper import get_is_capture_mode
from tokenspeed.runtime.utils import get_colorful_logger, get_device_module

logger = get_colorful_logger(__name__)
device_module = get_device_module()
CONCURRENT_WRITEBACK_BLOCK_QUOTA = 2


def _cache_stream_priorities() -> tuple[int | None, int | None]:
    priority_range = getattr(device_module.Stream, "priority_range", None)
    if priority_range is None:
        return None, None
    try:
        least_priority, greatest_priority = priority_range()
    except (RuntimeError, TypeError):
        return None, None
    return least_priority, greatest_priority


def _new_cache_stream(priority: int | None = None):
    if priority is None:
        return device_module.Stream()
    try:
        return device_module.Stream(priority=priority)
    except (RuntimeError, TypeError):
        return device_module.Stream()


def page_ids_to_token_indices(
    page_ids: list[int],
    page_size: int,
    device: str = "cpu",
) -> torch.Tensor:
    if len(page_ids) == 0:
        return torch.empty((0,), dtype=torch.int64, device=device)
    pages = torch.tensor(page_ids, dtype=torch.int64, device=device)
    offsets = torch.arange(page_size, dtype=torch.int64, device=device)
    return (pages[:, None] * page_size + offsets[None, :]).reshape(-1)


def _dedupe_page_pairs(
    src_pages: Iterable[int],
    dst_pages: Iterable[int],
) -> tuple[list[int], list[int]]:
    seen = set()
    deduped_src = []
    deduped_dst = []
    for src_page, dst_page in zip(src_pages, dst_pages):
        pair = (int(src_page), int(dst_page))
        if pair in seen:
            continue
        seen.add(pair)
        deduped_src.append(pair[0])
        deduped_dst.append(pair[1])
    return deduped_src, deduped_dst


def _ordered_unique(values: Iterable[int]) -> list[int]:
    seen = set()
    result = []
    for value in values:
        value = int(value)
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


class _Ack(NamedTuple):
    finish_event: object  # device_module.Event
    op_ids: list[int]


class HostExecutor:
    def __init__(
        self,
        page_size: int | None = None,
        device_pool=None,
        host_pool=None,
        io_backend: str = "kernel",
        layer_num: int | None = None,
        draft_device_pool=None,
        draft_host_pool=None,
        draft_layer_num: int = 0,
        pools: list[CachePool] | None = None,
        paged_pool=None,
    ):
        self.io_backend = io_backend
        if pools is None:
            has_kv_inputs = (
                page_size is not None
                and device_pool is not None
                and host_pool is not None
                and layer_num is not None
            )
            if not has_kv_inputs:
                if paged_pool is None:
                    raise ValueError(
                        "HostExecutor requires either pools, KV pool inputs, "
                        "or a paged_pool"
                    )
                pools = []
            else:
                pools = [
                    KVCachePool(
                        device_pool=device_pool,
                        host_pool=host_pool,
                        io_backend=io_backend,
                        layer_num=layer_num,
                        draft_device_pool=draft_device_pool,
                        draft_host_pool=draft_host_pool,
                        draft_layer_num=draft_layer_num,
                    )
                ]
        if not pools and paged_pool is None:
            raise ValueError("HostExecutor requires at least one cache pool")

        self.pools = {CacheKind(pool.kind): pool for pool in pools}
        self.paged_pool = paged_pool
        self.device = (
            next(iter(self.pools.values())).device
            if self.pools
            else self.paged_pool.device
        )

        write_priority, load_priority = _cache_stream_priorities()
        self.write_stream = _new_cache_stream(write_priority)
        self.load_stream = _new_cache_stream(load_priority)
        self._writeback_block_quota: int | None = None

        self.write_queues: dict[CacheKind, list[TransferUnit]] = {
            kind: [] for kind in self.pools
        }
        self.load_queues: dict[CacheKind, list[TransferUnit]] = {
            kind: [] for kind in self.pools
        }
        self.paged_write_queue: list[PagedCacheTransferUnit] = []
        self.paged_load_queue: list[PagedCacheTransferUnit] = []

        self.ack_write_queue: list[_Ack] = []
        self.ack_load_queue: list[_Ack] = []
        self.completed_writebacks: list[int] = []

        self._counters = {
            kind: pool.get_layer_done_counter() for kind, pool in self.pools.items()
        }
        self._paged_counter = (
            paged_pool.get_layer_done_counter() if paged_pool is not None else None
        )
        self._producer_map: dict[CacheKind, OrderedDict[int, int]] = {
            kind: OrderedDict() for kind in self.pools
        }
        self._paged_producer_map: OrderedDict[int, int] = OrderedDict()
        self._producer_map_limit = 1024

    def enqueue_writeback(
        self,
        op_id,
        src_pages,
        dst_pages,
        is_retract: bool = False,
        kind: CacheKind | str = CacheKind.KV,
    ) -> None:
        kind = CacheKind(kind)
        pool = self.pools[kind]
        src_pages, dst_pages = _dedupe_page_pairs(src_pages, dst_pages)
        if not src_pages:
            self.completed_writebacks.append(op_id)
            return
        device_indices = page_ids_to_token_indices(src_pages, pool.page_size(), "cpu")
        host_indices = page_ids_to_token_indices(dst_pages, pool.page_size(), "cpu")
        self.write_queues[kind].append(
            TransferUnit(
                kind=kind,
                src_loc=Location.DEVICE,
                dst_loc=Location.HOST,
                src_indices=device_indices,
                dst_indices=host_indices,
                op_id=op_id,
                is_retract=is_retract,
            )
        )

    def enqueue_loadback(
        self,
        op_id,
        src_pages,
        dst_pages,
        kind: CacheKind | str = CacheKind.KV,
        layerwise_cow_dst_pages_by_src: dict[int, list[int]] | None = None,
    ) -> None:
        kind = CacheKind(kind)
        pool = self.pools[kind]
        src_pages, dst_pages = _dedupe_page_pairs(src_pages, dst_pages)
        if not src_pages:
            return
        host_indices = page_ids_to_token_indices(src_pages, pool.page_size(), "cpu")
        device_indices = page_ids_to_token_indices(dst_pages, pool.page_size(), "cpu")
        cow_src_indices = None
        cow_dst_indices = None
        if layerwise_cow_dst_pages_by_src:
            cow_src_pages: list[int] = []
            cow_dst_pages: list[int] = []
            for dst_page in dst_pages:
                for cow_dst in layerwise_cow_dst_pages_by_src.get(int(dst_page), []):
                    cow_src_pages.append(int(dst_page))
                    cow_dst_pages.append(int(cow_dst))
            if cow_src_pages:
                cow_src_indices = page_ids_to_token_indices(
                    cow_src_pages, pool.page_size(), "cpu"
                )
                cow_dst_indices = page_ids_to_token_indices(
                    cow_dst_pages, pool.page_size(), "cpu"
                )
        self.load_queues[kind].append(
            TransferUnit(
                kind=kind,
                src_loc=Location.HOST,
                dst_loc=Location.DEVICE,
                src_indices=host_indices,
                dst_indices=device_indices,
                op_id=op_id,
                layerwise_cow_src_indices=cow_src_indices,
                layerwise_cow_dst_indices=cow_dst_indices,
            )
        )

    def enqueue_paged_cache_writeback(
        self,
        op_id: int,
        transfers: list,
        is_retract: bool = False,
    ) -> None:
        if self.paged_pool is None:
            if transfers:
                raise ValueError("paged-cache writeback requires a paged_pool")
            self.completed_writebacks.append(op_id)
            return
        if not transfers:
            self.completed_writebacks.append(op_id)
            return
        self.paged_write_queue.append(
            PagedCacheTransferUnit(
                op_id=int(op_id),
                transfers=list(transfers),
                is_retract=is_retract,
            )
        )

    def enqueue_paged_cache_loadback(self, op_id: int, transfers: list) -> None:
        if self.paged_pool is None:
            if transfers:
                raise ValueError("paged-cache loadback requires a paged_pool")
            return
        if not transfers:
            return
        self.paged_load_queue.append(
            PagedCacheTransferUnit(op_id=int(op_id), transfers=list(transfers))
        )

    def flush(self) -> None:
        has_load_work = self._has_work(self.load_queues) or bool(self.paged_load_queue)
        has_retract_writeback = any(
            unit.is_retract for units in self.write_queues.values() for unit in units
        ) or any(unit.is_retract for unit in self.paged_write_queue)
        throttle_writeback = has_load_work and not has_retract_writeback
        writeback_block_quota = (
            CONCURRENT_WRITEBACK_BLOCK_QUOTA if throttle_writeback else None
        )
        previous_writeback_block_quota = getattr(self, "_writeback_block_quota", None)
        self._writeback_block_quota = writeback_block_quota
        try:
            self._start_loading()
            self._start_writing()
        finally:
            self._writeback_block_quota = previous_writeback_block_quota

    def _start_writing(self) -> None:
        if not self._has_work(self.write_queues) and not self.paged_write_queue:
            return

        start_event = device_module.Event()
        finish_event = device_module.Event()
        op_ids: list[int] = []

        start_event.record()
        with device_module.stream(self.write_stream):
            start_event.wait(self.write_stream)
            for kind, units in self.write_queues.items():
                if not units:
                    continue
                pool = self.pools[kind]
                unit = self._merge_units(units)
                src_indices, dst_indices = self._prepare_indices(unit, pool)
                self._pool_writeback(
                    pool, src_indices.to(torch.int64), dst_indices.to(torch.int64)
                )
                self._record_if_cuda(src_indices, self.write_stream)
                self._record_if_cuda(dst_indices, self.write_stream)
                op_ids.extend(unit.op_id for unit in units)
            if self.paged_write_queue:
                assert self.paged_pool is not None
                transfers = [
                    transfer
                    for unit in self.paged_write_queue
                    for transfer in unit.transfers
                ]
                self.paged_pool.writeback_paged(transfers)
                op_ids.extend(unit.op_id for unit in self.paged_write_queue)
            finish_event.record()

        self._clear_queues(self.write_queues)
        self.paged_write_queue.clear()
        self.ack_write_queue.append(_Ack(finish_event, _ordered_unique(op_ids)))

    def _start_loading(self) -> None:
        if not self._has_work(self.load_queues) and not self.paged_load_queue:
            return
        assert (
            not get_is_capture_mode()
        ), "cache loadback must run in eager admission iter"

        with device_module.stream(self.load_stream):
            for kind, units in self.load_queues.items():
                if not units:
                    continue
                pool = self.pools[kind]
                counter = self._counters[kind]
                producer_id = counter.update_producer()
                producer_event = counter.events[producer_id]
                producer_event.start_event.record()
                producer_event.start_event.wait(self.load_stream)

                unit = self._merge_units(units)
                src_indices, dst_indices = self._prepare_indices(unit, pool)
                layerwise_copy = getattr(pool, "copy_layer", None)
                cow_src_indices = unit.layerwise_cow_src_indices
                cow_dst_indices = unit.layerwise_cow_dst_indices
                for layer_index in range(pool.num_layers()):
                    pool.loadback(
                        src_indices.to(torch.int64),
                        dst_indices.to(torch.int64),
                        layer_index,
                    )
                    if (
                        layerwise_copy is not None
                        and cow_src_indices is not None
                        and cow_dst_indices is not None
                    ):
                        layerwise_copy(
                            cow_src_indices.to(torch.int64),
                            cow_dst_indices.to(torch.int64),
                            layer_index,
                        )
                    producer_event.complete(layer_index)
                self._record_if_cuda(src_indices, self.load_stream)
                self._record_if_cuda(dst_indices, self.load_stream)
                if cow_src_indices is not None:
                    self._record_if_cuda(cow_src_indices, self.load_stream)
                if cow_dst_indices is not None:
                    self._record_if_cuda(cow_dst_indices, self.load_stream)

                op_ids = _ordered_unique(unit.op_id for unit in units)
                self.ack_load_queue.append(_Ack(producer_event.finish_event, op_ids))
                producer_map = self._producer_map[kind]
                for op_id in op_ids:
                    producer_map[op_id] = producer_id
                while len(producer_map) > self._producer_map_limit:
                    producer_map.popitem(last=False)

            if self.paged_load_queue:
                assert self.paged_pool is not None
                assert self._paged_counter is not None
                producer_id = self._paged_counter.update_producer()
                producer_event = self._paged_counter.events[producer_id]
                producer_event.start_event.record()
                producer_event.start_event.wait(self.load_stream)
                transfers = [
                    transfer
                    for unit in self.paged_load_queue
                    for transfer in unit.transfers
                ]
                for layer_index in range(self.paged_pool.num_layers()):
                    self.paged_pool.loadback_paged(transfers, layer_index)
                    producer_event.complete(layer_index)
                op_ids = _ordered_unique(unit.op_id for unit in self.paged_load_queue)
                self.ack_load_queue.append(_Ack(producer_event.finish_event, op_ids))
                for op_id in op_ids:
                    self._paged_producer_map[op_id] = producer_id
                while len(self._paged_producer_map) > self._producer_map_limit:
                    self._paged_producer_map.popitem(last=False)

        self._clear_queues(self.load_queues)
        self.paged_load_queue.clear()

    @staticmethod
    def _has_work(queues: dict[CacheKind, list[TransferUnit]]) -> bool:
        return any(bool(units) for units in queues.values())

    @staticmethod
    def _clear_queues(queues: dict[CacheKind, list[TransferUnit]]) -> None:
        for units in queues.values():
            units.clear()

    @staticmethod
    def _merge_units(units: list[TransferUnit]) -> TransferUnit:
        assert units
        if len(units) == 1:
            return units[0]
        first = units[0]
        cow_src_indices = [
            unit.layerwise_cow_src_indices
            for unit in units
            if unit.layerwise_cow_src_indices is not None
        ]
        cow_dst_indices = [
            unit.layerwise_cow_dst_indices
            for unit in units
            if unit.layerwise_cow_dst_indices is not None
        ]
        return TransferUnit(
            kind=first.kind,
            src_loc=first.src_loc,
            dst_loc=first.dst_loc,
            src_indices=torch.cat([unit.src_indices for unit in units]),
            dst_indices=torch.cat([unit.dst_indices for unit in units]),
            op_id=-1,
            is_retract=any(unit.is_retract for unit in units),
            layerwise_cow_src_indices=(
                torch.cat(cow_src_indices) if cow_src_indices else None
            ),
            layerwise_cow_dst_indices=(
                torch.cat(cow_dst_indices) if cow_dst_indices else None
            ),
        )

    def _prepare_indices(
        self, unit: TransferUnit, pool: CachePool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if unit.src_loc == Location.HOST:
            host_indices = unit.src_indices
            device_indices = unit.dst_indices
        elif unit.dst_loc == Location.HOST:
            host_indices = unit.dst_indices
            device_indices = unit.src_indices
        else:
            raise ValueError(f"unsupported transfer direction: {unit.direction}")

        io_backend = getattr(pool, "io_backend", self.io_backend)
        if io_backend == "kernel":
            target_device = pool.device
            if device_indices.device != target_device:
                device_indices = device_indices.to(target_device, non_blocking=True)
            if host_indices.device != target_device:
                host_indices = host_indices.to(target_device, non_blocking=True)
        elif io_backend == "direct":
            if pool.host_layout == "layer_first":
                device_indices = device_indices.cpu()
                host_indices, idx = host_indices.sort()
                device_indices = device_indices.index_select(0, idx)
            else:
                raise ValueError(f"Unsupported host layout: {pool.host_layout}")
        else:
            raise ValueError(f"Unsupported io_backend={io_backend}")

        if unit.src_loc == Location.HOST:
            return host_indices, device_indices
        return device_indices, host_indices

    def _pool_writeback(
        self, pool: CachePool, src_indices: torch.Tensor, dst_indices: torch.Tensor
    ) -> None:
        try:
            pool.writeback(
                src_indices, dst_indices, block_quota=self._writeback_block_quota
            )
        except TypeError as exc:
            if "block_quota" not in str(exc):
                raise
            pool.writeback(src_indices, dst_indices)

    @staticmethod
    def _record_if_cuda(tensor: torch.Tensor, stream) -> None:
        if tensor.is_cuda:
            tensor.record_stream(stream)

    def drain(self) -> list:
        results: list = []
        results.extend(self._poll_write_acks())
        results.extend(self._poll_load_acks())
        return results

    def _poll_write_acks(self) -> list:
        results = []
        completed_writebacks = getattr(self, "completed_writebacks", [])
        for op_id in completed_writebacks:
            logger.debug("[cache_op] writeback done op_id=%s immediate=True", op_id)
            evt = Cache.WriteBackDoneEvent()
            evt.op_id = op_id
            evt.success = True
            results.append(evt)
        completed_writebacks.clear()
        remaining = []
        for ack in self.ack_write_queue:
            if ack.finish_event.query():
                logger.debug(
                    "[cache_op] writeback done op_ids=%s immediate=False", ack.op_ids
                )
                for op_id in ack.op_ids:
                    evt = Cache.WriteBackDoneEvent()
                    evt.op_id = op_id
                    evt.success = True
                    results.append(evt)
            else:
                remaining.append(ack)
        self.ack_write_queue[:] = remaining
        return results

    def _poll_load_acks(self) -> list:
        results = []
        remaining = []
        for ack in self.ack_load_queue:
            if not ack.finish_event.query():
                remaining.append(ack)
        self.ack_load_queue[:] = remaining
        return results

    def get_producer_index(
        self, kind_or_op_id: CacheKind | str | int, op_id: int | None = None
    ) -> int | None:
        if op_id is None:
            kind = CacheKind.KV
            op_id = int(kind_or_op_id)
        else:
            if self._is_paged_kind(kind_or_op_id):
                return self._paged_producer_map.pop(int(op_id), None)
            kind = CacheKind(kind_or_op_id)
        return self._producer_map[kind].pop(int(op_id), None)

    def set_consumer(
        self,
        kind_or_producer_index: CacheKind | str | int | Iterable[int],
        producer_index: int | Iterable[int] | None = None,
    ) -> None:
        if producer_index is None:
            kind = CacheKind.KV
            producer_index = kind_or_producer_index
        else:
            if self._is_paged_kind(kind_or_producer_index):
                assert self._paged_counter is not None
                self._paged_counter.set_consumer(producer_index)
                return
            kind = CacheKind(kind_or_producer_index)
        self._counters[kind].set_consumer(producer_index)

    def _is_paged_kind(self, kind: CacheKind | str | int | Iterable[int]) -> bool:
        if self.paged_pool is None:
            return False
        return str(kind) == PAGED_CACHE_KIND

    def shutdown(self) -> None:
        self.write_stream.synchronize()
        self.load_stream.synchronize()
        for pool in self.pools.values():
            shutdown = getattr(pool, "shutdown", None)
            if shutdown is not None:
                shutdown()
        if self.paged_pool is not None:
            shutdown = getattr(self.paged_pool, "shutdown", None)
            if shutdown is not None:
                shutdown()

    def reset(self) -> None:
        self.write_stream.synchronize()
        self.load_stream.synchronize()
        self._clear_queues(self.write_queues)
        self._clear_queues(self.load_queues)
        self.paged_write_queue.clear()
        self.paged_load_queue.clear()
        self.ack_write_queue.clear()
        self.ack_load_queue.clear()
        for producer_map in self._producer_map.values():
            producer_map.clear()
        self._paged_producer_map.clear()
        for counter in self._counters.values():
            counter.reset()
        if self._paged_counter is not None:
            self._paged_counter.reset()
