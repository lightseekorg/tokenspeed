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

"""Storage-backed executor for cache prefetch and backup operations."""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from queue import Empty, Queue
from typing import Callable, Optional

import torch
import torch.distributed as dist
from tokenspeed_scheduler import Cache

from tokenspeed.runtime.cache.executor.host_executor import (
    page_ids_to_token_indices,
)
from tokenspeed.runtime.cache.kvstore_storage import KVStoreStorageConfig
from tokenspeed.runtime.cache.storage import StorageBackendFactory
from tokenspeed.runtime.distributed.process_group_manager import _make_all_groups
from tokenspeed.runtime.layers.attention.kv_cache.mla import MLATokenToKVPool
from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)


def _parse_storage_backend_extra_config(raw: Optional[str]):
    extra_config = {}
    if raw:
        try:
            extra_config = json.loads(raw)
        except Exception as exc:
            logger.error("Invalid backend extra config JSON: %s", exc)
            raise

    prefetch_threshold = extra_config.pop("prefetch_threshold", 256)
    prefetch_timeout_base = extra_config.pop("prefetch_timeout_base", 1)
    prefetch_timeout_per_ki_token = extra_config.pop(
        "prefetch_timeout_per_ki_token", 0.25
    )

    if not isinstance(prefetch_threshold, int):
        raise ValueError(
            f"prefetch_threshold must be int, got {type(prefetch_threshold).__name__}"
        )
    if not isinstance(prefetch_timeout_base, (int, float)):
        raise ValueError(
            f"prefetch_timeout_base must be number, got {type(prefetch_timeout_base).__name__}"
        )
    if not isinstance(prefetch_timeout_per_ki_token, (int, float)):
        raise ValueError(
            f"prefetch_timeout_per_ki_token must be number, got {type(prefetch_timeout_per_ki_token).__name__}"
        )

    return (
        extra_config,
        prefetch_threshold,
        float(prefetch_timeout_base),
        float(prefetch_timeout_per_ki_token),
    )


def _generate_storage_config(
    device_pool,
    host_pool,
    model_name: Optional[str],
    extra_config_dict: dict,
    is_dp_attention_enabled: bool,
) -> KVStoreStorageConfig:
    tp_rank = dist.get_rank() if dist.is_initialized() else 0
    tp_size = dist.get_world_size() if dist.is_initialized() else 1

    is_mla_backend = isinstance(device_pool, MLATokenToKVPool)

    return KVStoreStorageConfig(
        tp_rank=tp_rank,
        tp_size=tp_size,
        is_mla_model=is_mla_backend,
        is_page_first_layout=host_pool.layout == "page_first",
        model_name=model_name,
        extra_config=extra_config_dict,
    )


class StorageExecutor:
    """Execute L3 storage prefetch and backup operations asynchronously."""

    def __init__(
        self,
        page_size: int,
        device_pool,
        host_pool,
        storage_backend_type: Optional[str],
        storage_backend_extra_config: Optional[str] = None,
        model_name: Optional[str] = None,
        is_dp_attention_enabled: bool = False,
        storage_batch_size: int = 128,
        tp_group=None,
        on_l3_blocks_stored: Callable[[list[str]], None] | None = None,
    ):
        self.page_size = page_size
        self.host_pool = host_pool
        self.storage_batch_size = storage_batch_size
        self._on_l3_blocks_stored = on_l3_blocks_stored
        # op_id → rolling_page_hashes retained until backup completion callback.
        self._backup_hashes: dict[int, list[str]] = {}
        (
            extra_config_dict,
            _prefetch_threshold,
            prefetch_timeout_base,
            prefetch_timeout_per_ki_token,
        ) = _parse_storage_backend_extra_config(storage_backend_extra_config)

        self.prefetch_timeout_base = prefetch_timeout_base
        self.prefetch_timeout_per_page = (
            page_size / 1024 * prefetch_timeout_per_ki_token
        )

        self.storage_backend = None
        if storage_backend_type is not None:
            storage_config = _generate_storage_config(
                device_pool,
                host_pool,
                model_name,
                extra_config_dict,
                is_dp_attention_enabled,
            )
            try:
                self.storage_backend = StorageBackendFactory.create_backend(
                    storage_backend_type, storage_config, host_pool
                )
            except ValueError as exc:
                raise ValueError(f"Failed to create storage backend: {exc}") from exc

            self.storage_backend.register_mem_pool_host(host_pool)
        self.tp_size = (
            torch.distributed.get_world_size(group=tp_group)
            if tp_group is not None
            else 1
        )
        # Dedicated subgroup for kvstore collectives. The shared ``tp_group``
        # is used by the engine main thread elsewhere (event_loop,
        # request_handler). Issuing collectives on it from
        # the aggregator thread would race those callers — different threads
        # on the same rank issuing on the same group can pair up out-of-order
        # across ranks and hang.
        #
        # ``new_group`` is itself a world-wide collective, so when there is
        # more than one TP-shaped group in the world (e.g. DP attention with
        # attn_tp groups [0..3] and [4..7]) we have to call it for *every*
        # such group in the same deterministic order on every rank — calling
        # it only with the local rank's TP group would deadlock other ranks.
        # ``_make_all_groups`` enumerates the full set with the same size and
        # stride pattern, mirroring ``pg_manager.init_process_group``.
        self._tp_group = None
        if tp_group is not None and self.tp_size > 1:
            my_ranks = tuple(torch.distributed.get_process_group_ranks(tp_group))
            for g in _make_all_groups(my_ranks):
                pg = torch.distributed.new_group(ranks=list(g), backend="gloo")
                if g == my_ranks:
                    self._tp_group = pg
        self._results: Queue = Queue()
        self._prefetch_op_to_rid: dict = {}  # op_id → request_id
        self._executor: Optional[ThreadPoolExecutor] = None
        # All collectives on the dedicated kvstore subgroup are funneled
        # through a single aggregator thread so they issue in deterministic
        # submit order across ranks (Gloo and NCCL groups both require
        # callers to agree on issuance order; concurrent issuance from
        # worker threads can deadlock when ops finish in different order on
        # each rank).
        self._aggregator_pending: Queue = Queue()
        self._aggregator_stop = threading.Event()
        self._aggregator_thread: Optional[threading.Thread] = None
        if self.storage_backend is not None:
            self._executor = ThreadPoolExecutor(
                max_workers=2,
                thread_name_prefix="tokenspeed-mem-l3-io",
            )
            self._aggregator_thread = threading.Thread(
                target=self._aggregator_loop,
                name="tokenspeed-mem-l3-aggr",
                daemon=True,
            )
            self._aggregator_thread.start()

    @property
    def enabled(self) -> bool:
        return self.storage_backend is not None

    def submit_prefetch(self, op) -> None:
        # Extract request_id from the op and remember mapping
        rid = op.request_id
        self._prefetch_op_to_rid[op.op_id] = rid

        if not self.enabled:
            evt = Cache.PrefetchDoneEvent()
            evt.op_id = op.op_id
            evt.request_id = rid
            evt.success = False
            evt.completed_pages = 0
            self._results.put(evt)
            return
        future = self._executor.submit(self._run_prefetch, op)
        # Enqueue at submit time (not completion time) so both ranks see the
        # same aggregator order, which guarantees the per-op TP all_reduce
        # pairs up correctly.
        self._aggregator_pending.put(("prefetch", op.op_id, rid, future))

    def submit_backup(self, op) -> None:
        if not self.enabled:
            evt = Cache.BackUpDoneEvent()
            evt.op_id = op.op_id
            evt.success = False
            self._results.put(evt)
            return
        self._backup_hashes[op.op_id] = list(op.rolling_page_hashes)
        future = self._executor.submit(self._run_backup, op)
        future.add_done_callback(
            lambda fut, oid=op.op_id: self._on_backup_done(oid, fut)
        )

    def _prefetch_deadline(self, n_pages: int) -> float:
        return (
            time.monotonic()
            + self.prefetch_timeout_base
            + n_pages * self.prefetch_timeout_per_page
        )

    def _run_prefetch(self, op) -> int:
        hashes = op.rolling_page_hashes
        if not hashes:
            logger.debug("[cache_op] prefetch_exec op_id=%s no hashes, skip", op.op_id)
            return 0
        dst_pages = op.dst_pages
        assert len(hashes) == len(dst_pages), (
            f"prefetch key/page mismatch: {len(hashes)} hashes "
            f"vs {len(dst_pages)} dst_pages"
        )

        deadline = self._prefetch_deadline(len(hashes))
        completed_pages = 0

        for i in range(0, len(hashes), self.storage_batch_size):
            if time.monotonic() > deadline:
                logger.warning(
                    "prefetch op %s timed out after %s/%s pages",
                    op.op_id,
                    completed_pages,
                    len(hashes),
                )
                break

            batch_hashes = hashes[i : i + self.storage_batch_size]
            batch_dst = dst_pages[i : i + len(batch_hashes)]
            host_indices = page_ids_to_token_indices(batch_dst, self.page_size, "cpu")
            try:
                results = self.storage_backend.batch_get_v1(batch_hashes, host_indices)
            except Exception as exc:
                logger.warning(
                    "prefetch op %s batch IO error at offset %s: %s", op.op_id, i, exc
                )
                break
            result_ok = 0
            for ok in results:
                if not ok:
                    break
                result_ok += 1
            completed_pages += result_ok
            if result_ok < len(results):
                logger.warning(
                    "prefetch op %s: %s/%s pages missed in batch at offset %s",
                    op.op_id,
                    len(results) - result_ok,
                    len(results),
                    i,
                )
                break

        # NOTE: TP all_reduce moved to the aggregator thread; the worker only
        # performs storage I/O and returns the local count. See _aggregator_loop.
        logger.debug(
            "[cache_op] prefetch_exec op_id=%s completed %s/%s pages (local)",
            op.op_id,
            completed_pages,
            len(hashes),
        )
        return completed_pages

    def _run_backup(self, op) -> None:
        hashes = op.rolling_page_hashes
        src_pages = op.src_pages
        total_num_pages = len(hashes)
        logger.debug(
            "[cache_op] backup_exec op_id=%s pages=%s", op.op_id, total_num_pages
        )
        completed_pages = 0
        for i in range(0, total_num_pages, self.storage_batch_size):
            batch_hashes = hashes[i : i + self.storage_batch_size]
            batch_src = src_pages[i : i + len(batch_hashes)]
            host_indices = page_ids_to_token_indices(batch_src, self.page_size, "cpu")
            results = self.storage_backend.batch_set_v1(batch_hashes, host_indices)
            result_ok = sum(1 for ok in results if ok)
            completed_pages += result_ok
            if result_ok < len(results):
                failed_count = len(results) - result_ok
                logger.warning(
                    "backup op %s: %s/%s pages failed in batch at offset %s",
                    op.op_id,
                    failed_count,
                    len(results),
                    i,
                )
                raise RuntimeError(
                    f"backup op {op.op_id}: {completed_pages}/{total_num_pages} pages ok, "
                    f"{failed_count} failed in batch at offset {i}"
                )
        logger.debug(
            "[cache_op] backup_exec op_id=%s done, all %s/%s pages ok",
            op.op_id,
            completed_pages,
            total_num_pages,
        )

    def _aggregator_loop(self) -> None:
        """Single thread that owns all collectives on ``self._tp_group``.

        ``self._tp_group`` is a dedicated kvstore-only Gloo subgroup, separate
        from the ``tp_group`` used by the main thread elsewhere in the engine
        — so this thread's collectives don't race main-thread collectives.

        Both prefetch completions (which need a TP MIN all_reduce on
        ``completed_pages``) and ``query_exists`` calls are funneled through
        ``_aggregator_pending`` in submit order. Because both ranks observe
        the same submit order (deterministic from the C++ scheduler +
        main-thread engine loop), the aggregator on each rank issues
        collectives in the same order, so the MIN all_reduces pair correctly
        across ranks.
        """
        while not self._aggregator_stop.is_set():
            try:
                item = self._aggregator_pending.get(block=True, timeout=1)
            except Empty:
                continue
            kind = item[0]
            if kind == "prefetch":
                _, op_id, rid, future = item
                evt = Cache.PrefetchDoneEvent()
                evt.op_id = op_id
                evt.request_id = self._prefetch_op_to_rid.pop(op_id, rid)
                # Encode local outcome with a -1 sentinel for "this rank
                # failed locally". A bare ``continue`` on local failure would
                # skip the all_reduce below and deadlock peers that *did*
                # succeed locally — the collective must run in lockstep on
                # every rank. ReduceOp.MIN propagates the sentinel across all
                # ranks, so every rank agrees the op failed.
                try:
                    local = future.result()
                except Exception as exc:
                    logger.error("prefetch op %s local I/O failed: %s", op_id, exc)
                    local = -1
                completed_pages = local
                if self.tp_size > 1:
                    try:
                        t = torch.tensor(local, dtype=torch.int)
                        torch.distributed.all_reduce(
                            t,
                            op=torch.distributed.ReduceOp.MIN,
                            group=self._tp_group,
                        )
                        completed_pages = t.item()
                    except Exception as exc:
                        logger.warning("prefetch op %s TP sync failed: %s", op_id, exc)
                        completed_pages = -1
                if completed_pages < 0:
                    evt.success = False
                    evt.completed_pages = 0
                else:
                    evt.success = True
                    evt.completed_pages = completed_pages
                logger.debug(
                    "[prefetch_done] op_id=%s request_id=%s success=%s completed_pages=%s",
                    op_id,
                    evt.request_id,
                    evt.success,
                    evt.completed_pages,
                )
                self._results.put(evt)
            elif kind == "query_exists":
                _, hashes, result_future = item
                # Same reasoning as the prefetch branch: never short-circuit
                # before the collective.
                try:
                    local = self._query_exists_local(hashes)
                except Exception as exc:
                    logger.error("query_exists local I/O failed: %s", exc)
                    local = -1
                total_hit = local
                if self.tp_size > 1:
                    try:
                        t = torch.tensor(local, dtype=torch.int)
                        torch.distributed.all_reduce(
                            t,
                            op=torch.distributed.ReduceOp.MIN,
                            group=self._tp_group,
                        )
                        total_hit = t.item()
                    except Exception as exc:
                        logger.warning("query_exists TP sync failed: %s", exc)
                        total_hit = -1
                if total_hit < 0:
                    result_future.set_exception(
                        RuntimeError("query_exists failed on at least one rank")
                    )
                else:
                    result_future.set_result(total_hit)
            else:
                logger.error("unknown aggregator item kind: %s", kind)

    def _on_backup_done(self, op_id: int, future) -> None:
        evt = Cache.BackUpDoneEvent()
        evt.op_id = op_id
        hashes = self._backup_hashes.pop(op_id, [])
        try:
            future.result()
            evt.success = True
        except Exception as exc:
            evt.success = False
            logger.error("backup op %s failed: %s", op_id, exc)
        self._results.put(evt)
        if evt.success and self._on_l3_blocks_stored and hashes:
            try:
                self._on_l3_blocks_stored(hashes)
            except Exception:
                logger.exception("L3 KV event callback failed")

    def drain(self) -> list:
        results = []
        while True:
            try:
                results.append(self._results.get_nowait())
            except Empty:
                return results

    def query_exists(self, hashes: list[str]) -> int:
        if not self.enabled or not hashes:
            return 0
        if self.tp_size <= 1 or self._aggregator_thread is None:
            return self._query_exists_local(hashes)
        # Route through the aggregator so the all_reduce on ``tp_group`` is
        # serialized with prefetch-completion all_reduces on the same group.
        result_future: Future = Future()
        self._aggregator_pending.put(("query_exists", list(hashes), result_future))
        return result_future.result()

    def _query_exists_local(self, hashes: list[str]) -> int:
        total_hit = 0
        for i in range(0, len(hashes), self.storage_batch_size):
            batch = hashes[i : i + self.storage_batch_size]
            hit = self.storage_backend.batch_exists(batch)
            total_hit += hit
            if hit < len(batch):
                break
        return total_hit

    def shutdown(self) -> None:
        if self._aggregator_thread is not None:
            self._aggregator_stop.set()
            self._aggregator_thread.join(timeout=10)
            self._aggregator_thread = None
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        if self.storage_backend is not None and hasattr(self.storage_backend, "close"):
            try:
                self.storage_backend.close()
            except Exception:
                logger.exception("Failed to close storage backend")
            self.storage_backend = None
