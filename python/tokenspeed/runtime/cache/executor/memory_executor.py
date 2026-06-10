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

from __future__ import annotations

"""Top-level memory executor that coordinates host and storage executors."""

import logging
from dataclasses import dataclass
from typing import Iterable, Optional

try:
    from tokenspeed.runtime.layers.attention.kv_cache.mha import (
        MHATokenToKVPool as MHATokenToKVPoolPaged,
    )
except ImportError:
    MHATokenToKVPoolPaged = None
try:
    from tokenspeed.runtime.layers.attention.kv_cache.mla import (
        MLATokenToKVPool as MLATokenToKVPoolPaged,
    )
except (ImportError, AttributeError):
    MLATokenToKVPoolPaged = None

from tokenspeed_scheduler import Cache

from tokenspeed.runtime.cache.deepseek_v4_cache_host import (
    DeepseekV4TokenToKVPoolHost,
)
from tokenspeed.runtime.cache.executor.host_executor import HostExecutor
from tokenspeed.runtime.cache.executor.storage_executor import StorageExecutor
from tokenspeed.runtime.cache.kv_cache_host import (
    MHATokenToKVPoolHost,
    MLATokenToKVPoolHost,
)
from tokenspeed.runtime.cache.mamba_cache_host import MambaPoolHost
from tokenspeed.runtime.cache.transfer.deepseek_v4_pool import (
    DeepseekV4CachePool,
    _count_page_spans,
    _ordered_page_pairs,
)
from tokenspeed.runtime.cache.transfer.kv_pool import KVCachePool
from tokenspeed.runtime.cache.transfer.mamba_pool import MambaCachePool
from tokenspeed.runtime.cache.transfer.types import CacheKind
from tokenspeed.runtime.layers.attention.kv_cache.deepseek_v4 import (
    DeepseekV4TokenToKVPool,
)
from tokenspeed.runtime.layers.attention.kv_cache.mha import MHATokenToKVPool
from tokenspeed.runtime.layers.attention.kv_cache.mla import MLATokenToKVPool
from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)


@dataclass(slots=True)
class MemoryExecutorConfig:
    layer_num: int
    page_size: int = 64
    host_ratio: float = 2.0
    host_size_gb: int = 0
    io_backend: str = "kernel"
    host_layout: str = "layer_first"
    storage_backend: Optional[str] = "mooncake"
    storage_backend_extra_config: Optional[str] = None
    model_name: Optional[str] = None
    enable_mamba_l2: bool = False
    mamba_l2_host_slots: int = 0
    mamba_l2_layout: str = "layer_first"
    mamba_l2_io_backend: str = "kernel"


class MemoryExecutor:
    """Coordinate host-memory and storage-backed cache operations."""

    def __init__(
        self,
        device_pool,
        config: MemoryExecutorConfig,
        is_dp_attention_enabled: bool,
        tp_group=None,
        draft_device_pool=None,
        mamba_pool=None,
    ):
        self.page_size = config.page_size

        _mha_types = (MHATokenToKVPool,)
        if MHATokenToKVPoolPaged is not None:
            _mha_types = (MHATokenToKVPool, MHATokenToKVPoolPaged)

        _mla_types = (MLATokenToKVPool,)
        if MLATokenToKVPoolPaged is not None:
            _mla_types = (MLATokenToKVPool, MLATokenToKVPoolPaged)

        # Unwrap LayerMappedKVPool (hybrid GDN models) to get the inner MHA pool.
        actual_pool = device_pool
        if hasattr(device_pool, "inner") and not isinstance(
            device_pool, (*_mha_types, *_mla_types)
        ):
            actual_pool = device_pool.inner

        self.paged_cache_pool = None
        actual_draft_pool = None
        if isinstance(actual_pool, DeepseekV4TokenToKVPool):
            if config.storage_backend is not None:
                raise NotImplementedError(
                    "DeepSeek V4 KVStore currently supports L2 host memory only; "
                    "L3 storage backends are out of scope."
                )
            self.host_pool = DeepseekV4TokenToKVPoolHost(
                actual_pool,
                config.host_ratio,
                config.host_size_gb,
                config.host_layout,
            )
            self.paged_cache_pool = DeepseekV4CachePool(
                device_pool=actual_pool,
                host_pool=self.host_pool,
                io_backend=config.io_backend,
            )
        elif isinstance(actual_pool, _mha_types):
            self.host_pool = MHATokenToKVPoolHost(
                actual_pool,
                config.host_ratio,
                config.host_size_gb,
                config.page_size,
                config.host_layout,
            )
        elif isinstance(actual_pool, _mla_types):
            self.host_pool = MLATokenToKVPoolHost(
                actual_pool,
                config.host_ratio,
                config.host_size_gb,
                config.page_size,
                config.host_layout,
            )
        else:
            raise ValueError(
                "host_pool only supports MHA, MLA, and DeepSeek V4, "
                f"got {type(actual_pool)} from module {type(actual_pool).__module__}"
            )

        # Draft model L2 cache: draft shares the same page mapping as the base
        # model, so its host pool must hold exactly the same number of tokens.
        # Pass host_size_tokens directly to bypass ratio/GB recalculation.
        if draft_device_pool is not None and self.paged_cache_pool is None:
            actual_draft_pool = draft_device_pool
            if hasattr(draft_device_pool, "inner") and not isinstance(
                draft_device_pool, (*_mha_types, *_mla_types)
            ):
                actual_draft_pool = draft_device_pool.inner
            if isinstance(actual_draft_pool, _mha_types):
                self.draft_host_pool = MHATokenToKVPoolHost(
                    actual_draft_pool,
                    config.host_ratio,
                    config.host_size_gb,
                    config.page_size,
                    config.host_layout,
                    host_size_tokens=self.host_pool.size,
                )
            elif isinstance(actual_draft_pool, _mla_types):
                self.draft_host_pool = MLATokenToKVPoolHost(
                    actual_draft_pool,
                    config.host_ratio,
                    config.host_size_gb,
                    config.page_size,
                    config.host_layout,
                    host_size_tokens=self.host_pool.size,
                )
            else:
                raise ValueError(
                    f"draft_device_pool only supports MHA and MLA, "
                    f"got {type(actual_draft_pool)}"
                )
            draft_host_bytes = (
                self.draft_host_pool.size * self.draft_host_pool.size_per_token
            )
            logger.info(
                "Allocating %.2f GB host memory for draft model L2 cache (pool_type=%s size_tokens=%s size_per_token=%s layer_num=%s)",
                draft_host_bytes / 1e9,
                type(self.draft_host_pool).__name__,
                self.draft_host_pool.size,
                self.draft_host_pool.size_per_token,
                actual_draft_pool.layer_num,
            )
            draft_layer_num = actual_draft_pool.layer_num
        else:
            self.draft_host_pool = None
            draft_layer_num = 0

        pools = None
        self.mamba_host_pool = None
        if (
            self.paged_cache_pool is None
            and config.enable_mamba_l2
            and mamba_pool is not None
            and config.mamba_l2_host_slots > 0
        ):
            self.mamba_host_pool = MambaPoolHost(
                mamba_pool,
                host_size_slots=config.mamba_l2_host_slots,
                layout=config.mamba_l2_layout,
            )
            pools = [
                KVCachePool(
                    device_pool=device_pool,
                    host_pool=self.host_pool,
                    io_backend=config.io_backend,
                    layer_num=actual_pool.layer_num,
                    draft_device_pool=(
                        actual_draft_pool if draft_device_pool is not None else None
                    ),
                    draft_host_pool=self.draft_host_pool,
                    draft_layer_num=draft_layer_num,
                ),
                MambaCachePool(
                    device_pool=mamba_pool,
                    host_pool=self.mamba_host_pool,
                    io_backend=config.mamba_l2_io_backend,
                ),
            ]
            logger.debug(
                "[cache_op] MemoryExecutor init pools=%s host_pools=%s draft=%s mamba=%s io_backend=%s host_layout=%s",
                [pool.kind.value for pool in pools],
                [type(self.host_pool).__name__, type(self.mamba_host_pool).__name__],
                self.draft_host_pool is not None,
                True,
                config.io_backend,
                config.host_layout,
            )

        if self.paged_cache_pool is not None:
            self.host_exec = HostExecutor(
                pools=pools or [],
                paged_pool=self.paged_cache_pool,
                io_backend=config.io_backend,
            )
        elif pools is not None:
            self.host_exec = HostExecutor(pools=pools, io_backend=config.io_backend)
        else:
            self.host_exec = HostExecutor(
                page_size=config.page_size,
                device_pool=device_pool,
                host_pool=self.host_pool,
                io_backend=config.io_backend,
                layer_num=actual_pool.layer_num,
                draft_device_pool=(
                    actual_draft_pool if draft_device_pool is not None else None
                ),
                draft_host_pool=self.draft_host_pool,
                draft_layer_num=draft_layer_num,
            )
        self.storage_exec = StorageExecutor(
            page_size=config.page_size,
            device_pool=device_pool,
            host_pool=self.host_pool,
            storage_backend_type=config.storage_backend,
            storage_backend_extra_config=config.storage_backend_extra_config,
            model_name=config.model_name,
            is_dp_attention_enabled=is_dp_attention_enabled,
            tp_group=tp_group,
        )
        self._pending_mamba_layerwise_cow: dict[int, list[int]] | None = None

    @staticmethod
    def _page_groups_by_kind(op) -> dict[CacheKind, tuple[list, list]]:
        src_by_kind = getattr(op, "src_pages_by_kind", None)
        dst_by_kind = getattr(op, "dst_pages_by_kind", None)
        if src_by_kind is None or dst_by_kind is None:
            return {CacheKind.KV: (op.src_pages, op.dst_pages)}
        groups: dict[CacheKind, tuple[list, list]] = {}
        for kind in CacheKind:
            src_pages = src_by_kind.get(kind.value, [])
            dst_pages = dst_by_kind.get(kind.value, [])
            groups[kind] = (src_pages, dst_pages)
        return groups

    @staticmethod
    def _paged_transfer_summary(
        transfers: list,
    ) -> tuple[int, int, int, list[str], list[str]]:
        group_pages: dict[str, int] = {}
        group_spans: dict[str, int] = {}
        for transfer in transfers:
            group_id = str(getattr(transfer, "group_id", "unknown"))
            src_pages = getattr(transfer, "src_pages", [])
            dst_pages = getattr(transfer, "dst_pages", [])
            page_pairs = _ordered_page_pairs(src_pages, dst_pages)
            group_pages[group_id] = group_pages.get(group_id, 0) + len(page_pairs)
            group_spans[group_id] = group_spans.get(group_id, 0) + _count_page_spans(
                page_pairs
            )
        return (
            len(group_pages),
            sum(group_pages.values()),
            sum(group_spans.values()),
            [f"{group_id}:{pages}" for group_id, pages in sorted(group_pages.items())],
            [f"{group_id}:{spans}" for group_id, spans in sorted(group_spans.items())],
        )

    def set_mamba_layerwise_cow(
        self, cow_dst_pages_by_src: dict[int, list[int]] | None
    ) -> None:
        self._pending_mamba_layerwise_cow = cow_dst_pages_by_src or None

    def submit_plan(self, plan) -> None:
        if plan.cache:
            logger.debug("[cache_op] submit_plan: %s cache ops", len(plan.cache))
        try:
            for op in plan.cache:
                self.submit(op)
            self.host_exec.flush()
        finally:
            self._pending_mamba_layerwise_cow = None

    def submit(self, op) -> None:
        if isinstance(op, Cache.WriteBackOp):
            logger.debug(
                "[cache_op] writeback op_id=%s src_pages=%s dst_pages=%s",
                op.op_ids,
                len(op.src_pages),
                len(op.dst_pages),
            )
            groups = self._page_groups_by_kind(op)
            paged_transfers_by_op = getattr(op, "paged_cache_transfers", [])
            is_retract_flags = getattr(op, "is_retract", [])
            for i in range(len(op.op_ids)):
                op_id = op.op_ids[i]
                is_retract = (
                    bool(is_retract_flags[i]) if i < len(is_retract_flags) else False
                )
                submitted = False
                for kind, (src_groups, dst_groups) in groups.items():
                    if kind not in self.host_exec.pools:
                        continue
                    src_pages = src_groups[i] if i < len(src_groups) else []
                    dst_pages = dst_groups[i] if i < len(dst_groups) else []
                    if not src_pages:
                        continue
                    if kind == CacheKind.MAMBA:
                        logger.debug(
                            "[cache_op][mamba_l2] writeback schedule "
                            "op_id=%s slots=%s device_slots=%s host_slots=%s "
                            "is_retract=%s",
                            op_id,
                            len(src_pages),
                            src_pages[:8],
                            dst_pages[:8],
                            is_retract,
                        )
                    self.host_exec.enqueue_writeback(
                        op_id,
                        src_pages,
                        dst_pages,
                        is_retract=is_retract,
                        kind=kind,
                    )
                    submitted = True
                paged_transfers = (
                    paged_transfers_by_op[i] if i < len(paged_transfers_by_op) else []
                )
                if paged_transfers:
                    if logger.isEnabledFor(logging.DEBUG):
                        (
                            group_count,
                            page_count,
                            span_count,
                            group_pages,
                            group_spans,
                        ) = self._paged_transfer_summary(paged_transfers)
                        logger.debug(
                            "[cache_op][paged_l2] writeback schedule "
                            "op_id=%s groups=%s pages=%s spans=%s "
                            "group_pages=%s group_spans=%s "
                            "is_retract=%s",
                            op_id,
                            group_count,
                            page_count,
                            span_count,
                            group_pages,
                            group_spans,
                            is_retract,
                        )
                    self.host_exec.enqueue_paged_cache_writeback(
                        op_id,
                        paged_transfers,
                        is_retract=is_retract,
                    )
                    submitted = True
                if not submitted and all(
                    i >= len(src_groups) or not src_groups[i]
                    for kind, (src_groups, _) in groups.items()
                    if kind in self.host_exec.pools
                ):
                    self.host_exec.completed_writebacks.append(op_id)
        elif isinstance(op, Cache.LoadBackOp):
            logger.debug(
                "[cache_op] loadback op_id=%s src_pages=%s dst_pages=%s",
                op.op_ids,
                len(op.src_pages),
                len(op.dst_pages),
            )
            groups = self._page_groups_by_kind(op)
            paged_transfers_by_op = getattr(op, "paged_cache_transfers", [])
            for i in range(len(op.op_ids)):
                op_id = op.op_ids[i]
                for kind, (src_groups, dst_groups) in groups.items():
                    if kind not in self.host_exec.pools:
                        continue
                    src_pages = src_groups[i] if i < len(src_groups) else []
                    dst_pages = dst_groups[i] if i < len(dst_groups) else []
                    if not src_pages:
                        continue
                    if kind == CacheKind.MAMBA:
                        logger.debug(
                            "[cache_op][mamba_l2] loadback schedule "
                            "op_id=%s slots=%s host_slots=%s device_slots=%s",
                            op_id,
                            len(src_pages),
                            src_pages[:8],
                            dst_pages[:8],
                        )
                    loadback_kwargs = {}
                    mamba_layerwise_cow = getattr(
                        self, "_pending_mamba_layerwise_cow", None
                    )
                    if kind == CacheKind.MAMBA and mamba_layerwise_cow:
                        loadback_kwargs["layerwise_cow_dst_pages_by_src"] = (
                            mamba_layerwise_cow
                        )
                    self.host_exec.enqueue_loadback(
                        op_id, src_pages, dst_pages, kind=kind, **loadback_kwargs
                    )
                paged_transfers = (
                    paged_transfers_by_op[i] if i < len(paged_transfers_by_op) else []
                )
                if paged_transfers:
                    if logger.isEnabledFor(logging.DEBUG):
                        (
                            group_count,
                            page_count,
                            span_count,
                            group_pages,
                            group_spans,
                        ) = self._paged_transfer_summary(paged_transfers)
                        logger.debug(
                            "[cache_op][paged_l2] loadback schedule "
                            "op_id=%s groups=%s pages=%s spans=%s "
                            "group_pages=%s group_spans=%s",
                            op_id,
                            group_count,
                            page_count,
                            span_count,
                            group_pages,
                            group_spans,
                        )
                    self.host_exec.enqueue_paged_cache_loadback(
                        op_id,
                        paged_transfers,
                    )

        elif isinstance(op, Cache.PrefetchOp):
            logger.debug(
                "[cache_op] prefetch op_id=%s dst_pages=%s", op.op_id, len(op.dst_pages)
            )
            self.storage_exec.submit_prefetch(op)
        elif isinstance(op, Cache.BackUpOp):
            logger.debug(
                "[cache_op] backup op_id=%s src_pages=%s", op.op_id, len(op.src_pages)
            )
            self.storage_exec.submit_backup(op)
        else:
            raise ValueError("unsupported cache op kind")

    def poll_results(self) -> list:
        results: list = []
        results.extend(self.host_exec.drain())
        results.extend(self.storage_exec.drain())
        if results:
            for r in results:
                logger.debug(
                    "[cache_op] done op_id=%s success=%s type=%s",
                    r.op_id,
                    r.success,
                    type(r).__name__,
                )
        return results

    def get_producer_index(
        self, kind_or_op_id: CacheKind | str | int, op_id: int | None = None
    ) -> Optional[int]:
        return self.host_exec.get_producer_index(kind_or_op_id, op_id)

    def set_consumer(
        self,
        kind_or_producer_index: CacheKind | str | int | Iterable[int],
        producer_index: int | Iterable[int] | None = None,
    ) -> None:
        self.host_exec.set_consumer(kind_or_producer_index, producer_index)

    def query_l3_pages(self, hashes: list[str]) -> int:
        return self.storage_exec.query_exists(hashes)

    def shutdown(self) -> None:
        self.host_exec.shutdown()
        self.storage_exec.shutdown()

    def reset(self) -> None:
        self.host_exec.reset()
        self.storage_exec.drain()
