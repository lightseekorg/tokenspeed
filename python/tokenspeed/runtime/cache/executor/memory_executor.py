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

from tokenspeed.runtime.cache.executor.host_executor import HostExecutor
from tokenspeed.runtime.cache.executor.storage_executor import StorageExecutor
from tokenspeed.runtime.cache.kv_cache_host import (
    MHATokenToKVPoolHost,
    MLATokenToKVPoolHost,
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


class MemoryExecutor:
    """Coordinate host-memory and storage-backed cache operations."""

    def __init__(
        self,
        device_pool,
        config: MemoryExecutorConfig,
        is_dp_attention_enabled: bool,
        tp_group=None,
        draft_device_pool=None,
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

        if isinstance(actual_pool, _mha_types):
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
                f"host_pool only supports MHA and MLA, got {type(actual_pool)} "
                f"from module {type(actual_pool).__module__}"
            )

        # Draft model L2 cache: draft shares the same page mapping as the base
        # model, so its host pool must hold exactly the same number of tokens.
        # Pass host_size_tokens directly to bypass ratio/GB recalculation.
        if draft_device_pool is not None:
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

    def submit_plan(self, plan) -> None:
        if plan.cache:
            logger.debug("[cache_op] submit_plan: %s cache ops", len(plan.cache))
        for op in plan.cache:
            self.submit(op)
        self.host_exec.flush()

    def submit(self, op) -> None:
        if isinstance(op, Cache.WriteBackOp):
            logger.debug(
                "[cache_op] writeback op_id=%s src_pages=%s dst_pages=%s",
                op.op_ids,
                len(op.src_pages),
                len(op.dst_pages),
            )
            for i in range(len(op.op_ids)):
                op_id = op.op_ids[i]
                src_pages = op.src_pages[i]
                dst_pages = op.dst_pages[i]
                is_retract = bool(getattr(op, "is_retract", [False])[i])
                self.host_exec.enqueue_writeback(
                    op_id, src_pages, dst_pages, is_retract=is_retract
                )
        elif isinstance(op, Cache.LoadBackOp):
            logger.debug(
                "[cache_op] loadback op_id=%s src_pages=%s dst_pages=%s",
                op.op_ids,
                len(op.src_pages),
                len(op.dst_pages),
            )
            for i in range(len(op.op_ids)):
                op_id = op.op_ids[i]
                src_pages = op.src_pages[i]
                dst_pages = op.dst_pages[i]
                self.host_exec.enqueue_loadback(op_id, src_pages, dst_pages)
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

    def get_producer_index(self, op_id: int) -> Optional[int]:
        return self.host_exec.get_producer_index(op_id)

    def set_consumer(self, producer_index: int | Iterable[int]) -> None:
        self.host_exec.set_consumer(producer_index)

    def get_draft_producer_index(self, op_id: int) -> Optional[int]:
        return self.host_exec.get_draft_producer_index(op_id)

    def set_draft_consumer(self, producer_index: int | Iterable[int]) -> None:
        self.host_exec.set_draft_consumer(producer_index)

    def query_l3_pages(self, hashes: list[str]) -> int:
        return self.storage_exec.query_exists(hashes)

    def shutdown(self) -> None:
        self.host_exec.shutdown()
        self.storage_exec.shutdown()

    def reset(self) -> None:
        self.host_exec.reset()
        self.storage_exec.drain()
