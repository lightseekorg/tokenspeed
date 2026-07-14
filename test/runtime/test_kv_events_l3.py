"""Tests for Mooncake L3 (disk-tier) KV event publishing on backup success."""

from __future__ import annotations

import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from queue import Queue
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tokenspeed.runtime.pd.kv_events import (
    BlockStored,
    KVEventsConfig,
    l3_storage_keys_to_disk_events,
)


def test_kv_events_config_defaults_publish_tiers_to_gpu() -> None:
    config = KVEventsConfig()

    assert config.publish_tiers == ["gpu"]


def test_l3_keys_to_disk_events_chains_parents() -> None:
    events = l3_storage_keys_to_disk_events(["aa" * 32, "bb" * 32], block_size=16)

    assert len(events) == 2
    assert isinstance(events[0], BlockStored)
    assert events[0].parent_block_hash is None
    assert events[0].token_ids == []
    assert events[0].block_size == 16
    assert events[0].medium is None  # envelope applied by caller
    assert len(events[0].block_hashes) == 1
    assert events[1].parent_block_hash == events[0].block_hashes[0]
    assert events[1].block_hashes[0] != events[0].block_hashes[0]
    # Interim: stable u64 from XXH3-64(seed=1337) over the hex key bytes.
    import xxhash

    expected0 = xxhash.xxh3_64_intdigest(("aa" * 32).encode("utf-8"), seed=1337)
    assert events[0].block_hashes[0] == expected0


def test_l3_keys_to_disk_events_empty() -> None:
    assert l3_storage_keys_to_disk_events([]) == []


def _stub_storage_executor_deps() -> None:
    """Install lightweight stubs so StorageExecutor imports without torch/Mooncake."""
    if "tokenspeed.runtime.cache.executor.storage_executor" in sys.modules:
        return

    torch_mod = MagicMock()
    torch_mod.distributed = MagicMock()
    sys.modules.setdefault("torch", torch_mod)
    sys.modules.setdefault("torch.distributed", torch_mod.distributed)

    scheduler = MagicMock()
    cache = MagicMock()
    cache.BackUpDoneEvent = lambda: SimpleNamespace(
        op_id=0, success=False, request_id=None
    )
    cache.PrefetchDoneEvent = lambda: SimpleNamespace(
        op_id=0, success=False, request_id=None, completed_pages=0
    )
    scheduler.Cache = cache
    sys.modules.setdefault("tokenspeed_scheduler", scheduler)

    sys.modules.setdefault(
        "tokenspeed.runtime.layers.attention.kv_cache.mla", MagicMock()
    )
    sys.modules.setdefault(
        "tokenspeed.runtime.distributed.process_group_manager", MagicMock()
    )
    sys.modules.setdefault("tokenspeed.runtime.cache.kvstore_storage", MagicMock())
    sys.modules.setdefault("tokenspeed.runtime.cache.storage", MagicMock())
    host_exec = MagicMock()
    host_exec.page_ids_to_token_indices = MagicMock(
        return_value=MagicMock(name="host_indices")
    )
    sys.modules.setdefault("tokenspeed.runtime.cache.executor.host_executor", host_exec)
    utils = MagicMock()
    utils.get_colorful_logger = MagicMock(
        return_value=MagicMock(
            debug=MagicMock(),
            warning=MagicMock(),
            error=MagicMock(),
            exception=MagicMock(),
            info=MagicMock(),
        )
    )
    sys.modules.setdefault("tokenspeed.runtime.utils", utils)


def test_backup_done_enqueues_disk_block_stored() -> None:
    """StorageExecutor invokes on_l3_blocks_stored with rolling hashes on success."""
    _stub_storage_executor_deps()
    from tokenspeed.runtime.cache.executor.storage_executor import StorageExecutor

    received: list[list[str]] = []

    def on_l3(hashes: list[str]) -> None:
        received.append(list(hashes))

    # Build a minimal executor without running the full __init__ (avoids Mooncake).
    executor = object.__new__(StorageExecutor)
    executor.page_size = 16
    executor.storage_batch_size = 128
    executor.host_pool = MagicMock()
    executor.tp_size = 1
    executor._tp_group = None
    executor._results = Queue()
    executor._prefetch_op_to_rid = {}
    executor._aggregator_pending = Queue()
    executor._aggregator_stop = MagicMock()
    executor._aggregator_thread = None
    executor._on_l3_blocks_stored = on_l3
    executor._backup_hashes = {}
    executor.prefetch_timeout_base = 1.0
    executor.prefetch_timeout_per_page = 0.0

    backend = MagicMock()
    backend.batch_set_v1 = MagicMock(return_value=[True, True])
    executor.storage_backend = backend
    executor._executor = ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="test-l3-backup"
    )

    hashes = ["aa" * 32, "bb" * 32]
    op = SimpleNamespace(
        op_id=42,
        rolling_page_hashes=hashes,
        src_pages=[0, 1],
    )

    try:
        executor.submit_backup(op)
        # Wait until BackUpDoneEvent is drained (backup thread finished).
        deadline = time.monotonic() + 5.0
        done_evt = None
        while time.monotonic() < deadline:
            results = executor.drain()
            if results:
                done_evt = results[0]
                break
            time.sleep(0.01)
        assert done_evt is not None
        assert done_evt.success is True
        # Callback may race slightly after put; wait briefly for it.
        deadline = time.monotonic() + 2.0
        while not received and time.monotonic() < deadline:
            time.sleep(0.01)
        assert received == [hashes]
    finally:
        executor._executor.shutdown(wait=True)


def test_backup_done_skips_callback_on_failure() -> None:
    _stub_storage_executor_deps()
    from tokenspeed.runtime.cache.executor.storage_executor import StorageExecutor

    received: list[list[str]] = []
    executor = object.__new__(StorageExecutor)
    executor._results = Queue()
    executor._on_l3_blocks_stored = received.append
    executor._backup_hashes = {7: ["cc" * 32]}
    executor.storage_backend = MagicMock()

    failed = Future()
    failed.set_exception(RuntimeError("backup failed"))
    executor._on_backup_done(7, failed)

    results = executor.drain()
    assert len(results) == 1
    assert results[0].success is False
    assert received == []
    assert 7 not in executor._backup_hashes
