"""Tests for Mooncake L3 (disk-tier) KV event publishing on backup/clear."""

from __future__ import annotations

import importlib.util
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from queue import Queue
from types import SimpleNamespace
from unittest.mock import MagicMock

from tokenspeed.runtime.pd.kv_events import (
    AllBlocksCleared,
    BlockStored,
    KVEventsConfig,
    apply_envelope,
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


_MISSING = object()


def _load_mooncake_store_cls():
    """Load MooncakeStore without importing the stubbed ``cache.storage`` package."""
    import types

    sys.modules.setdefault("torch", MagicMock())
    sys.modules.setdefault("requests", MagicMock())
    sys.modules.setdefault("tokenspeed_kernel", MagicMock())
    sys.modules.setdefault("tokenspeed_kernel.platform", MagicMock())
    sys.modules.setdefault("tokenspeed.runtime.cache.kv_cache_host", MagicMock())

    mod_name = "mooncake_store_kv_events_l3"
    existing = sys.modules.get(mod_name)
    if existing is not None and hasattr(existing, "MooncakeStore"):
        return existing.MooncakeStore

    path = (
        Path(__file__).resolve().parents[2]
        / "python"
        / "tokenspeed"
        / "runtime"
        / "cache"
        / "storage"
        / "mooncake_store"
        / "mooncake_store.py"
    )

    # Temporarily stub import deps for this load only, then restore so sibling
    # tests (e.g. ``test_kv_events.py``) can still import real ``utils.env``.
    saved: dict[str, object] = {}
    kvstore = types.ModuleType("tokenspeed.runtime.cache.kvstore_storage")
    kvstore.KVStoreStorage = object
    kvstore.KVStoreStorageConfig = type("KVStoreStorageConfig", (), {})
    kvstore.KVStoreStorageExtraInfo = type("KVStoreStorageExtraInfo", (), {})

    env_mod = types.ModuleType("tokenspeed.runtime.utils.env")
    env_mod.envs = MagicMock()

    stubs = {
        "tokenspeed.runtime.cache.kvstore_storage": kvstore,
        "tokenspeed.runtime.utils.env": env_mod,
    }

    for key, stub in stubs.items():
        saved[key] = sys.modules.get(key, _MISSING)
        sys.modules[key] = stub

    try:
        spec = importlib.util.spec_from_file_location(mod_name, path)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
        return mod.MooncakeStore
    finally:
        for key, prev in saved.items():
            if prev is _MISSING:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = prev


def test_mooncake_store_clear_invokes_on_clear_callback() -> None:
    """MooncakeStore.clear() calls optional on_clear after store.remove_all()."""
    MooncakeStore = _load_mooncake_store_cls()

    store = object.__new__(MooncakeStore)
    store.store = MagicMock()
    called: list[bool] = []
    store.on_clear = lambda: called.append(True)

    store.clear()

    store.store.remove_all.assert_called_once_with()
    assert called == [True]


def test_mooncake_store_clear_without_on_clear_is_safe() -> None:
    MooncakeStore = _load_mooncake_store_cls()

    store = object.__new__(MooncakeStore)
    store.store = MagicMock()
    store.on_clear = None

    store.clear()

    store.store.remove_all.assert_called_once_with()


def test_storage_executor_wires_on_l3_all_cleared_to_backend() -> None:
    """StorageExecutor assigns on_l3_all_cleared onto backend.on_clear."""
    _stub_storage_executor_deps()
    from tokenspeed.runtime.cache.executor.storage_executor import StorageExecutor

    received: list[bool] = []

    def on_cleared() -> None:
        received.append(True)

    executor = object.__new__(StorageExecutor)
    executor._on_l3_all_cleared = on_cleared
    backend = SimpleNamespace(on_clear=None)
    executor.storage_backend = backend
    executor._wire_l3_clear_callback()

    assert backend.on_clear is on_cleared
    backend.on_clear()
    assert received == [True]


def test_mooncake_clear_enqueues_all_blocks_cleared_via_callback() -> None:
    """End-to-end: Mooncake clear → on_clear → pending AllBlocksCleared(disk)."""
    MooncakeStore = _load_mooncake_store_cls()

    pending: Queue = Queue()

    def on_l3_all_cleared() -> None:
        pending.put(AllBlocksCleared())

    store = object.__new__(MooncakeStore)
    store.store = MagicMock()
    store.on_clear = on_l3_all_cleared

    store.clear()

    event = pending.get_nowait()
    assert isinstance(event, AllBlocksCleared)
    config = KVEventsConfig(
        wire_format="rfc1527",
        backend_id="worker-0",
        publish_medium=True,
        publish_tiers=["gpu", "disk"],
    )
    apply_envelope(event, config, medium="disk", dp_rank=0)
    assert event.medium == "disk"
    assert event.backend_id == "worker-0"


def test_apply_envelope_all_blocks_cleared_medium_disk() -> None:
    event = AllBlocksCleared()
    config = KVEventsConfig(
        wire_format="rfc1527",
        backend_id="w",
        publish_medium=True,
    )

    annotated = apply_envelope(event, config, medium="disk", dp_rank=0)

    assert annotated.medium == "disk"
    assert annotated.backend_id == "w"
