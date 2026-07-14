"""Tests for Mooncake master KV-events config parse and relay scaffold."""

from __future__ import annotations

import importlib.util
import logging
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tokenspeed.runtime.pd.kv_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
)
from tokenspeed.runtime.pd.mooncake_kv_events import (
    MooncakeKvEventsConfig,
    MooncakeMasterEventSubscriber,
    engine_publishes_l3_disk,
    normalize_master_event,
    parse_mooncake_kv_events_config,
)

_MISSING = object()


def _load_mooncake_store_config_cls():
    """Load MooncakeStoreConfig without requiring torch (stubs for deps)."""
    sys.modules.setdefault("torch", MagicMock())
    sys.modules.setdefault("requests", MagicMock())
    sys.modules.setdefault("tokenspeed.runtime.cache.kv_cache_host", MagicMock())

    mod_name = "mooncake_store_kv_events_config"
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

    kvstore = types.ModuleType("tokenspeed.runtime.cache.kvstore_storage")
    kvstore.KVStoreStorage = object
    kvstore.KVStoreStorageConfig = type("KVStoreStorageConfig", (), {})
    kvstore.KVStoreStorageExtraInfo = type("KVStoreStorageExtraInfo", (), {})

    env_mod = types.ModuleType("tokenspeed.runtime.utils.env")
    env_mod.envs = MagicMock()
    # Defaults used by load_from_extra_config
    for name in (
        "MOONCAKE_LOCAL_HOSTNAME",
        "MOONCAKE_TE_META_DATA_SERVER",
        "MOONCAKE_GLOBAL_SEGMENT_SIZE",
        "MOONCAKE_PROTOCOL",
        "MOONCAKE_DEVICE",
        "MOONCAKE_MASTER_METRICS_PORT",
        "MOONCAKE_CHECK_SERVER",
    ):
        attr = MagicMock()
        attr.default = (
            "localhost"
            if "HOSTNAME" in name
            else (
                "P2PHANDSHAKE"
                if "META" in name
                else (
                    16 * 1024 * 1024 * 1024
                    if "SEGMENT" in name
                    else (
                        "tcp"
                        if "PROTOCOL" in name
                        else ("" if "DEVICE" in name else (9003 if "METRICS" in name else True))
                    )
                )
            )
        )
        setattr(env_mod.envs, name, attr)

    stubs = {
        "tokenspeed.runtime.cache.kvstore_storage": kvstore,
        "tokenspeed.runtime.utils.env": env_mod,
    }
    saved: dict[str, object] = {}
    for key, stub in stubs.items():
        saved[key] = sys.modules.get(key, _MISSING)
        sys.modules[key] = stub

    try:
        # Force fresh load so optional field additions are visible across runs.
        sys.modules.pop(mod_name, None)
        spec = importlib.util.spec_from_file_location(mod_name, path)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
        return mod.MooncakeStoreConfig
    finally:
        for key, prev in saved.items():
            if prev is _MISSING:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = prev


def test_parse_mooncake_kv_events_config_defaults_to_engine() -> None:
    cfg = parse_mooncake_kv_events_config({})
    assert cfg.source == "engine"
    assert cfg.master_subscribe_endpoint is None
    assert cfg.backend_id is None


def test_parse_mooncake_kv_events_config_from_nested_section() -> None:
    cfg = parse_mooncake_kv_events_config(
        {
            "master_server_address": "127.0.0.1:50051",
            "kv_events": {
                "source": "master",
                "master_subscribe_endpoint": "tcp://mooncake-master:6000",
                "backend_id": "node-3-cache-daemon",
            },
        }
    )
    assert cfg == MooncakeKvEventsConfig(
        source="master",
        master_subscribe_endpoint="tcp://mooncake-master:6000",
        backend_id="node-3-cache-daemon",
    )


def test_parse_mooncake_kv_events_config_none_and_missing_are_engine() -> None:
    assert parse_mooncake_kv_events_config(None).source == "engine"
    assert parse_mooncake_kv_events_config({"kv_events": {}}).source == "engine"


def test_parse_mooncake_kv_events_config_rejects_invalid_source() -> None:
    with pytest.raises(ValueError, match="source"):
        parse_mooncake_kv_events_config({"kv_events": {"source": "invalid"}})


def test_mooncake_store_config_loads_kv_events_optional_fields() -> None:
    MooncakeStoreConfig = _load_mooncake_store_config_cls()
    store_cfg = MooncakeStoreConfig.load_from_extra_config(
        {
            "master_server_address": "10.0.0.1:50051",
            "kv_events": {
                "source": "both",
                "master_subscribe_endpoint": "tcp://127.0.0.1:6000",
                "backend_id": "cache-daemon-1",
            },
        }
    )
    assert store_cfg.kv_events_source == "both"
    assert store_cfg.kv_events_master_subscribe_endpoint == "tcp://127.0.0.1:6000"
    assert store_cfg.kv_events_backend_id == "cache-daemon-1"


def test_mooncake_store_config_without_kv_events_stays_compatible() -> None:
    MooncakeStoreConfig = _load_mooncake_store_config_cls()
    store_cfg = MooncakeStoreConfig.load_from_extra_config(
        {"master_server_address": "10.0.0.1:50051"}
    )
    assert store_cfg.kv_events_source == "engine"
    assert store_cfg.kv_events_master_subscribe_endpoint is None
    assert store_cfg.kv_events_backend_id is None


def test_engine_publishes_l3_disk_by_source() -> None:
    assert engine_publishes_l3_disk(parse_mooncake_kv_events_config(None)) is True
    assert (
        engine_publishes_l3_disk(
            parse_mooncake_kv_events_config({"kv_events": {"source": "engine"}})
        )
        is True
    )
    assert (
        engine_publishes_l3_disk(
            parse_mooncake_kv_events_config({"kv_events": {"source": "both"}})
        )
        is True
    )
    assert (
        engine_publishes_l3_disk(
            parse_mooncake_kv_events_config({"kv_events": {"source": "master"}})
        )
        is False
    )


def test_normalize_master_event_stored_rfc1527_fields() -> None:
    event = normalize_master_event(
        {
            "event_type": "stored",
            "seq_hashes": [111, 222],
            "parent_hash": 999,
            "token_ids": [1, 2, 3, 4],
            "block_size": 2,
            "backend_id": "node-3-cache-daemon",
            "medium": "disk",
            "dp_rank": 0,
            "model_name": "test-model",
            "tenant_id": "default",
            "event_id": 42,
        }
    )
    assert isinstance(event, BlockStored)
    assert event.block_hashes == [111, 222]
    assert event.parent_block_hash == 999
    assert event.token_ids == [1, 2, 3, 4]
    assert event.block_size == 2
    assert event.backend_id == "node-3-cache-daemon"
    assert event.medium == "disk"
    assert event.dp_rank == 0
    assert event.model_name == "test-model"
    assert event.tenant_id == "default"
    assert event.event_id == 42


def test_normalize_master_event_removed_accepts_legacy_aliases() -> None:
    event = normalize_master_event(
        {
            "type": "BlockRemoved",
            "block_hashes": [7, 8],
            "backend_id": "daemon",
            "medium": "disk",
        }
    )
    assert isinstance(event, BlockRemoved)
    assert event.block_hashes == [7, 8]
    assert event.backend_id == "daemon"
    assert event.medium == "disk"


def test_normalize_master_event_cleared() -> None:
    event = normalize_master_event(
        {"event_type": "cleared", "backend_id": "daemon", "medium": "disk"}
    )
    assert isinstance(event, AllBlocksCleared)
    assert event.backend_id == "daemon"
    assert event.medium == "disk"


def test_normalize_master_event_rejects_unknown_type() -> None:
    with pytest.raises(ValueError, match="event_type|type"):
        normalize_master_event({"event_type": "unknown", "seq_hashes": [1]})


def test_subscriber_start_engine_source_is_noop() -> None:
    sub = MooncakeMasterEventSubscriber(
        parse_mooncake_kv_events_config({"kv_events": {"source": "engine"}})
    )
    sub.start()
    assert sub.is_active is False
    sub.stop()


def test_subscriber_start_master_without_endpoint_fail_open(
    caplog: pytest.LogCaptureFixture,
) -> None:
    cfg = MooncakeKvEventsConfig(source="master", master_subscribe_endpoint=None)
    sub = MooncakeMasterEventSubscriber(cfg)
    with caplog.at_level(logging.WARNING):
        sub.start()
    assert sub.is_active is False
    assert any("endpoint" in r.message.lower() for r in caplog.records)
    sub.stop()


def test_subscriber_start_master_with_endpoint_does_not_hang(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Missing Mooncake event SDK → warn and stay idle (fail-open), no hang."""
    cfg = MooncakeKvEventsConfig(
        source="both",
        master_subscribe_endpoint="tcp://127.0.0.1:9",
        backend_id="test-daemon",
    )
    sub = MooncakeMasterEventSubscriber(cfg)
    with caplog.at_level(logging.WARNING):
        sub.start()
    assert sub.is_active is False
    assert len(caplog.records) >= 1
    sub.stop()
