import json

import pytest

from tokenspeed.runtime.cache.storage.mooncake_store.mooncake_store import (
    DEFAULT_GLOBAL_SEGMENT_SIZE,
    MooncakeStore,
    MooncakeStoreConfig,
)


def test_inline_config_uses_stable_defaults():
    config = MooncakeStoreConfig.load_from_extra_config(
        {"master_server_address": "10.0.0.1:50051"}
    )

    assert config == MooncakeStoreConfig(
        local_hostname="localhost",
        metadata_server="P2PHANDSHAKE",
        global_segment_size=DEFAULT_GLOBAL_SEGMENT_SIZE,
        protocol="tcp",
        device_name="",
        master_server_address="10.0.0.1:50051",
        master_metrics_port=9003,
        check_server=False,
    )


def test_inline_config_does_not_read_mooncake_environment(monkeypatch):
    monkeypatch.setenv("MOONCAKE_MASTER", "environment.invalid:1")
    monkeypatch.setenv("MOONCAKE_LOCAL_HOSTNAME", "environment-host")
    monkeypatch.setenv("LOCAL_HOSTNAME", "legacy-environment-host")
    monkeypatch.setenv("MOONCAKE_PROTOCOL", "rdma")

    config = MooncakeStoreConfig.load_from_extra_config(
        {"master_server_address": "explicit.example:50051"}
    )

    assert config.master_server_address == "explicit.example:50051"
    assert config.local_hostname == "localhost"
    assert config.protocol == "tcp"


def test_config_path_loads_explicit_json(tmp_path):
    path = tmp_path / "mooncake.json"
    path.write_text(
        json.dumps(
            {
                "local_hostname": "worker-0",
                "metadata_server": "metadata.example:2379",
                "global_segment_size": "8gb",
                "protocol": "rdma",
                "device_name": "mlx5_0",
                "master_server_address": "master.example:50051",
                "master_metrics_port": 9103,
                "check_server": True,
            }
        ),
        encoding="utf-8",
    )

    config = MooncakeStoreConfig.load_from_extra_config({"config_path": str(path)})

    assert config.local_hostname == "worker-0"
    assert config.global_segment_size == 8 * 1024**3
    assert config.protocol == "rdma"
    assert config.check_server is True


def test_config_path_cannot_mix_with_inline_fields(tmp_path):
    path = tmp_path / "mooncake.json"
    path.write_text(
        json.dumps({"master_server_address": "master.example:50051"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="cannot be combined"):
        MooncakeStoreConfig.load_from_extra_config(
            {
                "config_path": str(path),
                "master_server_address": "other.example:50051",
            }
        )


def test_zero_global_segment_size_keeps_zero_copy_only_mode():
    config = MooncakeStoreConfig.load_from_extra_config(
        {
            "master_server_address": "master.example:50051",
            "global_segment_size": 0,
        }
    )

    assert config.global_segment_size == 0


@pytest.mark.parametrize(
    ("extra_config", "message"),
    [
        ({}, "non-empty extra_config"),
        ({"protocol": "tcp"}, "master_server_address is required"),
        (
            {
                "master_server_address": "master.example:50051",
                "master_server_adress": "typo.example:50051",
            },
            "Unknown Mooncake extra_config field",
        ),
        (
            {
                "master_server_address": "master.example:50051",
                "global_segment_size": -1,
            },
            "must be non-negative",
        ),
        (
            {
                "master_server_address": "master.example:50051",
                "check_server": "false",
            },
            "must be a boolean",
        ),
        (
            {
                "master_server_address": "master.example:50051",
                "extra_backend_tag": "",
            },
            "must be a non-empty string",
        ),
    ],
)
def test_invalid_extra_config_fails_closed(extra_config, message):
    with pytest.raises(ValueError, match=message):
        MooncakeStoreConfig.load_from_extra_config(extra_config)


def test_store_requires_explicit_config_before_loading_mooncake():
    with pytest.raises(ValueError, match="requires explicit configuration"):
        MooncakeStore()
