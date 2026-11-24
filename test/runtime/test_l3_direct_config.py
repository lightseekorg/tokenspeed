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

"""Direct source settings validate the storage/admission ownership boundary."""

import json
from dataclasses import replace

import pytest

from tokenspeed.runtime.cache.l3.config import (
    L3Settings,
    MooncakeStoreSettings,
    StorageRuntimeConfig,
)
from tokenspeed.runtime.cache.l3.direct import DirectSourceSettings


def test_registration_bound_is_explicit_and_validated():
    values = {"master_server_address": "test-master:1234", "global_segment_size": 0}

    def resolve_store():
        return MooncakeStoreSettings.resolve(
            values, library="store.so", tp_size=1, pp_size=1
        )

    assert resolve_store().registration_max_bytes is None
    values["registration_max_bytes"] = 2**30
    assert resolve_store().registration_max_bytes == 2**30
    for invalid in [0, -1, True, "1024", 2**63]:
        values["registration_max_bytes"] = invalid
        with pytest.raises(ValueError, match="registration_max_bytes"):
            resolve_store()


def settings():
    runtime = StorageRuntimeConfig.from_extra_config(
        None, host_blocks=16, prefix_granularity=128, restore_to_host=False
    )
    return L3Settings(
        runtime=runtime,
        deployment_id="schema",
        has_draft=False,
        weight_version="v1",
        host_buffer=4096,
        host_buffer_bytes=1024,
        groups=((1, 64, 1),),
        lcm_block_bytes=64,
        num_lcm_blocks=16,
        owner_rank=0,
        tp_rank=0,
        backend="mooncake",
        mooncake=MooncakeStoreSettings(
            library="test-store.so",
            local_hostname="test-reader",
            metadata_server="P2PHANDSHAKE",
            global_segment_size=0,
            local_buffer_size=4096,
            protocol="tcp",
            device_name="",
            master_server_address="test-master:1234",
            tenant_id="test",
            registration_max_bytes=None,
        ),
    )


def config():
    return dict(
        library="test-offload.so",
        master="test-master:1234",
        tenant="test",
        rpc_timeout_ms=1000,
        max_query_keys=256,
        max_candidates=4,
        max_prepare_bytes=2**20,
        max_source_batch_bytes=2**19,
        max_source_batches=8,
        source_budget_bytes=2**21,
        max_records=16,
        record_lifetime_ms=10000,
        read_budget_ms=2000,
        max_restore_tokens=8192,
    )


def resolve(values, l3):
    return DirectSourceSettings.resolve(
        json.dumps(values),
        l3,
        prefix_granularity=128,
        max_model_len=131072,
    )


def test_direct_storage_writes_keep_their_namespace_and_restore_policy():
    l3 = settings()
    result = resolve(config(), l3)
    assert result.l3 is l3
    assert not result.l3.runtime.scheduler.restore_to_host
    assert json.loads(result.config)["master"] == l3.mooncake.master_server_address
    assert result.scheduler_limits["storage_min_fetch_tokens"] == 4096
    assert result.prefix_granularity == 128


def test_omitted_deadlines_use_defaults_and_explicit_values_override():
    values = config()
    deadlines = ["rpc_timeout_ms", "record_lifetime_ms", "read_budget_ms"]
    for key in deadlines:
        del values[key]
    resolved = json.loads(resolve(values, settings()).config)
    assert [resolved[key] for key in deadlines] == [2000, 15000, 5000]
    values["rpc_timeout_ms"] = 750
    values["read_budget_ms"] = 3000
    resolved = json.loads(resolve(values, settings()).config)
    assert [resolved[key] for key in deadlines] == [750, 15000, 3000]
    values["read_budget_ms"] = 15000
    with pytest.raises(ValueError, match="fit inside record lifetime"):
        resolve(values, settings())


@pytest.mark.parametrize(
    "key,value",
    [
        ("port", 17005),
        ("port", 65530),
        ("max_query_keys", 1025),
        ("rpc_timeout_ms", 10001),
        ("max_source_batch_bytes", 2**31 + 1),
        ("source_budget_bytes", 512),
        ("read_budget_ms", 10000),
        ("tenant", "bad\0tenant"),
        ("max_records", True),
        ("unknown", 1),
    ],
)
def test_invalid_source_budgets_are_rejected(key, value):
    values = config()
    values[key] = value
    with pytest.raises(ValueError):
        resolve(values, settings())


def test_a_host_get_and_direct_preparation_cannot_be_selected_together():
    l3 = settings()
    host = StorageRuntimeConfig.from_extra_config(
        None, host_blocks=16, prefix_granularity=128, restore_to_host=True
    )
    with pytest.raises(ValueError, match="without Host restore"):
        resolve(config(), replace(l3, runtime=host))


def test_minimal_direct_config_inherits_store_without_an_endpoint():
    l3 = settings()
    result = DirectSourceSettings.resolve(
        None, l3, prefix_granularity=128, max_model_len=131072
    )
    values = json.loads(result.config)
    assert values["master"] == l3.mooncake.master_server_address
    assert values["tenant"] == l3.mooncake.tenant_id
    assert "port" not in values
    assert values["max_restore_tokens"] == 65536
    assert values["library"] == "libtokenspeed_mooncake_offload.so"
    assert result.scheduler_limits["storage_lookup_timeout_us"] == 500000


def test_direct_connection_cannot_diverge_from_store():
    for field in ("master", "tenant"):
        with pytest.raises(ValueError, match="share master and tenant"):
            resolve({field: "another"}, settings())


def test_small_context_does_not_force_storage_queries():
    result = DirectSourceSettings.resolve(
        None, settings(), prefix_granularity=128, max_model_len=1024
    )
    assert result.scheduler_limits["storage_max_tokens"] == 1023
    assert result.scheduler_limits["storage_min_fetch_tokens"] == 4096


def test_content_identity_remains_the_default_and_release_ids_are_explicit(
    tmp_path, monkeypatch
):
    from tokenspeed.runtime.cache.l3.identity import l3_checkpoint_id

    snapshot = tmp_path / "models--example--model" / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True)
    (snapshot.parent.parent / "refs").mkdir()

    def identify():
        return l3_checkpoint_id(
            str(snapshot),
            revision=None,
            hf_config=None,
            load_format="auto",
            model_loader_extra_config={},
            ext_yaml=None,
        )

    monkeypatch.delenv("TOKENSPEED_CHECKPOINT_IDENTITY_MODE", raising=False)
    default = identify()
    monkeypatch.setenv("TOKENSPEED_CHECKPOINT_IDENTITY_MODE", "content")
    assert identify() == default
    monkeypatch.setenv("TOKENSPEED_CHECKPOINT_IDENTITY_MODE", "release")
    monkeypatch.setenv(
        "TOKENSPEED_CHECKPOINT_RELEASES", json.dumps({str(snapshot): "immutable-v1"})
    )
    first = identify()
    assert first != default
    monkeypatch.setenv(
        "TOKENSPEED_CHECKPOINT_RELEASES", json.dumps({str(snapshot): "immutable-v2"})
    )
    assert identify() != first
    monkeypatch.setenv("TOKENSPEED_CHECKPOINT_RELEASES", "{}")
    with pytest.raises(KeyError):
        identify()
