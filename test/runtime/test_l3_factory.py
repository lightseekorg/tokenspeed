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

"""The L3 settings the data plane resolves for the engine host: namespace
identity, the Host pool and the store, and the executor they start
(``tokenspeed_engine.L3Executor``)."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from test.l3_test_utils import StorageOp, run_storage

import pytest
import torch
from tokenspeed_engine import L3Executor
from tokenspeed_scheduler import StorageKind
from transformers import Qwen3Config

from tokenspeed.runtime.cache.host_storage import HostCacheResources, HostCacheStorage
from tokenspeed.runtime.cache.l3.factory import l3_settings
from tokenspeed.runtime.cache.transfer.layout import (
    CacheField,
    CacheGroupLayout,
    CacheTransferLayout,
)
from tokenspeed.runtime.cache.transfer.ownership import BlockOwnerTranslation
from tokenspeed.runtime.configs.model_config import ModelConfig
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
    CacheFieldLayout,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
    CacheGroupLayout as MemoryGroupLayout,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
    CacheMemoryPlan,
    CachePlaneLayout,
)
from tokenspeed.runtime.utils.server_args import ServerArgs

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=10,
    suite="runtime-1gpu",
    nightly=False,
    disabled=None,
    disabled_on_runners=None,
    disabled_on_runners_reason=None,
)


def checkpoint(path: Path, *, payload: bytes) -> None:
    path.mkdir()
    Qwen3Config(
        architectures=["Qwen3ForCausalLM"],
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=32,
    ).save_pretrained(path)
    # The factory fingerprints local bytes; it never deserializes model weights.
    (path / "model.safetensors").write_bytes(payload)


@pytest.fixture
def construction(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("the pinned Host pool needs a CUDA device")
    path = tmp_path / "target"
    checkpoint(path, payload=b"target-checkpoint")
    args = ServerArgs(
        model=str(path),
        device="cuda",
        dtype="bfloat16",
        world_size=1,
        nprocs_per_node=1,
        attention_backend="triton",
        kvstore_storage_backend="memory",
        weight_version="v1",
    )
    args.mapping.rank = 0
    model = ModelConfig(
        str(path),
        server_args=args,
        model_override_args='{"rope_scaling": null}',
        dtype="bfloat16",
    )
    groups = (CacheGroupLayout("full", 1, (CacheField("layer.0.k", 0, 0, 4, 4),)),)
    # Sixteen 4-byte LCM blocks; the device buffer only shapes the layout.
    storage = HostCacheStorage(
        CacheTransferLayout(
            num_lcm_blocks=16,
            groups=groups,
            buffers=(torch.zeros(68, dtype=torch.uint8),),
            consumers=(("layer.0.k",),),
        ),
        num_host_lcm_blocks=16,
    )
    host_cache = HostCacheResources(
        storage=storage,
        ownership=BlockOwnerTranslation(
            shard_counts=(1,),
            device_virtual_counts=(17,),
            host_virtual_counts=(17,),
            rank=0,
        ),
        groups=groups,
        memory_plan=CacheMemoryPlan(
            prefix_granularity=2,
            lcm_block_bytes=4,
            num_lcm_blocks=16,
            groups=(MemoryGroupLayout("full", 1, 16),),
            planes=(CachePlaneLayout("kv", 4, 0),),
            fields=(
                CacheFieldLayout(
                    "full", "layer.0.k", "kv", (1, 1, 2), "bfloat16", 0, 4
                ),
            ),
        ),
        attention_backend_name="triton",
        draft_attention_backend_name="",
    )
    return args, model, storage, {"host_cache": host_cache, "world_cpu_group": None}


def test_settings_place_the_shared_host_pool(construction):
    args, model, storage, factory_kwargs = construction
    settings = l3_settings(args, model, None, **factory_kwargs)
    assert settings.host_buffer == storage.host_buffer.data_ptr()
    assert settings.host_buffer_bytes == 64
    assert settings.groups == ((1, 4, 1),)
    assert (settings.lcm_block_bytes, settings.num_lcm_blocks) == (4, 16)
    assert (settings.owner_rank, settings.tp_rank) == (0, 0)
    assert (settings.backend, settings.mooncake) == ("memory", None)
    assert settings.runtime.scheduler.max_host_blocks <= storage.num_host_lcm_blocks


def test_settings_start_an_executor_over_the_shared_host_pool(construction):
    args, model, storage, factory_kwargs = construction
    executor = L3Executor(l3_settings(args, model, None, **factory_kwargs))
    # The pinned pool starts uninitialized.
    host = storage.host_buffer
    host.zero_()
    key = [(0, "prefix", 0)]
    try:
        host[:4] = torch.frombuffer(bytearray(b"KV01"), dtype=torch.uint8)
        put = run_storage(executor, StorageOp(1, StorageKind.Put, key, [1]))
        assert put[1:] == ([True], True, [False])
        host[:4] = torch.frombuffer(bytearray(b"xxxx"), dtype=torch.uint8)
        get = run_storage(executor, StorageOp(2, StorageKind.Get, key, [1]))
        assert get[1:] == ([True], True, [False])
        assert bytes(host[:8].tolist()) == b"KV01" + bytes(4)
        assert executor.delete_namespace()
    finally:
        executor.close(5.0)


def test_settings_freeze_namespace_inputs_before_weight_version_switch(construction):
    args, model, _, factory_kwargs = construction
    executor = L3Executor(l3_settings(args, model, None, **factory_kwargs))
    try:
        original = executor.key_prefix()
        executor.set_weight_version("v2")
        expected = executor.key_prefix()
        assert expected != original
        executor.set_weight_version("v1")
        model.model_override_args["new_override"] = {"nested": [1, 2]}
        args.disable_tf32 = not args.disable_tf32
        factory_kwargs["host_cache"] = replace(
            factory_kwargs["host_cache"], attention_backend_name="changed-backend"
        )
        executor.set_weight_version("v2")
        assert executor.key_prefix() == expected
        changed = L3Executor(l3_settings(args, model, None, **factory_kwargs))
        try:
            changed.set_weight_version("v2")
            assert changed.key_prefix() != expected
        finally:
            changed.close(5.0)
    finally:
        executor.close(5.0)


def test_settings_include_the_draft_checkpoint_identity(construction, tmp_path):
    args, model, _, factory_kwargs = construction
    path = tmp_path / "draft"
    checkpoint(path, payload=b"draft-checkpoint")
    draft = ModelConfig(
        str(path), server_args=args, model_override_args="{}", dtype="bfloat16"
    )
    target_only = l3_settings(args, model, None, **factory_kwargs)
    with_draft = l3_settings(args, model, draft, **factory_kwargs)
    assert not target_only.has_draft and with_draft.has_draft
    assert target_only.deployment_id != with_draft.deployment_id


def test_settings_refuse_head_partitioned_payload_on_cyclic_owners(construction):
    args, model, _, factory_kwargs = construction
    ownership = BlockOwnerTranslation(
        shard_counts=(2,),
        device_virtual_counts=(17,),
        host_virtual_counts=(17,),
        rank=0,
    )
    factory_kwargs["host_cache"] = replace(
        factory_kwargs["host_cache"], ownership=ownership
    )
    with pytest.raises(ValueError, match="layer.0.k is head-partitioned"):
        l3_settings(args, model, None, **factory_kwargs)


def test_settings_validate_limits_against_actual_host_capacity(construction):
    args, model, storage, factory_kwargs = construction
    args.kvstore_storage_backend_extra_config = json.dumps(
        {"async": {"max_host_blocks": storage.num_host_lcm_blocks + 1}}
    )
    with pytest.raises(ValueError, match="exceeds usable Host parents"):
        l3_settings(args, model, None, **factory_kwargs)


def test_unconfigured_storage_needs_neither_host_memory_nor_checkpoint_io(construction):
    args, model, _, _ = construction
    args.kvstore_storage_backend = None
    Path(model.model_path, "model.safetensors").unlink()
    assert l3_settings(args, model, None, host_cache=None, world_cpu_group=None) is None


def test_configured_storage_requires_host_memory(construction):
    args, model, _, _ = construction
    with pytest.raises(ValueError, match="requires a Host L2 allocation"):
        l3_settings(args, model, None, host_cache=None, world_cpu_group=None)


def test_an_unknown_store_is_refused(construction):
    args, model, _, factory_kwargs = construction
    args.kvstore_storage_backend = "nfs"
    with pytest.raises(ValueError, match="unsupported KVStore storage backend"):
        l3_settings(args, model, None, **factory_kwargs)


def test_mooncake_settings_take_this_ranks_segment_share(construction, monkeypatch):
    args, model, _, factory_kwargs = construction
    monkeypatch.setenv("TOKENSPEED_MOONCAKE_STORE_LIBRARY", "/opt/libmooncake_store.so")
    args.kvstore_storage_backend = "mooncake"
    args.kvstore_storage_backend_extra_config = json.dumps(
        {
            "master_server_address": "127.0.0.1:50051",
            "global_segment_size": "2gb",
            "tenant_id": "tenant-a",
        }
    )
    mooncake = l3_settings(args, model, None, **factory_kwargs).mooncake
    assert mooncake.library == "/opt/libmooncake_store.so"
    assert mooncake.master_server_address == "127.0.0.1:50051"
    assert mooncake.global_segment_size == 2 * 1024**3
    assert mooncake.tenant_id == "tenant-a"


def test_settings_resolve_draft_eagle_layers_and_explicit_override(
    construction, tmp_path
):
    args, model, _, factory_kwargs = construction
    path = tmp_path / "draft-eagle"
    checkpoint(path, payload=b"draft-checkpoint")
    config_path = path / "config.json"
    config = json.loads(config_path.read_text())
    config["eagle_config"] = {"eagle_aux_hidden_state_layer_ids": [0]}
    config_path.write_text(json.dumps(config))
    draft = ModelConfig(
        str(path), server_args=args, model_override_args="{}", dtype="bfloat16"
    )
    args.speculative_algorithm = "EAGLE3"
    args.eagle3_layers_to_capture = None
    automatic = l3_settings(args, model, draft, **factory_kwargs)
    args.eagle3_layers_to_capture = [0]
    explicit = l3_settings(args, model, draft, **factory_kwargs)
    args.eagle3_layers_to_capture = [1]
    changed = l3_settings(args, model, draft, **factory_kwargs)
    assert automatic.deployment_id == explicit.deployment_id
    assert changed.deployment_id != automatic.deployment_id
