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

"""Host-only L3 construction preserves namespace and ownership behavior."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest
from transformers import Qwen3Config

from tokenspeed.runtime.cache.host_storage import HostCacheResources
from tokenspeed.runtime.cache.l3.executor import StorageJob
from tokenspeed.runtime.cache.l3.factory import create_l3_executor
from tokenspeed.runtime.cache.transfer.layout import CacheField, CacheGroupLayout
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


class Host:
    """CPU byte storage substitutes only for a GPU-pinned allocation."""

    def __init__(self):
        self.host_buffer = bytearray(64)
        self.host_cache_block_bytes = (4,)
        self.num_host_lcm_blocks = 16

    def host_block_range(self, group: int, block: int) -> tuple[int, int]:
        assert group == 0 and 1 <= block <= 16
        return (block - 1) * 4, 4


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
    host = Host()
    host_cache = HostCacheResources(
        storage=host,
        ownership=BlockOwnerTranslation(
            shard_counts=(1,),
            device_virtual_counts=(17,),
            host_virtual_counts=(17,),
            rank=0,
        ),
        groups=(CacheGroupLayout("full", 1, (CacheField("layer.0.k", 0, 0, 4, 4),)),),
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
    return args, model, host, {"host_cache": host_cache, "world_cpu_group": None}


def complete(executor):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        rows = executor.poll().completions
        if rows:
            assert len(rows) == 1
            return rows[0]
        time.sleep(0.001)
    raise AssertionError("L3 worker did not finish")


def test_factory_roundtrips_through_the_shared_host_resource(construction):
    args, model, host, factory_kwargs = construction
    executor = create_l3_executor(args, model, None, **factory_kwargs)
    try:
        assert executor.config.scheduler.max_host_blocks <= host.num_host_lcm_blocks
        host.host_buffer[:4] = b"KV01"
        key = executor._object_key("prefix", 0, 0)
        executor.submit(StorageJob(1, "put", (key,), (0,), (4,)))
        put = complete(executor)
        assert put.success == (True,) and put.retired
        host.host_buffer[:4] = b"xxxx"
        executor.submit(StorageJob(2, "get", (key,), (0,), (4,)))
        get = complete(executor)
        assert get.success == (True,) and get.retired
        assert host.host_buffer[:8] == b"KV01" + bytes(4)
    finally:
        executor.close(timeout_s=5)


def test_factory_freezes_namespace_inputs_before_weight_version_switch(construction):
    args, model, _, factory_kwargs = construction
    executor = create_l3_executor(args, model, None, **factory_kwargs)
    try:
        original = executor.key_prefix
        executor.set_weight_version("v2")
        expected = executor.key_prefix
        assert expected != original
        executor.set_weight_version("v1")
        model.model_override_args["new_override"] = {"nested": [1, 2]}
        args.disable_tf32 = not args.disable_tf32
        factory_kwargs["host_cache"] = replace(
            factory_kwargs["host_cache"], attention_backend_name="changed-backend"
        )
        executor.set_weight_version("v2")
        assert executor.key_prefix == expected
        changed = create_l3_executor(args, model, None, **factory_kwargs)
        try:
            changed.set_weight_version("v2")
            assert changed.key_prefix != expected
        finally:
            changed.close(timeout_s=5)
    finally:
        executor.close(timeout_s=5)


def test_factory_includes_the_draft_checkpoint_identity(construction, tmp_path):
    args, model, _, factory_kwargs = construction
    path = tmp_path / "draft"
    checkpoint(path, payload=b"draft-checkpoint")
    draft = ModelConfig(
        str(path), server_args=args, model_override_args="{}", dtype="bfloat16"
    )
    target_only = create_l3_executor(args, model, None, **factory_kwargs)
    with_draft = create_l3_executor(args, model, draft, **factory_kwargs)
    try:
        assert target_only.key_prefix != with_draft.key_prefix
    finally:
        target_only.close(timeout_s=5)
        with_draft.close(timeout_s=5)


def test_factory_refuses_head_partitioned_payload_on_cyclic_owners(construction):
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
        create_l3_executor(args, model, None, **factory_kwargs)


def test_factory_validates_limits_against_actual_host_capacity(construction):
    args, model, host, factory_kwargs = construction
    args.kvstore_storage_backend_extra_config = json.dumps(
        {"async": {"max_host_blocks": host.num_host_lcm_blocks + 1}}
    )
    with pytest.raises(ValueError, match="exceeds usable Host parents"):
        create_l3_executor(args, model, None, **factory_kwargs)


def test_unconfigured_storage_needs_neither_host_memory_nor_checkpoint_io(construction):
    args, model, _, _ = construction
    args.kvstore_storage_backend = None
    Path(model.model_path, "model.safetensors").unlink()
    assert (
        create_l3_executor(args, model, None, host_cache=None, world_cpu_group=None)
        is None
    )


def test_configured_storage_requires_host_memory(construction):
    args, model, _, _ = construction
    with pytest.raises(ValueError, match="requires a Host L2 allocation"):
        create_l3_executor(args, model, None, host_cache=None, world_cpu_group=None)


def test_factory_resolves_draft_eagle_layers_and_explicit_override(
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
    automatic = create_l3_executor(args, model, draft, **factory_kwargs)
    args.eagle3_layers_to_capture = [0]
    explicit = create_l3_executor(args, model, draft, **factory_kwargs)
    args.eagle3_layers_to_capture = [1]
    changed = create_l3_executor(args, model, draft, **factory_kwargs)
    try:
        assert automatic.key_prefix == explicit.key_prefix
        assert changed.key_prefix != automatic.key_prefix
    finally:
        automatic.close(timeout_s=5)
        explicit.close(timeout_s=5)
        changed.close(timeout_s=5)
