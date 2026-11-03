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

"""Resident memory limits and agreement, using real recipes and Gloo ranks."""

import json
from dataclasses import dataclass
from datetime import timedelta
from functools import partial
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
from tokenspeed.runtime.layers.attention.configs.dsa import DSAConfig
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
from tokenspeed.runtime.layers.attention.kv_cache.recipes.capacity import (
    agree_cache_capacity,
    local_cache_capacity,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.ordinary import OrdinaryRecipe
from tokenspeed.runtime.layers.attention.kv_cache.recipes.ownership import (
    pipeline_cache_ownership,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import CacheFieldSpec


@dataclass
class ModelMetadata:
    # The tests have no checkpoint; only its declared layer count is needed.
    num_attention_layers: int


@dataclass
class LaunchMetadata:
    max_total_tokens: int | None = None
    kv_offload_config: None = None
    disaggregation_mode: str = "null"
    chunked_prefill_size: int = 64
    enable_prefix_caching: bool = True


def mha_recipe(
    *, layers=8, draft_layers=0, budget=32 << 20, token_limit=None, probe=None
):
    config = AttnConfig(
        device="cpu",
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        kv_cache_quant_method="none",
        kv_cache_mxfp8=False,
        kernel_page_size=64,
        context_len=1024,
        max_bs=1,
        components=(
            MHAConfig(
                backend_name="fa2",
                num_attention_heads=8,
                num_kv_heads=2,
                head_dim=128,
                attn_tp_size=1,
            ),
        ),
    )
    return OrdinaryRecipe(
        family="mha",
        server_args=LaunchMetadata(max_total_tokens=token_limit),
        model_config=ModelMetadata(layers),
        attn_config=config,
        draft_model_config=ModelMetadata(draft_layers) if draft_layers else None,
        draft_attn_config=config if draft_layers else None,
        cache_budget_bytes=budget,
        probe_batch_rows=probe,
        decode_input_tokens=1,
        overlap_schedule_depth=0,
    )


def test_pp_cost_precedes_budget_floor_and_keeps_null_parent():
    # The budget cannot hold a full-model null+usable pair, but fits one stage.
    recipe = mha_recipe(budget=2 * 65536)
    with pytest.raises(ValueError, match="null parent"):
        recipe.setup()
    setup = recipe.setup_for_layers(
        resident_cache_window=(0, 1), synchronize_capacity=local_cache_capacity
    )
    local = setup.spec.memory_plan.narrow_to_layers(0, 1)
    assert local.num_lcm_blocks == 1
    assert local.arena_bytes == recipe.cache_budget_bytes
    assert local.lcm_block_bytes == 8 * 65536
    recipe.cache_budget_bytes -= 1
    with pytest.raises(ValueError, match="null parent"):
        recipe.setup_for_layers(
            resident_cache_window=(0, 1), synchronize_capacity=local_cache_capacity
        )


def test_draft_ownership_and_token_cap_survive_resident_sizing():
    recipe = mha_recipe(layers=4, draft_layers=1, token_limit=64 * 7)
    owners = pipeline_cache_ownership(4, 1, ((0, 2), (2, 4)))
    for owner, expected_bytes in zip(owners, (2 * 65536, 3 * 65536)):
        setup = recipe.setup_for_layers(
            resident_cache_window=owner.resident_cache_window,
            synchronize_capacity=local_cache_capacity,
        )
        plan = setup.spec.memory_plan.narrow_to_layers(*owner.resident_cache_window)
        assert plan.resident_block_bytes == expected_bytes
        assert plan.num_lcm_blocks == 7
        assert setup.spec.token_capacity == 7 * 64
        for field in plan.fields:
            assert (
                plan.field_page_byte_offset(field.field_id, 7) + field.payload_bytes
                <= plan.arena_bytes
            )


class SharedPlaneRecipe(OrdinaryRecipe):
    def fields_for_layer(self, layer_id, group_id, occurrence):
        return (
            CacheFieldSpec(
                f"layer.{layer_id}.kv",
                "shared",
                (64, 128),
                "bfloat16",
                exact_page_stride=False,
            ),
        )


def test_shared_plane_is_not_priced_as_local_field_payload():
    template = mha_recipe(layers=2)
    recipe = SharedPlaneRecipe(
        family="mha",
        server_args=template.server_args,
        model_config=template.model_config,
        attn_config=template.attn_config,
        draft_model_config=None,
        draft_attn_config=None,
        cache_budget_bytes=template.cache_budget_bytes,
        probe_batch_rows=None,
        decode_input_tokens=1,
        overlap_schedule_depth=0,
    )
    full = recipe.setup()
    local = recipe.setup_for_layers(
        resident_cache_window=(0, 1), synchronize_capacity=local_cache_capacity
    )
    assert full.spec.memory_plan.num_lcm_blocks == local.spec.memory_plan.num_lcm_blocks
    resident = local.spec.memory_plan.narrow_to_layers(0, 1)
    assert resident.resident_block_bytes == full.spec.memory_plan.lcm_block_bytes
    assert len(resident.fields) == 1
    assert resident.resident_block_bytes > resident.fields[0].payload_bytes


@pytest.mark.parametrize("draft", [False, True])
@pytest.mark.parametrize("qcp", [1, 8])
def test_qcp_reserves_one_shared_workspace_before_sizing(qcp, draft):
    config = AttnConfig(
        device="cuda",
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        kv_cache_quant_method="none",
        kv_cache_mxfp8=False,
        kernel_page_size=64,
        context_len=65536,
        max_bs=1,
        kvp_size=8,
        qcp_size=qcp,
        components=(
            DSAConfig(
                backend_name="dsa",
                num_attention_heads=16,
                num_kv_heads=1,
                head_dim=192,
                attn_tp_size=8,
                kv_lora_rank=512,
                qk_nope_head_dim=128,
                qk_rope_head_dim=64,
                v_head_dim=128,
                scaling=192**-0.5,
                kv_cache_dim=576,
                index_topk=2048,
                index_head_dim=128,
                index_n_heads=16,
                index_k_format="bf16",
            ),
        ),
    )
    recipe = OrdinaryRecipe(
        family="dsa",
        server_args=LaunchMetadata(),
        model_config=ModelMetadata(2),
        attn_config=config,
        draft_model_config=ModelMetadata(1) if draft else None,
        draft_attn_config=config if draft else None,
        cache_budget_bytes=256 << 20,
        probe_batch_rows=None,
        decode_input_tokens=1,
        overlap_schedule_depth=0,
    )
    setup = recipe.setup()
    plan = setup.spec.memory_plan
    workspace = 65536 * (1152 + 256) if qcp > 1 else 0
    assert setup.fixed_workspace_bytes == workspace
    assert (
        plan.num_lcm_blocks
        == (recipe.cache_budget_bytes - workspace) // plan.lcm_block_bytes - 1
    )
    assert plan.arena_bytes + workspace <= recipe.cache_budget_bytes
    assert (
        plan.arena_bytes + workspace + plan.lcm_block_bytes > recipe.cache_budget_bytes
    )


def test_probe_uses_demand_and_reports_resident_bytes_with_zero_budget():
    recipe = mha_recipe(budget=0, probe=3)
    setup = recipe.setup_for_layers(
        resident_cache_window=(0, 2), synchronize_capacity=local_cache_capacity
    )
    resident = setup.spec.memory_plan.narrow_to_layers(0, 2)
    assert resident.num_lcm_blocks == 3
    assert setup.cache_budget_bytes == resident.arena_bytes


def _capacity_worker(rank, init_file, output_dir):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=60),
    )
    try:
        windows = ((0, 1), (1, 3), (3, 6), (6, 8))
        recipe = mha_recipe()
        sync = partial(agree_cache_capacity, cpu_group=dist.group.WORLD, probe=False)
        setup = recipe.setup_for_layers(
            resident_cache_window=windows[rank], synchronize_capacity=sync
        )
        plan = setup.spec.memory_plan.narrow_to_layers(*windows[rank])
        assert plan.num_lcm_blocks == (32 << 20) // (3 * 65536) - 1
        assert plan.arena_bytes <= 32 << 20
        # Different budgets can make the smallest stage the bottleneck.
        recipe.cache_budget_bytes = 11 * 65536 if rank == 0 else 32 << 20
        smaller = recipe.setup_for_layers(
            resident_cache_window=windows[rank], synchronize_capacity=sync
        )
        assert smaller.spec.memory_plan.num_lcm_blocks == 10
        # One rank's infeasible budget must fail on every rank, without hanging.
        recipe.cache_budget_bytes = 1 if rank == 2 else 32 << 20
        try:
            recipe.setup_for_layers(
                resident_cache_window=windows[rank], synchronize_capacity=sync
            )
        except ValueError as exc:
            assert "rank 2:" in str(exc)
        else:
            raise AssertionError("a failed rank must fail the whole build")
        assert (
            agree_cache_capacity(rank + 1, None, cpu_group=dist.group.WORLD, probe=True)
            == 4
        )
        Path(output_dir, f"rank{rank}.json").write_text(
            json.dumps(
                {"parents": plan.num_lcm_blocks, "arena_bytes": plan.arena_bytes}
            )
        )
    finally:
        dist.destroy_process_group()


def test_four_ranks_agree_on_resident_capacity_and_errors(tmp_path):
    mp.spawn(
        _capacity_worker,
        args=(str(tmp_path / "gloo"), str(tmp_path)),
        nprocs=4,
        join=True,
    )
    results = [json.loads((tmp_path / f"rank{r}.json").read_text()) for r in range(4)]
    assert len({result["parents"] for result in results}) == 1
    assert len({result["arena_bytes"] for result in results}) == 3


class PackedRecipe(OrdinaryRecipe):
    def packing(self, groups):
        return {spec.group_id: 3 for spec, _ in groups}


def test_parent_limit_keeps_null_slots_inside_int32():
    template = mha_recipe(budget=1 << 65)
    recipe = PackedRecipe(
        family="mha",
        server_args=template.server_args,
        model_config=template.model_config,
        attn_config=template.attn_config,
        draft_model_config=None,
        draft_attn_config=None,
        cache_budget_bytes=template.cache_budget_bytes,
        probe_batch_rows=None,
        decode_input_tokens=1,
        overlap_schedule_depth=0,
    )
    plan = recipe.setup().spec.memory_plan
    assert (plan.num_lcm_blocks + 1) * 3 - 1 <= 2**31 - 1
    assert (plan.num_lcm_blocks + 2) * 3 - 1 > 2**31 - 1
