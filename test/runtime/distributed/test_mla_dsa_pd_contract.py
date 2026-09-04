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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

"""MLA / DSA pools expose the PD cache-group contract.

DeepSeek-R1 (MLA) and V3.2 (DSA) share a single replicated latent KV per TP
rank packed into the LCM arena; DSA adds a second per-layer field (``index_k``)
into the SAME history group. PD must move whole physical parent pages over the
cache-group contract, never the legacy per-layer-buffer path (which assumed
each field view was an independently contiguous ``[pages, item_len]`` slab and
corrupted every page of a strided multi-field arena). These tests assert the
pools advertise the contract and that it covers every planned field.
"""

from __future__ import annotations

import os
import sys

import torch

_TEST_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _TEST_DIR)
sys.path.insert(0, os.path.dirname(_TEST_DIR))

# CPU-only test (builds pools on CPU, no kernels) scheduled in runtime-1gpu
# because it imports the full runtime; mirrors test_cache_pd_manifest.py.
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=20, suite="runtime-1gpu")

from tokenspeed.runtime.layers.attention.kv_cache.recipes.ordinary import (
    OrdinaryRecipe,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
    pack,
)

_P = 64
_NUM_LAYERS = 3
_NUM_LCM_BLOCKS = 6


def _attn_config(family: str, pd_enabled: bool):
    """A real MLA/DSA config: the recipe dispatches fields on the spec type."""
    from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
    from tokenspeed.runtime.layers.attention.configs.dsa import DSAConfig
    from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig

    common = dict(
        backend_name="mla",
        num_attention_heads=64,
        num_kv_heads=64,
        attn_tp_size=1,
        head_dim=192,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        scaling=1.0,
        kv_cache_dim=576,
    )
    if family == "dsa":
        spec = DSAConfig(index_topk=1, index_head_dim=128, index_n_heads=1, **common)
    else:
        spec = MLAConfig(**common)
    return AttnConfig(
        device="cpu",
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        context_len=1024,
        max_bs=1,
        prefix_granularity=_P,
        kernel_page_size=_P,
        kv_cache_quant_method="",
        pd_disaggregation_enabled=pd_enabled,
        components=(spec,),
    )


def _recipe(family: str, pd_enabled: bool = False):
    """An ordinary recipe over ``_NUM_LAYERS`` MLA/DSA layers."""
    from types import SimpleNamespace

    config = _attn_config(family, pd_enabled)
    return OrdinaryRecipe(
        family=family,
        server_args=SimpleNamespace(max_total_tokens=_NUM_LCM_BLOCKS * _P),
        model_config=SimpleNamespace(num_attention_layers=_NUM_LAYERS),
        attn_config=config,
        draft_model_config=None,
        draft_attn_config=None,
        cache_budget_bytes=1 << 30,
        decode_input_tokens=1,
        overlap_schedule_depth=0,
    )


def _plan_and_specs(family: str, pd_enabled: bool = False):
    recipe = _recipe(family, pd_enabled)
    groups = recipe.groups()
    layout = pack(
        groups,
        prefix_granularity=recipe.prefix_granularity,
        cache_blocks_per_lcm_block=recipe.packing(groups),
        alignment=recipe.alignment,
        max_padding_fraction=recipe.max_padding_fraction,
    )
    return layout.bind(_NUM_LCM_BLOCKS), tuple(spec for spec, _ in groups)


def _make_mla_pool(pd_enabled: bool):
    from tokenspeed.runtime.layers.attention.kv_cache.mla import MLATokenToKVPool

    plan, specs = _plan_and_specs("mla", pd_enabled)
    from cache_pool_test_utils import make_pool

    _, pool = make_pool(
        MLATokenToKVPool,
        plan,
        device="cpu",
        model_dtype=torch.bfloat16,
        dtype=torch.bfloat16,
        quant_method="",
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        layer_num=_NUM_LAYERS,
        rank=0,
        cache_group_specs=specs,
        token_capacity=_NUM_LCM_BLOCKS * _P,
    )
    return pool


def _make_dsa_pool(pd_enabled: bool):
    from tokenspeed.runtime.layers.attention.kv_cache.dsa import DSATokenToKVPool

    plan, specs = _plan_and_specs("dsa", pd_enabled)
    from cache_pool_test_utils import make_pool

    _, pool = make_pool(
        DSATokenToKVPool,
        plan,
        device="cpu",
        model_dtype=torch.bfloat16,
        dtype=torch.bfloat16,
        quant_method="",
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        layer_num=_NUM_LAYERS,
        rank=0,
        index_head_dim=128,
        cache_group_specs=specs,
        token_capacity=_NUM_LCM_BLOCKS * _P,
    )
    return pool


def test_mla_pool_advertises_pd_contract_when_enabled() -> None:
    from tokenspeed.runtime.pd.cache_protocol import (
        build_arena_cache_transfer_contract,
    )

    pool = _make_mla_pool(pd_enabled=True)
    assert pool.arena.supports_disaggregation is True
    contract, base_addr = build_arena_cache_transfer_contract(pool.arena)
    assert base_addr == pool.arena.buffer.data_ptr()
    field_ids = {
        field.field_id for field in contract.fields_for_group("full_attention")
    }
    assert field_ids == {f"layer.{i}.latent_kv" for i in range(_NUM_LAYERS)}


def test_dsa_pool_contract_covers_latent_and_index_k() -> None:
    from tokenspeed.runtime.pd.cache_protocol import (
        build_arena_cache_transfer_contract,
    )

    pool = _make_dsa_pool(pd_enabled=True)
    assert pool.arena.supports_disaggregation is True
    contract, base_addr = build_arena_cache_transfer_contract(pool.arena)
    assert base_addr == pool.arena.buffer.data_ptr()
    fields = contract.fields_for_group("full_attention")
    field_ids = {field.field_id for field in fields}
    # DSA packs BOTH latent_kv and index_k for every layer into the one
    # history group; the legacy per-buffer path dropped/mis-strided these.
    expected = {f"layer.{i}.latent_kv" for i in range(_NUM_LAYERS)} | {
        f"layer.{i}.index_k" for i in range(_NUM_LAYERS)
    }
    assert field_ids == expected
    assert contract.group_specs[0].transfer_policy == "full_suffix"
    assert all(field.page_stride_bytes > 0 for field in fields)


def test_pd_contract_refused_when_disabled() -> None:
    import pytest

    from tokenspeed.runtime.pd.cache_protocol import (
        build_arena_cache_transfer_contract,
    )

    pool = _make_mla_pool(pd_enabled=False)
    assert pool.arena.supports_disaggregation is False
    with pytest.raises(RuntimeError, match="no transfer policy"):
        build_arena_cache_transfer_contract(pool.arena)
