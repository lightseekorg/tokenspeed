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
    _dsa_fields,
    _mla_fields,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
    solve_cache_layout,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    build_paged_cache_group_specs,
)

_P = 64
_NUM_LAYERS = 3
_NUM_LCM_BLOCKS = 6


class _MLACfg:
    kv_lora_rank = 512
    qk_rope_head_dim = 64
    kv_cache_dtype = torch.bfloat16
    kv_cache_quant_method = ""
    index_head_dim = 128
    page_size = _P
    dtype = torch.bfloat16


def _plan(fields):
    layout = solve_cache_layout(
        fields,
        prefix_granularity=_P,
        cache_blocks_per_lcm_block={f.group_id: 1 for f in fields},
        alignment=1,
        max_padding_fraction=1.0,
    )
    return layout.with_num_lcm_blocks(_NUM_LCM_BLOCKS)


def _specs(plan, pd_enabled: bool):
    return build_paged_cache_group_specs(
        layer_types=("full_attention",) * _NUM_LAYERS,
        group_ids=("full_attention",) * _NUM_LAYERS,
        sliding_window_tokens=None,
        prefix_granularity=_P,
        pd_disaggregation_enabled=pd_enabled,
    )


def _make_mla_pool(pd_enabled: bool):
    from tokenspeed.runtime.layers.attention.kv_cache.mla import MLATokenToKVPool

    fields = _mla_fields(_MLACfg(), _NUM_LAYERS)
    plan = _plan(fields)
    return MLATokenToKVPool(
        size=_NUM_LCM_BLOCKS * _P,
        model_dtype=torch.bfloat16,
        dtype=torch.bfloat16,
        quant_method="",
        kv_lora_rank=_MLACfg.kv_lora_rank,
        qk_rope_head_dim=_MLACfg.qk_rope_head_dim,
        layer_num=_NUM_LAYERS,
        device="cpu",
        enable_memory_saver=False,
        prefix_granularity=_P,
        rank=0,
        memory_plan=plan,
        paged_cache_group_specs=_specs(plan, pd_enabled),
        token_capacity=_NUM_LCM_BLOCKS * _P,
        layer_group_ids=("full_attention",) * _NUM_LAYERS,
    )


def _make_dsa_pool(pd_enabled: bool):
    from tokenspeed.runtime.layers.attention.kv_cache.dsa import DSATokenToKVPool

    fields = _dsa_fields(_MLACfg(), _NUM_LAYERS)
    plan = _plan(fields)
    return DSATokenToKVPool(
        size=_NUM_LCM_BLOCKS * _P,
        model_dtype=torch.bfloat16,
        dtype=torch.bfloat16,
        quant_method="",
        kv_lora_rank=_MLACfg.kv_lora_rank,
        qk_rope_head_dim=_MLACfg.qk_rope_head_dim,
        layer_num=_NUM_LAYERS,
        device="cpu",
        enable_memory_saver=False,
        prefix_granularity=_P,
        rank=0,
        index_head_dim=_MLACfg.index_head_dim,
        memory_plan=plan,
        paged_cache_group_specs=_specs(plan, pd_enabled),
        token_capacity=_NUM_LCM_BLOCKS * _P,
        layer_group_ids=("full_attention",) * _NUM_LAYERS,
    )


def test_mla_pool_advertises_pd_contract_when_enabled() -> None:
    from tokenspeed.runtime.pd.cache_protocol import (
        build_pool_cache_transfer_contract,
    )

    pool = _make_mla_pool(pd_enabled=True)
    assert pool.supports_disaggregation is True
    contract, base_addr = build_pool_cache_transfer_contract(pool)
    assert base_addr == pool.buffer.data_ptr()
    field_ids = {
        field.field_id for field in contract.fields_for_group("full_attention")
    }
    assert field_ids == {f"layer.{i}.latent_kv" for i in range(_NUM_LAYERS)}


def test_dsa_pool_contract_covers_latent_and_index_k() -> None:
    from tokenspeed.runtime.pd.cache_protocol import (
        build_pool_cache_transfer_contract,
    )

    pool = _make_dsa_pool(pd_enabled=True)
    assert pool.supports_disaggregation is True
    contract, base_addr = build_pool_cache_transfer_contract(pool)
    assert base_addr == pool.buffer.data_ptr()
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
        build_pool_cache_transfer_contract,
    )

    pool = _make_mla_pool(pd_enabled=False)
    assert pool.supports_disaggregation is False
    with pytest.raises(RuntimeError, match="no transfer policy"):
        build_pool_cache_transfer_contract(pool)
