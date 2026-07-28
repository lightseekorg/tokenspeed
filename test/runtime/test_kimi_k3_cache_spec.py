from __future__ import annotations

import importlib
from dataclasses import replace
from test.runtime.conftest import TP8_PAGE_SET_BYTES as _TP8_PAGE_SET_BYTES
from test.runtime.conftest import TP8_PHYSICAL_PAGE_BYTES as _TP8_PAGE_BYTES
from test.runtime.conftest import (
    kimi_tp8_plan,
)

import pytest
import torch

from tokenspeed.runtime.configs.kimi_k3_config import KimiLinearConfig


def build_specs_for(
    config: KimiLinearConfig,
    *,
    tp_size: int = 8,
    dtype: torch.dtype = torch.float8_e4m3fn,
    quant_method: str | None = None,
):
    cache_spec = importlib.import_module(
        "tokenspeed.runtime.configs.kimi_k3_cache_spec"
    )
    return cache_spec.build_kimi_k3_cache_specs(
        config,
        tp_size=tp_size,
        mla_cache_dtype=dtype,
        mla_quant_method=quant_method,
        preferred_block_size=128,
        kernel_alignment=128,
    )


def build_specs(tp_size: int = 8):
    return build_specs_for(KimiLinearConfig(), tp_size=tp_size)


def build_plan(*, budget: int = _TP8_PAGE_SET_BYTES * 8, flat: bool = True):
    return kimi_tp8_plan(budget=budget, flat=flat)


def test_tp8_reference_plan_is_exact() -> None:
    plan = build_plan()
    assert [group.group_id for group in plan.groups] == [
        "full_attention",
        "linear_attention_0",
        "linear_attention_1",
        "linear_attention_2",
    ]
    assert plan.block_size == 1_536
    assert plan.physical_page_bytes == _TP8_PAGE_BYTES
    assert len(plan.physical_slots) == 24
    assert plan.usable_pages == 7
    assert plan.diagnostics.padding_binding_count == 3
    assert plan.diagnostics.group_layer_counts == (
        ("full_attention", 24),
        ("linear_attention_0", 23),
        ("linear_attention_1", 23),
        ("linear_attention_2", 23),
    )
    assert dict(plan.diagnostics.component_padding_bytes) == {
        "full_attention": 0,
        "linear_attention_0": 70_656,
        "linear_attention_1": 70_656,
        "linear_attention_2": 70_656,
    }


def test_shared_pool_demand_discovers_groups_and_matches_128k_regression() -> None:
    cache_spec = importlib.import_module(
        "tokenspeed.runtime.configs.kimi_k3_cache_spec"
    )
    plan = build_plan(budget=_TP8_PAGE_SET_BYTES * 108)
    sizing = dict(
        max_scheduled_tokens=8_192,
        max_live_requests=1,
        decode_input_tokens=1,
        overlap_schedule_depth=0,
    )

    assert (
        cache_spec.kimi_k3_shared_pool_pages(plan, token_capacity=131_072, **sizing)
        == 107
    )
    assert (
        cache_spec.kimi_k3_token_capacity_for_shared_pool(
            plan,
            usable_pages=107,
            upper_bound_tokens=131_072,
            **sizing,
        )
        == 131_072
    )
    clamped = cache_spec.kimi_k3_token_capacity_for_shared_pool(
        plan,
        usable_pages=106,
        upper_bound_tokens=131_072,
        **sizing,
    )
    assert clamped < 131_072
    assert (
        cache_spec.kimi_k3_shared_pool_pages(plan, token_capacity=clamped, **sizing)
        <= 106
    )
    assert (
        cache_spec.kimi_k3_shared_pool_pages(plan, token_capacity=clamped + 1, **sizing)
        > 106
    )

    history = tuple(group for group in plan.groups if group.family == "history")
    state = tuple(group for group in plan.groups if group.family == "state")
    assert len(history) == 1
    for state_count in range(len(state) + 1):
        dynamic_plan = replace(plan, groups=history + state[:state_count])
        assert (
            cache_spec.kimi_k3_shared_pool_pages(
                dynamic_plan, token_capacity=131_072, **sizing
            )
            == 86 + state_count * 7
        )


def test_shared_pool_demand_includes_fragmentation_and_overlap() -> None:
    cache_spec = importlib.import_module(
        "tokenspeed.runtime.configs.kimi_k3_cache_spec"
    )
    plan = build_plan(budget=_TP8_PAGE_SET_BYTES * 200)

    baseline = cache_spec.kimi_k3_shared_pool_pages(
        plan,
        token_capacity=131_072,
        max_scheduled_tokens=8_192,
        max_live_requests=1,
    )
    concurrent = cache_spec.kimi_k3_shared_pool_pages(
        plan,
        token_capacity=131_072,
        max_scheduled_tokens=8_192,
        max_live_requests=2,
    )
    overlapped = cache_spec.kimi_k3_shared_pool_pages(
        plan,
        token_capacity=131_072,
        max_scheduled_tokens=8_192,
        max_live_requests=1,
        overlap_schedule_depth=1,
    )

    assert concurrent > baseline
    assert overlapped == baseline + len(plan.groups)


def test_every_real_layer_has_one_correct_binding() -> None:
    plan = build_plan()
    assert len(plan.layer_bindings) == 93
    assert [binding.layer_id for binding in plan.layer_bindings] == list(range(93))
    config = KimiLinearConfig()
    linear = config.linear_attn_config
    kda_layer_ids = sorted(layer_id - 1 for layer_id in linear["kda_layers"])
    mla_layer_ids = sorted(set(range(config.num_hidden_layers)) - set(kda_layer_ids))
    expected_bindings = {
        layer_id: ("full_attention", physical_slot)
        for physical_slot, layer_id in enumerate(mla_layer_ids)
    }
    for group_index in range(3):
        for physical_slot, layer_id in enumerate(
            kda_layer_ids[group_index * 23 : (group_index + 1) * 23]
        ):
            expected_bindings[layer_id] = (
                f"linear_attention_{group_index}",
                physical_slot,
            )
    assert {
        binding.layer_id: (binding.group_id, binding.physical_slot)
        for binding in plan.layer_bindings
    } == expected_bindings
    for binding in plan.layer_bindings:
        if config.is_kda_layer(binding.layer_id):
            assert binding.group_id.startswith("linear_attention_")
            assert {component.name for component in binding.components} == {
                "conv_state",
                "recurrent_state",
            }
        else:
            assert binding.group_id == "full_attention"
            assert [component.name for component in binding.components] == ["latent_kv"]


def test_planning_is_deterministic() -> None:
    assert build_plan() == build_plan()


def test_non_flat_scheduler_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="FlatKV-only"):
        build_plan(flat=False)


@pytest.mark.parametrize(
    "dtype,quant_method,message",
    [
        (torch.bfloat16, None, "float8_e4m3fn"),
        (torch.float8_e4m3fn, "per_token_head", "per_token_head"),
    ],
)
def test_unsupported_mla_layout_is_rejected(dtype, quant_method, message) -> None:
    cache_spec = importlib.import_module(
        "tokenspeed.runtime.configs.kimi_k3_cache_spec"
    )
    with pytest.raises(ValueError, match=message):
        cache_spec.plan_kimi_k3_flat_cache(
            KimiLinearConfig(),
            flat_kvcache_enabled=True,
            tp_size=8,
            mla_cache_dtype=dtype,
            mla_quant_method=quant_method,
            preferred_block_size=128,
            kernel_alignment=128,
            cache_budget_bytes=_TP8_PAGE_SET_BYTES * 8,
        )


def test_tp_must_divide_kda_heads() -> None:
    with pytest.raises(ValueError, match="divisible"):
        build_specs(tp_size=5)


@pytest.mark.parametrize("tp_size", [0, -1])
def test_tp_size_must_be_positive_integer(tp_size: int) -> None:
    with pytest.raises(ValueError, match=r"tp_size.*positive integer"):
        build_specs(tp_size=tp_size)


def test_budget_below_one_usable_page_is_rejected() -> None:
    with pytest.raises(ValueError, match="minimum_usable_pages=1"):
        build_plan(budget=_TP8_PAGE_SET_BYTES * 2 - 1)


def test_reference_topology_uses_global_layer_ids() -> None:
    specs = build_specs()
    assert len(specs) == 93
    assert [spec.layer_id for spec in specs] == list(range(93))
    assert sum(spec.family == "history" for spec in specs) == 24
    assert sum(spec.family == "state" for spec in specs) == 69


def test_mla_component_is_single_fp8_576_byte_row() -> None:
    mla = next(spec for spec in build_specs() if spec.family == "history")
    assert mla.group_id_prefix == "full_attention"
    assert mla.group_order == 0
    assert len(mla.components) == 1
    component = mla.components[0]
    assert component.name == "latent_kv"
    assert component.shape == (1, 576)
    assert component.dtype == torch.float8_e4m3fn
    assert component.bytes_per_token == 576


@pytest.mark.parametrize(
    "tp_size,conv_shape,recurrent_shape",
    [
        (1, (36_864, 3), (96, 128, 128)),
        (8, (4_608, 3), (12, 128, 128)),
    ],
)
def test_kda_shapes_are_derived_from_explicit_tp(
    tp_size: int, conv_shape: tuple[int, ...], recurrent_shape: tuple[int, ...]
) -> None:
    kda = next(spec for spec in build_specs(tp_size) if spec.family == "state")
    assert kda.group_id_prefix == "linear_attention"
    assert kda.group_order == 1
    by_name = {component.name: component for component in kda.components}
    assert by_name["conv_state"].shape == conv_shape
    assert by_name["conv_state"].dtype == torch.bfloat16
    assert by_name["recurrent_state"].shape == recurrent_shape
    assert by_name["recurrent_state"].dtype == torch.float32


def test_tp8_kda_component_bytes_total_814080() -> None:
    kda = next(spec for spec in build_specs() if spec.family == "state")
    assert sum(component.constant_bytes for component in kda.components) == 814_080
