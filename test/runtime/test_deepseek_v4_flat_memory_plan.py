# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY.

"""DeepSeek V4 contracts exercised through the production plan builder."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
_v4 = pytest.importorskip("tokenspeed.runtime.layers.attention.kv_cache.deepseek_v4")

from tokenspeed.runtime.configs.deepseek_v4_flat_memory_plan import (  # noqa: E402
    assert_v4_flat_plan_agreement,
    make_v4_flat_plan_agreement_record,
)
from tokenspeed.runtime.configs.flat_kv_contract import (  # noqa: E402
    CACHE_OWNER_DRAFT,
    CACHE_OWNER_TARGET,
)
from tokenspeed.runtime.configs.flat_memory_plan import (  # noqa: E402
    FLAT_PACKED_UNPACK_META_FIELDS,
)

PAGE_SIZE = 64


def _hf_config(sliding_window: int = 128) -> SimpleNamespace:
    return SimpleNamespace(
        compress_ratios=(1, 4, 128),
        head_dim=512,
        qk_rope_head_dim=64,
        index_head_dim=128,
        sliding_window=sliding_window,
    )


def _layout(layer_indices=None):
    return _v4.deepseek_v4_cache_layout_from_config(
        _hf_config(),
        page_size=PAGE_SIZE,
        use_fp4_indexer_cache=True,
        layer_indices=layer_indices,
    )


def _builder_kwargs() -> dict:
    return {
        "target_layout": _layout(),
        "target_hf_config": _hf_config(),
        "target_layer_num": 3,
        "target_max_live_requests": 2,
        "target_max_context_len": 256,
        "target_max_graph_bs": 2,
        "max_scheduled_tokens": 64,
        "max_total_tokens": 256,
        "draft_layout": _layout((0, 1)),
        "draft_hf_config": _hf_config(),
        "draft_layer_num": 2,
        "draft_max_live_requests": 3,
        "draft_max_context_len": 256,
        "draft_max_graph_bs": 2,
        "decode_input_tokens": 4,
        "overlap_schedule_depth": 1,
    }


@pytest.fixture
def production_plan():
    return _v4.build_deepseek_v4_flat_memory_plan(**_builder_kwargs())


def test_production_specs_pin_v4_storage_geometry(production_plan) -> None:
    expected = (
        ("v4.swa_kv", 64, "v4.swa", "continuation_state", "bounded_window"),
        (
            "v4.c4a.compressor_state",
            4,
            "v4.c4.state",
            "continuation_state",
            "bounded_window",
        ),
        (
            "v4.c4a.compressed_kv",
            256,
            "v4.c4.history",
            "history_anchor",
            "absolute",
        ),
        (
            "v4.c128a.compressor_state",
            8,
            "v4.c128.state",
            "continuation_state",
            "bounded_window",
        ),
        (
            "v4.c128a.compressed_kv",
            256,
            "v4.c128.history",
            "history_anchor",
            "absolute",
        ),
        (
            "v4.c4a.indexer_compressor_state",
            4,
            "v4.index.state",
            "continuation_state",
            "bounded_window",
        ),
    )
    specs = production_plan.target_owner_group_specs

    assert (
        tuple(
            (
                spec.group_id,
                spec.block_size,
                spec.pool_id,
                spec.prefix_role,
                spec.table_layout,
            )
            for spec in specs
        )
        == expected
    )
    assert all(
        spec.block_size == spec.rows_per_page * spec.entry_stride_tokens
        and spec.owner_mask & CACHE_OWNER_TARGET
        for spec in specs
    )


@pytest.mark.parametrize(
    ("group_id", "rows_per_page"),
    (
        ("v4.swa_kv", 32),
        ("v4.c4a.compressor_state", 5),
        ("v4.c4a.compressed_kv", 32),
        ("v4.c128a.compressor_state", 7),
        ("v4.c128a.compressed_kv", 3),
        ("v4.c4a.indexer_compressor_state", 5),
    ),
)
def test_flat_component_geometry_follows_only_group_specs(
    production_plan,
    monkeypatch,
    group_id,
    rows_per_page,
) -> None:
    layout = _layout()
    specs = tuple(
        (
            replace(
                spec,
                rows_per_page=rows_per_page,
                block_size=rows_per_page * spec.entry_stride_tokens,
            )
            if spec.group_id == group_id
            else spec
        )
        for spec in production_plan.target_owner_group_specs
    )
    expected_components = _v4.deepseek_v4_flat_component_plans(
        layout=layout,
        specs=specs,
        layer_num=3,
        owner="target",
    )
    stale = {
        (component.group_id, component.layer, component.component): component
        for pool in production_plan.pools
        for component in pool.tensors
        if component.owner == "target"
    }
    expected = {
        (component.group_id, component.layer, component.component): component
        for component in expected_components
    }
    affected = [identity for identity in expected if identity[0] == group_id]

    assert stale.keys() == expected.keys()
    assert affected
    assert all(
        expected[identity].shape_per_block != stale[identity].shape_per_block
        for identity in affected
    )
    assert (
        _v4._deepseek_v4_cache_group_page_bytes(layout, specs, 3)[group_id]
        != _v4._deepseek_v4_cache_group_page_bytes(
            layout, production_plan.target_owner_group_specs, 3
        )[group_id]
    )

    monkeypatch.setattr(_v4, "build_v4_cache_specs", lambda *_args, **_kwargs: specs)
    monkeypatch.setattr(
        _v4,
        "_build_v4_flat_memory_plan",
        lambda **_kwargs: production_plan,
    )
    with pytest.raises(ValueError, match="component geometry disagrees with specs"):
        _v4.build_deepseek_v4_flat_memory_plan(
            target_layout=_layout(),
            target_hf_config=_hf_config(),
            target_layer_num=3,
            target_max_live_requests=2,
            target_max_context_len=256,
            target_max_graph_bs=2,
            max_scheduled_tokens=64,
            max_total_tokens=256,
        )


def test_owner_union_and_pool_accounting_come_from_one_plan(production_plan) -> None:
    target_ids = {spec.group_id for spec in production_plan.target_owner_group_specs}
    draft_ids = {spec.group_id for spec in production_plan.draft_owner_group_specs}
    scheduler = {spec.group_id: spec for spec in production_plan.scheduler_group_specs}

    assert set(scheduler) == target_ids | draft_ids
    assert all(
        spec is scheduler[spec.group_id]
        for spec in production_plan.target_owner_group_specs
    )
    assert all(
        spec is scheduler[spec.group_id]
        for spec in production_plan.draft_owner_group_specs
    )
    for group_id, spec in scheduler.items():
        expected_mask = CACHE_OWNER_TARGET
        if group_id in draft_ids:
            expected_mask |= CACHE_OWNER_DRAFT
        assert spec.owner_mask == expected_mask

    pools = {pool.pool_id: pool for pool in production_plan.pools}
    assert {tensor.owner for tensor in pools["v4.swa"].tensors} == {
        "target",
        "draft",
    }
    assert {tensor.owner for tensor in pools["v4.c128.history"].tensors} == {"target"}
    assert all(
        pool.bytes_per_block == sum(tensor.bytes_per_block for tensor in pool.tensors)
        for pool in production_plan.pools
    )
    assert production_plan.payload_bytes == sum(
        pool.total_blocks * pool.bytes_per_block for pool in production_plan.pools
    )


def test_plan_rejects_noncanonical_scheduler_membership(production_plan) -> None:
    specs = production_plan.scheduler_group_specs
    cases = (
        ("duplicate", (specs[0], *specs), "unique group ids"),
        (
            "ownerless",
            (replace(specs[0], owner_mask=0), *specs[1:]),
            "assign every group to an owner",
        ),
    )

    for _name, invalid_specs, error in cases:
        with pytest.raises(ValueError, match=error):
            replace(production_plan, scheduler_group_specs=invalid_specs)


def test_graph_and_forward_metadata_are_exact_owner_local_shapes(
    production_plan,
) -> None:
    metadata = production_plan.runtime_metadata
    target_cols = metadata.graph_capture_cols_by_group("target")
    draft_cols = metadata.graph_capture_cols_by_group("draft")
    target_bytes = 4 * 2 * sum(cols + 1 for cols in target_cols.values())
    draft_bytes = 4 * 2 * sum(cols + 1 for cols in draft_cols.values())

    assert set(target_cols) == {
        spec.group_id for spec in production_plan.target_owner_group_specs
    }
    assert set(draft_cols) == {
        spec.group_id for spec in production_plan.draft_owner_group_specs
    }
    assert metadata.graph_metadata_bytes_for_owner("target") == target_bytes
    assert metadata.graph_metadata_bytes_for_owner("draft") == draft_bytes
    assert metadata.graph_metadata_bytes == target_bytes + draft_bytes
    assert metadata.forward_buffer_depth == 2
    assert metadata.max_scheduled_batch_rows == 3
    payload_bytes = (
        4
        * metadata.forward_buffer_depth
        * metadata.max_scheduled_batch_rows
        * sum(item.max_export_cols + 1 for item in metadata.group_table_plans)
    )
    header_bytes = (
        4
        * metadata.forward_buffer_depth
        * FLAT_PACKED_UNPACK_META_FIELDS
        * (len(target_cols) + len(draft_cols))
    )
    assert metadata.forward_input_bytes == payload_bytes + header_bytes
    assert production_plan.device_cache_total_bytes == (
        production_plan.payload_bytes
        + metadata.graph_metadata_bytes
        + metadata.forward_input_bytes
    )


def test_fingerprint_and_rank_agreement_use_production_plan() -> None:
    first = _v4.build_deepseek_v4_flat_memory_plan(**_builder_kwargs())
    second = _v4.build_deepseek_v4_flat_memory_plan(**_builder_kwargs())
    assert first == second
    assert first.plan_fingerprint == second.plan_fingerprint

    rank0 = make_v4_flat_plan_agreement_record(first, rank=0)
    rank1 = make_v4_flat_plan_agreement_record(second, rank=1)
    assert_v4_flat_plan_agreement((rank0, rank1))

    divergent_kwargs = _builder_kwargs()
    divergent_kwargs["max_total_tokens"] = 320
    divergent = _v4.build_deepseek_v4_flat_memory_plan(**divergent_kwargs)
    with pytest.raises(RuntimeError, match=r"rank 1 fingerprint=.*rank 0 fingerprint"):
        assert_v4_flat_plan_agreement(
            (rank0, make_v4_flat_plan_agreement_record(divergent, rank=1))
        )


@pytest.mark.parametrize(
    ("overrides", "error"),
    (
        ({"max_total_tokens": 257}, "must be page-aligned"),
        ({"target_layer_num": 2}, "target layer count"),
        ({"target_max_graph_bs": 3}, "target graph rows exceed"),
        ({"draft_max_graph_bs": 3}, "target/draft graph rows must match"),
        ({"draft_hf_config": _hf_config(256)}, "sliding_window_tokens"),
    ),
    ids=(
        "unaligned-budget",
        "layer-drift",
        "target-graph-row-drift",
        "cross-owner-graph-row-drift",
        "window-drift",
    ),
)
def test_production_builder_rejects_cross_owner_plan_drift(overrides, error) -> None:
    kwargs = _builder_kwargs()
    kwargs.update(overrides)

    with pytest.raises(ValueError, match=error):
        _v4.build_deepseek_v4_flat_memory_plan(**kwargs)
