from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

# CPU-only tests scheduled in runtime-1gpu because they import the full runtime.
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
    CacheFieldLayout,
    CacheGroupLayout,
    CacheMemoryPlan,
    CachePlaneLayout,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    CacheGroupSpec,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.transfer import (
    CacheFieldPartition,
    CacheFieldTransferSpec,
    CacheTransferSchema,
    build_cache_transfer_schema,
)
from tokenspeed.runtime.pd.cache_protocol import (
    CacheContractError,
    CachePDBlockManifest,
    CachePDGroupBlocks,
    CacheTransferContract,
    _logical_slots,
    build_cache_block_manifest,
    build_cache_fields_by_producer_step,
    build_cache_transfer_contract,
)
from tokenspeed.runtime.pd.cache_protocol import (  # noqa: E402
    build_cache_transfer_schema as compatibility_build_cache_transfer_schema,
)
from tokenspeed.runtime.pd.cache_protocol import (
    validate_cache_manifest,
    validate_cache_peer_layout,
)


def _group_spec(
    group_id: str,
    family: str,
    policy: str,
    *,
    prefix_granularity: int = 4,
    retention: str = "full_history",
    sliding_window_tokens: int | None = None,
) -> CacheGroupSpec:
    return CacheGroupSpec(
        group_id=group_id,
        retention=retention,
        rows_per_page=prefix_granularity,
        entry_stride_tokens=1,
        sliding_window_tokens=sliding_window_tokens,
        family=family,
        transfer_policy=policy,
    )


def _field(
    group_id: str,
    field_id: str,
    *,
    plane_id: str = "cache",
    shape: tuple[int, ...] = (1,),
    dtype: str = "uint8",
    offset: int = 0,
    stride: int = 128,
) -> CacheFieldLayout:
    return CacheFieldLayout(
        group_id=group_id,
        field_id=field_id,
        plane_id=plane_id,
        shape=shape,
        dtype=dtype,
        field_offset_bytes=offset,
        page_stride_bytes=stride,
    )


def _contract(
    specs: tuple[CacheGroupSpec, ...],
    fields: tuple[CacheFieldLayout, ...],
    *,
    block_size: int = 4,
    capacity: int = 64,
    page_bytes: int = 128,
    planes: tuple[CachePlaneLayout, ...] | None = None,
    transfer_schema: CacheTransferSchema = CacheTransferSchema(),
    packing: int = 1,
) -> CacheTransferContract:
    plan = CacheMemoryPlan(
        prefix_granularity=block_size,
        lcm_block_bytes=page_bytes,
        num_lcm_blocks=capacity - 1,
        groups=tuple(
            CacheGroupLayout(
                group_id=spec.group_id,
                cache_blocks_per_lcm_block=packing,
                page_count=(1 + (capacity - 1) * packing),
            )
            for spec in specs
        ),
        planes=planes
        or (
            CachePlaneLayout(
                plane_id="cache",
                bytes_per_lcm_block=page_bytes,
                arena_offset_bytes=0,
            ),
        ),
        fields=fields,
    )
    return CacheTransferContract(
        plan=plan,
        group_specs=specs,
        transfer_schema=transfer_schema,
    )


def _layout(capacity: int = 64) -> CacheTransferContract:
    specs = (
        _group_spec("attention", "history", "full_suffix"),
        _group_spec("linear-a", "state", "latest_snapshot"),
        _group_spec("linear-b", "state", "latest_snapshot"),
    )
    fields = (
        _field("attention", "attention.k", offset=0),
        _field("linear-a", "linear-a.state", offset=1),
        _field("linear-b", "linear-b.state", offset=2),
    )
    return _contract(specs, fields, capacity=capacity)


def _op(page_offset: int = 0) -> SimpleNamespace:
    tables = {
        "attention": np.array([[1, 2, 3, 4, -1]], dtype=np.int32) + page_offset,
        "linear-a": np.array([[11, 12, 13, 14, -1]], dtype=np.int32) + page_offset,
        "linear-b": np.array([[21, 22, 23, 24, -1]], dtype=np.int32) + page_offset,
    }
    return SimpleNamespace(block_tables_arrays=lambda: tables)


def test_recipe_layer_owns_transfer_schema_api() -> None:
    schema = build_cache_transfer_schema(
        _two_plane_lcm_plan(3),
        model_config=SimpleNamespace(
            num_attention_layers=1,
            num_key_value_heads=8,
            hf_config=SimpleNamespace(),
        ),
    )

    assert schema == CacheTransferSchema(
        tuple(
            CacheFieldTransferSpec(
                f"layer.0.{suffix}",
                CacheFieldPartition(axis=1, global_extent=8),
            )
            for suffix in ("k", "v")
        )
    )


def test_cache_protocol_compatibility_reexports_recipe_compiler() -> None:
    assert compatibility_build_cache_transfer_schema is build_cache_transfer_schema


def test_manifest_selects_history_suffix_and_only_final_state_slot() -> None:
    manifest = build_cache_block_manifest(
        _op(),
        layout=_layout(),
        request_row=0,
        prefix_len=4,
        prompt_len=14,
    )

    assert manifest.groups[0] == CachePDGroupBlocks("attention", (2, 3, 4))
    assert manifest.groups[1:] == (
        CachePDGroupBlocks("linear-a", (14,)),
        CachePDGroupBlocks("linear-b", (24,)),
    )


def test_contract_serializes_recipe_specs_without_a_pd_group_projection() -> None:
    contract = _layout()
    assert contract.group_specs[1].transfer_policy == "latest_snapshot"
    payload = json.loads(contract.to_wire_bytes())
    assert "groups" not in payload
    assert [spec["group_id"] for spec in payload["group_specs"]] == [
        spec.group_id for spec in contract.group_specs
    ]


def test_source_and_destination_block_ids_are_independent() -> None:
    source_layout = _layout(64)
    destination_layout = _layout(128)
    source = build_cache_block_manifest(
        _op(),
        layout=source_layout,
        request_row=0,
        prefix_len=4,
        prompt_len=14,
    )
    destination = build_cache_block_manifest(
        _op(40),
        layout=destination_layout,
        request_row=0,
        prefix_len=4,
        prompt_len=14,
    )

    validate_cache_peer_layout(source_layout, destination_layout)
    assert source.groups[0].block_ids != destination.groups[0].block_ids


@pytest.mark.parametrize("bad_block", [0, -1, 64])
def test_manifest_rejects_null_padding_and_out_of_capacity_block(
    bad_block: int,
) -> None:
    operation = _op()
    operation.block_tables_arrays()["linear-a"][0, 3] = bad_block
    with pytest.raises(CacheContractError, match="invalid block ID"):
        build_cache_block_manifest(
            operation,
            layout=_layout(),
            request_row=0,
            prefix_len=4,
            prompt_len=14,
        )


def test_manifest_accepts_reordered_mapping_but_rejects_wrong_keys() -> None:
    with pytest.raises(CacheContractError, match="prefix_granularity"):
        build_cache_block_manifest(
            _op(),
            layout=_layout(),
            request_row=0,
            prefix_len=3,
            prompt_len=14,
        )

    reordered = _op()
    tables = reordered.block_tables_arrays()
    reordered.block_tables_arrays = lambda: {
        "linear-a": tables["linear-a"],
        "attention": tables["attention"],
        "linear-b": tables["linear-b"],
    }
    manifest = build_cache_block_manifest(
        reordered,
        layout=_layout(),
        request_row=0,
        prefix_len=4,
        prompt_len=14,
    )
    assert tuple(group.group_id for group in manifest.groups) == (
        "attention",
        "linear-a",
        "linear-b",
    )

    for wrong_tables in (
        {"attention": tables["attention"], "linear-a": tables["linear-a"]},
        {**tables, "extra": tables["attention"]},
    ):
        wrong = SimpleNamespace(block_tables_arrays=lambda: wrong_tables)
        with pytest.raises(CacheContractError, match="group IDs disagree"):
            build_cache_block_manifest(
                wrong,
                layout=_layout(),
                request_row=0,
                prefix_len=4,
                prompt_len=14,
            )


def test_contract_and_manifest_wire_round_trip_has_no_version_field() -> None:
    layout = _layout()
    manifest = build_cache_block_manifest(
        _op(),
        layout=layout,
        request_row=0,
        prefix_len=4,
        prompt_len=14,
    )
    layout_wire = layout.to_wire_bytes()
    manifest_wire = manifest.to_wire_bytes()
    contract_payload = json.loads(layout_wire)
    plan_payload = contract_payload["plan"]
    assert plan_payload["num_lcm_blocks"] == layout.plan.num_lcm_blocks
    assert plan_payload["lcm_block_bytes"] == layout.plan.lcm_block_bytes
    assert [spec["group_id"] for spec in contract_payload["group_specs"]] == [
        "attention",
        "linear-a",
        "linear-b",
    ]
    assert "version" not in contract_payload
    manifest_payload = json.loads(manifest_wire)
    assert "version" not in manifest_payload
    assert manifest_payload["groups"][0]["block_ids"] == [2, 3, 4]
    assert "page_ids" not in manifest_payload["groups"][0]
    assert CacheTransferContract.from_wire_bytes(layout_wire) == layout
    assert CachePDBlockManifest.from_wire_bytes(manifest_wire) == manifest


def test_lcm_group_capacity_uses_its_cache_blocks_per_parent() -> None:
    spec = _group_spec("history", "history", "full_suffix")
    # Null parent plus three usable parents, each packing two child pages.
    layout = _contract(
        (spec,),
        (
            _field(
                "history",
                "layer.0.k",
                shape=(64,),
                dtype="bfloat16",
                stride=128,
            ),
        ),
        capacity=4,
        page_bytes=1024,
        packing=2,
    )
    tables = {"history": np.array([[5, 6]], dtype=np.int32)}
    operation = SimpleNamespace(block_tables_arrays=lambda: tables)

    build_cache_block_manifest(
        operation,
        layout=layout,
        request_row=0,
        prefix_len=0,
        prompt_len=8,
    )
    tables["history"][0, 1] = 7
    with pytest.raises(CacheContractError, match="invalid block ID"):
        build_cache_block_manifest(
            operation,
            layout=layout,
            request_row=0,
            prefix_len=0,
            prompt_len=8,
        )


def test_destination_manifest_uses_peer_local_group_packing() -> None:
    def layout(*, packing: int, physical_page_bytes: int) -> CacheTransferContract:
        spec = _group_spec(
            "history",
            "history",
            "full_suffix",
            prefix_granularity=2,
        )
        return _contract(
            (spec,),
            (
                _field(
                    "history",
                    "layer.0.k",
                    shape=(4,),
                    dtype="bfloat16",
                    stride=8,
                ),
            ),
            block_size=2,
            capacity=4,
            page_bytes=physical_page_bytes,
            packing=packing,
        )

    source_layout = layout(packing=1, physical_page_bytes=8)
    destination_layout = layout(packing=2, physical_page_bytes=16)
    source = CachePDBlockManifest(
        groups=(CachePDGroupBlocks("history", (3,)),),
        prefix_len=0,
        prompt_len=2,
    )
    destination = CachePDBlockManifest(
        groups=(CachePDGroupBlocks("history", (6,)),),
        prefix_len=0,
        prompt_len=2,
    )

    validate_cache_peer_layout(source_layout, destination_layout)
    validate_cache_manifest(source, layout=source_layout, peer="source")
    validate_cache_manifest(
        destination,
        layout=destination_layout,
        peer="destination",
    )
    with pytest.raises(CacheContractError, match="out-of-bounds"):
        validate_cache_manifest(
            destination,
            layout=source_layout,
            peer="destination",
        )


def test_logical_slots_accepts_block_granularity() -> None:
    assert _logical_slots(
        "full_suffix",
        prefix_len=4,
        prompt_len=14,
        block_granularity=4,
        retention="full_history",
        sliding_window_tokens=None,
    ) == (1, 2, 3)


def _cache_buffer(nbytes: int, data_ptr: int = 0x10000):
    class _Dtype:
        def __str__(self):
            return "torch.uint8"

    class _Backing:
        dtype = _Dtype()

        def __init__(self):
            self.nbytes = nbytes

        @staticmethod
        def is_contiguous():
            return True

        @staticmethod
        def storage_offset():
            return 0

        def data_ptr(self):
            return data_ptr

        def untyped_storage(self):
            return SimpleNamespace(data_ptr=lambda: data_ptr)

    return _Backing()


def _two_plane_lcm_plan(num_lcm_blocks: int) -> CacheMemoryPlan:
    plane_bytes = 512
    lcm_block_bytes = 1024
    return CacheMemoryPlan(
        prefix_granularity=4,
        lcm_block_bytes=lcm_block_bytes,
        num_lcm_blocks=num_lcm_blocks,
        groups=(
            CacheGroupLayout(
                group_id="history",
                cache_blocks_per_lcm_block=2,
                page_count=1 + num_lcm_blocks * 2,
            ),
        ),
        planes=(
            CachePlaneLayout(
                plane_id="k",
                bytes_per_lcm_block=plane_bytes,
                arena_offset_bytes=0,
            ),
            CachePlaneLayout(
                plane_id="v",
                bytes_per_lcm_block=plane_bytes,
                arena_offset_bytes=(num_lcm_blocks + 1) * plane_bytes,
            ),
        ),
        fields=(
            CacheFieldLayout(
                group_id="history",
                field_id="layer.0.k",
                plane_id="k",
                shape=(4, 8),
                dtype="bfloat16",
                field_offset_bytes=0,
                page_stride_bytes=256,
            ),
            CacheFieldLayout(
                group_id="history",
                field_id="layer.0.v",
                plane_id="v",
                shape=(4, 8),
                dtype="bfloat16",
                field_offset_bytes=0,
                page_stride_bytes=256,
            ),
        ),
    )


_LCM_SPECS = (
    CacheGroupSpec(
        group_id="history",
        retention="full_history",
        rows_per_page=4,
        entry_stride_tokens=1,
        sliding_window_tokens=None,
        family="history",
        transfer_policy="full_suffix",
    ),
)


def _recipe_contract(
    groups,
    *,
    ptr=0x10000,
    q=128,
    transfer_schema=CacheTransferSchema(),
    **solve,
):
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import pack

    specs = tuple(spec for spec, _ in groups)
    plan = pack(
        groups,
        prefix_granularity=q,
        max_padding_fraction=1.0,
        **solve,
    ).bind(3)
    return build_cache_transfer_contract(
        plan=plan,
        buffer=_cache_buffer(plan.arena_bytes, data_ptr=ptr),
        group_specs=specs,
        transfer_schema=transfer_schema,
    )[0]


def test_lcm_contract_registers_one_arena_and_preserves_group_geometry() -> None:
    plan = _two_plane_lcm_plan(3)
    layout, base_addr = build_cache_transfer_contract(
        plan=plan,
        buffer=_cache_buffer(plan.arena_bytes),
        group_specs=_LCM_SPECS,
    )

    assert layout.plan.num_lcm_blocks + 1 == 4
    assert layout.plan.groups[0].cache_blocks_per_lcm_block == 2
    fields = layout.fields_for_group("history")
    assert tuple(field.field_id for field in fields) == (
        "layer.0.k",
        "layer.0.v",
    )
    assert layout.plan.field_page_byte_offset(fields[0].field_id, 0) == 256
    assert base_addr == 0x10000


def test_peer_layout_allows_capacity_and_offsets_to_differ() -> None:
    small, _ = build_cache_transfer_contract(
        plan=_two_plane_lcm_plan(3),
        buffer=_cache_buffer(4096),
        group_specs=_LCM_SPECS,
    )
    large, _ = build_cache_transfer_contract(
        plan=_two_plane_lcm_plan(5),
        buffer=_cache_buffer(6144, data_ptr=0x20000),
        group_specs=_LCM_SPECS,
    )

    assert small.plan.num_lcm_blocks != large.plan.num_lcm_blocks
    assert small.plan.field_page_byte_offset(
        "layer.0.v", 0
    ) != large.plan.field_page_byte_offset("layer.0.v", 0)
    validate_cache_peer_layout(small, large)


def test_peer_layout_reports_prefix_granularity_mismatch() -> None:
    local = _layout()
    peer = replace(
        local,
        plan=replace(local.plan, prefix_granularity=8),
    )

    with pytest.raises(
        CacheContractError,
        match="^Cache P/D contract mismatch: prefix_granularity$",
    ):
        validate_cache_peer_layout(local, peer)


@pytest.mark.parametrize("family", ("mha", "mla", "dsa"))
def test_pd_derives_ordinary_transfer_metadata_from_physical_plan(
    family: str,
) -> None:
    import torch

    from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
    from tokenspeed.runtime.layers.attention.configs.dsa import DSAConfig
    from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
    from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.ordinary import (
        OrdinaryRecipe,
    )

    common = {
        "backend_name": "test",
        "num_attention_heads": 8,
        "num_kv_heads": 4,
        "head_dim": 8,
        "attn_tp_size": 2,
    }
    classes = {"mha": MHAConfig, "mla": MLAConfig, "dsa": DSAConfig}
    extras = {}
    if family != "mha":
        extras.update(
            kv_lora_rank=6,
            qk_nope_head_dim=4,
            qk_rope_head_dim=2,
            v_head_dim=4,
            scaling=1.0,
            kv_cache_dim=8,
        )
    if family == "dsa":
        extras.update(index_topk=4, index_head_dim=128, index_n_heads=1)
    spec = classes[family](
        **common,
        **extras,
        layer_types=(),
    )
    config = AttnConfig(
        device="cpu",
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        prefix_granularity=16,
        context_len=256,
        max_bs=4,
        max_graph_bs=4,
        kv_cache_quant_method="none",
        pd_disaggregation_enabled=True,
        components=(spec,),
    )

    recipe = OrdinaryRecipe(
        family=family,
        server_args=SimpleNamespace(max_total_tokens=None),
        model_config=SimpleNamespace(num_attention_layers=2),
        attn_config=config,
        draft_model_config=None,
        draft_attn_config=None,
        cache_budget_bytes=1 << 24,
        decode_input_tokens=1,
        overlap_schedule_depth=0,
    )
    groups = recipe.groups()
    layout = _recipe_contract(
        groups,
        q=config.prefix_granularity,
        cache_blocks_per_lcm_block=recipe.packing(groups),
        alignment=1,
    )
    model_config = SimpleNamespace(
        num_attention_layers=2,
        num_key_value_heads=4,
        hf_config=SimpleNamespace(),
    )
    layout = replace(
        layout,
        transfer_schema=build_cache_transfer_schema(
            layout.plan,
            model_config=model_config,
        ),
    )

    assert layout.plan.prefix_granularity == 16
    assert len(layout.group_specs) == 1
    group = layout.group_specs[0]
    assert (group.group_id, group.family) == ("full_attention", "history")
    # Packing lives on the plan, not on the group spec.
    assert layout.plan.group("full_attention").cache_blocks_per_lcm_block == 1

    fields_by_id = {
        field.field_id: field for field in layout.fields_for_group(group.group_id)
    }
    suffixes = {
        "mha": ("k", "v"),
        "mla": ("latent_kv",),
        "dsa": ("index_k", "latent_kv"),
    }[family]
    expected_field_ids = tuple(
        f"layer.{layer_id}.{suffix}" for layer_id in range(2) for suffix in suffixes
    )
    assert tuple(sorted(fields_by_id)) == expected_field_ids
    for field_id, field in fields_by_id.items():
        is_index = field_id.endswith(".index_k")
        expected_shape = (
            (16, 132)
            if family == "dsa" and is_index
            else (16, 1, 8) if field_id.endswith(".latent_kv") else (16, 2, 8)
        )
        # The recipe put the dtype in the plan; the contract reads it back.
        expected_dtype = "uint8" if family == "dsa" and is_index else "bfloat16"
        assert layout.field_dtype(field_id) == expected_dtype
        assert field.shape == expected_shape
        assert field.element_size == (1 if expected_dtype == "uint8" else 2)
        partition = layout.transfer_schema.partition_for(field_id)
        if field_id.endswith((".k", ".v")):
            assert partition is not None
            assert partition.axis == 1
            assert partition.global_extent == 4
        else:
            assert partition is None


def test_producer_schedule_groups_draft_fields_in_the_final_step() -> None:
    spec = _group_spec("history", "history", "full_suffix")
    layout = _contract(
        (spec,),
        tuple(
            _field("history", f"layer.{layer_id}.kv", offset=layer_id)
            for layer_id in range(4)
        ),
    )

    schedule = build_cache_fields_by_producer_step(layout.plan, num_target_layers=2)

    assert schedule.fields_by_step == (
        ("layer.0.kv",),
        ("layer.1.kv",),
        ("layer.2.kv", "layer.3.kv"),
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
