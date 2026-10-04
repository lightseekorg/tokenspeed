import os
import sys
from types import SimpleNamespace

import pytest

# CPU-only tests scheduled in runtime-1gpu because they import the full runtime.
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from runtime.cache_pd_test_utils import block_manifest  # noqa: E402
from runtime.cache_pd_test_utils import group  # noqa: E402
from runtime.cache_pd_test_utils import segment  # noqa: E402
from runtime.cache_pd_test_utils import layout as make_layout  # noqa: E402

from tokenspeed.runtime.pd.transfer_plan import (
    CachePageOwnerFilter,
    CacheTransferPlanner,
    UnsupportedPDLayoutError,
)


def _paged_layout(
    *,
    local_heads: int,
    page_stride: int,
    page_zero_offset: int,
    global_heads: int = 4,
):
    segments = [
        segment(
            "layer.0.k",
            dtype="bfloat16",
            shape=(2, local_heads, 2),
            offset=page_zero_offset,
            stride=page_stride,
            axis=1,
            extent=global_heads,
        )
    ]
    segments.append(
        segment(
            "layer.1.latent",
            dtype="bfloat16",
            shape=(2, 1),
            offset=page_zero_offset + 1024,
            stride=16,
        )
    )
    return make_layout(group("history", *segments), page_bytes=128)


def _composite_paged_layout(
    *,
    shape: tuple[int, ...],
    partition_axis: int,
    global_parts: tuple[int, ...],
    page_stride: int,
):
    return make_layout(
        group(
            "state",
            segment(
                "layer.0.conv",
                dtype="bfloat16",
                shape=shape,
                stride=page_stride,
                axis=partition_axis,
                extent=sum(global_parts),
                parts=global_parts,
            ),
            family="state",
        ),
        page_bytes=128,
    )


def _planner(prefill_tp, decode_tp, prefill_layout, decode_layout):
    return CacheTransferPlanner(
        prefill_tp_size=prefill_tp,
        decode_tp_size=decode_tp,
        prefill_layout=prefill_layout,
        decode_layout=decode_layout,
        prefill_field_ids=None,
    )


def _paged_planner(
    prefill_tp,
    decode_tp,
    prefill_heads,
    decode_heads,
    prefill_stride,
    decode_stride,
    *,
    global_heads=4,
):
    return _planner(
        prefill_tp,
        decode_tp,
        _paged_layout(
            local_heads=prefill_heads,
            global_heads=global_heads,
            page_stride=prefill_stride,
            page_zero_offset=128,
        ),
        _paged_layout(
            local_heads=decode_heads,
            global_heads=global_heads,
            page_stride=decode_stride,
            page_zero_offset=256,
        ),
    )


def _composite_planner(
    prefill_tp,
    decode_tp,
    prefill_shape,
    decode_shape,
    partition_axis,
    global_parts,
    prefill_stride,
    decode_stride,
):
    return _planner(
        prefill_tp,
        decode_tp,
        _composite_paged_layout(
            shape=prefill_shape,
            partition_axis=partition_axis,
            global_parts=global_parts,
            page_stride=prefill_stride,
        ),
        _composite_paged_layout(
            shape=decode_shape,
            partition_axis=partition_axis,
            global_parts=global_parts,
            page_stride=decode_stride,
        ),
    )


def test_cache_planner_splits_token_major_rows_from_tp1_to_tp2():
    planner = _paged_planner(1, 2, 4, 2, 64, 32)

    first = planner.plan_for_decode_rank(0)
    second = planner.plan_for_decode_rank(1)

    assert first.target_prefill_ranks == (0,)
    assert second.target_prefill_ranks == (0,)
    second_k = next(
        fragment
        for fragment in second.fragments_by_prefill_rank[0]
        if fragment.field_id == "layer.0.k"
    )
    assert second_k.rows_per_page == 2
    assert second_k.src_row_stride_bytes == 16
    assert second_k.dst_row_stride_bytes == 8
    assert second_k.src_byte_offset == 8
    assert second_k.dst_byte_offset == 0
    assert second_k.bytes_per_row == 8


def test_cache_planner_merges_tp2_to_tp1():
    planner = _paged_planner(2, 1, 2, 4, 32, 64)

    plan = planner.plan_for_decode_rank(0)

    assert plan.target_prefill_ranks == (0, 1)
    second_k = next(
        fragment
        for fragment in plan.fragments_by_prefill_rank[1]
        if fragment.field_id == "layer.0.k"
    )
    assert second_k.src_byte_offset == 0
    assert second_k.dst_byte_offset == 8
    assert second_k.rows_per_page == 2
    assert all(
        fragment.field_id != "layer.1.latent"
        for fragment in plan.fragments_by_prefill_rank[1]
    )


def test_cache_planner_handles_gqa_replicas_and_idle_prefill_ranks() -> None:
    planner = _paged_planner(4, 1, 1, 2, 32, 64, global_heads=2)

    plan = planner.plan_for_decode_rank(0)

    assert plan.target_prefill_ranks == (0, 2)


def test_cache_planner_maps_non_multiple_tp_sizes() -> None:
    planner = _paged_planner(2, 3, 3, 2, 96, 64, global_heads=6)

    plans = tuple(planner.plan_for_decode_rank(rank) for rank in range(3))

    assert [plan.target_prefill_ranks for plan in plans] == [(0,), (0, 1), (1,)]


def test_cache_planner_splits_each_qkv_part_from_tp1_to_tp2():
    planner = _composite_planner(1, 2, (16, 2), (8, 2), 0, (4, 4, 8), 64, 32)

    plan = planner.plan_for_decode_rank(1)

    fragments = plan.fragments_by_prefill_rank[0]
    assert len(fragments) == 3
    assert [fragment.src_byte_offset for fragment in fragments] == [8, 24, 48]
    assert [fragment.dst_byte_offset for fragment in fragments] == [0, 8, 16]
    assert [fragment.bytes_per_row for fragment in fragments] == [8, 8, 16]


def test_composite_partition_on_inner_axis_keeps_full_parent_row_stride():
    planner = _composite_planner(1, 2, (2, 8, 3), (2, 4, 3), 1, (4, 4), 96, 48)

    fragments = planner.plan_for_decode_rank(1).fragments_by_prefill_rank[0]

    assert len(fragments) == 2
    assert [fragment.src_byte_offset for fragment in fragments] == [12, 36]
    assert [fragment.dst_byte_offset for fragment in fragments] == [0, 12]
    assert all(fragment.rows_per_page == 2 for fragment in fragments)
    assert all(fragment.src_row_stride_bytes == 48 for fragment in fragments)
    assert all(fragment.dst_row_stride_bytes == 24 for fragment in fragments)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


# ---- prefill chunk-pipeline (PP) layer-window routing ----


def test_pp_layer_window_filters_fragments():
    """A stage's planner only routes fields inside its layer window."""
    layout = _paged_layout(
        local_heads=4, global_heads=4, page_stride=4096, page_zero_offset=128
    )
    stage0 = CacheTransferPlanner(
        prefill_tp_size=1,
        decode_tp_size=1,
        prefill_layout=layout,
        decode_layout=layout,
        prefill_field_ids=frozenset({"layer.0.k"}),
    )
    stage1 = CacheTransferPlanner(
        prefill_tp_size=1,
        decode_tp_size=1,
        prefill_layout=layout,
        decode_layout=layout,
        prefill_field_ids=frozenset({"layer.1.latent"}),
    )
    frags0 = stage0.plan_for_decode_rank(0).fragments_by_prefill_rank[0]
    frags1 = stage1.plan_for_decode_rank(0).fragments_by_prefill_rank[0]
    assert {f.field_id for f in frags0} == {"layer.0.k"}
    assert {f.field_id for f in frags1} == {"layer.1.latent"}
    # The stage union covers exactly the plan's full field set.
    all_fields = {field.field_id for field in layout.plan.fields}
    assert {f.field_id for f in frags0} | {f.field_id for f in frags1} == all_fields


def test_pp_receiver_calc_merges_stage_routes():
    """Decode's route plan spans pp*tp source ranks with disjoint fields."""
    from types import SimpleNamespace

    from tokenspeed.runtime.pd.mooncake.decode import PrefillParallelInfo
    from tokenspeed.runtime.pd.mooncake.receiver import _calc

    layout = _paged_layout(
        local_heads=4, global_heads=4, page_stride=4096, page_zero_offset=128
    )
    kv_mgr = SimpleNamespace(
        topology=SimpleNamespace(tp_size=1, tp_rank=0),
        kv_args=SimpleNamespace(cache_layout=layout),
    )
    info = PrefillParallelInfo(
        tp_size=2,  # registered world = pp(2) x tp(1)
        dp_size=1,
        cache_fields_by_stage=(("layer.0.k",), ("layer.1.latent",)),
        cache_layout=layout,
        pp_size=2,
    )
    assert info.prefill_tp_size_per_dp_rank == 1
    plan = _calc(kv_mgr, info)
    frags = plan.transfer_plan.fragments_by_prefill_rank
    # Stage-major dense ranks: stage0 -> rank 0 (layer.0), stage1 -> rank 1 (layer.1).
    assert set(frags) == {0, 1}
    assert {f.field_id for f in frags[0]} == {"layer.0.k"}
    assert {f.field_id for f in frags[1]} == {"layer.1.latent"}


def test_pp_receiver_calc_honors_layer_partition():
    """An explicit prefill layer partition moves fields between stage routes."""
    from types import SimpleNamespace

    from tokenspeed.runtime.pd.mooncake.decode import PrefillParallelInfo
    from tokenspeed.runtime.pd.mooncake.receiver import _calc

    # Three layers so partition (1, 2) differs from the even split (2, 1).
    layout = make_layout(
        group(
            "history",
            segment(
                "layer.0.latent", dtype="bfloat16", shape=(2, 1), offset=0, stride=16
            ),
            segment(
                "layer.1.latent", dtype="bfloat16", shape=(2, 1), offset=512, stride=16
            ),
            segment(
                "layer.2.latent", dtype="bfloat16", shape=(2, 1), offset=1024, stride=16
            ),
        ),
        page_bytes=128,
    )
    kv_mgr = SimpleNamespace(
        topology=SimpleNamespace(tp_size=1, tp_rank=0),
        kv_args=SimpleNamespace(cache_layout=layout),
    )

    def stage_fields(partition):
        info = PrefillParallelInfo(
            tp_size=2,
            dp_size=1,
            cache_fields_by_stage=(
                (("layer.0.latent", "layer.1.latent"), ("layer.2.latent",))
                if partition is None
                else (("layer.0.latent",), ("layer.1.latent", "layer.2.latent"))
            ),
            cache_layout=layout,
            pp_size=2,
        )
        frags = _calc(kv_mgr, info).transfer_plan.fragments_by_prefill_rank
        return {rank: {f.field_id for f in fields} for rank, fields in frags.items()}

    # Even split (partition None): stage0 gets layers 0-1, stage1 layer 2.
    assert stage_fields(None) == {
        0: {"layer.0.latent", "layer.1.latent"},
        1: {"layer.2.latent"},
    }
    # Explicit (1, 2): stage0 gets layer 0 only, stage1 layers 1-2.
    assert stage_fields((1, 2)) == {
        0: {"layer.0.latent"},
        1: {"layer.1.latent", "layer.2.latent"},
    }


@pytest.mark.parametrize(
    "placement",
    [
        (("layer.0.k", "layer.1.latent"), ("layer.0.k",)),
        (("layer.0.k",),),
        (("layer.0.k", "unknown"), ("layer.1.latent",)),
    ],
)
def test_stage_placement_rejects_duplicate_missing_or_unknown_fields(placement):
    from tokenspeed.runtime.pd.transfer_plan import build_pipeline_transfer_plan

    layout = _paged_layout(
        local_heads=4, global_heads=4, page_stride=4096, page_zero_offset=128
    )
    with pytest.raises(ValueError, match="exactly once"):
        build_pipeline_transfer_plan(
            prefill_tp_size=1,
            decode_tp_size=1,
            decode_tp_rank=0,
            prefill_layout=layout,
            decode_layout=layout,
            cache_fields_by_stage=placement,
        )


def test_stage_placement_supports_noncontiguous_fields_without_model_counts():
    from tokenspeed.runtime.pd.transfer_plan import build_pipeline_transfer_plan

    layout = make_layout(
        group(
            "history",
            *[
                segment(
                    f"layer.{index}.latent",
                    dtype="bfloat16",
                    shape=(2, 1),
                    offset=index * 512,
                    stride=16,
                )
                for index in range(4)
            ],
        ),
        page_bytes=128,
    )
    fields = (
        ("layer.0.latent", "layer.3.latent"),
        ("layer.1.latent", "layer.2.latent"),
    )
    plan, _ = build_pipeline_transfer_plan(
        prefill_tp_size=1,
        decode_tp_size=1,
        decode_tp_rank=0,
        prefill_layout=layout,
        decode_layout=layout,
        cache_fields_by_stage=fields,
    )
    assert {
        rank: {fragment.field_id for fragment in fragments}
        for rank, fragments in plan.fragments_by_prefill_rank.items()
    } == {rank: set(stage) for rank, stage in enumerate(fields)}


# ---- DCP page-sharded prefill (sharded source, whole destination) ----


def _latent_layout(*, shard_count: int, extra_replicated: bool = False):
    """MLA-style layout: replicated latent fields, optionally a second group."""
    groups = [
        group(
            "history",
            segment("layer.0.latent", dtype="bfloat16", shape=(2, 1), stride=16),
            segment(
                "layer.1.latent", dtype="bfloat16", shape=(2, 1), offset=512, stride=16
            ),
            shard_count=shard_count,
        )
    ]
    if extra_replicated:
        groups.append(
            group(
                "swa",
                segment(
                    "layer.2.swa",
                    dtype="bfloat16",
                    shape=(2, 1),
                    offset=1024,
                    stride=16,
                ),
            )
        )
    return make_layout(*groups, page_bytes=128)


def test_sharded_prefill_fans_every_subgroup_rank_to_one_decode_rank():
    """P TP4 x DCP4 -> D TP1: all four P ranks send, each tagged as an owner."""
    planner = _planner(
        4, 1, _latent_layout(shard_count=4), _latent_layout(shard_count=1)
    )
    assert planner.has_sharded_groups

    plan = planner.plan_for_decode_rank(0)

    assert plan.target_prefill_ranks == (0, 1, 2, 3)
    assert plan.owner_filters_by_prefill_rank == {
        rank: {"history": CachePageOwnerFilter(rank, 4)} for rank in range(4)
    }
    for rank in range(4):
        fragments = plan.fragments_by_prefill_rank[rank]
        assert [f.field_id for f in fragments] == ["layer.0.latent", "layer.1.latent"]
        assert all(f.rows_per_page == 1 and f.bytes_per_row == 4 for f in fragments)
    # Every P rank serves the only D rank; nobody is a dummy rendezvous rank.
    assert planner.decode_ranks_by_prefill_rank == {
        rank: frozenset({0}) for rank in range(4)
    }


def test_sharded_prefill_picks_the_replica_subgroup_per_decode_rank():
    """P TP4 x DCP2 -> D TP2: each D rank reads one whole DCP subgroup."""
    planner = _planner(
        4, 2, _latent_layout(shard_count=2), _latent_layout(shard_count=1)
    )

    first = planner.plan_for_decode_rank(0)
    second = planner.plan_for_decode_rank(1)

    assert first.target_prefill_ranks == (0, 1)
    assert second.target_prefill_ranks == (2, 3)
    assert first.owner_filters_by_prefill_rank == {
        0: {"history": CachePageOwnerFilter(0, 2)},
        1: {"history": CachePageOwnerFilter(1, 2)},
    }
    assert second.owner_filters_by_prefill_rank == {
        2: {"history": CachePageOwnerFilter(0, 2)},
        3: {"history": CachePageOwnerFilter(1, 2)},
    }
    assert planner.decode_ranks_by_prefill_rank == {
        0: frozenset({0}),
        1: frozenset({0}),
        2: frozenset({1}),
        3: frozenset({1}),
    }


def test_sharded_prefill_leaves_the_equal_tp_fast_path():
    """Equal TP still fans a sharded group out; a replicated group does not."""
    planner = _planner(
        2,
        2,
        _latent_layout(shard_count=2, extra_replicated=True),
        _latent_layout(shard_count=1, extra_replicated=True),
    )

    plan = planner.plan_for_decode_rank(1)

    assert plan.target_prefill_ranks == (0, 1)
    assert plan.owner_filters_by_prefill_rank == {
        0: {"history": CachePageOwnerFilter(0, 2)},
        1: {"history": CachePageOwnerFilter(1, 2)},
    }
    # The replicated group keeps its one same-index source and no owner filter.
    assert {f.field_id for f in plan.fragments_by_prefill_rank[0]} == {
        "layer.0.latent",
        "layer.1.latent",
    }
    assert {f.field_id for f in plan.fragments_by_prefill_rank[1]} == {
        "layer.0.latent",
        "layer.1.latent",
        "layer.2.swa",
    }
    assert planner.decode_ranks_by_prefill_rank == {
        0: frozenset({0, 1}),
        1: frozenset({0, 1}),
    }


def test_equal_tp_sharded_route_sends_through_the_pages_api():
    """P TP2 x DCP2 -> D TP2 leaves the empty-fragment route, yet every copy
    still goes out as one pages x fields grid per group."""
    from tokenspeed.runtime.pd.mooncake.pack import PageFieldCopies

    prefill_layout = _latent_layout(shard_count=2, extra_replicated=True)
    decode_layout = _latent_layout(shard_count=1, extra_replicated=True)
    plan = _planner(2, 2, prefill_layout, decode_layout).plan_for_decode_rank(1)
    src_ptr, dst_ptr = 0x10000, 0x20000
    source_manifest = block_manifest(("history", (1, 2, 3, 4)), ("swa", (5, 6)))
    destination_manifest = block_manifest(
        ("history", (10, 11, 12, 13)), ("swa", (14, 15))
    )

    def grids(rank):
        items = list(
            _sender(prefill_layout, src_ptr)._cache_transfer_blocks(
                dst_ptr=dst_ptr,
                src_block_manifest=source_manifest,
                dst_block_manifest=destination_manifest,
                transfer_fragments=plan.fragments_by_prefill_rank[rank],
                owner_filters=plan.owner_filters_by_prefill_rank[rank],
                dst_cache_layout=decode_layout,
            )
        )
        assert all(isinstance(item, PageFieldCopies) for item in items)
        return [
            (item.src_pages.tolist(), item.dst_pages.tolist(), item.fields.shape[0])
            for item in items
        ]

    # Rank 0 owns virtual blocks 1, 3 (local pages 1, 2) of the two-field
    # sharded group and nothing of the replicated one.
    assert grids(0) == [([1, 2], [10, 12], 2)]
    # Rank 1 owns 2, 4 and, as decode rank 1's same-index source, sends the
    # replicated group whole.
    assert grids(1) == [([1, 2], [11, 13], 2), ([5, 6], [14, 15], 1)]


def test_unsharded_layouts_keep_the_equal_tp_fast_path():
    planner = _planner(
        2, 2, _latent_layout(shard_count=1), _latent_layout(shard_count=1)
    )
    plan = planner.plan_for_decode_rank(1)
    assert plan.fragments_by_prefill_rank == {1: ()}
    # The one source rank still carries its (empty) set of decisions.
    assert plan.owner_filters_by_prefill_rank == {1: {}}
    assert planner.decode_ranks_by_prefill_rank == {
        0: frozenset({0}),
        1: frozenset({1}),
    }


def test_equal_tp_stage_subset_routes_gqa_replicas_like_the_full_plan():
    """One predicate picks the route: a PP stage at equal TP leaves the
    whole-copy path, so its served-rank sets follow the fragment route."""
    layout = _paged_layout(
        local_heads=1, global_heads=2, page_stride=4096, page_zero_offset=128
    )
    full = CacheTransferPlanner(
        prefill_tp_size=4,
        decode_tp_size=4,
        prefill_layout=layout,
        decode_layout=layout,
        prefill_field_ids=None,
    )
    stage = CacheTransferPlanner(
        prefill_tp_size=4,
        decode_tp_size=4,
        prefill_layout=layout,
        decode_layout=layout,
        prefill_field_ids=frozenset({"layer.0.k"}),
    )
    # Whole copy: every rank serves itself.
    assert full.decode_ranks_by_prefill_rank == {
        rank: frozenset({rank}) for rank in range(4)
    }
    # Fragment route: the representative rank of each GQA replica pair serves
    # both of its decode ranks, exactly as plan_for_decode_rank routes them.
    assert stage.decode_ranks_by_prefill_rank == {
        0: frozenset({0, 1}),
        1: frozenset(),
        2: frozenset({2, 3}),
        3: frozenset(),
    }
    for decode_rank in range(4):
        plan = stage.plan_for_decode_rank(decode_rank)
        for prefill_rank in plan.target_prefill_ranks:
            assert decode_rank in stage.decode_ranks_by_prefill_rank[prefill_rank]


def test_sharded_decode_destination_is_rejected():
    with pytest.raises(UnsupportedPDLayoutError, match="sharded on Decode"):
        _planner(2, 2, _latent_layout(shard_count=1), _latent_layout(shard_count=2))


def test_shard_count_must_divide_prefill_tp():
    with pytest.raises(UnsupportedPDLayoutError, match="does not divide"):
        _planner(3, 1, _latent_layout(shard_count=2), _latent_layout(shard_count=1))


def test_head_partitioned_field_in_a_sharded_group_is_rejected():
    def layout(shard_count):
        return make_layout(
            group(
                "history",
                segment(
                    "layer.0.k",
                    dtype="bfloat16",
                    shape=(2, 2, 2),
                    stride=32,
                    axis=1,
                    extent=4,
                ),
                shard_count=shard_count,
            ),
            page_bytes=128,
        )

    with pytest.raises(UnsupportedPDLayoutError, match="head-partitioned"):
        _planner(2, 2, layout(2), layout(1))


def test_pipeline_plan_offsets_owner_filters_by_stage():
    from tokenspeed.runtime.pd.transfer_plan import build_pipeline_transfer_plan

    plan, dummy_ranks = build_pipeline_transfer_plan(
        prefill_tp_size=2,
        decode_tp_size=1,
        decode_tp_rank=0,
        prefill_layout=_latent_layout(shard_count=2),
        decode_layout=_latent_layout(shard_count=1),
        cache_fields_by_stage=(("layer.0.latent",), ("layer.1.latent",)),
    )

    assert dummy_ranks == ()
    assert {
        rank: [f.field_id for f in fragments]
        for rank, fragments in plan.fragments_by_prefill_rank.items()
    } == {
        0: ["layer.0.latent"],
        1: ["layer.0.latent"],
        2: ["layer.1.latent"],
        3: ["layer.1.latent"],
    }
    assert plan.owner_filters_by_prefill_rank == {
        0: {"history": CachePageOwnerFilter(0, 2)},
        1: {"history": CachePageOwnerFilter(1, 2)},
        2: {"history": CachePageOwnerFilter(0, 2)},
        3: {"history": CachePageOwnerFilter(1, 2)},
    }


def _copies(items) -> list[tuple[int, int, int]]:
    """Expand sender items to ``(src, dst, length)`` triples."""
    from tokenspeed.runtime.pd.mooncake.pack import (
        PackedCopy,
        PageFieldCopies,
        expand_packed_copy,
    )

    copies = []
    for item in items:
        if isinstance(item, PageFieldCopies):
            copies.extend(
                (
                    int(src_base + page * src_stride),
                    int(dst_base + peer * dst_stride),
                    int(length),
                )
                for src_base, src_stride, dst_base, dst_stride, length in item.fields.tolist()
                for page, peer in zip(
                    item.src_pages.tolist(), item.dst_pages.tolist(), strict=True
                )
            )
        elif isinstance(item, PackedCopy):
            copies.extend(expand_packed_copy(item))
        else:
            copies.append(item)
    return copies


def _sender(layout, src_ptr: int):
    from tokenspeed.runtime.pd.mooncake.prefill import MooncakeKVManagerPrefill

    manager = object.__new__(MooncakeKVManagerPrefill)
    manager.kv_args = SimpleNamespace(cache_layout=layout, kv_data_ptr=src_ptr)
    return manager


def _hybrid_layout(*, local_kda_heads: int, shard_count: int):
    """Kimi K3 shape: a page-sharded MLA group plus head-partitioned KDA state."""
    return make_layout(
        group(
            "history",
            segment("layer.0.latent", dtype="bfloat16", shape=(2, 1), stride=16),
            shard_count=shard_count,
        ),
        group(
            "kda",
            segment(
                "layer.1.state",
                dtype="bfloat16",
                shape=(local_kda_heads, 4),
                offset=512,
                stride=64,
                axis=0,
                extent=8,
            ),
            family="state",
        ),
        page_bytes=128,
    )


def test_every_target_rank_of_a_hybrid_route_can_send_with_its_own_decisions():
    """P TP8 x DCP2 -> D TP2: the KDA head partition routes ranks outside the
    MLA subgroup; they hold no 'history' fragments and must say so."""
    prefill_layout = _hybrid_layout(local_kda_heads=1, shard_count=2)
    decode_layout = _hybrid_layout(local_kda_heads=4, shard_count=1)
    planner = _planner(8, 2, prefill_layout, decode_layout)
    src_ptr, dst_ptr = 0x10000, 0x20000
    kda_dst_base = dst_ptr + 512
    source_manifest = block_manifest(("history", (1, 2, 3, 4)), ("kda", (5,)))
    destination_manifest = block_manifest(("history", (10, 11, 12, 13)), ("kda", (7,)))

    for decode_rank, subgroup in ((0, (0, 1)), (1, (4, 5))):
        plan = planner.plan_for_decode_rank(decode_rank)
        assert plan.target_prefill_ranks == tuple(range(subgroup[0], subgroup[0] + 4))
        for rank in plan.target_prefill_ranks:
            decisions = plan.owner_filters_by_prefill_rank[rank]
            assert set(decisions) == {"history"}
            if rank in subgroup:
                assert decisions["history"] == CachePageOwnerFilter(rank % 2, 2)
            else:
                assert decisions["history"] is None
            copies = _copies(
                _sender(prefill_layout, src_ptr)._cache_transfer_blocks(
                    dst_ptr=dst_ptr,
                    src_block_manifest=source_manifest,
                    dst_block_manifest=destination_manifest,
                    transfer_fragments=plan.fragments_by_prefill_rank[rank],
                    owner_filters=decisions,
                    dst_cache_layout=decode_layout,
                )
            )
            history_copies = [copy for copy in copies if copy[1] < kda_dst_base]
            kda_copies = [copy for copy in copies if copy[1] >= kda_dst_base]
            # Every rank of the four sends its KDA head slice of the one page.
            assert len(kda_copies) == 1
            if rank in subgroup:
                # Owned virtual blocks 1,3 (owner 0) or 2,4 (owner 1) come from
                # local pages 1,2 and land at the matching manifest positions.
                owner = rank % 2
                assert history_copies == [
                    (src_ptr + local * 16, dst_ptr + remote * 16, 4)
                    for local, remote in ((1, 10 + owner), (2, 12 + owner))
                ]
            else:
                assert history_copies == []


def test_pipeline_stage_without_a_sharded_group_decides_none_and_sends_rest():
    """Two sharded groups, stage 0 holding only one: its ranks send that
    group's owned pages and nothing of the other."""
    from tokenspeed.runtime.pd.transfer_plan import build_pipeline_transfer_plan

    def layout(shard_count):
        return make_layout(
            group(
                "history",
                segment("layer.0.latent", dtype="bfloat16", shape=(2, 1), stride=16),
                shard_count=shard_count,
            ),
            group(
                "index",
                segment(
                    "layer.1.index",
                    dtype="bfloat16",
                    shape=(2, 1),
                    offset=512,
                    stride=16,
                ),
                shard_count=shard_count,
            ),
            page_bytes=128,
        )

    prefill_layout, decode_layout = layout(2), layout(1)
    plan, dummy_ranks = build_pipeline_transfer_plan(
        prefill_tp_size=2,
        decode_tp_size=1,
        decode_tp_rank=0,
        prefill_layout=prefill_layout,
        decode_layout=decode_layout,
        cache_fields_by_stage=(("layer.0.latent",), ("layer.1.index",)),
    )
    assert dummy_ranks == ()
    assert plan.owner_filters_by_prefill_rank == {
        0: {"history": CachePageOwnerFilter(0, 2), "index": None},
        1: {"history": CachePageOwnerFilter(1, 2), "index": None},
        2: {"history": None, "index": CachePageOwnerFilter(0, 2)},
        3: {"history": None, "index": CachePageOwnerFilter(1, 2)},
    }

    src_ptr, dst_ptr = 0x10000, 0x20000
    copies = _copies(
        _sender(prefill_layout, src_ptr)._cache_transfer_blocks(
            dst_ptr=dst_ptr,
            src_block_manifest=block_manifest(("history", (1, 2)), ("index", (3, 4))),
            dst_block_manifest=block_manifest(("history", (5, 6)), ("index", (7, 8))),
            transfer_fragments=plan.fragments_by_prefill_rank[1],
            owner_filters=plan.owner_filters_by_prefill_rank[1],
            dst_cache_layout=decode_layout,
        )
    )
    # Stage-0 rank 1 owns virtual block 2 (local page 1) of 'history' only.
    assert copies == [(src_ptr + 1 * 16, dst_ptr + 6 * 16, 4)]


def test_receiver_calc_targets_every_rank_of_a_sharded_prefill():
    from types import SimpleNamespace

    from tokenspeed.runtime.pd.mooncake.decode import PrefillParallelInfo
    from tokenspeed.runtime.pd.mooncake.receiver import _calc

    kv_mgr = SimpleNamespace(
        topology=SimpleNamespace(tp_size=1, tp_rank=0),
        kv_args=SimpleNamespace(cache_layout=_latent_layout(shard_count=1)),
    )
    info = PrefillParallelInfo(
        tp_size=4,
        dp_size=1,
        cache_fields_by_stage=(("layer.0.latent", "layer.1.latent"),),
        cache_layout=_latent_layout(shard_count=4),
    )

    route = _calc(kv_mgr, info)

    assert route.target_tp_ranks == (0, 1, 2, 3)
    assert route.dummy_tp_ranks == ()
