import os
import sys

import pytest

# CPU-only tests scheduled in runtime-1gpu because they import the full runtime.
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from runtime.cache_pd_test_utils import group  # noqa: E402
from runtime.cache_pd_test_utils import segment  # noqa: E402
from runtime.cache_pd_test_utils import layout as make_layout  # noqa: E402

from tokenspeed.runtime.pd.transfer_plan import (
    CacheTransferPlanner,
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
