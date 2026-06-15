from tokenspeed.runtime.pd.mooncake.receiver import _build_buffer_layout_pair
from tokenspeed.runtime.pd.transfer_plan import (
    BufferKind,
    ParallelLayout,
    PDTransferPlanner,
)


def test_replicated_prefill_kv_heads_transfer_to_decode_full_kv_heads():
    prefill_buffer, decode_buffer = _build_buffer_layout_pair(
        buffer_index=0,
        buffer_kind=BufferKind.TARGET_K,
        sharded_axis="kv_head",
        prefill_item_len=16_384,
        decode_item_len=32_768,
        prefill_unit_len=256,
        decode_unit_len=256,
        prefill_tp_size=4,
        decode_tp_size=1,
    )

    assert prefill_buffer.logical_size == 128
    assert prefill_buffer.tp_replica_group_size == 2
    assert decode_buffer.logical_size == 128
    assert decode_buffer.tp_replica_group_size == 1

    planner = PDTransferPlanner(
        prefill_layout=ParallelLayout(role="prefill", world_size=4),
        decode_layout=ParallelLayout(role="decode", world_size=1),
        prefill_buffers=(prefill_buffer,),
        decode_buffers=(decode_buffer,),
    )
    plan = planner.plan_for_decode_rank(0)

    assert plan.plan_kind == "fragmented"
    assert plan.target_prefill_ranks == (0, 2)
    assert plan.required_prefill_response_num == 2
    assert plan.required_dst_info_num_by_prefill_rank == {0: 1, 2: 1}

    first_head = plan.fragments_by_prefill_rank[0][0]
    second_head = plan.fragments_by_prefill_rank[2][0]
    assert first_head.src_byte_offset == 0
    assert first_head.dst_byte_offset == 0
    assert first_head.bytes_per_page == 16_384
    assert second_head.src_byte_offset == 0
    assert second_head.dst_byte_offset == 16_384
    assert second_head.bytes_per_page == 16_384
