from types import SimpleNamespace

from tokenspeed.runtime.pd.decode_executor import DisaggDecodeExecutor
from tokenspeed.runtime.pd.mooncake.receiver import (
    _build_buffer_layout_pair,
)
from tokenspeed.runtime.pd.prefill_executor import DisaggPrefillExecutor
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


def test_prefill_transfer_uses_decode_prefix_and_sender_progress():
    executor = DisaggPrefillExecutor.__new__(DisaggPrefillExecutor)
    executor.page_size = 64
    executor._decode_prefix_len = lambda _room: 128
    sender = SimpleNamespace(bootstrap_room=1, curr_idx=7)
    op = SimpleNamespace(
        extend_prefix_lens=[8192],
        input_lengths=[8109],
        prefill_lengths=[16301],
        occupied_pages=[list(range(255))],
    )

    indices, index_slice, is_last = executor._prefill_page_window(op, 0, sender)

    assert indices.tolist() == list(range(9, 255))
    assert index_slice == slice(7, 253)
    assert is_last


def test_decode_transfer_excludes_reserved_page_after_aligned_prompt():
    executor = DisaggDecodeExecutor.__new__(DisaggDecodeExecutor)
    executor.page_size = 64
    executor._request_pool_indices = {}
    calls = []
    executor.receivers = {
        "request": SimpleNamespace(prefill=lambda *args: calls.append(args))
    }
    op = SimpleNamespace(
        request_ids=["request"],
        occupied_pages=[list(range(129))],
        begins=[0],
        sizes=[129],
        request_pool_indices=[3],
        extend_prefix_lens=[4096],
        prefill_lengths=[8192],
    )

    executor._prefill(op)

    assert len(calls) == 1
    assert calls[0][0].tolist() == list(range(64, 128))
