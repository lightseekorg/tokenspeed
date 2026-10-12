from __future__ import annotations

import os
import sys
from contextlib import nullcontext
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

from runtime.cache_pd_test_utils import block_manifest as make_block_manifest
from runtime.cache_pd_test_utils import group as make_group  # noqa: E402
from runtime.cache_pd_test_utils import layout as make_layout
from runtime.cache_pd_test_utils import operation as make_operation
from runtime.cache_pd_test_utils import segment as make_segment

from tokenspeed.runtime.pd.cache_protocol import (  # noqa: E402
    CachePDBlockManifest,
    CacheTransferContract,
)
from tokenspeed.runtime.pd.mooncake.entities import (  # noqa: E402
    KVArgsRegisterInfo,
    TransferInfo,
    TransferKVChunk,
)
from tokenspeed.runtime.pd.mooncake.pack import (  # noqa: E402
    PageFieldCopies,
    flatten_transfer_blocks,
)
from tokenspeed.runtime.pd.topology import PDParallelTopology  # noqa: E402
from tokenspeed.runtime.pd.transfer_plan import CacheTransferFragment  # noqa: E402


def _topology(
    *,
    tp_size: int = 1,
    tp_rank: int = 0,
    dp_size: int = 1,
    dp_rank: int = 0,
    world_size: int | None = None,
    global_rank: int = 0,
) -> PDParallelTopology:
    return PDParallelTopology(
        tp_size=tp_size,
        tp_rank=tp_rank,
        dp_size=dp_size,
        dp_rank=dp_rank,
        world_size=world_size or tp_size * dp_size,
        global_rank=global_rank,
    )


def _layout(
    *,
    capacity: int = 16,
    physical_page_bytes: int = 32,
    page_stride_bytes: int = 32,
    history_offset: int = 0,
    state_offset: int = 16,
) -> CacheTransferContract:
    return make_layout(
        make_group(
            "history",
            make_segment(
                "layer.0.kv",
                dtype="bfloat16",
                shape=(8,),
                offset=history_offset,
                stride=page_stride_bytes,
            ),
        ),
        make_group(
            "state",
            make_segment(
                "layer.1.state",
                dtype="bfloat16",
                shape=(8,),
                offset=state_offset,
                stride=page_stride_bytes,
            ),
            family="state",
        ),
        capacity=capacity,
        page_bytes=physical_page_bytes,
    )


def _typed_layout(
    *,
    local_heads: int,
    global_heads: int,
) -> CacheTransferContract:
    payload_bytes = 2 * local_heads * 2 * 2
    return make_layout(
        make_group(
            "history",
            make_segment(
                "layer.0.k",
                dtype="bfloat16",
                shape=(2, local_heads, 2),
                axis=1,
                extent=global_heads,
            ),
        ),
        capacity=5,
        page_bytes=payload_bytes,
    )


def _op(*, state_page: int = 6, spec_candidate_ids=((),)):
    tables = {
        "history": np.asarray([[1, 2, 3]], dtype=np.int32),
        "state": np.asarray([[4, 5, state_page]], dtype=np.int32),
    }
    # The remote-decode op is self-contained: the scheduler stamps the
    # bootstrap token and drafter candidates onto its rows.
    return make_operation(
        tables,
        request_ids=["request-0"],
        request_pool_indices=[7],
        extend_prefix_lens=[2],
        prefill_lengths=[5],
        num_extends=lambda: 1,
        decode_input_ids=[42],
        spec_candidate_ids=[list(ids) for ids in spec_candidate_ids],
    )


def _destination_block_manifest() -> CachePDBlockManifest:
    return make_block_manifest(
        ("history", (10, 11)), ("state", (12,)), prefix=2, prompt=5
    )


def _single_group_block_manifest(
    group_id: str, block_ids: tuple[int, ...]
) -> CachePDBlockManifest:
    return make_block_manifest((group_id, block_ids))


def _destination_transfer_frames() -> list[bytes]:
    return [
        b"9",
        b"session",
        _destination_block_manifest().to_wire_bytes(),
    ]


def _destination_transfer_info() -> TransferInfo:
    return TransferInfo.from_zmq(_destination_transfer_frames())


def _registration_frames(
    layout: CacheTransferContract,
    *,
    pointer: int = 0x1000,
    decode_tp_size: int = 1,
    decode_tp_rank: int = 0,
) -> list[bytes]:
    return [
        b"None",
        b"127.0.0.1",
        b"9000",
        b"session",
        np.asarray([pointer], dtype=np.uint64).tobytes(),
        layout.to_wire_bytes(),
        str(decode_tp_size).encode("ascii"),
        str(decode_tp_rank).encode("ascii"),
    ]


def _registration(
    layout: CacheTransferContract,
    *,
    rank: int = 0,
    decode_tp_size: int = 1,
    session: str | None = None,
    endpoint: str | None = None,
    pointer: int = 0x2000,
    expected_decode_ranks=(),
) -> KVArgsRegisterInfo:
    return KVArgsRegisterInfo(
        endpoint=endpoint or f"decode-{rank}",
        dst_port=9000 + rank,
        mooncake_session_id=session or f"session-{rank}",
        dst_kv_ptr=pointer,
        peer_cache_layout=layout,
        decode_tp_size=decode_tp_size,
        decode_tp_rank=rank,
        expected_decode_ranks=frozenset(expected_decode_ranks),
    )


def _transfer_cache(
    manager,
    session: str,
    dst_ptr: int,
    transfer_fragments: tuple[CacheTransferFragment, ...],
    *,
    src_block_manifest: CachePDBlockManifest,
    dst_block_manifest: CachePDBlockManifest,
    dst_cache_layout,
    packer=None,
) -> int:
    return manager._transfer_data(
        session,
        manager._cache_transfer_blocks(
            dst_ptr=dst_ptr,
            src_block_manifest=src_block_manifest,
            dst_block_manifest=dst_block_manifest,
            transfer_fragments=transfer_fragments,
            owner_filters={},
            dst_cache_layout=dst_cache_layout,
        ),
        packer,
    )


def _recording_transfer_manager(layout: CacheTransferContract, src_ptr: int):
    from tokenspeed.runtime.pd.mooncake.prefill import MooncakeKVManagerPrefill

    calls = []
    manager = object.__new__(MooncakeKVManagerPrefill)
    manager.kv_args = SimpleNamespace(cache_layout=layout, kv_data_ptr=src_ptr)

    # Whole-field descriptors arrive as int64 array columns, fragment rows as
    # lists; record both as plain lists so the geometry asserts read the same.
    # The fake engine records per-descriptor WRITEs as lists and expands the
    # page-gathered WRITE the way Mooncake does (field-major, batched), so the
    # geometry asserts read the same for both.
    def record_pages(session, src_pages, dst_pages, fields, *, max_batch_size):
        src, dst, lengths = zip(
            *_expand_page_fields(src_pages, dst_pages, fields), strict=True
        )
        for start in range(0, len(src), max_batch_size):
            stop = start + max_batch_size
            calls.append(
                (
                    session,
                    list(src[start:stop]),
                    list(dst[start:stop]),
                    list(lengths[start:stop]),
                )
            )
        return 0

    manager.engine = SimpleNamespace(
        batch_transfer_sync=lambda session, src, dst, lengths: (
            calls.append((session, list(src), list(dst), list(lengths))) or 0
        ),
        batch_transfer_sync_pages=record_pages,
    )
    return manager, calls


def _expand_page_fields(src_pages, dst_pages, fields) -> list[tuple[int, int, int]]:
    """Reference expansion of a PageFieldCopies grid, field-major."""
    return [
        (
            int(src_base + page * src_stride),
            int(dst_base + peer * dst_stride),
            int(length),
        )
        for src_base, src_stride, dst_base, dst_stride, length in fields.tolist()
        for page, peer in zip(src_pages.tolist(), dst_pages.tolist(), strict=True)
    ]


def _route_manager():
    from tokenspeed.runtime.pd.mooncake.prefill import MooncakeKVManagerPrefill

    manager = object.__new__(MooncakeKVManagerPrefill)
    layout = _typed_layout(local_heads=4, global_heads=4)
    manager.kv_args = SimpleNamespace(
        cache_layout=layout,
        kv_data_ptr=0x1000,
        # A single stage owns every field, as a non-PP prefill declares.
        cache_fields_by_stage=(tuple(field.field_id for field in layout.plan.fields),),
    )
    manager.topology = _topology()
    return manager


def _real_destinations(manager, block_manifest=None):
    destination_layout = _typed_layout(local_heads=2, global_heads=4)
    registrations = tuple(
        manager._prepare_decode_registration(
            _registration(destination_layout, rank=rank, decode_tp_size=2)
        )
        for rank in range(2)
    )
    manager.decode_kv_args_table = {
        registration.mooncake_session_id: registration for registration in registrations
    }
    block_manifest = block_manifest or _single_group_block_manifest("history", (2,))
    requests = tuple(
        TransferInfo(9, registration.mooncake_session_id, block_manifest)
        for registration in registrations
    )
    return destination_layout, registrations, requests


class _RecordingSender:
    bootstrap_room = 9

    def __init__(self, calls) -> None:
        self.calls = calls

    def send(self, *args, **kwargs) -> None:
        self.calls.append((args, kwargs))

    def layerwise_final_chunk_submitted(self) -> bool:
        return False


class _FinalLayerwiseSender(_RecordingSender):
    def layerwise_final_chunk_submitted(self) -> bool:
        return True


class _StopWorker(BaseException):
    pass


class _OneChunkQueue:
    def __init__(self, chunk) -> None:
        self.chunk = chunk

    def get(self):
        if self.chunk is None:
            raise _StopWorker
        chunk, self.chunk = self.chunk, None
        return chunk


def test_decode_receiver_sends_static_registration_then_three_frame_request() -> None:
    from tokenspeed.runtime.pd.base.status import TransferPoll
    from tokenspeed.runtime.pd.mooncake.receiver import MooncakeKVReceiver

    layout = _layout()
    messages = []
    statuses = []

    class _Socket:
        def send_multipart(self, frames):
            messages.append(frames)

    receiver = object.__new__(MooncakeKVReceiver)
    receiver.kv_mgr = SimpleNamespace(
        kv_args=SimpleNamespace(
            kv_data_ptr=0x1000,
            cache_layout=layout,
            engine_rank=0,
        ),
        topology=_topology(tp_size=2, tp_rank=1, global_rank=1),
        rank_port=9000,
        update_status=lambda room, status: statuses.append((room, status)),
    )
    room = (1 << 63) - 1
    receiver.bootstrap_room = room
    receiver.session_id = "session"
    receiver.bootstrap_infos = [
        {"rank_ip": "127.0.0.1", "rank_port": 9100, "is_dummy": False}
    ]
    receiver._connect = lambda _endpoint: (_Socket(), nullcontext())

    receiver._register_kv_args()
    receiver.prefill(block_manifest=_destination_block_manifest())

    assert len(messages[0]) == 8
    registration = KVArgsRegisterInfo.from_zmq(messages[0])
    assert registration.decode_tp_size == 2
    assert registration.decode_tp_rank == 1
    assert len(messages[1]) == 3
    transfer = TransferInfo.from_zmq(messages[1])
    assert transfer.room == room
    assert transfer.block_manifest == _destination_block_manifest()
    assert receiver.init_time is not None
    assert statuses == [(room, TransferPoll.WaitingForInput)]


def test_rejected_registration_fails_later_request_without_stopping_handler() -> None:
    from tokenspeed.runtime.pd.base.status import TransferPoll
    from tokenspeed.runtime.pd.mooncake.prefill import MooncakeKVManagerPrefill

    layout = _layout()
    registration_frames = _registration_frames(layout)
    manager = object.__new__(MooncakeKVManagerPrefill)
    manager.decode_kv_args_table = {}
    manager.rejected_decode_sessions = {}
    manager.transfer_infos = {}
    manager.session_lock = nullcontext()
    manager._prepare_decode_registration = lambda _registration: (_ for _ in ()).throw(
        ValueError("incompatible layout")
    )
    aborts = []
    notifications = []
    manager.abort_room = lambda room, reason: aborts.append((room, reason))
    manager.sync_status_to_decode_endpoint = (
        lambda endpoint, port, room, status, rank: (
            notifications.append((endpoint, port, room, status, rank))
        )
    )
    manager.topology = _topology(tp_size=2, tp_rank=1, global_rank=1)

    manager._handle_bootstrap_message(registration_frames)
    manager._handle_bootstrap_message(
        [b"9", b"session", _destination_block_manifest().to_wire_bytes()]
    )
    manager._handle_bootstrap_message([b"9"])

    assert manager.rejected_decode_sessions["session"][:2] == (
        "127.0.0.1",
        9000,
    )
    assert aborts and aborts[0][0] == 9
    assert notifications == [("127.0.0.1", 9000, 9, TransferPoll.Failed, 1)]


@pytest.mark.parametrize(
    "fail_during_commit",
    (False, True),
    ids=("already-failed", "commit-race"),
)
def test_failed_room_fanout_never_restores_state(
    fail_during_commit: bool,
) -> None:
    from tokenspeed.runtime.pd.base.status import TransferPoll
    from tokenspeed.runtime.pd.mooncake.prefill import MooncakeKVManagerPrefill

    layout = _layout()
    # A registration the table holds has had its route planned.
    registration = replace(
        _registration(
            layout,
            session="session",
            endpoint="127.0.0.1",
            pointer=0x1000,
            expected_decode_ranks=(0,),
        ),
        transfer_owner_filters={},
    )
    manager = object.__new__(MooncakeKVManagerPrefill)
    manager.decode_kv_args_table = {"session": registration}
    manager.rejected_decode_sessions = {}
    manager.transfer_infos = {}
    manager.request_status = {
        9: (TransferPoll.WaitingForInput if fail_during_commit else TransferPoll.Failed)
    }
    manager.topology = _topology()
    if fail_during_commit:
        manager._validate_cache_room_fanout = lambda _requests: (
            manager.request_status.__setitem__(9, TransferPoll.Failed)
        )
    notifications = []
    manager.sync_status_to_decode_endpoint = (
        lambda endpoint, port, room, status, rank: (
            notifications.append((endpoint, port, room, status, rank))
        )
    )

    manager._handle_bootstrap_message(
        [b"9", b"session", _destination_block_manifest().to_wire_bytes()]
    )

    assert manager.transfer_infos == {}
    assert manager.request_status[9] == TransferPoll.Failed
    assert notifications == [("127.0.0.1", 9000, 9, TransferPoll.Failed, 0)]


def test_cache_factory_exposes_only_typed_arena() -> None:
    import torch

    from tokenspeed.runtime.pd.factory import get_kv_args

    layout = _layout()
    buffer = torch.zeros(layout.plan.arena_bytes, dtype=torch.uint8)
    pool = SimpleNamespace(
        arena=SimpleNamespace(
            supports_disaggregation=True,
            plan=layout.plan,
            cache_group_specs=layout.group_specs,
            contract_binding=lambda: buffer,
        ),
    )

    kv_args = get_kv_args(
        0,
        0,
        "mlx5_0",
        pool,
        draft_model_config=None,
        cache_fields_by_stage=(tuple(field.field_id for field in layout.plan.fields),),
        producer_fields_by_step=tuple(
            (field.field_id,) for field in layout.plan.fields
        ),
        logical_plan=None,
        model_config=SimpleNamespace(
            num_attention_layers=2,
            num_key_value_heads=1,
            hf_config=SimpleNamespace(),
            hf_text_config=SimpleNamespace(),
        ),
    )

    assert kv_args.engine_rank == 0
    assert kv_args.kv_data_ptr == buffer.data_ptr()
    assert kv_args.ib_device == "mlx5_0"
    assert kv_args.gpu_id == 0
    assert kv_args.cache_layout == layout
    assert kv_args.cache_producer_schedule.fields_by_step == (
        ("layer.0.kv",),
        ("layer.1.state",),
    )


@pytest.mark.parametrize("remote_hits", [0, 1, 4])
def test_terminal_events_clear_transport_room_state(
    monkeypatch: pytest.MonkeyPatch,
    remote_hits: int,
) -> None:
    from tokenspeed.runtime.pd import decode_executor as decode_module
    from tokenspeed.runtime.pd.base.status import TransferPoll
    from tokenspeed.runtime.pd.mooncake.receiver import MooncakeKVReceiver

    monkeypatch.setattr(
        decode_module,
        "poll_and_all_reduce",
        lambda _values, _group: [TransferPoll.Success],
    )
    decode_manager = SimpleNamespace(
        request_status={9: TransferPoll.Success},
        failure_records={9: "stale"},
        failure_lock=nullcontext(),
        prefill_response_tracker={9: {0}},
        expected_prefill_ranks_table={9: frozenset({0})},
        bootstrap_token_table={9: 42},
        spec_candidate_ids_table={9: [1]},
        cached_tokens_table={9: remote_hits},
        bootstrap_logprob_table={9: -0.25},
        _pending_bootstrap_token_table={},
        _pending_spec_candidate_ids_table={},
        _pending_bootstrap_logprob_table={},
        connection_lock=nullcontext(),
        addr_to_rooms_tracker={"bootstrap": {9}},
    )
    from tokenspeed.runtime.pd.mooncake.decode import MooncakeKVManagerDecode

    decode_manager.pop_prefill_metadata = lambda room: (
        MooncakeKVManagerDecode.pop_prefill_metadata(decode_manager, room)
    )
    receiver = object.__new__(MooncakeKVReceiver)
    receiver.prefill = lambda *, block_manifest: None
    receiver.kv_mgr = decode_manager
    receiver.bootstrap_room = 9
    receiver.bootstrap_addr = "bootstrap"
    decode = object.__new__(decode_module.DisaggDecodeExecutor)
    decode.receivers = {"request": receiver}
    decode.gloo_group = None
    decode._local_states = {"request": TransferPoll.Bootstrapped}
    decode.kv_manager = decode_manager
    decode._admissions = {}
    decode._remote_cache_slots = {}
    decode._remote_cached_tokens = {}
    decode._remote_bootstrap_logprobs = {}
    decode._remote_spec_candidate_ids = {}
    decode.cache_layout = _layout()
    admission = _op()
    admission.request_ids = ["request"]
    decode._cache_prefill(admission)

    assert len(decode.generate_events()) == 1
    assert decode.pop_remote_cache_slot("request") == 7
    assert decode.pop_remote_cached_tokens("request") == max(2, remote_hits)
    assert decode.pop_remote_bootstrap_logprob("request") == -0.25
    assert decode.pop_remote_spec_candidate_ids("request") == (7, [1])
    assert decode._admissions == {}
    assert decode._remote_cache_slots == {}
    assert decode._remote_cached_tokens == {}
    assert decode._remote_bootstrap_logprobs == {}
    assert decode._remote_spec_candidate_ids == {}
    assert decode_manager.cached_tokens_table == {}
    assert decode_manager.bootstrap_logprob_table == {}
    assert decode.receivers == {}
    assert decode_manager.request_status == {}
    assert decode_manager.expected_prefill_ranks_table == {}
    assert decode_manager.addr_to_rooms_tracker == {"bootstrap": set()}


def test_terminal_cleanup_wakes_prefill_metadata_waiter() -> None:
    import threading

    from tokenspeed.runtime.pd.base.status import TransferPoll
    from tokenspeed.runtime.pd.mooncake.prefill import MooncakeKVManagerPrefill

    manager = object.__new__(MooncakeKVManagerPrefill)
    manager.bootstrap_token_cond = threading.Condition()
    manager.prefill_metadata = {}
    manager.cached_tokens = {}
    manager.bootstrap_logprobs = {}
    manager.transfer_infos = {9: {}}
    manager.request_status = {9: TransferPoll.WaitingForInput}
    result = []
    waiter = threading.Thread(
        target=lambda: result.append(manager._wait_prefill_metadata(9, -1, [1, 2]))
    )
    waiter.start()
    manager.discard_room(9)
    waiter.join(timeout=1)

    assert not waiter.is_alive()
    assert result == [(-1, [1, 2])]
    assert manager.transfer_infos == {}
    manager.set_prefill_metadata(9, 99, [3])
    assert manager.prefill_metadata == {}


def test_decode_publishes_manifest_through_contract_receiver() -> None:
    import tokenspeed.runtime.pd.decode_executor as decode_module

    calls = []
    receiver = SimpleNamespace(
        prefill=lambda *args, **kwargs: calls.append((args, kwargs))
    )
    executor = object.__new__(decode_module.DisaggDecodeExecutor)
    executor.cache_layout = _layout()
    executor.receivers = {"request-0": receiver}
    executor._admissions = {}

    executor._cache_prefill(_op())

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == ()
    assert kwargs["block_manifest"].groups[0].block_ids == (2, 3)
    assert executor._admissions == {"request-0": (7, 2)}


def test_prefill_submits_manifest_through_contract_sender() -> None:
    import tokenspeed.runtime.pd.prefill_executor as prefill_module

    layout = _layout()
    destination = _destination_transfer_info()
    calls = []

    executor = object.__new__(prefill_module.DisaggPrefillExecutor)
    executor._layerwise_enabled = False
    executor.cache_layout = layout
    executor.senders = {"request-0": _RecordingSender(calls)}
    executor.kv_manager = SimpleNamespace(
        transfer_infos={9: {destination.mooncake_session_id: destination}},
        get_decode_registration=lambda _destination: SimpleNamespace(
            peer_cache_layout=layout
        ),
    )
    executor._cache_decode(_op())

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == (True,)
    assert kwargs["bootstrap_token"] == 42
    assert kwargs["block_manifest"].prompt_len == 5


def test_idle_prefill_rank_submits_final_dummy_rendezvous() -> None:
    import tokenspeed.runtime.pd.prefill_executor as prefill_module

    calls = []

    executor = object.__new__(prefill_module.DisaggPrefillExecutor)
    executor._layerwise_enabled = False
    executor.senders = {"request-0": _RecordingSender(calls)}
    executor.kv_manager = SimpleNamespace(
        transfer_infos={9: {"dummy": SimpleNamespace(is_dummy=True)}}
    )
    executor._cache_decode(_op())

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == (True,)
    assert kwargs == {
        "bootstrap_token": 42,
        "spec_candidate_ids": None,
        "block_manifest": None,
    }


def test_idle_layerwise_prefill_rank_submits_final_dummy_rendezvous() -> None:
    import tokenspeed.runtime.pd.prefill_executor as prefill_module

    calls = []

    executor = object.__new__(prefill_module.DisaggPrefillExecutor)
    executor._layerwise_enabled = True
    executor.senders = {"request-0": _RecordingSender(calls)}
    executor.kv_manager = SimpleNamespace(
        transfer_infos={9: {"dummy": SimpleNamespace(is_dummy=True)}}
    )
    executor._cache_decode(_op(spec_candidate_ids=[[7, 8]]))

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == (True,)
    assert kwargs == {
        "bootstrap_token": 42,
        "spec_candidate_ids": [7, 8],
        "block_manifest": None,
    }


def test_layerwise_final_preserves_speculative_candidates() -> None:
    import tokenspeed.runtime.pd.prefill_executor as prefill_module

    metadata_calls = []
    executor = object.__new__(prefill_module.DisaggPrefillExecutor)
    executor._layerwise_enabled = True
    executor.senders = {"request-0": _FinalLayerwiseSender([])}
    executor.kv_manager = SimpleNamespace(
        set_prefill_metadata=lambda *args: metadata_calls.append(args)
    )
    executor._cache_decode(_op(spec_candidate_ids=[[7, 8]]))

    assert metadata_calls == [(9, 42, [7, 8])]


@pytest.mark.parametrize("pp_rank", [0, 1])
def test_every_pipeline_stage_reports_under_its_stage_major_rank(pp_rank) -> None:
    """On a prefill pipeline every stage publishes the same broadcast bootstrap
    payload (the executor path above is stage-agnostic); each manager reports
    it under the stage-major prefill rank Decode counts completions by."""
    from tokenspeed.runtime.pd.mooncake.prefill import MooncakeKVManagerPrefill

    manager = object.__new__(MooncakeKVManagerPrefill)
    manager.topology = PDParallelTopology(
        tp_size=2,
        tp_rank=1,
        dp_size=1,
        dp_rank=0,
        world_size=4,
        global_rank=pp_rank * 2 + 1,
        pp_size=2,
        pp_rank=pp_rank,
    )
    assert manager._status_prefill_rank == pp_rank * 2 + 1


def test_shared_manager_executes_strided_cache_tp_fragment() -> None:
    source_segment = make_segment(
        "layer.0.k",
        dtype="bfloat16",
        shape=(2, 2, 2),
        stride=32,
        axis=1,
        extent=4,
    )
    destination_segment = make_segment(
        "layer.0.k",
        dtype="bfloat16",
        shape=(2, 4, 2),
        stride=64,
        axis=1,
        extent=4,
    )

    def one_field_layout(field):
        return make_layout(make_group("history", field), capacity=5, page_bytes=64)

    source_layout = one_field_layout(source_segment)
    destination_layout = one_field_layout(destination_segment)
    source_manifest = _single_group_block_manifest("history", (1,))
    destination_manifest = _single_group_block_manifest("history", (2,))
    fragment = CacheTransferFragment(
        group_id="history",
        field_id="layer.0.k",
        src_byte_offset=0,
        dst_byte_offset=8,
        src_row_stride_bytes=8,
        dst_row_stride_bytes=16,
        bytes_per_row=8,
        rows_per_page=2,
    )
    manager, calls = _recording_transfer_manager(source_layout, 0x1000)

    assert (
        _transfer_cache(
            manager,
            "session",
            0x2000,
            (fragment,),
            src_block_manifest=source_manifest,
            dst_block_manifest=destination_manifest,
            dst_cache_layout=destination_layout,
        )
        == 0
    )
    assert calls == [
        (
            "session",
            [0x1020, 0x1028],
            [0x2088, 0x2098],
            [8, 8],
        )
    ]


def test_transfer_blocks_for_many_pages_match_the_per_page_geometry() -> None:
    # Two fields and hundreds of pages: the generator resolves each field's
    # geometry once and expands pages in bulk, so check it against the plain
    # per-page formula in both the whole-field and the fragment path.
    def two_field_layout(capacity: int):
        return make_layout(
            make_group(
                "history",
                make_segment("layer.0.k", dtype="bfloat16", shape=(2, 4, 2)),
                make_segment("layer.1.k", dtype="bfloat16", shape=(2, 4, 2)),
            ),
            capacity=capacity,
            page_bytes=64,
        )

    source_layout = two_field_layout(700)
    destination_layout = two_field_layout(900)
    pages = 300
    src_pages = tuple(range(1, 2 * pages, 2))
    dst_pages = tuple(range(899, 899 - 2 * pages, -2))
    source_manifest = _single_group_block_manifest("history", src_pages)
    destination_manifest = _single_group_block_manifest("history", dst_pages)

    def field_pages(layout, ptr, field_id, page_ids):
        segment = next(f for f in layout.plan.fields if f.field_id == field_id)
        return [
            ptr + layout.plan.field_page_byte_offset(field_id, page)
            for page in page_ids
        ], segment.payload_bytes

    manager, _ = _recording_transfer_manager(source_layout, 0x1000)
    fields = tuple(f.field_id for f in source_layout.fields_for_group("history"))
    expected = []
    for field_id in fields:
        src, size = field_pages(source_layout, 0x1000, field_id, src_pages)
        dst, _ = field_pages(destination_layout, 0x2000, field_id, dst_pages)
        expected.extend(zip(src, dst, [size] * pages, strict=True))
    blocks = list(
        manager._cache_transfer_blocks(
            dst_ptr=0x2000,
            src_block_manifest=source_manifest,
            dst_block_manifest=destination_manifest,
            owner_filters={},
            dst_cache_layout=destination_layout,
        )
    )
    # One pages x fields item for the group, no per-page Python objects.
    (item,) = blocks
    assert isinstance(item, PageFieldCopies)
    assert item.fields.shape == (2, 5) and len(item) == 2 * pages
    assert _expand_page_fields(item.src_pages, item.dst_pages, item.fields) == expected
    with pytest.raises(TypeError):
        list(flatten_transfer_blocks(blocks))

    fragment = CacheTransferFragment(
        group_id="history",
        field_id=fields[1],
        src_byte_offset=4,
        dst_byte_offset=8,
        src_row_stride_bytes=16,
        dst_row_stride_bytes=16,
        bytes_per_row=8,
        rows_per_page=2,
    )
    src, _ = field_pages(source_layout, 0x1000, fields[1], src_pages)
    dst, _ = field_pages(destination_layout, 0x2000, fields[1], dst_pages)
    fragment_blocks = list(
        manager._cache_transfer_blocks(
            dst_ptr=0x2000,
            src_block_manifest=source_manifest,
            dst_block_manifest=destination_manifest,
            transfer_fragments=(fragment,),
            owner_filters={},
            dst_cache_layout=destination_layout,
        )
    )
    assert fragment_blocks == [
        (s + 4 + row * 16, d + 8 + row * 16, 8)
        for s, d in zip(src, dst, strict=True)
        for row in range(2)
    ]


def test_transfer_data_writes_page_grids_between_descriptor_batches() -> None:
    from tokenspeed.runtime.pd.mooncake import prefill as prefill_module

    manager, calls = _recording_transfer_manager(
        _typed_layout(local_heads=4, global_heads=4), 0
    )
    page_calls = []
    manager.engine.batch_transfer_sync_pages = (
        lambda session, src, dst, fields, *, max_batch_size: (
            page_calls.append((session, src, dst, fields, max_batch_size)) or 0
        )
    )
    item = PageFieldCopies(
        np.asarray([1, 2, 3], dtype=np.int64),
        np.asarray([9, 8, 7], dtype=np.int64),
        np.asarray([[100, 10, 200, 20, 5], [300, 30, 400, 40, 6]], dtype=np.int64),
    )
    # Pending per-descriptor rows are flushed ahead of the page item, and
    # rows after it start a new batch; order on the wire is preserved.
    blocks = [(1, 2, 3), (4, 5, 6), item, (7, 8, 9)]
    assert manager._transfer_data("session", iter(blocks)) == 0
    assert [call[1] for call in calls] == [[1, 4], [7]]
    ((session, src, dst, fields, max_batch_size),) = page_calls
    assert session == "session" and src is item.src_pages and fields is item.fields
    assert max_batch_size == prefill_module._TRANSFER_DESCRIPTOR_BATCH_SIZE

    manager.engine.batch_transfer_sync_pages = lambda *args, **kwargs: -3
    assert manager._transfer_data("session", iter([item])) == -3


def test_page_field_copies_validates_its_grid() -> None:
    item = PageFieldCopies(
        np.asarray([1, 5], dtype=np.int64),
        np.asarray([9, 8], dtype=np.int64),
        np.asarray(
            [[1000, 64, 5000, 128, 16], [2000, 4096, 3000, 4096, 3000]], dtype=np.int64
        ),
    )
    assert len(item) == 4
    with pytest.raises(TypeError):
        PageFieldCopies(item.src_pages, item.dst_pages, item.fields[:, :4].copy())
    with pytest.raises(ValueError):
        PageFieldCopies(item.src_pages, item.dst_pages[:1], item.fields)
    # Neither the packer nor the SGE flattener may see one.
    with pytest.raises(TypeError):
        list(flatten_transfer_blocks([item]))


def test_engine_wrapper_requires_and_forwards_the_page_gathered_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys
    import types

    from tokenspeed.runtime.pd.base.mooncake_engine import MooncakeTransferEngine

    class FakeTransferEngine:
        def __init__(self) -> None:
            self.initialized = None

        def initialize(self, *args):
            self.initialized = args
            return 0

        def get_rpc_port(self):
            return 4321

    fake_module = types.ModuleType("mooncake.engine")
    fake_module.TransferEngine = FakeTransferEngine
    monkeypatch.setitem(sys.modules, "mooncake", types.ModuleType("mooncake"))
    monkeypatch.setitem(sys.modules, "mooncake.engine", fake_module)
    # An engine without the page-gathered WRITE is a wrong install.
    with pytest.raises(RuntimeError, match="batch_transfer_sync_write_pages"):
        MooncakeTransferEngine("10.0.0.1", gpu_id=0, ib_device=None)
    # With it, bring-up completes and the session id names this rank.
    FakeTransferEngine.batch_transfer_sync_write_pages = lambda self, *a: 0
    engine = MooncakeTransferEngine("10.0.0.1", gpu_id=0, ib_device="mlx5_0")
    assert engine.session_id == "10.0.0.1:4321"
    assert engine.engine.initialized == ("10.0.0.1", "P2PHANDSHAKE", "rdma", "mlx5_0")

    src = np.asarray([1, 2], dtype=np.int64)
    dst = np.asarray([3, 4], dtype=np.int64)
    fields = np.zeros((2, 5), dtype=np.int64)
    seen = []
    wrapper = object.__new__(MooncakeTransferEngine)
    wrapper.engine = SimpleNamespace(
        batch_transfer_sync_write_pages=lambda session, s, d, f, batch: seen.append(
            ("pages", s, d, f, batch)
        )
        or 0
    )
    assert (
        wrapper.batch_transfer_sync_pages("s", src, dst, fields, max_batch_size=4096)
        == 0
    )
    assert seen == [("pages", src, dst, fields, 4096)]
    # A raising engine reports failure instead of propagating.
    wrapper.engine = SimpleNamespace(
        batch_transfer_sync_write_pages=lambda *args: (_ for _ in ()).throw(
            RuntimeError("x")
        )
    )
    assert (
        wrapper.batch_transfer_sync_pages("s", src, dst, fields, max_batch_size=1) == -1
    )


class _FakePackScratch:
    def __init__(self, base: int = 0xB000) -> None:
        self.base = base

    def materialize(self, blocks):
        from tokenspeed.runtime.pd.mooncake.pack import PackedCopy

        sges = []
        offset = 0
        for block in blocks:
            if isinstance(block, PackedCopy):
                sges.append((self.base + offset, block.dst, block.nbytes))
                offset += block.nbytes
            else:
                src, dst, length = block
                sges.append((int(src), int(dst), int(length)))
        return sges


def test_dest_contiguous_strided_src_packs_into_one_sge() -> None:
    source_segment = make_segment(
        "layer.0.k",
        dtype="bfloat16",
        shape=(2, 2, 2),
        stride=32,
        axis=1,
        extent=4,
    )
    destination_segment = make_segment(
        "layer.0.k",
        dtype="bfloat16",
        shape=(2, 1, 2),
        stride=16,
        axis=1,
        extent=4,
    )

    def one_field_layout(field):
        return make_layout(make_group("history", field), capacity=5, page_bytes=64)

    source_layout = one_field_layout(source_segment)
    destination_layout = one_field_layout(destination_segment)
    source_manifest = _single_group_block_manifest("history", (1,))
    destination_manifest = _single_group_block_manifest("history", (2,))
    fragment = CacheTransferFragment(
        group_id="history",
        field_id="layer.0.k",
        src_byte_offset=0,
        dst_byte_offset=0,
        src_row_stride_bytes=16,
        dst_row_stride_bytes=8,
        bytes_per_row=8,
        rows_per_page=2,
    )
    manager, calls = _recording_transfer_manager(source_layout, 0x1000)

    assert (
        _transfer_cache(
            manager,
            "session",
            0x2000,
            (fragment,),
            src_block_manifest=source_manifest,
            dst_block_manifest=destination_manifest,
            dst_cache_layout=destination_layout,
            packer=_FakePackScratch(),
        )
        == 0
    )
    assert len(calls) == 1
    session, src, dst, lengths = calls[0]
    assert session == "session"
    assert lengths == [16]
    assert src == [0xB000]
    assert len(dst) == 1


def test_flatten_packed_copy_emits_contiguous_dest_rows() -> None:
    from tokenspeed.runtime.pd.mooncake.pack import PackedCopy, flatten_transfer_blocks

    copy = PackedCopy(src=0x10, dst=0x20, width=8, src_pitch=16, rows=2)
    flattened = flatten_transfer_blocks([copy, (1, 2, 3)])
    assert not isinstance(flattened, list)
    assert list(flattened) == [
        (0x10, 0x20, 8),
        (0x20, 0x28, 8),
        (1, 2, 3),
    ]


def test_pack_scratch_device_uses_explicit_gpu_index() -> None:
    import torch

    from tokenspeed.runtime.pd.mooncake.pack import PrefillPackScratch, scratch_device

    assert scratch_device(3) == torch.device("cuda", 3)
    assert scratch_device(None).type == "cuda"
    engine = SimpleNamespace(
        register=lambda *_a, **_k: None, deregister=lambda *_a: None
    )
    assert PrefillPackScratch(engine, gpu_id=3)._gpu_id == 3


def test_pack_scratch_keeps_old_buffer_when_growth_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    from tokenspeed.runtime.pd.mooncake.pack import PrefillPackScratch

    events: list[tuple] = []

    def register(ptr, nbytes):
        events.append(("register", int(ptr), int(nbytes)))

    def deregister(ptr):
        events.append(("deregister", int(ptr)))

    class _Tensor:
        def __init__(self, ptr: int) -> None:
            self._ptr = ptr

        def data_ptr(self) -> int:
            return self._ptr

    scratch = PrefillPackScratch(
        SimpleNamespace(register=register, deregister=deregister), gpu_id=0
    )
    monkeypatch.setattr(
        torch.cuda, "Stream", lambda device=None: SimpleNamespace(cuda_stream=1)
    )

    def fake_empty(nbytes, dtype=None, device=None):
        if nbytes > 64:
            raise RuntimeError("OOM")
        return _Tensor(0x1000)

    monkeypatch.setattr(torch, "empty", fake_empty)
    scratch._ensure(64)
    with pytest.raises(RuntimeError, match="OOM"):
        scratch._ensure(256)
    assert scratch._ptr == 0x1000
    assert scratch._nbytes == 64
    assert events == [("register", 0x1000, 64)]


def test_pack_cuda_skips_non_nvidia(monkeypatch: pytest.MonkeyPatch) -> None:
    from tokenspeed.runtime.pd.mooncake.pack import PackedCopy, PrefillPackScratch

    monkeypatch.setattr(
        "tokenspeed_kernel.platform.current_platform",
        lambda: SimpleNamespace(is_nvidia=False),
    )
    copy = PackedCopy(src=0x10, dst=0x20, width=8, src_pitch=16, rows=2)
    engine = SimpleNamespace(
        register=lambda *_a, **_k: None, deregister=lambda *_a: None
    )
    assert list(PrefillPackScratch(engine).materialize([copy])) == [
        (0x10, 0x20, 8),
        (0x20, 0x28, 8),
    ]


def test_pack_scratch_keeps_old_buffer_when_register_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    from tokenspeed.runtime.pd.mooncake.pack import PrefillPackScratch

    events: list[tuple] = []

    def register(ptr, nbytes):
        events.append(("register", int(ptr), int(nbytes)))
        return 0 if nbytes <= 64 else -1

    def deregister(ptr):
        events.append(("deregister", int(ptr)))

    class _Tensor:
        def __init__(self, ptr: int) -> None:
            self._ptr = ptr

        def data_ptr(self) -> int:
            return self._ptr

    scratch = PrefillPackScratch(
        SimpleNamespace(register=register, deregister=deregister), gpu_id=0
    )
    monkeypatch.setattr(
        torch.cuda, "Stream", lambda device=None: SimpleNamespace(cuda_stream=1)
    )

    def fake_empty(nbytes, dtype=None, device=None):
        return _Tensor(0x1000 if nbytes <= 64 else 0x2000)

    monkeypatch.setattr(torch, "empty", fake_empty)
    scratch._ensure(64)
    with pytest.raises(RuntimeError, match="registration failed"):
        scratch._ensure(256)
    assert scratch._ptr == 0x1000
    assert scratch._nbytes == 64
    assert events == [
        ("register", 0x1000, 64),
        ("register", 0x2000, 256),
    ]


def test_materialize_falls_back_when_register_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    from tokenspeed.runtime.pd.mooncake.pack import PackedCopy, PrefillPackScratch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        "tokenspeed_kernel.platform.current_platform",
        lambda: SimpleNamespace(is_nvidia=True),
    )
    monkeypatch.setattr(
        torch,
        "empty",
        lambda *a, **k: SimpleNamespace(data_ptr=lambda: 0x1000),
    )
    events: list[str] = []
    copy = PackedCopy(src=0x10, dst=0x20, width=8, src_pitch=16, rows=2)
    engine = SimpleNamespace(
        register=lambda *_a, **_k: -1,
        deregister=lambda *_a: events.append("deregister"),
    )
    scratch = PrefillPackScratch(engine, gpu_id=0)
    assert list(scratch.materialize([copy])) == [
        (0x10, 0x20, 8),
        (0x20, 0x28, 8),
    ]
    assert scratch._ptr == 0
    assert events == []


def test_fallback_write_batches_expanded_row_sges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tokenspeed.runtime.pd.mooncake.pack import PackedCopy, PrefillPackScratch
    from tokenspeed.runtime.pd.mooncake.prefill import (
        _TRANSFER_DESCRIPTOR_BATCH_SIZE,
        MooncakeKVManagerPrefill,
    )

    monkeypatch.setattr(
        "tokenspeed_kernel.platform.current_platform",
        lambda: SimpleNamespace(is_nvidia=False),
    )
    batch_sizes: list[int] = []
    manager = object.__new__(MooncakeKVManagerPrefill)
    manager.engine = SimpleNamespace(
        batch_transfer_sync=lambda _s, src, _d, _l: batch_sizes.append(len(src)) or 0
    )
    packer = PrefillPackScratch(
        SimpleNamespace(register=lambda *_a, **_k: None, deregister=lambda *_a: None)
    )
    copies = [
        PackedCopy(src=index * 16, dst=index * 8, width=8, src_pitch=16, rows=2)
        for index in range(3000)
    ]
    assert manager._transfer_data("session", copies, packer) == 0
    assert batch_sizes == [4096, 1904]
    assert sum(batch_sizes) == 6000
    assert all(size <= _TRANSFER_DESCRIPTOR_BATCH_SIZE for size in batch_sizes)


def test_transfer_worker_completes_real_heterogeneous_fanout_before_status() -> None:
    from tokenspeed.runtime.pd.base.status import TransferPoll

    source_manifest = _single_group_block_manifest("history", (1,))
    chunk = TransferKVChunk(
        room=9,
        is_last=True,
        bootstrap_token=42,
        block_manifest=source_manifest,
    )

    manager = _route_manager()
    _, _, destinations = _real_destinations(manager)
    requests = {request.mooncake_session_id: request for request in destinations}
    sends = []
    statuses = {9: TransferPoll.WaitingForInput}
    notifications = []
    manager.transfer_infos = {9: requests}
    manager.session_lock = nullcontext()
    manager._is_session_failed = lambda _session: False
    manager._transfer_data = lambda session, blocks, packer=None: (
        sends.append((session, tuple(blocks))) or 0
    )
    manager.update_status = lambda room, status: statuses.__setitem__(room, status)
    manager.check_status = lambda room: statuses[room]
    manager.request_status = statuses
    manager.sync_status_to_decode_endpoint = (
        lambda endpoint, port, room, status, rank, **kwargs: notifications.append(
            (endpoint, port, room, status, rank, kwargs)
        )
    )
    manager.kv_transfer_metrics = None
    manager.topology = _topology()

    with pytest.raises(_StopWorker):
        manager.transfer_worker(_OneChunkQueue(chunk), None)

    assert [session for session, _blocks in sends] == ["session-0", "session-1"]
    assert all(blocks for _session, blocks in sends)
    assert statuses[9] == TransferPoll.Success
    assert [(endpoint, port) for endpoint, port, *_ in notifications] == [
        ("decode-0", 9000),
        ("decode-1", 9001),
    ]
    assert all(item[3] == TransferPoll.Success for item in notifications)
    assert all(item[5]["bootstrap_token"] == 42 for item in notifications)
    assert manager.transfer_infos == {}


def test_shared_manager_lazily_bounds_application_descriptor_batches() -> None:
    from tokenspeed.runtime.pd.mooncake.prefill import (
        _TRANSFER_DESCRIPTOR_BATCH_SIZE,
        MooncakeKVManagerPrefill,
    )

    batch_sizes = []
    pulled_at_first_write = []
    pulled = []
    manager = object.__new__(MooncakeKVManagerPrefill)

    def record_write(_session, src, dst, lengths):
        if not pulled_at_first_write:
            pulled_at_first_write.append(len(pulled))
        batch_sizes.append((len(src), len(dst), len(lengths)))
        return 0

    manager.engine = SimpleNamespace(batch_transfer_sync=record_write)
    block_count = 2 * _TRANSFER_DESCRIPTOR_BATCH_SIZE + 17

    def blocks():
        for index in range(block_count):
            pulled.append(index)
            yield (0x1000 + index * 64, 0x2000 + index * 64, 64)

    assert manager._transfer_data("decode-session", blocks()) == 0
    assert pulled_at_first_write == [_TRANSFER_DESCRIPTOR_BATCH_SIZE]
    assert batch_sizes == [
        (_TRANSFER_DESCRIPTOR_BATCH_SIZE,) * 3,
        (_TRANSFER_DESCRIPTOR_BATCH_SIZE,) * 3,
        (17, 17, 17),
    ]


def test_shared_manager_uses_destination_page_zero_offsets() -> None:
    source_layout = _layout(capacity=8)
    destination_layout = _layout(
        capacity=8,
        physical_page_bytes=64,
        page_stride_bytes=64,
        history_offset=8,
        state_offset=40,
    )
    source_manifest = make_block_manifest(
        ("history", (1, 2)), ("state", (4,)), prompt=4
    )
    destination_manifest = make_block_manifest(
        ("history", (5, 6)), ("state", (3,)), prompt=4
    )
    manager, calls = _recording_transfer_manager(source_layout, 0x10000)

    assert (
        _transfer_cache(
            manager,
            "session",
            0x20000,
            (),
            src_block_manifest=source_manifest,
            dst_block_manifest=destination_manifest,
            dst_cache_layout=destination_layout,
        )
        == 0
    )
    # One page-gathered WRITE per cache group.
    assert calls == [
        (
            "session",
            [0x10000 + 1 * 32, 0x10000 + 2 * 32],
            [0x20000 + 8 + 5 * 64, 0x20000 + 8 + 6 * 64],
            [16, 16],
        ),
        ("session", [0x10000 + 16 + 4 * 32], [0x20000 + 40 + 3 * 64], [16]),
    ]


def test_cache_heterogeneous_gqa_route_rendezvous_idle_prefill_ranks() -> None:
    from tokenspeed.runtime.pd.mooncake.decode import PrefillParallelInfo
    from tokenspeed.runtime.pd.mooncake.receiver import _calc

    decode_layout = _typed_layout(local_heads=2, global_heads=2)
    prefill_layout = _typed_layout(local_heads=1, global_heads=2)
    manager = SimpleNamespace(
        topology=_topology(),
        kv_args=SimpleNamespace(
            engine_rank=0,
            cache_layout=decode_layout,
        ),
    )
    prefill = PrefillParallelInfo(
        tp_size=4,
        dp_size=1,
        cache_fields_by_stage=(
            tuple(field.field_id for field in prefill_layout.plan.fields),
        ),
        cache_layout=prefill_layout,
    )

    route = _calc(manager, prefill)

    assert route.target_tp_ranks == (0, 1, 2, 3)
    assert route.dummy_tp_ranks == (1, 3)


def test_decode_accepts_only_the_planned_prefill_rank_completion_set() -> None:
    from collections import defaultdict

    from tokenspeed.runtime.pd.base.status import TransferPoll
    from tokenspeed.runtime.pd.mooncake.decode import MooncakeKVManagerDecode

    def manager():
        value = object.__new__(MooncakeKVManagerDecode)
        value.request_status = {9: TransferPoll.WaitingForInput}
        value.expected_prefill_ranks_table = {9: frozenset((0, 2))}
        value.prefill_response_tracker = defaultdict(set)
        value.bootstrap_token_table = {}
        value.spec_candidate_ids_table = {}
        value.cached_tokens_table = {}
        value.bootstrap_logprob_table = {}
        value._pending_bootstrap_token_table = {}
        value._pending_spec_candidate_ids_table = {}
        value._pending_bootstrap_logprob_table = {}
        value.failure_records = {}
        value.record_failure = lambda room, reason: value.failure_records.__setitem__(
            room, reason
        )
        return value

    complete = manager()
    complete._handle_prefill_status(9, TransferPoll.Success, 0, 42, None, 1280, -0.5)
    assert complete.request_status[9] == TransferPoll.WaitingForInput
    complete._handle_prefill_status(9, TransferPoll.Success, 0, 42, None, 1280, -0.5)
    assert complete.cached_tokens_table[9] == 1280
    # The rank completing last carries no bootstrap metadata; the first valid
    # token and logprob seen are kept.
    complete._handle_prefill_status(9, TransferPoll.Success, 2, -1, None, 1280, None)
    assert complete.request_status[9] == TransferPoll.Success
    assert complete.bootstrap_token_table[9] == 42
    assert complete.pop_prefill_metadata(9) == (42, None, 1280, -0.5)
    assert complete.cached_tokens_table == {}
    assert complete.bootstrap_logprob_table == {}

    wrong_rank = manager()
    wrong_rank._handle_prefill_status(9, TransferPoll.Success, 0, -1, None, 1280, None)
    wrong_rank._handle_prefill_status(9, TransferPoll.Success, 1, -1, None, 1280, None)
    assert wrong_rank.request_status[9] == TransferPoll.Failed
    assert wrong_rank.prefill_response_tracker[9] == {0}
    assert "unexpected Prefill TP rank" in wrong_rank.failure_records[9]


@pytest.mark.parametrize("failure_point", ("parallel_info", "bootstrap_info"))
def test_receiver_bootstrap_failure_is_not_overwritten(
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    from tokenspeed.runtime.pd.base.status import TransferPoll
    from tokenspeed.runtime.pd.mooncake import receiver as receiver_module

    statuses = []
    failures = []
    manager = SimpleNamespace(
        get_session_id=lambda: "decode-session",
        update_status=lambda room, status: statuses.append((room, status)),
        record_failure=lambda room, reason: failures.append((room, reason)),
        expected_prefill_ranks_table={},
        connection_pool={},
        kv_args=SimpleNamespace(engine_rank=0),
    )
    route = SimpleNamespace(
        transfer_plan=SimpleNamespace(
            target_prefill_ranks=(0,),
        ),
        target_tp_ranks=(0,),
        is_dummy_tp_rank=lambda _rank: False,
    )
    if failure_point == "parallel_info":
        monkeypatch.setattr(
            receiver_module.MooncakeKVReceiver,
            "_get_prefill_parallel_info",
            lambda _self: None,
        )
        monkeypatch.setattr(
            receiver_module,
            "_calc",
            lambda *_args: pytest.fail("route planning must not run after failure"),
        )
    else:
        monkeypatch.setattr(
            receiver_module.MooncakeKVReceiver,
            "_get_prefill_parallel_info",
            lambda _self: SimpleNamespace(dp_size=1),
        )
        monkeypatch.setattr(receiver_module, "_calc", lambda *_args: route)
        monkeypatch.setattr(
            receiver_module.MooncakeKVReceiver,
            "_get_bootstrap_infos",
            lambda *_args: None,
        )

    receiver_module.MooncakeKVReceiver(manager, "127.0.0.1:8998", 9)

    assert statuses == [
        (9, TransferPoll.Bootstrapping),
        (9, TransferPoll.Failed),
    ]
    assert failures and failures[0][0] == 9


def test_prefill_usage_status_wire_roundtrip():
    import threading

    from tokenspeed.runtime.pd.base.status import TransferPoll
    from tokenspeed.runtime.pd.mooncake.decode import parse_prefill_status_message
    from tokenspeed.runtime.pd.mooncake.prefill import MooncakeKVManagerPrefill

    manager = object.__new__(MooncakeKVManagerPrefill)
    manager.bootstrap_token_cond = threading.Condition()
    manager.request_status = {9: TransferPoll.Bootstrapped}
    manager.prefill_metadata = {}
    manager.cached_tokens = {}
    manager.bootstrap_logprobs = {}
    messages = []
    manager._connect = lambda endpoint: (
        SimpleNamespace(send_multipart=messages.append),
        nullcontext(),
    )
    manager.record_cached_tokens(9, 1280)
    manager.record_bootstrap_logprob(9, -0.123456789012345678)
    manager.sync_status_to_decode_endpoint(
        "127.0.0.1",
        1234,
        9,
        TransferPoll.Success,
        0,
        bootstrap_token=42,
        spec_candidate_ids=[5, 6],
    )
    parsed = parse_prefill_status_message(messages[0])
    # The logprob frame is an IEEE double: exact, not a decimal rendering.
    assert parsed == (
        9,
        TransferPoll.Success,
        0,
        42,
        [5, 6],
        1280,
        -0.123456789012345678,
    )
    # Older senders lack the optional trailing frames: no logprob, then no usage.
    assert parse_prefill_status_message(messages[0][:-1])[-2:] == (1280, None)
    assert parse_prefill_status_message(messages[0][:-2])[-2:] == (0, None)
    manager.begin_room(9)
    assert manager.prefill_metadata == {}
    assert manager.cached_tokens == {}
    assert manager.bootstrap_logprobs == {}

    # A request without logprobs ships an empty logprob frame.
    manager.request_status = {9: TransferPoll.Bootstrapped}
    manager.sync_status_to_decode_endpoint(
        "127.0.0.1", 1234, 9, TransferPoll.Success, 0, bootstrap_token=42
    )
    assert messages[1][-1] == b""
    assert parse_prefill_status_message(messages[1])[-1] is None


def test_usage_alone_does_not_release_layerwise_bootstrap_waiter():
    import threading

    from tokenspeed.runtime.pd.base.status import TransferPoll
    from tokenspeed.runtime.pd.mooncake.prefill import MooncakeKVManagerPrefill

    manager = object.__new__(MooncakeKVManagerPrefill)
    manager.bootstrap_token_cond = threading.Condition()
    manager.request_status = {9: TransferPoll.WaitingForInput}
    manager.prefill_metadata = {}
    manager.cached_tokens = {}
    manager.bootstrap_logprobs = {}
    manager.record_cached_tokens(9, 1280)
    manager.record_bootstrap_logprob(9, -1.5)
    done = threading.Event()
    result = []

    def wait():
        result.append(manager._wait_prefill_metadata(9, -1, None))
        done.set()

    waiter = threading.Thread(target=wait)
    waiter.start()
    try:
        assert not done.wait(0.05)
    finally:
        manager.set_prefill_metadata(9, 42, [5, 6])
        waiter.join(timeout=1)
    assert not waiter.is_alive()
    assert result == [(42, [5, 6])]
    assert manager.prefill_metadata[9] == (42, [5, 6])
    assert manager.cached_tokens[9] == 1280
    assert manager.bootstrap_logprobs[9] == -1.5


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


# ---- KVP page-sharded prefill: owner filtering in the sender ----


def _sharded_history_layout(shard_count: int) -> CacheTransferContract:
    return make_layout(
        make_group(
            "history",
            make_segment("layer.0.kv", dtype="bfloat16", shape=(8,), stride=32),
            shard_count=shard_count,
        ),
        capacity=16,
        page_bytes=32,
    )


def test_transfer_blocks_keep_only_owned_pages_translated_to_local() -> None:
    from tokenspeed.runtime.pd.transfer_plan import CachePageOwnerFilter

    source_layout = _sharded_history_layout(2)
    destination_layout = _sharded_history_layout(1)
    # Virtual IDs 1..6 are dealt to two owners; rank 1 holds only the blocks
    # whose (v - 1) % 2 == 1, that is 2, 4, 6, in its local pages 1, 2, 3.
    # The destination holds every block, so its entries stay at the same
    # manifest positions.
    source_manifest = _single_group_block_manifest("history", (1, 2, 3, 4, 5, 6))
    destination_manifest = _single_group_block_manifest(
        "history", (10, 11, 12, 13, 14, 15)
    )
    owner_filters = {"history": CachePageOwnerFilter(1, 2)}
    manager, _ = _recording_transfer_manager(source_layout, 0x1000)

    (item,) = manager._cache_transfer_blocks(
        dst_ptr=0x2000,
        src_block_manifest=source_manifest,
        dst_block_manifest=destination_manifest,
        owner_filters=owner_filters,
        dst_cache_layout=destination_layout,
    )
    assert isinstance(item, PageFieldCopies)
    assert item.src_pages.tolist() == [1, 2, 3]
    assert item.dst_pages.tolist() == [11, 13, 15]

    # The fragment route (what the planner emits for a sharded source) lands
    # on the same page-gathered WRITE: a whole-field fragment is one row of
    # the group's pages x fields grid, not a descriptor per page.
    fragment = CacheTransferFragment(
        group_id="history",
        field_id="layer.0.kv",
        src_byte_offset=0,
        dst_byte_offset=0,
        src_row_stride_bytes=16,
        dst_row_stride_bytes=16,
        bytes_per_row=16,
        rows_per_page=1,
    )
    (item,) = manager._cache_transfer_blocks(
        dst_ptr=0x2000,
        src_block_manifest=source_manifest,
        dst_block_manifest=destination_manifest,
        transfer_fragments=(fragment,),
        owner_filters=owner_filters,
        dst_cache_layout=destination_layout,
    )
    assert isinstance(item, PageFieldCopies)
    assert _expand_page_fields(item.src_pages, item.dst_pages, item.fields) == [
        (0x1000 + local * 32, 0x2000 + remote * 32, 16)
        for local, remote in ((1, 11), (2, 13), (3, 15))
    ]

    # The other owner sends the complementary subsequence.
    (item,) = manager._cache_transfer_blocks(
        dst_ptr=0x2000,
        src_block_manifest=source_manifest,
        dst_block_manifest=destination_manifest,
        owner_filters={"history": CachePageOwnerFilter(0, 2)},
        dst_cache_layout=destination_layout,
    )
    assert item.src_pages.tolist() == [1, 2, 3]
    assert item.dst_pages.tolist() == [10, 12, 14]


def test_transfer_blocks_apply_owner_filter_to_layerwise_selection() -> None:
    from tokenspeed.runtime.pd.cache_protocol import (
        CachePDLayerwiseBlockSelection,
        CachePDLayerwiseGroupSelection,
    )
    from tokenspeed.runtime.pd.transfer_plan import CachePageOwnerFilter

    manager, _ = _recording_transfer_manager(_sharded_history_layout(2), 0x1000)
    destination_manifest = _single_group_block_manifest("history", (10, 11, 12, 13))
    selection = CachePDLayerwiseBlockSelection(
        groups=(CachePDLayerwiseGroupSelection((3, 4), (2, 3)),),
    )

    def blocks(owner_rank):
        return list(
            manager._cache_transfer_blocks(
                dst_ptr=0x2000,
                src_block_manifest=None,
                dst_block_manifest=destination_manifest,
                owner_filters={"history": CachePageOwnerFilter(owner_rank, 2)},
                dst_cache_layout=_sharded_history_layout(1),
                block_selection=selection,
            )
        )

    (item,) = blocks(0)
    assert item.src_pages.tolist() == [2] and item.dst_pages.tolist() == [12]
    (item,) = blocks(1)
    assert item.src_pages.tolist() == [2] and item.dst_pages.tolist() == [13]


def test_transfer_blocks_send_nothing_for_a_sharded_group_decided_none() -> None:
    """A None decision skips the group; the sender never infers from absence."""
    manager, _ = _recording_transfer_manager(_sharded_history_layout(2), 0x1000)
    manifest = _single_group_block_manifest("history", (1, 2))

    assert (
        list(
            manager._cache_transfer_blocks(
                dst_ptr=0x2000,
                src_block_manifest=manifest,
                dst_block_manifest=manifest,
                owner_filters={"history": None},
                dst_cache_layout=_sharded_history_layout(1),
            )
        )
        == []
    )
    with pytest.raises(KeyError):
        list(
            manager._cache_transfer_blocks(
                dst_ptr=0x2000,
                src_block_manifest=manifest,
                dst_block_manifest=manifest,
                owner_filters={},
                dst_cache_layout=_sharded_history_layout(1),
            )
        )


def test_transfer_blocks_reject_an_out_of_range_virtual_block() -> None:
    from tokenspeed.runtime.pd.transfer_plan import CachePageOwnerFilter

    manager, _ = _recording_transfer_manager(_sharded_history_layout(2), 0x1000)
    # A virtual ID past the group's virtual count never reaches the wire.
    with pytest.raises(IndexError):
        list(
            manager._cache_transfer_blocks(
                dst_ptr=0x2000,
                src_block_manifest=_single_group_block_manifest("history", (31,)),
                dst_block_manifest=_single_group_block_manifest("history", (1,)),
                owner_filters={"history": CachePageOwnerFilter(0, 2)},
                dst_cache_layout=_sharded_history_layout(1),
            )
        )


def test_registration_validates_owner_filter_decisions_once() -> None:
    from tokenspeed.runtime.pd.transfer_plan import (
        CachePageOwnerFilter,
        validate_rank_owner_filters,
    )

    group_specs = make_layout(
        make_group(
            "history",
            make_segment("layer.0.kv", dtype="bfloat16", shape=(8,), stride=32),
            shard_count=2,
        ),
        make_group(
            "state",
            make_segment("layer.1.state", dtype="bfloat16", shape=(8,), stride=32),
        ),
        capacity=16,
        page_bytes=64,
    ).group_specs
    history = CacheTransferFragment(
        group_id="history",
        field_id="layer.0.kv",
        src_byte_offset=0,
        dst_byte_offset=0,
        src_row_stride_bytes=16,
        dst_row_stride_bytes=16,
        bytes_per_row=16,
        rows_per_page=1,
    )
    state = replace(history, group_id="state", field_id="layer.1.state")

    def check(fragments, owner_filters):
        validate_rank_owner_filters(
            group_specs=group_specs, fragments=fragments, owner_filters=owner_filters
        )

    check((history, state), {"history": CachePageOwnerFilter(1, 2)})
    check((state,), {"history": None})
    with pytest.raises(ValueError, match="no owner-filter decision"):
        check((history,), {})
    with pytest.raises(ValueError, match="disagrees with its shard count"):
        check((history,), {"history": CachePageOwnerFilter(0, 4)})
    with pytest.raises(ValueError, match="but no owner filter"):
        check((history,), {"history": None})
    with pytest.raises(ValueError, match="carries no fragment of it"):
        check((state,), {"history": CachePageOwnerFilter(0, 2)})
    with pytest.raises(ValueError, match="which is not sharded"):
        check((state,), {"history": None, "state": CachePageOwnerFilter(0, 2)})


def test_every_sharded_prefill_rank_serves_every_decode_rank() -> None:
    from tokenspeed.runtime.pd.mooncake.prefill import MooncakeKVManagerPrefill
    from tokenspeed.runtime.pd.transfer_plan import CachePageOwnerFilter

    source_layout = _sharded_history_layout(4)
    destination_layout = _sharded_history_layout(1)
    registrations = []
    for tp_rank in range(4):
        manager = object.__new__(MooncakeKVManagerPrefill)
        manager.kv_args = SimpleNamespace(
            cache_layout=source_layout,
            kv_data_ptr=0x1000,
            cache_fields_by_stage=(("layer.0.kv",),),
        )
        manager.topology = _topology(tp_size=4, tp_rank=tp_rank)
        registration = manager._prepare_decode_registration(
            _registration(destination_layout, rank=0, decode_tp_size=1)
        )
        registrations.append(registration)
        assert not registration.is_dummy
        assert registration.expected_decode_ranks == frozenset({0})
        assert registration.transfer_owner_filters == {
            "history": CachePageOwnerFilter(tp_rank, 4)
        }
        assert [f.field_id for f in registration.transfer_fragments] == ["layer.0.kv"]
        manager.decode_kv_args_table = {registration.mooncake_session_id: registration}
        # One decode rank completes this rank's fan-out.
        manager._validate_cache_room_fanout(
            (
                TransferInfo(
                    9,
                    registration.mooncake_session_id,
                    _single_group_block_manifest("history", (2,)),
                ),
            )
        )
