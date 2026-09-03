from __future__ import annotations

import os
import sys
from contextlib import nullcontext
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
        cp_size=1,
        cp_rank=0,
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
            dst_cache_layout=dst_cache_layout,
        ),
        packer,
    )


def _recording_transfer_manager(layout: CacheTransferContract, src_ptr: int):
    from tokenspeed.runtime.pd.mooncake.prefill import MooncakeKVManagerPrefill

    calls = []
    manager = object.__new__(MooncakeKVManagerPrefill)
    manager.kv_args = SimpleNamespace(cache_layout=layout, kv_data_ptr=src_ptr)
    manager.engine = SimpleNamespace(
        batch_transfer_sync=lambda session, src, dst, lengths: (
            calls.append((session, src, dst, lengths)) or 0
        )
    )
    return manager, calls


def _route_manager():
    from tokenspeed.runtime.pd.mooncake.prefill import MooncakeKVManagerPrefill

    manager = object.__new__(MooncakeKVManagerPrefill)
    manager.kv_args = SimpleNamespace(
        cache_layout=_typed_layout(local_heads=4, global_heads=4),
        kv_data_ptr=0x1000,
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
    registration = _registration(
        layout,
        session="session",
        endpoint="127.0.0.1",
        pointer=0x1000,
        expected_decode_ranks=(0,),
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
        model_config=SimpleNamespace(
            num_attention_layers=2,
            num_key_value_heads=1,
            hf_config=SimpleNamespace(),
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


def test_terminal_events_clear_transport_room_state(
    monkeypatch: pytest.MonkeyPatch,
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
        _pending_bootstrap_token_table={9: 42},
        _pending_spec_candidate_ids_table={9: [1]},
        connection_lock=nullcontext(),
        addr_to_rooms_tracker={"bootstrap": {9}},
        pop_prefill_metadata=lambda _room: (-1, None),
    )
    receiver = object.__new__(MooncakeKVReceiver)
    receiver.kv_mgr = decode_manager
    receiver.bootstrap_room = 9
    receiver.bootstrap_addr = "bootstrap"
    decode = object.__new__(decode_module.DisaggDecodeExecutor)
    decode.receivers = {"request": receiver}
    decode.gloo_group = None
    decode._local_states = {"request": TransferPoll.Bootstrapped}
    decode.kv_manager = decode_manager
    decode._request_pool_indices = {"request": 7}
    decode._remote_cache_slots = {}
    decode._remote_spec_candidate_ids = {}

    assert len(decode.generate_events()) == 1
    assert decode.pop_remote_cache_slot("request") == 7
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
    executor._request_pool_indices = {}

    executor._cache_prefill(_op())

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == ()
    assert kwargs["block_manifest"].groups[0].block_ids == (2, 3)
    assert executor._request_pool_indices == {"request-0": 7}


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
    assert calls == [
        (
            "session",
            [0x10000 + 1 * 32, 0x10000 + 2 * 32, 0x10000 + 16 + 4 * 32],
            [0x20000 + 8 + 5 * 64, 0x20000 + 8 + 6 * 64, 0x20000 + 40 + 3 * 64],
            [16, 16, 16],
        )
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
        value._pending_bootstrap_token_table = {}
        value._pending_spec_candidate_ids_table = {}
        value.failure_records = {}
        value.record_failure = lambda room, reason: value.failure_records.__setitem__(
            room, reason
        )
        return value

    complete = manager()
    complete._handle_prefill_status(9, TransferPoll.Success, 0, 42, None)
    assert complete.request_status[9] == TransferPoll.WaitingForInput
    complete._handle_prefill_status(9, TransferPoll.Success, 2, -1, None)
    assert complete.request_status[9] == TransferPoll.Success
    assert complete.bootstrap_token_table == {9: 42}

    wrong_rank = manager()
    wrong_rank._handle_prefill_status(9, TransferPoll.Success, 0, -1, None)
    wrong_rank._handle_prefill_status(9, TransferPoll.Success, 1, -1, None)
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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
