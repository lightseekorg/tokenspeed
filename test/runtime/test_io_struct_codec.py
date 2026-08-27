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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Engine-IPC msgpack codec tests.

Covers the tagged-union roundtrip for every migrated message family, the
tensor scheme (inline ext raw views vs out-of-band aux frames), the typed
multimodal payload, the SamplingParams struct semantics, and the pickle
fallback flag on the socket wrappers.
"""

from __future__ import annotations

import msgspec
import pytest
import zmq

from tokenspeed.runtime.engine import io_struct
from tokenspeed.runtime.engine.io_struct import (
    CUSTOM_TYPE_RAW_VIEW,
    AbortReq,
    AsyncIpcReceiver,
    BatchEmbeddingOut,
    BatchStrOut,
    BatchTokenIDOut,
    BatchTokenIDOutSlim,
    BlockReqInput,
    BlockReqType,
    CloseSessionReqInput,
    ConfigureLoggingReq,
    DestroyWeightsUpdateGroupReqInput,
    DestroyWeightsUpdateGroupReqOutput,
    ExpertDistributionReq,
    ExpertDistributionReqOutput,
    ExpertDistributionReqType,
    FlushCacheReqInput,
    FlushCacheReqOutput,
    GetInternalStateReq,
    GetInternalStateReqOutput,
    GetLoadReqOutput,
    GetWeightsByNameReqInput,
    GetWeightsByNameReqOutput,
    HealthCheckOutput,
    InitWeightsUpdateGroupReqInput,
    InitWeightsUpdateGroupReqOutput,
    IpcReceiver,
    IpcSender,
    IsSchedulerPausedReqInput,
    IsSchedulerPausedReqOutput,
    IsSleepingReqInput,
    IsSleepingReqOutput,
    LoadSnapshot,
    MsgpackDecoder,
    MsgpackEncoder,
    OpenSessionReqInput,
    OpenSessionReqOutput,
    PauseSchedulerReqInput,
    PauseSchedulerReqOutput,
    PickleWrapper,
    ProfileReq,
    ProfileReqOutput,
    ProfileReqType,
    ReleaseMemoryOccupationReqInput,
    ReleaseMemoryOccupationReqOutput,
    ResumeMemoryOccupationReqInput,
    ResumeMemoryOccupationReqOutput,
    ResumeSchedulerReqInput,
    ResumeSchedulerReqOutput,
    RpcReqInput,
    RpcReqOutput,
    SessionParams,
    SetInternalStateReq,
    SetInternalStateReqOutput,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightFromDiskReqOutput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromDistributedReqOutput,
    UpdateWeightsFromTensorReqInput,
    UpdateWeightsFromTensorReqOutput,
    ipc_message_union,
)
from tokenspeed.runtime.sampling.sampling_params import (
    _TOP_K_DISABLED,
    SamplingParams,
)

torch = pytest.importorskip("torch")

from tokenspeed.runtime.multimodal.inputs import (  # noqa: E402
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from tokenspeed.runtime.multimodal.shm_transport import ShmTensorHandle  # noqa: E402


def _roundtrip(obj):
    enc = MsgpackEncoder()
    dec = MsgpackDecoder(ipc_message_union())
    return dec.decode(enc.encode(obj))


def test_load_snapshot_round_trips_as_one_tagged_frame():
    """The scheduler snapshot stays a standalone, single-frame IPC message."""
    snapshot = LoadSnapshot("boot-a", 1, 0, 2, 3, 4, 5, 6, 1_000)

    frames = MsgpackEncoder().encode(snapshot)

    assert len(frames) == 1
    assert _roundtrip(snapshot) == snapshot


def _batch_token_id_out(**overrides) -> BatchTokenIDOut:
    fields = dict(
        rids=["a", "b"],
        finished_reasons=[None, {"type": "stop", "matched": 7}],
        decoded_texts=["", "x"],
        decode_ids=[[1, 2], [3]],
        read_offsets=[0, 0],
        output_ids=[[1, 2], [3]],
        output_multi_ids=[[], []],
        skip_special_tokens=[True, True],
        spaces_between_special_tokens=[True, False],
        no_stop_trim=[False, False],
        prompt_tokens=[4, 5],
        completion_tokens=[2, 1],
        cached_tokens=[0, 3],
        spec_verify_ct=[0, 0],
        input_token_logprobs_val=[],
        input_token_logprobs_idx=[],
        output_token_logprobs_val=[[-0.5, -0.25], []],
        output_token_logprobs_idx=[[1, 2], []],
        input_top_logprobs_val=[],
        input_top_logprobs_idx=[],
        output_top_logprobs_val=[],
        output_top_logprobs_idx=[],
        input_token_ids_logprobs_val=[],
        input_token_ids_logprobs_idx=[],
        output_token_ids_logprobs_val=[],
        output_token_ids_logprobs_idx=[],
        output_hidden_states=[],
        batch_accept_draft_tokens=[],
        output_extra_infos=[{"decode_prefix_len": 4}, {}],
        generated_time=12.5,
    )
    fields.update(overrides)
    return BatchTokenIDOut(**fields)


# --------------------------------------------------------------------------
# Tagged roundtrip: one representative instance per message family
# --------------------------------------------------------------------------

_MESSAGES = [
    TokenizedGenerateReqInput(
        rid="r", input_ids=[1, 2], sampling_params=SamplingParams(), stream=True
    ),
    TokenizedEmbeddingReqInput(
        rid="r", input_text="t", input_ids=[1], sampling_params=SamplingParams()
    ),
    AbortReq(rid="r"),
    FlushCacheReqInput(),
    FlushCacheReqOutput(success=True),
    PauseSchedulerReqInput(mode="wait"),
    PauseSchedulerReqOutput(success=True, message="ok"),
    ResumeSchedulerReqInput(),
    ResumeSchedulerReqOutput(success=True),
    IsSchedulerPausedReqInput(),
    IsSchedulerPausedReqOutput(is_paused=False),
    UpdateWeightFromDiskReqInput(model_path="/m"),
    UpdateWeightFromDiskReqOutput(success=True, message="m", num_paused_requests=2),
    UpdateWeightsFromDistributedReqInput(
        names=["w"], dtype_names=["bfloat16"], shapes=[[2, 2]]
    ),
    UpdateWeightsFromDistributedReqOutput(success=True, message=""),
    UpdateWeightsFromTensorReqInput(
        serialized_named_tensors=[b"\x00\x01"], load_format=None, flush_cache=True
    ),
    UpdateWeightsFromTensorReqOutput(success=True, message=""),
    InitWeightsUpdateGroupReqInput(
        master_address="h", master_port=1, rank_offset=0, world_size=2
    ),
    InitWeightsUpdateGroupReqOutput(success=True, message=""),
    DestroyWeightsUpdateGroupReqInput(),
    DestroyWeightsUpdateGroupReqOutput(success=True, message=""),
    GetWeightsByNameReqInput(name="w"),
    GetWeightsByNameReqOutput(parameter=[1.0, 2.0]),
    ReleaseMemoryOccupationReqInput(tags=["weights"]),
    ReleaseMemoryOccupationReqOutput(),
    ResumeMemoryOccupationReqInput(),
    ResumeMemoryOccupationReqOutput(),
    IsSleepingReqInput(),
    IsSleepingReqOutput(is_sleeping=False),
    GetInternalStateReq(),
    GetInternalStateReqOutput(internal_state={"k": 1}),
    SetInternalStateReq(server_args={"x": 2}),
    SetInternalStateReqOutput(updated=False, server_args={}),
    ExpertDistributionReq(action=ExpertDistributionReqType.DUMP_RECORD),
    ExpertDistributionReqOutput(),
    ProfileReq(type=ProfileReqType.START_PROFILE, activities=["CPU"]),
    ProfileReqOutput(success=True, message="ok"),
    ConfigureLoggingReq(log_requests=True),
    OpenSessionReqInput(capacity_of_str_len=8),
    CloseSessionReqInput(session_id="s"),
    OpenSessionReqOutput(session_id="s", success=True),
    HealthCheckOutput(),
    RpcReqInput(method="save", parameters={"p": 1}),
    RpcReqOutput(success=True, message=""),
    GetLoadReqOutput(dp_rank=1, num_reqs=2, num_waiting_reqs=1, num_pages=3),
    BlockReqInput(type=BlockReqType.UNBLOCK),
    _batch_token_id_out(),
    BatchStrOut(
        rids=["a"],
        finished_reasons=[{"type": "length"}],
        output_strs=["hi"],
        output_ids=[1],
        prompt_tokens=[1],
        completion_tokens=[1],
        cached_tokens=[0],
        spec_verify_ct=[0],
        input_token_logprobs_val=[],
        input_token_logprobs_idx=[],
        output_token_logprobs_val=[],
        output_token_logprobs_idx=[],
        input_top_logprobs_val=[],
        input_top_logprobs_idx=[],
        output_top_logprobs_val=[],
        output_top_logprobs_idx=[],
        input_token_ids_logprobs_val=[],
        input_token_ids_logprobs_idx=[],
        output_token_ids_logprobs_val=[],
        output_token_ids_logprobs_idx=[],
        output_hidden_states=[],
        batch_accept_draft_tokens=[],
        output_extra_infos=[{}],
        generated_time=1.0,
    ),
    BatchEmbeddingOut(
        rids=["a"], finished_reasons=[None], embeddings=[[0.5, 0.5]], prompt_tokens=[3]
    ),
    PickleWrapper.wrap({"arbitrary": object}),
]


@pytest.mark.parametrize("msg", _MESSAGES, ids=lambda m: type(m).__name__)
def test_tagged_union_roundtrip(msg):
    rt = _roundtrip(msg)
    assert type(rt) is type(msg)
    # Structs with tensor-free fields must roundtrip exactly.
    if type(msg) is not PickleWrapper:
        assert msgspec.structs.asdict(rt).keys() == msgspec.structs.asdict(msg).keys()


def test_every_message_tag_is_the_class_name():
    enc = MsgpackEncoder()
    for msg in _MESSAGES:
        raw = msgspec.msgpack.decode(enc.encode(msg)[0])
        assert raw[0] == type(msg).__name__


def test_batch_token_id_out_field_values_roundtrip():
    rt = _roundtrip(_batch_token_id_out())
    assert rt.rids == ["a", "b"]
    assert rt.finished_reasons == [None, {"type": "stop", "matched": 7}]
    assert rt.output_ids == [[1, 2], [3]]
    assert rt.output_token_logprobs_val[0] == [
        pytest.approx(-0.5),
        pytest.approx(-0.25),
    ]
    assert rt.output_extra_infos == [{"decode_prefix_len": 4}, {}]
    assert rt.generated_time == pytest.approx(12.5)


def test_accept_draft_tokens_none_mid_stream_roundtrips():
    # Speculative decoding reports acceptance only when a request finishes;
    # mid-stream entries are None and must survive the typed wire.
    rt = _roundtrip(_batch_token_id_out(batch_accept_draft_tokens=[None, 2.5]))
    assert rt.batch_accept_draft_tokens == [None, pytest.approx(2.5)]


def test_pickle_wrapper_roundtrip():
    rt = _roundtrip(PickleWrapper.wrap({"a": (1, 2)}))
    assert rt.unwrap() == {"a": (1, 2)}
    assert PickleWrapper.wrap(None) is None


def test_session_params_nested():
    req = TokenizedGenerateReqInput(
        rid="r",
        input_ids=[1],
        sampling_params=SamplingParams(),
        session_params=SessionParams(id="s", offset=3),
    )
    rt = _roundtrip(req)
    assert rt.session_params.id == "s"
    assert rt.session_params.offset == 3


def test_opaque_object_is_rejected_loudly():
    class Opaque:
        pass

    req = RpcReqInput(method="m", parameters={"bad": Opaque()})
    with pytest.raises(TypeError, match="PickleWrapper"):
        MsgpackEncoder().encode(req)


# --------------------------------------------------------------------------
# Tensor scheme: inline ext raw views vs out-of-band aux frames
# --------------------------------------------------------------------------


def _mm_request(feature: "torch.Tensor") -> TokenizedGenerateReqInput:
    item = MultimodalDataItem(
        modality=Modality.IMAGE,
        hash=0xDEADBEEF,
        offsets=[(2, 5)],
        feature=feature,
        model_specific_data={"grid_thw": torch.tensor([[1, 2, 2]])},
    )
    return TokenizedGenerateReqInput(
        rid="mm",
        input_ids=[1, 2, 3],
        sampling_params=SamplingParams(),
        multimodal_inputs=MultimodalInputs(mm_items=[item], im_token_id=9),
    )


def test_small_tensor_is_inlined_as_raw_view_ext():
    req = _mm_request(torch.arange(4, dtype=torch.float32))
    frames = MsgpackEncoder().encode(req)
    assert len(frames) == 1  # everything inline

    # The tensor slot is (dtype, shape, Ext(CUSTOM_TYPE_RAW_VIEW, bytes)).
    raw = msgspec.msgpack.decode(frames[0], ext_hook=lambda code, data: (code, data))
    mm = raw[-3]  # multimodal_inputs (then validation_error, data_parallel_rank)
    feature_slot = mm[0][0][4]
    dtype, shape, data = feature_slot
    assert dtype == "float32"
    assert shape == [4]
    assert data[0] == CUSTOM_TYPE_RAW_VIEW
    assert bytes(data[1]) == torch.arange(4, dtype=torch.float32).numpy().tobytes()


def test_large_tensor_rides_out_of_band_aux_frame():
    feature = torch.arange(1024, dtype=torch.bfloat16).reshape(4, 256)
    req = _mm_request(feature)
    frames = MsgpackEncoder().encode(req)
    assert len(frames) == 2  # primary + one aux frame for the 2 KiB tensor

    raw = msgspec.msgpack.decode(frames[0], ext_hook=lambda code, data: (code, data))
    dtype, shape, data = raw[-3][0][0][4]
    assert dtype == "bfloat16"
    assert shape == [4, 256]
    assert data == 1  # one-based aux frame index (frame 0 = primary buffer)

    rt = MsgpackDecoder(ipc_message_union()).decode(frames)
    item = rt.multimodal_inputs.mm_items[0]
    assert item.feature.dtype == torch.bfloat16
    assert torch.equal(item.feature, feature)
    assert torch.equal(item.grid_thw, torch.tensor([[1, 2, 2]]))
    assert item.offsets == [(2, 5)]
    assert item.hash == 0xDEADBEEF


def test_mm_struct_field_order_is_pinned():
    """MM structs are untagged positional arrays nested inside the tokenized
    request; their declaration order is cross-language wire contract."""
    assert [f.name for f in msgspec.structs.fields(MultimodalDataItem)] == [
        "modality",
        "hash",
        "pad_value",
        "offsets",
        "feature",
        "feature_shm",
        "model_specific_data",
        "encoded",
        "encoded_deepstack",
        "encode_handshake",
    ]
    assert [f.name for f in msgspec.structs.fields(MultimodalInputs)] == [
        "mm_items",
        "im_token_id",
        "video_token_id",
        "mrope_positions",
        "mrope_position_delta",
        "mrope_position_delta_scalar",
        "mrope_position_delta_repeated_cache",
    ]


def test_shm_handle_rides_typed_and_consumable():
    tensor = torch.full((8,), 2.0, dtype=torch.float16)
    handle = ShmTensorHandle.publish(tensor)
    item = MultimodalDataItem(modality=Modality.IMAGE, hash=1, feature_shm=handle)
    req = TokenizedGenerateReqInput(
        rid="shm",
        input_ids=[1],
        sampling_params=SamplingParams(),
        multimodal_inputs=MultimodalInputs(mm_items=[item]),
    )
    rt = _roundtrip(req)
    rt_handle = rt.multimodal_inputs.mm_items[0].feature_shm
    assert rt_handle.shm_name == handle.shm_name
    assert rt_handle.dtype == torch.float16
    assert tuple(rt_handle.shape) == (8,)
    rt_handle.attach()
    assert torch.equal(rt_handle.consume(), tensor)


def test_mm_item_reslots_handle_passed_as_feature():
    # Compatibility: constructors that predate the feature/feature_shm split
    # may pass a SHM handle via ``feature``.
    handle = ShmTensorHandle.publish(torch.ones(4))
    try:
        item = MultimodalDataItem(modality=Modality.IMAGE, feature=handle)
        assert item.feature is None
        assert item.feature_shm is handle
    finally:
        handle.release()


def test_mrope_tensors_roundtrip():
    mm = MultimodalInputs(
        mm_items=[MultimodalDataItem(modality=Modality.IMAGE, hash=1)],
        mrope_positions=torch.arange(9, dtype=torch.int64).reshape(3, 3),
        mrope_position_delta_scalar=-2,
    )
    req = TokenizedGenerateReqInput(
        rid="mr", input_ids=[1], sampling_params=SamplingParams(), multimodal_inputs=mm
    )
    rt = _roundtrip(req)
    assert torch.equal(rt.multimodal_inputs.mrope_positions, mm.mrope_positions)
    assert rt.multimodal_inputs.mrope_position_delta_scalar == -2


# --------------------------------------------------------------------------
# SamplingParams struct semantics
# --------------------------------------------------------------------------


def test_sampling_params_post_init_special_cases():
    greedy = SamplingParams(temperature=0.0)
    assert greedy.temperature == 1.0 and greedy.top_k == 1

    disabled = SamplingParams(top_k=-1)
    assert disabled.top_k == _TOP_K_DISABLED

    listy = SamplingParams(stop="END", stop_token_ids=[3, 3, 5])
    assert listy.stop_strs == "END"
    assert listy.stop_token_ids == {3, 5}


def test_sampling_params_decode_preserves_normalized_state():
    # normalize() runs on the sender; decode must not re-derive stop_strs
    # from the raw ``stop`` alias and lose the resolved fields.
    sp = SamplingParams(stop=["END"], seed=None)
    sp.resolve_seed("rid-1")
    sp.normalize(tokenizer=None)
    assert sp.is_normalized

    rt = msgspec.msgpack.Decoder(SamplingParams).decode(
        msgspec.msgpack.Encoder().encode(sp)
    )
    assert rt.stop_strs == ["END"]
    assert rt.stop_str_max_len == 3
    assert rt.seed == sp.seed
    assert rt.is_normalized


def test_sampling_params_dict_construction_accepts_n():
    # OpenAI-compat serving layers forward "n" inside sampling params dicts.
    sp = SamplingParams(**{"temperature": 0.5, "n": 4})
    assert sp.n == 4


# --------------------------------------------------------------------------
# Socket wrappers + pickle fallback flag
# --------------------------------------------------------------------------


class _LoopbackSocket:
    """Minimal in-memory socket honoring the multipart + pyobj surfaces."""

    def __init__(self):
        self.queue = []
        self.pickled = 0

    def send(self, data, flags=0):
        self.queue.append(bytes(data))

    def send_multipart(self, frames, flags=0, copy=True):
        self.queue.append([bytes(f) for f in frames])

    def recv_multipart(self, flags=0):
        if not self.queue:
            raise zmq.Again()
        return self.queue.pop(0)

    def send_pyobj(self, obj, flags=0):
        import pickle

        self.pickled += 1
        self.queue.append(pickle.dumps(obj))

    def recv_pyobj(self, flags=0):
        import pickle

        self.pickled += 1
        return pickle.loads(self.queue.pop(0))

    def close(self, linger=None):
        pass


def test_ipc_sender_receiver_msgpack_loopback():
    sock = _LoopbackSocket()
    sender, receiver = IpcSender(sock), IpcReceiver(sock)
    sender.send_pyobj(AbortReq(rid="z"))
    rt = receiver.recv_pyobj()
    assert isinstance(rt, AbortReq) and rt.rid == "z"
    assert sock.pickled == 0
    with pytest.raises(zmq.Again):
        receiver.recv_pyobj(zmq.NOBLOCK)


def test_ipc_wrappers_pickle_fallback_flag(monkeypatch):
    monkeypatch.setattr(io_struct, "USE_PICKLE_IPC", True)
    sock = _LoopbackSocket()
    IpcSender(sock).send_pyobj(AbortReq(rid="p"))
    rt = IpcReceiver(sock).recv_pyobj()
    assert isinstance(rt, AbortReq) and rt.rid == "p"
    assert sock.pickled == 2


def test_async_ipc_receiver_decodes_multipart():
    import asyncio

    class _AsyncSock(_LoopbackSocket):
        async def recv_multipart(self, flags=0):
            return self.queue.pop(0)

    sock = _AsyncSock()
    IpcSender(sock).send_pyobj(GetLoadReqOutput(dp_rank=3))
    rt = asyncio.run(AsyncIpcReceiver(sock).recv_pyobj())
    assert isinstance(rt, GetLoadReqOutput) and rt.dp_rank == 3


def test_encode_request_joins_the_tagged_union():
    # Defined in the EPD module, registered by subclassing BaseReq; a decoder
    # built after the import must accept it over the shared channel.
    from tokenspeed.runtime.epd.encode_worker import EncodeRequest

    req = EncodeRequest(
        request_id="e1",
        bootstrap_host="h",
        bootstrap_port=1,
        bootstrap_room=2,
        items=[
            MultimodalDataItem(
                modality=Modality.IMAGE, hash=5, feature=torch.ones(2, 2)
            )
        ],
    )
    rt = _roundtrip(req)
    assert type(rt) is EncodeRequest
    assert rt.request_id == "e1"
    assert torch.equal(rt.items[0].feature, torch.ones(2, 2))


def _gateway_encode_request():
    """The real struct crossing the gateway->encode ZMQ hop, with a feature
    tensor large enough to exercise the out-of-band aux-frame path."""
    from tokenspeed.runtime.epd.encode_worker import EncodeRequest

    return EncodeRequest(
        request_id="e2",
        bootstrap_host="h",
        bootstrap_port=7,
        bootstrap_room=11,
        items=[
            MultimodalDataItem(
                modality=Modality.IMAGE,
                hash=9,
                pad_value=3,
                offsets=[(0, 4)],
                feature=torch.arange(128 * 128, dtype=torch.float32).reshape(128, 128),
            )
        ],
    )


def test_prepickled_send_transcodes_onto_the_msgpack_wire():
    # Gateway->encode hop regression: the encode servicer serializes the
    # EncodeRequest off-thread (pickle) and pushes the raw bytes via ``send``,
    # while the encode loop decodes with the tagged-union msgpack decoder.
    # Without transcoding, pickle's PROTO byte (0x80) reads as a msgpack map
    # and the receiver dies with "Expected `array`, got `object`".
    import pickle

    from tokenspeed.runtime.epd.encode_worker import EncodeRequest

    req = _gateway_encode_request()
    sock = _LoopbackSocket()
    IpcSender(sock).send(pickle.dumps(req, protocol=pickle.DEFAULT_PROTOCOL))
    rt = IpcReceiver(sock).recv_pyobj()
    assert type(rt) is EncodeRequest
    assert (rt.request_id, rt.bootstrap_host, rt.bootstrap_port, rt.bootstrap_room) == (
        "e2",
        "h",
        7,
        11,
    )
    item = rt.items[0]
    assert item.modality == Modality.IMAGE
    assert (item.hash, item.pad_value, item.offsets) == (9, 3, [(0, 4)])
    assert torch.equal(item.feature, req.items[0].feature)
    assert sock.pickled == 0  # the wire itself stayed msgpack


def test_prepickled_send_passes_through_under_pickle_flag(monkeypatch):
    import pickle

    from tokenspeed.runtime.epd.encode_worker import EncodeRequest

    monkeypatch.setattr(io_struct, "USE_PICKLE_IPC", True)
    req = _gateway_encode_request()
    sock = _LoopbackSocket()
    IpcSender(sock).send(pickle.dumps(req, protocol=pickle.DEFAULT_PROTOCOL))
    rt = IpcReceiver(sock).recv_pyobj()
    assert type(rt) is EncodeRequest and rt.bootstrap_room == 11
