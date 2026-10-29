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

from tokenspeed.runtime.engine.io_struct import (
    CUSTOM_TYPE_RAW_VIEW,
    AbortReq,
    BatchTokenIDOutSlim,
    DestroyWeightsUpdateGroupReqInput,
    DestroyWeightsUpdateGroupReqOutput,
    FlushCacheReqInput,
    FlushCacheReqOutput,
    GetInternalStateReq,
    GetInternalStateReqOutput,
    InitWeightsUpdateGroupReqInput,
    InitWeightsUpdateGroupReqOutput,
    IsSchedulerPausedReqInput,
    IsSchedulerPausedReqOutput,
    IsSleepingReqInput,
    IsSleepingReqOutput,
    MsgpackDecoder,
    MsgpackEncoder,
    PauseSchedulerReqInput,
    PauseSchedulerReqOutput,
    ProfileReq,
    ProfileReqOutput,
    ProfileReqType,
    ReleaseMemoryOccupationReqInput,
    ReleaseMemoryOccupationReqOutput,
    ResumeMemoryOccupationReqInput,
    ResumeMemoryOccupationReqOutput,
    ResumeSchedulerReqInput,
    ResumeSchedulerReqOutput,
    SessionParams,
    SetInternalStateReq,
    SetInternalStateReqOutput,
    TokenizedGenerateReqInput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightFromDiskReqOutput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromDistributedReqOutput,
    UpdateWeightsFromMooncakeReqInput,
    UpdateWeightsFromMooncakeReqOutput,
    UpdateWeightsFromTensorReqInput,
    UpdateWeightsFromTensorReqOutput,
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
    # Decoding into the message's own type also checks its tag.
    return MsgpackDecoder(type(obj)).decode(MsgpackEncoder().encode(obj))


def _batch_token_id_out(**overrides) -> BatchTokenIDOutSlim:
    fields = dict(
        rids=["a", "b"],
        output_ids=[[1, 2], [3]],
        finished_reasons=["", "stop"],
        prompt_tokens=[4, 5],
        completion_tokens=[2, 1],
        cached_tokens=[0, 3],
        output_token_logprobs_val=[[-0.5, -0.25], []],
        output_token_logprobs_idx=[[1, 2], []],
        finish_reason_details=[None, {"type": "stop", "matched": 7}],
        spec_verify_ct=[0, 0],
        accept_draft_tokens=[None, None],
    )
    fields.update(overrides)
    return BatchTokenIDOutSlim(**fields)


# --------------------------------------------------------------------------
# Tagged roundtrip: one representative instance per message family
# --------------------------------------------------------------------------

_MESSAGES = [
    TokenizedGenerateReqInput(
        rid="r", input_ids=[1, 2], sampling_params=SamplingParams(), stream=True
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
        names=["w"],
        dtype_names=["bfloat16"],
        shapes=[[2, 2]],
        group_name="weight_update_group",
        flush_cache=True,
        weight_version=None,
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
    UpdateWeightsFromMooncakeReqInput(version=7, flush_cache=True, weight_version="7"),
    UpdateWeightsFromMooncakeReqOutput(success=True, message=""),
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
    ProfileReq(type=ProfileReqType.START_PROFILE, activities=["CPU"]),
    ProfileReqOutput(success=True, message="ok"),
    _batch_token_id_out(),
]


@pytest.mark.parametrize("msg", _MESSAGES, ids=lambda m: type(m).__name__)
def test_tagged_union_roundtrip(msg):
    rt = _roundtrip(msg)
    assert type(rt) is type(msg)
    assert msgspec.structs.asdict(rt) == msgspec.structs.asdict(msg)


def test_every_message_tag_is_the_class_name():
    enc = MsgpackEncoder()
    for msg in _MESSAGES:
        raw = msgspec.msgpack.decode(enc.encode(msg)[0])
        assert raw[0] == type(msg).__name__


def test_batch_token_id_out_field_values_roundtrip():
    rt = _roundtrip(_batch_token_id_out())
    assert rt.rids == ["a", "b"]
    assert rt.finished_reasons == ["", "stop"]
    assert rt.finish_reason_details == [None, {"type": "stop", "matched": 7}]
    assert rt.output_ids == [[1, 2], [3]]
    assert rt.output_token_logprobs_val[0] == [
        pytest.approx(-0.5),
        pytest.approx(-0.25),
    ]


def test_accept_draft_tokens_none_mid_stream_roundtrips():
    # Speculative decoding reports acceptance only when a request finishes;
    # mid-stream entries are None and must survive the typed wire.
    rt = _roundtrip(_batch_token_id_out(accept_draft_tokens=[None, 2.5]))
    assert rt.accept_draft_tokens == [None, pytest.approx(2.5)]


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

    req = SetInternalStateReq(server_args={"bad": Opaque()})
    with pytest.raises(TypeError, match="Cannot msgpack-encode"):
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

    rt = MsgpackDecoder(TokenizedGenerateReqInput).decode(frames)
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
        # Reserved wire slot (retired cache field); see MultimodalInputs.
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
# EPD: an encode server's request
# --------------------------------------------------------------------------


def test_encode_request_roundtrips_with_its_feature_out_of_band():
    # The struct an EPD frontend pushes to an encode server's ranks, with a
    # feature tensor large enough to ride an out-of-band aux frame.
    from tokenspeed.runtime.epd.encode_worker import EncodeRequest

    feature = torch.arange(128 * 128, dtype=torch.float32).reshape(128, 128)
    req = EncodeRequest(
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
                feature=feature,
            )
        ],
    )
    frames = MsgpackEncoder().encode(req)
    assert len(frames) == 2
    rt = MsgpackDecoder(EncodeRequest).decode(frames)
    assert (rt.request_id, rt.bootstrap_host, rt.bootstrap_port, rt.bootstrap_room) == (
        "e2",
        "h",
        7,
        11,
    )
    item = rt.items[0]
    assert item.modality == Modality.IMAGE
    assert (item.hash, item.pad_value, item.offsets) == (9, 3, [(0, 4)])
    assert torch.equal(item.feature, feature)
