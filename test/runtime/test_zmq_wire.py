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

"""The Python side of the msgpack ZMQ wire (``zmq_wire``): the codec the
data plane decodes requests with and the frontend encodes them with.

Pure CPU: native tagged request structs, map-encoded handshake structs and
request-type dispatch. The engine side's codec is ``ts-wire``, checked
against these structs by its conformance fixtures.
"""

from __future__ import annotations

import msgspec
import pytest

from tokenspeed.runtime.engine import zmq_wire
from tokenspeed.runtime.engine.io_struct import (
    AbortReq,
    MsgpackEncoder,
    TokenizedGenerateReqInput,
)
from tokenspeed.runtime.sampling.sampling_params import SamplingParams

_ENCODER = MsgpackEncoder()


def _make_request(rid: str = "r1", **overrides) -> TokenizedGenerateReqInput:
    fields = dict(
        rid=rid,
        input_ids=[1, 2, 3],
        sampling_params=SamplingParams(),
        stream=True,
    )
    fields.update(overrides)
    return TokenizedGenerateReqInput(**fields)


def _encode_payload(obj) -> bytes:
    frames = _ENCODER.encode(obj)
    assert len(frames) == 1, "text requests must not need aux frames"
    return frames[0]


# --------------------------------------------------------------------------
# Codec
# --------------------------------------------------------------------------


def test_generate_request_roundtrip():
    sp = SamplingParams(
        temperature=0.7, max_new_tokens=16, stop_token_ids=[2], seed=1234
    )
    req = _make_request(sampling_params=sp, return_logprob=True)

    [io] = zmq_wire.decode_request_frames([zmq_wire.REQ_TYPE_ADD, _encode_payload(req)])
    assert io.rid == "r1"
    assert io.input_ids == [1, 2, 3]
    assert io.return_logprob is True
    assert io.stream is True
    assert io.sampling_params.temperature == pytest.approx(0.7)
    assert io.sampling_params.max_new_tokens == 16
    assert io.sampling_params.stop_token_ids == {2}
    assert io.sampling_params.seed == 1234


def test_decode_resolves_seed_and_normalizes():
    # The pickle path's input processor runs resolve_seed + normalize; the
    # msgpack path must do the same after decode. Pin the side effects:
    # a None seed derives deterministically from crc32(rid) (rank-agreement),
    # and normalize fills the scheduler-required stop-string defaults.
    import zlib

    req = _make_request()
    [io] = zmq_wire.decode_request_frames([zmq_wire.REQ_TYPE_ADD, _encode_payload(req)])
    assert io.sampling_params.seed == zlib.crc32(b"r1") & 0xFFFFFFFF
    assert io.sampling_params.stop_strs == []
    assert io.sampling_params.stop_str_max_len == 0
    assert io.sampling_params.is_normalized is True


def test_decode_marks_parallel_sampling_for_abort():
    # Belt-and-braces with the frontend, which already rejects n > 1. Marked
    # via validation_error (terminal abort) rather than raised, so a skewed
    # frontend gets a terminated stream instead of a dropped message.
    req = _make_request(sampling_params=SamplingParams(n=2))
    [io] = zmq_wire.decode_request_frames([zmq_wire.REQ_TYPE_ADD, _encode_payload(req)])
    assert io.validation_error is not None
    assert "n=2" in io.validation_error


def test_generate_request_is_tagged_positional_tuple():
    """The wire form is ``[tag, fields...]`` — the frontend's codec depends on
    the tag name, the arity, and the field order. Pin all three."""
    req = _make_request(rid="x", input_ids=[9])
    raw = msgspec.msgpack.decode(_encode_payload(req))
    assert isinstance(raw, list)
    assert raw[0] == "TokenizedGenerateReqInput"
    assert len(raw) == 25  # tag + 24 fields (data_parallel_rank appended last)
    assert raw[1] == "x"  # rid
    assert raw[2] is None  # http_worker_ipc
    assert raw[3] is None  # input_text
    assert raw[4] == [9]  # input_ids
    assert isinstance(raw[5], list) and len(raw[5]) == 29  # SamplingParams
    assert raw[6] is False  # return_logprob
    assert raw[10] is True  # stream


def test_sampling_params_wire_order():
    """SamplingParams rides nested as an UNtagged positional array; its
    declaration order is cross-language wire contract."""
    names = [f.name for f in msgspec.structs.fields(SamplingParams)]
    assert names == [
        "max_new_tokens",
        "stop",
        "stop_token_ids",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "frequency_penalty",
        "presence_penalty",
        "repetition_penalty",
        "min_new_tokens",
        "json_schema",
        "regex",
        "ebnf",
        "structural_tag",
        "ignore_eos",
        "skip_special_tokens",
        "spaces_between_special_tokens",
        "no_stop_trim",
        "thinking_budget",
        "custom_params",
        "stream_interval",
        "logit_bias",
        "seed",
        "logprobs",
        "n",
        "stop_strs",
        "stop_str_max_len",
        "is_normalized",
    ]


def test_generate_request_prefix_decode():
    """The frontend may omit trailing default fields: an 11-element array
    (tag through ``stream``) must decode with the rest defaulted."""
    sp_wire = msgspec.msgpack.decode(msgspec.msgpack.encode(SamplingParams()))
    raw = msgspec.msgpack.encode(
        [
            "TokenizedGenerateReqInput",
            "p1",  # rid
            None,  # http_worker_ipc
            None,  # input_text
            [5, 6],  # input_ids
            sp_wire,  # sampling_params
            False,  # return_logprob
            -1,  # logprob_start_len
            0,  # top_logprobs_num
            None,  # token_ids_logprob
            True,  # stream
        ]
    )
    [io] = zmq_wire.decode_request_frames([zmq_wire.REQ_TYPE_ADD, raw])
    assert io.rid == "p1"
    assert io.input_ids == [5, 6]
    assert io.stream is True
    assert io.multimodal_inputs is None
    assert io.validation_error is None


def test_decode_request_frames_add_and_abort():
    add = zmq_wire.decode_request_frames(
        [zmq_wire.REQ_TYPE_ADD, _encode_payload(_make_request())]
    )
    assert len(add) == 1 and add[0].rid == "r1"

    abort_payload = msgspec.msgpack.encode(["a", "b"])
    aborts = zmq_wire.decode_request_frames([zmq_wire.REQ_TYPE_ABORT, abort_payload])
    assert [a.rid for a in aborts] == ["a", "b"]
    assert all(isinstance(a, AbortReq) for a in aborts)


def test_decode_request_frames_rejects_unknown_type():
    with pytest.raises(ValueError):
        zmq_wire.decode_request_frames([b"\x09", b""])


def test_ready_response_is_map_encoded():
    """The ready response uses named-key maps (the frontend decodes named
    fields, not positions)."""
    engine_ready = zmq_wire.WireEngineCoreReadyResponse(
        dtype="float16", multimodal_encoder_dtype="bfloat16"
    )
    raw_engine_ready = msgspec.msgpack.decode(zmq_wire.encode(engine_ready))
    assert raw_engine_ready["dtype"] == "float16"
    assert raw_engine_ready["multimodal_encoder_dtype"] == "bfloat16"
