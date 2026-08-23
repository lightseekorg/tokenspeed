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

"""msgpack ZMQ wire + transport tests for the direct frontend <-> scheduler path.

Pure CPU: exercises the ``zmq_wire`` codec (native tagged request/output
structs, map-encoded handshake structs, request-type dispatch) and a full
loopback of ``zmq_msgpack.connect_msgpack_engine`` against a hand-rolled
frontend (ROUTER handshake/input + PULL output), with no model or GPU.
"""

from __future__ import annotations

import struct
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace

import msgspec
import pytest
import zmq

from tokenspeed.runtime.engine import zmq_msgpack, zmq_wire
from tokenspeed.runtime.engine.io_struct import (
    AbortReq,
    BatchTokenIDOut,
    BatchTokenIDOutSlim,
    MsgpackEncoder,
    TokenizedGenerateReqInput,
)
from tokenspeed.runtime.engine.request_types import FINISH_ABORT, FINISH_LENGTH
from tokenspeed.runtime.sampling.sampling_params import SamplingParams

_ENCODER = MsgpackEncoder()


def _make_batch_out(**overrides) -> BatchTokenIDOut:
    """A minimal single-request BatchTokenIDOut as stream_output builds it."""
    fields = dict(
        rids=["r1"],
        finished_reasons=[FINISH_LENGTH(length=2).to_json()],
        decoded_texts=[""],
        decode_ids=[[10, 11]],
        read_offsets=[0],
        output_ids=[[10, 11]],
        output_multi_ids=[[]],
        skip_special_tokens=[True],
        spaces_between_special_tokens=[True],
        no_stop_trim=[False],
        prompt_tokens=[3],
        completion_tokens=[2],
        cached_tokens=[1],
        spec_verify_ct=[0],
        input_token_logprobs_val=[],
        input_token_logprobs_idx=[],
        # Per-request list[list], parallel to rids; empty inner list when
        # logprobs are off (OutputProcesser builds it this way).
        output_token_logprobs_val=[[]],
        output_token_logprobs_idx=[[]],
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
        generated_time=0.0,
    )
    fields.update(overrides)
    return BatchTokenIDOut(**fields)


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
    assert len(raw) == 24  # tag + 23 fields
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


def test_slim_out_from_full_roundtrip():
    # Logprobs off: the two logprob columns stay non-ragged (one empty inner
    # list per request).
    out = _make_batch_out()
    slim = BatchTokenIDOutSlim.from_full(out)
    rt = msgspec.msgpack.Decoder(BatchTokenIDOutSlim).decode(_encode_payload(slim))
    assert rt.rids == ["r1"]
    assert rt.output_ids == [[10, 11]]
    assert rt.finished_reasons == ["length"]
    assert rt.prompt_tokens == [3]
    assert rt.completion_tokens == [2]
    assert rt.cached_tokens == [1]
    assert rt.output_token_logprobs_val == [[]]
    assert rt.output_token_logprobs_idx == [[]]
    assert rt.engine_index == 0
    assert rt.num_running == 0
    assert rt.num_waiting == 0
    assert rt.kv_active_pages == 0
    assert rt.kv_total_pages == 0


def test_slim_out_carries_the_producing_engine_index():
    # The output PULL socket has no routing identity, so under DP the batch
    # itself names its producing rank; 0 stays the single-engine default.
    slim = BatchTokenIDOutSlim.from_full(_make_batch_out(), engine_index=3)
    rt = msgspec.msgpack.Decoder(BatchTokenIDOutSlim).decode(_encode_payload(slim))
    assert rt.engine_index == 3


@pytest.mark.parametrize(("prefix_length", "expected_engine_index"), [(9, 0), (10, 7)])
def test_slim_out_load_tail_defaults_for_older_senders(
    prefix_length: int, expected_engine_index: int
):
    # Pre-DP 9-element and pre-load 10-element senders both omit the load tail.
    slim = BatchTokenIDOutSlim.from_full(_make_batch_out(), engine_index=7)
    raw = msgspec.msgpack.decode(_encode_payload(slim))
    rt = msgspec.msgpack.Decoder(BatchTokenIDOutSlim).decode(
        msgspec.msgpack.encode(raw[:prefix_length])
    )
    assert rt.engine_index == expected_engine_index
    assert rt.num_running == 0
    assert rt.num_waiting == 0
    assert rt.kv_active_pages == 0
    assert rt.kv_total_pages == 0


def test_slim_out_sources_output_ids_not_the_detok_window():
    # ``decode_ids`` on the io_struct is the incremental-detokenization window,
    # which starts at the prompt tail for context; ``output_ids`` is the
    # not-yet-sent slice of generated ids. The wire must carry the latter, or
    # prompt tokens leak into the frontend's output.
    out = _make_batch_out(
        decode_ids=[[7, 8, 9, 10, 11]],  # detok window: prompt tail + output
        output_ids=[[10, 11]],  # newly generated only
    )
    slim = BatchTokenIDOutSlim.from_full(out)
    assert slim.output_ids == [[10, 11]]


def test_slim_out_output_ids_none_is_rejected():
    # Substituting empty lists while completion_tokens advances would silently
    # lose tokens on the frontend; from_full must fail loud instead.
    out = _make_batch_out(output_ids=None)
    with pytest.raises(ValueError, match="output_ids"):
        BatchTokenIDOutSlim.from_full(out)


def test_slim_out_carries_logprob_columns():
    # Logprobs on: one value/token-id per newly-decoded token, parallel to rids.
    out = _make_batch_out(
        output_token_logprobs_val=[[-0.5, -1.25]],
        output_token_logprobs_idx=[[10, 11]],
    )
    slim = BatchTokenIDOutSlim.from_full(out)
    rt = msgspec.msgpack.Decoder(BatchTokenIDOutSlim).decode(_encode_payload(slim))
    assert rt.output_token_logprobs_val == [[pytest.approx(-0.5), pytest.approx(-1.25)]]
    assert rt.output_token_logprobs_idx == [[10, 11]]


def test_slim_out_is_tagged_positional_tuple():
    """The wire form pins SMG's exact 14-element positional contract."""
    out = _make_batch_out(
        output_token_logprobs_val=[[-0.5]], output_token_logprobs_idx=[[10]]
    )
    slim = BatchTokenIDOutSlim.from_full(
        out,
        engine_index=1,
        num_running=2,
        num_waiting=3,
        kv_active_pages=4,
        kv_total_pages=20,
    )
    raw = msgspec.msgpack.decode(_encode_payload(slim))
    assert isinstance(raw, list)
    assert raw[0] == "BatchTokenIDOutSlim"
    assert len(raw) == 14
    assert raw[1] == ["r1"]  # rids
    assert raw[2] == [[10, 11]]  # output_ids
    assert raw[3] == ["length"]  # finished_reasons
    assert raw[7] == [[pytest.approx(-0.5)]]
    assert raw[8] == [[10]]
    assert raw[9] == 1  # engine_index (appended)
    assert raw[10] == 2  # num_running
    assert raw[11] == 3  # num_waiting
    assert raw[12] == 4  # kv_active_pages
    assert raw[13] == 20  # kv_total_pages


def test_slim_out_finish_reason_none_maps_to_empty():
    out = _make_batch_out(finished_reasons=[None])
    slim = BatchTokenIDOutSlim.from_full(out)
    assert slim.finished_reasons == [""]


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


class _FakeInputSocket:
    """Yields queued frame lists, then raises zmq.Again like a drained socket."""

    def __init__(self, messages: list[list[bytes]]) -> None:
        self._messages = list(messages)

    def recv_multipart(self, flags: int = 0) -> list[bytes]:
        if not self._messages:
            raise zmq.Again()
        return self._messages.pop(0)

    def close(self) -> None:
        pass


def _add_frames(rid: str, sp: SamplingParams, **overrides) -> list[bytes]:
    payload = _encode_payload(_make_request(rid=rid, sampling_params=sp, **overrides))
    return [zmq_wire.REQ_TYPE_ADD, payload]


def test_recv_socket_aborts_request_with_invalid_sampling_params():
    # top_k=0 is out of range: verify() rejects it. Dropping the request would
    # hang the frontend's stream with no terminal frame, so it still flows —
    # marked via validation_error, which RequestHandler turns into FINISH_ABORT
    # and OutputProcesser streams as a terminal "abort" output.
    bad = _add_frames("bad", SamplingParams(top_k=0))
    good = _add_frames("good", SamplingParams(top_k=-1))
    recv = zmq_msgpack.MsgpackRecvSocket(_FakeInputSocket([bad, good]), vocab_size=32)

    io_bad = recv.recv_pyobj()
    assert io_bad.rid == "bad"
    assert io_bad.validation_error and "top_k" in io_bad.validation_error
    io_good = recv.recv_pyobj()
    assert io_good.rid == "good"
    assert io_good.validation_error is None
    with pytest.raises(zmq.Again):
        recv.recv_pyobj(zmq.NOBLOCK)

    # The marker's terminal frame on the wire: FINISH_ABORT's to_json() dict
    # reduces to finished_reasons=["abort"], which the frontend treats as
    # terminal.
    out = _make_batch_out(
        finished_reasons=[FINISH_ABORT("Invalid request: bad top_k").to_json()]
    )
    slim = BatchTokenIDOutSlim.from_full(out)
    assert slim.finished_reasons == ["abort"]


def test_recv_socket_aborts_logprob_request_when_gate_is_off():
    # The pickle path rejects return_logprob when the server was started
    # without --enable-output-logprobs (input_processor); the msgpack path
    # must enforce the same gate via the terminal-abort marker.
    def _logprob_frames(rid: str) -> list[bytes]:
        return _add_frames(rid, SamplingParams(), return_logprob=True)

    recv_off = zmq_msgpack.MsgpackRecvSocket(
        _FakeInputSocket([_logprob_frames("r1")]),
        vocab_size=32,
        enable_output_logprobs=False,
    )
    io = recv_off.recv_pyobj()
    assert io.rid == "r1"
    assert "--enable-output-logprobs" in io.validation_error

    recv_on = zmq_msgpack.MsgpackRecvSocket(
        _FakeInputSocket([_logprob_frames("r2")]),
        vocab_size=32,
        enable_output_logprobs=True,
    )
    assert recv_on.recv_pyobj().validation_error is None


def test_recv_socket_drops_malformed_message_and_keeps_draining():
    # A version-skewed frontend must not kill the engine: a message that fails
    # to decode is dropped (with a warning) and the drain loop continues.
    malformed = [zmq_wire.REQ_TYPE_ADD, b"\x00garbage"]
    short = [b"\x07"]  # unknown type byte AND too few frames
    good = _add_frames("good", SamplingParams())
    recv = zmq_msgpack.MsgpackRecvSocket(
        _FakeInputSocket([malformed, short, good]), vocab_size=32
    )
    io = recv.recv_pyobj()
    assert io.rid == "good"
    with pytest.raises(zmq.Again):
        recv.recv_pyobj(zmq.NOBLOCK)


def test_recv_socket_verify_does_not_apply_to_abort():
    abort = [zmq_wire.REQ_TYPE_ABORT, msgspec.msgpack.encode(["a", "b"])]
    recv = zmq_msgpack.MsgpackRecvSocket(_FakeInputSocket([abort]), vocab_size=32)
    a1 = recv.recv_pyobj()
    a2 = recv.recv_pyobj()
    assert {a1.rid, a2.rid} == {"a", "b"}


class _FakeOutputSocket:
    """Records sends; optionally raises to exercise best-effort paths."""

    def __init__(self, raise_on_send: bool = False) -> None:
        self.sent: list[list[bytes]] = []
        self._raise = raise_on_send

    def send(self, payload: bytes, flags: int = 0) -> None:
        if self._raise:
            raise zmq.ZMQError()
        self.sent.append([payload])

    def send_multipart(self, frames, flags: int = 0, copy: bool = True) -> None:
        if self._raise:
            raise zmq.ZMQError()
        self.sent.append([bytes(f) for f in frames])

    def close(self) -> None:
        pass


def _decode_last_slim_output(socket: _FakeOutputSocket) -> BatchTokenIDOutSlim:
    return msgspec.msgpack.Decoder(BatchTokenIDOutSlim).decode(socket.sent[-1][-1])


def test_send_socket_uses_zero_load_tail_before_observation():
    socket = _FakeOutputSocket()
    sender = zmq_msgpack.MsgpackSendSocket(socket, engine_index=5)

    sender.send_pyobj(_make_batch_out())

    output = _decode_last_slim_output(socket)
    assert output.engine_index == 5
    assert (
        output.num_running,
        output.num_waiting,
        output.kv_active_pages,
        output.kv_total_pages,
    ) == (0, 0, 0, 0)


def test_send_socket_load_snapshot_setter_sends_no_frame():
    socket = _FakeOutputSocket()
    sender = zmq_msgpack.MsgpackSendSocket(socket)

    sender.set_load_snapshot(1, 2, 3, 4)

    assert socket.sent == []


def test_send_socket_next_output_uses_latest_load_snapshot():
    socket = _FakeOutputSocket()
    sender = zmq_msgpack.MsgpackSendSocket(socket)
    sender.set_load_snapshot(1, 2, 3, 4)
    sender.set_load_snapshot(5, 6, 7, 8)

    sender.send_pyobj(_make_batch_out())

    output = _decode_last_slim_output(socket)
    assert (
        output.num_running,
        output.num_waiting,
        output.kv_active_pages,
        output.kv_total_pages,
    ) == (5, 6, 7, 8)


def test_event_loop_load_snapshot_projects_active_pages_to_direct_output():
    """The direct tail carries active pages, not cached/used pages."""
    pytest.importorskip("tokenspeed_scheduler")
    from tokenspeed.runtime.engine.event_loop import EventLoop

    published = []
    projected = []
    loop = EventLoop.__new__(EventLoop)
    loop.load_snapshot_publisher = SimpleNamespace(
        observe=lambda values: published.append(values)
    )
    loop.send_to_tokenizer = SimpleNamespace(
        set_load_snapshot=lambda *values: projected.append(values)
    )
    loop.output_processor = SimpleNamespace(rid_to_state={"a": object(), "b": object()})
    loop._scheduler_cache_geometry = SimpleNamespace(num_usable_pages=20)

    loop._observe_load_snapshot(
        {"num_queue_reqs": 3, "num_active_pages": 4, "num_cached_pages": 17}
    )

    assert published == [(2, 3, 4, 17, 20)]
    assert projected == [(2, 3, 4, 20)]


def test_send_socket_engine_dead_sentinel():
    # The frontend's output loop treats the raw ENGINE_CORE_DEAD frame as
    # terminal; the transport sends it on engine shutdown/crash cleanup.
    sock = _FakeOutputSocket()
    zmq_msgpack.MsgpackSendSocket(sock).send_engine_dead()
    assert sock.sent == [[zmq_msgpack.ENGINE_CORE_DEAD]]

    # Best-effort: a dead socket on the cleanup path must not raise.
    zmq_msgpack.MsgpackSendSocket(
        _FakeOutputSocket(raise_on_send=True)
    ).send_engine_dead()


def test_handshake_structs_are_map_encoded():
    """Handshake structs use named-key maps (the frontend decodes named
    fields, not positions)."""
    ready = zmq_wire.WireReadyMessage(status="HELLO", local=True, headless=True)
    raw = msgspec.msgpack.decode(zmq_wire.encode(ready))
    assert isinstance(raw, dict)
    assert raw["status"] == "HELLO"

    engine_ready = zmq_wire.WireEngineCoreReadyResponse(
        dtype="float16", multimodal_encoder_dtype="bfloat16"
    )
    raw_engine_ready = msgspec.msgpack.decode(zmq_wire.encode(engine_ready))
    assert raw_engine_ready["dtype"] == "float16"
    assert raw_engine_ready["multimodal_encoder_dtype"] == "bfloat16"

    init = zmq_wire.WireHandshakeInitMessage(
        addresses=zmq_wire.WireHandshakeAddresses(
            inputs=["ipc://in"], outputs=["ipc://out"]
        )
    )
    decoded = zmq_wire.decode_init(zmq_wire.encode(init))
    assert decoded.addresses.inputs == ["ipc://in"]
    assert decoded.addresses.outputs == ["ipc://out"]


def test_init_decode_ignores_unknown_fields():
    """The frontend may add handshake fields; msgspec must tolerate unknown keys."""
    payload = msgspec.msgpack.encode(
        {
            "addresses": {
                "inputs": ["ipc://in"],
                "outputs": ["ipc://out"],
                "coordinator_input": None,
                "some_future_field": 42,
            },
            "parallel_config": {"tp": 1},
            "another_future_field": "ok",
        }
    )
    decoded = zmq_wire.decode_init(payload)
    assert decoded.addresses.inputs == ["ipc://in"]


# --------------------------------------------------------------------------
# Transport loopback (engine CONNECTs; frontend BINDs)
# --------------------------------------------------------------------------


class _FakeSmgFrontend:
    """Hand-rolled frontend side: binds handshake/input ROUTERs + output PULL
    and drives HELLO -> INIT -> READY -> input registration."""

    def __init__(self, ctx: zmq.Context, base: Path) -> None:
        self.handshake = ctx.socket(zmq.ROUTER)
        self.handshake_addr = f"ipc://{base / 'handshake.sock'}"
        self.handshake.bind(self.handshake_addr)

        self.input = ctx.socket(zmq.ROUTER)
        self.input_addr = f"ipc://{base / 'input.sock'}"
        self.input.bind(self.input_addr)

        self.output = ctx.socket(zmq.PULL)
        self.output_addr = f"ipc://{base / 'output.sock'}"
        self.output.bind(self.output_addr)

        self.engine_id: bytes | None = None
        self._encoder = MsgpackEncoder()

    def run_handshake(self) -> None:
        # HELLO (the frontend only needs the routing identity here).
        engine_id, _hello_payload = self.handshake.recv_multipart()
        self.engine_id = engine_id
        # INIT with the bound data-plane addresses.
        init = zmq_wire.WireHandshakeInitMessage(
            addresses=zmq_wire.WireHandshakeAddresses(
                inputs=[self.input_addr], outputs=[self.output_addr]
            )
        )
        self.handshake.send_multipart([engine_id, zmq_wire.encode(init)])
        # READY
        rid_ready, _ = self.handshake.recv_multipart()
        assert rid_ready == engine_id
        # Input-socket registration: [engine_id, EngineCoreReadyResponse].
        reg_id, reg_payload = self.input.recv_multipart()
        assert reg_id == engine_id
        assert len(reg_payload) > 0

    def send_add(self, req: TokenizedGenerateReqInput) -> None:
        frames = self._encoder.encode(req)
        self.input.send_multipart([self.engine_id, zmq_wire.REQ_TYPE_ADD, *frames])

    def send_abort(self, rids: list[str]) -> None:
        self.input.send_multipart(
            [self.engine_id, zmq_wire.REQ_TYPE_ABORT, msgspec.msgpack.encode(rids)]
        )

    def recv_output(self) -> BatchTokenIDOutSlim:
        assert self.output.poll(10_000, zmq.POLLIN), "no output from engine"
        frames = self.output.recv_multipart()
        return msgspec.msgpack.Decoder(BatchTokenIDOutSlim).decode(frames[-1])

    def close(self) -> None:
        for sock in (self.handshake, self.input, self.output):
            sock.close(linger=0)


def test_connect_and_data_plane_roundtrip():
    ctx = zmq.Context()
    tmp = tempfile.TemporaryDirectory()
    base = Path(tmp.name)
    frontend = _FakeSmgFrontend(ctx, base)

    engine_holder: dict = {}
    error_holder: dict = {}

    def _engine():
        try:
            recv, send = zmq_msgpack.connect_msgpack_engine(
                ctx,
                frontend.handshake_addr,
                engine_index=0,
                ready_response=zmq_wire.WireEngineCoreReadyResponse(
                    max_model_len=4096, dtype="bfloat16", vllm_version="tokenspeed-test"
                ),
                vocab_size=32000,
            )
            engine_holder["recv"] = recv
            engine_holder["send"] = send
        except Exception as exc:  # surface into the test thread
            error_holder["err"] = exc

    engine_thread = threading.Thread(target=_engine)
    engine_thread.start()
    frontend.run_handshake()
    engine_thread.join(timeout=15)
    assert not error_holder, error_holder.get("err")
    assert "recv" in engine_holder, "engine handshake did not complete"

    recv = engine_holder["recv"]
    send = engine_holder["send"]
    try:
        # Engine identity is the 2-byte LE index the frontend routes by.
        assert frontend.engine_id == struct.pack("<H", 0)

        # ADD -> the engine decodes a TokenizedGenerateReqInput.
        sp = SamplingParams(temperature=0.5, max_new_tokens=8)
        frontend.send_add(
            _make_request(rid="req-1", input_ids=[5, 6, 7], sampling_params=sp)
        )
        io = recv.recv_pyobj()
        assert io.rid == "req-1"
        assert io.input_ids == [5, 6, 7]
        assert io.sampling_params.max_new_tokens == 8

        # ABORT of two ids -> two AbortReq, one per recv_pyobj call.
        frontend.send_abort(["req-1", "req-2"])
        a1 = recv.recv_pyobj()
        a2 = recv.recv_pyobj()
        assert isinstance(a1, AbortReq) and isinstance(a2, AbortReq)
        assert {a1.rid, a2.rid} == {"req-1", "req-2"}

        # NOBLOCK drain contract: zmq.Again once the buffer + socket are empty.
        with pytest.raises(zmq.Again):
            recv.recv_pyobj(zmq.NOBLOCK)

        # Output path: BatchTokenIDOut -> slim tagged struct on the PULL socket.
        send.send_pyobj(_make_batch_out())
        out = frontend.recv_output()
        assert out.rids == ["r1"]
        assert out.output_ids == [[10, 11]]
        assert out.finished_reasons == ["length"]

        # Non-output control replies are dropped, not crashed on.
        send.send_pyobj(object())
        with pytest.raises(zmq.Again):
            frontend.output.recv(zmq.NOBLOCK)
    finally:
        recv.close()
        send.close()
        frontend.close()
        ctx.term()
        tmp.cleanup()


def test_two_dp_ranks_connect_with_distinct_identities():
    """DP ranks share one frontend socket set: each dials with its own
    engine-index identity, inputs route to exactly one rank, and outputs name
    their producing rank in the batch (the PULL side has no identity)."""
    ctx = zmq.Context()
    tmp = tempfile.TemporaryDirectory()
    frontend = _FakeSmgFrontend(ctx, Path(tmp.name))

    engines: dict[int, tuple] = {}
    errors: dict[int, Exception] = {}

    def _engine(index: int) -> None:
        try:
            engines[index] = zmq_msgpack.connect_msgpack_engine(
                ctx,
                frontend.handshake_addr,
                engine_index=index,
                ready_response=zmq_wire.WireEngineCoreReadyResponse(
                    max_model_len=4096,
                    dtype="bfloat16",
                    vllm_version="tokenspeed-test",
                    data_parallel_size=2,
                    data_parallel_rank=index,
                ),
                vocab_size=32000,
            )
        except Exception as exc:
            errors[index] = exc

    threads = [threading.Thread(target=_engine, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()

    # Phase-tracked handshake: HELLO and READY interleave arbitrarily across
    # ranks on the one ROUTER, so drive it per-identity rather than in order.
    init = zmq_wire.WireHandshakeInitMessage(
        addresses=zmq_wire.WireHandshakeAddresses(
            inputs=[frontend.input_addr], outputs=[frontend.output_addr]
        )
    )
    inits_sent: set[bytes] = set()
    ready: set[bytes] = set()
    while len(ready) < 2:
        engine_id, _payload = frontend.handshake.recv_multipart()
        if engine_id not in inits_sent:
            frontend.handshake.send_multipart([engine_id, zmq_wire.encode(init)])
            inits_sent.add(engine_id)
        else:
            ready.add(engine_id)
    registered: set[bytes] = set()
    while len(registered) < 2:
        reg_id, reg_payload = frontend.input.recv_multipart()
        assert len(reg_payload) > 0
        registered.add(reg_id)

    for t in threads:
        t.join(timeout=15)
    assert not errors, errors
    expected_ids = {struct.pack("<H", 0), struct.pack("<H", 1)}
    assert inits_sent == expected_ids
    assert registered == expected_ids

    try:
        # Inputs route by identity: only rank 1 sees a request addressed to it.
        frames = MsgpackEncoder().encode(_make_request(rid="to-rank-1"))
        frontend.input.send_multipart(
            [struct.pack("<H", 1), zmq_wire.REQ_TYPE_ADD, *frames]
        )
        io = engines[1][0].recv_pyobj()
        assert io.rid == "to-rank-1"
        with pytest.raises(zmq.Again):
            engines[0][0].recv_pyobj(zmq.NOBLOCK)

        # Outputs name their rank: rank 1's batch arrives tagged engine_index=1.
        engines[1][1].send_pyobj(_make_batch_out())
        out = frontend.recv_output()
        assert out.engine_index == 1
    finally:
        for recv, send in engines.values():
            recv.close()
            send.close()
        frontend.close()
        ctx.term()
        tmp.cleanup()
