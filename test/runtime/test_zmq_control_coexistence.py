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

"""CPU sockets exercising scheduler request ownership and control coexistence."""

from test.runtime.test_zmq_msgpack import _make_batch_out, _make_request

import msgspec
import pytest
import zmq

from tokenspeed.runtime.engine import zmq_msgpack, zmq_wire
from tokenspeed.runtime.engine.io_struct import (
    AbortReq,
    FlushCacheReqInput,
    FlushCacheReqOutput,
    IpcReceiver,
    IpcSender,
    MsgpackDecoder,
    MsgpackEncoder,
)


@pytest.fixture
def channels():
    assert hasattr(
        zmq_msgpack, "SharedFrontendReceiver"
    ), "shared scheduler ingress missing"
    assert hasattr(zmq_msgpack, "SharedFrontendSender"), "owned output routing missing"
    context = zmq.Context()
    sockets = []

    def pair(name, left_type, right_type):
        left, right = context.socket(left_type), context.socket(right_type)
        sockets.extend((left, right))
        left.bind(f"inproc://{name}")
        right.connect(f"inproc://{name}")
        return left, right

    control_in, control_engine = pair("control-in", zmq.PUSH, zmq.PULL)
    direct_in, direct_engine = pair("direct-in", zmq.PAIR, zmq.PAIR)
    control_out, control_client = pair("control-out", zmq.PUSH, zmq.PULL)
    direct_out, direct_client = pair("direct-out", zmq.PUSH, zmq.PULL)
    receiver = zmq_msgpack.SharedFrontendReceiver(
        IpcReceiver(control_engine),
        zmq_msgpack.MsgpackRecvSocket(
            direct_engine,
            100,
            enable_output_logprobs=True,
            supports_prompt_logprobs=True,
        ),
        max_requests_per_round=4,
    )
    sender = zmq_msgpack.SharedFrontendSender(
        IpcSender(control_out),
        zmq_msgpack.MsgpackSendSocket(
            direct_out, engine_index=2, weight_version=lambda: "checkpoint-2"
        ),
    )
    yield (
        receiver,
        sender,
        IpcSender(control_in),
        direct_in,
        IpcReceiver(control_client),
        direct_client,
    )
    receiver.close()
    sender.close()
    for socket in sockets:
        socket.close(linger=0)
    context.term()


def send_direct(socket, obj, kind=zmq_wire.REQ_TYPE_ADD):
    socket.send_multipart([kind, *MsgpackEncoder().encode(obj)])


def receive_ready(receiver):
    # Inproc delivery is immediate; poll helps make the test's precondition explicit.
    return receiver.recv_pyobj(zmq.NOBLOCK)


def test_same_rid_from_both_frontends_has_distinct_scheduler_identity(channels):
    receiver, _, control, direct, _, _ = channels
    control.send_pyobj(_make_request("same"))
    send_direct(direct, _make_request("same"))
    first, second = receive_ready(receiver), receive_ready(receiver)
    assert first.rid != second.rid
    control.send_pyobj(AbortReq(rid="same"))
    send_direct(direct, ["same"], kind=zmq_wire.REQ_TYPE_ABORT)
    assert receive_ready(receiver).rid == first.rid
    assert receive_ready(receiver).rid == second.rid


@pytest.mark.parametrize("owner", ["control", "direct"])
def test_admin_abort_targets_only_selected_owner_and_requests_terminal(channels, owner):
    receiver, _, control, direct, _, _ = channels
    control.send_pyobj(_make_request("same"))
    send_direct(direct, _make_request("same"))
    native, smg = receive_ready(receiver), receive_ready(receiver)
    control.send_pyobj(AbortReq(rid="same", admin_owner=owner, notify_client=True))
    abort = receive_ready(receiver)
    assert abort.rid == (native.rid if owner == "control" else smg.rid)
    assert abort.notify_client is True
    assert abort.admin_owner is None
    with pytest.raises(zmq.Again):
        receive_ready(receiver)


def test_legacy_abort_prefix_preserves_ordinary_cancellation_defaults():
    abort = MsgpackDecoder(AbortReq).decode(
        msgspec.msgpack.encode(["AbortReq", "old", None])
    )
    assert abort.rid == "old"
    assert abort.admin_owner is None
    assert abort.notify_client is False


def test_admin_abort_dispatch_preserves_terminal_notification(monkeypatch):
    from tokenspeed.runtime.engine.request_handler import RequestHandler

    handler = RequestHandler.__new__(RequestHandler)
    # No weight/flush operation is submitted: isolate dispatch from the GPU
    # collective gate, which has its own multi-rank tests.
    monkeypatch.setattr(
        handler,
        "_rendezvous_replica_flush",
        lambda *, pending_flush_outputs: (True, None, None),
    )
    abort = AbortReq(rid="direct-target", notify_client=True)
    specs, states, bootstraps, abort_requests = handler.process_requests([abort])
    assert (specs, states, bootstraps) == ([], [], [])
    assert abort_requests == [abort]
    assert abort_requests[0] is abort
    assert abort_requests[0].notify_client is True


def test_mixed_batch_and_control_reply_go_back_to_correct_frontend(channels):
    receiver, sender, control, direct, control_out, direct_out = channels
    control.send_pyobj(_make_request("same"))
    send_direct(direct, _make_request("same"))
    native, smg = receive_ready(receiver), receive_ready(receiver)
    batch = _make_batch_out()
    values = {}
    for field in msgspec.structs.fields(type(batch)):
        value = getattr(batch, field.name)
        values[field.name] = value * 2 if isinstance(value, list) else value
    values.update(
        rids=[native.rid, smg.rid],
        output_ids=[[101], [202]],
        finished_reasons=[None, {"type": "length", "length": 1}],
    )
    batch = type(batch)(**values)
    sender.send_pyobj(batch)
    control_reply = control_out.recv_pyobj()
    direct_reply = msgspec.msgpack.decode(direct_out.recv_multipart()[0])
    assert control_reply.rids == ["same"]
    assert control_reply.output_ids == [[101]]
    assert control_reply.finished_reasons == [None]
    assert direct_reply[1] == ["same"]
    assert direct_reply[2] == [[202]]
    assert direct_reply[14] == "checkpoint-2"
    assert batch.rids == [native.rid, smg.rid]
    sender.send_pyobj(FlushCacheReqOutput(success=False))
    assert control_out.recv_pyobj().success is False
    assert not direct_out.poll(0)


def test_input_drain_is_fair_and_bounded(channels):
    receiver, _, control, direct, _, _ = channels
    for i in range(10):
        send_direct(direct, _make_request(f"d{i}"))
    control.send_pyobj(FlushCacheReqInput())
    assert isinstance(receive_ready(receiver), FlushCacheReqInput)
    assert all(receive_ready(receiver).rid for _ in range(3))
    with pytest.raises(zmq.Again):
        receive_ready(receiver)
    assert receive_ready(receiver).rid


def test_invalid_direct_request_and_shutdown_keep_control_channel_live(channels):
    receiver, sender, control, direct, control_out, direct_out = channels
    direct.send_multipart([b"unknown", b"invalid"])
    control.send_pyobj(FlushCacheReqInput())
    send_direct(direct, _make_request("bad", return_logprob=True, top_logprobs_num=2))
    assert isinstance(receive_ready(receiver), FlushCacheReqInput)
    assert receive_ready(receiver).validation_error
    sender.send_engine_dead()
    assert direct_out.recv() == zmq_msgpack.ENGINE_CORE_DEAD
    sender.send_pyobj(FlushCacheReqOutput(success=True))
    assert control_out.recv_pyobj().success is True


def test_load_reporter_feeds_native_heartbeat_and_direct_batch_tail(tmp_path):
    from tokenspeed.runtime.engine import load_snapshot

    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    port = socket.bind_to_random_port("tcp://127.0.0.1")
    endpoint = f"tcp://127.0.0.1:{port}"
    direct_values = []
    reporter = load_snapshot.create_load_reporter(
        enabled=True,
        direct_setter=lambda *v: direct_values.append(v),
        endpoint=endpoint,
        dp_rank=0,
        heartbeat_interval=0.05,
        num_total_pages=100,
        sample_stats=lambda: {
            "num_running_requests": 2,
            "num_queue_reqs": 3,
            "num_active_pages": 4,
            "num_cached_pages": 1,
        },
    )
    try:
        reporter.sample_and_observe(2)
        assert socket.poll(1000), "control frontend requires heartbeat in direct mode"
        assert direct_values == [(2, 3, 4, 100)]
    finally:
        reporter.close()
        socket.close(linger=0)
        context.term()


def test_malformed_direct_flood_does_not_hold_a_scheduler_round(channels):
    receiver, _, control, direct, _, _ = channels
    control.send_pyobj(FlushCacheReqInput())
    assert isinstance(receive_ready(receiver), FlushCacheReqInput)
    for _ in range(100):
        direct.send_multipart([b"invalid", b"payload"])
    send_direct(direct, _make_request("after-flood"))
    with pytest.raises(zmq.Again):
        receive_ready(receiver)
    control.send_pyobj(FlushCacheReqInput())
    assert isinstance(receive_ready(receiver), FlushCacheReqInput)


def test_control_embedding_ids_are_restored(channels):
    from tokenspeed.runtime.engine.io_struct import (
        BatchEmbeddingOut,
        TokenizedEmbeddingReqInput,
    )
    from tokenspeed.runtime.sampling.sampling_params import SamplingParams

    receiver, sender, control, _, control_out, _ = channels
    control.send_pyobj(
        TokenizedEmbeddingReqInput(
            rid="embed",
            input_text="hello",
            input_ids=[1],
            sampling_params=SamplingParams(),
        )
    )
    request = receive_ready(receiver)
    sender.send_pyobj(
        BatchEmbeddingOut(
            rids=[request.rid],
            embeddings=[[0.5]],
            prompt_tokens=[1],
            finished_reasons=[{"type": "length", "length": 1}],
        )
    )
    assert control_out.recv_pyobj().rids == ["embed"]


def test_handshake_failure_closes_its_socket(monkeypatch):
    context = zmq.Context()
    sockets = []

    class TrackedContext:
        def socket(self, kind):
            socket = context.socket(kind)
            sockets.append(socket)
            return socket

    bad_init = zmq_wire.encode(
        zmq_wire.WireHandshakeInitMessage(
            addresses=zmq_wire.WireHandshakeAddresses(),
        )
    )
    monkeypatch.setattr(
        zmq_msgpack, "_recv_init_with_timeout", lambda socket, **kwargs: [bad_init]
    )
    try:
        with pytest.raises(ValueError, match="input and output addresses"):
            zmq_msgpack.connect_msgpack_engine(
                TrackedContext(),
                "inproc://missing-frontend",
                0,
                zmq_wire.WireEngineCoreReadyResponse(),
                100,
                enable_output_logprobs=True,
                supports_prompt_logprobs=True,
            )
        assert all(socket.closed for socket in sockets)
    finally:
        context.destroy(linger=0)


def test_shutdown_cancels_handshake_without_waiting_for_late_frontend():
    context = zmq.Context()
    socket = context.socket(zmq.DEALER)
    try:
        with pytest.raises(InterruptedError, match="shutdown"):
            zmq_msgpack._recv_init_with_timeout(socket, shutdown_requested=lambda: True)
    finally:
        socket.close(linger=0)
        context.term()


def test_pause_resume_and_memory_controls_keep_their_replies(channels):
    from tokenspeed.runtime.engine.io_struct import (
        PauseSchedulerReqInput,
        PauseSchedulerReqOutput,
        ReleaseMemoryOccupationReqInput,
        ReleaseMemoryOccupationReqOutput,
        ResumeMemoryOccupationReqInput,
        ResumeMemoryOccupationReqOutput,
        ResumeSchedulerReqInput,
        ResumeSchedulerReqOutput,
    )

    receiver, sender, control, _, control_out, direct_out = channels
    pairs = [
        (PauseSchedulerReqInput(mode="keep"), PauseSchedulerReqOutput(success=True)),
        (ResumeSchedulerReqInput(), ResumeSchedulerReqOutput(success=True)),
        (
            ReleaseMemoryOccupationReqInput(tags=["weights"]),
            ReleaseMemoryOccupationReqOutput(success=False, message="busy"),
        ),
        (ResumeMemoryOccupationReqInput(tags=[]), ResumeMemoryOccupationReqOutput()),
    ]
    for request, reply in pairs:
        control.send_pyobj(request)
        # Controls stay identical so RequestHandler's existing pause/memory state
        # machine and DP same-round gate receive exactly their existing schemas.
        assert receive_ready(receiver) == request
        sender.send_pyobj(reply)
        assert control_out.recv_pyobj() == reply
    assert not direct_out.poll(0)


def test_dp_control_fanout_sends_once_per_worker_over_native_sockets():
    from tokenspeed.runtime.engine.data_parallel_controller import (
        DataParallelController,
    )
    from tokenspeed.runtime.engine.io_struct import PauseSchedulerReqInput

    context = zmq.Context()
    receivers, senders, sockets = [], [], []
    try:
        for rank in range(2):
            push, pull = context.socket(zmq.PUSH), context.socket(zmq.PULL)
            sockets.extend((push, pull))
            push.bind(f"inproc://dp-control-{rank}")
            pull.connect(f"inproc://dp-control-{rank}")
            senders.append(IpcSender(push))
            receivers.append(IpcReceiver(pull))
        controller = DataParallelController.__new__(DataParallelController)
        controller.workers = senders
        controller.dispatching = controller.round_robin_scheduler
        controller.init_dispatcher()
        request = PauseSchedulerReqInput(mode="wait")
        controller._request_dispatcher(request)
        for receiver in receivers:
            assert receiver.recv_pyobj(zmq.NOBLOCK) == request
            with pytest.raises(zmq.Again):
                receiver.recv_pyobj(zmq.NOBLOCK)
    finally:
        for socket in sockets:
            socket.close(linger=0)
        context.term()


def test_loop_handshake_reports_pipeline_layout_and_live_weight_version(monkeypatch):
    import threading
    from types import SimpleNamespace

    captured = {}

    def connect(context, address, index, ready, vocab_size, **kwargs):
        captured.update(ready=ready, index=index, **kwargs)
        return None, None

    monkeypatch.setattr(zmq_msgpack, "connect_msgpack_engine", connect)
    args = SimpleNamespace(
        kv_parallel_size=1,
        max_num_seqs=10,
        chunked_prefill_size=256,
        served_model_name="model",
        model="model",
        zmq_engine_index=0,
        enable_output_logprobs=True,
        weight_version="first",
        speculative_algorithm="NEXTN",
        speculative_num_draft_tokens=4,
        zmq_handshake_endpoint=lambda: "tcp://127.0.0.1:30500",
    )
    loop = SimpleNamespace(
        server_args=args,
        _scheduler_cache_geometry=SimpleNamespace(
            num_device_pages=100, prefix_granularity=16
        ),
        model_config=SimpleNamespace(context_len=4096, dtype="float16", vocab_size=100),
        multimodal_encoder_dtype=None,
        world_size=4,
        dp_size=1,
        attn_tp_size=2,
        pp_size=2,
        dp_rank=0,
        max_total_num_tokens=1600,
        supports_prompt_logprobs=True,
        shutdown_event=threading.Event(),
    )
    zmq_msgpack.connect_msgpack_engine_for_loop(None, loop)
    assert captured["ready"].pipeline_parallel_size == 2
    assert captured["speculative_num_draft_tokens"] == 4
    args.weight_version = "second"
    assert captured["weight_version"]() == "second"
