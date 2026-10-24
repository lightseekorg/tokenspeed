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

"""A prompt the scheduler cannot take ends as that request's error, never as
the scheduler process's exit.

CPU-only: the real request handler, output processor, msgpack wire adapters
and the C++ scheduler binding; no model, device or process group.

The scheduler's admission (``SubmitRequests``) rejects a record it cannot
take -- an empty prompt, a negative generation budget, a token limit (prompt
plus the decode reserve) over the cache's single-request capacity -- and
that rejection ends the scheduler process. ``RequestHandler`` refuses such a
prompt before anything is registered, with the request's own client error,
so every record that reaches the scheduler is admitted. Over the msgpack
wire the refused ADD answers the terminal abort frame on the output socket
and the ADD after it reaches the scheduler's plan.
"""

from __future__ import annotations

import os
import sys
import time
from http import HTTPStatus
from unittest import mock

import msgspec
import zmq
from tokenspeed_scheduler import (
    AttnKind,
    CacheGroupConfig,
    Scheduler,
    SchedulerConfig,
)

from tokenspeed.runtime.distributed.process_group_manager import process_group_manager
from tokenspeed.runtime.engine import zmq_msgpack, zmq_wire
from tokenspeed.runtime.engine.event_loop import EventLoop
from tokenspeed.runtime.engine.generation_output_processor import (
    OutputProcesser,
    RequestState,
)
from tokenspeed.runtime.engine.io_struct import (
    BatchTokenIDOutSlim,
    MsgpackEncoder,
    TokenizedGenerateReqInput,
)
from tokenspeed.runtime.engine.request_handler import RequestHandler
from tokenspeed.runtime.engine.scheduler_utils import (
    UNBOUNDED_CACHED_PREFIX_TOKENS,
    make_spec,
)
from tokenspeed.runtime.sampling.sampling_params import SamplingParams
from tokenspeed.runtime.utils.server_args import ServerArgs

# CPU-only tests scheduled in runtime-1gpu because they import the full runtime.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

# The tiny scheduler below holds 28 tokens per request, and the window is
# that capacity (the loop takes the smaller of the context and the cache):
# the loop passes max_req_len = window - 1 and max_req_input_len = window -
# the decode reserve.
_CAPACITY = 28
_DECODE_RESERVE = 4
_MAX_REQ_LEN = _CAPACITY - 1


class _Metrics:
    enabled = False


def _state(prompt: list[int], **params) -> RequestState:
    return RequestState(
        prompt_input_ids=prompt,
        sampling_params=SamplingParams(**params),
        stream=False,
        tokenizer=None,
        computes_prompt_logprobs=True,
    )


def _tiny_scheduler(decode_input_tokens: int) -> Scheduler:
    """A fused-role scheduler over eight usable pages of four tokens: one
    request may hold ``_CAPACITY`` tokens (``max_single_request_tokens``)."""
    cfg = SchedulerConfig()
    cfg.prefix_granularity = 4
    cfg.max_scheduled_tokens = 64
    cfg.max_batch_size = 8
    cfg.decode_input_tokens = decode_input_tokens
    cfg.num_device_pages = 9
    cfg.num_snapshot_pages = 1  # never retracts
    cfg.disable_l2_cache = True
    cfg.cache_groups = [
        CacheGroupConfig(
            group_id="full",
            block_granularity=4,
            total_pages=9,
            attn_kind=AttnKind.Full,
        )
    ]
    scheduler = Scheduler(cfg)
    assert scheduler.max_single_request_tokens() == _CAPACITY
    return scheduler


def _planned_ids(scheduler: Scheduler) -> set[str]:
    ids: set[str] = set()
    for op in scheduler.next_execution_plan().forward:
        ids.update(op.request_ids)
    return ids


def _handler_bounds(decode_reserve: int) -> RequestHandler:
    """A handler carrying the two bounds the loop passes."""
    handler = object.__new__(RequestHandler)
    handler.max_req_len = _MAX_REQ_LEN
    handler.max_req_input_len = _CAPACITY - decode_reserve
    return handler


def _budget_of(
    handler: RequestHandler, prompt: list[int], requested: int | None
) -> tuple[RequestState, object]:
    state = _state(prompt, max_new_tokens=requested)
    spec = make_spec(
        rid="r", tokens=prompt, max_cached_prefix_tokens=UNBOUNDED_CACHED_PREFIX_TOKENS
    )
    handler._apply_generation_budget(spec, state)
    return state, spec


def _refusal(state: RequestState) -> str:
    reason = state.finished_reason.to_json()
    assert reason["type"] == "abort"
    assert reason["status_code"] == HTTPStatus.BAD_REQUEST
    return reason["message"]


# --------------------------------------------------------------------------
# The handler's bound
# --------------------------------------------------------------------------


def test_the_budget_is_the_room_left_behind_the_prompt():
    handler = _handler_bounds(decode_reserve=1)

    state, spec = _budget_of(handler, list(range(4)), requested=100)
    assert not state.finished
    assert spec.max_new_tokens == _MAX_REQ_LEN - 4 - 1
    assert state.sampling_params.max_new_tokens == _MAX_REQ_LEN - 4 - 1

    # An unset budget resolves to all the room there is.
    state, spec = _budget_of(handler, list(range(4)), requested=None)
    assert not state.finished
    assert spec.max_new_tokens == _MAX_REQ_LEN - 4 - 1

    # The longest prompt the engine admits without a decode reserve: the
    # window less two, where the budget's own term reaches zero.
    state, spec = _budget_of(handler, list(range(_MAX_REQ_LEN - 1)), requested=1)
    assert not state.finished
    assert spec.max_new_tokens == 0


def test_a_prompt_at_the_window_edge_is_refused_as_a_client_error():
    # A prompt of window - 1 tokens asking for one token: the field case.
    # The frontend's own bound (window - 1 without a decode reserve) admits
    # it; the budget would be declared as -1 and the scheduler's admission
    # would end the process.
    handler = _handler_bounds(decode_reserve=1)
    state, spec = _budget_of(handler, list(range(_MAX_REQ_LEN)), requested=1)
    assert (
        f"the input ({_MAX_REQ_LEN} tokens) exceeds the engine's maximum input "
        f"length ({_MAX_REQ_LEN - 1} tokens)"
    ) in _refusal(state)
    # Nothing is declared for a request that never reaches the scheduler.
    assert spec.max_new_tokens == 0
    assert state.sampling_params.max_new_tokens == 1


def test_a_prompt_inside_the_decode_reserve_is_refused_as_a_client_error():
    # With a decode reserve of four, a prompt of window - 3 tokens leaves a
    # budget of one, but its admission record (prompt + the reserve) would
    # exceed the cache's single-request capacity: the scheduler's admission
    # would reject it. The handler's bound is the frontend's.
    handler = _handler_bounds(decode_reserve=_DECODE_RESERVE)
    prompt = list(range(_CAPACITY - 3))
    state, _spec = _budget_of(handler, prompt, requested=1)
    assert (
        f"the input ({len(prompt)} tokens) exceeds the engine's maximum input "
        f"length ({_CAPACITY - _DECODE_RESERVE} tokens)"
    ) in _refusal(state)

    # The longest admitted prompt fits with its reserve.
    prompt = list(range(_CAPACITY - _DECODE_RESERVE))
    state, spec = _budget_of(handler, prompt, requested=100)
    assert not state.finished
    assert len(prompt) + max(spec.max_new_tokens, _DECODE_RESERVE) <= _CAPACITY


def test_an_empty_prompt_is_refused_as_a_client_error():
    handler = _handler_bounds(decode_reserve=1)
    state, _spec = _budget_of(handler, [], requested=4)
    assert "the input is empty" in _refusal(state)


def test_every_record_the_handler_admits_is_taken_by_the_scheduler():
    """The bound's contract against the real scheduler: with the reserve the
    scheduler reserves, every admitted prompt length submits, and the three
    refused shapes are exactly the ones the scheduler would reject."""
    handler = _handler_bounds(decode_reserve=_DECODE_RESERVE)
    scheduler = _tiny_scheduler(decode_input_tokens=_DECODE_RESERVE)
    admitted = []
    refused = []
    for length in range(0, _CAPACITY + 1):
        prompt = list(range(length))
        state = _state(prompt, max_new_tokens=1)
        spec = make_spec(
            rid=f"len-{length}",
            tokens=prompt,
            max_cached_prefix_tokens=UNBOUNDED_CACHED_PREFIX_TOKENS,
        )
        handler._apply_generation_budget(spec, state)
        (refused if state.finished else admitted).append(spec)
    assert [spec.request_id for spec in refused] == ["len-0"] + [
        f"len-{length}"
        for length in range(_CAPACITY - _DECODE_RESERVE + 1, _CAPACITY + 1)
    ]
    scheduler.submit_requests(admitted)
    assert scheduler.waiting_size() == len(admitted)


# --------------------------------------------------------------------------
# Over the msgpack wire
# --------------------------------------------------------------------------


class _SubmitFunnel:
    """Only the state ``EventLoop._submit_scheduler_requests`` reads."""

    _submit_scheduler_requests = EventLoop._submit_scheduler_requests

    def __init__(self, sender, scheduler: Scheduler) -> None:
        self.metrics = _Metrics()
        self.output_processor = OutputProcesser(
            sender, attn_tp_rank=0, metrics=_Metrics()
        )
        self.scheduler = scheduler


def _real_handler(recv, send) -> RequestHandler:
    """The real request handler over the wire adapters; patched out are the
    two collaborators this host lacks (the tokenizer load, the gloo group)."""
    server_args = ServerArgs(model="x")
    server_args.mapping.rank = 0
    with (
        mock.patch(
            "tokenspeed.runtime.engine.request_handler.get_tokenizer",
            return_value=None,
        ),
        mock.patch.object(
            process_group_manager, "get_process_group", return_value=None
        ),
    ):
        return RequestHandler(
            server_args,
            hf_eos_token_id=[2],
            max_req_len=_MAX_REQ_LEN,
            vocab_size=32000,
            recv_func=recv,
            send_func=send,
            can_clear_cache_fn=lambda: True,
            max_req_input_len=_CAPACITY - _DECODE_RESERVE,
            tokenizer_kwargs={},
            device=None,
            l3=None,
        )


def _drain(handler: RequestHandler, engine_in: zmq.Socket, count: int) -> list:
    """``recv_reqs`` until ``count`` requests arrived (inproc delivery is
    asynchronous)."""
    reqs: list = []
    deadline = time.monotonic() + 10.0
    while len(reqs) < count:
        assert time.monotonic() < deadline, f"{len(reqs)} of {count} requests arrived"
        engine_in.poll(100, zmq.POLLIN)
        reqs.extend(handler.recv_reqs())
    return reqs


def _recv_frame(frontend_in: zmq.Socket) -> BatchTokenIDOutSlim:
    assert frontend_in.poll(10_000, zmq.POLLIN), "no output frame"
    return msgspec.msgpack.Decoder(BatchTokenIDOutSlim).decode(
        frontend_in.recv_multipart()[-1]
    )


def test_a_refused_add_answers_an_abort_and_the_next_add_is_served():
    """The loop's admission slice over the real wire adapters: an ADD the
    handler refuses is published at admission as the terminal abort frame,
    the ADD after it is submitted to the scheduler and planned."""
    ctx = zmq.Context()
    engine_in = ctx.socket(zmq.PULL)
    engine_in.bind("inproc://admission-refusal-in")
    frontend_out = ctx.socket(zmq.PUSH)
    frontend_out.connect("inproc://admission-refusal-in")
    frontend_in = ctx.socket(zmq.PULL)
    frontend_in.bind("inproc://admission-refusal-out")
    engine_out = ctx.socket(zmq.PUSH)
    engine_out.connect("inproc://admission-refusal-out")
    recv = zmq_msgpack.MsgpackRecvSocket(
        engine_in,
        vocab_size=32000,
        enable_output_logprobs=False,
        supports_prompt_logprobs=True,
    )
    send = zmq_msgpack.MsgpackSendSocket(engine_out)
    encoder = MsgpackEncoder()

    def add(rid: str, prompt: list[int], max_new_tokens: int) -> None:
        req = TokenizedGenerateReqInput(
            rid=rid,
            input_ids=prompt,
            sampling_params=SamplingParams(max_new_tokens=max_new_tokens),
            stream=True,
        )
        frontend_out.send_multipart([zmq_wire.REQ_TYPE_ADD, *encoder.encode(req)])

    try:
        handler = _real_handler(recv, send)
        loop = _SubmitFunnel(send, _tiny_scheduler(_DECODE_RESERVE))

        # A prompt at the window's edge asking for one token (the field
        # case), one inside the decode reserve, then an ordinary request.
        add("edge", list(range(_MAX_REQ_LEN)), 1)
        add("reserve", list(range(_CAPACITY - 3)), 1)
        add("next", [1, 2, 3], 4)
        specs, states, _bootstraps, aborts = handler.process_requests(
            _drain(handler, engine_in, 3)
        )
        assert aborts == []
        admitted = []
        for spec, state in zip(specs, states):
            if state.finished:
                loop.output_processor.publish_finished_at_admission(
                    spec.request_id, state
                )
                continue
            loop.output_processor.register(spec.request_id, state)
            admitted.append(spec)
        loop._submit_scheduler_requests(admitted)

        for rid, prompt_tokens in (("edge", _MAX_REQ_LEN), ("reserve", _CAPACITY - 3)):
            frame = _recv_frame(frontend_in)
            assert frame.rids == [rid]
            assert frame.finished_reasons == ["abort"]
            assert frame.output_ids == [[]]
            assert frame.prompt_tokens == [prompt_tokens]
        assert _planned_ids(loop.scheduler) == {"next"}
        next_state = loop.output_processor.rid_to_state["next"]
        assert next_state.sampling_params.max_new_tokens == 4
        # Nothing else went out: a refused request's frame is its only trace.
        assert not frontend_in.poll(0, zmq.POLLIN)
    finally:
        recv.close()
        send.close()
        frontend_out.close(linger=0)
        frontend_in.close(linger=0)
        ctx.term()
