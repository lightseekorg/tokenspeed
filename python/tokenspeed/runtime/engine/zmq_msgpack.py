# Copyright (c) 2026 LightSeek Foundation
#
# Engine-side ZMQ transport for the direct SMG <-> scheduler msgpack path.
#
# In the default (pickle) mode the scheduler BINDS a PULL/PUSH pair and the
# Python tokenizer_manager connects in. SMG cannot speak that pickle protocol,
# so this module inverts the topology: SMG (the frontend) BINDS the handshake/
# input/output sockets and the scheduler CONNECTS in, completing the
# HELLO -> INIT -> READY handshake and then exchanging the msgpack frames
# defined in ``zmq_wire``.
#
# The adapters below expose ``recv_pyobj``/``send_pyobj`` so the existing
# RequestHandler and OutputProcesser drive them unchanged; only the wire codec
# differs. A control call off the wire (``zmq_wire.REQ_TYPE_CONTROL``) becomes
# the scheduler's own control request here and its output goes back as the
# call's reply (``ControlCalls``). This module is additive and only reached
# when ``--zmq-msgpack`` is on.

from __future__ import annotations

import collections
import dataclasses
import logging
import math
import struct
import time
from collections.abc import Callable, Sequence

import msgspec
import zmq

from tokenspeed.runtime.engine import zmq_wire
from tokenspeed.runtime.engine.io_struct import (
    BatchTokenIDOut,
    BatchTokenIDOutSlim,
    FlushCacheReqInput,
    FlushCacheReqOutput,
    MsgpackEncoder,
    ProfileReq,
    ProfileReqOutput,
    ProfileReqType,
    TokenizedGenerateReqInput,
)
from tokenspeed.runtime.engine.logprobs import (
    resolve_logprob_start_len,
    top_logprobs_launch_refusal,
    top_logprobs_request_refusal,
)
from tokenspeed.runtime.utils.env import envs

logger = logging.getLogger(__name__)

# SMG allows a registering worker a long connect window (~600 s), so the
# frontend may start the handshake well after this engine finished model load.
# Wait for INIT in slices, logging progress, instead of dying at 60 s.
_INIT_TIMEOUT_MS = 600_000
_INIT_POLL_SLICE_MS = 30_000

# Raw single-frame sentinel SMG's output loop treats as terminal: the worker
# is marked dead instead of staying healthy-idle after this engine exits.
ENGINE_CORE_DEAD = b"ENGINE_CORE_DEAD"

# A control call's cache flush is the frontend's retry loop moved onto the
# scheduler's own thread. The scheduler refuses a flush while any request pins
# blocks, and a finished request is retired a round after its last token was
# streamed, so one attempt made right after a completion is refused although
# nothing is in flight: the attempt is repeated every FLUSH_RETRY_INTERVAL_S
# until it is accepted or max(timeout_s, FLUSH_RETIRE_GRACE_S) has passed
# since the call arrived, and the call gets exactly one answer.
FLUSH_RETRY_INTERVAL_S = 0.05
FLUSH_RETIRE_GRACE_S = 2.0

# The positional structs the send adapter emits, by tag, declared in the
# engine's ready response (``positional_layouts``) so the frontend checks its
# decoders against them: the control reply on every role (a stray call is
# answered as not served even where none is served), the output batch where
# generate requests are served (``emitted_layouts``).
OUTPUT_LAYOUTS: dict[str, list[str]] = {
    zmq_wire.wire_tag(struct): zmq_wire.positional_layout(struct)
    for struct in (BatchTokenIDOutSlim, zmq_wire.ControlReplySlim)
}


def emitted_layouts(request_types: Sequence[str]) -> dict[str, list[str]]:
    """The entries of ``OUTPUT_LAYOUTS`` a role serving ``request_types`` can
    emit on its output socket."""
    emitted = {zmq_wire.wire_tag(zmq_wire.ControlReplySlim)}
    if zmq_wire.REQUEST_KIND_ADD in request_types:
        emitted.add(zmq_wire.wire_tag(BatchTokenIDOutSlim))
    return {tag: layout for tag, layout in OUTPUT_LAYOUTS.items() if tag in emitted}


# The request types each engine role serves, declared in its ready response
# (``request_types``, the frontend's signal for what a connection takes,
# control calls included) and enforced by its receive adapter: the scheduler
# loop takes generate, abort and control requests; the EPD encode loop takes
# encode requests only (it answers a control call as not served, so the
# caller does not wait out its timeout, and drops the rest).
SCHEDULER_REQUEST_TYPES: tuple[str, ...] = (
    zmq_wire.REQUEST_KIND_ADD,
    zmq_wire.REQUEST_KIND_ABORT,
    zmq_wire.REQUEST_KIND_CONTROL,
)
ENCODE_REQUEST_TYPES: tuple[str, ...] = (zmq_wire.REQUEST_KIND_ENCODE,)


class ControlCallError(ValueError):
    """A control call this transport cannot turn into a scheduler request;
    answered under its call id so the frontend's caller does not wait out its
    timeout."""

    def __init__(self, call_id: int, message: str) -> None:
        super().__init__(message)
        self.call_id = call_id


def _flush_timeout_s(call: zmq_wire.ControlCall) -> float:
    """The ``timeout_s`` of a ``flush_cache`` call (0 when not given)."""
    if not call.args:
        return 0.0
    try:
        timeout_s = float(call.args[0])
    except (TypeError, ValueError) as exc:
        raise ControlCallError(
            call.call_id,
            f"flush_cache: timeout_s must be a number, got {call.args[0]!r}",
        ) from exc
    if not math.isfinite(timeout_s) or timeout_s < 0:
        raise ControlCallError(
            call.call_id, "flush_cache: timeout_s must be finite and non-negative"
        )
    return timeout_s


# The ``start_profile`` options this engine takes (``ProfileReq``'s fields the
# in-process ``AsyncLLM.start_profile`` exposes). A key outside this set is
# refused, never silently dropped: a frontend would otherwise read success
# while its option did nothing.
_START_PROFILE_OPTIONS = frozenset(
    {
        "output_dir",
        "start_step",
        "num_steps",
        "activities",
        "with_stack",
        "record_shapes",
        "profile_by_stage",
        "profile_id",
    }
)


def _profile_option(options: dict, name: str, types: tuple[type, ...], call_id: int):
    """One ``start_profile`` option, None when absent, refused when mistyped."""
    value = options.get(name)
    if value is None:
        return None
    # bool is an int subclass; an integer option does not take a flag.
    if not isinstance(value, types) or (bool not in types and isinstance(value, bool)):
        expected = " | ".join(t.__name__ for t in types)
        raise ControlCallError(
            call_id, f"start_profile: {name} must be {expected}, got {value!r}"
        )
    return value


def _start_profile_request(call: zmq_wire.ControlCall, rid: str) -> ProfileReq:
    """The scheduler's ``ProfileReq`` for a ``start_profile`` call.

    The options take the defaults the in-process frontend applies
    (``AsyncLLM.start_profile``): the stack is recorded unless the caller or
    ``TOKENSPEED_PROFILE_WITH_STACK`` says no, an unnamed profile is named by
    its start time, and the handler fills ``activities`` and ``output_dir``.
    """
    options = call.args[0] if call.args else {}
    if not isinstance(options, dict):
        raise ControlCallError(
            call.call_id, f"start_profile: options must be a map, got {options!r}"
        )
    unknown = sorted(str(key) for key in options.keys() - _START_PROFILE_OPTIONS)
    if unknown:
        raise ControlCallError(
            call.call_id,
            f"start_profile: unknown options {unknown!r}; this engine takes "
            f"{sorted(_START_PROFILE_OPTIONS)!r}",
        )
    activities = _profile_option(options, "activities", (list,), call.call_id)
    if activities is not None and not all(isinstance(a, str) for a in activities):
        raise ControlCallError(
            call.call_id,
            f"start_profile: activities must be strings, got {activities!r}",
        )
    with_stack = _profile_option(options, "with_stack", (bool,), call.call_id)
    env_with_stack = envs.TOKENSPEED_PROFILE_WITH_STACK.get()
    profile_id = _profile_option(options, "profile_id", (str,), call.call_id)
    return ProfileReq(
        rid=rid,
        type=ProfileReqType.START_PROFILE,
        output_dir=_profile_option(options, "output_dir", (str,), call.call_id),
        start_step=_profile_option(options, "start_step", (int,), call.call_id),
        num_steps=_profile_option(options, "num_steps", (int,), call.call_id),
        activities=list(activities) if activities else None,
        with_stack=not (with_stack is False or env_with_stack is False),
        record_shapes=_profile_option(options, "record_shapes", (bool,), call.call_id),
        profile_by_stage=bool(
            _profile_option(options, "profile_by_stage", (bool,), call.call_id)
        ),
        profile_id=profile_id or time.strftime("%Y%m%d-%H%M%S"),
    )


@dataclasses.dataclass
class _FlushCall:
    """A ``flush_cache`` call in flight: its deadline and attempt count."""

    call_id: int
    deadline: float
    attempts: int = 1
    # When a refused attempt is re-issued; None while one is with the scheduler.
    retry_at: float | None = None


class ControlCalls:
    """The control calls in flight on this rank's wire.

    Shared by the receive and send adapters, which both run on the scheduler
    thread. ``requests_for`` turns a decoded call into the scheduler's own
    control request with the call id in its ``rid``; ``reply_for`` turns that
    request's output back into the ``ControlReplySlim`` the frontend awaits;
    ``due_requests`` re-issues a refused flush once its retry interval has
    passed (the receiver asks at every drain). ``send_reply`` is bound by the
    send adapter so a call refused at decode time is answered at once.
    """

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._flushes: dict[str, _FlushCall] = {}
        self.send_reply: Callable[[int, bool, str | None], None] | None = None

    def requests_for(self, call: zmq_wire.ControlCall) -> list:
        """The scheduler request(s) ``call`` becomes, to hand to the handler now.

        Raises:
            ControlCallError: an unknown method or malformed arguments; the
                caller answers it under the call id.
        """
        rid = zmq_wire.control_rid(call.call_id)
        if call.method == "flush_cache":
            timeout_s = _flush_timeout_s(call)
            self._flushes[rid] = _FlushCall(
                call_id=call.call_id,
                deadline=self._clock() + max(timeout_s, FLUSH_RETIRE_GRACE_S),
            )
            return [FlushCacheReqInput(rid=rid)]
        if call.method == "start_profile":
            return [_start_profile_request(call, rid)]
        if call.method == "stop_profile":
            return [ProfileReq(rid=rid, type=ProfileReqType.STOP_PROFILE)]
        raise ControlCallError(call.call_id, f"unknown control method {call.method!r}")

    def due_requests(self) -> list:
        """The refused flushes whose retry interval has passed, re-issued."""
        now = self._clock()
        due = []
        for rid, flush in self._flushes.items():
            if flush.retry_at is not None and now >= flush.retry_at:
                flush.retry_at = None
                flush.attempts += 1
                due.append(FlushCacheReqInput(rid=rid))
        return due

    def refuse(self, call_id: int, message: str) -> None:
        """Answer a call that never reached the scheduler."""
        if self.send_reply is None:
            logger.warning(
                f"msgpack input: control call {call_id:d} refused with no sender bound: "
                f"{message!s}"
            )
            return
        self.send_reply(call_id, False, message)

    def reply_for(
        self, output: FlushCacheReqOutput | ProfileReqOutput, *, engine_index: int
    ) -> zmq_wire.ControlReplySlim | None:
        """The reply ``output`` earns now, or None: a refused flush inside its
        deadline is re-armed instead (answered later), and an output of no call
        of the frontend's (rid None: the scheduler's own) earns none."""
        call_id = zmq_wire.control_call_id(output.rid)
        if call_id is None:
            return None
        if isinstance(output, ProfileReqOutput):
            return zmq_wire.ControlReplySlim(
                call_id=call_id,
                success=bool(output.success),
                message=output.message,
                engine_index=engine_index,
            )
        flush = self._flushes.get(output.rid)
        if flush is None:
            return None
        if output.success:
            del self._flushes[output.rid]
            return zmq_wire.ControlReplySlim(
                call_id=call_id, success=True, message=None, engine_index=engine_index
            )
        now = self._clock()
        if now < flush.deadline:
            flush.retry_at = now + FLUSH_RETRY_INTERVAL_S
            return None
        del self._flushes[output.rid]
        return zmq_wire.ControlReplySlim(
            call_id=call_id,
            success=False,
            message=(
                "Cache flush refused: requests still in flight after "
                f"{flush.attempts:d} attempts"
            ),
            engine_index=engine_index,
        )


def _engine_identity(engine_index: int) -> bytes:
    """Two-byte little-endian ROUTER identity for this engine (its engine index)."""
    return struct.pack("<H", engine_index)


class MsgpackRecvSocket:
    """Adapts a DEALER input socket to the ``recv_pyobj`` contract.

    Each ZMQ message is ``[type_byte, msgpack payload]`` (the DEALER strips the
    routing identity). One ABORT message can expand to several ``AbortReq``, so
    decoded io_structs are buffered and handed out one per ``recv_pyobj`` call,
    preserving the "one object per recv, ``zmq.Again`` when empty" contract the
    RequestHandler's drain loop relies on.

    ``vocab_size`` lets this socket run ``sampling_params.verify`` on each
    generate request, ``enable_output_logprobs`` gates ``return_logprob`` and
    ``supports_prompt_logprobs`` (the engine's startup verdict, see
    ``DeviceSpecs``) gates a ``logprob_start_len`` that asks for prompt
    logprobs (the pickle path enforces all three in the tokenizer_manager's
    input processor, which the msgpack path bypasses; the pure wire
    ``to_io_struct`` has none of these values). An invalid request is not
    dropped — that would hang the frontend's stream with no terminal frame. It
    is marked via ``validation_error`` so RequestHandler admits it pre-finished
    (FINISH_ABORT) and OutputProcesser emits a terminal "abort" output.

    ``top_logprobs_unservable`` is the launch-level reason this engine computes
    no ranked alternatives at all (None when it does); a request's
    ``top_logprobs_num`` is taken under the rules the input processor applies
    (``logprobs.top_logprobs_request_refusal``) and the alternatives ride the
    slim output's appended columns.

    A control call becomes the scheduler's own control request through
    ``control`` (shared with the send adapter, which answers the request's
    output under the call id); a call this side cannot place is answered as
    refused rather than dropped.

    ``request_types`` names the request kinds this engine role serves (the
    ones its ready response declares): a request of another kind is dropped
    with a warning instead of reaching a loop that has no handler for it, and
    a control call to a role that serves none is answered as not served.
    """

    def __init__(
        self,
        socket: zmq.Socket,
        vocab_size: int,
        *,
        enable_output_logprobs: bool,
        supports_prompt_logprobs: bool,
        top_logprobs_unservable: str | None,
        control: ControlCalls,
        request_types: Sequence[str],
    ) -> None:
        self._socket = socket
        self._vocab_size = vocab_size
        self._enable_output_logprobs = enable_output_logprobs
        self._supports_prompt_logprobs = supports_prompt_logprobs
        self._top_logprobs_unservable = top_logprobs_unservable
        self._control = control
        self._request_types = frozenset(request_types)
        self._pending: collections.deque = collections.deque()

    def _validation_error(self, obj: TokenizedGenerateReqInput) -> str | None:
        """Return the reason ``obj`` must be rejected, or None if it is valid.

        Mirrors the tokenizer-side ``InputProcessor`` gate for the SGLang
        logprob knobs. ``logprob_start_len`` is resolved in place (-1 -> last
        prompt token). A start that asks for prompt logprobs is refused when
        the engine cannot compute them (``supports_prompt_logprobs``), and
        otherwise because the slim per-step output (``BatchTokenIDOutSlim``)
        has no prompt-logprob columns -- computing them would silently drop
        them on the wire. ``top_logprobs_num`` takes the input processor's
        rules (a request sampled entirely in its prefill, on a launch that
        computes alternatives); the slim output carries them.
        """
        if obj.return_logprob and not self._enable_output_logprobs:
            return (
                "logprobs were requested but the server was started without "
                "--enable-output-logprobs"
            )
        if obj.return_logprob:
            if obj.top_logprobs_num:
                reason = (
                    top_logprobs_request_refusal(
                        obj.top_logprobs_num, obj.sampling_params.max_new_tokens
                    )
                    or self._top_logprobs_unservable
                )
                if reason is not None:
                    return reason
            if obj.token_ids_logprob:
                return "token_ids_logprob is not supported yet"
            if obj.input_ids is None:
                return "return_logprob requires token inputs"
            try:
                obj.logprob_start_len = resolve_logprob_start_len(
                    obj.logprob_start_len, len(obj.input_ids)
                )
            except ValueError as exc:
                return str(exc)
            if obj.logprob_start_len < len(obj.input_ids) - 1:
                if not self._supports_prompt_logprobs:
                    return (
                        "logprob_start_len >= 0 (prompt logprobs) is not supported by "
                        "this engine: the model narrows its prefill rows or runs "
                        "pipeline parallel; use logprob_start_len=-1"
                    )
                return (
                    "logprob_start_len >= 0 (prompt logprobs) is not carried on the "
                    "msgpack output wire; use logprob_start_len=-1"
                )
        try:
            obj.sampling_params.verify(self._vocab_size)
        except ValueError as exc:
            return str(exc)
        return None

    def recv_pyobj(self, flags: int = 0):
        while not self._pending:
            # A refused flush whose retry interval has passed goes back to the
            # scheduler before the socket is read, so it is re-attempted
            # whether or not the frontend sent anything new this round.
            self._pending.extend(self._control.due_requests())
            if self._pending:
                break
            # Raises zmq.Again (a zmq.ZMQError) under NOBLOCK when idle, which
            # the caller catches to end the drain loop.
            frames = self._socket.recv_multipart(flags)
            try:
                decoded = zmq_wire.decode_request_frames(frames)
            except Exception as exc:
                # A malformed frame (e.g. a version-skewed frontend) must not
                # escape the RequestHandler's zmq.ZMQError-only catch and kill
                # the engine; drop the message and keep draining.
                logger.warning(
                    "msgpack input: dropping malformed message "
                    f"({len(frames):d} frames, sizes="
                    f"{[len(frame) for frame in frames]!s}, head="
                    f"{(frames[0][:8].hex() if frames else '')!s}): {exc!s}",
                )
                continue
            for obj in decoded:
                kind = zmq_wire.request_kind(obj)
                if kind not in self._request_types:
                    if isinstance(obj, zmq_wire.ControlCall):
                        self._control.refuse(
                            obj.call_id,
                            f"{obj.method}: control calls are not served on this "
                            "engine role",
                        )
                    else:
                        logger.warning(
                            f"msgpack input: dropping a request of kind {kind!s}, "
                            "which this engine role does not serve",
                        )
                    continue
                if isinstance(obj, zmq_wire.ControlCall):
                    try:
                        self._pending.extend(self._control.requests_for(obj))
                    except ControlCallError as exc:
                        logger.warning(
                            f"msgpack input: control call {exc.call_id:d} refused: {exc!s}"
                        )
                        self._control.refuse(exc.call_id, str(exc))
                    continue
                # AbortReq carries no sampling_params; only generate requests
                # are validated. A rejected request still flows, pre-marked.
                # The wire layer may have marked it already (e.g. n != 1).
                if isinstance(obj, TokenizedGenerateReqInput):
                    reason = obj.validation_error or self._validation_error(obj)
                    if reason is not None:
                        logger.warning(
                            f"msgpack input: aborting invalid request {obj.rid!s}: "
                            f"{reason!s}",
                        )
                        obj.validation_error = reason
                self._pending.append(obj)
        return self._pending.popleft()

    def close(self) -> None:
        self._socket.close()


class MsgpackSendSocket:
    """Adapts a PUSH output socket to the ``send_pyobj`` contract.

    ``BatchTokenIDOut`` is sliced down to the tagged ``BatchTokenIDOutSlim``
    (the frontend detokenizes itself, so the incremental-detokenization
    columns would be dead weight every step). The output of a control call's
    request (``FlushCacheReqOutput``, ``ProfileReqOutput`` under the call's
    rid) goes out as the call's ``ControlReplySlim``. Every other control
    reply (GetLoad/...) has no SMG consumer in this mode and is dropped with
    a warning rather than crashing the scheduler.
    """

    def __init__(
        self, socket: zmq.Socket, engine_index: int = 0, *, control: ControlCalls
    ) -> None:
        self._socket = socket
        self._engine_index = engine_index
        self._encoder = MsgpackEncoder()
        self._load_snapshot = (0, 0, 0, 0)
        self._control = control
        control.send_reply = self._send_control_reply

    def set_load_snapshot(
        self,
        num_running: int,
        num_waiting: int,
        kv_active_pages: int,
        kv_total_pages: int,
    ) -> None:
        """Replace the load tail for the next ordinary output batch.

        Observation and output forwarding share the scheduler thread, so this
        latest-wins assignment needs neither a lock nor a standalone send.
        """
        self._load_snapshot = (
            num_running,
            num_waiting,
            kv_active_pages,
            kv_total_pages,
        )

    def send_pyobj(self, obj) -> None:
        if isinstance(obj, BatchTokenIDOut):
            # The PULL side carries no routing identity, so the batch itself
            # names its producing rank; the frontend attributes per-rank
            # outputs and load by this index under DP.
            (
                num_running,
                num_waiting,
                kv_active_pages,
                kv_total_pages,
            ) = self._load_snapshot
            slim = BatchTokenIDOutSlim.from_full(
                obj,
                engine_index=self._engine_index,
                num_running=num_running,
                num_waiting=num_waiting,
                kv_active_pages=kv_active_pages,
                kv_total_pages=kv_total_pages,
            )
            self._socket.send_multipart(self._encoder.encode(slim), copy=False)
        elif isinstance(obj, (FlushCacheReqOutput, ProfileReqOutput)):
            # None: a refused flush re-armed for another attempt (answered
            # later), or an output of no call of the frontend's.
            reply = self._control.reply_for(obj, engine_index=self._engine_index)
            if reply is not None:
                self._socket.send(zmq_wire.encode(reply))
        else:
            logger.warning(
                "msgpack output: dropping unsupported control reply "
                f"{type(obj).__name__!s}",
            )

    def _send_control_reply(
        self, call_id: int, success: bool, message: str | None
    ) -> None:
        self._socket.send(
            zmq_wire.encode(
                zmq_wire.ControlReplySlim(
                    call_id=call_id,
                    success=success,
                    message=message,
                    engine_index=self._engine_index,
                )
            )
        )

    def send_engine_dead(self) -> None:
        """Best-effort ENGINE_CORE_DEAD sentinel on engine shutdown/crash.

        Never raises: this runs on cleanup paths where the socket may already
        be unusable, and failing to notify must not mask the original error.
        """
        try:
            self._socket.send(ENGINE_CORE_DEAD, zmq.NOBLOCK)
        except Exception as exc:
            logger.warning(f"msgpack output: ENGINE_CORE_DEAD send failed: {exc!s}")

    def close(self) -> None:
        self._socket.close()


def _recv_init_with_timeout(socket: zmq.Socket) -> list[bytes]:
    """Wait for SMG's INIT reply, logging progress each poll slice so operators
    can tell a still-registering frontend from a wedged one."""
    waited_ms = 0
    while waited_ms < _INIT_TIMEOUT_MS:
        if socket.poll(_INIT_POLL_SLICE_MS, zmq.POLLIN):
            return socket.recv_multipart()
        waited_ms += _INIT_POLL_SLICE_MS
        logger.info(
            f"msgpack handshake: waiting for SMG INIT ({waited_ms // 1000:d}s elapsed)",
        )
    raise TimeoutError(
        f"SMG msgpack handshake timed out waiting for INIT after {_INIT_TIMEOUT_MS} ms"
    )


def connect_msgpack_engine(
    context: zmq.Context,
    handshake_address: str,
    engine_index: int,
    ready_response: "zmq_wire.WireEngineCoreReadyResponse",
    vocab_size: int,
    *,
    enable_output_logprobs: bool,
    supports_prompt_logprobs: bool,
    top_logprobs_unservable: str | None,
    request_types: Sequence[str],
) -> tuple[MsgpackRecvSocket, MsgpackSendSocket]:
    """Run the startup handshake against SMG and return the wrapped
    data-plane sockets.

    SMG binds ``handshake_address`` (ROUTER) plus the input (ROUTER) and output
    (PULL) sockets; this engine connects in and drives:

      1. HELLO on the handshake DEALER (identity = engine index).
      2. recv INIT -> learn SMG's input/output addresses.
      3. READY on the handshake DEALER.
      4. connect the input DEALER, register with the ready response, which
         declares the request types this role serves (``request_types``) and
         the output structs this transport can emit for them
         (``emitted_layouts``) beside the caller's own entries.
      5. connect the output PUSH.
    """
    identity = _engine_identity(engine_index)
    ready_response = msgspec.structs.replace(
        ready_response,
        positional_layouts={
            **emitted_layouts(request_types),
            **ready_response.positional_layouts,
        },
        request_types=list(request_types),
    )

    handshake = context.socket(zmq.DEALER)
    handshake.setsockopt(zmq.IDENTITY, identity)
    handshake.connect(handshake_address)
    logger.info(f"msgpack handshake: connected to {handshake_address!s}")

    handshake.send(
        zmq_wire.encode(
            zmq_wire.WireReadyMessage(status="HELLO", local=True, headless=True)
        )
    )
    init_frames = _recv_init_with_timeout(handshake)
    init = zmq_wire.decode_init(init_frames[-1])
    handshake.send(
        zmq_wire.encode(
            zmq_wire.WireReadyMessage(status="READY", local=True, headless=True)
        )
    )

    if not init.addresses.inputs or not init.addresses.outputs:
        raise ValueError(
            "SMG INIT message did not carry both input and output addresses: "
            f"inputs={init.addresses.inputs} outputs={init.addresses.outputs}"
        )
    input_address = init.addresses.inputs[0]
    output_address = init.addresses.outputs[0]
    logger.info(
        f"msgpack handshake: INIT input={input_address!s} output={output_address!s}",
    )

    input_socket = context.socket(zmq.DEALER)
    input_socket.setsockopt(zmq.IDENTITY, identity)
    input_socket.connect(input_address)
    # Register on the input socket so SMG learns this engine's post-init config.
    input_socket.send(zmq_wire.encode(ready_response))

    output_socket = context.socket(zmq.PUSH)
    output_socket.connect(output_address)

    handshake.close()
    logger.info(f"msgpack handshake: complete (engine_index={engine_index!s})")
    control = ControlCalls()
    return (
        MsgpackRecvSocket(
            input_socket,
            vocab_size,
            enable_output_logprobs=enable_output_logprobs,
            supports_prompt_logprobs=supports_prompt_logprobs,
            top_logprobs_unservable=top_logprobs_unservable,
            control=control,
            request_types=request_types,
        ),
        MsgpackSendSocket(output_socket, engine_index=engine_index, control=control),
    )


def _tokenspeed_version() -> str:
    try:
        from tokenspeed.version import __version__

        return __version__
    except Exception:
        return "unknown"


def connect_msgpack_engine_for_loop(
    context: zmq.Context, loop
) -> tuple[MsgpackRecvSocket, MsgpackSendSocket]:
    """Build the engine's ready response from an event loop's state and run
    the SMG startup handshake (see ``connect_msgpack_engine``).

    Args:
        context: The loop's ZMQ context.
        loop: The scheduler EventLoop; supplies the model/cache geometry and
            parallel layout the ready response reports to the frontend.

    Returns:
        The wrapped msgpack (input, output) sockets.

    Each DP rank dials SMG with its own engine identity (zmq_engine_index +
    dp_rank): the frontend's grouped worker awaits dp_size engines on one
    socket set and tells the ranks apart — and routes inputs back — by this
    index. The ready response carries the true dp rank/size alongside.
    """
    server_args = loop.server_args
    geometry = loop._scheduler_cache_geometry
    ready_response = zmq_wire.WireEngineCoreReadyResponse(
        max_model_len=loop.model_config.context_len,
        num_gpu_blocks=geometry.num_device_pages,
        prefix_granularity=geometry.prefix_granularity,
        dtype=zmq_wire.wire_dtype(loop.model_config.dtype),
        multimodal_encoder_dtype=loop.multimodal_encoder_dtype,
        vllm_version=f"tokenspeed-{_tokenspeed_version()}",
        world_size=loop.world_size,
        data_parallel_size=loop.dp_size,
        tensor_parallel_size=loop.attn_tp_size,
        decode_context_parallel_size=server_args.kv_parallel_size,
        data_parallel_rank=loop.dp_rank,
        max_num_seqs=server_args.max_num_seqs,
        # chunked_prefill_size=-1 means "disabled"; the wire field is a
        # non-negative integer for the frontend, so clamp to 0 (= no cap).
        max_num_batched_tokens=max(0, server_args.chunked_prefill_size),
        instance_id=server_args.served_model_name or server_args.model,
        kv_cache_size_tokens=loop.max_total_num_tokens,
    )
    return connect_msgpack_engine(
        context,
        server_args.zmq_handshake_endpoint(),
        server_args.zmq_engine_index + loop.dp_rank,
        ready_response,
        loop.model_config.vocab_size,
        enable_output_logprobs=server_args.enable_output_logprobs,
        supports_prompt_logprobs=loop.supports_prompt_logprobs,
        top_logprobs_unservable=top_logprobs_launch_refusal(server_args),
        request_types=SCHEDULER_REQUEST_TYPES,
    )
