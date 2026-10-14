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

"""Engine sockets for direct generation alongside the existing control frontend."""

from __future__ import annotations

import collections
import logging
import struct
from collections.abc import Callable

import msgspec
import zmq

from tokenspeed.runtime.engine import zmq_wire
from tokenspeed.runtime.engine.io_struct import (
    AbortReq,
    BaseBatchReq,
    BatchTokenIDOut,
    BatchTokenIDOutSlim,
    MsgpackEncoder,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
)
from tokenspeed.runtime.engine.logprobs import resolve_logprob_start_len

logger = logging.getLogger(__name__)

# SMG allows a registering worker a long connect window (~600 s), so the
# frontend may start the handshake well after this engine finished model load.
# Wait for INIT in slices, logging progress, instead of dying at 60 s.
_INIT_TIMEOUT_MS = 600_000
_INIT_POLL_SLICE_MS = 30_000

# Raw single-frame sentinel SMG's output loop treats as terminal: the worker
# is marked dead instead of staying healthy-idle after this engine exits.
ENGINE_CORE_DEAD = b"ENGINE_CORE_DEAD"


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
    """

    def __init__(
        self,
        socket: zmq.Socket,
        vocab_size: int,
        *,
        enable_output_logprobs: bool,
        supports_prompt_logprobs: bool,
    ) -> None:
        self._socket = socket
        self._vocab_size = vocab_size
        self._enable_output_logprobs = enable_output_logprobs
        self._supports_prompt_logprobs = supports_prompt_logprobs
        self._pending: collections.deque = collections.deque()

    def _validation_error(self, obj: TokenizedGenerateReqInput) -> str | None:
        """Return the reason ``obj`` must be rejected, or None if it is valid.

        Mirrors the tokenizer-side ``InputProcessor`` gate for the SGLang
        logprob knobs. ``logprob_start_len`` is resolved in place (-1 -> last
        prompt token). Prompt scores require the engine's native capability;
        supported engines return nullable rows on the appended slim output tail.
        """
        if obj.return_logprob and not self._enable_output_logprobs:
            return (
                "logprobs were requested but the server was started without "
                "--enable-output-logprobs"
            )
        if obj.return_logprob:
            if obj.top_logprobs_num:
                return (
                    "top_logprobs_num > 0 (output top-k logprobs) is not supported yet"
                )
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
        try:
            obj.sampling_params.verify(self._vocab_size)
        except ValueError as exc:
            return str(exc)
        return None

    def recv_pyobj(self, flags: int = 0):
        malformed_count = 0
        while not self._pending:
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
                malformed_count += 1
                if malformed_count >= 32:
                    raise zmq.Again()
                continue
            for obj in decoded:
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
        self._socket.close(linger=0)


class MsgpackSendSocket:
    """Adapts a PUSH output socket to the ``send_pyobj`` contract.

    Only ``BatchTokenIDOut`` is on SMG's output path; it is sliced down to the
    tagged ``BatchTokenIDOutSlim`` (the frontend detokenizes itself, so the
    incremental-detokenization columns would be dead weight every step).
    Control replies (Profile/GetLoad/...) have no SMG consumer in this mode
    and are dropped with a warning rather than crashing the scheduler.
    """

    def __init__(
        self,
        socket: zmq.Socket,
        engine_index: int = 0,
        *,
        weight_version: Callable[[], str | None] | None = None,
        speculative_num_draft_tokens: int | None = None,
    ) -> None:
        self._socket = socket
        self._engine_index = engine_index
        self._weight_version = weight_version
        self._speculative_num_draft_tokens = speculative_num_draft_tokens
        self._encoder = MsgpackEncoder()
        self._load_snapshot = (0, 0, 0, 0)

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
                weight_version=(
                    self._weight_version() if self._weight_version is not None else None
                ),
                speculative_num_draft_tokens=self._speculative_num_draft_tokens,
                engine_index=self._engine_index,
                num_running=num_running,
                num_waiting=num_waiting,
                kv_active_pages=kv_active_pages,
                kv_total_pages=kv_total_pages,
            )
            self._socket.send_multipart(self._encoder.encode(slim), copy=False)
        else:
            logger.warning(
                "msgpack output: dropping unsupported control reply "
                f"{type(obj).__name__!s}",
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
        self._socket.close(linger=0)


def _recv_init_with_timeout(
    socket: zmq.Socket,
    *,
    shutdown_requested: Callable[[], bool] | None = None,
) -> list[bytes]:
    """Wait for SMG's INIT reply, logging progress each poll slice so operators
    can tell a still-registering frontend from a wedged one."""
    waited_ms = 0
    while waited_ms < _INIT_TIMEOUT_MS:
        if shutdown_requested is not None and shutdown_requested():
            raise InterruptedError("SMG msgpack handshake interrupted by shutdown")
        poll_ms = min(1000, _INIT_TIMEOUT_MS - waited_ms)
        if socket.poll(poll_ms, zmq.POLLIN):
            return socket.recv_multipart()
        waited_ms += poll_ms
        if waited_ms % _INIT_POLL_SLICE_MS != 0:
            continue
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
    weight_version: Callable[[], str | None] | None = None,
    speculative_num_draft_tokens: int | None = None,
    shutdown_requested: Callable[[], bool] | None = None,
) -> tuple[MsgpackRecvSocket, MsgpackSendSocket]:
    """Run the startup handshake against SMG and return the wrapped
    data-plane sockets.

    SMG binds ``handshake_address`` (ROUTER) plus the input (ROUTER) and output
    (PULL) sockets; this engine connects in and drives:

      1. HELLO on the handshake DEALER (identity = engine index).
      2. recv INIT -> learn SMG's input/output addresses.
      3. READY on the handshake DEALER.
      4. connect the input DEALER, register with the ready response.
      5. connect the output PUSH.
    """
    identity = _engine_identity(engine_index)
    handshake = context.socket(zmq.DEALER)
    input_socket: zmq.Socket | None = None
    output_socket: zmq.Socket | None = None
    handshake_completed = False
    try:
        handshake.setsockopt(zmq.IDENTITY, identity)
        handshake.connect(handshake_address)
        logger.info(f"msgpack handshake: connected to {handshake_address!s}")
        handshake.send(
            zmq_wire.encode(
                zmq_wire.WireReadyMessage(status="HELLO", local=True, headless=True)
            )
        )
        init_frames = _recv_init_with_timeout(
            handshake, shutdown_requested=shutdown_requested
        )
        init = zmq_wire.decode_init(init_frames[-1])
        if not init.addresses.inputs or not init.addresses.outputs:
            raise ValueError(
                "SMG INIT message did not carry both input and output addresses: "
                f"inputs={init.addresses.inputs} outputs={init.addresses.outputs}"
            )
        handshake.send(
            zmq_wire.encode(
                zmq_wire.WireReadyMessage(status="READY", local=True, headless=True)
            )
        )
        input_address = init.addresses.inputs[0]
        output_address = init.addresses.outputs[0]
        logger.info(
            f"msgpack handshake: INIT input={input_address!s} output={output_address!s}"
        )
        input_socket = context.socket(zmq.DEALER)
        input_socket.setsockopt(zmq.IDENTITY, identity)
        input_socket.connect(input_address)
        input_socket.send(zmq_wire.encode(ready_response))
        output_socket = context.socket(zmq.PUSH)
        # A dead frontend must not indefinitely block the scheduler at HWM.
        output_socket.setsockopt(zmq.SNDTIMEO, 1000)
        output_socket.connect(output_address)
        handshake_completed = True
        logger.info(f"msgpack handshake: complete (engine_index={engine_index!s})")
        return (
            MsgpackRecvSocket(
                input_socket,
                vocab_size,
                enable_output_logprobs=enable_output_logprobs,
                supports_prompt_logprobs=supports_prompt_logprobs,
            ),
            MsgpackSendSocket(
                output_socket,
                engine_index=engine_index,
                weight_version=weight_version,
                speculative_num_draft_tokens=speculative_num_draft_tokens,
            ),
        )
    except BaseException:
        if input_socket is not None:
            input_socket.close(linger=0)
        if output_socket is not None:
            output_socket.close(linger=0)
        raise
    finally:
        # READY may still be queued when the local send returns. Allow a
        # bounded flush on success; failed startup has no frames to preserve.
        handshake.close(linger=1000 if handshake_completed else 0)


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
        pipeline_parallel_size=loop.pp_size,
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
        weight_version=lambda: server_args.weight_version,
        speculative_num_draft_tokens=(
            server_args.speculative_num_draft_tokens
            if server_args.speculative_algorithm
            else None
        ),
        shutdown_requested=loop.shutdown_event.is_set,
    )


# Separate scheduler IDs let both frontends use the same public request ID.
# The prefixes are deterministic so PD peers retain the same request identity.
_CONTROL_RID_PREFIX = "\x00control:"
_DIRECT_RID_PREFIX = "\x00direct:"


class SharedFrontendReceiver:
    """Fair, bounded ingress from control IPC and direct generation sockets.

    Generation and cancellation IDs are namespaced before scheduler admission;
    the sender restores the original IDs. No per-request ownership table is
    retained, so cancellation and failed requests cannot leak routing state.
    ``recv_pyobj`` is nonblocking, as required by RequestHandler's round drain.
    ``max_requests_per_round`` bounds admission work even under continuous input.
    """

    def __init__(self, control, direct, *, max_requests_per_round: int) -> None:
        if max_requests_per_round <= 0:
            raise ValueError("max_requests_per_round must be positive")
        self._receivers = (control, direct)
        self._prefixes = (_CONTROL_RID_PREFIX, _DIRECT_RID_PREFIX)
        self._next_receiver = 0
        self._received = 0
        self._max_requests_per_round = max_requests_per_round

    def recv_pyobj(self, flags: int = 0):
        if not flags & zmq.NOBLOCK:
            raise ValueError("SharedFrontendReceiver requires zmq.NOBLOCK")
        if self._received == self._max_requests_per_round:
            self._received = 0
            raise zmq.Again()
        for offset in range(2):
            index = (self._next_receiver + offset) % 2
            try:
                obj = self._receivers[index].recv_pyobj(zmq.NOBLOCK)
            except zmq.Again:
                continue
            self._next_receiver = 1 - index
            self._received += 1
            if isinstance(
                obj, (TokenizedGenerateReqInput, TokenizedEmbeddingReqInput, AbortReq)
            ):
                prefix = self._prefixes[index]
                if isinstance(obj, AbortReq) and obj.admin_owner is not None:
                    if index != 0:
                        raise ValueError(
                            "Admin abort must enter through control ingress"
                        )
                    prefix = self._prefixes[0 if obj.admin_owner == "control" else 1]
                    obj = msgspec.structs.replace(obj, admin_owner=None)
                obj = msgspec.structs.replace(obj, rid=prefix + obj.rid)
                if (
                    isinstance(obj, TokenizedGenerateReqInput)
                    and obj.session_params is not None
                ):
                    reference = obj.session_params.rid
                    if reference is not None:
                        obj.session_params = msgspec.structs.replace(
                            obj.session_params, rid=prefix + reference
                        )
            return obj
        self._received = 0
        raise zmq.Again()

    def close(self) -> None:
        for receiver in self._receivers:
            receiver.close()


class SharedFrontendSender:
    """Return generation rows to their owner and control replies to AsyncLLM.

    Output columns retain their original order and empty optional columns.
    Splitting creates new structs, leaving the scheduler's batch untouched.
    """

    def __init__(self, control, direct: MsgpackSendSocket) -> None:
        self._control = control
        self._direct = direct

    def send_pyobj(self, obj) -> None:
        if not isinstance(obj, BatchTokenIDOut):
            if isinstance(obj, BaseBatchReq) and obj.rids is not None:
                if any(not rid.startswith(_CONTROL_RID_PREFIX) for rid in obj.rids):
                    raise ValueError(
                        "Non-token output requires a control frontend owner"
                    )
                obj = msgspec.structs.replace(
                    obj, rids=[rid[len(_CONTROL_RID_PREFIX) :] for rid in obj.rids]
                )
            self._control.send_pyobj(obj)
            return
        owners = ([], [])
        public_ids = []
        for index, rid in enumerate(obj.rids):
            if rid.startswith(_CONTROL_RID_PREFIX):
                owners[0].append(index)
                public_ids.append(rid[len(_CONTROL_RID_PREFIX) :])
            elif rid.startswith(_DIRECT_RID_PREFIX):
                owners[1].append(index)
                public_ids.append(rid[len(_DIRECT_RID_PREFIX) :])
            else:
                raise ValueError(f"Generation output has no frontend owner: {rid!r}")
        for sender, indices in zip((self._control, self._direct), owners, strict=True):
            if not indices:
                continue
            columns = {}
            for field in msgspec.structs.fields(type(obj)):
                value = getattr(obj, field.name)
                if isinstance(value, list) and len(value) == len(obj.rids):
                    value = [value[index] for index in indices]
                columns[field.name] = value
            columns["rids"] = [public_ids[index] for index in indices]
            sender.send_pyobj(type(obj)(**columns))

    def set_load_snapshot(
        self,
        num_running: int,
        num_waiting: int,
        kv_active_pages: int,
        kv_total_pages: int,
    ) -> None:
        self._direct.set_load_snapshot(
            num_running, num_waiting, kv_active_pages, kv_total_pages
        )

    def send_engine_dead(self) -> None:
        self._direct.send_engine_dead()

    def close(self) -> None:
        self._control.close(linger=0)
        self._direct.close()
