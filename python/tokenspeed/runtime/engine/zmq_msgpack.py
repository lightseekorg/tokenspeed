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
# differs. This module is additive and only reached when ``--zmq-msgpack`` is on.

from __future__ import annotations

import collections
import logging
import struct

import zmq

from tokenspeed.runtime.engine import zmq_wire
from tokenspeed.runtime.engine.io_struct import (
    BatchTokenIDOut,
    BatchTokenIDOutSlim,
    MsgpackEncoder,
    TokenizedGenerateReqInput,
)

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
    generate request, and ``enable_output_logprobs`` gates ``return_logprob``
    (the pickle path enforces both in the tokenizer_manager's input processor,
    which the msgpack path bypasses; the pure wire ``to_io_struct`` has neither
    value). An invalid request is not dropped — that would hang the frontend's
    stream with no terminal frame. It is marked via ``validation_error`` so
    RequestHandler admits it pre-finished (FINISH_ABORT) and OutputProcesser
    emits a terminal "abort" output.
    """

    def __init__(
        self,
        socket: zmq.Socket,
        vocab_size: int,
        enable_output_logprobs: bool = False,
    ) -> None:
        self._socket = socket
        self._vocab_size = vocab_size
        self._enable_output_logprobs = enable_output_logprobs
        self._pending: collections.deque = collections.deque()

    def _validation_error(self, obj: TokenizedGenerateReqInput) -> str | None:
        """Return the reason ``obj`` must be rejected, or None if it is valid."""
        if obj.return_logprob and not self._enable_output_logprobs:
            return (
                "logprobs were requested but the server was started without "
                "--enable-output-logprobs"
            )
        try:
            obj.sampling_params.verify(self._vocab_size)
        except ValueError as exc:
            return str(exc)
        return None

    def recv_pyobj(self, flags: int = 0):
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
                    "(%d frames, sizes=%s, head=%s): %s",
                    len(frames),
                    [len(frame) for frame in frames],
                    frames[0][:8].hex() if frames else "",
                    exc,
                )
                continue
            for obj in decoded:
                # AbortReq carries no sampling_params; only generate requests
                # are validated. A rejected request still flows, pre-marked.
                # The wire layer may have marked it already (e.g. n != 1).
                if isinstance(obj, TokenizedGenerateReqInput):
                    reason = obj.validation_error or self._validation_error(obj)
                    if reason is not None:
                        logger.warning(
                            "msgpack input: aborting invalid request %s: %s",
                            obj.rid,
                            reason,
                        )
                        obj.validation_error = reason
                self._pending.append(obj)
        return self._pending.popleft()

    def close(self) -> None:
        self._socket.close()


class MsgpackSendSocket:
    """Adapts a PUSH output socket to the ``send_pyobj`` contract.

    Only ``BatchTokenIDOut`` is on SMG's output path; it is sliced down to the
    tagged ``BatchTokenIDOutSlim`` (the frontend detokenizes itself, so the
    incremental-detokenization columns would be dead weight every step).
    Control replies (Profile/GetLoad/...) have no SMG consumer in this mode
    and are dropped with a warning rather than crashing the scheduler.
    """

    def __init__(self, socket: zmq.Socket, engine_index: int = 0) -> None:
        self._socket = socket
        self._engine_index = engine_index
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
                engine_index=self._engine_index,
                num_running=num_running,
                num_waiting=num_waiting,
                kv_active_pages=kv_active_pages,
                kv_total_pages=kv_total_pages,
            )
            self._socket.send_multipart(self._encoder.encode(slim), copy=False)
        else:
            logger.warning(
                "msgpack output: dropping unsupported control reply %s",
                type(obj).__name__,
            )

    def send_engine_dead(self) -> None:
        """Best-effort ENGINE_CORE_DEAD sentinel on engine shutdown/crash.

        Never raises: this runs on cleanup paths where the socket may already
        be unusable, and failing to notify must not mask the original error.
        """
        try:
            self._socket.send(ENGINE_CORE_DEAD, zmq.NOBLOCK)
        except Exception as exc:
            logger.warning("msgpack output: ENGINE_CORE_DEAD send failed: %s", exc)

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
            "msgpack handshake: waiting for SMG INIT (%ds elapsed)",
            waited_ms // 1000,
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
    enable_output_logprobs: bool = False,
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
    handshake.setsockopt(zmq.IDENTITY, identity)
    handshake.connect(handshake_address)
    logger.info("msgpack handshake: connected to %s", handshake_address)

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
        "msgpack handshake: INIT input=%s output=%s", input_address, output_address
    )

    input_socket = context.socket(zmq.DEALER)
    input_socket.setsockopt(zmq.IDENTITY, identity)
    input_socket.connect(input_address)
    # Register on the input socket so SMG learns this engine's post-init config.
    input_socket.send(zmq_wire.encode(ready_response))

    output_socket = context.socket(zmq.PUSH)
    output_socket.connect(output_address)

    handshake.close()
    logger.info("msgpack handshake: complete (engine_index=%s)", engine_index)
    return (
        MsgpackRecvSocket(input_socket, vocab_size, enable_output_logprobs),
        MsgpackSendSocket(output_socket, engine_index=engine_index),
    )
