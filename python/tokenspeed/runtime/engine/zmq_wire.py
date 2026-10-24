# Copyright (c) 2026 LightSeek Foundation
#
# msgpack wire format for the direct SMG <-> scheduler ZMQ path.
#
# The data plane carries the NATIVE tagged io_struct types (see io_struct):
# a request frame decodes straight into ``TokenizedGenerateReqInput`` and a
# per-step output is the slim ``BatchTokenIDOutSlim`` slice, so there is no
# duplicate wire-struct layer to keep in sync. This module keeps only the
# transport-level pieces of the contract:
#
#   * the single-byte request-type frame (ADD/ABORT/CONTROL/ENCODE) that lets
#     the receiver dispatch without decoding the payload;
#   * the control call ``[call_id, method, args]`` behind the CONTROL byte and
#     the tagged ``ControlReplySlim`` that answers it on the output socket;
#   * the map-encoded startup-handshake structs (HELLO/INIT/READY plus the
#     engine's ready response), which match the frontend's
#     ``rmp_serde::to_vec_named`` codec and must NOT be positional.

from __future__ import annotations

import msgspec

from tokenspeed.runtime.engine.io_struct import (
    AbortReq,
    MsgpackDecoder,
    TokenizedGenerateReqInput,
)
from tokenspeed.runtime.epd.encode_worker import EncodeRequest

# Single-byte request-type frame: sent as a raw ZMQ frame ahead of the msgpack
# payload so the receiver can dispatch without decoding first.
REQ_TYPE_ADD = b"\x00"
REQ_TYPE_ABORT = b"\x01"
# A control call for the scheduler's own control requests (a cache flush, the
# profiler's start and stop): the payload is the msgpack array ``[call_id,
# method, args]`` and the answer rides the output socket as a
# ``ControlReplySlim`` under the same call id. The byte and both shapes are
# the ones the frontend already speaks to its SGLang plugin, so one encoder and
# one reply reader drive either scheduler.
REQ_TYPE_CONTROL = b"\x02"
# An encode request for the EPD encode role: the payload is the tagged
# positional ``EncodeRequest`` (``runtime/epd/encode_worker.py``) with its
# items' tensors as ext-encoded buffers or aux frames, exactly as an ADD's
# multimodal payload rides. No reply: the embedding reaches the prefill peer
# over the transfer engine.
REQ_TYPE_ENCODE = b"\x03"

# The request types by name, as an engine role lists the ones it serves in its
# ready response (``WireEngineCoreReadyResponse.request_types``) and as the
# transport tells a decoded request apart (``request_kind``).
REQUEST_KIND_ADD = "add"
REQUEST_KIND_ABORT = "abort"
REQUEST_KIND_CONTROL = "control"
REQUEST_KIND_ENCODE = "encode"

# The rid of the scheduler request a control call becomes: the frontend's call
# id behind a prefix no generate request uses, so the request's output is
# answered under that id (``control_call_id``).
CONTROL_RID_PREFIX = "smg-control-"


# ----------------------------------------------------------------------------
# Startup-handshake structs (SMG <-> engine), encoded as msgpack MAPS with named
# keys (msgspec's default struct encoding), matching the Rust
# `rmp_serde::to_vec_named` codec. Do NOT make these ``array_like``; only the
# request/output data-plane structs are positional tuples.
# ----------------------------------------------------------------------------


class WireReadyMessage(msgspec.Struct):
    """Engine -> frontend handshake status frame (``status`` = HELLO | READY)."""

    status: str | None = None
    local: bool | None = None
    headless: bool | None = None
    parallel_config_hash: str | None = None


class WireHandshakeAddresses(msgspec.Struct):
    """Frontend-owned data-plane addresses delivered to the engine in INIT."""

    inputs: list[str] = []
    outputs: list[str] = []
    coordinator_input: str | None = None
    coordinator_output: str | None = None
    frontend_stats_publish_address: str | None = None


class WireHandshakeInitMessage(msgspec.Struct):
    """Frontend -> engine INIT payload (sent in reply to HELLO)."""

    addresses: WireHandshakeAddresses
    # Opaque to the engine; decoded loosely so unknown keys are ignored.
    parallel_config: dict = {}


class WireEngineCoreReadyResponse(msgspec.Struct):
    """Engine -> frontend post-init config, sent on the input-socket
    registration once the HELLO/INIT/READY handshake completes.

    Field names (not order) are the wire contract for this map-encoded struct.
    """

    max_model_len: int = 0
    num_gpu_blocks: int = 0
    prefix_granularity: int = 0
    dp_stats_address: str | None = None
    dtype: str = "bfloat16"
    multimodal_encoder_dtype: str | None = None
    vllm_version: str = ""
    world_size: int = 1
    data_parallel_size: int = 1
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    # The frontend's key for KV parallelism (``--kv-parallel-size``); the
    # wire name is the frontend's contract, not this project's vocabulary.
    decode_context_parallel_size: int = 1
    data_parallel_rank: int = 0
    max_num_seqs: int = 0
    max_num_batched_tokens: int = 0
    instance_id: str = ""
    kv_cache_size_tokens: int | None = None
    kv_cache_max_concurrency: float | None = None
    kv_events_config: dict | None = None
    # The field order of the positional (``array_like``) structs this engine
    # can emit on the output socket, by tag: the frontend checks its decoders
    # here (the control reply's tuple, an appended output column) with no
    # version string to compare. Empty from an engine that predates it.
    positional_layouts: dict[str, list[str]] = {}
    # The request types this engine role serves (``REQUEST_KIND_*``), the
    # signal for what a connection takes, control calls included: the
    # scheduler loop lists add, abort and control, the EPD encode loop lists
    # encode. Empty from an engine that predates it, which serves add and
    # abort (and control when ``positional_layouts`` names the reply).
    request_types: list[str] = []
    # Whether this engine fills a multimodal request's MRoPE position slots
    # (``multimodal_inputs`` positions, delta, delta scalar) with its own
    # position function when the frontend sends them empty. False from an
    # engine that predates it: that engine runs an MRoPE model on linear
    # positions when the slots arrive empty, so a frontend fills them itself
    # or refuses the request.
    computes_mrope_positions: bool = False


# ----------------------------------------------------------------------------
# Control structs of the data plane: the call is the input socket's CONTROL
# payload (a plain msgpack array, decoded by ``decode_control``), the reply a
# positional (``array_like``) tagged tuple on the output socket, like the
# output batches. The handshake banner above does not apply to them.
# ----------------------------------------------------------------------------


class ControlCall(msgspec.Struct, kw_only=True):
    """One control call off the input socket: ``[call_id, method, args]``.

    ``method`` is ``flush_cache`` (args ``[timeout_s]``), ``start_profile``
    (args ``[options]``, a map of the profiler's start options) or
    ``stop_profile`` (no args). The transport turns the call into the
    scheduler's own control request and answers its output under the call id
    (``zmq_msgpack.ControlCalls``).
    """

    call_id: int
    method: str
    args: list


class ControlReplySlim(msgspec.Struct, tag=True, array_like=True):
    """The answer to a control call, on the output socket:
    ``["ControlReplySlim", call_id, success, message, engine_index]``.

    ``message`` is None for an accepted cache flush and the scheduler's own
    words otherwise; ``engine_index`` names the answering rank as the output
    batches do (the PULL side carries no routing identity).
    """

    call_id: int
    success: bool
    message: str | None = None
    engine_index: int = 0


def control_rid(call_id: int) -> str:
    """The rid the scheduler request of control call ``call_id`` carries."""
    return f"{CONTROL_RID_PREFIX}{call_id:d}"


def control_call_id(rid: object) -> int | None:
    """The call id a control request's rid carries; None for any other rid."""
    if not isinstance(rid, str) or not rid.startswith(CONTROL_RID_PREFIX):
        return None
    try:
        return int(rid[len(CONTROL_RID_PREFIX) :])
    except ValueError:
        return None


def request_kind(request: object) -> str:
    """The request type (``REQUEST_KIND_*``) a decoded request belongs to."""
    if isinstance(request, TokenizedGenerateReqInput):
        return REQUEST_KIND_ADD
    if isinstance(request, AbortReq):
        return REQUEST_KIND_ABORT
    if isinstance(request, ControlCall):
        return REQUEST_KIND_CONTROL
    if isinstance(request, EncodeRequest):
        return REQUEST_KIND_ENCODE
    raise TypeError(f"not a request of the msgpack wire: {type(request).__name__}")


def wire_tag(struct_type: type[msgspec.Struct]) -> str:
    """The tag string a tagged positional struct carries at element 0."""
    return struct_type.__struct_config__.tag


def positional_layout(struct_type: type[msgspec.Struct]) -> list[str]:
    """The field names of a positional struct in wire order, tag excluded."""
    return [field.name for field in msgspec.structs.fields(struct_type)]


_ENC = msgspec.msgpack.Encoder()
_DEC_ABORT = msgspec.msgpack.Decoder(list[str])
_DEC_CONTROL = msgspec.msgpack.Decoder(tuple[int, str, list])
_DEC_INIT = msgspec.msgpack.Decoder(WireHandshakeInitMessage)
# Native tagged request decode, tensor/aux-frame aware for multimodal payloads.
_DEC_ADD = MsgpackDecoder(TokenizedGenerateReqInput)
_DEC_ENCODE = MsgpackDecoder(EncodeRequest)


# SMG decodes the ready response's dtype into a fixed enum of these strings, so
# map tokenspeed's dtype onto the nearest one.
_WIRE_DTYPE_MAP = {
    "bfloat16": "bfloat16",
    "bf16": "bfloat16",
    "float16": "float16",
    "half": "float16",
    "fp16": "float16",
    "float32": "float32",
    "float": "float32",
    "fp32": "float32",
}


def wire_dtype(dtype) -> str:
    """Map a tokenspeed/torch dtype onto SMG's wire dtype enum string.

    Args:
        dtype: A torch dtype or its string form (e.g. ``torch.bfloat16``,
            ``"bf16"``).

    Returns:
        The wire enum string (``"bfloat16"`` / ``"float16"`` / ``"float32"``).

    Raises:
        ValueError: for unmapped dtypes — fail at handshake time: misreporting
            the dtype to the frontend is worse than refusing to start.
    """
    key = str(dtype).lower().replace("torch.", "")
    mapped = _WIRE_DTYPE_MAP.get(key)
    if mapped is None:
        raise ValueError(
            f"dtype {dtype!r} has no SMG wire mapping; extend _WIRE_DTYPE_MAP"
        )
    return mapped


def encode(obj: msgspec.Struct) -> bytes:
    """Encode a wire struct to a msgpack payload: a handshake map, or a
    positional data-plane struct such as ``ControlReplySlim``."""
    return _ENC.encode(obj)


def decode_abort(payload: bytes) -> list[str]:
    """Decode an ABORT payload (a msgpack array of request ids)."""
    return _DEC_ABORT.decode(payload)


def decode_control(payload: bytes) -> ControlCall:
    """Decode a CONTROL payload (the msgpack array ``[call_id, method, args]``)."""
    call_id, method, args = _DEC_CONTROL.decode(payload)
    return ControlCall(call_id=call_id, method=method, args=args)


def decode_init(payload: bytes) -> WireHandshakeInitMessage:
    return _DEC_INIT.decode(payload)


def _finalize_generate_request(io: TokenizedGenerateReqInput) -> None:
    """Run the request-materialization steps the in-process input processor
    performs on the pickle-era path, which this transport bypasses.

    ``resolve_seed`` derives a None seed deterministically from the rid so all
    TP/DP ranks agree on it; ``normalize`` fills the scheduler-required
    defaults (e.g. stop_strs). The frontend sends token ids and stop_token_ids
    (never stop strings), so no tokenizer is needed. ``verify(vocab_size)``
    is also part of that pipeline, but it needs vocab_size, which this pure
    wire layer does not have; it runs in the transport (MsgpackRecvSocket).
    """
    sampling_params = io.sampling_params
    sampling_params.resolve_seed(io.rid)
    sampling_params.normalize(tokenizer=None)
    # The engine does not multiplex one request into n completions (the
    # frontend fans out n > 1 itself and rejects it on its side). Mark the
    # request instead of raising so it terminates with an abort output
    # rather than being dropped as a malformed frame.
    if sampling_params.n != 1:
        io.validation_error = (
            f"n={sampling_params.n} is not supported on the msgpack "
            "wire; the frontend must fan out n > 1 itself"
        )


def decode_request_frames(frames: list[bytes]) -> list:
    """Decode a ``[type_byte, payload, *aux]`` request into io_structs.

    Returns a list because a single ABORT frame may carry several request ids;
    an ADD always yields exactly one ``TokenizedGenerateReqInput``, a CONTROL
    exactly one ``ControlCall`` and an ENCODE exactly one ``EncodeRequest``.
    ADD and ENCODE aux frames hold out-of-band tensor buffers (frame indices
    are relative to the payload, which is buffer 0).
    """
    if len(frames) < 2:
        raise ValueError(
            f"msgpack request needs >=2 frames (type, payload), got {len(frames)}"
        )
    type_byte = frames[0]
    if type_byte == REQ_TYPE_ADD:
        io = _DEC_ADD.decode(frames[1:])
        _finalize_generate_request(io)
        return [io]
    if type_byte == REQ_TYPE_ABORT:
        return [AbortReq(rid=rid) for rid in decode_abort(frames[1])]
    if type_byte == REQ_TYPE_CONTROL:
        return [decode_control(frames[1])]
    if type_byte == REQ_TYPE_ENCODE:
        return [_DEC_ENCODE.decode(frames[1:])]
    raise ValueError(f"unknown msgpack request type byte: {type_byte!r}")
