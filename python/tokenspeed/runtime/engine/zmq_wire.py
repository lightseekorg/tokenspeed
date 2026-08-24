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
#   * the single-byte request-type frame (ADD/ABORT) that lets the receiver
#     dispatch without decoding the payload;
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

# Single-byte request-type frame: sent as a raw ZMQ frame ahead of the msgpack
# payload so the receiver can dispatch without decoding first.
REQ_TYPE_ADD = b"\x00"
REQ_TYPE_ABORT = b"\x01"


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
    decode_context_parallel_size: int = 1
    data_parallel_rank: int = 0
    max_num_seqs: int = 0
    max_num_batched_tokens: int = 0
    instance_id: str = ""
    kv_cache_size_tokens: int | None = None
    kv_cache_max_concurrency: float | None = None
    kv_events_config: dict | None = None


_ENC = msgspec.msgpack.Encoder()
_DEC_ABORT = msgspec.msgpack.Decoder(list[str])
_DEC_INIT = msgspec.msgpack.Decoder(WireHandshakeInitMessage)
# Native tagged request decode, tensor/aux-frame aware for multimodal payloads.
_DEC_ADD = MsgpackDecoder(TokenizedGenerateReqInput)


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
    """Encode a handshake struct to a msgpack payload."""
    return _ENC.encode(obj)


def decode_abort(payload: bytes) -> list[str]:
    """Decode an ABORT payload (a msgpack array of request ids)."""
    return _DEC_ABORT.decode(payload)


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
    an ADD always yields exactly one ``TokenizedGenerateReqInput``. ADD aux
    frames hold out-of-band tensor buffers (frame indices are relative to the
    payload, which is buffer 0).
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
    raise ValueError(f"unknown msgpack request type byte: {type_byte!r}")
