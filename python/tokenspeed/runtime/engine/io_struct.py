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

"""
The messages of the engine's msgpack ZMQ wire, plus the msgpack codec that
carries them.

Every message is a ``msgspec.Struct`` deriving from ``BaseReq`` /
``BaseBatchReq`` (tagged, keyword-only, array-encoded): the engine host's
Rust half (``ts-wire``) shares the schema, and these structs are the source
of its conformance fixtures. Tensors ride as msgpack ext-typed raw buffers
with large payloads moved to out-of-band ZMQ frames (see
``MsgpackEncoder``).
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, Literal

import msgspec
import numpy as np
import torch

from tokenspeed.runtime.multimodal.inputs import MultimodalInputs
from tokenspeed.runtime.sampling.sampling_params import SamplingParams


class BaseReq(
    msgspec.Struct, tag=True, tag_field="_tag", kw_only=True, array_like=True
):
    """Base for single-request IPC payloads.

    ``tag=True`` prefixes the class name so a tagged-union decoder can
    dispatch on it; ``array_like=True`` encodes fields
    positionally (declaration order IS the wire contract — append only).
    ``tag_field`` is renamed off the default so subclasses may declare a
    field literally named ``type`` (it never appears in the array encoding).
    """

    rid: str | list[str] | None = None
    http_worker_ipc: str | None = None

    def regenerate_rid(self):
        """Generate a new request ID and return it."""
        if isinstance(self.rid, list):
            self.rid = [uuid.uuid4().hex for _ in range(len(self.rid))]
        else:
            self.rid = uuid.uuid4().hex
        return self.rid


class BaseBatchReq(
    msgspec.Struct, tag=True, tag_field="_tag", kw_only=True, array_like=True
):
    """Base for batched IPC payloads (parallel per-request columns)."""

    rids: list[str] | None = None


class SessionParams(msgspec.Struct, kw_only=True, array_like=True):
    id: str | None = None
    rid: str | None = None
    offset: int | None = None
    replace: bool | None = None


class TokenizedGenerateReqInput(BaseReq, kw_only=True):
    # The input text (None on the token-id-only and input-embeds paths)
    input_text: str | None = None
    # The input token ids (None on the input-embeds path)
    input_ids: list[int] | None = None
    # The sampling parameters
    sampling_params: SamplingParams
    # Whether to return the sampled token's logprob for this request.
    return_logprob: bool = False
    # SGLang dialect: prompt (input) logprobs are returned for positions
    # ``[logprob_start_len, len(input_ids))``; -1 selects the last prompt
    # token only (no prompt logits needed). The ingress resolves -1 and
    # rejects starts past the prompt. Top-k logprobs (``top_logprobs_num``)
    # are taken only for a request whose tokens are all sampled in its
    # prefill; token-id logprobs are not supported: the ingress refuses
    # token_ids_logprob.
    logprob_start_len: int = -1
    top_logprobs_num: int = 0
    token_ids_logprob: list[int] | None = None
    # Whether to stream output
    stream: bool = False

    # The input embeds (nested float lists; shape varies by caller)
    input_embeds: list | None = None

    # Session info for continual prompting
    session_params: SessionParams | None = None

    # Custom logit processor for advanced sampling control. Must be a serialized instance
    # of `CustomLogitProcessor` in python/tokenspeed/runtime/sampling/custom_logit_processor.py
    # Use the processor's `to_str()` method to generate the serialized string.
    custom_logit_processor: str | None = None

    # Whether to return hidden states
    return_hidden_states: bool = False

    # Time at object instantiated
    created_time: float = 0.0

    # For disaggregated inference
    bootstrap_host: str | None = None
    bootstrap_port: int | None = None
    bootstrap_room: int | None = None

    input_multi_ids: list[list[int]] | None = None
    input_extra_infos: list[dict] | None = None
    # Original prompt ids before multimodal pad/hash replacement. The scheduler
    # uses input_ids, while detokenization must use these tokenizer-valid ids.
    input_ids_unpadded: list[int] | None = None
    # Typed multimodal payload; tensors ride as ext-encoded raw buffers or
    # out-of-band frames.
    multimodal_inputs: MultimodalInputs | None = None

    # Set by the wire decode when it finds the request invalid. A non-None
    # value makes the engine admit the request pre-finished with an abort so
    # the client receives a terminal abort instead of a silent drop.
    validation_error: str | None = None

    # Hard-pin to an attention-DP rank. Keep last: array_like declaration
    # order is the wire contract.
    data_parallel_rank: int | None = None


# Serialized form of BaseFinishReason.to_json() (or None while streaming) —
# all values are msgpack-native primitives.
FinishReasonDict = dict


class BatchTokenIDOutSlim(BaseBatchReq, kw_only=True):
    """One step's outputs as an engine sends them to its frontend, which
    detokenizes itself (the msgpack wire).

    The engine (``ts-wire``) encodes them; this struct is their schema as a
    Python consumer decodes them, and the source of ``ts-wire``'s conformance
    fixtures. Field ORDER is a cross-language wire contract — do not
    reorder.
    """

    # Newly generated token ids per request this step.
    output_ids: list[list[int]]
    # "" (not finished) or the finish-reason type ("stop", "length", "abort").
    finished_reasons: list[str]
    prompt_tokens: list[int]
    completion_tokens: list[int]
    cached_tokens: list[int]
    # Sampled-token logprobs, parallel to rids: one inner list per request,
    # holding the value/token-id of each newly-decoded token this step. Empty []
    # for a request that did not ask for logprobs, so the columns stay
    # non-ragged (always length == len(rids)). Prompt logprobs ride the tail.
    output_token_logprobs_val: list[list[float]]
    output_token_logprobs_idx: list[list[int]]
    # Producing DP rank's engine index (the identity it dialed the frontend
    # with). The output PULL socket carries no routing identity, so under
    # DP the batch itself names its rank. Appended field: defaults to 0 so
    # older peers on either side stay compatible.
    engine_index: int = 0
    # Latest scheduler load observed before the output is forwarded. Appended
    # wire tail: older 9- and 10-element prefixes decode all four values as 0.
    num_running: int = 0
    num_waiting: int = 0
    kv_active_pages: int = 0
    kv_total_pages: int = 0
    # Appended wire tail: what a frontend renders beyond the tokens. An
    # older producer omits these and they decode as [].
    # The finish reason as ``BaseFinishReason.to_json()`` gives it (the
    # matched token, the abort's message and codes), None mid-stream; one
    # entry per request.
    finish_reason_details: list[FinishReasonDict | None] = []
    # With speculative decoding, one entry per request: its verify rounds
    # so far and, once finished, its accepted tokens per round (None before);
    # [] without speculation.
    spec_verify_ct: list[int] = []
    accept_draft_tokens: list[float | None] = []
    # Prompt logprobs, shipped once per request (on the first frame after
    # the prompt finished), and the top-k alternatives (``top_logprobs_num``). Each column has one
    # entry per request in a frame where some request carries one, and is []
    # for the whole frame otherwise.
    input_token_logprobs_val: list[list[float | None]] = []
    input_token_logprobs_idx: list[list[int]] = []
    input_top_logprobs_val: list[list] = []
    input_top_logprobs_idx: list[list] = []
    output_top_logprobs_val: list[list] = []
    output_top_logprobs_idx: list[list] = []


class FlushCacheReqInput(BaseReq, kw_only=True):
    pass


class FlushCacheReqOutput(BaseReq, kw_only=True):
    success: bool


# How a pause should treat in-flight requests.
# - "abort": kill in-flight requests immediately, then stop admitting new ones.
# - "wait":  stop admitting new ones, keep stepping until running requests drain.
# - "keep":  freeze everything in place; resume picks up where it left off.
PauseMode = Literal["abort", "wait", "keep"]


class PauseSchedulerReqInput(BaseReq, kw_only=True):
    # See PauseMode for how each mode treats in-flight requests.
    mode: PauseMode = "abort"


class PauseSchedulerReqOutput(BaseReq, kw_only=True):
    success: bool
    message: str = ""


class ResumeSchedulerReqInput(BaseReq, kw_only=True):
    pass


class ResumeSchedulerReqOutput(BaseReq, kw_only=True):
    success: bool
    message: str = ""


class IsSchedulerPausedReqInput(BaseReq, kw_only=True):
    pass


class IsSchedulerPausedReqOutput(BaseReq, kw_only=True):
    is_paused: bool


class UpdateWeightFromDiskReqInput(BaseReq, kw_only=True):
    # The model path with the new weights
    model_path: str
    # The format to load the weights
    load_format: str | None = None
    # Optional: update the weight version after a successful load.
    weight_version: str | None = None
    # Drop cached KV before loading: it was computed under the old weights.
    flush_cache: bool = True


class UpdateWeightFromDiskReqOutput(BaseReq, kw_only=True):
    success: bool
    message: str
    # Number of paused requests during weight sync.
    num_paused_requests: int | None = 0


class UpdateWeightsFromDistributedReqInput(BaseReq, kw_only=True):
    # Weight-update metadata shared with the trainer's NCCL sender.
    names: list[str]
    dtype_names: list[str]
    shapes: list[list[int]]
    group_name: str = "weight_update_group"
    flush_cache: bool = True
    # Required. Pass ``None`` to keep the current namespace. Flushed L3
    # updates must supply a shared checkpoint identity.
    weight_version: str | None


class UpdateWeightsFromDistributedReqOutput(BaseReq, kw_only=True):
    success: bool
    message: str


class UpdateWeightsFromTensorReqInput(BaseReq, kw_only=True):
    # One serialized ``Dict[str, torch.Tensor]`` per world rank (engine.py fans
    # the payload out across ``mapping.world_size``).
    serialized_named_tensors: list[bytes]
    load_format: str | None
    flush_cache: bool
    # Optional: update the weight version after a successful push.
    weight_version: str | None = None


class UpdateWeightsFromTensorReqOutput(BaseReq, kw_only=True):
    success: bool
    message: str


class InitWeightsUpdateGroupReqInput(BaseReq, kw_only=True):
    # The master address
    master_address: str
    # The master port
    master_port: int
    # The rank offset
    rank_offset: int
    # The world size
    world_size: int
    # The group name
    group_name: str = "weight_update_group"
    # The backend
    backend: str = "nccl"


class InitWeightsUpdateGroupReqOutput(BaseReq, kw_only=True):
    success: bool
    message: str


class DestroyWeightsUpdateGroupReqInput(BaseReq, kw_only=True):
    # The group name to tear down (must match the init group_name).
    group_name: str = "weight_update_group"


class DestroyWeightsUpdateGroupReqOutput(BaseReq, kw_only=True):
    success: bool
    message: str


class UpdateWeightsFromMooncakeReqInput(BaseReq, kw_only=True):
    """Load one committed checkpoint version through the Model Updater SDK.

    The trainer publishes weights to a Mooncake weight store and names the
    version to serve; every scheduler reads its own shard. Requires the
    server to be started with ``--model-update-config``.
    """

    # The committed weight-store version to load.
    version: int
    # Required: whether Device/Host KV is flushed before the load.
    flush_cache: bool
    # Required. The L3 namespace to publish after a successful load, or
    # ``None`` for the default ``mooncake_load_weight_version`` resolves.
    weight_version: str | None


class UpdateWeightsFromMooncakeReqOutput(BaseReq, kw_only=True):
    success: bool
    message: str


class RebalanceExpertsReqInput(BaseReq, kw_only=True):
    """Start one online expert rebalance now (``--enable-eplb``).

    Takes the load snapshot the periodic trigger would take at its next
    interval; the commit and the per-chunk weight moves follow on the same
    schedule. Refused while a rebalance is already in progress.
    """


class RebalanceExpertsReqOutput(BaseReq, kw_only=True):
    success: bool
    message: str


class ReleaseMemoryOccupationReqInput(BaseReq, kw_only=True):
    # Memory regions to release. None ⇒ all ("weights" and "kv_cache").
    tags: list[str] | None = None


class ReleaseMemoryOccupationReqOutput(BaseReq, kw_only=True):
    success: bool = True
    message: str = ""


class ResumeMemoryOccupationReqInput(BaseReq, kw_only=True):
    # Memory regions to resume. None ⇒ all previously released tags.
    tags: list[str] | None = None


class ResumeMemoryOccupationReqOutput(BaseReq, kw_only=True):
    success: bool = True
    message: str = ""


class IsSleepingReqInput(BaseReq, kw_only=True):
    pass


class IsSleepingReqOutput(BaseReq, kw_only=True):
    is_sleeping: bool


class AbortReq(BaseReq, kw_only=True):
    # The request id rides in the ``rid`` base field.
    pass


class GetInternalStateReq(BaseReq, kw_only=True):
    pass


class GetInternalStateReqOutput(BaseReq, kw_only=True):
    internal_state: dict[str, Any]


class SetInternalStateReq(BaseReq, kw_only=True):
    server_args: dict[str, Any]


class SetInternalStateReqOutput(BaseReq, kw_only=True):
    updated: bool
    server_args: dict[str, Any]


class ProfileReqType(Enum):
    START_PROFILE = 1
    STOP_PROFILE = 2


class ProfileReq(BaseReq, kw_only=True):
    type: ProfileReqType
    output_dir: str | None = None
    start_step: int | None = None
    num_steps: int | None = None
    activities: list[str] | None = None
    profile_by_stage: bool = False
    with_stack: bool | None = None
    record_shapes: bool | None = None
    profile_id: str | None = None


class ProfileReqOutput(BaseReq, kw_only=True):
    success: bool
    message: str


# ============================================================================
# msgpack codec for engine IPC.
#
# Tensor scheme (shared with the SMG Rust engine-side codec): a tensor or
# ndarray encodes as the tuple ``(dtype, shape, data)`` where ``data`` is
# either an ext-typed raw byte view (``CUSTOM_TYPE_RAW_VIEW``) for small
# payloads, or an integer index into the message's out-of-band ZMQ frames
# (frame 0 is the primary msgpack buffer, so indices are one-based).
# ============================================================================

CUSTOM_TYPE_RAW_VIEW = 3

# Tensors/ndarrays below this many bytes are inlined in the primary buffer;
# larger ones become dedicated zero-copy frames.
MSGPACK_ZERO_COPY_THRESHOLD = 256


def _tensor_bytes(tensor: torch.Tensor) -> memoryview:
    """A flat uint8 view of ``tensor``'s data (CPU, contiguous)."""
    t = tensor.detach()
    if t.device.type != "cpu":
        t = t.cpu()
    if not t.is_contiguous():
        t = t.contiguous()
    return memoryview(t.reshape(-1).view(torch.uint8).numpy()).cast("B")


class MsgpackEncoder:
    """msgpack encoder with tensor/ndarray support and out-of-band buffers.

    ``encode`` returns the list of ZMQ frames to send: the primary msgpack
    buffer first, then one frame per large tensor (referenced by index from
    the primary buffer). Not thread-safe while encoding.
    """

    def __init__(self, size_threshold: int = MSGPACK_ZERO_COPY_THRESHOLD) -> None:
        self._encoder = msgspec.msgpack.Encoder(enc_hook=self._enc_hook)
        self._aux_buffers: list | None = None
        self._size_threshold = size_threshold

    def encode(self, obj: Any) -> list:
        try:
            self._aux_buffers = bufs = [b""]
            bufs[0] = self._encoder.encode(obj)
            return bufs
        finally:
            self._aux_buffers = None

    def _enc_hook(self, obj: Any) -> Any:
        if isinstance(obj, torch.Tensor):
            return self._encode_tensor(obj)
        if isinstance(obj, np.ndarray) and obj.dtype.kind not in ("O", "V"):
            return self._encode_ndarray(obj)
        if isinstance(obj, torch.dtype):
            return str(obj).removeprefix("torch.")
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        raise TypeError(
            f"Cannot msgpack-encode object of type {type(obj)}. Give the field "
            "a precise msgspec-compatible type."
        )

    def _stash(self, buf) -> "msgspec.msgpack.Ext | int":
        """Inline small buffers as an ext raw view; frame out large ones."""
        if buf.nbytes < self._size_threshold:
            return msgspec.msgpack.Ext(CUSTOM_TYPE_RAW_VIEW, buf)
        index = len(self._aux_buffers)
        self._aux_buffers.append(buf)
        return index

    def _encode_tensor(self, obj: torch.Tensor) -> tuple:
        assert self._aux_buffers is not None
        dtype = str(obj.dtype).removeprefix("torch.")
        return dtype, tuple(obj.shape), self._stash(_tensor_bytes(obj))

    def _encode_ndarray(self, obj: np.ndarray) -> tuple:
        assert self._aux_buffers is not None
        arr = np.ascontiguousarray(obj)
        return obj.dtype.str, obj.shape, self._stash(arr.reshape(-1).view("B").data)


class MsgpackDecoder:
    """msgpack decoder resolving tensors from inline ext views or aux frames.

    ``decode`` accepts either a single buffer or the full frame list (frame 0
    = primary buffer). Not thread-safe while decoding.
    """

    def __init__(self, ty: Any = None) -> None:
        args = () if ty is None else (ty,)
        self._decoder = msgspec.msgpack.Decoder(
            *args, dec_hook=self._dec_hook, ext_hook=self._ext_hook
        )
        self._aux_buffers: Any = ()

    def decode(self, bufs: Any) -> Any:
        if isinstance(bufs, (bytes, bytearray, memoryview)):
            return self._decoder.decode(bufs)
        self._aux_buffers = bufs
        try:
            return self._decoder.decode(bufs[0])
        finally:
            self._aux_buffers = ()

    def _dec_hook(self, ty: type, obj: Any) -> Any:
        if ty is torch.dtype:
            return getattr(torch, obj)
        if isinstance(ty, type):
            if issubclass(ty, torch.Tensor):
                return self._decode_tensor(obj)
            if issubclass(ty, np.ndarray):
                return self._decode_ndarray(obj)
        raise TypeError(f"Cannot msgpack-decode into unsupported type {ty}.")

    def _ext_hook(self, code: int, data: memoryview) -> Any:
        if code == CUSTOM_TYPE_RAW_VIEW:
            return data
        raise NotImplementedError(f"Extension type code {code} is not supported")

    def _resolve_buffer(self, data) -> memoryview:
        buf = self._aux_buffers[data] if isinstance(data, int) else data
        return buf if isinstance(buf, memoryview) else memoryview(buf)

    def _decode_tensor(self, arr: Any) -> torch.Tensor:
        dtype, shape, data = arr
        buffer = self._resolve_buffer(data)
        torch_dtype = getattr(torch, dtype, None)
        if not isinstance(torch_dtype, torch.dtype):
            # numpy typestring (a field typed torch.Tensor fed an ndarray).
            return torch.from_numpy(
                np.frombuffer(buffer, dtype=np.dtype(dtype)).copy()
            ).reshape(shape)
        if not buffer.nbytes:  # torch.frombuffer rejects empty buffers
            return torch.empty(shape, dtype=torch_dtype)
        # frombuffer needs a writable buffer; received bytes are read-only, so
        # copy into a bytearray the tensor then owns.
        writable = buffer if not buffer.readonly else bytearray(buffer)
        flat = torch.frombuffer(writable, dtype=torch.uint8)
        return flat.view(torch_dtype).view(shape)

    def _decode_ndarray(self, arr: Any) -> np.ndarray:
        dtype, shape, data = arr
        buffer = self._resolve_buffer(data)
        return np.frombuffer(buffer, dtype=np.dtype(dtype)).copy().reshape(shape)
