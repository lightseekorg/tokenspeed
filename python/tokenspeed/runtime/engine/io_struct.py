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
The definition of objects transferred between different
processes (frontend, controller, scheduler), plus the msgpack codec
that carries them over ZMQ.

Every type that crosses engine IPC is a ``msgspec.Struct`` deriving from
``BaseReq``/``BaseBatchReq`` (tagged, keyword-only, array-encoded), so one
tagged-union decoder per receiving socket replaces pickle. Tensors ride as
msgpack ext-typed raw buffers with large payloads moved to out-of-band ZMQ
frames (see ``MsgpackEncoder``). Pickle IPC remains available as a rollout
escape hatch via ``TOKENSPEED_USE_PICKLE_IPC=1`` (read once at import).

``GenerateReqInput``/``EmbeddingReqInput`` are the HTTP-layer request shapes
(pre-tokenization, in-process only) and intentionally stay dataclasses.
"""

from __future__ import annotations

import copy
import pickle
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Union

import msgspec
import numpy as np
import torch

from tokenspeed.runtime.multimodal.inputs import MultimodalInputs
from tokenspeed.runtime.sampling.sampling_params import SamplingParams
from tokenspeed.runtime.utils.env import envs


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


class BaseReq(
    msgspec.Struct, tag=True, tag_field="_tag", kw_only=True, array_like=True
):
    """Base for single-request IPC payloads.

    ``tag=True`` prefixes the class name so a tagged-union decoder can
    dispatch without pickled type info; ``array_like=True`` encodes fields
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


class PickleWrapper(msgspec.Struct, tag=True, tag_field="_tag", array_like=True):
    """Explicit escape hatch: an opaque Python object as pickled bytes.

    Only for fields whose values are genuinely arbitrary Python objects.
    Multimodal payloads must NOT use this — they have typed structs with
    ext-encoded tensor buffers.
    """

    data: bytes

    @classmethod
    def wrap(cls, obj: Any) -> "PickleWrapper | None":
        if obj is None:
            return None
        return cls(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))

    def unwrap(self) -> Any:
        return pickle.loads(self.data)


class SessionParams(msgspec.Struct, kw_only=True, array_like=True):
    id: str | None = None
    rid: str | None = None
    offset: int | None = None
    replace: bool | None = None


@dataclass
class GenerateReqInput:
    # The input prompt. It can be a single prompt or a batch of prompts.
    text: list[str] | str | None = None
    # The token ids for text; one can specify either text or input_ids
    input_ids: list[list[int]] | list[int] | None = None
    input_multi_ids: list[list[int]] | list[list[int]] | None = None
    # The embeddings for input_ids; one can specify either text or input_ids or input_embeds.
    input_embeds: list[list[list[float]]] | list[list[float]] | None = None
    # Pre-built MultimodalInputs (already produced by an upstream preprocessor,
    # e.g. SMG's Rust crates/multimodal pipeline). The engine's InputProcessor
    # uses this directly (it does no in-process image preprocessing). input_ids
    # must already contain expanded image placeholder tokens at the right
    # offsets — the gateway is responsible for that. Typed as Any to avoid a
    # circular import on MultimodalInputs.
    precomputed_multimodal_inputs: Any | None = None
    # The sampling_params. See descriptions below.
    sampling_params: list[dict] | dict | None = None
    input_extra_infos: list[dict] | dict | None = None
    # Optional client label for logging; defaults to `rid`. Safe to reuse.
    user_rid: list[str] | str | None = None
    # Routing id; always server-assigned during normalize, never caller-settable.
    rid: list[str] | str | None = field(default=None, init=False)
    # --- Logprob request (two dialects, one compute path) ---
    # vLLM-compatible requests use ``sampling_params["logprobs"]``;
    # SGLang-compatible requests use the legacy fields below. A request uses
    # one dialect; the response is rendered to match (override with
    # ``logprob_format``).
    return_logprob: list[bool] | bool | None = None
    # Start location in the prompt for prompt logprobs. -1 (default) = output
    # tokens only.
    logprob_start_len: list[int] | int | None = None
    # Number of top logprobs per position.
    top_logprobs_num: list[int] | int | None = None
    # Specific token ids to score per position.
    token_ids_logprob: list[list[int]] | list[int] | None = None
    # Detokenize tokens in the returned logprobs.
    return_text_in_logprobs: bool = False
    # Output rendering dialect: "vllm" | "sglang" | "both". None = auto (match
    # the request dialect: vllm if sampling_params.logprobs is set, else sglang).
    logprob_format: list[str | None] | str | None = None
    # Whether to stream output.
    stream: bool = False
    # Whether to log metrics for this request (e.g. health_generate calls do not log metrics)
    log_metrics: bool = True

    # Session info for continual prompting
    session_params: list[dict] | dict | None = None

    # Custom logit processor for advanced sampling control. Must be a serialized instance
    # of `CustomLogitProcessor` in python/tokenspeed/runtime/sampling/custom_logit_processor.py
    # Use the processor's `to_str()` method to generate the serialized string.
    custom_logit_processor: list[str | None] | str | None = None

    # Whether to return hidden states
    return_hidden_states: bool = False

    # For disaggregated inference
    bootstrap_host: list[str] | str | None = None
    bootstrap_port: list[int] | int | None = None
    bootstrap_room: list[int] | int | None = None

    def normalize_batch_and_arguments(self):
        if (
            self.text is None and self.input_ids is None and self.input_embeds is None
        ) or (
            self.text is not None
            and self.input_ids is not None
            and self.input_embeds is not None
        ):
            raise ValueError(
                "Either text, input_ids or input_embeds should be provided."
            )

        # Derive the batch size
        if self.text is not None:
            if isinstance(self.text, str):
                self.is_single = True
                self.batch_size = 1
            else:
                self.is_single = False
                self.batch_size = len(self.text)
            self.input_embeds = None
        elif self.input_ids is not None:
            if isinstance(self.input_ids[0], int):
                self.is_single = True
                self.batch_size = 1
            else:
                self.is_single = False
                self.batch_size = len(self.input_ids)
            self.input_embeds = None
        else:
            _require(
                isinstance(self.input_embeds, list), "input_embeds should be a list."
            )
            if isinstance(self.input_embeds[0][0], float):
                # list[list[float]]
                self.is_single = True
                self.batch_size = 1
            else:
                # list[list[list[float]]]
                _require(
                    isinstance(self.input_embeds[0][0], list),
                    "input_embeds should be a list of float lists.",
                )
                _require(
                    isinstance(self.input_embeds[0][0][0], float),
                    "input_embeds should contain floats.",
                )
                self.is_single = False
                self.batch_size = len(self.input_embeds)

        # Handle parallel sampling. Pop "n" out of sampling_params so the
        # downstream SamplingParams(**dict) construction doesn't see it —
        # "n" is a request-level fan-out knob, not a per-sample field.
        if self.sampling_params is None:
            self.parallel_sample_num = 1
        elif isinstance(self.sampling_params, dict):
            self.parallel_sample_num = self.sampling_params.get("n", 1)
        else:  # isinstance(self.sampling_params, list):
            self.parallel_sample_num = self.sampling_params[0].get("n", 1)
            for sp in self.sampling_params[1:]:
                _require(
                    self.parallel_sample_num == sp.get("n", 1),
                    "The parallel_sample_num should be the same for all samples in sample params.",
                )

        if self.parallel_sample_num > 1 and self.is_single:
            self.is_single = False
            if self.text is not None:
                self.text = [self.text]
            if self.input_ids is not None:
                self.input_ids = [self.input_ids]
            if self.input_multi_ids is not None:
                self.input_multi_ids = [self.input_multi_ids]
            if self.input_embeds is not None:
                self.input_embeds = [self.input_embeds]

        # Fill in default arguments
        if self.is_single:
            if self.sampling_params is None:
                self.sampling_params = {}
            if self.rid is None:
                self.rid = uuid.uuid4().hex
            if self.user_rid is None:
                self.user_rid = self.rid
            else:
                if isinstance(self.user_rid, list):
                    _require(
                        len(self.user_rid) == 1,
                        "user_rid list should have length 1 for single request.",
                    )
                    self.user_rid = self.user_rid[0]
                _require(isinstance(self.user_rid, str), "user_rid should be a str.")
            if self.return_logprob is None:
                self.return_logprob = False
            if self.logprob_start_len is None:
                self.logprob_start_len = -1
            if self.top_logprobs_num is None:
                self.top_logprobs_num = 0
            if not self.token_ids_logprob:  # covers both None and []
                self.token_ids_logprob = None
            if isinstance(self.input_extra_infos, dict):
                self.input_extra_infos = [self.input_extra_infos]
        else:
            if self.parallel_sample_num == 1:
                num = self.batch_size
            else:
                # Expand parallel_sample_num
                num = self.batch_size * self.parallel_sample_num

            if self.sampling_params is None:
                self.sampling_params = [{} for _ in range(num)]
            elif not isinstance(self.sampling_params, list):
                self.sampling_params = [dict(self.sampling_params) for _ in range(num)]

            if self.rid is None:
                self.rid = [uuid.uuid4().hex for _ in range(num)]
            else:
                _require(isinstance(self.rid, list), "The rid should be a list.")
            if self.user_rid is None:
                self.user_rid = list(self.rid)
            elif isinstance(self.user_rid, str):
                self.user_rid = [self.user_rid] * num
            else:
                _require(
                    isinstance(self.user_rid, list) and len(self.user_rid) == num,
                    "user_rid should be a str or a list of matching length.",
                )

            if self.return_logprob is None:
                self.return_logprob = [False] * num
            elif not isinstance(self.return_logprob, list):
                self.return_logprob = [self.return_logprob] * num
            else:
                _require(
                    self.parallel_sample_num == 1,
                    "return_logprob cannot be a list when n > 1.",
                )

            if self.logprob_start_len is None:
                self.logprob_start_len = [-1] * num
            elif not isinstance(self.logprob_start_len, list):
                self.logprob_start_len = [self.logprob_start_len] * num
            else:
                _require(
                    self.parallel_sample_num == 1,
                    "logprob_start_len cannot be a list when n > 1.",
                )

            if self.top_logprobs_num is None:
                self.top_logprobs_num = [0] * num
            elif not isinstance(self.top_logprobs_num, list):
                self.top_logprobs_num = [self.top_logprobs_num] * num
            else:
                _require(
                    self.parallel_sample_num == 1,
                    "top_logprobs_num cannot be a list when n > 1.",
                )

            if not self.token_ids_logprob:  # covers both None and []
                self.token_ids_logprob = [None] * num
            elif not isinstance(self.token_ids_logprob, list):
                self.token_ids_logprob = [[self.token_ids_logprob] for _ in range(num)]
            elif not isinstance(self.token_ids_logprob[0], list):
                self.token_ids_logprob = [
                    copy.deepcopy(self.token_ids_logprob) for _ in range(num)
                ]
            else:
                _require(
                    self.parallel_sample_num == 1,
                    "token_ids_logprob cannot be nested lists when n > 1.",
                )

            if self.logprob_format is None or isinstance(self.logprob_format, str):
                self.logprob_format = [self.logprob_format] * num

            if self.custom_logit_processor is None:
                self.custom_logit_processor = [None] * num
            elif not isinstance(self.custom_logit_processor, list):
                self.custom_logit_processor = [self.custom_logit_processor] * num
            else:
                _require(
                    self.parallel_sample_num == 1,
                    "custom_logit_processor cannot be a list when n > 1.",
                )

            if self.bootstrap_host is None:
                self.bootstrap_host = [None] * num
            elif not isinstance(self.bootstrap_host, list):
                self.bootstrap_host = [self.bootstrap_host] * num
            else:
                _require(
                    self.parallel_sample_num == 1,
                    "bootstrap_host cannot be a list when n > 1.",
                )

            if self.bootstrap_port is None:
                self.bootstrap_port = [None] * num
            elif not isinstance(self.bootstrap_port, list):
                self.bootstrap_port = [self.bootstrap_port] * num
            else:
                _require(
                    self.parallel_sample_num == 1,
                    "bootstrap_port cannot be a list when n > 1.",
                )

            if self.bootstrap_room is None:
                self.bootstrap_room = [None] * num
            elif not isinstance(self.bootstrap_room, list):
                self.bootstrap_room = [self.bootstrap_room] * num
            else:
                _require(
                    self.parallel_sample_num == 1,
                    "bootstrap_room cannot be a list when n > 1.",
                )

        # Other checks
        if self.session_params is not None:
            _require(
                isinstance(self.session_params, dict)
                or isinstance(self.session_params[0], dict),
                "session_params should be a dict or a list of dicts.",
            )

    def regenerate_rid(self):
        self.rid = uuid.uuid4().hex
        return self.rid

    def __getitem__(self, i):
        sub = GenerateReqInput(
            text=self.text[i] if self.text is not None else None,
            input_ids=self.input_ids[i] if self.input_ids is not None else None,
            # precomputed_multimodal_inputs is a single prompt's MM; the SMG
            # path only clears is_single via n>1 (batch_size == 1), so all n
            # parallel samples correctly share it. Without this the image is
            # silently dropped on the n>1 fan-out (placeholders -> text path).
            precomputed_multimodal_inputs=self.precomputed_multimodal_inputs,
            input_multi_ids=(
                self.input_multi_ids[i] if self.input_multi_ids is not None else None
            ),
            input_embeds=(
                self.input_embeds[i] if self.input_embeds is not None else None
            ),
            input_extra_infos=(
                self.input_extra_infos[i]
                if self.input_extra_infos is not None
                else None
            ),
            sampling_params=self.sampling_params[i],
            user_rid=self.user_rid[i],
            return_logprob=self.return_logprob[i],
            logprob_start_len=self.logprob_start_len[i],
            top_logprobs_num=self.top_logprobs_num[i],
            token_ids_logprob=self.token_ids_logprob[i],
            return_text_in_logprobs=self.return_text_in_logprobs,
            logprob_format=self.logprob_format[i],
            stream=self.stream,
            log_metrics=self.log_metrics,
            custom_logit_processor=(
                self.custom_logit_processor[i]
                if self.custom_logit_processor is not None
                else None
            ),
            return_hidden_states=self.return_hidden_states,
            # if `__getitem__` is called, the bootstrap_host, bootstrap_port, bootstrap_room must be a list
            bootstrap_host=(
                self.bootstrap_host[i] if self.bootstrap_host is not None else None
            ),
            bootstrap_port=(
                self.bootstrap_port[i] if self.bootstrap_port is not None else None
            ),
            bootstrap_room=(
                self.bootstrap_room[i] if self.bootstrap_room is not None else None
            ),
        )
        sub.rid = self.rid[i]
        return sub


class TokenizedGenerateReqInput(BaseReq, kw_only=True):
    # The input text (None on the token-id-only and input-embeds paths)
    input_text: str | None = None
    # The input token ids (None on the input-embeds path)
    input_ids: list[int] | None = None
    # The sampling parameters
    sampling_params: SamplingParams
    # Whether to return the sampled token's logprob for this request.
    return_logprob: bool = False
    # Internal carry-over fields kept for pipeline/PD compatibility. The
    # output-logprob API only drives ``return_logprob``; InputProcessor sets
    # these to neutral values (logprob_start_len=-1, top_logprobs_num=0,
    # token_ids_logprob=None) since prompt logprobs, output top-k, and token-id
    # logprobs are not supported.
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
    # out-of-band frames (or SHM handles on the in-host pickle-era path).
    multimodal_inputs: MultimodalInputs | None = None

    # Set by a transport that validates requests itself (the msgpack ZMQ path,
    # which bypasses the tokenizer_manager's input processor). A non-None value
    # makes RequestHandler admit the request pre-finished with FINISH_ABORT so
    # the client receives a terminal abort instead of a silent drop.
    validation_error: str | None = None


@dataclass
class EmbeddingReqInput:
    # The input prompt. It can be a single prompt or a batch of prompts.
    text: list[str] | str | None = None
    # The token ids for text; one can either specify text or input_ids.
    input_ids: list[list[int]] | list[int] | None = None
    # Optional client label for logging; defaults to `rid`. Safe to reuse.
    user_rid: list[str] | str | None = None
    # Routing id; always server-assigned during normalize, never caller-settable.
    rid: list[str] | str | None = field(default=None, init=False)
    # Optional placeholder so non-generation callers can still instantiate the
    # shared request shape without real sampling params.
    sampling_params: list[dict] | dict = None
    # Optional placeholder for callers that do not provide input embeddings.
    input_embeds: list[list[list[float]]] | list[list[float]] | None = None
    # Whether to log metrics for this request (e.g. health_generate calls do not log metrics)
    log_metrics: bool = True

    def normalize_batch_and_arguments(self):
        if (self.text is None and self.input_ids is None) or (
            self.text is not None and self.input_ids is not None
        ):
            raise ValueError("Either text or input_ids should be provided.")

        # Derive the batch size
        if self.text is not None:
            if isinstance(self.text, str):
                self.is_single = True
                self.batch_size = 1
            else:
                self.is_single = False
                self.batch_size = len(self.text)
        else:
            if isinstance(self.input_ids[0], int):
                self.is_single = True
                self.batch_size = 1
            else:
                self.is_single = False
                self.batch_size = len(self.input_ids)

        # Fill in default arguments
        if self.is_single:
            if self.rid is None:
                self.rid = uuid.uuid4().hex
            if self.user_rid is None:
                self.user_rid = self.rid
            else:
                if isinstance(self.user_rid, list):
                    _require(
                        len(self.user_rid) == 1,
                        "user_rid list should have length 1 for single request.",
                    )
                    self.user_rid = self.user_rid[0]
                _require(isinstance(self.user_rid, str), "user_rid should be a str.")
            if self.sampling_params is None:
                self.sampling_params = {}
            self.sampling_params["max_new_tokens"] = 0
        else:
            if self.rid is None:
                self.rid = [uuid.uuid4().hex for _ in range(self.batch_size)]
            else:
                _require(isinstance(self.rid, list), "The rid should be a list.")
            if self.user_rid is None:
                self.user_rid = list(self.rid)
            elif isinstance(self.user_rid, str):
                self.user_rid = [self.user_rid] * self.batch_size
            else:
                _require(
                    isinstance(self.user_rid, list)
                    and len(self.user_rid) == self.batch_size,
                    "user_rid should be a str or a list of matching length.",
                )

            if self.sampling_params is None:
                self.sampling_params = [{} for _ in range(self.batch_size)]
            for i in range(self.batch_size):
                self.sampling_params[i]["max_new_tokens"] = 0

    def regenerate_rid(self):
        self.rid = uuid.uuid4().hex
        return self.rid

    def __getitem__(self, i):
        sub = EmbeddingReqInput(
            text=self.text[i] if self.text is not None else None,
            input_ids=self.input_ids[i] if self.input_ids is not None else None,
            sampling_params=self.sampling_params[i],
            user_rid=self.user_rid[i],
        )
        sub.rid = self.rid[i]
        return sub


class TokenizedEmbeddingReqInput(BaseReq, kw_only=True):
    # The input text
    input_text: str | None = None
    # The input token ids
    input_ids: list[int] | None = None
    # Placeholder sampling params field so request metadata can share one shape
    # with generation-oriented code paths.
    sampling_params: SamplingParams
    # Time at object instantiated
    created_time: float = 0.0


# Serialized form of BaseFinishReason.to_json() (or None while streaming) —
# all values are msgpack-native primitives.
FinishReasonDict = dict


class BatchTokenIDOut(BaseBatchReq, kw_only=True):
    # The finish reason (``BaseFinishReason.to_json()`` dicts, None mid-stream)
    finished_reasons: list[FinishReasonDict | None]
    # For incremental decoding
    decoded_texts: list[str]
    decode_ids: list[list[int]]
    read_offsets: list[int]
    # Only used when `--skip-tokenizer-init` is on. Per-request lists: the
    # not-yet-sent slice of each request's generated ids (see stream_output).
    output_ids: list[list[int]] | None
    output_multi_ids: list[list[int]] | None
    # Detokenization configs
    skip_special_tokens: list[bool]
    spaces_between_special_tokens: list[bool]
    no_stop_trim: list[bool]

    # Token counts
    prompt_tokens: list[int]
    completion_tokens: list[int]
    cached_tokens: list[int]
    spec_verify_ct: list[int]

    # Logprobs
    input_token_logprobs_val: list[float]
    input_token_logprobs_idx: list[int]
    # Per-request lists, parallel to rids: the newly-decoded tokens' sampled
    # logprobs/token ids this step, [] when logprobs are off (see stream_output).
    output_token_logprobs_val: list[list[float]]
    output_token_logprobs_idx: list[list[int]]
    input_top_logprobs_val: list[list]
    input_top_logprobs_idx: list[list]
    output_top_logprobs_val: list[list]
    output_top_logprobs_idx: list[list]
    input_token_ids_logprobs_val: list[list]
    input_token_ids_logprobs_idx: list[list]
    output_token_ids_logprobs_val: list[list]
    output_token_ids_logprobs_idx: list[list]

    # Hidden states
    output_hidden_states: list[list[float]]
    # Per-request draft-token acceptance (speculative decoding); None until
    # the request finishes, empty when spec decoding is off.
    batch_accept_draft_tokens: list[float | None]

    # Store some custom information, such as decoding status in multimodal scenarios, etc.
    output_extra_infos: list[dict[str, Any]]

    generated_time: float


def _finish_type(finished_reason) -> str:
    """Reduce an OutputProcesser finish reason to its wire type string."""
    if not finished_reason:
        return ""
    if isinstance(finished_reason, dict):
        return finished_reason.get("type", "")
    return str(finished_reason)


class BatchTokenIDOutSlim(BaseBatchReq, kw_only=True):
    """Per-request slice of ``BatchTokenIDOut`` for a frontend that
    detokenizes itself (the direct ZMQ scheduler drive).

    Carries only what that frontend consumes, so per-step outputs skip the
    incremental-detokenization columns (decoded_texts, decode_ids, ...) the
    in-process frontend needs. Field ORDER is a cross-language wire
    contract — do not reorder.
    """

    # Newly generated token ids per request this step, sourced from
    # ``BatchTokenIDOut.output_ids``.
    output_ids: list[list[int]]
    # "" (not finished) or the finish-reason type ("stop", "length", "abort").
    finished_reasons: list[str]
    prompt_tokens: list[int]
    completion_tokens: list[int]
    cached_tokens: list[int]
    # Sampled-token logprobs, parallel to rids: one inner list per request,
    # holding the value/token-id of each newly-decoded token this step. Empty []
    # for a request that did not ask for logprobs, so the columns stay
    # non-ragged (always length == len(rids)).
    output_token_logprobs_val: list[list[float]]
    output_token_logprobs_idx: list[list[int]]
    # Producing DP rank's engine index (the identity it dialed the frontend
    # with). The output PULL socket carries no routing identity, so under
    # DP the batch itself names its rank. Appended field: defaults to 0 so
    # older peers on either side stay compatible.
    engine_index: int = 0

    @classmethod
    def from_full(
        cls, out: BatchTokenIDOut, engine_index: int = 0
    ) -> "BatchTokenIDOutSlim":
        # Token source: ``out.output_ids`` — the not-yet-sent slice of each
        # request's generated ids. NOT ``out.decode_ids``: that is the
        # incremental-detokenization window, which starts at the prompt tail
        # for context and would leak prompt tokens to a frontend that
        # detokenizes from scratch.
        if out.output_ids is None:
            # Substituting empty lists here while completion_tokens advances
            # would silently lose tokens on the frontend; fail loud instead.
            raise ValueError(
                "BatchTokenIDOut.output_ids is None; the msgpack wire needs "
                "the per-request generated token ids"
            )
        return cls(
            engine_index=engine_index,
            rids=list(out.rids),
            output_ids=[list(ids) for ids in out.output_ids],
            finished_reasons=[_finish_type(fr) for fr in out.finished_reasons],
            prompt_tokens=list(out.prompt_tokens),
            completion_tokens=list(out.completion_tokens),
            cached_tokens=list(out.cached_tokens),
            output_token_logprobs_val=(
                list(out.output_token_logprobs_val)
                if out.output_token_logprobs_val is not None
                else [[] for _ in out.rids]
            ),
            output_token_logprobs_idx=(
                list(out.output_token_logprobs_idx)
                if out.output_token_logprobs_idx is not None
                else [[] for _ in out.rids]
            ),
        )


class BatchStrOut(BaseBatchReq, kw_only=True):
    # The finish reason
    finished_reasons: list[FinishReasonDict | None]
    # The output decoded strings
    output_strs: list[str]
    # The token ids
    output_ids: list[int] | None

    # Token counts
    prompt_tokens: list[int]
    completion_tokens: list[int]
    cached_tokens: list[int]
    spec_verify_ct: list[int]

    # Logprobs
    input_token_logprobs_val: list[float]
    input_token_logprobs_idx: list[int]
    output_token_logprobs_val: list[float]
    output_token_logprobs_idx: list[int]
    input_top_logprobs_val: list[list]
    input_top_logprobs_idx: list[list]
    output_top_logprobs_val: list[list]
    output_top_logprobs_idx: list[list]
    input_token_ids_logprobs_val: list[list]
    input_token_ids_logprobs_idx: list[list]
    output_token_ids_logprobs_val: list[list]
    output_token_ids_logprobs_idx: list[list]

    # Hidden states
    output_hidden_states: list[list[float]]
    # See BatchTokenIDOut.batch_accept_draft_tokens (None mid-stream).
    batch_accept_draft_tokens: list[float | None]

    # Store some custom information, such as decoding status in multimodal scenarios, etc.
    output_extra_infos: list[dict[str, Any]]

    generated_time: float


class BatchEmbeddingOut(BaseBatchReq, kw_only=True):
    # The finish reason
    finished_reasons: list[FinishReasonDict | None]
    # The output embedding (dense rows or sparse dicts)
    embeddings: list
    # Token counts
    prompt_tokens: list[int]


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
    # Optional: update the weight version after a successful push. When provided,
    # subsequent generation responses will carry this version in meta_info.
    weight_version: str | None = None


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


class GetWeightsByNameReqInput(BaseReq, kw_only=True):
    name: str
    truncate_size: int = 100


class GetWeightsByNameReqOutput(BaseReq, kw_only=True):
    parameter: list


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


class ExpertDistributionReqType(Enum):
    START_RECORD = 1
    STOP_RECORD = 2
    DUMP_RECORD = 3


class ExpertDistributionReq(BaseReq, kw_only=True):
    action: ExpertDistributionReqType


class ExpertDistributionReqOutput(BaseReq, kw_only=True):
    pass


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


class ConfigureLoggingReq(BaseReq, kw_only=True):
    log_requests: bool | None = None
    log_requests_level: int | None = None
    dump_requests_folder: str | None = None
    dump_requests_threshold: int | None = None


class OpenSessionReqInput(BaseReq, kw_only=True):
    capacity_of_str_len: int
    session_id: str | None = None


class CloseSessionReqInput(BaseReq, kw_only=True):
    session_id: str


class OpenSessionReqOutput(BaseReq, kw_only=True):
    session_id: str | None
    success: bool


class HealthCheckOutput(BaseReq, kw_only=True):
    pass


class RpcReqInput(BaseReq, kw_only=True):
    method: str
    parameters: dict | None = None


class RpcReqOutput(BaseReq, kw_only=True):
    success: bool
    message: str


class GetLoadReqInput(BaseReq, kw_only=True):
    pass


class GetLoadReqOutput(BaseReq, kw_only=True):
    dp_rank: int = 0
    num_reqs: int = 0
    num_waiting_reqs: int = 0
    num_pages: int = 0


class LoadSnapshot(
    msgspec.Struct, frozen=True, tag=True, tag_field="_tag", array_like=True
):
    """Immutable scheduler load replica sent over engine IPC.

    Fields are positional on the wire. Append any future fields only at the
    end so older decoders can retain their prefix compatibility.
    """

    epoch: str
    sequence: int
    dp_rank: int
    num_running_reqs: int
    num_waiting_reqs: int
    num_active_pages: int
    num_used_pages: int
    max_total_pages: int
    valid_for_ms: int


class WatchLoadUpdateReq(BaseReq, kw_only=True):
    loads: list[GetLoadReqOutput] = []


class BlockReqType(Enum):
    BLOCK = 1
    UNBLOCK = 2


class BlockReqInput(BaseReq, kw_only=True):
    type: BlockReqType = BlockReqType.BLOCK


# ============================================================================
# msgpack codec for engine IPC.
#
# Tensor scheme (shared with the SMG Rust engine-side codec): a tensor or
# ndarray encodes as the tuple ``(dtype, shape, data)`` where ``data`` is
# either an ext-typed raw byte view (``CUSTOM_TYPE_RAW_VIEW``) for small
# payloads, or an integer index into the message's out-of-band ZMQ frames
# (frame 0 is the primary msgpack buffer, so indices are one-based).
# ============================================================================

# msgpack extension type codes. 1 and 2 are reserved for pickled payloads in
# the shared cross-codec numbering; this codec never emits them (opaque
# objects must use an explicit PickleWrapper field instead).
CUSTOM_TYPE_PICKLE = 1
CUSTOM_TYPE_CLOUDPICKLE = 2
CUSTOM_TYPE_RAW_VIEW = 3

# Tensors/ndarrays below this many bytes are inlined in the primary buffer;
# larger ones become dedicated zero-copy frames.
MSGPACK_ZERO_COPY_THRESHOLD = 256

# Rollout escape hatch: force the legacy pickle IPC end-to-end. Read once at
# import; the senders/receivers below all consult it.
USE_PICKLE_IPC = envs.TOKENSPEED_USE_PICKLE_IPC.get()


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
            "a precise msgspec-compatible type, or wrap a genuinely opaque "
            "payload in an explicit PickleWrapper field."
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


def _walk_subclasses(base: type) -> list:
    out = []
    for sub in base.__subclasses__():
        out.append(sub)
        out.extend(_walk_subclasses(sub))
    return out


def ipc_message_union():
    """The tagged union of every registered IPC message type.

    Computed lazily so process roles that define extra ``BaseReq`` subclasses
    (e.g. the EPD encode worker) are included as long as the defining module
    is imported before the receiving socket is constructed.
    """
    types = tuple(
        dict.fromkeys(
            _walk_subclasses(BaseReq)
            + _walk_subclasses(BaseBatchReq)
            + [LoadSnapshot, PickleWrapper]
        )
    )
    return Union[types]  # noqa: UP007 — dynamic union over a runtime tuple


class IpcSender:
    """Engine-IPC sender wrapping a ZMQ PUSH/DEALER socket.

    Exposes ``send_pyobj`` so existing call sites are unchanged; the payload
    is msgpack (multipart, with out-of-band tensor frames) unless the
    ``USE_PICKLE_IPC`` escape hatch is on. ``send`` accepts a pre-pickled
    engine message from legacy off-thread serializers and transcodes it onto
    the same wire. Works over both sync and asyncio sockets (asyncio sends
    return the socket's future).
    """

    def __init__(self, socket) -> None:
        self._socket = socket
        self._encoder = MsgpackEncoder()

    def send_pyobj(self, obj: Any, flags: int = 0):
        if USE_PICKLE_IPC:
            return self._socket.send_pyobj(obj, flags)
        return self._socket.send_multipart(self._encoder.encode(obj), flags, copy=False)

    def send(self, data, flags: int = 0):
        # Legacy raw-frame surface: external callers that serialize an engine
        # message off-thread hand this seam pickled bytes (the pre-msgpack wire).
        # Transcode through the shared codec so every receiver on the channel
        # sees one wire format; under the pickle escape hatch the bytes already
        # match the wire and pass through untouched.
        if USE_PICKLE_IPC:
            return self._socket.send(data, flags)
        return self.send_pyobj(pickle.loads(data), flags)

    def close(self, linger: int | None = None) -> None:
        self._socket.close(linger=linger)

    def __getattr__(self, name: str):
        # Raw socket surface (poll, setsockopt, ...) for callers that bypass
        # the object codec.
        return getattr(self._socket, name)


class IpcReceiver:
    """Engine-IPC receiver wrapping a sync ZMQ PULL socket.

    ``recv_pyobj`` preserves the drain-loop contract: ``zmq.Again`` under
    NOBLOCK when the socket is empty.
    """

    def __init__(self, socket) -> None:
        self._socket = socket
        self._decoder = MsgpackDecoder(ipc_message_union())

    def recv_pyobj(self, flags: int = 0) -> Any:
        if USE_PICKLE_IPC:
            return self._socket.recv_pyobj(flags)
        return self._decoder.decode(self._socket.recv_multipart(flags))

    def close(self, linger: int | None = None) -> None:
        self._socket.close(linger=linger)

    def __getattr__(self, name: str):
        return getattr(self._socket, name)


class AsyncIpcReceiver:
    """Engine-IPC receiver wrapping a ``zmq.asyncio`` PULL socket."""

    def __init__(self, socket) -> None:
        self._socket = socket
        self._decoder = MsgpackDecoder(ipc_message_union())

    async def recv_pyobj(self, flags: int = 0) -> Any:
        if USE_PICKLE_IPC:
            return await self._socket.recv_pyobj(flags)
        return self._decoder.decode(await self._socket.recv_multipart(flags))

    def close(self, linger: int | None = None) -> None:
        self._socket.close(linger=linger)

    def __getattr__(self, name: str):
        return getattr(self._socket, name)
