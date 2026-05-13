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
processes (TokenizerManager, DetokenizerManager, Controller).
"""

import copy
import uuid
from abc import ABC
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from tokenspeed.runtime.engine.request import BaseFinishReason
from tokenspeed.runtime.sampling.sampling_params import SamplingParams
from tokenspeed.runtime.utils import ImageData

# Handle serialization of Image for pydantic
if TYPE_CHECKING:
    from PIL.Image import Image
else:
    Image = Any


@dataclass
class BaseReq(ABC):
    rid: str | list[str] | None = field(default=None)
    http_worker_ipc: str | None = field(default=None)

    def regenerate_rid(self):
        """Generate a new request ID and return it."""
        if isinstance(self.rid, list):
            self.rid = [uuid.uuid4().hex for _ in range(len(self.rid))]
        else:
            self.rid = uuid.uuid4().hex
        return self.rid


@dataclass
class SessionParams:
    id: str | None = None
    rid: str | None = None
    offset: int | None = None
    replace: bool | None = None


# Type definitions for multimodal input data
# Individual data item types for each modality
ImageDataInputItem = Image | str | ImageData | dict
AudioDataInputItem = str | dict
VideoDataInputItem = str | dict
# Union type for any multimodal data item
MultimodalDataInputItem = ImageDataInputItem | VideoDataInputItem | AudioDataInputItem
# Format types supporting single items, lists, or nested lists for batch processing
MultimodalDataInputFormat = (
    list[list[MultimodalDataInputItem]]
    | list[MultimodalDataInputItem]
    | MultimodalDataInputItem
)


@dataclass
class GenerateReqInput:
    # The input prompt. It can be a single prompt or a batch of prompts.
    text: list[str] | str | None = None
    # The token ids for text; one can specify either text or input_ids
    input_ids: list[list[int]] | list[int] | None = None
    input_multi_ids: list[list[int]] | list[list[int]] | None = None
    # The embeddings for input_ids; one can specify either text or input_ids or input_embeds.
    input_embeds: list[list[list[float]]] | list[list[float]] | None = None
    # The image input. It can be a file name, a url, or base64 encoded string.
    # See also python/tokenspeed/runtime/utils/common.py:load_image.
    image_data: list[str] | str | None = None
    # The video input. Like image data, it can be a file name, a url, or base64 encoded string.
    video_data: MultimodalDataInputFormat | None = None
    # The audio input. Like image data, it can be a file name, a url, or base64 encoded string.
    audio_data: MultimodalDataInputFormat | None = None
    # The sampling_params. See descriptions below.
    sampling_params: list[dict] | dict | None = None
    input_extra_infos: list[dict] | dict | None = None
    # Optional client label for logging; defaults to `rid`. Safe to reuse.
    user_rid: list[str] | str | None = None
    # Routing id; always server-assigned during normalize, never caller-settable.
    rid: list[str] | str | None = field(default=None, init=False)
    # Whether to return logprobs.
    return_logprob: list[bool] | bool | None = None
    # If return logprobs, the start location in the prompt for returning logprobs.
    # By default, this value is "-1", which means it will only return logprobs for output tokens.
    logprob_start_len: list[int] | int | None = None
    # If return logprobs, the number of top logprobs to return at each position.
    top_logprobs_num: list[int] | int | None = None
    # If return logprobs, the token ids to return logprob for.
    token_ids_logprob: list[list[int]] | list[int] | None = None
    # Whether to detokenize tokens in text in the returned logprobs.
    return_text_in_logprobs: bool = False
    # Whether to stream output.
    stream: bool = False
    # Whether to log metrics for this request (e.g. health_generate calls do not log metrics)
    log_metrics: bool = True

    # The modalities of the image data [image, multi-images, video]
    modalities: list[str] | None = None

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

    # LoRA adapter to use for this request.
    # Supply the name under which the adapter was registered via
    # Engine.load_lora_adapter(), or a filesystem path when the engine
    # is configured with --enable-lora.
    # None means use the base model (no adapter).
    lora_path: list[str | None] | str | None = None

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
            assert isinstance(self.input_embeds, list)
            if isinstance(self.input_embeds[0][0], float):
                # list[list[float]]
                self.is_single = True
                self.batch_size = 1
            else:
                # list[list[list[float]]]
                assert isinstance(self.input_embeds[0][0], list)
                assert isinstance(self.input_embeds[0][0][0], float)
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
                assert self.parallel_sample_num == sp.get(
                    "n", 1
                ), "The parallel_sample_num should be the same for all samples in sample params."

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
                    assert (
                        len(self.user_rid) == 1
                    ), "user_rid list should have length 1 for single request."
                    self.user_rid = self.user_rid[0]
                assert isinstance(self.user_rid, str), "user_rid should be a str."
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

            if not self.image_data:
                self.image_data = [None] * num
            elif not isinstance(self.image_data, list):
                self.image_data = [self.image_data] * num
            elif isinstance(self.image_data, list):
                pass

            if self.sampling_params is None:
                self.sampling_params = [{}] * num
            elif not isinstance(self.sampling_params, list):
                self.sampling_params = [self.sampling_params] * num

            if self.rid is None:
                self.rid = [uuid.uuid4().hex for _ in range(num)]
            else:
                assert isinstance(self.rid, list), "The rid should be a list."
            if self.user_rid is None:
                self.user_rid = list(self.rid)
            elif isinstance(self.user_rid, str):
                self.user_rid = [self.user_rid] * num
            else:
                assert (
                    isinstance(self.user_rid, list) and len(self.user_rid) == num
                ), "user_rid should be a str or a list of matching length."

            if self.return_logprob is None:
                self.return_logprob = [False] * num
            elif not isinstance(self.return_logprob, list):
                self.return_logprob = [self.return_logprob] * num
            else:
                assert self.parallel_sample_num == 1

            if self.logprob_start_len is None:
                self.logprob_start_len = [-1] * num
            elif not isinstance(self.logprob_start_len, list):
                self.logprob_start_len = [self.logprob_start_len] * num
            else:
                assert self.parallel_sample_num == 1

            if self.top_logprobs_num is None:
                self.top_logprobs_num = [0] * num
            elif not isinstance(self.top_logprobs_num, list):
                self.top_logprobs_num = [self.top_logprobs_num] * num
            else:
                assert self.parallel_sample_num == 1

            if not self.token_ids_logprob:  # covers both None and []
                self.token_ids_logprob = [None] * num
            elif not isinstance(self.token_ids_logprob, list):
                self.token_ids_logprob = [[self.token_ids_logprob] for _ in range(num)]
            elif not isinstance(self.token_ids_logprob[0], list):
                self.token_ids_logprob = [
                    copy.deepcopy(self.token_ids_logprob) for _ in range(num)
                ]
            else:
                assert self.parallel_sample_num == 1

            if self.custom_logit_processor is None:
                self.custom_logit_processor = [None] * num
            elif not isinstance(self.custom_logit_processor, list):
                self.custom_logit_processor = [self.custom_logit_processor] * num
            else:
                assert self.parallel_sample_num == 1

            if self.bootstrap_host is None:
                self.bootstrap_host = [None] * num
            elif not isinstance(self.bootstrap_host, list):
                self.bootstrap_host = [self.bootstrap_host] * num
            else:
                assert self.parallel_sample_num == 1

            if self.bootstrap_port is None:
                self.bootstrap_port = [None] * num
            elif not isinstance(self.bootstrap_port, list):
                self.bootstrap_port = [self.bootstrap_port] * num
            else:
                assert self.parallel_sample_num == 1

            if self.bootstrap_room is None:
                self.bootstrap_room = [None] * num
            elif not isinstance(self.bootstrap_room, list):
                self.bootstrap_room = [self.bootstrap_room] * num
            else:
                assert self.parallel_sample_num == 1

        # Other checks
        if self.session_params is not None:
            assert isinstance(self.session_params, dict) or isinstance(
                self.session_params[0], dict
            )

    def regenerate_rid(self):
        self.rid = uuid.uuid4().hex
        return self.rid

    def __getitem__(self, i):
        sub = GenerateReqInput(
            text=self.text[i] if self.text is not None else None,
            input_ids=self.input_ids[i] if self.input_ids is not None else None,
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
            image_data=self.image_data[i],
            sampling_params=self.sampling_params[i],
            user_rid=self.user_rid[i],
            return_logprob=self.return_logprob[i],
            logprob_start_len=self.logprob_start_len[i],
            top_logprobs_num=self.top_logprobs_num[i],
            token_ids_logprob=self.token_ids_logprob[i],
            return_text_in_logprobs=self.return_text_in_logprobs,
            stream=self.stream,
            log_metrics=self.log_metrics,
            modalities=self.modalities[i] if self.modalities else None,
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
            # ``lora_path`` may be a list (one entry per batched request) or
            # a single str/None applied to every request.  Without this
            # propagation each per-request sub-object would silently lose
            # its adapter binding and run as base model.
            lora_path=(
                self.lora_path[i]
                if isinstance(self.lora_path, list)
                else self.lora_path
            ),
        )
        sub.rid = self.rid[i]
        return sub


@dataclass
class TokenizedGenerateReqInput:
    # The request id
    rid: str
    # The input text
    input_text: str
    # The input token ids
    input_ids: list[int]
    # The sampling parameters
    sampling_params: SamplingParams
    # Whether to return the logprobs
    return_logprob: bool
    # If return logprobs, the start location in the prompt for returning logprobs.
    logprob_start_len: int
    # If return logprobs, the number of top logprobs to return at each position.
    top_logprobs_num: int
    # If return logprobs, the token id to return logprob for
    token_ids_logprob: list[int]
    # Whether to stream output
    stream: bool

    # The input embeds
    input_embeds: list[list[list[float]]] | list[list[float]] | None = None

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

    input_multi_ids: list[list[int]] = None
    input_extra_infos: list[dict] | None = None
    # Integer lora_id resolved from lora_path (0 = base model)
    lora_id: int = 0


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
                    assert (
                        len(self.user_rid) == 1
                    ), "user_rid list should have length 1 for single request."
                    self.user_rid = self.user_rid[0]
                assert isinstance(self.user_rid, str), "user_rid should be a str."
            if self.sampling_params is None:
                self.sampling_params = {}
            self.sampling_params["max_new_tokens"] = 0
        else:
            if self.rid is None:
                self.rid = [uuid.uuid4().hex for _ in range(self.batch_size)]
            else:
                assert isinstance(self.rid, list), "The rid should be a list."
            if self.user_rid is None:
                self.user_rid = list(self.rid)
            elif isinstance(self.user_rid, str):
                self.user_rid = [self.user_rid] * self.batch_size
            else:
                assert (
                    isinstance(self.user_rid, list)
                    and len(self.user_rid) == self.batch_size
                ), "user_rid should be a str or a list of matching length."

            if self.sampling_params is None:
                self.sampling_params = [{}] * self.batch_size
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


@dataclass
class TokenizedEmbeddingReqInput:
    # The request id
    rid: str
    # The input text
    input_text: str
    # The input token ids
    input_ids: list[int]
    # Placeholder sampling params field so request metadata can share one shape
    # with generation-oriented code paths.
    sampling_params: SamplingParams
    # Time at object instantiated
    created_time: float


@dataclass
class BatchTokenIDOut:
    # The request id
    rids: list[str]
    # The finish reason
    finished_reasons: list[BaseFinishReason]
    # For incremental decoding
    decoded_texts: list[str]
    decode_ids: list[list[int]]
    read_offsets: list[int]
    # Only used when `--skip-tokenizer-init` is on
    output_ids: list[int] | None
    output_multi_ids: list[int] | None
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
    batch_accept_draft_tokens: list[float]

    # Store some custom information, such as decoding status in multimodal scenarios, etc.
    output_extra_infos: list[dict[str, Any]]

    generated_time: int


@dataclass
class BatchMultimodalDecodeReq:
    # The request id
    rids: list[str]


@dataclass
class BatchStrOut:
    # The request id
    rids: list[str]
    # The finish reason
    finished_reasons: list[dict]
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
    batch_accept_draft_tokens: list[float]

    # Store some custom information, such as decoding status in multimodal scenarios, etc.
    output_extra_infos: list[dict[str, Any]]

    generated_time: int


@dataclass
class BatchMultimodalOut:
    # The request id
    rids: list[str]


@dataclass
class BatchEmbeddingOut:
    # The request id
    rids: list[str]
    # The finish reason
    finished_reasons: list[BaseFinishReason]
    # The output embedding
    embeddings: list[list[float]] | list[dict]
    # Token counts
    prompt_tokens: list[int]


@dataclass
class FlushCacheReqInput:
    pass


@dataclass
class FlushCacheReqOutput:
    success: bool


@dataclass
class UpdateWeightFromDiskReqInput:
    # The model path with the new weights
    model_path: str
    # The format to load the weights
    load_format: str | None = None


@dataclass
class UpdateWeightFromDiskReqOutput:
    success: bool
    message: str
    # Number of paused requests during weight sync.
    num_paused_requests: int | None = 0


@dataclass
class UpdateWeightsFromDistributedReqInput:
    name: str
    dtype: str
    shape: list[int]


@dataclass
class UpdateWeightsFromDistributedReqOutput:
    success: bool
    message: str


@dataclass
class UpdateWeightsFromTensorReqInput:
    serialized_named_tensors: bytes  # indeed Dict[str, torch.Tensor]
    load_format: str | None
    flush_cache: bool


@dataclass
class UpdateWeightsFromTensorReqOutput:
    success: bool
    message: str


@dataclass
class InitWeightsUpdateGroupReqInput:
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


@dataclass
class InitWeightsUpdateGroupReqOutput:
    success: bool
    message: str


@dataclass
class GetWeightsByNameReqInput:
    name: str
    truncate_size: int = 100


@dataclass
class GetWeightsByNameReqOutput:
    parameter: list


@dataclass
class ReleaseMemoryOccupationReqInput:
    pass


@dataclass
class ReleaseMemoryOccupationReqOutput:
    pass


@dataclass
class ResumeMemoryOccupationReqInput:
    pass


@dataclass
class ResumeMemoryOccupationReqOutput:
    pass


@dataclass
class AbortReq:
    # The request id
    rid: str


@dataclass
class GetInternalStateReq:
    pass


@dataclass
class GetInternalStateReqOutput:
    internal_state: dict[Any, Any]


@dataclass
class SetInternalStateReq:
    server_args: dict[str, Any]


@dataclass
class SetInternalStateReqOutput:
    updated: bool
    server_args: dict[str, Any]


class ExpertDistributionReq(Enum):
    START_RECORD = 1
    STOP_RECORD = 2
    DUMP_RECORD = 3


@dataclass
class ExpertDistributionReqOutput:
    pass


class ProfileReqType(Enum):
    START_PROFILE = 1
    STOP_PROFILE = 2


@dataclass
class ProfileReq:
    type: ProfileReqType
    output_dir: str | None = None
    start_step: int | None = None
    num_steps: int | None = None
    activities: list[str] | None = None
    profile_by_stage: bool = False
    with_stack: bool | None = None
    record_shapes: bool | None = None
    profile_id: str | None = None


@dataclass
class ProfileReqOutput:
    success: bool
    message: str


@dataclass
class ConfigureLoggingReq:
    log_requests: bool | None = None
    log_requests_level: int | None = None
    dump_requests_folder: str | None = None
    dump_requests_threshold: int | None = None


@dataclass
class OpenSessionReqInput:
    capacity_of_str_len: int
    session_id: str | None = None


@dataclass
class CloseSessionReqInput:
    session_id: str


@dataclass
class OpenSessionReqOutput:
    session_id: str | None
    success: bool


@dataclass
class HealthCheckOutput:
    pass


@dataclass
class RpcReqInput:
    method: str
    parameters: dict | None = None


@dataclass
class RpcReqOutput:
    success: bool
    message: str


@dataclass
class LoadLoraReqInput:
    lora_name: str
    lora_path: str
    pinned: bool = False


@dataclass
class LoadLoraReqOutput:
    success: bool
    lora_id: int = 0
    message: str = ""


@dataclass
class UnloadLoraReqInput:
    lora_name: str


@dataclass
class UnloadLoraReqOutput:
    success: bool
    message: str = ""


@dataclass
class GetLoadReqInput(BaseReq):
    pass


@dataclass
class GetLoadReqOutput(BaseReq):
    dp_rank: int = 0
    num_reqs: int = 0
    num_waiting_reqs: int = 0
    num_pages: int = 0


@dataclass
class WatchLoadUpdateReq(BaseReq):
    loads: list[GetLoadReqOutput] = field(default_factory=list)


class BlockReqType(Enum):
    BLOCK = 1
    UNBLOCK = 2


@dataclass
class BlockReqInput(BaseReq):
    type: BlockReqType = field(default_factory=BlockReqType.BLOCK)
