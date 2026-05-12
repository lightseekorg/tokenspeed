# Adapted from meituan-longcat/SGLang-FluentLLM.
# This file has been modified for this repository.
# This file may incorporate material from ModelTC/lightllm,
# vllm-project/vllm, and sgl-project/sglang, as identified in
# python/THIRDPARTYNOTICES.
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

"""Pydantic models for OpenAI API protocol"""

import time
import uuid
from dataclasses import dataclass
from typing import Any, Literal, Union

from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseInputItemParam,
    ResponseOutputItem,
    ResponseReasoningItem,
)
from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_serializer,
    model_validator,
)


class ModelCard(BaseModel):
    """Model cards."""

    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "tokenspeed"
    root: str | None = None
    max_model_len: int | None = None


class ModelList(BaseModel):
    """Model list consists of model cards."""

    object: str = "list"
    data: list[ModelCard] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    object: str = "error"
    message: str
    type: str
    param: str | None = None
    code: int


class LogProbs(BaseModel):
    text_offset: list[int] = Field(default_factory=list)
    token_logprobs: list[float | None] = Field(default_factory=list)
    tokens: list[str] = Field(default_factory=list)
    top_logprobs: list[dict[str, float] | None] = Field(default_factory=list)


class TopLogprob(BaseModel):
    token: str
    bytes: list[int]
    logprob: float


class ChatCompletionTokenLogprob(BaseModel):
    token: str
    bytes: list[int]
    logprob: float
    top_logprobs: list[TopLogprob]


class ChoiceLogprobs(BaseModel):
    # build for v1/chat/completions response
    content: list[ChatCompletionTokenLogprob]


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: int | None = 0
    accept_draft_tokens: float | None = 0
    # only used to return cached tokens when --enable-cache-report is set
    prompt_tokens_details: dict[str, int] | None = None
    reasoning_tokens: int | None = 0


class StreamOptions(BaseModel):
    include_usage: bool | None = True


class JsonSchemaResponseFormat(BaseModel):
    name: str
    description: str | None = None
    # use alias to workaround pydantic conflict
    schema_: dict[str, object] | None = Field(alias="schema", default=None)
    strict: bool | None = False


class CompletionRequest(BaseModel):
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/completions/create
    model: str
    prompt: list[int] | list[list[int]] | str | list[str]
    best_of: int | None = None
    echo: bool = False
    frequency_penalty: float = 0.0
    logit_bias: dict[str, float] | None = None
    logprobs: int | None = None
    max_tokens: int = 16
    n: int = 1
    presence_penalty: float = 0.0
    seed: int | None = None
    stop: str | list[str] | None = None
    stream: bool = False
    stream_options: StreamOptions | None = None
    suffix: str | None = None
    temperature: float = 1.0
    top_p: float = 1.0
    user: str | None = None
    return_hidden_states: bool = False

    # Extra parameters for SRT backend only and will be ignored by OpenAI models.
    top_k: int = -1
    min_p: float = 0.0
    min_tokens: int = 0
    json_schema: str | None = None
    regex: str | None = None
    ebnf: str | None = None
    repetition_penalty: float = 1.0
    stop_token_ids: list[int] | None = None
    no_stop_trim: bool = False
    ignore_eos: bool = False
    skip_special_tokens: bool = True
    session_params: dict | None = None

    # For PD disaggregation
    bootstrap_host: list[str] | str | None = None
    bootstrap_port: list[int | None] | int | None = None
    bootstrap_room: list[int] | int | None = None

    # For request id
    rid: list[str] | str | None = None

    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens_positive(cls, v):
        if v is not None and v <= 0:
            raise ValueError("max_tokens must be positive")
        return v


class CompletionResponseChoice(BaseModel):
    index: int
    text: str
    logprobs: LogProbs | None = None
    finish_reason: Literal["stop", "length", "content_filter", "abort"] | None = None
    matched_stop: None | int | str = None
    hidden_states: object | None = None

    @model_serializer(mode="wrap")
    def _serialize(self, handler):
        data = handler(self)
        if self.hidden_states is None:
            data.pop("hidden_states", None)
        return data


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[CompletionResponseChoice]
    usage: UsageInfo


class CompletionResponseStreamChoice(BaseModel):
    index: int
    text: str
    logprobs: LogProbs | None = None
    finish_reason: Literal["stop", "length", "content_filter", "abort"] | None = None
    matched_stop: None | int | str = None
    hidden_states: object | None = None

    @model_serializer(mode="wrap")
    def _serialize(self, handler):
        data = handler(self)
        if self.hidden_states is None:
            data.pop("hidden_states", None)
        return data


class CompletionStreamResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[CompletionResponseStreamChoice]
    usage: UsageInfo | None = None


class ChatCompletionMessageContentTextPart(BaseModel):
    type: Literal["text"]
    text: str


class ChatCompletionMessageContentImageURL(BaseModel):
    url: str
    detail: Literal["auto", "low", "high"] | None = "auto"


class ChatCompletionMessageContentVideoURL(BaseModel):
    url: str


class ChatCompletionMessageContentAudioURL(BaseModel):
    url: str


class ChatCompletionMessageContentImagePart(BaseModel):
    type: Literal["image_url"]
    image_url: ChatCompletionMessageContentImageURL
    modalities: Literal["image", "multi-images", "video"] | None = "image"


class ChatCompletionMessageContentVideoPart(BaseModel):
    type: Literal["video_url"]
    video_url: ChatCompletionMessageContentVideoURL


class ChatCompletionMessageContentAudioPart(BaseModel):
    type: Literal["audio_url"]
    audio_url: ChatCompletionMessageContentAudioURL


ChatCompletionMessageContentPart = (
    ChatCompletionMessageContentTextPart
    | ChatCompletionMessageContentImagePart
    | ChatCompletionMessageContentVideoPart
    | ChatCompletionMessageContentAudioPart
)


class FunctionResponse(BaseModel):
    """Function response."""

    name: str | None = None
    arguments: str | None = None


class ToolCall(BaseModel):
    """Tool call response."""

    id: str | None = None
    index: int | None = None
    type: Literal["function"] = "function"
    function: FunctionResponse


class ChatCompletionMessageGenericParam(BaseModel):
    role: Literal["system", "assistant", "tool", "developer"]
    content: str | list[ChatCompletionMessageContentTextPart] | None = Field(
        default=None
    )
    tool_call_id: str | None = None
    name: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[ToolCall] | None = Field(default=None, examples=[None])

    @field_validator("role", mode="before")
    @classmethod
    def _normalize_role(cls, v):
        if isinstance(v, str):
            v_lower = v.lower()
            if v_lower not in {"system", "assistant", "tool", "developer"}:
                raise ValueError(
                    "'role' must be one of 'system', 'assistant', or 'tool' (case-insensitive)."
                )
            return v_lower
        raise ValueError("'role' must be a string")


class ChatCompletionMessageUserParam(BaseModel):
    role: Literal["user"]
    content: str | list[ChatCompletionMessageContentPart]


ChatCompletionMessageParam = (
    ChatCompletionMessageGenericParam | ChatCompletionMessageUserParam
)


class ResponseFormat(BaseModel):
    type: Literal["text", "json_object", "json_schema"]
    json_schema: JsonSchemaResponseFormat | None = None


class StructuresResponseFormat(BaseModel):
    begin: str
    schema_: dict[str, object] | None = Field(alias="schema", default=None)
    end: str


class StructuralTagResponseFormat(BaseModel):
    type: Literal["structural_tag"]
    structures: list[StructuresResponseFormat]
    triggers: list[str]


class Function(BaseModel):
    """Function descriptions."""

    description: str | None = Field(default=None, examples=[None])
    name: str | None = None
    parameters: object | None = None
    strict: bool = False


class Tool(BaseModel):
    """Function wrapper."""

    type: str = Field(default="function", examples=["function"])
    function: Function


class ToolChoiceFuncName(BaseModel):
    """The name of tool choice function."""

    name: str | None = None


class ToolChoice(BaseModel):
    """The tool choice definition."""

    function: ToolChoiceFuncName
    type: Literal["function"] = Field(default="function", examples=["function"])


class ChatCompletionRequest(BaseModel):
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/chat/create
    messages: list[ChatCompletionMessageParam]
    model: str
    frequency_penalty: float = 0.0
    logit_bias: dict[str, float] | None = None
    logprobs: bool = False
    top_logprobs: int | None = None
    max_tokens: int | None = Field(
        default=None,
        deprecated="max_tokens is deprecated in favor of the max_completion_tokens field",
        description="The maximum number of tokens that can be generated in the chat completion. ",
    )
    max_completion_tokens: int | None = Field(
        default=None,
        description="The maximum number of completion tokens for a chat completion request, "
        "including visible output tokens and reasoning tokens. Input tokens are not included. ",
    )
    n: int = 1
    presence_penalty: float = 0.0
    response_format: ResponseFormat | StructuralTagResponseFormat | None = None
    seed: int | None = None
    stop: str | list[str] | None = None
    stream: bool = False
    stream_options: StreamOptions | None = StreamOptions()
    temperature: float = 0.7
    top_p: float = 1.0
    user: str | None = None
    tools: list[Tool] | None = Field(default=None, examples=[None])
    tool_choice: ToolChoice | Literal["auto", "required", "none", "bypass_check"] = (
        Field(default="auto", examples=["none"])
    )  # noqa
    return_hidden_states: bool = False
    reasoning_effort: Literal["low", "medium", "high"] | None = Field(
        default="medium",
        description="Constrains effort on reasoning for reasoning models. "
        "'low' is the least effort, 'high' is the most effort. Reducing reasoning effort can "
        "result in faster responses and fewer tokens used on reasoning in a response. "
        "Currently only supported for OpenAI models.",
    )

    @model_validator(mode="before")
    @classmethod
    def set_tool_choice_default(cls, values):
        if values.get("tool_choice") is None:
            if values.get("tools") is None:
                values["tool_choice"] = "none"
            else:
                values["tool_choice"] = "auto"
        return values

    # Extra parameters for SRT backend only and will be ignored by OpenAI models.
    top_k: int = -1
    min_p: float = 0.0
    min_tokens: int = 0
    regex: str | None = None
    ebnf: str | None = None
    repetition_penalty: float = 1.0
    stop_token_ids: list[int] | None = None
    no_stop_trim: bool = False
    ignore_eos: bool = False
    continue_final_message: bool = False
    skip_special_tokens: bool = True
    session_params: dict | None = None
    separate_reasoning: bool = True
    stream_reasoning: bool = True
    chat_template_kwargs: dict | None = None

    # For request id
    rid: list[str] | str | None = None

    # For PD disaggregation
    bootstrap_host: str | None = None
    bootstrap_port: int | None = None
    bootstrap_room: int | None = None


class ChatMessage(BaseModel):
    role: str | None = None
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[ToolCall] | None = Field(default=None, examples=[None])


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    logprobs: LogProbs | ChoiceLogprobs | None = None
    finish_reason: (
        Literal[
            "stop", "length", "tool_calls", "content_filter", "function_call", "abort"
        ]
        | None
    ) = None
    matched_stop: None | int | str = None
    hidden_states: object | None = None

    @model_serializer(mode="wrap")
    def _serialize(self, handler):
        data = handler(self)
        if self.hidden_states is None:
            data.pop("hidden_states", None)
        return data


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionResponseChoice]
    usage: UsageInfo


class DeltaMessage(BaseModel):
    role: str | None = None
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[ToolCall] | None = Field(default=None, examples=[None])
    hidden_states: object | None = None

    @model_serializer(mode="wrap")
    def _serialize(self, handler):
        data = handler(self)
        if self.hidden_states is None:
            data.pop("hidden_states", None)
        return data


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    logprobs: LogProbs | ChoiceLogprobs | None = None
    finish_reason: (
        Literal[
            "stop", "length", "tool_calls", "content_filter", "function_call", "abort"
        ]
        | None
    ) = None
    matched_stop: None | int | str = None


class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionResponseStreamChoice]
    usage: UsageInfo | None = None


OpenAIServingRequest = ChatCompletionRequest | CompletionRequest


# Response API protocol definitions
class ResponseReasoningParam(BaseModel):
    """Reasoning parameters for responses."""

    effort: Literal["low", "medium", "high"] | None = Field(
        default="medium",
        description="Constrains effort on reasoning for reasoning models.",
    )


class ResponseTool(BaseModel):
    """Tool definition for responses."""

    type: Literal["web_search_preview", "code_interpreter"] = Field(
        description="Type of tool to enable"
    )


ResponseInputOutputItem = Union[
    ResponseInputItemParam,
    "ResponseReasoningItem",
    ResponseFunctionToolCall,
]


class ResponsesRequest(BaseModel):
    """Request body for v1/responses endpoint."""

    # Core OpenAI API fields (ordered by official documentation)
    background: bool | None = False
    include: (
        list[
            Literal[
                "code_interpreter_call.outputs",
                "computer_call_output.output.image_url",
                "file_search_call.results",
                "message.input_image.image_url",
                "message.output_text.logprobs",
                "reasoning.encrypted_content",
            ]
        ]
        | None
    ) = None
    input: str | list[ResponseInputOutputItem]
    instructions: str | None = None
    max_output_tokens: int | None = None
    max_tool_calls: int | None = None
    metadata: dict[str, Any] | None = None
    model: str | None = None
    parallel_tool_calls: bool | None = True
    previous_response_id: str | None = None
    reasoning: ResponseReasoningParam | None = None
    service_tier: Literal["auto", "default", "flex", "scale", "priority"] = "auto"
    store: bool | None = True
    stream: bool | None = False
    temperature: float | None = None
    tool_choice: Literal["auto", "required", "none", "bypass_check"] = "auto"
    tools: list[ResponseTool] = Field(default_factory=list)
    top_logprobs: int | None = 0
    top_p: float | None = None
    truncation: Literal["auto", "disabled"] | None = "disabled"
    user: str | None = None

    # Extra TokenSpeed parameters
    request_id: str = Field(
        default_factory=lambda: f"resp_{uuid.uuid4().hex}",
        description="The request_id related to this request. If the caller does not set it, a random uuid will be generated.",
    )
    priority: int = Field(default=0, description="Request priority")

    # TokenSpeed-specific sampling parameters
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: str | list[str] | None = None
    top_k: int = -1
    min_p: float = 0.0
    repetition_penalty: float = 1.0

    # Default sampling parameters
    _DEFAULT_SAMPLING_PARAMS = {
        "temperature": 0.7,
        "top_p": 1.0,
        "top_k": -1,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
    }

    def to_sampling_params(
        self, default_max_tokens: int, default_params: dict | None = None
    ) -> dict[str, Any]:
        """Convert to sampling parameters for generation."""
        if default_params is None:
            default_params = {}

        # Use max_output_tokens if available, otherwise use max_tokens for backwards compatibility
        if self.max_output_tokens is not None:
            max_tokens = min(self.max_output_tokens, default_max_tokens)
        else:
            max_tokens = default_max_tokens

        # Avoid exceed the context length by minus 1 token
        max_tokens -= 1

        # Get parameters with defaults
        temperature = self.temperature
        if temperature is None:
            temperature = default_params.get(
                "temperature", self._DEFAULT_SAMPLING_PARAMS["temperature"]
            )

        top_p = self.top_p
        if top_p is None:
            top_p = default_params.get("top_p", self._DEFAULT_SAMPLING_PARAMS["top_p"])

        params = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "stop": self.stop,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "repetition_penalty": self.repetition_penalty,
        }

        # Apply any additional default parameters
        for key, value in default_params.items():
            if key not in params or params[key] is None:
                params[key] = value

        return params


class PromptTokenUsageInfo(BaseModel):
    """Prompt token usage details."""

    cached_tokens: int = 0


class ResponsesResponse(BaseModel):
    """Response body for v1/responses endpoint."""

    id: str = Field(default_factory=lambda: f"resp_{time.time()}")
    object: Literal["response"] = "response"
    created_at: int = Field(default_factory=lambda: int(time.time()))
    model: str

    output: list[
        ResponseOutputItem | ResponseReasoningItem | ResponseFunctionToolCall
    ] = Field(default_factory=list)
    status: Literal["queued", "in_progress", "completed", "failed", "cancelled"]
    usage: UsageInfo | None = None
    parallel_tool_calls: bool = True
    tool_choice: str = "auto"
    tools: list[ResponseTool] = Field(default_factory=list)

    @classmethod
    def from_request(
        cls,
        request: ResponsesRequest,
        sampling_params: Any,
        model_name: str,
        created_time: int,
        output: list[
            ResponseOutputItem | ResponseReasoningItem | ResponseFunctionToolCall
        ],
        status: str,
        usage: UsageInfo | None,
    ) -> "ResponsesResponse":
        """Create a response from a request."""
        return cls(
            id=request.request_id,
            created_at=created_time,
            model=model_name,
            output=output,
            status=status,
            usage=usage,
            parallel_tool_calls=request.parallel_tool_calls or True,
            tool_choice=request.tool_choice,
            tools=request.tools,
        )


class RequestResponseMetadata(BaseModel):
    """Metadata for request/response tracking."""

    request_id: str
    final_usage_info: UsageInfo | None = None


@dataclass
class MessageProcessingResult:
    """Result of processing chat messages and applying templates.

    This dataclass encapsulates all the outputs from message processing including
    prompt generation, multimodal data extraction, and constraint preparation.
    Used internally by OpenAIServingChat to pass processed data between methods.

    Args:
        prompt: The final text prompt after applying chat template
        prompt_ids: Either the text prompt (str) or tokenized IDs (List[int])
        image_data: Extracted image data from messages, if any
        audio_data: Extracted audio data from messages, if any
        modalities: List of modality types present in the messages
        stop: Combined stop strings from template and request
        tool_call_constraint: Optional constraint for structured tool calls
    """

    prompt: str
    prompt_ids: str | list[int]
    image_data: Any | None
    audio_data: Any | None
    video_data: Any | None
    modalities: list[str]
    stop: list[str]
    tool_call_constraint: Any | None = None


class ResponseReasoningTextContent(BaseModel):
    text: str
    type: Literal["reasoning_text"] = "reasoning_text"


ResponseInputOutputItem = Union[
    ResponseInputItemParam, "ResponseReasoningItem", ResponseFunctionToolCall
]
