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

import copy
import json
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Any, Literal

import jinja2
import orjson
from fastapi import Request
from fastapi.responses import ORJSONResponse, StreamingResponse

from tokenspeed.runtime.engine.io_struct import GenerateReqInput
from tokenspeed.runtime.engine.protocol import EngineClient
from tokenspeed.runtime.engine.request_types import ABORT_CODE
from tokenspeed.runtime.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatCompletionTokenLogprob,
    ChatMessage,
    ChoiceLogprobs,
    DeltaMessage,
    ErrorResponse,
    FunctionResponse,
    LogProbs,
    MessageProcessingResult,
    ToolCall,
    ToolChoice,
    TopLogprob,
)
from tokenspeed.runtime.entrypoints.openai.serving_base import (
    OpenAIServingBase,
    await_with_disconnect_watchdog,
)
from tokenspeed.runtime.entrypoints.openai.usage_processor import UsageProcessor
from tokenspeed.runtime.entrypoints.openai.utils import (
    process_hidden_states_from_ret,
    to_openai_style_logprobs,
    validate_sampling_params,
)
from tokenspeed.runtime.grammar.function_call_parser import FunctionCallParser
from tokenspeed.runtime.grammar.reasoning_structural_tag import (
    structural_tag_for_reasoning_json_schema,
)
from tokenspeed.runtime.inputs.conversation import generate_chat_conv
from tokenspeed.runtime.inputs.jinja_template_utils import (
    process_content_for_template_format,
)
from tokenspeed.runtime.inputs.reasoning_parser import ReasoningParser
from tokenspeed.runtime.inputs.template_manager import TemplateManager
from tokenspeed.runtime.sampling.sampling_params import SamplingParams
from tokenspeed.runtime.utils import convert_json_schema_to_str, get_colorful_logger

logger = get_colorful_logger(__name__)


def _resolve_chat_template_for_tools(
    template: Any, tokenizer_chat_template: Any = None
) -> str | None:
    """Reduce a chat-template value to the template body transformers will
    render for a tool request — mirrors transformers' own resolution.

    Two callers:
    * At init, ``template`` is the tokenizer's own ``chat_template`` and
      ``tokenizer_chat_template`` is unused.
    * On a per-request override, ``template`` is the override and
      ``tokenizer_chat_template`` is the tokenizer's named-template
      registry (a dict / list-of-dicts), used to look up *named* string
      overrides like ``{"chat_template": "tool_use"}``.

    Resolution order:
      1. String + name match in ``tokenizer_chat_template`` → that entry.
      2. String otherwise → treat as the template body.
      3. Dict / list → tools-path selection: ``tool_use`` → ``default`` →
         singleton entry. Same logic transformers applies when ``tools``
         is present.
    """
    if isinstance(template, str):
        if isinstance(tokenizer_chat_template, dict):
            body = tokenizer_chat_template.get(template)
            if isinstance(body, str):
                return body
        elif isinstance(tokenizer_chat_template, list):
            for entry in tokenizer_chat_template:
                if (
                    isinstance(entry, dict)
                    and entry.get("name") == template
                    and isinstance(entry.get("template"), str)
                ):
                    return entry["template"]
        return template
    if isinstance(template, dict):
        for key in ("tool_use", "default"):
            entry = template.get(key)
            if isinstance(entry, str):
                return entry
        if len(template) == 1:
            (only,) = template.values()
            if isinstance(only, str):
                return only
    if isinstance(template, list):
        by_name = {}
        for entry in template:
            if (
                isinstance(entry, dict)
                and isinstance(entry.get("name"), str)
                and isinstance(entry.get("template"), str)
            ):
                by_name[entry["name"]] = entry["template"]
        for key in ("tool_use", "default"):
            if key in by_name:
                return by_name[key]
        if len(by_name) == 1:
            return next(iter(by_name.values()))
    return None


def _chat_template_expects_wrapped_tools(template_source: str | None) -> bool:
    """Decide whether a Jinja chat template reads tools in the OpenAI-wrapped
    form (``tool.function.X``) or the flat form (``tool.X``).

    Returning the wrong shape lets the template render but emits a tool
    schema the model was not trained on — silent quality degradation. We
    avoid that by scanning the template AST once and dispatching on the
    actual access pattern, then caching by template source.

    Tracks aliasing so that templates which indirect through
    ``{% set %}`` rebinds or call macros (MiniMax-M2's
    ``render_tool_namespace`` does the latter) still resolve correctly.
    """
    if not isinstance(template_source, str) or not template_source:
        return False
    try:
        ast = jinja2.Environment().parse(template_source)
    except Exception:
        return False

    macros = {
        m.name: [a.name for a in m.args] for m in ast.find_all(jinja2.nodes.Macro)
    }

    # Names bound to the tools *list* — start with ``tools`` itself, then
    # propagate through ``{% set X = Y %}`` rebinds and macro parameters
    # whose macro is called with a list-alias positional argument.
    list_aliases: set[str] = {"tools"}
    for _ in range(8):  # fixpoint guard; bounded because aliases only grow
        before = len(list_aliases)
        for assign in ast.find_all(jinja2.nodes.Assign):
            if (
                isinstance(assign.node, jinja2.nodes.Name)
                and assign.node.name in list_aliases
                and isinstance(assign.target, jinja2.nodes.Name)
            ):
                list_aliases.add(assign.target.name)
        for call in ast.find_all(jinja2.nodes.Call):
            if not (
                isinstance(call.node, jinja2.nodes.Name) and call.node.name in macros
            ):
                continue
            params = macros[call.node.name]
            for i, arg in enumerate(call.args):
                if (
                    isinstance(arg, jinja2.nodes.Name)
                    and arg.name in list_aliases
                    and i < len(params)
                ):
                    list_aliases.add(params[i])
        if len(list_aliases) == before:
            break

    # Names bound to a tool *element* — produced by ``for X in <list_alias>``.
    elem_aliases: set[str] = set()
    for for_node in ast.find_all(jinja2.nodes.For):
        if (
            isinstance(for_node.iter, jinja2.nodes.Name)
            and for_node.iter.name in list_aliases
            and isinstance(for_node.target, jinja2.nodes.Name)
        ):
            elem_aliases.add(for_node.target.name)

    if not elem_aliases:
        return False

    # Any ``elem.function`` access means the template wants the wrapped shape.
    for getattr_node in ast.find_all(jinja2.nodes.Getattr):
        if (
            isinstance(getattr_node.node, jinja2.nodes.Name)
            and getattr_node.node.name in elem_aliases
            and getattr_node.attr == "function"
        ):
            return True
    return False


class OpenAIServingChat(OpenAIServingBase):
    """Handler for /v1/chat/completions requests"""

    def __init__(
        self,
        engine_client: EngineClient,
        template_manager: TemplateManager,
    ):
        super().__init__(engine_client)
        self.template_manager = template_manager
        self.tool_call_parser = self.engine_client.server_args.tool_call_parser
        # Inspect the chat template once at startup so per-request dispatch is
        # a constant-time bool lookup. The tokenizer's ``chat_template`` may
        # be a string, a dict (``{"tool_use": ..., "default": ...}``), or a
        # list-of-dicts; the resolver mirrors transformers' tools-path
        # selection so we score the entry that will actually render.
        self._tools_use_wrapped_shape = _chat_template_expects_wrapped_tools(
            _resolve_chat_template_for_tools(
                getattr(self.engine_client.tokenizer, "chat_template", None)
            )
        )

    def _validate_request(self, request: ChatCompletionRequest) -> str | None:
        """Validate that the input is valid."""
        if not request.messages:
            return "Messages cannot be empty."

        if (
            isinstance(request.tool_choice, str)
            and request.tool_choice.lower() == "required"
            and not request.tools
        ):
            return "Tools cannot be empty if tool choice is set to required."

        max_output_tokens = request.max_completion_tokens or request.max_tokens
        server_context_length = self.engine_client.server_args.max_model_len
        if (
            max_output_tokens
            and server_context_length
            and max_output_tokens > server_context_length
        ):
            return (
                f"max_completion_tokens is too large: {max_output_tokens}."
                f"This model supports at most {server_context_length} completion tokens."
            )

        return None

    def _convert_to_internal_request(
        self,
        request: ChatCompletionRequest,
    ) -> tuple[GenerateReqInput, ChatCompletionRequest]:
        """Convert OpenAI chat completion request to internal format"""
        is_multimodal = self.engine_client.model_config.is_multimodal

        # Process messages and apply chat template
        processed_messages = self._process_messages(request, is_multimodal)
        # Build sampling parameters
        sampling_params = self._build_sampling_params(
            request,
            processed_messages.stop,
            processed_messages.tool_call_constraint,
        )

        # Handle single vs multiple requests
        if is_multimodal:
            prompt_kwargs = {"text": processed_messages.prompt}
        else:
            if isinstance(processed_messages.prompt_ids, str):
                prompt_kwargs = {"text": processed_messages.prompt_ids}
            else:
                prompt_kwargs = {"input_ids": processed_messages.prompt_ids}

        adapted_request = GenerateReqInput(
            **prompt_kwargs,
            image_data=processed_messages.image_data,
            video_data=processed_messages.video_data,
            audio_data=processed_messages.audio_data,
            sampling_params=sampling_params,
            return_logprob=request.logprobs,
            logprob_start_len=-1,
            top_logprobs_num=request.top_logprobs or 0,
            stream=request.stream,
            return_text_in_logprobs=True,
            modalities=processed_messages.modalities,
            bootstrap_host=request.bootstrap_host,
            bootstrap_port=request.bootstrap_port,
            bootstrap_room=request.bootstrap_room,
            return_hidden_states=request.return_hidden_states,
            user_rid=request.rid,
            lora_path=request.lora_path,
        )

        return adapted_request, request

    def _process_messages(
        self, request: ChatCompletionRequest, is_multimodal: bool
    ) -> MessageProcessingResult:
        """Process chat messages and apply chat template"""

        # Apply chat template and its stop strings
        tools = None
        tool_call_constraint = None
        # gpt-oss reasoning detector keys off the literal ``<|end|>`` special
        # token in the detokenized stream, so detokenization must keep
        # special tokens. Same reason as the tools branch below.
        if self.engine_client.server_args.reasoning_parser == "gpt-oss":
            request.skip_special_tokens = False
        if request.tools and request.tool_choice != "none":
            request.skip_special_tokens = False

            tools = request.tools
            if not isinstance(request.tool_choice, str):
                tools = [
                    item
                    for item in tools
                    if item.function.name == request.tool_choice.function.name
                ]
            # Wrapped (``tool.function.X``: MiniMax-M2, DeepSeek) vs flat
            # (``tool.X``: Llama-3, Hermes, Qwen, gpt-oss). __init__ scored
            # the tokenizer's default template; an explicit per-request
            # override (which may be a body or a *name* into the tokenizer's
            # template registry) needs re-resolution.
            override = (request.chat_template_kwargs or {}).get("chat_template")
            if override is None:
                use_wrapped = self._tools_use_wrapped_shape
            else:
                use_wrapped = _chat_template_expects_wrapped_tools(
                    _resolve_chat_template_for_tools(
                        override,
                        getattr(self.engine_client.tokenizer, "chat_template", None),
                    )
                )
            if use_wrapped:
                tools = [item.model_dump() for item in tools]
            else:
                tools = [item.function.model_dump() for item in tools]

            if self.tool_call_parser:
                parser = FunctionCallParser(request.tools, self.tool_call_parser)
                tool_call_constraint = parser.get_structure_constraint(
                    request.tool_choice
                )

        # Use chat template
        if self.template_manager.chat_template_name is None:
            result = self._apply_jinja_template(request, tools, is_multimodal)
        else:
            result = self._apply_conversation_template(request, is_multimodal)

        result.tool_call_constraint = tool_call_constraint
        return result

    def _apply_jinja_template(
        self,
        request: ChatCompletionRequest,
        tools: list[dict] | None,
        is_multimodal: bool,
    ) -> MessageProcessingResult:
        """Apply Jinja chat template"""
        prompt = ""
        prompt_ids = []
        openai_compatible_messages = []
        image_data = []
        video_data = []
        audio_data = []
        modalities = []

        template_content_format = self.template_manager.jinja_template_content_format

        for message in request.messages:
            if message.content is None:
                message.content = ""
            msg_dict = message.model_dump()

            # Process content based on detected template format
            processed_msg = process_content_for_template_format(
                msg_dict,
                template_content_format,
                image_data,
                video_data,
                audio_data,
                modalities,
            )

            openai_compatible_messages.append(processed_msg)

        # Handle assistant prefix for continue_final_message
        assistant_prefix = None
        if (
            openai_compatible_messages
            and openai_compatible_messages[-1]["role"] == "assistant"
        ):
            if request.continue_final_message:
                assistant_prefix = openai_compatible_messages[-1]["content"]
                openai_compatible_messages = openai_compatible_messages[:-1]

        try:
            prompt_ids = self.engine_client.tokenizer.apply_chat_template(
                openai_compatible_messages,
                tokenize=True,
                add_generation_prompt=True,
                tools=tools,
                reasoning_effort=request.reasoning_effort,
                builtin_tools=[],
                return_dict=False,
                **(
                    request.chat_template_kwargs if request.chat_template_kwargs else {}
                ),
            )
        except Exception:
            for processed_msg in openai_compatible_messages:
                if (
                    processed_msg["role"] == "assistant"
                    and "tool_calls" in processed_msg
                    and isinstance(processed_msg["tool_calls"], list)
                ):
                    for item in processed_msg["tool_calls"]:
                        if "arguments" in item["function"] and isinstance(
                            item["function"]["arguments"], str
                        ):
                            try:
                                arguments = item["function"]["arguments"]
                                if not arguments or arguments.strip() in (
                                    "",
                                    "null",
                                    "None",
                                ):
                                    item["function"]["arguments"] = {}
                                else:
                                    arguments = arguments.strip()
                                    item["function"]["arguments"] = orjson.loads(
                                        arguments
                                    )
                            except orjson.JSONDecodeError as e:
                                logger.warning(
                                    "function_call arguments JSON parse error: %s | Original request arguments: %s",
                                    e,
                                    repr(arguments),
                                )
                                raise ValueError(
                                    f"function_call arguments JSON parse error: {e} | Original request arguments: {repr(arguments[:200])}"
                                )
            try:
                prompt_ids = self.engine_client.tokenizer.apply_chat_template(
                    openai_compatible_messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    tools=tools,
                    reasoning_effort=request.reasoning_effort,
                    builtin_tools=[],
                    return_dict=False,
                    **(
                        request.chat_template_kwargs
                        if request.chat_template_kwargs
                        else {}
                    ),
                )
            except Exception as e:
                logger.warning(
                    "apply_chat_template error: %s| Original request: openai_compatible_messages=%r",
                    e,
                    openai_compatible_messages,
                )
                raise RuntimeError(
                    f"apply_chat_template error: {e} | Please check request data format"
                )

        if assistant_prefix:
            encoded = self.engine_client.tokenizer.encode(assistant_prefix)
            if encoded and encoded[0] == self.engine_client.tokenizer.bos_token_id:
                encoded = encoded[1:]
            prompt_ids += encoded

        if is_multimodal:
            prompt = self.engine_client.tokenizer.decode(prompt_ids)

        stop = request.stop
        image_data = image_data if image_data else None
        audio_data = audio_data if audio_data else None
        video_data = video_data if video_data else None
        modalities = modalities if modalities else []
        return MessageProcessingResult(
            prompt=prompt,
            prompt_ids=prompt_ids,
            image_data=image_data,
            video_data=video_data,
            audio_data=audio_data,
            modalities=modalities,
            stop=stop,
        )

    def _apply_conversation_template(
        self,
        request: ChatCompletionRequest,
        is_multimodal: bool,
    ) -> MessageProcessingResult:
        """Apply conversation template"""
        prompt = ""
        prompt_ids = []
        conv = generate_chat_conv(request, self.template_manager.chat_template_name)

        # If we should continue the final assistant message, adjust the conversation.
        if (
            request.continue_final_message
            and request.messages
            and request.messages[-1].role == "assistant"
        ):
            # Remove the auto-added blank assistant turn, if present.
            if conv.messages and conv.messages[-1][1] is None:
                conv.messages.pop()
            # Rebuild the prompt from the conversation.
            prompt = conv.get_prompt()
            # Strip trailing stop tokens or separators that indicate end-of-assistant.
            if isinstance(conv.stop_str, list):
                for stop_token in conv.stop_str:
                    if prompt.endswith(stop_token):
                        prompt = prompt[: -len(stop_token)]
            elif isinstance(conv.stop_str, str) and prompt.endswith(conv.stop_str):
                prompt = prompt[: -len(conv.stop_str)]
            if conv.sep and prompt.endswith(conv.sep):
                prompt = prompt[: -len(conv.sep)]
            if getattr(conv, "sep2", None) and prompt.endswith(conv.sep2):
                prompt = prompt[: -len(conv.sep2)]
        else:
            prompt = conv.get_prompt()
            if (
                hasattr(request, "chat_template_kwargs")
                and request.chat_template_kwargs
            ):
                if self._get_enable_thinking_from_request(request):
                    raise RuntimeError("Should never be here.")

        image_data = conv.image_data if conv.image_data else None
        video_data = conv.video_data if conv.video_data else None
        audio_data = conv.audio_data if conv.audio_data else None
        modalities = conv.modalities if conv.modalities else []
        stop = copy.copy(conv.stop_str or [] if not request.ignore_eos else [])

        if request.stop:
            if isinstance(request.stop, str):
                stop.append(request.stop)
            else:
                stop.extend(request.stop)

        if not is_multimodal:
            prompt_ids = self.engine_client.tokenizer.encode(prompt)

        return MessageProcessingResult(
            prompt=prompt,
            prompt_ids=prompt_ids,
            image_data=image_data,
            video_data=video_data,
            audio_data=audio_data,
            modalities=modalities,
            stop=stop,
        )

    def _build_sampling_params(
        self,
        request: ChatCompletionRequest,
        stop: list[str],
        tool_call_constraint: Any | None,
    ) -> dict[str, Any]:
        """Build sampling parameters for the request"""

        sampling_params = {
            "temperature": request.temperature,
            "max_new_tokens": request.max_tokens or request.max_completion_tokens,
            "min_new_tokens": request.min_tokens,
            "stop": stop,
            "stop_token_ids": request.stop_token_ids,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "min_p": request.min_p,
            "presence_penalty": request.presence_penalty,
            "frequency_penalty": request.frequency_penalty,
            "repetition_penalty": request.repetition_penalty,
            "regex": request.regex,
            "ebnf": request.ebnf,
            "n": request.n,
            "no_stop_trim": request.no_stop_trim,
            "ignore_eos": request.ignore_eos,
            "skip_special_tokens": request.skip_special_tokens,
            "logit_bias": request.logit_bias,
            "seed": request.seed,
        }

        if request.response_format and request.response_format.type == "json_schema":
            sampling_params["json_schema"] = convert_json_schema_to_str(
                request.response_format.json_schema.schema_
            )
        elif request.response_format and request.response_format.type == "json_object":
            sampling_params["json_schema"] = '{"type": "object"}'
        elif (
            request.response_format and request.response_format.type == "structural_tag"
        ):
            sampling_params["structural_tag"] = convert_json_schema_to_str(
                request.response_format.model_dump(by_alias=True)
            )

        # When a reasoning parser is configured, wrap the user's JSON
        # schema in xgrammar's builtin structural tag for that model so
        # JSON enforcement is scoped to the response channel only —
        # reasoning content stays free-form. Falls through unchanged for
        # parsers without an xgrammar mapping.
        rp: str | None = self.engine_client.server_args.reasoning_parser
        if rp is not None and sampling_params.get("json_schema") is not None:
            user_schema: Any = json.loads(sampling_params["json_schema"])
            wrapped: str | None = structural_tag_for_reasoning_json_schema(
                rp, user_schema
            )
            if wrapped is not None:
                sampling_params.pop("json_schema")
                sampling_params["structural_tag"] = wrapped

        # Check if there are already existing output constraints
        has_existing_constraints = (
            sampling_params.get("regex")
            or sampling_params.get("ebnf")
            or sampling_params.get("structural_tag")
            or sampling_params.get("json_schema")
        )

        if tool_call_constraint and has_existing_constraints:
            logger.warning("Constrained decoding is not compatible with tool calls.")
        elif tool_call_constraint:
            constraint_type, constraint_value = tool_call_constraint
            if constraint_type == "structural_tag":
                sampling_params[constraint_type] = convert_json_schema_to_str(
                    constraint_value.model_dump(by_alias=True)
                )
            elif constraint_type == "json_schema":
                sampling_params[constraint_type] = convert_json_schema_to_str(
                    constraint_value
                )
            else:
                sampling_params[constraint_type] = constraint_value

        try:
            # Sampling parameter type check.
            n_value = sampling_params.pop("n", 1)
            validate_sampling_params(sampling_params, SamplingParams)
            SamplingParams(**sampling_params)
            sampling_params["n"] = n_value
        except ValueError:
            raise

        return sampling_params

    async def _handle_streaming_request(
        self,
        adapted_request: GenerateReqInput,
        request: ChatCompletionRequest,
        raw_request: Request,
    ) -> StreamingResponse:
        """Handle streaming chat completion request"""
        return StreamingResponse(
            self._generate_chat_stream(adapted_request, request, raw_request),
            media_type="text/event-stream",
        )

    async def _generate_chat_stream(
        self,
        adapted_request: GenerateReqInput,
        request: ChatCompletionRequest,
        raw_request: Request,
    ) -> AsyncGenerator[str, None]:
        """Generate streaming chat completion response"""
        # Parsers for tool calls and reasoning
        parser_dict = {}
        reasoning_parser_dict = {}

        # State tracking for streaming
        is_firsts = {}
        stream_buffers = {}
        has_tool_calls = {}
        finish_reasons = {}

        # Usage tracking
        prompt_tokens = {}
        completion_tokens = {}
        cached_tokens = {}
        spec_verify_tokens = {}
        hidden_states = {}

        if self.engine_client.server_args.speculative_algorithm is not None:
            adapted_request.return_logprob = None
            adapted_request.top_logprobs_num = None
            request.logprobs = False
            request.top_logprobs = None
            if raw_request is not None:
                raw_request.logprobs = False
                raw_request.top_logprobs = None

        try:
            async for content in self.engine_client.generate_request(adapted_request):
                index = content.get("index", 0)

                prompt_tokens[index] = content["meta_info"]["prompt_tokens"]
                completion_tokens[index] = content["meta_info"]["completion_tokens"]
                cached_tokens[index] = content["meta_info"].get("cached_tokens", 0)
                spec_verify_tokens[index] = content["meta_info"].get(
                    "spec_verify_ct", 0
                )
                hidden_states[index] = content["meta_info"].get("hidden_states", None)

                # Handle logprobs
                choice_logprobs = None
                if request.logprobs:
                    choice_logprobs = self._process_streaming_logprobs(content, 0)

                finish_reason = content["meta_info"]["finish_reason"]
                finish_reason_type = finish_reason["type"] if finish_reason else None

                # Track finish_reason for each index
                if finish_reason_type:
                    finish_reasons[index] = finish_reason
                    if finish_reason_type == "abort" and finish_reason["err_type"] in [
                        ABORT_CODE.TransferFailed
                    ]:
                        error = self.create_streaming_error_response(
                            message=finish_reason["message"],
                            err_type=finish_reason["err_type"].name,
                            status_code=finish_reason["err_type"].value,
                        )
                        yield f"data: {error}\n\n"
                        break

                # First chunk with role
                if is_firsts.get(index, True):
                    is_firsts[index] = False
                    delta = DeltaMessage(role="assistant", content="")
                    choice_data = ChatCompletionResponseStreamChoice(
                        index=index,
                        delta=delta,
                        finish_reason=None,
                        logprobs=None,
                    )
                    chunk = ChatCompletionStreamResponse(
                        id=content["meta_info"]["id"],
                        created=int(time.time()),
                        choices=[choice_data],
                        model=request.model,
                    )
                    yield f"data: {chunk.model_dump_json()}\n\n"

                stream_buffer = stream_buffers.get(index, "")
                delta = content["text"][len(stream_buffer) :]
                stream_buffers[index] = stream_buffer + delta

                # Handle reasoning content
                # gpt-oss output is always multi-channel with special tokens
                # (skip_special_tokens=False), so the raw delta is never
                # directly usable. Always run the parser; suppress the
                # reasoning_content frame when separate_reasoning=False.
                rp_name = self.engine_client.server_args.reasoning_parser
                must_parse_for_clean_content = rp_name == "gpt-oss"
                if rp_name and (
                    request.separate_reasoning or must_parse_for_clean_content
                ):
                    reasoning_text, delta = self._process_reasoning_stream(
                        index, delta, reasoning_parser_dict, content, request
                    )
                    if reasoning_text and request.separate_reasoning:
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=index,
                            delta=DeltaMessage(reasoning_content=reasoning_text),
                            finish_reason=None,
                        )
                        chunk = ChatCompletionStreamResponse(
                            id=content["meta_info"]["id"],
                            created=int(time.time()),
                            choices=[choice_data],
                            model=request.model,
                        )
                        yield f"data: {chunk.model_dump_json()}\n\n"

                # Handle tool calls
                if request.tool_choice != "none" and request.tools:
                    async for chunk in self._process_tool_call_stream(
                        index,
                        delta,
                        parser_dict,
                        content,
                        request,
                        has_tool_calls,
                    ):
                        if chunk:
                            yield chunk

                    # Send any remaining tool call arguments when generation finishes
                    if finish_reason_type is not None and index in parser_dict:
                        parser = parser_dict[index]
                        remaining_chunk = self._check_for_unstreamed_tool_args(
                            parser, content, request, index
                        )
                        if remaining_chunk:
                            yield remaining_chunk
                else:
                    # Regular content
                    if delta:
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=index,
                            delta=DeltaMessage(content=delta),
                            finish_reason=None,
                            matched_stop=None,
                            logprobs=choice_logprobs,
                        )
                        chunk = ChatCompletionStreamResponse(
                            id=content["meta_info"]["id"],
                            created=int(time.time()),
                            choices=[choice_data],
                            model=request.model,
                        )
                        yield f"data: {chunk.model_dump_json()}\n\n"

            # Send finish_reason chunks for each index that completed
            for idx, finish_reason_data in finish_reasons.items():
                finish_reason_type = finish_reason_data["type"]

                # Change finish_reason to "tool_calls" if we had tool calls and stopped naturally
                final_finish_reason = finish_reason_type
                if has_tool_calls.get(idx, False) and finish_reason_type == "stop":
                    final_finish_reason = "tool_calls"

                usage = None
                if final_finish_reason is not None:
                    # Internal streaming interface requires finish_reason && usage in same package return, also keep logic for subsequent usage separate return
                    usage = UsageProcessor.calculate_streaming_usage(
                        prompt_tokens,
                        completion_tokens,
                        cached_tokens,
                        spec_verify_tokens,
                        n_choices=request.n,
                        enable_cache_report=self.engine_client.server_args.enable_cache_report,
                    )

                finish_reason_chunk = ChatCompletionStreamResponse(
                    id=content["meta_info"][
                        "id"
                    ],  #  openai uses the same chatcmpl-id for all indices
                    created=int(time.time()),
                    choices=[
                        ChatCompletionResponseStreamChoice(
                            index=idx,
                            delta=DeltaMessage(),
                            finish_reason=final_finish_reason,
                            matched_stop=(
                                finish_reason_data["matched"]
                                if "matched" in finish_reason_data
                                else None
                            ),
                        )
                    ],
                    model=request.model,
                    usage=usage,
                )
                yield f"data: {finish_reason_chunk.model_dump_json()}\n\n"

            # Send hidden states if requested
            if request.return_hidden_states and hidden_states:
                for index, choice_hidden_states in hidden_states.items():
                    if choice_hidden_states:
                        last_token_hidden_states = (
                            choice_hidden_states[-1]
                            if len(choice_hidden_states) > 1
                            else []
                        )
                        hidden_states_chunk = ChatCompletionStreamResponse(
                            id=content["meta_info"]["id"],
                            created=int(time.time()),
                            choices=[
                                ChatCompletionResponseStreamChoice(
                                    index=index,
                                    delta=DeltaMessage(
                                        hidden_states=last_token_hidden_states
                                    ),
                                    finish_reason=None,  # Hidden states don't need finish_reason
                                )
                            ],
                            model=request.model,
                        )
                        yield f"data: {hidden_states_chunk.model_dump_json()}\n\n"

            # Additional usage chunk
            if request.stream_options and request.stream_options.include_usage:
                usage = UsageProcessor.calculate_streaming_usage(
                    prompt_tokens,
                    completion_tokens,
                    cached_tokens,
                    spec_verify_tokens,
                    n_choices=request.n,
                    enable_cache_report=self.engine_client.server_args.enable_cache_report,
                )
                usage_chunk = ChatCompletionStreamResponse(
                    id=content["meta_info"]["id"],
                    created=int(time.time()),
                    choices=[],  # Empty choices array as per OpenAI spec
                    model=request.model,
                    usage=usage,
                )
                yield f"data: {usage_chunk.model_dump_json()}\n\n"

        except ValueError as e:
            error = self.create_streaming_error_response(str(e))
            yield f"data: {error}\n\n"

        yield "data: [DONE]\n\n"

    async def _handle_non_streaming_request(
        self,
        adapted_request: GenerateReqInput,
        request: ChatCompletionRequest,
        raw_request: Request,
    ) -> ChatCompletionResponse | ErrorResponse | ORJSONResponse:
        """Handle non-streaming chat completion request"""
        if self.engine_client.server_args.speculative_algorithm is not None:
            adapted_request.return_logprob = None
            adapted_request.top_logprobs_num = None
            request.logprobs = False
            request.top_logprobs = None
            if raw_request is not None:
                raw_request.logprobs = False
                raw_request.top_logprobs = None

        try:
            # Wrap the await in a disconnect watchdog so client cancellation
            # (openai-python retry timeout, browser close, ...) cancels the
            # in-flight engine call and lets it call ``abort_request`` from
            # its ``finally`` block. Without this, Starlette/Uvicorn don't
            # auto-cancel non-streaming handlers and the cancelled request
            # keeps generating up to ``max_tokens`` while holding a
            # ``--max-num-seqs`` slot.
            ret = await await_with_disconnect_watchdog(
                self.engine_client.generate_request(adapted_request).__anext__(),
                raw_request,
            )
        except ValueError as e:
            return self.create_error_response(str(e))

        if not isinstance(ret, list):
            ret = [ret]

        response = self._build_chat_response(
            request,
            ret,
            int(time.time()),
        )

        return response

    def _build_chat_response(
        self,
        request: ChatCompletionRequest,
        ret: list[dict[str, Any]],
        created: int,
    ) -> ChatCompletionResponse | ORJSONResponse:
        """Build chat completion response from generation results"""
        choices = []

        for idx, ret_item in enumerate(ret):
            # Process logprobs
            choice_logprobs = None
            if request.logprobs:
                choice_logprobs = self._process_response_logprobs(ret_item)

            # Handle hidden states
            hidden_states = process_hidden_states_from_ret(ret_item, request)

            finish_reason = ret_item["meta_info"]["finish_reason"]
            text = ret_item["text"]

            finish_reason_type = finish_reason["type"] if finish_reason else None
            if (
                finish_reason_type
                and finish_reason_type == "abort"
                and finish_reason["err_type"] in [ABORT_CODE.TransferFailed]
            ):
                error = self.create_error_response(
                    message=finish_reason["message"],
                    err_type=finish_reason["err_type"].name,
                    status_code=finish_reason["err_type"].value,
                )
                return error

            # Handle reasoning content
            reasoning_text: str | None = None
            reasoning_parser: str | None = (
                self.engine_client.server_args.reasoning_parser
            )
            # gpt-oss output is always multi-channel with special tokens (we
            # keep them via skip_special_tokens=False), so the raw text is
            # never directly usable. Run the parser even when
            # separate_reasoning=False — we just drop reasoning_text in that
            # case so it never appears as a separate field.
            must_parse_for_clean_content: bool = reasoning_parser == "gpt-oss"
            if reasoning_parser and (
                request.separate_reasoning or must_parse_for_clean_content
            ):
                is_force_reasoning = self.template_manager.force_reasoning
                if (
                    hasattr(request, "chat_template_kwargs")
                    and request.chat_template_kwargs
                ):
                    request_signal = self._get_enable_thinking_from_request(request)
                    if request_signal is not None:
                        is_force_reasoning = request_signal
                try:
                    parser = ReasoningParser(
                        model_type=reasoning_parser,
                        stream_reasoning=False,
                        force_reasoning=is_force_reasoning,
                        request=request,
                    )
                    reasoning_text, text = parser.parse_non_stream(text)
                    if not request.separate_reasoning:
                        reasoning_text = None
                    # The harmony parser leaves the trailing ``<|end|>``
                    # (and any subsequent special tokens) on the normal
                    # text — strip them so JSON validators on the
                    # client side see a clean payload.
                    if reasoning_parser == "gpt-oss" and text:
                        for marker in (
                            "<|end|>",
                            "<|start|>",
                            "<|return|>",
                            "<|call|>",
                        ):
                            text = text.replace(marker, "")
                        text = text.rstrip()
                except Exception as e:
                    logger.error("Reasoning parsing error: %s", e)
                    return self.create_error_response(
                        "Failed to parse reasoning content",
                        err_type="InternalServerError",
                        status_code=500,
                    )

            # Handle tool calls
            tool_calls = None
            if request.tool_choice != "none" and request.tools:
                tool_call_parser = self.engine_client.server_args.tool_call_parser
                tool_calls, text, finish_reason = self._process_tool_calls(
                    text,
                    request.tools,
                    request.tool_choice,
                    tool_call_parser,
                    finish_reason,
                )

            choice_data = ChatCompletionResponseChoice(
                index=idx,
                message=ChatMessage(
                    role="assistant",
                    content=text if text else None,
                    tool_calls=tool_calls,
                    reasoning_content=reasoning_text if reasoning_text else None,
                ),
                logprobs=choice_logprobs,
                finish_reason=finish_reason["type"] if finish_reason else None,
                matched_stop=(
                    finish_reason["matched"]
                    if finish_reason and "matched" in finish_reason
                    else None
                ),
                hidden_states=hidden_states,
            )
            choices.append(choice_data)

        # Calculate usage
        usage = UsageProcessor.calculate_response_usage(
            ret,
            n_choices=request.n,
            enable_cache_report=self.engine_client.server_args.enable_cache_report,
        )

        return ChatCompletionResponse(
            id=ret[0]["meta_info"]["id"],
            created=created,
            model=request.model,
            choices=choices,
            usage=usage,
        )

    def _process_logprobs_tokens(
        self, logprobs: LogProbs, use_token_index: bool = False
    ) -> list[ChatCompletionTokenLogprob]:
        """Common helper to process logprobs tokens for both streaming and non-streaming

        Args:
            logprobs: LogProbs data from model
            use_token_index: True for non-streaming (use token_idx), False for streaming (use index 0)
        """
        token_logprobs = []

        for token_idx, (token, logprob) in enumerate(
            zip(logprobs.tokens, logprobs.token_logprobs)
        ):
            token_bytes = list(token.encode("utf-8"))
            top_logprobs = []
            if logprobs.top_logprobs:
                # - Non-streaming (use_token_index=True): uses token_idx for full data
                # - Streaming (use_token_index=False): uses index 0 for pre-sliced data
                top_logprobs_idx = token_idx if use_token_index else 0
                for top_token, top_logprob in logprobs.top_logprobs[
                    top_logprobs_idx
                ].items():
                    top_token_bytes = list(top_token.encode("utf-8"))
                    top_logprobs.append(
                        TopLogprob(
                            token=top_token,
                            bytes=top_token_bytes,
                            logprob=top_logprob,
                        )
                    )
            token_logprobs.append(
                ChatCompletionTokenLogprob(
                    token=token,
                    bytes=token_bytes,
                    logprob=logprob,
                    top_logprobs=top_logprobs,
                )
            )

        return token_logprobs

    def _process_response_logprobs(self, ret_item: dict[str, Any]) -> ChoiceLogprobs:
        """Process logprobs for non-streaming response"""
        logprobs = to_openai_style_logprobs(
            output_token_logprobs=ret_item["meta_info"]["output_token_logprobs"],
            output_top_logprobs=ret_item["meta_info"].get("output_top_logprobs", None),
        )

        token_logprobs = self._process_logprobs_tokens(logprobs, use_token_index=True)
        return ChoiceLogprobs(content=token_logprobs)

    def _process_tool_calls(
        self,
        text: str,
        tools: list[Any],
        toll_choice: ToolChoice | Literal["auto", "required", "none", "bypass_check"],
        tool_call_parser: str | None,
        finish_reason: dict[str, Any],
    ) -> tuple[list[ToolCall] | None, str, dict[str, Any]]:
        """Process tool calls in the response"""
        parser = FunctionCallParser(tools, tool_call_parser)
        if parser.has_tool_call(text):
            if finish_reason["type"] == "stop":
                finish_reason["type"] = "tool_calls"
                finish_reason["matched"] = None
            try:
                text, call_info_list = parser.parse_non_stream(text, toll_choice)
                tool_calls = [
                    ToolCall(
                        id=f"call_{uuid.uuid4().hex[:24]}",
                        function=FunctionResponse(
                            name=call_info.name, arguments=call_info.parameters
                        ),
                    )
                    for call_info in call_info_list
                ]
                return tool_calls, text, finish_reason
            except Exception as e:
                logger.error("Tool call parsing error: %s", e)
                # Return error but don't fail the whole request
                return None, text, finish_reason

        return None, text, finish_reason

    def _process_streaming_logprobs(
        self, content: dict[str, Any], n_prev_token: int
    ) -> ChoiceLogprobs:
        """Process logprobs for streaming response"""
        logprobs = to_openai_style_logprobs(
            output_token_logprobs=content["meta_info"]["output_token_logprobs"][
                n_prev_token:
            ],
            output_top_logprobs=content["meta_info"].get("output_top_logprobs", [])[
                n_prev_token:
            ],
        )

        token_logprobs = self._process_logprobs_tokens(logprobs, use_token_index=False)
        return ChoiceLogprobs(content=token_logprobs)

    def _process_reasoning_stream(
        self,
        index: int,
        delta: str,
        reasoning_parser_dict: dict[int, ReasoningParser],
        content: dict[str, Any],
        request: ChatCompletionRequest,
    ) -> tuple[str | None, str]:
        """Process reasoning content in streaming response"""
        if index not in reasoning_parser_dict:
            is_force_reasoning = self.template_manager.force_reasoning
            if (
                hasattr(request, "chat_template_kwargs")
                and request.chat_template_kwargs
            ):
                request_signal = self._get_enable_thinking_from_request(request)
                if request_signal is not None:
                    is_force_reasoning = request_signal
            reasoning_parser_dict[index] = ReasoningParser(
                self.engine_client.server_args.reasoning_parser,
                request.stream_reasoning,
                is_force_reasoning,
                request=request,
            )
        reasoning_parser = reasoning_parser_dict[index]
        return reasoning_parser.parse_stream_chunk(delta)

    def _get_enable_thinking_from_request(
        self, request: ChatCompletionRequest
    ) -> bool | None:
        """Extract the 'enable_thinking'/'thinking' flag from
        request.chat_template_kwargs.

        This parameter is only useful for models that support an explicit
        thinking toggle (e.g. Qwen3, DeepSeek-V3.1).

        Returns:
            The boolean value of 'enable_thinking' / 'thinking' if either
            key is set on the request, otherwise None — `None` means the
            request gives no signal, and the caller should fall back to
            the template-detection / model-type default rather than
            silently disabling reasoning.
        """
        # For Qwen3 models, `enable_thinking` is supported.
        if request.chat_template_kwargs.get("enable_thinking") is not None:
            return request.chat_template_kwargs.get("enable_thinking")
        # For DeepSeek-V3.1 models, `thinking` is supported.
        if request.chat_template_kwargs.get("thinking") is not None:
            return request.chat_template_kwargs.get("thinking")
        return None

    async def _process_tool_call_stream(
        self,
        index: int,
        delta: str,
        parser_dict: dict[int, FunctionCallParser],
        content: dict[str, Any],
        request: ChatCompletionRequest,
        has_tool_calls: dict[int, bool],
    ):
        """Process tool calls in streaming response"""
        if index not in parser_dict:
            parser_dict[index] = FunctionCallParser(
                tools=request.tools,
                tool_call_parser=self.engine_client.server_args.tool_call_parser,
            )
        parser = parser_dict[index]

        normal_text, calls = parser.parse_stream_chunk(delta, request.tool_choice)

        # Yield normal text
        if normal_text:
            choice_data = ChatCompletionResponseStreamChoice(
                index=index,
                delta=DeltaMessage(content=normal_text),
                finish_reason=None,
            )
            chunk = ChatCompletionStreamResponse(
                id=content["meta_info"]["id"],
                created=int(time.time()),
                choices=[choice_data],
                model=request.model,
            )
            yield f"data: {chunk.model_dump_json()}\n\n"

        # Yield tool calls
        for call_item in calls:
            # Mark that this choice has tool calls
            has_tool_calls[index] = True

            # Tool call ID should be generated only once per tool call
            if call_item.name:
                # First chunk: include ID and function name
                tool_call_id = f"call_{uuid.uuid4().hex[:24]}"
                function_name = call_item.name
            else:
                # Subsequent chunks: null ID and name for argument deltas
                tool_call_id = None
                function_name = None

            tool_call = ToolCall(
                id=tool_call_id,
                index=call_item.tool_index,
                function=FunctionResponse(
                    name=function_name,
                    arguments=call_item.parameters,
                ),
            )

            choice_data = ChatCompletionResponseStreamChoice(
                index=index,
                delta=DeltaMessage(tool_calls=[tool_call]),
                finish_reason=None,
            )
            chunk = ChatCompletionStreamResponse(
                id=content["meta_info"]["id"],
                created=int(time.time()),
                choices=[choice_data],
                model=request.model,
            )
            yield f"data: {chunk.model_dump_json()}\n\n"

    def _check_for_unstreamed_tool_args(
        self,
        parser: FunctionCallParser,
        content: dict[str, Any],
        request: ChatCompletionRequest,
        index: int,
    ) -> str | None:
        """
        Check for any remaining tool call arguments that need to be streamed
        when generation finishes. This ensures tool calls are properly completed
        even if the model generates the final arguments in the last chunk.
        """
        # Only check if we have tool calls and the parser has tracked data
        if (
            not hasattr(parser.detector, "prev_tool_call_arr")
            or not parser.detector.prev_tool_call_arr
        ):
            return None

        if (
            not hasattr(parser.detector, "streamed_args_for_tool")
            or not parser.detector.streamed_args_for_tool
        ):
            return None

        # Get the last tool call that was being processed
        tool_index = len(parser.detector.prev_tool_call_arr) - 1
        if tool_index < 0 or tool_index >= len(parser.detector.streamed_args_for_tool):
            return None

        # Get expected vs actual arguments
        expected_args = parser.detector.prev_tool_call_arr[tool_index].get(
            "arguments", {}
        )
        expected_call = json.dumps(expected_args, ensure_ascii=False)
        actual_call = parser.detector.streamed_args_for_tool[tool_index]

        # Check if there are remaining arguments to send
        remaining_call = (
            expected_call.replace(actual_call, "", 1)
            if actual_call in expected_call
            else ""
        )

        if remaining_call:
            # Create tool call chunk with remaining arguments
            tool_call = ToolCall(
                id=None,  # No ID for argument deltas
                index=tool_index,
                function=FunctionResponse(
                    name=None,  # No name for argument deltas
                    arguments=remaining_call,
                ),
            )

            choice_data = ChatCompletionResponseStreamChoice(
                index=index,
                delta=DeltaMessage(tool_calls=[tool_call]),
                finish_reason=None,  # Don't send finish_reason with this chunk
            )

            chunk = ChatCompletionStreamResponse(
                id=content["meta_info"]["id"],
                created=int(time.time()),
                choices=[choice_data],
                model=request.model,
            )

            return f"data: {chunk.model_dump_json()}\n\n"

        return None
