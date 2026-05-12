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

"""High-level function-call parser wrapper around format detectors."""

import logging
from typing import Any, Literal

from tokenspeed.runtime.entrypoints.openai.protocol import (
    StructuralTagResponseFormat,
    StructuresResponseFormat,
    Tool,
    ToolChoice,
)
from tokenspeed.runtime.grammar.base_format_detector import BaseFormatDetector
from tokenspeed.runtime.grammar.core_types import ToolCallItem
from tokenspeed.runtime.grammar.deepseekv3_detector import DeepSeekV3Detector
from tokenspeed.runtime.grammar.deepseekv31_detector import (
    DeepSeekV31Detector,
)
from tokenspeed.runtime.grammar.gpt_oss_detector import GptOssDetector
from tokenspeed.runtime.grammar.kimik2_detector import KimiK2Detector
from tokenspeed.runtime.grammar.minimax_m2 import MinimaxM2Detector
from tokenspeed.runtime.grammar.qwen3_coder_detector import (
    Qwen3CoderDetector,
)
from tokenspeed.runtime.grammar.qwen_detector import QwenDetector
from tokenspeed.runtime.grammar.utils import get_json_schema_constraint

logger = logging.getLogger(__name__)


class FunctionCallParser:
    """Parse function and tool calls from model outputs.

    This wrapper handles both streaming and non-streaming parsing by delegating
    the concrete syntax details to a detector implementation.
    """

    ToolCallParserEnum: dict[str, type[BaseFormatDetector]] = {
        "deepseekv3": DeepSeekV3Detector,
        "deepseekv31": DeepSeekV31Detector,
        "deepseek_v4": DeepSeekV31Detector,
        "openai": GptOssDetector,
        "gpt-oss": GptOssDetector,
        "kimi_k2": KimiK2Detector,
        "minimax_m2": MinimaxM2Detector,
        "minimax-m2": MinimaxM2Detector,
        "qwen": QwenDetector,
        "qwen3": QwenDetector,
        "qwen3_coder": Qwen3CoderDetector,
        "qwen3.5": QwenDetector,
    }

    def __init__(self, tools: list[Tool], tool_call_parser: str):
        detector_class = self.ToolCallParserEnum.get(tool_call_parser)
        if detector_class:
            self.detector = detector_class()
        else:
            raise ValueError(f"Unsupported tool_call_parser: {tool_call_parser}")
        self.tools = tools

    def has_tool_call(self, text: str) -> bool:
        """
        Check if the given text contains a tool call in the format supported by this parser.
        This delegates to the detector's implementation.

        Args:
            text: The text to check for tool calls

        Returns:
            True if the text contains a tool call, False otherwise.
        """
        if not self.tools:
            return False
        return self.detector.has_tool_call(text)

    def parse_non_stream(
        self,
        full_text: str,
        tool_choice: ToolChoice | Literal["auto", "required", "none", "bypass_check"],
    ) -> tuple[str, list[ToolCallItem]]:
        """
        One-time parsing of the full text to extract tool calls.

        Args:
            full_text: The complete text to parse

        Returns:
            A tuple containing:
            - The remaining text after parsing that was not consumed by the detector (can be treated as normal text)
            - A list of tool calls parsed from the text
        """
        if not self.tools:
            return full_text, []
        parsed_result = self.detector.detect_and_parse(
            text=full_text, tools=self.tools, tool_choice=tool_choice
        )
        tool_call_list = parsed_result.calls
        if tool_call_list:
            return parsed_result.normal_text, tool_call_list
        return full_text, []

    def parse_stream_chunk(
        self,
        chunk_text: str,
        tool_choice: ToolChoice | Literal["auto", "required", "none", "bypass_check"],
    ) -> tuple[str, list[ToolCallItem]]:
        """
        Streaming incremental parsing of chunks of text as they arrive.

        Args:
            chunk_text: The new chunk of text to parse

        Returns:
            A tuple containing:
            - The normal text that should be displayed to the user
            - A list of tool calls parsed from the chunk
        """
        if not self.tools:
            return chunk_text, []
        final_normal_text = ""
        final_calls = []

        sp_result = self.detector.parse_streaming_increment(
            chunk_text, self.tools, tool_choice
        )
        if sp_result.normal_text:
            final_normal_text = sp_result.normal_text
        if sp_result.calls:
            final_calls.extend(sp_result.calls)
            final_normal_text = sp_result.normal_text

        return final_normal_text, final_calls

    def get_structure_tag(self) -> StructuralTagResponseFormat:
        """
        Generate a structural tag response format for all available tools.

        This creates the necessary structural tags that guide the model's output format.
        """
        tool_structures: list[StructuresResponseFormat] = list()
        tool_trigger_set: set[str] = set()

        get_structure_info = self.detector.structure_info()
        for tool in self.tools:
            function = tool.function
            name = function.name
            assert name is not None
            info = get_structure_info(name)

            # accept all if not strict, otherwise only accept the schema
            schema = function.parameters if function.strict else {}

            tool_structures.append(
                StructuresResponseFormat(
                    begin=info.begin,
                    schema=schema,  # type: ignore
                    end=info.end,
                )
            )
            tool_trigger_set.add(info.trigger)

        return StructuralTagResponseFormat(
            type="structural_tag",
            structures=tool_structures,
            triggers=list(tool_trigger_set),
        )

    def get_structure_constraint(
        self, tool_choice: ToolChoice | Literal["auto", "required"]
    ) -> tuple[str, Any] | None:
        """
        Returns the appropriate structure constraint for tool calls based on the tool_choice.
        The constraint is used to guide the model's output format.

        Args:
            tool_choice: The tool choice setting from the request

        Returns:
            A tuple of (constraint_type, constraint_value) to be added to sampling parameters,
            or None if no constraint applies.
        """
        #  structural_tag only supports JSON-compatible content between the begin and end.
        # It cannot parse or validate function call Pythonic or XML-ish syntax.
        if (
            self.detector.supports_structural_tag()
            and tool_choice == "auto"
            and any(tool.function.strict for tool in self.tools)
        ):
            strict_tag = self.get_structure_tag()
            return ("structural_tag", strict_tag)
        elif tool_choice == "required" or isinstance(tool_choice, ToolChoice):
            json_schema = get_json_schema_constraint(self.tools, tool_choice)
            return ("json_schema", json_schema)
