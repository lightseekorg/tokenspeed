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

import json
import logging
import re
from typing import Any, Literal

from tokenspeed.runtime.entrypoints.openai.protocol import Tool, ToolChoice
from tokenspeed.runtime.grammar.base_format_detector import BaseFormatDetector
from tokenspeed.runtime.grammar.core_types import (
    StreamingParseResult,
    StructureInfo,
    ToolCallItem,
    _GetInfoFunc,
)

logger = logging.getLogger(__name__)


class MinimaxM2Detector(BaseFormatDetector):
    """Detector for MiniMax M2 tool calls.

    Expected format:
      <minimax:tool_call>
      <invoke name="tool_name">
      <parameter name="arg">value</parameter>
      ...
      </invoke>
      </minimax:tool_call>
    """

    def __init__(self):
        super().__init__()
        self.tool_call_start_token = "<minimax:tool_call>"
        self.tool_call_end_token = "</minimax:tool_call>"
        self.invoke_start_pattern = re.compile(r'<invoke name="([^"]+)">', re.DOTALL)
        self.invoke_end_token = "</invoke>"
        self.parameter_pattern = re.compile(
            r'<parameter name="([^"]+)">(.*?)</parameter>', re.DOTALL
        )

    def has_tool_call(self, text: str) -> bool:
        return self.tool_call_start_token in text

    def _tool_indices(self, tools: list[Tool]) -> dict[str, int]:
        return {
            tool.function.name: i for i, tool in enumerate(tools) if tool.function.name
        }

    def _convert_param_value(self, raw_value: str, tool: Tool, param_name: str) -> Any:
        parameters = tool.function.parameters or {}
        properties = (
            parameters.get("properties", {}) if isinstance(parameters, dict) else {}
        )
        schema = properties.get(param_name, {}) if isinstance(properties, dict) else {}
        expected_type = schema.get("type") if isinstance(schema, dict) else None

        value = raw_value.strip()
        if expected_type == "integer":
            try:
                return int(value)
            except ValueError:
                return value
        if expected_type == "number":
            try:
                return float(value)
            except ValueError:
                return value
        if expected_type == "boolean":
            lowered = value.lower()
            if lowered in {"true", "1", "yes", "on"}:
                return True
            if lowered in {"false", "0", "no", "off"}:
                return False
            return value
        if expected_type in {"object", "array"}:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return value

    def _parse_block(
        self,
        block: str,
        tools: list[Tool],
        tool_choice: ToolChoice | Literal["auto", "required", "none", "bypass_check"],
    ) -> list[ToolCallItem]:
        tool_indices = self._tool_indices(tools)
        results: list[ToolCallItem] = []

        for invoke_match in re.finditer(
            r'<invoke name="([^"]+)">(.*?)</invoke>', block, re.DOTALL
        ):
            function_name = invoke_match.group(1).strip()
            body = invoke_match.group(2)

            if function_name not in tool_indices and tool_choice != "bypass_check":
                logger.warning(
                    "Model attempted to call undefined MiniMax tool: %s", function_name
                )
                continue

            tool = next(
                (tool for tool in tools if tool.function.name == function_name), None
            )
            params: dict[str, Any] = {}
            for param_match in self.parameter_pattern.finditer(body):
                param_name = param_match.group(1).strip()
                raw_value = param_match.group(2)
                params[param_name] = (
                    self._convert_param_value(raw_value, tool, param_name)
                    if tool is not None
                    else raw_value.strip()
                )

            results.append(
                ToolCallItem(
                    tool_index=tool_indices.get(function_name, -1),
                    name=function_name,
                    parameters=json.dumps(params, ensure_ascii=False),
                )
            )

        return results

    def detect_and_parse(
        self,
        text: str,
        tools: list[Tool],
        tool_choice: ToolChoice | Literal["auto", "required", "none", "bypass_check"],
    ) -> StreamingParseResult:
        if self.tool_call_start_token not in text:
            return StreamingParseResult(normal_text=text)

        normal_parts: list[str] = []
        calls: list[ToolCallItem] = []
        cursor = 0
        while True:
            start = text.find(self.tool_call_start_token, cursor)
            if start == -1:
                normal_parts.append(text[cursor:])
                break
            normal_parts.append(text[cursor:start])
            end = text.find(self.tool_call_end_token, start)
            if end == -1:
                normal_parts.append(text[start:])
                break
            block = text[start : end + len(self.tool_call_end_token)]
            calls.extend(self._parse_block(block, tools, tool_choice))
            cursor = end + len(self.tool_call_end_token)

        return StreamingParseResult(normal_text="".join(normal_parts), calls=calls)

    def parse_streaming_increment(
        self,
        new_text: str,
        tools: list[Tool],
        tool_choice: ToolChoice | Literal["auto", "required", "none", "bypass_check"],
    ) -> StreamingParseResult:
        self._buffer += new_text

        # No start token at all — flush entire buffer as normal text.
        if self.tool_call_start_token not in self._buffer:
            normal_text = self._buffer
            self._buffer = ""
            return StreamingParseResult(normal_text=normal_text)

        # Check if we have at least one complete tool_call block.
        last_end = self._buffer.rfind(self.tool_call_end_token)
        if last_end == -1:
            # Incomplete block — flush any normal text before the start tag,
            # keep the partial block in the buffer.
            start = self._buffer.find(self.tool_call_start_token)
            normal_text = self._buffer[:start]
            self._buffer = self._buffer[start:]
            return StreamingParseResult(normal_text=normal_text)

        # We have complete block(s). Parse everything up to (and including)
        # the last complete end token.
        consume_end = last_end + len(self.tool_call_end_token)
        consumable = self._buffer[:consume_end]
        self._buffer = self._buffer[consume_end:]

        result = self.detect_and_parse(consumable, tools, tool_choice)
        return result

    def supports_structural_tag(self) -> bool:
        return False

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin=f'{self.tool_call_start_token}<invoke name="{name}">',
            end=f"{self.invoke_end_token}{self.tool_call_end_token}",
            trigger=self.tool_call_start_token,
        )
