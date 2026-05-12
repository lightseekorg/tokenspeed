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

"""Shared helpers for function-call parsing and schema construction."""

from json import JSONDecodeError, JSONDecoder
from json.decoder import WHITESPACE
from typing import Any, Literal

import orjson
import partial_json_parser
from partial_json_parser.core.options import Allow

from tokenspeed.runtime.entrypoints.openai.protocol import Tool, ToolChoice


def _find_common_prefix(s1: str, s2: str) -> str:
    min_length = min(len(s1), len(s2))
    prefix_chars = []
    for i in range(0, min_length):
        if s1[i] == s2[i]:
            prefix_chars.append(s1[i])
        else:
            break
    return "".join(prefix_chars)


def _partial_json_loads(input_str: str, flags: Allow) -> tuple[Any, int]:
    """
    Parse incomplete or partial JSON strings commonly encountered during streaming.

    Args:
        input_str (str): The potentially incomplete JSON string to parse.
        flags (Allow): Bitwise flags controlling what types of partial data are allowed.
            Common flags include:
            - Allow.STR: Allow partial strings (e.g., '"hello wo' -> 'hello wo')
            - Allow.OBJ: Allow partial objects (e.g., '{"key":' -> {'key': None})
            - Allow.ARR: Allow partial arrays (e.g., '[1, 2,' -> [1, 2])
            - Allow.ALL: Allow all types of partial data

    Returns:
        Tuple[Any, int]: A tuple containing:
            - parsed_object: The Python object parsed from the JSON
            - consumed_length: Number of characters consumed from input_str
    """
    try:
        return (partial_json_parser.loads(input_str, flags), len(input_str))
    except (JSONDecodeError, IndexError) as e:
        msg = getattr(e, "msg", str(e))
        if "Extra data" in msg or "pop from empty list" in msg:
            start = WHITESPACE.match(input_str, 0).end()
            obj, end = JSONDecoder().raw_decode(input_str, start)
            return obj, end
        raise


def _is_complete_json(input_str: str) -> bool:
    try:
        orjson.loads(input_str)
        return True
    except JSONDecodeError:
        return False


def _get_tool_schema_defs(tools: list[Tool]) -> dict:
    """
    Get consolidated $defs from all tools, validating for conflicts.

    Args:
        tools: List of tools to process

    Returns:
        Dictionary of consolidated $defs from all tools

    Raises:
        ValueError: If conflicting $defs are found
    """
    all_defs = {}
    for tool in tools:
        if tool.function.parameters is None:
            continue
        defs = tool.function.parameters.get("$defs", {})
        for def_name, def_schema in defs.items():
            if def_name in all_defs and all_defs[def_name] != def_schema:
                raise ValueError(
                    f"Tool definition '{def_name}' has "
                    "multiple schemas, which is not "
                    "supported."
                )
            else:
                all_defs[def_name] = def_schema
    return all_defs


def _get_tool_schema(tool: Tool) -> dict:
    return {
        "properties": {
            "name": {"type": "string", "enum": [tool.function.name]},
            "parameters": (
                tool.function.parameters
                if tool.function.parameters
                else {"type": "object", "properties": {}}
            ),
        },
        "required": ["name", "parameters"],
    }


def get_json_schema_constraint(
    tools: list[Tool], tool_choice: ToolChoice | Literal["required"]
) -> dict | None:
    """
    Get the JSON schema constraint for the specified tool choice.

    Args:
        tool_choice: The tool choice specification

    Returns:
        JSON schema dict, or None if no valid tools found
    """

    if isinstance(tool_choice, ToolChoice):
        # For specific function choice, return the user's parameters schema directly
        fn_name = tool_choice.function.name
        for tool in tools:
            if tool.function.name == fn_name:
                return {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 1,
                    "items": _get_tool_schema(tool),
                }
        return None
    if tool_choice == "required":
        json_schema = {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "anyOf": [_get_tool_schema(tool) for tool in tools],
            },
        }
        json_schema_defs = _get_tool_schema_defs(tools)
        if json_schema_defs:
            json_schema["$defs"] = json_schema_defs
        return json_schema

    return None
