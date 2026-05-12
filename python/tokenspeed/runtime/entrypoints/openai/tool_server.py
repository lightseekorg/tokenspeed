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

from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any

from openai_harmony import ToolNamespaceConfig


class ToolServer(ABC):

    @abstractmethod
    def has_tool(self, tool_name: str):
        pass

    @abstractmethod
    def get_tool_description(self, tool_name: str):
        pass

    @abstractmethod
    def get_tool_session(self, tool_name: str) -> AbstractAsyncContextManager[Any]: ...


class DemoToolServer(ToolServer):

    def __init__(self):
        from tokenspeed.runtime.entrypoints.tool import (
            HarmonyBrowserTool,
            HarmonyPythonTool,
            Tool,
        )

        self.tools: dict[str, Tool] = {}
        browser_tool = HarmonyBrowserTool()
        if browser_tool.enabled:
            self.tools["browser"] = browser_tool
        python_tool = HarmonyPythonTool()
        if python_tool.enabled:
            self.tools["python"] = python_tool

    def has_tool(self, tool_name: str):
        return tool_name in self.tools

    def get_tool_description(self, tool_name: str):
        if tool_name not in self.tools:
            return None
        if tool_name == "browser":
            return ToolNamespaceConfig.browser()
        elif tool_name == "python":
            return ToolNamespaceConfig.python()
        else:
            raise ValueError(f"Unknown tool {tool_name}")

    @asynccontextmanager
    async def get_tool_session(self, tool_name: str):
        yield self.tools[tool_name]
