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

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from openai_harmony import Message, Role, StreamState

from tokenspeed.runtime.entrypoints.harmony_utils import (
    get_encoding,
    get_streamable_parser_for_assistant,
    render_for_completion,
)
from tokenspeed.runtime.entrypoints.tool import Tool

logger = logging.getLogger(__name__)


class ConversationContext(ABC):
    @abstractmethod
    def append_output(self, output) -> None:
        """Append a model or tool output into the conversation state."""

    @abstractmethod
    async def call_tool(self) -> list[Message]:
        """Execute the currently requested tool call."""

    @abstractmethod
    def need_builtin_tool_call(self) -> bool:
        """Return whether the next step should invoke a built-in tool."""

    @abstractmethod
    def render_for_completion(self) -> list[int]:
        """Render the conversation into completion token ids."""


class SimpleContext(ConversationContext):
    def __init__(self):
        self.last_output = None

    def append_output(self, output) -> None:
        self.last_output = output

    def need_builtin_tool_call(self) -> bool:
        return False

    async def call_tool(self) -> list[Message]:
        raise NotImplementedError("Should not be called.")

    def render_for_completion(self) -> list[int]:
        raise NotImplementedError("Should not be called.")


class HarmonyContext(ConversationContext):
    def __init__(
        self,
        messages: list,
        tool_sessions: dict[str, Tool],
    ):
        self._messages = messages
        self.tool_sessions = tool_sessions

        self.parser = get_streamable_parser_for_assistant()
        self.num_init_messages = len(messages)
        self.num_prompt_tokens = 0
        self.num_cached_tokens = 0
        self.num_output_tokens = 0
        self.num_reasoning_tokens = 0

    def append_output(self, output) -> None:
        if isinstance(output, dict) and "output_ids" in output:
            output_token_ids = output["output_ids"]

            for token_id in output_token_ids:
                self.parser.process(token_id)
            output_msgs = self.parser.messages

            meta_info = output["meta_info"]

            if isinstance(meta_info, dict):
                if "prompt_token_ids" in meta_info:
                    self.num_prompt_tokens = meta_info["prompt_tokens"]
                if "cached_tokens" in meta_info:
                    self.num_cached_tokens = meta_info["cached_tokens"]
                if "completion_tokens" in meta_info:
                    self.num_output_tokens += meta_info["completion_tokens"]

        else:
            output_msgs = output

        self._messages.extend(output_msgs)

    @property
    def messages(self) -> list:
        return self._messages

    def need_builtin_tool_call(self) -> bool:
        last_msg = self.messages[-1]
        recipient = last_msg.recipient
        return recipient is not None and (
            recipient.startswith("browser.") or recipient.startswith("python")
        )

    async def call_tool(self) -> list[Message]:
        if not self.messages:
            return []
        last_msg = self.messages[-1]
        recipient = last_msg.recipient
        if recipient is None:
            raise ValueError("No tool call found")
        if recipient.startswith("browser."):
            return await self.call_search_tool(self.tool_sessions["browser"], last_msg)
        if recipient.startswith("python"):
            return await self.call_python_tool(self.tool_sessions["python"], last_msg)
        raise ValueError("No tool call found")

    def render_for_completion(self) -> list[int]:
        return render_for_completion(self.messages)

    async def call_search_tool(
        self, tool_session: Tool, last_msg: Message
    ) -> list[Message]:
        return await tool_session.get_result(self)

    async def call_python_tool(
        self, tool_session: Tool, last_msg: Message
    ) -> list[Message]:
        return await tool_session.get_result(self)


class StreamingHarmonyContext(HarmonyContext):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_output = None

        self.parser = get_streamable_parser_for_assistant()
        self.encoding = get_encoding()
        self.last_tok = None

    @property
    def messages(self) -> list:
        return self.parser.messages

    def append_output(self, output) -> None:
        if isinstance(output, dict) and "output_ids" in output:
            # RequestOutput from TokenSpeed with outputs
            output_token_ids = output["output_ids"]

            for token_id in output_token_ids:
                self.parser.process(token_id)

        else:
            # Handle the case of tool output in direct message format
            assert len(output) == 1, "Tool output should be a single message"
            msg = output[0]
            # Sometimes the recipient is not set for tool messages,
            # so we set it to "assistant"
            if msg.author.role == Role.TOOL and msg.recipient is None:
                msg.recipient = "assistant"
            toks = self.encoding.render(msg)
            for tok in toks:
                self.parser.process(tok)
            self.last_tok = toks[-1]

    def is_expecting_start(self) -> bool:
        return self.parser.state == StreamState.EXPECT_START

    def is_assistant_action_turn(self) -> bool:
        return self.last_tok in self.encoding.stop_tokens_for_assistant_actions()

    def render_for_completion(self) -> list[int]:
        # now this list of tokens as next turn's starting tokens
        # `<|start|>assistant``,
        # we need to process them in parser.
        rendered_tokens = super().render_for_completion()

        last_n = -1
        to_process = []
        while rendered_tokens[last_n] != self.last_tok:
            to_process.append(rendered_tokens[last_n])
            last_n -= 1
        for tok in reversed(to_process):
            self.parser.process(tok)

        return rendered_tokens
