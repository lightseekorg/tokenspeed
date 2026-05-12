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
Centralized template management for chat templates and completion templates.

This module provides a unified interface for managing both chat conversation templates
and code completion templates, eliminating global state and improving modularity.
"""

import json
import os
import re

from tokenspeed.runtime.inputs.code_completion_parser import (
    CompletionTemplate,
    FimPosition,
    completion_template_exists,
    register_completion_template,
)
from tokenspeed.runtime.inputs.conversation import (
    Conversation,
    SeparatorStyle,
    chat_template_exists,
    register_conv_template,
)
from tokenspeed.runtime.inputs.jinja_template_utils import (
    detect_jinja_template_content_format,
)
from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)


class TemplateManager:
    """
    Centralized manager for chat and completion templates.

    This class encapsulates all template-related state and operations,
    eliminating the need for global variables and providing a clean
    interface for template management.
    """

    def __init__(self):
        self._chat_template_name: str | None = None
        self._completion_template_name: str | None = None
        self._jinja_template_content_format: str | None = "openai"
        # Tri-state: True (template explicitly forces reasoning), False
        # (template explicitly disables it), None (template signals neither —
        # let the gateway-side reasoning parser fall back to its
        # model-type default).
        self._force_reasoning: bool | None = None

    @property
    def chat_template_name(self) -> str | None:
        """Get the current chat template name."""
        return self._chat_template_name

    @property
    def completion_template_name(self) -> str | None:
        """Get the current completion template name."""
        return self._completion_template_name

    @property
    def jinja_template_content_format(self) -> str | None:
        """Get the detected template content format ('string' or 'openai' or None)."""
        return self._jinja_template_content_format

    @property
    def force_reasoning(self) -> bool | None:
        """
        Tri-state signal of whether the chat template forces reasoning/thinking.

        Returns:
            True  — template explicitly forces reasoning (e.g. injects
                    `<think>` after the assistant header, or sets
                    `enable_thinking | default(true)`).
            False — template explicitly disables reasoning
                    (`enable_thinking | default(false)`).
            None  — template gives no signal either way; the caller should
                    fall back to model-type defaults in the gateway-side
                    reasoning parser.
        """
        return self._force_reasoning

    def _detect_reasoning_pattern(self, template: str) -> bool | None:
        """
        Detect if the chat template contains reasoning/thinking patterns.
        Returns Optional[bool]: True/False when the template is explicit,
        None when it gives no signal.
        """
        if template is None:
            return None

        # Explicit enable_thinking flag in the template wins over the
        # `<think>` heuristic — covers Qwen3-style toggleable templates.
        enable_thinking_pattern = r"enable_thinking\s*\|\s*default\((true|false|.*?)\)"
        enable_thinking_match = re.search(enable_thinking_pattern, template)
        if enable_thinking_match is not None:
            value = enable_thinking_match.group(1)
            if value == "true":
                logger.warning("Detected the force reasoning pattern in chat template.")
                return True
            if value == "false":
                return False
            # Non-boolean default (e.g. variable reference) — treat as no signal.
            return None

        force_reasoning_pattern = r"<\|im_start\|>assistant\\n<think>\\n"
        if re.search(force_reasoning_pattern, template) is not None:
            logger.warning("Detected the force reasoning pattern in chat template.")
            return True

        return None

    def load_chat_template(
        self, tokenizer_manager, chat_template_arg: str | None, model_path: str
    ) -> None:
        """
        Load a chat template from various sources.

        Args:
            tokenizer_manager: The tokenizer manager instance
            chat_template_arg: Template name, file path, or None to auto-detect
            model_path: Path to the model
        """
        logger.warning(
            "chat_template_arg=%r model_path=%r", chat_template_arg, model_path
        )
        if chat_template_arg:
            self._load_explicit_chat_template(tokenizer_manager, chat_template_arg)
        else:
            # If no pre-defined template was found, fallback to HuggingFace template
            if self._chat_template_name is None:
                # Default to string content format if no template was found
                self._jinja_template_content_format = "string"
                logger.warning(
                    "chat template will be detected by hf in function _apply_jinja_template"
                )

        # Detect reasoning pattern from chat template
        if tokenizer_manager.tokenizer:
            self._force_reasoning = self._detect_reasoning_pattern(
                tokenizer_manager.tokenizer.chat_template
            )

    def _load_jinja_template(self, tokenizer_manager, template_path: str) -> None:
        """Load a Jinja template file."""
        with open(template_path) as f:
            chat_template = "".join(f.readlines()).strip("\n")
        tokenizer_manager.tokenizer.chat_template = chat_template.replace("\\n", "\n")
        self._chat_template_name = None
        # Detect content format from the loaded template
        self._jinja_template_content_format = detect_jinja_template_content_format(
            chat_template
        )
        logger.info(
            "Detected user specified Jinja chat template with content format: %s",
            self._jinja_template_content_format,
        )

    def _load_explicit_chat_template(
        self, tokenizer_manager, chat_template_arg: str
    ) -> None:
        """Load explicitly specified chat template."""
        logger.warning("Loading chat template from argument: %s", chat_template_arg)

        if chat_template_exists(chat_template_arg):
            self._chat_template_name = chat_template_arg
            return

        if not os.path.exists(chat_template_arg):
            raise RuntimeError(
                f"Chat template {chat_template_arg} is not a built-in template name "
                "or a valid chat template file path."
            )

        if chat_template_arg.endswith(".jinja"):
            self._load_jinja_template(tokenizer_manager, chat_template_arg)
        else:
            self._load_json_chat_template(chat_template_arg)

    def load_completion_template(self, completion_template_arg: str) -> None:
        """
        Load completion template for code completion.

        Args:
            completion_template_arg: Template name or file path
        """
        logger.warning("Loading completion template: %s", completion_template_arg)

        if not completion_template_exists(completion_template_arg):
            if not os.path.exists(completion_template_arg):
                raise RuntimeError(
                    f"Completion template {completion_template_arg} is not a built-in template name "
                    "or a valid completion template file path."
                )

            self._load_json_completion_template(completion_template_arg)
        else:
            self._completion_template_name = completion_template_arg

    def initialize_templates(
        self,
        tokenizer_manager,
        model_path: str,
        chat_template: str | None = None,
        completion_template: str | None = None,
    ) -> None:
        """
        Initialize all templates based on provided configuration.

        Args:
            tokenizer_manager: The tokenizer manager instance
            model_path: Path to the model
            chat_template: Optional chat template name/path
            completion_template: Optional completion template name/path
        """
        # Load chat template
        self.load_chat_template(tokenizer_manager, chat_template, model_path)

        # Load completion template
        if completion_template:
            self.load_completion_template(completion_template)

    def _load_json_chat_template(self, template_path: str) -> None:
        """Load a JSON chat template file."""
        assert template_path.endswith(
            ".json"
        ), "unrecognized format of chat template file"

        with open(template_path) as filep:
            template = json.load(filep)
            try:
                sep_style = SeparatorStyle[template["sep_style"]]
            except KeyError:
                raise ValueError(
                    f"Unknown separator style: {template['sep_style']}"
                ) from None

            register_conv_template(
                Conversation(
                    name=template["name"],
                    system_template=template["system"] + "\n{system_message}",
                    system_message=template.get("system_message", ""),
                    roles=(template["user"], template["assistant"]),
                    sep_style=sep_style,
                    sep=template.get("sep", "\n"),
                    stop_str=template["stop_str"],
                ),
                override=True,
            )
        self._chat_template_name = template["name"]

    def _load_json_completion_template(self, template_path: str) -> None:
        """Load a JSON completion template file."""
        assert template_path.endswith(
            ".json"
        ), "unrecognized format of completion template file"

        with open(template_path) as filep:
            template = json.load(filep)
            try:
                fim_position = FimPosition[template["fim_position"]]
            except KeyError:
                raise ValueError(
                    f"Unknown fim position: {template['fim_position']}"
                ) from None

            register_completion_template(
                CompletionTemplate(
                    name=template["name"],
                    fim_begin_token=template["fim_begin_token"],
                    fim_middle_token=template["fim_middle_token"],
                    fim_end_token=template["fim_end_token"],
                    fim_position=fim_position,
                ),
                override=True,
            )
        self._completion_template_name = template["name"]
