# Adapted from meituan-longcat/SGLang-FluentLLM.
# This file has been modified for this repository.
# Upstream lineage includes ModelTC/lightllm, vllm-project/vllm,
# and sgl-project/sglang. See python/THIRDPARTYNOTICES.
# Licensed under the Apache License, Version 2.0
#
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

"""Completion templates."""

import dataclasses
import logging
from enum import auto

from tokenspeed.runtime.entrypoints.openai.protocol import CompletionRequest

logger = logging.getLogger(__name__)


class FimPosition:
    """Position of fim middle token."""

    MIDDLE = auto()
    END = auto()


@dataclasses.dataclass
class CompletionTemplate:
    """A class that manages completion prompt templates. only for code completion currently."""

    # The name of this template
    name: str

    # the fim begin token
    fim_begin_token: str

    # The fim middle token
    fim_middle_token: str

    # The fim end token
    fim_end_token: str

    # The position of the fim middle token
    fim_position: FimPosition


# A global registry for all completion templates
completion_templates: dict[str, CompletionTemplate] = {}


def register_completion_template(template: CompletionTemplate, override: bool = False):
    """Register a new completion template."""
    if not override:
        assert (
            template.name not in completion_templates
        ), f"{template.name} has been registered."

    completion_templates[template.name] = template


def completion_template_exists(template_name: str) -> bool:
    """Return whether the named completion template is registered."""
    return template_name in completion_templates


def generate_completion_prompt_from_request(
    request: CompletionRequest, template_name: str
) -> str:
    if request.suffix == "":
        return request.prompt

    return generate_completion_prompt(request.prompt, request.suffix, template_name)


def generate_completion_prompt(prompt: str, suffix: str, template_name: str) -> str:
    completion_template = completion_templates[template_name]
    fim_begin_token = completion_template.fim_begin_token
    fim_middle_token = completion_template.fim_middle_token
    fim_end_token = completion_template.fim_end_token
    fim_position = completion_template.fim_position

    if fim_position == FimPosition.MIDDLE:
        prompt = f"{fim_begin_token}{prompt}{fim_middle_token}{suffix}{fim_end_token}"
    elif fim_position == FimPosition.END:
        prompt = f"{fim_begin_token}{prompt}{fim_end_token}{suffix}{fim_middle_token}"

    return prompt


register_completion_template(
    CompletionTemplate(
        name="deepseek_coder",
        fim_begin_token="<｜fim▁begin｜>",
        fim_middle_token="<｜fim▁hole｜>",
        fim_end_token="<｜fim▁end｜>",
        fim_position=FimPosition.MIDDLE,
    )
)


register_completion_template(
    CompletionTemplate(
        name="star_coder",
        fim_begin_token="<fim_prefix>",
        fim_middle_token="<fim_middle>",
        fim_end_token="<fim_suffix>",
        fim_position=FimPosition.END,
    )
)

register_completion_template(
    CompletionTemplate(
        name="qwen_coder",
        fim_begin_token="<|fim_prefix|>",
        fim_middle_token="<|fim_middle|>",
        fim_end_token="<|fim_suffix|>",
        fim_position=FimPosition.END,
    )
)
