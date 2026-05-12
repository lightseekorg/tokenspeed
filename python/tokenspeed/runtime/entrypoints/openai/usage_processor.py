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

from collections.abc import Mapping
from typing import Any, final

from tokenspeed.runtime.entrypoints.openai.protocol import UsageInfo


@final
class UsageProcessor:
    """Stateless helpers that turn raw token counts into a UsageInfo."""

    @staticmethod
    def _details_if_cached(
        count: int, enable_cache_report: bool = False
    ) -> dict[str, int] | None:
        """Return {"cached_tokens": N} only when N > 0 (keeps JSON slim)."""
        if enable_cache_report:
            # When cache report is enabled, always return the details even if count is 0
            return {"cached_tokens": count}
        else:
            # When cache report is disabled, only return details when count > 0
            return {"cached_tokens": count} if count > 0 else None

    @staticmethod
    def calculate_response_usage(
        responses: list[dict[str, Any]],
        n_choices: int = 1,
        enable_cache_report: bool = False,
    ) -> UsageInfo:
        completion_tokens = sum(r["meta_info"]["completion_tokens"] for r in responses)

        prompt_tokens = sum(
            responses[i]["meta_info"]["prompt_tokens"]
            for i in range(0, len(responses), n_choices)
        )

        cached_details = None
        if enable_cache_report:
            # For batch requests (n > 1), all choices share the same prompt,
            # so cached_tokens should only be counted once (from the first choice)
            cached_total = sum(
                responses[i]["meta_info"].get("cached_tokens", 0)
                for i in range(0, len(responses), n_choices)
            )
            cached_details = UsageProcessor._details_if_cached(
                cached_total, enable_cache_report
            )

        spec_verify_ct_list = []
        for r in responses:
            if r["meta_info"].get("spec_verify_ct"):
                spec_verify_ct_list.append(r["meta_info"]["spec_verify_ct"])
        spec_verify_ct = sum(spec_verify_ct_list)
        accept_draft_tokens = (
            (completion_tokens - len(responses)) / spec_verify_ct
            if len(spec_verify_ct_list) > 0 and spec_verify_ct > 0
            else None
        )

        return UsageProcessor.calculate_token_usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            accept_draft_tokens=accept_draft_tokens,
            cached_tokens=cached_details,
        )

    @staticmethod
    def calculate_streaming_usage(
        prompt_tokens: Mapping[int, int],
        completion_tokens: Mapping[int, int],
        cached_tokens: Mapping[int, int],
        spec_verify_tokens: Mapping[int, int],
        n_choices: int,
        enable_cache_report: bool = False,
    ) -> UsageInfo:
        # index % n_choices == 0 marks the first choice of a prompt
        total_prompt_tokens = sum(
            tok for idx, tok in prompt_tokens.items() if idx % n_choices == 0
        )
        total_completion_tokens = sum(completion_tokens.values())
        # Count accept_draft_tokens, some may be None and some may not
        # If spec_verify_tokens are all None, then accept_draft_tokens is None, otherwise calculate based on spec_verify_tokens
        accept_draft_tokens = None
        if spec_verify_tokens is not None and len(spec_verify_tokens) > 0:
            valid_completion_tokens = []
            valid_verify_tokens = []
            for key in spec_verify_tokens.keys():
                if spec_verify_tokens[key] is not None:
                    valid_completion_tokens.append(completion_tokens[key])
                    valid_verify_tokens.append(spec_verify_tokens[key])
            if len(valid_verify_tokens) > 0:
                verify_tokens = sum(valid_verify_tokens)
                completion_tokens = sum(valid_completion_tokens)
                accept_draft_tokens = (
                    (completion_tokens - len(valid_completion_tokens)) / verify_tokens
                    if verify_tokens > 0
                    else None
                )

        # For batch requests (n > 1), all choices share the same prompt,
        # so cached_tokens should only be counted once (from the first choice)
        total_cached_tokens = (
            sum(tok for idx, tok in cached_tokens.items() if idx % n_choices == 0)
            if enable_cache_report
            else 0
        )
        cached_details = (
            UsageProcessor._details_if_cached(total_cached_tokens, enable_cache_report)
            if enable_cache_report
            else None
        )

        return UsageProcessor.calculate_token_usage(
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            accept_draft_tokens=accept_draft_tokens,
            cached_tokens=cached_details,
        )

    @staticmethod
    def calculate_token_usage(
        prompt_tokens: int,
        completion_tokens: int,
        accept_draft_tokens: dict[str, int] | None = None,
        cached_tokens: dict[str, int] | None = None,
    ) -> UsageInfo:
        """Calculate token usage information"""
        return UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            accept_draft_tokens=accept_draft_tokens,
            prompt_tokens_details=cached_tokens,
        )
