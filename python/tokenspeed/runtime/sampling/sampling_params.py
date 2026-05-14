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

"""Sampling parameters for text generation."""

import zlib
from typing import Any

_SAMPLING_EPS = 1e-6

# Sentinel for "top_k is disabled" (sample from whole vocab). We rewrite
# top_k=-1 (API convention) to this value so downstream code can pass it
# unchanged to top_k kernels that expect a positive cutoff.
_TOP_K_DISABLED = 1 << 30


class SamplingParams:
    """
    The sampling parameters.

    See docs/backend/sampling_params.md or
    https://docs.tokenspeed.ai/backend/sampling_params.html
    for the documentation.
    """

    def __init__(
        self,
        max_new_tokens: int | None = None,
        stop: str | list[str] | None = None,
        stop_token_ids: list[int] | None = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        min_p: float = 0.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
        min_new_tokens: int = 0,
        json_schema: str | None = None,
        regex: str | None = None,
        ebnf: str | None = None,
        structural_tag: str | None = None,
        ignore_eos: bool = False,
        skip_special_tokens: bool = True,
        spaces_between_special_tokens: bool = True,
        no_stop_trim: bool = False,
        thinking_budget: int | None = None,
        custom_params: dict[str, Any] | None = None,
        stream_interval: int | None = None,
        logit_bias: dict[str, float] | None = None,
        seed: int | None = None,
        # OpenAI-compat: `n` is a request-level fanout (number of choices)
        # that the serving layer forwards on every sampling_params dict.
        # TokenSpeed does not multiplex a single request into n completions,
        # so accept and ignore.
        n: int = 1,
    ) -> None:
        self.max_new_tokens = max_new_tokens
        self.stop_strs = stop
        if stop_token_ids:
            self.stop_token_ids = set(stop_token_ids)
        else:
            self.stop_token_ids = None
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.repetition_penalty = repetition_penalty
        self.min_new_tokens = min_new_tokens
        self.regex = regex
        self.json_schema = json_schema
        self.ebnf = ebnf
        self.structural_tag = structural_tag
        self.ignore_eos = ignore_eos
        self.skip_special_tokens = skip_special_tokens
        self.spaces_between_special_tokens = spaces_between_special_tokens
        self.no_stop_trim = no_stop_trim
        self.custom_params = custom_params
        self.thinking_budget = thinking_budget
        self.stream_interval = stream_interval
        self.logit_bias = logit_bias
        self.seed = seed

        # Process some special cases
        if self.temperature < _SAMPLING_EPS:
            # top_k = 1 means greedy sampling
            self.temperature = 1.0
            self.top_k = 1
        if self.top_k == -1:
            self.top_k = _TOP_K_DISABLED

    def verify(self, vocab_size: int) -> None:
        if self.temperature < 0.0:
            raise ValueError(
                f"temperature must be non-negative, got {self.temperature}."
            )
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}.")
        if not 0.0 <= self.min_p <= 1.0:
            raise ValueError(f"min_p must be in [0, 1], got {self.min_p}.")
        if self.top_k < -1 or self.top_k == 0:
            raise ValueError(
                f"top_k must be -1 (disable), or at least 1, " f"got {self.top_k}."
            )
        if not -2.0 <= self.frequency_penalty <= 2.0:
            raise ValueError(
                "frequency_penalty must be in [-2, 2], got "
                f"{self.frequency_penalty}."
            )
        if not -2.0 <= self.presence_penalty <= 2.0:
            raise ValueError(
                "presence_penalty must be in [-2, 2], got " f"{self.presence_penalty}."
            )
        if not 0.0 <= self.repetition_penalty <= 2.0:
            raise ValueError(
                "repetition_penalty must be in (0, 2], got "
                f"{self.repetition_penalty}."
            )
        if not 0 <= self.min_new_tokens:
            raise ValueError(
                f"min_new_tokens must be in (0, max_new_tokens], got "
                f"{self.min_new_tokens}."
            )
        if self.max_new_tokens is not None:
            if self.max_new_tokens < 0:
                raise ValueError(
                    f"max_new_tokens must be at least 0, got {self.max_new_tokens}."
                )
            if not self.min_new_tokens <= self.max_new_tokens:
                raise ValueError(
                    f"min_new_tokens must be in (0, max_new_tokens({self.max_new_tokens})], got "
                    f"{self.min_new_tokens}."
                )
        if self.logit_bias is not None:
            for token_id in self.logit_bias:
                if not 0 <= int(token_id) < vocab_size:
                    raise ValueError(
                        f"logit_bias must has keys in [0, {vocab_size - 1}], got "
                        f"{token_id}."
                    )

        grammars = [
            self.json_schema,
            self.regex,
            self.ebnf,
        ]  # since mutually exclusive, only one can be set
        if sum(x is not None for x in grammars) > 1:
            raise ValueError("Only one of regex, json_schema, or ebnf can be set.")

    def requested_features(self) -> "set[str]":
        """Return the set of backend-facing feature names this request needs.

        `temperature`, `top_k`, `top_p`, `min_p` each appear only when the
        corresponding field is not at its neutral default. Used by
        SamplingBackend.register() to reject requests asking for features
        the active backend does not implement."""
        out: set[str] = set()
        if abs(self.temperature - 1.0) > _SAMPLING_EPS:
            out.add("temperature")
        # top_k=_TOP_K_DISABLED and top_k=1 (greedy short-circuit from __init__) are neutral.
        if self.top_k != _TOP_K_DISABLED and self.top_k != 1:
            out.add("top_k")
        if self.top_p < 1.0:
            out.add("top_p")
        if self.min_p > 0.0:
            out.add("min_p")
        if self.frequency_penalty != 0.0:
            out.add("frequency_penalty")
        if self.presence_penalty != 0.0:
            out.add("presence_penalty")
        if self.repetition_penalty != 1.0:
            out.add("repetition_penalty")
        if self.logit_bias:
            out.add("logit_bias")
        return out

    def resolve_seed(self, rid: str) -> None:
        """If the caller didn't supply a seed, derive one deterministically
        from rid. Called at the single request-materialization point so all
        TP/DP ranks agree on the seed."""
        if self.seed is None:
            self.seed = zlib.crc32(rid.encode("utf-8")) & 0xFFFFFFFF

    def normalize(self, tokenizer) -> None:
        # Process stop strings
        if self.stop_strs is None:
            self.stop_strs = []
            self.stop_str_max_len = 0
        else:
            if isinstance(self.stop_strs, str):
                self.stop_strs = [self.stop_strs]

            stop_str_max_len = 0
            for stop_str in self.stop_strs:
                if tokenizer is not None:
                    stop_str_ids = tokenizer.encode(stop_str, add_special_tokens=False)
                    stop_str_max_len = max(stop_str_max_len, len(stop_str_ids))
                else:
                    stop_str_max_len = max(stop_str_max_len, len(stop_str))
            self.stop_str_max_len = stop_str_max_len
