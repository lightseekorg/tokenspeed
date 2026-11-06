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

"""Logical vocabulary domains shared by model construction and execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PretrainedConfig


@dataclass(frozen=True, kw_only=True)
class VocabularySpec:
    """Logical input-ID capacity and output width, before parallel padding.

    Target models use output columns as token IDs in the input space. Draft
    adapters own any draft-column to target-token mapping. Neither size is a
    tokenizer length, a local shard width, or a padded parameter dimension.
    """

    input_size: int
    output_size: int

    def __post_init__(self) -> None:
        for domain, size in (("input", self.input_size), ("output", self.output_size)):
            if type(size) is not int or size <= 0:
                raise ValueError(
                    f"The {domain} vocabulary size must be a positive integer"
                )

    @classmethod
    def from_hf_config(
        cls, config: PretrainedConfig, *, is_draft: bool
    ) -> VocabularySpec:
        """Normalize the common checkpoint convention at the loading boundary.

        Model profiles provide their own spec when a checkpoint uses other
        field meanings. A missing/null output size means the full vocabulary;
        a draft's explicitly reduced head takes precedence over that fallback.
        """
        get_text_config = getattr(config, "get_text_config", None)
        config = (
            get_text_config()
            if get_text_config is not None
            else getattr(config, "text_config", config)
        )
        output_size = getattr(config, "output_vocab_size", None)
        if output_size is None and is_draft:
            output_size = getattr(config, "draft_vocab_size", None)
        return cls(
            input_size=config.vocab_size,
            output_size=config.vocab_size if output_size is None else output_size,
        )
