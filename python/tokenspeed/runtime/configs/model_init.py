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

"""Explicit checkpoint and resolved vocabulary inputs for model construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from tokenspeed.runtime.configs.vocabulary import VocabularySpec

HFConfigT = TypeVar("HFConfigT", covariant=True)
OtherHFConfigT = TypeVar("OtherHFConfigT")


@dataclass(frozen=True, kw_only=True)
class ModelInitConfig(Generic[HFConfigT]):
    """Model construction inputs, without loading or serving policy.

    The vocabulary is resolved before construction. The frozen envelope does
    not freeze its model-specific HF config; adapters that modify checkpoint
    metadata must copy that config before passing it to another model.
    """

    hf_config: HFConfigT
    vocabulary: VocabularySpec

    def __post_init__(self) -> None:
        if not isinstance(self.vocabulary, VocabularySpec):
            raise TypeError("ModelInitConfig.vocabulary must be a VocabularySpec")

    def with_hf_config(
        self, hf_config: OtherHFConfigT
    ) -> ModelInitConfig[OtherHFConfigT]:
        """Select checkpoint metadata while preserving the resolved vocabulary.

        This neither copies nor normalizes the supplied HF config. The caller
        owns text-config selection and any draft-specific checkpoint changes.
        """
        return ModelInitConfig(hf_config=hf_config, vocabulary=self.vocabulary)
