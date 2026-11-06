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

"""Checkpoint normalization and immutable vocabulary domains."""

import os
import sys
from dataclasses import FrozenInstanceError

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="runtime-1gpu")

import pytest
from transformers import PretrainedConfig

from tokenspeed.runtime.configs.vocabulary import VocabularySpec


@pytest.mark.parametrize("output_size", [None, 64, 128])
def test_checkpoint_output_domain_keeps_input_configuration(output_size):
    config = PretrainedConfig(vocab_size=128, output_vocab_size=output_size)
    before = config.to_dict()
    vocabulary = VocabularySpec.from_hf_config(config, is_draft=False)
    assert vocabulary.input_size == 128
    assert vocabulary.output_size == (128 if output_size is None else output_size)
    assert config.to_dict() == before


def test_missing_output_domain_uses_input_size():
    assert VocabularySpec.from_hf_config(
        PretrainedConfig(vocab_size=128), is_draft=False
    ) == VocabularySpec(input_size=128, output_size=128)


@pytest.mark.parametrize("size", [0, -1, True, 3.5])
@pytest.mark.parametrize("domain", ["input", "output"])
def test_invalid_domain_sizes_fail_before_model_construction(size, domain):
    with pytest.raises(ValueError, match=f"{domain} vocabulary"):
        VocabularySpec(
            input_size=size if domain == "input" else 128,
            output_size=size if domain == "output" else 64,
        )


def test_reduced_draft_head_is_resolved_only_for_drafts():
    config = PretrainedConfig(vocab_size=128, draft_vocab_size=32)
    assert VocabularySpec.from_hf_config(config, is_draft=True).output_size == 32
    assert VocabularySpec.from_hf_config(config, is_draft=False).output_size == 128
    config.output_vocab_size = 64
    assert VocabularySpec.from_hf_config(config, is_draft=True).output_size == 64


def test_shared_domain_definition_cannot_be_mutated():
    vocabulary = VocabularySpec(input_size=128, output_size=64)
    with pytest.raises(FrozenInstanceError):
        vocabulary.output_size = 128
