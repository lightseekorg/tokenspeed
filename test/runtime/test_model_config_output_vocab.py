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

"""Loaded output vocabularies agree with the runtime logits configuration."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")


@pytest.mark.parametrize("tp_size", [1, 2])
@pytest.mark.parametrize("actual,expected", [(128, 128), (64, 64), (128, 64)])
def test_loaded_output_vocabulary_matches_runtime(actual, expected, tp_size):
    import torch
    from transformers import PretrainedConfig

    from tokenspeed.runtime.execution.model_runner import validate_output_vocabulary
    from tokenspeed.runtime.layers.logits_processor import LogitsProcessor
    from tokenspeed.runtime.layers.vocab_parallel_embedding import ParallelLMHead

    model = torch.nn.Module()
    model.lm_head = ParallelLMHead(
        actual,
        16,
        padding_size=256,
        tp_rank=0,
        tp_size=tp_size,
        tp_group=tuple(range(tp_size)),
    )
    model.logits_processor = LogitsProcessor(
        PretrainedConfig(vocab_size=actual),
        attn_tp_size=tp_size,
        tp_rank=0,
        tp_size=tp_size,
        tp_group=tuple(range(tp_size)),
        dp_lm_head_tp=False,
    )
    if actual == expected:
        validate_output_vocabulary(model, expected)
    else:
        with pytest.raises(ValueError, match="logical output vocabulary of 128.*64"):
            validate_output_vocabulary(model, expected)


def test_workers_without_logits_have_no_output_vocabulary():
    import torch

    from tokenspeed.runtime.execution.model_runner import validate_output_vocabulary

    model = torch.nn.Module()
    model.logits_processor = None
    validate_output_vocabulary(model, 64)
