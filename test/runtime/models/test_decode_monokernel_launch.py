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

"""``--decode-monokernel`` at launch: a model without decode monokernels fails
startup instead of serving without them, and a model with them reads the
switch from its config. CPU only."""

from __future__ import annotations

import json
import os
import sys

import pytest

# The CI runner executes each registered test file as a standalone script.
_TEST_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _TEST_DIR)
sys.path.insert(0, os.path.dirname(_TEST_DIR))
from test.runtime.models.conftest import model_config, server_args  # noqa: E402

from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")


def test_the_switch_fails_a_model_without_monokernels(checkpoints) -> None:
    with pytest.raises(ValueError, match="no decode monokernel for.*Qwen3ForCausalLM"):
        server_args(checkpoints["other"], decode_monokernel=True)


def test_without_the_switch_the_model_launches(checkpoints) -> None:
    assert not server_args(checkpoints["other"]).decode_monokernel


@pytest.mark.parametrize("switch", [False, True])
def test_the_switch_reaches_the_model_config(tmp_path, switch) -> None:
    """A Qwen4-Exp model builds its monokernel layers when its text config
    says so: ModelConfig sets it from the switch."""
    config = {
        "architectures": ["Qwen4ExpForCausalLM"],
        "model_type": "qwen4_exp_text",
        "vocab_size": 128,
        "hidden_size": 128,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "max_position_embeddings": 2048,
        "torch_dtype": "bfloat16",
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    args = server_args(tmp_path, decode_monokernel=switch)
    assert model_config(args).hf_text_config.decode_monokernel is switch
