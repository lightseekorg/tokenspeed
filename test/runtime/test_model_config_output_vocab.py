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

"""Output vocabulary controls sampling without shrinking input embeddings."""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed.runtime.configs.model_config import ModelConfig  # noqa: E402
from tokenspeed.runtime.utils.server_args import ServerArgs  # noqa: E402


@pytest.mark.parametrize("output_vocab_size", [None, 64, 128])
def test_output_vocab_preserves_input_vocabulary(tmp_path, output_vocab_size):
    config = dict(
        model_type="qwen3",
        architectures=["Qwen3ForCausalLM"],
        torch_dtype="bfloat16",
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=256,
        bos_token_id=1,
        eos_token_id=2,
    )
    if output_vocab_size is not None:
        config["output_vocab_size"] = output_vocab_size
    (tmp_path / "config.json").write_text(json.dumps(config))
    args = ServerArgs(
        model=str(tmp_path),
        device="cpu",
        dtype="bfloat16",
        trust_remote_code=False,
        max_model_len=256,
        world_size=1,
        max_num_seqs=2,
        chunked_prefill_size=128,
        enforce_eager=True,
        disable_prefill_graph=True,
    )
    model = ModelConfig(
        model_path=args.model,
        trust_remote_code=False,
        model_override_args="{}",
        server_args=args,
        dtype=args.dtype,
        context_length=args.max_model_len,
    )
    assert model.vocab_size == (128 if output_vocab_size is None else output_vocab_size)
    assert model.hf_text_config.vocab_size == 128
