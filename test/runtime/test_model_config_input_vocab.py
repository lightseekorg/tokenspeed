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

"""Input vocabulary resolution, cross-domain bounds, and runtime propagation."""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed.runtime.configs.model_config import ModelConfig  # noqa: E402
from tokenspeed.runtime.utils.server_args import ServerArgs  # noqa: E402


def _load(tmp_path, *, input_vocab_size, output_fields):
    config = dict(
        model_type="qwen3",
        architectures=["Qwen3ForCausalLM"],
        torch_dtype="bfloat16",
        vocab_size=input_vocab_size,
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
    config.update(output_fields)
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
    return ModelConfig(
        model_path=args.model,
        trust_remote_code=False,
        model_override_args="{}",
        server_args=args,
        dtype=args.dtype,
        context_length=args.max_model_len,
    )


@pytest.mark.parametrize(
    "output_fields",
    [
        {},
        {"output_vocab_size": None},
        {"output_vocab_size": 64},
        {"output_vocab_size": 128},
    ],
)
def test_input_and_output_vocabularies_are_independent(tmp_path, output_fields):
    model = _load(tmp_path, input_vocab_size=128, output_fields=output_fields)
    assert model.input_vocab_size == 128
    assert model.vocab_size == (output_fields.get("output_vocab_size") or 128)
    assert model.hf_text_config.vocab_size == 128


@pytest.mark.parametrize("size", [0, -1, True, 96.5])
@pytest.mark.parametrize("domain", ["input", "output"])
def test_invalid_vocabulary_is_rejected(tmp_path, size, domain):
    with pytest.raises(
        ValueError,
        match="Vocabulary sizes must be integers satisfying 0 < output <= input",
    ) as error:
        _load(
            tmp_path,
            input_vocab_size=size if domain == "input" else 128,
            output_fields={"output_vocab_size": size if domain == "output" else 64},
        )
    assert f"input_vocab_size={size if domain == 'input' else 128!r}" in str(
        error.value
    )
    assert f"vocab_size={size if domain == 'output' else 64!r}" in str(error.value)


def test_executor_config_keeps_both_vocabularies(tmp_path):
    import torch

    from tokenspeed.runtime.execution.model_executor import ModelExecutorConfig

    model = _load(
        tmp_path, input_vocab_size=128, output_fields={"output_vocab_size": 64}
    )
    args = ServerArgs(
        model=str(tmp_path),
        device="cpu",
        dtype="bfloat16",
        world_size=1,
        max_num_seqs=2,
        chunked_prefill_size=128,
        enforce_eager=True,
        disable_prefill_graph=True,
    )
    args.mapping.rank = 0
    executor = ModelExecutorConfig.from_server_args(
        args,
        model,
        torch.nn.Module(),
        max_req_pool_size=4,
        gpu_id=0,
        global_rank=0,
        prefix_granularity=1,
    )
    assert executor.input_vocab_size == 128
    assert executor.vocab_size == 64


def test_draft_media_substitute_uses_input_vocabulary(tmp_path):
    import torch

    from tokenspeed.runtime.execution.input_buffer import InputBuffers
    from tokenspeed.runtime.execution.multimodal_runtime import MultimodalRuntime
    from tokenspeed.runtime.multimodal.inputs import Modality, MultimodalDataItem

    model = _load(
        tmp_path,
        input_vocab_size=128,
        output_fields={"output_vocab_size": 64, "image_token_id": 100},
    )
    buffers = InputBuffers(
        max_bs=1, max_num_tokens=4, state_write_padding_pool_index=2, device="cpu"
    )
    MultimodalRuntime.wire_drafter(buffers, model)
    item = MultimodalDataItem(modality=Modality.IMAGE, hash=123)
    item.set_pad_value()
    tokens = torch.tensor([10, item.pad_value], dtype=torch.int64)
    # The real substitution keeps the checkpoint's input-only token ID.
    from tokenspeed.runtime.multimodal.inputs import substitute_mm_pad_

    substitute_mm_pad_(tokens, buffers.mm_pad_substitute_ids)
    assert tokens.tolist() == [10, 100]


def test_output_vocabulary_must_be_embeddable(tmp_path):
    with pytest.raises(
        ValueError,
        match="Vocabulary sizes must be integers satisfying 0 < output <= input",
    ):
        _load(tmp_path, input_vocab_size=64, output_fields={"output_vocab_size": 128})
