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

"""Runtime feature flags must flow through ``ServerArgs``, never process env."""

from __future__ import annotations

import asyncio
import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed.runtime.configs.model_config import ModelConfig
from tokenspeed.runtime.engine.input_processor import InputProcessor
from tokenspeed.runtime.engine.io_struct import GenerateReqInput
from tokenspeed.runtime.multimodal.inputs import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from tokenspeed.runtime.utils.server_args import ServerArgs


def _build_model_config(*, allow_longer_context: bool) -> ModelConfig:
    hf_config = SimpleNamespace(
        architectures=["LlamaForCausalLM"],
        dtype=torch.bfloat16,
        model_type="llama",
        hidden_size=16,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_hidden_layers=1,
        vocab_size=32,
        eos_token_id=None,
    )
    with patch.object(ServerArgs, "__post_init__"):
        server_args = ServerArgs(
            model="stub",
            allow_overwrite_longer_context_len=allow_longer_context,
        )
    server_args.mapping = None

    with (
        patch(
            "tokenspeed.runtime.configs.model_config.get_config",
            return_value=hf_config,
        ),
        patch(
            "tokenspeed.runtime.configs.model_config.get_generation_config",
            return_value=None,
        ),
        patch(
            "tokenspeed.runtime.configs.model_config.get_hf_text_config",
            return_value=hf_config,
        ),
        patch(
            "tokenspeed.runtime.configs.model_config.get_context_length",
            return_value=1024,
        ),
        patch.object(ModelConfig, "_verify_quantization"),
    ):
        return ModelConfig(
            "stub",
            context_length=2048,
            model_override_args="{}",
            server_args=server_args,
        )


def test_longer_context_override_only_reads_server_args() -> None:
    legacy_name = "TOKENSPEED_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"

    with patch.dict(os.environ, {legacy_name: "1"}):
        with pytest.raises(ValueError, match="--allow-overwrite-longer-context-len"):
            _build_model_config(allow_longer_context=False)

    with patch.dict(os.environ, {legacy_name: "0"}):
        config = _build_model_config(allow_longer_context=True)

    assert config.context_len == 2048


def test_default_multimodal_hashing_ignores_legacy_environment() -> None:
    first = MultimodalDataItem(modality=Modality.IMAGE, feature=b"same image")
    second = MultimodalDataItem(modality=Modality.IMAGE, feature=b"same image")
    inputs = MultimodalInputs(mm_items=[first, second])

    with patch.dict(os.environ, {"TOKENSPEED_MM_SKIP_COMPUTE_HASH": "1"}):
        inputs.ensure_pad_values()

    assert first.hash == second.hash
    assert first.pad_value == second.pad_value


class _StubTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [0] * len(text)


def test_input_processor_propagates_explicit_multimodal_hash_policy() -> None:
    item = MultimodalDataItem(
        modality=Modality.IMAGE,
        hash=42,
        feature=b"image",
        offsets=[(1, 1)],
    )
    inputs = MultimodalInputs(
        mm_items=[item],
        mrope_positions=torch.zeros((3, 3), dtype=torch.int64),
    )
    engine = SimpleNamespace(
        context_len=100,
        is_generation=True,
        tokenizer=_StubTokenizer(),
        logger=SimpleNamespace(warning=lambda *args, **kwargs: None),
        server_args=SimpleNamespace(
            reasoning_parser=None,
            enable_prefix_caching=False,
            enable_output_logprobs=False,
            mm_skip_compute_hash=True,
        ),
        model_config=SimpleNamespace(
            vocab_size=32000,
            is_multimodal_active=True,
            hf_config=SimpleNamespace(),
        ),
    )
    request = GenerateReqInput(
        input_ids=[10, 99, 11],
        sampling_params={},
        precomputed_multimodal_inputs=inputs,
    )
    request.normalize_batch_and_arguments()

    random_hash = 12345
    with (
        patch.dict(os.environ, {"TOKENSPEED_MM_SKIP_COMPUTE_HASH": "0"}),
        patch(
            "tokenspeed.runtime.multimodal.inputs.uuid.uuid4",
            return_value=SimpleNamespace(int=random_hash),
        ),
    ):
        output = asyncio.run(InputProcessor(engine).tokenize_one_request(request))

    assert item.hash == random_hash
    assert output.input_ids[1] == item.pad_value
