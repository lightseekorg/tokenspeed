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
import json
import os
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

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
from tokenspeed.runtime.utils.common import (
    configure_logger,
    maybe_inference_mode,
    maybe_model_redirect,
    maybe_set_numa_aware_cpu_affinity,
    prepare_model_and_tokenizer,
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


def test_model_redirect_uses_only_the_explicit_path(tmp_path) -> None:
    redirect_path = tmp_path / "redirects.json"
    redirect_path.write_text(json.dumps({"source/model": "/models/local"}))

    with patch.dict(
        os.environ,
        {"TOKENSPEED_MODEL_REDIRECT_PATH": str(redirect_path)},
    ):
        assert maybe_model_redirect("source/model") == "source/model"
    assert maybe_model_redirect("source/model", str(redirect_path)) == "/models/local"

    with patch.object(ServerArgs, "__post_init__"):
        server_args = ServerArgs(
            model="source/model",
            model_redirect_path=str(redirect_path),
        )
    server_args.resolve_basic_defaults()
    assert server_args.model == "/models/local"
    assert server_args.tokenizer == "/models/local"


def test_modelscope_download_is_explicit() -> None:
    snapshot_download = Mock(side_effect=["/cache/model", "/cache/tokenizer"])
    modelscope = ModuleType("modelscope")
    modelscope.snapshot_download = snapshot_download

    with patch.dict(os.environ, {"TOKENSPEED_USE_MODELSCOPE": "1"}):
        assert prepare_model_and_tokenizer(
            "remote/model",
            "remote/tokenizer",
        ) == ("remote/model", "remote/tokenizer")

    with (
        patch.dict(sys.modules, {"modelscope": modelscope}),
        patch("tokenspeed.runtime.utils.common.os.path.exists", return_value=False),
    ):
        resolved = prepare_model_and_tokenizer(
            "remote/model",
            "remote/tokenizer",
            use_modelscope=True,
        )

    assert resolved == ("/cache/model", "/cache/tokenizer")
    assert snapshot_download.call_args_list[0].args == ("remote/model",)
    assert snapshot_download.call_args_list[1].args == ("remote/tokenizer",)


def test_inference_context_is_a_stable_runtime_policy() -> None:
    assert not torch.is_inference_mode_enabled()
    with patch.dict(os.environ, {"TOKENSPEED_ENABLE_TORCH_INFERENCE_MODE": "0"}):
        with maybe_inference_mode():
            assert torch.is_inference_mode_enabled()


def test_numa_affinity_can_be_disabled_explicitly() -> None:
    with patch("tokenspeed.runtime.utils.common.current_platform") as platform:
        maybe_set_numa_aware_cpu_affinity(0, enabled=False)

    platform.assert_not_called()


def test_logging_config_uses_the_explicit_path(tmp_path) -> None:
    config = {
        "version": 1,
        "handlers": {"null": {"class": "logging.NullHandler"}},
        "root": {"handlers": ["null"], "level": "INFO"},
    }
    config_path = tmp_path / "logging.json"
    config_path.write_text(json.dumps(config))
    server_args = SimpleNamespace(
        log_level="info",
        logging_config_path=str(config_path),
    )

    with (
        patch("logging.config.dictConfig") as dict_config,
        patch("tokenspeed._logging.suppress_noisy_third_party_logs"),
    ):
        configure_logger(server_args)

    dict_config.assert_called_once_with(config)


def test_logging_config_ignores_the_legacy_environment(tmp_path) -> None:
    config_path = tmp_path / "logging.json"
    config_path.write_text(json.dumps({"version": 1}))
    server_args = SimpleNamespace(log_level="info", logging_config_path=None)

    with (
        patch.dict(
            os.environ,
            {"TOKENSPEED_LOGGING_CONFIG_PATH": str(config_path)},
        ),
        patch("logging.basicConfig") as basic_config,
        patch("logging.config.dictConfig") as dict_config,
        patch("tokenspeed._logging.suppress_noisy_third_party_logs"),
    ):
        configure_logger(server_args)

    basic_config.assert_called_once()
    dict_config.assert_not_called()
