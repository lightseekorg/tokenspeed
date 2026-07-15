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
import dataclasses
import inspect
import json
import os
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
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
    DEFAULT_AUDIO_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_IMAGE_REQUEST_TIMEOUT_SECONDS,
    configure_logger,
    get_bool_env_var,
    get_image_bytes,
    load_audio,
    maybe_inference_mode,
    maybe_model_redirect,
    maybe_set_numa_aware_cpu_affinity,
    prepare_model_and_tokenizer,
)
from tokenspeed.runtime.utils.network import get_ip
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


def test_vendor_environment_projection_overrides_inherited_values() -> None:
    from tokenspeed.runtime.entrypoints.engine import _set_envs_and_config

    inherited = {
        "NCCL_NVLS_ENABLE": "1",
        "NVIDIA_TF32_OVERRIDE": "1",
        "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "1",
    }
    args = SimpleNamespace(
        enable_symm_mem=True,
        enable_nccl_nvls=False,
        disable_tf32=True,
        enable_metrics=False,
    )
    with (
        patch.dict(os.environ, inherited, clear=False),
        patch("tokenspeed.runtime.entrypoints.engine.set_ulimit"),
        patch("tokenspeed.runtime.entrypoints.engine.signal.signal"),
        patch("tokenspeed.runtime.entrypoints.engine.mp.set_start_method"),
    ):
        _set_envs_and_config(args)

        assert os.environ["NCCL_CUMEM_ENABLE"] == "1"
        assert os.environ["NCCL_NVLS_ENABLE"] == "1"
        assert os.environ["NVIDIA_TF32_OVERRIDE"] == "0"
        assert os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] == "0"


def test_network_discovery_ignores_legacy_host_environment() -> None:
    with (
        patch.dict(
            os.environ,
            {"TOKENSPEED_HOST_IP": "198.51.100.1", "HOST_IP": "198.51.100.2"},
        ),
        patch(
            "tokenspeed.runtime.utils.network.socket.gethostname", return_value="node"
        ),
        patch(
            "tokenspeed.runtime.utils.network.socket.gethostbyname",
            return_value="192.0.2.10",
        ),
    ):
        assert get_ip() == "192.0.2.10"

    assert get_ip("203.0.113.7") == "203.0.113.7"
    with pytest.raises(ValueError, match="non-empty string"):
        get_ip("  ")


def test_mooncake_prefill_registration_prefers_explicit_advertised_host() -> None:
    from tokenspeed.runtime.pd.mooncake.prefill import MooncakeKVManagerPrefill

    manager = MooncakeKVManagerPrefill.__new__(MooncakeKVManagerPrefill)
    manager.dist_init_addr = "198.51.100.10:29500"
    manager.bootstrap_port = 8998
    manager.world_size = 4
    manager.dp_size = 1
    manager.rank_port = 9001
    manager.args = SimpleNamespace(
        advertised_host="203.0.113.7",
        enable_mla_l1_5_cache=False,
    )
    manager.kv_args = SimpleNamespace(
        engine_rank=0,
        kv_item_lens=[],
        kv_unit_lens=[],
        state_item_lens=[],
        state_unit_lens=[],
    )
    response = SimpleNamespace(status_code=200, text="")

    with (
        patch(
            "tokenspeed.runtime.pd.mooncake.prefill.get_local_ip_by_remote",
            return_value="192.0.2.8",
        ),
        patch(
            "tokenspeed.runtime.pd.mooncake.prefill.requests.put",
            return_value=response,
        ) as put,
    ):
        manager._register_to_bootstrap()

    assert put.call_args.args[0] == "http://198.51.100.10:8998/route"
    assert put.call_args.kwargs["json"]["rank_ip"] == "203.0.113.7"


def test_remote_image_timeout_is_an_explicit_api() -> None:
    response = Mock(content=b"image")
    with (
        patch.dict(os.environ, {"REQUEST_TIMEOUT": "99"}),
        patch(
            "tokenspeed.runtime.utils.common.requests.get", return_value=response
        ) as request,
    ):
        assert get_image_bytes("https://example.test/image") == b"image"

    request.assert_called_once_with(
        "https://example.test/image",
        timeout=DEFAULT_IMAGE_REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status.assert_called_once_with()
    response.close.assert_called_once_with()


def test_remote_audio_timeout_is_an_explicit_api() -> None:
    response = Mock(content=b"audio")
    audio = np.array([0.0, 0.5], dtype=np.float32)
    soundfile = SimpleNamespace(read=Mock(return_value=(audio, 16000)))
    with (
        patch.dict(os.environ, {"REQUEST_TIMEOUT": "99"}),
        patch.dict(sys.modules, {"soundfile": soundfile}),
        patch(
            "tokenspeed.runtime.utils.common.requests.get", return_value=response
        ) as request,
    ):
        output = load_audio(
            "https://example.test/audio",
            request_timeout=7.5,
        )

    request.assert_called_once_with(
        "https://example.test/audio", stream=True, timeout=7.5
    )
    np.testing.assert_array_equal(output, audio)
    response.raise_for_status.assert_called_once_with()
    response.close.assert_called_once_with()


def test_environment_boolean_reader_is_limited_to_ci_contracts() -> None:
    with pytest.raises(ValueError, match="Unsupported environment boolean"):
        get_bool_env_var("TOKENSPEED_HIDDEN_FEATURE")


def test_expert_recorder_file_dump_uses_explicit_directory(tmp_path) -> None:
    from tokenspeed.runtime.moe.distribution_recorder import _dump_to_file

    output_dir = tmp_path / "expert-records"
    with patch("tokenspeed.runtime.moe.distribution_recorder.torch.save") as save:
        _dump_to_file("record.pt", {"count": 1}, str(output_dir))

    save.assert_called_once_with({"count": 1}, str(output_dir / "record.pt"))
    assert output_dir.is_dir()


def test_new_scheduler_and_mooncake_options_preserve_positional_compatibility():
    from tokenspeed.runtime.engine.scheduler_utils import make_config
    from tokenspeed.runtime.pd.mooncake.entities import KVManagerArgs

    parameters = list(inspect.signature(make_config).parameters)
    assert parameters[-2:] == [
        "prefix_cache_adjunct",
        "enable_memory_debug_checks",
    ]

    fields = [field.name for field in dataclasses.fields(KVManagerArgs)]
    assert fields[-2:] == ["runtime_config", "advertised_host"]


def _tokenspeed_mla_config(*, prefill_backend: str, binary_so_path=None):
    from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig

    return MLAConfig(
        device="cpu",
        backend_name="tokenspeed_mla",
        num_attention_heads=2,
        num_kv_heads=2,
        head_dim=4,
        attn_tp_size=1,
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.float8_e4m3fn,
        page_size=64,
        context_len=1024,
        max_bs=4,
        max_graph_bs=4,
        kv_cache_quant_method="none",
        kv_lora_rank=2,
        qk_nope_head_dim=2,
        qk_rope_head_dim=2,
        v_head_dim=3,
        scaling=0.5,
        kv_cache_dim=4,
        tokenspeed_mla_prefill_backend=prefill_backend,
        tokenspeed_mla_prefill_binary_so_path=binary_so_path,
    )


def test_tokenspeed_mla_binary_prefill_skips_cutedsl_warmup_and_is_wired():
    from tokenspeed.runtime.layers.attention.backends import (
        tokenspeed_mla as backend_mod,
    )

    config = _tokenspeed_mla_config(
        prefill_backend="binary",
        binary_so_path="/opt/tokenspeed/fmha.so",
    )
    workspace = torch.empty(1, dtype=torch.int8)
    with (
        patch.dict(
            backend_mod.global_server_args_dict,
            {"kv_cache_dtype": "fp8_e4m3"},
        ),
        patch.object(
            backend_mod,
            "get_cutedsl_workspace_buffer",
            return_value=workspace,
        ),
        patch.object(
            backend_mod,
            "has_binary_prefill",
            return_value=True,
        ) as has_binary,
        patch.object(backend_mod, "warmup_compile_prefill") as warmup,
    ):
        backend = backend_mod.CuteDSLMLABackend(config)

    has_binary.assert_called_once_with("/opt/tokenspeed/fmha.so")
    warmup.assert_not_called()

    query = torch.zeros((2, 2, 4), dtype=torch.bfloat16)
    key = torch.zeros_like(query)
    value = torch.zeros((2, 2, 3), dtype=torch.bfloat16)
    expected = torch.zeros_like(value)
    cumulative = torch.tensor([0, 2], dtype=torch.int32)
    sequence_lengths = torch.tensor([2], dtype=torch.int32)
    with patch.object(
        backend_mod,
        "tokenspeed_mla_prefill",
        return_value=(expected, None),
    ) as prefill:
        output, _ = backend.forward_extend_chunked(
            query,
            key,
            value,
            scaling=0.5,
            logits_soft_cap=None,
            cum_seq_lens_q=cumulative,
            cum_seq_lens_kv=cumulative,
            max_q_len=2,
            max_kv_len=2,
            seq_lens=sequence_lengths,
            batch_size=1,
            causal=False,
        )

    assert output is expected
    assert prefill.call_args.kwargs["backend"] == "binary"
    assert prefill.call_args.kwargs["binary_so_path"] == "/opt/tokenspeed/fmha.so"


def test_tokenspeed_mla_cutedsl_prefill_keeps_explicit_warmup():
    from tokenspeed.runtime.layers.attention.backends import (
        tokenspeed_mla as backend_mod,
    )

    config = _tokenspeed_mla_config(prefill_backend="cutedsl")
    with (
        patch.dict(
            backend_mod.global_server_args_dict,
            {"kv_cache_dtype": "fp8_e4m3"},
        ),
        patch.object(
            backend_mod,
            "get_cutedsl_workspace_buffer",
            return_value=torch.empty(1, dtype=torch.int8),
        ),
        patch.object(backend_mod, "has_binary_prefill") as has_binary,
        patch.object(backend_mod, "warmup_compile_prefill") as warmup,
    ):
        backend_mod.CuteDSLMLABackend(config)

    has_binary.assert_not_called()
    warmup.assert_called_once()
