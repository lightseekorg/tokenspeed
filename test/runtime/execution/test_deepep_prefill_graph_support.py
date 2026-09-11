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

"""Static DeepEP prefill opt-in checks against loaded model operations."""

import os
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci

from tokenspeed.runtime.execution.model_executor import (
    _validate_deepep_prefill_graph_support,
)
from tokenspeed.runtime.execution.model_runner import ModelRunner
from tokenspeed.runtime.layers.moe.expert import MoELayer
from tokenspeed.runtime.models.qwen3_5_moe import Qwen3_5MoeSparseMoeBlock

register_cuda_ci(est_time=5, suite="runtime-1gpu")


def _experts() -> MoELayer:
    experts = MoELayer.__new__(MoELayer)
    torch.nn.Module.__init__(experts)
    experts.plan = {
        "apply_kernel_name": "deep_gemm_deepep_fp8_moe_apply",
        "solution": "deep_gemm",
        "weight_dtype": "fp8",
        "a2a_backend": "deepep",
        "deepep_mode": "auto",
        "internal_activation_dtype": "input",
    }
    return experts


def _block() -> Qwen3_5MoeSparseMoeBlock:
    block = Qwen3_5MoeSparseMoeBlock.__new__(Qwen3_5MoeSparseMoeBlock)
    torch.nn.Module.__init__(block)
    block.use_deepep = True
    block.experts = _experts()
    return block


def _runner() -> ModelRunner:
    runner = ModelRunner.__new__(ModelRunner)
    runner.model = torch.nn.ModuleList([_block(), _block()])
    runner.model_config = SimpleNamespace(dtype=torch.bfloat16)
    runner.server_args = SimpleNamespace(
        all2all_backend="deepep",
        deepep_mode="auto",
        disaggregation_mode="null",
    )
    runner.mapping = SimpleNamespace(
        attn=SimpleNamespace(tp_size=2, dp_size=2, cp_size=1),
        moe=SimpleNamespace(tp_size=1, dp_size=1),
        pp_size=1,
        nnodes=1,
    )
    runner.is_draft_worker = False
    runner.is_generation = True
    runner.is_multimodal = True
    runner.is_multimodal_active = False
    runner.deepep_prefill_graph_enabled = False
    return runner


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        disable_prefill_graph=False,
        prefill_graph_max_tokens=2048,
        spec_algo=None,
        enforce_eager=False,
    )


def test_loaded_plan_and_whole_qwen_operation_are_supported():
    runner = _runner()
    assert runner.deepep_prefill_graph_unsupported_reason() is None
    assert _validate_deepep_prefill_graph_support(_config(), runner, True)
    # The architectural multimodal fact does not disable text-only serving.
    assert runner.is_multimodal
    assert not runner.is_multimodal_active


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("apply_kernel_name", "flashinfer_cutedsl_deepep_nvfp4_moe_apply"),
        ("apply_kernel_name", "deep_gemm_fp8_moe_apply"),
        ("solution", "auto"),
        ("weight_dtype", "unquant"),
        ("a2a_backend", "none"),
        ("deepep_mode", "normal"),
        ("deepep_mode", "low_latency"),
        ("internal_activation_dtype", "fp8"),
    ],
)
def test_every_resolved_plan_must_match_the_supported_kernel(key, value):
    runner = _runner()
    # Keep one layer supported: checking only the first would miss this.
    runner.model[1].experts.plan[key] = value
    with pytest.raises(ValueError, match="must resolve.*DeepGEMM"):
        _validate_deepep_prefill_graph_support(_config(), runner, True)


def test_requested_backend_cannot_enable_a_tp_model_path():
    runner = _runner()
    runner.model[1].use_deepep = False
    with pytest.raises(ValueError, match="did not select its DeepEP"):
        _validate_deepep_prefill_graph_support(_config(), runner, True)


def test_unwrapped_moe_operation_is_rejected():
    runner = _runner()
    runner.model.append(_experts())
    with pytest.raises(ValueError, match="complete Qwen graph break"):
        _validate_deepep_prefill_graph_support(_config(), runner, True)


def test_unrecognized_deepep_operation_is_rejected():
    runner = _runner()
    custom = torch.nn.Module()
    custom.plan = {"a2a_backend": "deepep"}
    runner.model.append(custom)
    with pytest.raises(ValueError, match="complete Qwen graph break"):
        _validate_deepep_prefill_graph_support(_config(), runner, True)


def test_dense_model_is_not_enabled_by_deepep_argument():
    runner = _runner()
    runner.model = torch.nn.Linear(4, 4)
    with pytest.raises(ValueError, match="no supported Qwen"):
        _validate_deepep_prefill_graph_support(_config(), runner, True)


def test_an_overridden_qwen_block_requires_its_own_integration_audit():
    class OtherQwenBlock(Qwen3_5MoeSparseMoeBlock):
        pass

    runner = _runner()
    other = OtherQwenBlock.__new__(OtherQwenBlock)
    torch.nn.Module.__init__(other)
    other.experts = _experts()
    other.use_deepep = True
    runner.model.append(other)
    with pytest.raises(ValueError, match="complete Qwen graph break"):
        _validate_deepep_prefill_graph_support(_config(), runner, True)


def test_fp16_activations_are_rejected():
    runner = _runner()
    runner.model_config.dtype = torch.float16
    with pytest.raises(ValueError, match="BF16 activations"):
        _validate_deepep_prefill_graph_support(_config(), runner, True)


@pytest.mark.parametrize("mode", ["normal", "low_latency"])
def test_pinned_dispatch_mode_is_rejected_even_with_auto_kernel_plan(mode):
    runner = _runner()
    runner.server_args.deepep_mode = mode
    with pytest.raises(ValueError, match="--deepep-mode auto"):
        _validate_deepep_prefill_graph_support(_config(), runner, True)


@pytest.mark.parametrize("mode", ["prefill", "decode", "encode"])
def test_disaggregation_is_rejected(mode):
    runner = _runner()
    runner.server_args.disaggregation_mode = mode
    config = _config()
    config.enforce_eager = True
    with pytest.raises(ValueError, match="disaggregation is not supported"):
        _validate_deepep_prefill_graph_support(config, runner, True)


@pytest.mark.parametrize(
    ("group", "field", "value", "reason"),
    [
        ("attn", "cp_size", 2, "context and pipeline"),
        (None, "pp_size", 2, "context and pipeline"),
        ("moe", "tp_size", 2, "MoE TP must be 1"),
        ("moe", "dp_size", 2, "one EP group"),
        (None, "nnodes", 2, "single-node"),
    ],
)
def test_unsupported_parallelism_is_rejected(group, field, value, reason):
    runner = _runner()
    mapping = runner.mapping if group is None else getattr(runner.mapping, group)
    setattr(mapping, field, value)
    with pytest.raises(ValueError, match=reason):
        _validate_deepep_prefill_graph_support(_config(), runner, True)


@pytest.mark.parametrize("algorithm", ["MTP", "EAGLE3", "DFLASH"])
def test_speculative_decoding_is_rejected(algorithm):
    config = _config()
    config.spec_algo = algorithm
    with pytest.raises(ValueError, match="speculative decoding"):
        _validate_deepep_prefill_graph_support(config, _runner(), True)


def test_draft_worker_is_rejected():
    runner = _runner()
    runner.is_draft_worker = True
    with pytest.raises(ValueError, match="draft workers"):
        _validate_deepep_prefill_graph_support(_config(), runner, True)


@pytest.mark.parametrize("dp_size", [1, 2])
def test_actual_multimodal_execution_is_rejected(dp_size):
    runner = _runner()
    runner.mapping.attn.dp_size = dp_size
    runner.is_multimodal_active = True
    with pytest.raises(ValueError, match="--language-model-only"):
        _validate_deepep_prefill_graph_support(_config(), runner, True)


def test_attention_capability_exclusions_are_preserved():
    with pytest.raises(ValueError, match="attention backend"):
        _validate_deepep_prefill_graph_support(_config(), _runner(), False)


def test_explicit_eager_conflict_has_an_actionable_error():
    config = _config()
    config.enforce_eager = True
    with pytest.raises(ValueError, match="--enforce-eager.*max-tokens 0"):
        _validate_deepep_prefill_graph_support(config, _runner(), True)


@pytest.mark.parametrize("disabled", ["zero", "flag", "other_all2all"])
def test_disabled_graphs_do_not_validate_or_import_model_kernels(disabled):
    config = _config()
    runner = _runner()
    runner.deepep_prefill_graph_unsupported_reason = Mock(
        side_effect=AssertionError("disabled graph must not inspect model plans")
    )
    if disabled == "zero":
        config.prefill_graph_max_tokens = 0
    elif disabled == "flag":
        config.disable_prefill_graph = True
    else:
        runner.server_args.all2all_backend = "flashinfer_nvlink_two_sided"
    assert not _validate_deepep_prefill_graph_support(config, runner, True)
    runner.deepep_prefill_graph_unsupported_reason.assert_not_called()


@pytest.mark.parametrize("active", [False, True])
def test_runner_preserves_authoritative_multimodal_fact(active):
    model_config = SimpleNamespace(
        is_generation=True,
        is_multimodal=True,
        is_multimodal_active=active,
    )
    server_args = SimpleNamespace(
        device="cpu",
        mapping=object(),
        draft_moe_backend=None,
        moe_backend="deep_gemm",
        kv_cache_dtype="bfloat16",
        enable_memory_saver=False,
    )
    module = "tokenspeed.runtime.execution.model_runner"
    with (
        patch(f"{module}.global_server_args_dict_update"),
        patch(f"{module}.initialize_moe_config"),
        patch(f"{module}.TorchMemorySaverAdapter.create"),
        patch.object(ModelRunner, "load_model"),
    ):
        runner = ModelRunner(model_config, server_args, 0, 0, False)
    assert runner.is_multimodal
    assert runner.is_multimodal_active is active
    assert runner.deepep_prefill_graph_enabled is False
