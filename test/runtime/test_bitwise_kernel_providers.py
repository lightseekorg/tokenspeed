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

"""Numerical requirements work with a registered provider of any name."""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=2, suite="runtime-1gpu")

import pytest
import torch
from tokenspeed_kernel.ops.moe import moe_apply, moe_plan
from tokenspeed_kernel.registry import KernelRegistry, KernelSpec
from tokenspeed_kernel.selection import NoKernelFoundError, kernel_override
from tokenspeed_kernel.signature import format_signatures
from transformers import PretrainedConfig

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.dense.unquant import UnquantizedLinearMethod
from tokenspeed.runtime.layers.linear import ReplicatedLinear
from tokenspeed.runtime.layers.logits_processor import LogitsMetadata, LogitsProcessor
from tokenspeed.runtime.layers.moe.utils import MoeBackend
from tokenspeed.runtime.layers.quantization.fp8 import Fp8Config
from tokenspeed.runtime.layers.vocab_parallel_embedding import ParallelLMHead
from tokenspeed.runtime.models import deepseek_v41 as v41
from tokenspeed.runtime.utils.env import global_server_args_dict


@pytest.fixture
def registry(monkeypatch):
    registry = KernelRegistry()
    monkeypatch.setattr(KernelRegistry, "_instance", registry)
    monkeypatch.setitem(global_server_args_dict, "numerics", "rl-bitwise")
    monkeypatch.setitem(global_server_args_dict, "logprob_order", "torch")
    return registry


def _mm(a, b, a_scales, b_scales, out_dtype, *, alpha, block_size, out=None):
    # Explicit K order makes this tiny reference independent of token count.
    result = torch.zeros(a.shape[0], b.shape[0], dtype=out_dtype, device=a.device)
    for k in range(a.shape[1]):
        result.add_(a[:, k, None] * b[None, :, k])
    if out is not None:
        out.copy_(result)
        return out
    return result


def _dense_provider(registry, invariant):
    registry.register(
        KernelSpec(
            name="public_reference_mm",
            family="gemm",
            mode="mm",
            solution="public_reference",
            format_signatures=format_signatures(("a", "b"), "dense", {torch.float32}),
            features=frozenset({"batch_invariant"}) if invariant else frozenset(),
        ),
        _mm,
    )


@pytest.mark.parametrize("with_bias", [False, True])
def test_lm_head_uses_its_planned_method_for_all_batch_sizes(registry, with_bias):
    _dense_provider(registry, True)
    head = ParallelLMHead(
        8, 4, params_dtype=torch.float32, tp_rank=0, tp_size=1, tp_group=(0,)
    )
    with torch.no_grad():
        head.weight.copy_(torch.arange(head.weight.numel()).reshape_as(head.weight) % 7)
    processor = LogitsProcessor(
        PretrainedConfig(vocab_size=8, hidden_size=4), dp_lm_head_tp=False
    )
    metadata = LogitsMetadata(ForwardMode.DECODE, query_shard=None)
    x = torch.arange(12, dtype=torch.float32).view(3, 4)
    bias = (
        torch.arange(head.weight.shape[0], dtype=torch.float32) if with_bias else None
    )
    expected = x @ head.weight[:8].t()
    if bias is not None:
        expected = expected + bias[:8]
    for inputs, reference in [
        (x, expected),
        (x[1:2], expected[1:2]),
        (x.flip(0), expected.flip(0)),
    ]:
        actual = processor._get_logits(
            inputs, head, metadata, embedding_bias=bias, require_full_vocab=True
        )
        assert torch.equal(actual, reference)
    assert head.quant_method.kernel_plan.kernel.name == "public_reference_mm"


def test_unquantized_initialization_rejects_a_provider_without_the_feature(registry):
    _dense_provider(registry, False)
    method = UnquantizedLinearMethod()
    with pytest.raises(NoKernelFoundError):
        method.create_weights(torch.nn.Module(), 4, [8], 4, 8, torch.float32)


def test_v41_bf16_dequantized_linear_plans_after_loading(registry, monkeypatch):
    registry.register(
        KernelSpec(
            name="public_reference_mm",
            family="gemm",
            mode="mm",
            solution="public_reference",
            format_signatures=format_signatures(("a", "b"), "dense", {torch.bfloat16}),
            features=frozenset({"batch_invariant"}),
        ),
        _mm,
    )
    # Hopper loads the FP8 codes into BF16 and folds the scales after loading.
    monkeypatch.setattr(
        v41, "current_platform", lambda: SimpleNamespace(is_hopper=True)
    )
    quant = v41.v41_mxfp8_config(
        Fp8Config.from_config(
            {
                "quant_method": "fp8",
                "activation_scheme": "dynamic",
                "weight_block_size": [32, 32],
                "scale_fmt": "ue8m0",
            }
        )
    )
    layer = ReplicatedLinear(
        64,
        32,
        bias=False,
        quant_config=quant,
        params_dtype=torch.bfloat16,
        prefix="model.proj",
    )
    codes = ((torch.arange(32 * 64).reshape(32, 64) % 9) - 4).to(torch.float8_e4m3fn)
    # E8M0 127 is 2**0, 128 is 2**1.
    scales = (torch.arange(2).reshape(1, 2) + 127).to(torch.uint8)
    layer.weight.weight_loader(layer.weight, codes)
    layer.weight_scale_inv.weight_loader(
        layer.weight_scale_inv, scales.view(torch.float8_e8m0fnu)
    )
    layer.quant_method.process_weights_after_loading(layer)

    assert isinstance(layer.quant_method, UnquantizedLinearMethod)
    assert layer.quant_method.kernel_plan.kernel.name == "public_reference_mm"
    weight = codes.float() * torch.tensor([1.0, 2.0]).repeat_interleave(32)
    x = (torch.arange(3 * 64).reshape(3, 64) % 5 - 2).to(torch.bfloat16)
    expected = _mm(
        x,
        weight.to(torch.bfloat16),
        None,
        None,
        torch.bfloat16,
        alpha=None,
        block_size=None,
    )
    for inputs, reference in [(x, expected), (x[1:2], expected[1:2])]:
        actual = layer.quant_method.apply(layer, inputs)
        assert torch.equal(actual, reference)


def test_dense_override_cannot_bypass_the_requirement(registry):
    _dense_provider(registry, False)
    with kernel_override("gemm", "mm", "public_reference"):
        with pytest.raises(NoKernelFoundError):
            ParallelLMHead(
                8, 4, params_dtype=torch.float32, tp_rank=0, tp_size=1, tp_group=(0,)
            )


def _expert_reference(*, x, w, topk_weights, topk_ids, **kwargs):
    # A single linear expert with a precomputed route, evaluated numerically.
    assert torch.all(topk_ids == 0)
    return (x @ w.weight.t()) * topk_weights[:, :1]


def _expert_provider(registry, invariant):
    registry.register(
        KernelSpec(
            name="public_reference_expert",
            family="moe",
            mode="apply",
            solution="public_reference",
            format_signatures=format_signatures("x", "dense", {torch.float32}),
            features=frozenset({"batch_invariant"}) if invariant else frozenset(),
            traits={
                "routing_mode": frozenset({"precomputed_topk"}),
                "combine_order": frozenset({"rank"}),
            },
        ),
        _expert_reference,
    )


def _plan(solution):
    return moe_plan(
        "unquant",
        input_dtype=torch.float32,
        routing_mode="precomputed_topk",
        a2a_backend="none",
        ep_size=1,
        ispp=8,
        hidden=4,
        swiglu_form=None,
        activation_clamped=False,
        expert_id_repeats=False,
        fast_math=False,
        combine_order="rank",
        features=frozenset({"batch_invariant"}),
        solution=solution,
    )


def test_expert_plan_keeps_the_callable_after_registration_changes(registry):
    _expert_provider(registry, True)
    assert MoeBackend("public_reference").value == "public_reference"
    plan = _plan(None)
    registry._unregister("public_reference_expert")
    weight = torch.nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        weight.weight.copy_(torch.eye(4) * 3)
    x = torch.arange(12, dtype=torch.float32).view(3, 4)
    for inputs in [x, x[1:2], x.flip(0)]:
        actual = moe_apply(
            plan,
            inputs,
            weight,
            None,
            topk_ids=torch.zeros(len(inputs), 1, dtype=torch.int32),
            topk_weights=torch.full((len(inputs), 1), 0.5),
        )
        assert torch.equal(actual, inputs * 1.5)


@pytest.mark.parametrize("solution", [None, "public_reference"])
def test_expert_plan_rejects_missing_invariance(registry, solution):
    _expert_provider(registry, False)
    with pytest.raises(NoKernelFoundError):
        _plan(solution)
