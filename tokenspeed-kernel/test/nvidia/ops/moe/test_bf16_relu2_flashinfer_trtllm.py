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
"""flashinfer TRTLLM-Gen BF16 squared-ReLU MoE (non-gated) vs an fp32 reference.

Nemotron-H's MTP layer keeps its experts in BF16. GEMM1 holds only the up
projection (``w13`` is ``[E, I, H]``) and the activation is ``relu(x)**2``.
"""

from __future__ import annotations

import pytest
import torch

HIDDEN = 256
ISPP = 256


def _runtime_reason() -> str | None:
    if not torch.cuda.is_available():
        return "requires CUDA"
    import tokenspeed_kernel.ops.moe.flashinfer.trtllm_unquant  # noqa: F401
    from tokenspeed_kernel.platform import current_platform
    from tokenspeed_kernel.registry import KernelRegistry

    spec = KernelRegistry.get().get_by_name(
        "flashinfer_trtllm_unquant_relu2_routed_moe_apply"
    )
    if spec is None or not spec.capability.satisfied_by(current_platform()):
        return "outside the kernel's registered capability range"
    return None


_reason = _runtime_reason()
requires_relu2 = pytest.mark.skipif(_reason is not None, reason=str(_reason))


class _MoEWeights(torch.nn.Module):
    """Minimal module carrying what the preprocessor and apply consume."""

    def __init__(self, w13: torch.Tensor, w2: torch.Tensor, top_k: int):
        super().__init__()
        self.w13_weight = torch.nn.Parameter(w13.clone(), requires_grad=False)
        self.w2_weight = torch.nn.Parameter(w2.clone(), requires_grad=False)
        self.num_experts = w13.shape[0]
        self.num_local_experts = w13.shape[0]
        self.top_k = top_k
        self.intermediate_size = w13.shape[1]
        self.tp_size = 1
        self.ep_rank = 0
        self.routing_config: dict = {}


def _make_weights(generator: torch.Generator, num_experts: int):
    w13 = torch.randn(num_experts, ISPP, HIDDEN, generator=generator) * 0.1
    w2 = torch.randn(num_experts, HIDDEN, ISPP, generator=generator) * 0.1
    return w13.bfloat16().cuda(), w2.bfloat16().cuda()


def _reference_moe(x, w13, w2, topk_ids, topk_weights):
    x = x.float()
    out = torch.zeros_like(x)
    for e in topk_ids.unique().tolist():
        act = torch.relu(x @ w13[e].float().t()).square().bfloat16().float()
        weight = torch.where(topk_ids == e, topk_weights.float(), 0.0).sum(dim=-1)
        out += weight[:, None] * (act @ w2[e].float().t())
    return out


def _rel_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return ((actual.float() - expected).norm() / expected.norm()).item()


@requires_relu2
@pytest.mark.parametrize("num_tokens", [1, 37])
def test_bf16_relu2_routed_moe_matches_reference(num_tokens):
    from tokenspeed_kernel.ops.moe.flashinfer.trtllm_unquant import (
        flashinfer_trtllm_unquant_relu2_moe_weights,
        flashinfer_trtllm_unquant_relu2_routed_moe_apply,
    )

    num_experts, top_k = 16, 6
    generator = torch.Generator().manual_seed(20260930)
    w13, w2 = _make_weights(generator, num_experts)
    x = (torch.randn(num_tokens, HIDDEN, generator=generator) * 0.5).bfloat16().cuda()
    topk_ids = (
        torch.stack(
            [
                torch.randperm(num_experts, generator=generator)[:top_k]
                for _ in range(num_tokens)
            ]
        )
        .to(torch.int32)
        .cuda()
    )
    topk_weights = torch.rand(num_tokens, top_k, generator=generator).softmax(-1)
    topk_weights = topk_weights.bfloat16().cuda()

    w = _MoEWeights(w13, w2, top_k).cuda()
    flashinfer_trtllm_unquant_relu2_moe_weights({}, w)
    actual = flashinfer_trtllm_unquant_relu2_routed_moe_apply(
        {}, x, w, router_logits=None, topk_weights=topk_weights, topk_ids=topk_ids
    )
    torch.cuda.synchronize()
    err = _rel_l2(actual, _reference_moe(x, w13, w2, topk_ids, topk_weights))
    assert err < 0.02, f"{err=:.4f}"


@requires_relu2
@pytest.mark.parametrize("num_tokens", [3, 64])
def test_bf16_relu2_kernel_routing_matches_sigmoid_bias_top22(num_tokens):
    """Nemotron-H routing: 512 experts, sigmoid + correction bias, one group, top-22, scale 5."""
    from flashinfer.fused_moe import RoutingMethodType
    from tokenspeed_kernel.ops.moe.flashinfer.trtllm_unquant import (
        flashinfer_trtllm_unquant_relu2_moe_apply,
        flashinfer_trtllm_unquant_relu2_moe_weights,
    )

    num_experts, top_k, scale = 512, 22, 5.0
    generator = torch.Generator().manual_seed(88)
    w13, w2 = _make_weights(generator, num_experts)
    x = (torch.randn(num_tokens, HIDDEN, generator=generator) * 0.5).bfloat16().cuda()
    logits = torch.randn(num_tokens, num_experts, generator=generator).cuda()
    bias = (torch.randn(num_experts, generator=generator) * 0.1).cuda()

    w = _MoEWeights(w13, w2, top_k).cuda()
    w.routing_config = {
        "routing_method_type": RoutingMethodType.DeepSeekV3,
        "correction_bias": bias,
        "n_group": 1,
        "topk_group": 1,
        "routed_scaling_factor": scale,
    }
    flashinfer_trtllm_unquant_relu2_moe_weights({}, w)
    actual = flashinfer_trtllm_unquant_relu2_moe_apply({}, x, w, logits)
    torch.cuda.synchronize()

    scores = logits.sigmoid()
    topk_ids = (scores + bias).topk(top_k, dim=-1).indices
    picked = scores.gather(1, topk_ids)
    topk_weights = picked / picked.sum(dim=-1, keepdim=True) * scale
    err = _rel_l2(actual, _reference_moe(x, w13, w2, topk_ids, topk_weights))
    assert err < 0.02, f"{err=:.4f}"


@requires_relu2
@pytest.mark.parametrize("routing_mode", [None, "precomputed_topk"])
def test_moe_plan_selects_the_bf16_relu2_kernels(routing_mode):
    import tokenspeed_kernel

    plan = tokenspeed_kernel.moe_plan(
        "unquant",
        input_dtype=torch.bfloat16,
        activation="relu2",
        routing_mode=routing_mode,
        ep_size=1,
        ispp=2688,
        internal_activation_dtype="input",
        hidden=None,
        swiglu_form=None,
        activation_clamped=False,
        expert_id_repeats=False,
        fast_math=True,
    )
    expected = {
        None: "flashinfer_trtllm_unquant_relu2_moe_apply",
        "precomputed_topk": "flashinfer_trtllm_unquant_relu2_routed_moe_apply",
    }[routing_mode]
    assert plan["apply_kernel_name"] == expected
    assert (
        plan["weight_preprocessor"].__name__
        == "flashinfer_trtllm_unquant_relu2_moe_weights"
    )
