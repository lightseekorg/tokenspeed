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

from __future__ import annotations

import pytest
import tokenspeed_kernel
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401
from tokenspeed_kernel.platform import current_platform


class _Weights(torch.nn.Module):
    def __init__(self, num_experts: int, num_local_experts: int) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.ep_rank = 0
        self.ep_size = num_experts // num_local_experts
        self.hidden_size = 768
        self.intermediate_size = 512
        self.w13_weight = torch.nn.Parameter(
            torch.randn(
                num_local_experts,
                2 * self.intermediate_size,
                self.hidden_size,
                device="npu:0",
                dtype=torch.bfloat16,
            )
            * 0.005,
            requires_grad=False,
        )
        self.w2_weight = torch.nn.Parameter(
            torch.randn(
                num_local_experts,
                self.hidden_size,
                self.intermediate_size,
                device="npu:0",
                dtype=torch.bfloat16,
            )
            * 0.005,
            requires_grad=False,
        )


@pytest.fixture(autouse=True, scope="module")
def _requires_npu():
    if not current_platform().is_npu:
        pytest.skip("requires an Ascend NPU")
    torch.npu.set_device(0)


def _plan(weights: _Weights, *, num_zero_experts: int = 0) -> dict:
    plan = tokenspeed_kernel.moe_plan(
        "unquant",
        input_dtype=torch.bfloat16,
        activation="silu",
        routing_mode="precomputed_topk",
        ep_size=weights.ep_size,
        num_zero_experts=num_zero_experts,
        ispp=weights.intermediate_size,
    )
    assert plan["apply_kernel_name"] == "ascend_bf16_precomputed_moe_apply"
    tokenspeed_kernel.moe_process_weights(plan, weights)
    return plan


def _apply(
    plan: dict,
    weights: _Weights,
    hidden: torch.Tensor,
    route_weights: torch.Tensor,
    route_ids: torch.Tensor,
) -> torch.Tensor:
    return tokenspeed_kernel.moe_apply(
        plan,
        hidden,
        weights,
        torch.empty(
            hidden.shape[0],
            weights.num_experts,
            device=hidden.device,
            dtype=torch.float32,
        ),
        topk_weights=route_weights,
        topk_ids=route_ids,
    )


def _oracle(
    weights: _Weights,
    hidden: torch.Tensor,
    route_weights: torch.Tensor,
    route_ids: torch.Tensor,
) -> torch.Tensor:
    w13 = weights.w13_weight.transpose(-1, -2)
    w2 = weights.w2_weight.transpose(-1, -2)
    output = torch.zeros_like(hidden, dtype=torch.float32)
    for token in range(hidden.shape[0]):
        for route in range(route_ids.shape[1]):
            expert = int(route_ids[token, route])
            if 0 <= expert < weights.num_local_experts:
                gate_up = F.linear(hidden[token : token + 1], w13[expert])
                gate, up = gate_up.chunk(2, dim=-1)
                expert_output = F.linear(F.silu(gate) * up, w2[expert])
                output[token] += (
                    expert_output[0].float() * route_weights[token, route].float()
                )
    return output.to(hidden.dtype)


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    relative_l2 = torch.linalg.vector_norm(
        actual.float() - expected.float()
    ) / torch.linalg.vector_norm(expected.float()).clamp_min(1e-12)
    assert relative_l2 <= 5e-3


@pytest.mark.parametrize("tokens", [1, 2, 32])
def test_ascend_softmax_bias_topk_matches_torch(tokens: int):
    torch.manual_seed(6100 + tokens)
    logits = torch.randn(tokens, 416, device="npu:0", dtype=torch.float32)
    bias = torch.randn(416, device="npu:0", dtype=torch.float32) * 0.01
    weights, ids = tokenspeed_kernel.moe_softmax_bias_topk(
        logits,
        bias,
        12,
        routed_scaling_factor=6.0,
    )
    probabilities = logits.softmax(-1)
    expected_ids = torch.topk(probabilities + bias, 12, dim=-1).indices
    expected_weights = probabilities.gather(1, expected_ids) * 6.0
    assert torch.equal(ids.long(), expected_ids.long())
    torch.testing.assert_close(weights, expected_weights, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("routing", ["all", "partial", "none"])
def test_ascend_local_expert_matches_fp32_oracle(routing: str):
    torch.manual_seed(6200)
    weights = _Weights(384, 48)
    plan = _plan(weights)
    assert tuple(weights.w13_weight.shape) == (48, 768, 1024)
    assert tuple(weights.w2_weight.shape) == (48, 512, 768)

    tokens, topk = 4, 12
    hidden = torch.randn(tokens, 768, device="npu:0", dtype=torch.bfloat16)
    route_weights = torch.rand(tokens, topk, device="npu:0", dtype=torch.float32)
    local = torch.arange(tokens * topk, device="npu:0", dtype=torch.int32).view(
        tokens, topk
    )
    if routing == "partial":
        route_ids = torch.where(
            torch.arange(topk, device="npu:0").remainder(2).bool().unsqueeze(0),
            local + 48,
            local,
        )
        route_ids[0, -1] = 385
        route_ids[1, -1] = -1
    elif routing == "none":
        route_ids = local + 48
    else:
        route_ids = local

    actual = _apply(plan, weights, hidden, route_weights, route_ids)
    expected = _oracle(weights, hidden, route_weights, route_ids)
    _assert_close(actual, expected)
    if routing == "none":
        assert torch.count_nonzero(actual) == 0


def test_ascend_flattened_192_local_experts():
    torch.manual_seed(6300)
    weights = _Weights(1536, 192)
    plan = _plan(weights)
    tokens, topk = 8, 12
    hidden = torch.randn(tokens, 768, device="npu:0", dtype=torch.bfloat16)
    route_weights = torch.rand(tokens, topk, device="npu:0", dtype=torch.float32)
    route_ids = torch.arange(tokens * topk, device="npu:0", dtype=torch.int32).view(
        tokens, topk
    )
    route_ids[:, 1::2] += 192
    actual = _apply(plan, weights, hidden, route_weights, route_ids)
    expected = _oracle(weights, hidden, route_weights, route_ids)
    _assert_close(actual, expected)


def test_ascend_ep1_zeros_model_side_special_routes():
    torch.manual_seed(6350)
    weights = _Weights(32, 32)
    plan = _plan(weights, num_zero_experts=8)
    hidden = torch.randn(4, 768, device="npu:0", dtype=torch.bfloat16)
    route_weights = torch.rand(4, 4, device="npu:0", dtype=torch.float32)
    route_ids = torch.tensor(
        [[0, 1, 32, 33], [2, 34, 3, 35], [36, 4, 5, 37], [6, 7, 38, 39]],
        device="npu:0",
        dtype=torch.int32,
    )

    actual = _apply(plan, weights, hidden, route_weights, route_ids)
    expected = _oracle(weights, hidden, route_weights, route_ids)
    _assert_close(actual, expected)


def test_ascend_ep1_zero_ids_are_outside_active_expert_range():
    hidden = torch.randn(2, 128, device="npu:0", dtype=torch.bfloat16)
    route_ids = torch.tensor(
        [[0, 1, 32, 33], [2, 34, 3, 35]],
        device="npu:0",
        dtype=torch.int32,
    )

    _, _, expert_counts, _ = torch_npu.npu_moe_init_routing_v2(
        hidden,
        route_ids,
        scale=None,
        active_num=route_ids.numel(),
        expert_num=40,
        quant_mode=-1,
        active_expert_range=[0, 32],
        expert_tokens_num_type=1,
        expert_tokens_num_flag=True,
        row_idx_type=0,
        drop_pad_mode=0,
    )

    assert tuple(expert_counts.shape) == (32,)
    assert expert_counts.dtype == torch.int64
    assert int(expert_counts.sum().cpu()) == 4


@pytest.mark.parametrize("batch_size", [1, 2])
def test_ascend_local_expert_npu_graph_replay(batch_size: int):
    torch.manual_seed(6400 + batch_size)
    weights = _Weights(384, 48)
    plan = _plan(weights)
    hidden = torch.randn(batch_size, 768, device="npu:0", dtype=torch.bfloat16)
    route_weights = torch.rand(batch_size, 12, device="npu:0", dtype=torch.float32)
    route_ids = torch.arange(batch_size * 12, device="npu:0", dtype=torch.int32).view(
        batch_size, 12
    )
    route_ids[:, 1::2] += 48

    for _ in range(4):
        output = _apply(plan, weights, hidden, route_weights, route_ids)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(
        graph,
        stream=torch.npu.Stream(),
        auto_dispatch_capture=True,
    ):
        output = _apply(plan, weights, hidden, route_weights, route_ids)
    torch.npu.synchronize()
    graph.replay()
    torch.npu.synchronize()
    first = output.clone()

    hidden.copy_(torch.randn_like(hidden))
    route_weights.copy_(torch.rand_like(route_weights))
    route_ids.copy_((route_ids + 1).remainder(96))
    expected = _oracle(weights, hidden, route_weights, route_ids)
    graph.replay()
    torch.npu.synchronize()
    assert not torch.equal(first, output)
    _assert_close(output, expected)
