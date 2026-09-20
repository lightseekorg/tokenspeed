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


"""The MoE reference kernels: registration bands and agreement with oracles."""

from __future__ import annotations

import pytest
import tokenspeed_kernel
import torch
import torch.nn.functional as F
from tokenspeed_kernel.registry import KernelRegistry, Priority, load_builtin_kernels
from tokenspeed_kernel.selection import is_ground_truth


@pytest.mark.parametrize(
    "name,band",
    [
        ("torch_pack_topk_router_logits", Priority.PORTABLE),
        ("torch_softmax_topk", Priority.PORTABLE),
        ("torch_sigmoid_bias_topk", Priority.PORTABLE),
        ("torch_precomputed_moe_apply", Priority.REFERENCE),
    ],
)
def test_moe_reference_bands(name: str, band: Priority) -> None:
    load_builtin_kernels()
    spec = KernelRegistry.get().get_by_name(name)
    assert spec is not None
    assert spec.solution == "torch"
    assert spec.priority == band
    assert is_ground_truth(spec) == (band == Priority.REFERENCE)


def test_softmax_topk_reference_matches_torch_on_cpu() -> None:
    torch.manual_seed(0)
    logits = torch.randn(5, 16, dtype=torch.float32)
    weights, ids = tokenspeed_kernel.moe_softmax_topk(
        logits,
        4,
        topk_indices_dtype=torch.int64,
        renormalize=False,
        routed_scaling_factor=1.5,
        solution="reference",
    )
    expected_ids = torch.topk(logits, 4, dim=-1).indices
    assert torch.equal(ids, expected_ids)
    torch.testing.assert_close(
        weights, torch.softmax(logits, dim=-1).gather(-1, expected_ids) * 1.5
    )


@pytest.mark.parametrize("activation", ["silu", "situ"])
def test_precomputed_moe_apply_matches_per_token_oracle(
    device: str, activation: str
) -> None:
    torch.manual_seed(1)
    tokens, hidden, intermediate, experts, top_k = 6, 128, 32, 5, 2
    x = torch.randn(tokens, hidden, device=device, dtype=torch.bfloat16)
    w13 = (
        torch.randn(
            experts, 2 * intermediate, hidden, device=device, dtype=torch.bfloat16
        )
        * 0.05
    )
    w2 = (
        torch.randn(experts, hidden, intermediate, device=device, dtype=torch.bfloat16)
        * 0.05
    )
    # One route points past the expert table and must contribute nothing.
    topk_ids = torch.randint(0, experts, (tokens, top_k), device=device)
    topk_ids[0, 1] = experts
    topk_weights = torch.rand(tokens, top_k, device=device, dtype=torch.float32)

    w = torch.nn.Module()
    w.w13_weight = w13
    w.w2_weight = w2
    w.top_k = top_k
    w.activation_situ_beta = 4.0
    w.activation_situ_linear_beta = 25.0
    plan = tokenspeed_kernel.moe_plan(
        "unquant",
        input_dtype=torch.bfloat16,
        activation=activation,
        routing_mode="precomputed_topk",
        ispp=intermediate,
        solution="reference",
    )
    assert plan["apply_kernel_name"] == "torch_precomputed_moe_apply"
    out = tokenspeed_kernel.moe_apply(
        plan, x, w, None, topk_weights=topk_weights, topk_ids=topk_ids
    )
    assert out.dtype == x.dtype

    expected = torch.zeros(tokens, hidden, device=device, dtype=torch.float32)
    for token in range(tokens):
        for slot in range(top_k):
            expert = int(topk_ids[token, slot])
            if expert >= experts:
                continue
            gate_up = x[token].float() @ w13[expert].float().T
            gate, up = gate_up[:intermediate], gate_up[intermediate:]
            if activation == "situ":
                gate = 4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)
                up = 25.0 * torch.tanh(up / 25.0)
                inter = gate * up
            else:
                inter = F.silu(gate) * up
            expected[token] += float(topk_weights[token, slot]) * (
                inter @ w2[expert].float().T
            )
    torch.testing.assert_close(out.float(), expected, rtol=2e-2, atol=2e-2)
