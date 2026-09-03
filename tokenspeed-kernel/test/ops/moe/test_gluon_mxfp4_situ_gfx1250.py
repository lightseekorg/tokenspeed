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
import torch
from kimi3_reference import mxfp4_moe_reference
from utils import is_cdna5, make_mxfp4_moe_weights, make_round_robin_topk

if not is_cdna5():
    pytest.skip(
        "AMD CDNA5 is required for gfx1250 Gluon MXFP4 SiTU tests",
        allow_module_level=True,
    )

import tokenspeed_kernel  # noqa: E402

_KERNEL_NAME = "gluon_mxfp4_a8w4_situ_gfx1250_precomputed_moe_apply"


def _make_mxfp4_module(
    *,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    top_k: int,
    generator: torch.Generator,
) -> tuple[torch.nn.Module, dict[str, torch.Tensor]]:
    raw = make_mxfp4_moe_weights(
        num_experts,
        hidden_size,
        intermediate_size,
        generator,
    )
    module = torch.nn.Module()
    module.w13_weight = torch.nn.Parameter(
        raw["w13_weight"].clone(), requires_grad=False
    )
    module.w13_weight_scale = torch.nn.Parameter(
        raw["w13_scale"].clone(), requires_grad=False
    )
    module.w2_weight = torch.nn.Parameter(raw["w2_weight"].clone(), requires_grad=False)
    module.w2_weight_scale = torch.nn.Parameter(
        raw["w2_scale"].clone(), requires_grad=False
    )
    module.top_k = top_k
    module.num_experts = num_experts
    module.ep_size = 1
    module.activation_situ_beta = 4.0
    module.activation_situ_linear_beta = 25.0
    return module, raw


def _make_case(
    num_tokens: int,
) -> tuple[
    torch.nn.Module,
    dict[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    generator = torch.Generator(device="cuda").manual_seed(20260829 + num_tokens)
    num_experts, hidden_size, intermediate_size, top_k = 16, 3584, 384, 16
    module, raw = _make_mxfp4_module(
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        top_k=top_k,
        generator=generator,
    )
    hidden_states = 0.1 * torch.randn(
        num_tokens,
        hidden_size,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    topk_weights, topk_ids = make_round_robin_topk(
        num_tokens,
        num_experts,
        top_k,
    )
    router_logits = torch.zeros(
        (num_tokens, num_experts),
        dtype=torch.float32,
        device="cuda",
    )
    return module, raw, hidden_states, topk_weights, topk_ids, router_logits


def _make_plan() -> dict:
    plan = tokenspeed_kernel.moe_plan(
        "mxfp4",
        input_dtype=torch.bfloat16,
        activation="situ",
        routing_mode="precomputed_topk",
        ep_size=1,
        ispp=384,
        internal_activation_dtype="input",
        solution="gluon",
    )
    assert plan["apply_kernel_name"] == _KERNEL_NAME
    return plan


@pytest.mark.parametrize("num_tokens", [1, 4, 32])
def test_kimi_k3_tp_situ_matches_a8w4_reference_gfx1250(
    num_tokens: int,
) -> None:
    module, raw, hidden_states, topk_weights, topk_ids, router_logits = _make_case(
        num_tokens
    )
    plan = _make_plan()
    tokenspeed_kernel.moe_process_weights(plan, module)

    actual = tokenspeed_kernel.moe_apply(
        plan,
        hidden_states,
        module,
        router_logits,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    )
    expected = mxfp4_moe_reference(
        hidden_states,
        raw["w13_weight"],
        raw["w13_scale"],
        raw["w2_weight"],
        raw["w2_scale"],
        topk_ids,
        topk_weights,
        activation_dtype=torch.float8_e4m3fn,
        situ_beta=4.0,
        situ_linear_beta=25.0,
    )

    torch.cuda.synchronize()
    assert torch.count_nonzero(expected).item() > 0
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=5e-2)


def test_kimi_k3_tp_situ_is_cuda_graph_capturable_gfx1250() -> None:
    module, _, hidden_states, topk_weights, topk_ids, router_logits = _make_case(1)
    plan = _make_plan()
    tokenspeed_kernel.moe_process_weights(plan, module)

    expected = tokenspeed_kernel.moe_apply(
        plan,
        hidden_states,
        module,
        router_logits,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    ).clone()
    output = torch.empty_like(expected)
    module._situ_output_buffer = output

    def apply() -> torch.Tensor:
        result = tokenspeed_kernel.moe_apply(
            plan,
            hidden_states,
            module,
            router_logits,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
        )
        assert result.data_ptr() == output.data_ptr()
        return result

    for _ in range(3):
        apply()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = apply()
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(captured, expected, atol=0.0, rtol=0.0)
