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

from types import SimpleNamespace

import pytest
import tokenspeed_kernel
import torch
import torch_npu

from tokenspeed.runtime.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)


@pytest.fixture(autouse=True, scope="module")
def _requires_npu():
    if not torch.npu.is_available():
        pytest.skip("requires an Ascend NPU")
    torch.npu.set_device(0)


def test_compressed_tensors_dynamic_token_int8_selects_int8_moe() -> None:
    quant_config = CompressedTensorsConfig.from_config(
        {
            "format": "int-quantized",
            "enable_smooth_quant": True,
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": 8,
                        "type": "int",
                        "symmetric": True,
                        "strategy": "channel",
                        "dynamic": False,
                    },
                    "input_activations": {
                        "num_bits": 8,
                        "type": "int",
                        "symmetric": True,
                        "strategy": "token",
                        "dynamic": True,
                    },
                }
            },
        }
    )
    assert quant_config.moe_weight_dtype("model.layers.0.moe") == "int8"


@pytest.mark.parametrize("smooth_quant", ("none", "w13", "w2", "both"))
def test_w8a8_moe_uses_nz_weights_and_runs(smooth_quant: str) -> None:
    tokens, hidden, intermediate, experts, top_k = 16, 64, 128, 8, 4
    device = torch.device("npu:0")
    weights = SimpleNamespace(
        num_local_experts=experts,
        num_experts=experts,
        hidden_size=hidden,
        intermediate_size=intermediate,
        ep_rank=0,
        ep_size=1,
    )
    weights.w13_weight = torch.nn.Parameter(
        torch.randint(
            -8,
            8,
            (experts, 2 * intermediate, hidden),
            dtype=torch.int8,
            device=device,
        ),
        requires_grad=False,
    )
    weights.w2_weight = torch.nn.Parameter(
        torch.randint(
            -8,
            8,
            (experts, hidden, intermediate),
            dtype=torch.int8,
            device=device,
        ),
        requires_grad=False,
    )
    weights.w13_weight_scale = torch.nn.Parameter(
        torch.full(
            (experts, 2 * intermediate),
            1.0 / 128.0,
            dtype=torch.float32,
            device=device,
        ),
        requires_grad=False,
    )
    weights.w2_weight_scale = torch.nn.Parameter(
        torch.full(
            (experts, hidden),
            1.0 / 128.0,
            dtype=torch.bfloat16,
            device=device,
        ),
        requires_grad=False,
    )
    if smooth_quant in {"w13", "both"}:
        weights.w13_smooth_scale = torch.nn.Parameter(
            torch.ones(experts, hidden, dtype=torch.float32, device=device),
            requires_grad=False,
        )
    if smooth_quant in {"w2", "both"}:
        weights.w2_smooth_scale = torch.nn.Parameter(
            torch.ones(experts, intermediate, dtype=torch.float32, device=device),
            requires_grad=False,
        )

    plan = tokenspeed_kernel.moe_plan(
        "int8",
        input_dtype=torch.bfloat16,
        activation="silu",
        routing_mode="precomputed_topk",
        ep_size=1,
        ispp=intermediate,
        internal_activation_dtype="int8",
        num_zero_experts=2,
    )
    assert plan["apply_kernel_name"] == "ascend_int8_precomputed_moe_apply"
    tokenspeed_kernel.moe_process_weights(plan, weights)
    assert torch_npu.get_npu_format(weights.w13_weight) == 29
    assert torch_npu.get_npu_format(weights.w2_weight) == 29

    x = torch.randn(tokens, hidden, dtype=torch.bfloat16, device=device)
    ids = torch.arange(tokens * top_k, dtype=torch.int32, device=device).view(
        tokens, top_k
    )
    ids = ids.remainder(experts + 2)
    route_weights = torch.rand(tokens, top_k, dtype=torch.float32, device=device)
    output = tokenspeed_kernel.moe_apply(
        plan,
        x,
        weights,
        route_weights,
        topk_weights=route_weights,
        topk_ids=ids,
    )
    torch.npu.synchronize()
    assert output.shape == x.shape
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()
