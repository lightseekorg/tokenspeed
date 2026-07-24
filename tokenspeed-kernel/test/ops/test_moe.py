# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

import importlib

import tokenspeed_kernel
import torch
import torch.nn.functional as F
from tokenspeed_kernel.ops.moe.triton import unquant as _triton_unquant


def _torch_unquant_moe_reference(
    x: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    out = torch.zeros_like(x, dtype=torch.float32)
    for token_idx in range(x.shape[0]):
        for topk_idx in range(topk_ids.shape[1]):
            expert_id = int(topk_ids[token_idx, topk_idx])
            gate_up = F.linear(x[token_idx].float(), w13_weight[expert_id].float())
            gate, up = gate_up.chunk(2, dim=-1)
            down = F.linear(F.silu(gate) * up, w2_weight[expert_id].float())
            out[token_idx] += topk_weights[token_idx, topk_idx].float() * down
    return out.to(x.dtype)


def test_triton_unquant_moe_matches_torch_reference(device: str, require) -> None:
    # MTP draft experts are bf16; this verifies the registry path and weight layout.
    importlib.reload(_triton_unquant)
    require("moe", "apply", "triton", torch.bfloat16, "x")

    torch.manual_seed(1234)
    num_tokens = 5
    hidden_size = 16
    intermediate_size = 32
    num_experts = 4
    top_k = 2
    dtype = torch.bfloat16

    x = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype) * 0.1
    w13_weight = (
        torch.randn(
            num_experts,
            2 * intermediate_size,
            hidden_size,
            device=device,
            dtype=dtype,
        )
        * 0.1
    )
    w2_weight = (
        torch.randn(
            num_experts,
            hidden_size,
            intermediate_size,
            device=device,
            dtype=dtype,
        )
        * 0.1
    )
    topk_ids = torch.tensor(
        [[0, 1], [2, 3], [1, 2], [3, 0], [0, 2]],
        device=device,
        dtype=torch.int64,
    )
    topk_weights = torch.tensor(
        [[0.7, 0.3], [0.6, 0.4], [0.8, 0.2], [0.5, 0.5], [0.9, 0.1]],
        device=device,
        dtype=torch.float32,
    )

    plan = tokenspeed_kernel.moe_plan(
        "unquant",
        input_dtype=dtype,
        activation="silu",
        ep_size=1,
        ispp=intermediate_size,
        solution="triton",
    )
    assert plan["apply_kernel_name"] == "triton_unquant_precomputed_moe_apply"

    w = torch.nn.Module()
    w.top_k = top_k
    w.num_experts = num_experts
    w.num_local_experts = num_experts
    w.ep_size = 1
    w.ep_rank = 0
    w.w13_weight = torch.nn.Parameter(w13_weight.clone(), requires_grad=False)
    w.w2_weight = torch.nn.Parameter(w2_weight.clone(), requires_grad=False)
    tokenspeed_kernel.moe_process_weights(plan, w)

    out = tokenspeed_kernel.moe_apply(
        plan,
        x,
        w,
        torch.empty(num_tokens, num_experts, device=device, dtype=torch.float32),
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    )
    ref = _torch_unquant_moe_reference(
        x,
        w13_weight,
        w2_weight,
        topk_weights,
        topk_ids,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(out.float(), ref.float(), rtol=2e-2, atol=2e-2)
