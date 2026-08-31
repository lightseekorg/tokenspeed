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

"""Numerical check: Marlin W4A16 MXFP4 MoE apply vs a bf16 dequant reference.

Marlin dequantizes the packed E2M1 weights inside the GEMM, so the kernel
output must track a plain bf16 reference that dequantizes the same weights and
runs the two linear layers with the SiTU epilogue. Runs on SM90+.
"""

from __future__ import annotations

import pytest
import torch
from kimi3_reference import mxfp4_moe_reference
from tokenspeed_kernel.ops.moe.marlin.mxfp4 import (
    marlin_mxfp4_moe_weights,
    marlin_mxfp4_precomputed_moe_apply,
)
from tokenspeed_kernel.platform import current_platform
from utils import make_mxfp4_moe_weights


def _requires_sm90():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not current_platform().is_nvidia:
        pytest.skip("NVIDIA required")
    if current_platform().arch_version < type(current_platform().arch_version)(9, 0):
        pytest.skip("SM90+ required")


class _Weights(torch.nn.Module):
    """Minimal module carrying MXFP4 expert params for the apply kernel."""

    def __init__(self, raw, num_local_experts, ep_size, ep_rank, beta, linear_beta):
        super().__init__()
        self.w13_weight = torch.nn.Parameter(raw["w13_weight"], requires_grad=False)
        self.w13_weight_scale = torch.nn.Parameter(
            raw["w13_scale"], requires_grad=False
        )
        self.w2_weight = torch.nn.Parameter(raw["w2_weight"], requires_grad=False)
        self.w2_weight_scale = torch.nn.Parameter(raw["w2_scale"], requires_grad=False)
        self.num_local_experts = num_local_experts
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.activation = "situ"
        self.activation_situ_beta = beta
        self.activation_situ_linear_beta = linear_beta


@pytest.mark.parametrize("num_tokens", [1, 8, 64])
def test_marlin_mxfp4_situ_matches_reference(num_tokens: int) -> None:
    _requires_sm90()
    generator = torch.Generator(device="cuda").manual_seed(11)
    num_experts, top_k = 8, 2
    hidden_size, intermediate_size = 256, 128
    beta, linear_beta = 4.0, 25.0

    x = (
        torch.randn(num_tokens, hidden_size, generator=generator, device="cuda") * 0.2
    ).to(torch.bfloat16)
    raw = make_mxfp4_moe_weights(
        num_experts, hidden_size, intermediate_size, generator, device="cuda"
    )
    # topk ids/weights: round-robin ids, normalized random weights.
    topk_ids = (
        torch.arange(num_tokens * top_k, device="cuda").reshape(num_tokens, top_k)
        % num_experts
    ).to(torch.int32)
    topk_weights = torch.rand(
        num_tokens, top_k, generator=generator, device="cuda", dtype=torch.float32
    )
    topk_weights = topk_weights / topk_weights.sum(-1, keepdim=True)

    expected = mxfp4_moe_reference(
        x,
        raw["w13_weight"],
        raw["w13_scale"],
        raw["w2_weight"],
        raw["w2_scale"],
        topk_ids,
        topk_weights,
        activation_dtype=torch.bfloat16,
        situ_beta=beta,
        situ_linear_beta=linear_beta,
    )

    # Repack in place, then apply (no EP: one rank owns all experts).
    w = _Weights(
        {k: v.clone() for k, v in raw.items()},
        num_local_experts=num_experts,
        ep_size=1,
        ep_rank=0,
        beta=beta,
        linear_beta=linear_beta,
    )
    plan = {"activation": "situ"}
    marlin_mxfp4_moe_weights(plan, w)
    actual = marlin_mxfp4_precomputed_moe_apply(
        plan, x, w, None, topk_weights=topk_weights, topk_ids=topk_ids
    )

    torch.testing.assert_close(actual.float(), expected.float(), atol=5e-2, rtol=5e-2)


def test_marlin_mxfp4_ep_masks_nonlocal_experts() -> None:
    """EP: each rank owns half the experts; summed ranks == full reference."""
    _requires_sm90()
    generator = torch.Generator(device="cuda").manual_seed(23)
    num_experts, top_k = 8, 2
    hidden_size, intermediate_size = 256, 128
    num_tokens = 16
    beta, linear_beta = 4.0, 25.0
    ep_size = 2
    num_local = num_experts // ep_size

    x = (
        torch.randn(num_tokens, hidden_size, generator=generator, device="cuda") * 0.2
    ).to(torch.bfloat16)
    raw = make_mxfp4_moe_weights(
        num_experts, hidden_size, intermediate_size, generator, device="cuda"
    )
    topk_ids = (
        torch.arange(num_tokens * top_k, device="cuda").reshape(num_tokens, top_k)
        % num_experts
    ).to(torch.int32)
    topk_weights = torch.rand(
        num_tokens, top_k, generator=generator, device="cuda", dtype=torch.float32
    )
    topk_weights = topk_weights / topk_weights.sum(-1, keepdim=True)

    expected = mxfp4_moe_reference(
        x,
        raw["w13_weight"],
        raw["w13_scale"],
        raw["w2_weight"],
        raw["w2_scale"],
        topk_ids,
        topk_weights,
        activation_dtype=torch.bfloat16,
        situ_beta=beta,
        situ_linear_beta=linear_beta,
    )

    acc = torch.zeros_like(x, dtype=torch.float32)
    plan = {"activation": "situ"}
    for ep_rank in range(ep_size):
        lo = ep_rank * num_local
        shard = {
            "w13_weight": raw["w13_weight"][lo : lo + num_local].clone(),
            "w13_scale": raw["w13_scale"][lo : lo + num_local].clone(),
            "w2_weight": raw["w2_weight"][lo : lo + num_local].clone(),
            "w2_scale": raw["w2_scale"][lo : lo + num_local].clone(),
        }
        w = _Weights(
            shard,
            num_local_experts=num_local,
            ep_size=ep_size,
            ep_rank=ep_rank,
            beta=beta,
            linear_beta=linear_beta,
        )
        marlin_mxfp4_moe_weights(plan, w)
        acc += marlin_mxfp4_precomputed_moe_apply(
            plan, x, w, None, topk_weights=topk_weights, topk_ids=topk_ids
        ).float()

    torch.testing.assert_close(acc, expected.float(), atol=5e-2, rtol=5e-2)
