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


def _is_gfx950() -> bool:
    if not torch.cuda.is_available():
        return False
    arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
    return "gfx950" in arch


if not _is_gfx950():
    pytest.skip(
        "Gluon bf16 MoE apply is gfx950 (CDNA4) only",
        allow_module_level=True,
    )

import tokenspeed_kernel  # noqa: E402

# DeepSeek-V3 TP=8 MoE reference shape.
E = 256
D = 7168
I_R = 256
TOPK = 8


def _torch_moe_ref(x, w13, w2, topk_ids, topk_weights) -> torch.Tensor:
    xf, w13f, w2f = x.float(), w13.float(), w2.float()
    ids, wts = topk_ids.cpu(), topk_weights.float().cpu()
    out = torch.zeros(x.shape[0], D, dtype=torch.float32, device=x.device)
    for t in range(x.shape[0]):
        for s in range(TOPK):
            e = int(ids[t, s])
            g = xf[t] @ w13f[e].T
            inter = torch.nn.functional.silu(g[:I_R]) * g[I_R:]
            out[t] += float(wts[t, s]) * (inter @ w2f[e].T)
    return out


@pytest.mark.parametrize("num_tokens", [1, 8, 16])
def test_gluon_bf16_moe_apply_matches_reference(num_tokens):
    """moe_plan selects the gluon bf16 apply and it matches the fp32 oracle."""
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(0)
    x = torch.randn(num_tokens, D, dtype=torch.bfloat16, device=dev, generator=g)
    w13 = (
        torch.randn(E, 2 * I_R, D, dtype=torch.bfloat16, device=dev, generator=g) * 0.05
    )
    w2 = torch.randn(E, D, I_R, dtype=torch.bfloat16, device=dev, generator=g) * 0.05
    logits = torch.randn(num_tokens, E, dtype=torch.float32, device=dev, generator=g)
    probs = torch.softmax(logits, dim=-1)
    topk_weights, topk_ids = torch.topk(probs, TOPK, dim=-1)
    topk_weights = (topk_weights / topk_weights.sum(-1, keepdim=True)).to(torch.float32)
    topk_ids = topk_ids.to(torch.int32)

    plan = tokenspeed_kernel.moe_plan(
        "bf16",
        input_dtype=torch.bfloat16,
        activation="swiglu",
        ispp=I_R,  # I_R=256 satisfies the kernel's ispp_alignment gate
        solution="gluon",
    )
    assert plan["apply_kernel_name"] == "gluon_bf16_precomputed_moe_apply"
    assert plan["support_routing"] is False  # precomputed_topk

    w = torch.nn.Module()
    w.w13_weight = w13  # [E, 2*I, D], gate rows [0:I], up rows [I:2I]
    w.w2_weight = w2  # [E, D, I]
    w.top_k = TOPK
    tokenspeed_kernel.moe_process_weights(plan, w)  # no-op for this plan

    out = tokenspeed_kernel.moe_apply(
        plan, x, w, logits, topk_weights=topk_weights, topk_ids=topk_ids
    )
    torch.cuda.synchronize()

    ref = _torch_moe_ref(x, w13, w2, topk_ids, topk_weights)
    peak = ref.abs().max().item()
    max_abs = (out.float() - ref).abs().max().item()
    assert max_abs <= 2e-2 * peak + 1e-2, f"max_abs={max_abs}, peak={peak}"
