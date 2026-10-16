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

"""FlashInfer TRTLLM-Gen MXFP4 MoE with a plain SiLU-gated activation.

A ``silu`` layer (MiMo-V2) computes ``silu(gate) * up``; the gpt-oss SwiGLU
defaults (alpha 1.702, ``up + 1``, clamp 7) belong to ``swiglu`` layers only.
"""

from __future__ import annotations

import pytest
import torch
from kimi3_reference import dequantize_mxfp4
from tokenspeed_kernel.ops.moe import moe_apply as kernel_moe_apply
from tokenspeed_kernel.ops.moe import moe_plan as kernel_moe_plan
from tokenspeed_kernel.ops.moe import moe_process_weights as kernel_moe_process
from utils import make_mxfp4_moe_weights

NUM_EXPERTS = 8
TOP_K = 2
HIDDEN = 256
ISPP = 128


def _runtime_reason() -> str | None:
    if not torch.cuda.is_available():
        return "requires CUDA"
    if not (10, 0) <= torch.cuda.get_device_capability() <= (10, 3):
        return "flashinfer TRTLLM-Gen MXFP4 MoE targets the sm_100 family"
    return None


_reason = _runtime_reason()
requires_flashinfer_mxfp4 = pytest.mark.skipif(_reason is not None, reason=str(_reason))


class _MoEWeights(torch.nn.Module):
    def __init__(self, raw: dict[str, torch.Tensor]) -> None:
        super().__init__()
        self.w13_weight = torch.nn.Parameter(raw["w13_weight"], requires_grad=False)
        self.w13_weight_scale = torch.nn.Parameter(
            raw["w13_scale"], requires_grad=False
        )
        self.w2_weight = torch.nn.Parameter(raw["w2_weight"], requires_grad=False)
        self.w2_weight_scale = torch.nn.Parameter(raw["w2_scale"], requires_grad=False)
        self.activation = "silu"
        self.w13_input_layout = "concatenated"
        self.num_experts = NUM_EXPERTS
        self.num_local_experts = NUM_EXPERTS
        self.top_k = TOP_K
        self.hidden_size = HIDDEN


def _reference(raw, x, topk_ids, topk_weights, *, gpt_oss_swiglu: bool):
    out = torch.zeros(x.shape[0], HIDDEN, dtype=torch.float32)
    for expert in range(NUM_EXPERTS):
        token, slot = (topk_ids == expert).nonzero(as_tuple=True)
        if not token.numel():
            continue
        w13 = dequantize_mxfp4(raw["w13_weight"][expert], raw["w13_scale"][expert])
        w2 = dequantize_mxfp4(raw["w2_weight"][expert], raw["w2_scale"][expert])
        gate, up = (x[token].float() @ w13.T).bfloat16().float().chunk(2, dim=-1)
        if gpt_oss_swiglu:
            gate, up = gate.clamp(max=7.0), up.clamp(-7.0, 7.0)
            act = gate * torch.sigmoid(1.702 * gate) * (up + 1.0)
        else:
            act = torch.nn.functional.silu(gate) * up
        y = act.bfloat16().float() @ w2.T
        out.index_add_(0, token, y * topk_weights[token, slot].float()[:, None])
    return out


@requires_flashinfer_mxfp4
def test_flashinfer_mxfp4_silu_experts_use_plain_swiglu() -> None:
    generator = torch.Generator().manual_seed(20261016)
    raw = make_mxfp4_moe_weights(NUM_EXPERTS, HIDDEN, ISPP, generator, device="cpu")
    x = (torch.randn(32, HIDDEN, generator=generator) * 0.5).bfloat16()
    logits = torch.randn(32, NUM_EXPERTS, generator=generator)
    topk_weights, topk_ids = torch.topk(logits.softmax(-1), TOP_K, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(-1, keepdim=True)
    topk_ids = topk_ids.to(torch.int32)

    plan = kernel_moe_plan(
        "mxfp4",
        input_dtype=torch.bfloat16,
        activation="silu",
        routing_mode="precomputed_topk",
        ep_size=1,
        ispp=ISPP,
        hidden=HIDDEN,
        swiglu_form=None,
        activation_clamped=False,
        expert_id_repeats=False,
        fast_math=False,
        combine_order="rank",
        internal_activation_dtype="input",
        solution="flashinfer_trtllm",
    )
    assert plan["apply_kernel_name"] == "flashinfer_trtllm_mxfp4_moe_apply"
    w = _MoEWeights({k: v.clone() for k, v in raw.items()}).cuda()
    kernel_moe_process(plan, w)
    assert w.gemm1_alpha is None and w.gemm1_beta is None
    assert w.gemm1_clamp_limit is None

    actual = (
        kernel_moe_apply(
            plan,
            x.cuda(),
            w,
            logits.cuda(),
            topk_weights=topk_weights.cuda(),
            topk_ids=topk_ids.cuda(),
        )
        .float()
        .cpu()
    )

    expected = _reference(raw, x, topk_ids.long(), topk_weights, gpt_oss_swiglu=False)
    gpt_oss = _reference(raw, x, topk_ids.long(), topk_weights, gpt_oss_swiglu=True)
    scale = expected.abs().mean()
    err = (actual - expected).abs().mean() / scale
    err_gpt_oss = (actual - gpt_oss).abs().mean() / scale
    # MXFP8 activations bound the agreement; the other activation is far off.
    assert err < 0.05, f"relative error {err:.4f}"
    assert err_gpt_oss > 4 * err, (err, err_gpt_oss)
