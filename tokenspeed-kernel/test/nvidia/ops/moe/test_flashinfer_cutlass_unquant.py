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

"""FlashInfer CUTLASS unquantized MoE: PDL stays off, and EP numerics.

PDL raced inside this fused-MoE kernel chain on SM90 at decode-sized
batches: a routed GEMM row transiently read NaN while rerunning the
identical call was clean (LongCat Flash-Lite, TP8/EP8 H20, bs=1 decode).
The wrapper therefore serializes the chain; the first test pins that. The
second covers an expert-parallel rank whose tokens repeat a zero-weight
placeholder expert, as LongCat's zero experts do.
"""

from __future__ import annotations

from importlib.util import find_spec
from types import SimpleNamespace

import pytest
import torch
from tokenspeed_kernel.platform import current_platform


def _requires_flashinfer_hopper() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not current_platform().is_nvidia or not current_platform().is_hopper_plus:
        pytest.skip("NVIDIA Hopper or newer required")
    if find_spec("flashinfer") is None:
        pytest.skip("requires flashinfer")


def test_wrapper_never_enables_pdl(monkeypatch) -> None:
    _requires_flashinfer_hopper()
    from tokenspeed_kernel.ops.moe.flashinfer import cutlass_unquant

    seen = {}

    def fake_cutlass_fused_moe(**kwargs):
        seen.update(kwargs)
        return [kwargs["input"]]

    monkeypatch.setattr(cutlass_unquant, "cutlass_fused_moe", fake_cutlass_fused_moe)
    x = torch.zeros(2, 8, dtype=torch.bfloat16)
    cutlass_unquant.flashinfer_cutlass_unquant_moe_apply(
        {},
        x,
        SimpleNamespace(w13_weight=None, w2_weight=None),
        None,
        topk_weights=torch.ones(2, 1),
        topk_ids=torch.zeros(2, 1, dtype=torch.int32),
        enable_pdl=True,
    )
    assert seen["enable_pdl"] is False


def _reference(x, ids, weights, w13, w2, first_expert):
    """SwiGLU experts in fp32; the w13 halves are [gate | up] as loaded."""
    out = torch.zeros(x.shape[0], w2.shape[1], dtype=torch.float32, device=x.device)
    inter = w2.shape[2]
    for token in range(x.shape[0]):
        for slot in range(ids.shape[1]):
            expert = int(ids[token, slot]) - first_expert
            if not 0 <= expert < w13.shape[0]:
                continue
            gate_up = x[token].float() @ w13[expert].float().T
            act = torch.nn.functional.silu(gate_up[:inter]) * gate_up[inter:]
            out[token] += weights[token, slot] * (act @ w2[expert].float().T)
    return out


@pytest.mark.parametrize("ep_rank", [0, 1])
def test_ep_rank_with_repeated_placeholder_experts(ep_rank: int) -> None:
    _requires_flashinfer_hopper()
    import tokenspeed_kernel

    torch.manual_seed(0)
    device = "cuda"
    num_experts, ep_size, hidden, inter, top_k, tokens = 16, 2, 256, 128, 6, 5
    local = num_experts // ep_size
    w13 = (torch.randn(local, 2 * inter, hidden, device=device) * 0.05).bfloat16()
    w2 = (torch.randn(local, hidden, inter, device=device) * 0.05).bfloat16()
    x = torch.randn(tokens, hidden, device=device).bfloat16()
    # Every token routes a few real experts; the remaining slots repeat
    # placeholder expert 0 at weight zero.
    ids = torch.zeros(tokens, top_k, dtype=torch.int32, device=device)
    weights = torch.zeros(tokens, top_k, device=device)
    for token in range(tokens):
        real = 1 + token % 4
        ids[token, :real] = torch.randperm(num_experts - 1, device=device)[:real] + 1
        weights[token, :real] = torch.rand(real, device=device)

    plan = tokenspeed_kernel.moe_plan(
        "unquant",
        input_dtype=torch.bfloat16,
        activation="swiglu",
        routing_mode="precomputed_topk",
        ep_size=ep_size,
        ispp=inter,
        hidden=hidden,
        swiglu_form="standard",
        activation_clamped=False,
        expert_id_repeats=True,
        internal_activation_dtype="input",
        fast_math=True,
        solution="flashinfer_cutlass",
    )
    reference_w13 = w13.clone()
    layer = SimpleNamespace(
        w13_weight=torch.nn.Parameter(w13, requires_grad=False),
        w2_weight=torch.nn.Parameter(w2, requires_grad=False),
        ep_size=ep_size,
        ep_rank=ep_rank,
        tp_size=1,
        tp_rank=0,
    )
    tokenspeed_kernel.moe_process_weights(plan, layer)
    out = tokenspeed_kernel.moe_apply(
        plan,
        x,
        layer,
        None,
        topk_weights=weights,
        topk_ids=ids,
        num_tokens_global=tokens,
        max_num_tokens_per_gpu=tokens,
    )
    expected = _reference(
        x, ids, weights, reference_w13, w2, first_expert=ep_rank * local
    )
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out.float(), expected, atol=2e-2, rtol=2e-2)
