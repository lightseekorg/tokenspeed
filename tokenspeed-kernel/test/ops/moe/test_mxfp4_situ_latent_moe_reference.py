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

import torch
import torch.nn.functional as F
from kimi3_reference import dequantize_mxfp4 as dequantize_mxfp4_reference
from kimi3_reference import (
    mxfp4_moe_reference,
)
from utils import make_mxfp4_moe_weights


def test_a16w4_moe_matches_explicit_token_slot_reference() -> None:
    generator = torch.Generator().manual_seed(7)
    num_tokens, num_experts = 4, 3
    latent_size = intermediate_size = 32
    top_k = 2

    hidden_states = (
        torch.randn(num_tokens, latent_size, generator=generator) * 0.2
    ).to(torch.bfloat16)
    raw = make_mxfp4_moe_weights(
        num_experts,
        latent_size,
        intermediate_size,
        generator,
        device="cpu",
    )
    w13_packed, w13_scales = raw["w13_weight"], raw["w13_scale"]
    w2_packed, w2_scales = raw["w2_weight"], raw["w2_scale"]
    topk_ids = torch.tensor([[0, 2], [1, 0], [2, 1], [0, 1]], dtype=torch.int32)
    topk_weights = torch.tensor(
        [[0.6, 0.4], [0.75, 0.25], [0.2, 0.8], [0.5, 0.5]],
        dtype=torch.float32,
    )
    beta = 2.0
    linear_beta = 3.0

    actual = mxfp4_moe_reference(
        hidden_states,
        w13_packed,
        w13_scales,
        w2_packed,
        w2_scales,
        topk_ids,
        topk_weights,
        activation_dtype=torch.bfloat16,
        situ_beta=beta,
        situ_linear_beta=linear_beta,
    )

    w13 = dequantize_mxfp4_reference(w13_packed, w13_scales)
    w2 = dequantize_mxfp4_reference(w2_packed, w2_scales)
    expected = torch.zeros_like(hidden_states, dtype=torch.float32)
    for token in range(num_tokens):
        for slot in range(top_k):
            expert = int(topk_ids[token, slot])
            gate_up = F.linear(
                hidden_states[token : token + 1].float(), w13[expert]
            ).to(hidden_states.dtype)
            gate, up = gate_up.float().chunk(2, dim=-1)
            gate = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
            up = linear_beta * torch.tanh(up / linear_beta)
            intermediate = (gate * up).to(hidden_states.dtype)
            expert_output = F.linear(intermediate.float(), w2[expert]).to(
                hidden_states.dtype
            )
            expected[token] += topk_weights[token, slot] * expert_output[0].float()
    expected = expected.to(hidden_states.dtype)

    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
