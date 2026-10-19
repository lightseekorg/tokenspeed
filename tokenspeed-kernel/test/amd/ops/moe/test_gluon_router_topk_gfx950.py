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

"""GFX950 Gluon fused router GEMV and sqrt-softplus top-k."""

import pytest
import torch
from utils import is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)

from tokenspeed_kernel.ops.gemm import dsv4_linear_fp32  # noqa: E402
from tokenspeed_kernel.ops.moe import moe_router_topk, moe_topk  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.moe.router_gemv import (  # noqa: E402
    launch_gluon_router_gemv_splitk_gfx950,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.sqrt_softplus_topk import (  # noqa: E402
    launch_gluon_sqrt_softplus_topk_gfx950,
)

HIDDEN = 4096
TOP_K = 6
VOCAB = 1024


def _inputs(tokens, experts, routing_kind):
    generator = torch.Generator(device="cuda").manual_seed(tokens * experts)
    x = torch.randn(
        tokens, HIDDEN, device="cuda", generator=generator, dtype=torch.float32
    ).to(torch.bfloat16)
    weight = (
        torch.randn(
            experts, HIDDEN, device="cuda", generator=generator, dtype=torch.float32
        )
        * 0.05
    ).to(torch.bfloat16)
    bias = hash_table = input_ids = None
    if routing_kind == "bias":
        bias = torch.randn(
            experts, device="cuda", generator=generator, dtype=torch.float32
        )
    elif routing_kind == "hash":
        hash_table = torch.stack(
            [
                torch.randperm(experts, device="cuda", generator=generator)[:TOP_K]
                for _ in range(VOCAB)
            ]
        ).to(torch.int32)
        input_ids = torch.randint(
            0, VOCAB, (tokens,), device="cuda", generator=generator, dtype=torch.int64
        )
    return x, weight, bias, hash_table, input_ids


@pytest.mark.parametrize("routing_kind", ["plain", "bias", "hash"])
@pytest.mark.parametrize("experts", [256, 384])
@pytest.mark.parametrize("tokens", [1, 15, 16, 17, 128])
def test_router_topk_matches_gemm_then_topk(tokens, experts, routing_kind):
    x, weight, bias, hash_table, input_ids = _inputs(tokens, experts, routing_kind)
    weights, ids = moe_router_topk(
        x,
        weight,
        TOP_K,
        True,
        1.5,
        correction_bias=bias,
        hash_indices_table=hash_table,
        input_ids=input_ids,
        solution="gluon",
    )
    logits = dsv4_linear_fp32(x, weight)
    expected_weights, expected_ids = moe_topk(
        logits,
        TOP_K,
        "sqrt_softplus",
        "hash" if routing_kind == "hash" else "topk",
        True,
        1.5,
        correction_bias=bias,
        hash_indices_table=hash_table,
        input_ids=input_ids,
    )
    torch.testing.assert_close(ids, expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(weights, expected_weights, rtol=0, atol=1e-6)


@pytest.mark.parametrize("tokens", [1, 15, 16, 17, 128])
def test_topk_split_partials_match_summed_logits(tokens):
    x, weight, bias, _, _ = _inputs(tokens, 256, "bias")
    partials = launch_gluon_router_gemv_splitk_gfx950(x, weight)
    assert partials.shape == (HIDDEN // 512, tokens, 256)
    logits = partials[0].clone()
    for partial in partials[1:]:
        logits += partial
    weights, ids, _ = launch_gluon_sqrt_softplus_topk_gfx950(
        partials, TOP_K, True, correction_bias=bias, need_scores=False
    )
    expected_weights, expected_ids, _ = launch_gluon_sqrt_softplus_topk_gfx950(
        logits, TOP_K, True, correction_bias=bias, need_scores=False
    )
    torch.testing.assert_close(ids, expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(weights, expected_weights, rtol=0, atol=1e-6)
