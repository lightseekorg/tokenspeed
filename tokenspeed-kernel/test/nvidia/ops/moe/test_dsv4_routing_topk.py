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
from tokenspeed_kernel import dsv4_linear_fp32, moe_topk
from tokenspeed_kernel.platform import pdl_enabled
from tokenspeed_kernel.thirdparty.cuda.routing import softplus_sqrt_topk_flash


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires NVIDIA GPU")
@pytest.mark.parametrize(
    "experts,kind", [(256, "bias"), (384, "bias"), (384, "hash32"), (384, "hash64")]
)
@pytest.mark.parametrize("tokens", [0, 17])
@pytest.mark.parametrize("scaling_factor", [0.0, 2.5])
def test_weight_dtype_scales_in_fp32_before_store(
    experts, kind, tokens, scaling_factor
):
    torch.manual_seed(17)
    logits = torch.randn(tokens, experts, device="cuda") * 8
    bias = torch.randn(experts, device="cuda") if kind == "bias" else None
    table = None
    input_ids = None
    if kind != "bias":
        table = torch.stack(
            [torch.randperm(experts, device="cuda")[:6] for _ in range(19)]
        ).int()
        input_ids = (torch.arange(tokens, device="cuda") % 19).to(
            torch.int32 if kind == "hash32" else torch.int64
        )
    args = dict(
        top_k=6,
        score_function="sqrt_softplus",
        selection_method="topk" if kind == "bias" else "hash",
        renormalize=True,
        correction_bias=bias,
        hash_indices_table=table,
        input_ids=input_ids,
        topk_indices_dtype=torch.int32,
        override="cuda_sqrt_softplus_topk",
        solution=None,
    )
    unscaled, ids = moe_topk(
        logits, routed_scaling_factor=1.0, topk_weights_dtype=torch.float32, **args
    )

    def run():
        return moe_topk(
            logits,
            routed_scaling_factor=scaling_factor,
            topk_weights_dtype=torch.bfloat16,
            **args,
        )

    actual, ids_bf16 = run()
    assert actual.dtype == torch.bfloat16
    assert actual.shape == ids_bf16.shape == (tokens, 6)
    torch.testing.assert_close(
        actual, (unscaled * scaling_factor).bfloat16(), atol=0, rtol=0
    )
    torch.testing.assert_close(ids_bf16, ids, atol=0, rtol=0)
    if tokens:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured, captured_ids = run()
        logits.normal_()
        graph.replay()
        unscaled, ids = moe_topk(
            logits, routed_scaling_factor=1.0, topk_weights_dtype=torch.float32, **args
        )
        torch.testing.assert_close(
            captured, (unscaled * scaling_factor).bfloat16(), atol=0, rtol=0
        )
        torch.testing.assert_close(captured_ids, ids, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires NVIDIA GPU")
@pytest.mark.parametrize("experts", [256, 384])
@pytest.mark.parametrize("pdl", [False, True])
@pytest.mark.parametrize(
    "case", ["signed_zero", "boundary_tie", "nonfinite_bias", "infinite_logits"]
)
def test_router_preserves_score_order_and_ties(experts, pdl, case):
    # Seventeen rows exercise complete CTAs and a final partial CTA.
    logits = torch.zeros(17, experts, device="cuda")
    bias = torch.zeros(experts, device="cuda")
    if case == "signed_zero":
        logits[:, ::2] = -0.0
        bias[::2] = -0.0
        selected = [0, 1, 2, 3, 4, 5]
    elif case == "boundary_tie":
        # The tie spans lanes and each lane's candidates. Only six of eight win.
        bias[[0, 31, 32, 63, 64, 95, 96, 127]] = 1.0
        selected = [0, 31, 32, 63, 64, 95]
    elif case == "nonfinite_bias":
        # Preserve the existing ordered-bit representation's positive-NaN rank.
        bias[[31, 63]] = float("nan")
        bias[95] = float("inf")
        bias[0] = -float("inf")
        selected = [31, 63, 95, 1, 2, 3]
    else:
        logits[:, [31, 63]] = float("inf")
        logits[:, 0] = -float("inf")
        selected = [31, 63, 1, 2, 3, 4]
    ids = torch.empty(17, 6, device="cuda", dtype=torch.int32)
    weights = torch.empty(17, 6, device="cuda", dtype=torch.bfloat16)
    original_pdl = pdl_enabled()
    pdl_enabled(pdl)
    try:
        softplus_sqrt_topk_flash(logits, bias, ids, weights, 1.5, True)
        expected_ids = torch.tensor(selected, device="cuda", dtype=torch.int32)
        torch.testing.assert_close(ids, expected_ids.expand_as(ids), atol=0, rtol=0)
        if case == "infinite_logits":
            assert torch.isnan(weights[:, :2]).all()
            assert (weights[:, 2:] == 0).all()
        else:
            torch.testing.assert_close(
                weights, torch.full_like(weights, 0.25), atol=0, rtol=0
            )
    finally:
        pdl_enabled(original_pdl)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires NVIDIA GPU")
@pytest.mark.parametrize("pdl", [False, True])
def test_router_pdl_chain_reads_updated_graph_input(pdl):
    torch.manual_seed(119)
    x = torch.randn(4, 5120, device="cuda", dtype=torch.bfloat16)
    gate = (torch.randn(384, 5120, device="cuda") * 0.01).bfloat16()
    bias = torch.randn(384, device="cuda") * 0.2
    ids = torch.empty(4, 6, device="cuda", dtype=torch.int32)
    weights = torch.empty(4, 6, device="cuda", dtype=torch.bfloat16)
    original_pdl = pdl_enabled()
    pdl_enabled(pdl)
    try:

        def run():
            logits = dsv4_linear_fp32(x, gate)
            softplus_sqrt_topk_flash(logits, bias, ids, weights, 1.0, True)
            # A dependent consumer also must observe the current route outputs.
            return ids.clone(), weights.clone()

        run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual_ids, actual_weights = run()
        for _ in range(3):
            x.normal_()
            expected_ids, expected_weights = run()
            graph.replay()
            torch.testing.assert_close(actual_ids, expected_ids, atol=0, rtol=0)
            torch.testing.assert_close(actual_weights, expected_weights, atol=0, rtol=0)
            scores = torch.nn.functional.softplus(dsv4_linear_fp32(x, gate)).sqrt()
            reference_ids = (scores + bias).argsort(
                dim=-1, descending=True, stable=True
            )[:, :6]
            selected = scores.gather(1, reference_ids)
            reference_weights = (
                selected / selected.sum(dim=-1, keepdim=True)
            ).bfloat16()
            torch.testing.assert_close(actual_ids, reference_ids.int(), atol=0, rtol=0)
            torch.testing.assert_close(
                actual_weights, reference_weights, atol=2e-3, rtol=1e-2
            )
    finally:
        pdl_enabled(original_pdl)
