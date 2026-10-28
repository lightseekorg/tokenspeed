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

"""Common BF16 expert activation and weighted-route arithmetic."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.moe._triton import bf16_epilogue
from tokenspeed_kernel.ops.moe._triton.bf16_epilogue import (
    clamped_swiglu_bf16,
    finalize_bf16_routes,
)
from utils import assert_no_triton_compile

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")


def _finalize_reference(values, weights, ids, *, weighted_routes):
    values = values.cpu().view(*ids.shape, -1)
    weights, ids = weights.cpu(), ids.cpu()
    total = torch.zeros((ids.shape[0], values.shape[-1]), dtype=torch.float32)
    for token in range(ids.shape[0]):
        for slot in range(ids.shape[1]):
            if not 2 <= ids[token, slot].item() < 4:
                continue
            value = values[token, slot].bfloat16().float()
            if not weighted_routes:
                weight = weights[token, slot].bfloat16().float()
                value = (value * weight).bfloat16().float()
            total[token] += value
    return total.bfloat16()


def _activation_reference(gate_up, ids, *, limit):
    width = gate_up.shape[-1] // 2
    gate_up, ids = gate_up.cpu(), ids.cpu().flatten()
    output = torch.zeros((gate_up.shape[0], width), dtype=torch.bfloat16)
    for row, expert in enumerate(ids.tolist()):
        if not 2 <= expert < 4:
            continue
        gate, up = gate_up[row].bfloat16().float().chunk(2)
        if limit is not None:
            gate = gate.clamp(max=limit)
            up = up.clamp(-limit, limit)
        output[row] = (torch.nn.functional.silu(gate) * up).bfloat16()
    return output


@pytest.mark.parametrize("weighted_routes", [False, True])
@pytest.mark.parametrize("ids_dtype", [torch.int32, torch.int64])
def test_finalize_rounds_values_weights_and_product_before_fp32_slot_sum(
    weighted_routes, ids_dtype
):
    vector = torch.tensor(
        [1.00390625, 0.7318954467773438, 3.1415927, -1.00390625]
    ).repeat(32)
    values = torch.full((2, 4, 128), float("nan"))
    values[0, 0] = vector
    values[0, 1] = -vector * 0.7
    values[1, 0] = vector * 0.31
    values[1, 2] = vector
    ids = torch.tensor(
        [[2, 2, -1, torch.iinfo(ids_dtype).min], [3, 0, 2, torch.iinfo(ids_dtype).max]],
        dtype=ids_dtype,
    )
    weights = torch.tensor(
        [
            [0.7, 0.31, float("nan"), float("nan")],
            [0.31, float("nan"), 0.7319, float("nan")],
        ]
    )
    expected = _finalize_reference(
        values, weights, ids, weighted_routes=weighted_routes
    )
    if not weighted_routes:
        # This data distinguishes the required BF16 weight cast from merely
        # rounding a BF16 value times an unrounded FP32 weight.
        wrong = torch.zeros((2, 128), dtype=torch.float32)
        for token in range(2):
            for slot in range(4):
                if 2 <= ids[token, slot].item() < 4:
                    wrong[token] += (
                        (values[token, slot].bfloat16().float() * weights[token, slot])
                        .bfloat16()
                        .float()
                    )
        assert not torch.equal(expected, wrong.bfloat16())
    actual = finalize_bf16_routes(
        values.flatten(0, 1).cuda(),
        weights.cuda(),
        ids.cuda(),
        expert_start=2,
        num_local_experts=2,
        weighted_routes=weighted_routes,
    )
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    assert torch.isfinite(actual).all()


@pytest.mark.parametrize(
    "route_values, expected",
    [([256.0, 1.0, -256.0, 0.5], 1.5), ([2**25, 1.0, -(2**25), 0.5], 0.5)],
)
def test_finalize_accumulates_in_fp32_in_slot_order(route_values, expected):
    values = torch.tensor(route_values).view(4, 1).expand(4, 128).contiguous().cuda()
    weights = torch.ones((1, 4), device="cuda")
    ids = torch.full((1, 4), 2, device="cuda", dtype=torch.int32)
    output = finalize_bf16_routes(
        values, weights, ids, expert_start=2, num_local_experts=2, weighted_routes=False
    )
    torch.testing.assert_close(
        output, torch.full_like(output, expected), rtol=0, atol=0
    )


@pytest.mark.parametrize("limit", [None, 3.0])
@pytest.mark.parametrize("input_dtype", [torch.float32, torch.bfloat16])
def test_activation_rounds_operands_clamps_and_ignores_invalid_rows(limit, input_dtype):
    gate = torch.tensor(
        [-50.0, -3.1, -0.731895, 0.731895, 3.1, 50.0, 1.00390625, -1.00390625]
    ).repeat(33)
    up = torch.tensor(
        [50.0, -50.0, 0.7, -0.31, 5.0, -5.0, 1.00390625, 0.731895]
    ).repeat(33)
    gate_up = torch.full((4, 2 * gate.numel()), float("nan"))
    gate_up[0] = torch.cat((gate, up))
    gate_up[2] = torch.cat((-gate, -up))
    gate_up = gate_up.to(input_dtype)
    ids = torch.tensor([[2, -1, 3, -(2**31)]], dtype=torch.int32)
    expected = _activation_reference(gate_up, ids, limit=limit)
    actual = clamped_swiglu_bf16(
        gate_up.cuda(), ids.cuda(), expert_start=2, num_local_experts=2, limit=limit
    )
    # CPU SiLU and the device sigmoid may differ within one BF16 ULP.
    magnitude = expected.abs()
    ulp = (
        torch.nextafter(magnitude, torch.full_like(magnitude, float("inf"))).float()
        - magnitude.float()
    )
    assert torch.all((actual.cpu().float() - expected.float()).abs() <= ulp)
    rounded_input_output = clamped_swiglu_bf16(
        gate_up.bfloat16().cuda(),
        ids.cuda(),
        expert_start=2,
        num_local_experts=2,
        limit=limit,
    )
    torch.testing.assert_close(actual, rounded_input_output, rtol=0, atol=0)
    assert torch.count_nonzero(actual[[1, 3]]).item() == 0
    if limit is not None:
        unclamped = _activation_reference(gate_up, ids, limit=None)
        assert not torch.equal(expected, unclamped)


@pytest.mark.parametrize("weighted_routes", [False, True])
def test_empty_routes_have_empty_bf16_outputs(weighted_routes):
    ids = torch.empty((0, 4), dtype=torch.int32, device="cuda")
    weights = torch.empty((0, 4), device="cuda")
    values = torch.empty((0, 128), device="cuda")
    output = finalize_bf16_routes(
        values,
        weights,
        ids,
        expert_start=2,
        num_local_experts=2,
        weighted_routes=weighted_routes,
    )
    activation = clamped_swiglu_bf16(
        values, ids, expert_start=2, num_local_experts=2, limit=3.0
    )
    assert output.shape == (0, 128) and output.dtype == torch.bfloat16
    assert activation.shape == (0, 64) and activation.dtype == torch.bfloat16


def test_helpers_accept_strided_inputs():
    generator = torch.Generator().manual_seed(71)
    values = torch.randn((8, 256), generator=generator).cuda()[:, ::2]
    weights = torch.rand((2, 8), generator=generator).cuda()[:, ::2]
    ids = torch.tensor(
        [[2, -1, 3, -1, 2, -1, 3, -1]] * 2, dtype=torch.int64, device="cuda"
    )[:, ::2]
    output = finalize_bf16_routes(
        values, weights, ids, expert_start=2, num_local_experts=2, weighted_routes=False
    )
    expected = _finalize_reference(values, weights, ids, weighted_routes=False)
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
    activation = clamped_swiglu_bf16(
        values, ids, expert_start=2, num_local_experts=2, limit=3.0
    )
    expected_activation = clamped_swiglu_bf16(
        values.contiguous(),
        ids.contiguous(),
        expert_start=2,
        num_local_experts=2,
        limit=3.0,
    )
    torch.testing.assert_close(activation, expected_activation, rtol=0, atol=0)


@pytest.mark.parametrize("weighted_routes", [False, True])
def test_helpers_do_not_recompile_for_token_count(weighted_routes):
    generator = torch.Generator(device="cuda").manual_seed(83)
    max_tokens, top_k, hidden = 40, 4, 128
    gate_up = torch.randn(
        (max_tokens * top_k, 2 * hidden), generator=generator, device="cuda"
    )
    weights = torch.rand((max_tokens, top_k), generator=generator, device="cuda")
    ids = torch.full((max_tokens, top_k), 2, dtype=torch.int32, device="cuda")

    def run(tokens):
        activation = clamped_swiglu_bf16(
            gate_up[: tokens * top_k],
            ids[:tokens],
            expert_start=2,
            num_local_experts=2,
            limit=3.0,
        )
        return finalize_bf16_routes(
            activation,
            weights[:tokens],
            ids[:tokens],
            expert_start=2,
            num_local_experts=2,
            weighted_routes=weighted_routes,
        )

    # The runtime row count has Triton's divisible-by-16 specialization classes.
    for tokens in (1, 4):
        run(tokens)
    # A token's result must not depend on how many other tokens share its batch.
    expected = run(max_tokens)
    with assert_no_triton_compile(
        bf16_epilogue._clamped_swiglu_bf16_kernel,
        bf16_epilogue._finalize_bf16_routes_kernel,
    ):
        for tokens in (2, 3, 5, 8, 11, 16, 33):
            torch.testing.assert_close(run(tokens), expected[:tokens], rtol=0, atol=0)
