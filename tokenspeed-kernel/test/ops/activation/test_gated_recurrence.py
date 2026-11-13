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

"""Bitwise checks of the gated recurrence kernels against Torch's BF16 statements."""

from __future__ import annotations

import pytest
import tokenspeed_kernel.ops.activation._triton.gated_recurrence as gated_recurrence
import torch
from tokenspeed_kernel.ops.activation._triton.gated_recurrence import (
    capped_logit_sum,
    gated_tanh_recurrence,
    projected_gated_tanh_recurrence,
)
from utils import assert_no_triton_compile

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires an admitted GPU"
)


@pytest.mark.parametrize("cap", [30.0, None])
def test_capped_logit_sum_matches_the_torch_statement_bitwise(cap):
    generator = torch.Generator(device="cuda").manual_seed(31)
    rows, cols, valid, width = 5, 50256, 49290, 50272
    base = (
        torch.randn((rows, cols), generator=generator, device="cuda") * 40
    ).bfloat16()
    correction = (
        torch.randn((rows, cols), generator=generator, device="cuda") * 3
    ).bfloat16()
    out = torch.empty((rows, width), dtype=torch.bfloat16, device="cuda")
    capped_logit_sum(base, correction, cap, valid, out)
    expected = base + correction
    if cap is not None:
        expected = (cap * torch.tanh(expected.float() / cap)).to(torch.bfloat16)
    assert torch.equal(out[:, :valid], expected[:, :valid])
    assert torch.isneginf(out[:, valid:]).all()


@pytest.mark.parametrize("scale", [0.01, 1.0, 100.0])
def test_gated_tanh_recurrence_matches_the_torch_statements_bitwise(scale):
    generator = torch.Generator(device="cuda").manual_seed(7)
    rows, width = 9, 384
    projected = (
        torch.randn((rows, 3 * width), generator=generator, device="cuda") * scale
    ).bfloat16()
    state = (
        torch.randn((rows, width), generator=generator, device="cuda") * scale
    ).bfloat16()
    new_state, output = gated_tanh_recurrence(projected, state)
    gate, candidate, expected_output = projected.chunk(3, dim=-1)
    gate = torch.sigmoid(gate)
    expected_state = gate * state + (1.0 - gate) * torch.tanh(candidate)
    assert torch.equal(new_state.view(torch.int16), expected_state.view(torch.int16))
    assert torch.equal(
        output.view(torch.int16), torch.tanh(expected_output).view(torch.int16)
    )


def test_capped_logit_sum_graph_replay_reads_updated_inputs():
    generator = torch.Generator(device="cuda").manual_seed(31)
    rows, cols, valid, width = 16, 50256, 49290, 50272
    base = torch.randn((rows, cols), generator=generator, device="cuda").bfloat16()
    correction = torch.randn_like(base)
    out = torch.empty((rows, width), dtype=torch.bfloat16, device="cuda")
    capped_logit_sum(base, correction, 30.0, valid, out)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        capped_logit_sum(base, correction, 30.0, valid, out)
    for scale in (3.0, 40.0):
        base.normal_(mean=0.0, std=scale, generator=generator)
        correction.normal_(mean=0.0, std=3.0, generator=generator)
        graph.replay()
        expected = (30.0 * torch.tanh((base + correction).float() / 30.0)).bfloat16()
        assert torch.equal(out[:, :valid], expected[:, :valid])
        assert torch.isneginf(out[:, valid:]).all()


@pytest.mark.parametrize("token_dtype", [torch.int32, torch.int64])
def test_projected_gated_tanh_recurrence_rounds_the_whole_projection_once(
    token_dtype,
):
    generator = torch.Generator(device="cuda").manual_seed(11)
    rows, width, embed, vocab = 5, 256, 256, 1000

    def randn(*shape, scale=1.0):
        return torch.randn(shape, generator=generator, device="cuda") * scale

    static = randn(rows, 3 * width)
    state = randn(rows, width).bfloat16()
    embedding = randn(vocab, embed).bfloat16()
    # A wider weight's leading columns, as a projection's recurrent inputs.
    weight = randn(3 * width, width + embed + 64, scale=0.05).bfloat16()
    weight = weight[:, : width + embed]
    tokens = torch.tensor([3, 999, 0, 3, 517], device="cuda", dtype=token_dtype)
    new_state, output = projected_gated_tanh_recurrence(
        static, state, tokens, embedding, weight, shard_padding=None
    )
    # Indices over shards of 250 rows padded to 400 read the same rows.
    padded = tokens + tokens.div(250, rounding_mode="floor") * 150
    for actual, expected in zip(
        projected_gated_tanh_recurrence(
            static, state, padded, embedding, weight, shard_padding=(400, 250)
        ),
        (new_state, output),
    ):
        assert torch.equal(actual, expected)
    inputs = torch.cat((state, embedding[tokens.long()]), dim=-1)
    exact = static.double() + inputs.double() @ weight.double().T
    # Another FP32 sum order can move a rounded projection, and its update, by an ulp.
    projected = exact.bfloat16()
    expected_state, expected_output = gated_tanh_recurrence(projected, state)
    for actual, expected in ((new_state, expected_state), (output, expected_output)):
        assert actual.dtype == torch.bfloat16 and actual.is_contiguous()
        mismatched = (actual != expected).float().mean().item()
        assert mismatched < 0.01
        torch.testing.assert_close(actual, expected, rtol=1.6e-2, atol=1.6e-2)
    # Rows are independent of their neighbors, and batch sizes share one compilation.
    with assert_no_triton_compile(
        gated_recurrence._projected_gated_tanh_recurrence_kernel
    ):
        for count in (1, 2, 4):
            partial = projected_gated_tanh_recurrence(
                static[:count],
                state[:count],
                tokens[:count],
                embedding,
                weight,
                shard_padding=None,
            )
            assert torch.equal(partial[0], new_state[:count])
            assert torch.equal(partial[1], output[:count])
