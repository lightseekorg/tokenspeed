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

"""CPU behavior tests for depth residual ownership and normalization."""

from __future__ import annotations

import pytest
import torch

from tokenspeed.runtime.layers.attention_residual import (
    AttentionResidual,
    AttentionResidualState,
    PrecisionRMSNorm,
    norm_gains,
)


def _fp32_norms(count: int, width: int) -> tuple[PrecisionRMSNorm, ...]:
    return tuple(
        PrecisionRMSNorm(width, 1e-6, False, torch.float32, False, False)
        for _ in range(count)
    )


def _rms(x: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + 1e-6)


@pytest.mark.parametrize("normalize_values", [False, True])
def test_depth_state_boundaries_and_three_attention_inputs(normalize_values):
    owner = AttentionResidual(
        12, 4, 6, [0, 1, 2, 3, 4, 6, 7, 8, 9, 10], normalize_values
    )
    owner.derive_score_weights()
    embedding = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.bfloat16)
    state = AttentionResidualState(owner, embedding, None, None)
    boundary_counts = []
    for layer in range(12):
        norms = _fp32_norms(3 if layer in (5, 11) else 1, 4)
        attention_inputs = state.attention_input(layer, norms, norm_gains(norms))
        assert len(attention_inputs) == len(norms)
        for value in attention_inputs:
            assert value.shape == embedding.shape
        if layer == 0:
            torch.testing.assert_close(
                attention_inputs[0], _rms(embedding.float()), rtol=0, atol=0
            )
        attention = torch.full_like(embedding, 1.0)
        mlp = torch.full_like(embedding, 2.0)
        mlp_input = state.mlp_input(layer, attention, _fp32_norms(1, 4)[0])
        assert mlp_input.dtype == torch.float32
        state.finalize_layer(layer, attention, mlp)
        if layer in (0, 5, 6, 11):
            boundary_counts.append(
                (layer, state.num_blocks, float(state.partial[0, 0]))
            )
    assert boundary_counts == [(0, 1, 3.0), (5, 1, 18.0), (6, 2, 3.0), (11, 2, 18.0)]
    assert torch.equal(state.blocks[0], embedding)
    final = state.compute_output(_fp32_norms(1, 4)[0])
    sources = torch.stack(
        [embedding, torch.full_like(embedding, 18), torch.full_like(embedding, 18)]
    ).float()
    if normalize_values:
        sources = _rms(sources)
    torch.testing.assert_close(final, _rms(sources.mean(dim=0)), rtol=1e-6, atol=1e-6)
    assert final.dtype == torch.float32


def test_attention_mix_rounds_the_pending_sum_once_and_stores_the_block():
    owner = AttentionResidual(4, 2, 2, [0, 1, 2, 3], False)
    owner.derive_score_weights()
    embedding = torch.tensor([[256.0, 1.0]], dtype=torch.bfloat16)
    state = AttentionResidualState(owner, embedding, None, None)
    norms = _fp32_norms(1, 2)
    for layer, (attention, mlp) in enumerate(
        [((256.0, 1.0), (0.0, 0.0)), ((1.0, 0.0), (-256.0, 0.5))]
    ):
        state.attention_input(layer, norms, norm_gains(norms))
        attention = torch.tensor([attention], dtype=torch.bfloat16)
        state.mlp_input(layer, attention, norms[0])
        state.finalize_layer(
            layer, attention, torch.tensor([mlp], dtype=torch.bfloat16)
        )
    # Layer 2's attention mix stores 256 + 1 - 256 rounded once (256 + 1 first gives 0).
    state.attention_input(2, norms, norm_gains(norms))
    assert state.num_blocks == 1
    assert state.blocks[1].tolist() == [[1.0, 1.5]]
    assert state.partial.data_ptr() == state.blocks[1].data_ptr()


def test_nonboundary_residual_sum_rounds_only_once():
    owner = AttentionResidual(2, 1, 2, [0, 1], False)
    state = AttentionResidualState(
        owner, torch.tensor([[256]], dtype=torch.bfloat16), None, None
    )
    state.finalize_layer(
        1,
        torch.tensor([[1]], dtype=torch.bfloat16),
        torch.tensor([[-256]], dtype=torch.bfloat16),
    )
    assert state.partial.item() == 1.0


@pytest.mark.parametrize("pre_affine", [False, True])
@pytest.mark.parametrize("elementwise_affine", [False, True])
def test_norm_preprocessing_outputs_bf16_from_fp32(pre_affine, elementwise_affine):
    norm = PrecisionRMSNorm(
        4, 1e-6, pre_affine, torch.bfloat16, elementwise_affine, False
    )
    x = torch.tensor([[0.1251, -0.9917, 2.1234, 8.7654]], dtype=torch.float32)
    values = x.clone()
    if pre_affine:
        with torch.no_grad():
            norm.pre_weight.copy_(torch.tensor([1.0, 2.0, 0.5, 0.25]))
        values = values * norm.pre_weight.float()
    expected = values * torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + 1e-6)
    if elementwise_affine:
        with torch.no_grad():
            norm.weight.copy_(torch.tensor([0.5, 1.0, 2.0, -1.0]))
        expected = expected * norm.weight.float()
        assert norm.weight.dtype == torch.bfloat16
    actual = norm(x)
    torch.testing.assert_close(actual, expected.to(torch.bfloat16), rtol=0, atol=0)
    assert actual.dtype == torch.bfloat16
    assert norm(torch.empty((0, 4), dtype=torch.float32)).shape == (0, 4)


def test_query_loading_preserves_names_and_full_layer_geometry():
    owner = AttentionResidual(12, 4, 6, [0, 1, 2, 3, 4, 6, 7, 8, 9, 10], True)
    parameters = dict(owner.named_parameters())
    assert parameters["attn_queries.4"].shape == (3, 4)
    assert parameters["attn_queries.10"].shape == (3, 4)
    assert parameters["attn_queries.0"].shape == (4,)
    assert parameters["output_query"].shape == (4,)
    assert all(value.dtype == torch.float32 for value in parameters.values())
    owner.load_state_dict(
        {name: torch.full_like(value, 0.25) for name, value in parameters.items()},
        strict=True,
    )
    assert owner.attn_query(5).tolist() == [[0.25] * 4] * 3


@pytest.mark.parametrize(
    "arguments",
    [
        (0, 4, 1, [], True),
        (2, 0, 1, [0], True),
        (2, 4, 3, [0], True),
        (2, 4, 1, [0, 0], True),
        (2, 4, 1, [2], True),
    ],
)
def test_invalid_depth_geometry_is_rejected(arguments):
    with pytest.raises(ValueError):
        AttentionResidual(*arguments)


def _twelve_layer_mixer(tokens: int):
    """A run of every mix of a 12-layer, two-block depth stack (full layers
    5 and 11) over fixed sublayer outputs, as ``run(aux_stream)``."""
    torch.manual_seed(3)
    width = 64
    owner = AttentionResidual(12, width, 6, [0, 1, 2, 3, 4, 6, 7, 8, 9, 10], True)
    with torch.no_grad():
        for query in owner.parameters():
            query.normal_()
    owner = owner.cuda()
    owner.derive_score_weights()
    embedding = torch.randn((tokens, width), device="cuda").to(torch.bfloat16)
    outputs = [
        torch.randn((tokens, width), device="cuda").to(torch.bfloat16)
        for _ in range(24)
    ]

    def bf16_norm():
        return PrecisionRMSNorm(width, 1e-6, False, torch.bfloat16, True, False).cuda()

    attention_norms = [
        tuple(bf16_norm() for _ in range(3 if layer in (5, 11) else 1))
        for layer in range(12)
    ]
    norm = bf16_norm()

    def run(aux_stream):
        state = AttentionResidualState(owner, embedding, aux_stream, None)
        mixed = []
        for layer in range(12):
            norms = attention_norms[layer]
            mixed.extend(state.attention_input(layer, norms, norm_gains(norms)))
            attention, mlp = outputs[2 * layer], outputs[2 * layer + 1]
            mixed.append(state.mlp_input(layer, attention, norm))
            state.finalize_layer(layer, attention, mlp)
        mixed.append(state.compute_output(norm))
        return mixed

    return run


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_block_set_partials_on_an_aux_stream_mix_like_inline_ones():
    """Partials started on an aux stream at each block boundary give the same
    mixtures as partials computed inline, eagerly and replayed in a graph."""
    run = _twelve_layer_mixer(8)
    expected = run(None)
    aux = torch.cuda.Stream()
    actual = run(aux)
    torch.cuda.synchronize()
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    graph = torch.cuda.CUDAGraph()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        run(aux)
    torch.cuda.current_stream().wait_stream(side)
    with torch.cuda.graph(graph):
        captured = run(aux)
    graph.replay()
    torch.cuda.synchronize()
    assert all(torch.equal(a, b) for a, b in zip(captured, expected))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_forwards_above_the_sweep_cap_mix_like_block_set_sweeps(monkeypatch):
    """Without block-set sweeps each query mixes in one pass: the same
    mixtures up to FP32 summation order, so at most one BF16 rounding apart."""
    run = _twelve_layer_mixer(8)
    swept = run(torch.cuda.Stream())
    monkeypatch.setattr(AttentionResidualState, "set_sweep_max_tokens", 0)
    one_pass = run(torch.cuda.Stream())
    torch.cuda.synchronize()
    assert len(one_pass) == len(swept)
    for a, b in zip(one_pass, swept):
        assert a.dtype == b.dtype == torch.bfloat16
        torch.testing.assert_close(a.float(), b.float(), rtol=8e-3, atol=0)
