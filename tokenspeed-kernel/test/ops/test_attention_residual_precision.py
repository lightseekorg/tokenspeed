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

"""CPU and optional GPU comparisons for depth mixtures and explicit precision."""

from __future__ import annotations

import math

import pytest
import torch
from l2_clip_helpers import triton_l2_clip
from tokenspeed_kernel.ops.layernorm.precision import (
    add_rmsnorm_precision,
    l2_clip,
    rmsnorm_precision,
)
from tokenspeed_kernel.ops.residual.attention import (
    attention_residual_combine,
    attention_residual_mix,
    attention_residual_partials,
    residual_sum,
)


def _device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm device unavailable")
    return torch.device(name)


def attention_residual(
    blocks: list[torch.Tensor],
    partial: torch.Tensor,
    queries: torch.Tensor,
    *,
    partial_addend: torch.Tensor | None,
    normalize_values: bool,
    eps: float,
) -> tuple[torch.Tensor, ...]:
    """One FP32 mixture of ``blocks`` and ``partial + partial_addend`` per query."""
    width = partial.shape[1]
    weights = (queries.float().reshape(-1, width) * width**-0.5).contiguous()
    stacked = torch.stack(blocks) if blocks else partial.new_empty((0, *partial.shape))
    partials = attention_residual_partials(
        stacked, weights, normalize_values=normalize_values, eps=eps
    )
    terms = (partial,) if partial_addend is None else (partial, partial_addend)
    mixed = attention_residual_combine(
        terms,
        weights,
        partials,
        candidate_out=None,
        normalize_values=normalize_values,
        eps=eps,
        norm_weights=None,
        norm_eps=eps,
        output_dtype=torch.float32,
    )
    return tuple(mixed.unbind(0))


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
@pytest.mark.parametrize("normalize_values", [False, True])
@pytest.mark.parametrize("query_count", [1, 3])
def test_zero_query_uniform_depth_mix(device_name, normalize_values, query_count):
    device = _device(device_name)
    sources = [
        torch.tensor([[1.0, 2.0], [0.0, 0.0]], device=device),
        torch.tensor([[6.0, -2.0], [0.0, 0.0]], device=device),
    ]
    query = torch.zeros((query_count, 2), dtype=torch.float32, device=device)
    result = attention_residual(
        [sources[0]],
        sources[1],
        query,
        partial_addend=None,
        normalize_values=normalize_values,
        eps=1e-6,
    )
    expected = torch.stack(sources).float()
    if normalize_values:
        expected = expected * torch.rsqrt(
            expected.square().mean(dim=-1, keepdim=True) + 1e-6
        )
    for row in result:
        torch.testing.assert_close(row, expected.mean(dim=0), rtol=2e-6, atol=2e-6)
        assert row.dtype == torch.float32
    assert len(result) == query_count


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_learned_queries_weight_sources_independently(device_name):
    device = _device(device_name)
    a = torch.tensor([[1.0, 0.0]], dtype=torch.float32, device=device)
    b = torch.tensor([[0.0, 1.0]], dtype=torch.float32, device=device)
    queries = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], device=device)
    output = attention_residual(
        [a], b, queries, partial_addend=None, normalize_values=False, eps=1e-6
    )
    logit = (1.0 / math.sqrt(0.5 + 1e-6)) / math.sqrt(2.0)
    p = math.exp(logit) / (1 + math.exp(logit))
    expected = torch.tensor([[[p, 1 - p]], [[1 - p, p]], [[0.5, 0.5]]], device=device)
    torch.testing.assert_close(torch.stack(output), expected, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_partial_addend_avoids_bf16_rounding_and_inputs_stay_unchanged(device_name):
    device = _device(device_name)
    partial = torch.ones((2, 4), dtype=torch.bfloat16, device=device)
    addend = torch.full_like(partial, 0.00390625)
    result = attention_residual(
        [],
        partial,
        torch.zeros(4, device=device),
        partial_addend=addend,
        normalize_values=False,
        eps=1e-6,
    )[0]
    torch.testing.assert_close(
        result, torch.full((2, 4), 1.00390625, device=device), rtol=0, atol=0
    )
    assert torch.equal(partial, torch.ones_like(partial))
    summed = residual_sum((partial * 256, partial, partial * -256))
    assert torch.equal(summed, partial)


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
@pytest.mark.parametrize("width", [16, 128, 4608])
def test_nine_source_multiquery_matches_cpu_equation(device_name, width):
    device = _device(device_name)
    generator = torch.Generator(device="cpu").manual_seed(2026)
    sources = torch.randn((9, 3, width), generator=generator, dtype=torch.float32).to(
        torch.bfloat16
    )
    queries = torch.randn((3, width), generator=generator, dtype=torch.float32)
    inverse = torch.rsqrt(sources.float().square().mean(dim=-1) + 1e-6)
    normalized = sources.float() * inverse.unsqueeze(-1)
    # Score the dot before applying its inverse RMS, as the reference does.
    logits = torch.einsum("sth,qh->qts", sources.float(), queries / math.sqrt(width))
    logits = logits * inverse.T.unsqueeze(0)
    expected = torch.einsum("qts,sth->qth", logits.softmax(dim=-1), normalized)
    moved = [value.to(device) for value in sources.unbind(0)]
    actual = attention_residual(
        moved[:-1],
        moved[-1],
        queries.to(device),
        partial_addend=None,
        normalize_values=True,
        eps=1e-6,
    )
    torch.testing.assert_close(
        torch.stack(actual).cpu(), expected, rtol=2e-5, atol=3e-6
    )


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_clip_preserves_strides_and_materialization_before_token_scale(device_name):
    device = _device(device_name)
    # On GPUs, the standalone kernel that pins the fused head norms bitwise.
    clip = l2_clip if device_name == "cpu" else triton_l2_clip
    generator = torch.Generator(device="cpu").manual_seed(81)
    packed = torch.randn((4, 3, 14), generator=generator).to(torch.bfloat16).to(device)
    x = packed[..., :7]
    scale = torch.linspace(0.5, 1.5, 21, device=device).reshape(3, 7)
    tau = torch.tensor([1.1, 1.2, 1.3, 1.4], device=device)
    value = x.float() * scale.float()
    clipped = value * (
        1.2 / (torch.linalg.vector_norm(value, dim=-1, keepdim=True) + 1e-6)
    ).clamp(max=1)
    expected = (clipped.to(torch.bfloat16).float() * tau[:, None, None]).to(
        torch.bfloat16
    )
    wrong = (clipped * tau[:, None, None]).to(torch.bfloat16)
    assert not torch.equal(expected, wrong)
    actual = clip(x, 1.2, 1e-6, scale, tau)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.is_contiguous()
    below = clip(torch.ones((2, 1, 4), device=device), 10.0, 1e-6, None, None)
    assert torch.equal(below, torch.ones_like(below))


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_rmsnorm_fp32_to_bf16_and_weightless_heads(device_name):
    device = _device(device_name)
    x = torch.arange(1, 25, dtype=torch.float32, device=device).reshape(2, 3, 4) / 7
    weight = torch.tensor([0.5, 1.0, 1.5, 2.0], dtype=torch.bfloat16, device=device)
    expected = (
        x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + 1e-6) * weight.float()
    )
    actual = rmsnorm_precision(
        x, weight, 1e-6, torch.bfloat16, mean_by_reciprocal=False
    )
    torch.testing.assert_close(actual, expected.to(torch.bfloat16), rtol=0, atol=0)
    value_norm = rmsnorm_precision(
        x, None, 1e-6, torch.bfloat16, mean_by_reciprocal=False
    )
    torch.testing.assert_close(
        value_norm,
        (x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + 1e-6)).to(
            torch.bfloat16
        ),
        rtol=0,
        atol=0,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an admitted GPU")
def test_rmsnorm_mean_rounds_by_the_requested_rule():
    # Rows of k ones have an exact sum of squares in any order; these k round
    # k / 4608 and k * fl32(1 / 4608) differently. An elementwise tensor
    # divisor divides; a scalar one may become a reciprocal.
    width = 4608
    counts = torch.arange(1, width + 1, device="cuda", dtype=torch.float32)
    divided = counts / torch.full_like(counts, width)
    multiplied = counts * (1.0 / width)
    picked = torch.nonzero(divided != multiplied).flatten()[:64]
    assert picked.numel() == 64
    x = (torch.arange(width, device="cuda")[None, :] < counts[picked][:, None]).float()
    outputs = []
    for by_reciprocal, mean in ((False, divided), (True, multiplied)):
        actual = rmsnorm_precision(
            x, None, 1e-6, torch.float32, mean_by_reciprocal=by_reciprocal
        )
        expected = x * torch.rsqrt(mean[picked][:, None] + 1e-6)
        assert torch.equal(actual, expected)
        outputs.append(actual)
    assert not torch.equal(*outputs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an admitted GPU")
def test_rmsnorm_reads_strided_head_rows_in_place():
    """Per-head rows of a column slice normalize exactly as their contiguous
    copy, into a contiguous output, without recompiling per token count."""
    from tokenspeed_kernel.ops.layernorm._triton.precision import (
        _rmsnorm_precision_kernel,
    )
    from utils import assert_no_triton_compile

    weight = torch.randn(128, device="cuda").bfloat16()

    def run(tokens):
        # [K | V] projection rows; the norm reads the K half's two heads.
        rows = torch.randn(tokens, 512, device="cuda").bfloat16()
        heads = rows[:, :256].unflatten(-1, (2, 128))
        actual = rmsnorm_precision(
            heads, weight, 1e-6, torch.bfloat16, mean_by_reciprocal=False
        )
        expected = rmsnorm_precision(
            heads.contiguous(),
            weight,
            1e-6,
            torch.bfloat16,
            mean_by_reciprocal=False,
        )
        assert actual.is_contiguous() and actual.shape == heads.shape
        assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))

    run(1)
    run(16)
    with assert_no_triton_compile(_rmsnorm_precision_kernel):
        for tokens in (2, 3, 8, 13):
            run(tokens)


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_empty_tokens(device_name):
    device = _device(device_name)
    x = torch.empty((0, 4), dtype=torch.bfloat16, device=device)
    mixture = attention_residual(
        [],
        x,
        torch.zeros((3, 4), device=device),
        partial_addend=None,
        normalize_values=True,
        eps=1e-6,
    )
    assert len(mixture) == 3 and mixture[0].shape == (0, 4)
    assert residual_sum((x, x)).shape == x.shape
    assert (
        rmsnorm_precision(
            x.float(), None, 1e-6, torch.bfloat16, mean_by_reciprocal=False
        ).shape
        == x.shape
    )
    assert l2_clip(x.view(0, 1, 4), 1.0, 1e-6, None, None).shape == (0, 1, 4)


def test_invalid_contracts_fail_before_execution_cpu():
    x = torch.ones((1, 4), dtype=torch.bfloat16)
    weights = torch.zeros((1, 4))
    with pytest.raises(ValueError, match="FP32 \\[Q, H\\] weights"):
        attention_residual_partials(
            x.view(1, 1, 4), torch.zeros((1, 3)), normalize_values=True, eps=1e-6
        )
    partials = attention_residual_partials(
        x.new_empty((0, 1, 4)), weights, normalize_values=True, eps=1e-6
    )
    combine = dict(
        candidate_out=None,
        normalize_values=True,
        eps=1e-6,
        norm_weights=None,
        norm_eps=1e-6,
    )
    with pytest.raises(ValueError, match="one to three terms"):
        attention_residual_combine(
            (), weights, partials, output_dtype=torch.float32, **combine
        )
    with pytest.raises(ValueError, match="unnormalized mixture"):
        attention_residual_combine(
            (x,), weights, partials, output_dtype=torch.bfloat16, **combine
        )
    with pytest.raises(ValueError, match="contiguous"):
        attention_residual_combine(
            (x,),
            torch.zeros((1, 8))[:, ::2],
            partials,
            output_dtype=torch.float32,
            **combine,
        )
    with pytest.raises(ValueError, match="epsilon"):
        rmsnorm_precision(x, None, -1.0, torch.bfloat16, mean_by_reciprocal=False)
    with pytest.raises(ValueError, match="scale"):
        l2_clip(x.view(1, 1, 4), 1.0, 1e-6, None, torch.ones(2))


def test_changed_input_cuda_graph_replay():
    device = _device("cuda")
    first = torch.ones((4, 16), dtype=torch.bfloat16, device=device)
    partial = torch.full_like(first, 2.0)
    query = torch.zeros((3, 16), device=device)

    def forward():
        mixed = attention_residual(
            [first],
            partial,
            query,
            partial_addend=None,
            normalize_values=True,
            eps=1e-6,
        )[0]
        return rmsnorm_precision(
            mixed, None, 1e-6, torch.bfloat16, mean_by_reciprocal=False
        )

    for _ in range(3):
        forward()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = forward()
    before = actual.clone()
    replacement = (
        torch.arange(1, 65, dtype=torch.float32, device=device)
        .reshape(4, 16)
        .to(torch.bfloat16)
    )
    first.copy_(replacement)
    graph.replay()
    torch.cuda.synchronize()
    expected_mixed = attention_residual(
        [first.cpu()],
        partial.cpu(),
        query.cpu(),
        partial_addend=None,
        normalize_values=True,
        eps=1e-6,
    )[0]
    expected = rmsnorm_precision(
        expected_mixed, None, 1e-6, torch.bfloat16, mean_by_reciprocal=False
    )
    assert not torch.equal(actual, before)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an admitted GPU")
def test_weighted_sum_matches_the_source_order_fp32_loop_bitwise():
    from tokenspeed_kernel.ops.residual.attention import residual_weighted_sum

    generator = torch.Generator().manual_seed(5)
    values = (torch.randn((19, 8, 4608), generator=generator) * 30).to(torch.bfloat16)
    weights = torch.randn(8, generator=generator).to(torch.bfloat16)
    gpu_values, gpu_weights = values.cuda(), weights.cuda()
    expected = torch.zeros((19, 4608), device="cuda")
    for source in range(8):
        expected = (
            expected + gpu_values[:, source].float() * gpu_weights[source].float()
        )
    assert torch.equal(residual_weighted_sum(gpu_values, gpu_weights), expected)
    assert torch.equal(residual_weighted_sum(values, weights), expected.cpu())


def _split_inputs(device, tokens, sources, queries, width=4608):
    generator = torch.Generator(device="cpu").manual_seed(tokens * 31 + queries)
    blocks = (torch.randn((sources, tokens, width), generator=generator) * 3).to(
        torch.bfloat16
    )
    terms = tuple(
        (torch.randn((tokens, width), generator=generator) * (i + 1)).to(torch.bfloat16)
        for i in range(3)
    )
    weights = torch.randn((queries, width), generator=generator) * width**-0.5
    gains = (torch.rand((queries, width), generator=generator) + 0.5).to(torch.bfloat16)
    return (x.to(device) for x in (blocks, *terms, weights, gains))


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
@pytest.mark.parametrize("candidate", ["stored", "addend"])
def test_split_mix_matches_the_equation_with_fused_norm(device_name, candidate):
    device = _device(device_name)
    blocks, t0, t1, t2, weights, gains = _split_inputs(device, 5, 4, 3)
    partials = attention_residual_partials(
        blocks, weights, normalize_values=True, eps=1e-6
    )
    stored = torch.empty_like(t0) if candidate == "stored" else None
    terms = (t0, t1, t2) if candidate == "stored" else (t0, t1)
    mixed = attention_residual_combine(
        terms,
        weights,
        partials,
        candidate_out=stored,
        normalize_values=True,
        eps=1e-6,
        norm_weights=None,
        norm_eps=1e-6,
        output_dtype=torch.float32,
    )
    normed = attention_residual_combine(
        terms,
        weights,
        partials,
        candidate_out=stored,
        normalize_values=True,
        eps=1e-6,
        norm_weights=gains,
        norm_eps=1e-6,
        output_dtype=torch.bfloat16,
    )
    if stored is not None:
        # The stored candidate is residual_sum's single rounding.
        assert torch.equal(stored, residual_sum(terms))
        partial, addend = stored, None
    else:
        partial, addend = t0, t1
    # Score weights are queries already scaled by width**-0.5.
    queries = weights * blocks.shape[-1] ** 0.5
    expected = torch.stack(
        attention_residual(
            list(blocks.unbind(0)),
            partial,
            queries,
            partial_addend=addend,
            normalize_values=True,
            eps=1e-6,
        )
    )
    torch.testing.assert_close(mixed, expected, rtol=2e-5, atol=3e-6)
    expected_normed = torch.stack(
        [
            rmsnorm_precision(row, gain, 1e-6, torch.bfloat16, mean_by_reciprocal=False)
            for row, gain in zip(mixed, gains)
        ]
    )
    torch.testing.assert_close(
        normed.float(), expected_normed.float(), rtol=8e-3, atol=0
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an admitted GPU")
def test_a_query_partial_does_not_depend_on_its_launch():
    blocks, *_, weights, _ = _split_inputs(torch.device("cuda"), 7, 6, 14)
    together = attention_residual_partials(
        blocks, weights, normalize_values=True, eps=1e-6
    )
    for query in (0, 5, 13):
        alone = attention_residual_partials(
            blocks, weights[query : query + 1], normalize_values=True, eps=1e-6
        )
        for x, y in zip(together, alone):
            assert torch.equal(x[query], y[0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an admitted GPU")
def test_split_mix_token_counts_do_not_recompile():
    from tokenspeed_kernel.ops.activation.triton import (
        _attnres_combine_kernel,
        _attnres_partial_kernel,
    )
    from utils import assert_no_triton_compile

    def run(tokens):
        blocks, t0, t1, _, weights, gains = _split_inputs(
            torch.device("cuda"), tokens, 3, 1
        )
        partials = attention_residual_partials(
            blocks, weights, normalize_values=True, eps=1e-6
        )
        attention_residual_combine(
            (t0, t1),
            weights,
            partials,
            candidate_out=torch.empty_like(t0),
            normalize_values=True,
            eps=1e-6,
            norm_weights=gains,
            norm_eps=1e-6,
            output_dtype=torch.bfloat16,
        )

    run(1)
    run(16)
    with assert_no_triton_compile(_attnres_partial_kernel, _attnres_combine_kernel):
        for tokens in (2, 3, 8, 13, 32, 57):
            run(tokens)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an admitted GPU")
@pytest.mark.parametrize("candidate", ["stored", "addend"])
@pytest.mark.parametrize(
    "width,sources,queries,normalize_values",
    [(4608, 4, 3, True), (4608, 9, 1, False), (64, 0, 1, True), (100, 2, 2, True)],
)
def test_one_pass_mix_matches_the_split_equations(
    candidate, width, sources, queries, normalize_values
):
    blocks, t0, t1, t2, weights, gains = _split_inputs(
        torch.device("cuda"), 5, sources, queries, width
    )
    terms = (t0, t1, t2) if candidate == "stored" else (t0, t1)
    for norm_weights, dtype in ((None, torch.float32), (gains, torch.bfloat16)):
        results = []
        for device in ("cuda", "cpu"):
            stored = (
                torch.empty_like(t0, device=device) if candidate == "stored" else None
            )
            mixed = attention_residual_mix(
                blocks.to(device),
                tuple(x.to(device) for x in terms),
                weights.to(device),
                candidate_out=stored,
                normalize_values=normalize_values,
                eps=1e-6,
                norm_weights=None if norm_weights is None else norm_weights.to(device),
                norm_eps=1e-6,
                output_dtype=dtype,
                tokens=None,
                out=None,
            )
            results.append((mixed.cpu(), None if stored is None else stored.cpu()))
        (actual, stored), (expected, expected_stored) = results
        if candidate == "stored":
            # The stored candidate is residual_sum's single rounding.
            assert torch.equal(stored, expected_stored)
        if dtype == torch.float32:
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=3e-6)
        else:
            torch.testing.assert_close(
                actual.float(), expected.float(), rtol=8e-3, atol=0
            )


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_a_token_range_mixes_like_the_whole_batch(device_name):
    """A sequence-parallel rank mixes its share of the tokens into a shared
    destination: those rows (and their stored candidate rows) carry the bits
    of the whole batch's mix; the other rows are not written."""
    device = _device(device_name)
    blocks, t0, t1, t2, weights, gains = _split_inputs(device, 11, 3, 3)
    args = dict(
        normalize_values=True,
        eps=1e-6,
        norm_weights=gains,
        norm_eps=1e-6,
        output_dtype=torch.bfloat16,
    )
    whole_candidate = torch.empty_like(t0)
    whole = attention_residual_mix(
        blocks,
        (t0, t1, t2),
        weights,
        candidate_out=whole_candidate,
        tokens=None,
        out=None,
        **args,
    )
    out = torch.full_like(whole, float("nan"))
    candidate = torch.full_like(t0, float("nan"))
    result = attention_residual_mix(
        blocks,
        (t0, t1, t2),
        weights,
        candidate_out=candidate,
        tokens=(4, 5),
        out=out,
        **args,
    )
    assert result is out
    assert torch.equal(out[:, 4:9], whole[:, 4:9])
    assert torch.equal(candidate[4:9], whole_candidate[4:9])
    assert out[:, :4].isnan().all() and out[:, 9:].isnan().all()
    assert candidate[:4].isnan().all() and candidate[9:].isnan().all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an admitted GPU")
def test_one_pass_mix_token_source_and_query_counts_do_not_recompile():
    from tokenspeed_kernel.ops.residual._triton.attention import _attnres_mix_kernel
    from utils import assert_no_triton_compile

    def run(tokens, sources, queries):
        blocks, t0, t1, _, weights, gains = _split_inputs(
            torch.device("cuda"), tokens, sources, queries
        )
        attention_residual_mix(
            blocks,
            (t0, t1),
            weights,
            candidate_out=torch.empty_like(t0),
            normalize_values=True,
            eps=1e-6,
            norm_weights=gains,
            norm_eps=1e-6,
            output_dtype=torch.bfloat16,
            tokens=(tokens // 2, tokens - tokens // 2),
            out=None,
        )

    run(1, 1, 1)
    run(16, 3, 1)
    with assert_no_triton_compile(_attnres_mix_kernel):
        for tokens, sources, queries in (
            (2, 0, 3),
            (3, 2, 1),
            (13, 8, 3),
            (57, 5, 2),
            (300, 1, 16),
        ):
            run(tokens, sources, queries)


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
@pytest.mark.parametrize("width", [2048, 4608])
@pytest.mark.parametrize("mean_by_reciprocal", [False, True])
def test_add_rmsnorm_precision_is_the_sum_then_the_norm(
    device_name, width, mean_by_reciprocal
):
    """One pass over the FP32 sum gives the separate statements' bits: the
    normalized rows and the sum stored once in BF16."""
    device = _device(device_name)
    generator = torch.Generator(device=device).manual_seed(width)
    x = (torch.randn((7, width), generator=generator, device=device) * 3).bfloat16()
    residual = (
        torch.randn((7, width), generator=generator, device=device) * 30
    ).bfloat16()
    weight = (torch.rand(width, generator=generator, device=device) + 0.5).bfloat16()
    summed = x.float() + residual.float()
    normalized, stored = add_rmsnorm_precision(
        x,
        residual,
        weight,
        1e-6,
        torch.bfloat16,
        torch.bfloat16,
        mean_by_reciprocal=mean_by_reciprocal,
    )
    expected = rmsnorm_precision(
        summed,
        weight,
        1e-6,
        torch.bfloat16,
        mean_by_reciprocal=mean_by_reciprocal,
    )
    assert torch.equal(normalized.view(torch.int16), expected.view(torch.int16))
    assert torch.equal(stored, summed.to(torch.bfloat16))
