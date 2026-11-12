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

"""gfx950 decode projection fused with the two AttnRes history reductions."""

from __future__ import annotations

import pytest
import torch
from utils import assert_no_triton_compile, is_cdna4

if not is_cdna4():
    pytest.skip(
        "AMD CDNA4 is required for the gfx950 AttnRes projection tests",
        allow_module_level=True,
    )

from tokenspeed_kernel.ops.activation.triton import attnres_partial_dual  # noqa: E402
from tokenspeed_kernel.ops.gemm import linear_attnres_partials_available  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.gemv import (  # noqa: E402
    launch_gluon_gemv_a16w16_gfx950,
)
from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.linear_attnres_partials_gfx950 import (  # noqa: E402
    gluon_linear_attnres_partials_gemv_gfx950,
    gluon_linear_attnres_partials_gfx950,
    gluon_linear_attnres_partials_m4_gfx950,
    launch_gluon_linear_attnres_partials_gfx950,
)

_HIDDEN = 7168
_EPS = 1e-6


def _inputs(tokens: int, output_size: int, num_blocks: int, seed: int):
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape, scale=1.0):
        return (
            torch.randn(*shape, device="cuda", generator=generator) * scale
        ).bfloat16()

    hidden = randn(tokens, _HIDDEN)
    weight = randn(output_size, _HIDDEN, scale=0.02)
    blocks = randn(num_blocks, tokens, _HIDDEN)
    scores = (randn(_HIDDEN, scale=0.05), randn(_HIDDEN, scale=0.05))
    return hidden, weight, blocks, scores


def _scratch(tokens: int):
    return tuple(
        (
            torch.empty(tokens, device="cuda"),
            torch.empty(tokens, device="cuda"),
            torch.empty(tokens, _HIDDEN, device="cuda"),
        )
        for _ in range(2)
    )


def _run(hidden, weight, blocks, scores, scratch, out):
    return launch_gluon_linear_attnres_partials_gfx950(
        hidden, weight, blocks, *scores, *scratch, eps=_EPS, out=out
    )


def _check(hidden, weight, blocks, scores, scratch, out):
    expected = (hidden.float() @ weight.float().T).bfloat16()
    torch.testing.assert_close(out, expected, rtol=2**-7, atol=2e-3)
    values = blocks.float()
    inverse_rms = torch.rsqrt(values.square().mean(dim=-1) + _EPS)
    for score, (maxima, sums, accumulators) in zip(scores, scratch, strict=True):
        logits = torch.einsum("bth,h->bt", values, score.float()) * inverse_rms
        expected_max = logits.max(dim=0).values
        weights = torch.exp(logits - expected_max)
        torch.testing.assert_close(maxima, expected_max, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(sums, weights.sum(dim=0), rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(
            accumulators,
            torch.einsum("bt,bth->th", weights, values),
            rtol=1e-4,
            atol=1e-4,
        )


@pytest.mark.parametrize("output_size", [3648, 6288])
@pytest.mark.parametrize("tokens", [1, 2, 4])
@pytest.mark.parametrize("num_blocks", [1, 8, 11])
def test_projection_and_attnres_partials_match_torch(
    output_size: int, tokens: int, num_blocks: int
) -> None:
    hidden, weight, blocks, scores = _inputs(
        tokens, output_size, num_blocks, seed=tokens
    )
    scratch = _scratch(tokens)
    out = torch.empty(tokens, output_size, device="cuda", dtype=torch.bfloat16)

    _run(hidden, weight, blocks, scores, scratch, out)

    _check(hidden, weight, blocks, scores, scratch, out)


@pytest.mark.parametrize("output_size", [3648, 6288])
def test_four_token_launch_replays_in_a_graph_across_block_counts(
    output_size: int,
) -> None:
    """The K3 verify window's M=4 kernel: one compile, graph-capturable."""
    tokens = 4
    hidden, weight, blocks, scores = _inputs(tokens, output_size, 8, seed=29)
    scratch = _scratch(tokens)
    out = torch.empty(tokens, output_size, device="cuda", dtype=torch.bfloat16)
    for num_blocks in (1, 2):
        _run(hidden, weight, blocks[:num_blocks], scores, scratch, out)

    with assert_no_triton_compile(
        gluon_linear_attnres_partials_m4_gfx950, gluon_linear_attnres_partials_gfx950
    ):
        for num_blocks in range(1, 9):
            _run(hidden, weight, blocks[:num_blocks], scores, scratch, out)
            _check(hidden, weight, blocks[:num_blocks], scores, scratch, out)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _run(hidden, weight, blocks, scores, scratch, out)
    hidden.copy_(torch.randn_like(hidden, dtype=torch.float32).bfloat16())
    blocks.copy_(torch.randn_like(blocks, dtype=torch.float32).bfloat16())
    graph.replay()
    torch.cuda.synchronize()

    _check(hidden, weight, blocks, scores, scratch, out)


def _unfused(hidden, weight, blocks, scores):
    """Today's 5-16 token path: the decode GEMV, then the Triton partials."""
    tokens = hidden.shape[0]
    scratch = _scratch(tokens)
    out = launch_gluon_gemv_a16w16_gfx950(
        hidden,
        weight,
        torch.empty(tokens, weight.shape[0], device="cuda", dtype=torch.bfloat16),
    )
    attnres_partial_dual(blocks, *scores, _EPS, *scratch)
    return out, scratch


def _assert_bitwise(out, scratch, expected_out, expected_scratch):
    assert torch.equal(out.view(torch.int16), expected_out.view(torch.int16))
    for actual, expected in zip(scratch, expected_scratch, strict=True):
        for a, e in zip(actual, expected, strict=True):
            assert torch.equal(a.view(torch.int32), e.view(torch.int32))


@pytest.mark.parametrize("output_size", [3648, 6288])
@pytest.mark.parametrize("tokens", [5, 8, 16])
@pytest.mark.parametrize("num_blocks", [1, 6, 11])
def test_gemv_tokens_match_unfused_projection_and_partials_bitwise(
    output_size: int, tokens: int, num_blocks: int
) -> None:
    hidden, weight, blocks, scores = _inputs(
        tokens, output_size, num_blocks, seed=tokens
    )
    scratch = _scratch(tokens)
    out = torch.empty(tokens, output_size, device="cuda", dtype=torch.bfloat16)

    _run(hidden, weight, blocks, scores, scratch, out)

    _check(hidden, weight, blocks, scores, scratch, out)
    _assert_bitwise(out, scratch, *_unfused(hidden, weight, blocks, scores))


def test_gemv_tokens_share_one_compile_and_replay_in_a_graph() -> None:
    """Token and block counts are runtime arguments of one kernel."""
    hidden, weight, blocks, scores = _inputs(16, 6288, 8, seed=31)
    outs = {
        tokens: torch.empty(tokens, 6288, device="cuda", dtype=torch.bfloat16)
        for tokens in (8, 16)
    }
    scratches = {tokens: _scratch(tokens) for tokens in outs}
    views = {
        tokens: (hidden[:tokens], blocks[:, :tokens].contiguous()) for tokens in outs
    }
    for num_blocks in (1, 2):
        _run(
            views[8][0], weight, views[8][1][:num_blocks], scores, scratches[8], outs[8]
        )

    with assert_no_triton_compile(gluon_linear_attnres_partials_gemv_gfx950):
        for tokens in (5, 12):
            _run(
                hidden[:tokens],
                weight,
                blocks[:3, :tokens].contiguous(),
                scores,
                _scratch(tokens),
                torch.empty(tokens, 6288, device="cuda", dtype=torch.bfloat16),
            )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for tokens, (view_hidden, view_blocks) in views.items():
                _run(
                    view_hidden,
                    weight,
                    view_blocks,
                    scores,
                    scratches[tokens],
                    outs[tokens],
                )
    for tokens, (view_hidden, view_blocks) in views.items():
        view_hidden.copy_(torch.randn_like(view_hidden, dtype=torch.float32).bfloat16())
        view_blocks.copy_(torch.randn_like(view_blocks, dtype=torch.float32).bfloat16())
    graph.replay()
    torch.cuda.synchronize()

    for tokens, (view_hidden, view_blocks) in views.items():
        _assert_bitwise(
            outs[tokens],
            scratches[tokens],
            *_unfused(view_hidden, weight, view_blocks, scores),
        )


@pytest.mark.parametrize("tokens", [8, 16])
def test_registry_fuses_the_mla_and_kda_projections_above_four_tokens(
    tokens: int,
) -> None:
    available = {}
    for output_size in (3648, 6288, 4096):
        hidden, weight, blocks, scores = _inputs(tokens, output_size, 2, seed=3)
        available[output_size] = linear_attnres_partials_available(
            hidden, weight, blocks, *scores, *_scratch(tokens), eps=_EPS
        )
    assert available == {3648: True, 6288: True, 4096: False}

    hidden, weight, blocks, scores = _inputs(tokens, 4096, 2, seed=3)
    out = torch.empty(tokens, 4096, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="output size in"):
        _run(hidden, weight, blocks, scores, _scratch(tokens), out)
