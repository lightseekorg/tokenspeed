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

"""CuTe DSL cluster kernels for a gated boundary and routed normalization."""

from __future__ import annotations

import pytest
import torch
from fused_boundary_helpers import (
    armed_mailbox,
    assert_child_keeps_waiting,
    boundary_case,
)
from tokenspeed_kernel.numerics.torch.residual import (
    torch_post_norm_residual_pre_norm,
)
from tokenspeed_kernel.ops.moe import (
    routed_input_scale_rmsnorm_sum_gathered,
)
from tokenspeed_kernel.ops.moe._cute_dsl import routed_norm as routed_cute_dsl
from tokenspeed_kernel.ops.moe._cute_dsl.latent_down import HALF_POLLED_SENTINEL
from tokenspeed_kernel.ops.moe._triton.routed_norm import (
    triton_routed_input_scale_rmsnorm_sum_gathered as triton_routed_sum_gathered,
)
from tokenspeed_kernel.ops.residual import (
    boundary_gathers_mailbox,
    post_norm_residual_pre_norm,
)
from tokenspeed_kernel.ops.residual._cute_dsl import gated_boundary as boundary_cute_dsl
from tokenspeed_kernel.ops.residual._triton.gated_boundary import (
    triton_post_norm_residual_pre_norm as triton_boundary,
)
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

pytestmark = pytest.mark.skipif(
    not (boundary_cute_dsl.is_available() and routed_cute_dsl.is_available()),
    reason="needs the CuTe DSL cluster kernels",
)

WIDTH = 8192
MAX_ROWS = boundary_cute_dsl.CLUSTER_BOUNDARY_MAX_ROWS


def _supported(
    args: tuple, branch: torch.Tensor, sentinel: int | None, ranks: int
) -> bool:
    stream, _, post_weight, _, _, alpha, beta, pre_weight = args[:8]
    return boundary_cute_dsl.cluster_boundary_supported(
        stream, branch, post_weight, alpha, beta, pre_weight, sentinel, ranks
    )


def _routed_case(tokens: int, topk: int, device: str, padded_width: int) -> tuple:
    torch.manual_seed(23)
    rows = tokens * topk + 16
    gemm2_out = (torch.randn(rows, padded_width, device=device) * 0.5).to(
        torch.bfloat16
    )
    permute_map = torch.randperm(rows, device=device)[: tokens * topk].to(torch.int32)
    # A route another rank owns, on the last route of the last token.
    permute_map[-1] = -1
    if tokens > 1:
        # An entire token no rank owns: every inverse RMS is rsqrt(eps).
        permute_map[topk : 2 * topk] = -1
    route_weights = torch.rand(tokens, topk, device=device)
    norm_weight = torch.randn(4096, device=device, dtype=torch.bfloat16) * 0.1
    return gemm2_out, permute_map, tokens, topk, route_weights, norm_weight, 1e-5, 1.0


def _assert_agree(got: torch.Tensor, twin: torch.Tensor) -> None:
    # A variance ULP moves a whole row; a post-norm rounding flip moves single channels.
    torch.testing.assert_close(got.float(), twin.float(), rtol=8e-3, atol=8e-3)
    relative = (got.float() - twin.float()).abs() / twin.float().abs().clamp_min(1e-2)
    assert (relative > 1e-3).float().mean().item() < 1e-3


@pytest.mark.parametrize(
    "family, mode, signature",
    [
        (
            "residual",
            "post_norm_residual_pre_norm",
            format_signature(
                stream=dense_tensor_format(torch.float32),
                branch=dense_tensor_format(torch.bfloat16),
            ),
        ),
        (
            "moe",
            "routed_input_scale_rmsnorm_sum_gathered",
            format_signature(gemm2_out=dense_tensor_format(torch.bfloat16)),
        ),
    ],
)
def test_registry_routes_cluster_batches_to_cute_dsl(family, mode, signature) -> None:
    """The cluster trait picks the CuTe DSL kernel; every other batch gets Triton."""
    for supported, solution in ((True, "cute_dsl"), (False, "triton")):
        kernel = select_kernel(
            family, mode, signature, traits={"cluster_supported": supported}
        )
        assert kernel.name == f"{solution}_{mode}"


@pytest.mark.parametrize("device", ["cuda"])
@pytest.mark.parametrize("rows", [1, 7, 64])
@pytest.mark.parametrize("boundary", ["attention", "ffn", "ffn_linked"])
def test_cluster_boundary_agrees_with_triton_up_to_variance_rounding(
    device: str, rows: int, boundary: str
) -> None:
    """Same math as the Triton kernel; only the variance sum order differs."""
    args = boundary_case(boundary, rows, device)
    assert _supported(args, args[1], None, 1)
    cluster = boundary_cute_dsl.cute_dsl_post_norm_residual_pre_norm(
        *args, False, None, 1
    )
    pdl = boundary_cute_dsl.cute_dsl_post_norm_residual_pre_norm(*args, True, None, 1)
    for got, twin in zip(cluster, pdl, strict=True):
        assert (got is None and twin is None) or torch.equal(got, twin)
    triton = triton_boundary(*args, False, None, 1)
    for got, twin, ref in zip(
        cluster[:2],
        triton[:2],
        torch_post_norm_residual_pre_norm(*args, False, None, 1)[:2],
    ):
        if got is None:
            assert twin is None and ref is None
            continue
        # A BF16 round trip of the post-norm would show as ~2^-9; FP32 stays within 2^-20.
        torch.testing.assert_close(got.double(), ref, rtol=2.0**-20, atol=1e-6)
        _assert_agree(got, twin)
    if cluster[2] is not None:
        assert torch.equal(cluster[2], cluster[1].to(torch.bfloat16))
    # The dispatcher picks the cluster kernel here.
    assert torch.equal(
        post_norm_residual_pre_norm(*args, False, None, 1)[0], cluster[0]
    )


@pytest.mark.parametrize("device", ["cuda"])
@pytest.mark.parametrize("topk", [4, 8])
@pytest.mark.parametrize("tokens", [1, 6, 64])
def test_cluster_routed_sum_agrees_with_triton_up_to_variance_rounding(
    device: str, topk: int, tokens: int
) -> None:
    """One cluster per token agrees with the two Triton launches up to variance rounding."""
    args = _routed_case(tokens, topk, device, 4096 + 128)
    assert routed_cute_dsl.cluster_routed_sum_supported(*args[:6])
    cluster = routed_cute_dsl.cute_dsl_routed_input_scale_rmsnorm_sum_gathered(
        *args, False, None
    )
    assert torch.equal(
        cluster,
        routed_cute_dsl.cute_dsl_routed_input_scale_rmsnorm_sum_gathered(
            *args, True, None
        ),
    )
    _assert_agree(cluster, triton_routed_sum_gathered(*args, False, None))
    if tokens > 1:
        assert cluster[1].eq(0).all()
    assert torch.equal(
        routed_input_scale_rmsnorm_sum_gathered(*args, False, None), cluster
    )


@pytest.mark.parametrize("device", ["cuda"])
def test_cluster_kernels_decline_what_they_cannot_serve(device: str) -> None:
    """Declined shapes run the Triton kernels bit for bit."""
    # 2048 channels do not split into 16 CTAs of 256 threads.
    alpha, beta = torch.ones(2048, device=device), torch.zeros(2048, device=device)
    stream = torch.randn(2, 2048, device=device)
    branch = torch.randn(2, 2048, device=device, dtype=torch.bfloat16)
    narrow = (stream, branch, None, 1e-5, 0.0, alpha, beta, None, 0.0, 0.0)
    assert not _supported(narrow, branch, None, 1)
    assert torch.equal(
        post_norm_residual_pre_norm(*narrow, False, None, 1)[0],
        triton_boundary(*narrow, False, None, 1)[0],
    )
    wide = boundary_case("ffn_linked", MAX_ROWS + 1, device)
    assert not _supported(wide, wide[1], None, 1)
    assert torch.equal(
        post_norm_residual_pre_norm(*wide, False, None, 1)[0],
        triton_boundary(*wide, False, None, 1)[0],
    )
    # A misaligned alpha view declines too.
    shifted = torch.ones(WIDTH + 1, device=device)[1:]
    assert not _supported((*wide[:5], shifted, *wide[6:]), wide[1], None, 1)

    routed = _routed_case(2, 8, device, 4096 + 128)
    narrow_weight = torch.randn(1024, device=device, dtype=torch.bfloat16)
    assert not routed_cute_dsl.cluster_routed_sum_supported(*routed[:5], narrow_weight)
    # Non-contiguous route weights decline rather than being copied silently.
    strided = torch.rand(2, 16, device=device)[:, ::2]
    assert not routed_cute_dsl.cluster_routed_sum_supported(
        *routed[:4], strided, routed[5]
    )
    # A padded row pitch that breaks 16-byte row alignment, and a batch past the ceiling.
    odd = _routed_case(2, 8, device, 4096 + 4)
    tokens = routed_cute_dsl.CLUSTER_ROUTED_SUM_MAX_TOKENS + 1
    past = (
        routed[0],
        torch.zeros(tokens * 8, device=device, dtype=torch.int32),
        tokens,
        8,
        torch.rand(tokens, 8, device=device),
        *routed[5:],
    )
    for args in (odd, past):
        assert not routed_cute_dsl.cluster_routed_sum_supported(*args[:6])
        assert torch.equal(
            routed_input_scale_rmsnorm_sum_gathered(*args, False, None),
            triton_routed_sum_gathered(*args, False, None),
        )


@pytest.mark.parametrize("device", ["cuda"])
def test_cluster_kernels_replay_from_a_cuda_graph(device: str) -> None:
    """The production path captures both kernels; replay must equal eager."""
    boundary_args = (*boundary_case("ffn_linked", 4, device), True, None, 1)
    routed_args = (*_routed_case(4, 8, device, 4096 + 128), True, None)
    eager = (
        boundary_cute_dsl.cute_dsl_post_norm_residual_pre_norm(*boundary_args)[
            0
        ].clone(),
        routed_cute_dsl.cute_dsl_routed_input_scale_rmsnorm_sum_gathered(
            *routed_args
        ).clone(),
    )
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        boundary_cute_dsl.cute_dsl_post_norm_residual_pre_norm(*boundary_args)
        routed_cute_dsl.cute_dsl_routed_input_scale_rmsnorm_sum_gathered(*routed_args)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side):
        captured = (
            boundary_cute_dsl.cute_dsl_post_norm_residual_pre_norm(*boundary_args)[0],
            routed_cute_dsl.cute_dsl_routed_input_scale_rmsnorm_sum_gathered(
                *routed_args
            ),
        )
    for out in captured:
        out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured[0], eager[0]) and torch.equal(captured[1], eager[1])


@pytest.mark.parametrize("device", ["cuda"])
@pytest.mark.parametrize("rows", [1, 5, 64])
def test_cluster_boundary_gathers_an_armed_mailbox_and_re_arms_it(
    device: str, rows: int
) -> None:
    """A mailbox branch gives the plain kernel's bits; the rows read are re-armed."""
    args = boundary_case("ffn_linked", rows, device)
    stream, branch = args[:2]
    plain = boundary_cute_dsl.cute_dsl_post_norm_residual_pre_norm(*args, True, None, 1)
    mailbox = armed_mailbox(branch, MAX_ROWS)
    assert _supported(args, mailbox, HALF_POLLED_SENTINEL, 1)
    got = post_norm_residual_pre_norm(
        stream, mailbox, *args[2:], True, HALF_POLLED_SENTINEL, 1
    )
    for ours, theirs in zip(got, plain, strict=True):
        assert torch.equal(ours, theirs)
    assert (mailbox.view(torch.int32) == -1).all()
    # A short mailbox declines; past the ceiling is refused, never handed to Triton.
    assert not _supported(args, mailbox[:0], HALF_POLLED_SENTINEL, 1)
    wide = torch.cat([stream] * 65)[: MAX_ROWS + 1]
    wide_mailbox = armed_mailbox(torch.cat([branch] * 65)[: MAX_ROWS + 1], MAX_ROWS + 1)
    with pytest.raises(ValueError, match="cluster boundary"):
        post_norm_residual_pre_norm(
            wide, wide_mailbox, *args[2:], True, HALF_POLLED_SENTINEL, 1
        )
    # The producer asks before publishing; 4096 channels leave no whole word per thread.
    assert boundary_gathers_mailbox(MAX_ROWS, WIDTH, 1)
    assert not boundary_gathers_mailbox(MAX_ROWS + 1, WIDTH, 1)
    assert not boundary_gathers_mailbox(1, 4096, 1)


@pytest.mark.parametrize("device", ["cuda"])
@pytest.mark.parametrize("rows", [1, 5, MAX_ROWS])
def test_cluster_boundary_reduces_one_partial_per_rank_like_the_oneshot_allreduce(
    device: str, rows: int
) -> None:
    """Four partials side by side give the plain kernel's bits for their rank-order BF16 sum."""
    ranks = 4
    args = boundary_case("attention", rows, device)
    stream = args[0]
    torch.manual_seed(29)
    partials = [
        (torch.randn(rows, WIDTH, device=device) * 0.5).to(torch.bfloat16)
        for _ in range(ranks)
    ]
    partials[1][:, ::3] = -0.0
    # The one-shot all-reduce strips -0.0, then adds in BF16 in rank order.
    sanitized = [torch.where(p == 0, torch.zeros_like(p), p) for p in partials]
    summed = sanitized[0]
    for partial in sanitized[1:]:
        summed = summed + partial
    plain = boundary_cute_dsl.cute_dsl_post_norm_residual_pre_norm(
        stream, summed, *args[2:], True, None, 1
    )
    mailbox = armed_mailbox(torch.cat(partials, dim=1), MAX_ROWS)
    assert _supported(args, mailbox, HALF_POLLED_SENTINEL, ranks)
    got = post_norm_residual_pre_norm(
        stream, mailbox, *args[2:], True, HALF_POLLED_SENTINEL, ranks
    )
    for ours, theirs in zip(got, plain, strict=True):
        assert torch.equal(ours, theirs)
    assert (mailbox.view(torch.int32) == -1).all()
    # A mailbox holding one partial per row cannot be read as four.
    narrow = armed_mailbox(partials[0], MAX_ROWS)
    assert not _supported(args, narrow, HALF_POLLED_SENTINEL, ranks)
    with pytest.raises(ValueError, match="cluster boundary"):
        post_norm_residual_pre_norm(
            stream, narrow, *args[2:], True, HALF_POLLED_SENTINEL, ranks
        )
    assert boundary_gathers_mailbox(MAX_ROWS, WIDTH, ranks)
    assert not boundary_gathers_mailbox(1, WIDTH, 0)


_BOUNDARY_WAITER = """
from tokenspeed_kernel.ops.residual import residual_gate_coefficients
from tokenspeed_kernel.ops.residual._cute_dsl import gated_boundary
width = 8192
stream = torch.ones(1, width, device=device)
row = torch.full((1, width), 0.5, dtype=torch.bfloat16, device=device)
alpha, beta = residual_gate_coefficients(torch.ones(width, device=device), 0.3, 1e-3)
weight = torch.ones(width, dtype=torch.bfloat16, device=device)
mailbox = torch.empty(2, width, dtype=torch.bfloat16, device=device)
tail = (weight, 1e-8, 1.0, alpha, beta, weight, 1e-5, 1.0, True, HALF_POLLED_SENTINEL, 1)
def consume():
    gated_boundary.cute_dsl_post_norm_residual_pre_norm(stream, mailbox, *tail)
"""


@pytest.mark.parametrize("word, empty", [(3333, 0xFFFF), (4095, -65536)])
def test_cluster_boundary_waits_for_a_word_still_holding_a_sentinel_half(
    word: int, empty: int
) -> None:
    """A word with either BF16 half still empty stalls the whole row."""
    assert_child_keeps_waiting(_BOUNDARY_WAITER, word, empty)
