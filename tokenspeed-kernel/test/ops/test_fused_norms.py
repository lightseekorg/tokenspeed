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

import itertools

import pytest
import torch
from fused_boundary_helpers import boundary_case
from tokenspeed_kernel.numerics.torch.layernorm import (
    torch_affine_rmsnorm,
    torch_weightless_qk_rmsnorm,
    torch_weightless_rmsnorm,
    torch_weightless_rmsnorm_gated_sigmoid,
)
from tokenspeed_kernel.numerics.torch.residual import (
    torch_post_norm_residual_pre_norm,
)
from tokenspeed_kernel.ops.layernorm import (
    affine_rmsnorm,
    weightless_qk_rmsnorm,
    weightless_rmsnorm,
    weightless_rmsnorm_gated_sigmoid,
)
from tokenspeed_kernel.ops.moe import bf16_moe_routes
from tokenspeed_kernel.ops.moe._triton.routed_norm import (
    _gather_moe_routes_kernel,
    _launch_routed_input_scale_sum,
)
from tokenspeed_kernel.ops.moe._triton.routed_norm import (
    triton_routed_input_scale_rmsnorm_sum_gathered as triton_routed_sum_gathered,
)
from tokenspeed_kernel.ops.moe.routed_norm import (
    gather_moe_routes,
    routed_input_scale_rmsnorm_sum,
)
from tokenspeed_kernel.ops.residual import (
    post_norm_residual_pre_norm,
    post_norm_residual_pre_norm_bf16,
    residual_gate_coefficients,
)
from tokenspeed_kernel.ops.residual._triton.gated_boundary import (
    triton_post_norm_residual_pre_norm as triton_boundary,
)
from tokenspeed_kernel.platform import current_platform

pytestmark = pytest.mark.skipif(
    not (current_platform().is_amd or current_platform().is_nvidia),
    reason="Fused normalization tests require an AMD or NVIDIA GPU.",
)


def test_head_norms_match_reference(device: str) -> None:
    torch.manual_seed(11)
    tokens, query_heads, kv_heads, head_dim = 13, 7, 3, 48
    q_width, kv_width = query_heads * head_dim, kv_heads * head_dim
    packed = torch.randn(
        tokens, q_width + 2 * kv_width, dtype=torch.bfloat16, device=device
    )
    q, k, _ = packed.split((q_width, kv_width, kv_width), dim=-1)
    eps, q_scale = 1e-5, 43.7840518911 / head_dim**0.5
    q_out, k_out = weightless_qk_rmsnorm(q, k, head_dim, eps, q_scale, True)
    q_ref, k_ref = torch_weightless_qk_rmsnorm(q, k, head_dim, eps, q_scale, True)
    torch.testing.assert_close(q_out, q_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(k_out, k_ref, atol=2e-2, rtol=2e-2)
    x, gate = q.contiguous(), torch.randn_like(q)
    output = weightless_rmsnorm_gated_sigmoid(x, gate, query_heads, head_dim, eps, True)
    reference = torch_weightless_rmsnorm_gated_sigmoid(
        x, gate, query_heads, head_dim, eps, True
    )
    torch.testing.assert_close(output, reference, atol=2e-2, rtol=2e-2)


def test_rmsnorm_variants_match_reference(device: str) -> None:
    torch.manual_seed(12)
    x = torch.randn(7, 4096, dtype=torch.bfloat16, device=device)
    weight = torch.randn(4096, dtype=torch.bfloat16, device=device) * 0.03
    eps = 1e-8

    plain = weightless_rmsnorm(x, eps, x.dtype)
    affine = affine_rmsnorm(x, weight, eps, 1.0, x.dtype)
    fp32_affine = affine_rmsnorm(x.float(), weight, eps, 1.0, torch.float32)

    plain_ref = torch_weightless_rmsnorm(x, eps, x.dtype)
    affine_ref = torch_affine_rmsnorm(x, weight, eps, 1.0, x.dtype)
    torch.testing.assert_close(plain, plain_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(affine, affine_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(
        fp32_affine,
        torch_affine_rmsnorm(x.float(), weight, eps, 1.0, torch.float32),
        atol=1e-5,
        rtol=1e-5,
    )
    # The output dtype only picks the final rounding; the math stays FP32.
    narrowed = affine_rmsnorm(x.float(), weight, eps, 1.0, torch.bfloat16)
    assert torch.equal(narrowed, fp32_affine.to(torch.bfloat16))
    # The input dtype sets the load layout, so the variance sum order may differ.
    widened = weightless_rmsnorm(x, eps, torch.float32)
    torch.testing.assert_close(
        widened,
        weightless_rmsnorm(x.float(), eps, torch.float32),
        atol=0.0,
        rtol=2**-20,
    )


@pytest.mark.parametrize("device", ["cuda"])
@pytest.mark.parametrize("gate_dtype", [torch.float32, torch.bfloat16])
def test_gate_coefficients_feed_the_fused_residual(
    device: str, gate_dtype: torch.dtype
) -> None:
    """The boundary combines alpha * stream + beta * post-norm in FP32, within 2^-20 of FP64."""
    torch.manual_seed(13)
    width = 8192
    gate = (torch.randn(width, device=device) * 4.0).to(gate_dtype)
    stream = torch.randn(3, width, device=device) * 3.0
    branch = (torch.randn(3, width, device=device) * 0.5).to(torch.bfloat16)
    alpha, beta = residual_gate_coefficients(gate, 0.3, 1e-3)
    assert alpha.dtype == beta.dtype == torch.float32
    scaled_gate = gate.float() / 0.3
    beta_ref = scaled_gate.sigmoid()
    alpha_ref = ((-scaled_gate).sigmoid() * (1.0 + beta_ref)).clamp_min(1e-3).sqrt()
    torch.testing.assert_close(alpha, alpha_ref, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(beta, beta_ref, atol=1e-6, rtol=1e-5)
    post_norm = (stream, branch, None, 1e-5, 0.0)
    no_pre_norm = (None, 0.0, 0.0, False, None, 1)
    fused = triton_boundary(*post_norm, alpha, beta, *no_pre_norm)[0]
    # alpha=0, beta=1 hands back the kernel's own FP32 post-norm.
    ones = (torch.zeros_like(alpha), torch.ones_like(beta))
    post = triton_boundary(*post_norm, *ones, *no_pre_norm)[0]
    reference = alpha.double() * stream.double() + beta.double() * post.double()
    torch.testing.assert_close(fused.double(), reference, atol=1e-6, rtol=2.0**-20)
    with pytest.raises(ValueError, match="FP32 vector"):
        post_norm_residual_pre_norm(
            stream,
            branch,
            None,
            1e-5,
            0.0,
            alpha.half(),
            beta,
            None,
            0.0,
            0.0,
            False,
            None,
            1,
        )


def test_boundary_can_omit_router_output_during_replay(device: str) -> None:
    """Dropping the unused FP32 output preserves residual and BF16 next input."""
    torch.manual_seed(41)
    stream = torch.randn(17, 8192, device=device)
    branch = torch.randn_like(stream).bfloat16()
    post_weight = torch.randn(8192, device=device).bfloat16() * 0.1
    pre_weight = torch.randn_like(post_weight) * 0.1
    alpha, beta = residual_gate_coefficients(torch.randn_like(post_weight), 0.3, 1e-3)

    def run(return_pre_fp32: bool, has_pre: bool):
        boundary = (
            post_norm_residual_pre_norm
            if return_pre_fp32
            else post_norm_residual_pre_norm_bf16
        )
        return boundary(
            stream,
            branch,
            post_weight,
            1e-6,
            1.0,
            alpha,
            beta,
            pre_weight if has_pre else None,
            1e-6,
            1.0,
            False,
            None,
            1,
        )

    run(True, True)
    run(False, True)
    run(True, False)
    run(False, False)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run(False, True)
        final = run(False, False)
    assert final[1] is None
    for step in range(3):
        stream.add_((step + 1) / 16)
        branch.mul_(-0.5)
        expected = run(True, True)
        final_expected = run(True, False)
        graph.replay()
        torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
        torch.testing.assert_close(actual[1], expected[2], rtol=0, atol=0)
        torch.testing.assert_close(final[0], final_expected[0], rtol=0, atol=0)


@pytest.mark.parametrize("device", ["cuda"])
def test_bf16_moe_routes_beyond_int32_expert_stride(device: str) -> None:
    """A local expert past a 2^31 weight offset addresses correctly; remote experts stay zero."""
    experts, width, inter = 80, 4096, 4096
    assert experts * 2 * inter * width > 2**31

    torch.manual_seed(0)
    w13 = (torch.randn(experts, 2 * inter, width, device=device) * 0.02).to(
        torch.bfloat16
    )
    w2 = (torch.randn(experts, width, inter, device=device) * 0.02).to(torch.bfloat16)
    x = (torch.randn(2, width, device=device) * 0.5).to(torch.bfloat16)
    # Token 0 lands on the highest local expert; token 1 on experts below and above the shard.
    ids = torch.tensor(
        [[experts, experts], [0, experts + 1]], device=device, dtype=torch.int32
    )

    routes = bf16_moe_routes(x.contiguous(), w13, w2, ids, 1, batch_invariant=False)

    assert torch.equal(
        routes, bf16_moe_routes(x.contiguous(), w13, w2, ids, 1, batch_invariant=False)
    )
    assert not torch.isnan(routes.float()).any()
    assert not routes[1].any()
    top = experts - 1
    gate = w13[top, :inter].float() @ x[0].float()
    up = w13[top, inter:].float() @ x[0].float()
    expected = w2[top].float() @ (torch.nn.functional.silu(gate) * up)
    torch.testing.assert_close(routes[0, 0].float(), expected, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("device", ["cuda"])
def test_portable_kernels_capture_and_replay_with_fresh_inputs(device: str) -> None:
    """The Muse Spark Triton layer chain replays bitwise with fresh inputs."""
    torch.manual_seed(17)
    tokens, heads, head_dim, topk, experts, inter = 16, 8, 64, 4, 2, 64
    width = heads * head_dim
    bf16 = {"dtype": torch.bfloat16, "device": device}
    q, gate = torch.randn(tokens, width, **bf16), torch.randn(tokens, width, **bf16)
    k = torch.randn(tokens, width // 2, **bf16)
    residual = torch.randn(tokens, width, device=device)
    alpha, beta = residual_gate_coefficients(
        torch.randn(width, device=device), 0.3, 1e-3
    )
    norm_weight = torch.randn(width, **bf16) * 0.02
    w13 = torch.randn(experts, 2 * inter, width, **bf16) * 0.02
    w2 = torch.randn(experts, width, inter, **bf16) * 0.02
    topk_ids = torch.randint(0, experts + 2, (tokens, topk), device=device).int()
    route_weights = torch.rand(tokens, topk, device=device).softmax(-1)

    def run_chain() -> tuple[torch.Tensor, ...]:
        q_norm, k_norm = weightless_qk_rmsnorm(q, k, head_dim, 1e-5, 1.25, False)
        attention = weightless_rmsnorm_gated_sigmoid(
            q_norm, gate, heads, head_dim, 1e-5, False
        )
        stream, pre_fp32, pre = triton_boundary(
            residual,
            attention,
            None,
            1e-8,
            1.0,
            alpha,
            beta,
            norm_weight,
            1e-5,
            1.0,
            False,
            None,
            1,
        )
        routes = bf16_moe_routes(pre, w13, w2, topk_ids, 1, batch_invariant=False)
        combined = routed_input_scale_rmsnorm_sum(
            routes, route_weights, norm_weight, 1e-8, 1.0, False, None
        )
        return k_norm, stream, pre_fp32, combined

    for _ in range(3):
        run_chain()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run_chain()
    for tensor in (q, k, gate, residual):
        tensor.copy_(torch.randn_like(tensor))
    topk_ids.copy_(torch.randint_like(topk_ids, 0, experts + 2))
    graph.replay()
    expected = run_chain()
    torch.cuda.synchronize()
    for actual, reference in zip(captured, expected, strict=True):
        assert torch.equal(actual, reference)


@pytest.mark.parametrize("device", ["cuda"])
@pytest.mark.parametrize("enable_pdl", [False, True])
@pytest.mark.parametrize("boundary", ["attention", "ffn", "ffn_linked"])
def test_fused_boundary_keeps_the_post_norm_in_fp32(
    device: str, enable_pdl: bool, boundary: str
) -> None:
    """A BF16 round trip of the post-norm would show as ~2^-9; FP32 stays within 2^-20 of FP64."""
    args = boundary_case(boundary, 5, device)
    outputs = triton_boundary(*args, enable_pdl, None, 1)
    for got, plain in zip(
        outputs,
        triton_boundary(*args, False, None, 1),
        strict=True,
    ):
        assert (got is None and plain is None) or torch.equal(got, plain)
    new_stream, pre_fp32, pre_bf16 = outputs
    stream_ref, pre_ref, _ = torch_post_norm_residual_pre_norm(*args, False, None, 1)
    ulps = 2.0**-20
    torch.testing.assert_close(new_stream.double(), stream_ref, atol=1e-6, rtol=ulps)
    if pre_ref is None:
        assert pre_fp32 is None and pre_bf16 is None
        return
    torch.testing.assert_close(pre_fp32.double(), pre_ref, atol=1e-6, rtol=ulps)
    assert torch.equal(pre_bf16, pre_fp32.to(torch.bfloat16))


@pytest.mark.parametrize("device", ["cuda"])
def test_gathered_routes_replay_and_vary_token_count(device: str) -> None:
    """Route alignment and destination storage survive capture and batch changes."""
    from utils import assert_no_triton_compile

    topk, width, padded_width = 8, 257, 288

    def make_case(tokens: int):
        routes = torch.randn(
            tokens * topk + 7, padded_width, device=device, dtype=torch.bfloat16
        )
        mapping = torch.randperm(routes.shape[0], device=device)[: tokens * topk].int()
        mapping[::3] = -1
        output = torch.full(
            (tokens, topk, width), torch.nan, device=device, dtype=routes.dtype
        )
        return routes, mapping, tokens, topk, width, output

    def check(case):
        routes, mapping, tokens, topk, width, output = case
        actual = gather_moe_routes(*case)
        assert actual is output
        indices = mapping.long()
        expected = routes[indices.clamp_min(0), :width]
        expected[indices < 0] = 0
        torch.testing.assert_close(
            actual, expected.reshape(tokens, topk, width), atol=0, rtol=0
        )

    # Warm each runtime integer alignment class, including a partial workgroup.
    for tokens in (1, 2):
        check(make_case(tokens))
    with assert_no_triton_compile(_gather_moe_routes_kernel):
        for tokens in (3, 5, 16, 33):
            check(make_case(tokens))
    case = make_case(5)
    gather_moe_routes(*case)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gather_moe_routes(*case)
    routes, mapping, _, _, _, output = case
    routes.neg_()
    mapping.copy_(mapping.flip(0))
    output.fill_(torch.nan)
    graph.replay()
    torch.cuda.synchronize()
    indices = mapping.long()
    expected = routes[indices.clamp_min(0), :width]
    expected[indices < 0] = 0
    torch.testing.assert_close(output, expected.reshape(output.shape), atol=0, rtol=0)

    # Rejection must happen before a permuted gather can corrupt its source.
    source = torch.randn(8, width, device=device, dtype=torch.bfloat16)
    before = source.clone()
    mapping = torch.arange(7, -1, -1, device=device, dtype=torch.int32)
    with pytest.raises(ValueError, match="overlap"):
        gather_moe_routes(source, mapping, 1, 8, width, source.view(1, 8, width))
    torch.testing.assert_close(source, before, atol=0, rtol=0)

    destination = torch.zeros(1, 8, width, device=device, dtype=torch.bfloat16)
    mapping = destination.flatten().view(torch.int32)[:8]
    mapping.copy_(torch.arange(7, -1, -1, device=device, dtype=torch.int32))
    before_destination = destination.clone()
    with pytest.raises(ValueError, match="overlap"):
        gather_moe_routes(source, mapping, 1, 8, width, destination)
    torch.testing.assert_close(
        destination.view(torch.int16),
        before_destination.view(torch.int16),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(source, before, atol=0, rtol=0)


@pytest.mark.parametrize("device", ["cuda"])
def test_bf16_moe_routes_write_a_supplied_destination(device: str) -> None:
    """A caller-owned destination holds the same routes, zeroing stale and remote rows."""
    experts, width, inter, tokens, topk = 4, 256, 128, 5, 2
    torch.manual_seed(3)
    w13 = (torch.randn(experts, 2 * inter, width, device=device) * 0.05).bfloat16()
    w2 = (torch.randn(experts, width, inter, device=device) * 0.05).bfloat16()
    x = (torch.randn(tokens, width, device=device) * 0.5).bfloat16()
    # Expert 5 is outside this rank's [1, 5) shard, so its route stays zero.
    ids = torch.tensor(
        [[1, 2], [3, 4], [5, 1], [2, 2], [4, 3]], device=device, dtype=torch.int32
    )
    expected = bf16_moe_routes(x, w13, w2, ids, 1, batch_invariant=False)

    storage = torch.full(
        (tokens * topk + 2, width), 7.0, device=device, dtype=torch.bfloat16
    )
    destination = storage[1:-1]
    actual = bf16_moe_routes(
        x, w13, w2, ids, 1, output=destination, batch_invariant=False
    )

    assert actual.data_ptr() == destination.data_ptr()
    assert actual.shape == (tokens, topk, width)
    assert torch.equal(actual, expected)
    assert not actual[2, 0].any()
    assert torch.equal(storage[[0, -1]], torch.full_like(storage[[0, -1]], 7.0))
    with pytest.raises(ValueError, match="overlap"):
        bf16_moe_routes(
            x,
            w13,
            w2,
            ids,
            1,
            output=w2.view(-1)[: tokens * topk * width].view(tokens * topk, width),
            batch_invariant=False,
        )


@pytest.mark.parametrize("device", ["cuda"])
def test_routed_sum_writes_a_supplied_output(device: str) -> None:
    """The route sum stored into caller-owned storage equals the allocated result."""
    torch.manual_seed(4)
    tokens, topk, width = 3, 8, 4096
    routes = (torch.randn(tokens, topk, width, device=device) * 0.5).bfloat16()
    route_weights = torch.rand(tokens, topk, device=device).softmax(-1)
    norm_weight = torch.randn(width, device=device, dtype=torch.bfloat16) * 0.1
    expected = routed_input_scale_rmsnorm_sum(
        routes, route_weights, norm_weight, 1e-8, 1.0, False, None
    )
    output = torch.full_like(expected, float("nan"))
    actual = routed_input_scale_rmsnorm_sum(
        routes, route_weights, norm_weight, 1e-8, 1.0, False, output
    )
    assert actual is output
    assert torch.equal(actual, expected)
    with pytest.raises(ValueError, match="overlap"):
        routed_input_scale_rmsnorm_sum(
            routes,
            route_weights,
            norm_weight,
            1e-8,
            1.0,
            False,
            routes.view(-1)[: tokens * width].view(tokens, width),
        )


@pytest.mark.parametrize("device", ["cuda"])
def test_gathered_route_reduction_matches_materialized_routes(device: str) -> None:
    """Reading routes through the permute map must equal gathering them first."""
    torch.manual_seed(5)
    tokens, topk, width, padded_width = 6, 8, 4096, 4096 + 128
    rows = tokens * topk + 16
    gemm2_out = (torch.randn(rows, padded_width, device=device) * 0.5).to(
        torch.bfloat16
    )
    permute_map = torch.randperm(rows, device=device)[: tokens * topk].to(torch.int32)
    permute_map[torch.tensor([3, 17, 40], device=device)] = -1  # remote routes
    route_weights = torch.rand(tokens, topk, device=device).softmax(-1)
    norm_weight = torch.randn(width, device=device, dtype=torch.bfloat16) * 0.1

    permuted = permute_map.long()
    routes = gemm2_out.index_select(0, permuted.clamp_min(0))[:, :width]
    routes = torch.where((permuted >= 0).unsqueeze(-1), routes, routes.new_zeros(()))
    routes = routes.view(tokens, topk, width)
    gathered = gather_moe_routes(gemm2_out, permute_map, tokens, topk, width)
    torch.testing.assert_close(gathered, routes, atol=0, rtol=0)
    expected = routed_input_scale_rmsnorm_sum(
        routes, route_weights, norm_weight, 1e-8, 1.0, False, None
    )
    actual = triton_routed_sum_gathered(
        gemm2_out,
        permute_map,
        tokens,
        topk,
        route_weights,
        norm_weight,
        1e-8,
        1.0,
        False,
        None,
    )
    # CDNA4 lowers the mapped pointer path separately from the materialized
    # path. A few values can straddle a BF16 boundary even though both keep
    # their reductions in FP32; the measured worst case is 2**-14.
    if current_platform().is_amd:
        torch.testing.assert_close(actual, expected, atol=2**-13, rtol=0)
    else:
        assert torch.equal(actual, expected)
    supplied = torch.empty_like(actual)
    written = triton_routed_sum_gathered(
        gemm2_out,
        permute_map,
        tokens,
        topk,
        route_weights,
        norm_weight,
        1e-8,
        1.0,
        False,
        supplied,
    )
    assert written is supplied
    assert torch.equal(written, actual)
    # Every step stays FP32; only the final store rounds to BF16.
    scaled = routes.float() * (norm_weight + 1.0).float()
    normalized = scaled * torch.rsqrt(scaled.square().mean(dim=-1, keepdim=True) + 1e-8)
    reference = (normalized * route_weights[..., None]).sum(dim=1).to(routes.dtype)
    torch.testing.assert_close(expected, reference, atol=5e-4, rtol=0.0)


@pytest.mark.parametrize("device", ["cuda"])
@pytest.mark.parametrize("use_map", [False, True])
def test_single_pass_route_reduction_matches_two_pass(
    device: str, use_map: bool
) -> None:
    """Single- and two-launch reductions must agree to the platform contract."""
    torch.manual_seed(7)
    tokens, topk, width = 6, 8, 4096
    rows = tokens * topk
    routes = (torch.randn(rows, width, device=device) * 0.5).to(torch.bfloat16)
    permute_map = None
    if use_map:
        permute_map = torch.randperm(rows, device=device).to(torch.int32)
        permute_map[torch.tensor([2, 9, 30], device=device)] = -1
    route_weights = torch.rand(tokens, topk, device=device, dtype=torch.float32)
    norm_weight = torch.randn(width, device=device, dtype=torch.bfloat16) * 0.1
    outputs = []
    for single_pass, enable_pdl in itertools.product((False, True), (False, True)):
        output = torch.empty((tokens, width), dtype=torch.bfloat16, device=device)
        _launch_routed_input_scale_sum(
            routes,
            permute_map,
            width,
            route_weights,
            norm_weight,
            output,
            tokens,
            topk,
            1e-5,
            1.0,
            width,
            single_pass,
            enable_pdl,
        )
        outputs.append(output)
    for output in outputs[1:]:
        if current_platform().is_amd and use_map:
            # Separate CDNA4 lowering of the mapped single-pass loop can move
            # a handful of values by one BF16 ULP.
            torch.testing.assert_close(
                output,
                outputs[0],
                atol=2**-8,
                rtol=2**-7,
            )
        else:
            assert torch.equal(output, outputs[0])
