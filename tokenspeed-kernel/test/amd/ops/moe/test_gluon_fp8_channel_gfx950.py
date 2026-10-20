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

"""Device qualification prepared on CPU; no collection-time device queries."""

import pytest
import torch
from utils import is_cdna4


def _require_device():
    if not is_cdna4():
        pytest.skip("requires gfx950")


def _raw_down(a, b, block_m):
    """Same production MFMA body, FP32 diagnostic output and unit scales."""
    from tokenspeed_kernel_amd.ops.gfx950.moe.fp16.stage2_kernel import (
        gluon_bf16_moe_stage2_kernel,
    )

    rows, reduction = a.shape
    columns = b.shape[0]
    sorted_ids = torch.arange(rows, device=a.device, dtype=torch.int32)
    experts = torch.zeros(1, device=a.device, dtype=torch.int32)
    valid = torch.full((1,), rows, device=a.device, dtype=torch.int32)
    row_scales = torch.ones(rows, device=a.device)
    channel_scales = torch.ones(columns, device=a.device)
    output = torch.empty(rows, 1, columns, device=a.device)
    compiled = gluon_bf16_moe_stage2_kernel[(1,)](
        a,
        b,
        channel_scales,
        row_scales,
        output,
        sorted_ids,
        experts,
        row_scales,
        valid,
        columns,
        reduction,
        rows,
        rows,
        1,
        *a.stride(),
        b.numel(),
        *b.stride(),
        0,
        0,
        0,
        *output.stride(),
        BLOCK_M=block_m,
        BLOCK_N=64,
        BLOCK_K=64,
        NUM_WARPS=4,
        WEIGHT_FP8=False,
        CHANNEL_FP8=True,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output[:, 0], compiled


@pytest.mark.parametrize("block_m", [16, 32, 64])
def test_raw_code_bf16_exact_mixed_magnitude_products(block_m):
    _require_device()
    for first, second in ((0, 1), (0, 2), (7, 8), (31, 63)):
        a = torch.zeros(block_m, 64, device="cuda")
        a[:, first], a[:, second] = 256.0, 0.25
        for sign in (-1.0, 1.0):
            b = torch.zeros(64, 64, device="cuda")
            b[:, first], b[:, second] = 256.0, sign * 0.25
            actual, _ = _raw_down(
                a.to(torch.float8_e4m3fn), b.to(torch.float8_e4m3fn), block_m
            )
            assert torch.all(actual == 65536.0 + sign * 0.0625)


@pytest.mark.parametrize("block_m", [16, 32, 64])
def test_raw_code_bf16_short_k_fp32_bound(block_m):
    _require_device()
    generator = torch.Generator(device="cuda").manual_seed(29)
    a = (
        (torch.randn(block_m, 64, device="cuda", generator=generator) * 100)
        .clamp(-448, 448)
        .to(torch.float8_e4m3fn)
    )
    b = (
        (torch.randn(64, 64, device="cuda", generator=generator) * 100)
        .clamp(-448, 448)
        .to(torch.float8_e4m3fn)
    )
    actual, compiled = _raw_down(a, b, block_m)
    reference = a.double() @ b.double().T
    sum_abs = a.double().abs() @ b.double().abs().T
    gamma = (63 * 2.0**-24) / (1 - 63 * 2.0**-24)
    assert torch.isfinite(actual).all()
    assert torch.all((actual.double() - reference).abs() <= gamma * sum_abs)
    # The actual compiled kernel is exposed to an external evidence plugin;
    # inspecting it does not launch a second, stand-in projection.
    assert compiled is not None


@pytest.mark.parametrize("block_m", [16, 32, 64])
@pytest.mark.parametrize("intermediate", [320, 640])
def test_grouped_channel_projection_boundaries(block_m, intermediate):
    _require_device()
    from tokenspeed_kernel.ops.moe._triton.fp8_channel_fused import _quantize_routes
    from tokenspeed_kernel_amd.ops.gfx950.moe.fp16.moe_align_device import (
        moe_align_block_size_device,
    )
    from tokenspeed_kernel_amd.ops.gfx950.moe.fp16.stage1_kernel import (
        launch_gluon_fp8_channel_stage1_gfx950,
    )
    from tokenspeed_kernel_amd.ops.gfx950.moe.fp16.stage2_kernel import (
        launch_gluon_fp8_channel_stage2_gfx950,
    )

    generator = torch.Generator(device="cuda").manual_seed(1)
    tokens, experts, hidden, topk = 5, 8, 5120, 8
    xq = torch.randn(tokens, hidden, device="cuda", generator=generator).to(
        torch.float8_e4m3fn
    )
    w13 = torch.randn(
        experts,
        2 * intermediate,
        hidden,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    ).to(torch.float8_e4m3fn)
    w2 = torch.randn(
        experts,
        hidden,
        intermediate,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    ).to(torch.float8_e4m3fn)
    xs = torch.rand(tokens, 1, device="cuda", generator=generator) * 0.01
    s13 = (
        torch.rand(experts, 2 * intermediate, device="cuda", generator=generator) * 0.05
    )
    s2 = torch.rand(experts, hidden, device="cuda", generator=generator) * 0.05
    ids = (
        torch.arange(experts, device="cuda", dtype=torch.int32)
        .expand(tokens, -1)
        .contiguous()
    )
    weights = torch.rand(tokens, topk, device="cuda", generator=generator)
    sorted_ids, sorted_experts, sorted_weights, valid = moe_align_block_size_device(
        ids, weights, experts, block_m, expert_start=0
    )
    activated = torch.empty(
        tokens * topk, intermediate, device="cuda", dtype=torch.bfloat16
    )
    launch_gluon_fp8_channel_stage1_gfx950(
        xq,
        w13,
        xs,
        s13,
        sorted_ids,
        sorted_experts,
        valid,
        activated,
        topk,
        block_m=block_m,
    )
    expected = torch.empty_like(activated)
    for expert in range(experts):
        raw = xq.double() @ w13[expert].double().T
        scaled = (raw.float() * xs) * s13[expert]
        gate, up = scaled.chunk(2, dim=1)
        expected[expert::topk] = (gate * torch.sigmoid(gate) * up).bfloat16()
    error = torch.linalg.vector_norm(
        activated.float() - expected.float()
    ) / torch.linalg.vector_norm(expected.float())
    assert error < 2e-3
    route_codes, route_scales = _quantize_routes(activated)
    partials = torch.empty(tokens, topk, hidden, device="cuda", dtype=torch.bfloat16)
    launch_gluon_fp8_channel_stage2_gfx950(
        route_codes,
        w2,
        route_scales.view(-1, 1),
        s2,
        sorted_ids,
        sorted_experts,
        sorted_weights,
        valid,
        partials,
        block_m=block_m,
    )
    for expert in range(experts):
        raw = route_codes[expert::topk].double() @ w2[expert].double().T
        reference = (
            (raw.float() * route_scales[expert::topk, None]) * s2[expert]
        ).bfloat16()
        error = torch.linalg.vector_norm(
            partials[:, expert].float() - reference.float()
        ) / torch.linalg.vector_norm(reference.float())
        assert error < 2e-3
    # Route weights must not affect the projection's BF16 rounding boundary.
    changed_weights = sorted_weights * 3.14159
    repeated = torch.empty_like(partials)
    launch_gluon_fp8_channel_stage2_gfx950(
        route_codes,
        w2,
        route_scales.view(-1, 1),
        s2,
        sorted_ids,
        sorted_experts,
        changed_weights,
        valid,
        repeated,
        block_m=block_m,
    )
    assert torch.equal(partials, repeated)


@pytest.mark.parametrize("intermediate", [320, 640])
def test_channel_down_graph_refreshes_live_expert_populations(intermediate):
    _require_device()
    from tokenspeed_kernel_amd.ops.gfx950.moe.fp16.moe_align_device import (
        moe_align_block_size_device,
    )
    from tokenspeed_kernel_amd.ops.gfx950.moe.fp16.stage2_kernel import (
        gluon_bf16_moe_stage2_kernel,
        launch_gluon_fp8_channel_stage2_gfx950,
    )
    from utils import assert_no_triton_compile

    tokens, hidden = 33, 5120
    codes = torch.ones(tokens, intermediate, device="cuda").to(torch.float8_e4m3fn)
    w2 = torch.ones(2, hidden, intermediate, device="cuda").to(torch.float8_e4m3fn)
    w2[1] = torch.full((hidden, intermediate), 2.0, device="cuda").to(
        torch.float8_e4m3fn
    )
    ids = torch.zeros(tokens, 1, device="cuda", dtype=torch.int32)
    ids[31:] = 1
    weights = torch.ones(tokens, 1, device="cuda")
    row_scales = torch.ones(tokens, 1, device="cuda")
    channel_scales = torch.ones(2, hidden, device="cuda")
    partials = torch.empty(tokens, 1, hidden, device="cuda", dtype=torch.bfloat16)

    def run():
        sorted_ids, sorted_experts, sorted_weights, valid = moe_align_block_size_device(
            ids, weights, 2, 32, expert_start=0
        )
        launch_gluon_fp8_channel_stage2_gfx950(
            codes,
            w2,
            row_scales,
            channel_scales,
            sorted_ids,
            sorted_experts,
            sorted_weights,
            valid,
            partials,
            block_m=32,
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for population in (31, 32, 33, 31):
        ids.fill_(1)
        ids[:population] = 0
        with assert_no_triton_compile(gluon_bf16_moe_stage2_kernel):
            run()
        graph.replay()
        assert torch.all(partials[:population] == float(intermediate))
        assert torch.all(partials[population:] == float(2 * intermediate))


@pytest.fixture(scope="module")
def tp8_channel_fixture():
    _require_device()
    from tokenspeed_kernel.benchmark.generators.moe import _prepare_routed_weights
    from tokenspeed_kernel.ops.moe import moe_plan

    candidate, weights = _prepare_routed_weights(
        weight_dtype_name="fp8_channel",
        weight_dtype=torch.float8_e4m3fn,
        input_dtype=torch.bfloat16,
        activation="silu",
        routing_mode="precomputed_topk",
        ep_size=1,
        hidden_size=5120,
        intermediate_size_per_partition=320,
        num_experts=256,
        num_local_experts=256,
        top_k=8,
        ep_rank=0,
        swiglu_limit=None,
        situ_beta=None,
        situ_linear_beta=None,
        block_shape=None,
        internal_activation_dtype="input",
        fast_math=False,
        weight_seed=1861433681,
    )
    candidate["precision_policy"] = "default"
    native = moe_plan(
        weight_dtype="fp8_channel",
        input_dtype=torch.bfloat16,
        activation="silu",
        routing_mode="precomputed_topk",
        ep_size=1,
        ispp=320,
        hidden=5120,
        num_experts=256,
        top_k=8,
        swiglu_form="standard",
        internal_activation_dtype="input",
        expert_id_repeats=False,
        activation_clamped=False,
        combine_order="rank",
        solution="triton",
        fast_math=False,
    )
    native["precision_policy"] = "default"
    return native, candidate, weights


@pytest.mark.parametrize("tokens", [16, 32, 64, 128, 256, 848, 896, 8192])
def test_tp8_public_operation_reference_and_changed_route_graph(
    tp8_channel_fixture, tokens
):
    from ops.moe.test_triton_fp8_channel_fused import _native_fp8_mm, _reference
    from tokenspeed_kernel.ops.moe import moe_apply

    native_plan, candidate_plan, layer = tp8_channel_fixture
    generator = torch.Generator(device="cuda").manual_seed(19 + tokens)
    x = torch.randn(tokens, 5120, device="cuda", generator=generator).bfloat16()
    ids = (
        (
            torch.arange(tokens, device="cuda")[:, None] * 13
            + torch.arange(8, device="cuda")[None, :] * 17
        )
        % 256
    ).to(torch.int32)
    weights = torch.rand(tokens, 8, device="cuda", generator=generator) * 0.37

    def apply(plan):
        return moe_apply(plan, x, layer, None, topk_ids=ids, topk_weights=weights)

    def bf16_projection_reference(a, b):
        # Gluon converts FP8 codes to BF16 before the matrix instruction.
        return (a.double() @ b.double().T).float()

    for plan, matmul in (
        (native_plan, _native_fp8_mm),
        (candidate_plan, bf16_projection_reference),
    ):
        expected = _reference(x, layer, weights, ids, matmul=matmul)
        actual = apply(plan)
        assert torch.isfinite(actual).all()
        error = (actual.float() - expected.float()).norm() / expected.float().norm()
        assert error < 2e-3
    del actual, expected
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            apply(candidate_plan)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            captured = apply(candidate_plan)
        for concentrated in (False, True):
            if concentrated:
                # Same buffers and distinct top8, radically different live counts.
                ids.copy_(
                    torch.arange(8, device="cuda", dtype=torch.int32).expand(tokens, -1)
                )
                weights.mul_(0.5)
                x.mul_(0.9)
            eager = apply(candidate_plan)
            graph.replay()
            torch.testing.assert_close(captured, eager, atol=0, rtol=0)
    finally:
        graph.reset()
