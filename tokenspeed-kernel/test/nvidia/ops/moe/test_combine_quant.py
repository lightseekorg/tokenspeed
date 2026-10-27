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

import pytest
import torch
from tokenspeed_kernel.ops.moe import (
    combine_quant_fp8,
    prepare_sm90_fp8_moe,
    prepare_sm90_fp8_moe_routes,
)

TOKENS = (1, 8, 16, 32, 64, 128, 256, 1024, 4096)


def require_hopper():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("Requires Hopper")


def assert_bytes(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    actual_bytes = actual.reshape(-1).contiguous().view(torch.uint8)
    expected_bytes = expected.reshape(-1).contiguous().view(torch.uint8)
    assert torch.equal(actual_bytes, expected_bytes), {
        "shape": tuple(actual.shape),
        "dtype": str(actual.dtype),
        "actual": actual.flatten()[:16].float().tolist(),
        "expected": expected.flatten()[:16].float().tolist(),
        "actual_bytes": actual_bytes.flatten()[:64].tolist(),
        "expected_bytes": expected_bytes.flatten()[:64].tolist(),
        "mismatched_bytes": int((actual_bytes != expected_bytes).sum()),
    }


def reference_quantize(combined):
    values = combined.view(combined.shape[0], combined.shape[1] // 128, 128).float()
    epsilon = torch.tensor(1e-10, dtype=torch.bfloat16, device=combined.device).float()
    maximum = values.abs().amax(-1).clamp_min(epsilon)
    multiplier = (448.0 / maximum.double()).float()
    scales = (1.0 / multiplier.double()).float()
    q = (values * multiplier.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q.view_as(combined), scales


def reference_combine(partial, inverse, weights, valid_rows):
    # BF16 * FP32 products fit in FP64. Round once after each ordered FMA.
    result = torch.zeros(
        (weights.shape[0], partial.shape[1]), device=partial.device, dtype=torch.float32
    )
    for slot in range(weights.shape[1]):
        index = inverse[:, slot].long()
        valid = (index >= 0) & (index < valid_rows[0]) & (index < partial.shape[0])
        value = partial[index.clamp(0, partial.shape[0] - 1)].double()
        value = torch.where(valid[:, None], value, 0.0)
        result = (result.double() + weights[:, slot, None].double() * value).float()
    return result.to(torch.bfloat16)


def exercise_combined_quantization(tokens, column_map, topk, hidden):
    require_hopper()
    generator = torch.Generator(device="cuda").manual_seed(812)
    partial = torch.randn(
        (tokens * topk, hidden),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    inverse = (
        torch.randperm(tokens * topk, device="cuda", generator=generator)
        .to(torch.int32)
        .view(tokens, topk)
    )
    if column_map:
        inverse = inverse.T.contiguous().T
    # Dyadic weights exercise exact ordered accumulation without a reference GEMM.
    weights = (
        torch.randint(-4, 5, (tokens, topk), device="cuda", generator=generator).float()
        / 16
    )
    valid_rows = torch.tensor([tokens * topk], device="cuda", dtype=torch.int64)
    out = combine_quant_fp8(partial, inverse, weights, valid_rows, solution="cute_dsl")
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            combine_quant_fp8(
                partial, inverse, weights, valid_rows, solution="cute_dsl", out=out
            )
        for limit in (tokens * topk, tokens * topk // 2, 0, tokens * topk):
            valid_rows.fill_(limit)
            partial.normal_(generator=generator)
            if limit < tokens * topk:
                partial[limit:].fill_(float("nan"))
            inverse[0, 0] = -1
            out[0].view(torch.uint8).fill_(255)
            out[1].fill_(float("nan"))
            graph.replay()
            expected = reference_quantize(
                reference_combine(partial, inverse, weights, valid_rows)
            )
            assert_bytes(out[0], expected[0])
            try:
                assert_bytes(out[1], expected[1])
            except AssertionError as error:
                raise AssertionError(
                    {
                        "tokens": tokens,
                        "column_map": column_map,
                        "valid_rows": limit,
                        "scale_difference": error.args,
                    }
                ) from error
    finally:
        torch.cuda.synchronize()
        graph.reset()


@pytest.mark.parametrize("tokens", TOKENS)
@pytest.mark.parametrize("column_map", (False, True))
def test_combined_quantization_and_graph_refresh(tokens, column_map):
    exercise_combined_quantization(tokens, column_map, 5, 256)


@pytest.mark.parametrize("tokens", (1, 256))
@pytest.mark.parametrize("topk", (1, 2, 4, 8, 16, 32))
@pytest.mark.parametrize("column_map", (False, True))
def test_model_topk_and_graph_refresh(tokens, topk, column_map):
    exercise_combined_quantization(tokens, column_map, topk, 256)


@pytest.fixture(scope="module")
def native_routes():
    require_hopper()
    settings = dict(mode="optimized", max_num_stages_fc1=-1, max_num_stages_fc2=-1)
    return prepare_sm90_fp8_moe(**settings), prepare_sm90_fp8_moe_routes(**settings)


def make_weights(experts, hidden, intermediate):
    require_hopper()
    generator = torch.Generator(device="cuda").manual_seed(1701)
    matrices, scales = [], []
    for rows, columns in ((2 * intermediate, hidden), (hidden, intermediate)):
        value = torch.randn(
            (experts, rows, columns),
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        matrices.append((value * 8).to(torch.float8_e4m3fn))
        scales.append(
            torch.exp2(
                torch.randint(
                    -8,
                    -3,
                    (experts, rows // 128, columns // 128),
                    device="cuda",
                    generator=generator,
                ).float()
            )
        )
    return *matrices, *scales


@pytest.fixture(scope="module")
def weights():
    return make_weights(65, 256, 768)


@pytest.fixture(scope="module", params=(33, 65, 129, 257))
def varied_expert_weights(request):
    return make_weights(request.param, 256, 768)


def exercise_deferred_native(native_routes, weights, tokens, ep_size, ep_rank, topk):
    native, deferred = native_routes
    w1, w2, s1, s2 = weights
    experts, hidden = w1.shape[0], w1.shape[2]
    x = torch.randn((tokens, hidden), device="cuda", dtype=torch.bfloat16) * 0.1
    ids = (
        torch.arange(tokens * topk, device="cuda", dtype=torch.int32).view(tokens, topk)
        % experts
        + ep_rank * experts
    )
    route_weights = torch.softmax(torch.randn((tokens, topk), device="cuda"), -1)
    hint = max(1, (tokens * topk + experts - 1) // experts)
    args = dict(
        input=x,
        token_selected_experts=ids,
        token_final_scales=route_weights,
        fc1_expert_weights=w1,
        fc2_expert_weights=w2,
        ep_size=ep_size,
        ep_rank=ep_rank,
        expected_m=hint,
        enable_pdl=False,
    )

    def whole():
        return native(
            **args,
            quant_scales=[s1, s2],
            output_dtype=torch.bfloat16,
            use_deepseek_fp8_block_scale=True,
            use_fused_finalize=False,
        )[0]

    routes = deferred(**args, fc1_scales=s1, fc2_scales=s2)
    destinations = (routes.partial, routes.inverse.T.reshape(-1), routes.expert_offsets)
    quantized = combine_quant_fp8(
        routes.partial,
        routes.inverse,
        routes.weights,
        routes.valid_rows,
        solution="cute_dsl",
    )
    whole()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            captured = deferred(**args, fc1_scales=s1, fc2_scales=s2, out=destinations)
            combine_quant_fp8(
                captured.partial,
                captured.inverse,
                captured.weights,
                captured.valid_rows,
                solution="cute_dsl",
                out=quantized,
            )
        for pattern in ("balanced", "concentrated", "mixed", "remote", "balanced"):
            local = torch.arange(tokens * topk, device="cuda", dtype=torch.int32).view(
                tokens, topk
            )
            local.remainder_(
                min(topk, experts) if pattern == "concentrated" else experts
            )
            ids.copy_(local + ep_rank * experts)
            if ep_size > 1 and pattern in ("mixed", "remote"):
                ids[:, :: 2 if pattern == "mixed" else 1] = 0
            x.normal_(mean=0, std=0.1)
            route_weights.copy_(torch.softmax(torch.randn_like(route_weights), -1))
            destinations[0].fill_(float("nan"))
            destinations[1].fill_(-999)
            destinations[2].fill_(-999)
            graph.replay()
            expected = whole()
            assert_bytes(
                reference_combine(
                    captured.partial,
                    captured.inverse,
                    captured.weights,
                    captured.valid_rows,
                ),
                expected,
            )
            q, scales = reference_quantize(expected)
            assert_bytes(quantized[0], q)
            assert_bytes(quantized[1], scales)
    finally:
        torch.cuda.synchronize()
        graph.reset()


@pytest.mark.parametrize("tokens", TOKENS)
@pytest.mark.parametrize("ep_size,ep_rank", ((1, 0), (3, 2)))
def test_deferred_native_matches_whole_moe(
    native_routes, weights, tokens, ep_size, ep_rank
):
    exercise_deferred_native(native_routes, weights, tokens, ep_size, ep_rank, 5)


@pytest.mark.parametrize("tokens", (1, 256))
@pytest.mark.parametrize("topk", (1, 2, 4, 8, 16, 32))
def test_deferred_native_model_dimensions(
    native_routes, varied_expert_weights, tokens, topk
):
    exercise_deferred_native(native_routes, varied_expert_weights, tokens, 3, 2, topk)


@pytest.mark.parametrize("hidden", (128, 256, 768, 1024, 2048))
@pytest.mark.parametrize("tokens", (1, 17, 256))
@pytest.mark.parametrize("column_map", (False, True))
def test_combined_quantization_widths(hidden, tokens, column_map):
    exercise_combined_quantization(tokens, column_map, 5, hidden)


@pytest.mark.parametrize(
    "hidden,intermediate", ((128, 256), (256, 768), (768, 512), (1024, 2048))
)
def test_deferred_native_widths(native_routes, hidden, intermediate):
    matrices = make_weights(33, hidden, intermediate)
    exercise_deferred_native(native_routes, matrices, 8, 3, 2, 5)
