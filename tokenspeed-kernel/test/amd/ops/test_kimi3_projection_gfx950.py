# Copyright (c) 2026 LightSeek Foundation

from __future__ import annotations

import pytest
import tokenspeed_kernel
import torch
from tokenspeed_kernel.ops.gemm.kimi3 import (
    _use_gluon_largem,
    _use_gluon_mediumm,
    _use_gluon_smallm,
)
from tokenspeed_kernel.ops.moe import moe_sigmoid_bias_topk
from utils import is_cdna4

if not is_cdna4():
    pytest.skip(
        "AMD CDNA4 is required for Kimi K3 projection tests",
        allow_module_level=True,
    )


@pytest.mark.parametrize(
    "input_size,output_size,num_tokens",
    [
        (7168, 3584, 1),
        (3584, 7168, 1),
        (3584, 7168, 2),
        (3584, 7168, 4),
        (7168, 3584, 128),
        (3584, 7168, 128),
        (7168, 3584, 768),
        (7168, 3584, 1024),
        (3584, 7168, 384),
        (3584, 7168, 512),
        (3584, 7168, 2048),
        (7168, 3584, 4096),
        (3584, 7168, 4096),
        (7168, 3584, 8192),
        (3584, 7168, 8192),
    ],
)
def test_kimi3_latent_projection_matches_torch(
    input_size: int,
    output_size: int,
    num_tokens: int,
) -> None:
    torch.manual_seed(input_size + num_tokens)
    hidden_states = torch.randn(
        num_tokens, input_size, device="cuda", dtype=torch.bfloat16
    )
    weight = torch.randn(output_size, input_size, device="cuda", dtype=torch.bfloat16)

    expected = torch.nn.functional.linear(hidden_states, weight)
    actual = tokenspeed_kernel.kimi3_latent_projection(hidden_states, weight)

    if _use_gluon_largem(num_tokens, input_size, output_size):
        assert torch.equal(actual, expected)
    else:
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize(
    "m,k,n,uses_medium,uses_large",
    [
        (512, 7168, 3584, False, False),
        (768, 7168, 3584, True, False),
        (1024, 7168, 3584, True, False),
        (2048, 7168, 3584, False, False),
        (4096, 7168, 3584, False, True),
        (320, 3584, 7168, False, False),
        (384, 3584, 7168, True, False),
        (512, 3584, 7168, True, False),
        (1024, 3584, 7168, False, False),
        (2048, 3584, 7168, False, True),
    ],
)
def test_kimi3_latent_projection_dispatch_boundaries(
    m: int,
    k: int,
    n: int,
    uses_medium: bool,
    uses_large: bool,
) -> None:
    assert _use_gluon_mediumm(m, k, n) is uses_medium
    assert _use_gluon_largem(m, k, n) is uses_large


@pytest.mark.parametrize(
    "m,k,n,expected",
    [
        (1, 3584, 7168, False),
        (2, 3584, 7168, True),
        (4, 3584, 7168, True),
        (4, 7168, 3584, False),
        (8, 3584, 7168, False),
    ],
)
def test_kimi3_latent_projection_smallm_dispatch(
    m: int, k: int, n: int, expected: bool
) -> None:
    assert _use_gluon_smallm(m, k, n) is expected


@pytest.mark.parametrize("input_size,output_size", [(7168, 3584), (3584, 7168)])
def test_kimi3_latent_projection_writes_out_and_captures(
    input_size: int,
    output_size: int,
) -> None:
    hidden_states = torch.randn(1, input_size, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(output_size, input_size, device="cuda", dtype=torch.bfloat16)
    output = torch.empty(1, output_size, device="cuda", dtype=torch.bfloat16)
    expected = torch.nn.functional.linear(hidden_states, weight)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        returned = tokenspeed_kernel.kimi3_latent_projection(
            hidden_states, weight, out=output
        )
    assert returned.data_ptr() == output.data_ptr()
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(output, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("num_tokens", [1, 2, 4, 16])
def test_kimi3_latent_projection_add3_matches_torch_and_captures(
    num_tokens: int,
) -> None:
    torch.manual_seed(31)
    hidden_states = torch.randn(num_tokens, 3584, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(7168, 3584, device="cuda", dtype=torch.bfloat16)
    prefix = torch.randn(num_tokens, 7168, device="cuda", dtype=torch.bfloat16)
    lane = torch.randn(num_tokens, 10752, device="cuda", dtype=torch.bfloat16)
    shared_output = lane[:, 3584:]
    expected = (
        prefix + torch.nn.functional.linear(hidden_states, weight) + shared_output
    )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = tokenspeed_kernel.kimi3_latent_projection_add3(
            hidden_states,
            weight,
            prefix,
            shared_output,
        )
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("num_tokens", [1, 2, 3, 4])
def test_kimi3_rmsnorm_linear_add_matches_composed_and_captures(
    num_tokens: int,
) -> None:
    torch.manual_seed(37)
    latent = torch.randn(num_tokens, 3584, device="cuda", dtype=torch.bfloat16)
    norm_weight = torch.randn(3584, device="cuda", dtype=torch.bfloat16)
    projection_weight = torch.randn(7168, 3584, device="cuda", dtype=torch.bfloat16)
    prefix = torch.randn(num_tokens, 7168, device="cuda", dtype=torch.bfloat16)
    shared_lane = torch.randn(num_tokens, 10752, device="cuda", dtype=torch.bfloat16)
    shared = shared_lane[:, 3584:]
    source = latent.float()
    normalized = (
        source
        * torch.rsqrt(source.square().mean(dim=-1, keepdim=True) + 1e-6)
        * norm_weight.float()
    ).to(latent.dtype)
    projected = torch.nn.functional.linear(normalized, projection_weight)
    expected = (projected.float() + prefix.float() + shared.float()).to(latent.dtype)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = tokenspeed_kernel.kimi3_latent_projection_add3(
            latent,
            projection_weight,
            prefix,
            shared,
            norm_weight=norm_weight,
            eps=1e-6,
        )
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def test_kimi3_latent_projection_rejects_forced_kernel_for_non_k3_shape() -> None:
    hidden_states = torch.empty(1, 128, device="cuda", dtype=torch.bfloat16)
    weight = torch.empty(64, 128, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="requires a contiguous gfx950"):
        tokenspeed_kernel.kimi3_latent_projection(
            hidden_states,
            weight,
            solution="triton_gemv",
        )


def test_kimi3_qkvfab_projection_matches_torch_and_captures() -> None:
    hidden_states = torch.randn(1, 7168, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(6288, 7168, device="cuda", dtype=torch.bfloat16)
    output = torch.empty(1, 6288, device="cuda", dtype=torch.bfloat16)
    expected = torch.nn.functional.linear(hidden_states, weight)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        returned = tokenspeed_kernel.kimi3_qkvfab_projection(
            hidden_states, weight, out=output
        )
    assert returned.data_ptr() == output.data_ptr()
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(output, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("num_tokens", [0, 1, 2, 4, 8, 33])
def test_kimi3_qkvfab_projection_dispatches_all_token_counts(
    num_tokens: int,
) -> None:
    hidden_states = torch.randn(num_tokens, 7168, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(6288, 7168, device="cuda", dtype=torch.bfloat16)
    expected = torch.nn.functional.linear(hidden_states, weight)
    actual = tokenspeed_kernel.kimi3_qkvfab_projection(hidden_states, weight)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def test_kimi3_router_projection_matches_torch_and_captures() -> None:
    hidden_states = torch.randn(1, 7168, device="cuda", dtype=torch.bfloat16)
    weight = (torch.randn(896, 7168, device="cuda", dtype=torch.float32) * 0.01).to(
        torch.bfloat16
    )
    correction_bias = torch.randn(896, device="cuda", dtype=torch.float32) * 0.01
    output = torch.empty(1, 896, device="cuda", dtype=torch.float32)
    expected = torch.nn.functional.linear(hidden_states.float(), weight.float())

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        returned = tokenspeed_kernel.kimi3_router_projection(
            hidden_states, weight, out=output
        )
    assert returned.data_ptr() == output.data_ptr()
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-4)
    expected_weights, expected_ids = moe_sigmoid_bias_topk(
        expected,
        correction_bias,
        16,
        routed_scaling_factor=1.0,
        normalize_topk_weights=True,
    )
    actual_weights, actual_ids = moe_sigmoid_bias_topk(
        output,
        correction_bias,
        16,
        routed_scaling_factor=1.0,
        normalize_topk_weights=True,
    )
    assert torch.equal(actual_ids, expected_ids)
    torch.testing.assert_close(actual_weights, expected_weights, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("num_tokens", [0, 1, 2, 4, 8, 33])
def test_kimi3_router_projection_dispatches_all_token_counts(
    num_tokens: int,
) -> None:
    hidden_states = torch.randn(num_tokens, 7168, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(896, 7168, device="cuda", dtype=torch.bfloat16)
    expected = torch.nn.functional.linear(hidden_states.float(), weight.float())
    actual = tokenspeed_kernel.kimi3_router_projection(hidden_states, weight)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("linear_beta", [None, 2.5])
def test_kimi3_shared_situ_projection_matches_reference_and_captures(
    linear_beta: float | None,
) -> None:
    torch.manual_seed(17)
    hidden_states = torch.randn(1, 7168, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(1536, 7168, device="cuda", dtype=torch.bfloat16) * 0.01
    output = torch.empty(1, 768, device="cuda", dtype=torch.bfloat16)
    gate_up = torch.nn.functional.linear(hidden_states, weight)
    expected = tokenspeed_kernel.situ_and_mul(
        gate_up,
        beta=1.5,
        linear_beta=linear_beta,
    )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        returned = tokenspeed_kernel.kimi3_shared_situ_projection(
            hidden_states,
            weight,
            beta=1.5,
            linear_beta=linear_beta,
            out=output,
        )
    assert returned.data_ptr() == output.data_ptr()
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(output, expected, rtol=2e-2, atol=2e-2)


def test_kimi3_shared_down_projection_matches_torch_and_captures() -> None:
    torch.manual_seed(23)
    hidden_states = torch.randn(1, 768, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(7168, 768, device="cuda", dtype=torch.bfloat16)
    output = torch.empty(1, 7168, device="cuda", dtype=torch.bfloat16)
    expected = torch.nn.functional.linear(hidden_states, weight)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        returned = tokenspeed_kernel.kimi3_shared_down_projection(
            hidden_states,
            weight,
            out=output,
        )
    assert returned.data_ptr() == output.data_ptr()
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(output, expected, rtol=2e-2, atol=2e-2)
