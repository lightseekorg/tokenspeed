# Copyright (c) 2026 LightSeek Foundation

from __future__ import annotations

import pytest
import tokenspeed_kernel
import torch
from tokenspeed_kernel.ops.attention.mla import mla_normalize_project_query
from utils import is_cdna5

if not is_cdna5():
    pytest.skip(
        "AMD CDNA5 is required for Kimi K3 gfx1250 projection tests",
        allow_module_level=True,
    )


def test_kimi3_m16_add3_auto_matches_composed_and_captures() -> None:
    torch.manual_seed(1250)
    hidden_states = torch.randn(16, 3584, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(7168, 3584, device="cuda", dtype=torch.bfloat16)
    prefix = torch.randn(16, 7168, device="cuda", dtype=torch.bfloat16)
    lane = torch.randn(16, 10752, device="cuda", dtype=torch.bfloat16)
    shared_output = lane[:, 3584:]

    composed = tokenspeed_kernel.kimi3_latent_projection_add3(
        hidden_states,
        weight,
        prefix,
        shared_output,
        solution="composed",
    )
    forced = tokenspeed_kernel.kimi3_latent_projection_add3(
        hidden_states,
        weight,
        prefix,
        shared_output,
        solution="gluon_wmma_add3",
    )
    triton_control = tokenspeed_kernel.kimi3_latent_projection_add3(
        hidden_states,
        weight,
        prefix,
        shared_output,
        solution="triton_wmma_add3",
    )
    automatic = tokenspeed_kernel.kimi3_latent_projection_add3(
        hidden_states,
        weight,
        prefix,
        shared_output,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(forced, composed, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(triton_control, composed, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(automatic, composed, rtol=2e-2, atol=2e-2)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = tokenspeed_kernel.kimi3_latent_projection_add3(
            hidden_states,
            weight,
            prefix,
            shared_output,
        )
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured, composed, rtol=2e-2, atol=2e-2)

    hidden_states.copy_(torch.randn_like(hidden_states))
    prefix.copy_(torch.randn_like(prefix))
    shared_output.copy_(torch.randn_like(shared_output))
    mutated_expected = tokenspeed_kernel.kimi3_latent_projection_add3(
        hidden_states,
        weight,
        prefix,
        shared_output,
        solution="composed",
    )
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured, mutated_expected, rtol=2e-2, atol=2e-2)


def test_kimi3_m16_add3_rejects_non_target_projection() -> None:
    hidden_states = torch.empty(16, 7168, device="cuda", dtype=torch.bfloat16)
    weight = torch.empty(3584, 7168, device="cuda", dtype=torch.bfloat16)
    addend = torch.empty(16, 3584, device="cuda", dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="3584->7168"):
        tokenspeed_kernel.kimi3_latent_projection_add3(
            hidden_states,
            weight,
            addend,
            addend,
            solution="gluon_wmma_add3",
        )


@pytest.mark.parametrize("num_tokens", [2, 4, 8, 16, 32])
def test_kimi3_mla_qkv_gate_tdm_auto_matches_and_captures(
    num_tokens: int,
) -> None:
    torch.manual_seed(3648 + num_tokens)
    hidden_states = torch.randn(
        num_tokens,
        7168,
        device="cuda",
        dtype=torch.bfloat16,
    )
    weight = torch.randn(3648, 7168, device="cuda", dtype=torch.bfloat16)
    expected = torch.nn.functional.linear(hidden_states, weight)

    forced = tokenspeed_kernel.kimi3_mla_qkv_gate_projection(
        hidden_states,
        weight,
        2112,
        solution="gluon_wmma_gfx1250",
    )
    automatic = tokenspeed_kernel.kimi3_mla_qkv_gate_projection(
        hidden_states,
        weight,
        2112,
    )
    torch.cuda.synchronize()
    assert forced.packed is not None
    assert automatic.packed is not None
    torch.testing.assert_close(forced.packed, expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(automatic.packed, expected, rtol=2e-2, atol=2e-2)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = tokenspeed_kernel.kimi3_mla_qkv_gate_projection(
            hidden_states,
            weight,
            2112,
        )
    graph.replay()
    torch.cuda.synchronize()
    assert captured.packed is not None
    torch.testing.assert_close(captured.packed, expected, rtol=2e-2, atol=2e-2)

    hidden_states.copy_(torch.randn_like(hidden_states))
    mutated_expected = torch.nn.functional.linear(hidden_states, weight)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        captured.packed,
        mutated_expected,
        rtol=2e-2,
        atol=2e-2,
    )


@pytest.mark.parametrize("num_tokens", [512, 4095, 4096, 4097, 8192, 12289])
def test_kimi3_gfx1250_large_m_latent_projection_matches_and_captures(
    num_tokens: int,
) -> None:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(1250 + num_tokens)
    hidden_states = torch.randn(
        num_tokens,
        3584,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    weight = torch.randn(
        7168,
        3584,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    expected = torch.nn.functional.linear(hidden_states, weight)
    output = torch.empty_like(expected)

    actual = tokenspeed_kernel.kimi3_latent_projection(
        hidden_states,
        weight,
        out=output,
    )
    torch.cuda.synchronize()
    assert actual.data_ptr() == output.data_ptr()
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = tokenspeed_kernel.kimi3_latent_projection(
            hidden_states,
            weight,
        )
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("num_tokens", [512, 4095, 4097, 8192, 12289])
def test_kimi3_gfx1250_large_m_shared_down_matches(
    num_tokens: int,
) -> None:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(2500 + num_tokens)
    hidden_states = torch.randn(
        num_tokens,
        768,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    weight = torch.randn(
        7168,
        768,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    expected = torch.nn.functional.linear(hidden_states, weight)
    output = torch.empty_like(expected)

    actual = tokenspeed_kernel.kimi3_shared_down_projection(
        hidden_states,
        weight,
        out=output,
    )
    torch.cuda.synchronize()
    assert actual.data_ptr() == output.data_ptr()
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("num_tokens", [513, 4095, 8192])
def test_kimi3_gfx1250_large_m_mla_qkv_gate_matches(
    num_tokens: int,
) -> None:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(3250 + num_tokens)
    hidden_states = torch.randn(
        num_tokens,
        7168,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    weight = torch.randn(
        3648,
        7168,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    expected = torch.nn.functional.linear(hidden_states, weight)

    projection = tokenspeed_kernel.kimi3_mla_qkv_gate_projection(
        hidden_states,
        weight,
        2112,
    )
    torch.cuda.synchronize()
    assert projection.packed is None
    torch.testing.assert_close(
        projection.qkv,
        expected[:, :2112],
        rtol=2e-2,
        atol=2e-2,
    )
    torch.testing.assert_close(
        projection.gate,
        expected[:, 2112:],
        rtol=2e-2,
        atol=2e-2,
    )


def test_kimi3_gfx1250_large_m_mla_query_matches() -> None:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(3400)
    query = torch.randn(
        513,
        1536,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    kv = torch.randn(
        513,
        512,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    query_norm_weight = torch.randn(
        1536,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    kv_norm_weight = torch.randn(
        512,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    projection_weight = torch.randn(
        2304,
        1536,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    eps = 1e-6
    query_fp32 = query.float()
    expected_query_norm = (
        query_fp32
        * torch.rsqrt(query_fp32.square().mean(dim=-1, keepdim=True) + eps)
        * query_norm_weight.float()
    ).to(query.dtype)
    kv_fp32 = kv.float()
    expected_kv = (
        kv_fp32
        * torch.rsqrt(kv_fp32.square().mean(dim=-1, keepdim=True) + eps)
        * kv_norm_weight.float()
    ).to(kv.dtype)
    expected_query = torch.nn.functional.linear(
        expected_query_norm,
        projection_weight,
    )

    actual_query, absorbed_query = mla_normalize_project_query(
        query,
        kv,
        query_norm_weight,
        kv_norm_weight,
        projection_weight,
        eps=eps,
    )
    torch.cuda.synchronize()
    assert absorbed_query is None
    torch.testing.assert_close(
        actual_query,
        expected_query,
        rtol=2e-2,
        atol=2e-2,
    )
    torch.testing.assert_close(kv, expected_kv, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("num_tokens", [2, 4, 8, 16, 32])
def test_kimi3_kda_qkvfab_tdm_auto_matches_and_captures(
    num_tokens: int,
) -> None:
    torch.manual_seed(6288 + num_tokens)
    hidden_states = torch.randn(
        num_tokens,
        7168,
        device="cuda",
        dtype=torch.bfloat16,
    )
    weight = torch.randn(6288, 7168, device="cuda", dtype=torch.bfloat16)
    expected = torch.nn.functional.linear(hidden_states, weight)
    output = torch.empty_like(expected)

    forced = tokenspeed_kernel.kimi3_qkvfab_projection(
        hidden_states,
        weight,
        solution="gluon_wmma_gfx1250",
    )
    automatic = tokenspeed_kernel.kimi3_qkvfab_projection(
        hidden_states,
        weight,
        out=output,
    )
    torch.cuda.synchronize()
    assert automatic.data_ptr() == output.data_ptr()
    torch.testing.assert_close(forced, expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(automatic, expected, rtol=2e-2, atol=2e-2)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = tokenspeed_kernel.kimi3_qkvfab_projection(
            hidden_states,
            weight,
        )
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured, expected, rtol=2e-2, atol=2e-2)

    hidden_states.copy_(torch.randn_like(hidden_states))
    mutated_expected = torch.nn.functional.linear(hidden_states, weight)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        captured,
        mutated_expected,
        rtol=2e-2,
        atol=2e-2,
    )


@pytest.mark.parametrize("num_tokens", [513, 4095, 8192])
def test_kimi3_gfx1250_large_m_qkvfab_matches(
    num_tokens: int,
) -> None:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(3500 + num_tokens)
    hidden_states = torch.randn(
        num_tokens,
        7168,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    weight = torch.randn(
        6288,
        7168,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    expected = torch.nn.functional.linear(hidden_states, weight)

    actual = tokenspeed_kernel.kimi3_qkvfab_projection(
        hidden_states,
        weight,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def test_kimi3_gfx1250_large_m_linear_o_proj_shape_matches() -> None:
    from tokenspeed_kernel.ops.gemm.kimi3 import _try_gluon_largem_gfx1250

    generator = torch.Generator(device="cuda")
    generator.manual_seed(3600)
    hidden_states = torch.randn(
        512,
        1536,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    weight = torch.randn(
        7168,
        1536,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    expected = torch.nn.functional.linear(hidden_states, weight)

    actual = _try_gluon_largem_gfx1250(hidden_states, weight)
    torch.cuda.synchronize()
    assert actual is not None
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
