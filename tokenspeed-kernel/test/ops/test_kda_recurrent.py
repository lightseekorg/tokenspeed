"""CUDA-graph padding coverage for the KDA recurrent kernel."""

from __future__ import annotations

import pytest
import torch
from kimi3_reference import kda_gate
from kimi3_reference import kda_recurrent as reference_kda_recurrent
from tokenspeed_kernel.ops.attention import (
    kda_paged_decode,
    kda_paged_prefill,
    kda_recurrent,
    kda_recurrent_decode,
)
from tokenspeed_kernel.ops.attention.triton.kda_dispatch import (
    triton_nvidia_kda_paged_decode,
    triton_nvidia_kda_paged_prefill,
)
from tokenspeed_kernel.platform import current_platform


def test_k3_safe_gate_reference_matches_sigmoid_contract() -> None:
    """Distinguish K3's safe sigmoid gate from a clamped softplus gate."""
    raw_g = torch.tensor([[[-2.0, 0.0, 2.0]]])
    a_log = torch.tensor([0.25])
    dt_bias = torch.tensor([[0.5, -0.5, 1.0]])
    lower_bound = -5.0

    gate_input = raw_g + dt_bias
    expected = lower_bound * torch.sigmoid(torch.exp(a_log)[None, :, None] * gate_input)
    actual = kda_gate(raw_g, a_log, dt_bias, lower_bound=lower_bound)
    torch.testing.assert_close(actual, expected)

    legacy = torch.clamp_min(
        -torch.exp(a_log)[None, :, None] * torch.nn.functional.softplus(gate_input),
        lower_bound,
    )
    assert not torch.allclose(actual, legacy)


def test_kda_recurrent_matches_safe_gate_reference() -> None:
    """Packed recurrence must use K3's safe sigmoid decay gate."""
    device = "cuda"
    torch.manual_seed(5)
    tokens, heads, dim = 3, 2, 8
    q = torch.randn(tokens, heads, dim, device=device, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    raw_g = torch.randn_like(q)
    beta = torch.randn(tokens, heads, device=device, dtype=torch.bfloat16)
    state = torch.randn(1, heads, dim, dim, device=device, dtype=torch.float32)
    a_log = torch.randn(heads, device=device, dtype=torch.float32)
    dt_bias = torch.randn(heads, dim, device=device, dtype=torch.float32)
    lower_bound = -5.0

    expected_out, expected_state = reference_kda_recurrent(
        q,
        k,
        v,
        raw_g,
        beta,
        state[0],
        a_log,
        dt_bias,
        lower_bound=lower_bound,
    )
    actual_out, actual_state = kda_recurrent(
        q,
        k,
        v,
        raw_g,
        beta,
        state,
        a_log,
        dt_bias,
        lower_bound=lower_bound,
        cu_seqlens=torch.tensor([0, tokens], device=device, dtype=torch.int32),
        state_indices=torch.tensor([0], device=device, dtype=torch.int32),
    )

    torch.testing.assert_close(
        actual_out.float(), expected_out.float(), atol=5e-2, rtol=5e-2
    )
    torch.testing.assert_close(actual_state[0], expected_state, atol=5e-2, rtol=5e-2)


def test_kda_recurrent_zeroes_invalid_and_empty_graph_rows() -> None:
    """Invalid capture rows and trailing replay padding must be defined zeros."""

    device = "cuda"
    torch.manual_seed(7)
    tokens, heads, key_dim, value_dim = 4, 2, 8, 8
    q = torch.randn(tokens, heads, key_dim, device=device, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(tokens, heads, value_dim, device=device, dtype=torch.bfloat16)
    raw_g = torch.randn_like(q)
    beta = torch.randn(tokens, heads, device=device, dtype=torch.bfloat16)
    state = torch.randn(
        tokens,
        heads,
        value_dim,
        key_dim,
        device=device,
        dtype=torch.float32,
    )
    a_log = torch.randn(heads, device=device, dtype=torch.float32)
    dt_bias = torch.randn(heads, key_dim, device=device, dtype=torch.float32)

    # Capture uses one physical token per row but no live state slots.
    cu_seqlens = torch.arange(tokens + 1, device=device, dtype=torch.int32)
    state_indices = torch.full((tokens,), -1, device=device, dtype=torch.int32)

    # Compile before entering capture.
    kda_recurrent(
        q,
        k,
        v,
        raw_g,
        beta,
        state,
        a_log,
        dt_bias,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output, _ = kda_recurrent(
            q,
            k,
            v,
            raw_g,
            beta,
            state,
            a_log,
            dt_bias,
            cu_seqlens=cu_seqlens,
            state_indices=state_indices,
        )
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, torch.zeros_like(output))

    # Replay one live row plus multiple trailing empty padded rows. The live row
    # is computed normally; no physical padded row may retain allocator data.
    cu_seqlens.copy_(torch.tensor([0, 1, 1, 1, 1], device=device, dtype=torch.int32))
    state_indices.copy_(torch.tensor([0, -1, -1, -1], device=device, dtype=torch.int32))
    graph.replay()
    torch.cuda.synchronize()

    assert torch.isfinite(output[0]).all()
    assert torch.count_nonzero(output[0]) > 0
    torch.testing.assert_close(output[1:], torch.zeros_like(output[1:]))


def test_kda_recurrent_decode_updates_indexed_state_pool() -> None:
    """Fused decode must match recurrent math and remain graph replay safe."""

    device = "cuda"
    torch.manual_seed(11)
    tokens, active, heads, dim = 4, 2, 2, 8
    q = torch.randn(tokens, heads, dim, device=device, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    raw_g = torch.randn_like(q)
    beta = torch.randn(tokens, heads, device=device, dtype=torch.bfloat16)
    state_pool = torch.randn(4, heads, dim, dim, device=device, dtype=torch.float32)
    initial_pool = state_pool.clone()
    a_log = torch.randn(heads, device=device, dtype=torch.float32)
    dt_bias = torch.randn(heads, dim, device=device, dtype=torch.float32)
    cu_seqlens = torch.tensor([0, 1, 2, 2, 2], device=device, dtype=torch.int32)
    # Two requests may share a cached source page but must write independent
    # copy-on-write destinations.
    read_indices = torch.tensor([0, 0, -1, -1], device=device, dtype=torch.int32)
    write_indices = torch.tensor([2, 3, -1, -1], device=device, dtype=torch.int32)

    gathered = initial_pool[read_indices[:active].to(torch.int64)].contiguous()
    expected_out, expected_state = kda_recurrent(
        q[:active],
        k[:active],
        v[:active],
        raw_g[:active],
        beta[:active],
        gathered,
        a_log,
        dt_bias,
        cu_seqlens=torch.arange(active + 1, device=device, dtype=torch.int32),
        state_indices=torch.arange(active, device=device, dtype=torch.int32),
    )

    actual_out = kda_recurrent_decode(
        q,
        k,
        v,
        raw_g,
        beta,
        state_pool,
        a_log,
        dt_bias,
        cu_seqlens=cu_seqlens,
        read_indices=read_indices,
        write_indices=write_indices,
    )
    torch.testing.assert_close(
        actual_out[:active].float(), expected_out.float(), atol=5e-2, rtol=5e-2
    )
    torch.testing.assert_close(
        actual_out[active:], torch.zeros_like(actual_out[active:])
    )
    torch.testing.assert_close(
        state_pool[write_indices[:active].to(torch.int64)],
        expected_state,
        atol=5e-2,
        rtol=5e-2,
    )

    state_pool.copy_(initial_pool)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_out = kda_recurrent_decode(
            q,
            k,
            v,
            raw_g,
            beta,
            state_pool,
            a_log,
            dt_bias,
            cu_seqlens=cu_seqlens,
            read_indices=read_indices,
            write_indices=write_indices,
        )
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        captured_out[:active].float(), expected_out.float(), atol=5e-2, rtol=5e-2
    )


def test_kda_paged_decode_dispatches_to_fla_kernel_on_amd() -> None:
    """AMD dispatch must select the portable FLA-derived indexed kernel."""
    if not current_platform().is_amd:
        pytest.skip("AMD KDA dispatch test")

    device = "cuda"
    torch.manual_seed(13)
    tokens, heads, dim = 3, 2, 8
    q = torch.randn(1, tokens, heads, dim, device=device, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    raw_g = torch.randn_like(q)
    beta = torch.randn(1, tokens, heads, device=device, dtype=torch.bfloat16)
    state_pool = torch.randn(6, heads, dim, dim, device=device, dtype=torch.float32)
    expected_pool = state_pool.clone()
    a_log = torch.randn(heads, device=device, dtype=torch.float32)
    dt_bias = torch.randn(heads, dim, device=device, dtype=torch.float32)
    cu_seqlens = torch.arange(tokens + 1, device=device, dtype=torch.int32)
    read_indices = torch.tensor([0, 1, 1], device=device, dtype=torch.int32)
    write_indices = torch.tensor([2, 3, 4], device=device, dtype=torch.int32)

    expected_out = triton_nvidia_kda_paged_decode(
        q,
        k,
        v,
        raw_g,
        beta,
        a_log,
        dt_bias,
        state_pool=expected_pool,
        cu_seqlens=cu_seqlens,
        read_indices=read_indices,
        write_indices=write_indices,
        lower_bound=-5.0,
    )
    actual_out = kda_paged_decode(
        q,
        k,
        v,
        raw_g,
        beta,
        a_log,
        dt_bias,
        state_pool=state_pool,
        read_indices=read_indices,
        write_indices=write_indices,
        cu_seqlens=cu_seqlens,
    )

    torch.testing.assert_close(actual_out, expected_out)
    torch.testing.assert_close(state_pool, expected_pool)


def test_kda_paged_decode_uses_fla_kernel_for_compound_decode_on_amd() -> None:
    """The FLA-derived AMD dispatch must preserve packed compound decode."""
    if not current_platform().is_amd:
        pytest.skip("AMD KDA dispatch test")

    device = "cuda"
    torch.manual_seed(19)
    tokens, heads, dim = 3, 2, 8
    q = torch.randn(1, tokens, heads, dim, device=device, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    raw_g = torch.randn_like(q)
    beta = torch.randn(1, tokens, heads, device=device, dtype=torch.bfloat16)
    state_pool = torch.randn(6, heads, dim, dim, device=device, dtype=torch.float32)
    expected_pool = state_pool.clone()
    a_log = torch.randn(heads, device=device, dtype=torch.float32)
    dt_bias = torch.randn(heads, dim, device=device, dtype=torch.float32)
    cu_seqlens = torch.tensor([0, 2, 3], device=device, dtype=torch.int32)
    read_indices = torch.tensor([0, 1], device=device, dtype=torch.int32)
    write_indices = torch.tensor([4, 5], device=device, dtype=torch.int32)

    expected_out = triton_nvidia_kda_paged_decode(
        q,
        k,
        v,
        raw_g,
        beta,
        a_log,
        dt_bias,
        state_pool=expected_pool,
        cu_seqlens=cu_seqlens,
        read_indices=read_indices,
        write_indices=write_indices,
        lower_bound=-5.0,
    )
    actual_out = kda_paged_decode(
        q,
        k,
        v,
        raw_g,
        beta,
        a_log,
        dt_bias,
        state_pool=state_pool,
        read_indices=read_indices,
        write_indices=write_indices,
        cu_seqlens=cu_seqlens,
    )

    torch.testing.assert_close(
        actual_out.float(), expected_out.float(), atol=5e-2, rtol=5e-2
    )
    torch.testing.assert_close(
        state_pool,
        expected_pool,
        atol=5e-2,
        rtol=5e-2,
    )


def test_kda_paged_prefill_dispatches_to_fla_kernel_on_amd() -> None:
    """AMD prefill must select the portable FLA-derived chunk kernel."""
    if not current_platform().is_amd:
        pytest.skip("AMD KDA dispatch test")

    device = "cuda"
    torch.manual_seed(29)
    tokens, heads, dim = 5, 2, 8
    q = torch.randn(1, tokens, heads, dim, device=device, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    raw_g = torch.randn_like(q)
    beta = torch.randn(1, tokens, heads, device=device, dtype=torch.bfloat16)
    initial_state = torch.randn(2, heads, dim, dim, device=device, dtype=torch.float32)
    a_log = torch.randn(heads, device=device, dtype=torch.float32)
    dt_bias = torch.randn(heads, dim, device=device, dtype=torch.float32)
    cu_seqlens = torch.tensor([0, 3, 5], device=device, dtype=torch.int32)

    expected = triton_nvidia_kda_paged_prefill(
        q=q,
        k=k,
        v=v,
        g_raw=raw_g,
        beta_logits=beta,
        A_log=a_log,
        dt_bias=dt_bias,
        initial_state=initial_state.clone(),
        cu_seqlens=cu_seqlens,
        lower_bound=-5.0,
    )
    actual = kda_paged_prefill(
        q,
        k,
        v,
        raw_g,
        beta,
        a_log,
        dt_bias,
        initial_state=initial_state.clone(),
        cu_seqlens=cu_seqlens,
    )

    torch.testing.assert_close(actual.out, expected.out)
    torch.testing.assert_close(actual.final_state, expected.final_state)


def test_kda_recurrent_decode_updates_page_strided_state_pool() -> None:
    """Fused decode supports FlatKV's padded physical-page stride."""

    device = "cuda"
    torch.manual_seed(17)
    tokens, active, heads, dim = 4, 2, 2, 8
    state_elements = heads * dim * dim
    raw_pool = torch.randn(4, state_elements + 17, device=device, dtype=torch.float32)
    state_pool = raw_pool[:, :state_elements].view(4, heads, dim, dim)
    assert not state_pool.is_contiguous()
    initial_pool = state_pool.clone()

    q = torch.randn(tokens, heads, dim, device=device, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    raw_g = torch.randn_like(q)
    beta = torch.randn(tokens, heads, device=device, dtype=torch.bfloat16)
    a_log = torch.randn(heads, device=device, dtype=torch.float32)
    dt_bias = torch.randn(heads, dim, device=device, dtype=torch.float32)
    cu_seqlens = torch.tensor([0, 1, 2, 2, 2], device=device, dtype=torch.int32)
    read_indices = torch.tensor([0, 0, -1, -1], device=device, dtype=torch.int32)
    write_indices = torch.tensor([2, 3, -1, -1], device=device, dtype=torch.int32)

    gathered = initial_pool[read_indices[:active].to(torch.int64)].contiguous()
    expected_out, expected_state = kda_recurrent(
        q[:active],
        k[:active],
        v[:active],
        raw_g[:active],
        beta[:active],
        gathered,
        a_log,
        dt_bias,
        cu_seqlens=torch.arange(active + 1, device=device, dtype=torch.int32),
        state_indices=torch.arange(active, device=device, dtype=torch.int32),
    )

    actual_out = kda_recurrent_decode(
        q,
        k,
        v,
        raw_g,
        beta,
        state_pool,
        a_log,
        dt_bias,
        cu_seqlens=cu_seqlens,
        read_indices=read_indices,
        write_indices=write_indices,
    )

    torch.testing.assert_close(
        actual_out[:active].float(), expected_out.float(), atol=5e-2, rtol=5e-2
    )
    torch.testing.assert_close(
        actual_out[active:], torch.zeros_like(actual_out[active:])
    )
    torch.testing.assert_close(
        state_pool[write_indices[:active].to(torch.int64)],
        expected_state,
        atol=5e-2,
        rtol=5e-2,
    )
