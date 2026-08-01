"""KDA chunk-prefill and dispatch coverage."""

from __future__ import annotations

import pytest
import torch
from kimi3_reference import kda_gate
from kimi3_reference import kda_recurrent as reference_kda_recurrent
from tokenspeed_kernel.ops.attention import (
    _attention_format_signature,
    kda_chunk_prefill,
    kda_paged_decode,
)
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.selection import NoKernelFoundError, select_kernel


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


def test_kda_chunk_prefill_uses_canonical_k_major_state() -> None:
    """Native prefill preserves the public [N,H,K,V] state layout."""
    device = "cuda"
    torch.manual_seed(3)
    tokens, heads, key_dim, value_dim = 65, 2, 16, 4
    q = torch.randn(tokens, heads, key_dim, device=device, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(
        tokens,
        heads,
        value_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    raw_g = torch.randn_like(q)
    beta = torch.randn(tokens, heads, device=device, dtype=torch.bfloat16)
    state = torch.randn(
        1,
        heads,
        key_dim,
        value_dim,
        device=device,
        dtype=torch.float32,
    )
    a_log = torch.randn(heads, device=device, dtype=torch.float32)
    dt_bias = torch.randn(heads, key_dim, device=device, dtype=torch.float32)

    expected_out, expected_state = reference_kda_recurrent(
        q,
        k,
        v,
        raw_g,
        beta,
        state[0].transpose(-1, -2).contiguous(),
        a_log,
        dt_bias,
    )
    actual_out, actual_state = kda_chunk_prefill(
        q,
        k,
        v,
        raw_g,
        beta,
        state,
        a_log,
        dt_bias,
        block_value=8,
    )

    torch.testing.assert_close(
        actual_out.float(), expected_out.float(), atol=6e-2, rtol=6e-2
    )
    torch.testing.assert_close(
        actual_state,
        expected_state.transpose(-1, -2).unsqueeze(0),
        atol=6e-2,
        rtol=6e-2,
    )


@pytest.mark.parametrize("lower_bound", [-5.0, None])
@pytest.mark.parametrize("value_dim", [8, 5])
def test_kda_paged_decode_defaults_to_specialized_kernel_on_amd(
    lower_bound: float | None,
    value_dim: int,
) -> None:
    """AMD single-token dispatch must select Gluon and preserve native math."""
    if not current_platform().is_cdna4:
        pytest.skip("gfx950 KDA dispatch test")

    device = "cuda"
    torch.manual_seed(13)
    tokens, heads, key_dim = 3, 2, 8
    q = torch.randn(1, tokens, heads, key_dim, device=device, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(1, tokens, heads, value_dim, device=device, dtype=torch.bfloat16)
    raw_g = torch.randn_like(q)
    beta = torch.randn(1, tokens, heads, device=device, dtype=torch.bfloat16)
    state_pool = torch.randn(
        6,
        heads,
        key_dim,
        value_dim,
        device=device,
        dtype=torch.float32,
    )
    initial_pool = state_pool.clone()
    expected_pool = state_pool.clone()
    a_log = torch.randn(heads, device=device, dtype=torch.float32)
    dt_bias = torch.randn(heads, key_dim, device=device, dtype=torch.float32)
    cu_seqlens = torch.arange(tokens + 1, device=device, dtype=torch.int32)
    read_indices = torch.tensor([0, 1, 1], device=device, dtype=torch.int32)
    write_indices = torch.tensor([2, 3, 4], device=device, dtype=torch.int32)

    selected = select_kernel(
        "attention",
        "kda_paged_decode",
        _attention_format_signature(q=q, k=k, v=v),
        traits={"indexed_state": True, "single_token": True},
    )
    assert selected.name == "gluon_kda_paged_decode_gfx950"

    expected_out = []
    for row in range(tokens):
        out, final_state = reference_kda_recurrent(
            q[0, row : row + 1],
            k[0, row : row + 1],
            v[0, row : row + 1],
            raw_g[0, row : row + 1],
            beta[0, row : row + 1],
            initial_pool[read_indices[row].long()].transpose(-1, -2),
            a_log,
            dt_bias,
            lower_bound=lower_bound,
        )
        expected_out.append(out[0])
        expected_pool[write_indices[row].long()] = final_state.transpose(-1, -2)
    expected_out = torch.stack(expected_out).unsqueeze(0)
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
        lower_bound=lower_bound,
    )

    torch.testing.assert_close(
        actual_out.float(), expected_out.float(), atol=2e-2, rtol=2e-2
    )
    torch.testing.assert_close(state_pool, expected_pool, atol=2e-5, rtol=2e-5)


def test_kda_paged_decode_graph_padding_and_page_stride() -> None:
    """Gluon decode supports padded graph batches and strided state pages."""
    if not current_platform().is_cdna4:
        pytest.skip("gfx950 KDA dispatch test")

    torch.manual_seed(23)
    batch, active, heads, key_dim, value_dim = 4, 2, 2, 8, 5
    state_elements = heads * key_dim * value_dim
    raw_pool = torch.randn(7, state_elements + 11, device="cuda", dtype=torch.float32)
    state_pool = raw_pool[:, :state_elements].view(7, heads, key_dim, value_dim)
    q = torch.randn(1, batch, heads, key_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(1, batch, heads, value_dim, device="cuda", dtype=torch.bfloat16)
    raw_g = torch.randn_like(q)
    beta = torch.randn(1, batch, heads, device="cuda", dtype=torch.bfloat16)
    a_log = torch.randn(heads, device="cuda", dtype=torch.float32)
    dt_bias = torch.randn(heads, key_dim, device="cuda", dtype=torch.float32)
    read_indices = torch.tensor([1, 2, -1, -1], device="cuda", dtype=torch.int32)
    write_indices = torch.tensor([3, 4, -1, -1], device="cuda", dtype=torch.int32)
    cu_seqlens = torch.tensor([0, 1, 2, 2, 2], device="cuda", dtype=torch.int32)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = kda_paged_decode(
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
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(
        captured[:, active:],
        torch.zeros_like(captured[:, active:]),
    )


def test_kda_paged_decode_does_not_select_nvidia_kernel_on_amd() -> None:
    """The NVIDIA portable adapter remains outside the AMD dispatch surface."""
    if not current_platform().is_amd:
        pytest.skip("AMD KDA dispatch test")

    q = torch.randn(1, 3, 2, 8, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    with pytest.raises(NoKernelFoundError):
        select_kernel(
            "attention",
            "kda_paged_decode",
            _attention_format_signature(q=q, k=k, v=v),
            traits={"indexed_state": True, "single_token": False},
        )
