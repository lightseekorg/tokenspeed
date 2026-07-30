# Copyright (c) 2026 LightSeek Foundation

from __future__ import annotations

import pytest
import torch


def _is_gfx950() -> bool:
    if not torch.cuda.is_available():
        return False
    arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
    return "gfx950" in arch


if not _is_gfx950():
    pytest.skip("AMD GFX950 is required for Gluon KDA tests", allow_module_level=True)


from tokenspeed_kernel_amd.ops.attention.gluon.kda_decode_gfx950 import (  # noqa: E402
    kda_recurrent_decode_gfx950,
)


def _reference_step(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    q = q.float()
    k = k.float()
    v = v.float()
    g = g.float() + dt_bias
    q = torch.nn.functional.normalize(q, dim=-1) * q.shape[-1] ** -0.5
    k = torch.nn.functional.normalize(k, dim=-1)
    if lower_bound is None:
        log_decay = -a_log.exp()[:, None] * torch.nn.functional.softplus(g)
    else:
        log_decay = lower_bound * torch.sigmoid(a_log.exp()[:, None] * g)
    state = state.float() * log_decay.exp()[:, :, None]
    prediction = torch.einsum("hkv,hk->hv", state, k)
    delta = beta.float().sigmoid()[:, None] * (v - prediction)
    state = state + torch.einsum("hk,hv->hkv", k, delta)
    out = torch.einsum("hkv,hk->hv", state, q)
    return out, state


@pytest.mark.parametrize("lower_bound", [-5.0, None])
def test_kda_decode_matches_non_square_reference(lower_bound: float | None) -> None:
    torch.manual_seed(17)
    batch, heads, key_dim, value_dim = 3, 2, 8, 5
    q = torch.randn(1, batch, heads, key_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(1, batch, heads, value_dim, device="cuda", dtype=torch.bfloat16)
    g = torch.randn_like(q)
    beta = torch.randn(1, batch, heads, device="cuda", dtype=torch.bfloat16)
    a_log = torch.randn(heads, device="cuda", dtype=torch.float32)
    dt_bias = torch.randn(heads, key_dim, device="cuda", dtype=torch.float32)
    state_pool = torch.randn(
        2 * batch, heads, key_dim, value_dim, device="cuda", dtype=torch.float32
    )
    initial_pool = state_pool.clone()
    read_indices = torch.arange(batch, device="cuda", dtype=torch.int32)
    write_indices = read_indices + batch
    cu_seqlens = torch.arange(batch + 1, device="cuda", dtype=torch.int32)

    expected_out = []
    expected_states = []
    for row in range(batch):
        out, state = _reference_step(
            q[0, row],
            k[0, row],
            v[0, row],
            g[0, row],
            beta[0, row],
            initial_pool[row],
            a_log,
            dt_bias,
            lower_bound,
        )
        expected_out.append(out)
        expected_states.append(state)

    actual = kda_recurrent_decode_gfx950(
        q,
        k,
        v,
        g,
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
        actual[0].float(),
        torch.stack(expected_out),
        atol=2e-2,
        rtol=2e-2,
    )
    torch.testing.assert_close(
        state_pool[write_indices.long()],
        torch.stack(expected_states),
        atol=2e-5,
        rtol=2e-5,
    )


def test_kda_decode_graph_padding_and_page_stride() -> None:
    torch.manual_seed(23)
    batch, active, heads, key_dim, value_dim = 4, 2, 2, 8, 5
    state_elements = heads * key_dim * value_dim
    raw_pool = torch.randn(7, state_elements + 11, device="cuda", dtype=torch.float32)
    state_pool = raw_pool[:, :state_elements].view(7, heads, key_dim, value_dim)
    q = torch.randn(1, batch, heads, key_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(1, batch, heads, value_dim, device="cuda", dtype=torch.bfloat16)
    g = torch.randn_like(q)
    beta = torch.randn(1, batch, heads, device="cuda", dtype=torch.bfloat16)
    a_log = torch.randn(heads, device="cuda", dtype=torch.float32)
    dt_bias = torch.randn(heads, key_dim, device="cuda", dtype=torch.float32)
    read_indices = torch.tensor([1, 2, -1, -1], device="cuda", dtype=torch.int32)
    write_indices = torch.tensor([3, 4, -1, -1], device="cuda", dtype=torch.int32)
    cu_seqlens = torch.tensor([0, 1, 2, 2, 2], device="cuda", dtype=torch.int32)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = kda_recurrent_decode_gfx950(
            q,
            k,
            v,
            g,
            beta,
            a_log,
            dt_bias,
            state_pool=state_pool,
            read_indices=read_indices,
            write_indices=write_indices,
            cu_seqlens=cu_seqlens,
            lower_bound=-5.0,
        )
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(
        captured[:, active:],
        torch.zeros_like(captured[:, active:]),
    )
