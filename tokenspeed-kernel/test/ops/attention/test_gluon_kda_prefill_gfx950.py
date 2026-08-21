# Copyright (c) 2026 LightSeek Foundation

from __future__ import annotations

from itertools import accumulate

import pytest
import torch
from utils import is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required for Gluon KDA tests", allow_module_level=True)


from tokenspeed_kernel_amd.ops.gfx950.attention.kda.prefill import (  # noqa: E402
    gluon_kda_paged_prefill_gfx950,
)


def _reference_sequence(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    beta_logits: torch.Tensor,
    state: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    q = q.float()
    k = k.float()
    v = v.float()
    q *= torch.rsqrt(torch.sum(q * q, dim=-1, keepdim=True) + 1e-6)
    q *= q.shape[-1] ** -0.5
    k *= torch.rsqrt(torch.sum(k * k, dim=-1, keepdim=True) + 1e-6)
    gate_input = raw_g.float() + dt_bias
    if lower_bound is None:
        gate = -a_log.exp()[None, :, None] * torch.nn.functional.softplus(gate_input)
    else:
        gate = lower_bound * torch.sigmoid(a_log.exp()[None, :, None] * gate_input)
    beta = beta_logits.float().sigmoid()

    outputs = []
    # V-major: state is [heads, V, K] (value-major), matching gfx950's
    # physical recurrent-state pool layout.
    state = state.float().clone()
    for token in range(q.shape[0]):
        state *= gate[token].exp()[:, None, :]
        prediction = torch.einsum("hvk,hk->hv", state, k[token])
        delta = beta[token, :, None] * (v[token] - prediction)
        state += torch.einsum("hv,hk->hvk", delta, k[token])
        outputs.append(torch.einsum("hvk,hk->hv", state, q[token]))
    if not outputs:
        return v.new_empty((0, *v.shape[1:]), dtype=torch.float32), state
    return torch.stack(outputs), state


@pytest.mark.parametrize("lower_bound", [-5.0, None])
def test_kda_prefill_matches_packed_recurrent_reference(
    lower_bound: float | None,
) -> None:
    torch.manual_seed(31)
    lengths = [0, 1, 15, 16, 17, 63, 64, 65]
    total_tokens = sum(lengths)
    heads = 2
    dim = 128
    q = torch.randn(1, total_tokens, heads, dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    raw_g = torch.randn_like(q)
    beta = torch.randn(1, total_tokens, heads, device="cuda", dtype=torch.bfloat16)
    initial_state = torch.randn(
        len(lengths), heads, dim, dim, device="cuda", dtype=torch.float32
    )
    a_log = torch.randn(heads, device="cuda", dtype=torch.float32) * 0.1 - 2.0
    dt_bias = torch.randn(heads, dim, device="cuda", dtype=torch.float32)
    cu_seqlens = torch.tensor(
        [0, *accumulate(lengths)], device="cuda", dtype=torch.int32
    )

    expected_outputs = []
    expected_states = []
    for sequence, (begin, end) in enumerate(
        zip(cu_seqlens.tolist(), cu_seqlens.tolist()[1:])
    ):
        output, state = _reference_sequence(
            q[0, begin:end],
            k[0, begin:end],
            v[0, begin:end],
            raw_g[0, begin:end],
            beta[0, begin:end],
            initial_state[sequence],
            a_log,
            dt_bias,
            lower_bound,
        )
        expected_outputs.append(output)
        expected_states.append(state)

    actual_output, actual_state = gluon_kda_paged_prefill_gfx950(
        q,
        k,
        v,
        raw_g,
        beta,
        a_log,
        dt_bias,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        lower_bound=lower_bound,
    )

    torch.testing.assert_close(
        actual_output[0].float(),
        torch.cat(expected_outputs),
        atol=2e-3,
        rtol=2e-2,
    )
    torch.testing.assert_close(
        actual_state,
        torch.stack(expected_states),
        atol=4e-3,
        rtol=2e-2,
    )
    assert actual_state.dtype == torch.float32
    assert actual_state.shape == (len(lengths), heads, dim, dim)
    assert actual_state.stride()[-2:] == (dim, 1)
