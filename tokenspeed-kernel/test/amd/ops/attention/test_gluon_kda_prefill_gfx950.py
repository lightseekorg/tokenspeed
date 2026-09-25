# Copyright (c) 2026 LightSeek Foundation

from __future__ import annotations

from itertools import accumulate

import pytest
import torch
from utils import is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required for Gluon KDA tests", allow_module_level=True)


from tokenspeed_kernel_amd.ops.gfx950.attention.kda.prefill import (  # noqa: E402
    launch_gluon_kda_paged_prefill_gfx950,
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

    actual_output, actual_state = launch_gluon_kda_paged_prefill_gfx950(
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


def _capacity_inputs(capacity: int, sequences: int, heads: int = 4, dim: int = 128):
    q = torch.randn(1, capacity, heads, dim, device="cuda", dtype=torch.bfloat16)
    beta = torch.randn(1, capacity, heads, device="cuda", dtype=torch.bfloat16)
    return dict(
        q=q,
        k=torch.randn_like(q),
        v=torch.randn_like(q),
        g_raw=torch.randn_like(q),
        beta_logits=beta,
        A_log=torch.randn(heads, device="cuda", dtype=torch.float32) * 0.1 - 2.0,
        dt_bias=torch.randn(heads, dim, device="cuda", dtype=torch.float32),
        initial_state=torch.randn(
            sequences, heads, dim, dim, device="cuda", dtype=torch.float32
        ),
    )


def _exact(inputs: dict, lengths: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    live = sum(lengths)
    sliced = {
        name: tensor[:, :live] if tensor.shape[0] == 1 else tensor
        for name, tensor in inputs.items()
        if name not in ("A_log", "dt_bias", "initial_state")
    }
    return launch_gluon_kda_paged_prefill_gfx950(
        **sliced,
        A_log=inputs["A_log"],
        dt_bias=inputs["dt_bias"],
        initial_state=inputs["initial_state"],
        cu_seqlens=torch.tensor(
            [0, *accumulate(lengths)], device="cuda", dtype=torch.int32
        ),
        lower_bound=-5.0,
    )


def test_kda_prefill_capacity_is_selected() -> None:
    from tokenspeed_kernel.ops.attention.kda import kda_prefill_capacity_supported

    assert kda_prefill_capacity_supported(torch.bfloat16, solution=None)


@pytest.mark.parametrize("lengths", [[300], [1, 64, 65, 129], [2045, 1, 1]])
def test_kda_prefill_capacity_padding_matches_exact(lengths: list[int]) -> None:
    from tokenspeed_kernel.ops.attention.kda import (
        KdaPrefillCapacity,
        kda_paged_prefill,
    )

    torch.manual_seed(7)
    capacity = 2048 if sum(lengths) <= 2048 else 4096
    inputs = _capacity_inputs(capacity, len(lengths))
    expected_output, expected_state = _exact(inputs, lengths)
    # Padding rows are undefined in serving; NaN proves they are never read.
    live = sum(lengths)
    for name in ("q", "k", "v", "g_raw", "beta_logits"):
        inputs[name][:, live:] = float("nan")
    boundaries = [0, *accumulate(lengths)]
    result = kda_paged_prefill(
        **inputs,
        cu_seqlens=torch.tensor(boundaries, device="cuda", dtype=torch.int64),
        cu_seqlens_cpu=torch.tensor(boundaries, dtype=torch.int64),
        capacity=KdaPrefillCapacity(capacity, len(lengths)),
        inputs_packed=False,
        lower_bound=-5.0,
        recurrent_layout="v_major",
    )
    torch.testing.assert_close(result.out[:, :live], expected_output, rtol=0, atol=0)
    torch.testing.assert_close(result.final_state, expected_state, rtol=0, atol=0)


def test_kda_prefill_graph_replays_changing_boundaries() -> None:
    torch.manual_seed(11)
    capacity = 1024
    inputs = _capacity_inputs(capacity, 3)
    cu_seqlens = torch.tensor([0, 1, 2, 3], device="cuda", dtype=torch.int32)
    captured = dict(inputs, cu_seqlens=cu_seqlens, lower_bound=-5.0)

    # Warm the JIT outside capture; capture fails on any host synchronization.
    launch_gluon_kda_paged_prefill_gfx950(**captured)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output, state = launch_gluon_kda_paged_prefill_gfx950(**captured)

    for lengths in ([1000, 20, 4], [64, 64, 64], [1, 1, 1022], [5, 300, 1]):
        cu_seqlens.copy_(
            torch.tensor([0, *accumulate(lengths)], device="cuda", dtype=torch.int32)
        )
        graph.replay()
        expected_output, expected_state = _exact(inputs, lengths)
        live = sum(lengths)
        torch.testing.assert_close(output[:, :live], expected_output, rtol=0, atol=0)
        torch.testing.assert_close(state, expected_state, rtol=0, atol=0)
