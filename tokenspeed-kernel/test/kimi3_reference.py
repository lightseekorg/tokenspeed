"""Torch references shared by the Kimi K3 MXFP4 tests."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def situ_and_mul(
    x: torch.Tensor,
    *,
    beta: float = 4.0,
    linear_beta: float | None = 25.0,
) -> torch.Tensor:
    """Moonshot SiTU gated activation."""

    if x.shape[-1] % 2:
        raise ValueError("SiTU input width must be even")
    gate, up = x.float().chunk(2, dim=-1)
    gate = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    return (gate * up).to(x.dtype)


def kda_gate(
    raw_g: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    lower_bound: float | None = -5.0,
) -> torch.Tensor:
    """Reference KDA log-decay gate."""

    if raw_g.ndim != 3:
        raise ValueError("raw_g must be [T, H, D]")
    heads = raw_g.shape[1]
    if a_log.shape != (heads,):
        raise ValueError("a_log must have one value per local head")
    if dt_bias.shape != raw_g.shape[1:]:
        raise ValueError("dt_bias must be [H, D]")

    gate_input = raw_g.float() + dt_bias.float()[None, :, :]
    if lower_bound is not None:
        return lower_bound * torch.sigmoid(
            torch.exp(a_log.float())[None, :, None] * gate_input
        )
    return -torch.exp(a_log.float())[None, :, None] * F.softplus(gate_input)


def kda_recurrent(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    lower_bound: float | None = -5.0,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sequential KDA oracle for both prefill and decode."""

    if q.shape != k.shape or q.shape != raw_g.shape:
        raise ValueError("q, k, and raw_g must have matching shapes")
    tokens, heads, key_dim = q.shape
    if v.shape[:2] != (tokens, heads):
        raise ValueError("v must have the same token and head dimensions as q")
    value_dim = v.shape[-1]
    if state.shape != (heads, value_dim, key_dim):
        raise ValueError("invalid KDA state shape")
    if beta.shape != (tokens, heads):
        raise ValueError("beta must be [T, H]")

    gates = kda_gate(raw_g, a_log, dt_bias, lower_bound=lower_bound)
    running = state.float().clone()
    outputs = []
    scale = key_dim**-0.5
    for token_idx in range(tokens):
        q_t = F.normalize(q[token_idx].float(), dim=-1, eps=eps) * scale
        k_t = F.normalize(k[token_idx].float(), dim=-1, eps=eps)
        v_t = v[token_idx].float()
        beta_t = torch.sigmoid(beta[token_idx].float())

        running = running * torch.exp(gates[token_idx])[:, None, :]
        prediction = torch.einsum("hvk,hk->hv", running, k_t)
        delta = beta_t[:, None] * (v_t - prediction)
        running = running + torch.einsum("hv,hk->hvk", delta, k_t)
        outputs.append(torch.einsum("hvk,hk->hv", running, q_t))

    return torch.stack(outputs).to(q.dtype), running.to(state.dtype)


_E2M1_VALUES = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)


def dequantize_mxfp4(
    packed: torch.Tensor,
    scales: torch.Tensor,
    *,
    group_size: int = 32,
) -> torch.Tensor:
    """Decode packed OCP E2M1 values with one UE8M0 scale per group."""

    if packed.dtype != torch.uint8 or scales.dtype != torch.uint8:
        raise ValueError("MXFP4 payload and scales must use uint8 storage")
    low = packed & 0x0F
    high = packed >> 4
    codes = torch.stack((low, high), dim=-1).flatten(-2)
    values = _E2M1_VALUES.to(packed.device)[codes.long()]
    scale_values = torch.exp2(scales.float() - 127.0)
    expanded_scales = scale_values.repeat_interleave(group_size, dim=-1)
    if values.shape != expanded_scales.shape:
        raise ValueError("packed values and scale groups describe different shapes")
    return values * expanded_scales


def _mxfp4_linear(
    x: torch.Tensor,
    packed_weight: torch.Tensor,
    scales: torch.Tensor,
    *,
    activation_dtype: torch.dtype,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    x = x.to(activation_dtype).float()
    return F.linear(x, dequantize_mxfp4(packed_weight, scales)).to(output_dtype)


def mxfp4_moe_reference(
    hidden_states: torch.Tensor,
    w13_packed: torch.Tensor,
    w13_scales: torch.Tensor,
    w2_packed: torch.Tensor,
    w2_scales: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    activation_dtype: torch.dtype,
    situ_beta: float = 1.0,
    situ_linear_beta: float | None = None,
) -> torch.Tensor:
    """Routed MXFP4 experts with explicit activation-quantization boundaries."""

    combined = torch.zeros_like(hidden_states, dtype=torch.float32)
    for expert_id in range(w13_packed.shape[0]):
        routes = (topk_ids == expert_id).nonzero(as_tuple=False)
        if not routes.numel():
            continue
        token_ids, slot_ids = routes.unbind(dim=1)
        gate_up = _mxfp4_linear(
            hidden_states.index_select(0, token_ids),
            w13_packed[expert_id],
            w13_scales[expert_id],
            activation_dtype=activation_dtype,
            output_dtype=hidden_states.dtype,
        )
        output = _mxfp4_linear(
            situ_and_mul(
                gate_up,
                beta=situ_beta,
                linear_beta=situ_linear_beta,
            ),
            w2_packed[expert_id],
            w2_scales[expert_id],
            activation_dtype=activation_dtype,
            output_dtype=hidden_states.dtype,
        )
        route_weights = topk_weights[token_ids, slot_ids].float().unsqueeze(-1)
        combined.index_add_(0, token_ids, output.float() * route_weights)
    return combined.to(hidden_states.dtype)
