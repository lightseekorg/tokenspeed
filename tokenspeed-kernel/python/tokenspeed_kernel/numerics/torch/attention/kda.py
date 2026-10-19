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

"""PyTorch references for ``tokenspeed_kernel.ops.attention.kda`` (Kimi Delta
Attention)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures


def kda_gate(
    raw_g: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    lower_bound: float | None,
) -> torch.Tensor:
    """KDA log-decay gate in FP32.

    ``lower_bound * sigmoid(exp(a_log) * (raw_g + dt_bias))`` with a lower
    bound, else ``-exp(a_log) * softplus(raw_g + dt_bias)``.

    Args:
        raw_g: ``[T, H, D]`` gate projections.
        a_log: ``[H]`` per-head log decay rates.
        dt_bias: ``[H, D]`` gate bias.
        lower_bound: Bound of the sigmoid form, or None for the softplus form.
    """
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


@register_kernel(
    "attention",
    "kda_recurrent",
    name="torch_kda_recurrent",
    solution="torch",
    signatures=format_signatures(
        ("q", "k", "v"), "dense", {torch.bfloat16, torch.float16, torch.float32}
    ),
    priority=Priority.REFERENCE,
)
def torch_kda_recurrent(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    output_dtype: torch.dtype,
    lower_bound: float | None,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sequential KDA recurrence of one sequence, in FP32.

    Per token: L2-normalize q (scaled by ``D ** -0.5``) and k, decay the
    ``[H, V, K]`` state by ``exp(gate)``, then apply the delta rule with
    ``sigmoid(beta)`` and read the output with q.

    Args:
        q, k, raw_g: ``[T, H, K]``.
        v: ``[T, H, V]``.
        beta: ``[T, H]`` pre-sigmoid write strengths.
        state: ``[H, V, K]`` initial state.
        a_log, dt_bias: Gate parameters, see :func:`kda_gate`.
        output_dtype: Dtype of the returned outputs.
        lower_bound: Gate form, see :func:`kda_gate`.
        eps: L2-normalization epsilon.

    Returns:
        ``([T, H, V]`` outputs in ``output_dtype``, final state in
        ``state``'s dtype).
    """
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

    outputs, running = kda_delta_rule(
        F.normalize(q.float(), dim=-1, eps=eps) * key_dim**-0.5,
        F.normalize(k.float(), dim=-1, eps=eps),
        v,
        kda_gate(raw_g, a_log, dt_bias, lower_bound=lower_bound),
        torch.sigmoid(beta.float()),
        state,
    )
    return outputs.to(output_dtype), running.to(state.dtype)


def kda_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """KDA's per-channel gated delta rule over one sequence, in FP32.

    Per token: decay the value-major ``[H, V, K]`` state by ``exp(gate)`` per
    key channel, write ``beta * (v - S k)`` along ``k``, and read ``S q``.
    Pass ``q`` normalized and scaled and ``k`` normalized.

    Args:
        q, k, gate: ``[T, H, K]``.
        v: ``[T, H, V]``.
        beta: ``[T, H]`` write strengths (post-sigmoid).
        state: ``[H, V, K]`` initial state.

    Returns:
        FP32 ``(outputs [T, H, V], final state [H, V, K])``.
    """
    q, k, v, gate, beta = (x.float() for x in (q, k, v, gate, beta))
    running = state.float().clone()
    outputs = []
    for token in range(q.shape[0]):
        running = running * torch.exp(gate[token])[:, None, :]
        prediction = torch.einsum("hvk,hk->hv", running, k[token])
        delta = beta[token, :, None] * (v[token] - prediction)
        running = running + torch.einsum("hv,hk->hvk", delta, k[token])
        outputs.append(torch.einsum("hvk,hk->hv", running, q[token]))
    if not outputs:
        return v.new_empty((0, *v.shape[1:])), running
    return torch.stack(outputs), running


@register_kernel(
    "attention",
    "kda_conv_decode",
    name="torch_kda_conv_decode",
    solution="torch",
    signatures=format_signatures("qkv", "dense", {torch.bfloat16}),
    priority=Priority.REFERENCE,
)
def torch_kda_conv_decode(
    qkv: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_pool: torch.Tensor,
    state_pool: torch.Tensor,
    read_indices: torch.Tensor,
    gate_weight: torch.Tensor,
    gate_input: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: torch.Tensor,
    steps: torch.Tensor,
    *,
    tokens_per_request: int,
    num_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    lower_bound: float,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """``steps[i]`` ordinary KDA decode steps of request ``i``, in FP32.

    Each step slides the request's 4-tap causal conv window over its
    ``qkv`` row, applies SiLU, L2-normalizes the key heads, and applies the
    delta rule with the sigmoid-bounded gate ``lower_bound * sigmoid(exp(
    a_log) * (gate_weight @ gate_input + dt_bias))`` and ``sigmoid(beta)``.

    Args:
        qkv: ``[requests * tokens_per_request, 3 * heads * K]`` raw projections
            (q, k, v blocks); only k and v enter the state.
        conv_weight: ``[channels, 4]`` conv taps, oldest first.
        conv_pool: ``[pages, channels, 3]`` conv windows.
        state_pool: ``[pages, heads, V, K]`` value-major states.
        read_indices: ``[requests]`` pages holding each request's start.
        gate_weight: ``[heads * K, gate_dim]`` gate projection.
        gate_input: ``[tokens, gate_dim]``.
        a_log: ``[heads]``; dt_bias: ``[heads * K]``; beta: ``[tokens, heads]``.
        steps: ``[requests]`` steps to run.

    Returns:
        Per request, the final conv window ``[channels, 3]`` and key-major
        state ``[heads, K, V]``.
    """
    width = num_heads * head_k_dim
    conv_weight = conv_weight.float()
    gate_weight = gate_weight.float()
    exp_a = torch.exp(a_log)[:, None]
    windows, states = [], []
    for request in range(read_indices.numel()):
        page = int(read_indices[request])
        window = conv_pool[page].float()
        h = state_pool[page].transpose(-1, -2).float().clone()
        for step in range(int(steps[request])):
            token = request * tokens_per_request + step
            x = qkv[token].float()
            acc = (window * conv_weight[:, :3]).sum(-1) + x * conv_weight[:, 3]
            y = acc * torch.sigmoid(acc)
            window = torch.cat([window[:, 1:], x[:, None]], dim=1)
            key = y[width : 2 * width].view(num_heads, head_k_dim)
            value = y[2 * width :].view(num_heads, head_v_dim)
            key = key / torch.sqrt((key * key).sum(-1, keepdim=True) + 1e-6)
            gate = ((gate_weight @ gate_input[token].float()) + dt_bias).view(
                num_heads, head_k_dim
            )
            h = h * torch.exp(lower_bound * torch.sigmoid(exp_a * gate))[:, :, None]
            residual = value - torch.einsum("hkv,hk->hv", h, key)
            residual = residual * torch.sigmoid(beta[token].float())[:, None]
            h = h + torch.einsum("hk,hv->hkv", key, residual)
        windows.append(window)
        states.append(h)
    return windows, states


@register_kernel(
    "attention",
    "verify_state_blocks",
    name="torch_verify_state_blocks",
    solution="torch",
    signatures=format_signatures("table", "dense", {torch.int32}),
    priority=Priority.REFERENCE,
)
def torch_verify_state_blocks(table: torch.Tensor, *, batch_size: int) -> torch.Tensor:
    """The state page each verify row reads: column 0 when it holds a page,
    else column 1 of the two-slot table."""
    rows = table[:batch_size]
    return torch.where(rows[:, 0] > 0, rows[:, 0], rows[:, 1]).to(torch.int32)


@register_kernel(
    "attention",
    "commit_state_pages",
    name="torch_commit_state_pages",
    solution="torch",
    signatures=format_signatures("table", "dense", {torch.int32}),
    priority=Priority.REFERENCE,
)
def torch_commit_state_pages(
    accepted: torch.Tensor, table: torch.Tensor, *, bs: int, draft_tokens: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """The page each request commits to (column 1) and its accepted steps,
    clamped to ``[1, draft_tokens]``."""
    steps = accepted[:bs].to(torch.int64).clamp(min=1, max=draft_tokens)
    return table[:bs, 1].to(torch.int32), steps.to(torch.int32)
