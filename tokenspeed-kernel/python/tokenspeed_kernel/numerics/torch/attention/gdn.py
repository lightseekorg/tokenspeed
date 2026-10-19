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

"""PyTorch references for ``tokenspeed_kernel.ops.attention.gdn`` (gated
delta net)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures


def gdn_gate(
    a: torch.Tensor, dt_bias: torch.Tensor, A_log: torch.Tensor
) -> torch.Tensor:
    """Log-decay ``-exp(A_log) * softplus(a + dt_bias)`` (softplus threshold 20),
    in the inputs' promoted dtype."""
    return -torch.exp(A_log) * F.softplus(a + dt_bias)


def gdn_l2norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    """``x / sqrt(sum(x**2) + eps)`` over the last dimension."""
    return x / torch.sqrt(x.square().sum(-1, keepdim=True) + eps)


def gdn_l2norm_clamped(x: torch.Tensor, eps: float) -> torch.Tensor:
    """``x * rsqrt(max(sum(x**2), eps))`` over the last dimension in FP32,
    rounded to ``x``'s dtype."""
    x_float = x.float()
    return (
        x_float * torch.rsqrt(x_float.square().sum(dim=-1, keepdim=True).clamp_min(eps))
    ).to(x.dtype)


@register_kernel(
    "attention",
    "gdn_recurrence",
    name="torch_gdn_recurrence",
    solution="torch",
    signatures=format_signatures(
        ("q", "k", "v"),
        "dense",
        {torch.bfloat16, torch.float16, torch.float32, torch.float64},
    ),
    priority=Priority.REFERENCE,
)
def torch_gdn_recurrence(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    *,
    scale: float,
    parents: list[list[int]] | None,
    compute_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Token-by-token gated delta rule.

    Each node ``t`` starts from its parent's state (``parents[b][t]``, or the
    previous token when ``parents`` is None; ``-1`` is the initial state),
    decays it by ``exp(g)``, writes ``beta * (v - k @ S)`` along ``k``, and
    reads ``scale * q @ S``. Query/key heads are shared by groups of value
    heads. Pass ``q`` and ``k`` already normalized.

    Args:
        q, k: ``[B, T, H, K]``.
        v: ``[B, T, HV, V]``.
        g: ``[B, T, HV]`` log-decay, e.g. :func:`gdn_gate`.
        beta: ``[B, T, HV]`` write strength (post-sigmoid).
        initial_state: K-last ``[B, HV, V, K]``.
        scale: Output scale.
        parents: Per batch, the parent node of each node, or None for a chain.
        compute_dtype: FP32 or FP64 arithmetic.

    Returns:
        ``(out [B, T, HV, V], states [B, T, HV, V, K])`` in ``compute_dtype``,
        the K-last state after each node.
    """
    B, T, H, _ = q.shape
    HV = v.shape[2]
    q, k = (x.to(compute_dtype).repeat_interleave(HV // H, dim=2) for x in (q, k))
    v, g, beta = (x.to(compute_dtype) for x in (v, g, beta))
    start = initial_state.to(compute_dtype).transpose(-2, -1)  # [B, HV, K, V]
    out = torch.empty(B, T, HV, v.shape[-1], dtype=compute_dtype, device=q.device)
    states = torch.empty(B, T, *start.shape[1:], dtype=compute_dtype, device=q.device)
    for bi in range(B):
        for t in range(T):
            parent = t - 1 if parents is None else parents[bi][t]
            h = start[bi] if parent < 0 else states[bi, parent]
            h = h * torch.exp(g[bi, t])[:, None, None]
            delta = beta[bi, t][:, None] * (
                v[bi, t] - torch.einsum("hk,hkv->hv", k[bi, t], h)
            )
            states[bi, t] = h + k[bi, t][:, :, None] * delta[:, None, :]
            out[bi, t] = scale * torch.einsum("hk,hkv->hv", q[bi, t], states[bi, t])
    return out, states.transpose(-2, -1)


@register_kernel(
    "attention",
    "prepare_prefill_state_inputs",
    name="torch_prepare_prefill_state_inputs",
    solution="torch",
    signatures=format_signatures("src", "dense", {torch.int32, torch.int64}),
    priority=Priority.REFERENCE,
)
def torch_prepare_prefill_state_inputs(
    conv: torch.Tensor, state: torch.Tensor, src: torch.Tensor, dst: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Seed each prefill's conv window and recurrent state from its source.

    A request with history (``src > 0``) copies conv row ``src`` to ``dst``
    and reads state ``src``; one without keeps conv row ``dst`` and starts
    from a zero state. ``conv`` is updated in place.

    Returns:
        ``(initial states [requests, ...], has_history [requests])``.
    """
    history = src > 0
    safe = torch.where(history, src, dst).long()
    conv[dst.long()] = conv[safe]
    recurrent = state[safe]
    recurrent.masked_fill_(~history[:, None, None, None], 0)
    return recurrent, history


@register_kernel(
    "attention",
    "causal_conv1d_prefill_metadata",
    name="torch_causal_conv1d_prefill_metadata",
    solution="torch",
    signatures=format_signatures("lengths", "dense", {torch.int32, torch.int64}),
    priority=Priority.REFERENCE,
)
def torch_causal_conv1d_prefill_metadata(
    lengths: list[int], block_m: int
) -> tuple[list[int], list[int]]:
    """The (sequence, chunk) pair of every ``block_m``-token prefill chunk.

    Returns:
        ``(rows, chunks)``: the sequence and chunk index of each chunk, in
        sequence order.
    """
    counts = [(length + block_m - 1) // block_m for length in lengths]
    return (
        [row for row, count in enumerate(counts) for _ in range(count)],
        [chunk for count in counts for chunk in range(count)],
    )
