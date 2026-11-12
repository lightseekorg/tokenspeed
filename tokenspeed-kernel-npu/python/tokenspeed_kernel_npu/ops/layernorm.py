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

"""Ascend normalization operators."""

from __future__ import annotations

import torch
import torch_npu


def rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    residual: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Apply RMSNorm, optionally adding and updating a residual tensor."""
    if x.shape[0] == 0:
        result = x if out is None else out
        return (result, residual) if residual is not None else result
    if residual is not None:
        result, _, residual_out = torch_npu.npu_add_rms_norm(
            x, residual, weight, epsilon=eps
        )
        residual.copy_(residual_out)
        return result, residual
    result, _ = torch_npu.npu_rms_norm(x, weight, epsilon=eps)
    if out is not None:
        out.copy_(result)
        result = out
    return result


def qk_rmsnorm(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    enable_pdl: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply per-head RMSNorm to query and key tensors."""
    del enable_pdl
    head_dim = q_weight.numel()
    if k_weight.numel() != head_dim:
        raise ValueError("q and k weights must have the same head dimension")
    if q.shape[-1] % head_dim or k.shape[-1] % head_dim:
        raise ValueError("q and k widths must be divisible by the head dimension")

    q_shape, k_shape = q.shape, k.shape
    q_heads = q.reshape(-1, q.shape[-1] // head_dim, head_dim)
    k_heads = k.reshape(-1, k.shape[-1] // head_dim, head_dim)
    q_scale = torch.rsqrt(q_heads.float().square().mean(-1, keepdim=True) + eps)
    k_scale = torch.rsqrt(k_heads.float().square().mean(-1, keepdim=True) + eps)
    q_out = (q_heads * q_scale.to(q.dtype) * q_weight).reshape(q_shape)
    k_out = (k_heads * k_scale.to(k.dtype) * k_weight).reshape(k_shape)
    return q_out, k_out


def gemma_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    residual: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Gemma RMSNorm (multiplier ``1 + weight``) through the native RMSNorm ops.

    Args:
        x: ``[..., hidden]`` input; rows may be strided, columns dense.
        weight: Gemma checkpoint weight offset, ``[hidden]``.
        eps: Epsilon added to the mean square.
        residual: Optional ``x``-shaped residual stream; when given the FP32 sum
            ``x + residual`` is normalized and its rounded copy becomes the new
            residual.

    Returns:
        The normalized tensor, or ``(normalized, new_residual)``.
    """
    scale = (weight.float() + 1.0).to(weight.dtype)
    return rmsnorm(x, scale, eps, residual)


def gated_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    z: torch.Tensor | None,
    *,
    eps: float,
    group_size: int | None,
    norm_before_gate: bool,
    sigmoid_gate: bool,
    fp8_scale: torch.Tensor | None,
    weights_independent: bool,
) -> torch.Tensor:
    """Grouped gated RMSNorm composed from the native RMSNorm op and elementwise math.

    The native op normalizes the whole last dimension, so grouped statistics
    are formed by viewing the row as ``[groups, group_size]``.

    Args:
        x: ``[..., width]`` input.
        weight: ``[width]`` norm weight, or ``[group_size]`` shared per group.
        z: Gate input shaped like ``x``, or ``None`` for a plain grouped norm.
        eps: Epsilon added to each group's mean square.
        group_size: Features sharing one RMS statistic; ``None`` means the whole
            last dimension.
        norm_before_gate: ``norm(x) * gate(z)``; otherwise
            ``norm(x * gate(z))``.
        sigmoid_gate: Gate with ``sigmoid(z)``; otherwise ``silu(z)``.
        fp8_scale: Rejected; the native path returns ``x``'s dtype.
        weights_independent: Unused; the native op has no PDL producer.

    Returns:
        The normalized tensor, shaped like ``x``.
    """
    del weights_independent
    if fp8_scale is not None:
        raise NotImplementedError("Ascend gated RMSNorm does not emit FP8")
    width = x.shape[-1]
    group = width if group_size is None else int(group_size)
    if width % group:
        raise ValueError(
            f"gated RMSNorm width {width} is not a multiple of group {group}"
        )
    if weight.numel() not in (width, group):
        raise ValueError(
            f"gated RMSNorm weight must hold {width} or {group} values, "
            f"got {weight.numel()}"
        )
    groups = width // group
    # A weight of ``group`` values is shared by every group.
    per_group_weight = weight.float().reshape(
        1 if weight.numel() == group else groups, group
    )

    # ``[..., groups, group]``: one RMS statistic per group.
    shaped = x.reshape(*x.shape[:-1], groups, group).float()
    gate = None
    if z is not None:
        gate_input = z.reshape(*x.shape[:-1], groups, group).float()
        gate = (
            torch.sigmoid(gate_input)
            if sigmoid_gate
            else gate_input * torch.sigmoid(gate_input)
        )
        if not norm_before_gate:
            shaped = shaped * gate

    rstd = torch.rsqrt(shaped.square().mean(-1, keepdim=True) + eps)
    normalized = (shaped * rstd) * per_group_weight
    if gate is not None and norm_before_gate:
        normalized = normalized * gate
    # The portable leaf emits the activation dtype whatever the weight dtype.
    return normalized.reshape(*x.shape).to(x.dtype)


__all__ = ["gated_rmsnorm", "gemma_rmsnorm", "qk_rmsnorm", "rmsnorm"]
