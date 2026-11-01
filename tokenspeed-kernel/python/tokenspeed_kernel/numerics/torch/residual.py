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

"""PyTorch references for ``tokenspeed_kernel.ops.residual``: manifold
hyper-connections (mHC), hyper-connection mixes, attention residuals and the
gated residual boundary."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from tokenspeed_kernel.numerics.torch.layernorm import torch_rmsnorm
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature


def mhc_mix_coefficients(
    mixes: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    hc_eps: float,
    sinkhorn_iters: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """mHC pre/post/combine coefficients from RMS-normalized projections.

    Args:
        mixes: ``[..., 2 * hc + hc**2]`` projections of the flattened streams,
            already scaled by their inverse RMS.
        scale: Pre, post and combine scales ``[3]``.
        base: Mixing biases shaped like the last dim of ``mixes``.
        hc_eps: Epsilon of the pre mix and the Sinkhorn normalization.
        sinkhorn_iters: Sinkhorn iterations; the first is the column
            normalization after the softmax.

    Returns:
        ``pre`` and ``post`` ``[..., hc]`` and ``comb`` ``[..., hc, hc]`` (input
        streams by output streams), in ``mixes``'s dtype.
    """
    hc = int(round((1 + mixes.shape[-1]) ** 0.5)) - 1
    scale, base = scale.to(mixes.dtype), base.to(mixes.dtype)
    pre = torch.sigmoid(mixes[..., :hc] * scale[0] + base[:hc]) + hc_eps
    post = torch.sigmoid(mixes[..., hc : 2 * hc] * scale[1] + base[hc : 2 * hc]) * 2.0
    comb = mixes[..., 2 * hc :].unflatten(-1, (hc, hc)) * scale[2] + base[
        2 * hc :
    ].view(hc, hc)
    comb = torch.softmax(comb, dim=-1) + hc_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
    for _ in range(1, sinkhorn_iters):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + hc_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
    return pre, post, comb


@register_kernel(
    "residual",
    "mhc_mixes",
    name="torch_mhc_mixes",
    solution="torch",
    signatures=frozenset(
        {format_signature(residual=dense_tensor_format(torch.bfloat16))}
    ),
    priority=Priority.REFERENCE,
)
def torch_mhc_mixes(
    residual: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """mHC coefficients of residual streams ``[..., hc, H]``.

    The flattened streams are projected by ``weight`` and scaled by their
    inverse RMS, then mapped by :func:`mhc_mix_coefficients`. Math runs in
    ``weight``'s dtype (FP32, or FP64 for an FP64 oracle).
    """
    flat = residual.to(weight.dtype).flatten(-2)
    inv_rms = torch.rsqrt(flat.square().mean(dim=-1, keepdim=True) + rms_eps)
    mixes = F.linear(flat, weight) * inv_rms
    return mhc_mix_coefficients(mixes, scale, base, hc_eps, sinkhorn_iters)


@register_kernel(
    "residual",
    "mhc_apply_pre",
    name="torch_mhc_apply_pre",
    solution="torch",
    signatures=frozenset(
        {
            format_signature(
                residual=dense_tensor_format(torch.bfloat16),
                pre=dense_tensor_format(torch.float32),
            )
        }
    ),
    priority=Priority.REFERENCE,
)
def torch_mhc_apply_pre(
    residual: torch.Tensor,
    pre: torch.Tensor,
    *,
    norm_weight: torch.Tensor | None,
    norm_eps: float | None,
) -> torch.Tensor:
    """Collapse streams ``[..., hc, H]`` by ``pre`` ``[..., hc]`` into a layer
    input in ``residual``'s dtype (BF16 for the kernels).

    A ``norm_weight`` RMSNorms the rounded sum with ``norm_eps``.
    """
    layer = (pre.float().unsqueeze(-1) * residual.float()).sum(dim=-2)
    layer = layer.to(residual.dtype)
    if norm_weight is None:
        return layer
    return torch_rmsnorm(layer, norm_weight, norm_eps)


@register_kernel(
    "residual",
    "mhc_pre",
    name="torch_mhc_pre",
    solution="torch",
    signatures=frozenset(
        {
            format_signature(
                residual=dense_tensor_format(torch.bfloat16),
                fn=dense_tensor_format(torch.float32),
                hc_scale=dense_tensor_format(torch.float32),
                hc_base=dense_tensor_format(torch.float32),
            )
        }
    ),
    priority=Priority.REFERENCE,
)
def torch_mhc_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    norm_weight: torch.Tensor | None,
    norm_eps: float | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """mHC pre-mapping: :func:`torch_mhc_mixes`, then
    :func:`torch_mhc_apply_pre` of its ``pre``.

    Returns:
        The layer input ``[..., H]`` in ``residual``'s dtype, post mix
        ``[..., hc, 1]`` and combine mix ``[..., hc, hc]``.
    """
    pre, post, comb = torch_mhc_mixes(
        residual, fn, hc_scale, hc_base, rms_eps, hc_eps, sinkhorn_iters
    )
    layer = torch_mhc_apply_pre(
        residual, pre, norm_weight=norm_weight, norm_eps=norm_eps
    )
    return layer, post.unsqueeze(-1), comb


@register_kernel(
    "residual",
    "hyperconnection_mix",
    name="torch_hyperconnection_mix",
    solution="torch",
    signatures=frozenset(
        format_signature(
            normalized=dense_tensor_format(dtype),
            projection_weight=dense_tensor_format(dtype),
            up_weight=dense_tensor_format(dtype),
        )
        for dtype in (torch.bfloat16, torch.float16, torch.float32)
    ),
    priority=Priority.REFERENCE,
)
def torch_hyperconnection_mix(
    normalized: torch.Tensor,
    projection_weight: torch.Tensor,
    up_weight: torch.Tensor,
    hc_count: int,
    hidden_size: int,
    lowrank: int,
    projection_rows: int,
    projection_scale: float,
    weights_independent: bool,
    packed_up_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Low-rank gated mix of ``hc_count`` streams, in FP64.

    ``projected = normalized @ projection_weight.T * projection_scale``; its
    first ``lowrank`` columns gate the streams through
    ``sigmoid(silu(.) @ up_weight.T)`` and the mix is the mean of the gated
    streams. Projection rows past ``lowrank`` are the inject logits.
    ``weights_independent`` is a launch hint and has no effect here.

    Returns:
        FP64 ``(mixed [T, hidden_size], inject [T, projection_rows - lowrank]
        or None)``.
    """
    del weights_independent
    x = normalized.double()
    projected = (x @ projection_weight[:projection_rows].double().T) * projection_scale
    gate = F.silu(projected[:, :lowrank]) @ up_weight.double().T
    mixed = (
        torch.sigmoid(gate).unflatten(-1, (hc_count, hidden_size))
        * x.unflatten(-1, (hc_count, hidden_size))
    ).mean(dim=-2)
    inject = projected[:, lowrank:] if projection_rows > lowrank else None
    return mixed, inject


@register_kernel(
    "residual",
    "post_norm_residual_pre_norm",
    name="torch_post_norm_residual_pre_norm",
    solution="torch",
    signatures=frozenset(
        {
            format_signature(
                stream=dense_tensor_format(torch.float32),
                branch=dense_tensor_format(torch.bfloat16),
            )
        }
    ),
    priority=Priority.REFERENCE,
)
def torch_post_norm_residual_pre_norm(
    stream: torch.Tensor,
    branch: torch.Tensor,
    post_weight: torch.Tensor | None,
    post_eps: float,
    post_weight_offset: float,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    pre_weight: torch.Tensor | None,
    pre_eps: float,
    pre_weight_offset: float,
    enable_pdl: bool,
    branch_sentinel: int | None,
    branch_ranks: int,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """A decoder-layer boundary in FP64 from FP32 gate coefficients.

    ``new_stream = alpha * stream + beta * rmsnorm(branch * post_weight)``,
    then ``pre = rmsnorm(new_stream) * pre_weight``; each checkpoint weight
    plus its offset rounds through BF16 as the kernels load it, and nothing
    else rounds. Only an ordinary branch is supported (``branch_sentinel``
    None, ``branch_ranks`` 1). ``enable_pdl`` is a launch hint.

    Returns:
        ``(new_stream, pre, pre_bf16)`` with the first two in FP64, or
        ``None`` for the pre-norm outputs when ``pre_weight`` is None.
    """
    del enable_pdl
    if branch_sentinel is not None or branch_ranks != 1:
        raise ValueError("the reference reads an ordinary branch only")

    def centered(weight: torch.Tensor, offset: float) -> torch.Tensor:
        return (weight.float() + offset).to(torch.bfloat16).double()

    values = branch.double()
    if post_weight is not None:
        values = values * centered(post_weight, post_weight_offset)
    values = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + post_eps)
    new_stream = alpha.double() * stream.double() + beta.double() * values
    if pre_weight is None:
        return new_stream, None, None
    pre = new_stream * torch.rsqrt(new_stream.square().mean(-1, keepdim=True) + pre_eps)
    pre = pre * centered(pre_weight, pre_weight_offset)
    return new_stream, pre, pre.to(torch.bfloat16)


@register_kernel(
    "residual",
    "mhc_post",
    name="torch_mhc_post",
    solution="torch",
    signatures=frozenset(
        {
            format_signature(
                hidden_states=dense_tensor_format(torch.bfloat16),
                residual=dense_tensor_format(torch.bfloat16),
            )
        }
    ),
    priority=Priority.REFERENCE,
)
def torch_mhc_post(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    """mHC post-mapping: ``comb``-mixed streams plus ``post`` times the layer
    output, in ``hidden_states``'s dtype.

    Args:
        hidden_states: ``[T, H]`` layer output.
        residual: ``[T, hc, H]`` streams.
        post: ``[T, hc]`` or ``[T, hc, 1]``.
        comb: ``[T, hc, hc]``, input streams by output streams.

    Returns:
        ``[T, hc, H]`` new streams.
    """
    if post.dim() == 2:
        post = post.unsqueeze(-1)
    mixed_residual = torch.einsum("tnm,tnh->tmh", comb.float(), residual.float())
    block_update = post.float() * hidden_states.float().unsqueeze(1)
    return (mixed_residual + block_update).to(hidden_states.dtype)
