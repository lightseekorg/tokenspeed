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


"""PyTorch reference kernels for residual-stream operators.

``torch_attn_res_fwd`` registers as the ``"torch"`` solution in the PORTABLE
band: it is the numeric ground truth for the fused AttnRes kernels and the
last-resort path when no fused kernel covers an input. The mHC reference sits
in the REFERENCE band and is reachable only by naming a solution
(``"torch"``, or the ``"reference"`` meta solution that resolves to it).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import (
    dense_tensor_format,
    format_signature,
    format_signatures,
)


@register_kernel(
    "residual",
    "attn_res_fwd",
    name="torch_attn_res_fwd",
    solution="torch",
    signatures=format_signatures(
        ("layer_residual", "block_residual"), "dense", {torch.bfloat16}
    ),
    traits={},
    priority=Priority.PORTABLE,
    tags={"determinism", "portability"},
)
def torch_attn_res_fwd(
    *,
    layer_residual: torch.Tensor,
    block_residual: torch.Tensor,
    res_weight: torch.Tensor,
    rms_weight: torch.Tensor,
    eps: float,
    out_norm_weight: torch.Tensor | None = None,
    out_norm_eps: float | None = None,
    delta: torch.Tensor | None = None,
    num_valid_blocks: int | None = None,
    block_write_idx: int = -1,
) -> torch.Tensor:
    """Attention-Residual forward over snapshot candidates.

    Candidates are ``block_residual[:num_valid_blocks]`` followed by
    ``layer_residual``. Each candidate is scored by
    ``<RMSNorm(candidate), rms_weight * res_weight>``; the softmax over
    candidates weights the raw candidates. Scores and the mix run in FP32
    on the global residual backbone and round to BF16 once.

    Args:
        layer_residual: BF16 ``[T, H]`` current residual stream.
        block_residual: BF16 ``[K, T, H]`` periodic-snapshot storage.
        res_weight: BF16 ``[H]`` scorer projection weight.
        rms_weight: BF16 ``[H]`` RMSNorm weight.
        eps: RMSNorm epsilon.
        out_norm_weight: Optional BF16 ``[H]`` weight of the following
            RMSNorm, fused into the epilogue when given.
        out_norm_eps: Optional epsilon of the following RMSNorm; defaults to
            ``eps``.
        delta: Optional BF16 ``[T, H]`` update added in place to
            ``layer_residual`` before it participates in the mix.
        num_valid_blocks: Number of leading snapshots to include; defaults
            to every row of ``block_residual``.
        block_write_idx: Snapshot row receiving the updated layer residual,
            or ``-1`` to skip the write.

    Returns:
        BF16 ``[T, H]`` mixed residual, normed when ``out_norm_weight`` is
        given.
    """
    valid_blocks = (
        block_residual.shape[0] if num_valid_blocks is None else num_valid_blocks
    )
    if delta is not None:
        layer_residual.add_(delta)
    if block_write_idx >= 0:
        block_residual[block_write_idx].copy_(layer_residual)

    # Candidates [N, T, H] = blocks then layer; RMSNorm + softmax score + mix in
    # fp32 (this sits on the global residual backbone), bf16 out.
    values = torch.cat(
        (block_residual[:valid_blocks], layer_residual.unsqueeze(0)), dim=0
    ).float()
    rs = (values.square().mean(-1, keepdim=True) + eps).rsqrt()
    score_weight = rms_weight.float() * res_weight.float()  # [H]
    logits = (values * rs * score_weight).sum(-1)  # [N, T]
    probs = logits.softmax(0)  # over candidates
    out = (probs.unsqueeze(-1) * values).sum(0)  # [T, H]
    out = out.to(layer_residual.dtype)
    if out_norm_weight is not None:
        # Fused following RMSNorm: stats over the bf16-rounded mix (matches the
        # separate rmsnorm-kernel path).
        of = out.float()
        output_eps = eps if out_norm_eps is None else out_norm_eps
        rs = (of.square().mean(-1, keepdim=True) + output_eps).rsqrt()
        out = (of * rs * out_norm_weight.float()).to(layer_residual.dtype)
    return out


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
    traits={},
    priority=Priority.REFERENCE,
    tags={"determinism", "portability"},
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
    """mHC pre-mapping in FP32.

    The RMS-normalized flattened residual streams project through ``fn``
    into ``hc_mult`` pre gates (sigmoid, plus ``hc_eps``), ``hc_mult`` post
    gates (``2 * sigmoid``) and an ``hc_mult x hc_mult`` combine matrix
    (softmax, then ``sinkhorn_iters`` rounds of column/row normalization).

    Args:
        residual: BF16 ``[..., hc_mult, hidden_size]`` residual streams.
        fn: FP32 ``[2 * hc_mult + hc_mult**2, hc_mult * hidden_size]``
            mixing projection.
        hc_scale: FP32 ``[3]`` pre, post and combine scales.
        hc_base: FP32 ``[2 * hc_mult + hc_mult**2]`` mixing biases.
        rms_eps: Epsilon of the residual RMS normalization.
        hc_eps: Epsilon added during pre-mix and Sinkhorn normalization.
        sinkhorn_iters: Number of Sinkhorn normalization rounds.
        norm_weight: Optional RMSNorm weight applied to the layer input.
        norm_eps: RMSNorm epsilon; required with ``norm_weight``.

    Returns:
        BF16 layer input ``[..., hidden_size]``, FP32 post mix
        ``[..., hc_mult, 1]`` and FP32 combine mix ``[..., hc_mult, hc_mult]``.
    """
    if (norm_weight is None) != (norm_eps is None):
        raise ValueError("norm_weight and norm_eps must be provided together")
    hc_mult, hidden_size = residual.shape[-2:]
    outer_shape = residual.shape[:-2]
    flat = residual.float().reshape(-1, hc_mult * hidden_size)
    num_tokens = flat.shape[0]
    inv_rms = torch.rsqrt(flat.square().mean(dim=-1, keepdim=True) + rms_eps)
    mixes = F.linear(flat, fn) * inv_rms
    pre_raw, post_raw, comb_raw = torch.split(
        mixes, [hc_mult, hc_mult, hc_mult * hc_mult], dim=-1
    )
    pre = torch.sigmoid(pre_raw * hc_scale[0] + hc_base[:hc_mult]) + hc_eps
    post = torch.sigmoid(post_raw * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult]) * 2.0
    comb = torch.softmax(
        comb_raw.view(num_tokens, hc_mult, hc_mult) * hc_scale[2]
        + hc_base[2 * hc_mult :].view(1, hc_mult, hc_mult),
        dim=-1,
    )
    comb = comb + hc_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
    for _ in range(1, sinkhorn_iters):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + hc_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
    layer_input = (pre.unsqueeze(-1) * flat.view(num_tokens, hc_mult, hidden_size)).sum(
        dim=1
    )
    layer_input = layer_input.to(residual.dtype)
    if norm_weight is not None:
        layer_input = F.rms_norm(layer_input, (hidden_size,), norm_weight, norm_eps)
    return (
        layer_input.view(*outer_shape, hidden_size),
        post.view(*outer_shape, hc_mult, 1),
        comb.view(*outer_shape, hc_mult, hc_mult),
    )


__all__ = ["torch_attn_res_fwd", "torch_mhc_pre"]
