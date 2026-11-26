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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""Identity Hyper-Connections (iHC) kernels for Hy4-flash.

iHC maintains ``hc_mult`` (N=4) residual streams per token without Sinkhorn
stream-mixing:
- ``ihc_pre``: computes FP32 RMS-normalized linear logits from flattened
  streams, derives sigmoid ``pre`` and ``post`` weights, and reduces the N
  streams into one branch input ``[T, D]``.
- ``ihc_post``: adds ``post[..., None] * branch_output[..., None, :]`` into the
  residual stream tensor ``[T, N, D]`` in FP32 and casts back to stream dtype.
- ``ihc_head``: collapses the final N streams into one ``[T, D]`` hidden tensor
  before the shared final RMSNorm.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from tokenspeed_kernel._triton import tl, triton


@triton.jit(do_not_specialize=["num_tokens"])
def _ihc_pre_reduce_kernel(
    x_ptr,
    mixes_ptr,
    scale_ptr,
    base_ptr,
    out_ptr,
    post_ptr,
    num_tokens,
    stride_xt,
    stride_xn,
    stride_ot,
    stride_mt,
    stride_pt,
    rms_norm_eps: tl.constexpr,
    hc_magnitude: tl.constexpr,
    hc_eps: tl.constexpr,
    HC_MULT: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    if token >= num_tokens:
        return

    # Pass 1: unweighted RMS norm scalar over the flattened [HC_MULT * HIDDEN_SIZE] row.
    sum_sq = tl.zeros((), dtype=tl.float32)
    offs = tl.arange(0, BLOCK_D)
    for n in tl.static_range(0, HC_MULT):
        base_x = token * stride_xt + n * stride_xn
        for start in range(0, HIDDEN_SIZE, BLOCK_D):
            cols = start + offs
            mask = cols < HIDDEN_SIZE
            val = tl.load(x_ptr + base_x + cols, mask=mask, other=0.0).to(tl.float32)
            sum_sq += tl.sum(val * val, axis=0)
    inv_rms = tl.rsqrt(sum_sq / (HC_MULT * HIDDEN_SIZE) + rms_norm_eps)

    pre_scale = tl.load(scale_ptr + 0).to(tl.float32)
    post_scale = tl.load(scale_ptr + 1).to(tl.float32)

    n_idx = tl.arange(0, HC_MULT)
    pre_logits = tl.load(mixes_ptr + token * stride_mt + n_idx).to(tl.float32) * inv_rms
    post_logits = (
        tl.load(mixes_ptr + token * stride_mt + HC_MULT + n_idx).to(tl.float32)
        * inv_rms
    )
    pre_b = tl.load(base_ptr + n_idx).to(tl.float32)
    post_b = tl.load(base_ptr + HC_MULT + n_idx).to(tl.float32)

    pre = tl.sigmoid(pre_logits * pre_scale + pre_b) + hc_eps
    post = hc_magnitude * tl.sigmoid(post_logits * post_scale + post_b) + hc_eps
    tl.store(post_ptr + token * stride_pt + n_idx, post)

    # Pass 2: weighted stream reduction -> [HIDDEN_SIZE].
    for start in range(0, HIDDEN_SIZE, BLOCK_D):
        cols = start + offs
        mask = cols < HIDDEN_SIZE
        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for n in tl.static_range(0, HC_MULT):
            w_n = tl.sum(tl.where(n_idx == n, pre, 0.0), axis=0)
            val = tl.load(
                x_ptr + token * stride_xt + n * stride_xn + cols,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            acc += w_n * val
        tl.store(
            out_ptr + token * stride_ot + cols,
            acc.to(out_ptr.dtype.element_ty),
            mask=mask,
        )


@triton.jit(do_not_specialize=["num_tokens"])
def _ihc_post_kernel(
    residual_ptr,
    branch_ptr,
    post_ptr,
    out_ptr,
    num_tokens,
    stride_rt,
    stride_rn,
    stride_bt,
    stride_pt,
    stride_ot,
    stride_on,
    HC_MULT: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    block = tl.program_id(1).to(tl.int64)
    if token >= num_tokens:
        return
    cols = block * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = cols < HIDDEN_SIZE
    branch = tl.load(branch_ptr + token * stride_bt + cols, mask=mask, other=0.0).to(
        tl.float32
    )
    for n in tl.static_range(0, HC_MULT):
        p = tl.load(post_ptr + token * stride_pt + n).to(tl.float32)
        res = tl.load(
            residual_ptr + token * stride_rt + n * stride_rn + cols,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        updated = res + p * branch
        tl.store(
            out_ptr + token * stride_ot + n * stride_on + cols,
            updated.to(out_ptr.dtype.element_ty),
            mask=mask,
        )


@triton.jit(do_not_specialize=["num_tokens"])
def _ihc_head_reduce_kernel(
    x_ptr,
    mixes_ptr,
    scale_ptr,
    base_ptr,
    out_ptr,
    num_tokens,
    stride_xt,
    stride_xn,
    stride_ot,
    stride_mt,
    rms_norm_eps: tl.constexpr,
    hc_eps: tl.constexpr,
    HC_MULT: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    if token >= num_tokens:
        return

    sum_sq = tl.zeros((), dtype=tl.float32)
    offs = tl.arange(0, BLOCK_D)
    for n in tl.static_range(0, HC_MULT):
        base_x = token * stride_xt + n * stride_xn
        for start in range(0, HIDDEN_SIZE, BLOCK_D):
            cols = start + offs
            mask = cols < HIDDEN_SIZE
            val = tl.load(x_ptr + base_x + cols, mask=mask, other=0.0).to(tl.float32)
            sum_sq += tl.sum(val * val, axis=0)
    inv_rms = tl.rsqrt(sum_sq / (HC_MULT * HIDDEN_SIZE) + rms_norm_eps)

    head_scale = tl.load(scale_ptr + 0).to(tl.float32)
    n_idx = tl.arange(0, HC_MULT)
    logits = tl.load(mixes_ptr + token * stride_mt + n_idx).to(tl.float32) * inv_rms
    base = tl.load(base_ptr + n_idx).to(tl.float32)
    pre = tl.sigmoid(logits * head_scale + base) + hc_eps

    for start in range(0, HIDDEN_SIZE, BLOCK_D):
        cols = start + offs
        mask = cols < HIDDEN_SIZE
        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for n in tl.static_range(0, HC_MULT):
            w_n = tl.sum(tl.where(n_idx == n, pre, 0.0), axis=0)
            val = tl.load(
                x_ptr + token * stride_xt + n * stride_xn + cols,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            acc += w_n * val
        tl.store(
            out_ptr + token * stride_ot + cols,
            acc.to(out_ptr.dtype.element_ty),
            mask=mask,
        )


def _ihc_pre_torch(
    hidden_streams: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    *,
    rms_norm_eps: float,
    hc_magnitude: float,
    hc_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact FP32 reference matching HYV4FlashHyperConnection."""
    orig_dtype = hidden_streams.dtype
    num_tokens, hc_mult, hidden_size = hidden_streams.shape
    flat = hidden_streams.reshape(num_tokens, hc_mult * hidden_size).float()
    inv_rms = torch.rsqrt(flat.pow(2).mean(dim=-1, keepdim=True) + rms_norm_eps)
    mixes = F.linear(flat, fn.float()) * inv_rms
    pre_logits, post_logits = mixes.split(hc_mult, dim=-1)
    pre_b, post_b = base.float().split(hc_mult, dim=-1)
    pre_scale, post_scale = scale.float().unbind(0)
    pre = torch.sigmoid(pre_logits * pre_scale + pre_b) + hc_eps
    post = hc_magnitude * torch.sigmoid(post_logits * post_scale + post_b) + hc_eps
    branch_input = (pre.unsqueeze(-1) * hidden_streams.float()).sum(dim=1)
    return post, branch_input.to(orig_dtype)


def ihc_pre(
    hidden_streams: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    *,
    rms_norm_eps: float = 1e-5,
    hc_magnitude: float = 2.0,
    hc_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute iHC post weights (FP32) and reduced branch input ``[T, D]``.

    Args:
        hidden_streams: ``[T, hc_mult, hidden_size]`` residual stream tensor.
        fn: ``[2 * hc_mult, hc_mult * hidden_size]`` projection weight.
        scale: ``[2]`` scalar scales ``(pre_scale, post_scale)``.
        base: ``[2 * hc_mult]`` bias terms ``(pre_b, post_b)``.
        rms_norm_eps: Unweighted RMSNorm epsilon.
        hc_magnitude: Post-gate multiplier (default 2.0).
        hc_eps: Epsilon added to both sigmoid gates (default 1e-6).

    Returns:
        ``(post_weights, branch_input)`` with shapes ``[T, hc_mult]`` (FP32)
        and ``[T, hidden_size]`` (same dtype as ``hidden_streams``).
    """
    if hidden_streams.ndim != 3:
        raise ValueError(
            f"ihc_pre expects [T, hc_mult, D], got {tuple(hidden_streams.shape)}"
        )
    num_tokens, hc_mult, hidden_size = hidden_streams.shape
    if num_tokens == 0:
        return (
            torch.empty(
                (0, hc_mult), dtype=torch.float32, device=hidden_streams.device
            ),
            torch.empty(
                (0, hidden_size),
                dtype=hidden_streams.dtype,
                device=hidden_streams.device,
            ),
        )
    if (
        not hidden_streams.is_cuda
        or (hc_mult & (hc_mult - 1)) != 0
        or hidden_streams.stride(-1) != 1
    ):
        return _ihc_pre_torch(
            hidden_streams,
            fn,
            scale,
            base,
            rms_norm_eps=rms_norm_eps,
            hc_magnitude=hc_magnitude,
            hc_eps=hc_eps,
        )

    flat = hidden_streams.reshape(num_tokens, hc_mult * hidden_size).float()
    mixes = F.linear(flat, fn.float())
    out = torch.empty(
        (num_tokens, hidden_size),
        dtype=hidden_streams.dtype,
        device=hidden_streams.device,
    )
    post = torch.empty(
        (num_tokens, hc_mult),
        dtype=torch.float32,
        device=hidden_streams.device,
    )
    block_d = min(1024, triton.next_power_of_2(hidden_size))
    _ihc_pre_reduce_kernel[(num_tokens,)](
        hidden_streams,
        mixes,
        scale,
        base,
        out,
        post,
        num_tokens,
        hidden_streams.stride(0),
        hidden_streams.stride(1),
        out.stride(0),
        mixes.stride(0),
        post.stride(0),
        rms_norm_eps=float(rms_norm_eps),
        hc_magnitude=float(hc_magnitude),
        hc_eps=float(hc_eps),
        HC_MULT=hc_mult,
        HIDDEN_SIZE=hidden_size,
        BLOCK_D=block_d,
        num_warps=4,
    )
    return post, out


def ihc_post(
    residual: torch.Tensor,
    branch_output: torch.Tensor,
    post_weights: torch.Tensor,
) -> torch.Tensor:
    """Update residual streams ``(post[..., None] * branch[..., None, :] + residual)``."""
    if residual.ndim != 3:
        raise ValueError(
            f"ihc_post expects [T, hc_mult, D], got {tuple(residual.shape)}"
        )
    num_tokens, hc_mult, hidden_size = residual.shape
    if num_tokens == 0:
        return residual.clone()
    if (
        not residual.is_cuda
        or residual.stride(-1) != 1
        or branch_output.stride(-1) != 1
    ):
        updated = (
            post_weights.float().unsqueeze(-1) * branch_output.float().unsqueeze(1)
            + residual.float()
        )
        return updated.to(residual.dtype)

    out = torch.empty_like(residual)
    block_d = min(512, triton.next_power_of_2(hidden_size))
    grid = (num_tokens, triton.cdiv(hidden_size, block_d))
    _ihc_post_kernel[grid](
        residual,
        branch_output,
        post_weights,
        out,
        num_tokens,
        residual.stride(0),
        residual.stride(1),
        branch_output.stride(0),
        post_weights.stride(0),
        out.stride(0),
        out.stride(1),
        HC_MULT=hc_mult,
        HIDDEN_SIZE=hidden_size,
        BLOCK_D=block_d,
        num_warps=4,
    )
    return out


def ihc_head(
    hidden_streams: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    *,
    rms_norm_eps: float = 1e-5,
    hc_eps: float = 1e-6,
) -> torch.Tensor:
    """Collapse ``[T, hc_mult, D]`` streams into ``[T, D]`` before final RMSNorm."""
    if hidden_streams.ndim != 3:
        raise ValueError(
            f"ihc_head expects [T, hc_mult, D], got {tuple(hidden_streams.shape)}"
        )
    num_tokens, hc_mult, hidden_size = hidden_streams.shape
    if num_tokens == 0:
        return torch.empty(
            (0, hidden_size),
            dtype=hidden_streams.dtype,
            device=hidden_streams.device,
        )
    if (
        not hidden_streams.is_cuda
        or (hc_mult & (hc_mult - 1)) != 0
        or hidden_streams.stride(-1) != 1
    ):
        flat = hidden_streams.reshape(num_tokens, hc_mult * hidden_size).float()
        inv_rms = torch.rsqrt(flat.pow(2).mean(dim=-1, keepdim=True) + rms_norm_eps)
        mixes = F.linear(flat, hc_fn.float()) * inv_rms
        pre = torch.sigmoid(mixes * hc_scale.float() + hc_base.float()) + hc_eps
        return (
            (pre.unsqueeze(-1) * hidden_streams.float())
            .sum(dim=1)
            .to(hidden_streams.dtype)
        )

    flat = hidden_streams.reshape(num_tokens, hc_mult * hidden_size).float()
    mixes = F.linear(flat, hc_fn.float())
    out = torch.empty(
        (num_tokens, hidden_size),
        dtype=hidden_streams.dtype,
        device=hidden_streams.device,
    )
    block_d = min(1024, triton.next_power_of_2(hidden_size))
    _ihc_head_reduce_kernel[(num_tokens,)](
        hidden_streams,
        mixes,
        hc_scale,
        hc_base,
        out,
        num_tokens,
        hidden_streams.stride(0),
        hidden_streams.stride(1),
        out.stride(0),
        mixes.stride(0),
        rms_norm_eps=float(rms_norm_eps),
        hc_eps=float(hc_eps),
        HC_MULT=hc_mult,
        HIDDEN_SIZE=hidden_size,
        BLOCK_D=block_d,
        num_warps=4,
    )
    return out


__all__ = ["ihc_head", "ihc_post", "ihc_pre"]
