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

"""Portable Triton kernels for norm-preserving gated residuals."""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import (
    dense_tensor_format,
    format_signature,
    format_signatures,
)

_FLOAT_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


@triton.jit
def _centered_bf16(weight, offset: tl.constexpr):
    """Add the checkpoint weight center with BF16 rounding semantics."""
    return (weight.to(tl.float32) + offset).to(tl.bfloat16).to(tl.float32)


@triton.jit
def _residual_gate_coefficients_kernel(
    gate_ptr,
    alpha_ptr,
    beta_ptr,
    width: tl.constexpr,
    temperature: tl.constexpr,
    eps: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < width
    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    scaled_gate = gate / temperature
    beta = tl.sigmoid(scaled_gate)
    alpha = tl.sqrt(tl.maximum(tl.sigmoid(-scaled_gate) * (1.0 + beta), eps))
    tl.store(alpha_ptr + offsets, alpha, mask=mask)
    tl.store(beta_ptr + offsets, beta, mask=mask)


@register_kernel(
    "residual",
    "residual_gate_coefficients",
    name="triton_residual_gate_coefficients",
    solution="triton",
    signatures=format_signatures("gate", "dense", _FLOAT_DTYPES),
    priority=Priority.PORTABLE,
)
def triton_residual_gate_coefficients(
    gate: torch.Tensor, temperature: float, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Triton kernel of :func:`tokenspeed_kernel.ops.residual.residual_gate_coefficients`."""
    if gate.ndim != 1:
        raise ValueError("gate must be a [width] vector")
    if not gate.is_cuda:
        raise ValueError("gate must be a GPU tensor")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    width = gate.shape[0]
    alpha = torch.empty(width, dtype=torch.float32, device=gate.device)
    beta = torch.empty_like(alpha)
    if width == 0:
        return alpha, beta
    block = 1024
    _residual_gate_coefficients_kernel[(triton.cdiv(width, block),)](
        gate.contiguous(),
        alpha,
        beta,
        width=width,
        temperature=temperature,
        eps=eps,
        BLOCK=block,
        num_warps=4,
    )
    return alpha, beta


@triton.jit
def _post_norm_residual_pre_norm_kernel(
    stream_ptr,
    branch_ptr,
    post_weight_ptr,
    alpha_ptr,
    beta_ptr,
    pre_weight_ptr,
    out_stream_ptr,
    out_pre_fp32_ptr,
    out_pre_bf16_ptr,
    width: tl.constexpr,
    post_eps: tl.constexpr,
    post_weight_offset: tl.constexpr,
    pre_eps: tl.constexpr,
    pre_weight_offset: tl.constexpr,
    sentinel_word: tl.constexpr,
    sentinel_half: tl.constexpr,
    POST_HAS_WEIGHT: tl.constexpr,
    HAS_PRE: tl.constexpr,
    BLOCK: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
    RETURN_PRE_FP32: tl.constexpr,
    MAILBOX: tl.constexpr,
    MAILBOX_BLOCK_WIDTH: tl.constexpr,
    MAILBOX_BLOCK_STRIDE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < width
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
        # Dependents may launch now; their own wait still covers this grid's completion.
        tl.extra.cuda.gdc_launch_dependents()
    stream = tl.load(stream_ptr + row * width + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    if MAILBOX:
        # One row-wide poll load over the blocks keeps the plain path's variance sum order bitwise.
        element = (
            (offsets // MAILBOX_BLOCK_WIDTH) * MAILBOX_BLOCK_STRIDE
            + row * MAILBOX_BLOCK_WIDTH
            + offsets % MAILBOX_BLOCK_WIDTH
        )
        half_ptr = branch_ptr.to(tl.pointer_type(tl.int16)) + element
        halves = tl.load(half_ptr, mask=mask, other=0, volatile=True)
        while tl.max((halves == sentinel_half).to(tl.int32), axis=0) != 0:
            halves = tl.load(half_ptr, mask=mask, other=0, volatile=True)
        values = halves.to(tl.bfloat16, bitcast=True).to(tl.float32)
        words = tl.arange(0, BLOCK // 2)
        word_ptr = branch_ptr.to(tl.pointer_type(tl.int32)) + (
            (words // (MAILBOX_BLOCK_WIDTH // 2)) * (MAILBOX_BLOCK_STRIDE // 2)
            + row * (MAILBOX_BLOCK_WIDTH // 2)
            + words % (MAILBOX_BLOCK_WIDTH // 2)
        )
        tl.store(
            word_ptr,
            tl.full([BLOCK // 2], sentinel_word, tl.int32),
            mask=words < width // 2,
        )
    else:
        values = tl.load(branch_ptr + row * width + offsets, mask=mask, other=0.0).to(
            tl.float32
        )
    if POST_HAS_WEIGHT:
        weight = tl.load(post_weight_ptr + offsets, mask=mask, other=0.0)
        values *= _centered_bf16(weight, post_weight_offset)
    variance = tl.sum(values * values, axis=0) / width
    values *= tl.rsqrt(variance + post_eps)
    alpha = tl.load(alpha_ptr + offsets, mask=mask, other=0.0)
    beta = tl.load(beta_ptr + offsets, mask=mask, other=0.0)
    # Pinned: the other FMA choice differs by an ULP on a third of the channels.
    stream = tl.fma(beta, values, alpha * stream)
    tl.store(out_stream_ptr + row * width + offsets, stream, mask=mask)
    if HAS_PRE:
        variance = tl.sum(stream * stream, axis=0) / width
        pre = stream * tl.rsqrt(variance + pre_eps)
        weight = tl.load(pre_weight_ptr + offsets, mask=mask, other=0.0)
        pre *= _centered_bf16(weight, pre_weight_offset)
        if RETURN_PRE_FP32:
            tl.store(out_pre_fp32_ptr + row * width + offsets, pre, mask=mask)
        tl.store(
            out_pre_bf16_ptr + row * width + offsets, pre.to(tl.bfloat16), mask=mask
        )


@register_kernel(
    "residual",
    "post_norm_residual_pre_norm",
    name="triton_post_norm_residual_pre_norm",
    solution="triton",
    signatures=frozenset(
        {
            format_signature(
                stream=dense_tensor_format(torch.float32),
                branch=dense_tensor_format(torch.bfloat16),
            )
        }
    ),
    priority=Priority.PORTABLE,
)
def triton_post_norm_residual_pre_norm(
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
    """Return the residual stream and both normalized input dtypes."""
    return _launch_post_norm_residual_pre_norm(
        stream,
        branch,
        post_weight,
        post_eps,
        post_weight_offset,
        alpha,
        beta,
        pre_weight,
        pre_eps,
        pre_weight_offset,
        enable_pdl,
        branch_sentinel,
        branch_ranks,
        return_pre_fp32=True,
    )


def _launch_post_norm_residual_pre_norm(
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
    *,
    return_pre_fp32: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """The fused layer boundary as one Triton program per row.

    Same contract as the dispatcher ``post_norm_residual_pre_norm``, except that
    a mailbox branch holds one block per rank to gather, never partials to
    reduce; every step stays FP32 and only the BF16 pre-norm rounds.
    """
    if stream.ndim != 2:
        raise ValueError("the residual stream must be [rows, width]")
    rows, width = stream.shape
    block_width = block_stride = width
    if branch_sentinel is None:
        if stream.shape != branch.shape or branch_ranks != 1:
            raise ValueError("stream and branch must share a [rows, width] shape")
    else:
        if branch_ranks != 1:
            raise ValueError("partials per rank need the cluster boundary kernel")
        if not 0 <= branch_sentinel <= 0xFFFFFFFF or (branch_sentinel & 0xFFFF) != (
            branch_sentinel >> 16
        ):
            raise ValueError("the mailbox sentinel is one word whose halves agree")
        if branch.ndim != 3 or branch.shape[1] != rows:
            raise ValueError("a mailbox branch is a [blocks, rows, block_width] view")
        block_width, block_stride = branch.shape[2], branch.stride(0)
        if branch.shape[0] * block_width != width or block_width & (block_width - 1):
            raise ValueError("the mailbox blocks must tile the row in powers of two")
        if branch.stride()[1:] != (block_width, 1) or block_stride < rows * block_width:
            raise ValueError(
                "a mailbox must be the first rows of whole contiguous blocks"
            )
        # The kernel polls halves but re-arms 32-bit words, which must not straddle two blocks.
        if block_width < 2 or block_stride % 2:
            raise ValueError("a mailbox block must hold whole 32-bit words")
    if stream.dtype != torch.float32:
        raise ValueError("the residual stream must use FP32")
    if branch.dtype != torch.bfloat16:
        raise ValueError("the branch must use BF16")
    for name, coefficient in (("alpha", alpha), ("beta", beta)):
        if coefficient.shape != (width,) or coefficient.dtype != torch.float32:
            raise ValueError(f"{name} must be an FP32 vector over the final dimension")
    for name, weight in (("post_weight", post_weight), ("pre_weight", pre_weight)):
        if weight is not None and weight.shape != (width,):
            raise ValueError(f"{name} must match the final dimension")
    if not stream.is_cuda:
        raise ValueError("all operands must be GPU tensors")
    stream_values = stream.contiguous()
    branch_values = branch if branch_sentinel is not None else branch.contiguous()
    out_stream = torch.empty_like(stream_values)
    has_pre = pre_weight is not None
    pre_fp32 = torch.empty_like(stream_values) if has_pre and return_pre_fp32 else None
    pre_bf16 = (
        torch.empty((rows, width), dtype=torch.bfloat16, device=stream.device)
        if has_pre
        else None
    )
    sentinel = 0 if branch_sentinel is None else branch_sentinel
    if rows == 0:
        return out_stream, pre_fp32, pre_bf16
    block = triton.next_power_of_2(width)
    if block > 65536:
        raise ValueError("the boundary width is too wide for one Triton block")
    use_pdl = enable_pdl and current_platform().is_hopper_plus
    launch = {"launch_pdl": True} if use_pdl else {}
    _post_norm_residual_pre_norm_kernel[(rows,)](
        stream_values,
        branch_values,
        post_weight if post_weight is not None else stream_values,
        alpha.contiguous(),
        beta.contiguous(),
        pre_weight if has_pre else stream_values,
        out_stream,
        pre_fp32 if pre_fp32 is not None else out_stream,
        pre_bf16 if has_pre else branch_values,
        width=width,
        post_eps=post_eps,
        post_weight_offset=post_weight_offset,
        pre_eps=pre_eps,
        pre_weight_offset=pre_weight_offset,
        # The kernel compares int16 halves and stores int32 words: signed images.
        sentinel_word=sentinel - (1 << 32) if sentinel >= 1 << 31 else sentinel,
        sentinel_half=(sentinel & 0xFFFF) - (1 << 16 if sentinel & 0x8000 else 0),
        POST_HAS_WEIGHT=post_weight is not None,
        HAS_PRE=has_pre,
        BLOCK=block,
        ENABLE_PDL=use_pdl,
        RETURN_PRE_FP32=return_pre_fp32,
        MAILBOX=branch_sentinel is not None,
        MAILBOX_BLOCK_WIDTH=block_width,
        MAILBOX_BLOCK_STRIDE=block_stride,
        # Six live row-wide vectors spill at 4 warps on an 8192 row.
        num_warps=16,
        **launch,
    )
    return out_stream, pre_fp32, pre_bf16


__all__ = [
    "triton_post_norm_residual_pre_norm",
    "triton_residual_gate_coefficients",
]
