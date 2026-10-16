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

"""Portable Triton kernels for weight-free and centered-affine RMSNorm."""

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
    """The checkpoint weight plus its center, rounded through BF16, in FP32."""
    return (weight.to(tl.float32) + offset).to(tl.bfloat16).to(tl.float32)


@triton.jit
def _weightless_qk_rmsnorm_kernel(
    q_ptr,
    k_ptr,
    q_out_ptr,
    k_out_ptr,
    q_stride,
    k_stride,
    q_out_stride,
    k_out_stride,
    num_q_heads: tl.constexpr,
    head_dim: tl.constexpr,
    eps: tl.constexpr,
    q_scale: tl.constexpr,
    BLOCK: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    is_k = head >= num_q_heads
    local_head = tl.where(is_k, head - num_q_heads, head)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < head_dim
    if is_k:
        src = k_ptr + token * k_stride + local_head * head_dim + offsets
        dst = k_out_ptr + token * k_out_stride + local_head * head_dim + offsets
        scale = 1.0
    else:
        src = q_ptr + token * q_stride + local_head * head_dim + offsets
        dst = q_out_ptr + token * q_out_stride + local_head * head_dim + offsets
        scale = q_scale
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
        # Dependents may launch now; their own wait still covers this grid's completion.
        tl.extra.cuda.gdc_launch_dependents()
    values = tl.load(src, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(values * values, axis=0) / head_dim
    output = values * tl.rsqrt(variance + eps) * scale
    tl.store(dst, output, mask=mask)


@register_kernel(
    "layernorm",
    "weightless_qk_rmsnorm",
    name="triton_weightless_qk_rmsnorm",
    solution="triton",
    signatures=format_signatures(("q", "k"), "dense", _FLOAT_DTYPES),
    priority=Priority.PORTABLE,
)
def triton_weightless_qk_rmsnorm(
    q: torch.Tensor,
    k: torch.Tensor,
    head_dim: int,
    eps: float,
    q_scale: float,
    enable_pdl: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Triton kernel of :func:`tokenspeed_kernel.ops.layernorm.weightless_qk_rmsnorm`."""
    if q.ndim != 2 or k.ndim != 2 or q.shape[0] != k.shape[0]:
        raise ValueError("q and k must be rank-2 tensors with equal token counts")
    if head_dim <= 0 or q.shape[1] % head_dim or k.shape[1] % head_dim:
        raise ValueError("head_dim must divide both q and k widths")
    if q.dtype != k.dtype or q.device != k.device:
        raise ValueError("q and k must share dtype and device")
    if not q.is_cuda:
        raise ValueError("q and k must be GPU tensors")
    if q.stride(1) != 1 or k.stride(1) != 1:
        raise ValueError("q and k must have contiguous inner dimensions")
    q_out = torch.empty_like(q, memory_format=torch.contiguous_format)
    k_out = torch.empty_like(k, memory_format=torch.contiguous_format)
    if q.shape[0] == 0:
        return q_out, k_out
    num_q_heads = q.shape[1] // head_dim
    num_kv_heads = k.shape[1] // head_dim
    block = triton.next_power_of_2(head_dim)
    if block > 65536:
        raise ValueError("head_dim is too wide for one Triton block")
    use_pdl = enable_pdl and current_platform().is_hopper_plus
    launch = {"launch_pdl": True} if use_pdl else {}
    _weightless_qk_rmsnorm_kernel[(q.shape[0], num_q_heads + num_kv_heads)](
        q,
        k,
        q_out,
        k_out,
        q.stride(0),
        k.stride(0),
        q_out.stride(0),
        k_out.stride(0),
        num_q_heads=num_q_heads,
        head_dim=head_dim,
        eps=eps,
        q_scale=q_scale,
        BLOCK=block,
        ENABLE_PDL=use_pdl,
        **launch,
    )
    return q_out, k_out


@triton.jit
def _rmsnorm_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    width: tl.constexpr,
    eps: tl.constexpr,
    weight_offset: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < width
    values = tl.load(x_ptr + row * width + offsets, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(values * values, axis=0) / width
    values *= tl.rsqrt(variance + eps)
    if HAS_WEIGHT:
        weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
        values *= _centered_bf16(weight, weight_offset)
    tl.store(out_ptr + row * width + offsets, values, mask=mask)


def _rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float,
    weight_offset: float,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    if x.ndim < 1:
        raise ValueError("x must have at least one dimension")
    width = x.shape[-1]
    if not x.is_cuda:
        raise ValueError("x must be a GPU tensor")
    if weight is not None:
        if weight.shape != (width,) or weight.device != x.device:
            raise ValueError("weight must be colocated and match x's final dimension")
    if out_dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("out_dtype must be BF16, FP16 or FP32")
    values = x.contiguous()
    output = torch.empty(values.shape, dtype=out_dtype, device=values.device)
    if values.numel() == 0:
        return output.view_as(x)
    block = triton.next_power_of_2(width)
    if block > 65536:
        raise ValueError("the normalized dimension is too wide for one Triton block")
    _rmsnorm_kernel[(values.numel() // width,)](
        values,
        weight if weight is not None else values,
        output,
        width=width,
        eps=eps,
        weight_offset=weight_offset,
        HAS_WEIGHT=weight is not None,
        BLOCK=block,
    )
    return output.view_as(x)


@register_kernel(
    "layernorm",
    "weightless_rmsnorm",
    name="triton_weightless_rmsnorm",
    solution="triton",
    signatures=format_signatures("x", "dense", _FLOAT_DTYPES),
    priority=Priority.PORTABLE,
)
def triton_weightless_rmsnorm(
    x: torch.Tensor, eps: float, out_dtype: torch.dtype
) -> torch.Tensor:
    """Triton kernel of :func:`tokenspeed_kernel.ops.layernorm.weightless_rmsnorm`."""
    return _rmsnorm(x, None, eps, 0.0, out_dtype)


@register_kernel(
    "layernorm",
    "affine_rmsnorm",
    name="triton_affine_rmsnorm",
    solution="triton",
    signatures=format_signatures("x", "dense", _FLOAT_DTYPES),
    priority=Priority.PORTABLE,
)
def triton_affine_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    weight_offset: float,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Triton kernel of :func:`tokenspeed_kernel.ops.layernorm.affine_rmsnorm`."""
    return _rmsnorm(x, weight, eps, weight_offset, out_dtype)


@triton.jit
def _weightless_rmsnorm_gated_sigmoid_kernel(
    x_ptr,
    gate_ptr,
    out_ptr,
    gate_stride,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    token = tl.program_id(0)
    heads = tl.arange(0, BLOCK_H)
    dims = tl.arange(0, BLOCK_D)
    mask = (heads[:, None] < num_heads) & (dims[None, :] < head_dim)
    offsets = token * num_heads * head_dim + heads[:, None] * head_dim + dims[None, :]
    gate_offsets = token * gate_stride + heads[:, None] * head_dim + dims[None, :]
    gate = tl.load(gate_ptr + gate_offsets, mask=mask, other=0.0).to(tl.float32)
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
        # Dependents may launch now; their own wait still covers this grid's completion.
        tl.extra.cuda.gdc_launch_dependents()
    values = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(values * values, axis=1) / head_dim
    output = values * tl.rsqrt(variance[:, None] + eps) * tl.sigmoid(gate)
    tl.store(out_ptr + offsets, output, mask=mask)


@register_kernel(
    "layernorm",
    "weightless_rmsnorm_gated_sigmoid",
    name="triton_weightless_rmsnorm_gated_sigmoid",
    solution="triton",
    signatures=format_signatures(("x", "gate"), "dense", _FLOAT_DTYPES),
    priority=Priority.PORTABLE,
)
def triton_weightless_rmsnorm_gated_sigmoid(
    x: torch.Tensor,
    gate: torch.Tensor,
    num_heads: int,
    head_dim: int,
    eps: float,
    enable_pdl: bool,
) -> torch.Tensor:
    """Triton kernel of :func:`tokenspeed_kernel.ops.layernorm.weightless_rmsnorm_gated_sigmoid`."""
    if num_heads <= 0 or head_dim <= 0:
        raise ValueError("num_heads and head_dim must be positive")
    if x.ndim != 2 or gate.shape != x.shape or x.shape[1] != num_heads * head_dim:
        raise ValueError("x and gate must have shape [tokens, num_heads * head_dim]")
    if x.dtype != gate.dtype or x.device != gate.device:
        raise ValueError("x and gate must share dtype and device")
    if not x.is_cuda:
        raise ValueError("x and gate must be GPU tensors")
    if not x.is_contiguous() or gate.stride(1) != 1:
        raise ValueError("x must be contiguous and gate must have unit inner stride")
    output = torch.empty_like(x)
    if x.shape[0] == 0:
        return output
    block_h = triton.next_power_of_2(num_heads)
    block_d = triton.next_power_of_2(head_dim)
    if block_h * block_d > 65536:
        raise ValueError(
            "the per-token attention tile is too wide for one Triton block"
        )
    use_pdl = enable_pdl and current_platform().is_hopper_plus
    launch = {"launch_pdl": True} if use_pdl else {}
    _weightless_rmsnorm_gated_sigmoid_kernel[(x.shape[0],)](
        x,
        gate,
        output,
        gate.stride(0),
        num_heads=num_heads,
        head_dim=head_dim,
        eps=eps,
        BLOCK_H=block_h,
        BLOCK_D=block_d,
        ENABLE_PDL=use_pdl,
        num_warps=4,
        **launch,
    )
    return output


__all__ = [
    "triton_affine_rmsnorm",
    "triton_weightless_qk_rmsnorm",
    "triton_weightless_rmsnorm",
    "triton_weightless_rmsnorm_gated_sigmoid",
]
