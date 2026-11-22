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

"""The fused layer boundary that also writes a per-row FP8 copy of its BF16
pre-norm rows, for a next projection that reads FP8 rows.

A separate kernel from ``gated_boundary.py``'s, so that kernel and its compile
key stay as they are for every caller. Its stream and pre-norm arithmetic is
that kernel's, step for step, so both write the same stream and BF16 bits; it
takes no mailbox branch (no FP8 reader follows a mailbox boundary).
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.quantization._triton_fp8_rows import fp8_row_codes
from tokenspeed_kernel.ops.residual._triton.gated_boundary import _centered_bf16
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature


@triton.jit
def _post_norm_residual_pre_norm_fp8_kernel(
    stream_ptr,
    branch_ptr,
    post_weight_ptr,
    alpha_ptr,
    beta_ptr,
    pre_weight_ptr,
    out_stream_ptr,
    out_pre_bf16_ptr,
    out_pre_fp8_ptr,
    out_pre_scale_ptr,
    width: tl.constexpr,
    post_eps: tl.constexpr,
    post_weight_offset: tl.constexpr,
    pre_eps: tl.constexpr,
    pre_weight_offset: tl.constexpr,
    POST_HAS_WEIGHT: tl.constexpr,
    BLOCK: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
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
    # Pinned as in the BF16 boundary kernel.
    stream = tl.fma(beta, values, alpha * stream)
    tl.store(out_stream_ptr + row * width + offsets, stream, mask=mask)
    variance = tl.sum(stream * stream, axis=0) / width
    pre = stream * tl.rsqrt(variance + pre_eps)
    weight = tl.load(pre_weight_ptr + offsets, mask=mask, other=0.0)
    pre *= _centered_bf16(weight, pre_weight_offset)
    pre_bf16 = pre.to(tl.bfloat16)
    tl.store(out_pre_bf16_ptr + row * width + offsets, pre_bf16, mask=mask)
    # Per-row FP8 E4M3 copy of the BF16 row, by the token quantizer's row
    # code, so both write the same bits.
    codes, scale = fp8_row_codes(pre_bf16.to(tl.float32), mask)
    tl.store(out_pre_fp8_ptr + row * width + offsets, codes, mask=mask)
    tl.store(out_pre_scale_ptr + row, scale)


@register_kernel(
    "residual",
    "post_norm_residual_pre_norm_fp8",
    name="triton_post_norm_residual_pre_norm_fp8",
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
def triton_post_norm_residual_pre_norm_fp8(
    stream: torch.Tensor,
    branch: torch.Tensor,
    post_weight: torch.Tensor | None,
    post_eps: float,
    post_weight_offset: float,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    pre_weight: torch.Tensor,
    pre_eps: float,
    pre_weight_offset: float,
    enable_pdl: bool,
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Registered entry of :func:`launch_post_norm_residual_pre_norm_fp8`."""
    return launch_post_norm_residual_pre_norm_fp8(
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
    )


def launch_post_norm_residual_pre_norm_fp8(
    stream: torch.Tensor,
    branch: torch.Tensor,
    post_weight: torch.Tensor | None,
    post_eps: float,
    post_weight_offset: float,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    pre_weight: torch.Tensor,
    pre_eps: float,
    pre_weight_offset: float,
    enable_pdl: bool,
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """The boundary with the FP8 copy of its BF16 pre-norm rows, one Triton
    program per row.

    Arguments as ``post_norm_residual_pre_norm`` for a plain ``[rows, width]``
    BF16 branch, with ``pre_weight`` required.

    Returns:
        ``(new_stream, pre_bf16, (codes, scales))``: FP8 E4M3
        ``[rows, width]`` codes and FP32 ``[rows]`` scales.
    """
    if stream.ndim != 2 or stream.shape != branch.shape:
        raise ValueError("stream and branch must share a [rows, width] shape")
    rows, width = stream.shape
    if stream.dtype != torch.float32:
        raise ValueError("the residual stream must use FP32")
    if branch.dtype != torch.bfloat16:
        raise ValueError("the branch must use BF16")
    for name, coefficient in (("alpha", alpha), ("beta", beta)):
        if coefficient.shape != (width,) or coefficient.dtype != torch.float32:
            raise ValueError(f"{name} must be an FP32 vector over the final dimension")
    if pre_weight is None:
        raise ValueError("an FP8 pre-norm copy needs the pre-norm")
    for name, weight in (("post_weight", post_weight), ("pre_weight", pre_weight)):
        if weight is not None and weight.shape != (width,):
            raise ValueError(f"{name} must match the final dimension")
    if not stream.is_cuda:
        raise ValueError("all operands must be GPU tensors")
    stream_values = stream.contiguous()
    branch_values = branch.contiguous()
    out_stream = torch.empty_like(stream_values)
    pre_bf16 = torch.empty((rows, width), dtype=torch.bfloat16, device=stream.device)
    codes = torch.empty((rows, width), dtype=torch.float8_e4m3fn, device=stream.device)
    scales = torch.empty(rows, dtype=torch.float32, device=stream.device)
    if rows == 0:
        return out_stream, pre_bf16, (codes, scales)
    block = triton.next_power_of_2(width)
    if block > 65536:
        raise ValueError("the boundary width is too wide for one Triton block")
    use_pdl = enable_pdl and current_platform().is_hopper_plus
    launch = {"launch_pdl": True} if use_pdl else {}
    _post_norm_residual_pre_norm_fp8_kernel[(rows,)](
        stream_values,
        branch_values,
        post_weight if post_weight is not None else stream_values,
        alpha.contiguous(),
        beta.contiguous(),
        pre_weight,
        out_stream,
        pre_bf16,
        codes,
        scales,
        width=width,
        post_eps=post_eps,
        post_weight_offset=post_weight_offset,
        pre_eps=pre_eps,
        pre_weight_offset=pre_weight_offset,
        POST_HAS_WEIGHT=post_weight is not None,
        BLOCK=block,
        ENABLE_PDL=use_pdl,
        # As the BF16 boundary kernel: row-wide vectors spill at 4 warps.
        num_warps=16,
        **launch,
    )
    return out_stream, pre_bf16, (codes, scales)
