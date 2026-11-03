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

"""Portable Triton kernels for per-route normalization and reduction."""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

_FLOAT_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


@triton.jit
def _gather_moe_routes_kernel(
    routes_ptr,
    map_ptr,
    output_ptr,
    numel,
    row_stride,
    width: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < numel
    row = tl.load(map_ptr + offsets // width, mask=valid, other=-1)
    source = tl.maximum(row, 0).to(tl.int64) * row_stride + offsets % width
    values = tl.load(routes_ptr + source, mask=valid & (row >= 0), other=0.0)
    tl.store(output_ptr + offsets, values, mask=valid)


@register_kernel(
    "moe",
    "gather_moe_routes",
    name="triton_gather_moe_routes",
    solution="triton",
    signatures=format_signatures("routes", "dense", _FLOAT_DTYPES),
    priority=Priority.PORTABLE,
)
def triton_gather_moe_routes(
    routes: torch.Tensor,
    route_map: torch.Tensor,
    num_tokens: int,
    topk: int,
    hidden_size: int,
    output: torch.Tensor | None,
) -> torch.Tensor:
    """Triton kernel of :func:`tokenspeed_kernel.ops.moe.gather_moe_routes`."""
    if routes.ndim != 2 or routes.shape[1] < hidden_size or routes.stride(1) != 1:
        raise ValueError(
            "routes must have shape [rows, padded_width] with contiguous columns"
        )
    if route_map.shape != (num_tokens * topk,) or not route_map.is_contiguous():
        raise ValueError("route_map must be contiguous with shape [tokens * topk]")
    if not routes.is_cuda or route_map.device != routes.device:
        raise ValueError("routes and route_map must be colocated GPU tensors")
    if num_tokens < 0 or topk <= 0 or hidden_size <= 0:
        raise ValueError("tokens must be nonnegative and topk and hidden_size positive")
    shape = (num_tokens, topk, hidden_size)
    if output is None:
        output = torch.empty(shape, device=routes.device, dtype=routes.dtype)
    elif (
        output.shape != shape
        or output.dtype != routes.dtype
        or output.device != routes.device
        or not output.is_contiguous()
    ):
        raise ValueError(
            "output must be contiguous [tokens, topk, hidden_size] matching routes"
        )
    if output.numel():
        output_end = output.data_ptr() + output.numel() * output.element_size()
        if routes.numel():
            source_end = (
                routes.data_ptr()
                + ((routes.shape[0] - 1) * routes.stride(0) + routes.shape[1])
                * routes.element_size()
            )
            if routes.data_ptr() < output_end and output.data_ptr() < source_end:
                raise ValueError("output must not overlap routes")
        map_end = route_map.data_ptr() + route_map.numel() * route_map.element_size()
        if route_map.data_ptr() < output_end and output.data_ptr() < map_end:
            raise ValueError("output must not overlap route_map")
    if output.numel() >= 2**31:
        raise ValueError(
            f"triton_gather_moe_routes uses 32-bit offsets; got {output.numel()} "
            "output elements"
        )
    if output.numel():
        _gather_moe_routes_kernel[(triton.cdiv(output.numel(), 1024),)](
            routes,
            route_map,
            output,
            output.numel(),
            routes.stride(0),
            hidden_size,
            1024,
        )
    return output


@triton.jit
def _centered_bf16(weight, offset: tl.constexpr):
    """Add the checkpoint weight center with BF16 rounding semantics."""
    return (weight.to(tl.float32) + offset).to(tl.bfloat16).to(tl.float32)


@triton.jit
def _routed_input_scale_inv_rms_kernel(
    routes_ptr,
    map_ptr,
    weight_ptr,
    inv_rms_ptr,
    row_stride,
    width: tl.constexpr,
    eps: tl.constexpr,
    weight_offset: tl.constexpr,
    HAS_MAP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    route = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < width
    if HAS_MAP:
        # -1 marks a route this rank does not own; it contributes zeros.
        row = tl.load(map_ptr + route)
        mask = mask & (row >= 0)
        row = tl.maximum(row, 0).to(tl.int64)
    else:
        row = route.to(tl.int64)
    values = tl.load(routes_ptr + row * row_stride + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    values *= _centered_bf16(weight, weight_offset)
    variance = tl.sum(values * values, axis=0) / width
    tl.store(inv_rms_ptr + route, tl.rsqrt(variance + eps))


@triton.jit
def _routed_input_scale_sum_kernel(
    routes_ptr,
    map_ptr,
    route_weights_ptr,
    norm_weight_ptr,
    inv_rms_ptr,
    out_ptr,
    num_tokens,
    row_stride,
    width: tl.constexpr,
    topk: tl.constexpr,
    weight_offset: tl.constexpr,
    HAS_MAP: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    tokens = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    dims = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (tokens[:, None] < num_tokens) & (dims[None, :] < width)
    norm_weight = tl.load(norm_weight_ptr + dims, mask=dims < width, other=0.0)
    norm_weight = _centered_bf16(norm_weight, weight_offset)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for slot in range(topk):
        route = tokens * topk + slot
        if HAS_MAP:
            row = tl.load(map_ptr + route, mask=tokens < num_tokens, other=-1)
            row_mask = mask & (row >= 0)[:, None]
            row = tl.maximum(row, 0).to(tl.int64)
        else:
            row_mask = mask
            row = route.to(tl.int64)
        values = tl.load(
            routes_ptr + row[:, None] * row_stride + dims[None, :],
            mask=row_mask,
            other=0.0,
        ).to(tl.float32)
        route_weight = tl.load(
            route_weights_ptr + route, mask=tokens < num_tokens, other=0.0
        ).to(tl.float32)
        inv_rms = tl.load(inv_rms_ptr + route, mask=tokens < num_tokens, other=0.0)
        scaled = values * norm_weight[None, :]
        acc += scaled * inv_rms[:, None] * route_weight[:, None]
    tl.store(out_ptr + tokens[:, None] * width + dims[None, :], acc, mask=mask)


def _routed_sum_block(
    routes: torch.Tensor,
    num_tokens: int,
    topk: int,
    route_weights: torch.Tensor,
    norm_weight: torch.Tensor,
) -> int:
    """Check the operands both routed sums share and return the Triton block."""
    width = norm_weight.shape[0]
    if route_weights.shape != (num_tokens, topk):
        raise ValueError("route_weights must be [tokens, topk]")
    if routes.dtype != norm_weight.dtype:
        raise TypeError("routes and norm_weight must have the same dtype")
    if not routes.is_cuda:
        raise ValueError("all operands must be GPU tensors")
    if topk <= 0 or width <= 0:
        raise ValueError("topk and width must be positive")
    block = triton.next_power_of_2(width)
    if block > 65536:
        raise ValueError("the route width is too large for one Triton reduction")
    return block


# Cold-L2 B300: one program per token (one launch) wins below five tokens.
ROUTED_SUM_SINGLE_PASS_MAX_TOKENS = 4


@register_kernel(
    "moe",
    "routed_input_scale_rmsnorm_sum",
    name="triton_routed_input_scale_rmsnorm_sum",
    solution="triton",
    signatures=format_signatures("routes", "dense", _FLOAT_DTYPES),
    priority=Priority.PORTABLE,
)
def triton_routed_input_scale_rmsnorm_sum(
    routes: torch.Tensor,
    route_weights: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    weight_offset: float,
    enable_pdl: bool,
    output: torch.Tensor | None,
) -> torch.Tensor:
    """Triton kernel of :func:`tokenspeed_kernel.ops.moe.routed_input_scale_rmsnorm_sum`."""
    if routes.ndim != 3:
        raise ValueError("routes must have shape [tokens, topk, width]")
    num_tokens, topk, width = routes.shape
    if norm_weight.shape != (width,):
        raise ValueError("norm_weight must match routes' final dimension")
    if routes.device != route_weights.device or routes.device != norm_weight.device:
        raise ValueError("routes, route_weights, and norm_weight must be colocated")
    if not route_weights.is_floating_point():
        raise TypeError("route_weights must have a floating-point dtype")
    block = _routed_sum_block(routes, num_tokens, topk, route_weights, norm_weight)
    routes_values = routes.contiguous()
    weights_values = route_weights.contiguous()
    if output is None:
        output = torch.empty(
            (num_tokens, width), dtype=routes.dtype, device=routes.device
        )
    elif (
        output.shape != (num_tokens, width)
        or output.dtype != routes.dtype
        or output.device != routes.device
        or not output.is_contiguous()
    ):
        raise ValueError(
            "output must be contiguous [tokens, width] matching routes' dtype"
        )
    elif output.numel():
        output_end = output.data_ptr() + output.numel() * output.element_size()
        for source in (routes_values, weights_values):
            source_end = source.data_ptr() + source.numel() * source.element_size()
            if source.data_ptr() < output_end and output.data_ptr() < source_end:
                raise ValueError("output must not overlap routes or route weights")
    if num_tokens == 0:
        return output
    _launch_routed_input_scale_sum(
        routes_values,
        None,
        width,
        weights_values,
        norm_weight,
        output,
        num_tokens,
        topk,
        eps,
        weight_offset,
        block,
        num_tokens <= ROUTED_SUM_SINGLE_PASS_MAX_TOKENS,
        enable_pdl,
    )
    return output


@triton.jit
def _routed_input_scale_rmsnorm_sum_token_kernel(
    routes_ptr,
    map_ptr,
    route_weights_ptr,
    norm_weight_ptr,
    out_ptr,
    row_stride,
    width: tl.constexpr,
    topk: tl.constexpr,
    eps: tl.constexpr,
    weight_offset: tl.constexpr,
    HAS_MAP: tl.constexpr,
    BLOCK: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    # Same FP32 product and rounding as the two-pass path; only the store rounds.
    token = tl.program_id(0)
    dims = tl.arange(0, BLOCK)
    dim_mask = dims < width
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
        # Dependents may launch now; their own wait still covers this grid's completion.
        tl.extra.cuda.gdc_launch_dependents()
    norm_weight = tl.load(norm_weight_ptr + dims, mask=dim_mask, other=0.0)
    norm_weight = _centered_bf16(norm_weight, weight_offset)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for slot in tl.static_range(topk):
        route = token * topk + slot
        if HAS_MAP:
            row = tl.load(map_ptr + route)
            row_mask = dim_mask & (row >= 0)
            row = tl.maximum(row, 0).to(tl.int64)
        else:
            row_mask = dim_mask
            row = route.to(tl.int64)
        values = tl.load(
            routes_ptr + row * row_stride + dims, mask=row_mask, other=0.0
        ).to(tl.float32)
        product = values * norm_weight
        variance = tl.sum(product * product, axis=0) / width
        inv_rms = tl.rsqrt(variance + eps)
        route_weight = tl.load(route_weights_ptr + route).to(tl.float32)
        acc += product * inv_rms * route_weight
    tl.store(out_ptr + token * width + dims, acc, mask=dim_mask)


def _launch_routed_input_scale_sum(
    routes: torch.Tensor,
    permute_map: torch.Tensor | None,
    row_stride: int,
    route_weights: torch.Tensor,
    norm_weight: torch.Tensor,
    output: torch.Tensor,
    num_tokens: int,
    topk: int,
    eps: float,
    weight_offset: float,
    block: int,
    single_pass: bool,
    enable_pdl: bool,
) -> None:
    """Reduce routes into ``output``: one program per token, or two tiled launches."""
    has_map = permute_map is not None
    map_arg = permute_map if has_map else routes
    width = norm_weight.shape[0]
    use_pdl = enable_pdl and current_platform().is_hopper_plus
    launch = {"launch_pdl": True} if use_pdl else {}
    if single_pass:
        _routed_input_scale_rmsnorm_sum_token_kernel[(num_tokens,)](
            routes,
            map_arg,
            route_weights,
            norm_weight,
            output,
            row_stride,
            width=width,
            topk=topk,
            eps=eps,
            weight_offset=weight_offset,
            HAS_MAP=has_map,
            BLOCK=block,
            ENABLE_PDL=use_pdl,
            num_warps=8,
            **launch,
        )
        return
    inv_rms = torch.empty(num_tokens * topk, dtype=torch.float32, device=routes.device)
    _routed_input_scale_inv_rms_kernel[(num_tokens * topk,)](
        routes,
        map_arg,
        norm_weight,
        inv_rms,
        row_stride,
        width=width,
        eps=eps,
        weight_offset=weight_offset,
        HAS_MAP=has_map,
        BLOCK=block,
        num_warps=8,
    )
    _routed_input_scale_sum_kernel[
        (triton.cdiv(num_tokens, 4), triton.cdiv(width, 256))
    ](
        routes,
        map_arg,
        route_weights,
        norm_weight,
        inv_rms,
        output,
        num_tokens,
        row_stride,
        width=width,
        topk=topk,
        weight_offset=weight_offset,
        HAS_MAP=has_map,
        BLOCK_M=4,
        BLOCK_N=256,
        num_warps=4,
    )


@register_kernel(
    "moe",
    "routed_input_scale_rmsnorm_sum_gathered",
    name="triton_routed_input_scale_rmsnorm_sum_gathered",
    solution="triton",
    signatures=format_signatures("gemm2_out", "dense", _FLOAT_DTYPES),
    priority=Priority.PORTABLE,
)
def triton_routed_input_scale_rmsnorm_sum_gathered(
    gemm2_out: torch.Tensor,
    permute_map: torch.Tensor,
    num_tokens: int,
    topk: int,
    route_weights: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    weight_offset: float,
    enable_pdl: bool,
    output: torch.Tensor | None,
) -> torch.Tensor:
    """Triton kernel of the routed normalized reduction dispatcher."""
    width = norm_weight.shape[0]
    if gemm2_out.ndim != 2 or gemm2_out.stride(1) != 1 or gemm2_out.shape[1] < width:
        raise ValueError("gemm2_out must be row-major [rows, >= width]")
    if permute_map.dtype != torch.int32 or permute_map.numel() != num_tokens * topk:
        raise ValueError("permute_map must be int32 with tokens * topk entries")
    block = _routed_sum_block(gemm2_out, num_tokens, topk, route_weights, norm_weight)
    output_shape = (num_tokens, width)
    if output is None:
        output = torch.empty(
            output_shape,
            dtype=gemm2_out.dtype,
            device=gemm2_out.device,
        )
    elif (
        output.shape != output_shape
        or output.dtype != gemm2_out.dtype
        or output.device != gemm2_out.device
        or not output.is_contiguous()
    ):
        raise ValueError(
            "output must be a contiguous tensor matching the route-sum shape, "
            "dtype, and device"
        )
    if num_tokens == 0:
        return output
    _launch_routed_input_scale_sum(
        gemm2_out,
        permute_map.contiguous(),
        gemm2_out.stride(0),
        route_weights.contiguous(),
        norm_weight,
        output,
        num_tokens,
        topk,
        eps,
        weight_offset,
        block,
        num_tokens <= ROUTED_SUM_SINGLE_PASS_MAX_TOKENS,
        enable_pdl,
    )
    return output


__all__ = [
    "triton_routed_input_scale_rmsnorm_sum",
    "triton_routed_input_scale_rmsnorm_sum_gathered",
]
