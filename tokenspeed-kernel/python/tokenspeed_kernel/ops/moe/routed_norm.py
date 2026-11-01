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

"""Per-route normalization and weighted reduction entry points."""

import torch
from tokenspeed_kernel.ops.moe._cute_dsl import routed_norm as _cute_dsl
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import dense_format_signature


def gather_moe_routes(
    routes: torch.Tensor,
    route_map: torch.Tensor,
    num_tokens: int,
    topk: int,
    hidden_size: int,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gather deferred expert rows into token/route order; absent routes are zero.

    Args:
        routes: Expert outputs ``[rows, padded_width]``.
        route_map: Contiguous ``[tokens * topk]`` row indices; ``-1`` marks absence.
        num_tokens: Number of tokens.
        topk: Routes per token.
        hidden_size: Output width (``<= padded_width``).
        output: Optional contiguous destination that overlaps no input.

    Returns:
        ``[tokens, topk, hidden_size]`` routes in ``routes.dtype``.
    """
    kernel = select_kernel(
        "moe",
        "gather_moe_routes",
        dense_format_signature(("routes",), (routes.dtype,)),
    )
    return kernel(routes, route_map, num_tokens, topk, hidden_size, output)


def routed_input_scale_rmsnorm_sum(
    routes: torch.Tensor,
    route_weights: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    weight_offset: float,
    enable_pdl: bool,
    output: torch.Tensor | None,
) -> torch.Tensor:
    """Normalize each expert route before its weighted top-k reduction.

    Scaling, RMS normalization, route weighting, and top-k accumulation use
    FP32. The result is rounded once when stored in ``routes.dtype``.

    Args:
        routes: Expert results shaped ``[tokens, topk, width]``.
        route_weights: FP32 route weights shaped ``[tokens, topk]``.
        norm_weight: Learned input-scale RMSNorm weight shaped ``[width]``.
        eps: Epsilon for each route's variance.
        weight_offset: Center added to the checkpoint's norm weight.
        enable_pdl: Enable NVIDIA programmatic dependent launch.
        output: Optional contiguous ``[tokens, width]`` destination in
            ``routes.dtype`` that overlaps no input; None allocates.

    Returns:
        Weighted route sum shaped ``[tokens, width]``, backed by ``output`` when given.
    """
    kernel = select_kernel(
        "moe",
        "routed_input_scale_rmsnorm_sum",
        dense_format_signature(("routes",), (routes.dtype,)),
    )
    return kernel(
        routes, route_weights, norm_weight, eps, weight_offset, enable_pdl, output
    )


def routed_input_scale_rmsnorm_sum_gathered(
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
    """Normalize and reduce expert routes straight out of a deferred MoE finalize.

    Up to ``cute_dsl.CLUSTER_ROUTED_SUM_MAX_TOKENS`` tokens run on the CuTe
    DSL cluster kernel where it is available; everything else, and every
    other platform, runs the Triton launches, both under this contract; each
    route is read through ``permute_map``, where ``-1`` contributes zeros.
    They sum each route's variance in a different order, so batches on either side of the ceiling agree to the last bit of
    the inverse RMS rather than bitwise, and each is deterministic.

    Args:
        gemm2_out: Permuted expert outputs ``[rows, padded_width]`` in BF16.
        permute_map: Int32 ``[tokens * topk]`` expanded-to-permuted row map,
            ``-1`` for a route another rank owns.
        num_tokens: Number of tokens.
        topk: Routes per token.
        route_weights: FP32 route weights ``[tokens, topk]``.
        norm_weight: Learned input-scale RMSNorm weight ``[width]``.
        eps: Epsilon for each route's variance.
        weight_offset: Center added to the checkpoint's norm weight.
        enable_pdl: Enable NVIDIA programmatic dependent launch.
        output: Optional caller-owned contiguous output. Supplying it selects
            the portable Triton implementation so the producer can write
            directly into downstream storage.

    Returns:
        Weighted route sum ``[tokens, width]`` in ``gemm2_out``'s dtype.
    """
    kernel = select_kernel(
        "moe",
        "routed_input_scale_rmsnorm_sum_gathered",
        dense_format_signature(("gemm2_out",), (gemm2_out.dtype,)),
        traits={
            "cluster_supported": output is None
            and _cute_dsl.cluster_routed_sum_supported(
                gemm2_out,
                permute_map,
                num_tokens,
                topk,
                route_weights,
                norm_weight,
            )
        },
    )
    return kernel(
        gemm2_out,
        permute_map,
        num_tokens,
        topk,
        route_weights,
        norm_weight,
        eps,
        weight_offset,
        enable_pdl,
        output,
    )


__all__ = [
    "gather_moe_routes",
    "routed_input_scale_rmsnorm_sum",
    "routed_input_scale_rmsnorm_sum_gathered",
]
