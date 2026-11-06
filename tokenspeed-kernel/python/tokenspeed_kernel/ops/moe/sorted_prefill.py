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

"""MXFP4 MoE prefill split at the expert intermediate.

Under expert tensor parallelism each rank's gate/up GEMM produces one column
slice of every route's intermediate. ``moe_sorted_gate_up`` writes that slice
as MXFP4 rows in a sorted route order that depends only on the top-k IDs, so
the slices of all ranks line up row by row and concatenate along K.
``moe_sorted_down`` then projects complete rows through a down weight that
``prepare_moe_sorted_down_weight`` laid out, typically a slice of the down
projection's output columns.
"""

from typing import Any

import torch
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import dense_format_signature


def moe_sorted_gate_up(
    x_quantized: tuple[torch.Tensor, torch.Tensor],
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w: torch.nn.Module,
    intermediate: torch.Tensor,
    intermediate_scale: torch.Tensor,
) -> Any:
    """Write each route's gate/up activation as sorted MXFP4 rows.

    Args:
        x_quantized: Token-order E2M1 rows ``[tokens, hidden / 2]`` and their
            CDNA4-swizzled E8M0 scales.
        topk_weights: FP32 route weights ``[tokens, top_k]``.
        topk_ids: Expert IDs ``[tokens, top_k]``; every expert must be local.
        w: The processed MoE layer whose gate/up weight to apply.
        intermediate: uint8 ``[>= capacity, I / 2]`` destination, where I is
            the layer's local intermediate width and the capacity is the
            kernel's largest padded route count for this many tokens.
        intermediate_scale: Contiguous uint8 ``[>= capacity rounded up to 32,
            I / 32]`` scale destination.

    Returns:
        An opaque sorted-route handle for ``moe_sorted_down``.
    """
    kernel = select_kernel(
        "moe", "sorted_gate_up", dense_format_signature(("x",), (torch.uint8,))
    )
    return kernel(
        x_quantized, topk_weights, topk_ids, w, intermediate, intermediate_scale
    )


def _sorted_down_kernel(dtype: torch.dtype):
    return select_kernel("moe", "sorted_down", dense_format_signature(("x",), (dtype,)))


def moe_sorted_down(
    intermediate: torch.Tensor,
    intermediate_scale: torch.Tensor,
    routes: Any,
    weight: torch.Tensor,
    deferred_out: torch.Tensor,
) -> torch.Tensor:
    """Project sorted MXFP4 rows into unweighted routes in token/slot order.

    Args:
        intermediate: uint8 sorted rows covering the weight's full K; the row
            pitch may exceed ``K / 2`` bytes.
        intermediate_scale: Contiguous CDNA4 scale panels, ``K / 32`` columns.
        routes: Handle from ``moe_sorted_gate_up``.
        weight: Output of ``prepare_moe_sorted_down_weight`` with N columns.
        deferred_out: Contiguous BF16 ``[tokens * top_k, N]`` destination.

    Returns:
        ``deferred_out``, each route accumulated over K in FP32 and rounded
        once to BF16.
    """
    kernel = _sorted_down_kernel(intermediate.dtype)
    return kernel(intermediate, intermediate_scale, routes, weight, deferred_out)


def prepare_moe_sorted_down_weight(
    weight: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """Lay out MXFP4 expert down weights ``[experts, N, K / 2]`` for ``moe_sorted_down``.

    Args:
        weight: Packed E2M1 weights ``[experts, N, K / 2]``.
        scale: E8M0 block scales ``[experts, N, K / 32]``.

    Returns:
        The weight in the down kernel's layout; the inputs may be released.
    """
    kernel = _sorted_down_kernel(weight.dtype)
    return kernel.impl._tokenspeed_weight_preprocessor(weight, scale)


def warmup_moe_sorted_prefill(
    w: torch.nn.Module, weight: torch.Tensor, min_rows: int, max_rows: int
) -> None:
    """Compile both sorted halves for every token count in a range.

    Args:
        w: The processed MoE layer ``moe_sorted_gate_up`` will run.
        weight: A down weight from ``prepare_moe_sorted_down_weight``.
        min_rows: Smallest token count the caller runs.
        max_rows: Largest token count the caller runs.
    """
    kernel = _sorted_down_kernel(torch.uint8)
    kernel.impl._tokenspeed_warmup(w, weight, min_rows, max_rows)


__all__ = [
    "moe_sorted_down",
    "moe_sorted_gate_up",
    "prepare_moe_sorted_down_weight",
    "warmup_moe_sorted_prefill",
]
