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

"""Unfused grouped-expert MoE apply (two grouped GEMMs + external activation).

The fused MoE path (``moe_apply``) bakes the gate/up activation into the expert
kernel, which only supports SiLU / SwiGLU. To run an activation the fused kernels
don't have (e.g. Kimi-K3's SiTU), run the experts as two grouped GEMMs
(``matmul_ogs``) with the activation applied *between* them. This is the
boundary-correct entry a model uses: runtime code depends only on
``tokenspeed_kernel``, never ``triton_kernels`` directly.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from tokenspeed_kernel.ops.other.moe_grouped_routing.torch import moe_grouped_routing


def moe_unfused_apply(
    x: torch.Tensor,
    w13,
    w13_precision_config,
    w2,
    w2_precision_config,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
    top_k: int,
    activation: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Route top-k, then run ``gemm1 -> activation -> gemm2`` over ragged experts.

    Args:
        x: expert input activations ``[num_rows, K]`` (unquantized; weight-only
            MXFP4 keeps the activation in bf16).
        w13: gate/up expert weights as a ``triton_kernels`` tensor.
        w13_precision_config: ``triton_kernels`` ``PrecisionConfig`` for gemm1.
        w2: down-projection expert weights as a ``triton_kernels`` tensor.
        w2_precision_config: ``PrecisionConfig`` for gemm2.
        topk_weights: per-token routing weights ``[num_tokens, top_k]``.
        topk_ids: per-token selected expert ids ``[num_tokens, top_k]``.
        num_experts: total number of routed experts.
        top_k: experts selected per token.
        activation: applied to the gemm1 output between the two GEMMs (e.g. the
            model's SiTU gate/up activation).

    Returns:
        ``[num_tokens, out_dim]`` -- the expert output gate-weighted-combined over
        the ``top_k`` routed experts. NOT reduced across tensor-parallel ranks:
        when the expert weights are TP-sharded on the intermediate dim each rank
        returns a partial sum, so the caller must all-reduce.
    """
    # Lazy import: ``triton_kernels`` is an optional Triton backend, kept out of
    # module import time (and off the runtime's dependency surface).
    from triton_kernels.matmul import matmul

    meta, gather_indx, scatter_indx, gammas = moe_grouped_routing(
        topk_weights, topk_ids, num_experts
    )
    # gemm1: scatter tokens to their experts (gather_indx); gate/up projection.
    gemm1 = matmul(
        x,
        w13,
        None,
        a_ragged_metadata=meta,
        gather_indx=gather_indx,
        precision_config=w13_precision_config,
        fused_activation=None,
    )
    gemm1 = activation(gemm1)
    # gemm2: down projection; gather back to tokens + gate-weighted combine.
    out = matmul(
        gemm1,
        w2,
        None,
        a_ragged_metadata=meta,
        scatter_indx=scatter_indx,
        gammas=gammas,
        precision_config=w2_precision_config,
        fused_activation=None,
    )
    return out.view(x.size(0), top_k, -1).sum(dim=1)
