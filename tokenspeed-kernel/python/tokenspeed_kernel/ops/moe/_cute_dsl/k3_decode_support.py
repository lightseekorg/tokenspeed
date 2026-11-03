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

"""What the Kimi-K3 TP8 decode chain serves, importable on any platform.

``k3_decode`` holds the CuTe DSL kernels and needs ``cutlass``; callers decide
with this module first and import that one only where the chain applies.
"""

from __future__ import annotations

import torch

HIDDEN, LATENT, EXPERTS = 7168, 3584, 896
# Decode batches the chain is compiled for; others take the ordinary MoE.
BATCHES = (*range(1, 9), 16, 32, 64)


def supports(
    *,
    tp_size: int,
    hidden_size: int,
    latent_size: int,
    num_experts: int,
    top_k: int,
    num_groups: int,
    topk_groups: int,
    scoring_func: str,
    linear_beta: float | None,
    router_dtype: torch.dtype,
    expert_dtype: str,
    pdl: bool,
    batch_invariant: bool,
) -> bool:
    """Whether the chain serves a MoE layer.

    Args:
        tp_size: The tensor-parallel group the experts and latent are split over.
        hidden_size, latent_size: Model and routed-latent widths.
        num_experts, top_k, num_groups, topk_groups, scoring_func: Routing.
        linear_beta: SiTU's linear cap (the kernels need one).
        router_dtype: The router weight's dtype.
        expert_dtype: The routed experts' weight format.
        pdl: Whether programmatic dependent launch is enabled; the chain always
            launches with it.
        batch_invariant: Whether reductions must not depend on the batch; the
            chain reduces in its own order.

    Returns:
        True for TP8 MXFP4 experts behind an ungrouped sigmoid top-16 router over
        896 experts, with this model's widths, outside batch-invariant modes.
    """
    return (
        tp_size == 8
        and hidden_size == HIDDEN
        and latent_size == LATENT
        and num_experts == EXPERTS
        and top_k == 16
        and num_groups == topk_groups == 1
        and scoring_func == "sigmoid"
        and linear_beta is not None
        and router_dtype == torch.bfloat16
        and expert_dtype == "mxfp4"
        and pdl
        and not batch_invariant
    )
