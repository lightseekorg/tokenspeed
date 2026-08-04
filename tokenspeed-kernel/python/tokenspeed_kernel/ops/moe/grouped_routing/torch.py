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

"""Public routing-metadata entry point for the unfused grouped-expert MoE path.

Turns a top-k routing (``topk_weights`` / ``topk_ids``) into the ragged
per-expert segmentation + gather/scatter indices + gate scales that the grouped
GEMM (``matmul_ogs``) consumes; used by ``moe_unfused_apply`` to drive the two
expert GEMMs.

This is a thin public alias over the fused Triton MXFP4 MoE's own
``_routing_from_topk`` — the single source of truth already shared by the
mxfp4 / fp8 backends. Exposing it (rather than re-deriving the argsort /
scatter-add) keeps one implementation and gives models a stable import that is
not coupled to a specific backend module.
"""

from __future__ import annotations

import torch


def moe_grouped_routing(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
    dtype: torch.dtype | None = None,
):
    """Build ``(ragged_metadata, gather_indx, scatter_indx, gammas)`` from top-k.

    Args:
        topk_weights: ``[num_tokens, top_k]`` per-token expert weights.
        topk_ids: ``[num_tokens, top_k]`` selected expert ids (``< 0`` = dropped).
        num_experts: total number of (physical) experts.
        dtype: optional dtype for the returned gate scales.

    Returns:
        ``ragged_metadata`` (per-expert row segmentation), ``gather_indx``
        (gemm1 token->row), ``scatter_indx`` (gemm2 row->token), ``gammas``
        (per-row gate weights for the gemm2 combine).
    """
    # Lazy import: the Triton MXFP4 backend pulls in the optional
    # ``triton_kernels`` dependency, so keep it out of module import time.
    from tokenspeed_kernel.ops.moe.mxfp4.triton import _routing_from_topk

    return _routing_from_topk(topk_weights, topk_ids, num_experts, dtype=dtype)
