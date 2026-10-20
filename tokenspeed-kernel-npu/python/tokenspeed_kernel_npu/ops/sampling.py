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

"""Ascend graph-safe sampling primitives provided by flash-npu-kernel."""

from __future__ import annotations

import torch

try:
    from flash_ops.npu_sample import (
        SamplingWorkspace,
        chain_speculative_sampling_target_only,
        fused_topk_topp_renorm,
        sample_from_probs,
        verify_chain_greedy,
    )
except ImportError as exc:
    raise ImportError(
        "Ascend full sampling requires flash_ops built with the npu_sample "
        "operators (fused_topk_topp_renorm and sorted_topk_topp_renorm)"
    ) from exc

if not hasattr(torch.ops.custom, "fused_topk_topp_renorm"):
    raise RuntimeError(
        "flash_ops.npu_sample was imported, but the "
        "custom::fused_topk_topp_renorm operator is not registered; rebuild "
        "flash-npu-kernel with the npu_sample operators and matching OPP"
    )


def min_p_renorm_prob(
    probs: torch.Tensor,
    min_ps: torch.Tensor,
) -> torch.Tensor:
    """Apply row-wise min-p filtering and renormalize in FP32.

    The implementation is intentionally composed from capturable tensor ops.
    It is a correctness path until the operation is folded into the Ascend
    filtering kernel.
    """
    if probs.ndim != 2 or probs.dtype != torch.float32:
        raise ValueError("probs must be a 2D float32 tensor")
    if min_ps.shape != (probs.shape[0],) or min_ps.dtype != torch.float32:
        raise ValueError("min_ps must be float32 with one value per row")
    thresholds = probs.amax(dim=-1, keepdim=True) * min_ps.unsqueeze(-1)
    filtered = torch.where(probs >= thresholds, probs, torch.zeros_like(probs))
    normalizer = filtered.sum(dim=-1, keepdim=True)
    return filtered / normalizer.clamp_min(torch.finfo(torch.float32).tiny)


__all__ = [
    "SamplingWorkspace",
    "chain_speculative_sampling_target_only",
    "fused_topk_topp_renorm",
    "min_p_renorm_prob",
    "sample_from_probs",
    "verify_chain_greedy",
]
