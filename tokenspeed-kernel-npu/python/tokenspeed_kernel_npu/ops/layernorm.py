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

"""Ascend normalization operators."""

from __future__ import annotations

import torch
import torch_npu


def rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    residual: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Apply RMSNorm, optionally adding and updating a residual tensor."""
    if x.shape[0] == 0:
        result = x if out is None else out
        return (result, residual) if residual is not None else result
    if residual is not None:
        result, _, residual_out = torch_npu.npu_add_rms_norm(
            x, residual, weight, epsilon=eps
        )
        residual.copy_(residual_out)
        return result, residual
    result, _ = torch_npu.npu_rms_norm(x, weight, epsilon=eps)
    if out is not None:
        out.copy_(result)
        result = out
    return result


def qk_rmsnorm(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    enable_pdl: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply per-head RMSNorm to query and key tensors."""
    del enable_pdl
    head_dim = q_weight.numel()
    if k_weight.numel() != head_dim:
        raise ValueError("q and k weights must have the same head dimension")
    if q.shape[-1] % head_dim or k.shape[-1] % head_dim:
        raise ValueError("q and k widths must be divisible by the head dimension")

    q_out = rmsnorm(q.reshape(-1, head_dim), q_weight, eps).reshape_as(q)
    k_out = rmsnorm(k.reshape(-1, head_dim), k_weight, eps).reshape_as(k)
    return q_out, k_out


__all__ = ["qk_rmsnorm", "rmsnorm"]
