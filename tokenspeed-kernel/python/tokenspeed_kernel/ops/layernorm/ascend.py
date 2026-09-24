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

"""Ascend layernorm kernels."""

import torch
from tokenspeed_kernel_npu.ops.layernorm import qk_rmsnorm as _npu_qk_rmsnorm
from tokenspeed_kernel_npu.ops.layernorm import rmsnorm

__all__ = ["qk_rmsnorm", "rmsnorm"]


def qk_rmsnorm(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    *,
    weight_offset: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The NPU per-head norm, with ``weight_offset + weight`` formed in the weight dtype."""
    if weight_offset:
        q_weight = (q_weight.float() + weight_offset).to(q_weight.dtype)
        k_weight = (k_weight.float() + weight_offset).to(k_weight.dtype)
    return _npu_qk_rmsnorm(q, k, q_weight, k_weight, eps)
