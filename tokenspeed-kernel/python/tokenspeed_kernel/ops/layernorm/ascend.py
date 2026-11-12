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
from tokenspeed_kernel.platform import CapabilityRequirement, current_platform
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures
from tokenspeed_kernel_npu.ops.layernorm import gated_rmsnorm as _npu_gated_rmsnorm
from tokenspeed_kernel_npu.ops.layernorm import gemma_rmsnorm as _npu_gemma_rmsnorm
from tokenspeed_kernel_npu.ops.layernorm import qk_rmsnorm as _npu_qk_rmsnorm
from tokenspeed_kernel_npu.ops.layernorm import rmsnorm

__all__ = ["gated_rmsnorm", "gemma_rmsnorm", "qk_rmsnorm", "rmsnorm"]

_CAPABILITY = CapabilityRequirement(vendors=frozenset({"ascend"}))
_DTYPES = {torch.bfloat16, torch.float16, torch.float32}


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


if current_platform().is_npu:

    @register_kernel(
        "layernorm",
        "gemma_rmsnorm",
        name="ascend_gemma_rmsnorm",
        solution="torch_npu",
        capability=_CAPABILITY,
        signatures=format_signatures("x", "dense", _DTYPES),
        priority=Priority.SPECIALIZED,
        features={"ascend"},
    )
    def gemma_rmsnorm(
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        residual: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """The NPU Gemma norm, folding ``1 + weight`` into the scale first."""
        return _npu_gemma_rmsnorm(x, weight, eps, residual)

    @register_kernel(
        "layernorm",
        "gated_rmsnorm",
        name="ascend_gated_rmsnorm",
        solution="torch_npu",
        capability=_CAPABILITY,
        signatures=format_signatures("x", "dense", _DTYPES),
        priority=Priority.SPECIALIZED,
        traits={
            # The leaf composes the op from elementwise math, so it serves any
            # group size; only the FP8 output is unavailable, which keeps the
            # quantized call sites on the portable solution.
            "fp8_out": frozenset({False}),
        },
        features={"ascend"},
    )
    def gated_rmsnorm(
        x: torch.Tensor,
        weight: torch.Tensor,
        z: torch.Tensor | None,
        eps: float,
        group_size: int,
        norm_before_gate: bool,
        sigmoid_gate: bool,
        fp8_scale: torch.Tensor | None,
        weights_independent: bool,
    ) -> torch.Tensor:
        """The NPU grouped gated norm; see ``tokenspeed_kernel_npu.ops.layernorm``."""
        return _npu_gated_rmsnorm(
            x,
            weight,
            z,
            eps=eps,
            group_size=group_size,
            norm_before_gate=norm_before_gate,
            sigmoid_gate=sigmoid_gate,
            fp8_scale=fp8_scale,
            weights_independent=weights_independent,
        )
