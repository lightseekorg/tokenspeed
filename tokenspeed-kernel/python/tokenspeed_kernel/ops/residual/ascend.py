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


"""Ascend mHC mixing, fused normalization, and Engram residual gating."""

import torch
from tokenspeed_kernel.platform import CapabilityRequirement, current_platform
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

if current_platform().is_npu:

    @register_kernel(
        "residual",
        "mhc_mixes",
        name="ascend_mhc_mixes",
        solution="triton_ascend",
        capability=CapabilityRequirement(vendors=frozenset({"ascend"})),
        signatures=frozenset(
            {format_signature(residual=dense_tensor_format(torch.bfloat16))}
        ),
        priority=Priority.PERFORMANT,
    )
    def ascend_mhc_mixes(
        residual, weight, scale, base, rms_eps, hc_eps, sinkhorn_iters
    ):
        from tokenspeed_kernel_npu.ops.dsv41_mhc import mhc_mixes

        return mhc_mixes(residual, weight, scale, base, rms_eps, hc_eps, sinkhorn_iters)

    @register_kernel(
        "residual",
        "mhc_apply_pre_norm",
        name="ascend_mhc_apply_pre_norm",
        solution="triton_ascend",
        capability=CapabilityRequirement(vendors=frozenset({"ascend"})),
        signatures=frozenset(
            {format_signature(residual=dense_tensor_format(torch.bfloat16))}
        ),
        priority=Priority.PERFORMANT,
    )
    def ascend_mhc_apply_pre_norm(residual, pre, weight, eps):
        from tokenspeed_kernel_npu.ops.dsv41_mhc import mhc_pre_norm

        return mhc_pre_norm(residual, pre, weight, eps)

    @register_kernel(
        "residual",
        "mhc_post",
        name="ascend_mhc_post",
        solution="triton_ascend",
        capability=CapabilityRequirement(vendors=frozenset({"ascend"})),
        signatures=frozenset(
            {
                format_signature(
                    hidden_states=dense_tensor_format(torch.bfloat16),
                    residual=dense_tensor_format(torch.bfloat16),
                    post=dense_tensor_format(torch.float32),
                    comb=dense_tensor_format(torch.float32),
                )
            }
        ),
        priority=Priority.PERFORMANT,
    )
    def ascend_mhc_post(hidden_states, residual, post, comb):
        from tokenspeed_kernel_npu.ops.dsv41_mhc import mhc_post

        if residual.ndim != 3 or residual.shape[1] != 4:
            raise ValueError("Ascend mHC post requires residual [T,4,H]")
        n, _, h = residual.shape
        if (
            hidden_states.shape != (n, h)
            or post.shape != (n, 4, 1)
            or comb.shape != (n, 4, 4)
        ):
            raise ValueError("Ascend mHC post input shapes disagree")
        return mhc_post(hidden_states, residual, post, comb)

    @register_kernel(
        "residual",
        "normalized_dot_gate",
        name="ascend_normalized_dot_gate",
        solution="triton_ascend",
        capability=CapabilityRequirement(vendors=frozenset({"ascend"})),
        signatures=frozenset(
            {format_signature(residual=dense_tensor_format(torch.bfloat16))}
        ),
        priority=Priority.PERFORMANT,
    )
    def ascend_normalized_dot_gate(
        residual, key_value, query_weight, key_weight, mask, eps
    ):
        from tokenspeed_kernel_npu.ops.normalized_dot_gate import normalized_dot_gate

        return normalized_dot_gate(
            residual, key_value, query_weight, key_weight, mask, eps
        )
