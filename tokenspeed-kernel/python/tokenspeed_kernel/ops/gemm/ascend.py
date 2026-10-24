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

"""Ascend BF16 projections with FP32 accumulation."""

import torch
from tokenspeed_kernel.platform import CapabilityRequirement, current_platform
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

if current_platform().is_npu:

    @register_kernel(
        "gemm",
        "mm",
        name="ascend_bf16_wide_projection",
        solution="triton_ascend",
        capability=CapabilityRequirement(vendors=frozenset({"ascend"})),
        signatures=frozenset(
            {
                format_signature(
                    a=dense_tensor_format(torch.bfloat16),
                    b=dense_tensor_format(torch.bfloat16),
                )
            }
        ),
        traits={
            "n": frozenset({1600, 3200, 6400}),
            "k": frozenset({6144}),
            "mnk_problem_filter": frozenset({lambda m, n, k: 0 < m <= 384}),
            "a_inner_stride_one": frozenset({True}),
            "b_inner_stride_one": frozenset({True}),
            "b_layout": frozenset({"NK"}),
            "out_dtype": frozenset({torch.bfloat16}),
            "out_inner_stride_one": frozenset({True}),
            "pdl_enabled": frozenset({False}),
        },
        priority=Priority.PERFORMANT,
    )
    def ascend_bf16_wide_projection(
        A, B, A_scales, B_scales, out_dtype, *, alpha, block_size, out=None
    ):
        if any(value is not None for value in (A_scales, B_scales, alpha, block_size)):
            raise ValueError("BF16 projection does not accept quantization parameters")
        if out_dtype != torch.bfloat16:
            raise ValueError("BF16 projection requires BF16 output")
        from tokenspeed_kernel_npu.ops.bf16_linear_fp32 import bf16_wide_projection

        return bf16_wide_projection(A, B, out)

    @register_kernel(
        "gemm",
        "mm",
        name="ascend_bf16_short_projection",
        solution="triton_ascend",
        capability=CapabilityRequirement(vendors=frozenset({"ascend"})),
        signatures=frozenset(
            {
                format_signature(
                    a=dense_tensor_format(torch.bfloat16),
                    b=dense_tensor_format(torch.bfloat16),
                )
            }
        ),
        traits={
            "n": frozenset({5120}),
            "k": frozenset({288}),
            "mnk_problem_filter": frozenset({lambda m, n, k: 0 < m <= 384}),
            "a_inner_stride_one": frozenset({True}),
            "b_inner_stride_one": frozenset({True}),
            "b_layout": frozenset({"NK"}),
            "out_dtype": frozenset({torch.bfloat16}),
            "out_inner_stride_one": frozenset({True}),
            "pdl_enabled": frozenset({False}),
        },
        priority=Priority.PERFORMANT,
    )
    def ascend_bf16_short_projection(
        A, B, A_scales, B_scales, out_dtype, *, alpha, block_size, out=None
    ):
        if any(value is not None for value in (A_scales, B_scales, alpha, block_size)):
            raise ValueError("BF16 projection does not accept quantization parameters")
        if out_dtype != torch.bfloat16:
            raise ValueError("BF16 projection requires BF16 output")
        from tokenspeed_kernel_npu.ops.bf16_linear_fp32 import bf16_short_projection

        return bf16_short_projection(A, B, out)

    @register_kernel(
        "gemm",
        "dsv4_linear_fp32",
        name="ascend_bf16_linear_fp32",
        solution="triton_ascend",
        capability=CapabilityRequirement(vendors=frozenset({"ascend"})),
        signatures=frozenset(
            {
                format_signature(
                    hidden_states=dense_tensor_format(torch.bfloat16),
                    weight=dense_tensor_format(torch.bfloat16),
                )
            }
        ),
        traits={},
        priority=Priority.PERFORMANT,
    )
    def ascend_bf16_linear_fp32(hidden_states, weight, enable_pdl):
        if enable_pdl:
            raise ValueError("Ascend does not support CUDA PDL")
        from tokenspeed_kernel_npu.ops.bf16_linear_fp32 import bf16_linear_fp32

        return bf16_linear_fp32(hidden_states, weight)

    @register_kernel(
        "gemm",
        "grouped_bf16_projection",
        name="ascend_grouped_bf16_projection",
        solution="triton_ascend",
        capability=CapabilityRequirement(vendors=frozenset({"ascend"})),
        signatures=frozenset(
            {
                format_signature(
                    x=dense_tensor_format(torch.bfloat16),
                    weight=dense_tensor_format(torch.bfloat16),
                )
            }
        ),
        traits={
            "batch": frozenset({2}),
            "m": frozenset({48}),
            "n": frozenset({1024}),
            "k": frozenset({4096}),
            "a_inner_stride_one": frozenset({True}),
            "b_inner_stride_one": frozenset({True}),
            "is_cuda": frozenset({False}),
        },
        priority=Priority.PERFORMANT,
    )
    def ascend_grouped_bf16_projection(x, weight, out):
        from tokenspeed_kernel_npu.ops.grouped_bf16_projection import (
            grouped_bf16_projection,
        )

        return grouped_bf16_projection(x, weight, out)
