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

"""Triton residual kernels."""

"""SM120 compensated Tensor Core projection for V4.1 mHC coefficients."""

import torch
from tokenspeed_kernel._triton import triton
from tokenspeed_kernel.ops.residual.triton import (
    _mhc_mixes_impl,
    _mhc_prenorm_gemm_launch_config,
    _mhc_prenorm_gemm_triton_kernel,
    mhc_pre_mix_hc4,
)
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature


def _prenorm_gemm(x, fn, out_mul, out_sqrsum, n_splits):
    tokens, k = x.shape
    n = fn.shape[0]
    bm, bn, bk, warps, stages = _mhc_prenorm_gemm_launch_config(tokens, k, n, n_splits)
    split_k = triton.cdiv(triton.cdiv(k, n_splits), bk) * bk
    # Keep row tiles below 64: larger tiles may lower tf32x3 to plain TF32.
    # FP32 weights stay FP32; this is compensated multiplication, not a
    # permanent TF32/BF16 weight conversion. Squared sums are unchanged.
    _mhc_prenorm_gemm_triton_kernel[
        (n_splits, triton.cdiv(tokens, bm), triton.cdiv(n, bn))
    ](
        x,
        fn,
        out_mul,
        out_sqrsum,
        tokens,
        K=k,
        N=n,
        SPLIT_K=split_k,
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
        DOT_PRECISION="tf32x3",
        num_warps=warps,
        num_stages=stages,
    )


@register_kernel(
    "residual",
    "mhc_mixes",
    name="triton_sm120_mhc_mixes",
    solution="triton_sm120",
    capability=CapabilityRequirement(
        vendors=frozenset({"nvidia"}),
        min_arch_version=ArchVersion(12, 0),
        max_arch_version=ArchVersion(12, 1),
    ),
    signatures=frozenset(
        {format_signature(residual=dense_tensor_format(torch.bfloat16))}
    ),
    priority=Priority.SPECIALIZED,
)
def triton_sm120_mhc_mixes(
    residual, weight, scale, base, rms_eps, hc_eps, sinkhorn_iters
):
    return _mhc_mixes_impl(
        residual,
        weight,
        scale,
        base,
        rms_eps,
        hc_eps,
        sinkhorn_iters,
        _prenorm_gemm,
        mhc_pre_mix_hc4,
    )
