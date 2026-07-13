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

"""Manifold-constrained hyper-connection kernels."""

from tokenspeed_kernel.ops.mhc.deep_gemm import (
    deep_gemm_mhc_prenorm_gemm,
    has_deep_gemm_mhc,
)
from tokenspeed_kernel.ops.mhc.trtllm import (
    FUSED_HC_MAX_K_SPLITS,
    has_trtllm_mhc,
    supports_trtllm_mhc,
    trtllm_mhc_big_fuse,
    trtllm_mhc_fused_hc,
    trtllm_mhc_post_mapping,
)

__all__ = [
    "FUSED_HC_MAX_K_SPLITS",
    "deep_gemm_mhc_prenorm_gemm",
    "has_deep_gemm_mhc",
    "has_trtllm_mhc",
    "supports_trtllm_mhc",
    "trtllm_mhc_big_fuse",
    "trtllm_mhc_fused_hc",
    "trtllm_mhc_post_mapping",
]
