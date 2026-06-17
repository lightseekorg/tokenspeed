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

from tokenspeed_kernel.profiling import bootstrap_profiling_from_env

bootstrap_profiling_from_env()

from tokenspeed_kernel.ops.attention import (
    attn_merge_state,
    attn_plan,
    mha_decode_with_kvcache,
    mha_extend_with_kvcache,
    mha_prefill,
    mla_decode_with_kvcache,
    mla_prefill,
)
from tokenspeed_kernel.ops.gemm import mm
from tokenspeed_kernel.ops.moe import moe_apply, moe_plan, moe_process_weights
from tokenspeed_kernel.ops.quantization import (
    quantize_fp8,
    quantize_fp8_with_scale,
    quantize_mxfp4,
    quantize_mxfp8,
    quantize_nvfp4,
)
from tokenspeed_kernel.ops.sampling import argmax
from tokenspeed_kernel.selection import NoKernelFoundError

__all__ = [
    # exceptions
    "NoKernelFoundError",
    # gemm
    "mm",
    # attention
    "attn_plan",
    "mha_prefill",
    "mha_extend_with_kvcache",
    "mha_decode_with_kvcache",
    "mla_prefill",
    "mla_decode_with_kvcache",
    "attn_merge_state",
    # moe
    "moe_apply",
    "moe_plan",
    "moe_process_weights",
    # quantization
    "quantize_fp8",
    "quantize_fp8_with_scale",
    "quantize_mxfp8",
    "quantize_nvfp4",
    "quantize_mxfp4",
    # sampling
    "argmax",
]
