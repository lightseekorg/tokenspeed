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

try:
    from tokenspeed_kernel_nvidia.thirdparty.trtllm import *  # noqa: F401,F403
except ImportError:
    from tokenspeed_kernel.registry import error_fn

    dsv3_fused_a_gemm = error_fn
    fast_topk_v2 = error_fn
    fp8_blockwise_scaled_mm = error_fn
    moe_align_block_size = error_fn
    per_tensor_quant_fp8 = error_fn
    per_token_group_quant_8bit = error_fn
    per_token_quant_fp8 = error_fn

    __all__ = [
        "dsv3_fused_a_gemm",
        "fast_topk_v2",
        "fp8_blockwise_scaled_mm",
        "moe_align_block_size",
        "per_tensor_quant_fp8",
        "per_token_group_quant_8bit",
        "per_token_quant_fp8",
    ]
