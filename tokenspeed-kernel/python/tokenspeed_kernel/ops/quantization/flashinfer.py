# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""FlashInfer quantization kernels."""

import torch
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.registry import error_fn

fp4_quantize = error_fn
mxfp8_quantize = error_fn
nvfp4_block_scale_interleave = error_fn
get_fp8_blockscale_gemm_runner_sm90 = error_fn

if current_platform().is_nvidia:
    try:
        from flashinfer import (
            fp4_quantize,
            mxfp8_quantize,
            nvfp4_block_scale_interleave,
        )
    except ImportError:
        pass
    try:
        from flashinfer.gemm.gemm_base import get_fp8_blockscale_gemm_runner_sm90
    except ImportError:
        pass


def fp8_quantize_1x128_sm90(
    A: torch.Tensor,
    A_q: torch.Tensor,
    A_scales: torch.Tensor,
) -> None:
    """Quantize BF16 activations into FlashInfer/DeepGEMM's SM90 1x128 layout."""
    if get_fp8_blockscale_gemm_runner_sm90 is error_fn:
        raise RuntimeError("FlashInfer SM90 FP8 quantization runner is not available")
    runner = get_fp8_blockscale_gemm_runner_sm90()
    runner.fp8_quantize_1x128(A, A_q, A_scales, False)


__all__ = [
    "fp4_quantize",
    "mxfp8_quantize",
    "nvfp4_block_scale_interleave",
    "fp8_quantize_1x128_sm90",
]
