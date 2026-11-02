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

"""Public entry points for TRT-LLM blockwise FP8 GEMM."""


def blockwise_gemm(a, b, a_scales, b_scales, out):
    """Run block-FP8 GEMM with startup-tuned tiles and the current CUDA stream.

    Args:
        a: E4M3 activations [M,K].
        b: E4M3 weights [N,K].
        a_scales: FP32 activation scales [M,K/128].
        b_scales: FP32 weight scales [N/128,K/128].
        out: Contiguous BF16 output [M,N].

    Returns:
        The supplied output tensor. Tuning follows the shared autotune context.
    """
    if a.shape[0] == 0:
        return out
    from tokenspeed_kernel.thirdparty.trtllm_blockwise.runner import run

    return run(a, b, a_scales, b_scales, out)
