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

"""Activation-tile conversion matching the shared 128-value quantizer."""

from tokenspeed_kernel_amd._triton import gl, gluon


@gluon.jit
def quantize_intermediate_tile(value, USE_RECIPROCAL: gl.constexpr):
    # The separate path stores BF16 before its FP32 scale reduction.
    rounded = value.to(gl.bfloat16).to(gl.float32)
    scale = gl.maximum(gl.max(gl.abs(rounded), 1), 1.0e-10) / 448.0
    # Match the shared quantizer's size-selected arithmetic, including midpoints.
    if USE_RECIPROCAL:
        inverse_scale = 1.0 / scale
        scaled = rounded * inverse_scale[:, None]
    else:
        scaled = rounded / scale[:, None]
    quantized = gl.clamp(scaled, -448.0, 448.0).to(gl.float8e4nv)
    return quantized, scale
