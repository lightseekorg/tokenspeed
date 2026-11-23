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

"""The per-row FP8 E4M3 code of the dense FP8 GEMMs' activations and weights.

One definition, shared by the token quantizer, the weight conversion and the
layer boundary's FP8 copy, so all of them write the same bits: a row's scale
is ``max(amax / 448, 1e-12)`` by IEEE FP32 division, and its codes are
``x / scale`` (IEEE division) rounded once to nearest even and saturated at
+-448.
"""

from tokenspeed_kernel._triton import tl, triton

__all__ = ["fp8_row_codes"]

FP8_E4M3_MAX = tl.constexpr(448.0)
# Floors the scale of an all-zero row, whose codes are then all zero.
FP8_ROW_SCALE_MIN = tl.constexpr(1e-12)


@triton.jit
def fp8_row_codes(values, mask):
    """FP8 E4M3 codes and the FP32 dequant scale of one FP32 row.

    Args:
        values: FP32 ``[BLOCK]`` row; lanes outside ``mask`` are ignored.
        mask: ``[BLOCK]`` valid lanes.

    Returns:
        ``(codes, scale)``: FP8 E4M3 ``[BLOCK]`` codes and the FP32 scale.
    """
    amax = tl.max(tl.where(mask, tl.abs(values), 0.0), axis=0)
    # IEEE division (not a reciprocal multiply): the reference definition.
    scale = tl.maximum(tl.div_rn(amax, FP8_E4M3_MAX), FP8_ROW_SCALE_MIN)
    codes = tl.clamp(tl.div_rn(values, scale), -FP8_E4M3_MAX, FP8_E4M3_MAX)
    return codes.to(tl.float8e4nv), scale
