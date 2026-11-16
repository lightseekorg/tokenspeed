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

from __future__ import annotations

"""Shared Triton MXFP4 index-row codec, independent of attention registration."""

from tokenspeed_kernel._triton import tl, triton


@triton.jit
def _e2m1_encode(x):
    a = tl.minimum(tl.abs(x), 6.0)
    # Nearest, ties to even *code*, including signed zero. V4's codec is not RNE.
    code = (
        (a > 0.25).to(tl.int32)
        + (a >= 0.75).to(tl.int32)
        + (a > 1.25).to(tl.int32)
        + (a >= 1.75).to(tl.int32)
        + (a > 2.5).to(tl.int32)
        + (a >= 3.5).to(tl.int32)
        + (a > 5.0).to(tl.int32)
    )
    return (code | ((x.to(tl.int32, bitcast=True) >> 28) & 8)).to(tl.uint8)


@triton.jit
def _e2m1_decode(code):
    a = code & 7
    value = tl.where(
        a < 4,
        a.to(tl.float32) * 0.5,
        tl.where(a == 4, 2.0, tl.where(a == 5, 3.0, tl.where(a == 6, 4.0, 6.0))),
    )
    return tl.where((code & 8) != 0, -value, value)


@triton.jit
def _ue8m0_exponent(scaled_max):
    # Biased exponent of 2**ceil(log2(x)); matches the reference IEEE-754
    # ceil(log2) exactly, unlike an approximate log2.
    bits = scaled_max.to(tl.int32, bitcast=True)
    return ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(tl.int32)


@triton.jit
def _mxfp4_index_rows(groups, scaled_max, ROWS: tl.constexpr):
    """Encode FP32 [ROWS, 4, 32] index groups as MXFP4 rows.

    scaled_max is each group's floored amax / 6; callers keep the division
    their reference uses. Returns nibble-packed E2M1 codes uint8 [ROWS, 64]
    (even element in the low nibble) and the biased UE8M0 exponents
    int32 [ROWS, 4].
    """
    exponent = _ue8m0_exponent(scaled_max)
    scale = (exponent << 23).to(tl.float32, bitcast=True)
    codes = _e2m1_encode(tl.div_rn(groups, scale[:, :, None])).to(tl.int32)
    pairs = tl.reshape(codes, (ROWS, 64, 2))
    packed = tl.sum(pairs << (tl.arange(0, 2)[None, None, :] * 4), 2).to(tl.uint8)
    return packed, exponent


@triton.jit
def _ue8m0_word(exponent):
    # Four [ROWS, 4] exponent bytes as one little-endian int32 per row.
    word = tl.sum(exponent.to(tl.uint32) << (tl.arange(0, 4)[None, :] * 8), 1)
    return word.to(tl.int32)
