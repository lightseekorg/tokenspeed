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


"""INT8 block-16 payloads of the prefill collectives.

A producer encodes its BF16 rows; the pull collectives read the codes and
scales instead of BF16 and decode each element before using it, so each
element reaches the collective's arithmetic as decode(encode(x)):

  encode: per 16 consecutive elements, scale = max|x| / 127 in FP32 (an
          all-zero block keeps 1e-30 / 127); code = x / scale rounded half
          to even, clamped to [-127, 127].
  decode: code * scale in FP32, rounded once to BF16.

Every step is one correctly rounded FP32 operation, so the bits match the
reference encoder / decoder exactly. The payload is 1.25 bytes an element
(one code, plus one FP32 scale per 16) instead of 2.

Inputs must be finite: a block holding Inf or NaN has an Inf / NaN scale,
which the codes and the decoded block do not carry meaningfully. The owned-row
producers this serves (projections, routed sums) are finite.
"""

from tokenspeed_kernel._triton import gl, gluon, libdevice, tl, triton

# Elements that share one scale (the kernels below spell it as 16).
PAYLOAD_BLOCK = 16


# The element count follows the batch: unspecialized, so its alignment class
# does not select a new binary.
@triton.jit(do_not_specialize=["elements"])
def encode_int8_block16(x_ptr, codes_ptr, scales_ptr, elements, BLOCK: tl.constexpr):
    """INT8 codes and FP32 block-16 scales of the first ``elements`` BF16
    values of ``x_ptr`` (a multiple of 16), into ``codes_ptr`` and
    ``scales_ptr`` [elements / 16]. One program encodes BLOCK elements."""
    tl.static_assert(BLOCK % 16 == 0)
    # The program's base grows with the payload: int64, so payloads past
    # 2^31 elements index correctly; offsets inside a program stay int32.
    base = tl.program_id(0).to(tl.int64) * BLOCK
    offsets = tl.arange(0, BLOCK)
    mask = offsets < elements - base
    x = tl.load(x_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)
    blocks = tl.reshape(x, (BLOCK // 16, 16))
    amax = tl.max(tl.abs(blocks), axis=1)
    scale = tl.math.div_rn(tl.maximum(amax, 1e-30), 127.0)
    codes = libdevice.rint(tl.math.div_rn(blocks, scale[:, None]))
    codes = tl.minimum(tl.maximum(codes, -127.0), 127.0)
    codes = tl.reshape(codes, (BLOCK,)).to(tl.int8)
    tl.store(codes_ptr + base + offsets, codes, mask=mask)
    groups = tl.arange(0, BLOCK // 16)
    tl.store(
        scales_ptr + base // 16 + groups, scale, mask=groups < (elements - base) // 16
    )


@gluon.jit
def load_int8_block16(codes_ptr, scales_ptr, offsets, mask):
    """BF16 decode of the int8-block16 payload elements at ``offsets``
    (masked lanes read 0): the pulls' replacement for a BF16 load."""
    codes = gl.amd.cdna4.buffer_load(codes_ptr, offsets, mask, 0, cache=".cg")
    scales = gl.amd.cdna4.buffer_load(scales_ptr, offsets // 16, mask, 0.0, cache=".cg")
    return (codes.to(gl.float32) * scales).to(gl.bfloat16)
