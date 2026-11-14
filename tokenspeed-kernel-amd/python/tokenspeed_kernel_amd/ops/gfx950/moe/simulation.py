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

"""Bijections of final expert IDs that preserve scoring and selection work."""

from tokenspeed_kernel_amd._triton import gl, gluon


@gluon.jit
def permute_experts(ids, row, seed, experts: gl.constexpr):
    # Keep ids in the expression: selection and score gathers must still execute.
    # uint32 wrapping here is intentional PRNG arithmetic, never a memory offset.
    gl.static_assert(
        experts == 256
        or experts == 384
        or experts == 512
        or experts == 896
        or experts == 1024
    )
    x = row.to(gl.uint32) ^ seed.to(gl.uint32)
    x = (x ^ (x >> 16)) * 0x7FEB352D
    x = (x ^ (x >> 15)) * 0x846CA68B
    x = x ^ (x >> 16)
    multiplier = gl.where((x & 1) == 0, 17, 37)
    multiplier = gl.where((x & 2) != 0, 53, multiplier)
    return ((ids.to(gl.uint32) * multiplier + x % experts) % experts).to(gl.int32)
