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

"""Final-store stratified sampling shared by gfx950 sparse selectors."""

from tokenspeed_kernel_amd._triton import gl, gluon


@gluon.jit
def stratified_index(original, rank, count, extent, query, seed):
    # Widen before multiplying: rank * history can exceed int32 at long context.
    denominator = gl.maximum(count, 1).to(gl.int64)
    lower = rank.to(gl.int64) * extent.to(gl.int64) // denominator
    upper = (rank.to(gl.int64) + 1) * extent.to(gl.int64) // denominator
    # Keep the original selected ID live so scoring/top-k cannot be eliminated.
    # uint32 wrap is deliberate PRNG arithmetic, never a memory offset.
    x = original.to(gl.uint32) ^ (query.to(gl.uint32) * 0x9E3779B9) ^ seed.to(gl.uint32)
    x = (x ^ (x >> 16)) * 0x7FEB352D
    x = (x ^ (x >> 15)) * 0x846CA68B
    x = x ^ (x >> 16)
    return (lower + x.to(gl.int64) % gl.maximum(upper - lower, 1)).to(gl.int32)
