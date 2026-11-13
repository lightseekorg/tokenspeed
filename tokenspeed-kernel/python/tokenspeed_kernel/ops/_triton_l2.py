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

"""L2 prefetch hints for Triton kernels.

A kernel waiting on its PDL predecessor can ask L2 for the older rows it
reads after the wait while HBM is otherwise idle. A hint reads nothing into
registers and never changes a result; a hint of a line another kernel is
still writing only warms L2.
"""

from tokenspeed_kernel._triton import tl, triton


@triton.jit
def prefetch_l2(address, mask):
    """Ask L2 for the 128-byte lines at ``address`` where ``mask`` is set
    (NVIDIA; elsewhere nothing)."""
    if tl.target_info.is_cuda():
        tl.inline_asm_elementwise(
            "{ .reg .pred p; setp.ne.s32 p, $2, 0; @p prefetch.global.L2 [$1]; mov.u32 $0, 0; }",
            "=r,l,r",
            [address, mask.to(tl.int32)],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )
