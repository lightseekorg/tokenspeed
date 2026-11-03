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

"""Programs and the XCDs of a gfx950 GPU."""

from tokenspeed_kernel_amd._triton import gl, gluon


@gluon.jit
def _remap_xcd(pid, num_programs, NUM_XCDS: gl.constexpr):
    # Hardware dispatches program i to XCD i % NUM_XCDS; give each XCD a
    # contiguous range of program ids instead.
    pids_per_xcd = (num_programs + NUM_XCDS - 1) // NUM_XCDS
    tall_xcds = num_programs % NUM_XCDS
    tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
    xcd = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS
    if xcd < tall_xcds:
        pid = xcd * pids_per_xcd + local_pid
    else:
        pid = (
            tall_xcds * pids_per_xcd
            + (xcd - tall_xcds) * (pids_per_xcd - 1)
            + local_pid
        )
    return pid


@gluon.jit
def xcc_id():
    """Return the XCD (accelerator complex die) running this workgroup, 0-7.

    The hardware does not dispatch workgroup i to XCD i % 8 in general: the
    round-robin start depends on the kernels before. Callers that need the
    XCD read it here instead of deriving it from the program id.
    """
    return gl.inline_asm(
        "s_getreg_b32 $0, hwreg(HW_REG_XCC_ID, 0, 4)",
        "=s,~{memory}",
        [],
        gl.int32,
        is_pure=False,
    )
