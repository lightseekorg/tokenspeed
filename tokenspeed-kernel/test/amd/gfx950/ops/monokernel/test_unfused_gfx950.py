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

"""``unfused.log1p`` in a launch with FP fusion on against libdevice's
``log1p`` in a launch with it off, over FP32 bit patterns of every exponent
and sign, specials included."""

from __future__ import annotations

import pytest
import torch
from utils import is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)

from tokenspeed_kernel._triton import libdevice, tl, triton  # noqa: E402
from tokenspeed_kernel_amd._extern import extern_compile_options  # noqa: E402
from tokenspeed_kernel_amd._triton import gl, gluon  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.monokernel import unfused  # noqa: E402

BLOCK = 4096


@triton.jit
def _libdevice_log1p(x_ptr, y_ptr, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + i).to(tl.float32, bitcast=True)
    tl.store(y_ptr + i, libdevice.log1p(x).to(tl.int32, bitcast=True))


@gluon.jit
def _unfused_log1p(x_ptr, y_ptr, BLOCK: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([4], [64], [gl.num_warps()], [0])
    i = gl.program_id(0).to(gl.int64) * BLOCK + gl.arange(0, BLOCK, layout)
    x = gl.load(x_ptr + i).to(gl.float32, bitcast=True)
    gl.store(y_ptr + i, unfused.log1p(x).to(gl.int32, bitcast=True))


def test_log1p_equals_libdevice_without_fusion():
    # Every 509th bit pattern (a prime stride: all mantissa residues) and the
    # boundaries: zeros, -1 and its neighbours, 2^-24, infinities, NaN.
    sweep = torch.arange(-(2**31), 2**31, 509, dtype=torch.int64)
    special = torch.tensor(
        [0.0, -0.0, -1.0, -0.99999994, -1.0000001, 2.0**-24, -(2.0**-24)]
        + [2.0**-25, 1.0, float("inf"), float("-inf"), float("nan")]
    ).view(torch.int32)
    x = torch.cat([sweep.to(torch.int32), special])
    x = torch.cat([x, x.new_zeros(-x.numel() % BLOCK)]).cuda()
    want = torch.empty_like(x)
    got = torch.empty_like(x)
    grid = (x.numel() // BLOCK,)
    _libdevice_log1p[grid](x, want, BLOCK, num_warps=4, enable_fp_fusion=False)
    _unfused_log1p[grid](x, got, BLOCK, num_warps=4, **extern_compile_options())
    assert torch.equal(got, want)
