# MIT License
#
# Copyright (c) 2026 LightSeek Foundation <contact@lightseek.org>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Native contracts for shared AMD instruction-scheduling hints."""

import pytest
import torch
from utils import is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)

from tokenspeed_kernel_amd._scheduling import (  # noqa: E402
    sched_compile_options,
    sched_group,
)
from tokenspeed_kernel_amd._triton import gl, gluon  # noqa: E402


@gluon.jit
def _sched_group_probe(x, out):
    offset = gl.arange(0, 256, layout=gl.BlockedLayout([1], [64], [4], [0]))
    y = gl.exp2(gl.load(x + offset))
    sched_group(("trans",), 2)
    sched_group("mfma", 1)
    gl.store(out + offset, y)


def test_sched_group_accepts_class_names():
    x = torch.randn(256, device="cuda")
    out = torch.empty_like(x)
    compiled = _sched_group_probe[(1,)](x, out, num_warps=4, **sched_compile_options())
    llir = compiled.asm["llir"]
    assert "@llvm.amdgcn.sched.group.barrier(i32 1024, i32 2, i32 0)" in llir
    assert "@llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)" in llir
    torch.testing.assert_close(out, torch.exp2(x), atol=0, rtol=0)
