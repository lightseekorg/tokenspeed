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


"""Native contracts for compiler-inserted LDS barriers that kernels rely on."""

import re

import pytest
import torch
from utils import is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)

from tokenspeed_kernel_amd._triton import cdna4_async_copy, gl, gluon  # noqa: E402


@gluon.jit
def _relaxed_refill_probe(src, out):
    layout: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [4, 1], [1, 0])
    shared: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    rows = gl.arange(0, 32, layout=gl.SliceLayout(1, layout))
    cols = gl.arange(0, 64, layout=gl.SliceLayout(0, layout))
    offsets = rows[:, None] * 64 + cols[None, :]
    smem = gl.allocate_shared_memory(gl.float16, [32, 64], shared)
    cdna4_async_copy.buffer_load_to_shared(smem, src, offsets)
    cdna4_async_copy.commit_group()
    cdna4_async_copy.wait_group(0)
    first = cdna4_async_copy.load_shared_relaxed(smem, layout)
    # Refill the slot every wave just read: a write-after-read across waves.
    cdna4_async_copy.buffer_load_to_shared(smem, src, offsets + 32 * 64)
    cdna4_async_copy.commit_group()
    cdna4_async_copy.wait_group(0)
    second = cdna4_async_copy.load_shared_relaxed(smem, layout)
    gl.store(out + offsets, first + second)


def test_relaxed_load_then_async_refill_gets_barrier():
    # load_shared_relaxed pipelines rely on the compiler, not a manual
    # gl.barrier(), to order the read before the next copy into its slot.
    src = torch.randn(2 * 32 * 64, device="cuda", dtype=torch.float16)
    out = torch.empty(32 * 64, device="cuda", dtype=torch.float16)
    compiled = _relaxed_refill_probe[(1,)](src, out, num_warps=4)
    torch.testing.assert_close(out, src[:2048] + src[2048:], atol=0, rtol=0)
    asm = compiled.asm["amdgcn"]
    first_read = re.search(r"^\s*ds_read\w*", asm, re.M)
    refill = re.compile(r"^\s*buffer_load\w* .* lds$", re.M).search(
        asm, first_read.end()
    )
    assert re.search(r"^\s*s_barrier$", asm[first_read.end() : refill.start()], re.M)
