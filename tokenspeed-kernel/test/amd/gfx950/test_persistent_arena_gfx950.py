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

"""Uncached device buffers for persistent kernels' cross-XCD hand-offs."""

import pytest
import torch
from tokenspeed_kernel._triton import gl, gluon
from utils import is_cdna4

if not is_cdna4():
    pytest.skip("Persistent helpers target gfx950", allow_module_level=True)

from tokenspeed_kernel_amd.ops.gfx950.persistent.arena import (  # noqa: E402
    UncachedBuffer,
)


@gluon.jit
def _store_program_ids(out_ptr, WORDS: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
    offsets = gl.arange(0, WORDS, layout)
    gl.store(out_ptr + gl.program_id(0) * WORDS + offsets, offsets + gl.program_id(0))


def test_buffer_starts_zeroed_and_holds_kernel_stores():
    words, programs = 256, 64
    buffer = UncachedBuffer(words * programs, torch.int32, torch.device("cuda"))
    try:
        assert torch.equal(
            buffer.read(), torch.zeros(words * programs, dtype=torch.int32)
        )
        _store_program_ids[(programs,)](buffer, words, num_warps=4)
        torch.cuda.synchronize()
        expected = (
            torch.arange(words, dtype=torch.int32)[None, :]
            + torch.arange(programs, dtype=torch.int32)[:, None]
        ).flatten()
        assert torch.equal(buffer.read(), expected)
    finally:
        buffer.free()
