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

"""The code warm finds its kernel's code: code_pointer and text_range."""

from __future__ import annotations

import pytest
import torch
from utils import is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)

from tokenspeed_kernel_amd._triton import gl, gluon  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.monokernel import code_warm  # noqa: E402

BLOCK = 1024


@gluon.jit(do_not_specialize=("offset", "words"))
def _copy_text(out, offset, words, BLOCK: gl.constexpr):
    """Copy ``words`` int32 words of this kernel's code to ``out``."""
    code = code_warm.code_pointer(offset)
    i = gl.arange(0, BLOCK, gl.BlockedLayout([4], [64], [gl.num_warps()], [0]))
    for k in range(0, words, BLOCK):
        o = k + i
        gl.store(out + o, gl.load(code + o, mask=o < words), mask=o < words)


def _text_words(kernel) -> torch.Tensor:
    """The kernel's .text from its code object, as int32 words."""
    import struct

    elf = kernel.asm["hsaco"]
    (shoff,) = struct.unpack_from("<Q", elf, 0x28)
    entsize, count, names_index = struct.unpack_from("<HHH", elf, 0x3A)
    sections = [
        struct.unpack_from("<IIQQQQIIQQ", elf, shoff + k * entsize)
        for k in range(count)
    ]
    names = sections[names_index][4]
    for s in sections:
        if elf[names + s[0] : elf.index(b"\0", names + s[0])] == b".text":
            body = bytearray(elf[s[4] : s[4] + s[5]])
            return torch.frombuffer(body, dtype=torch.int32).clone()
    raise AssertionError("no .text")


def test_code_pointer_reads_the_kernels_own_text():
    out = torch.zeros(1 << 16, dtype=torch.int32, device="cuda")
    kernel = _copy_text.warmup(out, 0, 0, BLOCK, num_warps=4, grid=(1,))
    offset, quads = code_warm.text_range(kernel)
    text = _text_words(kernel)
    assert quads * 4 == text.numel()
    # The pc that s_getpc_b64 returns is the marker's s_mov_b32.
    marker = (text == code_warm.MARKER).nonzero().flatten().tolist()
    assert len(marker) == 1 and offset == 4 * marker[0] - 4
    _copy_text[(1,)](out, offset, quads * 4, BLOCK, num_warps=4)
    torch.cuda.synchronize()
    assert torch.equal(out[: text.numel()].cpu(), text)
    # Cached per kernel.
    assert code_warm.text_range(kernel) == (offset, quads)
