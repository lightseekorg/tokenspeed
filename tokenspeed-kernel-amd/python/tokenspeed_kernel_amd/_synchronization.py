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

"""AMD device synchronization primitives shared by compute and communication."""

from tokenspeed_kernel_amd._triton import gl, gluon


@gluon.jit
def wait_vmem(args, constraints: gl.constexpr):
    """Wait for this subgroup's outstanding vector memory operations to complete.

    ``args`` and ``constraints`` specify the assembly's compiler dependencies.
    Callers handle workgroup synchronization and cache writeback/invalidation.
    """
    gl.inline_asm_elementwise(
        asm="s_waitcnt vmcnt(0)",
        constraints=constraints,
        args=args,
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )
