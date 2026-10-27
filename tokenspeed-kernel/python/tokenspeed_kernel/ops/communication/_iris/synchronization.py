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

"""Vector-memory waits that Iris selects per architecture through its policy.

The policy carries each wait as its instruction string, not as a Gluon
function: Triton's compile hooks serialize constexpr arguments to JSON, and a
Gluon function does not serialize. Callers handle workgroup synchronization,
cache writeback/invalidation, and system-scope ordering.
"""

from tokenspeed_kernel._triton import gl, gluon


@gluon.jit
def iris_wait(asm: gl.constexpr, args, constraints: gl.constexpr):
    """Issue the wait instruction ``asm`` for this subgroup.

    ``args`` and ``constraints`` specify the assembly's compiler dependencies.
    """
    gl.inline_asm_elementwise(
        asm=asm,
        constraints=constraints,
        args=args,
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )
