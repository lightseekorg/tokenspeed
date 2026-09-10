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

# Single point of indirection for the Triton vendor release package used by
# tokenspeed-kernel-amd. All implementation code imports Triton symbols from
# here so the underlying distribution can be swapped in one place.

import tokenspeed_triton as triton
import tokenspeed_triton.experimental.gluon.language as gl
from tokenspeed_triton import language as tl
from tokenspeed_triton.experimental import gluon
from tokenspeed_triton.experimental.gluon.language.amd.cdna4 import (
    async_copy as cdna4_async_copy,
)
from tokenspeed_triton.language.core import _aggregate as aggregate

__all__ = [
    "aggregate",
    "cdna4_async_copy",
    "gl",
    "gluon",
    "tl",
    "triton",
]
