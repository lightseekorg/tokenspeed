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

# Single point of indirection for the Triton package used by
# tokenspeed-kernel-amd. Prefer the vendor release in production, while allowing
# compiler development directly against a public triton-lang source checkout.

import importlib

try:
    triton = importlib.import_module("tokenspeed_triton")
    _TRITON_PACKAGE = "tokenspeed_triton"
except ModuleNotFoundError as exc:
    if exc.name != "tokenspeed_triton":
        raise
    triton = importlib.import_module("triton")
    _TRITON_PACKAGE = "triton"

gl = importlib.import_module(f"{_TRITON_PACKAGE}.experimental.gluon.language")
tl = importlib.import_module(f"{_TRITON_PACKAGE}.language")
gluon = importlib.import_module(f"{_TRITON_PACKAGE}.experimental.gluon")
gluon_builtin = importlib.import_module(
    f"{_TRITON_PACKAGE}.experimental.gluon.language._core"
).builtin
cdna4_async_copy = importlib.import_module(
    f"{_TRITON_PACKAGE}.experimental.gluon.language.amd.cdna4.async_copy"
)
aggregate = importlib.import_module(f"{_TRITON_PACKAGE}.language.core")._aggregate

__all__ = [
    "aggregate",
    "cdna4_async_copy",
    "gl",
    "gluon",
    "gluon_builtin",
    "tl",
    "triton",
]
