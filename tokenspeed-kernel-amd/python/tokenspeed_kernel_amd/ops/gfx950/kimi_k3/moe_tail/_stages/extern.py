# SPDX-License-Identifier: MIT AND Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 LightSeek Foundation
# SPDX-FileCopyrightText: Copyright 2026 SGLang contributors
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (C) 2026, Advanced Micro Devices, Inc.
#
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


"""Device operations of the MoE tail that Gluon does not expose.

They come from one LLVM IR library, ``extern.ll`` next to this file, with no
inline machine instructions. The tail launches with :func:`compile_options`,
which adds that library to the shared intrinsic libraries of
:mod:`tokenspeed_kernel_amd._extern`.

Modified from sgl-amd-mono (https://github.com/sgl-project/sgl-amd-mono, Apache
License 2.0) at b9f40eb562f449ce08c030df200f97c7ea716511: ``common/ops.py``,
whose device primitives are adapted from ROCm/ATOM
(https://github.com/ROCm/ATOM, MIT License) at
97359d6df46c59736468168b7be8bc50b482a1a9.
"""

from pathlib import Path

from tokenspeed_kernel_amd._extern import extern_compile_options
from tokenspeed_kernel_amd._triton import gl, gluon, tl

LIBRARY = "tokenspeed_moe_tail"
PATH = str(Path(__file__).with_name("extern.ll"))


def compile_options() -> dict:
    """Return the tail's launch options: every external LLVM IR library."""
    options = extern_compile_options()
    options["extern_libs"] = {**options["extern_libs"], LIBRARY: PATH}
    return options


def _call(name, args, types, result, pure, semantic):
    return tl.core.extern_elementwise(
        LIBRARY,
        PATH,
        args,
        {types: ("__tokenspeed_moe_tail_" + name, result)},
        is_pure=pure,
        _semantic=semantic,
    )


@tl.core.extern
def readlane(value, lane, _semantic):
    return _call(
        "readlane", [value, lane], (tl.int32, tl.int32), tl.int32, True, _semantic
    )


@tl.core.extern
def writelane(value, lane, vector, _semantic):
    return _call(
        "writelane",
        [value, lane, vector],
        (tl.int32, tl.int32, tl.int32),
        tl.int32,
        True,
        _semantic,
    )


@tl.core.extern
def trailing_zeros(value, _semantic):
    return _call("cttz", [value], (tl.int64,), tl.int32, True, _semantic)


@tl.core.extern
def _uniform_bits(bits, _semantic):
    return _call("peer_uniform", [bits], (tl.uint64,), tl.uint64, True, _semantic)


@gluon.jit
def load_peer_base(address):
    # The peer table is immutable and every lane of a wave loads the same
    # entry, so the address is moved to scalar registers.
    bits = _uniform_bits(gl.load(address).to(gl.uint64))
    return bits.to(gl.pointer_type(gl.uint8))
