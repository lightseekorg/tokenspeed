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


"""Addresses of the routed experts' MXFP4 weights and scales (N16 layout).

The N16 layout is AITER's shuffled weight and scale arrangement, with the
gate and up rows interleaved. A new layout belongs here, not in the
projection stages.

Modified from sgl-amd-mono (https://github.com/sgl-project/sgl-amd-mono, Apache
License 2.0) at b9f40eb562f449ce08c030df200f97c7ea716511: ``common/mx.py`` and
``kimi_k3/gfx950/stages/moe.py``, whose device primitives are adapted from
ROCm/ATOM (https://github.com/ROCm/ATOM, MIT License) at
97359d6df46c59736468168b7be8bc50b482a1a9.
"""

from tokenspeed_kernel_amd._triton import gl, gluon


@gluon.jit
def expert_addresses(
    expert,
    row_group,
    step,
    lane,
    word,
    N: gl.constexpr,
    K: gl.constexpr,
):
    """Return the payload word index, and the scale word index and bit shift."""
    SCALE_COLS: gl.constexpr = gl.cdiv(K // 32, 8) * 8
    payload = expert * (N * K // 8) + row_group * (2 * K) + step * 256 + lane * 4 + word
    # The same shuffle_scale formula includes the padded 16 scale columns of
    # down (K=384) and the 112 scale columns of gate/up (K=3584).
    row = expert * N + row_group * 16 + lane % 16
    col = step * 4 + lane // 16
    byte = (
        ((((row // 32) * (SCALE_COLS // 8) + col // 8) * 4 + col % 4) * 16 + row % 16)
        * 2
        + (col // 4) % 2
    ) * 2 + (row // 16) % 2
    return payload, byte // 4, (byte % 4) * 8


@gluon.jit
def gate_up_row_group(group, wave):
    rgl = wave % 4
    row_pair, up = 2 * group + rgl % 2, rgl // 2
    return 2 * row_pair + up
