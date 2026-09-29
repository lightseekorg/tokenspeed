# /***************************************************************************************************
#  * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights
#  *reserved. SPDX-License-Identifier: BSD-3-Clause
#  *
#  * Redistribution and use in source and binary forms, with or without
#  * modification, are permitted provided that the following conditions are met:
#  *
#  * 1. Redistributions of source code must retain the above copyright notice,
#  *this list of conditions and the following disclaimer.
#  *
#  * 2. Redistributions in binary form must reproduce the above copyright notice,
#  * this list of conditions and the following disclaimer in the documentation
#  * and/or other materials provided with the distribution.
#  *
#  * 3. Neither the name of the copyright holder nor the names of its
#  * contributors may be used to endorse or promote products derived from
#  * this software without specific prior written permission.
#  *
#  * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  *POSSIBILITY OF SUCH DAMAGE.
#  *
#  **************************************************************************************************/

"""Petit tal/tensor/stride.h coordinate-to-index mapping."""

import triton.experimental.gluon as g
from triton.experimental.gluon import language as l


@g.constexpr_function
def _is_tuple(value):
    return isinstance(value, (tuple, l.tuple))


@g.constexpr_function
def _product(shape):
    if isinstance(shape, (tuple, l.tuple)):
        result = 1
        for mode in shape:
            result *= _product(mode)
        return result
    return shape


@g.jit
def _crd2idx_itt(coord, shape: l.constexpr, stride: l.constexpr, mode: l.constexpr):
    if mode + 1 == len(shape):
        # Native leaves the last coordinate unbounded (no final modulo).
        index = crd2idx(coord, shape[mode], stride[mode])
    elif isinstance(coord, l.constexpr) and coord == 0:
        index = crd2idx(0, shape[mode], stride[mode])
        for i in l.static_range(mode + 1, len(shape)):
            index += crd2idx(0, shape[i], stride[i])
    else:
        size: l.constexpr = _product(shape[mode])
        index = crd2idx(coord % size, shape[mode], stride[mode]) + _crd2idx_itt(
            coord // size, shape, stride, mode + 1
        )
    return index


@g.jit
def crd2idx(coord, shape: l.constexpr, stride: l.constexpr):
    if isinstance(coord, l.tuple):
        l.static_assert(_is_tuple(shape), "Tuple coordinate requires tuple shape")
        l.static_assert(_is_tuple(stride), "Tuple coordinate requires tuple stride")
        l.static_assert(len(coord) == len(shape), "Mismatched Ranks")
        l.static_assert(len(coord) == len(stride), "Mismatched Ranks")
        index = crd2idx(coord[0], shape[0], stride[0])
        for i in l.static_range(1, len(coord)):
            index += crd2idx(coord[i], shape[i], stride[i])
    elif _is_tuple(shape):
        l.static_assert(_is_tuple(stride), "Tuple shape requires tuple stride")
        l.static_assert(len(shape) == len(stride), "Mismatched Ranks")
        index = _crd2idx_itt(coord, shape, stride, 0)
    else:
        l.static_assert(not _is_tuple(stride), "Scalar shape requires scalar stride")
        index = coord * stride
    return index
