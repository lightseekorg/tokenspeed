# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2026 SonicSampler Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from tokenspeed_kernel._triton import tl, triton  # noqa.

cdiv, jit = triton.cdiv, triton.jit


@jit
def mask_pair(shape):

    sign = tl.full(shape, 0x8000, dtype=tl.uint16)
    full = tl.full(shape, 0xFFFF, dtype=tl.uint16)

    return sign, full


@jit
def encode_block(values):
    """
    Project the representable real numbers into the finite set of natural numbers, ℕ_0, such that:

        • Positive numbers are merely offset by 2^15 given their ascending ordering, while

        • Negative numbers, given the inverse ordering of their absolute values, are inverted
          via (2^16 - x - 1).

    Note that `values` here is inherently bit-casted to an unsigned 16-bit integer first and the above
    is achieved by encoding the value under the following scheme:

        • Positive --( Mask: 0x8000 )-> [ Sign Bit: 1, Lower 15 Bits As-Is ].

        • Negative --( Mask: 0xFFFF )-> [ Inverted 16 Bits ].

    With mask(s) applied by the XOR operation and the resulting block being upcasted to 32-bit
    unsigned integer(s).

    """
    shape: tl.constexpr = values.shape

    sign, full = mask_pair(shape)

    encoded = values.to(tl.uint16, bitcast=True)

    mask = tl.where((encoded & sign) != 0, full, sign)

    return (encoded ^ mask).to(tl.uint32)


@jit
def shift_pack(values, indices, block_n):
    """
    Shifts the lower 16 bits of `values` to the upper half and inverts the `indices` with
    respect to `block_n` to the lower half.

    """
    return (values << 16) | ((block_n - 1) ^ indices)


@jit
def bitpack_block(values, indices, block_n):
    """
    Packs each ℕ_0-projected unsigned 16-bit integer `value` into the upper half of an unsigned
    32-bit integer with its corresponding inverted `index` into the lower half.

    Note(s):

        • Here, the indices are inverted via (BLOCK_N - k - 1) to ensure stable sorting /
          selection i.e. earlier positioned tied-values are ordered first.

    Caveat(s):

        • All indices must no value greater than (2^16 - 1) i.e. 65,535.

    """
    encoded = encode_block(values)

    return shift_pack(encoded, indices, block_n)


@jit
def bitpack_scalar(value, index, block_n):

    raw = value.to(tl.uint16, bitcast=True)

    if (raw & 0x8000) != 0:

        raw ^= 0xFFFF

    else:

        raw ^= 0x8000

    encoded = raw.to(tl.uint32)

    return (encoded << 16) | ((block_n - 1) ^ index)


@jit
def decode_block(values):

    shape: tl.constexpr = values.shape

    sign, full = mask_pair(shape)

    decoded = values.to(tl.uint16)

    mask = tl.where((decoded & sign) == 0, full, sign)

    return (decoded ^ mask).to(tl.bfloat16, bitcast=True)
