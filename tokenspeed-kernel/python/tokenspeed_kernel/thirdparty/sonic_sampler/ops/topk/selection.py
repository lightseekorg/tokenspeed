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

from tokenspeed_kernel.thirdparty.sonic_sampler.ops.base import gdc_launch_dependents
from tokenspeed_kernel.thirdparty.sonic_sampler.ops.encoding import shift_pack
from tokenspeed_kernel.thirdparty.sonic_sampler.ops.topk.bitonic import (
    top_k as bitonic_top_k,
)


@jit
def bitonic_selection(
    y_ptr,
    batch_id,
    block_id,
    packed,
    max_k: tl.constexpr,
    enable_pdl: tl.constexpr,
    stride_y: tl.constexpr,
    block_n: tl.constexpr,
):

    # Apply bitonic top-k to reduce tile to top-k selections in block-alternating order.

    selections = bitonic_top_k(packed, block_id & 1, max_k, block_n)

    if enable_pdl:

        gdc_launch_dependents()

    targets = (batch_id * stride_y) + (block_id * max_k) + tl.arange(0, max_k)

    tl.store(y_ptr + targets, selections)


@jit
def radix_selection(
    y_ptr,
    batch_id,
    block_id,
    encoded,
    threshold,
    margin,
    indices,
    max_k: tl.constexpr,
    enable_pdl: tl.constexpr,
    stride_y: tl.constexpr,
    block_n: tl.constexpr,
):

    # Pack indices for stable ordering.

    packed = shift_pack(encoded, indices, block_n)

    # Accumulate mask for position(s).

    equal = encoded == threshold
    equal &= tl.cumsum(equal, axis=0) <= margin

    marked = (encoded > threshold) | equal

    # Ensure ordered (coalescer-friendly) bounded positions for storing top-k entries.

    positions = tl.cumsum(marked, axis=0) - marked

    if enable_pdl:

        gdc_launch_dependents()

    shifts = (batch_id * stride_y) + (block_id * max_k) + positions

    tl.store(y_ptr + shifts, packed, mask=marked)
