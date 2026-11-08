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

jit = triton.jit

from tokenspeed_kernel.thirdparty.sonic_sampler.ops.encoding import (
    encode_block,
    shift_pack,
)
from tokenspeed_kernel.thirdparty.sonic_sampler.ops.topk.radix import radix_partition
from tokenspeed_kernel.thirdparty.sonic_sampler.ops.topk.selection import (
    bitonic_selection,
    radix_selection,
)


@jit
def sparse_threshold(encoded, marked, threshold, max_k: tl.constexpr):

    # Construct mid-span mask and bins for partitioning.

    bits: tl.constexpr = 5
    bins: tl.constexpr = 1 << bits

    mask: tl.constexpr = bins - 1

    # Apply radix partitioning and update threshold.

    values = ((encoded >> bits) & mask) ^ mask
    head, margin = radix_partition(values, marked, max_k, bins)

    threshold = ((threshold << bits) | (head ^ mask)) << bits

    # Construct lower-span mask and bins for conditional partitioning.

    if margin > 0:

        marked &= values == head

        values = (encoded & mask) ^ mask
        tail, margin = radix_partition(values, marked, margin, bins)

        if head == 0 and tail == 0:

            threshold |= mask - 1
            margin &= 0

        else:

            threshold |= tail ^ mask

    else:

        threshold |= mask

    return threshold, margin


@jit
def screen_combine(max_a, count_a, max_b, count_b):
    """
    Fused maximum threshold and count in a single optimized IADD and IMNMX reduction tree
    that avoids a separate counting pass and halves inter-warp synchronization relative
    to a two-pass approach.

    """
    is_greater = max_a > max_b
    is_equal = max_a == max_b

    new_max = tl.maximum(max_a, max_b)

    peak_count = tl.where(is_equal, count_a + count_b, count_b)
    new_count = tl.where(is_greater, count_a, peak_count)

    return new_max, new_count


@jit
def radix_bitonic(
    y_ptr,
    batch_id,
    block_id,
    logits,
    indices,
    max_k: tl.constexpr,  # Maximum Top-K Value -> M.
    scale: tl.constexpr,  # Power-of-2 Integer Scale Factor.
    enable_pdl: tl.constexpr,
    stride_y: tl.constexpr,
    block_n: tl.constexpr,
):

    # Encode scaled logits into numerical lexicographic ordering i.e. uint32.

    encoded = encode_block(logits * scale)

    # Verify low-entropy of most-significant bit(s).

    upper = encoded >> 10
    ones = tl.full(upper.shape, 1, dtype=tl.int32)

    threshold, count = tl.reduce((upper, ones), axis=0, combine_fn=screen_combine)

    marked = upper == threshold
    sparse = count >= max_k

    if sparse:

        # Execute two-pass radix filtering based on lower bits.

        threshold, margin = sparse_threshold(encoded, marked, threshold, max_k)

        # Apply radix filtering for masked-scattering.

        radix_selection(
            y_ptr,
            batch_id,
            block_id,
            encoded,
            threshold,
            margin,
            indices,
            max_k,
            enable_pdl,
            stride_y,
            block_n,
        )

    else:

        # Fallback to optimized bitonic top-k.

        packed = shift_pack(encoded, indices, block_n)

        bitonic_selection(
            y_ptr,
            batch_id,
            block_id,
            packed,
            max_k,
            enable_pdl,
            stride_y,
            block_n,
        )
