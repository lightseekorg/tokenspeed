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

from tokenspeed_kernel.thirdparty.sonic_sampler.ops.encoding import encode_block


@jit
def radix_partition(values, mask, upper: tl.constexpr, bins: tl.constexpr):
    """
    Optimized radix pivot and margin computation via a packed masked-max reduction strategy
    with a single IMNMX reduction tree with IMAD.SHL.U32 pre-op.

    """
    ladder = tl.cumsum(tl.histogram(values, bins, mask=mask), axis=0)

    offsets = tl.arange(0, bins) + 1
    packed = tl.max(tl.where(ladder <= upper, (ladder << 8) | offsets, 0), axis=0)

    pivot = packed & 0xFF
    target = packed >> 8

    return pivot, upper - target


@jit
def radix_threshold(logits, max_k: tl.constexpr):

    # Encode logits into numerical lexicographic ordering i.e. uint32.

    encoded = encode_block(logits)

    # Filter candidates based on upper 8-bits in the first phase.

    values = (encoded >> 8) ^ 0xFF
    head, margin = radix_partition(values, None, max_k, 256)

    threshold = (head ^ 0xFF) << 8

    if margin > 0:

        # Filter candidates based on lower 8-bits in the second phase.

        mask = values == head
        values = (encoded & 0xFF) ^ 0xFF

        tail, margin = radix_partition(values, mask, margin, 256)

        if head == 0 and tail == 0:

            threshold |= 0xFE
            margin &= 0

        else:

            threshold |= tail ^ 0xFF

    else:

        threshold |= 0xFF

    return encoded, threshold, margin
