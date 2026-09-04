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

from tokenspeed_kernel.thirdparty.sonic.ops.encoding import bitpack_block
from tokenspeed_kernel.thirdparty.sonic.ops.topk.adaptive import radix_bitonic
from tokenspeed_kernel.thirdparty.sonic.ops.topk.radix import radix_threshold
from tokenspeed_kernel.thirdparty.sonic.ops.topk.selection import (
    bitonic_selection,
    radix_selection,
)


@jit
def top_k_reduction(
    y_ptr,
    row_id,
    col_id,
    logits,
    indices,
    max_k: tl.constexpr,
    scale: tl.constexpr,
    enable_adaptive: tl.constexpr,
    enable_radix: tl.constexpr,
    enable_pdl: tl.constexpr,
    stride_y: tl.constexpr,
    block_n: tl.constexpr,
):

    if enable_adaptive:

        radix_bitonic(
            y_ptr,
            row_id,
            col_id,
            logits,
            indices,
            max_k,
            scale,
            enable_pdl,
            stride_y,
            block_n,
        )

    elif enable_radix:

        # Compute the threshold and marginal via radix filtering.

        encoded, threshold, margin = radix_threshold(logits, max_k)

        # Packed radix top-k reduction.

        radix_selection(
            y_ptr,
            row_id,
            col_id,
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

        # Packed bitonic top-k reduction.

        packed = bitpack_block(logits, indices, block_n)

        bitonic_selection(
            y_ptr,
            row_id,
            col_id,
            packed,
            max_k,
            enable_pdl,
            stride_y,
            block_n,
        )
