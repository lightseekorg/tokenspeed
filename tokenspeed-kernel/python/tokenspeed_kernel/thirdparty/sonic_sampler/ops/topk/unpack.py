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

from tokenspeed_kernel.thirdparty.sonic_sampler.ops.base import gdc_wait
from tokenspeed_kernel.thirdparty.sonic_sampler.ops.encoding import decode_block
from tokenspeed_kernel.thirdparty.sonic_sampler.ops.topk.bitonic import (
    bitonic_reduction,
)
from tokenspeed_kernel.thirdparty.sonic_sampler.ops.topk.bitonic import (
    top_k as bitonic_top_k,
)


@jit
def unsharded_top_k(
    s_ptr,
    row_id,
    vocab_blocks: tl.constexpr,
    max_k: tl.constexpr,
    scale: tl.constexpr,
    total_cols: tl.constexpr,
    is_adaptive: tl.constexpr,
    is_alternating: tl.constexpr,
    enable_pdl: tl.constexpr,
    stride_s: tl.constexpr,
    block_n: tl.constexpr,
    block_v: tl.constexpr,
):
    """
    Note(s):

        • `total_cols` -> `block_v` • `max_k`

    """
    # Define the XOR-based index inverting mask(s).

    upper_mask = tl.constexpr(block_n - 1)
    total_mask = tl.constexpr(total_cols - 1)

    slice_mask = tl.constexpr(max_k - 1)

    # Prepare the layout block offset(s), absolute offset(s), and relative indices.

    block_ids = tl.arange(0, block_v)[:, None]
    block_mask = block_ids < vocab_blocks

    local_indices = (block_ids * max_k) + tl.arange(0, max_k)[None, :]
    block_offsets = (row_id * stride_s) + local_indices

    local_mask = total_mask

    if is_alternating:

        local_mask = total_mask ^ (((block_ids & 1) ^ 1) * slice_mask)

    local_indices ^= local_mask
    local_offsets = block_ids * block_n

    if enable_pdl:

        gdc_wait()

    # Load the uint32 packed [ Z_v', M ] block values from the scratch-pad.

    unsharded_packed = tl.load(s_ptr + block_offsets, mask=block_mask, other=0)

    # Prepare the absolute indices across the vocab block(s).

    unsharded_absolutes = (
        ((unsharded_packed & 0xFFFF) ^ upper_mask) + local_offsets
    ).reshape((total_cols,))

    # Replace packed indices with relative indices.

    unsharded_packed = (unsharded_packed & 0xFFFF_0000) | local_indices

    # Apply bitonic reduction across vocab block(s) to select top-k entries.

    if is_alternating:

        unsharded_reduced = bitonic_reduction(
            unsharded_packed, block_v, max_k, total_cols
        )

    else:

        unsharded_reduced = bitonic_top_k(unsharded_packed, 1, max_k, total_cols)

    # Extract values and indices from packed entries.

    unsharded_values = decode_block((unsharded_reduced >> 16).to(tl.uint16)).to(
        tl.bfloat16, bitcast=True
    )

    if is_adaptive:

        unsharded_values *= scale

    unsharded_indices = (unsharded_reduced & 0xFFFF) ^ total_mask

    if is_alternating:

        factor_k: tl.constexpr = tl.standard._log2(max_k)  # noqa.

        entry_block = unsharded_indices >> factor_k
        entry_index = (unsharded_indices & slice_mask) ^ (
            ((entry_block & 1) ^ 1) * slice_mask
        )

        unsharded_indices = (entry_block << factor_k) | entry_index

    unsharded_indices = tl.gather(unsharded_absolutes, unsharded_indices, axis=0)

    return unsharded_values, unsharded_indices


@jit
def sharded_top_k(
    s_ptr,
    row_id,
    vocab_blocks: tl.constexpr,
    world_size: tl.constexpr,
    shard_size: tl.constexpr,
    max_k: tl.constexpr,
    scale: tl.constexpr,
    total_rows: tl.constexpr,
    total_cols: tl.constexpr,
    is_adaptive: tl.constexpr,
    is_alternating: tl.constexpr,
    enable_pdl: tl.constexpr,
    shard_stride: tl.constexpr,
    block_stride: tl.constexpr,
    stride_s: tl.constexpr,
    block_n: tl.constexpr,
    block_v: tl.constexpr,
):
    """
    In the sharded case, each row of the scratch-pad can be logically viewed in the form of
    [ W, Z_v, M ] where the layout and logical offset(s) across the shards, { s_i | 0 ≤ i < W },
    are given by `vocab_blocks` • `max_k` and `shard_size` respectively.

    Note(s):

        • Z_v' i.e. `block_v` here is the next power-of-2 of `vocab_blocks`.

        • `shard_stride` -> `vocab_blocks` • `max_k`

        • `block_stride` -> `block_v` • `max_k`

        • `total_cols` -> `world_size` • `block_v` • `max_k`

        • `total_rows` -> `world_size` • `block_v`

    """
    # Define the logical constant(s) of the program.

    total_mask = tl.constexpr(total_cols - 1)
    upper_mask = tl.constexpr(block_n - 1)

    slice_mask = tl.constexpr(max_k - 1)

    # Prepare the layout block offset(s), absolute offset(s), and relative indices.

    shard_ids = tl.arange(0, world_size)[:, None, None]
    block_steps = tl.arange(0, block_v)[None, :, None]

    block_shifts = (block_steps * max_k) + tl.arange(0, max_k)[None, None, :]

    layout_offsets = (row_id * stride_s) + (shard_ids * shard_stride) + block_shifts
    step_mask = block_steps < vocab_blocks

    absolute_offsets = (shard_ids * shard_size) + (block_steps * block_n)
    relative_indices = (shard_ids * block_stride) + block_shifts

    local_mask = total_mask

    if is_alternating:

        local_mask = total_mask ^ (((block_steps & 1) ^ 1) * slice_mask)

    relative_indices ^= local_mask

    if enable_pdl:

        gdc_wait()

    # Load the uint32 packed [ W, Z_v', M ] block values from the scratch-pad.

    sharded_packed = tl.load(s_ptr + layout_offsets, mask=step_mask, other=0)

    # Prepare the absolute indices across the vocab block(s).

    sharded_absolutes = (
        ((sharded_packed & 0xFFFF) ^ upper_mask) + absolute_offsets
    ).reshape((total_cols,))

    # Replace packed indices with relative indices.

    sharded_packed = (sharded_packed & 0xFFFF_0000) | relative_indices

    # Apply bitonic reduction across vocab block(s) to select top-k entries.

    if is_alternating:

        sharded_packed = sharded_packed.reshape((total_rows, max_k))
        sharded_reduced = bitonic_reduction(
            sharded_packed, total_rows, max_k, total_cols
        )

    else:

        sharded_reduced = bitonic_top_k(sharded_packed, 1, max_k, total_cols)

    # Extract values and indices from packed entries.

    sharded_values = decode_block((sharded_reduced >> 16).to(tl.uint16)).to(
        tl.bfloat16, bitcast=True
    )

    if is_adaptive:

        sharded_values *= scale

    sharded_indices = (sharded_reduced & 0xFFFF) ^ total_mask

    if is_alternating:

        factor_k: tl.constexpr = tl.standard._log2(max_k)  # noqa.

        entry_block = sharded_indices >> factor_k
        entry_index = (sharded_indices & slice_mask) ^ (
            ((entry_block & 1) ^ 1) * slice_mask
        )

        sharded_indices = (entry_block << factor_k) | entry_index

    sharded_indices = tl.gather(sharded_absolutes, sharded_indices, axis=0)

    return sharded_values, sharded_indices


@jit
def unpack_top_k(
    s_ptr,
    row_id,
    vocab_blocks: tl.constexpr,
    world_size: tl.constexpr,
    shard_size: tl.constexpr,
    max_k: tl.constexpr,
    scale: tl.constexpr,
    total_rows: tl.constexpr,
    total_cols: tl.constexpr,
    sharded: tl.constexpr,
    is_adaptive: tl.constexpr,
    is_alternating: tl.constexpr,
    enable_pdl: tl.constexpr,
    shard_stride: tl.constexpr,
    block_stride: tl.constexpr,
    stride_s: tl.constexpr,
    block_n: tl.constexpr,
    block_v: tl.constexpr,
):

    if sharded:

        return sharded_top_k(
            s_ptr,
            row_id,
            vocab_blocks,
            world_size,
            shard_size,
            max_k,
            scale,
            total_rows,
            total_cols,
            is_adaptive,
            is_alternating,
            enable_pdl,
            shard_stride,
            block_stride,
            stride_s,
            block_n,
            block_v,
        )

    else:

        return unsharded_top_k(
            s_ptr,
            row_id,
            vocab_blocks,
            max_k,
            scale,
            total_cols,
            is_adaptive,
            is_alternating,
            enable_pdl,
            stride_s,
            block_n,
            block_v,
        )
