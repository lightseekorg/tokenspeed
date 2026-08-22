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

"""Triton kernels for the PyTorch symmetric-memory AR+RMSNorm example."""

from tokenspeed_kernel._triton import tl, triton


@triton.jit
def _direct_peer_ptr(peer_ptrs, peer: tl.constexpr, local_ptr):
    """Load a direct peer buffer base from PyTorch's device pointer table."""
    peer_ptrs = peer_ptrs.to(tl.pointer_type(tl.uint64))
    return tl.load(peer_ptrs + peer).to(local_ptr.dtype)


@triton.jit
def fused_ar_rmsnorm_oneshot_wholerow_kernel(
    input,
    input_peer_ptrs,
    residual,
    weight,
    norm_out,
    residual_out,
    M,
    EPS: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    NUM_PROGRAMS: tl.constexpr,
):
    """Pull every peer's row, then compute residual-add and RMSNorm locally."""
    tl.static_assert(
        (HIDDEN_SIZE & (HIDDEN_SIZE - 1)) == 0,
        "whole-row AR+RMSNorm requires a power-of-two hidden size",
    )
    pid = tl.program_id(0)
    columns = tl.max_contiguous(
        tl.multiple_of(tl.arange(0, HIDDEN_SIZE), HIDDEN_SIZE),
        HIDDEN_SIZE,
    )
    gamma = tl.load(weight + columns).to(tl.float32)

    for row in range(pid, M, NUM_PROGRAMS):
        offsets = row * HIDDEN_SIZE + columns
        reduced = tl.load(input + offsets).to(tl.float32)
        for peer in tl.static_range(0, WORLD_SIZE):
            if peer != RANK:
                peer_input = _direct_peer_ptr(input_peer_ptrs, peer, input)
                reduced += tl.load(peer_input + offsets).to(tl.float32)

        residual_value = reduced + tl.load(residual + offsets).to(tl.float32)
        tl.store(residual_out + offsets, residual_value)
        variance = tl.sum(residual_value * residual_value) / HIDDEN_SIZE
        tl.store(norm_out + offsets, residual_value * tl.rsqrt(variance + EPS) * gamma)


@triton.jit
def fused_ar_rmsnorm_oneshot_blocked_kernel(
    input,
    input_peer_ptrs,
    residual,
    weight,
    scratch,
    norm_out,
    residual_out,
    M,
    EPS: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    NUM_PROGRAMS: tl.constexpr,
):
    """One-shot pull for hidden sizes that do not fit a whole-row program."""
    tl.static_assert(
        (BLOCK_SIZE & (BLOCK_SIZE - 1)) == 0,
        "blocked AR+RMSNorm requires a power-of-two block size",
    )
    pid = tl.program_id(0)
    columns = tl.arange(0, BLOCK_SIZE)

    for row in range(pid, M, NUM_PROGRAMS):
        row_offset = row * HIDDEN_SIZE
        scratch_offset = pid * HIDDEN_SIZE
        sum_squares = tl.zeros((), tl.float32)

        for block_start in tl.static_range(0, HIDDEN_SIZE, BLOCK_SIZE):
            block_columns = block_start + columns
            mask = block_columns < HIDDEN_SIZE
            offsets = row_offset + block_columns
            reduced = tl.load(input + offsets, mask=mask, other=0.0).to(tl.float32)
            for peer in tl.static_range(0, WORLD_SIZE):
                if peer != RANK:
                    peer_input = _direct_peer_ptr(input_peer_ptrs, peer, input)
                    reduced += tl.load(
                        peer_input + offsets,
                        mask=mask,
                        other=0.0,
                    ).to(tl.float32)

            residual_value = reduced + tl.load(
                residual + offsets,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            tl.store(residual_out + offsets, residual_value, mask=mask)
            tl.store(
                scratch + scratch_offset + block_columns, residual_value, mask=mask
            )
            sum_squares += tl.sum(residual_value * residual_value)

        scale = tl.rsqrt(sum_squares / HIDDEN_SIZE + EPS)
        for block_start in tl.static_range(0, HIDDEN_SIZE, BLOCK_SIZE):
            block_columns = block_start + columns
            mask = block_columns < HIDDEN_SIZE
            offsets = row_offset + block_columns
            value = tl.load(
                scratch + scratch_offset + block_columns,
                mask=mask,
                other=0.0,
            )
            gamma = tl.load(weight + block_columns, mask=mask, other=0.0).to(tl.float32)
            tl.store(norm_out + offsets, value * scale * gamma, mask=mask)


@triton.jit
def fused_ar_rmsnorm_twoshot_blocked_kernel(
    input,
    input_peer_ptrs,
    residual,
    output,
    output_peer_ptrs,
    residual_out,
    residual_out_peer_ptrs,
    weight,
    scratch,
    M,
    EPS: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    NUM_PROGRAMS: tl.constexpr,
):
    """Reduce this rank's row shard, normalize it, and push it to every peer."""
    tl.static_assert(
        (BLOCK_SIZE & (BLOCK_SIZE - 1)) == 0,
        "blocked AR+RMSNorm requires a power-of-two block size",
    )
    pid = tl.program_id(0)
    shard_rows = tl.cdiv(M, WORLD_SIZE)
    first_row = shard_rows * RANK
    columns = tl.arange(0, BLOCK_SIZE)

    for shard_row in range(pid, shard_rows, NUM_PROGRAMS):
        row = first_row + shard_row
        if row < M:
            row_offset = row * HIDDEN_SIZE
            scratch_offset = shard_row * HIDDEN_SIZE
            sum_squares = tl.zeros((), tl.float32)

            for block_start in tl.static_range(0, HIDDEN_SIZE, BLOCK_SIZE):
                block_columns = block_start + columns
                mask = block_columns < HIDDEN_SIZE
                offsets = row_offset + block_columns
                reduced = tl.load(input + offsets, mask=mask, other=0.0).to(tl.float32)
                for peer in tl.static_range(0, WORLD_SIZE):
                    if peer != RANK:
                        peer_input = _direct_peer_ptr(input_peer_ptrs, peer, input)
                        reduced += tl.load(
                            peer_input + offsets,
                            mask=mask,
                            other=0.0,
                        ).to(tl.float32)

                residual_value = reduced + tl.load(
                    residual + offsets,
                    mask=mask,
                    other=0.0,
                ).to(tl.float32)
                tl.store(residual_out + offsets, residual_value, mask=mask)
                for peer in tl.static_range(0, WORLD_SIZE):
                    if peer != RANK:
                        peer_residual = _direct_peer_ptr(
                            residual_out_peer_ptrs,
                            peer,
                            residual_out,
                        )
                        tl.store(peer_residual + offsets, residual_value, mask=mask)

                tl.store(
                    scratch + scratch_offset + block_columns,
                    residual_value,
                    mask=mask,
                )
                sum_squares += tl.sum(residual_value * residual_value)

            scale = tl.rsqrt(sum_squares / HIDDEN_SIZE + EPS)
            for block_start in tl.static_range(0, HIDDEN_SIZE, BLOCK_SIZE):
                block_columns = block_start + columns
                mask = block_columns < HIDDEN_SIZE
                offsets = row_offset + block_columns
                value = tl.load(
                    scratch + scratch_offset + block_columns,
                    mask=mask,
                    other=0.0,
                )
                gamma = tl.load(weight + block_columns, mask=mask, other=0.0).to(
                    tl.float32
                )
                normalized = value * scale * gamma
                tl.store(output + offsets, normalized, mask=mask)
                for peer in tl.static_range(0, WORLD_SIZE):
                    if peer != RANK:
                        peer_output = _direct_peer_ptr(
                            output_peer_ptrs,
                            peer,
                            output,
                        )
                        tl.store(peer_output + offsets, normalized, mask=mask)
