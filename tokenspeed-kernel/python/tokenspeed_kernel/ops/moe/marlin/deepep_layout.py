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

"""Device-counted packing of DeepEP receive rows for the Marlin MoE.

The legacy low-latency wire layout reserves every expert's worst-case receive
capacity. Marlin intermediate tensors instead use a bound on the number of
actual routes in this forward, and GPU prefix sums schedule only valid rows.
All launch shapes and allocations depend on host-known batch bounds; receive
counts never leave the GPU. The receive buffer is reused for combine output.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.activation.triton import _situ_and_mul_values


def compact_row_capacity(
    num_global_tokens: int,
    top_k: int,
    num_local_experts: int,
    recv_m: int,
    block_m: int,
) -> int:
    """Bound aligned received routes without assuming balanced expert routing.

    Every source token visits at most min(top_k, local experts) distinct local
    experts. Alignment can add block_m-1 rows per expert. Return at least one
    block so even an all-idle graph can launch the same device-counted kernels.
    """
    if num_global_tokens < 0 or recv_m < 0:
        raise ValueError("token bounds must be nonnegative")
    if min(top_k, num_local_experts, block_m) <= 0:
        raise ValueError("expert, routing, and block dimensions must be positive")
    routes = min(
        num_global_tokens * min(top_k, num_local_experts),
        num_local_experts * recv_m,
    )
    rows = routes + num_local_experts * (block_m - 1)
    return max(block_m, ((rows + block_m - 1) // block_m) * block_m)


@triton.jit
def _layout_kernel(
    counts_ptr,
    offsets_ptr,
    total_ptr,
    NUM_EXPERTS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    EXPERTS_PAD: tl.constexpr,
):
    experts = tl.arange(0, EXPERTS_PAD)
    counts = tl.load(counts_ptr + experts, experts < NUM_EXPERTS, other=0)
    padded = tl.cdiv(counts, BLOCK_M) * BLOCK_M
    ends = tl.cumsum(padded)
    tl.store(offsets_ptr + experts, ends - padded, experts < NUM_EXPERTS)
    tl.store(total_ptr, tl.sum(padded, 0))


@triton.jit
def _pack_kernel(
    recv_ptr,
    counts_ptr,
    offsets_ptr,
    packed_ptr,
    sorted_ptr,
    expert_ptr,
    RECV_M: tl.constexpr,
    HIDDEN: tl.constexpr,
    ROW_CAPACITY: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    expert = tl.program_id(0)
    column = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    count = tl.load(counts_ptr + expert)
    offset = tl.load(offsets_ptr + expert)
    for block in range(tl.cdiv(count, BLOCK_M)):
        row = block * BLOCK_M + tl.arange(0, BLOCK_M)
        valid = (row[:, None] < count) & (column[None, :] < HIDDEN)
        values = tl.load(
            recv_ptr + (expert * RECV_M + row[:, None]) * HIDDEN + column[None, :],
            valid,
            other=0,
        )
        tl.store(
            packed_ptr + (offset + row[:, None]) * HIDDEN + column[None, :],
            values,
            valid,
        )
        if tl.program_id(1) == 0:
            # Marlin's sentinel is >= size_m * top_k. Padded rows are never
            # read by either GEMM, so their activation storage stays untouched.
            ids = tl.where(row < count, offset + row, ROW_CAPACITY)
            tl.store(sorted_ptr + offset + row, ids)
            tl.store(expert_ptr + offset // BLOCK_M + block, expert)


@triton.jit
def _activation_kernel(
    x_ptr,
    out_ptr,
    counts_ptr,
    offsets_ptr,
    WIDTH: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
    IS_SITU: tl.constexpr,
    BETA: tl.constexpr,
    LINEAR_BETA: tl.constexpr,
):
    expert = tl.program_id(0)
    column = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    count = tl.load(counts_ptr + expert)
    offset = tl.load(offsets_ptr + expert)
    for block in range(tl.cdiv(count, BLOCK_M)):
        row = block * BLOCK_M + tl.arange(0, BLOCK_M)
        valid = (row[:, None] < count) & (column[None, :] < WIDTH)
        address = (offset + row[:, None]) * (2 * WIDTH) + column[None, :]
        gate = tl.load(x_ptr + address, valid, other=0).to(tl.float32)
        up = tl.load(x_ptr + address + WIDTH, valid, other=0).to(tl.float32)
        if IS_SITU:
            values = _situ_and_mul_values(
                gate,
                up,
                BETA,
                LINEAR_BETA,
                HAS_LINEAR_BETA=LINEAR_BETA is not None,
            )
        else:
            values = gate * tl.sigmoid(gate) * up
        tl.store(
            out_ptr + (offset + row[:, None]) * WIDTH + column[None, :],
            values,
            valid,
        )


@triton.jit
def _unpack_kernel(
    packed_ptr,
    counts_ptr,
    offsets_ptr,
    recv_ptr,
    RECV_M: tl.constexpr,
    HIDDEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    expert = tl.program_id(0)
    column = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    count = tl.load(counts_ptr + expert)
    offset = tl.load(offsets_ptr + expert)
    for block in range(tl.cdiv(count, BLOCK_M)):
        row = block * BLOCK_M + tl.arange(0, BLOCK_M)
        valid = (row[:, None] < count) & (column[None, :] < HIDDEN)
        values = tl.load(
            packed_ptr + (offset + row[:, None]) * HIDDEN + column[None, :],
            valid,
            other=0,
        )
        tl.store(
            recv_ptr + (expert * RECV_M + row[:, None]) * HIDDEN + column[None, :],
            values,
            valid,
        )


def pack_recv_rows(
    recv_x: torch.Tensor,
    counts: torch.Tensor,
    num_global_tokens: int,
    top_k: int,
    block_m: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack valid expert rows and build Marlin's schedule on the GPU.

    Args:
        recv_x: BF16 [local_experts, receive_capacity, latent] DeepEP output.
        counts: int32 [local_experts] live receive counts on the GPU.
        num_global_tokens: Upper bound on all source tokens in this forward.
        top_k: Maximum distinct experts selected by each source token.
        block_m: Marlin token block size.

    Returns:
        Packed activations, aligned sorted token IDs, block expert IDs, the
        device scalar padded route count, and per-expert compact offsets.
    """
    experts, recv_m, hidden = recv_x.shape
    rows = compact_row_capacity(num_global_tokens, top_k, experts, recv_m, block_m)
    packed = torch.empty((rows, hidden), dtype=recv_x.dtype, device=recv_x.device)
    offsets = torch.empty((experts,), dtype=torch.int32, device=recv_x.device)
    sorted_ids = torch.empty((rows,), dtype=torch.int32, device=recv_x.device)
    expert_ids = torch.empty(
        (rows // block_m,), dtype=torch.int32, device=recv_x.device
    )
    total = torch.empty((1,), dtype=torch.int32, device=recv_x.device)
    _layout_kernel[(1,)](
        counts,
        offsets,
        total,
        NUM_EXPERTS=experts,
        BLOCK_M=block_m,
        EXPERTS_PAD=triton.next_power_of_2(experts),
    )
    # Detect a violated caller bound on device, including during graph replay.
    # This is asynchronous (no .item()/host receive-count synchronization).
    torch._assert_async(
        total[0] <= rows, "DeepEP receive rows exceed the compact Marlin bound"
    )
    _pack_kernel[(experts, triton.cdiv(hidden, 256))](
        recv_x,
        counts,
        offsets,
        packed,
        sorted_ids,
        expert_ids,
        RECV_M=recv_m,
        HIDDEN=hidden,
        ROW_CAPACITY=rows,
        BLOCK_M=block_m,
        BLOCK_H=256,
    )
    return packed, sorted_ids, expert_ids, total, offsets


def activate_recv_rows(
    x: torch.Tensor,
    counts: torch.Tensor,
    offsets: torch.Tensor,
    block_m: int,
    activation: str,
    beta: float,
    linear_beta: float | None,
) -> torch.Tensor:
    """Apply SiTU/SiLU only to valid compact rows, leaving alignment unread.

    ``x`` is [compact_capacity, 2 * intermediate]; counts and offsets are the
    GPU metadata returned by pack_recv_rows. The result has half that width.
    """
    width = x.shape[1] // 2
    out = torch.empty((x.shape[0], width), dtype=x.dtype, device=x.device)
    _activation_kernel[(counts.numel(), triton.cdiv(width, 256))](
        x,
        out,
        counts,
        offsets,
        WIDTH=width,
        BLOCK_M=block_m,
        BLOCK_H=256,
        IS_SITU=activation == "situ",
        BETA=beta,
        LINEAR_BETA=linear_beta,
    )
    return out


def unpack_recv_rows(
    packed: torch.Tensor,
    counts: torch.Tensor,
    offsets: torch.Tensor,
    recv_x: torch.Tensor,
    block_m: int,
) -> torch.Tensor:
    """Write valid compact outputs into the consumed DeepEP receive buffer.

    Returns recv_x, now containing expert outputs in the layout combine
    consumes. DeepEP combines only valid rows; unused capacity is untouched.
    """
    experts, recv_m, hidden = recv_x.shape
    _unpack_kernel[(experts, triton.cdiv(hidden, 256))](
        packed,
        counts,
        offsets,
        recv_x,
        RECV_M=recv_m,
        HIDDEN=hidden,
        BLOCK_M=block_m,
        BLOCK_H=256,
    )
    return recv_x
