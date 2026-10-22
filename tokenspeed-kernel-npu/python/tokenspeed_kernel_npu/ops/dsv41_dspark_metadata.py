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

"""Integer DSpark block metadata with aligned flat stores on Ascend."""

import torch
from tokenspeed_kernel_npu._triton import tl, triton


@triton.jit
def _block_inputs(
    Bonus,
    Start,
    Ids,
    Pos,
    Pre,
    Req,
    N,
    B0,
    S0,
    NOISE,
    BLOCK: tl.constexpr,
    HC: tl.constexpr,
    TILE: tl.constexpr,
):
    t = tl.arange(0, TILE)
    request, column = t // BLOCK, t % BLOCK
    live = t < N * BLOCK
    bonus = tl.load(Bonus + request * B0, live, 0).to(tl.int32)
    start = tl.load(Start + request * S0, live, 0).to(tl.int64)
    tl.store(Ids + t, tl.where(column == 0, bonus, NOISE), live)
    tl.store(Pos + t, start + 1 + column, live)
    tl.store(Req + t, request.to(tl.int64), live)
    p = tl.arange(0, TILE * HC)
    tl.store(Pre + p, tl.where(p % HC == 0, 1.0, 0.0), p < N * BLOCK * HC)


@triton.jit
def _block_address(
    H,
    Page,
    Row,
    Indices,
    N,
    H0,
    WINDOW: tl.constexpr,
    BLOCK: tl.constexpr,
    P: tl.constexpr,
    TILE: tl.constexpr,
):
    i = tl.program_id(0) * TILE + tl.arange(0, TILE)
    request, column = i // WINDOW, i % WINDOW
    slot = tl.load(H + request * H0 + column, i < N * WINDOW, -1).to(tl.int64)
    safe = tl.maximum(slot, 0)
    tl.store(Page + i, safe // P, i < N * WINDOW)
    tl.store(Row + i, safe % P, i < N * WINDOW)
    width: tl.constexpr = WINDOW + BLOCK
    request, column = i // (BLOCK * width), i % width
    live = i < N * BLOCK * width
    history = tl.load(H + request * H0 + column, live & (column < WINDOW), 0)
    value = tl.where((column < WINDOW) & (history < 0), -1, request * width + column)
    tl.store(Indices + i, value.to(tl.int32), live)


@triton.jit
def _anchors(
    Tokens,
    Accept,
    Positions,
    Next,
    Start,
    BS,
    EXTENDS,
    OUTPUTS,
    N0,
    WIDTH: tl.constexpr,
    SPEC: tl.constexpr,
    TILE: tl.constexpr,
):
    t = tl.arange(0, TILE)
    row, col = t // SPEC, t % SPEC
    live = row < BS
    accepted = tl.load(Accept + row, live, 1).to(tl.int64)
    accepted = tl.minimum(tl.maximum(accepted, 1), WIDTH).to(tl.int32)
    decode = row >= EXTENDS
    index = OUTPUTS + (row - EXTENDS) * WIDTH + accepted - 1
    active = live & ((row < OUTPUTS) | decode)
    bonus = tl.load(Tokens + tl.where(decode, index, row), active, 0)
    tl.store(Next + row * N0 + col, bonus.to(tl.int32), active)
    r = tl.arange(0, 32)
    live_decode = r < BS - EXTENDS
    accepted = tl.load(Accept + EXTENDS + r, live_decode, 1).to(tl.int64)
    accepted = tl.minimum(tl.maximum(accepted, 1), WIDTH).to(tl.int32)
    anchor = tl.load(Positions + r * WIDTH + accepted - 1, live_decode, -1)
    tl.store(Start + r, anchor.to(tl.int64), live_decode)


def dspark_block(
    bonus, start_pos, history_slots, noise_token, rows_per_page, hc_mult, block_size
):
    """Return the seven ordinary block tensors for at most 32 requests."""
    n, window = history_slots.shape
    device = bonus.device
    ids = torch.empty(n * block_size, device=device, dtype=torch.int32)
    positions = torch.empty(n * block_size, device=device, dtype=torch.int64)
    pre = torch.empty((n * block_size, hc_mult), device=device, dtype=torch.float32)
    requests = torch.empty_like(positions)
    page = torch.empty((n, window), device=device, dtype=torch.int64)
    row = torch.empty_like(page)
    indices = torch.empty(
        (n * block_size, window + block_size), device=device, dtype=torch.int32
    )
    if n:
        _block_inputs[(1,)](
            bonus,
            start_pos,
            ids,
            positions,
            pre,
            requests,
            n,
            bonus.stride(0),
            start_pos.stride(0),
            noise_token,
            block_size,
            hc_mult,
            triton.next_power_of_2(n * block_size),
            num_warps=1,
            enable_fp_fusion=False,
            multibuffer=False,
        )
        _block_address[(triton.cdiv(indices.numel(), 256),)](
            history_slots,
            page,
            row,
            indices,
            n,
            history_slots.stride(0),
            window,
            block_size,
            rows_per_page,
            256,
            num_warps=1,
            enable_fp_fusion=False,
            multibuffer=False,
        )
    return ids, positions, pre, requests, page, row, indices


def dspark_anchors(
    output_tokens,
    accept_lengths,
    positions,
    num_extends,
    num_prefill_outputs,
    width,
    next_tokens,
    start_pos,
):
    """Fill only output-bearing requests; incomplete prefill rows stay untouched."""
    if accept_lengths.numel():
        _anchors[(1,)](
            output_tokens,
            accept_lengths,
            positions,
            next_tokens,
            start_pos,
            accept_lengths.numel(),
            num_extends,
            num_prefill_outputs,
            next_tokens.stride(0),
            width,
            next_tokens.shape[1],
            triton.next_power_of_2(next_tokens.numel()),
            num_warps=1,
            enable_fp_fusion=False,
            multibuffer=False,
        )
