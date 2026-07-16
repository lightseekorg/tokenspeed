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

"""DSA top-k Gluon kernels for AMD GFX950."""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, triton
from tokenspeed_kernel_amd.ops.attention.gluon.dsa_score_gfx950 import (
    _check_packed_fp8_inputs,
    _dsa_decode_logits_fp8_kernel,
    _dsa_prefill_logits_fp8_kernel,
)

_RADIX_TOPK_MIN_COLS = 65536
_RADIX_TOPK_BLOCK_N = 4096

__all__ = [
    "gluon_dsa_decode_topk_fp8_gfx950",
    "gluon_dsa_prefill_topk_fp8_gfx950",
]


@gluon.constexpr_function
def _vector_layout(
    BLOCK: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    LOAD_ELEMS: gl.constexpr,
):
    return gl.BlockedLayout([LOAD_ELEMS], [64], [NUM_WARPS], [0])


@gluon.jit
def _fp32_to_ordered_key(x):
    bits = x.to(gl.uint32, bitcast=True)
    sign = bits & 0x80000000
    return bits ^ gl.where(sign != 0, 0xFFFFFFFF, 0x80000000)


@gluon.jit
def _topk_add(a, b):
    return a + b


@gluon.jit
def _find_topk_threshold_key(
    values,
    valid,
    topk: gl.constexpr,
    BLOCK_N: gl.constexpr,
    layout: gl.constexpr,
):
    keys = _fp32_to_ordered_key(values)
    prefix = 0
    remaining = topk

    # Ordered FP32 keys are searched from the most-significant 4-bit nibble down.
    for shift in gl.static_range(28, -1, -4):
        if shift == 28:
            prefix_match = valid
        else:
            prefix_match = valid & ((keys >> (shift + 4)) == prefix)
        bucket = (keys >> shift) & 0xF
        cumulative = 0
        selected = 0
        selected_remaining = remaining
        found = 0

        for bucket_id in gl.static_range(15, -1, -1):
            in_bucket = prefix_match & (bucket == bucket_id)
            count = gl.sum(
                gl.where(
                    in_bucket,
                    gl.full([BLOCK_N], 1, gl.int32, layout=layout),
                    gl.full([BLOCK_N], 0, gl.int32, layout=layout),
                ),
                axis=0,
            ).to(gl.int32)
            take = (found == 0) & (remaining <= cumulative + count)
            selected = gl.where(take, bucket_id, selected)
            selected_remaining = gl.where(
                take, remaining - cumulative, selected_remaining
            )
            cumulative += gl.where(found == 0, count, 0)
            found = gl.where(take, 1, found)

        prefix = (prefix << 4) | selected
        remaining = selected_remaining

    return prefix


@gluon.jit
def _dsa_decode_select_topk_kernel(
    logits,
    block_table,
    seq_lens,
    out,
    lens_out,
    logits_stride: gl.constexpr,
    block_table_stride: gl.constexpr,
    out_stride: gl.constexpr,
    block_table_cols: gl.constexpr,
    page_size: gl.constexpr,
    topk: gl.constexpr,
    q_len_per_req: gl.constexpr,
    BLOCK_N: gl.constexpr,
    LOAD_ELEMS: gl.constexpr,
    TOPK_LOAD_ELEMS: gl.constexpr,
):
    row = gl.program_id(0)
    layout: gl.constexpr = _vector_layout(BLOCK_N, gl.num_warps(), LOAD_ELEMS)
    topk_layout: gl.constexpr = _vector_layout(topk, gl.num_warps(), TOPK_LOAD_ELEMS)
    offsets = gl.arange(0, BLOCK_N, layout=layout)
    top_offsets = gl.arange(0, topk, layout=topk_layout)
    req = row // q_len_per_req
    q_offset = row - req * q_len_per_req
    seq_len = gl.load(seq_lens + req).to(gl.int32)
    if q_len_per_req != 1:
        seq_len = seq_len - (q_len_per_req - 1) + q_offset
    lens = gl.minimum(seq_len, topk).to(gl.int32)
    gl.store(lens_out + row, lens)
    gl.store(out + row * out_stride + top_offsets, -1)

    if seq_len <= topk:
        valid_top = top_offsets < seq_len
        local = top_offsets.to(gl.int32)
        block_idx = local // page_size
        block_offset = local - block_idx * page_size
        page = gl.load(
            block_table + req * block_table_stride + block_idx,
            mask=valid_top & (block_idx < block_table_cols),
            other=0,
        ).to(gl.int32)
        slots = page * page_size + block_offset
        gl.store(
            out + row * out_stride + top_offsets,
            gl.where(valid_top, slots, -1),
            mask=top_offsets < topk,
        )
        return

    valid = offsets < seq_len
    values = gl.load(
        logits + row * logits_stride + offsets,
        mask=valid,
        other=-float("inf"),
    )
    threshold = _find_topk_threshold_key(values, valid, topk, BLOCK_N, layout)
    keys = _fp32_to_ordered_key(values)
    greater = valid & (keys > threshold)
    equal = valid & (keys == threshold)
    greater_i32 = greater.to(gl.int32)
    equal_i32 = equal.to(gl.int32)
    count_greater = gl.sum(greater_i32, axis=0).to(gl.int32)
    greater_pos = gl.associative_scan(greater_i32, 0, _topk_add) - 1
    equal_pos = count_greater + gl.associative_scan(equal_i32, 0, _topk_add) - 1
    greater_write = greater & (greater_pos < topk)
    equal_write = equal & (equal_pos < topk)
    local = offsets.to(gl.int32)
    block_idx = local // page_size
    block_offset = local - block_idx * page_size
    page = gl.load(
        block_table + req * block_table_stride + block_idx,
        mask=(greater_write | equal_write) & (block_idx < block_table_cols),
        other=0,
    ).to(gl.int32)
    slots = page * page_size + block_offset
    gl.store(out + row * out_stride + greater_pos, slots, mask=greater_write)
    gl.store(out + row * out_stride + equal_pos, slots, mask=equal_write)


@gluon.jit
def _dsa_prefill_select_topk_kernel(
    logits,
    row_starts,
    row_ends,
    out,
    lens_out,
    logits_stride: gl.constexpr,
    out_stride: gl.constexpr,
    topk: gl.constexpr,
    BLOCK_N: gl.constexpr,
    LOAD_ELEMS: gl.constexpr,
    TOPK_LOAD_ELEMS: gl.constexpr,
):
    row = gl.program_id(0)
    layout: gl.constexpr = _vector_layout(BLOCK_N, gl.num_warps(), LOAD_ELEMS)
    topk_layout: gl.constexpr = _vector_layout(topk, gl.num_warps(), TOPK_LOAD_ELEMS)
    offsets = gl.arange(0, BLOCK_N, layout=layout)
    top_offsets = gl.arange(0, topk, layout=topk_layout)
    row_start = gl.load(row_starts + row).to(gl.int32)
    row_end = gl.load(row_ends + row).to(gl.int32)
    candidate_len = gl.maximum(row_end - row_start, 0)
    lens = gl.minimum(candidate_len, topk).to(gl.int32)
    gl.store(lens_out + row, lens)
    gl.store(out + row * out_stride + top_offsets, -1)

    if candidate_len <= topk:
        local = row_start + top_offsets.to(gl.int32)
        valid_top = top_offsets < candidate_len
        gl.store(
            out + row * out_stride + top_offsets,
            gl.where(valid_top, local, -1),
            mask=top_offsets < topk,
        )
        return

    valid = (offsets >= row_start) & (offsets < row_end)
    values = gl.load(
        logits + row * logits_stride + offsets,
        mask=valid,
        other=-float("inf"),
    )
    threshold = _find_topk_threshold_key(values, valid, topk, BLOCK_N, layout)
    keys = _fp32_to_ordered_key(values)
    greater = valid & (keys > threshold)
    equal = valid & (keys == threshold)
    greater_i32 = greater.to(gl.int32)
    equal_i32 = equal.to(gl.int32)
    count_greater = gl.sum(greater_i32, axis=0).to(gl.int32)
    greater_pos = gl.associative_scan(greater_i32, 0, _topk_add) - 1
    equal_pos = count_greater + gl.associative_scan(equal_i32, 0, _topk_add) - 1
    greater_write = greater & (greater_pos < topk)
    equal_write = equal & (equal_pos < topk)
    local = offsets.to(gl.int32)
    gl.store(out + row * out_stride + greater_pos, local, mask=greater_write)
    gl.store(out + row * out_stride + equal_pos, local, mask=equal_write)


@gluon.jit
def _dsa_decode_radix_init_kernel(
    seq_lens,
    out,
    lens_out,
    prefixes,
    remaining,
    counters,
    out_stride: gl.constexpr,
    topk: gl.constexpr,
    q_len_per_req: gl.constexpr,
    TOPK_LOAD_ELEMS: gl.constexpr,
):
    row = gl.program_id(0)
    top_layout: gl.constexpr = _vector_layout(topk, gl.num_warps(), TOPK_LOAD_ELEMS)
    top_offsets = gl.arange(0, topk, layout=top_layout)
    req = row // q_len_per_req
    q_offset = row - req * q_len_per_req
    seq_len = gl.load(seq_lens + req).to(gl.int32)
    if q_len_per_req != 1:
        seq_len = seq_len - (q_len_per_req - 1) + q_offset
    lens = gl.minimum(seq_len, topk).to(gl.int32)
    gl.store(lens_out + row, lens)
    gl.store(prefixes + row, 0)
    gl.store(remaining + row, lens)
    gl.store(counters + row * 2, 0)
    gl.store(counters + row * 2 + 1, 0)
    gl.store(out + row * out_stride + top_offsets, -1, mask=top_offsets < topk)


@gluon.jit
def _dsa_prefill_radix_init_kernel(
    row_starts,
    row_ends,
    out,
    lens_out,
    prefixes,
    remaining,
    counters,
    out_stride: gl.constexpr,
    topk: gl.constexpr,
    TOPK_LOAD_ELEMS: gl.constexpr,
):
    row = gl.program_id(0)
    top_layout: gl.constexpr = _vector_layout(topk, gl.num_warps(), TOPK_LOAD_ELEMS)
    top_offsets = gl.arange(0, topk, layout=top_layout)
    row_start = gl.load(row_starts + row).to(gl.int32)
    row_end = gl.load(row_ends + row).to(gl.int32)
    candidate_len = gl.maximum(row_end - row_start, 0)
    lens = gl.minimum(candidate_len, topk).to(gl.int32)
    gl.store(lens_out + row, lens)
    gl.store(prefixes + row, 0)
    gl.store(remaining + row, lens)
    gl.store(counters + row * 2, 0)
    gl.store(counters + row * 2 + 1, 0)
    gl.store(out + row * out_stride + top_offsets, -1, mask=top_offsets < topk)


@gluon.jit
def _dsa_radix_hist_kernel(
    logits,
    prefixes,
    hist,
    logits_stride: gl.constexpr,
    hist_tiles: gl.constexpr,
    n_cols: gl.constexpr,
    shift: gl.constexpr,
    BLOCK_N: gl.constexpr,
    LOAD_ELEMS: gl.constexpr,
):
    row = gl.program_id(0)
    tile = gl.program_id(1)
    layout: gl.constexpr = _vector_layout(BLOCK_N, gl.num_warps(), LOAD_ELEMS)
    offsets = tile * BLOCK_N + gl.arange(0, BLOCK_N, layout=layout)
    mask = offsets < n_cols
    values = gl.load(
        logits + row * logits_stride + offsets,
        mask=mask,
        other=-float("inf"),
    )
    keys = _fp32_to_ordered_key(values)
    prefix = gl.load(prefixes + row).to(gl.uint32)
    if shift == 28:
        prefix_match = mask
    else:
        prefix_match = (keys >> (shift + 4)) == prefix
    bucket = (keys >> shift) & 0xF
    base = (row * hist_tiles + tile) * 16
    for bucket_id in gl.static_range(0, 16):
        count = gl.sum(
            gl.where(
                mask & prefix_match & (bucket == bucket_id),
                gl.full([BLOCK_N], 1, gl.int32, layout=layout),
                gl.full([BLOCK_N], 0, gl.int32, layout=layout),
            ),
            axis=0,
        ).to(gl.int32)
        gl.store(hist + base + bucket_id, count)


@gluon.jit
def _dsa_radix_update_kernel(
    prefixes,
    remaining,
    hist,
    hist_tiles: gl.constexpr,
    BLOCK_TILES: gl.constexpr,
    LOAD_ELEMS: gl.constexpr,
):
    row = gl.program_id(0)
    layout: gl.constexpr = _vector_layout(BLOCK_TILES, gl.num_warps(), LOAD_ELEMS)
    tile_offsets = gl.arange(0, BLOCK_TILES, layout=layout)
    tile_mask = tile_offsets < hist_tiles
    row_hist = hist + row * hist_tiles * 16
    kth = gl.load(remaining + row).to(gl.int32)
    cumulative = 0
    selected = 0
    selected_remaining = kth
    found = 0

    for bucket_desc in gl.static_range(0, 16):
        bucket_id = 15 - bucket_desc
        counts = gl.load(
            row_hist + tile_offsets * 16 + bucket_id,
            mask=tile_mask,
            other=0,
        )
        count = gl.sum(counts, axis=0).to(gl.int32)
        take = (found == 0) & (kth <= cumulative + count)
        selected = gl.where(take, bucket_id, selected)
        selected_remaining = gl.where(take, kth - cumulative, selected_remaining)
        cumulative += gl.where(found == 0, count, 0)
        found = gl.where(take, 1, found)

    prefix = gl.load(prefixes + row).to(gl.uint32)
    gl.store(prefixes + row, ((prefix << 4) | selected).to(gl.int32))
    gl.store(remaining + row, selected_remaining)


@gluon.jit
def _dsa_decode_radix_scatter_slots_kernel(
    logits,
    prefixes,
    remaining,
    counters,
    block_table,
    seq_lens,
    out,
    logits_stride: gl.constexpr,
    block_table_stride: gl.constexpr,
    out_stride: gl.constexpr,
    block_table_cols: gl.constexpr,
    n_cols: gl.constexpr,
    page_size: gl.constexpr,
    topk: gl.constexpr,
    q_len_per_req: gl.constexpr,
    BLOCK_N: gl.constexpr,
    LOAD_ELEMS: gl.constexpr,
):
    row = gl.program_id(0)
    tile = gl.program_id(1)
    layout: gl.constexpr = _vector_layout(BLOCK_N, gl.num_warps(), LOAD_ELEMS)
    offsets = tile * BLOCK_N + gl.arange(0, BLOCK_N, layout=layout)
    req = row // q_len_per_req
    q_offset = row - req * q_len_per_req
    seq_len = gl.load(seq_lens + req).to(gl.int32)
    if q_len_per_req != 1:
        seq_len = seq_len - (q_len_per_req - 1) + q_offset
    mask = (offsets < n_cols) & (offsets < seq_len)
    values = gl.load(
        logits + row * logits_stride + offsets,
        mask=mask,
        other=-float("inf"),
    )
    threshold = gl.load(prefixes + row).to(gl.uint32)
    keep_equal = gl.load(remaining + row).to(gl.int32)
    count_greater = topk - keep_equal
    finite = values != -float("inf")
    keys = _fp32_to_ordered_key(values)
    greater = mask & finite & (keys > threshold)
    equal = mask & finite & (keys == threshold)

    greater_i32 = greater.to(gl.int32)
    equal_i32 = equal.to(gl.int32)
    tile_greater = gl.sum(greater_i32, axis=0).to(gl.int32)
    tile_equal = gl.sum(equal_i32, axis=0).to(gl.int32)
    greater_start = gl.atomic_add(
        counters + row * 2, tile_greater, sem="acq_rel", scope="gpu"
    )
    equal_start = gl.atomic_add(
        counters + row * 2 + 1, tile_equal, sem="acq_rel", scope="gpu"
    )

    greater_pos = greater_start + gl.associative_scan(greater_i32, 0, _topk_add) - 1
    equal_pos = (
        count_greater + equal_start + gl.associative_scan(equal_i32, 0, _topk_add) - 1
    )
    greater_write = greater & (greater_pos < topk)
    equal_write = equal & (equal_pos < topk) & (equal_pos < count_greater + keep_equal)

    block_idx = offsets.to(gl.int32) // page_size
    block_offset = offsets.to(gl.int32) - block_idx * page_size
    page = gl.load(
        block_table + req * block_table_stride + block_idx,
        mask=(greater_write | equal_write) & (block_idx < block_table_cols),
        other=0,
    ).to(gl.int32)
    slots = page * page_size + block_offset
    gl.store(out + row * out_stride + greater_pos, slots, mask=greater_write)
    gl.store(out + row * out_stride + equal_pos, slots, mask=equal_write)


@gluon.jit
def _dsa_prefill_radix_scatter_kernel(
    logits,
    prefixes,
    remaining,
    counters,
    row_starts,
    row_ends,
    out,
    logits_stride: gl.constexpr,
    out_stride: gl.constexpr,
    n_cols: gl.constexpr,
    topk: gl.constexpr,
    BLOCK_N: gl.constexpr,
    LOAD_ELEMS: gl.constexpr,
):
    row = gl.program_id(0)
    tile = gl.program_id(1)
    layout: gl.constexpr = _vector_layout(BLOCK_N, gl.num_warps(), LOAD_ELEMS)
    offsets = tile * BLOCK_N + gl.arange(0, BLOCK_N, layout=layout)
    row_start = gl.load(row_starts + row).to(gl.int32)
    row_end = gl.load(row_ends + row).to(gl.int32)
    mask = (offsets >= row_start) & (offsets < row_end) & (offsets < n_cols)
    values = gl.load(
        logits + row * logits_stride + offsets,
        mask=mask,
        other=-float("inf"),
    )
    threshold = gl.load(prefixes + row).to(gl.uint32)
    keep_equal = gl.load(remaining + row).to(gl.int32)
    count_greater = topk - keep_equal
    finite = values != -float("inf")
    keys = _fp32_to_ordered_key(values)
    greater = mask & finite & (keys > threshold)
    equal = mask & finite & (keys == threshold)

    greater_i32 = greater.to(gl.int32)
    equal_i32 = equal.to(gl.int32)
    tile_greater = gl.sum(greater_i32, axis=0).to(gl.int32)
    tile_equal = gl.sum(equal_i32, axis=0).to(gl.int32)
    greater_start = gl.atomic_add(
        counters + row * 2, tile_greater, sem="acq_rel", scope="gpu"
    )
    equal_start = gl.atomic_add(
        counters + row * 2 + 1, tile_equal, sem="acq_rel", scope="gpu"
    )

    greater_pos = greater_start + gl.associative_scan(greater_i32, 0, _topk_add) - 1
    equal_pos = (
        count_greater + equal_start + gl.associative_scan(equal_i32, 0, _topk_add) - 1
    )
    greater_write = greater & (greater_pos < topk)
    equal_write = equal & (equal_pos < topk) & (equal_pos < count_greater + keep_equal)
    local = offsets.to(gl.int32)
    gl.store(out + row * out_stride + greater_pos, local, mask=greater_write)
    gl.store(out + row * out_stride + equal_pos, local, mask=equal_write)


def _load_elems(block: int, num_warps: int) -> int:
    return max(1, triton.cdiv(int(block), 64 * int(num_warps)))


def _contiguous(tensor: torch.Tensor) -> torch.Tensor:
    return tensor if tensor.is_contiguous() else tensor.contiguous()


def _to_contiguous(
    tensor: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    tensor = tensor.to(device=device, dtype=dtype)
    return _contiguous(tensor)


def _validate_topk(topk: int) -> None:
    if topk <= 0:
        raise ValueError(f"topk must be positive, got {topk}")
    if topk & (topk - 1):
        raise ValueError(f"DSA Gluon top-k requires power-of-two topk, got {topk}")


def _use_radix_topk(cols: int) -> bool:
    return int(cols) >= _RADIX_TOPK_MIN_COLS


def _dsa_radix_scratch(
    rows: int,
    cols: int,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    tiles = triton.cdiv(int(cols), _RADIX_TOPK_BLOCK_N)
    hist = torch.empty((rows, tiles, 16), dtype=torch.int32, device=device)
    prefixes = torch.empty((rows,), dtype=torch.int32, device=device)
    remaining = torch.empty((rows,), dtype=torch.int32, device=device)
    counters = torch.empty((rows, 2), dtype=torch.int32, device=device)
    block_tiles = triton.next_power_of_2(tiles)
    return hist, prefixes, remaining, counters, tiles, block_tiles


def _run_radix_prefix_passes(
    logits: torch.Tensor,
    hist: torch.Tensor,
    prefixes: torch.Tensor,
    remaining: torch.Tensor,
    *,
    rows: int,
    cols: int,
    tiles: int,
    block_tiles: int,
) -> None:
    hist_load_elems = _load_elems(_RADIX_TOPK_BLOCK_N, 8)
    update_load_elems = _load_elems(block_tiles, 8)
    # Ordered FP32 keys have 8 nibbles; each pass fixes one more prefix nibble.
    for shift in range(28, -1, -4):
        _dsa_radix_hist_kernel[(rows, tiles)](
            logits,
            prefixes,
            hist,
            logits.stride(0),
            tiles,
            n_cols=cols,
            shift=shift,
            BLOCK_N=_RADIX_TOPK_BLOCK_N,
            LOAD_ELEMS=hist_load_elems,
            num_warps=8,
        )
        _dsa_radix_update_kernel[(rows,)](
            prefixes,
            remaining,
            hist,
            hist_tiles=tiles,
            BLOCK_TILES=block_tiles,
            LOAD_ELEMS=update_load_elems,
            num_warps=8,
        )


def _dsa_decode_radix_topk_slots(
    logits: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    page_size: int,
    topk: int,
    q_len_per_req: int,
    out: torch.Tensor,
    lens_out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, cols = logits.shape
    hist, prefixes, remaining, counters, tiles, block_tiles = _dsa_radix_scratch(
        rows, cols, device=logits.device
    )
    _dsa_decode_radix_init_kernel[(rows,)](
        seq_lens,
        out,
        lens_out,
        prefixes,
        remaining,
        counters,
        out.stride(0),
        topk=topk,
        q_len_per_req=q_len_per_req,
        TOPK_LOAD_ELEMS=_load_elems(topk, 8),
        num_warps=8,
    )
    _run_radix_prefix_passes(
        logits,
        hist,
        prefixes,
        remaining,
        rows=rows,
        cols=cols,
        tiles=tiles,
        block_tiles=block_tiles,
    )
    _dsa_decode_radix_scatter_slots_kernel[(rows, tiles)](
        logits,
        prefixes,
        remaining,
        counters,
        block_table,
        seq_lens,
        out,
        logits.stride(0),
        block_table.stride(0),
        out.stride(0),
        block_table.shape[1],
        n_cols=cols,
        page_size=int(page_size),
        topk=topk,
        q_len_per_req=q_len_per_req,
        BLOCK_N=_RADIX_TOPK_BLOCK_N,
        LOAD_ELEMS=_load_elems(_RADIX_TOPK_BLOCK_N, 8),
        num_warps=8,
    )
    return out, lens_out


def _dsa_prefill_radix_topk(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    *,
    topk: int,
    out: torch.Tensor,
    lens_out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, cols = logits.shape
    hist, prefixes, remaining, counters, tiles, block_tiles = _dsa_radix_scratch(
        rows, cols, device=logits.device
    )
    _dsa_prefill_radix_init_kernel[(rows,)](
        row_starts,
        row_ends,
        out,
        lens_out,
        prefixes,
        remaining,
        counters,
        out.stride(0),
        topk=topk,
        TOPK_LOAD_ELEMS=_load_elems(topk, 8),
        num_warps=8,
    )
    _run_radix_prefix_passes(
        logits,
        hist,
        prefixes,
        remaining,
        rows=rows,
        cols=cols,
        tiles=tiles,
        block_tiles=block_tiles,
    )
    _dsa_prefill_radix_scatter_kernel[(rows, tiles)](
        logits,
        prefixes,
        remaining,
        counters,
        row_starts,
        row_ends,
        out,
        logits.stride(0),
        out.stride(0),
        n_cols=cols,
        topk=topk,
        BLOCK_N=_RADIX_TOPK_BLOCK_N,
        LOAD_ELEMS=_load_elems(_RADIX_TOPK_BLOCK_N, 8),
        num_warps=8,
    )
    return out, lens_out


def gluon_dsa_decode_topk_fp8_gfx950(
    q: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    *,
    page_size: int,
    topk: int,
    softmax_scale: float,
    q_len_per_req: int = 1,
    index_k_cache: torch.Tensor | None = None,
    seq_lens_2d: torch.Tensor | None = None,
    plan: object | None = None,
    out: torch.Tensor | None = None,
    lens_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    del plan, seq_lens_2d
    topk = int(topk)
    q_len_per_req = int(q_len_per_req)
    _validate_topk(topk)
    if not 1 <= q_len_per_req <= 6:
        raise ValueError(f"q_len_per_req must be in [1, 6], got {q_len_per_req}")
    if index_k_cache is None:
        raise RuntimeError("Gluon DSA paged top-k requires packed FP8 index_k_cache")
    row_bytes = _check_packed_fp8_inputs(q, index_k_cache, weights, int(page_size))
    if seq_lens.dim() != 1:
        raise ValueError(
            f"seq_lens must be 1-D, got {tuple(seq_lens.shape)} for q={tuple(q.shape)}"
        )
    expected_tokens = int(seq_lens.numel()) * q_len_per_req
    if expected_tokens != q.shape[0]:
        raise ValueError(
            "q rows must equal seq_lens rows times q_len_per_req, got "
            f"q={tuple(q.shape)}, seq_lens={tuple(seq_lens.shape)}, "
            f"q_len_per_req={q_len_per_req}"
        )
    if block_table.dim() != 2 or block_table.shape[0] < seq_lens.numel():
        raise ValueError(
            "block_table must have at least one row per request, got "
            f"block_table={tuple(block_table.shape)}, q={tuple(q.shape)}"
        )
    if q.shape[0] == 0:
        empty_out = (
            torch.empty((0, int(topk)), dtype=torch.int32, device=q.device)
            if out is None
            else out
        )
        empty_lens = (
            torch.empty((0,), dtype=torch.int32, device=q.device)
            if lens_out is None
            else lens_out
        )
        return empty_out, empty_lens
    q = _contiguous(q)
    index_k_cache = _contiguous(index_k_cache)
    weights = _contiguous(weights)
    seq_lens = _to_contiguous(seq_lens, device=q.device, dtype=torch.int32)
    block_table = _to_contiguous(block_table, device=q.device, dtype=torch.int32)
    max_seq_len = int(block_table.shape[1]) * int(page_size)
    if out is None:
        out = torch.empty((q.shape[0], topk), dtype=torch.int32, device=q.device)
    if lens_out is None:
        lens_out = torch.empty((q.shape[0],), dtype=torch.int32, device=q.device)
    logits = torch.empty(
        (q.shape[0], max_seq_len), dtype=torch.float32, device=q.device
    )
    block_n = 32
    _dsa_decode_logits_fp8_kernel[(q.shape[0], triton.cdiv(max_seq_len, block_n))](
        q,
        index_k_cache.view(torch.float8_e4m3fn),
        index_k_cache.view(torch.float32),
        weights,
        seq_lens,
        block_table,
        logits,
        block_table.stride(0),
        logits.stride(0),
        page_size=int(page_size),
        row_bytes=row_bytes,
        max_seq_len=max_seq_len,
        num_heads=q.shape[1],
        head_dim=q.shape[2],
        num_groups=q.shape[2] // 128,
        softmax_scale=float(softmax_scale),
        q_len_per_req=q_len_per_req,
        BLOCK_N=block_n,
        BLOCK_D=128,
        num_warps=4,
    )
    if _use_radix_topk(max_seq_len):
        return _dsa_decode_radix_topk_slots(
            logits,
            block_table,
            seq_lens,
            page_size=int(page_size),
            topk=topk,
            q_len_per_req=q_len_per_req,
            out=out,
            lens_out=lens_out,
        )

    select_warps = 8
    select_block = triton.next_power_of_2(max(max_seq_len, topk))
    _dsa_decode_select_topk_kernel[(q.shape[0],)](
        logits,
        block_table,
        seq_lens,
        out,
        lens_out,
        logits.stride(0),
        block_table.stride(0),
        out.stride(0),
        block_table.shape[1],
        page_size=int(page_size),
        topk=topk,
        q_len_per_req=q_len_per_req,
        BLOCK_N=select_block,
        LOAD_ELEMS=_load_elems(select_block, select_warps),
        TOPK_LOAD_ELEMS=_load_elems(topk, select_warps),
        num_warps=select_warps,
    )
    return out, lens_out


def gluon_dsa_prefill_topk_fp8_gfx950(
    q: torch.Tensor,
    weights: torch.Tensor,
    kv_workspace_slots: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    *,
    topk: int,
    softmax_scale: float,
    index_k_cache: torch.Tensor | None = None,
    page_size: int | None = None,
    index_k_fp8: torch.Tensor | None = None,
    index_k_scale: torch.Tensor | None = None,
    max_logits_bytes: int | None = None,
    out: torch.Tensor | None = None,
    lens_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    del index_k_fp8, index_k_scale
    topk = int(topk)
    _validate_topk(topk)
    if index_k_cache is None or page_size is None:
        raise RuntimeError(
            "Gluon DSA top-k requires packed FP8 index_k_cache and page_size"
        )
    row_bytes = _check_packed_fp8_inputs(q, index_k_cache, weights, int(page_size))
    if kv_workspace_slots.dim() != 1:
        raise ValueError(
            f"kv_workspace_slots must be 1-D, got {tuple(kv_workspace_slots.shape)}"
        )
    if row_starts.shape != (q.shape[0],) or row_ends.shape != (q.shape[0],):
        raise ValueError(
            "row_starts/row_ends must be [tokens], got "
            f"row_starts={tuple(row_starts.shape)}, row_ends={tuple(row_ends.shape)}, "
            f"q={tuple(q.shape)}"
        )
    if out is None:
        out = torch.empty((q.shape[0], topk), dtype=torch.int32, device=q.device)
    if lens_out is None:
        lens_out = torch.empty((q.shape[0],), dtype=torch.int32, device=q.device)
    if q.shape[0] == 0:
        return out, lens_out
    q = _contiguous(q)
    index_k_cache = _contiguous(index_k_cache)
    weights = _contiguous(weights)
    kv_workspace_slots = _to_contiguous(
        kv_workspace_slots, device=q.device, dtype=torch.int64
    )
    row_starts = _to_contiguous(row_starts, device=q.device, dtype=torch.int32)
    row_ends = _to_contiguous(row_ends, device=q.device, dtype=torch.int32)
    seq_len_sum = int(kv_workspace_slots.numel())
    if seq_len_sum == 0:
        out.fill_(-1)
        lens_out.zero_()
        return out, lens_out

    if max_logits_bytes is None:
        max_query_rows = q.shape[0]
    else:
        max_query_rows = max(1, int(max_logits_bytes) // (max(seq_len_sum, 1) * 4))
    block_n = 32
    select_warps = 8
    select_block = triton.next_power_of_2(max(seq_len_sum, topk))
    for start in range(0, q.shape[0], max_query_rows):
        end = min(start + max_query_rows, q.shape[0])
        logits = torch.empty(
            (end - start, seq_len_sum), dtype=torch.float32, device=q.device
        )
        _dsa_prefill_logits_fp8_kernel[
            (end - start, triton.cdiv(seq_len_sum, block_n))
        ](
            q[start:end],
            index_k_cache.view(torch.float8_e4m3fn),
            index_k_cache.view(torch.float32),
            weights[start:end],
            kv_workspace_slots,
            row_starts[start:end],
            row_ends[start:end],
            logits,
            logits.stride(0),
            seq_len_sum=seq_len_sum,
            page_size=int(page_size),
            row_bytes=row_bytes,
            num_heads=q.shape[1],
            head_dim=q.shape[2],
            num_groups=q.shape[2] // 128,
            softmax_scale=float(softmax_scale),
            BLOCK_N=block_n,
            BLOCK_D=128,
            num_warps=4,
        )
        if _use_radix_topk(seq_len_sum):
            _dsa_prefill_radix_topk(
                logits,
                row_starts[start:end],
                row_ends[start:end],
                topk=topk,
                out=out[start:end],
                lens_out=lens_out[start:end],
            )
            continue

        _dsa_prefill_select_topk_kernel[(end - start,)](
            logits,
            row_starts[start:end],
            row_ends[start:end],
            out[start:end],
            lens_out[start:end],
            logits.stride(0),
            out.stride(0),
            topk=topk,
            BLOCK_N=select_block,
            LOAD_ELEMS=_load_elems(select_block, select_warps),
            TOPK_LOAD_ELEMS=_load_elems(topk, select_warps),
            num_warps=select_warps,
        )
    return out, lens_out
