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

"""NVIDIA Gluon scoring and exact bounded selection for QSA streams."""

from __future__ import annotations

import math

import torch
from tokenspeed_kernel._triton import GluonTensorDescriptor, gl, gluon, triton
from tokenspeed_kernel.platform import current_platform


@gluon.jit
def _stream_add(a, b):
    return a + b


@gluon.jit
def _stream_scan(values):
    return gl.associative_scan(values, 1, _stream_add)


@gluon.jit
def _stream_placement_prefix(values):
    """Compact in thread order; the public selection has no output order."""
    group: gl.constexpr = values.shape[0]
    columns: gl.constexpr = values.shape[1]
    threads_per_row: gl.constexpr = 32 * gl.num_warps() // group
    by_thread = gl.reshape(values, (group, columns // threads_per_row, threads_per_row))
    totals = gl.sum(by_thread, 1)
    totals_layout: gl.constexpr = totals.type.layout
    plain: gl.constexpr = gl.BlockedLayout(
        [1, 1], [1, 32], [group, gl.num_warps() // group], [1, 0]
    )
    totals_plain = gl.convert_layout(totals, plain)
    before = gl.convert_layout(_stream_scan(totals_plain) - totals_plain, totals_layout)
    within = gl.associative_scan(by_thread, 1, _stream_add)
    return gl.reshape(before[:, None, :] + within, (group, columns))


@gluon.jit
def _stream_store_candidates(
    destination,
    indices,
    values,
    mask,
    G: gl.constexpr,
    N: gl.constexpr,
    NW: gl.constexpr,
):
    # The candidate buffer has the fixed row-major, unswizzled uint64 layout
    # allocated below. Prefix ranks give each writer a unique destination.
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [G, NW // G], [1, 0])
    row = gl.arange(0, G, layout=gl.SliceLayout(1, layout))
    offsets = row[:, None] * 1024 + gl.convert_layout(indices, layout)
    payload = gl.convert_layout(values, layout)
    predicate = gl.convert_layout(mask.to(gl.int32), layout)
    count: gl.constexpr = G * N // (NW * 32)
    gl.inline_asm(
        _stream_store_asm,
        _stream_store_constraints(count),
        [destination, offsets, payload, predicate],
        result_types=(),
        is_pure=False,
    )


@gluon.constexpr_function
def _stream_store_constraints(count):
    return ",".join(
        ["r"] + ["r"] * count + ["l"] * count + ["r"] * count + ["~{memory}"]
    )


@gluon.constexpr_function
def _stream_store_asm(outputs, inputs):
    base, offsets, payloads, predicates = inputs
    lines = ["{", ".reg .u32 address;", ".reg .pred active;"]
    for offset, value, predicate in zip(offsets, payloads, predicates):
        lines.append(f"mad.lo.u32 address, {offset}, 8, {base[0]};")
        lines.append(f"setp.ne.u32 active, {predicate}, 0;")
        lines.append(f"@active st.shared.b64 [address], {value};")
    lines.append("}")
    return "\n".join(lines)


@gluon.constexpr_function
def _stream_histogram_constraints(count):
    return ",".join(["r"] + ["r"] * count + ["~{memory}"])


@gluon.constexpr_function
def _stream_histogram_asm(outputs, inputs):
    base, offsets = inputs
    lines = ["{", ".reg .u32 address;", ".reg .pred active;"]
    for offset in offsets:
        lines.append(f"setp.ge.s32 active, {offset}, 0;")
        lines.append(f"mad.lo.u32 address, {offset}, 4, {base[0]};")
        lines.append("@active red.relaxed.cta.shared.add.u32 [address], 1;")
    lines.append("}")
    return "\n".join(lines)


@gluon.jit
def _stream_histogram(
    digits, valid, G: gl.constexpr, N: gl.constexpr, NW: gl.constexpr
):
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [G, NW // G], [1, 0])
    hist_layout: gl.constexpr = gl.BlockedLayout(
        [1, 8 if NW == G else 4], [1, 32], [G, NW // G], [1, 0]
    )
    rows = gl.arange(0, G, layout=gl.SliceLayout(1, layout))
    offsets = gl.where(valid, digits + rows[:, None] * 256, -1)
    # The same storage may have held the preceding histogram's reads.
    gl.barrier()
    histogram = gl.allocate_shared_memory(
        gl.int32,
        (G, 256),
        gl.SwizzledSharedLayout(1, 1, 1, [1, 0]),
        gl.full((G, 256), 0, gl.int32, hist_layout),
    )
    gl.barrier()
    count: gl.constexpr = G * N // (NW * 32)
    gl.inline_asm(
        _stream_histogram_asm,
        _stream_histogram_constraints(count),
        (histogram, offsets),
        result_types=(),
        is_pure=False,
    )
    gl.barrier()
    return histogram.load(hist_layout)


@gluon.jit
def _stream_bucket(hist, rank, G: gl.constexpr, NW: gl.constexpr):
    l: gl.constexpr = gl.BlockedLayout(
        [1, 8 if NW == G else 4], [1, 32], [G, NW // G], [1, 0]
    )
    old_l: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [G, NW // G], [1, 0])
    rank = gl.convert_layout(rank, gl.SliceLayout(1, l))
    bins = gl.arange(0, 256, layout=gl.SliceLayout(0, l))
    before = gl.sum(hist, 1)[:, None] - _stream_scan(hist)
    digit = gl.max(
        gl.where(
            (before < rank[:, None]) & (before + hist >= rank[:, None]),
            bins[None, :],
            0,
        ),
        1,
    )
    remaining = rank - gl.sum(gl.where(bins[None, :] > digit[:, None], hist, 0), 1)
    return gl.convert_layout(digit, gl.SliceLayout(1, old_l)), gl.convert_layout(
        remaining, gl.SliceLayout(1, old_l)
    )


@gluon.jit
def _stream_exact_threshold(
    bits, valid, rank, G: gl.constexpr, N: gl.constexpr, NW: gl.constexpr
):
    l: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [G, NW // G], [1, 0])
    active = valid
    cutoff = gl.full((G,), 0, gl.uint32, gl.SliceLayout(1, l))
    remaining = rank
    for shift in gl.static_range(24, -1, -8):
        digits = ((bits >> shift) & 255).to(gl.int32)
        hist = _stream_histogram(digits, active, G, N, NW)
        digit, remaining = _stream_bucket(hist, remaining, G, NW)
        cutoff |= digit.to(gl.uint32) << shift
        active &= digits == digit[:, None]
    return cutoff


@gluon.jit
def _stream_coarse(bits):
    # Positive FP16 encoding is monotone, including saturation to infinity.
    # Use it only for pruning; exact FP32 scores determine final selection.
    return (
        bits.to(gl.float32, bitcast=True).to(gl.float16).to(gl.uint16, bitcast=True)
        >> 6
    ).to(gl.int32)


@gluon.jit
def _stream_prefetch_keys(
    key_cache,
    page_table,
    request,
    start,
    end,
    stride_k_n,
    stride_k_d,
    stride_pt_b,
    destination,
    PAGE_SIZE: gl.constexpr,
    BN: gl.constexpr,
    NW: gl.constexpr,
    USE_INT32_OFFSETS: gl.constexpr,
):
    layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [NW, 1], [1, 0])
    ids = start + gl.arange(0, BN, layout=gl.SliceLayout(1, layout))
    dims = gl.arange(0, 128, layout=gl.SliceLayout(0, layout))
    if USE_INT32_OFFSETS:
        request = request.to(gl.int32)
    pages = gl.load(
        page_table + request * stride_pt_b + ids // PAGE_SIZE, ids < end, 0
    ).to(gl.int32 if USE_INT32_OFFSETS else gl.int64)
    slots = pages * PAGE_SIZE + ids % PAGE_SIZE
    pointers = key_cache + slots[:, None] * stride_k_n + dims[None, :] * stride_k_d
    gl.nvidia.ampere.async_copy.async_load(
        destination, pointers, mask=ids[:, None] < end
    )
    gl.nvidia.ampere.async_copy.commit_group()


@gluon.jit
def _stream_output_pointer(selected):
    # Keep the row-pointer vector out of the scoring loop and partition
    # captures. This uniform identity is evaluated only at the write site.
    bits = gl.inline_asm(
        "mov.u64 $0, $1;",
        "=l,l,~{memory}",
        (selected,),
        result_types=gl.uint64,
        is_pure=False,
    )
    return bits.to(gl.pointer_type(gl.int32))


@gluon.jit
def _qsa_stream_score_partition(
    key_cache,
    page_table,
    request,
    max_count,
    qs,
    ks,
    tm,
    mma_bar,
    score_buffers,
    stride_k_n,
    stride_k_d,
    stride_pt_b,
    PAGE_SIZE: gl.constexpr,
    G: gl.constexpr,
    BN: gl.constexpr,
    NW: gl.constexpr,
    USE_INT32_OFFSETS: gl.constexpr,
    key_desc,
    USE_TMA: gl.constexpr,
):
    if USE_TMA:
        copy_bar = gl.allocate_shared_memory(
            gl.int64, (2, 1), gl.nvidia.hopper.mbarrier.MBarrierLayout()
        )
    for stage in gl.static_range(2):
        gl.nvidia.blackwell.mbarrier.init(mma_bar.index(stage), count=1)
        if USE_TMA:
            gl.nvidia.hopper.mbarrier.init(copy_bar.index(stage), count=1)
            _qsa_tma_prefetch_pages(
                key_desc,
                page_table,
                request,
                stage * BN,
                max_count,
                stride_pt_b,
                ks.index(stage),
                copy_bar.index(stage),
                BN,
                USE_INT32_OFFSETS,
            )
        else:
            _stream_prefetch_keys(
                key_cache,
                page_table,
                request,
                stage * BN,
                max_count,
                stride_k_n,
                stride_k_d,
                stride_pt_b,
                ks.index(stage),
                PAGE_SIZE,
                BN,
                NW,
                USE_INT32_OFFSETS,
            )
    if USE_TMA:
        gl.nvidia.hopper.mbarrier.wait(copy_bar.index(0), 0, deps=(ks.index(0),))
    else:
        gl.nvidia.ampere.async_copy.wait_group(1)
    gl.barrier()
    gl.nvidia.hopper.fence_async_shared()
    gl.nvidia.blackwell.tcgen05_mma(
        ks.index(0), qs, tm.index(0), use_acc=False, mbarriers=[mma_bar.index(0)]
    )
    for start in range(0, max_count, BN):
        slot = (start // BN) & 1
        phase = (start // (2 * BN)) & 1
        gl.nvidia.blackwell.mbarrier.wait(
            mma_bar.index(slot), phase, deps=(ks.index(slot), tm.index(slot))
        )
        matrix = tm.index(slot).load()
        # Store the direct TMEM epilogue view; the swizzle also permits
        # conflict-free reads along keys by each query's selector warp.
        scores = gl.sum(gl.reshape(gl.maximum(matrix, 0), (BN, G, 4)), 2)
        if start + BN < max_count:
            if USE_TMA:
                gl.nvidia.hopper.mbarrier.wait(
                    copy_bar.index(1 - slot),
                    ((start + BN) // (2 * BN)) & 1,
                    deps=(ks.index(1 - slot),),
                )
            else:
                gl.nvidia.ampere.async_copy.wait_group(0)
                gl.barrier()
                gl.nvidia.hopper.fence_async_shared()
            gl.nvidia.blackwell.tcgen05_mma(
                ks.index(1 - slot),
                qs,
                tm.index(1 - slot),
                use_acc=False,
                mbarriers=[mma_bar.index(1 - slot)],
            )
            if start + 2 * BN < max_count:
                if USE_TMA:
                    _qsa_tma_prefetch_pages(
                        key_desc,
                        page_table,
                        request,
                        start + 2 * BN,
                        max_count,
                        stride_pt_b,
                        ks.index(slot),
                        copy_bar.index(slot),
                        BN,
                        USE_INT32_OFFSETS,
                    )
                else:
                    _stream_prefetch_keys(
                        key_cache,
                        page_table,
                        request,
                        start + 2 * BN,
                        max_count,
                        stride_k_n,
                        stride_k_d,
                        stride_pt_b,
                        ks.index(slot),
                        PAGE_SIZE,
                        BN,
                        NW,
                        USE_INT32_OFFSETS,
                    )
        if start >= 2 * BN:
            _qsa_named_score_barrier(5 + slot, (G + NW) * 32, False)
        score_buffers.index(slot).store(scores)
        gl.barrier()
        _qsa_named_score_barrier(3 + slot, (G + NW) * 32, True)
    # Drain the final readers before the CTA starts another request owner.
    for stage in gl.static_range(2):
        if stage * BN < max_count:
            _qsa_named_score_barrier(5 + stage, (G + NW) * 32, False)
    if not USE_TMA:
        gl.nvidia.ampere.async_copy.wait_group(0)
    gl.barrier()
    for stage in gl.static_range(2):
        gl.nvidia.blackwell.mbarrier.invalidate(mma_bar.index(stage))
        if USE_TMA:
            gl.nvidia.hopper.mbarrier.invalidate(copy_bar.index(stage))


@gluon.jit
def _qsa_stream_select_partition(
    rows,
    owned,
    valid_count,
    max_count,
    selected,
    score_buffers,
    G: gl.constexpr,
    BN: gl.constexpr,
    NW: gl.constexpr,
):
    l: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [G, NW // G], [1, 0])
    rl: gl.constexpr = gl.SliceLayout(1, l)
    cl: gl.constexpr = gl.SliceLayout(0, l)
    running = gl.allocate_shared_memory(
        gl.uint64, (G, 1024), gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    )
    running_count = gl.full((G,), 0, gl.int32, rl)
    coarse_cutoff = gl.full((G,), 0, gl.int32, rl)
    ns = gl.arange(0, BN, layout=cl)
    all_offsets = gl.arange(0, 1024, layout=cl)
    outcols = gl.arange(0, 512, layout=cl)
    for start in range(0, max_count, BN):
        slot = (start // BN) & 1
        _qsa_named_score_barrier(3 + slot, (NW + (4 if G == 16 else NW)) * 32, False)
        scores = score_buffers.index(slot).permute((1, 0)).load(l)
        # Every selector finishes reading before the producer reuses this slot.
        gl.barrier()
        _qsa_named_score_barrier(5 + slot, (NW + (4 if G == 16 else NW)) * 32, True)
        block_ids = start + ns
        bits = scores.to(gl.uint32, bitcast=True)
        bucket = _stream_prefill_coarse(bits)
        eligible = (
            owned[:, None]
            & (block_ids[None, :] < valid_count[:, None])
            & (bucket >= coarse_cutoff[:, None])
        )
        added = gl.sum(eligible.to(gl.int32), 1)
        dest = (
            running_count[:, None] + _stream_placement_prefix(eligible.to(gl.int32)) - 1
        )
        packed = ((bits.to(gl.uint64) + 1) << 32) | block_ids[None, :].to(gl.uint64)
        _stream_store_candidates(
            running, gl.maximum(dest, 0), packed, eligible, G, BN, NW
        )
        running_count += added
        if gl.max(running_count, 0) > 768:
            # Complete appends before reading their candidate values.
            gl.barrier()
            values = running.load(l)
            valid = all_offsets[None, :] < running_count[:, None]
            encoded = (values >> 32).to(gl.uint32)
            buckets = _stream_prefill_coarse(encoded - 1)
            base = gl.maximum(gl.max(gl.where(valid, buckets, 0), 1) - 255, 0)
            digits = gl.maximum(buckets - base[:, None], 0)
            hist = _stream_histogram(digits, valid, G, 1024, NW)
            rank = gl.minimum(running_count, 512)
            digit, _ = _stream_bucket(hist, rank, G, NW)
            # The first bin includes all lower clamped values. Keep
            # the entire bin if it contains the kth score.
            cutoff = gl.where(digit > 0, digit + base, 0)
            keep = valid & (buckets >= cutoff[:, None])
            kept_count = gl.sum(keep.to(gl.int32), 1)
            if gl.max(kept_count, 0) > 768:
                exact = _stream_exact_threshold(encoded, valid, rank, G, 1024, NW)
                greater = valid & (encoded > exact[:, None])
                equal = valid & (encoded == exact[:, None])
                keep = greater | (
                    equal
                    & (
                        _stream_placement_prefix(equal.to(gl.int32))
                        <= (rank - gl.sum(greater.to(gl.int32), 1))[:, None]
                    )
                )
                kept_count = gl.sum(keep.to(gl.int32), 1)
            destinations = _stream_placement_prefix(keep.to(gl.int32)) - 1
            gl.barrier()
            _stream_store_candidates(
                running, gl.maximum(destinations, 0), values, keep, G, 1024, NW
            )
            # The next append starts at kept_count, beyond these writes.
            running_count = kept_count
            coarse_cutoff = cutoff
    # The final append may not have triggered a compaction read.
    gl.barrier()
    values = running.load(l)
    valid = all_offsets[None, :] < running_count[:, None]
    encoded = (values >> 32).to(gl.uint32)
    rank = gl.minimum(running_count, 512)
    exact = _stream_exact_threshold(encoded, valid, rank, G, 1024, NW)
    greater = valid & (encoded > exact[:, None])
    equal = valid & (encoded == exact[:, None])
    ngreater = gl.sum(greater.to(gl.int32), 1)
    # Each row has at most 1024 candidates, so two 16-bit counts
    # can share one scan without a carry between them.
    ranks = _stream_placement_prefix(greater.to(gl.int32) | (equal.to(gl.int32) << 16))
    dest = gl.where(
        greater, (ranks & 0xFFFF) - 1, ngreater[:, None] + (ranks >> 16) - 1
    )
    output_base = _stream_output_pointer(selected)
    gl.store(output_base + rows[:, None] * 512 + outcols[None, :], -1, owned[:, None])
    gl.barrier()
    gl.store(
        output_base + rows[:, None] * 512 + gl.maximum(dest, 0),
        (values & 0xFFFFFFFF).to(gl.int32),
        owned[:, None] & (greater | equal) & (dest < 512),
    )


@gluon.constexpr_function
def _soa_store_constraints(count, bases):
    return ",".join(["r"] * bases + ["r"] * count + ["r"] * count + ["~{memory}"])


@gluon.constexpr_function
def _soa_score_store_asm(outputs, inputs):
    base, offsets, values = inputs
    lines = ["{", ".reg .u32 address;", ".reg .pred active;"]
    for offset, value in zip(offsets, values):
        lines.extend(
            [
                f"setp.ge.s32 active, {offset}, 0;",
                f"mad.lo.u32 address, {offset}, 4, {base[0]};",
                f"@active st.shared.u32 [address], {value};",
            ]
        )
    lines.append("}")
    return "\n".join(lines)


@gluon.constexpr_function
def _soa_id_store_asm(outputs, inputs):
    low_base, high_base, offsets, values = inputs
    lines = [
        "{",
        ".reg .u32 low_address, high_address, upper;",
        ".reg .u16 low_value, high_value;",
        ".reg .pred active;",
    ]
    for offset, value in zip(offsets, values):
        lines.extend(
            [
                f"setp.ge.s32 active, {offset}, 0;",
                f"mad.lo.u32 low_address, {offset}, 2, {low_base[0]};",
                f"add.u32 high_address, {offset}, {high_base[0]};",
                f"cvt.u16.u32 low_value, {value};",
                f"shr.u32 upper, {value}, 16;",
                "cvt.u16.u32 high_value, upper;",
                "@active st.shared.u16 [low_address], low_value;",
                "@active st.shared.u8 [high_address], high_value;",
            ]
        )
    lines.append("}")
    return "\n".join(lines)


@gluon.jit
def _soa_store_scores(
    destination, offsets, values, G: gl.constexpr, N: gl.constexpr, NW: gl.constexpr
):
    count: gl.constexpr = G * N // (NW * 32)
    gl.inline_asm(
        _soa_score_store_asm,
        _soa_store_constraints(count, 1),
        (destination, offsets, values),
        result_types=(),
        is_pure=False,
    )


@gluon.jit
def _soa_store_ids(
    low,
    high,
    offsets,
    values,
    G: gl.constexpr,
    N: gl.constexpr,
    NW: gl.constexpr,
    ID_BITS: gl.constexpr,
):
    count: gl.constexpr = G * N // (NW * 32)
    if ID_BITS == 16:
        gl.inline_asm(
            _soa_low_store_asm,
            _soa_store_constraints(count, 1),
            (low, offsets, values),
            result_types=(),
            is_pure=False,
        )
    else:
        gl.inline_asm(
            _soa_id_store_asm,
            _soa_store_constraints(count, 2),
            (low, high, offsets, values),
            result_types=(),
            is_pure=False,
        )


@gluon.constexpr_function
def _soa_id_load_constraints(count):
    return ",".join(["=r"] * count + ["r", "r", "r", "~{memory}"])


@gluon.constexpr_function
def _soa_id_load_asm(outputs, inputs):
    (result,) = outputs
    low_base, high_base, row = inputs
    lines = [
        "{",
        ".reg .u32 lane, offset, low_address, high_address, lower, upper;",
        ".reg .u16 low_value, high_value;",
        "mov.u32 lane, %laneid;",
        f"mad.lo.u32 offset, {row[0]}, 1024, lane;",
        f"mad.lo.u32 low_address, offset, 2, {low_base[0]};",
        f"add.u32 high_address, offset, {high_base[0]};",
    ]
    for index, value in enumerate(result):
        lines.extend(
            [
                f"ld.shared.u16 low_value, [low_address+{index * 64}];",
                f"ld.shared.u8 high_value, [high_address+{index * 32}];",
                "cvt.u32.u16 lower, low_value;",
                "cvt.u32.u16 upper, high_value;",
                f"mad.lo.u32 {value}, upper, 65536, lower;",
            ]
        )
    lines.append("}")
    return "\n".join(lines)


@gluon.jit
def _soa_load_ids(
    low, high, G: gl.constexpr, N: gl.constexpr, NW: gl.constexpr, ID_BITS: gl.constexpr
):
    gl.static_assert(NW == G)
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [G, 1], [1, 0])
    if ID_BITS == 16:
        return low.load(layout).to(gl.uint32)
    rows = gl.arange(0, G, layout=gl.SliceLayout(1, layout))
    count: gl.constexpr = N // 32
    return gl.inline_asm(
        _soa_id_load_asm,
        _soa_id_load_constraints(count),
        (low, high, rows),
        result_types=gl.distributed_type(gl.uint32, (G, N), layout),
        is_pure=False,
    )


@gluon.jit
def _qsa_soa_select_partition(
    rows,
    owned,
    valid_count,
    max_count,
    selected,
    score_buffers,
    G: gl.constexpr,
    BN: gl.constexpr,
    NW: gl.constexpr,
    ID_BITS: gl.constexpr,
):
    gl.static_assert(G == NW)
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [G, 1], [1, 0])
    row_layout: gl.constexpr = gl.SliceLayout(1, layout)
    col_layout: gl.constexpr = gl.SliceLayout(0, layout)
    shared_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    running_scores = gl.allocate_shared_memory(gl.uint32, (G, 1024), shared_layout)
    running_low = gl.allocate_shared_memory(gl.uint16, (G, 1024), shared_layout)
    if ID_BITS == 16:
        running_high = running_low
    else:
        running_high = gl.allocate_shared_memory(gl.uint8, (G, 1024), shared_layout)
    query_offsets = gl.arange(0, G, layout=row_layout)
    all_columns = gl.arange(0, 1024, layout=col_layout)
    new_columns = gl.arange(0, BN, layout=col_layout)
    output_columns = gl.arange(0, 512, layout=col_layout)
    running_count = gl.full((G,), 0, gl.int32, row_layout)
    coarse_cutoff = gl.full((G,), 0, gl.int32, row_layout)
    for start in range(0, max_count, BN):
        slot = (start // BN) & 1
        _qsa_named_score_barrier(3 + slot, (NW + (4 if G == 16 else NW)) * 32, False)
        scores = score_buffers.index(slot).permute((1, 0)).load(layout)
        gl.barrier()
        _qsa_named_score_barrier(5 + slot, (NW + (4 if G == 16 else NW)) * 32, True)
        bits = scores.to(gl.uint32, bitcast=True)
        new_ids = start + new_columns
        eligible = (
            owned[:, None]
            & (new_ids[None, :] < valid_count[:, None])
            & (_stream_prefill_coarse(bits) >= coarse_cutoff[:, None])
        )
        added = gl.sum(eligible.to(gl.int32), 1)
        new_destinations = (
            running_count[:, None] + _stream_placement_prefix(eligible.to(gl.int32)) - 1
        )
        new_flattened = gl.where(
            eligible, query_offsets[:, None] * 1024 + new_destinations, -1
        )
        _soa_store_scores(running_scores, new_flattened, bits + 1, G, BN, NW)
        _soa_store_ids(
            running_low,
            running_high,
            new_flattened,
            (new_ids[None, :] + gl.full((G, BN), 0, gl.int32, layout)).to(gl.uint32),
            G,
            BN,
            NW,
            ID_BITS,
        )
        running_count += added
        if gl.max(running_count, 0) > 1024 - BN:
            gl.barrier()
            encoded = running_scores.load(layout)
            valid = all_columns[None, :] < running_count[:, None]
            buckets = _stream_prefill_coarse(encoded - 1)
            base = gl.maximum(gl.max(gl.where(valid, buckets, 0), 1) - 255, 0)
            hist = _stream_histogram(
                gl.maximum(buckets - base[:, None], 0), valid, G, 1024, NW
            )
            rank = gl.minimum(running_count, 512)
            digit, _ = _stream_bucket(hist, rank, G, NW)
            cutoff = gl.where(digit > 0, digit + base, 0)
            keep = valid & (buckets >= cutoff[:, None])
            kept_count = gl.sum(keep.to(gl.int32), 1)
            if gl.max(kept_count, 0) > 1024 - BN:
                exact = _stream_exact_threshold(encoded, valid, rank, G, 1024, NW)
                greater = valid & (encoded > exact[:, None])
                equal = valid & (encoded == exact[:, None])
                keep = greater | (
                    equal
                    & (
                        _stream_placement_prefix(equal.to(gl.int32))
                        <= (rank - gl.sum(greater.to(gl.int32), 1))[:, None]
                    )
                )
                kept_count = gl.sum(keep.to(gl.int32), 1)
            destinations = _stream_placement_prefix(keep.to(gl.int32)) - 1
            flattened = gl.where(keep, query_offsets[:, None] * 1024 + destinations, -1)
            # Read every old score before overwriting the score plane.
            gl.barrier()
            _soa_store_scores(running_scores, flattened, encoded, G, 1024, NW)
            # Score writes do not touch the two ID planes, so IDs load late.
            compact_ids = _soa_load_ids(running_low, running_high, G, 1024, NW, ID_BITS)
            gl.barrier()
            _soa_store_ids(
                running_low, running_high, flattened, compact_ids, G, 1024, NW, ID_BITS
            )
            running_count = kept_count
            coarse_cutoff = cutoff
    gl.barrier()
    encoded = running_scores.load(layout)
    valid = all_columns[None, :] < running_count[:, None]
    rank = gl.minimum(running_count, 512)
    threshold = _stream_exact_threshold(encoded, valid, rank, G, 1024, NW)
    greater = valid & (encoded > threshold[:, None])
    equal = valid & (encoded == threshold[:, None])
    ngreater = gl.sum(greater.to(gl.int32), 1)
    prefix = _stream_placement_prefix(greater.to(gl.int32) | (equal.to(gl.int32) << 16))
    destinations = gl.where(
        greater, (prefix & 0xFFFF) - 1, ngreater[:, None] + (prefix >> 16) - 1
    )
    chosen = owned[:, None] & (greater | equal) & (destinations < 512)
    # Scores are dead before the ID load; every score bit remains FP32 exact.
    ids = _soa_load_ids(running_low, running_high, G, 1024, NW, ID_BITS)
    output = _stream_output_pointer(selected) + rows[:, None] * 512
    gl.store(output + output_columns[None, :], -1, owned[:, None])
    gl.barrier()
    gl.store(output + gl.maximum(destinations, 0), ids.to(gl.int32), chosen)


@gluon.jit(
    do_not_specialize=[
        "num_rows",
        "num_blocks",
        "stride_q_n",
        "stride_q_h",
        "stride_pt_b",
    ]
)
def _qsa_soa_stream_kernel(
    query,
    key_cache,
    page_table,
    requests,
    complete,
    selected,
    num_rows,
    num_blocks,
    page_size: gl.constexpr,
    stride_q_n,
    stride_q_h,
    stride_q_d,
    stride_k_n,
    stride_k_d,
    stride_pt_b,
    G: gl.constexpr,
    BN: gl.constexpr,
    BD: gl.constexpr,
    NW: gl.constexpr,
    RAGGED: gl.constexpr,
    ENABLE_PDL: gl.constexpr,
    TCGEN: gl.constexpr,
    USE_INT32_OFFSETS: gl.constexpr,
    key_desc,
    USE_TMA: gl.constexpr,
    ID_BITS: gl.constexpr,
):
    l: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [G, NW // G], [1, 0])
    rl: gl.constexpr = gl.SliceLayout(1, l)
    cl: gl.constexpr = gl.SliceLayout(0, l)
    if ENABLE_PDL:
        gl.inline_asm_elementwise(
            "griddepcontrol.wait; mov.u32 $0, 0;",
            "=r",
            [],
            dtype=gl.int32,
            is_pure=False,
            pack=1,
        )
        gl.inline_asm_elementwise(
            "griddepcontrol.launch_dependents; mov.u32 $0, 0;",
            "=r",
            [],
            dtype=gl.int32,
            is_pure=False,
            pack=1,
        )
    rows = gl.program_id(0) * G + gl.arange(0, G, layout=rl)
    row_valid = rows < num_rows
    reqs = gl.load(requests + rows, row_valid, 0).to(gl.int64)
    counts = gl.minimum(
        gl.maximum(gl.load(complete + rows, row_valid, 0), 0), num_blocks
    )
    remaining_rows = row_valid
    # A boundary CTA handles each request separately, sharing K only between
    # matching requests. The common case is one pass through this loop.
    while gl.sum(remaining_rows.to(gl.int32), 0) > 0:
        first = gl.min(gl.where(remaining_rows, rows, 0x7FFFFFFF), 0)
        req = gl.sum(gl.where(rows == first, reqs, 0), 0)
        owned = remaining_rows & (reqs == req)
        valid_count = gl.where(owned, counts, 0)
        max_count = gl.max(valid_count, 0)
        if max_count <= 512:
            outcols = gl.arange(0, 512, layout=cl)
            output = (
                _stream_output_pointer(selected)
                + rows[:, None] * 512
                + outcols[None, :]
            )
            gl.store(
                output,
                gl.where(outcols[None, :] < valid_count[:, None], outcols[None, :], -1),
                owned[:, None],
            )
        else:
            tql: gl.constexpr = gl.BlockedLayout([8, 1], [16, 2], [1, NW], [0, 1])
            tdq = gl.arange(0, 128, layout=gl.SliceLayout(1, tql))
            thq = gl.arange(0, G * 4, layout=gl.SliceLayout(0, tql))
            owned_tc = gl.gather(
                gl.convert_layout(owned, gl.SliceLayout(0, tql)), thq // 4, 0
            )
            qvalue = gl.load(
                query
                + (gl.program_id(0) * G + thq[None, :] // 4) * stride_q_n
                + (thq[None, :] % 4) * stride_q_h
                + tdq[:, None] * stride_q_d,
                owned_tc[None, :],
                0,
            )
            qs = gl.allocate_shared_memory(
                gl.bfloat16,
                (128, G * 4),
                gl.NVMMASharedLayout(128, 16, transposed=True),
                qvalue,
            )
            ks = gl.allocate_shared_memory(
                gl.bfloat16, (2, BN, 128), gl.NVMMASharedLayout(128, 16)
            )
            tm = gl.nvidia.blackwell.allocate_tensor_memory(
                gl.float32,
                (2, BN, G * 4),
                gl.nvidia.blackwell.TensorMemoryLayout((128, G * 4), 1),
            )
            mma_bar = gl.allocate_shared_memory(
                gl.int64, (2, 1), gl.nvidia.blackwell.mbarrier.MBarrierLayout()
            )
            score_buffers = gl.allocate_shared_memory(
                gl.float32, (2, BN, G), gl.SwizzledSharedLayout(1, 32 // G, G, [1, 0])
            )
            gl.barrier()
            gl.warp_specialize(
                [
                    (
                        _qsa_soa_select_partition,
                        (
                            rows,
                            owned,
                            valid_count,
                            max_count,
                            selected,
                            score_buffers,
                            G,
                            BN,
                            NW,
                            ID_BITS,
                        ),
                    ),
                    (
                        _qsa_stream_score_partition,
                        (
                            key_cache,
                            page_table,
                            req,
                            max_count,
                            qs,
                            ks,
                            tm,
                            mma_bar,
                            score_buffers,
                            stride_k_n,
                            stride_k_d,
                            stride_pt_b,
                            page_size,
                            G,
                            BN,
                            4,
                            USE_INT32_OFFSETS,
                            key_desc,
                            USE_TMA,
                        ),
                    ),
                ],
                [4],
                [96],
            )
        remaining_rows &= ~owned


@gluon.jit(
    do_not_specialize=[
        "num_rows",
        "num_blocks",
        "stride_q_n",
        "stride_q_h",
        "stride_pt_b",
    ]
)
def _qsa_warp_stream_kernel(
    query,
    key_cache,
    page_table,
    requests,
    complete,
    selected,
    num_rows,
    num_blocks,
    page_size: gl.constexpr,
    stride_q_n,
    stride_q_h,
    stride_q_d,
    stride_k_n,
    stride_k_d,
    stride_pt_b,
    G: gl.constexpr,
    BN: gl.constexpr,
    BD: gl.constexpr,
    NW: gl.constexpr,
    RAGGED: gl.constexpr,
    ENABLE_PDL: gl.constexpr,
    TCGEN: gl.constexpr,
    USE_INT32_OFFSETS: gl.constexpr,
    key_desc,
    USE_TMA: gl.constexpr,
):
    l: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [G, NW // G], [1, 0])
    rl: gl.constexpr = gl.SliceLayout(1, l)
    cl: gl.constexpr = gl.SliceLayout(0, l)
    if ENABLE_PDL:
        gl.inline_asm_elementwise(
            "griddepcontrol.wait; mov.u32 $0, 0;",
            "=r",
            [],
            dtype=gl.int32,
            is_pure=False,
            pack=1,
        )
        gl.inline_asm_elementwise(
            "griddepcontrol.launch_dependents; mov.u32 $0, 0;",
            "=r",
            [],
            dtype=gl.int32,
            is_pure=False,
            pack=1,
        )
    rows = gl.program_id(0) * G + gl.arange(0, G, layout=rl)
    row_valid = rows < num_rows
    reqs = gl.load(requests + rows, row_valid, 0).to(gl.int64)
    counts = gl.minimum(
        gl.maximum(gl.load(complete + rows, row_valid, 0), 0), num_blocks
    )
    remaining_rows = row_valid
    # A boundary CTA handles each request separately, sharing K only between
    # matching requests. The common case is one pass through this loop.
    while gl.sum(remaining_rows.to(gl.int32), 0) > 0:
        first = gl.min(gl.where(remaining_rows, rows, 0x7FFFFFFF), 0)
        req = gl.sum(gl.where(rows == first, reqs, 0), 0)
        owned = remaining_rows & (reqs == req)
        valid_count = gl.where(owned, counts, 0)
        max_count = gl.max(valid_count, 0)
        if max_count <= 512:
            outcols = gl.arange(0, 512, layout=cl)
            output = (
                _stream_output_pointer(selected)
                + rows[:, None] * 512
                + outcols[None, :]
            )
            gl.store(
                output,
                gl.where(outcols[None, :] < valid_count[:, None], outcols[None, :], -1),
                owned[:, None],
            )
        else:
            tql: gl.constexpr = gl.BlockedLayout([8, 1], [16, 2], [1, NW], [0, 1])
            tdq = gl.arange(0, 128, layout=gl.SliceLayout(1, tql))
            thq = gl.arange(0, G * 4, layout=gl.SliceLayout(0, tql))
            owned_tc = gl.gather(
                gl.convert_layout(owned, gl.SliceLayout(0, tql)), thq // 4, 0
            )
            qvalue = gl.load(
                query
                + (gl.program_id(0) * G + thq[None, :] // 4) * stride_q_n
                + (thq[None, :] % 4) * stride_q_h
                + tdq[:, None] * stride_q_d,
                owned_tc[None, :],
                0,
            )
            qs = gl.allocate_shared_memory(
                gl.bfloat16,
                (128, G * 4),
                gl.NVMMASharedLayout(128, 16, transposed=True),
                qvalue,
            )
            ks = gl.allocate_shared_memory(
                gl.bfloat16, (2, BN, 128), gl.NVMMASharedLayout(128, 16)
            )
            tm = gl.nvidia.blackwell.allocate_tensor_memory(
                gl.float32,
                (2, BN, G * 4),
                gl.nvidia.blackwell.TensorMemoryLayout((128, G * 4), 1),
            )
            mma_bar = gl.allocate_shared_memory(
                gl.int64, (2, 1), gl.nvidia.blackwell.mbarrier.MBarrierLayout()
            )
            score_buffers = gl.allocate_shared_memory(
                gl.float32, (2, BN, G), gl.SwizzledSharedLayout(1, 32 // G, G, [1, 0])
            )
            gl.barrier()
            gl.warp_specialize(
                [
                    (
                        _qsa_stream_select_partition,
                        (
                            rows,
                            owned,
                            valid_count,
                            max_count,
                            selected,
                            score_buffers,
                            G,
                            BN,
                            NW,
                        ),
                    ),
                    (
                        _qsa_stream_score_partition,
                        (
                            key_cache,
                            page_table,
                            req,
                            max_count,
                            qs,
                            ks,
                            tm,
                            mma_bar,
                            score_buffers,
                            stride_k_n,
                            stride_k_d,
                            stride_pt_b,
                            page_size,
                            G,
                            BN,
                            NW,
                            USE_INT32_OFFSETS,
                            key_desc,
                            USE_TMA,
                        ),
                    ),
                ],
                [NW],
                [96],
            )
        remaining_rows &= ~owned


@gluon.jit(
    do_not_specialize=[
        "num_rows",
        "num_blocks",
        "stride_q_n",
        "stride_q_h",
        "stride_pt_b",
    ]
)
def _qsa_bounded_stream_kernel(
    query,
    key_cache,
    page_table,
    requests,
    complete,
    selected,
    num_rows,
    num_blocks,
    page_size: gl.constexpr,
    stride_q_n,
    stride_q_h,
    stride_q_d,
    stride_k_n,
    stride_k_d,
    stride_pt_b,
    G: gl.constexpr,
    BN: gl.constexpr,
    BD: gl.constexpr,
    NW: gl.constexpr,
    RAGGED: gl.constexpr,
    ENABLE_PDL: gl.constexpr,
    TCGEN: gl.constexpr,
    USE_INT32_OFFSETS: gl.constexpr,
):
    l: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [G, NW // G], [1, 0])
    rl: gl.constexpr = gl.SliceLayout(1, l)
    cl: gl.constexpr = gl.SliceLayout(0, l)
    if ENABLE_PDL:
        gl.inline_asm_elementwise(
            "griddepcontrol.wait; mov.u32 $0, 0;",
            "=r",
            [],
            dtype=gl.int32,
            is_pure=False,
            pack=1,
        )
        gl.inline_asm_elementwise(
            "griddepcontrol.launch_dependents; mov.u32 $0, 0;",
            "=r",
            [],
            dtype=gl.int32,
            is_pure=False,
            pack=1,
        )
    rows = gl.program_id(0) * G + gl.arange(0, G, layout=rl)
    row_valid = rows < num_rows
    reqs = gl.load(requests + rows, row_valid, 0).to(gl.int64)
    counts = gl.minimum(
        gl.maximum(gl.load(complete + rows, row_valid, 0), 0), num_blocks
    )
    remaining_rows = row_valid
    # A boundary CTA handles each request separately, sharing K only between
    # matching requests. The common case is one pass through this loop.
    while gl.sum(remaining_rows.to(gl.int32), 0) > 0:
        first = gl.min(gl.where(remaining_rows, rows, 0x7FFFFFFF), 0)
        req = gl.sum(gl.where(rows == first, reqs, 0), 0)
        owned = remaining_rows & (reqs == req)
        valid_count = gl.where(owned, counts, 0)
        max_count = gl.max(valid_count, 0)
        outcols = gl.arange(0, 512, layout=cl)
        output = selected + rows[:, None] * 512 + outcols[None, :]
        if max_count <= 512:
            gl.store(
                output,
                gl.where(outcols[None, :] < valid_count[:, None], outcols[None, :], -1),
                owned[:, None],
            )
        else:
            running = gl.allocate_shared_memory(
                gl.uint64, (G, 1024), gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
            )
            running_count = gl.full((G,), 0, gl.int32, rl)
            coarse_cutoff = gl.full((G,), 0, gl.int32, rl)
            ns = gl.arange(0, BN, layout=cl)
            all_offsets = gl.arange(0, 1024, layout=cl)
            mma_l: gl.constexpr = gl.NVMMADistributedLayout(
                version=[2, 0], warps_per_cta=[1, NW], instr_shape=[16, 8]
            )
            qm: gl.constexpr = max(16, G * 4)
            ql: gl.constexpr = gl.BlockedLayout([1, 4], [2, 16], [NW, 1], [1, 0])
            kl: gl.constexpr = gl.BlockedLayout([8, 1], [16, 2], [1, NW], [0, 1])
            hi = gl.arange(0, qm, layout=gl.SliceLayout(1, ql))
            diq = gl.arange(0, BD, layout=gl.SliceLayout(0, ql))
            dik = gl.arange(0, BD, layout=gl.SliceLayout(1, kl))
            nk = gl.arange(0, BN, layout=gl.SliceLayout(0, kl))
            # Reuse the ownership mask in the matrix view without reading
            # another row or creating a new request-dependent specialization.
            head_owned = gl.gather(owned, hi // 4 % G, 0)
            if TCGEN:
                tql: gl.constexpr = gl.BlockedLayout([8, 1], [16, 2], [1, NW], [0, 1])
                tkl: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [NW, 1], [1, 0])
                tdq = gl.arange(0, 128, layout=gl.SliceLayout(1, tql))
                thq = gl.arange(0, G * 4, layout=gl.SliceLayout(0, tql))
                owned_tc = gl.gather(
                    gl.convert_layout(owned, gl.SliceLayout(0, tql)), thq // 4, 0
                )
                qvalue = gl.load(
                    query
                    + (gl.program_id(0) * G + thq[None, :] // 4) * stride_q_n
                    + (thq[None, :] % 4) * stride_q_h
                    + tdq[:, None] * stride_q_d,
                    owned_tc[None, :],
                    0,
                )
                qs = gl.allocate_shared_memory(
                    gl.bfloat16,
                    (128, G * 4),
                    gl.NVMMASharedLayout(128, 16, transposed=True),
                    qvalue,
                )
                ks = gl.allocate_shared_memory(
                    gl.bfloat16, (2, BN, 128), gl.NVMMASharedLayout(128, 16)
                )
                tm = gl.nvidia.blackwell.allocate_tensor_memory(
                    gl.float32,
                    (2, BN, G * 4),
                    gl.nvidia.blackwell.TensorMemoryLayout((128, G * 4), 1),
                )
                mma_bar = gl.allocate_shared_memory(
                    gl.int64, (2, 1), gl.nvidia.blackwell.mbarrier.MBarrierLayout()
                )
                for stage in gl.static_range(2):
                    gl.nvidia.blackwell.mbarrier.init(mma_bar.index(stage), count=1)
                    _stream_prefetch_keys(
                        key_cache,
                        page_table,
                        req,
                        stage * BN,
                        max_count,
                        stride_k_n,
                        stride_k_d,
                        stride_pt_b,
                        ks.index(stage),
                        page_size,
                        BN,
                        NW,
                        USE_INT32_OFFSETS,
                    )
                # Wait for the first copy while allowing the second to stay
                # in flight. Each stage is reused only after its MMA completes.
                gl.nvidia.ampere.async_copy.wait_group(1)
                gl.barrier()
                gl.nvidia.hopper.fence_async_shared()
                gl.nvidia.blackwell.tcgen05_mma(
                    ks.index(0),
                    qs,
                    tm.index(0),
                    use_acc=False,
                    mbarriers=[mma_bar.index(0)],
                )
            for start in range(0, max_count, BN):
                if TCGEN:
                    slot = (start // BN) & 1
                    phase = (start // (2 * BN)) & 1
                    gl.nvidia.blackwell.mbarrier.wait(
                        mma_bar.index(slot),
                        phase,
                        deps=(ks.index(slot), tm.index(slot)),
                    )
                    matrix = tm.index(slot).load()
                    scores = gl.permute(
                        gl.sum(gl.reshape(gl.maximum(matrix, 0), (BN, G, 4)), 2), (1, 0)
                    )
                    if start + BN < max_count:
                        gl.nvidia.ampere.async_copy.wait_group(0)
                        gl.barrier()
                        gl.nvidia.hopper.fence_async_shared()
                        gl.nvidia.blackwell.tcgen05_mma(
                            ks.index(1 - slot),
                            qs,
                            tm.index(1 - slot),
                            use_acc=False,
                            mbarriers=[mma_bar.index(1 - slot)],
                        )
                        if start + 2 * BN < max_count:
                            _stream_prefetch_keys(
                                key_cache,
                                page_table,
                                req,
                                start + 2 * BN,
                                max_count,
                                stride_k_n,
                                stride_k_d,
                                stride_pt_b,
                                ks.index(slot),
                                page_size,
                                BN,
                                NW,
                                USE_INT32_OFFSETS,
                            )
                else:
                    ids = start + nk
                    pages = gl.load(
                        page_table + req * stride_pt_b + ids // page_size,
                        ids < max_count,
                        0,
                    ).to(gl.int64)
                    slots = pages * page_size + ids % page_size
                    acc = gl.full((qm, BN), 0, gl.float32, mma_l)
                    for ds in range(0, 128, BD):
                        q = gl.load(
                            query
                            + (gl.program_id(0) * G + hi[:, None] // 4) * stride_q_n
                            + (hi[:, None] % 4) * stride_q_h
                            + (ds + diq[None, :]) * stride_q_d,
                            (hi[:, None] < G * 4) & head_owned[:, None],
                            0,
                        )
                        k = gl.load(
                            key_cache
                            + slots[None, :] * stride_k_n
                            + (ds + dik[:, None]) * stride_k_d,
                            ids[None, :] < max_count,
                            0,
                        )
                        a = gl.convert_layout(q, gl.DotOperandLayout(0, mma_l, 2))
                        b = gl.convert_layout(k, gl.DotOperandLayout(1, mma_l, 2))
                        acc = gl.nvidia.ampere.mma_v2(a, b, acc, input_precision="ieee")
                    if G == 1:
                        hmask = gl.arange(0, qm, layout=gl.SliceLayout(1, mma_l)) < 4
                        acc = gl.where(hmask[:, None], gl.maximum(acc, 0), 0)
                        scores = gl.sum(acc, 0)[None, :]
                    else:
                        scores = gl.sum(gl.reshape(gl.maximum(acc, 0), (G, 4, BN)), 1)
                scores = gl.convert_layout(scores, l)
                block_ids = start + ns
                bits = scores.to(gl.uint32, bitcast=True)
                bucket = _stream_coarse(bits)
                eligible = (
                    owned[:, None]
                    & (block_ids[None, :] < valid_count[:, None])
                    & (bucket >= coarse_cutoff[:, None])
                )
                added = gl.sum(eligible.to(gl.int32), 1)
                dest = (
                    running_count[:, None]
                    + _stream_placement_prefix(eligible.to(gl.int32))
                    - 1
                )
                packed = ((bits.to(gl.uint64) + 1) << 32) | block_ids[None, :].to(
                    gl.uint64
                )
                running.atomic_scatter_xchg(
                    packed, gl.maximum(dest, 0), 1, mask=eligible
                )
                gl.barrier()
                running_count += added
                if gl.max(running_count, 0) > 768:
                    values = running.load(l)
                    valid = all_offsets[None, :] < running_count[:, None]
                    encoded = (values >> 32).to(gl.uint32)
                    buckets = _stream_coarse(encoded - 1)
                    base = gl.maximum(gl.max(gl.where(valid, buckets, 0), 1) - 255, 0)
                    digits = gl.maximum(buckets - base[:, None], 0)
                    hist = _stream_histogram(digits, valid, G, 1024, NW)
                    rank = gl.minimum(running_count, 512)
                    digit, _ = _stream_bucket(hist, rank, G, NW)
                    # The first bin includes all lower clamped values. Keep
                    # the entire bin if it contains the kth score.
                    cutoff = gl.where(digit > 0, digit + base, 0)
                    keep = valid & (buckets >= cutoff[:, None])
                    kept_count = gl.sum(keep.to(gl.int32), 1)
                    if gl.max(kept_count, 0) > 768:
                        exact = _stream_exact_threshold(
                            encoded, valid, rank, G, 1024, NW
                        )
                        greater = valid & (encoded > exact[:, None])
                        equal = valid & (encoded == exact[:, None])
                        keep = greater | (
                            equal
                            & (
                                _stream_placement_prefix(equal.to(gl.int32))
                                <= (rank - gl.sum(greater.to(gl.int32), 1))[:, None]
                            )
                        )
                        kept_count = gl.sum(keep.to(gl.int32), 1)
                    destinations = _stream_placement_prefix(keep.to(gl.int32)) - 1
                    gl.barrier()
                    running.atomic_scatter_xchg(
                        values, gl.maximum(destinations, 0), 1, mask=keep
                    )
                    gl.barrier()
                    running_count = kept_count
                    coarse_cutoff = cutoff
            if TCGEN:
                gl.nvidia.ampere.async_copy.wait_group(0)
                gl.barrier()
                for stage in gl.static_range(2):
                    gl.nvidia.blackwell.mbarrier.invalidate(mma_bar.index(stage))
            values = running.load(l)
            valid = all_offsets[None, :] < running_count[:, None]
            encoded = (values >> 32).to(gl.uint32)
            rank = gl.minimum(running_count, 512)
            exact = _stream_exact_threshold(encoded, valid, rank, G, 1024, NW)
            greater = valid & (encoded > exact[:, None])
            equal = valid & (encoded == exact[:, None])
            ngreater = gl.sum(greater.to(gl.int32), 1)
            # Each row has at most 1024 candidates, so two 16-bit counts
            # can share one scan without a carry between them.
            ranks = _stream_placement_prefix(
                greater.to(gl.int32) | (equal.to(gl.int32) << 16)
            )
            dest = gl.where(
                greater, (ranks & 0xFFFF) - 1, ngreater[:, None] + (ranks >> 16) - 1
            )
            gl.store(output, -1, owned[:, None])
            gl.barrier()
            gl.store(
                selected + rows[:, None] * 512 + gl.maximum(dest, 0),
                (values & 0xFFFFFFFF).to(gl.int32),
                owned[:, None] & (greater | equal) & (dest < 512),
            )
        remaining_rows &= ~owned


def launch_qsa_block_topk_stream(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    page_table: torch.Tensor,
    request_indices: torch.Tensor,
    complete_blocks: torch.Tensor,
    selected: torch.Tensor,
    *,
    page_size: int,
    queries_per_request: int | None,
    enable_pdl: bool,
) -> torch.Tensor:
    """Launch the NVIDIA BF16 four-head stream into the supplied output."""
    rows = query.shape[0]
    num_blocks = page_table.shape[1] * page_size
    requested_group = (
        math.gcd(queries_per_request, 16) if queries_per_request is not None else 16
    )
    use_soa = (
        current_platform().is_blackwell
        and requested_group == 16
        and rows >= 4096
        and num_blocks <= 2**24
    )
    group = (
        16
        if use_soa
        else (
            math.gcd(queries_per_request, 8) if queries_per_request is not None else 8
        )
    )
    # Bucket the allocation width while keeping individual capacities runtime.
    id_bits = 16 if triton.next_power_of_2(num_blocks) <= 65536 else 24
    tcgen = current_platform().is_blackwell and group >= 4
    warps = group if tcgen else max(4, 2 * group)
    stream_kernel = (
        _qsa_soa_stream_kernel
        if use_soa
        else (_qsa_warp_stream_kernel if tcgen else _qsa_bounded_stream_kernel)
    )
    use_tma = (
        tcgen
        and page_size == 64
        and group in (8, 16)
        and key_cache.dtype == torch.bfloat16
        and key_cache.shape[0] < 2**31
    )
    descriptor_kwargs = {"key_desc": None, "USE_TMA": False} if tcgen else {}
    if use_tma:
        key_desc = GluonTensorDescriptor(
            base=key_cache,
            shape=[key_cache.shape[0], 128],
            strides=[key_cache.stride(0), key_cache.stride(2)],
            block_shape=[64, 128],
            layout=gl.NVMMASharedLayout(128, 16),
            padding="zero",
            round_f32_to_tf32=False,
        )
        descriptor_kwargs = {"key_desc": key_desc, "USE_TMA": True}
    cache_offset = (key_cache.shape[0] - 1) * key_cache.stride(0) + (
        key_cache.shape[2] - 1
    ) * key_cache.stride(2)
    table_offset = (
        (page_table.shape[0] - 1) * page_table.stride(0) + page_table.shape[1] - 1
    )
    int32_offsets = max(cache_offset, table_offset) < 2**31
    stream_kernel[(triton.cdiv(rows, group),)](
        query,
        key_cache,
        page_table,
        request_indices,
        complete_blocks,
        selected,
        rows,
        num_blocks,
        page_size,
        *query.stride(),
        key_cache.stride(0),
        key_cache.stride(2),
        page_table.stride(0),
        G=group,
        BN=128 if use_soa else 256,
        BD=64,
        NW=warps,
        RAGGED=queries_per_request is None,
        ENABLE_PDL=enable_pdl,
        TCGEN=tcgen,
        USE_INT32_OFFSETS=int32_offsets,
        num_warps=warps,
        **({"maxnreg": 96 if use_soa else 128} if tcgen else {}),
        **descriptor_kwargs,
        **({"ID_BITS": id_bits} if use_soa else {}),
        **({"launch_pdl": True} if enable_pdl else {}),
    )
    return selected


@gluon.constexpr_function
def _warp_sort_asm(merge):
    lines = [
        "{",
        ".reg .b64 current, other, high, low;",
        ".reg .b32 lane, a, b, x, y, flip;",
        ".reg .pred greater, choose_high;",
        "mov.u32 lane, %tid.x;",
        "mov.b64 current, $1;",
    ]
    for stage in [6] if merge else range(1, 6):
        for step in range(4 if merge else stage - 1, -1, -1):
            lines += [
                "mov.b64 {a, b}, current;",
                f"shfl.sync.bfly.b32 x, a, {1 << step}, 31, -1;",
                f"shfl.sync.bfly.b32 y, b, {1 << step}, 31, -1;",
                "mov.b64 other, {x, y};",
                "setp.gt.u64 greater, current, other;",
                "selp.b64 high, current, other, greater;",
                "selp.b64 low, other, current, greater;",
                ("mov.u32 a, 0;" if merge else f"shr.u32 a, lane, {stage};"),
                f"shr.u32 b, lane, {step};",
                "xor.b32 flip, a, b;",
                "and.b32 flip, flip, 1;",
                "setp.eq.u32 choose_high, flip, 0;",
                "selp.b64 current, high, low, choose_high;",
            ]
    lines += ["mov.b64 $0, current;", "}"]
    return "\n".join(lines)


@gluon.jit(
    do_not_specialize=[
        "num_tiles",
        "num_blocks",
        "stride_q_n",
        "stride_q_h",
        "stride_pt_b",
    ]
)
def _qsa_native_score_candidates_kernel(
    query,
    key_cache,
    page_table,
    request_indices,
    complete_blocks,
    candidates,
    discarded,
    merge_frontiers,
    num_heads,
    head_dim,
    num_blocks,
    page_size,
    stride_q_n,
    stride_q_h,
    stride_q_d,
    stride_k_n,
    stride_k_d,
    stride_pt_b,
    num_tiles,
    BLOCK_H: gl.constexpr,
    BLOCK_D: gl.constexpr,
    BLOCK_TOPK: gl.constexpr,
    KEEP: gl.constexpr,
    ENABLE_PDL: gl.constexpr,
    USE_INT32_OFFSETS: gl.constexpr,
):
    if ENABLE_PDL:
        gl.inline_asm_elementwise(
            "griddepcontrol.wait; mov.u32 $0, 0;",
            "=r",
            [],
            dtype=gl.int32,
            is_pure=False,
            pack=1,
        )
        gl.inline_asm_elementwise(
            "griddepcontrol.launch_dependents; mov.u32 $0, 0;",
            "=r",
            [],
            dtype=gl.int32,
            is_pure=False,
            pack=1,
        )
    row = gl.program_id(0)
    tile = gl.program_id(1)
    count = gl.minimum(gl.maximum(gl.load(complete_blocks + row), 0), num_blocks)
    if tile == 0:
        gl.store(merge_frontiers + row, gl.where(count > 8192, count, 0))
    if count <= BLOCK_TOPK:
        return
    if tile * 128 >= count:
        if count > 8192:
            padding_layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0])
            padding = gl.arange(0, 128, layout=padding_layout)
            gl.store(
                candidates + row * num_tiles * KEEP + tile * KEEP + padding,
                gl.full((128,), 0, gl.uint64, padding_layout),
                padding < KEEP,
            )
            gl.store(discarded + row * num_tiles + tile, 0)
        return
    effective_keep = gl.where(count <= 4096, 64, KEEP)
    flat: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0])
    columns = gl.arange(0, 128, layout=flat)
    ids = tile * 128 + columns
    packed = gl.full((128,), 0, gl.uint64, flat)
    if count > BLOCK_TOPK and tile * 128 < count:
        request = gl.load(request_indices + row).to(
            gl.int32 if USE_INT32_OFFSETS else gl.int64
        )
        qlayout: gl.constexpr = gl.BlockedLayout([8, 1], [16, 2], [1, 4], [0, 1])
        dims_q = gl.arange(0, 128, layout=gl.SliceLayout(1, qlayout))
        heads = gl.arange(0, 8, layout=gl.SliceLayout(0, qlayout))
        qvalues = gl.load(
            query
            + row * stride_q_n
            + dims_q[:, None] * stride_q_d
            + heads[None, :] * stride_q_h,
            heads[None, :] < 4,
            0,
        )
        qs = gl.allocate_shared_memory(
            gl.bfloat16,
            (128, 8),
            gl.NVMMASharedLayout(128, 16, transposed=True),
            qvalues,
        )
        ks = gl.allocate_shared_memory(
            gl.bfloat16, (128, 128), gl.NVMMASharedLayout(128, 16)
        )
        klayout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0])
        kid = tile * 128 + gl.arange(0, 128, layout=gl.SliceLayout(1, klayout))
        dims = gl.arange(0, 128, layout=gl.SliceLayout(0, klayout))
        pages = gl.load(
            page_table + request * stride_pt_b + kid // page_size, kid < count, 0
        ).to(gl.int32 if USE_INT32_OFFSETS else gl.int64)
        slots = pages * page_size + kid % page_size
        gl.nvidia.ampere.async_copy.async_load(
            ks,
            key_cache + slots[:, None] * stride_k_n + dims[None, :] * stride_k_d,
            mask=kid[:, None] < count,
        )
        gl.nvidia.ampere.async_copy.commit_group()
        tm = gl.nvidia.blackwell.allocate_tensor_memory(
            gl.float32, (128, 8), gl.nvidia.blackwell.TensorMemoryLayout((128, 8), 1)
        )
        barrier = gl.allocate_shared_memory(
            gl.int64, (1,), gl.nvidia.blackwell.mbarrier.MBarrierLayout()
        )
        gl.nvidia.blackwell.mbarrier.init(barrier, count=1)
        gl.nvidia.ampere.async_copy.wait_group(0)
        gl.barrier()
        gl.nvidia.hopper.fence_async_shared()
        gl.nvidia.blackwell.tcgen05_mma(ks, qs, tm, use_acc=False, mbarriers=[barrier])
        gl.nvidia.blackwell.mbarrier.wait(barrier, 0, deps=(ks, qs, tm))
        scores = gl.convert_layout(gl.sum(gl.maximum(tm.load(), 0), 1), flat)
        gl.nvidia.blackwell.mbarrier.invalidate(barrier)
        packed = (
            (scores.to(gl.uint32, bitcast=True).to(gl.uint64) + 1) << 32
        ) | ids.to(gl.uint64)
        packed = gl.where(ids < count, packed, 0)
    first_pass = gl.inline_asm_elementwise(
        _warp_sort_asm(False), "=l,l", [packed], dtype=gl.uint64, is_pure=True, pack=1
    )
    exchange = gl.allocate_shared_memory(
        gl.uint64, (128,), gl.SwizzledSharedLayout(1, 1, 1, [0]), first_pass
    )
    gl.barrier()
    paired = gl.inline_asm(
        "{ .reg .b32 thread, address; mov.u32 thread, %tid.x; xor.b32 thread, thread, 32; mad.lo.u32 address, thread, 8, $1; ld.shared.b64 $0, [address]; }",
        "=l,r,~{memory}",
        [exchange],
        result_types=gl.distributed_type(gl.uint64, (128,), flat),
        is_pure=False,
    )
    # The following reduction can reuse the exchange buffer.
    gl.barrier()
    paired_first = gl.where(
        (columns & 32) == 0,
        gl.maximum(first_pass, paired),
        gl.minimum(first_pass, paired),
    )
    sorted_keys = gl.inline_asm_elementwise(
        _warp_sort_asm(True),
        "=l,l",
        [paired_first],
        dtype=gl.uint64,
        is_pure=True,
        pack=1,
    )
    per_group = effective_keep // 2
    offsets = (columns // 64) * per_group + columns % 64
    gl.store(
        candidates + row * num_tiles * KEEP + tile * effective_keep + offsets,
        sorted_keys,
        columns % 64 < per_group,
    )
    upper = gl.max(gl.where(columns % 64 == per_group, sorted_keys >> 32, 0), 0)
    gl.store(discarded + row * num_tiles + tile, upper.to(gl.uint32))


@gluon.jit
def _decode_add(a, b):
    return a + b


@gluon.jit
def _decode_scan(values):
    return gl.associative_scan(values, 1, _decode_add)


@gluon.constexpr_function
def _decode_histogram_constraints(count):
    return ",".join(["r"] + ["r"] * count + ["~{memory}"])


@gluon.constexpr_function
def _decode_histogram_asm(outputs, inputs):
    base, offsets = inputs
    lines = ["{", ".reg .u32 address;", ".reg .pred active;"]
    for offset in offsets:
        lines.append(f"setp.ge.s32 active, {offset}, 0;")
        lines.append(f"mad.lo.u32 address, {offset}, 4, {base[0]};")
        lines.append("@active red.relaxed.cta.shared.add.u32 [address], 1;")
    lines.append("}")
    return "\n".join(lines)


@gluon.constexpr_function
def _decode_histogram_dedup_asm(outputs, inputs):
    base, offsets, limit = inputs
    lines = [
        "{",
        ".reg .u32 address, thread;",
        ".reg .pred owned, active;",
        "mov.u32 thread, %tid.x;",
        f"setp.lt.u32 owned, thread, {limit[0]};",
    ]
    for offset in offsets:
        lines.extend(
            [
                f"setp.ge.s32 active, {offset}, 0;",
                "and.pred active, active, owned;",
                f"mad.lo.u32 address, {offset}, 4, {base[0]};",
                "@active red.relaxed.cta.shared.add.u32 [address], 1;",
            ]
        )
    lines.append("}")
    return "\n".join(lines)


@gluon.jit
def _decode_histogram(
    digits, valid, G: gl.constexpr, N: gl.constexpr, NW: gl.constexpr
):
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [G, NW // G], [1, 0])
    hist_layout: gl.constexpr = gl.BlockedLayout(
        [1, 8 if NW == G else 4], [1, 32], [G, NW // G], [1, 0]
    )
    rows = gl.arange(0, G, layout=gl.SliceLayout(1, layout))
    offsets = gl.where(valid, digits + rows[:, None] * 256, -1)
    # The same storage may have held the preceding histogram's reads.
    gl.barrier()
    histogram = gl.allocate_shared_memory(
        gl.int32,
        (G, 256),
        gl.SwizzledSharedLayout(1, 1, 1, [1, 0]),
        gl.full((G, 256), 0, gl.int32, hist_layout),
    )
    gl.barrier()
    count: gl.constexpr = max(1, G * N // (NW * 32))
    if G * N < NW * 32:
        gl.inline_asm(
            _decode_histogram_dedup_asm,
            "r,r,r,~{memory}",
            (histogram, offsets, G * N),
            result_types=(),
            is_pure=False,
        )
    else:
        gl.inline_asm(
            _decode_histogram_asm,
            _decode_histogram_constraints(count),
            (histogram, offsets),
            result_types=(),
            is_pure=False,
        )
    gl.barrier()
    return histogram.load(hist_layout)


@gluon.jit
def _decode_bucket(hist, rank, G: gl.constexpr, NW: gl.constexpr):
    l: gl.constexpr = gl.BlockedLayout(
        [1, 8 if NW == G else 4], [1, 32], [G, NW // G], [1, 0]
    )
    old_l: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [G, NW // G], [1, 0])
    rank = gl.convert_layout(rank, gl.SliceLayout(1, l))
    bins = gl.arange(0, 256, layout=gl.SliceLayout(0, l))
    before = gl.sum(hist, 1)[:, None] - _decode_scan(hist)
    digit = gl.max(
        gl.where(
            (before < rank[:, None]) & (before + hist >= rank[:, None]),
            bins[None, :],
            0,
        ),
        1,
    )
    remaining = rank - gl.sum(gl.where(bins[None, :] > digit[:, None], hist, 0), 1)
    return gl.convert_layout(digit, gl.SliceLayout(1, old_l)), gl.convert_layout(
        remaining, gl.SliceLayout(1, old_l)
    )


@gluon.jit
def _decode_exact_threshold(
    bits, valid, rank, G: gl.constexpr, N: gl.constexpr, NW: gl.constexpr
):
    l: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [G, NW // G], [1, 0])
    active = valid
    cutoff = gl.full((G,), 0, gl.uint32, gl.SliceLayout(1, l))
    remaining = rank
    for shift in gl.static_range(24, -1, -8):
        digits = ((bits >> shift) & 255).to(gl.int32)
        hist = _decode_histogram(digits, active, G, N, NW)
        digit, remaining = _decode_bucket(hist, remaining, G, NW)
        cutoff |= digit.to(gl.uint32) << shift
        active &= digits == digit[:, None]
    return cutoff


@gluon.jit
def _decode_placement_prefix(values):
    """Compact in thread order; the public selection has no output order."""
    group: gl.constexpr = values.shape[0]
    columns: gl.constexpr = values.shape[1]
    threads_per_row: gl.constexpr = 32 * gl.num_warps() // group
    if columns >= threads_per_row:
        by_thread = gl.reshape(
            values, (group, columns // threads_per_row, threads_per_row)
        )
        totals = gl.sum(by_thread, 1)
        totals_layout: gl.constexpr = totals.type.layout
        plain: gl.constexpr = gl.BlockedLayout(
            [1, 1], [1, 32], [group, gl.num_warps() // group], [1, 0]
        )
        totals_plain = gl.convert_layout(totals, plain)
        before = gl.convert_layout(
            _decode_scan(totals_plain) - totals_plain, totals_layout
        )
        within = gl.associative_scan(by_thread, 1, _decode_add)
        return gl.convert_layout(
            gl.reshape(before[:, None, :] + within, (group, columns)),
            values.type.layout,
        )
    else:
        return gl.convert_layout(_decode_scan(values), values.type.layout)


@gluon.jit
def _merge_prefix(values, N: gl.constexpr, NW: gl.constexpr):
    flat: gl.constexpr = gl.BlockedLayout([1], [32], [NW], [0])
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, NW], [1, 0])
    matrix = gl.convert_layout(values, gl.SliceLayout(0, layout))[None, :]
    ranks = _decode_placement_prefix(matrix)
    return gl.convert_layout(gl.reshape(ranks, (N,)), flat)


@gluon.constexpr_function
def _merge_score_load_constraints(count):
    return ",".join(["=r"] * count + ["l", "r", "r", "~{memory}"])


@gluon.constexpr_function
def _merge_score_load_asm(outputs, inputs):
    (values,) = outputs
    base, limit, threads = inputs
    lines = [
        "{",
        ".reg .u32 thread, index;",
        ".reg .u64 address;",
        ".reg .pred active;",
        "mov.u32 thread, %tid.x;",
    ]
    for item, value in enumerate(values):
        lines.extend(
            [
                f"mad.lo.u32 index, {threads[0]}, {item}, thread;",
                f"setp.lt.u32 active, index, {limit[0]};",
                f"mad.wide.u32 address, index, 8, {base[0]};",
                f"mov.u32 {value}, 0;",
                f"@active ld.global.u32 {value}, [address+4];",
            ]
        )
    lines.append("}")
    return "\n".join(lines)


@gluon.jit
def _merge_load_scores(base, limit, N: gl.constexpr, NW: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [NW], [0])
    count: gl.constexpr = N // (NW * 32)
    return gl.inline_asm(
        _merge_score_load_asm,
        _merge_score_load_constraints(count),
        (base, limit, NW * 32),
        result_types=gl.distributed_type(gl.uint32, (N,), layout),
        is_pure=False,
    )


@gluon.constexpr_function
def _merge_pool_store_constraints(count):
    return ",".join(["l", "r", "r", "r"] + ["r"] * count + ["~{memory}"])


@gluon.constexpr_function
def _merge_pool_store_asm(outputs, inputs):
    source, score_pool, index_pool, threads, destinations = inputs
    lines = [
        "{",
        ".reg .u32 thread, index, offset, address, bits;",
        ".reg .u64 global_address;",
        ".reg .pred active;",
        "mov.u32 thread, %tid.x;",
    ]
    for item, destination in enumerate(destinations):
        lines.extend(
            [
                f"setp.ge.s32 active, {destination}, 0;",
                f"mad.lo.u32 index, {threads[0]}, {item}, thread;",
                f"mad.wide.u32 global_address, index, 8, {source[0]};",
                "@active ld.global.u32 bits, [global_address+4];",
                f"shl.b32 offset, {destination}, 2;",
                f"add.u32 address, {score_pool[0]}, offset;",
                "@active st.shared.u32 [address], bits;",
                f"add.u32 address, {index_pool[0]}, offset;",
                "@active st.shared.u32 [address], index;",
            ]
        )
    lines.append("}")
    return "\n".join(lines)


@gluon.jit
def _merge_store_pool(
    source, scores, indices, destinations, N: gl.constexpr, NW: gl.constexpr
):
    gl.inline_asm(
        _merge_pool_store_asm,
        _merge_pool_store_constraints(N // (NW * 32)),
        (source, scores, indices, NW * 32, destinations),
        result_types=(),
        is_pure=False,
    )


@gluon.constexpr_function
def _merge_output_store_constraints(count):
    return ",".join(["l", "l", "r"] + ["r"] * count + ["~{memory}"])


@gluon.constexpr_function
def _merge_output_store_asm(outputs, inputs):
    source, output, threads, destinations = inputs
    lines = [
        "{",
        ".reg .u32 thread, index, id;",
        ".reg .u64 source_address, output_address;",
        ".reg .pred positive, bounded, active;",
        "mov.u32 thread, %tid.x;",
    ]
    for item, destination in enumerate(destinations):
        lines.extend(
            [
                f"setp.ge.s32 positive, {destination}, 0;",
                f"setp.lt.s32 bounded, {destination}, 512;",
                "and.pred active, positive, bounded;",
                f"mad.lo.u32 index, {threads[0]}, {item}, thread;",
                f"mad.wide.u32 source_address, index, 8, {source[0]};",
                "@active ld.global.u32 id, [source_address];",
                f"mad.wide.u32 output_address, {destination}, 4, {output[0]};",
                "@active st.global.u32 [output_address], id;",
            ]
        )
    lines.append("}")
    return "\n".join(lines)


@gluon.jit
def _merge_store_equal_output(source, output, valid, N: gl.constexpr, NW: gl.constexpr):
    destination = _merge_prefix(valid.to(gl.int32), N, NW) - 1
    destination = gl.where(valid, destination, -1)
    gl.inline_asm(
        _merge_output_store_asm,
        _merge_output_store_constraints(N // (NW * 32)),
        (source, output, NW * 32, destination),
        result_types=(),
        is_pure=False,
    )


@gluon.jit
def _merge_coarse_select(
    source,
    output,
    candidate_limit,
    total,
    topk,
    N: gl.constexpr,
    NW: gl.constexpr,
    PACKED_OUTPUT: gl.constexpr,
):
    flat: gl.constexpr = gl.BlockedLayout([1], [32], [NW], [0])
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, NW], [1, 0])
    row_layout: gl.constexpr = gl.SliceLayout(1, layout)
    bits = _merge_load_scores(source, candidate_limit, N, NW)
    valid = bits != 0
    cutoff = gl.full((), 0, gl.uint32, gl.SliceLayout(0, flat))
    enough = total >= topk
    if enough:
        highest = gl.max(bits, 0)
        if highest == 1:
            # Native candidate tiles have a contiguous valid prefix.
            cutoff = highest
            equal_offsets = gl.arange(0, 512, layout=flat)
            packed_equal = gl.load(source + equal_offsets, equal_offsets < topk, 0)
            if PACKED_OUTPUT:
                gl.store(output + equal_offsets, packed_equal, equal_offsets < topk)
            else:
                gl.store(
                    output + equal_offsets,
                    (packed_equal & 0xFFFFFFFF).to(gl.int32),
                    equal_offsets < topk,
                )
        else:
            base = gl.maximum((highest >> 16).to(gl.int32) - 255, 0)
            digits = gl.maximum((bits >> 16).to(gl.int32) - base, 0)
            digits_2d = gl.convert_layout(digits, gl.SliceLayout(0, layout))[None, :]
            valid_2d = gl.convert_layout(valid, gl.SliceLayout(0, layout))[None, :]
            hist = _decode_histogram(digits_2d, valid_2d, 1, N, NW)
            rank = gl.full((1,), 0, gl.int32, row_layout) + topk
            digit, remaining = _decode_bucket(hist, rank, 1, NW)
            digit_scalar = gl.sum(digit, 0)
            coarse_cutoff = gl.where(
                digit_scalar > 0, (digit_scalar + base).to(gl.uint32) << 16, 0
            )
            promising = valid & (bits >= coarse_cutoff)
            histogram_bins = gl.arange(
                0, 256, layout=gl.SliceLayout(0, hist.type.layout)
            )
            pool_count = gl.sum(
                gl.sum(gl.where(histogram_bins[None, :] >= digit_scalar, hist, 0), 1), 0
            )
            remaining_scalar = gl.sum(remaining, 0)
            boundary_count = pool_count - topk + remaining_scalar
            if boundary_count <= 32:
                cutoff = _merge_small_boundary(
                    source,
                    output,
                    bits,
                    valid,
                    base,
                    digit_scalar,
                    topk,
                    remaining_scalar,
                    N,
                    NW,
                    PACKED_OUTPUT,
                )
            else:
                if pool_count <= 1024:
                    destinations = _merge_prefix(promising.to(gl.int32), N, NW) - 1
                    destinations = gl.where(promising, destinations, -1)
                    scores = gl.allocate_shared_memory(
                        gl.uint32, (1024,), gl.SwizzledSharedLayout(1, 1, 1, [0])
                    )
                    indices = gl.allocate_shared_memory(
                        gl.uint32, (1024,), gl.SwizzledSharedLayout(1, 1, 1, [0])
                    )
                    gl.barrier()
                    _merge_store_pool(source, scores, indices, destinations, N, NW)
                    gl.barrier()
                    pooled = scores.load(flat)
                    offsets = gl.arange(0, 1024, layout=flat)
                    pooled_valid = offsets < pool_count
                    pooled_2d = gl.convert_layout(pooled, gl.SliceLayout(0, layout))[
                        None, :
                    ]
                    pooled_valid_2d = gl.convert_layout(
                        pooled_valid, gl.SliceLayout(0, layout)
                    )[None, :]
                    if digit_scalar > 0:
                        active = pooled_valid_2d & (
                            (pooled_2d >> 16) == (coarse_cutoff >> 16)
                        )
                        exact = gl.full((1,), 0, gl.uint32, row_layout) + coarse_cutoff
                        for shift in gl.static_range(8, -1, -8):
                            fine_digits = ((pooled_2d >> shift) & 255).to(gl.int32)
                            fine_hist = _decode_histogram(
                                fine_digits, active, 1, 1024, NW
                            )
                            fine_digit, remaining = _decode_bucket(
                                fine_hist, remaining, 1, NW
                            )
                            exact |= fine_digit.to(gl.uint32) << shift
                            active &= fine_digits == fine_digit[:, None]
                        cutoff = gl.sum(exact, 0)
                    else:
                        cutoff = gl.sum(
                            _decode_exact_threshold(
                                pooled_2d, pooled_valid_2d, rank, 1, 1024, NW
                            ),
                            0,
                        )
                    greater = pooled_valid & (pooled > cutoff)
                    equal = pooled_valid & (pooled == cutoff)
                    higher = gl.sum(greater.to(gl.int32), 0)
                    ranks = _merge_prefix(
                        greater.to(gl.int32) | (equal.to(gl.int32) << 16), 1024, NW
                    )
                    destination = gl.where(
                        greater, (ranks & 0xFFFF) - 1, higher + (ranks >> 16) - 1
                    )
                    chosen = (greater | equal) & (destination < topk)
                    source_indices = indices.load(flat)
                    packed = gl.load(source + source_indices, chosen, 0)
                    if PACKED_OUTPUT:
                        gl.store(output + destination, packed, chosen)
                    else:
                        gl.store(
                            output + destination,
                            (packed & 0xFFFFFFFF).to(gl.int32),
                            chosen,
                        )
                else:
                    # The original complete selection handles an oversized boundary.
                    enough = gl.full((), False, gl.int1, gl.SliceLayout(0, flat))
    return cutoff, enough


@gluon.jit(
    do_not_specialize=[
        "num_tiles",
        "num_parts",
        "num_blocks",
        "num_rows",
        "stride_q_n",
        "stride_q_h",
        "stride_pt_b",
    ]
)
def _qsa_coarse_merge_candidates_kernel(
    candidates,
    discarded,
    complete,
    selected,
    refine,
    num_tiles,
    num_blocks,
    partials,
    num_parts,
    query,
    key_cache,
    page_table,
    requests,
    num_rows,
    page_size: gl.constexpr,
    stride_q_n,
    stride_q_h,
    stride_q_d,
    stride_k_n,
    stride_k_d,
    stride_pt_b,
    TILE_SIZE: gl.constexpr,
    KEEP: gl.constexpr,
    N: gl.constexpr,
    NT: gl.constexpr,
    NW: gl.constexpr,
    ENABLE_PDL: gl.constexpr,
    ADAPTIVE: gl.constexpr,
    USE_PARTIAL: gl.constexpr,
):
    if ENABLE_PDL:
        gl.inline_asm_elementwise(
            "griddepcontrol.wait; mov.u32 $0, 0;",
            "=r",
            [],
            dtype=gl.int32,
            is_pure=False,
            pack=1,
        )
        gl.inline_asm_elementwise(
            "griddepcontrol.launch_dependents; mov.u32 $0, 0;",
            "=r",
            [],
            dtype=gl.int32,
            is_pure=False,
            pack=1,
        )
    row = gl.program_id(0)
    count = gl.minimum(gl.maximum(gl.load(complete + row), 0), num_blocks)
    source = candidates + row * num_tiles * KEEP
    output = selected + row * 512
    needs_refine = gl.full(
        (), False, gl.int1, gl.SliceLayout(0, gl.BlockedLayout([1], [32], [NW], [0]))
    )
    if count <= 512:
        offsets = gl.arange(0, 512, layout=gl.BlockedLayout([1], [32], [NW], [0]))
        gl.store(output + offsets, gl.where(offsets < count, offsets, -1))
    else:
        limit = num_tiles * KEEP
        active_tiles = num_tiles
        retained_keep = gl.where(count <= 4096, TILE_SIZE // 2, KEEP)
        remainder = count % TILE_SIZE
        retained_total = (
            count // TILE_SIZE * retained_keep
            + gl.minimum(remainder, retained_keep // 2)
            + gl.minimum(gl.maximum(remainder - TILE_SIZE // 2, 0), retained_keep // 2)
        )
        if USE_PARTIAL:
            if count > 8192:
                local_keep = gl.where(count >= 98304, 128, 512)
                source = partials + row * num_parts * 512
                limit = num_parts * local_keep
                retained_total = retained_total // 4096 * local_keep + gl.minimum(
                    retained_total % 4096, local_keep
                )
        if ADAPTIVE:
            if count <= 8192:
                active_tiles = gl.cdiv(count, TILE_SIZE)
                effective_keep = gl.where(count <= 4096, TILE_SIZE // 2, KEEP)
                limit = active_tiles * effective_keep
            if count <= 4096:
                cutoff, enough = _merge_coarse_select(
                    source,
                    output,
                    limit,
                    retained_total,
                    512,
                    min(N, 2048),
                    NW,
                    PACKED_OUTPUT=False,
                )
            elif count <= 8192:
                cutoff, enough = _merge_coarse_select(
                    source,
                    output,
                    limit,
                    retained_total,
                    512,
                    min(N, 4096),
                    NW,
                    PACKED_OUTPUT=False,
                )
            else:
                cutoff, enough = _merge_coarse_select(
                    source,
                    output,
                    limit,
                    retained_total,
                    512,
                    N,
                    NW,
                    PACKED_OUTPUT=False,
                )
        else:
            cutoff, enough = _merge_coarse_select(
                source, output, limit, retained_total, 512, N, NW, PACKED_OUTPUT=False
            )
        tiles = gl.arange(0, NT, layout=gl.BlockedLayout([1], [32], [NW], [0]))
        upper = gl.load(discarded + row * num_tiles + tiles, tiles < active_tiles, 0)
        needs_refine = (~enough) | (gl.max(upper, 0) > cutoff)
        if needs_refine:
            _qsa_decode_fallback_kernel(
                query,
                key_cache,
                page_table,
                requests,
                complete,
                selected,
                num_rows,
                num_blocks,
                page_size,
                stride_q_n,
                stride_q_h,
                stride_q_d,
                stride_k_n,
                stride_k_d,
                stride_pt_b,
                G=1,
                BN=256,
                BD=64,
                NW=NW,
                RAGGED=False,
                ENABLE_PDL=False,
                TCGEN=False,
                USE_INT32_OFFSETS=False,
            )
    gl.store(refine + row, needs_refine.to(gl.int32))


@gluon.jit
def _decode_coarse(bits):
    # Positive FP16 encoding is monotone, including saturation to infinity.
    # Use it only for pruning; exact FP32 scores determine final selection.
    return (
        bits.to(gl.float32, bitcast=True).to(gl.float16).to(gl.uint16, bitcast=True)
        >> 6
    ).to(gl.int32)


@gluon.jit
def _decode_prefetch_keys(
    key_cache,
    page_table,
    request,
    start,
    end,
    stride_k_n,
    stride_k_d,
    stride_pt_b,
    destination,
    PAGE_SIZE: gl.constexpr,
    BN: gl.constexpr,
    NW: gl.constexpr,
    USE_INT32_OFFSETS: gl.constexpr,
):
    layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [NW, 1], [1, 0])
    ids = start + gl.arange(0, BN, layout=gl.SliceLayout(1, layout))
    dims = gl.arange(0, 128, layout=gl.SliceLayout(0, layout))
    if USE_INT32_OFFSETS:
        request = request.to(gl.int32)
    pages = gl.load(
        page_table + request * stride_pt_b + ids // PAGE_SIZE, ids < end, 0
    ).to(gl.int32 if USE_INT32_OFFSETS else gl.int64)
    slots = pages * PAGE_SIZE + ids % PAGE_SIZE
    pointers = key_cache + slots[:, None] * stride_k_n + dims[None, :] * stride_k_d
    gl.nvidia.ampere.async_copy.async_load(
        destination, pointers, mask=ids[:, None] < end
    )
    gl.nvidia.ampere.async_copy.commit_group()


@gluon.jit(
    noinline=True,
    do_not_specialize=[
        "num_rows",
        "num_blocks",
        "stride_q_n",
        "stride_q_h",
        "stride_pt_b",
    ],
)
def _qsa_decode_fallback_kernel(
    query,
    key_cache,
    page_table,
    requests,
    complete,
    selected,
    num_rows,
    num_blocks,
    page_size: gl.constexpr,
    stride_q_n,
    stride_q_h,
    stride_q_d,
    stride_k_n,
    stride_k_d,
    stride_pt_b,
    G: gl.constexpr,
    BN: gl.constexpr,
    BD: gl.constexpr,
    NW: gl.constexpr,
    RAGGED: gl.constexpr,
    ENABLE_PDL: gl.constexpr,
    TCGEN: gl.constexpr,
    USE_INT32_OFFSETS: gl.constexpr,
):
    l: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [G, NW // G], [1, 0])
    rl: gl.constexpr = gl.SliceLayout(1, l)
    cl: gl.constexpr = gl.SliceLayout(0, l)
    if ENABLE_PDL:
        gl.inline_asm_elementwise(
            "griddepcontrol.wait; mov.u32 $0, 0;",
            "=r",
            [],
            dtype=gl.int32,
            is_pure=False,
            pack=1,
        )
        gl.inline_asm_elementwise(
            "griddepcontrol.launch_dependents; mov.u32 $0, 0;",
            "=r",
            [],
            dtype=gl.int32,
            is_pure=False,
            pack=1,
        )
    rows = gl.program_id(0) * G + gl.arange(0, G, layout=rl)
    row_valid = rows < num_rows
    reqs = gl.load(requests + rows, row_valid, 0).to(gl.int64)
    counts = gl.minimum(
        gl.maximum(gl.load(complete + rows, row_valid, 0), 0), num_blocks
    )
    remaining_rows = row_valid
    # A boundary CTA handles each request separately, sharing K only between
    # matching requests. The common case is one pass through this loop.
    while gl.sum(remaining_rows.to(gl.int32), 0) > 0:
        first = gl.min(gl.where(remaining_rows, rows, 0x7FFFFFFF), 0)
        req = gl.sum(gl.where(rows == first, reqs, 0), 0)
        owned = remaining_rows & (reqs == req)
        valid_count = gl.where(owned, counts, 0)
        max_count = gl.max(valid_count, 0)
        outcols = gl.arange(0, 512, layout=cl)
        output = selected + rows[:, None] * 512 + outcols[None, :]
        if max_count <= 512:
            gl.store(
                output,
                gl.where(outcols[None, :] < valid_count[:, None], outcols[None, :], -1),
                owned[:, None],
            )
        else:
            running = gl.allocate_shared_memory(
                gl.uint64, (G, 1024), gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
            )
            running_count = gl.full((G,), 0, gl.int32, rl)
            coarse_cutoff = gl.full((G,), 0, gl.int32, rl)
            ns = gl.arange(0, BN, layout=cl)
            all_offsets = gl.arange(0, 1024, layout=cl)
            mma_l: gl.constexpr = gl.NVMMADistributedLayout(
                version=[2, 0], warps_per_cta=[1, NW], instr_shape=[16, 8]
            )
            qm: gl.constexpr = max(16, G * 4)
            ql: gl.constexpr = gl.BlockedLayout([1, 4], [2, 16], [NW, 1], [1, 0])
            kl: gl.constexpr = gl.BlockedLayout([8, 1], [16, 2], [1, NW], [0, 1])
            hi = gl.arange(0, qm, layout=gl.SliceLayout(1, ql))
            diq = gl.arange(0, BD, layout=gl.SliceLayout(0, ql))
            dik = gl.arange(0, BD, layout=gl.SliceLayout(1, kl))
            nk = gl.arange(0, BN, layout=gl.SliceLayout(0, kl))
            # Reuse the ownership mask in the matrix view without reading
            # another row or creating a new request-dependent specialization.
            head_owned = gl.gather(owned, hi // 4 % G, 0)
            if TCGEN:
                tql: gl.constexpr = gl.BlockedLayout([8, 1], [16, 2], [1, NW], [0, 1])
                tkl: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [NW, 1], [1, 0])
                tdq = gl.arange(0, 128, layout=gl.SliceLayout(1, tql))
                thq = gl.arange(0, G * 4, layout=gl.SliceLayout(0, tql))
                owned_tc = gl.gather(
                    gl.convert_layout(owned, gl.SliceLayout(0, tql)), thq // 4, 0
                )
                qvalue = gl.load(
                    query
                    + (gl.program_id(0) * G + thq[None, :] // 4) * stride_q_n
                    + (thq[None, :] % 4) * stride_q_h
                    + tdq[:, None] * stride_q_d,
                    owned_tc[None, :],
                    0,
                )
                qs = gl.allocate_shared_memory(
                    gl.bfloat16,
                    (128, G * 4),
                    gl.NVMMASharedLayout(128, 16, transposed=True),
                    qvalue,
                )
                ks = gl.allocate_shared_memory(
                    gl.bfloat16, (2, BN, 128), gl.NVMMASharedLayout(128, 16)
                )
                tm = gl.nvidia.blackwell.allocate_tensor_memory(
                    gl.float32,
                    (2, BN, G * 4),
                    gl.nvidia.blackwell.TensorMemoryLayout((128, G * 4), 1),
                )
                mma_bar = gl.allocate_shared_memory(
                    gl.int64, (2, 1), gl.nvidia.blackwell.mbarrier.MBarrierLayout()
                )
                for stage in gl.static_range(2):
                    gl.nvidia.blackwell.mbarrier.init(mma_bar.index(stage), count=1)
                    _decode_prefetch_keys(
                        key_cache,
                        page_table,
                        req,
                        stage * BN,
                        max_count,
                        stride_k_n,
                        stride_k_d,
                        stride_pt_b,
                        ks.index(stage),
                        page_size,
                        BN,
                        NW,
                        USE_INT32_OFFSETS,
                    )
                # Wait for the first copy while allowing the second to stay
                # in flight. Each stage is reused only after its MMA completes.
                gl.nvidia.ampere.async_copy.wait_group(1)
                gl.barrier()
                gl.nvidia.hopper.fence_async_shared()
                gl.nvidia.blackwell.tcgen05_mma(
                    ks.index(0),
                    qs,
                    tm.index(0),
                    use_acc=False,
                    mbarriers=[mma_bar.index(0)],
                )
            for start in range(0, max_count, BN):
                if TCGEN:
                    slot = (start // BN) & 1
                    phase = (start // (2 * BN)) & 1
                    gl.nvidia.blackwell.mbarrier.wait(
                        mma_bar.index(slot),
                        phase,
                        deps=(ks.index(slot), tm.index(slot)),
                    )
                    matrix = tm.index(slot).load()
                    scores = gl.permute(
                        gl.sum(gl.reshape(gl.maximum(matrix, 0), (BN, G, 4)), 2), (1, 0)
                    )
                    if start + BN < max_count:
                        gl.nvidia.ampere.async_copy.wait_group(0)
                        gl.barrier()
                        gl.nvidia.hopper.fence_async_shared()
                        gl.nvidia.blackwell.tcgen05_mma(
                            ks.index(1 - slot),
                            qs,
                            tm.index(1 - slot),
                            use_acc=False,
                            mbarriers=[mma_bar.index(1 - slot)],
                        )
                        if start + 2 * BN < max_count:
                            _decode_prefetch_keys(
                                key_cache,
                                page_table,
                                req,
                                start + 2 * BN,
                                max_count,
                                stride_k_n,
                                stride_k_d,
                                stride_pt_b,
                                ks.index(slot),
                                page_size,
                                BN,
                                NW,
                                USE_INT32_OFFSETS,
                            )
                else:
                    ids = start + nk
                    pages = gl.load(
                        page_table + req * stride_pt_b + ids // page_size,
                        ids < max_count,
                        0,
                    ).to(gl.int64)
                    slots = pages * page_size + ids % page_size
                    acc = gl.full((qm, BN), 0, gl.float32, mma_l)
                    for ds in range(0, 128, BD):
                        q = gl.load(
                            query
                            + (gl.program_id(0) * G + hi[:, None] // 4) * stride_q_n
                            + (hi[:, None] % 4) * stride_q_h
                            + (ds + diq[None, :]) * stride_q_d,
                            (hi[:, None] < G * 4) & head_owned[:, None],
                            0,
                        )
                        k = gl.load(
                            key_cache
                            + slots[None, :] * stride_k_n
                            + (ds + dik[:, None]) * stride_k_d,
                            ids[None, :] < max_count,
                            0,
                        )
                        a = gl.convert_layout(q, gl.DotOperandLayout(0, mma_l, 2))
                        b = gl.convert_layout(k, gl.DotOperandLayout(1, mma_l, 2))
                        acc = gl.nvidia.ampere.mma_v2(a, b, acc, input_precision="ieee")
                    if G == 1:
                        hmask = gl.arange(0, qm, layout=gl.SliceLayout(1, mma_l)) < 4
                        acc = gl.where(hmask[:, None], gl.maximum(acc, 0), 0)
                        scores = gl.sum(acc, 0)[None, :]
                    else:
                        scores = gl.sum(gl.reshape(gl.maximum(acc, 0), (G, 4, BN)), 1)
                scores = gl.convert_layout(scores, l)
                block_ids = start + ns
                bits = scores.to(gl.uint32, bitcast=True)
                bucket = _decode_coarse(bits)
                eligible = (
                    owned[:, None]
                    & (block_ids[None, :] < valid_count[:, None])
                    & (bucket >= coarse_cutoff[:, None])
                )
                added = gl.sum(eligible.to(gl.int32), 1)
                dest = (
                    running_count[:, None]
                    + _decode_placement_prefix(eligible.to(gl.int32))
                    - 1
                )
                packed = ((bits.to(gl.uint64) + 1) << 32) | block_ids[None, :].to(
                    gl.uint64
                )
                running.atomic_scatter_xchg(
                    packed, gl.maximum(dest, 0), 1, mask=eligible
                )
                gl.barrier()
                running_count += added
                if gl.max(running_count, 0) > 768:
                    values = running.load(l)
                    valid = all_offsets[None, :] < running_count[:, None]
                    encoded = (values >> 32).to(gl.uint32)
                    buckets = _decode_coarse(encoded - 1)
                    base = gl.maximum(gl.max(gl.where(valid, buckets, 0), 1) - 255, 0)
                    digits = gl.maximum(buckets - base[:, None], 0)
                    hist = _decode_histogram(digits, valid, G, 1024, NW)
                    rank = gl.minimum(running_count, 512)
                    digit, _ = _decode_bucket(hist, rank, G, NW)
                    # The first bin includes all lower clamped values. Keep
                    # the entire bin if it contains the kth score.
                    cutoff = gl.where(digit > 0, digit + base, 0)
                    keep = valid & (buckets >= cutoff[:, None])
                    kept_count = gl.sum(keep.to(gl.int32), 1)
                    if gl.max(kept_count, 0) > 768:
                        exact = _decode_exact_threshold(
                            encoded, valid, rank, G, 1024, NW
                        )
                        greater = valid & (encoded > exact[:, None])
                        equal = valid & (encoded == exact[:, None])
                        keep = greater | (
                            equal
                            & (
                                _decode_placement_prefix(equal.to(gl.int32))
                                <= (rank - gl.sum(greater.to(gl.int32), 1))[:, None]
                            )
                        )
                        kept_count = gl.sum(keep.to(gl.int32), 1)
                    destinations = _decode_placement_prefix(keep.to(gl.int32)) - 1
                    gl.barrier()
                    running.atomic_scatter_xchg(
                        values, gl.maximum(destinations, 0), 1, mask=keep
                    )
                    gl.barrier()
                    running_count = kept_count
                    coarse_cutoff = cutoff
            if TCGEN:
                gl.nvidia.ampere.async_copy.wait_group(0)
                gl.barrier()
                for stage in gl.static_range(2):
                    gl.nvidia.blackwell.mbarrier.invalidate(mma_bar.index(stage))
            values = running.load(l)
            valid = all_offsets[None, :] < running_count[:, None]
            encoded = (values >> 32).to(gl.uint32)
            rank = gl.minimum(running_count, 512)
            exact = _decode_exact_threshold(encoded, valid, rank, G, 1024, NW)
            greater = valid & (encoded > exact[:, None])
            equal = valid & (encoded == exact[:, None])
            ngreater = gl.sum(greater.to(gl.int32), 1)
            # Each row has at most 1024 candidates, so two 16-bit counts
            # can share one scan without a carry between them.
            ranks = _decode_placement_prefix(
                greater.to(gl.int32) | (equal.to(gl.int32) << 16)
            )
            dest = gl.where(
                greater, (ranks & 0xFFFF) - 1, ngreater[:, None] + (ranks >> 16) - 1
            )
            gl.store(output, -1, owned[:, None])
            gl.barrier()
            gl.store(
                selected + rows[:, None] * 512 + gl.maximum(dest, 0),
                (values & 0xFFFFFFFF).to(gl.int32),
                owned[:, None] & (greater | equal) & (dest < 512),
            )
        remaining_rows &= ~owned


@gluon.constexpr_function
def _pair_sort_asm(merge):
    return _warp_sort_asm(merge).replace("mov.u32 lane, %tid.x;", "mov.u32 lane, $2;")


@gluon.constexpr_function
def _pair_exchange_asm(outputs, inputs):
    (values,) = outputs
    (base,) = inputs
    lines = [
        "{",
        ".reg .u32 thread, query, key, index, address, temporary;",
        "mov.u32 thread, %tid.x;",
        "shr.u32 query, thread, 6;",
        "and.b32 query, query, 1;",
        "and.b32 key, thread, 31;",
        "shr.u32 temporary, thread, 5;",
        "and.b32 temporary, temporary, 1;",
        "mad.lo.u32 key, temporary, 32, key;",
    ]
    for item, value in enumerate(values):
        lines.extend(
            [
                f"add.u32 index, key, {item * 64};",
                "xor.b32 index, index, 32;",
                "mad.lo.u32 index, query, 128, index;",
                f"mad.lo.u32 address, index, 8, {base[0]};",
                f"ld.shared.b64 {value}, [address];",
            ]
        )
    lines.append("}")
    return "\n".join(lines)


@gluon.jit
def _pair_mma_scores(
    query,
    key_cache,
    page_table,
    first_row,
    tile,
    request0,
    request1,
    count0,
    count1,
    active0,
    active1,
    stride_q_n,
    stride_q_h,
    stride_q_d,
    stride_k_n,
    stride_k_d,
    stride_pt_b,
    DOUBLE_K: gl.constexpr,
    USE_INT32_OFFSETS: gl.constexpr,
):
    key_rows: gl.constexpr = 256 if DOUBLE_K else 128
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [2, 2], [1, 0])
    qlayout: gl.constexpr = gl.BlockedLayout([8, 1], [16, 2], [1, 4], [0, 1])
    dimensions = gl.arange(0, 128, layout=gl.SliceLayout(1, qlayout))
    heads = gl.arange(0, 8, layout=gl.SliceLayout(0, qlayout))
    head_owned = gl.where(heads < 4, active0, active1)
    qvalues = gl.load(
        query
        + (first_row + heads[None, :] // 4) * stride_q_n
        + heads[None, :] % 4 * stride_q_h
        + dimensions[:, None] * stride_q_d,
        head_owned[None, :],
        0,
    )
    qs = gl.allocate_shared_memory(
        gl.bfloat16, (128, 8), gl.NVMMASharedLayout(128, 16, transposed=True), qvalues
    )
    ks = gl.allocate_shared_memory(
        gl.bfloat16, (key_rows, 128), gl.NVMMASharedLayout(128, 16)
    )
    klayout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0])
    key_offsets = gl.arange(0, key_rows, layout=gl.SliceLayout(1, klayout))
    key_dimensions = gl.arange(0, 128, layout=gl.SliceLayout(0, klayout))
    key_ids = tile * 128 + key_offsets % 128
    if DOUBLE_K:
        owner_request = gl.where(key_offsets < 128, request0, request1)
        owner_count = gl.where(key_offsets < 128, count0, count1)
    else:
        owner0 = active0 & ((~active1) | (count0 >= count1))
        owner_request = gl.where(owner0, request0, request1)
        owner_count = gl.where(owner0, count0, count1)
    pages = gl.load(
        page_table + owner_request * stride_pt_b + key_ids // 64,
        key_ids < owner_count,
        0,
    ).to(gl.int32 if USE_INT32_OFFSETS else gl.int64)
    slots = pages * 64 + key_ids % 64
    valid_keys = key_ids < owner_count
    gl.nvidia.ampere.async_copy.async_load(
        ks,
        key_cache + slots[:, None] * stride_k_n + key_dimensions[None, :] * stride_k_d,
        mask=valid_keys[:, None],
    )
    gl.nvidia.ampere.async_copy.commit_group()
    tm = gl.nvidia.blackwell.allocate_tensor_memory(
        gl.float32, (key_rows, 8), gl.nvidia.blackwell.TensorMemoryLayout((128, 8), 1)
    )
    barrier = gl.allocate_shared_memory(
        gl.int64, (1,), gl.nvidia.blackwell.mbarrier.MBarrierLayout()
    )
    gl.nvidia.blackwell.mbarrier.init(barrier, count=1)
    gl.nvidia.ampere.async_copy.wait_group(0)
    gl.barrier()
    gl.nvidia.hopper.fence_async_shared()
    gl.nvidia.blackwell.tcgen05_mma(ks, qs, tm, use_acc=False, mbarriers=[barrier])
    gl.nvidia.blackwell.mbarrier.wait(barrier, 0, deps=(ks, qs, tm))
    matrix = tm.load()
    if DOUBLE_K:
        matrix_rows = gl.arange(0, 256, layout=gl.SliceLayout(1, matrix.type.layout))
        matrix_heads = gl.arange(0, 8, layout=gl.SliceLayout(0, matrix.type.layout))
        diagonal = matrix_rows[:, None] // 128 == matrix_heads[None, :] // 4
        scores = gl.reshape(
            gl.sum(gl.where(diagonal, gl.maximum(matrix, 0), 0), 1), (2, 128)
        )
    else:
        scores = gl.permute(
            gl.sum(gl.reshape(gl.maximum(matrix, 0), (128, 2, 4)), 2), (1, 0)
        )
    scores = gl.convert_layout(scores, layout)
    gl.nvidia.blackwell.mbarrier.invalidate(barrier)
    return scores


@gluon.jit(
    do_not_specialize=[
        "num_tiles",
        "num_blocks",
        "num_rows",
        "stride_q_n",
        "stride_q_h",
        "stride_pt_b",
    ]
)
def _qsa_native_pair_score_candidates_kernel(
    query,
    key_cache,
    page_table,
    request_indices,
    complete_blocks,
    candidates,
    discarded,
    merge_frontiers,
    num_heads,
    head_dim,
    num_blocks,
    page_size: gl.constexpr,
    stride_q_n,
    stride_q_h,
    stride_q_d,
    stride_k_n,
    stride_k_d,
    stride_pt_b,
    num_tiles,
    num_rows,
    BLOCK_H: gl.constexpr,
    BLOCK_D: gl.constexpr,
    BLOCK_TOPK: gl.constexpr,
    KEEP: gl.constexpr,
    ENABLE_PDL: gl.constexpr,
    USE_INT32_OFFSETS: gl.constexpr,
):
    gl.static_assert(page_size == 64)
    gl.static_assert(gl.num_warps() == 4)
    if ENABLE_PDL:
        gl.inline_asm_elementwise(
            "griddepcontrol.wait; mov.u32 $0, 0;",
            "=r",
            [],
            dtype=gl.int32,
            is_pure=False,
            pack=1,
        )
        gl.inline_asm_elementwise(
            "griddepcontrol.launch_dependents; mov.u32 $0, 0;",
            "=r",
            [],
            dtype=gl.int32,
            is_pure=False,
            pack=1,
        )
    first_row = gl.program_id(0) * 2
    tile = gl.program_id(1)
    count0 = gl.minimum(gl.maximum(gl.load(complete_blocks + first_row), 0), num_blocks)
    count1 = gl.minimum(
        gl.maximum(
            gl.load(complete_blocks + first_row + 1, first_row + 1 < num_rows, 0), 0
        ),
        num_blocks,
    )
    active0 = (count0 > BLOCK_TOPK) & (tile * 128 < count0)
    active1 = (first_row + 1 < num_rows) & (count1 > BLOCK_TOPK) & (tile * 128 < count1)
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [2, 2], [1, 0])
    row_layout: gl.constexpr = gl.SliceLayout(1, layout)
    column_layout: gl.constexpr = gl.SliceLayout(0, layout)
    row_offsets = gl.arange(0, 2, layout=row_layout)
    rows = first_row + row_offsets
    counts = gl.where(row_offsets == 0, count0, count1)
    active = gl.where(row_offsets == 0, active0, active1)
    if tile == 0:
        gl.store(
            merge_frontiers + rows, gl.where(counts > 8192, counts, 0), rows < num_rows
        )
    columns = gl.arange(0, 128, layout=column_layout)
    inactive_long = (rows < num_rows) & (~active) & (counts > 8192)
    gl.store(
        candidates + rows[:, None] * num_tiles * KEEP + tile * KEEP + columns[None, :],
        gl.full((2, 128), 0, gl.uint64, layout),
        inactive_long[:, None] & (columns[None, :] < KEEP),
    )
    gl.store(discarded + rows * num_tiles + tile, 0, inactive_long)
    if not (active0 | active1):
        return
    request0 = gl.load(request_indices + first_row).to(
        gl.int32 if USE_INT32_OFFSETS else gl.int64
    )
    request1 = gl.load(request_indices + first_row + 1, first_row + 1 < num_rows, 0).to(
        gl.int32 if USE_INT32_OFFSETS else gl.int64
    )
    same_pages = request0 == request1
    if active0 & active1 & (~same_pages):
        page0 = gl.load(page_table + request0 * stride_pt_b + tile * 2)
        page1 = gl.load(page_table + request1 * stride_pt_b + tile * 2)
        both_second = gl.minimum(count0, count1) > tile * 128 + 64
        next0 = gl.load(
            page_table + request0 * stride_pt_b + tile * 2 + 1, both_second, 0
        )
        next1 = gl.load(
            page_table + request1 * stride_pt_b + tile * 2 + 1, both_second, 0
        )
        same_pages = (page0 == page1) & (next0 == next1)
    shared = same_pages | (~(active0 & active1))
    if shared:
        scores = _pair_mma_scores(
            query,
            key_cache,
            page_table,
            first_row,
            tile,
            request0,
            request1,
            count0,
            count1,
            active0,
            active1,
            stride_q_n,
            stride_q_h,
            stride_q_d,
            stride_k_n,
            stride_k_d,
            stride_pt_b,
            DOUBLE_K=False,
            USE_INT32_OFFSETS=USE_INT32_OFFSETS,
        )
    else:
        scores = _pair_mma_scores(
            query,
            key_cache,
            page_table,
            first_row,
            tile,
            request0,
            request1,
            count0,
            count1,
            active0,
            active1,
            stride_q_n,
            stride_q_h,
            stride_q_d,
            stride_k_n,
            stride_k_d,
            stride_pt_b,
            DOUBLE_K=True,
            USE_INT32_OFFSETS=USE_INT32_OFFSETS,
        )
    owned = active
    ids = tile * 128 + columns
    packed = ((scores.to(gl.uint32, bitcast=True).to(gl.uint64) + 1) << 32) | ids[
        None, :
    ].to(gl.uint64)
    packed = gl.where(owned[:, None] & (ids[None, :] < counts[:, None]), packed, 0)
    logical_columns = columns[None, :] + gl.full((2, 128), 0, gl.int32, layout)
    first_pass = gl.inline_asm_elementwise(
        _pair_sort_asm(False),
        "=l,l,r",
        [packed, logical_columns],
        dtype=gl.uint64,
        is_pure=True,
        pack=1,
    )
    exchange = gl.allocate_shared_memory(
        gl.uint64, (2, 128), gl.SwizzledSharedLayout(1, 1, 1, [1, 0]), first_pass
    )
    gl.barrier()

    paired = gl.inline_asm(
        _pair_exchange_asm,
        "=l,=l,r,~{memory}",
        (exchange,),
        result_types=gl.distributed_type(gl.uint64, (2, 128), layout),
        is_pure=False,
    )
    gl.barrier()
    merged = gl.where(
        (columns[None, :] & 32) == 0,
        gl.maximum(first_pass, paired),
        gl.minimum(first_pass, paired),
    )
    sorted_keys = gl.inline_asm_elementwise(
        _pair_sort_asm(True),
        "=l,l,r",
        [merged, logical_columns],
        dtype=gl.uint64,
        is_pure=True,
        pack=1,
    )
    effective_keep = gl.where(counts <= 4096, 64, KEEP)
    per_group = effective_keep // 2
    offsets = columns[None, :] // 64 * per_group[:, None] + columns[None, :] % 64
    gl.store(
        candidates
        + rows[:, None] * num_tiles * KEEP
        + tile * effective_keep[:, None]
        + offsets,
        sorted_keys,
        owned[:, None] & (columns[None, :] % 64 < per_group[:, None]),
    )
    upper = gl.max(
        gl.where(columns[None, :] % 64 == per_group[:, None], sorted_keys >> 32, 0), 1
    )
    gl.store(discarded + rows * num_tiles + tile, upper.to(gl.uint32), owned)
    gl.barrier()


@gluon.constexpr_function
def _merge_sparse_constraints(count):
    return ",".join(["l", "l", "r", "r"] + ["r"] * count + ["~{memory}"])


@gluon.constexpr_function
def _merge_sparse_asm(outputs, inputs, packed_output):
    source, output, boundary, threads, destinations = inputs
    lines = [
        "{",
        ".reg .u32 thread, index, slot, address, id, score;",
        ".reg .u64 source_address, output_address, value;",
        ".reg .pred active, above, equal;",
        "mov.u32 thread, %tid.x;",
    ]
    for item, destination in enumerate(destinations):
        lines.extend(
            [
                f"setp.ge.s32 active, {destination}, 0;",
                f"setp.lt.s32 above, {destination}, 512;",
                "and.pred above, above, active;",
                f"setp.ge.s32 equal, {destination}, 512;",
                f"mad.lo.u32 index, {threads[0]}, {item}, thread;",
                f"mad.wide.u32 source_address, index, 8, {source[0]};",
                "@active ld.global.u64 value, [source_address];",
            ]
        )
        if packed_output:
            lines.extend(
                [
                    f"mad.wide.u32 output_address, {destination}, 8, {output[0]};",
                    "@above st.global.u64 [output_address], value;",
                ]
            )
        else:
            lines.extend(
                [
                    "mov.b64 {id, score}, value;",
                    f"mad.wide.u32 output_address, {destination}, 4, {output[0]};",
                    "@above st.global.u32 [output_address], id;",
                ]
            )
        lines.extend(
            [
                f"sub.u32 slot, {destination}, 512;",
                f"mad.lo.u32 address, slot, 8, {boundary[0]};",
                "@equal st.shared.u64 [address], value;",
            ]
        )
    lines.append("}")
    return "\n".join(lines)


@gluon.constexpr_function
def _merge_sparse_ids_asm(outputs, inputs):
    return _merge_sparse_asm(outputs, inputs, False)


@gluon.constexpr_function
def _merge_sparse_packed_asm(outputs, inputs):
    return _merge_sparse_asm(outputs, inputs, True)


@gluon.jit
def _merge_small_boundary(
    source,
    output,
    bits,
    valid,
    base,
    digit,
    topk,
    remaining,
    N: gl.constexpr,
    NW: gl.constexpr,
    PACKED_OUTPUT: gl.constexpr,
):
    flat: gl.constexpr = gl.BlockedLayout([1], [32], [NW], [0])
    buckets = gl.maximum((bits >> 16).to(gl.int32) - base, 0)
    above = valid & (buckets > digit)
    equal = valid & (buckets == digit)
    # Both counters are bounded by512 or by the32-entry boundary.
    ranks = _merge_prefix(above.to(gl.int32) | (equal.to(gl.int32) << 16), N, NW)
    destinations = gl.where(
        above, (ranks & 0xFFFF) - 1, gl.where(equal, 512 + (ranks >> 16) - 1, -1)
    )
    gl.barrier()
    boundary = gl.allocate_shared_memory(
        gl.uint64,
        (32,),
        gl.SwizzledSharedLayout(1, 1, 1, [0]),
        gl.full((32,), 0, gl.uint64, flat),
    )
    gl.barrier()
    if PACKED_OUTPUT:
        gl.inline_asm(
            _merge_sparse_packed_asm,
            _merge_sparse_constraints(N // (NW * 32)),
            (source, output, boundary, NW * 32, destinations),
            result_types=(),
            is_pure=False,
        )
    else:
        gl.inline_asm(
            _merge_sparse_ids_asm,
            _merge_sparse_constraints(N // (NW * 32)),
            (source, output, boundary, NW * 32, destinations),
            result_types=(),
            is_pure=False,
        )
    gl.barrier()
    packed = boundary.load(flat)
    # All shared readers finish before a later reduction reuses this storage.
    gl.barrier()
    columns = gl.arange(0, 32, layout=flat)
    ordered = gl.inline_asm_elementwise(
        _pair_sort_asm(False),
        "=l,l,r",
        [packed, columns],
        dtype=gl.uint64,
        is_pure=True,
        pack=1,
    )
    destination = topk - remaining + columns
    if PACKED_OUTPUT:
        gl.store(output + destination, ordered, columns < remaining)
    else:
        gl.store(
            output + destination,
            (ordered & 0xFFFFFFFF).to(gl.int32),
            columns < remaining,
        )
    return gl.max(
        gl.where(columns == remaining - 1, (ordered >> 32).to(gl.uint32), 0), 0
    )


@gluon.jit(do_not_specialize=["num_tiles", "num_parts", "num_blocks"])
def _qsa_partial_merge_candidates_kernel(
    candidates,
    discarded,
    complete,
    partials,
    num_tiles,
    num_parts,
    num_blocks,
    KEEP: gl.constexpr,
    NW: gl.constexpr,
    ENABLE_PDL: gl.constexpr,
):
    gl.static_assert(KEEP == 32)
    if ENABLE_PDL:
        gl.inline_asm_elementwise(
            "griddepcontrol.wait; mov.u32 $0, 0;",
            "=r",
            [],
            dtype=gl.int32,
            is_pure=False,
            pack=1,
        )
        gl.inline_asm_elementwise(
            "griddepcontrol.launch_dependents; mov.u32 $0, 0;",
            "=r",
            [],
            dtype=gl.int32,
            is_pure=False,
            pack=1,
        )
    row = gl.program_id(0)
    part = gl.program_id(1)
    count = gl.minimum(gl.maximum(gl.load(complete + row), 0), num_blocks)
    if count <= 8192:
        return
    flat: gl.constexpr = gl.BlockedLayout([1], [32], [NW], [0])
    remainder = count % 128
    retained = (
        count // 128 * KEEP
        + gl.minimum(remainder, KEEP // 2)
        + gl.minimum(gl.maximum(remainder - 64, 0), KEEP // 2)
    )
    valid_count = gl.minimum(gl.maximum(retained - part * 4096, 0), 4096)
    local_keep = gl.where(count >= 98304, 128, 512)
    output = partials + row * num_parts * 512 + part * local_keep
    offsets = gl.arange(0, 512, layout=flat)
    gl.store(
        output + offsets, gl.full((512,), 0, gl.uint64, flat), offsets < local_keep
    )
    cutoff = gl.full((), 0, gl.uint32, gl.SliceLayout(0, flat))
    enough = gl.full((), True, gl.int1, gl.SliceLayout(0, flat))
    if valid_count > 0:
        target = gl.minimum(valid_count, local_keep)
        cutoff, enough = _merge_coarse_select(
            candidates + row * num_tiles * KEEP + part * 4096,
            output,
            valid_count,
            valid_count,
            target,
            4096,
            NW,
            PACKED_OUTPUT=True,
        )
    # Each part reads and updates only its own original discard group.
    # The final max therefore includes both original and new discarded keys.
    original_tiles = part * 128 + gl.arange(0, 128, layout=flat)
    upper = gl.load(
        discarded + row * num_tiles + original_tiles, original_tiles < num_tiles, 0
    )
    combined = gl.maximum(
        gl.max(upper, 0), gl.where(valid_count > local_keep, cutoff, 0)
    )
    combined = gl.where(enough, combined, 0xFFFFFFFF).to(gl.uint32)
    gl.store(discarded + row * num_tiles + part * 128, combined)


@gluon.jit
def _qsa_tma_prefetch_pages(
    key_desc,
    page_table,
    request,
    start,
    max_count,
    stride_pt_b,
    destination,
    copy_bar,
    BN: gl.constexpr,
    USE_INT32_OFFSETS: gl.constexpr,
):
    gl.nvidia.hopper.mbarrier.expect(copy_bar, BN * 128 * 2)
    request = request.to(gl.int32 if USE_INT32_OFFSETS else gl.int64)
    for part in gl.static_range(BN // 64):
        logical = start + part * 64
        physical = gl.load(
            page_table + request * stride_pt_b + logical // 64, logical < max_count, 0
        )
        # An invalid logical page still transfers its full byte count from
        # reserved physical page zero. Frontier masks discard those key rows.
        gl.nvidia.hopper.tma.async_load(
            key_desc,
            [physical * 64, 0],
            copy_bar,
            destination.slice(part * 64, 64, dim=0),
        )


@gluon.jit
def _qsa_named_score_barrier(barrier_id, THREADS: gl.constexpr, ARRIVE: gl.constexpr):
    """Join both warp partitions while allowing the sender to continue."""
    # Warp specialization reserves IDs 0..2. Each of the two score slots has
    # a ready barrier (3/4) and a consumed barrier (5/6); all CTA threads arrive.
    if ARRIVE:
        gl.inline_asm_elementwise(
            f"barrier.cta.arrive $1, {THREADS}; mov.u32 $0, 0;",
            "=r,r,~{memory}",
            [barrier_id],
            dtype=gl.int32,
            is_pure=False,
            pack=1,
        )
    else:
        gl.inline_asm_elementwise(
            f"barrier.cta.sync $1, {THREADS}; mov.u32 $0, 0;",
            "=r,r,~{memory}",
            [barrier_id],
            dtype=gl.int32,
            is_pure=False,
            pack=1,
        )


@gluon.jit
def _stream_prefill_coarse(bits):
    # Six mantissa bits retain more precise complete bins without converting
    # scores to FP16. The original FP32 bits still decide the final boundary.
    return (bits >> 17).to(gl.int32)


@gluon.constexpr_function
def _soa_low_store_asm(outputs, inputs):
    base, offsets, values = inputs
    lines = ["{", ".reg .u32 address;", ".reg .u16 low;", ".reg .pred active;"]
    for offset, value in zip(offsets, values):
        lines.extend(
            [
                f"setp.ge.s32 active, {offset}, 0;",
                f"mad.lo.u32 address, {offset}, 2, {base[0]};",
                f"cvt.u16.u32 low, {value};",
                "@active st.shared.u16 [address], low;",
            ]
        )
    lines.append("}")
    return "\n".join(lines)
