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

"""GPU prefix and group128 quantization for FlashInfer's compact scale ABI."""

from tokenspeed_kernel._triton import tl, triton


@triton.jit
def _prefix(
    counts, offsets, capacity, compact_capacity, E: tl.constexpr, BLOCK: tl.constexpr
):
    idx = tl.arange(0, BLOCK)
    values = tl.load(counts + idx - 1, (idx > 0) & (idx <= E), 0)
    tl.device_assert((values >= 0) & (values <= capacity), "Invalid receive counts")
    tl.device_assert(tl.sum(values) <= compact_capacity, "Compact capacity exceeded")
    tl.store(offsets + idx, tl.cumsum(values), idx <= E)


@triton.jit
def _pack_quant(
    x,
    counts,
    offsets,
    q,
    sf,
    x_stride_e,
    x_stride_m,
    sf_stride_k,
    ROW_SPLITS: tl.constexpr,
    K: tl.constexpr,
):
    expert = tl.program_id(1)
    pack = tl.program_id(0) // ROW_SPLITS
    row = tl.program_id(0) % ROW_SPLITS
    start = tl.load(offsets + expert)
    sf_start = (start + expert * 3) // 4 * 4
    live = tl.load(counts + expert)
    cols = pack * 512 + tl.arange(0, 512)
    groups = pack * 4 + tl.arange(0, 4)
    while row < live:
        values = tl.load(
            x + expert.to(tl.int64) * x_stride_e + row.to(tl.int64) * x_stride_m + cols,
            cols < K,
            0,
        ).to(tl.float32)
        values = tl.reshape(values, (4, 128))
        raw = tl.maximum(tl.max(tl.abs(values), 1), 1.0e-4) / 448.0
        bits = raw.to(tl.int32, bitcast=True)
        exp = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
        scale = (exp << 23).to(tl.float32, bitcast=True)
        quantized = tl.minimum(tl.maximum(values / scale[:, None], -448.0), 448.0)
        tl.store(
            q + (start + row).to(tl.int64) * K + cols,
            tl.reshape(quantized, (512,)),
            cols < K,
        )
        word = tl.sum(
            tl.where(groups < K // 128, exp, 0).to(tl.uint32) << (tl.arange(0, 4) * 8)
        )
        tl.store(sf + pack.to(tl.int64) * sf_stride_k + sf_start + row, word)
        row += ROW_SPLITS


def prepare_compact(recv, counts, offsets, q, sf, *, row_splits):
    experts = recv.shape[0]
    _prefix[(1,)](
        counts,
        offsets,
        recv.shape[1],
        q.shape[0],
        E=experts,
        BLOCK=triton.next_power_of_2(experts + 1),
        debug=True,
    )
    _pack_quant[(((recv.shape[-1] // 128 + 3) // 4) * row_splits, experts)](
        recv,
        counts,
        offsets,
        q,
        sf,
        recv.stride(0),
        recv.stride(1),
        sf.stride(0),
        ROW_SPLITS=row_splits,
        K=recv.shape[-1],
        num_warps=4,
        num_stages=1,
    )


@triton.jit
def _branch_offsets(
    offsets, low, high, threshold, E: tl.constexpr, BLOCK: tl.constexpr
):
    idx = tl.arange(0, BLOCK)
    values = tl.load(offsets + idx, idx <= E, 0)
    total = tl.load(offsets + E)
    tl.store(low + idx, tl.where(total <= threshold, values, 0), idx <= E)
    tl.store(high + idx, tl.where(total > threshold, values, 0), idx <= E)


def branch_offsets(offsets, low, high, *, threshold):
    experts = offsets.numel() - 1
    _branch_offsets[(1,)](
        offsets,
        low,
        high,
        threshold,
        E=experts,
        BLOCK=triton.next_power_of_2(experts + 1),
    )


@triton.jit
def _pack_wire(
    x,
    sf_in,
    global_in,
    counts,
    offsets,
    q,
    sf_out,
    xe,
    xm,
    se,
    sm,
    sk,
    ge,
    gm,
    out_stride,
    K: tl.constexpr,
    FORMAT: tl.constexpr,
    ROW_SPLITS: tl.constexpr,
):
    expert = tl.program_id(1)
    pack = tl.program_id(0) // ROW_SPLITS
    row = tl.program_id(0) % ROW_SPLITS
    start = tl.load(offsets + expert)
    sf_start = (start + expert * 3) // 4 * 4
    live = tl.load(counts + expert)
    cols = pack * 512 + tl.arange(0, 512)
    groups = pack * 4 + tl.arange(0, 4)
    while row < live:
        base = expert.to(tl.int64) * xe + row.to(tl.int64) * xm
        if FORMAT == 2:
            code = tl.load(x + base + cols // 2, cols < K, 0).to(tl.uint32)
            code = (code >> ((cols % 2) * 4)) & 15
            mag = code & 7
            value = tl.where(
                mag < 4,
                mag * 0.5,
                tl.where(
                    mag == 4, 2.0, tl.where(mag == 5, 3.0, tl.where(mag == 6, 4.0, 6.0))
                ),
            )
            value = tl.where((code & 8) != 0, -value, value)
            scale = tl.load(
                sf_in
                + expert.to(tl.int64) * se
                + row.to(tl.int64) * sm
                + (cols // 16) * sk,
                cols < K,
                0.0,
            ).to(tl.float32)
            global_scale = tl.load(
                global_in + expert.to(tl.int64) * ge + row.to(tl.int64) * gm
            )
            value = (value * scale * global_scale).to(tl.bfloat16).to(tl.float32)
        else:
            value = tl.load(x + base + cols, cols < K, 0.0).to(tl.float32)
            if FORMAT == 1:
                scale = tl.load(
                    sf_in
                    + expert.to(tl.int64) * se
                    + row.to(tl.int64) * sm
                    + (cols // 128) * sk,
                    cols < K,
                    0.0,
                )
                value = (value * scale).to(tl.bfloat16).to(tl.float32)
        value = tl.reshape(value, (4, 128))
        raw = tl.maximum(tl.max(tl.abs(value), 1), 1e-4) / 448.0
        bits = raw.to(tl.int32, bitcast=True)
        exp = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
        scale = (exp << 23).to(tl.float32, bitcast=True)
        quant = tl.minimum(tl.maximum(value / scale[:, None], -448.0), 448.0)
        tl.store(
            q + (start + row).to(tl.int64) * K + cols,
            tl.reshape(quant, (512,)),
            cols < K,
        )
        word = tl.sum(
            tl.where(groups < K // 128, exp, 0).to(tl.uint32) << (tl.arange(0, 4) * 8)
        )
        tl.store(sf_out + pack.to(tl.int64) * out_stride + sf_start + row, word)
        row += ROW_SPLITS


def prepare_wire(
    recv, scales, globals_, counts, offsets, q, sf, *, row_splits, wire_dtype, capacity
):
    experts = counts.numel()
    _prefix[(1,)](
        counts,
        offsets,
        capacity,
        q.shape[0],
        E=experts,
        BLOCK=triton.next_power_of_2(experts + 1),
        debug=True,
    )
    ss = scales.stride() if scales.numel() else (0, 0, 0)
    gs = globals_.stride() if globals_.numel() else (0, 0)
    _pack_wire[(((q.shape[-1] // 128 + 3) // 4) * row_splits, experts)](
        recv,
        scales,
        globals_,
        counts,
        offsets,
        q,
        sf,
        recv.stride(0),
        recv.stride(1),
        *ss,
        *gs,
        sf.stride(0),
        K=q.shape[-1],
        FORMAT={"bf16": 0, "fp8": 1, "nvfp4": 2}[wire_dtype],
        ROW_SPLITS=row_splits,
        num_warps=4,
        num_stages=1,
    )
