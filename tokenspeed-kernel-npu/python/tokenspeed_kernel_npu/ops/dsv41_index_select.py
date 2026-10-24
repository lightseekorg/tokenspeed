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


"""Exact native TopK over order-preserving keys with a device-side fallback."""

import torch
from tokenspeed_kernel_npu._triton import tl, triton


@triton.jit
def _keys(values):
    bits = values.to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32) & 65535
    return tl.where((bits & 32768) != 0, bits ^ 65535, bits ^ 32768)


@triton.jit(do_not_specialize=["SS", "BS", "VS", "WIDTH"])
def _block_maxima(
    Scores,
    Visible,
    Blocks,
    SS,
    BS,
    VS,
    WIDTH,
    TILE: tl.constexpr,
    TORCH_SEMANTICS: tl.constexpr,
):
    row = tl.program_id(0)
    width = tl.minimum(tl.maximum(tl.load(Visible + row * VS), 0), WIDTH)
    cols = tl.arange(0, TILE)
    lanes = tl.arange(0, 8)
    latest = -1
    for start in range(0, tl.cdiv(width, 8), TILE):
        block = start + cols
        offsets = block[:, None] * 8 + lanes[None, :]
        vals = tl.load(
            Scores + row * SS + offsets, offsets < width, other=-float("inf")
        )
        if TORCH_SEMANTICS:
            maxima = tl.max(vals, 1)
        else:
            maxima = tl.max(tl.where(vals > -float("inf"), vals, -float("inf")), 1)
        valid = block * 8 < width
        tl.store(Blocks + row * BS + block, maxima, valid)
        resident = valid & (maxima > -float("inf"))
        if TORCH_SEMANTICS:
            # Match amax/isfinite: NaNs propagate and only finite blocks can
            # be forced. The ordered-key path instead forces the last valid one.
            resident &= maxima < float("inf")
        latest = tl.maximum(latest, tl.max(tl.where(resident, block, -1), 0))
    tl.store(Blocks + row * BS + latest, float("inf"), latest >= 0)


def block_scores_for_topk(scores, visible):
    """Compute eight-token maxima and force the last finite block for native TopK.

    Scores beyond each live extent are ignored. Unused output blocks stay -inf;
    NaNs, infinities and the subsequent native TopK tie behavior are unchanged.
    """
    n, width = scores.shape
    if width % 8:
        raise ValueError("Index score storage must contain complete eight-token blocks")
    # Isolate each program's masked stores, including short rows, within a
    # complete output tile so neighboring programs cannot overwrite its tail.
    stride = triton.cdiv(width // 8, 128) * 128
    output = torch.full(
        (n, stride), -torch.inf, device=scores.device, dtype=scores.dtype
    )[:, : width // 8]
    if n and width:
        _block_maxima[(n,)](
            scores,
            visible,
            output,
            scores.stride(0),
            output.stride(0),
            visible.stride(0),
            width,
            128,
            True,
            num_warps=1,
            multibuffer=False,
        )
    return output


@triton.jit
def _seed_keys(Scores, Visible, Seed, SS, VS, KS, WIDTH, CAP, TILE: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, TILE)
    width = tl.minimum(tl.maximum(tl.load(Visible + row * VS), 0), WIDTH)
    vals = tl.load(
        Scores + row * SS + col, (col < CAP) & (col < width), other=-float("inf")
    )
    key = tl.where(
        (col < width) & (vals > -float("inf")),
        _keys(vals).to(tl.float32),
        -float("inf"),
    )
    tl.store(Seed + row * KS + col, key, col < CAP)


@triton.jit
def _encode_keys(
    Scores,
    Visible,
    Floor,
    Keys,
    Fast,
    SS,
    VS,
    KS,
    WIDTH,
    CAP,
    UNIT: tl.constexpr,
    COL_BITS: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0)
    width = tl.cdiv(tl.minimum(tl.maximum(tl.load(Visible + row * VS), 0), WIDTH), UNIT)
    lane = tl.arange(0, TILE)
    # Offset the 16-bit score key so score and position fit in a normal FP32
    # bit pattern. These are sorting keys, never model arithmetic.
    floor = 0
    fast = width <= CAP
    if UNIT == 1:
        seed_floor = tl.load(Floor + row)
        floor = tl.where(seed_floor > -float("inf"), seed_floor, 0).to(tl.int32)
        largest = 0
        for start in range(0, tl.minimum(width, CAP), TILE):
            col = start + lane
            vals = tl.load(Scores + row * SS + col, col < width, other=-float("inf"))
            largest = tl.maximum(
                largest,
                tl.max(
                    tl.where((col < width) & (vals > -float("inf")), _keys(vals), 0), 0
                ),
            )
        # The sampled kth key is a lower bound on the global kth key.
        # A wide range or too few sampled valid entries takes the exact fallback.
        fast = (
            fast
            & (seed_floor > -float("inf"))
            & (largest - floor < (0xFE000000 >> COL_BITS))
        )
    tl.store(Fast + row * 16, fast.to(tl.int32))
    for start in range(0, CAP, TILE):
        col = start + lane
        vals = tl.load(
            Scores + row * SS + col, (col < CAP) & (col < width), other=-float("inf")
        )
        key = _keys(vals)
        # Enumerate negative normals followed by positive normals. Skip both
        # subnormal intervals (hardware may flush them) and all NaN/Inf codes.
        # This provides 0xFE000000 unique ordered keys, including score ranges
        # crossing zero, without perturbing any actual model score.
        rank = ((key - floor) << COL_BITS) + ((1 << COL_BITS) - 1 - col)
        # Signed int32 arithmetic preserves the bits without an unsupported
        # uint32-to-uint64 promotion in the Ascend compiler.
        bits = tl.where(
            (rank >= 0) & (rank < 0x7F000000),
            -8388609 - rank,
            rank - 0x7E800000,
        ).to(tl.int32)
        encoded = bits.to(tl.float32, bitcast=True)
        valid = fast & (col < width) & (vals > -float("inf")) & (key >= floor)
        tl.store(
            Keys + row * KS + col, tl.where(valid, encoded, -float("inf")), col < CAP
        )


@triton.jit
def _gather_selected(
    Scores,
    NativeValues,
    NativeIds,
    Fast,
    Values,
    Ids,
    SS,
    NS,
    OS,
    NK,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    nv = tl.load(NativeValues + row * NS + col, col < NK, other=-float("inf"))
    ids = tl.load(NativeIds + row * NS + col, col < NK, other=0).to(tl.int32)
    valid = (tl.load(Fast + row * 16) != 0) & (col < NK) & (nv > -float("inf"))
    vals = tl.load(Scores + row * SS + ids, valid, other=-float("inf"))
    tl.store(Values + row * OS + col, vals, col < K)
    tl.store(Ids + row * OS + col, tl.where(valid, ids, -1), col < K)


@triton.jit
def _select_fallback(
    Fast,
    Scores,
    Visible,
    Values,
    Ids,
    SS,
    VS,
    OS,
    WIDTH,
    K: tl.constexpr,
    UNIT: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0)
    if tl.load(Fast + row * 16) == 0:
        width = tl.cdiv(
            tl.minimum(tl.maximum(tl.load(Visible + row * VS), 0), WIDTH), UNIT
        )
        lane = tl.arange(0, TILE)
        output_col = tl.arange(0, triton.next_power_of_2(K))
        tl.store(Values + row * OS + output_col, -float("inf"), output_col < K)
        tl.store(Ids + row * OS + output_col, -1, output_col < K)
        threshold = 0
        if width > K:
            # A BF16 score has only 16 ordered bits. Vector comparisons and sums
            # avoid Ascend's expensive per-element histogram updates. All bounds
            # remain device values and can change between graph replays.
            for step in range(16):
                trial = threshold + (1 << (15 - step))
                count = 0
                for start in range(0, width, TILE):
                    col = start + lane
                    vals = tl.load(
                        Scores + row * SS + col, col < width, other=-float("inf")
                    )
                    valid = (col < width) & (vals > -float("inf"))
                    count += tl.sum((valid & (_keys(vals) >= trial)).to(tl.int32), 0)
                threshold = tl.where(count >= K, trial, threshold)
        above = 0
        for start in range(0, width, TILE):
            col = start + lane
            vals = tl.load(Scores + row * SS + col, col < width, other=-float("inf"))
            valid = (col < width) & (vals > -float("inf"))
            above += tl.sum((valid & (_keys(vals) > threshold)).to(tl.int32), 0)
        strong_count = 0
        equal_count = 0
        for start in range(0, width, TILE):
            col = start + lane
            vals = tl.load(Scores + row * SS + col, col < width, other=-float("inf"))
            keys = _keys(vals)
            valid = (col < width) & (vals > -float("inf"))
            strong = valid & (keys > threshold)
            equal = valid & (keys == threshold)
            sp = strong_count + tl.cumsum(strong.to(tl.int32), 0) - 1
            ep = equal_count + tl.cumsum(equal.to(tl.int32), 0) - 1
            selected = strong | (equal & (ep < K - above))
            pos = tl.where(strong, sp, above + ep)
            tl.store(Values + row * OS + pos, vals, selected)
            tl.store(Ids + row * OS + pos, col, selected)
            strong_count += tl.sum(strong.to(tl.int32), 0)
            equal_count += tl.sum(equal.to(tl.int32), 0)


def select_full_scores(scores, visible, row_k, block_k, *, logical_capacity):
    """Return exact row/block TopK values and IDs for BF16-rounded scores.

    Native TopK sorts order-preserving FP32 bit patterns encoding score first,
    then earlier logical column. Original scores are gathered without changes.
    Device-side guards retain the full-width count selector for long histories
    or score ranges that cannot be encoded. Fixed buffers and live lengths
    support graph replay across both paths without host synchronization.
    """
    n = scores.shape[0]
    width = min(logical_capacity, scores.shape[1])
    outputs = []
    for unit, k in ((1, row_k), (8, block_k)):
        stride = triton.cdiv(k, 16) * 16
        values = torch.full(
            (n, stride), -torch.inf, device=scores.device, dtype=torch.float32
        )[:, :k]
        ids = torch.full((n, stride), -1, device=scores.device, dtype=torch.int32)[
            :, :k
        ]
        outputs.append((values, ids))
        if not n or not width or not k:
            continue
        source = scores
        if unit == 8:
            capacity = triton.cdiv(width, unit)
            source = torch.empty(
                (n, triton.cdiv(capacity, 256) * 256),
                device=scores.device,
                dtype=torch.float32,
            )[:, :capacity]
            _block_maxima[(n,)](
                scores,
                visible,
                source,
                scores.stride(0),
                source.stride(0),
                visible.stride(0),
                width,
                256,
                False,
                multibuffer=False,
            )
        cap = min(triton.cdiv(width, unit), 65536 if unit == 1 else 16384)
        floor = None
        if unit == 1:
            sample = min(cap, max(2048, k * 2))
            seed = torch.empty(
                (n, triton.cdiv(sample, 256) * 256),
                device=scores.device,
                dtype=torch.float32,
            )[:, :sample]
            _seed_keys[(n,)](
                source,
                visible,
                seed,
                source.stride(0),
                visible.stride(0),
                seed.stride(0),
                width,
                sample,
                triton.next_power_of_2(sample),
                multibuffer=False,
            )
            floor = torch.topk(seed, min(k, sample), dim=-1, sorted=False).values.amin(
                -1
            )
        keys = torch.empty(
            (n, triton.cdiv(cap, 256) * 256), device=scores.device, dtype=torch.float32
        )[:, :cap]
        fast = torch.empty((n, 16), device=scores.device, dtype=torch.int32)
        _encode_keys[(n,)](
            source,
            visible,
            floor,
            keys,
            fast,
            source.stride(0),
            visible.stride(0),
            keys.stride(0),
            width,
            cap,
            unit,
            16 if unit == 1 else 14,
            2048,
            multibuffer=False,
        )
        native_values, native_ids = torch.topk(keys, min(k, cap), dim=-1, sorted=False)
        _gather_selected[(n,)](
            source,
            native_values,
            native_ids,
            fast,
            values,
            ids,
            source.stride(0),
            native_values.stride(0),
            values.stride(0),
            min(k, cap),
            k,
            triton.next_power_of_2(k),
            multibuffer=False,
        )
        _select_fallback[(n,)](
            fast,
            source,
            visible,
            values,
            ids,
            source.stride(0),
            visible.stride(0),
            values.stride(0),
            width,
            k,
            unit,
            2048,
            multibuffer=False,
        )
    return tuple(outputs)
