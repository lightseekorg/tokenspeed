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

"""Row-major block-scaled expert GEMM with BF16 operand materialization."""

from typing import Any

from tokenspeed_kernel_amd._triton import cdna4_async_copy as async_copy
from tokenspeed_kernel_amd._triton import gl, gluon


def _expert_mm_metadata(
    grid: tuple[int, ...], kernel: Any, args: dict[str, Any]
) -> dict[str, str]:
    # Counts reside on the GPU. Do not report padded work as useful FLOPs.
    return {"name": kernel.name}


@gluon.jit
def _load_block(Values, Scales, rows, cols, mask, K: gl.constexpr, NVFP4: gl.constexpr):
    if NVFP4:
        packed = gl.load(Values + rows * (K // 2) + cols // 2, mask, other=0)
        code = (packed.to(gl.int32) >> ((cols % 2) * 4)) & 15
        magnitude = code & 7
        value = gl.where(magnitude < 4, magnitude.to(gl.float32) * 0.5, 2.0)
        value = gl.where(magnitude == 5, 3.0, value)
        value = gl.where(magnitude == 6, 4.0, value)
        value = gl.where(magnitude == 7, 6.0, value)
        value = gl.where((code & 8) != 0, -value, value)
        scale = gl.load(Scales + rows * (K // 16) + cols // 16, mask, other=0.0)
        value = value * scale.to(gl.float32)
    else:
        data = gl.load(Values + rows * K + cols, mask, other=0.0).to(gl.float32)
        exponent = gl.load(Scales + rows * (K // 32) + cols // 32, mask, other=0)
        exponent = exponent.to(gl.int32)
        scale = gl.maximum((exponent << 23).to(gl.float32, bitcast=True), 2.0**-127)
        scale = gl.where(exponent == 255, float("nan"), scale)
        value = data * scale
    return value.to(gl.bfloat16)


@gluon.constexpr_function
def _nvfp4_weight_layouts(BN, BK):
    """Register layouts of a ``[BN, BK]`` NVFP4 weight tile for 4 warps.

    Returns ``(load, packed, upcast, grouped, scale)``:

    - ``load``: the ``[BN, BK // 2]`` packed bytes as loaded. A lane reads
      16 contiguous bytes, so 4 lanes read 64 contiguous bytes of a row.
    - ``packed``: the same bytes where a lane holds the 4 bytes of each MFMA
      K step that the B operand (16x16x32, 8 values a lane) needs. Moving
      ``load`` to ``packed`` exchanges lane bits with register bits inside a
      warp: no barrier.
    - ``upcast``: ``[BN, BK]`` values after the FP4 upcast of ``packed``,
      which is the B operand layout itself.
    - ``grouped``: ``upcast`` viewed as ``[BN, BK // 16, 16]`` scale groups.
    - ``scale``: ``[BN, BK // 16]`` E4M3 group scales as loaded, a whole row's
      scales in each lane; selecting a lane's groups needs no data exchange.
    """
    assert BN % 64 == 0 and BK % 128 == 0, (BN, BK)
    rows = [[1, 0], [2, 0], [4, 0], [8, 0]]
    warps = [[16, 0], [32, 0]]
    more_rows = [[1 << bit, 0] for bit in range(6, BN.bit_length() - 1)]
    # Packed bytes from 64 on sit in registers in both byte layouts.
    more_bytes = [[0, 1 << bit] for bit in range(6, (BK // 2).bit_length() - 1)]
    load = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [0, 8]] + more_bytes + more_rows,
        lane_bases=rows + [[0, 16], [0, 32]],
        warp_bases=warps,
        block_bases=[],
        shape=[BN, BK // 2],
    )
    packed = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 16], [0, 32]] + more_bytes + more_rows,
        lane_bases=rows + [[0, 4], [0, 8]],
        warp_bases=warps,
        block_bases=[],
        shape=[BN, BK // 2],
    )
    upcast = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [0, 32], [0, 64]]
        + [[0, 2 * byte[1]] for byte in more_bytes]
        + more_rows,
        lane_bases=rows + [[0, 8], [0, 16]],
        warp_bases=warps,
        block_bases=[],
        shape=[BN, BK],
    )
    grouped = gl.DistributedLinearLayout(
        reg_bases=[[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 2, 0], [0, 4, 0]]
        + [[0, byte[1] // 8, 0] for byte in more_bytes]
        + [[row[0], 0, 0] for row in more_rows],
        lane_bases=[[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [0, 0, 8], [0, 1, 0]],
        warp_bases=[[16, 0, 0], [32, 0, 0]],
        block_bases=[],
        shape=[BN, BK // 16, 16],
    )
    scale = gl.DistributedLinearLayout(
        reg_bases=[[0, 1 << bit] for bit in range((BK // 16).bit_length() - 1)]
        + more_rows,
        lane_bases=rows + [[0, 0], [0, 0]],
        warp_bases=warps,
        block_bases=[],
        shape=[BN, BK // 16],
    )
    return load, packed, upcast, grouped, scale


@gluon.jit
def _load_nvfp4_tile(
    W, WS, row0, k0, rows_left, K: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr
):
    """Load a ``[BN, BK]`` NVFP4 weight tile of rows ``row0 + [0, BN)``: its
    packed bytes and E4M3 group scales, for ``_upcast_nvfp4_tile``. Rows at
    or past ``rows_left`` and columns at or past ``K`` read 0. Weights are
    read once a step, so they skip L1."""
    load_layout: gl.constexpr = _nvfp4_weight_layouts(BN, BK)[0]
    scale_layout: gl.constexpr = _nvfp4_weight_layouts(BN, BK)[4]
    rows = gl.arange(0, BN, layout=gl.SliceLayout(1, load_layout))
    cols = gl.arange(0, BK // 2, layout=gl.SliceLayout(0, load_layout))
    packed = gl.load(
        W + (row0 + rows.to(gl.int64))[:, None] * (K // 2) + (k0 // 2 + cols)[None, :],
        (rows < rows_left)[:, None] & (k0 // 2 + cols < K // 2)[None, :],
        other=0,
        cache_modifier=".cg",
    )
    srows = gl.arange(0, BN, layout=gl.SliceLayout(1, scale_layout))
    scols = gl.arange(0, BK // 16, layout=gl.SliceLayout(0, scale_layout))
    scale = gl.load(
        WS
        + (row0 + srows.to(gl.int64))[:, None] * (K // 16)
        + (k0 // 16 + scols)[None, :],
        (srows < rows_left)[:, None] & (k0 // 16 + scols < K // 16)[None, :],
        other=0.0,
        cache_modifier=".cg",
    )
    return packed, scale


@gluon.jit
def _upcast_nvfp4_tile(packed, scale, BN: gl.constexpr, BK: gl.constexpr):
    """``[BN, BK]`` BF16 weights of a ``_load_nvfp4_tile`` tile, in the MFMA
    B operand layout (after a permute to ``[BK, BN]``).

    The native FP4 upcast runs under a unit E8M0 scale (127); the E4M3 group
    scale multiplies after it. Both steps are exact in BF16, so the values
    equal ``_load_block``'s.
    """
    layouts: gl.constexpr = _nvfp4_weight_layouts(BN, BK)
    packed = gl.convert_layout(packed, layouts[1])
    scale = gl.convert_layout(scale, gl.SliceLayout(2, layouts[3]))
    unit = gl.full((BN, BK), 127, gl.uint8, layouts[2])
    value = gl.amd.cdna4.scaled_upcast(packed, unit, gl.bfloat16, axis=1)
    value = gl.convert_layout(gl.reshape(value, [BN, BK // 16, 16]), layouts[3])
    value = (value.to(gl.float32) * scale.to(gl.float32)[:, :, None]).to(gl.bfloat16)
    return gl.reshape(value, [BN, BK])


@gluon.constexpr_function
def _nvfp4_word_layout(BN, BK):
    """``[BN, BK // 16, 2]`` int32 words of packed NVFP4 weights (8 values a
    word; the axes: row, scale group, word in the group) where a lane holds
    the word of each MFMA K step that the B operand (16x16x32, 8 values a
    lane) needs. Its slice along the last axis holds a lane's group scales.
    """
    assert BN % 64 == 0 and BK % 128 == 0, (BN, BK)
    more_rows = [[1 << bit, 0, 0] for bit in range(6, BN.bit_length() - 1)]
    return gl.DistributedLinearLayout(
        reg_bases=[[0, 1 << bit, 0] for bit in range(1, (BK // 16).bit_length() - 1)]
        + more_rows,
        lane_bases=[[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [0, 0, 1], [0, 1, 0]],
        warp_bases=[[16, 0, 0], [32, 0, 0]],
        block_bases=[],
        shape=[BN, BK // 16, 2],
    )


@gluon.jit
def _e2m1_pair(pair, scale):
    """One byte's two E2M1 values (an FP32 pair in an int64) times ``scale``
    (FP32), each product exact, rounded to BF16."""
    low = (pair & 0xFFFFFFFF).to(gl.int32).to(gl.float32, bitcast=True) * scale
    high = (pair >> 32).to(gl.int32).to(gl.float32, bitcast=True) * scale
    return low.to(gl.bfloat16), high.to(gl.bfloat16)


@gluon.jit
def _upcast_nvfp4_words(words, scale, BN: gl.constexpr, BK: gl.constexpr):
    """``[BN, BK]`` BF16 weights of ``_nvfp4_word_layout`` words and their
    ``[BN, BK // 16]`` E4M3 group scales, in the MFMA B operand layout
    (after a permute to ``[BK, BN]``). Equal to ``_upcast_nvfp4_tile``.

    Each ``v_cvt_scalef32_pk_f32_fp4`` turns one byte (op_sel picks it from
    the word) into an FP32 pair at unit scale; the E4M3 scale multiplies in
    FP32 and the product rounds once. Inline asm because Gluon's scaled
    upcast yields only BF16/FP16, which costs two more instructions a pair to
    unpack to FP32 for the multiply.
    """
    scale = gl.convert_layout(scale, gl.SliceLayout(2, _nvfp4_word_layout(BN, BK)))
    scale = scale.to(gl.float32)[:, :, None]
    byte0, byte1, byte2, byte3 = gl.inline_asm_elementwise(
        "v_cvt_scalef32_pk_f32_fp4 $0, $4, 1.0\n"
        "v_cvt_scalef32_pk_f32_fp4 $1, $4, 1.0 op_sel:[1,0,0]\n"
        "v_cvt_scalef32_pk_f32_fp4 $2, $4, 1.0 op_sel:[0,1,0]\n"
        "v_cvt_scalef32_pk_f32_fp4 $3, $4, 1.0 op_sel:[1,1,0]",
        "=&v,=&v,=&v,=&v,v",
        [words],
        dtype=(gl.int64, gl.int64, gl.int64, gl.int64),
        is_pure=True,
        pack=1,
    )
    low0, high0 = _e2m1_pair(byte0, scale)
    low1, high1 = _e2m1_pair(byte1, scale)
    low2, high2 = _e2m1_pair(byte2, scale)
    low3, high3 = _e2m1_pair(byte3, scale)
    # Value 2 j + h of a word is byte j's low (h = 0) or high nibble.
    low = gl.join(gl.join(low0, low2), gl.join(low1, low3))
    high = gl.join(gl.join(high0, high2), gl.join(high1, high3))
    return gl.reshape(gl.join(low, high), [BN, BK])


@gluon.constexpr_function
def _ring_layouts(BM, BN, BK):
    """Copy and LDS layouts of one stage of the NVFP4 ring (4 warps): packed
    weights as int32 words ``[BN, BK // 8]``, group scales ``[BN, BK // 16]``
    and BF16 activations ``[BM, BK]``. Each lane copies 16 bytes (4 for
    scales), rows split across lanes. Weight and activation rows are swizzled
    so that the operand reads (one row a lane) spread over the LDS banks."""
    assert BK % 128 == 0 and BN % 64 == 0 and BM % 8 == 0, (BM, BN, BK)
    return (
        gl.BlockedLayout([1, 4], [64 // (BK // 32), BK // 32], [4, 1], [1, 0]),
        gl.BlockedLayout([1, 4], [64 // (BK // 64), BK // 64], [4, 1], [1, 0]),
        gl.BlockedLayout([1, 8], [64 // (BK // 8), BK // 8], [4, 1], [1, 0]),
        gl.SwizzledSharedLayout(4, 1, 8, [1, 0]),
        gl.SwizzledSharedLayout(1, 1, 1, [1, 0]),
        gl.SwizzledSharedLayout(8, 1, 16, [1, 0]),
    )


@gluon.constexpr_function
def _ring_depth(BM, BN, BK):
    """Stages of the NVFP4 ring: 4 while two programs fit a CU's 160 KB of
    LDS (the decode tile: 17 KB a stage), else as many as one program fits,
    at least 2."""
    stage = BN * BK // 2 + BN * BK // 16 + BM * BK * 2
    if 4 * stage <= 80 * 1024:
        return 4
    return max(2, min(4, 160 * 1024 // stage))


@gluon.jit
def _ring_stages(WS, BM: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr):
    """The LDS ring of ``_nvfp4_ring_dot`` (``WS``: the group scales)."""
    layouts: gl.constexpr = _ring_layouts(BM, BN, BK)
    depth: gl.constexpr = _ring_depth(BM, BN, BK)
    w_ring = gl.allocate_shared_memory(gl.int32, [depth, BN, BK // 8], layouts[3])
    s_ring = gl.allocate_shared_memory(
        WS.dtype.element_ty, [depth, BN, BK // 16], layouts[4]
    )
    x_ring = gl.allocate_shared_memory(gl.bfloat16, [depth, BM, BK], layouts[5])
    return w_ring, s_ring, x_ring


@gluon.jit
def _ring_weights(W, WS, w_row0, K: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr):
    """Per-lane copy addresses of K block 0 of weight rows ``w_row0 + [0,
    BN)``: packed words and group scales. Per-lane addresses, not buffer
    descriptors: the tile's base is not provably uniform, and a descriptor in
    vector registers costs a loop per copy."""
    layouts: gl.constexpr = _ring_layouts(16, BN, BK)
    # Offsets inside one expert's tile stay below BN * K / 2 bytes.
    rows = gl.arange(0, BN, layout=gl.SliceLayout(1, layouts[0]))
    cols = gl.arange(0, BK // 8, layout=gl.SliceLayout(0, layouts[0]))
    w_words = (W + w_row0 * (K // 2)).to(gl.pointer_type(gl.int32), bitcast=True)
    w_at = w_words + (rows[:, None] * (K // 8) + cols[None, :])
    srows = gl.arange(0, BN, layout=gl.SliceLayout(1, layouts[1]))
    scols = gl.arange(0, BK // 16, layout=gl.SliceLayout(0, layouts[1]))
    s_at = (WS + w_row0 * (K // 16)) + (srows[:, None] * (K // 16) + scols[None, :])
    return w_at, s_at


@gluon.jit
def _ring_activations(X, x_rows, rows_valid, K: gl.constexpr, BM: gl.constexpr, BK):
    """Per-lane copy addresses of K block 0 of the BF16 activation rows
    ``x_rows`` (int64, any layout) and the mask of rows to copy."""
    x_copy: gl.constexpr = _ring_layouts(BM, 64, BK)[2]
    xr = gl.convert_layout(x_rows, gl.SliceLayout(1, x_copy))
    valid = gl.convert_layout(rows_valid, gl.SliceLayout(1, x_copy))
    cols = gl.arange(0, BK, layout=gl.SliceLayout(0, x_copy))
    x_at = X + xr[:, None] * K + cols[None, :]
    x_mask = valid[:, None] & (cols < BK)[None, :]
    return x_at, x_mask


@gluon.jit
def _issue_weights(rings, slot, block, w_at, s_at, BK: gl.constexpr):
    w_ring, s_ring, _ = rings
    async_copy.global_load_to_shared(
        w_ring.index(slot), w_at + block * (BK // 8), cache_modifier=".cg"
    )
    async_copy.global_load_to_shared(
        s_ring.index(slot), s_at + block * (BK // 16), cache_modifier=".cg"
    )


@gluon.jit
def _issue_activations(rings, slot, block, x_at, x_mask, BK: gl.constexpr):
    # Rows past the expert's count stay unwritten: they feed only MFMA rows
    # that are never stored.
    _, _, x_ring = rings
    async_copy.global_load_to_shared(x_ring.index(slot), x_at + block * BK, mask=x_mask)


@gluon.jit
def _nvfp4_ring_dot(
    acc,
    rings,
    w_at,
    s_at,
    x_at,
    x_mask,
    K: gl.constexpr,
    BM: gl.constexpr,
    BN: gl.constexpr,
    BK: gl.constexpr,
):
    """``acc`` plus a tile's activations times its NVFP4 weights, K block by
    K block in order (the MFMA K order of the masked loop, so the same bits).

    Every block's operands reach LDS through direct-to-LDS copies that run
    ``depth - 1`` blocks ahead (``_ring_depth``): they hold no registers, so
    more bytes stay in flight than register loads allow at the decode shape,
    where a program walks one tile's K blocks back to back. The caller
    issued the first ``depth - 1`` blocks as commit groups of weights, then
    of activations (which wait on the routes); each step here issues one
    more, so waiting for all but the newest ``depth - 2`` groups finds the
    step's block complete. Copies past the last block reload it.
    """
    gl.static_assert(K % BK == 0, "whole K blocks only")
    weight_layouts: gl.constexpr = _nvfp4_weight_layouts(BN, BK)
    mfma_layout: gl.constexpr = acc.type.layout
    dot_a: gl.constexpr = gl.DotOperandLayout(0, mfma_layout, 8)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, mfma_layout, 8)
    w_ring, s_ring, x_ring = rings
    depth: gl.constexpr = _ring_depth(BM, BN, BK)
    blocks: gl.constexpr = K // BK
    for i in range(blocks):
        # Block i's copies are complete in this lane; the barrier makes every
        # lane's copies visible and frees the slot the next copy refills.
        async_copy.wait_group(depth - 2)
        gl.barrier()
        following = gl.minimum(i + depth - 1, blocks - 1)
        _issue_weights(rings, (i + depth - 1) % depth, following, w_at, s_at, BK)
        _issue_activations(rings, (i + depth - 1) % depth, following, x_at, x_mask, BK)
        async_copy.commit_group()
        slot = i % depth
        words = w_ring.index(slot).reshape([BN, BK // 16, 2])
        words = words.load(_nvfp4_word_layout(BN, BK))
        scale = s_ring.index(slot).load(weight_layouts[4])
        a = x_ring.index(slot).load(dot_a)
        value = _upcast_nvfp4_words(words, scale, BN, BK)
        acc = gl.amd.cdna4.mfma(
            a, gl.convert_layout(gl.permute(value, (1, 0)), dot_b), acc
        )
    # The ring is free again for the program's next tile.
    async_copy.wait_group(0)
    gl.barrier()
    return acc


@gluon.constexpr_function
def _next_power_of_2(value):
    return 1 << (value - 1).bit_length()


@gluon.jit(
    do_not_specialize=["route_count", "num_programs"],
    launch_metadata=_expert_mm_metadata,
)
def gluon_block_scaled_moe_routes_mm_gfx950(
    X,
    XS,
    XA,
    W,
    WS,
    WA,
    WD,
    Y,
    Routes,
    Counts,
    route_count,
    num_programs,
    N: gl.constexpr,
    K: gl.constexpr,
    E: gl.constexpr,
    TOP_K: gl.constexpr,
    INPUT_PER_TOKEN: gl.constexpr,
    NVFP4: gl.constexpr,
    BM: gl.constexpr,
    BN: gl.constexpr,
    BK: gl.constexpr,
    BF16_X: gl.constexpr,
):
    """Route rows of each local expert times its weights, FP32 accumulation.

    With ``BF16_X``, ``X`` holds BF16 activations (the exact values of the
    quantized input, ``XS`` unused) instead of codes to decode for every
    output tile. ``XA`` still gives the NVFP4 rows' outer factors.
    """
    load_layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 16], [4, 1], [1, 0])
    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[1, 4]
    )
    dot_a: gl.constexpr = gl.DotOperandLayout(0, mfma_layout, 8)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, mfma_layout, 8)
    # Tiles run expert by expert, each expert's row blocks by output column
    # tiles. Every program finds its tiles' experts from one load of all route
    # counts and their prefix sum, instead of a dependent load per expert.
    tiles_n: gl.constexpr = gl.cdiv(N, BN)
    count_layout: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])
    experts = gl.arange(0, _next_power_of_2(E), layout=count_layout)
    counts = gl.load(Counts + experts, experts < E, other=0)
    expert_tiles = gl.cdiv(counts, BM) * tiles_n
    expert_end = gl.cumsum(expert_tiles, 0)
    total_tiles = gl.sum(expert_tiles, 0)
    # Whole tiles of NVFP4 weights and BF16 activations stream through an LDS
    # ring (``_nvfp4_ring_dot``); other shapes and formats take the masked loop.
    RING: gl.constexpr = NVFP4 and BF16_X and N % BN == 0 and K % BK == 0
    if RING:
        rings = _ring_stages(WS, BM, BN, BK)
    tile = gl.program_id(0)
    while tile < total_tiles:
        expert = gl.sum((expert_end <= tile).to(gl.int32), 0)
        here = experts == expert
        count = gl.sum(gl.where(here, counts, 0), 0)
        local = tile - gl.sum(gl.where(here, expert_end - expert_tiles, 0), 0)
        m = (local // tiles_n) * BM + gl.arange(
            0, BM, layout=gl.SliceLayout(1, load_layout)
        )
        n0 = (local % tiles_n) * BN
        n = n0 + gl.arange(0, BN, layout=gl.SliceLayout(1, load_layout))
        if RING:
            # The weights' first blocks need only the expert: issue them
            # before the routes' load.
            w_at, s_at = _ring_weights(W, WS, expert.to(gl.int64) * N + n0, K, BN, BK)
            for s in gl.static_range(_ring_depth(BM, BN, BK) - 1):
                _issue_weights(rings, s, min(s, K // BK - 1), w_at, s_at, BK)
                async_copy.commit_group()
        route = gl.load(Routes + expert * route_count + m, m < count, other=0).to(
            gl.int64
        )
        xrow = route // TOP_K if INPUT_PER_TOKEN else route
        wrow = expert.to(gl.int64) * N + n.to(gl.int64)
        # The checkpoint's outer factor applies after the FP32 dot reduction;
        # its loads issue before the K loop.
        if NVFP4:
            xa = gl.load(XA + xrow, m < count, other=0.0)
            wa = gl.load(WA + expert)
            denominator = 1536.0 * gl.load(WD)
            scale_out = (xa * wa) * gl.div_rn(1.0, denominator)
        acc = gl.zeros((BM, BN), gl.float32, mfma_layout)
        if RING:
            x_at, x_mask = _ring_activations(X, xrow, m < count, K, BM, BK)
            for s in gl.static_range(_ring_depth(BM, BN, BK) - 1):
                _issue_activations(rings, s, min(s, K // BK - 1), x_at, x_mask, BK)
                async_copy.commit_group()
            acc = _nvfp4_ring_dot(acc, rings, w_at, s_at, x_at, x_mask, K, BM, BN, BK)
        else:
            for start in range(gl.cdiv(K, BK)):
                k = start * BK + gl.arange(0, BK, layout=gl.SliceLayout(0, load_layout))
                x_mask = (m < count)[:, None] & (k < K)[None, :]
                if BF16_X:
                    xv = gl.load(X + xrow[:, None] * K + k[None, :], x_mask, other=0.0)
                else:
                    xv = _load_block(X, XS, xrow[:, None], k[None, :], x_mask, K, NVFP4)
                if NVFP4:
                    packed, scale = _load_nvfp4_tile(
                        W,
                        WS,
                        expert.to(gl.int64) * N + n0,
                        start * BK,
                        N - n0,
                        K,
                        BN,
                        BK,
                    )
                    wv = gl.convert_layout(
                        _upcast_nvfp4_tile(packed, scale, BN, BK), load_layout
                    )
                else:
                    wv = _load_block(
                        W,
                        WS,
                        wrow[:, None],
                        k[None, :],
                        (n < N)[:, None] & (k < K)[None, :],
                        K,
                        NVFP4,
                    )
                a = gl.convert_layout(xv, dot_a)
                b = gl.convert_layout(gl.permute(wv, (1, 0)), dot_b)
                acc = gl.amd.cdna4.mfma(a, b, acc)
        acc = gl.convert_layout(acc, load_layout)
        if NVFP4:
            acc = acc * scale_out[:, None]
        # Output columns follow the K-contiguous load layout's second axis.
        out_n = n0 + gl.arange(0, BN, layout=gl.SliceLayout(0, load_layout))
        gl.store(
            Y + route[:, None] * N + out_n[None, :],
            acc,
            (m < count)[:, None] & (out_n < N)[None, :],
        )
        tile += num_programs
