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
def _nvfp4_weight_layouts(BN, BK, NW):
    """Register layouts of a ``[BN, BK]`` NVFP4 weight tile for ``NW`` warps,
    each warp 16 rows (rows from ``16 NW`` on repeat in registers).

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
    assert BN % (16 * NW) == 0 and BK % 128 == 0, (BN, BK, NW)
    rows = [[1, 0], [2, 0], [4, 0], [8, 0]]
    warp_bits = range(4, (16 * NW).bit_length() - 1)
    warps = [[1 << bit, 0] for bit in warp_bits]
    more_rows = [
        [1 << bit, 0] for bit in range((16 * NW).bit_length() - 1, BN.bit_length() - 1)
    ]
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
        warp_bases=[[row[0], 0, 0] for row in warps],
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
    load_layout: gl.constexpr = _nvfp4_weight_layouts(BN, BK, gl.num_warps())[0]
    scale_layout: gl.constexpr = _nvfp4_weight_layouts(BN, BK, gl.num_warps())[4]
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
    layouts: gl.constexpr = _nvfp4_weight_layouts(BN, BK, gl.num_warps())
    packed = gl.convert_layout(packed, layouts[1])
    scale = gl.convert_layout(scale, gl.SliceLayout(2, layouts[3]))
    unit = gl.full((BN, BK), 127, gl.uint8, layouts[2])
    value = gl.amd.cdna4.scaled_upcast(packed, unit, gl.bfloat16, axis=1)
    value = gl.convert_layout(gl.reshape(value, [BN, BK // 16, 16]), layouts[3])
    value = (value.to(gl.float32) * scale.to(gl.float32)[:, :, None]).to(gl.bfloat16)
    return gl.reshape(value, [BN, BK])


@gluon.constexpr_function
def _nvfp4_word_layout(BN, BK, NW):
    """``[BN, BK // 16, 2]`` int32 words of packed NVFP4 weights (8 values a
    word; the axes: row, scale group, word in the group) where a lane holds
    the word of each MFMA K step that the B operand (16x16x32, 8 values a
    lane) needs. Its slice along the last axis holds a lane's group scales.
    """
    assert BN % (16 * NW) == 0 and BK % 128 == 0, (BN, BK, NW)
    first_rep = (16 * NW).bit_length() - 1
    more_rows = [[1 << bit, 0, 0] for bit in range(first_rep, BN.bit_length() - 1)]
    return gl.DistributedLinearLayout(
        reg_bases=[[0, 1 << bit, 0] for bit in range(1, (BK // 16).bit_length() - 1)]
        + more_rows,
        lane_bases=[[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [0, 0, 1], [0, 1, 0]],
        warp_bases=[[1 << bit, 0, 0] for bit in range(4, first_rep)],
        block_bases=[],
        shape=[BN, BK // 16, 2],
    )


@gluon.jit
def _pack_bf16(pair):
    """An FP32 pair in an int64 rounded to a packed BF16 pair (an int32),
    the low value in the low half."""
    low = (pair & 0xFFFFFFFF).to(gl.int32).to(gl.float32, bitcast=True)
    high = (pair >> 32).to(gl.int32).to(gl.float32, bitcast=True)
    return gl.inline_asm_elementwise(
        "v_cvt_pk_bf16_f32 $0, $1, $2",
        "=v,v,v",
        [low, high],
        dtype=gl.int32,
        is_pure=True,
        pack=1,
    )


@gluon.jit
def _bf16_pair(word):
    """A packed BF16 pair (an int32) as a ``[.., 2]`` BF16 tensor, no data
    movement: each pair stays one register."""
    return gl.inline_asm_elementwise(
        "", "=v,0,v", [gl.join(word, word)], dtype=gl.bfloat16, is_pure=True, pack=2
    )


@gluon.jit
def _upcast_nvfp4_words(words, scale, BN: gl.constexpr, BK: gl.constexpr):
    """``[BN, BK]`` BF16 weights of ``_nvfp4_word_layout`` words and their
    ``[BN, BK // 16]`` E4M3 group scales, in the MFMA B operand layout
    (after a permute to ``[BK, BN]``). Equal to ``_upcast_nvfp4_tile``.

    Each ``v_cvt_scalef32_pk_f32_fp4`` turns one byte (op_sel picks it from
    the word) into an FP32 pair at unit scale; the E4M3 scale multiplies in
    FP32 and ``v_cvt_pk_bf16_f32`` rounds the pair once into one register,
    already the MFMA operand's packing (rounding a value at a time costs a
    ``v_perm_b32`` a pair to pack). Inline asm because Gluon's scaled upcast
    yields only BF16/FP16, which costs two more instructions a pair to
    unpack to FP32 for the multiply.
    """
    scale = gl.convert_layout(
        scale, gl.SliceLayout(2, _nvfp4_word_layout(BN, BK, gl.num_warps()))
    )
    scale = scale.to(gl.float32)[:, :, None]
    # The scale's bits in the low half of a register pair: v_pk_mul_f32
    # multiplies both values of a pair by it (op_sel_hi), one instruction a
    # pair, exact products as two v_mul_f32.
    bits = gl.broadcast(scale, words)[0].to(gl.int32, bitcast=True).to(gl.int64)
    byte0, byte1, byte2, byte3 = gl.inline_asm_elementwise(
        "v_cvt_scalef32_pk_f32_fp4 $0, $4, 1.0\n"
        "v_cvt_scalef32_pk_f32_fp4 $1, $4, 1.0 op_sel:[1,0,0]\n"
        "v_cvt_scalef32_pk_f32_fp4 $2, $4, 1.0 op_sel:[0,1,0]\n"
        "v_cvt_scalef32_pk_f32_fp4 $3, $4, 1.0 op_sel:[1,1,0]\n"
        "v_pk_mul_f32 $0, $0, $5 op_sel_hi:[1,0]\n"
        "v_pk_mul_f32 $1, $1, $5 op_sel_hi:[1,0]\n"
        "v_pk_mul_f32 $2, $2, $5 op_sel_hi:[1,0]\n"
        "v_pk_mul_f32 $3, $3, $5 op_sel_hi:[1,0]",
        "=&v,=&v,=&v,=&v,v,v",
        [words, bits],
        dtype=(gl.int64, gl.int64, gl.int64, gl.int64),
        is_pure=True,
        pack=1,
    )
    pair0 = _pack_bf16(byte0)
    pair1 = _pack_bf16(byte1)
    pair2 = _pack_bf16(byte2)
    pair3 = _pack_bf16(byte3)
    r0 = _bf16_pair(pair0)
    r1 = _bf16_pair(pair1)
    r2 = _bf16_pair(pair2)
    r3 = _bf16_pair(pair3)
    # Value 2 j + h of a word is byte j's low (h = 0) or high nibble: the
    # joins give [.., h, b, a] for j = 2 b + a, permuted to [.., b, a, h]
    # with h still within each register.
    v = gl.join(gl.join(r0, r2), gl.join(r1, r3))
    v = gl.permute(v, (0, 1, 2, 4, 5, 3))
    return gl.reshape(v, [BN, BK])


@gluon.constexpr_function
def _mxfp8_weight_layouts(BN, BK, NW):
    """Register layouts of a ``[BN, BK]`` MXFP8 weight tile for ``NW`` warps,
    each warp 16 rows (rows from ``16 NW`` on repeat in registers).

    Returns ``(value, grouped)``:

    - ``value``: E4M3 codes, and the BF16 values after their upcast, in the
      MFMA B operand layout itself (16x16x32, 8 values a lane): a lane holds
      8 contiguous bytes of each K step, one 8-byte LDS read.
    - ``grouped``: ``value`` viewed as ``[BN, BK // 32, 32]`` scale groups.
      Its slice along the last axis holds a lane's UE8M0 scales, a whole
      row's in each lane.
    """
    assert BN % (16 * NW) == 0 and BK % 128 == 0, (BN, BK, NW)
    first_rep = (16 * NW).bit_length() - 1
    warps = [[1 << bit, 0] for bit in range(4, first_rep)]
    more_rows = [[1 << bit, 0] for bit in range(first_rep, BN.bit_length() - 1)]
    # K steps of 32 values sit in registers, after a lane's 8 bytes.
    steps = [[0, 1 << bit] for bit in range(5, BK.bit_length() - 1)]
    value = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4]] + steps + more_rows,
        lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [0, 8], [0, 16]],
        warp_bases=warps,
        block_bases=[],
        shape=[BN, BK],
    )
    grouped = gl.DistributedLinearLayout(
        reg_bases=[[0, 0, 1], [0, 0, 2], [0, 0, 4]]
        + [[0, step[1] // 32, 0] for step in steps]
        + [[row[0], 0, 0] for row in more_rows],
        lane_bases=[[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [0, 0, 8], [0, 0, 16]],
        warp_bases=[[row[0], 0, 0] for row in warps],
        block_bases=[],
        shape=[BN, BK // 32, 32],
    )
    return value, grouped


@gluon.jit
def _upcast_mxfp8_tile(codes, exponents, BN: gl.constexpr, BK: gl.constexpr):
    """``[BN, BK]`` BF16 weights of E4M3 ``codes`` in ``_mxfp8_weight_layouts``
    order and their ``[BN, BK // 32]`` UE8M0 ``exponents``, in the MFMA B
    operand layout (after a permute to ``[BK, BN]``). The scale rule and the
    FP32 product are ``_load_block``'s, so the values equal the masked loop's.
    """
    grouped: gl.constexpr = _mxfp8_weight_layouts(BN, BK, gl.num_warps())[1]
    exponent = gl.convert_layout(exponents, gl.SliceLayout(2, grouped)).to(gl.int32)
    scale = gl.maximum((exponent << 23).to(gl.float32, bitcast=True), 2.0**-127)
    scale = gl.where(exponent == 255, float("nan"), scale)
    data = gl.convert_layout(gl.reshape(codes, [BN, BK // 32, 32]), grouped)
    value = (data.to(gl.float32) * scale[:, :, None]).to(gl.bfloat16)
    return gl.reshape(value, [BN, BK])


@gluon.constexpr_function
def _scale_copy_rows(BN, BK, NW, NVFP4):
    """Rows of a ring stage's group scales: ``BN``, or a multiple of it when a
    stage's scales fill fewer than the ``64 NW`` lanes of one 4-byte copy
    (the extra rows repeat the tile's rows and are never read)."""
    row_bytes = BK // 16 if NVFP4 else BK // 32
    return max(BN, 4 * 64 * NW // row_bytes)


@gluon.constexpr_function
def _ring_layouts(BN, BK, NW, NVFP4):
    """Copy and LDS layouts of one ring stage (``NW`` warps): the weights
    (NVFP4: packed int32 words ``[BN, BK // 8]``; MXFP8: E4M3 bytes
    ``[BN, BK]``), the group scales (``[_scale_copy_rows, BK // 16]`` E4M3 or
    ``[.., BK // 32]`` UE8M0) and the BF16 activations ``[BM, BK]``. Each lane
    copies 16 bytes (4 for scales), rows split across lanes. Weight and
    activation rows are swizzled so that the operand reads (one row a lane)
    spread over the LDS banks."""
    assert BK % 128 == 0 and BN % (16 * NW) == 0, (BN, BK, NW)
    if NVFP4:
        w_copy = gl.BlockedLayout([1, 4], [64 // (BK // 32), BK // 32], [NW, 1], [1, 0])
        s_copy = gl.BlockedLayout([1, 4], [64 // (BK // 64), BK // 64], [NW, 1], [1, 0])
        w_shared = gl.SwizzledSharedLayout(4, 1, 8, [1, 0])
    else:
        w_copy = gl.BlockedLayout(
            [1, 16], [64 // (BK // 16), BK // 16], [NW, 1], [1, 0]
        )
        s_copy = gl.BlockedLayout(
            [1, 4], [64 // (BK // 128), BK // 128], [NW, 1], [1, 0]
        )
        w_shared = gl.SwizzledSharedLayout(16, 1, min(16, BK // 16), [1, 0])
    return (
        w_copy,
        s_copy,
        gl.BlockedLayout([1, 8], [64 // (BK // 8), BK // 8], [NW, 1], [1, 0]),
        w_shared,
        gl.SwizzledSharedLayout(1, 1, 1, [1, 0]),
        gl.SwizzledSharedLayout(8, 1, 16, [1, 0]),
    )


@gluon.constexpr_function
def _ring_depth(BM, BN, BK, NVFP4):
    """Stages of the ring. NVFP4: 4 while two programs fit a CU's 160 KB of
    LDS (the decode tile: 17 KB a stage), else as many as one program fits,
    at least 2. MXFP8: 2, which keeps two programs on a CU at every tile
    (decode 50 KB, prefill 66 KB a program) and measured faster than 3 or 4
    at both shapes."""
    if not NVFP4:
        return 2
    stage = BN * BK // 2 + BN * BK // 16 + BM * BK * 2
    if 4 * stage <= 80 * 1024:
        return 4
    return max(2, min(4, 160 * 1024 // stage))


@gluon.jit
def ring_stages(
    WS,
    BM: gl.constexpr,
    BN: gl.constexpr,
    BK: gl.constexpr,
    depth: gl.constexpr,
    NVFP4: gl.constexpr,
):
    """The LDS ring of ``ring_dot`` (``WS``: the group scales), of ``depth``
    stages, for NVFP4 or MXFP8 weights."""
    NW: gl.constexpr = gl.num_warps()
    layouts: gl.constexpr = _ring_layouts(BN, BK, NW, NVFP4)
    if NVFP4:
        w_ring = gl.allocate_shared_memory(gl.int32, [depth, BN, BK // 8], layouts[3])
        s_cols: gl.constexpr = BK // 16
    else:
        w_ring = gl.allocate_shared_memory(gl.float8e4nv, [depth, BN, BK], layouts[3])
        s_cols: gl.constexpr = BK // 32
    s_ring = gl.allocate_shared_memory(
        WS.dtype.element_ty,
        [depth, _scale_copy_rows(BN, BK, NW, NVFP4), s_cols],
        layouts[4],
    )
    x_ring = gl.allocate_shared_memory(gl.bfloat16, [depth, BM, BK], layouts[5])
    return w_ring, s_ring, x_ring


@gluon.jit
def ring_weights(
    W,
    WS,
    w_row0,
    K: gl.constexpr,
    BN: gl.constexpr,
    BK: gl.constexpr,
    NVFP4: gl.constexpr,
):
    """Per-lane copy addresses of K block 0 of weight rows ``w_row0 + [0,
    BN)``: the weights and their group scales. Per-lane addresses, not buffer
    descriptors: the tile's base is not provably uniform, and a descriptor in
    vector registers costs a loop per copy."""
    NW: gl.constexpr = gl.num_warps()
    layouts: gl.constexpr = _ring_layouts(BN, BK, NW, NVFP4)
    # Offsets inside one expert's tile stay below BN * K bytes.
    rows = gl.arange(0, BN, layout=gl.SliceLayout(1, layouts[0]))
    if NVFP4:
        cols = gl.arange(0, BK // 8, layout=gl.SliceLayout(0, layouts[0]))
        w_words = (W + w_row0 * (K // 2)).to(gl.pointer_type(gl.int32), bitcast=True)
        w_at = w_words + (rows[:, None] * (K // 8) + cols[None, :])
        GROUP: gl.constexpr = 16
    else:
        cols = gl.arange(0, BK, layout=gl.SliceLayout(0, layouts[0]))
        w_at = (W + w_row0 * K) + (rows[:, None] * K + cols[None, :])
        GROUP: gl.constexpr = 32
    S_ROWS: gl.constexpr = _scale_copy_rows(BN, BK, NW, NVFP4)
    srows = gl.arange(0, S_ROWS, layout=gl.SliceLayout(1, layouts[1]))
    if S_ROWS > BN:
        srows = srows % BN
    scols = gl.arange(0, BK // GROUP, layout=gl.SliceLayout(0, layouts[1]))
    s_at = (WS + w_row0 * (K // GROUP)) + (
        srows[:, None] * (K // GROUP) + scols[None, :]
    )
    return w_at, s_at


@gluon.jit
def ring_activations(X, row_starts, rows_valid, BM: gl.constexpr, BK: gl.constexpr):
    """Per-lane copy addresses of K block 0 of ``BM`` BF16 activation rows
    starting at elements ``row_starts`` of ``X`` (int64, any layout), and the
    mask of rows to copy."""
    NW: gl.constexpr = gl.num_warps()
    x_copy: gl.constexpr = _ring_layouts(16 * NW, BK, NW, True)[2]
    # One copy instruction covers 64 // (BK // 8) rows a warp.
    gl.static_assert(BM % (NW * 64 // (BK // 8)) == 0)
    starts = gl.convert_layout(row_starts, gl.SliceLayout(1, x_copy))
    valid = gl.convert_layout(rows_valid, gl.SliceLayout(1, x_copy))
    cols = gl.arange(0, BK, layout=gl.SliceLayout(0, x_copy))
    x_at = X + starts[:, None] + cols[None, :]
    x_mask = valid[:, None] & (cols < BK)[None, :]
    return x_at, x_mask


@gluon.jit
def ring_issue_weights(
    rings, slot, block, w_at, s_at, BK: gl.constexpr, NVFP4: gl.constexpr
):
    """Copy K block ``block`` of a tile's weights and group scales into ring
    slot ``slot``; read once, they skip L1."""
    w_ring, s_ring, _ = rings
    if NVFP4:
        w_step: gl.constexpr = BK // 8
        s_step: gl.constexpr = BK // 16
    else:
        w_step: gl.constexpr = BK
        s_step: gl.constexpr = BK // 32
    async_copy.global_load_to_shared(
        w_ring.index(slot), w_at + block * w_step, cache_modifier=".cg"
    )
    async_copy.global_load_to_shared(
        s_ring.index(slot), s_at + block * s_step, cache_modifier=".cg"
    )


@gluon.jit
def ring_issue_activations(
    rings, slot, block, x_at, x_mask, BK: gl.constexpr, CACHE: gl.constexpr
):
    """Copy K block ``block`` of the activation rows into ring slot ``slot``
    with cache modifier ``CACHE``. Masked rows stay unwritten: they feed only
    MFMA rows that are never stored."""
    _, _, x_ring = rings
    async_copy.global_load_to_shared(
        x_ring.index(slot), x_at + block * BK, mask=x_mask, cache_modifier=CACHE
    )


@gluon.jit
def _ring_load_weights(
    rings, slot, BN: gl.constexpr, BK: gl.constexpr, NVFP4: gl.constexpr
):
    """The weights and group scales of ring slot ``slot`` in the layouts of
    ``_upcast_ring_weights``."""
    w_ring, s_ring, _ = rings
    NW: gl.constexpr = gl.num_warps()
    scales = s_ring.index(slot)
    if _scale_copy_rows(BN, BK, NW, NVFP4) > BN:
        scales = scales.slice(0, BN)
    if NVFP4:
        words = w_ring.index(slot).reshape([BN, BK // 16, 2])
        words = words.load(_nvfp4_word_layout(BN, BK, NW))
        scale = scales.load(_nvfp4_weight_layouts(BN, BK, NW)[4])
    else:
        layouts: gl.constexpr = _mxfp8_weight_layouts(BN, BK, NW)
        words = w_ring.index(slot).load(layouts[0])
        scale = scales.load(gl.SliceLayout(2, layouts[1]))
    return words, scale


@gluon.jit
def _upcast_ring_weights(
    words, scale, BN: gl.constexpr, BK: gl.constexpr, NVFP4: gl.constexpr
):
    """``[BN, BK]`` BF16 weights of ``_ring_load_weights`` values in the MFMA B
    operand layout (after a permute to ``[BK, BN]``)."""
    if NVFP4:
        value = _upcast_nvfp4_words(words, scale, BN, BK)
    else:
        value = _upcast_mxfp8_tile(words, scale, BN, BK)
    return value


@gluon.jit
def ring_block(
    acc, rings, slot, BN: gl.constexpr, BK: gl.constexpr, NVFP4: gl.constexpr
):
    """``acc`` plus the activations times the weights of ring slot ``slot``:
    one K block of ``ring_dot``."""
    mfma_layout: gl.constexpr = acc.type.layout
    dot_a: gl.constexpr = gl.DotOperandLayout(0, mfma_layout, 8)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, mfma_layout, 8)
    _, _, x_ring = rings
    words, scale = _ring_load_weights(rings, slot, BN, BK, NVFP4)
    a = x_ring.index(slot).load(dot_a)
    value = _upcast_ring_weights(words, scale, BN, BK, NVFP4)
    return gl.amd.cdna4.mfma(
        a, gl.convert_layout(gl.permute(value, (1, 0)), dot_b), acc
    )


@gluon.jit
def ring_block_pair(
    acc0, acc1, rings, halves, slot, BN: gl.constexpr, BK: gl.constexpr
):
    """``ring_block`` of NVFP4 weights on a ring of ``2 BN`` rows a stage:
    rows ``0 .. BN`` of slot ``slot`` into ``acc0``, ``BN .. 2 BN`` into
    ``acc1``. ``halves``: the same LDS as a ring of ``BN`` rows a stage
    (stage ``2 s + h``: half ``h`` of stage ``s``). Upcasting one half at a
    time keeps ``ring_block``'s registers."""
    mfma_layout: gl.constexpr = acc0.type.layout
    dot_a: gl.constexpr = gl.DotOperandLayout(0, mfma_layout, 8)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, mfma_layout, 8)
    _, _, x_ring = rings
    a = x_ring.index(slot).load(dot_a)
    words, scale = _ring_load_weights(halves, 2 * slot, BN, BK, True)
    value = _upcast_nvfp4_words(words, scale, BN, BK)
    acc0 = gl.amd.cdna4.mfma(
        a, gl.convert_layout(gl.permute(value, (1, 0)), dot_b), acc0
    )
    words, scale = _ring_load_weights(halves, 2 * slot + 1, BN, BK, True)
    value = _upcast_nvfp4_words(words, scale, BN, BK)
    acc1 = gl.amd.cdna4.mfma(
        a, gl.convert_layout(gl.permute(value, (1, 0)), dot_b), acc1
    )
    return acc0, acc1


@gluon.jit
def ring_dot(
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
    depth: gl.constexpr,
    X_CACHE: gl.constexpr,
    NVFP4: gl.constexpr,
):
    """``acc`` plus a tile's activations times its NVFP4 or MXFP8 weights, K
    block by K block in order (the MFMA K order of the masked loop, so the
    same bits).

    Every block's operands reach LDS through direct-to-LDS copies that run
    ``depth - 1`` blocks ahead: they hold no registers, so more bytes stay in
    flight than register loads allow at the decode shape, where a program
    walks one tile's K blocks back to back. Activations copy with cache
    modifier ``X_CACHE``. The caller
    issued the first ``depth - 1`` blocks as commit groups of weights, then
    of activations (which wait on the routes); each step here issues one
    more, so waiting for all but the newest ``depth - 2`` groups finds the
    step's block complete. Copies past the last block reload it.
    """
    gl.static_assert(K % BK == 0, "whole K blocks only")
    blocks: gl.constexpr = K // BK
    for i in range(blocks):
        # Block i's copies are complete in this lane; the barrier makes every
        # lane's copies visible and frees the slot the next copy refills.
        async_copy.wait_group(depth - 2)
        gl.barrier()
        following = gl.minimum(i + depth - 1, blocks - 1)
        ring_issue_weights(
            rings, (i + depth - 1) % depth, following, w_at, s_at, BK, NVFP4
        )
        ring_issue_activations(
            rings, (i + depth - 1) % depth, following, x_at, x_mask, BK, X_CACHE
        )
        async_copy.commit_group()
        acc = ring_block(acc, rings, i % depth, BN, BK, NVFP4)
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
    # Whole tiles of NVFP4 or MXFP8 weights and BF16 activations stream
    # through an LDS ring (``ring_dot``); other shapes take the masked loop.
    RING: gl.constexpr = BF16_X and N % BN == 0 and K % BK == 0
    DEPTH: gl.constexpr = _ring_depth(BM, BN, BK, NVFP4)
    if RING:
        rings = ring_stages(WS, BM, BN, BK, DEPTH, NVFP4)
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
            w_at, s_at = ring_weights(
                W, WS, expert.to(gl.int64) * N + n0, K, BN, BK, NVFP4
            )
            for s in gl.static_range(DEPTH - 1):
                ring_issue_weights(rings, s, min(s, K // BK - 1), w_at, s_at, BK, NVFP4)
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
            x_at, x_mask = ring_activations(X, xrow * K, m < count, BM, BK)
            for s in gl.static_range(DEPTH - 1):
                ring_issue_activations(
                    rings, s, min(s, K // BK - 1), x_at, x_mask, BK, ""
                )
                async_copy.commit_group()
            acc = ring_dot(
                acc,
                rings,
                w_at,
                s_at,
                x_at,
                x_mask,
                K,
                BM,
                BN,
                BK,
                DEPTH,
                "",
                NVFP4,
            )
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
