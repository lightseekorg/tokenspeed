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

"""MXFP4-weight x MXFP4-activation decode GEMV building blocks.

A decode GEMV ``y[n] = sum_k x[k] W[n, k]`` reads far more weight than it
computes. Thus a stage spreads a weight over many CTAs (a few 16-row blocks
each). It splits K over waves to keep enough loads in flight per CTA.

One ``v_mfma_scale_f32_16x16x128_f8f6f4`` multiplies a 16x128 MXFP4 A tile
by a 128x16 MXFP4 B tile (16 weight rows). The A tile packs up to 16
activation rows (MFMA row ``m`` holds row ``m % RP``). Thus a launch that
serves several rows reads each weight once and runs the MFMAs of one row.
A row's result depends only on its own A row: it is exactly what a one-row
GEMV gives.

The K split is the batch dimension of a rank-3 MFMA layout, and the warps own
this dimension. Wave ``s`` runs k-iterations ``s, s + S, s + 2 S, ...``. This
interleave makes the waves stream adjacent 1 KiB blocks at any moment. A
fixed-order sum over the batch dimension combines the waves.

Weights (host layouts, see ``weights.py``):

* tiles: N16 order. In each 16-row block, each 64-byte K chunk (one
  k-iteration) is a 1 KiB block in MFMA lane order. Lane ``g * 16 + n`` reads
  row ``n``, K bytes ``16 g .. 16 g + 15``, as one ``dwordx4``. The bytes are
  the same as in TokenSpeed's dense ``preshuffle_mxfp4_weight`` and MoE N16
  banks.
* scales: ``[N/16][ceil(KI/4)][64][4]`` UE8M0. Lane ``g * 16 + n``'s dword
  ``j`` holds the scales of k-iterations ``4 j .. 4 j + 3``, so one dword load
  serves four MFMAs.

Activations sit in LDS as ``[KP, 64]`` packed E2M1 bytes, plus ``[KP, 4]``
UE8M0 scales (k-iteration major). ``KP`` is a power of two that is at least
the iteration count. Scales follow TokenSpeed's quantizer, Quark's "even"
rule (:func:`mxfp4_exponent`).
"""

from __future__ import annotations

from tokenspeed_kernel_amd._triton import gl, gluon, triton
from tokenspeed_kernel_amd.ops.gfx950.megakernel import schedule

cdna4 = gl.amd.cdna4

KITER_BYTES = gl.constexpr(64)  # packed E2M1 bytes per row and k-iteration


@gluon.constexpr_function
def mfma_layout(split_k, n_waves):
    """Rank-3 scaled-MFMA layout: ``split_k`` K-split waves x ``n_waves`` N16 waves."""
    return gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 128],
        transposed=True,
        warps_per_cta=[split_k, 1, n_waves],
    )


@gluon.constexpr_function
def a_layout(split_k, n_waves):
    return gl.DotOperandLayout(0, mfma_layout(split_k, n_waves), 16)


@gluon.constexpr_function
def b_layout(split_k, n_waves):
    return gl.DotOperandLayout(1, mfma_layout(split_k, n_waves), 16)


@gluon.constexpr_function
def scale_layout(split_k, n_waves, operand):
    """E8M0 operand-scale layout of :func:`mfma_layout` (``operand`` 0 or 1).

    Shapes are ``[split_k, 16, 4]`` for A and ``[split_k, 16 * n_waves, 4]``
    for B. Lane ``l`` holds the scale of row ``l % 16``, K group ``l // 16``.
    ``get_mfma_scale_layout`` builds rank-2 layouts only, so this function
    adds the K-split dimension as warp bases. Like A, the A scale has a copy
    on each N wave. The N waves hold different parts of the B scale.
    """
    split_bits = split_k.bit_length() - 1
    n_bits = n_waves.bit_length() - 1
    n_bases = [[0, 16 << i, 0] if operand == 1 else [0, 0, 0] for i in range(n_bits)]
    # The MFMA layout numbers warps N-wave first, then K-split.
    return gl.DistributedLinearLayout(
        reg_bases=[],
        lane_bases=[[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 0, 1], [0, 0, 2]],
        warp_bases=n_bases + [[1 << i, 0, 0] for i in range(split_bits)],
        block_bases=[],
        shape=[split_k, 16 * (n_waves if operand == 1 else 1), 4],
    )


@gluon.constexpr_function
def _scale_extent(K, SCALES):
    """K extent of a scale layout: dwords of 4 k-iterations (``"k4"``) or
    256-byte cells of 2 k-iterations (``"n32"``)."""
    return -(-K // 512) if SCALES == "k4" else -(-K // 256)


@gluon.jit
def _scale_bytes(ws, block, kit, g, n, SCALES: gl.constexpr, KJ: gl.constexpr):
    """UE8M0 scale of (16-row ``block``, row ``n``, K group ``g``, ``kit``).

    ``SCALES`` names the scale layout:

    * ``"k4"`` (:func:`weights.mxfp4_scales`): int32 dwords
      ``[N/16][KJ][64]``. The scale is byte ``kit % 4`` of dword
      ``(block, kit // 4, 16 g + n)``. The function returns the whole dword.
      :func:`_scale_operand` takes the byte when the MFMA runs, so the load
      issue does not wait for it.
    * ``"n32h"``: the ``"n32"`` layout for a GEMV whose K-split waves each
      take one contiguous half of K (see :func:`load_weight_tiles`).
    * ``"n32"`` (TokenSpeed's N16 MoE banks, ``n16_weights._n32_scale_offset``):
      uint8, ``KJ`` 256-byte cells (two k-iterations) per pair of 16-row
      blocks. The scale is byte ``(block // 2) * KJ * 256 + (kit // 2) * 256
      + 64 g + 4 n + 2 (kit % 2) + block % 2``.
    """
    if SCALES == "k4":
        return cdna4.buffer_load(ws, (block * KJ + kit // 4) * 64 + g * 16 + n)
    else:
        offset = (block // 2) * (KJ * 256) + (kit // 2) * 256 + g * 64 + n * 4
        return cdna4.buffer_load(ws, offset + (kit % 2) * 2 + block % 2)


@gluon.jit
def load_weight_tiles(
    w,
    ws,
    nb0,
    K: gl.constexpr,
    kit0,
    KITS: gl.constexpr,
    SPLIT: gl.constexpr,
    NW: gl.constexpr,
    GROUP: gl.constexpr,
    GROUP_STRIDE: gl.constexpr,
    SCALES: gl.constexpr,
):
    """Issue every weight and scale load of one GEMV task. Nothing waits here.

    Call this before polling the activation so the weight stream overlaps
    the wait. N wave ``v`` reads 16-row block
    ``nb0 + (v // GROUP) * GROUP_STRIDE + v % GROUP``.

    Args:
        w: N16 E2M1 tiles (uint8).
        ws: Scales in the ``SCALES`` layout (see :func:`_scale_bytes`).
        nb0: First 16-row block of the task.
        K: The weight's (padded) reduction width.
        kit0, KITS: The task's k-iterations ``kit0 .. kit0 + KITS - 1``
            (``KITS`` is a multiple of ``SPLIT``). This is a split-K across
            tasks. With ``"k4"`` scales, ``kit0`` is a multiple of 4.
        SPLIT: K-split waves.
        NW: N waves (16 rows each).
        GROUP, GROUP_STRIDE: N waves come in groups of ``GROUP`` consecutive
            blocks, ``GROUP_STRIDE`` blocks apart.
        SCALES: Scale layout, ``"k4"`` or ``"n32"`` (see :func:`_scale_bytes`).

    Returns:
        ``(tiles, scales)``: one B tile and one B scale per chain step.
    """
    KI: gl.constexpr = K // 128
    KJ: gl.constexpr = _scale_extent(K, SCALES)
    gl.static_assert(KITS % SPLIT == 0 and KITS <= KI)
    bl: gl.constexpr = b_layout(SPLIT, NW)
    s = gl.arange(0, SPLIT, gl.SliceLayout(1, gl.SliceLayout(2, bl)))
    k = gl.arange(0, 64, gl.SliceLayout(0, gl.SliceLayout(2, bl)))
    n = gl.arange(0, 16 * NW, gl.SliceLayout(0, gl.SliceLayout(1, bl)))
    v = n // 16
    block = nb0 + (v // GROUP) * GROUP_STRIDE + v % GROUP
    # "n32h": K-split wave s takes the contiguous k-iterations s KITS / 2 ...
    split_stride: gl.constexpr = KITS // SPLIT if SCALES == "n32h" else 1
    step: gl.constexpr = 1 if SCALES == "n32h" else SPLIT
    tile = (
        block[None, None, :] * (16 * (K // 2))
        + s[:, None, None] * (1024 * split_stride)
        + (k // 16)[None, :, None] * 256
        + (n % 16)[None, None, :] * 16
        + (k % 16)[None, :, None]
    )
    sl: gl.constexpr = scale_layout(SPLIT, NW, 1)
    ss = gl.arange(0, SPLIT, gl.SliceLayout(1, gl.SliceLayout(2, sl)))
    sn = gl.arange(0, 16 * NW, gl.SliceLayout(0, gl.SliceLayout(2, sl)))
    sg = gl.arange(0, 4, gl.SliceLayout(0, gl.SliceLayout(1, sl)))
    sv = sn // 16
    sblock = nb0 + (sv // GROUP) * GROUP_STRIDE + sv % GROUP
    tiles = ()
    scales = ()
    for i in gl.static_range(KITS // SPLIT):
        # A launch reads each weight once: loads that skip L1 (".cg")
        # stream faster.
        tiles += (cdna4.buffer_load(w, tile + (kit0 + i * step) * 1024, cache=".cg"),)
        kit = kit0 + i * step + ss * split_stride
        if SCALES != "n32h":
            scales += (
                _scale_bytes(
                    ws,
                    sblock[None, :, None],
                    kit[:, None, None],
                    sg[None, None, :],
                    (sn % 16)[None, :, None],
                    SCALES,
                    KJ,
                ),
            )
        elif i % 2 == 0:
            # One dword holds both k-iterations of the pair for both blocks
            # of a block pair; _scale_operand takes the byte.
            cell = (sblock // 2)[None, :, None] * (KJ * 256) + (kit // 2)[
                :, None, None
            ] * 256
            dword = cell + sg[None, None, :] * 64 + (sn % 16)[None, :, None] * 4
            word = cdna4.buffer_load(
                ws.to(gl.pointer_type(gl.uint32), bitcast=True), dword // 4
            )
            scales += (word, word)
    return tiles, scales


@gluon.constexpr_function
def _lds_iter(i, split, seg, seg_stride):
    kit = i * split
    if seg == 0:
        return kit
    assert seg % split == 0, "a chain step must not straddle an LDS segment"
    return kit // seg * seg_stride + kit % seg


@gluon.jit
def gemv(
    x_smem,
    xs_smem,
    tiles,
    scales,
    SPLIT: gl.constexpr,
    NW: gl.constexpr,
    SEG: gl.constexpr,
    SEG_STRIDE: gl.constexpr,
    FIRST: gl.constexpr,
    ROWS: gl.constexpr,
):
    """Activation rows times the loaded weight tiles.

    Args:
        x_smem: ``[R, KP, 64]`` packed E2M1 bytes of activation rows in LDS.
        xs_smem: ``[R, KP, 4]`` UE8M0 scales of the rows.
        tiles, scales: From :func:`load_weight_tiles`.
        SEG, SEG_STRIDE: LDS placement of the k-iterations: iteration ``kit``
            sits at LDS row ``kit // SEG * SEG_STRIDE + kit % SEG`` (``SEG``
            0: row ``kit``). This lets a row that sits in padded segments
            (for example one per hyper-connection stream) feed the GEMV in
            place.
        FIRST, ROWS: The activation rows to multiply, ``FIRST .. FIRST +
            ROWS - 1`` of ``x_smem``: one row, or all ``R`` rows from 0.

    Returns:
        FP32 ``[RP, 16 * NW]``: row ``r`` is activation row ``FIRST + r``'s
        result. ``RP`` is 1 for one row, else the row count of ``x_smem``.
        Rows past ``ROWS`` are padding.
    """
    al: gl.constexpr = a_layout(SPLIT, NW)
    sal: gl.constexpr = scale_layout(SPLIT, NW, 0)
    RP: gl.constexpr = _packed_rows(x_smem.shape[0], ROWS)
    gl.static_assert(ROWS == 1 or (FIRST == 0 and ROWS <= RP and RP <= 16))
    acc = gemv_zero(SPLIT, NW)
    for i in gl.static_range(len(tiles)):
        a = _a_tile(x_smem, _lds_iter(i, SPLIT, SEG, SEG_STRIDE), FIRST, RP, SPLIT, al)
        a_s = _a_tile(
            xs_smem, _lds_iter(i, SPLIT, SEG, SEG_STRIDE), FIRST, RP, SPLIT, sal
        )
        acc = _mfma(acc, a, a_s, tiles[i], scales[i], i)
    return gemv_result(acc, RP)


@gluon.jit
def gemv_zero(SPLIT: gl.constexpr, NW: gl.constexpr):
    """A zero accumulator for :func:`gemv_row`."""
    return gl.zeros([SPLIT, 16, 16 * NW], gl.float32, mfma_layout(SPLIT, NW))


@gluon.jit
def gemv_row(acc, x_smem, xs_smem, tiles, scales, row: gl.constexpr):
    """Row ``row`` times its own weight tiles, into ``acc``'s MFMA
    rows ``m`` with ``m % RP == row`` (``RP`` is the row count of
    ``x_smem``).

    Use it for stages whose rows each need different weights (an expert per
    row). Each row's chain runs as a one-row GEMV. A select gathers the rows
    into one accumulator. Thus one :func:`gemv_result` and one epilogue serve
    all rows. Each row's result equals its one-row GEMV.

    Args:
        acc: From :func:`gemv_zero` or an earlier call.
        x_smem, xs_smem, tiles, scales: As :func:`gemv`.
        row: The row.
    """
    SPLIT: gl.constexpr = acc.shape[0]
    NW: gl.constexpr = acc.shape[2] // 16
    RP: gl.constexpr = x_smem.shape[0]
    al: gl.constexpr = a_layout(SPLIT, NW)
    sal: gl.constexpr = scale_layout(SPLIT, NW, 0)
    own = gl.zeros_like(acc)
    for i in gl.static_range(len(tiles)):
        a = _a_tile(x_smem, i * SPLIT, row, 1, SPLIT, al)
        a_s = _a_tile(xs_smem, i * SPLIT, row, 1, SPLIT, sal)
        own = _mfma(own, a, a_s, tiles[i], scales[i], i)
    m = gl.arange(0, 16, gl.SliceLayout(0, gl.SliceLayout(2, acc.type.layout)))
    return gl.where((m % RP == row)[None, :, None], own, acc)


@gluon.jit
def gemv_chain(x_smem, xs_smem, tiles, scales):
    """Every activation row times the loaded weight tiles, as an
    accumulator: MFMA row ``m`` holds row ``m % RP`` (``RP`` the row count
    of ``x_smem``). :func:`gemv_result` reads it; a select keeps rows of
    several chains in one accumulator (see :func:`gemv_row`)."""
    acc = gemv_zero(tiles[0].shape[0], tiles[0].shape[2] // 16)
    SPLIT: gl.constexpr = acc.shape[0]
    NW: gl.constexpr = acc.shape[2] // 16
    RP: gl.constexpr = x_smem.shape[0]
    al: gl.constexpr = a_layout(SPLIT, NW)
    sal: gl.constexpr = scale_layout(SPLIT, NW, 0)
    for i in gl.static_range(len(tiles)):
        a = _a_tile(x_smem, i * SPLIT, 0, RP, SPLIT, al)
        a_s = _a_tile(xs_smem, i * SPLIT, 0, RP, SPLIT, sal)
        acc = _mfma(acc, a, a_s, tiles[i], scales[i], i)
    return acc


@gluon.jit
def gemv_result(acc, RP: gl.constexpr):
    """``[RP, 16 * NW]`` FP32 rows of a GEMV accumulator (as :func:`gemv`)."""
    NW: gl.constexpr = acc.shape[2] // 16
    # Sum the K-split waves in a fixed order. MFMA row m holds the result of
    # activation row m % RP, so rows m and m + RP agree.
    out = gl.sum(acc, axis=0)
    return gl.max(gl.reshape(out, [16 // RP, RP, 16 * NW]), axis=0)


@gluon.jit
def _mfma(acc, a, a_s, tile, scale, i: gl.constexpr):
    """MFMA of chain step ``i`` (``tile`` and ``scale`` from
    :func:`load_weight_tiles`)."""
    gl.static_assert(a.shape[2] == KITER_BYTES)
    return cdna4.mfma_scaled(
        a,
        a_s,
        "e2m1",
        tile,
        _scale_operand(scale, i, acc.shape[0]),
        "e2m1",
        acc,
    )


@gluon.jit
def _scale_operand(scale, i: gl.constexpr, SPLIT: gl.constexpr):
    """The B scale bytes of chain step ``i``: ``"n32"`` bytes as loaded, or
    byte ``kit % 4`` of each ``"k4"`` dword (k-iteration ``kit = kit0 + i
    SPLIT + s`` on K-split wave ``s``, ``kit0 % 4 == 0``)."""
    if scale.dtype.primitive_bitwidth == 8:
        return scale
    elif scale.dtype.is_int_unsigned():
        # "n32h": byte 2 (kit % 2) + block % 2 of the cell's dword. Wave
        # blocks alternate with the N index (an even first block).
        n = gl.arange(
            0, scale.shape[1], gl.SliceLayout(0, gl.SliceLayout(2, scale.type.layout))
        )
        shift = 8 * ((i % 2) * 2 + (n // 16) % 2)
        return ((scale >> shift[None, :, None]) & 0xFF).to(gl.uint8)
    else:
        s = gl.arange(0, SPLIT, gl.SliceLayout(1, gl.SliceLayout(2, scale.type.layout)))
        shift = 8 * ((i * SPLIT + s) % 4)
        return ((scale >> shift[:, None, None]) & 0xFF).to(gl.uint8)


@gluon.jit
def gemv_template(SPLIT: gl.constexpr, NW: gl.constexpr, RP: gl.constexpr):
    """Zeros ``[RP, 16 * NW]`` in :func:`gemv`'s result layout, to build a
    stage's per-row inputs in that layout before the GEMV runs."""
    acc = gl.zeros([SPLIT, 16, 16 * NW], gl.float32, mfma_layout(SPLIT, NW))
    out = gl.sum(acc, axis=0)
    return gl.max(gl.reshape(out, [16 // RP, RP, 16 * NW]), axis=0)


@gluon.jit
def _a_tile(
    smem,
    kit: gl.constexpr,
    FIRST: gl.constexpr,
    RP: gl.constexpr,
    SPLIT: gl.constexpr,
    layout: gl.constexpr,
):
    """The A operand (``x_smem``) or its scales (``xs_smem``) of k-iterations
    ``kit ..``, ``[SPLIT, 16, K]`` in ``layout``: MFMA row ``m`` holds
    activation row ``FIRST`` (``RP`` 1) or row ``m % RP``.

    The rows load as ``[SPLIT, RP, K]`` in the operand's own lane order. The
    lanes of MFMA rows ``m`` and ``m + RP`` read the same row. Then a
    broadcast and a reshape make the 16 MFMA rows. These two steps move no
    data.
    """
    K: gl.constexpr = smem.shape[2]
    linear: gl.constexpr = gl.to_linear_layout(layout, [SPLIT, 16, K])
    rows_layout: gl.constexpr = _split_mfma_rows(linear, [SPLIT, 16, K], RP)
    view = smem.permute([1, 0, 2]).slice(kit, SPLIT, 0)
    if RP == 1:
        view = view.slice(FIRST, 1, 1)
    x = view.load(gl.SliceLayout(1, rows_layout))
    x, _ = gl.broadcast(
        gl.expand_dims(x, 1), gl.zeros([SPLIT, 16 // RP, RP, K], x.dtype, rows_layout)
    )
    return gl.convert_layout(gl.reshape(x, [SPLIT, 16, K]), layout)


@gluon.constexpr_function
def _packed_rows(allocated, rows):
    """Rows the A tile packs: every allocated row for a multi-row GEMV."""
    return allocated if rows > 1 else 1


@gluon.constexpr_function
def _split_mfma_rows(linear, shape, RP):
    """``linear`` (``[SPLIT, 16, K]``) as ``[SPLIT, 16 // RP, RP, K]``: MFMA
    row ``m`` becomes ``(m // RP, m % RP)``."""
    bits = RP.bit_length() - 1

    def split(basis):
        s, m, k = basis
        return [s, m >> bits, m & (RP - 1), k]

    return gl.DistributedLinearLayout(
        reg_bases=[split(b) for b in linear.reg_bases],
        lane_bases=[split(b) for b in linear.lane_bases],
        warp_bases=[split(b) for b in linear.warp_bases],
        block_bases=[split(b) for b in linear.block_bases],
        shape=[shape[0], 16 // RP, RP, shape[2]],
    )


@gluon.jit
def mxfp4_exponent(amax):
    """UE8M0 exponent of an MXFP4 block (TokenSpeed's quantizer, Quark's
    "even" mode): ``amax`` rounded up to a power of two when its significand
    is at least 1.75, minus two exponents (E2M1's 4.0 binade), clamped to
    ``[0, 254]``."""
    bits = amax.to(gl.uint32, bitcast=True)
    exponent = (((bits + 0x200000) & 0x7F800000) >> 23).to(gl.int32)
    return gl.minimum(gl.maximum(exponent - 2, 0), 254).to(gl.uint32)


@gluon.jit
def quantize_mxfp4(x):
    """Quantize FP32 ``[G, 32]`` blocks (one block per row) to E2M1 + UE8M0.

    The values are rounded to BF16 first, as the regular path's quantizer
    does. ``x``'s layout must hold at least 8 consecutive values of a row per
    lane. Returns ``(packed E2M1 [G, 16] uint8, uint8 exponents [G])``.
    """
    x = x.to(gl.bfloat16).to(gl.float32)
    exponent = mxfp4_exponent(gl.max(gl.abs(x), axis=1)).to(gl.uint8)
    return cdna4.scaled_downcast(x, exponent[:, None], "e2m1", axis=1), exponent


@gluon.jit
def sigmoid(x):
    """``1 / (1 + exp(-x))`` with a fast reciprocal (2.5 ulp). IEEE division
    costs ~25 VALU instructions per value."""
    return gl.fdiv(1.0, 1.0 + gl.exp(-x), approx=True)


@gluon.jit
def silu(x):
    return gl.fdiv(x, 1.0 + gl.exp(-x), approx=True)


@gluon.jit
def load_mxfp4_rows(
    mb,
    words: gl.constexpr,
    exponents: gl.constexpr,
    KI: gl.constexpr,
    R: gl.constexpr,
    HALVES: gl.constexpr,
    extra_regions: gl.constexpr = (),
    extra_indices=(),
):
    """Poll MXFP4 rows ``0 .. R - 1`` (``KI`` k-iterations each) into LDS for
    :func:`gemv`, in one round trip. With ``HALVES``, LDS k-iteration
    ``2 i + s`` holds k-iteration ``s KI / 2 + i``: the order of a
    ``SPLIT = 2`` GEMV whose waves each take one contiguous half of K
    (``"n32h"`` scales, :func:`load_weight_tiles`). ``words`` holds ``KI *
    16`` int32 words of packed E2M1 bytes a row. ``exponents`` holds ``KI *
    4`` UE8M0 exponents a row (one per 32 values).

    ``extra_regions`` / ``extra_indices`` name other inputs of the caller.
    The function polls them in the same round trips (``mb.poll_groups``).
    Their indices include the rows' offsets (``mb.row_offset``).

    Returns:
        ``(x_smem [RP, KP, 64] uint8, xs_smem [RP, KP, 4] uint8, extras)``,
        ``RP`` is the power of two at least ``R``. Nothing uses the rows past
        ``R``. ``extras[k]`` holds the payloads of ``extra_indices[k]``.
    """
    KP: gl.constexpr = triton.next_power_of_2(KI)
    RP: gl.constexpr = triton.next_power_of_2(R)
    WPK: gl.constexpr = KITER_BYTES // 4  # words per k-iteration
    lanes: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
    w = gl.arange(0, RP * KP * WPK, lanes)
    e = gl.arange(0, RP * KP * 4, lanes)
    w_row = gl.minimum(w // (KP * WPK), R - 1)
    e_row = gl.minimum(e // (KP * 4), R - 1)
    w_at = w % (KP * WPK)
    e_at = e % (KP * 4)
    w_src = w_at
    e_src = e_at
    if HALVES:
        gl.static_assert(KI % 2 == 0)
        w_kit = w_at // WPK
        e_kit = e_at // 4
        w_src = ((w_kit % 2) * (KI // 2) + w_kit // 2) * WPK + w_at % WPK
        e_src = ((e_kit % 2) * (KI // 2) + e_kit // 2) * 4 + e_at % 4
    stride: gl.constexpr = schedule.row_pairs(mb.plan)
    # Throttle: wave 0 waits on the exponents (a producer writes its block's
    # exponent after its words) before every wave polls the rows.
    mb.row(R - 1).pre_poll_spread(exponents, KI * 4)
    got = mb.poll_many(
        _with_extras(words, exponents, extra_regions),
        (
            w_row * stride + gl.minimum(w_src, KI * WPK - 1),
            e_row * stride + gl.minimum(e_src, KI * 4 - 1),
        )
        + extra_indices,
    )
    got_w = got[0]
    got_e = got[1]
    # Each row ends in 32 bytes of padding (4 bytes for the exponents). The
    # MFMA lanes of a multi-row GEMV read 4 rows at once, and rows that start
    # on the same LDS bank conflict: a 4-row GEMV took 2.7 times as long.
    x_words = gl.allocate_shared_memory(
        gl.int32, [RP, KP, WPK], _padded_rows([RP, KP, WPK], KP * WPK, 8)
    )
    xs_smem = gl.allocate_shared_memory(
        gl.uint8, [RP, KP, 4], _padded_rows([RP, KP, 4], KP * 4, 4)
    )
    x_words.store(gl.reshape(gl.where(w_at < KI * WPK, got_w, 0), [RP, KP, WPK]))
    xs_smem.store(
        gl.reshape(gl.where(e_at < KI * 4, got_e, 127).to(gl.uint8), [RP, KP, 4])
    )
    gl.barrier()
    x_smem = x_words.reinterpret(
        gl.uint8,
        [RP, KP, 4 * WPK],
        _padded_rows([RP, KP, 4 * WPK], KP * 4 * WPK, 32),
    )
    return x_smem, xs_smem, got[2:]


@gluon.constexpr_function
def _padded_rows(shape, row, pad):
    """Shared layout of ``shape`` (row-major) with ``pad`` elements after
    every ``row`` elements."""
    return gl.PaddedSharedLayout.with_identity_for([[row, pad]], shape, [2, 1, 0])


@gluon.constexpr_function
def _with_extras(words, exponents, extra_regions):
    return (words, exponents) + tuple(extra_regions)


@gluon.constexpr_function
def block_layout(num_warps):
    """``[blocks, 32]`` with 8 consecutive values per lane (the layout the
    scaled downcast needs). Each row holds one MX block."""
    return gl.BlockedLayout([1, 8], [16, 4], [num_warps, 1], [1, 0])


@gluon.jit
def pack_bytes(data):
    """Bytes ``[G, B]`` (at least 4 consecutive per lane), packed E2M1, into
    ``[G, B / 4]`` little-endian int32 words."""
    G: gl.constexpr = data.shape[0]
    B: gl.constexpr = data.shape[1]
    b = gl.reshape(data.to(gl.uint8, bitcast=True).to(gl.int32), [G, B // 4, 4])
    shift = gl.arange(0, 4, gl.SliceLayout(0, gl.SliceLayout(1, b.type.layout))) * 8
    return gl.sum(b << shift[None, None, :], axis=2)


@gluon.jit
def put_mxfp4(mb, words: gl.constexpr, exponents: gl.constexpr, block0, values):
    """Publish FP32 ``values [G, 32]`` as MXFP4 blocks ``block0 .. block0 + G - 1``.

    ``words`` receives 4 words per block and ``exponents`` one per block.
    """
    G: gl.constexpr = values.shape[0]
    v = gl.convert_layout(values, block_layout(gl.num_warps()))
    data, exponent = quantize_mxfp4(v)
    packed = pack_bytes(data)
    W: gl.constexpr = packed.shape[1]
    g = gl.arange(0, G, gl.SliceLayout(1, packed.type.layout))
    q = gl.arange(0, W, gl.SliceLayout(0, packed.type.layout))
    mb.put(words, (block0 + g)[:, None] * W + q[None, :], packed)
    mb.put(
        exponents, block0 + gl.arange(0, G, exponent.type.layout), exponent.to(gl.int32)
    )


@gluon.jit
def put_mxfp4_rows(mb, words: gl.constexpr, exponents: gl.constexpr, block, values):
    """Publish FP32 ``values [RP, 32]`` as MXFP4 block ``block`` of each row
    (4 words a block). Row ``r`` of ``values`` goes to row ``r``. Rows past
    the launch's rows are padding, and the function does not publish them."""
    RP: gl.constexpr = values.shape[0]
    R: gl.constexpr = schedule.rows(mb.plan)
    v = gl.convert_layout(values, block_layout(gl.num_warps()))
    data, exponent = quantize_mxfp4(v)
    packed = pack_bytes(data)
    W: gl.constexpr = packed.shape[1]
    r = gl.arange(0, RP, gl.SliceLayout(1, packed.type.layout))
    q = gl.arange(0, W, gl.SliceLayout(0, packed.type.layout))
    stride: gl.constexpr = schedule.row_pairs(mb.plan)
    mb.put(
        words,
        (r * stride + block * W)[:, None] + q[None, :],
        packed,
        mask=(r < R)[:, None],
    )
    re = gl.arange(0, RP, exponent.type.layout)
    mb.put(exponents, re * stride + block, exponent.to(gl.int32), mask=re < R)


@gluon.jit
def put_bf16_rows(mb, region: gl.constexpr, pair0, values):
    """Publish ``values [RP, N]`` (N even) as BF16 pairs ``pair0 ..`` of each
    row (rows past the launch's rows are padding)."""
    RP: gl.constexpr = values.shape[0]
    N: gl.constexpr = values.shape[1]
    R: gl.constexpr = schedule.rows(mb.plan)
    pairs: gl.constexpr = gl.BlockedLayout([1, 2], [1, 64], [gl.num_warps(), 1], [1, 0])
    values = gl.convert_layout(values.to(gl.bfloat16), pairs)
    word = pack_bf16_pairs(gl.reshape(values, [RP, N // 2, 2]))
    r = gl.arange(0, RP, gl.SliceLayout(1, word.type.layout))
    p = gl.arange(0, N // 2, gl.SliceLayout(0, word.type.layout))
    stride: gl.constexpr = schedule.row_pairs(mb.plan)
    mb.put(
        region, (r * stride + pair0)[:, None] + p[None, :], word, mask=(r < R)[:, None]
    )


@gluon.jit
def put_bf16(mb, region: gl.constexpr, pair0, values):
    """Publish ``values`` (1-D, even length) as BF16 pairs ``pair0 ..``."""
    N: gl.constexpr = values.shape[0]
    # split needs a pair's two values in one thread.
    pairs: gl.constexpr = gl.BlockedLayout([2], [64], [gl.num_warps()], [0])
    values = gl.convert_layout(values.to(gl.bfloat16), pairs)
    word = pack_bf16_pairs(gl.reshape(values, [N // 2, 2]))
    mb.put(region, pair0 + gl.arange(0, N // 2, word.type.layout), word)


@gluon.jit
def pack_bf16_pairs(values):
    """BF16 ``values [..., 2]`` (a pair in one thread) as int32 mailbox
    payloads, element 0 in the low half."""
    lo, hi = gl.split(values)
    word = lo.to(gl.int16, bitcast=True).to(gl.int32) & 0xFFFF
    return word | (hi.to(gl.int16, bitcast=True).to(gl.int32) << 16)


@gluon.jit
def bf16_half(word, element):
    """FP32 of BF16 element ``element % 2`` of a polled pair word."""
    bits = gl.where(element % 2 == 0, word << 16, word & -65536)
    return bits.to(gl.float32, bitcast=True)
