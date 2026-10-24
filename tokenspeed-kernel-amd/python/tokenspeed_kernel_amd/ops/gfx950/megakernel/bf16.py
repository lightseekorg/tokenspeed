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

"""BF16-weight x BF16-activation decode GEMV building blocks.

The BF16 sibling of ``mx.py``, for checkpoints that keep a projection in
BF16.

One ``v_mfma_f32_16x16x32_bf16`` multiplies a 16x32 A tile (up to 16
activation rows, MFMA row ``m`` holding row ``m % RP``) by a 32x16 B tile (16
weight rows). Lane ``g * 16 + n`` of the B operand holds row ``n``, K
values ``8 g .. 8 g + 7``, one ``dwordx4``. The weight is in the N16 tile
order of ``weights.bf16_tiles``: one k-iteration of one 16-row block is
1 KiB of contiguous memory in lane order, as in ``mx.py``. A GEMV load of
the row-major weight would read 64 bytes of each of 16 rows, and a fully
loaded GPU reads that pattern at only about 0.7 times the rate.

The K split works as in ``mx.py``: the waves own the batch dimension of a
rank-3 MFMA layout, wave ``s`` runs k-iterations ``s, s + S, ...``, and a
fixed-order sum over the batch dimension combines them
(:func:`mx.gemv_result`). A row's result depends only on its own A row.

Activations sit in LDS as ``[RP, KP, 32]`` BF16 (k-iteration major), the
task's own K range only: a BF16 row is four times the bytes of an MXFP4 row.
"""

from __future__ import annotations

from tokenspeed_kernel_amd._triton import gl, gluon, triton
from tokenspeed_kernel_amd.ops.gfx950.megakernel import mx, schedule

cdna4 = gl.amd.cdna4

KITER = gl.constexpr(32)  # K values per MFMA


@gluon.constexpr_function
def mfma_layout(split_k, n_waves):
    """Rank-3 BF16 MFMA layout: ``split_k`` K-split waves x ``n_waves`` N16 waves."""
    return gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[split_k, 1, n_waves],
    )


@gluon.constexpr_function
def a_layout(split_k, n_waves):
    return gl.DotOperandLayout(0, mfma_layout(split_k, n_waves), 8)


@gluon.constexpr_function
def b_layout(split_k, n_waves):
    return gl.DotOperandLayout(1, mfma_layout(split_k, n_waves), 8)


@gluon.jit
def load_weight_tiles(
    w,
    nb0,
    K: gl.constexpr,
    kit0,
    KITS: gl.constexpr,
    SPLIT: gl.constexpr,
    NW: gl.constexpr,
    GROUP: gl.constexpr,
    GROUP_STRIDE: gl.constexpr,
):
    """Issue every weight load of one GEMV task. Nothing waits here.

    Args:
        w: The BF16 weight ``[N, K]`` in ``weights.bf16_tiles`` order.
        nb0: First 16-row block of the task. N wave ``v`` reads block
            ``nb0 + (v // GROUP) * GROUP_STRIDE + v % GROUP``.
        K: The weight's reduction width (a multiple of 32).
        kit0, KITS: The task's k-iterations ``kit0 .. kit0 + KITS - 1`` (32
            values each, ``KITS`` a multiple of ``SPLIT``): a split-K across
            tasks.
        SPLIT: K-split waves.
        NW: N waves (16 rows each).
        GROUP, GROUP_STRIDE: As in :func:`mx.load_weight_tiles`.

    Returns:
        One ``[SPLIT, 32, 16 * NW]`` B tile per chain step.
    """
    gl.static_assert(K % 32 == 0 and KITS % SPLIT == 0 and KITS <= K // 32)
    bl: gl.constexpr = b_layout(SPLIT, NW)
    s = gl.arange(0, SPLIT, gl.SliceLayout(1, gl.SliceLayout(2, bl)))
    k = gl.arange(0, 32, gl.SliceLayout(0, gl.SliceLayout(2, bl)))
    n = gl.arange(0, 16 * NW, gl.SliceLayout(0, gl.SliceLayout(1, bl)))
    v = n // 16
    block = nb0 + (v // GROUP) * GROUP_STRIDE + v % GROUP
    # 512 values per (block, k-iteration): lane g * 16 + n reads values
    # 128 g + 8 n .. of it.
    lane = ((k // 8) * 128 + k % 8)[None, :, None] + ((n % 16) * 8)[None, None, :]
    offset = (block[None, None, :] * (K // 32) + (kit0 + s)[:, None, None]) * 512 + lane
    tiles = ()
    for i in gl.static_range(KITS // SPLIT):
        # A launch reads each weight once, and its whole 1 KiB rows: loads
        # that skip L1 (".cg") stream faster.
        tiles += (cdna4.buffer_load(w, offset + i * SPLIT * 512, cache=".cg"),)
    return tiles


@gluon.jit
def load_row_major_tiles(
    w,
    nb0,
    K: gl.constexpr,
    kit0,
    KITS: gl.constexpr,
    SPLIT: gl.constexpr,
    NW: gl.constexpr,
    GROUP: gl.constexpr,
    GROUP_STRIDE: gl.constexpr,
):
    """:func:`load_weight_tiles` of a row-major ``[N, K]`` weight, read in
    place (for weights that are too large to copy, such as a model's MoE
    banks). A load reads 64 bytes of each of 16 rows. ``w`` and ``N * K``
    must fit 32-bit element offsets from ``w``."""
    gl.static_assert(K % 32 == 0 and KITS % SPLIT == 0 and KITS <= K // 32)
    bl: gl.constexpr = b_layout(SPLIT, NW)
    s = gl.arange(0, SPLIT, gl.SliceLayout(1, gl.SliceLayout(2, bl)))
    k = gl.arange(0, 32, gl.SliceLayout(0, gl.SliceLayout(2, bl)))
    n = gl.arange(0, 16 * NW, gl.SliceLayout(0, gl.SliceLayout(1, bl)))
    v = n // 16
    row = (nb0 + (v // GROUP) * GROUP_STRIDE + v % GROUP) * 16 + n % 16
    offset = row[None, None, :] * K + (kit0 + s)[:, None, None] * 32 + k[None, :, None]
    tiles = ()
    for i in gl.static_range(KITS // SPLIT):
        tiles += (cdna4.buffer_load(w, offset + i * SPLIT * 32),)
    return tiles


@gluon.jit
def gemv(x_smem, tiles, FIRST: gl.constexpr, ROWS: gl.constexpr):
    """Activation rows times the loaded weight tiles.

    Args:
        x_smem: ``[R, KP, 32]`` BF16 activation rows in LDS, the task's K
            range (:func:`load_rows`).
        tiles: From :func:`load_weight_tiles`.
        FIRST, ROWS: The rows to multiply, ``FIRST .. FIRST + ROWS - 1``:
            one row, or all ``R`` rows from 0.

    Returns:
        FP32 ``[RP, 16 * NW]`` as :func:`mx.gemv`.
    """
    SPLIT: gl.constexpr = tiles[0].shape[0]
    NW: gl.constexpr = tiles[0].shape[2] // 16
    al: gl.constexpr = a_layout(SPLIT, NW)
    RP: gl.constexpr = x_smem.shape[0] if ROWS > 1 else 1
    gl.static_assert(ROWS == 1 or (FIRST == 0 and ROWS <= RP and RP <= 16))
    acc = gl.zeros([SPLIT, 16, 16 * NW], gl.float32, mfma_layout(SPLIT, NW))
    for i in gl.static_range(len(tiles)):
        a = _a_tile(x_smem, i * SPLIT, FIRST, RP, SPLIT, al)
        acc = cdna4.mfma(a, tiles[i], acc)
    return mx.gemv_result(acc, RP)


@gluon.jit
def gemv_template(SPLIT: gl.constexpr, NW: gl.constexpr, RP: gl.constexpr):
    """Zeros ``[RP, 16 * NW]`` in :func:`gemv`'s result layout (as
    :func:`mx.gemv_template`)."""
    acc = gl.zeros([SPLIT, 16, 16 * NW], gl.float32, mfma_layout(SPLIT, NW))
    return mx.gemv_result(acc, RP)


@gluon.jit
def _a_tile(
    smem,
    kit: gl.constexpr,
    FIRST: gl.constexpr,
    RP: gl.constexpr,
    SPLIT: gl.constexpr,
    layout: gl.constexpr,
):
    """The A operand of k-iterations ``kit ..``, ``[SPLIT, 16, 32]`` in
    ``layout``: MFMA row ``m`` holds row ``FIRST`` (``RP`` 1) or ``m % RP``
    (as :func:`mx._a_tile`)."""
    linear: gl.constexpr = gl.to_linear_layout(layout, [SPLIT, 16, 32])
    rows_layout: gl.constexpr = mx._split_mfma_rows(linear, [SPLIT, 16, 32], RP)
    view = smem.permute([1, 0, 2]).slice(kit, SPLIT, 0)
    if RP == 1:
        view = view.slice(FIRST, 1, 1)
    x = view.load(gl.SliceLayout(1, rows_layout))
    x, _ = gl.broadcast(
        gl.expand_dims(x, 1), gl.zeros([SPLIT, 16 // RP, RP, 32], x.dtype, rows_layout)
    )
    return gl.convert_layout(gl.reshape(x, [SPLIT, 16, 32]), layout)


@gluon.constexpr_function
def _with_extras(region, extra_regions):
    return (region,) + tuple(extra_regions)


@gluon.jit
def load_rows(
    mb,
    region: gl.constexpr,
    k0,
    KS: gl.constexpr,
    R: gl.constexpr,
    extra_regions: gl.constexpr = (),
    extra_indices=(),
):
    """Poll BF16 rows ``0 .. R - 1``, values ``k0 .. k0 + KS - 1`` of each,
    into LDS for :func:`gemv`, in one round trip.

    ``region`` holds each row as BF16 pairs (:func:`mx.put_bf16_rows`);
    ``k0`` and ``KS`` are even. ``extra_regions`` / ``extra_indices`` poll
    other inputs of the caller in the same round trips (as
    :func:`mx.load_mxfp4_rows`).

    Returns:
        ``(x_smem [RP, KP, 32] bf16, extras)``: k-iteration ``i`` holds
        values ``k0 + 32 i ..``. ``RP`` and ``KP`` are powers of two;
        values past ``KS`` are 0. Nothing uses the rows past ``R``.
    """
    gl.static_assert(KS % 32 == 0)
    KI: gl.constexpr = KS // 32
    KP: gl.constexpr = triton.next_power_of_2(KI)
    RP: gl.constexpr = triton.next_power_of_2(R)
    lanes: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
    p = gl.arange(0, RP * KP * 16, lanes)
    row = gl.minimum(p // (KP * 16), R - 1)
    at = p % (KP * 16)
    stride: gl.constexpr = schedule.row_pairs(mb.plan)
    got = mb.poll_many(
        _with_extras(region, extra_regions),
        (row * stride + k0 // 2 + gl.minimum(at, KS // 2 - 1),) + extra_indices,
    )
    # Each row ends in 32 bytes of padding: rows that start on the same LDS
    # bank conflict when the MFMA lanes of a multi-row GEMV read them.
    words = gl.allocate_shared_memory(
        gl.int32, [RP, KP, 16], mx._padded_rows([RP, KP, 16], KP * 16, 8)
    )
    words.store(gl.reshape(gl.where(at < KS // 2, got[0], 0), [RP, KP, 16]))
    gl.barrier()
    x_smem = words.reinterpret(
        gl.bfloat16, [RP, KP, 32], mx._padded_rows([RP, KP, 32], KP * 32, 16)
    )
    return x_smem, got[1:]
