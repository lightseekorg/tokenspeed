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

Split chains (:func:`load_split_tiles`, :func:`split_chain`,
:func:`split_sum`) give each output the arithmetic of the regular path's
split-K projection (``gemm/fp16/fp32_output.py``): K splits into ``SPLITS``
equal slices, each slice is one MFMA chain from zero in K order, and the
slices' FP32 partials add in split order, ``p0 + p1 + ...``. A slot of the
rank-3 MFMA layout's batch dimension runs one chain: the waves own the
slots, ``SLOTS`` of them a pass, and ``ceil(SPLITS / SLOTS)`` passes cover
the splits (slots past ``SPLITS`` load nothing and are dropped). One MFMA
step of a pass multiplies the same chain step of every slot, so the LDS
chunk of a pass holds ``C`` steps of every slot: kit ``c * SLOTS + s`` of
the chunk is step ``i0 + c`` of slot ``s`` (:func:`load_split_rows`). Past 16
activation rows, a chain's MFMA rows are the rows themselves: each 16-row
block of them is one more MFMA a step on the same weight tile, in the same K
order, so a row's result does not depend on the launch's rows.
"""

from __future__ import annotations

from tokenspeed_kernel_amd._triton import gl, gluon, triton
from tokenspeed_kernel_amd.ops.gfx950.monokernel import mx, schedule

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
def split_waves(n_waves, num_warps):
    """Waves along the slot dimension of a split chain with ``n_waves`` N16
    waves."""
    return num_warps // n_waves


@gluon.constexpr_function
def split_passes(splits, slots):
    return (splits + slots - 1) // slots


@gluon.jit
def load_split_tiles(
    w,
    nb0,
    K: gl.constexpr,
    SPLITS: gl.constexpr,
    FIRST: gl.constexpr,
    COUNT: gl.constexpr,
    SLOTS: gl.constexpr,
    NW: gl.constexpr,
):
    """Issue every weight load of one split-chain GEMV task. Nothing waits
    here.

    Args:
        w: The BF16 weight ``[N, K]`` in ``weights.bf16_tiles`` order.
        nb0: First 16-row block of the task; N wave ``v`` reads block
            ``nb0 + v``.
        K: The weight's reduction width.
        SPLITS: K slices (``K / (32 SPLITS)`` k-iterations each), as the
            regular projection's split-K.
        FIRST, COUNT: The task's splits ``FIRST .. FIRST + COUNT - 1``.
        SLOTS: Chains a pass (a power of two, at least the slot waves).
        NW: N waves (16 rows each).

    Returns:
        A tuple over passes ``p`` of tuples over chain steps ``i`` of
        ``[SLOTS, 32, 16 * NW]`` B tiles: slot ``s`` holds k-iteration
        ``i`` of split ``FIRST + p * SLOTS + s`` (zeros past the task's).
    """
    gl.static_assert(K % (32 * SPLITS) == 0)
    KS: gl.constexpr = K // 32 // SPLITS
    WS: gl.constexpr = split_waves(NW, gl.num_warps())
    gl.static_assert(SLOTS % WS == 0)
    bl: gl.constexpr = b_layout(WS, NW)
    s = gl.arange(0, SLOTS, gl.SliceLayout(1, gl.SliceLayout(2, bl)))
    k = gl.arange(0, 32, gl.SliceLayout(0, gl.SliceLayout(2, bl)))
    n = gl.arange(0, 16 * NW, gl.SliceLayout(0, gl.SliceLayout(1, bl)))
    block = nb0 + n // 16
    # As load_weight_tiles: lane g * 16 + n reads values 128 g + 8 n ..
    lane = ((k // 8) * 128 + k % 8)[None, :, None] + ((n % 16) * 8)[None, None, :]
    gl.static_assert(FIRST + COUNT <= SPLITS)
    passes = ()
    for p in gl.static_range(split_passes(COUNT, SLOTS)):
        split = p * SLOTS + s
        valid = (split < COUNT)[:, None, None] & (lane >= 0)
        first = (FIRST + gl.minimum(split, COUNT - 1)) * KS
        offset = (block[None, None, :] * (K // 32) + first[:, None, None]) * 512 + lane
        tiles = ()
        for i in gl.static_range(KS):
            tiles += (
                cdna4.buffer_load(
                    w, offset + i * 512, mask=valid, other=0, cache=".cg"
                ),
            )
        passes += (tiles,)
    return passes


@gluon.jit
def split_zeros(SLOTS: gl.constexpr, NW: gl.constexpr, RP: gl.constexpr):
    """A pass's zero accumulators ``[SLOTS, max(16, RP), 16 * NW]`` for ``RP``
    activation rows (:func:`split_chain`)."""
    WS: gl.constexpr = split_waves(NW, gl.num_warps())
    return gl.zeros([SLOTS, _mfma_rows(RP), 16 * NW], gl.float32, mfma_layout(WS, NW))


@gluon.constexpr_function
def _mfma_rows(rows):
    """MFMA rows of a split chain of ``rows`` activation rows."""
    return max(16, rows)


@gluon.jit
def split_chain(x_smem, tiles, i0: gl.constexpr, acc):
    """Chain steps ``i0 ..`` of one pass: ``acc`` (:func:`split_zeros`)
    plus the products of the steps that ``x_smem`` holds
    (:func:`load_split_rows`) with ``tiles`` (a pass of
    :func:`load_split_tiles`), one MFMA a step in K order."""
    SLOTS: gl.constexpr = tiles[0].shape[0]
    NW: gl.constexpr = tiles[0].shape[2] // 16
    al: gl.constexpr = a_layout(split_waves(NW, gl.num_warps()), NW)
    RP: gl.constexpr = x_smem.shape[0]
    C: gl.constexpr = x_smem.shape[1] // SLOTS
    for c in gl.static_range(C):
        if RP > 16:
            # MFMA row m is activation row m: one MFMA a 16-row block.
            a = x_smem.permute([1, 0, 2]).slice(c * SLOTS, SLOTS, 0).load(al)
        else:
            a = _a_tile(x_smem, c * SLOTS, 0, RP, SLOTS, al)
        acc = cdna4.mfma(a, tiles[i0 + c], acc)
    return acc


@gluon.constexpr_function
def sum_layout(num_warps):
    """``[RP, N]`` layout of :func:`split_sum` (two columns a thread, as
    ``mx.put_bf16_rows`` packs them)."""
    return gl.BlockedLayout([1, 2], [1, 64], [num_warps, 1], [1, 0])


@gluon.jit
def split_sum(accs, COUNT: gl.constexpr, RP: gl.constexpr, prev):
    """FP32 ``[RP, N]``: the partials of the ``COUNT`` chains of ``accs``
    (a tuple of passes of :func:`split_chain`) added in split order, from
    the first one's or onto ``prev`` (the sum of the splits before them, in
    :func:`sum_layout`; None for the first), as the regular projection's
    split-K reduction. The partials meet in LDS, then every thread adds its
    outputs' partials."""
    SLOTS: gl.constexpr = accs[0].shape[0]
    N: gl.constexpr = accs[0].shape[2]
    # Slot s of pass p at rows (p * SLOTS + s) * RP ..
    parts = gl.allocate_shared_memory(
        gl.float32,
        [triton.next_power_of_2(len(accs) * SLOTS) * RP, N],
        gl.SwizzledSharedLayout(1, 1, 1, [1, 0]),
    )
    for p in gl.static_range(len(accs)):
        if RP > 16:
            rows = accs[p]
        else:
            # MFMA row m holds the result of activation row m % RP, so rows
            # m and m + RP agree.
            rows = gl.max(gl.reshape(accs[p], [SLOTS, 16 // RP, RP, N]), axis=1)
        parts.slice(p * SLOTS * RP, SLOTS * RP, 0).store(
            gl.reshape(rows, [SLOTS * RP, N])
        )
    gl.barrier()
    layout: gl.constexpr = sum_layout(gl.num_warps())
    if prev is None:
        total = parts.slice(0, RP, 0).load(layout)
    else:
        total = prev + parts.slice(0, RP, 0).load(layout)
    for s in gl.static_range(1, COUNT):
        total += parts.slice(s * RP, RP, 0).load(layout)
    return total


@gluon.constexpr_function
def _with_extras(region, extra_regions):
    return (region,) + tuple(extra_regions)


# Pairs per thread of one poll of load_rows (see "Few pairs per thread").
POLL_PAIRS = 16


@gluon.constexpr_function
def _poll_group_rows(RP, KP, num_warps):
    """Rows of one poll of :func:`load_rows`: a power of two, at most
    ``POLL_PAIRS`` pairs per thread unless one row is wider."""
    group = RP
    while group > 1 and group * KP * 16 > POLL_PAIRS * 64 * num_warps:
        group //= 2
    return group


@gluon.jit
def load_split_rows(
    mb,
    region: gl.constexpr,
    p: gl.constexpr,
    i0: gl.constexpr,
    C: gl.constexpr,
    SLOTS: gl.constexpr,
    SPLITS: gl.constexpr,
    FIRST: gl.constexpr,
    COUNT: gl.constexpr,
    K: gl.constexpr,
    R: gl.constexpr,
    WORDS: gl.constexpr,
):
    """Poll the activation k-iterations of chain steps ``i0 .. i0 + C - 1``
    of pass ``p`` (:func:`load_split_tiles` of splits ``FIRST .. FIRST +
    COUNT - 1``) of rows ``0 .. R - 1`` into LDS, in one round trip per
    group of rows (as :func:`load_rows`).

    ``region`` holds each row's ``K`` values as BF16 pairs, or with
    ``WORDS`` as plain words (word ``j`` of a row holds the values of pair
    ``j``; ``Mailbox.store_words``) whose flags the caller polled: they load
    past the caches, half the bytes of tagged pairs and no tag checks.

    Returns:
        ``x_smem [RP, C * SLOTS, 32] bf16``: kit ``c * SLOTS + s`` holds
        step ``i0 + c`` of slot ``s``, k-iteration ``(FIRST + p * SLOTS +
        s) * KS + i0 + c`` of the row (zeros for slots past the task's).
    """
    KS: gl.constexpr = K // 32 // SPLITS
    KP: gl.constexpr = C * SLOTS
    gl.static_assert(KP == triton.next_power_of_2(KP) and i0 + C <= KS)
    RP: gl.constexpr = triton.next_power_of_2(R)
    G: gl.constexpr = _poll_group_rows(RP, KP, gl.num_warps())
    lanes: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
    q = gl.arange(0, G * KP * 16, lanes)
    at = q % (KP * 16)
    kit = at // 16
    split = p * SLOTS + kit % SLOTS
    # Slots past the task's splits poll a real pair and store zeros.
    pair = (
        (FIRST + gl.minimum(split, COUNT - 1)) * KS + i0 + kit // SLOTS
    ) * 16 + at % 16
    valid = split < COUNT
    stride: gl.constexpr = schedule.row_pairs(mb.plan)
    words = gl.allocate_shared_memory(
        gl.int32, [RP, KP, 16], mx._padded_rows([RP, KP, 16], KP * 16, 8)
    )
    for g in gl.static_range(RP // G):
        row = gl.minimum(g * G + q // (KP * 16), R - 1)
        if WORDS:
            got = mb.load_words(region, 2 * mb.row_offset(row) + pair)
        else:
            (got,) = mb.poll_many((region,), (row * stride + pair,))
        words.slice(g * G, G, 0).store(gl.reshape(gl.where(valid, got, 0), [G, KP, 16]))
    gl.barrier()
    return words.reinterpret(
        gl.bfloat16, [RP, KP, 32], mx._padded_rows([RP, KP, 32], KP * 32, 16)
    )


@gluon.jit
def load_word_rows(
    mb,
    region: gl.constexpr,
    flags: gl.constexpr,
    FLAGS: gl.constexpr,
    KS: gl.constexpr,
    R: gl.constexpr,
):
    """:func:`load_rows` of rows ``0 .. R - 1`` whose producers publish BF16
    pairs as plain words in ``region`` (word ``k / 2`` of a row holds values
    ``k, k + 1``; :meth:`Mailbox.store_words_peer`) and each raise one of
    the ``FLAGS`` flags of the row in ``flags``. The flags are polled first,
    then the words are loaded past the caches: half the bytes of tagged
    pairs.

    Returns:
        ``x_smem [RP, KP, 32] bf16``, as :func:`load_rows` with ``k0 = 0``.
    """
    gl.static_assert(KS % 32 == 0)
    KI: gl.constexpr = KS // 32
    KP: gl.constexpr = triton.next_power_of_2(KI)
    RP: gl.constexpr = triton.next_power_of_2(R)
    nw: gl.constexpr = gl.num_warps()
    lanes: gl.constexpr = gl.BlockedLayout([1], [64], [nw], [0])
    f = gl.arange(0, triton.next_power_of_2(R * FLAGS), lanes)
    mb.poll_flags(flags, mb.row_offset(gl.minimum(f // FLAGS, R - 1)) + f % FLAGS)
    # No load may issue before every wave's poll returned.
    gl.barrier()
    G: gl.constexpr = _poll_group_rows(RP, KP, nw)
    p = gl.arange(0, G * KP * 16, lanes)
    at = p % (KP * 16)
    word = gl.minimum(at, KS // 2 - 1)
    stride: gl.constexpr = schedule.row_pairs(mb.plan)
    words = gl.allocate_shared_memory(
        gl.int32, [RP, KP, 16], mx._padded_rows([RP, KP, 16], KP * 16, 8)
    )
    for g in gl.static_range(RP // G):
        row = gl.minimum(g * G + p // (KP * 16), R - 1)
        got = mb.load_words(region, 2 * row * stride + word)
        words.slice(g * G, G, 0).store(
            gl.reshape(gl.where(at < KS // 2, got, 0), [G, KP, 16])
        )
    gl.barrier()
    return words.reinterpret(
        gl.bfloat16, [RP, KP, 32], mx._padded_rows([RP, KP, 32], KP * 32, 16)
    )


@gluon.jit
def load_local_word_rows(
    mb, region: gl.constexpr, k0, KS: gl.constexpr, R: gl.constexpr
):
    """:func:`load_rows` of BF16 rows that producers published as plain
    words (``Mailbox.store_words`` / ``raise_flag``) in local region
    ``region``, values ``k0 .. k0 + KS - 1`` of rows ``0 .. R - 1``. Call it
    after the CTA polled the producers' flags (and met at a barrier). One
    round trip: a word holds two values, so all the rows fit the registers
    that :func:`load_rows` gives a group of rows' pairs. Unlike
    :func:`load_word_rows` the caller polls the flags (for a slice ``k0 ..``
    of the rows, its own subset of them).

    Returns:
        ``x_smem [RP, KP, 32] bf16``, as :func:`load_rows`.
    """
    gl.static_assert(KS % 32 == 0)
    KI: gl.constexpr = KS // 32
    KP: gl.constexpr = triton.next_power_of_2(KI)
    RP: gl.constexpr = triton.next_power_of_2(R)
    lanes: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
    p = gl.arange(0, RP * KP * 16, lanes)
    row = gl.minimum(p // (KP * 16), R - 1)
    at = p % (KP * 16)
    # Row r's words start at word 2 row_offset(r) (Mailbox.store_words).
    got = mb.load_words(
        region, 2 * mb.row_offset(row) + k0 // 2 + gl.minimum(at, KS // 2 - 1)
    )
    words = gl.allocate_shared_memory(
        gl.int32, [RP, KP, 16], mx._padded_rows([RP, KP, 16], KP * 16, 8)
    )
    words.store(gl.reshape(gl.where(at < KS // 2, got, 0), [RP, KP, 16]))
    gl.barrier()
    return words.reinterpret(
        gl.bfloat16, [RP, KP, 32], mx._padded_rows([RP, KP, 32], KP * 32, 16)
    )


@gluon.jit
def load_global_rows(x, row_stride, k0, KS: gl.constexpr, R: gl.constexpr):
    """:func:`load_rows` of BF16 rows in global memory that an earlier launch
    wrote (row ``r`` at ``x + r * row_stride``), values ``k0 .. k0 + KS - 1``.
    ``row_stride`` and ``k0`` are even.

    The loads are plain (cacheable), so this GPU must have written the rows
    in an earlier launch: a launch boundary makes its own writes visible to
    them. Rows that another GPU wrote need loads past the caches (``.cv``),
    which a stale line in this GPU's L2 cannot serve.

    Returns:
        ``x_smem [RP, KP, 32] bf16``, as :func:`load_rows`.
    """
    gl.static_assert(KS % 32 == 0)
    KI: gl.constexpr = KS // 32
    KP: gl.constexpr = triton.next_power_of_2(KI)
    RP: gl.constexpr = triton.next_power_of_2(R)
    lanes: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
    p = gl.arange(0, RP * KP * 16, lanes)
    row = gl.minimum(p // (KP * 16), R - 1)
    at = p % (KP * 16)
    words_ptr = x.to(gl.pointer_type(gl.int32), bitcast=True)
    got = gl.load(
        words_ptr + (row * row_stride + k0) // 2 + gl.minimum(at, KS // 2 - 1),
        mask=at < KS // 2,
        other=0,
    )
    words = gl.allocate_shared_memory(
        gl.int32, [RP, KP, 16], mx._padded_rows([RP, KP, 16], KP * 16, 8)
    )
    words.store(gl.reshape(got, [RP, KP, 16]))
    gl.barrier()
    return words.reinterpret(
        gl.bfloat16, [RP, KP, 32], mx._padded_rows([RP, KP, 32], KP * 32, 16)
    )


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
    into LDS for :func:`gemv`, in one round trip per group of rows (a wide
    poll of many rows would hold too many registers).

    ``region`` holds each row as BF16 pairs (:func:`mx.put_bf16_rows`);
    ``k0`` and ``KS`` are even. ``extra_regions`` / ``extra_indices`` poll
    other inputs of the caller with the first group (as
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
    G: gl.constexpr = _poll_group_rows(RP, KP, gl.num_warps())
    lanes: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
    p = gl.arange(0, G * KP * 16, lanes)
    at = p % (KP * 16)
    pair = k0 // 2 + gl.minimum(at, KS // 2 - 1)
    stride: gl.constexpr = schedule.row_pairs(mb.plan)
    # Each row ends in 32 bytes of padding: rows that start on the same LDS
    # bank conflict when the MFMA lanes of a multi-row GEMV read them.
    words = gl.allocate_shared_memory(
        gl.int32, [RP, KP, 16], mx._padded_rows([RP, KP, 16], KP * 16, 8)
    )
    row = gl.minimum(p // (KP * 16), R - 1)
    got = mb.poll_many(
        _with_extras(region, extra_regions), (row * stride + pair,) + extra_indices
    )
    words.slice(0, G, 0).store(
        gl.reshape(gl.where(at < KS // 2, got[0], 0), [G, KP, 16])
    )
    for g in gl.static_range(1, RP // G):
        row = gl.minimum(g * G + p // (KP * 16), R - 1)
        (more,) = mb.poll_many((region,), (row * stride + pair,))
        words.slice(g * G, G, 0).store(
            gl.reshape(gl.where(at < KS // 2, more, 0), [G, KP, 16])
        )
    gl.barrier()
    x_smem = words.reinterpret(
        gl.bfloat16, [RP, KP, 32], mx._padded_rows([RP, KP, 32], KP * 32, 16)
    )
    return x_smem, got[1:]
