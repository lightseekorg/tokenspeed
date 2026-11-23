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

"""BF16 projection stages and the tensor-parallel exchange of their partials.

A projection task multiplies every row of the launch by ``BLOCKS`` 16-row
blocks of a BF16 weight in ``weights.bf16_tiles`` order, so each weight is
read once a launch. Each output follows the regular path's FP32 projection
(``gluon_mm_bf16_fp32_gfx950``) bit for bit: K splits into that kernel's
``SPLITS`` slices (``projection_split_k`` of the weight's ``[N, K]``),
each one MFMA chain from zero in K order, and the slices' partials add in
split order (``bf16.py``, "Split chains"). The FP32 result is rounded once
to BF16, as the regular path's ``.to(bfloat16)`` after its GEMM does; the
router's logits stay FP32. The plan's dimensions give the shapes and the
splits (``Dims.qkvr_splits``, ``router_splits``, ``wo_splits``): at hidden
4608 the QKV projection splits K 4608 in 12, the router in 18 and the output
projection K 1536 in 3; at hidden 2304 K 2304 in 6 and 9, K 768 in 2. A full
layer's K and V source projections take the QKV projection's split (the
regular path's one stacked GEMM). Every split count works with any slot
count of a pass: slots past a group's splits load nothing.

A task loads every row's activations of the K splits it adds. Past 16 rows
those loads would outgrow its weight loads, so a projection splits K over
chained groups of tasks (:func:`chained_stages`): group ``g``'s tasks add
their splits' products onto group ``g - 1``'s FP32 sums (:func:`chain_stage`)
in split order, which leaves every sum as it is.

The activation rows sit in LDS one chunk at a time: ``C`` chain steps of
each of a pass's ``SLOTS`` chains, ``CHUNK_KITS`` k-iterations (a whole
4608-value row would take 128 KiB of LDS at 8 rows).

An exchanged projection (the attention output projection) is a row-parallel
partial: every rank holds a ``[rows, N]`` partial of the full sum. Its tasks
store the BF16 partial as plain words into every rank's copy of a peer region
(``[source rank][N]`` per row) and raise one flag per task on every rank.
A reduce task polls the four ranks' flags of its columns, loads the four
partials past the caches and adds them in rank order 0, 1, 2, 3 in FP32,
then rounds once to BF16: the regular path's rank-ordered fold, without a
second all-gather. Its tasks are the output's 32-column blocks (``hidden /
32``: 144 at hidden 4608, 72 at 2304); scattered, each rank owns the blocks
``world`` apart from its rank on, so the block count divides by ``world``.
"""

from __future__ import annotations

from tokenspeed_kernel_amd._triton import gl, gluon, triton
from tokenspeed_kernel_amd.ops.gfx950.monokernel import bf16, mx, schedule, trace
from tokenspeed_kernel_amd.ops.gfx950.monokernel.inkling2 import plan as ip

# Activation k-iterations a GEMV chunk holds in LDS at 16 rows or fewer (64
# KiB at 16 rows); more rows hold fewer.
CHUNK_KITS = 64


@gluon.constexpr_function
def chunk_steps(steps, slots, rows):
    """Chain steps a chunk holds: the largest power of two that divides
    ``steps`` and fits ``CHUNK_KITS`` (``16 / rows`` of it past 16 rows)
    with ``slots`` chains."""
    c = max(CHUNK_KITS * 16 // max(rows, 16) // slots, 1)
    while steps % c:
        c //= 2
    return c


@gluon.jit
def issue(
    w,
    task,
    K: gl.constexpr,
    BLOCKS: gl.constexpr,
    SPLITS: gl.constexpr,
    FIRST: gl.constexpr,
    COUNT: gl.constexpr,
    SLOTS: gl.constexpr,
):
    """Issue every weight load of projection task ``task``: ``BLOCKS``
    16-row blocks from block ``task * BLOCKS``, K splits ``FIRST .. FIRST +
    COUNT - 1`` (``bf16.load_split_tiles``)."""
    return bf16.load_split_tiles(
        w, task * BLOCKS, K, SPLITS, FIRST, COUNT, SLOTS, BLOCKS
    )


@gluon.jit
def gemv(
    mb,
    task,
    x: gl.constexpr,
    run: gl.constexpr,
    tiles,
    K: gl.constexpr,
    SPLITS: gl.constexpr,
    FIRST: gl.constexpr,
    COUNT: gl.constexpr,
    x_flag: gl.constexpr,
    X_FLAGS: gl.constexpr,
):
    """FP32 ``[RP, 16 * BLOCKS]`` sums over K splits ``.. FIRST + COUNT -
    1`` of every row of region ``x`` (``K`` values a row) times the tiles
    of :func:`issue`, in the regular projection's order: the task's splits'
    products added in split order onto the sum of the splits before
    ``FIRST`` (region ``run``, an earlier task's :func:`chain_stage`; None
    when ``FIRST`` is 0). ``x`` holds BF16 pairs, or plain words that its
    producers publish with ``X_FLAGS`` flags a row in region ``x_flag``
    (None for pairs)."""
    R: gl.constexpr = schedule.rows(mb.plan)
    SLOTS: gl.constexpr = tiles[0][0].shape[0]
    NW: gl.constexpr = tiles[0][0].shape[2] // 16
    KS: gl.constexpr = len(tiles[0])
    RP: gl.constexpr = triton.next_power_of_2(R)
    C: gl.constexpr = chunk_steps(KS, SLOTS, RP)
    WORDS: gl.constexpr = x_flag is not None
    if WORDS:
        # Every row's flags, then a barrier: no word loads before them.
        lanes: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
        f = gl.arange(0, triton.next_power_of_2(R * X_FLAGS), lanes)
        mb.poll_flags(
            x_flag, mb.row_offset(gl.minimum(f // X_FLAGS, R - 1)) + f % X_FLAGS
        )
        gl.barrier()
    accs = ()
    for p in gl.static_range(len(tiles)):
        acc = bf16.split_zeros(SLOTS, NW, RP)
        for i0 in gl.static_range(0, KS, C):
            if p + i0 > 0:
                # Every wave's reads of the last chunk end before the next
                # store.
                gl.barrier()
            x_smem = bf16.load_split_rows(
                mb, x, p, i0, C, SLOTS, SPLITS, FIRST, COUNT, K, R, WORDS
            )
            acc = bf16.split_chain(x_smem, tiles[p], i0, acc)
        accs += (acc,)
    gl.static_assert((run is None) == (FIRST == 0))
    prev = None
    if run is not None:
        # The earlier splits' sum, polled after this task's products.
        layout: gl.constexpr = bf16.sum_layout(gl.num_warps())
        r = gl.arange(0, RP, gl.SliceLayout(1, layout))
        n = task * (16 * NW) + gl.arange(0, 16 * NW, gl.SliceLayout(0, layout))
        got = mb.poll(run, mb.row_offset(r)[:, None] + n[None, :])
        prev = got.to(gl.float32, bitcast=True)
    # The partials' LDS may reuse the last chunk's.
    gl.barrier()
    return bf16.split_sum(accs, COUNT, RP, prev)


@gluon.jit
def _total(
    mb,
    task,
    w,
    x: gl.constexpr,
    run: gl.constexpr,
    K: gl.constexpr,
    BLOCKS: gl.constexpr,
    SPLITS: gl.constexpr,
    FIRST: gl.constexpr,
    COUNT: gl.constexpr,
    SLOTS: gl.constexpr,
    STAGE: gl.constexpr,
    x_flag: gl.constexpr,
    X_FLAGS: gl.constexpr,
):
    """:func:`gemv` of task ``task`` after :func:`issue`, with its trace
    points."""
    mb.stamp(STAGE, trace.START)
    tiles = issue(w, task, K, BLOCKS, SPLITS, FIRST, COUNT, SLOTS)
    gl.amd.hint.sched_barrier()
    acc = gemv(mb, task, x, run, tiles, K, SPLITS, FIRST, COUNT, x_flag, X_FLAGS)
    mb.stamp(STAGE, trace.COMPUTED)
    return acc


@gluon.jit
def _put_fp32_rows(mb, out: gl.constexpr, task, acc):
    """Publish FP32 ``acc [RP, N]`` as one pair a value: columns ``N task
    ..`` of every row of region ``out``."""
    RP: gl.constexpr = acc.shape[0]
    N: gl.constexpr = acc.shape[1]
    R: gl.constexpr = schedule.rows(mb.plan)
    layout: gl.constexpr = gl.BlockedLayout(
        [1, 1], [1, 64], [gl.num_warps(), 1], [1, 0]
    )
    acc = gl.convert_layout(acc, layout)
    r = gl.arange(0, RP, gl.SliceLayout(1, layout))
    n = task * N + gl.arange(0, N, gl.SliceLayout(0, layout))
    stride: gl.constexpr = schedule.row_pairs(mb.plan)
    mb.put(
        out,
        (r * stride)[:, None] + n[None, :],
        acc.to(gl.int32, bitcast=True),
        mask=(r < R)[:, None],
    )


@gluon.jit
def chain_stage(
    mb,
    task,
    w,
    x: gl.constexpr,
    run: gl.constexpr,
    run_out: gl.constexpr,
    K: gl.constexpr,
    BLOCKS: gl.constexpr,
    SPLITS: gl.constexpr,
    FIRST: gl.constexpr,
    COUNT: gl.constexpr,
    SLOTS: gl.constexpr,
    STAGE: gl.constexpr,
):
    """A K-split group of a projection: the FP32 sum of splits ``.. FIRST +
    COUNT - 1`` of task ``task``'s columns (:func:`gemv`), one pair a value
    in region ``run_out``, for the next group's task. A projection of many
    rows splits K over groups, so that each task loads only its splits'
    activations; the groups add in split order, so the sums do not change."""
    acc = _total(
        mb, task, w, x, run, K, BLOCKS, SPLITS, FIRST, COUNT, SLOTS, STAGE, None, 0
    )
    _put_fp32_rows(mb, run_out, task, acc)
    mb.stamp(STAGE, trace.END)


@gluon.jit
def projection_stage(
    mb,
    task,
    w,
    x: gl.constexpr,
    run: gl.constexpr,
    out: gl.constexpr,
    K: gl.constexpr,
    BLOCKS: gl.constexpr,
    SPLITS: gl.constexpr,
    FIRST: gl.constexpr,
    COUNT: gl.constexpr,
    SLOTS: gl.constexpr,
    STAGE: gl.constexpr,
):
    """Projection task ``task``: output columns ``16 * BLOCKS * task ..`` of
    every row, published as BF16 pairs in region ``out``. ``SPLITS`` is the
    regular projection's K split; the task adds splits ``FIRST .. SPLITS -
    1`` (``FIRST + COUNT == SPLITS``) onto the earlier groups' sum ``run``
    (:func:`gemv`); ``SLOTS`` is the chains a pass
    (``bf16.load_split_tiles``)."""
    gl.static_assert(FIRST + COUNT == SPLITS)
    acc = _total(
        mb, task, w, x, run, K, BLOCKS, SPLITS, FIRST, COUNT, SLOTS, STAGE, None, 0
    )
    mx.put_bf16_rows(mb, out, task * (8 * BLOCKS), acc)
    mb.stamp(STAGE, trace.END)


@gluon.jit
def logits_stage(
    mb,
    task,
    w,
    x: gl.constexpr,
    run: gl.constexpr,
    out: gl.constexpr,
    K: gl.constexpr,
    BLOCKS: gl.constexpr,
    SPLITS: gl.constexpr,
    FIRST: gl.constexpr,
    COUNT: gl.constexpr,
    SLOTS: gl.constexpr,
    STAGE: gl.constexpr,
):
    """:func:`projection_stage` that publishes FP32 values, one pair each
    (the router's logits stay FP32 on the regular path)."""
    gl.static_assert(FIRST + COUNT == SPLITS)
    acc = _total(
        mb, task, w, x, run, K, BLOCKS, SPLITS, FIRST, COUNT, SLOTS, STAGE, None, 0
    )
    _put_fp32_rows(mb, out, task, acc)
    mb.stamp(STAGE, trace.END)


# A projection's chained K split groups (``plan.projection_shape``): the
# stages ``name_k0 ..`` and ``name`` last.
@gluon.constexpr_function
def _groups(plan, name):
    """K split groups of projection ``name``."""
    return ip.PROJECTION_SHAPES[plan.plan.dims.hidden][name][plan.plan.rows][1]


@gluon.constexpr_function
def _group_stage(plan, name, g):
    """The stage of K split group ``g`` (the last group: ``name``)."""
    return name if g == _groups(plan, name) - 1 else f"{name}_k{g}"


@gluon.constexpr_function
def _group_run(name, g):
    """The running sum before group ``g``'s splits (None for the first)."""
    return None if g == 0 else f"{name}_run{g - 1}"


@gluon.constexpr_function
def _blocks(plan, name):
    """16-row weight blocks a task of projection ``name``."""
    return ip.PROJECTION_SHAPES[plan.plan.dims.hidden][name][plan.plan.rows][0]


@gluon.constexpr_function
def _slots(blocks, count, num_warps):
    """Chains a pass (``bf16.load_split_tiles``) for a group's ``count``
    splits: a power of two from the slot waves to 16 that leaves the fewest
    empty slots over the passes, the fewest passes among those."""
    best = None
    slots = num_warps // blocks
    while slots <= 16:
        total = (count + slots - 1) // slots * slots
        if best is None or total <= best[0]:
            best = (total, slots)
        slots *= 2
    return best[1]


@gluon.jit
def chained_stages(
    PLAN: gl.constexpr,
    mb,
    w,
    x: gl.constexpr,
    NAME: gl.constexpr,
    out: gl.constexpr,
    K: gl.constexpr,
    SPLITS: gl.constexpr,
    FP32: gl.constexpr,
):
    """This CTA's tasks of every K split group of projection ``NAME`` (a
    weight of reduction width ``K`` and ``SPLITS`` regular splits; the plan
    gives its blocks a task and groups): :func:`chain_stage` for each group but the last,
    whose tasks publish the result in ``out`` (:func:`projection_stage`, or
    :func:`logits_stage` with ``FP32``)."""
    G: gl.constexpr = _groups(PLAN, NAME)
    COUNT: gl.constexpr = SPLITS // G
    gl.static_assert(COUNT * G == SPLITS)
    BLOCKS: gl.constexpr = _blocks(PLAN, NAME)
    SLOTS: gl.constexpr = _slots(BLOCKS, COUNT, gl.num_warps())
    for g in gl.static_range(G):
        if schedule.has_stage(PLAN, _group_stage(PLAN, NAME, g)):
            task, active = schedule.stage_task(PLAN, _group_stage(PLAN, NAME, g))
            if active:
                if g < G - 1:
                    chain_stage(
                        mb,
                        task,
                        w,
                        x,
                        _group_run(NAME, g),
                        _group_run(NAME, g + 1),
                        K,
                        BLOCKS,
                        SPLITS,
                        g * COUNT,
                        COUNT,
                        SLOTS,
                        _group_stage(PLAN, NAME, g),
                    )
                elif FP32:
                    logits_stage(
                        mb,
                        task,
                        w,
                        x,
                        _group_run(NAME, g),
                        out,
                        K,
                        BLOCKS,
                        SPLITS,
                        g * COUNT,
                        COUNT,
                        SLOTS,
                        NAME,
                    )
                else:
                    projection_stage(
                        mb,
                        task,
                        w,
                        x,
                        _group_run(NAME, g),
                        out,
                        K,
                        BLOCKS,
                        SPLITS,
                        g * COUNT,
                        COUNT,
                        SLOTS,
                        NAME,
                    )


@gluon.jit
def _pack_rows(values):
    """BF16 pair words ``[RP, N / 2]`` of FP32 ``values [RP, N]``."""
    RP: gl.constexpr = values.shape[0]
    N: gl.constexpr = values.shape[1]
    pairs: gl.constexpr = gl.BlockedLayout([1, 2], [1, 64], [gl.num_warps(), 1], [1, 0])
    values = gl.convert_layout(values.to(gl.bfloat16), pairs)
    return mx.pack_bf16_pairs(gl.reshape(values, [RP, N // 2, 2]))


@gluon.jit
def partial_stage(
    mb,
    task,
    w,
    x: gl.constexpr,
    x_flag: gl.constexpr,
    X_FLAGS: gl.constexpr,
    run: gl.constexpr,
    part: gl.constexpr,
    flag: gl.constexpr,
    K: gl.constexpr,
    N: gl.constexpr,
    BLOCKS: gl.constexpr,
    SPLITS: gl.constexpr,
    FIRST: gl.constexpr,
    COUNT: gl.constexpr,
    SLOTS: gl.constexpr,
    RANK: gl.constexpr,
    SCATTER: gl.constexpr,
    STAGE: gl.constexpr,
):
    """Row-parallel projection task ``task`` of this rank (K splits as
    :func:`projection_stage`; ``x`` plain words with ``X_FLAGS`` flags a row
    in ``x_flag``, :func:`gemv`): its BF16 partial columns of every row go to
    every rank's copy of peer region ``part`` (``[source rank][N / 2
    words]`` a row), then flag ``RANK * tasks + task`` of ``flag`` rises on
    every rank. With ``SCATTER`` they go to rank ``task % world`` only, the
    owner of the columns, which folds them (:func:`scatter_fold_stage`)."""
    gl.static_assert(FIRST + COUNT == SPLITS)
    acc = _total(
        mb,
        task,
        w,
        x,
        run,
        K,
        BLOCKS,
        SPLITS,
        FIRST,
        COUNT,
        SLOTS,
        STAGE,
        x_flag,
        X_FLAGS,
    )
    word = _pack_rows(acc)
    RP: gl.constexpr = word.shape[0]
    NWORDS: gl.constexpr = word.shape[1]
    R: gl.constexpr = schedule.rows(mb.plan)
    r = gl.arange(0, RP, gl.SliceLayout(1, word.type.layout))
    c = gl.arange(0, NWORDS, gl.SliceLayout(0, word.type.layout))
    index = (2 * mb.row_offset(r) + RANK * (N // 2) + task * NWORDS)[:, None] + c[
        None, :
    ]
    W: gl.constexpr = schedule.world(mb.plan)
    TASKS: gl.constexpr = N // (16 * BLOCKS)
    if SCATTER:
        # Every rank owns as many column blocks (scatter_fold_stage).
        gl.static_assert(TASKS % W == 0)
        mb.store_words_peer(part, task % W, index, word, mask=(r < R)[:, None])
        mb.raise_flag_peer(flag, task % W, RANK * TASKS + task)
    else:
        for p in gl.static_range(W):
            mb.store_words_peer(part, p, index, word, mask=(r < R)[:, None])
        mb.raise_flag_peers(flag, RANK * TASKS + task)
    mb.stamp(STAGE, trace.END)


@gluon.jit
def poll_partials(mb, task, flag: gl.constexpr, TASKS: gl.constexpr):
    """Wait for every rank's flag of partial task ``task``
    (:func:`partial_stage`), then meet at a barrier so that no load of the
    partials issues before the flags are seen."""
    W: gl.constexpr = schedule.world(mb.plan)
    lanes: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
    s = gl.arange(0, W, lanes)
    mb.poll_flags(flag, s * TASKS + task)
    gl.barrier()


@gluon.constexpr_function
def word_layout(num_warps):
    """``[rows, words]`` layout of the reduce tasks' column words."""
    return gl.BlockedLayout([1, 1], [1, 64], [num_warps, 1], [1, 0])


@gluon.jit
def fold_rows(mb, rows, task, part: gl.constexpr, N: gl.constexpr, HALF: gl.constexpr):
    """The rank-ordered FP32 sums of words ``HALF * task ..`` (BF16 pairs) of
    the partials of rows ``rows`` (a ``[RP]`` tensor in the slice layout of
    :func:`word_layout`), each rounded once to BF16. Returns FP32
    ``(lo, hi) [RP, HALF]``: the even and odd columns. Call it after
    :func:`poll_partials`."""
    W: gl.constexpr = schedule.world(mb.plan)
    layout: gl.constexpr = word_layout(gl.num_warps())
    c = gl.arange(0, HALF, gl.SliceLayout(0, layout))
    base = (2 * mb.row_offset(rows))[:, None] + (task * HALF + c)[None, :]
    lo = gl.zeros([rows.shape[0], HALF], gl.float32, layout)
    hi = gl.zeros([rows.shape[0], HALF], gl.float32, layout)
    for p in gl.static_range(W):
        got = mb.load_words(part, base + p * (N // 2))
        # Rank order 0, 1, 2, 3 from +0.0, as the regular path's fold.
        lo += (got << 16).to(gl.float32, bitcast=True)
        hi += (got & -65536).to(gl.float32, bitcast=True)
    return lo.to(gl.bfloat16).to(gl.float32), hi.to(gl.bfloat16).to(gl.float32)


@gluon.constexpr_function
def wo_scatter(plan):
    """Whether ``plan``'s attention output exchange scatters
    (``plan.WO_SCATTER``)."""
    return ip.WO_SCATTER[plan.plan.rows]


@gluon.jit
def scatter_fold_stage(
    mb,
    task,
    part: gl.constexpr,
    flag: gl.constexpr,
    gather: gl.constexpr,
    gather_flag: gl.constexpr,
    N: gl.constexpr,
    COLUMNS: gl.constexpr,
    RANK: gl.constexpr,
    STAGE: gl.constexpr,
):
    """Fold task ``task`` of a scattered exchange (:func:`partial_stage`
    with ``SCATTER``): this rank owns columns ``COLUMNS * c ..`` of column
    block ``c = task * world + RANK``. It folds the ranks' partials of them
    in rank order (:func:`fold_rows`, the regular path's fold, rounded once
    to BF16), stores the folded words to every rank's copy of peer region
    ``gather`` (``[N / 2 words]`` a row) and raises flag ``c`` of
    ``gather_flag`` on every rank: a reduce-scatter and an all-gather, half
    the bytes of every rank sending its whole partial to every rank."""
    mb.stamp(STAGE, trace.START)
    W: gl.constexpr = schedule.world(mb.plan)
    R: gl.constexpr = schedule.rows(mb.plan)
    RP: gl.constexpr = triton.next_power_of_2(R)
    HALF: gl.constexpr = COLUMNS // 2
    block = task * W + RANK
    poll_partials(mb, block, flag, N // COLUMNS)
    mb.stamp(STAGE, trace.READY)
    layout: gl.constexpr = word_layout(gl.num_warps())
    r = gl.arange(0, RP, gl.SliceLayout(1, layout))
    lo, hi = fold_rows(mb, r, block, part, N, HALF)
    word = mx.pack_bf16_pairs(gl.join(lo.to(gl.bfloat16), hi.to(gl.bfloat16)))
    word = gl.convert_layout(word, layout)
    c = gl.arange(0, HALF, gl.SliceLayout(0, layout))
    index = (2 * mb.row_offset(r) + block * HALF)[:, None] + c[None, :]
    for p in gl.static_range(W):
        mb.store_words_peer(gather, p, index, word, mask=(r < R)[:, None])
    mb.raise_flag_peers(gather_flag, block)
    mb.stamp(STAGE, trace.END)


@gluon.jit
def reduce_columns(
    mb,
    task,
    part: gl.constexpr,
    flag: gl.constexpr,
    N: gl.constexpr,
    COLUMNS: gl.constexpr,
):
    """The rank-ordered FP32 sum of the partials of columns ``COLUMNS * task
    ..`` of every row (:func:`partial_stage` with ``16 * BLOCKS ==
    COLUMNS``), rounded once to BF16. Returns FP32 ``[RP, COLUMNS]``."""
    R: gl.constexpr = schedule.rows(mb.plan)
    RP: gl.constexpr = triton.next_power_of_2(R)
    poll_partials(mb, task, flag, N // COLUMNS)
    r = gl.arange(0, RP, gl.SliceLayout(1, word_layout(gl.num_warps())))
    lo, hi = fold_rows(mb, r, task, part, N, COLUMNS // 2)
    return gl.reshape(gl.join(lo, hi), [RP, COLUMNS])


@gluon.jit
def reduce_stage(
    mb,
    task,
    part: gl.constexpr,
    flag: gl.constexpr,
    out: gl.constexpr,
    N: gl.constexpr,
    COLUMNS: gl.constexpr,
    STAGE: gl.constexpr,
):
    """Reduce task ``task``: :func:`reduce_columns`, published as BF16
    pairs ``COLUMNS / 2 * task ..`` of region ``out``."""
    mb.stamp(STAGE, trace.START)
    total = reduce_columns(mb, task, part, flag, N, COLUMNS)
    mb.stamp(STAGE, trace.READY)
    mx.put_bf16_rows(mb, out, task * (COLUMNS // 2), total)
    mb.stamp(STAGE, trace.END)
