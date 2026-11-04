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

"""Cross-GPU state and exchange ends of the Muse Spark monokernel.

The roles named here (SENDER, AR_OWNER, ...) are defined in ``layer.py``.
This module owns the receive areas of the layer's cross-GPU exchanges:

- AR: the o_proj all-reduce, a reduce-scatter (RS) from the senders to the
  column owners, then an all-gather (AG) of the owned columns and their norm
  sums;
- POOLED_AG: the all-gather of the pooled (pre-expert) rows;
- ROUTE_PUSH: an expert down unit's output, pushed to its route's column
  owner;
- ROUTE_AG: a route owner's completed columns and sums of squares, sent to
  every rank;
- POST_AG: the all-gather of the post-expert columns and their norm sums.

Each area, like the local attention output area (``box``), has two copies
chosen by ``launch.copy = epoch & 1`` and is sized for ``MAX_ROWS`` rows.
Consumers never clear: at the entry of launch L the 128 senders re-arm copy
``(L + 1) & 1`` of every area, 1 / 128 each, and drain those stores before
their first push of L. A peer writes that copy again only in launch L + 1,
which it reaches only after all of this rank's pushes of L.

Device side: :class:`MuseSparkIrisView` (local and peer heap bases),
:func:`clear_next_copies`, and one function per exchange end, which calls the
standalone Iris bodies with ``copy`` as the mailbox stage. Host side:
:class:`MuseSparkIrisState` allocates the areas.
"""

from dataclasses import dataclass

import torch
from tokenspeed_kernel._triton import aggregate, gl, gluon
from tokenspeed_kernel.ops.communication._iris import allocation_bytes
from tokenspeed_kernel.ops.communication._iris.all_gather import (
    iris_lamport_allgather_wait,
)
from tokenspeed_kernel.ops.communication._iris.all_reduce import (
    _iris_group_heap_base,
    _iris_lamport_push,
    _iris_sanitize_lamport_bf16,
    _iris_sanitize_lamport_bf16x2,
    _iris_subgroup_any,
)
from tokenspeed_kernel.ops.communication._iris.muse_spark.attntail import (
    muse_attntail_own,
    muse_attntail_rs_push,
)
from tokenspeed_kernel.ops.communication._iris.muse_spark.moetail import (
    iris_muse_moetail_send,
)
from tokenspeed_kernel.ops.communication._iris.muse_spark.routed_moe import (
    iris_column_routes_combine,
    iris_column_routes_complete,
    iris_column_routes_push_words,
    iris_column_routes_wait,
)
from tokenspeed_kernel.platform import current_platform

if current_platform().is_cdna4:
    from tokenspeed_kernel_amd.ops.gfx950.monokernel.persistent import (
        ERROR_ABORT,
        ERROR_CALLER,
        ERROR_WORDS,
        FIRST_KERNEL_SITE,
        SITE_ABORTED,
        Launch,
        fail_stop,
        poll_check,
        poll_pairs,
        poll_stop_site,
        record_failure,
        store_pairs,
        trap,
        vote,
    )
else:  # host-side sizes only
    ERROR_ABORT, ERROR_CALLER, ERROR_WORDS, FIRST_KERNEL_SITE = 32, 64, 128, 16
    SITE_ABORTED = 3
    Launch = None  # a device type; RoutePushArgs only names it on the host

WORLD_SIZE = 8
# Failure sites of the hop waits (fail_stop records them).
SITE_AR_OWNER = FIRST_KERNEL_SITE + 16
SITE_POOLED_NORM = SITE_AR_OWNER + 1
SITE_ROUTE_OWNER_OWN = SITE_AR_OWNER + 2
SITE_ROUTE_OWNER = SITE_AR_OWNER + 3
SITE_FINISHER = SITE_AR_OWNER + 4
SITE_REDUCER = SITE_AR_OWNER + 5
SITE_POST_AG_SUMS = SITE_AR_OWNER + 6
_SITE_AR_OWNER = gl.constexpr(SITE_AR_OWNER)
_SITE_POOLED_NORM = gl.constexpr(SITE_POOLED_NORM)
_SITE_ROUTE_OWNER_OWN = gl.constexpr(SITE_ROUTE_OWNER_OWN)
_SITE_ROUTE_OWNER = gl.constexpr(SITE_ROUTE_OWNER)
_SITE_FINISHER = gl.constexpr(SITE_FINISHER)
_SITE_REDUCER = gl.constexpr(SITE_REDUCER)
_SITE_POST_AG_SUMS = gl.constexpr(SITE_POST_AG_SUMS)
_ERROR_ABORT = gl.constexpr(ERROR_ABORT)
_SITE_ABORTED = gl.constexpr(SITE_ABORTED)
_ERROR_CALLER = gl.constexpr(ERROR_CALLER)
# Rows every area is sized for, whatever a launch's rows, so an area's rows
# never shrink between launches (stale rows of a wider launch would read as
# data).
MAX_ROWS = 32
HIDDEN = 8192
ROUTE_WIDTH = 4096
TOPK = 8
SENDERS = 128
OWNER_COLUMNS = 128

# Empty markers, as 32-bit words: a BF16 pair whose high half is negative
# zero (receivers test the high half; senders never send negative zero).
LAMPORT_EMPTY_WORD = 0x80000000
# The attention output area tests both halves.
BOX_EMPTY_WORD = 0x80008000

# Elements and 32-bit words of one copy of each area (the copy stride).
_OWNED = HIDDEN // WORLD_SIZE
AR_SHARD = MAX_ROWS * _OWNED  # BF16 elements of one AR source slot
AR_WORDS = 2 * WORLD_SIZE * AR_SHARD // 2  # [RS, AG][source][shard]
AR_SUM_WORDS = WORLD_SIZE * (_OWNED // OWNER_COLUMNS) * MAX_ROWS * 4  # int64, tagged
BOX_WORDS = MAX_ROWS * 1024 // 2
POOLED_AG_LOCAL = 512
POOLED_AG_SLOT = MAX_ROWS * POOLED_AG_LOCAL
POOLED_AG_WORDS = WORLD_SIZE * POOLED_AG_SLOT // 2
_ROUTE_WORDS = ROUTE_WIDTH // 2
_SLICE_WORDS = _ROUTE_WORDS // WORLD_SIZE
ROUTES_WORDS = MAX_ROWS * (
    (2 * WORLD_SIZE - 1) * _ROUTE_WORDS + WORLD_SIZE * WORLD_SIZE * TOPK
)
ROUTE_PUSH_OWN_WORDS = MAX_ROWS * TOPK * _SLICE_WORDS
POST_AG_LOCAL = HIDDEN // WORLD_SIZE
POST_AG_SLOT = MAX_ROWS * POST_AG_LOCAL
POST_AG_WORDS = WORLD_SIZE * POST_AG_SLOT // 2
# The FFN boundary's per-row norm sums: post-expert senders -> their XCD's
# reducer (local pairs), reducer -> every rank (tagged int64 words).
SUMS = 4
REDUCER_SENDERS = 8
REDUCER_IN_WORDS = WORLD_SIZE * REDUCER_SENDERS * MAX_ROWS * SUMS  # int64, local
POST_AG_SUM_WORDS = WORLD_SIZE * WORLD_SIZE * MAX_ROWS * SUMS  # int64, symmetric

_WORLD = gl.constexpr(WORLD_SIZE)
_MAX_ROWS = gl.constexpr(MAX_ROWS)
_HIDDEN = gl.constexpr(HIDDEN)
_OWNER_COLUMNS = gl.constexpr(OWNER_COLUMNS)
_SHARES = gl.constexpr(SENDERS)
_BOX_WORDS = gl.constexpr(BOX_WORDS)
_AR_WORDS = gl.constexpr(AR_WORDS)
_AR_SHARD = gl.constexpr(AR_SHARD)
_POOLED_AG_WORDS = gl.constexpr(POOLED_AG_WORDS)
_POOLED_AG_SLOT = gl.constexpr(POOLED_AG_SLOT)
_ROUTES_WORDS = gl.constexpr(ROUTES_WORDS)
_ROUTE_PUSH_OWN_WORDS = gl.constexpr(ROUTE_PUSH_OWN_WORDS)
_POST_AG_WORDS = gl.constexpr(POST_AG_WORDS)
_POST_AG_SLOT = gl.constexpr(POST_AG_SLOT)
_SUMS = gl.constexpr(SUMS)
HIDDEN_LOCAL = gl.constexpr(POST_AG_LOCAL)
_SLICE = gl.constexpr(_SLICE_WORDS)
_LAMPORT_EMPTY = gl.constexpr(LAMPORT_EMPTY_WORD)
_BOX_EMPTY = gl.constexpr(BOX_EMPTY_WORD)


@dataclass(frozen=True)
class ClearedArea:
    """One two-copy area of the launch state.

    Attributes:
        name: The area (its exchange).
        copy_words: 32-bit words of one copy; the area holds two.
        sentinel: The empty word every copy is re-armed with.
        local: Only this GPU writes it (the attention output area, this
            rank's own route slices); the others are symmetric (Iris heap).
    """

    name: str
    copy_words: int
    sentinel: int
    local: bool


CLEARED_AREAS = (
    ClearedArea("att_box", BOX_WORDS, BOX_EMPTY_WORD, True),
    ClearedArea("ar", AR_WORDS, LAMPORT_EMPTY_WORD, False),
    ClearedArea("pooled_ag", POOLED_AG_WORDS, LAMPORT_EMPTY_WORD, False),
    ClearedArea("route_push_own", ROUTE_PUSH_OWN_WORDS, LAMPORT_EMPTY_WORD, True),
    ClearedArea("routes", ROUTES_WORDS, LAMPORT_EMPTY_WORD, False),
    ClearedArea("post_ag", POST_AG_WORDS, LAMPORT_EMPTY_WORD, False),
)
"""The areas :func:`clear_next_copies` re-arms (AR sums are epoch-tagged and
never cleared)."""


@aggregate
class MuseSparkIrisView:
    """This rank's view of the launch state's areas.

    Attributes:
        heaps: The group's Iris heap bases (int64, group order); a peer's
            address of a symmetric area is ``heaps[peer] + (local -
            heaps[rank])``.
        rank: This rank in the group (int32; the exchange ends that need it
            at compile time take a RANK constexpr too).
        box: The attention output area, uint32 words, two copies (local).
        ar: AR [copy][RS, AG][source][shard], BF16 (symmetric).
        ar_sums: AR norm sums, int64 tagged words (symmetric).
        pooled_ag: POOLED_AG [copy][source][row * 512], BF16 (symmetric).
        route_push_own: This rank's own route slices [copy][route row][256 words]
            (local).
        routes: ROUTE_PUSH / ROUTE_AG [copy][column-route stage words],
            uint32 (symmetric).
        post_ag: POST_AG [copy][source][row * 1024], BF16 (symmetric).
        reducer_in: The post-expert senders' norm-sum partials
            [xcd][sender][row][4], int64 pairs (local, epoch-tagged).
        post_ag_sums: The reducers' norm sums [rank][xcd][row][4], int64
            tagged words (symmetric, epoch-tagged).
    """

    heaps: gl.tuple
    rank: gl.tensor
    box: gl.tensor
    ar: gl.tensor
    ar_sums: gl.tensor
    pooled_ag: gl.tensor
    route_push_own: gl.tensor
    routes: gl.tensor
    post_ag: gl.tensor
    reducer_in: gl.tensor
    post_ag_sums: gl.tensor

    @gluon.constexpr_function
    def __init__(
        self,
        heaps,
        rank,
        box,
        ar,
        ar_sums,
        pooled_ag,
        route_push_own,
        routes,
        post_ag,
        reducer_in,
        post_ag_sums,
    ):
        self.heaps = heaps
        self.rank = rank
        self.box = box
        self.ar = ar
        self.ar_sums = ar_sums
        self.pooled_ag = pooled_ag
        self.route_push_own = route_push_own
        self.routes = routes
        self.post_ag = post_ag
        self.reducer_in = reducer_in
        self.post_ag_sums = post_ag_sums


@aggregate
class RoutePushArgs:
    """:func:`route_push_sink`'s arguments, as the MoE stream hands them over:
    the view, the launch and this rank at compile time (a tuple would turn it
    into a runtime value)."""

    view: MuseSparkIrisView
    launch: Launch
    RANK: gl.constexpr

    @gluon.constexpr_function
    def __init__(self, view, launch, RANK):
        self.view = view
        self.launch = launch
        self.RANK = gl.constexpr(RANK)


@gluon.jit
def _peer_addresses(local, heaps, RANK: gl.constexpr):
    """Every rank's address of a symmetric area, as int64 (group order)."""
    offset = local.to(gl.int64, bitcast=True) - heaps[RANK]
    addresses = ()
    for peer in gl.static_range(len(heaps)):
        addresses += (heaps[peer] + offset,)
    return addresses


@gluon.jit
def iris_clear_share(copy_base, share, WORDS: gl.constexpr, SENTINEL: gl.constexpr):
    """Re-arm share ``share`` of ``_SHARES`` of one area copy (written through).

    ``copy_base`` is the copy's first 32-bit word (16-byte aligned); the
    shares split its ``WORDS`` words evenly, 16 bytes per lane per store. No
    wait: the caller drains before the push its peers wait on.
    """
    PER_SHARE: gl.constexpr = WORDS // _SHARES
    LANES: gl.constexpr = 64 * gl.num_warps()
    gl.static_assert(WORDS % _SHARES == 0 and PER_SHARE % 4 == 0)
    layout: gl.constexpr = gl.BlockedLayout([4], [64], [gl.num_warps()], [0])
    lane = gl.arange(0, 4 * LANES, layout)
    empty = gl.full([4 * LANES], SENTINEL, gl.uint32, layout)
    base = gl.multiple_of(copy_base.to(gl.pointer_type(gl.uint32), bitcast=True), 16)
    for first in gl.static_range(0, PER_SHARE, 4 * LANES):
        if first + 4 * LANES <= PER_SHARE:
            gl.amd.cdna4.buffer_store(
                empty, base, share * PER_SHARE + first + lane, cache=".wt"
            )
        else:
            gl.amd.cdna4.buffer_store(
                empty,
                base,
                share * PER_SHARE + first + lane,
                mask=first + lane < PER_SHARE,
                cache=".wt",
            )


@gluon.jit
def clear_next_copies(view, launch, sender):
    """Sender ``sender`` (0-127) re-arms its share of copy ``(L + 1) & 1`` of
    every area in :data:`CLEARED_AREAS`, this rank's slots only. The caller
    drains (``release_drain``) before its reduce-scatter push."""
    nxt = 1 - launch.copy
    box = view.box.to(gl.pointer_type(gl.uint32), bitcast=True)
    ar = view.ar.to(gl.pointer_type(gl.uint32), bitcast=True)
    pooled_ag = view.pooled_ag.to(gl.pointer_type(gl.uint32), bitcast=True)
    route_push_own = view.route_push_own.to(gl.pointer_type(gl.uint32), bitcast=True)
    routes = view.routes.to(gl.pointer_type(gl.uint32), bitcast=True)
    post_ag = view.post_ag.to(gl.pointer_type(gl.uint32), bitcast=True)
    iris_clear_share(box + nxt * _BOX_WORDS, sender, _BOX_WORDS, _BOX_EMPTY)
    iris_clear_share(ar + nxt * _AR_WORDS, sender, _AR_WORDS, _LAMPORT_EMPTY)
    iris_clear_share(
        pooled_ag + nxt * _POOLED_AG_WORDS, sender, _POOLED_AG_WORDS, _LAMPORT_EMPTY
    )
    iris_clear_share(
        route_push_own + nxt * _ROUTE_PUSH_OWN_WORDS,
        sender,
        _ROUTE_PUSH_OWN_WORDS,
        _LAMPORT_EMPTY,
    )
    iris_clear_share(
        routes + nxt * _ROUTES_WORDS, sender, _ROUTES_WORDS, _LAMPORT_EMPTY
    )
    iris_clear_share(
        post_ag + nxt * _POST_AG_WORDS, sender, _POST_AG_WORDS, _LAMPORT_EMPTY
    )


@gluon.jit
def ar_rs_sink(
    view, launch, out, block, rows, RANK: gl.constexpr, PEER_CACHE: gl.constexpr
):
    """Sender ``block``'s reduce-scatter push of the o_proj all-reduce:
    ``out`` is its FP32 [16 tokens, 64] o_proj partial for columns
    [64 block, 64 block + 64)."""
    muse_attntail_rs_push(
        out, block, _peer_addresses(view.ar, view.heaps, RANK), launch.copy, rows,
        64, _HIDDEN, RANK, _WORLD, _AR_SHARD, PEER_CACHE,
    )  # fmt: skip


@gluon.jit
def ar_owner(
    view,
    launch,
    owner,
    rows,
    stream_ptr,
    post_weight_ptr,
    alpha_ptr,
    beta_ptr,
    on_expiry: gl.constexpr,
    RANK: gl.constexpr,
    POST_WEIGHT_OFFSET: gl.constexpr,
    POST_HAS_WEIGHT: gl.constexpr,
    PEER_CACHE: gl.constexpr,
):
    """All-reduce owner ``owner`` (0-7): wait for the 16 sources of its 128
    columns, sum them in rank order, and push the all-gather words and the
    four norm sums (tagged with ``launch.tag``) to every rank. Then zero its
    sums of rows [rows, MAX_ROWS) on every rank: 32-bit tags repeat after
    2**32 - 1 launches, so a row that a later, wider launch reads must not
    keep an old tag."""
    sums_peers = _peer_addresses(view.ar_sums, view.heaps, RANK)
    muse_attntail_own(
        owner, launch.cta, view.ar, _peer_addresses(view.ar, view.heaps, RANK),
        sums_peers, launch.tag, launch.copy, rows, stream_ptr, post_weight_ptr,
        alpha_ptr, beta_ptr, _HIDDEN, RANK, _WORLD, _AR_SHARD, _MAX_ROWS,
        _OWNER_COLUMNS, POST_WEIGHT_OFFSET, POST_HAS_WEIGHT, PEER_CACHE, False,
        launch, on_expiry, _SITE_AR_OWNER,
    )  # fmt: skip
    OWNERS: gl.constexpr = _HIDDEN // _WORLD // _OWNER_COLUMNS
    layout: gl.constexpr = gl.BlockedLayout(
        [1, 4], [32, 2], [gl.num_warps(), 1], [1, 0]
    )
    row = gl.arange(0, _MAX_ROWS, layout=gl.SliceLayout(1, layout))
    which = gl.arange(0, 4, layout=gl.SliceLayout(0, layout))
    slot = ((RANK * OWNERS + owner) * _MAX_ROWS + row[:, None]) * 4 + which[None, :]
    stale = (row >= rows)[:, None] & (which >= 0)[None, :]
    zero = gl.zeros([_MAX_ROWS, 4], gl.int64, layout)
    for peer in gl.static_range(_WORLD):
        destination = gl.multiple_of(sums_peers[peer].to(gl.pointer_type(gl.int64)), 16)
        gl.amd.cdna4.buffer_store(zero, destination, slot, mask=stale, cache=".wt")


@gluon.jit
def pooled_ag_sink(
    view,
    launch,
    values,
    tile,
    rows,
    ROWS: gl.constexpr,
    RANK: gl.constexpr,
    PEER_CACHE: gl.constexpr,
):
    """The pooler reducer of ``tile`` pushes its reduced [ROWS, 64] columns
    (any float layout) to slot RANK of every rank's POOLED_AG copy, its own
    included (written through: the pooled-row norm reads it from any XCD)."""
    layout: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [gl.num_warps(), 1], [1, 0])
    v = _iris_sanitize_lamport_bf16(gl.convert_layout(values, layout).to(gl.bfloat16))
    row = gl.arange(0, ROWS, layout=gl.SliceLayout(1, layout))
    column = gl.arange(0, 64, layout=gl.SliceLayout(0, layout))
    offsets = (
        launch.copy * (2 * _POOLED_AG_WORDS)
        + RANK * _POOLED_AG_SLOT
        + row[:, None] * 512
        + tile * 64
        + column[None, :]
    )
    valid = (row < rows)[:, None] & (column >= 0)[None, :]
    peers = _peer_addresses(view.pooled_ag, view.heaps, RANK)
    published = ()
    for _ in gl.static_range(1, _WORLD):
        published += (v,)
    _iris_lamport_push(
        published, peers, 0, offsets, valid, tile, RANK, _WORLD, PEER_CACHE
    )
    own = gl.multiple_of(
        view.pooled_ag.to(gl.pointer_type(gl.bfloat16), bitcast=True), 16
    )
    gl.amd.cdna4.buffer_store(v, own, offsets, mask=valid, cache=".wt")


@gluon.jit
def pooled_norm_wait(view, launch, row, on_expiry: gl.constexpr, RANK: gl.constexpr):
    """Pooled-row norm of ``row``: wait for every rank's 512 pooled columns
    (this rank's own slot is polled too). Returns
    ``iris_lamport_allgather_wait``'s words for ``iris_rmsnorm_mxfp4_row``."""
    words = view.pooled_ag.to(gl.pointer_type(gl.uint32), bitcast=True)
    return iris_lamport_allgather_wait(
        view.pooled_ag, words, row * 256, (launch.copy.to(gl.uint32),), RANK, _WORLD, 512,
        4096, _POOLED_AG_WORDS, _POOLED_AG_SLOT // 2, False, True, launch, on_expiry, _SITE_POOLED_NORM,
    )  # fmt: skip


@gluon.constexpr_function
def _wave_strips(layout, warps):
    """Whether ``layout`` gives wave w columns [16 w, 16 w + 16) of every row,
    4 adjacent columns per lane (the MoE stream's sink layout)."""
    return (
        isinstance(layout, gl.BlockedLayout)
        and layout.size_per_thread[-1] == 4
        and layout.threads_per_warp[-1] == 4
        and list(layout.warps_per_cta) == [1, warps]
    )


@gluon.jit
def _route_push_store(
    view, launch, words, route_row, slice_words, owner, RANK: gl.constexpr
):
    """One route-push store of BF16-pair ``words`` at ``slice_words`` of
    route rows ``route_row`` in ``owner``'s 512-column slice (``owner`` a
    per-wave scalar): to this rank's own slices, or to that owner's mailbox."""
    valid = (route_row >= 0) & (slice_words >= 0)
    if owner == RANK:
        own = gl.multiple_of(
            view.route_push_own.to(gl.pointer_type(gl.uint32), bitcast=True), 16
        )
        gl.amd.cdna4.buffer_store(
            _iris_sanitize_lamport_bf16x2(words),
            own,
            launch.copy * _ROUTE_PUSH_OWN_WORDS + route_row * _SLICE + slice_words,
            mask=valid,
            cache=".wt",
        )
    else:
        iris_column_routes_push_words(
            words, route_row, slice_words, owner, launch.copy, valid,
            _peer_addresses(view.routes, view.heaps, RANK), RANK, _WORLD,
            _MAX_ROWS, _ROUTES_WORDS, ".wt",
        )  # fmt: skip


@gluon.jit
def _bf16_pairs(even, odd):
    """32-bit words of BF16 column pairs: ``even`` in the low half."""
    low = even.to(gl.bfloat16).to(gl.uint16, bitcast=True)
    high = odd.to(gl.bfloat16).to(gl.uint16, bitcast=True)
    return low.to(gl.uint32) | (high.to(gl.uint32) << 16)


@gluon.jit
def route_push_sink(args, ys, route, cols, col0):
    """Push an MoE down unit's output to its route's column owner
    (ROUTE_PUSH): the MoE stream's ``sink(args, ys, route, cols, col0)`` with
    ``args`` a :class:`RoutePushArgs`. Called from per-wave code, so it moves
    no data between lanes (no layout conversion, LDS or barrier).

    ``ys`` holds FP32 partials [R, 16 * num_warps] in a BlockedLayout where
    wave w holds columns [16 w, 16 w + 16) of every row, 4 per lane
    (``BlockedLayout([1, 4], [16, 4], [1, num_warps], [0, 1])``);
    ``route``: [R, 1] route row (token * 8 + pick), -1 to skip, a slice of
    that layout; ``col0``: the wave's first column, a per-wave scalar whose
    512-column slice holds all of the wave's columns (its owner). Each
    column pair goes out as one 32-bit word (receivers test a word's high
    half). Columns this rank owns go to its own slices (read by
    :func:`route_owner_complete`), the rest to the owner's mailbox
    (iris_column_routes_push_words, copy ``launch.copy``), no wait.

    - One strip (``ys = (y,)``, ``col0`` a multiple of 16): a lane's 4
      adjacent columns, two words in one 8-byte store; ``cols`` is not read.
    - Four strips of one down unit (``ys``, ``cols`` 4-tuples, ``col0`` a
      multiple of 64): strips ``s`` and ``s + 2`` hold a lane's even and odd
      columns ``cols[s]`` (s = 0, 1), so each lane writes two 16-byte lines,
      one store each.
    """
    view = args.view
    launch = args.launch
    RANK: gl.constexpr = args.RANK
    owner = col0 // 512
    gl.static_assert(
        _wave_strips(ys[0].type.layout, gl.num_warps()),
        "route_push_sink: wave w must hold columns 16 w .. 16 w + 15, 4 per lane",
    )
    R: gl.constexpr = ys[0].shape[0]
    C: gl.constexpr = ys[0].shape[1]
    gl.static_assert(C == 16 * gl.num_warps(), "route_push_sink: 16 columns a wave")
    if len(ys) == 1:
        halves = gl.reshape(ys[0], [R, C // 2, 2])
        even, odd = gl.split(halves)
        words = _bf16_pairs(even, odd)
        words_l: gl.constexpr = words.type.layout
        route_row = gl.convert_layout(
            gl.reshape(route, [R]), gl.SliceLayout(1, words_l), assert_trivial=True
        )[:, None]
        # This wave's 8 words of the owner's slice, built from an arange so
        # the store sees each lane's two words as adjacent.
        within = gl.arange(0, C // 2, gl.SliceLayout(0, words_l)) % 8
        slice_words = (gl.multiple_of((col0 % 512) // 2, 8) + within)[None, :]
        _route_push_store(view, launch, words, route_row, slice_words, owner, RANK)
    else:
        gl.static_assert(len(ys) == 4, "route_push_sink: one or four strips")
        for line in gl.static_range(2):
            words = _bf16_pairs(ys[line], ys[line + 2])
            # A lane's 4 words are adjacent and 16-byte aligned.
            slice_words = gl.max_contiguous(
                gl.multiple_of((cols[line] % 512) // 2, [1, 4]), [1, 4]
            )
            _route_push_store(view, launch, words, route, slice_words, owner, RANK)


@gluon.jit
def route_owner_complete(
    view,
    launch,
    route_row,
    norm_weight,
    on_expiry: gl.constexpr,
    RANK: gl.constexpr,
    WEIGHT_OFFSET: gl.constexpr,
    PEER_CACHE: gl.constexpr,
):
    """Route owner of ``route_row``: wait for this rank's own 512-column slice
    (written by the local down units), complete it with the peers' partials,
    and send the completed columns and sum of squares to every rank
    (ROUTE_AG)."""
    layout: gl.constexpr = gl.BlockedLayout(
        [1, 4], [64, 1], [gl.num_warps(), 1], [1, 0]
    )
    lane = gl.arange(0, 64, layout=gl.SliceLayout(1, layout))
    word = gl.arange(0, 4, layout=gl.SliceLayout(0, layout))
    offsets = (
        launch.copy * _ROUTE_PUSH_OWN_WORDS
        + route_row * _SLICE
        + lane[:, None] * 4
        + word[None, :]
    )
    own_ptr = gl.multiple_of(
        view.route_push_own.to(gl.pointer_type(gl.uint32), bitcast=True), 16
    )
    own = gl.amd.cdna4.buffer_load(own_ptr, offsets, cache=".cv")
    start = gl.to_tensor(0).to(gl.int64)
    retries = 0
    stop = retries < 0
    while _iris_subgroup_any(gl.max(((own >> 16) == 0x8000).to(gl.int32), 1)) & (
        not stop
    ):
        own = gl.amd.cdna4.buffer_load(own_ptr, offsets, cache=".cv")
        retries += 1
        if retries % 64 == 0:
            stop, start = poll_check(launch, start, retries, False)
    if stop:
        on_expiry(launch, poll_stop_site(launch, _SITE_ROUTE_OWNER_OWN), route_row)
    iris_column_routes_complete(
        own, launch.copy, norm_weight, view.routes,
        _peer_addresses(view.routes, view.heaps, RANK), route_row, RANK, _WORLD,
        _MAX_ROWS, _ROUTES_WORDS, WEIGHT_OFFSET, PEER_CACHE, False, launch,
        on_expiry, _SITE_ROUTE_OWNER,
    )  # fmt: skip


@gluon.jit
def finisher(
    view,
    launch,
    token,
    owner,
    route_weights,
    norm_weight,
    on_expiry: gl.constexpr,
    EPS: gl.constexpr,
    WEIGHT_OFFSET: gl.constexpr,
):
    """Finisher of (token, owner): wait for the owner's completed columns of
    the token's 8 routes and every rank's sums of squares (ROUTE_AG), then
    return the routed RMSNorm and weighted route sum over those 512 columns
    (FP32 [512])."""
    loaded, squares = iris_column_routes_wait(
        view.routes, launch.copy, token, owner, _WORLD, _MAX_ROWS, _ROUTES_WORDS, False,
        launch, on_expiry, _SITE_FINISHER,
    )  # fmt: skip
    return iris_column_routes_combine(
        loaded, squares, route_weights, norm_weight, token, owner, EPS, WEIGHT_OFFSET
    )


@gluon.jit
def post_ag_sink(
    view,
    launch,
    values,
    first_output,
    rows,
    BLOCK_M: gl.constexpr,
    OUTPUTS: gl.constexpr,
    RANK: gl.constexpr,
    PEER_CACHE: gl.constexpr,
):
    """Push a post-expert sender's columns [first_output, first_output +
    OUTPUTS) of every row to slot RANK of every rank's POST_AG copy, its own
    included (the moetail push)."""
    iris_muse_moetail_send(
        values, first_output, launch.cta, launch.copy.to(gl.uint32),
        _peer_addresses(view.post_ag, view.heaps, RANK), rows, _iris_lamport_push, RANK,
        _WORLD, HIDDEN_LOCAL, _POST_AG_SLOT, BLOCK_M, OUTPUTS, gl.num_warps(), PEER_CACHE,
        ".wt",
    )  # fmt: skip


@gluon.jit
def post_sender_sums_publish(view, launch, sums, sender, rows):
    """Post-expert sender ``sender`` (0-7) of this XCD publishes its FP32
    [16, SUMS] norm-sum partials (rows < rows) to its XCD's reducer, one
    (value, tag) pair each; the pairs of rows [rows, 16) are zeroed."""
    layout: gl.constexpr = gl.BlockedLayout(
        [1, 4], [16, 4], [gl.num_warps(), 1], [1, 0]
    )
    values = gl.convert_layout(sums, layout).to(gl.int32, bitcast=True)
    row = gl.arange(0, 16, layout=gl.SliceLayout(1, layout))
    which = gl.arange(0, _SUMS, layout=gl.SliceLayout(0, layout))
    slot = ((launch.xcd * 8 + sender) * _MAX_ROWS + row[:, None]) * _SUMS + which[
        None, :
    ]
    valid = (row < rows)[:, None] & (which >= 0)[None, :]
    store_pairs(view.reducer_in + slot, values, launch, valid, "gpu")
    # Rows past ``rows`` get tag 0, which never matches (tags repeat after
    # 2**32 - 1 launches, so no slot may keep an old one).
    gl.atomic_store(
        view.reducer_in + slot,
        gl.zeros_like(slot).to(gl.int64),
        mask=(row >= rows)[:, None] & (which >= 0)[None, :],
        sem="relaxed",
        scope="gpu",
    )


@gluon.jit
def reducer_sums(view, launch, rows, on_expiry: gl.constexpr, RANK: gl.constexpr, PEER_CACHE: gl.constexpr,):  # fmt: skip
    """The norm-sum reducer of this XCD: wait for its 8 senders' partials of
    rows < rows (at most 16), add them in FP32 (the same order on every
    rank), and push the [row][SUMS] totals as tagged int64 words to slot
    (RANK, xcd) of every rank's ``post_ag_sums``. Rows [rows, MAX_ROWS) of the
    slot are zeroed, as tags repeat after 2**32 - 1 launches. Needs 8 waves
    (one pair per lane)."""
    gl.static_assert(gl.num_warps() == 8)
    poll: gl.constexpr = gl.BlockedLayout([1, 1], [1, 64], [8, 1], [1, 0])
    sender = gl.arange(0, 8, layout=gl.SliceLayout(1, poll))
    entry = gl.arange(0, 16 * _SUMS, layout=gl.SliceLayout(0, poll))
    slot = ((launch.xcd * 8 + sender[:, None]) * _MAX_ROWS) * _SUMS + entry[None, :]
    valid = (sender >= 0)[:, None] & (entry // _SUMS < rows)[None, :]
    values, seen = poll_pairs(view.reducer_in + slot, valid, launch, "gpu")
    if not seen:
        on_expiry(launch, poll_stop_site(launch, _SITE_REDUCER), 0)
    totals = gl.sum(values.to(gl.float32, bitcast=True), 0)  # [16 * SUMS]
    push: gl.constexpr = gl.BlockedLayout([1, 4], [32, 2], [gl.num_warps(), 1], [1, 0])
    row = gl.arange(0, _MAX_ROWS, layout=gl.SliceLayout(1, push))
    which = gl.arange(0, _SUMS, layout=gl.SliceLayout(0, push))
    flat = row[:, None] * _SUMS + which[None, :]
    total = gl.convert_layout(
        gl.reshape(totals, [16, _SUMS]),
        gl.BlockedLayout([1, 4], [16, 4], [gl.num_warps(), 1], [1, 0]),
    )
    # Rows 16-31 of a [32, SUMS] slot hold zeros (and tag 0): never current.
    padded = gl.join(total, gl.zeros_like(total))
    padded = gl.convert_layout(
        gl.reshape(gl.permute(padded, [2, 0, 1]), [_MAX_ROWS, _SUMS]), push
    )
    live = (row < rows)[:, None] & (which >= 0)[None, :]
    words = gl.where(
        live,
        (launch.tag.to(gl.int64) << 32)
        | (padded.to(gl.int32, bitcast=True).to(gl.int64) & 0xFFFFFFFF),
        0,
    )
    destination_slot = (RANK * 8 + launch.xcd) * (_MAX_ROWS * _SUMS) + flat
    peers = _peer_addresses(view.post_ag_sums, view.heaps, RANK)
    for peer in gl.static_range(_WORLD):
        destination = gl.multiple_of(peers[peer].to(gl.pointer_type(gl.int64)), 16)
        if peer == RANK:
            gl.amd.cdna4.buffer_store(words, destination, destination_slot, cache=".wt")
        else:
            gl.amd.cdna4.buffer_store(
                words, destination, destination_slot, cache=PEER_CACHE
            )


@gluon.jit
def post_ag_sums_wait(view, launch, row, on_expiry: gl.constexpr):
    """The FFN boundary's norm sums of ``row`` from every rank's 8 reducers:
    FP32 [64 sources (rank, xcd), SUMS], polled on their tags (one pair per
    lane), returned with the sources over a wave's lanes and the SUMS sums
    of a source in one lane (deferred_norm_scales_gfx950's layout)."""
    layout: gl.constexpr = gl.BlockedLayout(
        [1, 1], [16, 4], [gl.num_warps(), 1], [1, 0]
    )
    source = gl.arange(0, 64, layout=gl.SliceLayout(1, layout))
    which = gl.arange(0, _SUMS, layout=gl.SliceLayout(0, layout))
    slot = (source[:, None] * _MAX_ROWS + row) * _SUMS + which[None, :]
    valid = (source >= 0)[:, None] & (which >= 0)[None, :]
    values, seen = poll_pairs(view.post_ag_sums + slot, valid, launch, "sys")
    if not seen:
        on_expiry(launch, poll_stop_site(launch, _SITE_POST_AG_SUMS), row)
    sums_layout: gl.constexpr = gl.BlockedLayout(
        [1, 4], [64, 1], [gl.num_warps(), 1], [1, 0]
    )
    return gl.convert_layout(values.to(gl.float32, bitcast=True), sums_layout)


@gluon.jit
def iris_fail_stop(launch, site, detail):
    """The monokernel's ``on_expiry``: stop every rank's waits, then record
    the failure and trap; never returns.

    Callable from any subset of waves. A rank that failed on its own stores
    the abort marker ``(rank + 1) << 16 | site`` into every rank's abort word
    that is still zero (system-scope compare-and-swap, so the first origin
    stays), after claiming its own record with its site and detail. A rank
    stopped by a peer's marker records ``SITE_ABORTED`` with the marker as
    the detail and leaves the abort words alone, so every rank's record
    names the origin. The peers' addresses come from the error words' caller
    lines that :class:`MuseSparkIrisState` fills.
    """
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
    lane = gl.arange(0, 64 * gl.num_warps(), layout) % 64
    seen = gl.load(launch.error_ptr + _ERROR_ABORT + lane * 0, cache_modifier=".cv")
    # No cross-lane reduction here: the calling waves may be a subset.
    if vote((seen != 0).to(gl.int32)):
        fail_stop(launch, _SITE_ABORTED, seen)
    else:
        # The origin claims its own record before its marker can stop this
        # rank's other workgroups, whose records would otherwise win the
        # claim with SITE_ABORTED and hide the origin's site.
        record_failure(launch, site, detail)
        table = launch.error_ptr.to(gl.pointer_type(gl.int64), bitcast=True)
        peer = gl.load(table + _ERROR_CALLER // 2 + lane, mask=lane < 8, other=0)
        rank = gl.load(launch.error_ptr + _ERROR_CALLER + 16 + lane * 0)
        marker = ((rank + 1) << 16) | site
        # Lanes with no peer swap on this rank's own abort word instead.
        own = (launch.error_ptr + _ERROR_ABORT).to(gl.int64, bitcast=True)
        target = gl.where(lane < 8, peer + 4 * _ERROR_ABORT, own)
        gl.atomic_cas(
            target.to(gl.pointer_type(gl.int32)),
            gl.zeros_like(marker),
            marker,
            sem="relaxed",
            scope="sys",
        )
        trap()


class MuseSparkIrisState:
    """The launch state's areas for one group (allocate before capture).

    Symmetric areas come from the group's Iris heap at the same offset on
    every rank; local areas are device memory. Lamport areas start empty
    (every copy), the AR sums at zero (tags are never zero).

    Args:
        peers: The group's ``IrisGroup`` (its heap must hold
            :meth:`heap_bytes` more).
    """

    def __init__(self, peers) -> None:
        if peers.world_size != WORLD_SIZE:
            raise ValueError(
                f"MuseSparkIrisState needs {WORLD_SIZE} ranks, got {peers.world_size}"
            )
        empty = LAMPORT_EMPTY_WORD - (1 << 32)
        self.peers = peers
        self.ar = peers.ctx.empty((2 * AR_WORDS,), dtype=torch.int32)
        self.ar_sums = peers.ctx.zeros((AR_SUM_WORDS,), dtype=torch.int64)
        self.pooled_ag = peers.ctx.empty((2 * POOLED_AG_WORDS,), dtype=torch.int32)
        self.routes = peers.ctx.empty((2 * ROUTES_WORDS,), dtype=torch.int32)
        self.post_ag = peers.ctx.empty((2 * POST_AG_WORDS,), dtype=torch.int32)
        for area in (self.ar, self.pooled_ag, self.routes, self.post_ag):
            area.fill_(empty)
        self.post_ag_sums = peers.ctx.zeros((POST_AG_SUM_WORDS,), dtype=torch.int64)
        self.reducer_in = torch.zeros(
            (REDUCER_IN_WORDS,), dtype=torch.int64, device=peers.device
        )
        self.box = torch.full(
            (2 * BOX_WORDS,), BOX_EMPTY_WORD - (1 << 32), dtype=torch.int32,
            device=peers.device,
        )  # fmt: skip
        self.route_push_own = torch.full(
            (2 * ROUTE_PUSH_OWN_WORDS,), empty, dtype=torch.int32, device=peers.device
        )
        self.heaps = tuple(int(address) for address in peers.addresses[:WORLD_SIZE])
        # The launch's error words, on the heap so that iris_fail_stop can
        # store every rank's abort word; their caller lines hold the peers'
        # addresses of them (int64) and this rank.
        self.errors = peers.ctx.zeros((ERROR_WORDS,), dtype=torch.int32)
        offset = self.errors.data_ptr() - self.heaps[peers.rank]
        caller = self.errors[ERROR_CALLER:]
        caller[:16].view(torch.int64).copy_(
            torch.tensor([base + offset for base in self.heaps], dtype=torch.int64)
        )
        caller[16] = peers.rank

    @staticmethod
    def heap_bytes() -> int:
        """Iris heap bytes of the symmetric areas."""
        return (
            allocation_bytes((2 * AR_WORDS,), torch.int32)
            + allocation_bytes((AR_SUM_WORDS,), torch.int64)
            + allocation_bytes((2 * POOLED_AG_WORDS,), torch.int32)
            + allocation_bytes((2 * ROUTES_WORDS,), torch.int32)
            + allocation_bytes((2 * POST_AG_WORDS,), torch.int32)
            + allocation_bytes((ERROR_WORDS,), torch.int32)
            + allocation_bytes((POST_AG_SUM_WORDS,), torch.int64)
        )

    def view_args(self) -> tuple:
        """Kernel arguments that build :class:`MuseSparkIrisView` in order:
        ``heaps`` (a tuple of 8 int64), rank, then box, ar, ar_sums,
        pooled_ag, route_push_own, routes, post_ag, reducer_in, post_ag_sums.
        The Lamport areas are typed as the kernel reads them (BF16 halves, or
        32-bit words)."""
        return (
            self.heaps, self.peers.rank, self.box.view(torch.uint32),
            self.ar.view(torch.bfloat16), self.ar_sums, self.pooled_ag.view(torch.bfloat16),
            self.route_push_own.view(torch.uint32), self.routes.view(torch.uint32),
            self.post_ag.view(torch.bfloat16), self.reducer_in, self.post_ag_sums,
        )  # fmt: skip
