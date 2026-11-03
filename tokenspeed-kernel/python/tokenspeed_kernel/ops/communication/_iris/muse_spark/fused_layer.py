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

"""Iris state and hop bodies of the native fused Muse Spark layer (one launch per layer).

One launch kind owns every cross-GPU receive area and the local attention
box, each in two copies chosen by ``launch.copy = epoch & 1`` and sized for
``MAX_ROWS`` rows. Consumers never clear: at the entry of launch L the 128
senders of the all-to-all H1 reduce-scatter (SND) re-arm copy ``(L + 1) & 1``
of every area, each its 1 / 128 share, and drain the stores before their
first push of L. A peer writes that copy again only in L + 1, which it reaches
only after every one of this rank's L pushes (NATIVE-DESIGN §7.1).

Device side: :class:`MuseNativeView` (local bases and the peers' heap bases),
:func:`clear_next_copies`, and one function per hop end, each calling the
standalone kernels' bodies with ``copy`` as the mailbox stage and no in-body
clears. Host side: :class:`MuseNativeIrisState` allocates the areas.
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
    _iris_subgroup_any,
)
from tokenspeed_kernel.ops.communication._iris.muse_spark.attntail import (
    muse_attntail_own,
    muse_attntail_rs_push,
)
from tokenspeed_kernel.ops.communication._iris.muse_spark.moetail import (
    iris_muse_moetail_push,
)
from tokenspeed_kernel.ops.communication._iris.muse_spark.routed_moe import (
    iris_column_routes_combine,
    iris_column_routes_complete,
    iris_column_routes_wait,
)
from tokenspeed_kernel.platform import current_platform

if current_platform().is_cdna4:
    from tokenspeed_kernel_amd.ops.gfx950.persistent import (
        ERROR_ABORT,
        ERROR_CALLER,
        ERROR_WORDS,
        FIRST_KERNEL_SITE,
        fail_stop,
        poll_check,
        poll_pairs,
        poll_stop_site,
        store_pairs,
        vote,
    )
else:  # host-side sizes only
    ERROR_ABORT, ERROR_CALLER, ERROR_WORDS, FIRST_KERNEL_SITE = 32, 64, 128, 16

WORLD_SIZE = 8
# Failure sites of the hop waits (fail_stop records them).
SITE_OWN1 = FIRST_KERNEL_SITE + 16
SITE_XN = SITE_OWN1 + 1
SITE_OWN4_OWN = SITE_OWN1 + 2
SITE_OWN4 = SITE_OWN1 + 3
SITE_FIN = SITE_OWN1 + 4
SITE_RED = SITE_OWN1 + 5
SITE_H6_SUMS = SITE_OWN1 + 6
_SITE_OWN1 = gl.constexpr(SITE_OWN1)
_SITE_XN = gl.constexpr(SITE_XN)
_SITE_OWN4_OWN = gl.constexpr(SITE_OWN4_OWN)
_SITE_OWN4 = gl.constexpr(SITE_OWN4)
_SITE_FIN = gl.constexpr(SITE_FIN)
_SITE_RED = gl.constexpr(SITE_RED)
_SITE_H6_SUMS = gl.constexpr(SITE_H6_SUMS)
_ERROR_ABORT = gl.constexpr(ERROR_ABORT)
_ERROR_CALLER = gl.constexpr(ERROR_CALLER)
# Rows every area is sized for, whatever a launch's rows (8 or 16 today), so
# an area's rows never shrink between launches (stale rows of a wider launch
# would read as data).
MAX_ROWS = 32
HIDDEN = 8192
ROUTE_WIDTH = 4096
TOPK = 8
SENDERS = 128
OWNER_COLUMNS = 128

# Empty markers, as 32-bit words: a BF16 pair whose high half is negative
# zero (receivers test the high half; senders never send negative zero).
LAMPORT_EMPTY_WORD = 0x80000000
# The attention box tests both halves.
BOX_EMPTY_WORD = 0x80008000

# Elements and 32-bit words of one copy of each area (the copy stride).
_OWNED = HIDDEN // WORLD_SIZE
H1_SHARD = MAX_ROWS * _OWNED  # BF16 elements of one H1 source slot
H1_WORDS = 2 * WORLD_SIZE * H1_SHARD // 2  # [RS, AG][source][shard]
H1_SUM_WORDS = WORLD_SIZE * (_OWNED // OWNER_COLUMNS) * MAX_ROWS * 4  # int64, tagged
BOX_WORDS = MAX_ROWS * 1024 // 2
H3_LOCAL = 512
H3_SLOT = MAX_ROWS * H3_LOCAL
H3_WORDS = WORLD_SIZE * H3_SLOT // 2
_ROUTE_WORDS = ROUTE_WIDTH // 2
_SLICE_WORDS = _ROUTE_WORDS // WORLD_SIZE
ROUTES_WORDS = MAX_ROWS * (
    (2 * WORLD_SIZE - 1) * _ROUTE_WORDS + WORLD_SIZE * WORLD_SIZE * TOPK
)
H4_OWN_WORDS = MAX_ROWS * TOPK * _SLICE_WORDS
H6_LOCAL = HIDDEN // WORLD_SIZE
H6_SLOT = MAX_ROWS * H6_LOCAL
H6_WORDS = WORLD_SIZE * H6_SLOT // 2
# The FFN boundary's per-row norm sums: UNP (b, x) -> RED x (local pairs),
# RED x -> every rank (tagged int64 words), like the H1 sums.
SUMS = 4
RED_SENDERS = 8
RED_IN_WORDS = WORLD_SIZE * RED_SENDERS * MAX_ROWS * SUMS  # int64, local
H6_SUM_WORDS = WORLD_SIZE * WORLD_SIZE * MAX_ROWS * SUMS  # int64, symmetric

_WORLD = gl.constexpr(WORLD_SIZE)
_MAX_ROWS = gl.constexpr(MAX_ROWS)
_HIDDEN = gl.constexpr(HIDDEN)
_OWNER_COLUMNS = gl.constexpr(OWNER_COLUMNS)
_SHARES = gl.constexpr(SENDERS)
_BOX_WORDS = gl.constexpr(BOX_WORDS)
_H1_WORDS = gl.constexpr(H1_WORDS)
_H1_SHARD = gl.constexpr(H1_SHARD)
_H3_WORDS = gl.constexpr(H3_WORDS)
_H3_SLOT = gl.constexpr(H3_SLOT)
_ROUTES_WORDS = gl.constexpr(ROUTES_WORDS)
_H4_OWN_WORDS = gl.constexpr(H4_OWN_WORDS)
_H6_WORDS = gl.constexpr(H6_WORDS)
_H6_SLOT = gl.constexpr(H6_SLOT)
_SUMS = gl.constexpr(SUMS)
HIDDEN_LOCAL = gl.constexpr(H6_LOCAL)
_SLICE = gl.constexpr(_SLICE_WORDS)
_LAMPORT_EMPTY = gl.constexpr(LAMPORT_EMPTY_WORD)
_BOX_EMPTY = gl.constexpr(BOX_EMPTY_WORD)


@dataclass(frozen=True)
class ClearedArea:
    """One two-copy area of the launch state.

    Attributes:
        name: The area (its hop).
        copy_words: 32-bit words of one copy; the area holds two.
        sentinel: The empty word every copy is re-armed with.
        local: Only this GPU writes it (the attention box, the own H4
            slices); the others are peer-symmetric (Iris heap).
    """

    name: str
    copy_words: int
    sentinel: int
    local: bool


CLEARED_AREAS = (
    ClearedArea("att_box", BOX_WORDS, BOX_EMPTY_WORD, True),
    ClearedArea("h1", H1_WORDS, LAMPORT_EMPTY_WORD, False),
    ClearedArea("h3", H3_WORDS, LAMPORT_EMPTY_WORD, False),
    ClearedArea("h4_own", H4_OWN_WORDS, LAMPORT_EMPTY_WORD, True),
    ClearedArea("routes", ROUTES_WORDS, LAMPORT_EMPTY_WORD, False),
    ClearedArea("h6", H6_WORDS, LAMPORT_EMPTY_WORD, False),
)
"""The areas :func:`clear_next_copies` re-arms (H1 sums are epoch-tagged and
never cleared)."""


@aggregate
class MuseNativeView:
    """This rank's view of the launch state's areas.

    Attributes:
        heaps: The group's Iris heap bases (int64, group order); a peer's
            address of a symmetric area is ``heaps[peer] + (local -
            heaps[rank])``.
        rank: This rank in the group (int32; the hop ends that need it at
            compile time take a RANK constexpr too).
        box: The attention box, uint32 words, two copies (local).
        h1: H1 [copy][RS, AG][source][shard], BF16 (symmetric).
        h1_sums: H1 norm sums, int64 tagged words (symmetric).
        h3: H3 [copy][source][row * 512], BF16 (symmetric).
        h4_own: This rank's own route slices [copy][route row][256 words]
            (local).
        routes: H4 / H5 [copy][column-route stage words], uint32 (symmetric).
        h6: H6 [copy][source][row * 1024], BF16 (symmetric).
        red_in: UNP's norm-sum partials [xcd][sender][row][4], int64 pairs
            (local, epoch-tagged).
        h6_sums: RED's norm sums [rank][xcd][row][4], int64 tagged words
            (symmetric, epoch-tagged).
    """

    heaps: gl.tuple
    rank: gl.tensor
    box: gl.tensor
    h1: gl.tensor
    h1_sums: gl.tensor
    h3: gl.tensor
    h4_own: gl.tensor
    routes: gl.tensor
    h6: gl.tensor
    red_in: gl.tensor
    h6_sums: gl.tensor

    @gluon.constexpr_function
    def __init__(
        self, heaps, rank, box, h1, h1_sums, h3, h4_own, routes, h6, red_in, h6_sums
    ):
        self.heaps = heaps
        self.rank = rank
        self.box = box
        self.h1 = h1
        self.h1_sums = h1_sums
        self.h3 = h3
        self.h4_own = h4_own
        self.routes = routes
        self.h6 = h6
        self.red_in = red_in
        self.h6_sums = h6_sums


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
    """SND ``sender``'s (0-127) entry clear of copy ``(L + 1) & 1`` of every
    area in :data:`CLEARED_AREAS`, this rank's slots only. The caller drains
    (``release_drain``) before its H1 reduce-scatter push."""
    nxt = 1 - launch.copy
    box = view.box.to(gl.pointer_type(gl.uint32), bitcast=True)
    h1 = view.h1.to(gl.pointer_type(gl.uint32), bitcast=True)
    h3 = view.h3.to(gl.pointer_type(gl.uint32), bitcast=True)
    h4_own = view.h4_own.to(gl.pointer_type(gl.uint32), bitcast=True)
    routes = view.routes.to(gl.pointer_type(gl.uint32), bitcast=True)
    h6 = view.h6.to(gl.pointer_type(gl.uint32), bitcast=True)
    iris_clear_share(box + nxt * _BOX_WORDS, sender, _BOX_WORDS, _BOX_EMPTY)
    iris_clear_share(h1 + nxt * _H1_WORDS, sender, _H1_WORDS, _LAMPORT_EMPTY)
    iris_clear_share(h3 + nxt * _H3_WORDS, sender, _H3_WORDS, _LAMPORT_EMPTY)
    iris_clear_share(
        h4_own + nxt * _H4_OWN_WORDS, sender, _H4_OWN_WORDS, _LAMPORT_EMPTY
    )
    iris_clear_share(
        routes + nxt * _ROUTES_WORDS, sender, _ROUTES_WORDS, _LAMPORT_EMPTY
    )
    iris_clear_share(h6 + nxt * _H6_WORDS, sender, _H6_WORDS, _LAMPORT_EMPTY)


@gluon.jit
def h1_rs_sink(
    view, launch, out, block, rows, RANK: gl.constexpr, PEER_CACHE: gl.constexpr
):
    """SND ``block``'s H1 reduce-scatter push: ``out`` is its FP32 [16
    tokens, 64] o_proj partial for columns [64 block, ...)."""
    muse_attntail_rs_push(
        out, block, _peer_addresses(view.h1, view.heaps, RANK), launch.copy, rows,
        64, _HIDDEN, RANK, _WORLD, _H1_SHARD, PEER_CACHE,
    )  # fmt: skip


@gluon.jit
def own1(
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
    """OWN1 ``owner`` (0-7): wait for the 16 sources of its 128 columns, sum
    in rank order, push the AG words and the four norm sums (tagged with
    launch.tag) to every rank. Then zero its sums of rows [rows, MAX_ROWS)
    on every rank: 32-bit tags repeat after 2**32 - 1 launches, so a row a
    later, wider launch reads must not keep a tag that old (TAG-LIVE)."""
    sums_peers = _peer_addresses(view.h1_sums, view.heaps, RANK)
    muse_attntail_own(
        owner, launch.cta, view.h1, _peer_addresses(view.h1, view.heaps, RANK),
        sums_peers, launch.tag, launch.copy, rows, stream_ptr, post_weight_ptr,
        alpha_ptr, beta_ptr, _HIDDEN, RANK, _WORLD, _H1_SHARD, _MAX_ROWS,
        _OWNER_COLUMNS, POST_WEIGHT_OFFSET, POST_HAS_WEIGHT, PEER_CACHE, False,
        launch, on_expiry, _SITE_OWN1,
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
def h3_sink(
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
    (any float layout) to slot RANK of every rank's H3 copy, its own
    included (written through: XN reads it from any XCD)."""
    layout: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [gl.num_warps(), 1], [1, 0])
    v = _iris_sanitize_lamport_bf16(gl.convert_layout(values, layout).to(gl.bfloat16))
    row = gl.arange(0, ROWS, layout=gl.SliceLayout(1, layout))
    column = gl.arange(0, 64, layout=gl.SliceLayout(0, layout))
    offsets = (
        launch.copy * (2 * _H3_WORDS)
        + RANK * _H3_SLOT
        + row[:, None] * 512
        + tile * 64
        + column[None, :]
    )
    valid = (row < rows)[:, None] & (column >= 0)[None, :]
    peers = _peer_addresses(view.h3, view.heaps, RANK)
    published = ()
    for _ in gl.static_range(1, _WORLD):
        published += (v,)
    _iris_lamport_push(
        published, peers, 0, offsets, valid, tile, RANK, _WORLD, PEER_CACHE
    )
    own = gl.multiple_of(view.h3.to(gl.pointer_type(gl.bfloat16), bitcast=True), 16)
    gl.amd.cdna4.buffer_store(v, own, offsets, mask=valid, cache=".wt")


@gluon.jit
def xn_wait(view, launch, row, on_expiry: gl.constexpr, RANK: gl.constexpr):
    """XN of ``row``: wait for every rank's 512 pooled columns (the own slot
    polled too); returns iris_lamport_allgather_wait's words for
    iris_rmsnorm_mxfp4_row."""
    words = view.h3.to(gl.pointer_type(gl.uint32), bitcast=True)
    return iris_lamport_allgather_wait(
        view.h3, words, row * 256, (launch.copy.to(gl.uint32),), RANK, _WORLD, 512,
        4096, _H3_WORDS, _H3_SLOT // 2, False, True, launch, on_expiry, _SITE_XN,
    )  # fmt: skip


@gluon.constexpr_function
def _whole_pairs(layout):
    """Whether every lane of ``layout`` holds whole adjacent column pairs."""
    return isinstance(layout, gl.BlockedLayout) and layout.size_per_thread[-1] % 2 == 0


@gluon.jit
def h4_sink(args, y, route, cols):
    """The MoE down unit's output pushed to its route's column owner (H4):
    the MoE stream's ``sink(args, y, route, cols)`` with ``args = (view,
    launch)``. Called from per-wave code, so it moves no data between lanes
    (no layout conversion, LDS or barrier).

    ``y``: FP32 partial [R, C] in a BlockedLayout whose lanes each hold
    whole adjacent column pairs (an even run of columns, such as
    ``BlockedLayout([1, 4], [16, 4], [1, num_warps], [1, 0])``: a wave's
    [16, 16] tile); ``route``: [R, 1] route row (token * 8 + pick), -1 to
    skip; ``cols``: [1, C] output columns (0-4095, the first of each pair
    even), both slices of y's layout. Each pair goes out as one 32-bit word
    (receivers test a word's high half). Columns this rank owns go to the
    local own slices (read by own4_complete), the rest to slot (7 -
    distance) of the owner's mailbox: device scope locally, system scope to
    peers, no wait."""
    gl.static_assert(
        _whole_pairs(y.type.layout), "h4_sink: lanes must hold whole pairs"
    )
    view, launch = args
    rank = view.rank
    R: gl.constexpr = y.shape[0]
    C: gl.constexpr = y.shape[1]
    values = _iris_sanitize_lamport_bf16(y.to(gl.bfloat16))
    zeros = gl.zeros([R, C], gl.int32, y.type.layout)
    columns = cols + zeros
    routes = route + zeros
    low, high = gl.split(gl.reshape(values, [R, C // 2, 2]))
    words = low.to(gl.uint16, bitcast=True).to(gl.uint32) | (
        high.to(gl.uint16, bitcast=True).to(gl.uint32) << 16
    )
    column, _ = gl.split(gl.reshape(columns, [R, C // 2, 2]))
    route_row, _ = gl.split(gl.reshape(routes, [R, C // 2, 2]))
    owner = column // 512
    within = route_row * _SLICE + (column % 512) // 2
    valid = route_row >= 0
    own = view.h4_own.to(gl.pointer_type(gl.uint32), bitcast=True)
    gl.atomic_store(
        own + launch.copy * _H4_OWN_WORDS + within,
        words,
        mask=valid & (owner == rank),
        sem="relaxed",
        scope="gpu",
    )
    distance = (owner - rank + _WORLD) % _WORLD
    offset = view.routes.to(gl.int64, bitcast=True) - _iris_group_heap_base(
        rank, view.heaps
    )
    target = (
        _iris_group_heap_base(owner, view.heaps)
        + offset
        + 4
        * (
            launch.copy * _ROUTES_WORDS
            + (_WORLD - 1 - distance) * (_MAX_ROWS * 2 * 1024)
            + within
        ).to(gl.int64)
    )
    gl.atomic_store(
        target.to(gl.pointer_type(gl.uint32)),
        words,
        mask=valid & (owner != rank),
        sem="relaxed",
        scope="sys",
    )


@gluon.jit
def own4_complete(
    view,
    launch,
    route_row,
    norm_weight,
    on_expiry: gl.constexpr,
    RANK: gl.constexpr,
    WEIGHT_OFFSET: gl.constexpr,
    PEER_CACHE: gl.constexpr,
):
    """OWN4 of ``route_row``: wait for this rank's own 512-column slice (the
    local dn units' share), then complete it with the peers' partials and
    push the completed columns and sum of squares to every rank (H5)."""
    layout: gl.constexpr = gl.BlockedLayout(
        [1, 4], [64, 1], [gl.num_warps(), 1], [1, 0]
    )
    lane = gl.arange(0, 64, layout=gl.SliceLayout(1, layout))
    word = gl.arange(0, 4, layout=gl.SliceLayout(0, layout))
    offsets = (
        launch.copy * _H4_OWN_WORDS
        + route_row * _SLICE
        + lane[:, None] * 4
        + word[None, :]
    )
    own_ptr = gl.multiple_of(
        view.h4_own.to(gl.pointer_type(gl.uint32), bitcast=True), 16
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
        on_expiry(launch, poll_stop_site(launch, _SITE_OWN4_OWN), route_row)
    iris_column_routes_complete(
        own, launch.copy, norm_weight, view.routes,
        _peer_addresses(view.routes, view.heaps, RANK), route_row, RANK, _WORLD,
        _MAX_ROWS, _ROUTES_WORDS, WEIGHT_OFFSET, PEER_CACHE, False, launch,
        on_expiry, _SITE_OWN4,
    )  # fmt: skip


@gluon.jit
def fin(
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
    """FIN of (token, owner): wait for owner's completed columns of the
    token's 8 routes and every rank's sums of squares (H5), then the routed
    RMSNorm and weighted route sum over those 512 columns (FP32 [512])."""
    loaded, squares = iris_column_routes_wait(
        view.routes, launch.copy, token, owner, _WORLD, _MAX_ROWS, _ROUTES_WORDS, False,
        launch, on_expiry, _SITE_FIN,
    )  # fmt: skip
    return iris_column_routes_combine(
        loaded, squares, route_weights, norm_weight, token, owner, EPS, WEIGHT_OFFSET
    )


@gluon.jit
def h6_sink(
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
    """UNP's post-expert columns [first_output, + OUTPUTS) of every row to
    slot RANK of every rank's H6 copy, its own included (moetail's push)."""
    iris_muse_moetail_push(
        values, launch.copy.to(gl.uint32), first_output, launch.cta,
        _peer_addresses(view.h6, view.heaps, RANK), rows, _iris_lamport_push, RANK,
        _WORLD, HIDDEN_LOCAL, _H6_SLOT, BLOCK_M, OUTPUTS, gl.num_warps(), PEER_CACHE,
        ".wt",
    )  # fmt: skip


@gluon.jit
def unp_sums_publish(view, launch, sums, sender, rows):
    """UNP ``sender`` (0-7) of this XCD publishes its FP32 [16, SUMS] norm-sum
    partials (rows < rows) to its XCD's RED: one (value, tag) pair each; the
    pairs of rows [rows, 16) are zeroed."""
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
    store_pairs(view.red_in + slot, values, launch, valid, "gpu")
    # Rows past ``rows`` get tag 0, which never matches (TAG-LIVE).
    gl.atomic_store(
        view.red_in + slot,
        gl.zeros_like(slot).to(gl.int64),
        mask=(row >= rows)[:, None] & (which >= 0)[None, :],
        sem="relaxed",
        scope="gpu",
    )


@gluon.jit
def red_sums(view, launch, rows, on_expiry: gl.constexpr, RANK: gl.constexpr, PEER_CACHE: gl.constexpr,):  # fmt: skip
    """RED of this XCD: wait for its 8 senders' partials of rows < rows (at
    most 16), add them in FP32 (the same reduction on every rank), and push
    the [row][SUMS] totals as tagged int64 words to slot (RANK, xcd) of every
    rank's h6_sums; rows [rows, MAX_ROWS) of the slot are zeroed (TAG-LIVE).
    Needs 8 waves (one pair per lane)."""
    gl.static_assert(gl.num_warps() == 8)
    poll: gl.constexpr = gl.BlockedLayout([1, 1], [1, 64], [8, 1], [1, 0])
    sender = gl.arange(0, 8, layout=gl.SliceLayout(1, poll))
    entry = gl.arange(0, 16 * _SUMS, layout=gl.SliceLayout(0, poll))
    slot = ((launch.xcd * 8 + sender[:, None]) * _MAX_ROWS) * _SUMS + entry[None, :]
    valid = (sender >= 0)[:, None] & (entry // _SUMS < rows)[None, :]
    values, seen = poll_pairs(view.red_in + slot, valid, launch, "gpu")
    if not seen:
        on_expiry(launch, poll_stop_site(launch, _SITE_RED), 0)
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
    peers = _peer_addresses(view.h6_sums, view.heaps, RANK)
    for peer in gl.static_range(_WORLD):
        destination = gl.multiple_of(peers[peer].to(gl.pointer_type(gl.int64)), 16)
        if peer == RANK:
            gl.amd.cdna4.buffer_store(words, destination, destination_slot, cache=".wt")
        else:
            gl.amd.cdna4.buffer_store(
                words, destination, destination_slot, cache=PEER_CACHE
            )


@gluon.jit
def h6_sums_wait(view, launch, row, on_expiry: gl.constexpr):
    """The FFN boundary's norm sums of ``row`` from every rank's 8 REDs:
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
    values, seen = poll_pairs(view.h6_sums + slot, valid, launch, "sys")
    if not seen:
        on_expiry(launch, poll_stop_site(launch, _SITE_H6_SUMS), row)
    sums_layout: gl.constexpr = gl.BlockedLayout(
        [1, 4], [64, 1], [gl.num_warps(), 1], [1, 0]
    )
    return gl.convert_layout(values.to(gl.float32, bitcast=True), sums_layout)


@gluon.jit
def iris_fail_stop(launch, site, detail):
    """The fused layer's ``on_expiry``: stop every rank's waits, then record
    the failure and trap (fail_stop); never returns.

    Callable from any subset of waves. Each calling wave stores the abort
    marker ``(rank + 1) << 16 | site`` (the origin, for every rank's record)
    into every rank's abort word, system scope; the addresses come from the
    error words' caller lines that MuseNativeIrisState fills.
    """
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
    lane = gl.arange(0, 64 * gl.num_warps(), layout) % 64
    table = launch.error_ptr.to(gl.pointer_type(gl.int64), bitcast=True)
    peer = gl.load(table + _ERROR_CALLER // 2 + lane, mask=lane < 8, other=0)
    rank = gl.load(launch.error_ptr + _ERROR_CALLER + 16 + lane * 0)
    marker = ((rank + 1) << 16) | site
    gl.atomic_store(
        (peer + 4 * _ERROR_ABORT).to(gl.pointer_type(gl.int32)),
        marker,
        mask=lane < 8,
        sem="relaxed",
        scope="sys",
    )
    fail_stop(launch, site, detail)


class MuseNativeIrisState:
    """The launch state's areas for one group (allocate before capture).

    Symmetric areas come from the group's Iris heap at the same offset on
    every rank; local areas are device memory. Lamport areas start empty
    (every copy), the H1 sums at zero (tags are never zero).

    Args:
        peers: The group's ``IrisGroup`` (its heap must hold
            :meth:`heap_bytes` more).
    """

    def __init__(self, peers) -> None:
        if peers.world_size != WORLD_SIZE:
            raise ValueError(
                f"MuseNativeIrisState needs {WORLD_SIZE} ranks, got {peers.world_size}"
            )
        empty = LAMPORT_EMPTY_WORD - (1 << 32)
        self.peers = peers
        self.h1 = peers.ctx.empty((2 * H1_WORDS,), dtype=torch.int32)
        self.h1_sums = peers.ctx.zeros((H1_SUM_WORDS,), dtype=torch.int64)
        self.h3 = peers.ctx.empty((2 * H3_WORDS,), dtype=torch.int32)
        self.routes = peers.ctx.empty((2 * ROUTES_WORDS,), dtype=torch.int32)
        self.h6 = peers.ctx.empty((2 * H6_WORDS,), dtype=torch.int32)
        for area in (self.h1, self.h3, self.routes, self.h6):
            area.fill_(empty)
        self.h6_sums = peers.ctx.zeros((H6_SUM_WORDS,), dtype=torch.int64)
        self.red_in = torch.zeros(
            (RED_IN_WORDS,), dtype=torch.int64, device=peers.device
        )
        self.box = torch.full(
            (2 * BOX_WORDS,), BOX_EMPTY_WORD - (1 << 32), dtype=torch.int32,
            device=peers.device,
        )  # fmt: skip
        self.h4_own = torch.full(
            (2 * H4_OWN_WORDS,), empty, dtype=torch.int32, device=peers.device
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
            allocation_bytes((2 * H1_WORDS,), torch.int32)
            + allocation_bytes((H1_SUM_WORDS,), torch.int64)
            + allocation_bytes((2 * H3_WORDS,), torch.int32)
            + allocation_bytes((2 * ROUTES_WORDS,), torch.int32)
            + allocation_bytes((2 * H6_WORDS,), torch.int32)
            + allocation_bytes((ERROR_WORDS,), torch.int32)
            + allocation_bytes((H6_SUM_WORDS,), torch.int64)
        )

    def view_args(self) -> tuple:
        """Kernel arguments that build :class:`MuseNativeView` in order:
        ``heaps`` (a tuple of 8 int64), rank, then box, h1, h1_sums, h3,
        h4_own, routes, h6, red_in, h6_sums."""
        return (
            self.heaps, self.peers.rank, self.box, self.h1, self.h1_sums, self.h3,
            self.h4_own, self.routes, self.h6, self.red_in, self.h6_sums,
        )  # fmt: skip
