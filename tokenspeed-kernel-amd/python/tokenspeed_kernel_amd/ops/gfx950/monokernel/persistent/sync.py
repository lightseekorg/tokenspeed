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

"""Launch entry, polls, joins and fail-stop deadlines for persistent gfx950 kernels.

A persistent kernel keeps one workgroup on every CU for its whole launch and
hands work between workgroups through memory. These helpers assume a plain
launch of exactly ``GRID`` workgroups that are all resident.

Hand-off classes (each wait is an equality on an epoch-unique, never-zero
tag, so a stale or slow read only delays; wrong data needs a current tag with
an old value, which the classes rule out):

- same XCD, payload + flag: plain payload stores, :func:`release_drain` in
  every storing wave, ``gl.barrier()``, then the flag; consumers poll with
  ``sc1`` or ``smem`` and read the payload with ``.cg`` after the barrier;
- (value, tag) pairs, :func:`store_pairs` / :func:`poll_pairs`: one 8-byte
  store per pair, so a reader that sees the tag sees that store's value; no
  drain and no flag. ``cta`` scope for readers on the same XCD, ``gpu`` for
  any XCD;
- write-through payload + flag for other XCDs: ``.wt`` payload, drain,
  barrier, flag; consumers read with ``.cv``.

Polls are per wave (one owner lane, one ``v_cmp`` vote per retry, or scalar
loads), with no sleep. Every 64 retries a poll reads the clock (the first such
check starts its deadline, so a wait met sooner reads no clock) and the abort
word, with scalar loads in scalar polls so that they never wait behind the
wave's vector loads. One workgroup barrier after the loop is the acquire
point for loads that follow. Polls return whether the word was seen, per
wave; the caller runs ``on_expiry(launch, site, detail)`` otherwise, which
must not return (:func:`fail_stop` is the local handler).
"""

from tokenspeed_kernel_amd._triton import aggregate, gl, gluon
from tokenspeed_kernel_amd.ops.gfx950.monokernel.persistent._intrinsics import (
    memrealtime,
    release_drain,
    scalar_load_glc,
    scalar_load_glc_i64,
    scalar_tags16_differ,
    trap,
    vote,
    wave_acquire,
    xcc_id,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.persistent.timing import realtime_ticks

# Workgroups per launch, XCDs per GPU, and workgroups per XCD.
GRID = 256
XCDS = 8
SLOTS = GRID // XCDS
# int32 words per 128 B line; every counter, flag and epoch word owns a line.
LINE_WORDS = 32
_LINE = gl.constexpr(LINE_WORDS)
_XCDS = gl.constexpr(XCDS)
_SLOTS = gl.constexpr(SLOTS)

# Error words (int32), local to one GPU. The first line holds the record of a
# failure (site, logical workgroup, detail, epoch, the kernel's label such as
# a layer index) and, at ERROR_MIRROR, the int64 device address of a host
# copy of the record (0: none); peers set the second line to stop this GPU's
# waits; the caller owns the last two lines (for example, where to reach its
# peers).
ERROR_RECORD = 0
ERROR_RECORD_FIELDS = 5
ERROR_MIRROR = 16
ERROR_ABORT = LINE_WORDS
ERROR_CALLER = 2 * LINE_WORDS
ERROR_WORDS = 4 * LINE_WORDS
_ERROR_RECORD = gl.constexpr(ERROR_RECORD)
_ERROR_MIRROR = gl.constexpr(ERROR_MIRROR)
_RECORD_FIELDS = gl.constexpr(ERROR_RECORD_FIELDS)
_ERROR_ABORT = gl.constexpr(ERROR_ABORT)

# Entry counters: one int64 word per 128 B line, per program id (epochs) and
# per (xcd, slot) role (claims). Zeroed together once; never re-zeroed.
ENTRY_WORDS = GRID * LINE_WORDS // 2
_COUNTER_STRIDE = gl.constexpr(LINE_WORDS // 2)

# Failure sites raised by these helpers; kernels number their own from
# FIRST_KERNEL_SITE.
SITE_CLAIM = 1
SITE_JOIN = 2
SITE_ABORTED = 3
FIRST_KERNEL_SITE = 16
_SITE_CLAIM = gl.constexpr(SITE_CLAIM)
_SITE_JOIN = gl.constexpr(SITE_JOIN)
_SITE_ABORTED = gl.constexpr(SITE_ABORTED)

# Polls read the clock and the abort word once per this many retries; doing
# it on every retry costs ~0.2 us per handoff round trip.
_CHECK_RETRIES = gl.constexpr(64)

# Join counters: one line per XCD, then one line for the whole grid, in two
# sets that consecutive joins alternate.
JOIN_SET_WORDS = (XCDS + 1) * LINE_WORDS
JOIN_WORDS = 2 * JOIN_SET_WORDS
_JOIN_SET_WORDS = gl.constexpr(JOIN_SET_WORDS)
# A top-bit toggle join of N members: every member adds 1, the designated one
# adds 1 + 2**31 - N instead, so the N-th arrival (in any order) flips bit 31
# and leaves the low bits as they were. Never skipped, so it needs no reset.
_XCD_TOGGLE_ADD = gl.constexpr(1 + 2**31 - SLOTS)
_TOP_TOGGLE_ADD = gl.constexpr(1 + 2**31 - XCDS)


@aggregate
class Launch:
    """This workgroup's identity and fail-stop state for one launch.

    ``cta = slot * 8 + xcd`` is the logical workgroup id that roles use in
    place of the program id, so ``cta % 8`` is the physical XCD. ``epoch``
    (int64) never wraps; ``tag`` (int32, never zero) marks this launch's
    tagged words; ``copy = epoch & 1`` picks a two-copy area.
    """

    cta: gl.tensor
    xcd: gl.tensor
    slot: gl.tensor
    epoch: gl.tensor
    tag: gl.tensor
    copy: gl.tensor
    label: gl.tensor
    error_ptr: gl.tensor
    abort_ptr: gl.tensor
    deadline_ticks: gl.tensor
    arena_ptr: gl.tensor

    @gluon.constexpr_function
    def __init__(self, cta, xcd, slot, epoch, tag, copy, label, error_ptr, abort_ptr, deadline_ticks, arena_ptr,):  # fmt: skip
        self.cta = cta
        self.xcd = xcd
        self.slot = slot
        self.epoch = epoch
        self.tag = tag
        self.copy = copy
        self.label = label
        self.error_ptr = error_ptr
        self.abort_ptr = abort_ptr
        self.deadline_ticks = deadline_ticks
        self.arena_ptr = arena_ptr


@aggregate
class Entry:
    """The launch entry between :func:`enter_issue` and :func:`enter_check`:
    this workgroup's role and the lines of its two counters."""

    cta: gl.tensor
    xcd: gl.tensor
    slot: gl.tensor
    pid: gl.tensor
    epoch_line: gl.tensor
    claim_line: gl.tensor

    @gluon.constexpr_function
    def __init__(self, cta, xcd, slot, pid, epoch_line, claim_line):
        self.cta = cta
        self.xcd = xcd
        self.slot = slot
        self.pid = pid
        self.epoch_line = epoch_line
        self.claim_line = claim_line


@gluon.jit
def launch_tag(epoch, BITS: gl.constexpr):
    """Map a 64-bit launch epoch to a nonzero ``BITS``-bit tag, as int64.

    Zero-filled and long-idle words never match a live tag, and consecutive
    launches always differ. Tags repeat after 2**BITS - 1 launches, so a
    tagged slot that a later launch may poll is rewritten or scrubbed in every
    launch.
    """
    return epoch % ((1 << BITS) - 1) + 1


@gluon.jit
def logical_cta():
    """This workgroup's logical id, ``(pid // 8) * 8 + xcd``, from registers.

    The role it names may issue its first weight or KV loads before
    :func:`enter_check`; nothing else may happen before the check.
    """
    return (gl.program_id(0) // _XCDS) * _XCDS + xcc_id()


@gluon.jit
def _lanes():
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
    return gl.arange(0, 64 * gl.num_warps(), layout)


@gluon.jit
def enter_issue(epoch_ptr, claim_ptr):
    """Start the launch entry: this workgroup's role from the hardware XCD and
    the program id, and its counters' lines. No memory access: issue the
    role's first loads after it, then call :func:`enter_check`.

    Args:
        epoch_ptr: int64 words, one 128 B line per program id
            (``ENTRY_WORDS``).
        claim_ptr: int64 words, one 128 B line per ``(xcd, slot)``. Both are
            zeroed together once, when the state is created.
    """
    pid = gl.program_id(0)
    xcd = xcc_id()
    slot = pid // _XCDS
    epoch_line = epoch_ptr + pid * _COUNTER_STRIDE
    claim_line = claim_ptr + (xcd * _SLOTS + slot) * _COUNTER_STRIDE
    return Entry(slot * _XCDS + xcd, xcd, slot, pid, epoch_line, claim_line)


@gluon.jit
def enter_check(entry, error_ptr, arena_ptr, deadline_ticks, label, on_expiry):
    """Finish the launch entry; returns this wave's :class:`Launch`.

    Every workgroup advances its epoch word once per launch, so all
    workgroups hold the same epoch, and claims its role ``(xcd, slot)``; a
    second workgroup on a role stops with ``SITE_CLAIM`` (a missing role
    implies a duplicate: 256 workgroups, 256 roles). No wave stores, pushes
    or runs an atomic before the check. The claim words are always touched
    from their own XCD; an epoch word follows its program id to whichever
    XCD runs it, so its coherence across launches rests on the kernel
    boundary.

    Every wave reads the epoch word with a scalar ``glc`` load, the waves
    join at a barrier (so none reads the word after the store), and the
    epoch goes back with a relaxed atomic store, which does not wait for
    earlier loads. The workgroup then exchanges ``(tag, pid)`` into its
    claim word and stops if the old tag is this launch's. The exchange
    returns after the vector loads issued before it (results return in
    order), so the check waits for the role's first weights: issue only the
    weights the role uses first before the entry.

    Args:
        entry: From :func:`enter_issue`.
        error_ptr: int32 error words (``ERROR_WORDS``).
        arena_ptr: int64 words of the launch's uncached flag arena.
        deadline_ticks: Clock ticks a wait may last before it fails.
        label: int32 recorded with a failure, such as the layer index.
        on_expiry: ``on_expiry(launch, site, detail)``; must not return.
    """
    epoch = scalar_load_glc_i64(entry.epoch_line) + 1
    gl.barrier()
    gl.atomic_store(entry.epoch_line, epoch, sem="relaxed", scope="gpu")
    tag = launch_tag(epoch, 32).to(gl.int32)
    launch = Launch(
        entry.cta, entry.xcd, entry.slot, epoch, tag, (epoch & 1).to(gl.int32),
        gl.to_tensor(label), error_ptr, error_ptr + _ERROR_ABORT, deadline_ticks,
        arena_ptr,
    )  # fmt: skip
    mark = (tag.to(gl.int64) << 32) | entry.pid.to(gl.int64)
    old = gl.atomic_xchg(entry.claim_line, mark, sem="relaxed", scope="gpu")
    # The exchange's result reaches the other waves through compiler
    # scratch in LDS; the barrier keeps every wave's read of it before any
    # wave's next LDS write (the role's buffers may start there).
    gl.barrier()
    if ((old >> 32) & 0xFFFFFFFF) == tag.to(gl.int64):
        # The other workgroup's program id.
        on_expiry(launch, _SITE_CLAIM, (old & 0xFFFFFFFF).to(gl.int32))
    return launch


@gluon.jit
def fail_stop(launch, site, detail):
    """Record a failure in the error words and stop the queue; never returns.

    Callable from any subset of waves (a poll that expires in one wave):
    every calling lane tries to claim the record, so it never depends on a
    particular wave, and no workgroup barrier is involved. The record is also
    written to the host copy at ``ERROR_MIRROR``, if set, before the trap:
    after a queue error the device memory is no longer readable.

    Args:
        launch: The failing workgroup's :class:`Launch`.
        site: What failed: a ``SITE_*`` value, or a kernel's own site.
        detail: Site-specific detail, such as a mask of missing peers. For
            ``SITE_ABORTED`` (a peer stopped this wait) the record holds the
            abort word instead, the marker that names the origin.
    """
    record_failure(launch, site, detail)
    trap()


@gluon.jit
def record_failure(launch, site, detail):
    """:func:`fail_stop`'s record without the trap: claim the error record
    (the first claim on this GPU wins), write it and its host copy, and
    drain. A caller that must publish something after its record, such as a
    peer abort marker, calls this first and then traps itself."""
    lane = _lanes()
    # One load, on the failure path only.
    abort = gl.load(launch.abort_ptr + lane * 0, cache_modifier=".cv")
    detail = gl.where(site == _SITE_ABORTED, abort, detail)
    record = launch.error_ptr + _ERROR_RECORD + lane * 0
    # The first claim owns the record (sites are never zero), so failing
    # waves and workgroups cannot interleave their fields. (A scalar atomic
    # would run on the workgroup's first thread only, which may not be in a
    # calling wave.)
    owner = (
        gl.atomic_cas(
            record, gl.zeros_like(lane), gl.zeros_like(lane) + site, sem="relaxed",
            scope="gpu",
        )
        == 0
    )  # fmt: skip
    fields = gl.where(lane % 8 == 0, site, 0)
    fields = gl.where(lane % 8 == 1, launch.cta, fields)
    fields = gl.where(lane % 8 == 2, detail, fields)
    fields = gl.where(lane % 8 == 3, launch.epoch.to(gl.int32), fields)
    fields = gl.where(lane % 8 == 4, launch.label, fields)
    # The owner's wave stores the fields, one per lane (its lanes 1-4 here,
    # and 0-4 in the host copy).
    word = lane % 64
    wave_owns = vote(owner.to(gl.int32))
    field_of = gl.where(word < 8, fields, 0)
    own_fields = wave_owns & (word >= 1) & (word < _RECORD_FIELDS)
    gl.store(launch.error_ptr + word, field_of, mask=own_fields, cache_modifier=".wt")
    table = launch.error_ptr.to(gl.pointer_type(gl.int64), bitcast=True)
    mirror = gl.load(table + _ERROR_MIRROR // 2)
    host = mirror.to(gl.pointer_type(gl.int32), bitcast=True)
    to_host = wave_owns & (word < _RECORD_FIELDS) & (mirror != 0)
    gl.store(host + word, field_of, mask=to_host, cache_modifier=".wt")
    release_drain()


@gluon.jit
def poll_expired(launch, start, retries):
    """Return whether a poll that began at ``start`` must stop.

    Checked every ``_CHECK_RETRIES`` retries: the deadline has passed, or a
    peer stored its abort marker in this GPU's error words.
    """
    stop = retries < 0
    if retries % _CHECK_RETRIES == 0:
        late = realtime_ticks() - start > launch.deadline_ticks
        abort = gl.load(launch.abort_ptr, cache_modifier=".cv")
        stop = late | (abort != 0)
    return stop


@gluon.jit
def poll_stop_site(launch, site):
    """Return ``site`` for an expired poll, or the abort site if a peer stopped."""
    abort = gl.load(launch.abort_ptr, cache_modifier=".cv")
    return gl.where(abort != 0, _SITE_ABORTED, site)


@gluon.jit
def poll_check(launch, start, retries, SCALAR: gl.constexpr):
    """The deadline and abort check of a poll loop, for loops written
    elsewhere (the Iris hop waits); call it every 64 retries with ``start``
    initialized to 0. Returns ``(stop, start)``: the first check starts the
    deadline, so a wait met within 64 retries reads no clock. ``SCALAR``
    reads the abort word with a scalar load, which does not wait behind the
    wave's vector loads (a vector load here would drain them)."""
    now = memrealtime()
    start = gl.where(retries == _CHECK_RETRIES, now, start)
    late = now - start > launch.deadline_ticks
    if SCALAR:
        abort = scalar_load_glc(launch.abort_ptr)
    else:
        abort = gl.load(launch.abort_ptr, cache_modifier=".cv")
    return late | (abort != 0), start


@gluon.jit
def _poll_load(ptrs, owner, LOAD: gl.constexpr):
    if LOAD == "sc1":
        # Device scope: words this GPU wrote through a coherent path.
        return gl.atomic_load(ptrs, mask=owner, sem="relaxed", scope="gpu")
    else:
        gl.static_assert(LOAD == ".cv", "LOAD is sc1, .cv or smem")
        # System scope: words another GPU stored.
        return gl.load(ptrs, mask=owner, other=0, cache_modifier=".cv")


@gluon.jit
def poll_until(word_ptr, expected, launch, LOAD: gl.constexpr):
    """Wait until the int32 at ``word_ptr`` equals ``expected``.

    ``LOAD`` (no default; it states the word's hand-off class):

    - ``"sc1"``: lane 0 of the first wave reloads with relaxed device-scope
      loads and its wave exits on a one-instruction vote (the other waves
      skip the loop): words this GPU stored;
    - ``".cv"``: the same with system-scope loads: words another GPU stored;
    - ``"smem"``: scalar loads past the scalar cache (``glc``), which never
      wait behind the wave's vector loads, so a role may poll with its
      weights in flight. Only for words written on this XCD or in uncached
      memory.

    Wave independence: the waves that poll (lane 0's wave, or every wave
    with ``"smem"``) loop on their own reads with no barrier or LDS in the
    loop, so they need not agree on a trip count; the closing workgroup
    barrier is the one point that joins them, and loads after it see the
    word for the whole workgroup.

    Returns:
        Whether the word was seen; ``False`` only in the wave that stopped
        (every wave with ``"smem"``). After one workgroup barrier.
    """
    start = gl.to_tensor(0).to(gl.int64)
    retries = 0
    stop = retries < 0
    if LOAD == "smem":
        seen = scalar_load_glc(word_ptr)
        while (seen != expected) & (not stop):
            retries += 1
            seen = scalar_load_glc(word_ptr)
            if retries % _CHECK_RETRIES == 0:
                stop, start = poll_check(launch, start, retries, True)
    else:
        lane = _lanes()
        owner = lane == 0
        ptrs = word_ptr + lane * 0
        missing = owner & (_poll_load(ptrs, owner, LOAD) != expected)
        while vote(missing.to(gl.int32)) & (not stop):
            retries += 1
            missing = owner & (_poll_load(ptrs, owner, LOAD) != expected)
            if retries % _CHECK_RETRIES == 0:
                stop, start = poll_check(launch, start, retries, False)
    gl.barrier()
    return not stop


@gluon.jit
def poll_until_all(word_ptrs, valid, expected, launch, LOAD: gl.constexpr):
    """Wait until every valid lane's int32 word equals its ``expected``.

    ``word_ptrs``, ``valid`` and ``expected`` hold one word per lane
    (``[64 * num_warps]``). Each wave reloads only its lanes' missing words
    (``LOAD``: ``"sc1"`` or ``".cv"``, as in :func:`poll_until`) and exits on
    a one-instruction vote. Returns whether this wave saw all its words.
    """
    missing = valid & (_poll_load(word_ptrs, valid, LOAD) != expected)
    start = gl.to_tensor(0).to(gl.int64)
    retries = 0
    stop = retries < 0
    while vote(missing.to(gl.int32)) & (not stop):
        retries += 1
        missing = missing & (_poll_load(word_ptrs, missing, LOAD) != expected)
        if retries % _CHECK_RETRIES == 0:
            stop, start = poll_check(launch, start, retries, False)
    gl.barrier()
    return not stop


@gluon.jit
def store_pairs(pair_ptrs, values, launch, mask, SCOPE: gl.constexpr):
    """Publish int32 ``values`` as (value, tag) pairs: one 8-byte relaxed
    atomic store each (``dwordx2``), ``tag << 32 | value``, the H1 norm sums'
    format. No wait after it.

    ``SCOPE``: ``"cta"`` (``sc0``) for readers on this XCD only, ``"gpu"``
    (``sc1``) for readers on any XCD. Every pair is written at most once per
    launch.
    """
    word = (launch.tag.to(gl.int64) << 32) | (values.to(gl.int64) & 0xFFFFFFFF)
    gl.atomic_store(pair_ptrs, word, mask=mask, sem="relaxed", scope=SCOPE)


@gluon.jit
def poll_pairs(pair_ptrs, valid, launch, SCOPE: gl.constexpr):
    """Wait for this launch's (value, tag) pairs; one pair per lane.

    Each lane reloads its pair with one 8-byte relaxed load (``SCOPE``
    ``"gpu"``: ``sc1``, for pairs this GPU stored; ``"sys"`` for pairs
    another GPU stored) until the high (tag) half equals the launch tag; the
    low (value) half comes from that same load, so no second read is needed.

    Returns:
        ``(values, seen)``: the int32 values (0 in invalid lanes) and whether
        this wave saw all its pairs; after one workgroup barrier.
    """
    words = gl.atomic_load(pair_ptrs, mask=valid, sem="relaxed", scope=SCOPE)
    missing = valid & ((words >> 32).to(gl.int32) != launch.tag)
    start = gl.to_tensor(0).to(gl.int64)
    retries = 0
    stop = retries < 0
    while vote(missing.to(gl.int32)) & (not stop):
        retries += 1
        words = gl.where(
            missing,
            gl.atomic_load(pair_ptrs, mask=missing, sem="relaxed", scope=SCOPE),
            words,
        )
        missing = valid & ((words >> 32).to(gl.int32) != launch.tag)
        if retries % _CHECK_RETRIES == 0:
            stop, start = poll_check(launch, start, retries, False)
    gl.barrier()
    return gl.where(valid, words.to(gl.int32), 0), not stop


@gluon.jit
def poll_tags16(word_ptr, tag, launch, GLC: gl.constexpr, BARRIER: gl.constexpr):
    """Wait until all 16 int32 tags at ``word_ptr`` (64 B aligned) equal
    ``tag``, with scalar loads only (never behind the wave's vector loads).

    ``GLC`` (no default): ``True`` reads past the scalar cache (same-XCD
    words, or uncached memory, measured); ``False`` reads through it after
    ``s_dcache_inv`` (unmeasured). ``BARRIER``: end with a workgroup barrier
    (other waves use what the tags guard), or only with a per-wave acquire
    (this wave alone polls and reads, as in per-wave work loops). Returns
    whether the tags were seen, per wave.

    Wave independence (``BARRIER`` False): each wave polls with its own
    scalar loads and goes on alone, so per-wave loops with different trip
    counts may call it. That is safe only when the waiting wave itself reads
    what the tags guard; its ``wave_acquire`` keeps those reads after the
    poll. Data shared across waves needs ``BARRIER`` True.
    """
    start = gl.to_tensor(0).to(gl.int64)
    retries = 0
    stop = retries < 0
    differ = scalar_tags16_differ(word_ptr, tag, GLC)
    while (differ != 0) & (not stop):
        retries += 1
        differ = scalar_tags16_differ(word_ptr, tag, GLC)
        if retries % _CHECK_RETRIES == 0:
            stop, start = poll_check(launch, start, retries, True)
    if BARRIER:
        gl.barrier()
    else:
        wave_acquire()
    return not stop


@gluon.jit
def grid_join(join_ptr, launch, on_expiry, SET: gl.constexpr, detail):
    """Wait until every workgroup of the grid has reached this join.

    ``detail`` names the join in a failure record (such as the phase before
    it). Prefer data-driven waits: a grid join costs ~2.1 us.

    A two-level top-bit toggle (``JOIN_WORDS`` int32 words, zeroed once when
    the state is created): the 32 workgroups of an XCD arrive on their XCD's
    line, the arrival that completes an XCD arrives on the grid line, and
    everyone waits for the grid line's top bit to flip. Both levels flip once
    per join from the same start, so a waiter takes the bit to wait for from
    its own XCD arrival. A join must never be skipped: a flip that some
    workgroup missed is never undone. Consecutive joins of a launch alternate
    the two line sets (``SET``: the join's index in the launch, modulo 2).

    Stores before the join are visible after it to ``.cv`` loads: each wave
    drains its stores before its workgroup arrives.
    """
    lines = join_ptr + SET * _JOIN_SET_WORDS
    release_drain()
    gl.barrier()
    add = gl.where(launch.slot == 0, _XCD_TOGGLE_ADD, 1)
    old = gl.atomic_add(lines + launch.xcd * _LINE, add, sem="relaxed", scope="gpu")
    before = old < 0
    if ((old + add) < 0) != before:
        top_add = gl.where(launch.xcd == 0, _TOP_TOGGLE_ADD, 1)
        gl.atomic_add(lines + _XCDS * _LINE, top_add, sem="relaxed", scope="gpu")
    flipped = gl.where(before, 0, -(2**31))
    seen = poll_until(lines + _XCDS * _LINE, flipped, launch, "sc1")
    if not seen:
        on_expiry(launch, poll_stop_site(launch, _SITE_JOIN), detail)
