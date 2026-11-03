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

"""Launch entry, joins and fail-stop deadlines for persistent gfx950 kernels.

A persistent kernel keeps one workgroup on every CU for its whole launch and
hands work between workgroups through memory. These helpers assume a plain
launch of exactly ``GRID`` workgroups that are all resident; nothing here
waits on a kernel boundary.

Memory model (no acquire invalidate and no L2 write-back on these paths):

- A producer writes handoff data with write-through stores (``.wt``), waits
  for them in every wave (``s_waitcnt vmcnt(0)``) and joins its workgroup
  before it publishes. ``gl.barrier()`` alone emits no ``vmcnt`` wait.
- A consumer reads handoff data only with ``.cv`` loads (``sc0 sc1``), which
  bypass non-coherent cache lines on a GPU with one L2 per XCD.

Every wait carries a deadline. On expiry, or when a peer has stopped, the
wait ends and the caller's ``on_expiry(launch, site, detail)`` handler runs.
:func:`fail_stop` is the local handler: it records the failure in the error
words and raises a queue error, which aborts the process. A kernel must only
pass handlers that do not return.
"""

from tokenspeed_kernel_amd._triton import aggregate, gl, gluon
from tokenspeed_kernel_amd.ops.gfx950._xcd import xcc_id
from tokenspeed_kernel_amd.ops.gfx950.persistent.timing import realtime_ticks

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
# a layer index); peers set the second line to stop this GPU's waits; the
# caller owns the last two lines (for example, where to reach its peers).
ERROR_RECORD = 0
ERROR_ABORT = LINE_WORDS
ERROR_CALLER = 2 * LINE_WORDS
ERROR_WORDS = 4 * LINE_WORDS
_ERROR_RECORD = gl.constexpr(ERROR_RECORD)
_ERROR_ABORT = gl.constexpr(ERROR_ABORT)

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

# A word wait polls in slices of this many ns, then checks the deadline and
# the abort word.
_WAIT_SLICE_NS = gl.constexpr(50_000)

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

    ``cta = slot * 8 + xcd`` is the logical workgroup id that roles and
    virtual programs use in place of the program id, so ``cta % 8`` is the
    physical XCD.
    """

    cta: gl.tensor
    xcd: gl.tensor
    slot: gl.tensor
    epoch: gl.tensor
    label: gl.tensor
    error_ptr: gl.tensor
    deadline_ticks: gl.tensor

    @gluon.constexpr_function
    def __init__(self, cta, xcd, slot, epoch, label, error_ptr, deadline_ticks):
        self.cta = cta
        self.xcd = xcd
        self.slot = slot
        self.epoch = epoch
        self.label = label
        self.error_ptr = error_ptr
        self.deadline_ticks = deadline_ticks


@gluon.jit
def wait_vector_memory():
    """Wait for this wave's outstanding vector loads and stores."""
    gl.inline_asm("s_waitcnt vmcnt(0)", "~{memory}", [], (), is_pure=False)


@gluon.jit
def trap():
    """Raise a queue error (s_trap 2); the runtime then aborts the process."""
    gl.inline_asm("s_trap 2", "~{memory}", [], (), is_pure=False)


@gluon.jit
def launch_tag(epoch, BITS: gl.constexpr):
    """Map a launch epoch to a nonzero ``BITS``-bit tag, as int64.

    Zero-filled and long-idle words therefore never match a live tag.
    """
    wide = epoch.to(gl.uint32).to(gl.int64)
    return wide % ((1 << BITS) - 1) + 1


@gluon.jit
def fail_stop(launch, site, detail):
    """Record a failure in the error words and stop the queue; never returns.

    Callable from any subset of waves (a poll that expires in one wave):
    every calling lane tries to claim the record, so it never depends on a
    particular wave, and no workgroup barrier is involved.

    Args:
        launch: The failing workgroup's :class:`Launch`.
        site: What failed: a ``SITE_*`` value, or a kernel's own site.
        detail: Site-specific detail, such as a mask of missing peers.
    """
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
    lane = gl.arange(0, 64 * gl.num_warps(), layout)
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
    gl.store(
        record + 1, gl.zeros_like(lane) + launch.cta, mask=owner, cache_modifier=".wt"
    )
    gl.store(record + 2, gl.zeros_like(lane) + detail, mask=owner, cache_modifier=".wt")
    gl.store(
        record + 3, gl.zeros_like(lane) + launch.epoch, mask=owner, cache_modifier=".wt"
    )
    gl.store(
        record + 4, gl.zeros_like(lane) + launch.label, mask=owner, cache_modifier=".wt"
    )
    wait_vector_memory()
    trap()


@gluon.jit
def poll_expired(launch, start, retries):
    """Return whether a poll that began at ``start`` must stop.

    Checked every ``_CHECK_RETRIES`` retries: the deadline has passed, or a
    peer stored its abort marker in this GPU's error words.
    """
    stop = retries < 0
    if retries % _CHECK_RETRIES == 0:
        late = realtime_ticks() - start > launch.deadline_ticks
        abort = gl.load(launch.error_ptr + _ERROR_ABORT, cache_modifier=".cv")
        stop = late | (abort != 0)
    return stop


@gluon.jit
def poll_stop_site(launch, site):
    """Return ``site`` for an expired poll, or the abort site if a peer stopped."""
    abort = gl.load(launch.error_ptr + _ERROR_ABORT, cache_modifier=".cv")
    return gl.where(abort != 0, _SITE_ABORTED, site)


@gluon.jit
def enter_launch(epoch_ptr, claim_ptr, error_ptr, deadline_ticks, label, on_expiry):
    """Advance this workgroup's launch epoch and claim its logical role.

    The XCD comes from the hardware register and the slot from the program
    id, so ``(xcd, slot)`` is unique for any dispatch that hands the XCDs
    workgroups in turn from any start. Each workgroup exchanges
    ``(tag, pid)`` into ``claim[xcd][slot]``; finding this launch's tag there
    means two workgroups took one role, and ``on_expiry`` runs. The check
    completes before this returns, so it precedes every role-dependent store
    or atomic; only weight loads may be issued before the call.

    Args:
        epoch_ptr: int32 epoch words, one line per program id. Zeroed once
            when the state is created; every launch of the grid advances each.
        claim_ptr: int64 claim words, one line per ``(xcd, slot)``. Zeroed
            together with the epoch words, and only then.
        error_ptr: int32 error words (``ERROR_WORDS``).
        deadline_ticks: Clock ticks a wait may last before it fails.
        label: int32 recorded with a failure, such as the layer index.
        on_expiry: ``on_expiry(launch, site, detail)``.

    Returns:
        The workgroup's :class:`Launch`.
    """
    pid = gl.program_id(0)
    xcd = xcc_id()
    slot = pid // _XCDS
    cta = slot * _XCDS + xcd
    # The word of a program id is touched only by that program id, so the
    # epochs of all workgroups advance together. Its XCD changes from launch
    # to launch, so it is read and written past the L2s.
    epoch = gl.load(epoch_ptr + pid * _LINE, cache_modifier=".cv") + 1
    gl.store(epoch_ptr + pid * _LINE, epoch, cache_modifier=".wt")
    launch = Launch(
        cta, xcd, slot, epoch, gl.to_tensor(label), error_ptr, deadline_ticks
    )
    tag = launch_tag(epoch, 32)
    claim = claim_ptr + (xcd * _SLOTS + slot) * (_LINE // 2)
    old = gl.atomic_xchg(claim, (tag << 32) | pid, sem="relaxed", scope="gpu")
    if ((old >> 32) & 0xFFFFFFFF) == tag:
        on_expiry(launch, _SITE_CLAIM, (old & 0xFFFFFFFF).to(gl.int32))
    return launch


@gluon.jit
def wait_word(
    word_ptr, expected, launch, on_expiry: gl.constexpr, SITE: gl.constexpr, detail
):
    """Wait until the int32 at ``word_ptr`` equals ``expected``.

    One lane polls with relaxed device-scope loads (``gl.atomic_poll``) while
    the other waves wait at its closing barrier. Every ``_WAIT_SLICE_NS`` the
    workgroup checks the launch deadline and the abort word, and calls
    ``on_expiry(launch, site, detail)`` when either stops it. Returns after a
    workgroup barrier, so ``.cv`` loads after it see what the word's writer
    stored before writing it.
    """
    start = realtime_ticks()
    seen = gl.atomic_poll(
        word_ptr, expected, sem="relaxed", scope="gpu", timeout_ns=_WAIT_SLICE_NS
    )
    alive = start >= 0
    while (not seen) & alive:
        late = realtime_ticks() - start > launch.deadline_ticks
        abort = gl.load(launch.error_ptr + _ERROR_ABORT, cache_modifier=".cv")
        alive = not (late | (abort != 0))
        if alive:
            seen = gl.atomic_poll(
                word_ptr,
                expected,
                sem="relaxed",
                scope="gpu",
                timeout_ns=_WAIT_SLICE_NS,
            )
    if not seen:
        on_expiry(launch, poll_stop_site(launch, SITE), detail)
    gl.barrier()


@gluon.jit
def grid_join(join_ptr, launch, on_expiry, SET: gl.constexpr, detail):
    """Wait until every workgroup of the grid has reached this join.

    ``detail`` names the join in a failure record (such as the phase before
    it).

    A two-level top-bit toggle (``JOIN_WORDS`` int32 words, zeroed once when
    the state is created): the 32 workgroups of an XCD arrive on their XCD's
    line, the arrival that completes an XCD arrives on the grid line, and
    everyone waits for the grid line's top bit to flip. Both levels flip once
    per join from the same start, so a waiter takes the bit to wait for from
    its own XCD arrival. A join must never be skipped: a flip that some
    workgroup missed is never undone.

    Consecutive joins of a launch alternate the two line sets (``SET``: the
    join's index in the launch, modulo 2). A set is used again only after
    every workgroup passed the join in between, so a waiter sees its grid
    line at exactly 0 or ``-2**31`` and can wait for that value.

    Stores before the join are visible after it to ``.cv`` loads: each wave
    drains its stores before its workgroup arrives.
    """
    lines = join_ptr + SET * _JOIN_SET_WORDS
    wait_vector_memory()
    gl.barrier()
    add = gl.where(launch.slot == 0, _XCD_TOGGLE_ADD, 1)
    old = gl.atomic_add(lines + launch.xcd * _LINE, add, sem="relaxed", scope="gpu")
    before = old < 0
    if ((old + add) < 0) != before:
        top_add = gl.where(launch.xcd == 0, _TOP_TOGGLE_ADD, 1)
        gl.atomic_add(lines + _XCDS * _LINE, top_add, sem="relaxed", scope="gpu")
    flipped = gl.where(before, 0, -(2**31))
    wait_word(lines + _XCDS * _LINE, flipped, launch, on_expiry, _SITE_JOIN, detail)
