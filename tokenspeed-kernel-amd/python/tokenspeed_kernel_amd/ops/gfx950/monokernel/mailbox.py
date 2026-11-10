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

"""Tagged-pair mailboxes: how monokernel CTAs hand values to each other.

Each 32-bit value that one CTA gives another is an aligned 8-byte pair
``(tag << 32) | payload``. One 64-bit store writes the pair. The tag is unique
per launch (:func:`launch_tag`), so a pair is its own ready flag: the consumer
re-loads it until the tag matches. The protocol needs no flag, fence or
clearing. A stale pair from an earlier launch never matches.

Memory rules on gfx950 (coarse-grained VRAM):

* Stores and polls are agent-scope (``sc1``): relaxed ``gpu``-scope atomics.
  The vL1D serves a plain or ``sc0`` load, which can then spin forever on a
  stale line.
* The 64-bit store and load are single-copy atomic, so a pair never mixes an
  old payload with a new tag.
* Only throttled polling is safe. With every thread of every CTA spinning, the
  ``sc1`` load flood starves the producers' stores and the grid livelocks.
  :func:`pre_poll` lets one wave spin on one hint pair per producer before the
  payload :func:`poll`. The poll then usually succeeds on its first load.

Across ranks (a plan of ``world`` > 1 on Iris's symmetric heap), a stage
puts into another rank's copy of a peer region (:meth:`Mailbox.put_peer`):
a system-scope store over the GPU links. A poll of a peer region is
system-scope (``sc0 sc1``), so the local L2 never serves a stale line of
it.

Fail-stop: a poll gives up after about a minute of waiting (the realtime
clock, read every ``POLL_CHECK`` retries only) and executes ``s_trap 2``. The
runtime then reports a GPU exception and the process stops. A deadlock or a
missing producer thus stops the server with an error instead of hanging it.

Stuck-poll record: a mailbox made with a ``stuck`` buffer (``int64 [grid,
STUCK_WORDS]``, host-visible) has each poll write, only when it gives up and
before the trap, one record into its CTA's slot: the expected tag, the stale
pair as observed and its address, the clock at the poll's first check and at
the trap, the stale operand's index in the poll, and the record's offset. The host
decodes the region, row and offset from the address and the plan, and the
step and layer from the tag (``stuck.decode``). Without a ``stuck`` buffer
a poll writes nothing.
"""

from __future__ import annotations

from tokenspeed_kernel_amd import _extern
from tokenspeed_kernel_amd._triton import aggregate, gl, gluon
from tokenspeed_kernel_amd.ops.gfx950.monokernel import schedule, trace
from tokenspeed_kernel_amd.ops.gfx950.monokernel.plan import LAYER_SLOTS as _SLOTS

LAYER_SLOTS = gl.constexpr(_SLOTS)  # the device copy of plan.LAYER_SLOTS
# A waiting poll reads the realtime clock (100 MHz on gfx950) every POLL_CHECK
# retries (a retry is at least about 0.12 us: one check per ~0.5 ms, never in
# a launch's normal waits) and traps once it has waited POLL_TIMEOUT_MANT <<
# POLL_TIMEOUT_SHIFT ticks (59 s). Time, not a retry count: a rank whose host
# submits the launch ~10 s late must stall its peers, not crash them. Tests
# shrink POLL_TIMEOUT_SHIFT (formatted into the asm when the kernel compiles).
POLL_CHECK = 4096
POLL_TIMEOUT_MANT = 11
POLL_TIMEOUT_SHIFT = 29
# int64 words of a CTA's stuck-poll record: expected tag, observed pair, stale
# pair address, clock at the first check, clock at the trap, the stale
# operand's index in the poll, the record's byte offset (CTA * 64), unused.
STUCK_WORDS = 8
STUCK_RECORD_BYTES = gl.constexpr(STUCK_WORDS * 8)  # the device copy


@gluon.jit
def launch_tag(step_ptr, layer):
    """This launch's tag: ``step * LAYER_SLOTS + layer + 1`` (wraps at 2**32).

    ``step_ptr`` points at the device step counter that a step's metadata
    kernel advances, so graph replays advance it too.
    """
    return gl.load(step_ptr) * LAYER_SLOTS + layer + 1


@gluon.jit
def put(scratch, offsets, payload, tag, mask=None):
    """Publish ``payload`` (32-bit values) at pair ``offsets`` of ``scratch``.

    Args:
        scratch: ``int64`` pointer to the mailbox buffer.
        offsets: Pair indices, any distributed layout.
        payload: ``int32``/``uint32``/``float32`` values, layout of ``offsets``.
        tag: This launch's tag.
        mask: Optional lanes to publish.
    """
    _put(scratch + offsets, payload, tag, mask, "gpu")


@gluon.jit
def _put(ptrs, payload, tag, mask, SCOPE: gl.constexpr):
    bits = payload.to(gl.int32, bitcast=True).to(gl.int64) & 0xFFFFFFFF
    pair = (tag.to(gl.int64) << 32) | bits
    gl.atomic_store(ptrs, pair, mask=mask, sem="relaxed", scope=SCOPE)


@gluon.constexpr_function
def _lane_type(num_warps):
    """``int64`` tensor type with exactly one element per thread."""
    return gl.distributed_type(gl.int64, [64 * num_warps], lane_layout(num_warps))


@gluon.constexpr_function
def _poll_constraints(groups):
    """Constraints of a poll of ``groups`` index tensors, one per operand
    group (inline_asm repeats each for every element of its group).

    Outputs: every group's pairs, one scratch VGPR pair, one SGPR pair, the
    retry count (one SGPR), the saved EXEC mask (one SGPR pair), the clock at
    the first check (one SGPR pair). Inputs: every group's pointers, the tag,
    the wave's "spin" flag. The spin flag is one value per thread, so its
    entry also names the clobbers. A retry reloads only the late pairs' lanes
    (EXEC narrowed, then restored), so EXEC is among them.
    """
    return (
        ("=&v",) * groups
        + ("=&v", "=&s", "=&s", "=&s", "=&s")
        + ("v",) * groups
        + ("s", "v,~{vcc},~{scc},~{exec}")
    )


@gluon.constexpr_function
def _poll_constraints_stuck(groups):
    """:func:`_poll_constraints` of a poll that writes a stuck-poll record:
    the record buffer's address and the CTA's record offset in it (SGPR
    pairs) come in before the spin flag, and the record's stores clobber
    EXEC."""
    return (
        ("=&v",) * groups
        + ("=&v", "=&s", "=&s", "=&s", "=&s")
        + ("v",) * groups
        + ("s", "s", "s", "v,~{vcc},~{scc},~{exec}")
    )


# Load modifiers of a poll by scope: agent scope bypasses the vL1D; system
# scope also bypasses the L2, which another GPU's stores do not update.
_POLL_CACHE = {"agent": "sc1", "system": "sc0 sc1"}


@gluon.constexpr_function
def _poll_asm_agent(outputs, inputs):
    return _poll_asm(outputs, inputs, _POLL_CACHE["agent"])


@gluon.constexpr_function
def _poll_asm_system(outputs, inputs):
    return _poll_asm(outputs, inputs, _POLL_CACHE["system"])


@gluon.constexpr_function
def _poll_asm_stuck_agent(outputs, inputs):
    return _poll_asm(outputs, inputs, _POLL_CACHE["agent"], True)


@gluon.constexpr_function
def _poll_asm_stuck_system(outputs, inputs):
    return _poll_asm(outputs, inputs, _POLL_CACHE["system"], True)


@gluon.constexpr_function
def _poll_asm(outputs, inputs, cache, stuck=False):
    """One wave-uniform retry loop over all of a thread's pairs.

    All loads issue before one ``vmcnt`` wait. While any lane's tag differs,
    the wave loads again only the pairs whose tag differs: a retry of a
    large poll then carries only the late pairs, and a pair that lands
    during the retry is seen one memory round trip later. The wave traps
    after about a minute of waiting. A wave whose ``spin`` flag is zero skips
    the loop (``pre_poll`` keeps one wave per CTA spinning).

    Every ``POLL_CHECK`` retries the wave reads the realtime clock: the first
    reading starts the wait's clock, a later one past the timeout traps (the
    64-bit compare runs on the VALU, whose 64-bit add and compare take the
    SGPR pairs whole).
    """
    groups = len(inputs) - (4 if stuck else 2)
    pairs = [p for group in outputs[:groups] for p in group]
    ptrs = [p for group in inputs[:groups] for p in group]
    (tmp,), (acc,), (retries,), (save,), (start,) = outputs[groups : groups + 5]
    (tag,), (spin,) = inputs[groups], inputs[-1]
    # Branch on SCC, not VCCZ: the hardware does not interlock a VALU write
    # of VCC with s_cbranch_vccz. Also, LLVM's hazard recognizer cannot see
    # inside inline asm.
    lines = [
        f"v_cmp_ne_u32 vcc, 0, {spin}",
        "s_cmp_eq_u64 vcc, 0",
        "s_cbranch_scc1 2f",
        f"s_mov_b32 {retries}, 0",
        f"s_mov_b64 {start}, 0",
    ]
    lines += [f"global_load_dwordx2 {p}, {a}, off {cache}" for p, a in zip(pairs, ptrs)]
    lines += ["1:", "s_waitcnt vmcnt(0)", f"s_mov_b64 {acc}, 0"]
    for p in pairs:
        lines += [
            f"v_lshrrev_b64 {tmp}, 32, {p}",
            f"v_cmp_ne_u64 vcc, {tag}, {tmp}",
            f"s_or_b64 {acc}, {acc}, vcc",
        ]
    lines += [f"s_cmp_eq_u64 {acc}, 0", "s_cbranch_scc1 2f"]
    # Only a retry counts, so a poll that succeeds at once costs nothing more,
    # and only every POLL_CHECK-th retry reads the clock.
    lines += [
        f"s_add_u32 {retries}, {retries}, 1",
        f"s_cmp_lt_u32 {retries}, {POLL_CHECK}",
        "s_cbranch_scc1 4f",
        f"s_mov_b32 {retries}, 0",
        f"s_memrealtime {acc}",
        "s_waitcnt lgkmcnt(0)",
        f"s_cmp_eq_u64 {start}, 0",
        "s_cbranch_scc0 5f",
        f"s_mov_b64 {start}, {acc}",
        "s_branch 4f",
        "5:",
        f"s_mov_b64 {save}, {POLL_TIMEOUT_MANT}",
        f"s_lshl_b64 {save}, {save}, {POLL_TIMEOUT_SHIFT}",
        f"v_mov_b64 {tmp}, {start}",
        f"v_lshl_add_u64 {tmp}, {save}, 0, {tmp}",
        f"v_cmp_lt_u64 vcc, {tmp}, {acc}",
        "s_cmp_lg_u64 vcc, 0",
        "s_cbranch_scc1 3f",
        "4:",
        "s_sleep 1",
        f"s_mov_b64 {save}, exec",
    ]
    # Load again the stale pairs only: each load runs on the lanes whose
    # pair still holds another tag. EXEC returns to the saved mask after.
    for p, a in zip(pairs, ptrs):
        lines += [
            f"v_lshrrev_b64 {tmp}, 32, {p}",
            f"v_cmp_ne_u64 vcc, {tag}, {tmp}",
            f"s_and_b64 exec, {save}, vcc",
            f"global_load_dwordx2 {p}, {a}, off {cache}",
            f"s_mov_b64 exec, {save}",
        ]
    lines += ["s_branch 1b", "3:"]
    if stuck:
        (rec,), (slot,) = inputs[groups + 1], inputs[groups + 2]
        lines += _stuck_record(
            pairs, ptrs, tmp, retries, save, tag, start, acc, rec, slot
        )
    lines += ["s_trap 2", "2:"]
    return "\n".join(lines)


@gluon.constexpr_function
def _stuck_record(pairs, ptrs, tmp, lane, mask, tag, start, now, rec, slot):
    """The lines that write a stuck poll's record (see the module docstring)
    from the first lane of the first operand whose pair is stale, then wait
    for the stores to land: only the trap branch runs them. ``now`` holds the
    clock of the check that trapped; ``slot`` is the CTA's record offset in
    bytes (the kernel scales the CTA: v_lshl_add_u64 shifts by 4 at most)."""
    lines = []
    for j, p in enumerate(pairs):
        lines += [
            f"v_lshrrev_b64 {tmp}, 32, {p}",
            f"v_cmp_ne_u64 vcc, {tag}, {tmp}",
            "s_cmp_eq_u64 vcc, 0",
            f"s_cbranch_scc0 {10 + j}f",
        ]
    lines += ["s_branch 9f"]
    for j, (p, a) in enumerate(zip(pairs, ptrs)):
        lines += [
            f"{10 + j}:",
            # The stale lane alone writes: its pair and address are its own.
            f"s_ff1_i32_b64 {lane}, vcc",
            f"s_lshl_b64 {mask}, 1, {lane}",
            f"s_mov_b64 exec, {mask}",
            f"v_mov_b64 {tmp}, {rec}",
            f"v_lshl_add_u64 {tmp}, {slot}, 0, {tmp}",
            f"global_store_dwordx2 {tmp}, {p}, off offset:8 sc0 sc1",
            f"global_store_dwordx2 {tmp}, {a}, off offset:16 sc0 sc1",
            "s_nop 1",
        ]
        for offset, value in (
            (24, start),
            (32, now),
            (40, str(j)),
            (48, slot),
            (0, tag),
        ):
            lines += [
                f"v_mov_b64 {p}, {value}",
                f"global_store_dwordx2 {tmp}, {p}, off offset:{offset} sc0 sc1",
                "s_nop 1",
            ]
        lines += ["s_waitcnt vmcnt(0)", "s_branch 9f"]
    lines += ["9:"]
    return lines


@gluon.constexpr_function
def _no_buffer(buf):
    """Whether ``buf`` is the "no buffer" ``None`` (plain or constexpr)."""
    return buf is None or (isinstance(buf, gl.constexpr) and buf.value is None)


@gluon.jit
def _trap():
    """The fail-stop of a stuck poll (``s_trap 2``), for a broken invariant."""
    gl.inline_asm("s_trap 2\ns_mov_b32 $0, 0", "=s", [], gl.int32, is_pure=False)


@gluon.jit
def _spin_all():
    lanes = gl.arange(0, 64 * gl.num_warps(), lane_layout(gl.num_warps()))
    return gl.full_like(lanes, 1)


@gluon.jit
def _payload(pairs):
    return (pairs & 0xFFFFFFFF).to(gl.int32)


@gluon.constexpr_function
def _group_end(first, count):
    return min(first + 4, count)


@gluon.constexpr_function
def _poll_types(offsets, num_warps):
    """Result types of a poll of the index tensors of tuple type
    ``offsets``: their pairs, then the poll's four scratch operands."""
    pairs = [t.with_element_ty(gl.int64) for t in offsets.types]
    return pairs + [_lane_type(num_warps), gl.int64, gl.int32, gl.int64, gl.int64]


@gluon.jit
def _poll_n(scratch, offsets, tag, spin, SCOPE: gl.constexpr, stuck):
    """Spin until every pair at the index tensors ``offsets`` (a tuple, any
    layouts) carries ``tag``, in one round trip. Return the tuple of their
    pairs. Waves whose ``spin`` is zero skip the loop. ``stuck``: None, or
    the stuck-poll record buffer (see the module docstring)."""
    ptrs = ()
    for k in gl.static_range(len(offsets)):
        ptrs += (scratch + offsets[k],)
    # The pair's high dword holds the tag's 32 bits and the poll shifts it
    # down zero-extended, so compare with the zero-extended tag. A
    # sign-extended tag never matches once the tag wraps past bit 31.
    gl.static_assert(SCOPE == "agent" or SCOPE == "system", "scope: agent or system")
    # asm: a poll is a loop of uncached loads that retries only the late
    # pairs' lanes until every tag matches; Gluon has no volatile load, and
    # a load in a loop is hoisted or merged.
    if stuck is None:
        out = gl.inline_asm(
            _poll_asm_system if SCOPE == "system" else _poll_asm_agent,
            _poll_constraints(len(offsets)),
            ptrs + (tag.to(gl.int64) & 0xFFFFFFFF, spin),
            _poll_types(offsets.type, gl.num_warps()),
        )
    else:
        out = gl.inline_asm(
            _poll_asm_stuck_system if SCOPE == "system" else _poll_asm_stuck_agent,
            _poll_constraints_stuck(len(offsets)),
            ptrs
            + (
                tag.to(gl.int64) & 0xFFFFFFFF,
                stuck.to(gl.int64, bitcast=True),
                # The CTA's record, in bytes from the buffer's start.
                gl.program_id(0).to(gl.int64) * STUCK_RECORD_BYTES,
                spin,
            ),
            _poll_types(offsets.type, gl.num_warps()),
        )
    pairs = ()
    for k in gl.static_range(len(offsets)):
        pairs += (out[k],)
    return pairs


@gluon.jit
def poll(scratch, offsets, tag):
    """Spin until every pair at ``offsets`` carries ``tag``. Return the payloads.

    Every thread polls the pairs that it holds in ``offsets``'s layout. Keep a
    thread's pairs few (about 12) to bound live registers. For few pairs per
    CTA, a direct poll is faster than :func:`pre_poll` first (one round trip
    instead of two). For many pairs, pre-poll to throttle the load flood.

    Returns:
        ``int32`` payloads in ``offsets``'s layout.
    """
    return _payload(_poll_n(scratch, (offsets,), tag, _spin_all(), "agent", None)[0])


@gluon.jit
def poll_waves(scratch, offsets, tag, spin):
    """:func:`poll` by the waves whose ``spin`` (one value per thread, in
    :func:`lane_layout`) is non-zero. The other waves skip the loop, and
    their payloads are undefined."""
    return _payload(_poll_n(scratch, (offsets,), tag, spin, "agent", None)[0])


@gluon.jit
def poll_many_waves(scratch, offsets, tag, spin):
    """:func:`poll_waves` of a tuple of at most four index tensors (any
    layouts) in one round trip. Returns the tuple of their payloads."""
    return _poll_many_waves(scratch, offsets, tag, spin, "agent", None)


@gluon.jit
def _poll_many_waves(scratch, offsets, tag, spin, SCOPE: gl.constexpr, stuck):
    gl.static_assert(len(offsets) <= 4, "one round trip polls four tensors")
    pairs = _poll_n(scratch, offsets, tag, spin, SCOPE, stuck)
    payloads = ()
    for k in gl.static_range(len(offsets)):
        payloads += (_payload(pairs[k]),)
    return payloads


@gluon.jit
def poll_many(scratch, offsets, tag):
    """:func:`poll` of a tuple of index tensors (any layouts), four per round
    trip. Returns the tuple of their payloads."""
    return _poll_many(scratch, offsets, tag, "agent", None)


@gluon.jit
def _poll_many(scratch, offsets, tag, SCOPE: gl.constexpr, stuck):
    payloads = ()
    for first in gl.static_range(0, len(offsets), 4):
        group = ()
        for k in gl.static_range(first, _group_end(first, len(offsets))):
            group += (offsets[k],)
        pairs = _poll_n(scratch, group, tag, _spin_all(), SCOPE, stuck)
        for k in gl.static_range(len(group)):
            payloads += (_payload(pairs[k]),)
    return payloads


@gluon.constexpr_function
def _per_row(regions, rows):
    """``regions`` repeated once per row."""
    return tuple(regions) * rows


@gluon.jit
def _row_indices(indices, ROWS: gl.constexpr, plan: gl.constexpr):
    """``indices`` shifted to each row, rows major."""
    out = ()
    for r in gl.static_range(ROWS):
        for k in gl.static_range(len(indices)):
            out += (indices[k] + r * schedule.row_pairs(plan),)
    return out


@gluon.jit
def pre_poll(scratch, hints, tag):
    """Wave 0 spins on ``hints`` (one pair per producer), then the CTA waits at a
    barrier.

    A throttle only: :func:`poll` still checks every payload pair afterwards.
    Every thread must call it.

    Args:
        hints: Pair indices in the one-element-per-thread layout of
            :func:`lane_layout`. The function ignores the lanes of waves
            other than 0.
    """
    _pre_poll(scratch, hints, tag, "agent", None)


@gluon.jit
def _pre_poll(scratch, hints, tag, SCOPE: gl.constexpr, stuck):
    lanes = gl.arange(0, 64 * gl.num_warps(), lane_layout(gl.num_warps()))
    _poll_n(scratch, (hints,), tag, (lanes < 64).to(gl.int32), SCOPE, stuck)
    gl.barrier()


@gluon.constexpr_function
def lane_layout(num_warps):
    """One-element-per-thread 1-D layout: thread ``i`` holds element ``i``."""
    return gl.BlockedLayout([1], [64], [num_warps], [0])


@aggregate
class Peers:
    """A rank's view of the other ranks of a plan of ``world`` > 1.

    The mailbox buffer sits on Iris's symmetric heap, so a region has the
    same offset on every rank. ``deltas`` holds, for every rank, its heap
    base minus this rank's: :meth:`Peers.load` reads the bases once per
    launch (``IrisGroup.bases``), so a put to another rank needs no load.
    ``parity`` (0 or 1) selects this launch's copy of every peer region: it
    must alternate from one launch to the next on every rank.
    """

    deltas: gl.tensor
    rank: gl.constexpr
    parity: gl.tensor

    @gluon.constexpr_function
    def __init__(self, deltas, rank, parity):
        self.deltas = deltas
        self.rank = gl.constexpr(rank)
        self.parity = parity

    @gluon.jit
    def translate(self, ptrs, peer):
        """``ptrs`` (on this rank's heap) as seen on rank ``peer``'s heap."""
        i = gl.arange(0, self.deltas.shape[0], self.deltas.type.layout)
        delta = gl.sum(gl.where(i == peer, self.deltas, 0), axis=0)
        return (ptrs.to(gl.int64) + delta).to(ptrs.dtype)

    @gluon.jit
    def translate_lanes(self, ptrs, peers):
        """``ptrs`` (on this rank's heap), each as seen on the rank in
        ``peers`` (a tensor like ``ptrs``; values past the group map to the
        last rank)."""
        i = gl.arange(0, self.deltas.shape[0], self.deltas.type.layout)
        delta = gl.zeros(ptrs.shape, gl.int64, ptrs.type.layout)
        for p in gl.static_range(self.deltas.shape[0]):
            d = gl.sum(gl.where(i == p, self.deltas, 0), axis=0)
            delta = gl.where(gl.minimum(peers, self.deltas.shape[0] - 1) == p, d, delta)
        return (ptrs.to(gl.int64) + delta).to(ptrs.dtype)


@gluon.jit
def load_peers(bases, rank: gl.constexpr, parity, WORLD: gl.constexpr):
    """:class:`Peers` from the ``int64 [WORLD]`` heap bases of the group."""
    layout: gl.constexpr = gl.BlockedLayout([WORLD], [64], [gl.num_warps()], [0])
    b = gl.load(bases + gl.arange(0, WORLD, layout))
    return Peers(b - gl.load(bases + rank), rank, parity)


@gluon.constexpr_function
def _scope(plan, region):
    return "system" if plan.plan.is_peer(region) else "agent"


@gluon.constexpr_function
def _scopes(plan, regions):
    peer = any(plan.plan.is_peer(region) for region in regions)
    return "system" if peer else "agent"


# ``plan`` below is a ``PlanKey`` (see plan.py).
@aggregate
class Mailbox:
    """A launch's mailbox buffer, tag, plan, optional LDS stamp table
    (``trace.allocate``), for a plan of ``world`` > 1 its :class:`Peers`, and
    an optional stuck-poll record buffer (see the module docstring).

    A caller addresses a region by name, with pair indices relative to the
    region's row 0. :meth:`row` gives the view of another row, so a
    stage written for one row serves any row through that view.
    """

    scratch: gl.tensor
    tag: gl.tensor
    plan: gl.constexpr
    trace: gl.shared_memory_descriptor | gl.constexpr
    peers: Peers | gl.constexpr
    stuck: gl.tensor | gl.constexpr

    @gluon.constexpr_function
    def __init__(self, scratch, tag, plan, trace_table, peers=None, stuck=None):
        self.scratch = scratch
        self.tag = tag
        self.plan = gl.constexpr(plan)
        self.trace = gl.constexpr(None) if trace.disabled(trace_table) else trace_table
        self.peers = gl.constexpr(None) if peers is None else peers
        self.stuck = gl.constexpr(None) if _no_buffer(stuck) else stuck

    @gluon.jit
    def row_offset(self, row):
        """Pair offset of row ``row``'s copy of every region (any
        tensor of rows). Rows past the launch's rows are padding: they map to
        its last row, which a poll may read. A put must mask them."""
        return gl.minimum(row, schedule.rows(self.plan) - 1) * schedule.row_pairs(
            self.plan
        )

    @gluon.jit
    def row(self, row):
        """This mailbox as seen by row ``row``: every region index
        addresses that row's copy of the region."""
        return Mailbox(
            self.scratch + row * schedule.row_pairs(self.plan),
            self.tag,
            self.plan,
            self.trace,
            self.peers,
            self.stuck,
        )

    @gluon.jit
    def check_parity(self, parity):
        """Fail-stop (``s_trap 2``) unless ``parity`` differs from the last
        launch's on this rank: the peer regions' copy must alternate from one
        launch to the next (:class:`Peers`). The plan's one-pair region
        ``parity`` keeps it (plus one; 0: no launch yet). Call it from CTA 0
        before any put. Serving may let a regular layer's collectives order
        two same-parity launches, so only tests and benches turn it on."""
        at = self.scratch + schedule.region(self.plan, "parity")
        prev = gl.load(at)
        # Every wave reads the last launch's value before any wave writes
        # this launch's.
        gl.barrier()
        if prev == parity + 1:
            _trap()
        # prev * 0: an int64 value whether ``parity`` is a tensor or a
        # specialized constant.
        gl.store(at, prev * 0 + parity + 1)

    @gluon.jit
    def stamp(self, stage: gl.constexpr, point: gl.constexpr):
        """Trace ``point`` of ``stage`` for this CTA (see ``trace.py``)."""
        trace.stamp(self.trace, schedule.stage_event(self.plan, stage, point))

    @gluon.jit
    def _base(self, region: gl.constexpr):
        """Pair offset of ``region`` in this launch: its copy by parity for a
        peer region."""
        if schedule.is_peer(self.plan, region):
            return schedule.region(self.plan, region) + self.peers.parity * (
                schedule.peer_copy(self.plan, region)
            )
        else:
            return schedule.region(self.plan, region)

    @gluon.jit
    def put(self, region: gl.constexpr, index, payload, mask=None):
        """Publish ``payload`` at ``index`` of this rank's copy of ``region``."""
        _put(
            self.scratch + self._base(region) + index,
            payload,
            self.tag,
            mask,
            "sys" if schedule.is_peer(self.plan, region) else "gpu",
        )

    @gluon.jit
    def put_peer(self, region: gl.constexpr, peer, index, payload, mask=None):
        """Publish ``payload`` at ``index`` of rank ``peer``'s copy of peer
        region ``region`` (a system-scope store; ``peer`` may be this rank)."""
        gl.static_assert(
            schedule.is_peer(self.plan, region), "put_peer needs a peer region"
        )
        ptrs = self.peers.translate(self.scratch + self._base(region) + index, peer)
        _put(ptrs, payload, self.tag, mask, "sys")

    @gluon.jit
    def _words(self, region: gl.constexpr):
        """``int32`` base pointer of this rank's copy of ``region``'s 32-bit
        words (a region of ``n`` pairs holds ``2 n`` words)."""
        return (self.scratch + self._base(region)).to(
            gl.pointer_type(gl.int32), bitcast=True
        )

    @gluon.jit
    def store_words(self, region: gl.constexpr, index, values, mask=None):
        """Store 32-bit ``values`` at word ``index`` of ``region`` without
        tags, for a producer that writes many values and one flag
        (:meth:`raise_flag`) instead of a tagged pair each.

        The stores are vector buffer stores that write through to memory
        (``sc0 sc1``), so a consumer on another XCD reads them past its L2
        after it sees the flag (:meth:`load_words`). Word indices count
        from row 0's copy: row ``r``'s copy starts at word
        ``2 * row_offset(r)``."""
        gl.static_assert(
            not schedule.is_peer(self.plan, region),
            "store_words is local: use store_words_peer",
        )
        gl.amd.cdna4.buffer_store(
            values.to(gl.int32, bitcast=True),
            self._words(region),
            index,
            mask=mask,
            cache=".wt",
        )

    @gluon.jit
    def raise_flag(self, region: gl.constexpr, index):
        """Publish the CTA's earlier :meth:`store_words` with the flag pair
        ``index`` of ``region``: every wave waits for its stores to land (on
        gfx9 ``vmcnt`` counts stores), the CTA meets at a barrier, then one
        thread puts the pair. A consumer polls the flag, then loads the
        words."""
        # Wait for this wave's stores before the CTA publishes the flag.
        _extern.release_drain()
        gl.barrier()
        lanes = gl.arange(0, 64 * gl.num_warps(), lane_layout(gl.num_warps()))
        _put(
            self.scratch + self._base(region) + index + lanes * 0,
            gl.full(lanes.shape, 1, gl.int32, lanes.type.layout),
            self.tag,
            lanes == 0,
            "gpu",
        )

    @gluon.jit
    def store_words_peer(self, region: gl.constexpr, peer, index, values, mask=None):
        """:meth:`store_words` into rank ``peer``'s copy of peer region
        ``region`` (``peer`` may be this rank), published with
        :meth:`raise_flag_peers`. The write-through stores cross the links
        to the peer's memory; ``vmcnt`` counts them until they land there."""
        gl.static_assert(
            schedule.is_peer(self.plan, region), "store_words_peer needs a peer region"
        )
        base = self.peers.translate(self._words(region), peer)
        gl.amd.cdna4.buffer_store(
            values.to(gl.int32, bitcast=True), base, index, mask=mask, cache=".wt"
        )

    @gluon.jit
    def raise_flag_peers(self, region: gl.constexpr, index):
        """Publish the CTA's earlier :meth:`store_words_peer` with the flag
        pair ``index`` of every rank's copy of peer region ``region``: every
        wave waits for its stores to land (``vmcnt`` counts stores on gfx9,
        remote ones until the peer's memory acknowledges them), the CTA
        meets at a barrier, then thread ``p`` puts rank ``p``'s flag at
        system scope."""
        gl.static_assert(
            schedule.is_peer(self.plan, region), "raise_flag_peers needs a peer region"
        )
        # Wait for this wave's peer stores before the CTA publishes the flags.
        _extern.release_drain()
        gl.barrier()
        W: gl.constexpr = schedule.world(self.plan)
        lanes = gl.arange(0, 64 * gl.num_warps(), lane_layout(gl.num_warps()))
        ptrs = self.peers.translate_lanes(
            self.scratch + self._base(region) + index + lanes * 0, lanes
        )
        _put(
            ptrs,
            gl.full(lanes.shape, 1, gl.int32, lanes.type.layout),
            self.tag,
            lanes < W,
            "sys",
        )

    @gluon.jit
    def load_words(self, region: gl.constexpr, index):
        """Load the 32-bit words at ``index`` of ``region`` (int32) that a
        producer published with :meth:`raise_flag`, past the caches
        (``sc0 sc1``). Call it after the CTA polled the flags and met at a
        barrier, so no load issues before the poll returns."""
        return gl.amd.cdna4.buffer_load(self._words(region), index, cache=".cv")

    @gluon.jit
    def poll(self, region: gl.constexpr, index):
        return _payload(
            _poll_n(
                self.scratch + self._base(region),
                (index,),
                self.tag,
                _spin_all(),
                _scope(self.plan, region),
                self.stuck,
            )[0]
        )

    @gluon.jit
    def poll_waves(self, region: gl.constexpr, index, spin):
        return _payload(
            _poll_n(
                self.scratch + self._base(region),
                (index,),
                self.tag,
                spin,
                _scope(self.plan, region),
                self.stuck,
            )[0]
        )

    @gluon.jit
    def poll_many_waves(self, regions: gl.constexpr, indices, spin):
        """:meth:`poll_many` of at most four index tensors, by the waves
        whose ``spin`` is non-zero (:func:`poll_many_waves`)."""
        offsets = ()
        for k in gl.static_range(len(regions)):
            offsets += (self.pairs(regions[k], indices[k]),)
        return _poll_many_waves(
            self.scratch,
            offsets,
            self.tag,
            spin,
            _scopes(self.plan, regions),
            self.stuck,
        )

    @gluon.jit
    def poll_many(self, regions: gl.constexpr, indices):
        """Poll ``indices[k]`` (pairs from region ``regions[k]``'s row 0) for
        every ``k``, four index tensors per round trip. Returns the tuple of
        payloads. A poll that includes a peer region is system-scope."""
        offsets = ()
        for k in gl.static_range(len(regions)):
            offsets += (self.pairs(regions[k], indices[k]),)
        return _poll_many(
            self.scratch, offsets, self.tag, _scopes(self.plan, regions), self.stuck
        )

    @gluon.jit
    def pairs(self, region: gl.constexpr, index):
        """Mailbox pair offsets of pairs ``index`` of ``region``, for
        :meth:`poll_pairs` (one index tensor may then mix local regions)."""
        return index + self._base(region)

    @gluon.jit
    def poll_pairs(self, offsets):
        """Poll a tuple of tensors of :meth:`pairs` offsets of local regions,
        four per round trip. Poll peer regions with :meth:`poll_many`."""
        return _poll_many(self.scratch, offsets, self.tag, "agent", self.stuck)

    @gluon.jit
    def poll_rows(self, regions: gl.constexpr, indices, ROWS: gl.constexpr):
        """Poll ``indices[k]`` of region ``regions[k]`` in rows
        ``0 .. ROWS - 1`` (see :meth:`poll_many`).

        Args:
            regions: A tuple of region names.
            indices: A tuple of index tensors, one per region.

        Returns:
            ``payloads[row][k]``: a tuple per row of the regions' payloads.
        """
        polled = self.poll_many(
            _per_row(regions, ROWS), _row_indices(indices, ROWS, self.plan)
        )
        payloads = ()
        for r in gl.static_range(ROWS):
            payloads += (polled[r * len(regions) : (r + 1) * len(regions)],)
        return payloads

    @gluon.jit
    def pre_poll_spread(self, region: gl.constexpr, N: gl.constexpr):
        """:meth:`pre_poll` on up to 64 of a region's first ``N`` pairs,
        spread evenly. Use it for a region whose producers each end with one
        of those pairs, such as MXFP4 exponents."""
        lanes = gl.arange(0, 64 * gl.num_warps(), lane_layout(gl.num_warps()))
        self.pre_poll(region, gl.minimum(lanes * N // 64, N - 1))

    @gluon.jit
    def pre_poll(self, region: gl.constexpr, hints):
        _pre_poll(
            self.scratch + self._base(region),
            hints,
            self.tag,
            _scope(self.plan, region),
            self.stuck,
        )


__all__ = [
    "LAYER_SLOTS",
    "Mailbox",
    "Peers",
    "load_peers",
    "lane_layout",
    "launch_tag",
    "poll",
    "poll_many",
    "poll_waves",
    "pre_poll",
    "put",
]
