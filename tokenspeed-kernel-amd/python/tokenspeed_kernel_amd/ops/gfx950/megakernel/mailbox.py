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

"""Tagged-pair mailboxes: how megakernel CTAs hand values to each other.

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

Fail-stop: a poll gives up after ``MAX_RETRIES`` retries (at least about 8 s;
a launch takes about 100 us) and executes ``s_trap 2``. The runtime then reports
a GPU exception and the process stops. A deadlock or a missing producer thus
stops the server with an error instead of hanging it.
"""

from __future__ import annotations

from tokenspeed_kernel_amd._triton import aggregate, gl, gluon
from tokenspeed_kernel_amd.ops.gfx950.megakernel import schedule, trace
from tokenspeed_kernel_amd.ops.gfx950.megakernel.plan import LAYER_SLOTS as _SLOTS

LAYER_SLOTS = gl.constexpr(_SLOTS)  # the device copy of plan.LAYER_SLOTS
# Retries before a poll traps. A retry (an sc1 load round trip and an s_sleep)
# takes at least about 0.12 us on MI350X, so this is at least about 8 s.
MAX_RETRIES = 1 << 26


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
    bits = payload.to(gl.int32, bitcast=True).to(gl.int64) & 0xFFFFFFFF
    pair = (tag.to(gl.int64) << 32) | bits
    gl.atomic_store(scratch + offsets, pair, mask=mask, sem="relaxed", scope="gpu")


@gluon.constexpr_function
def _lane_type(num_warps):
    """``int64`` tensor type with exactly one element per thread."""
    return gl.distributed_type(gl.int64, [64 * num_warps], lane_layout(num_warps))


@gluon.constexpr_function
def _poll_constraints(groups):
    """Constraints of a poll of ``groups`` index tensors, one per operand
    group (inline_asm repeats each for every element of its group).

    Outputs: every group's pairs, one scratch VGPR pair, one SGPR pair, the
    retry count (one SGPR), the saved EXEC mask (one SGPR pair). Inputs:
    every group's pointers, the tag, the wave's "spin" flag. The spin flag is
    one value per thread, so its entry also names the clobbers.
    """
    return (
        ("=&v",) * groups
        + ("=&v", "=&s", "=&s", "=&s")
        + ("v",) * groups
        + ("s", "v,~{vcc},~{scc}")
    )


@gluon.constexpr_function
def _poll_asm(outputs, inputs):
    """One wave-uniform retry loop over all of a thread's pairs.

    All loads issue before one ``vmcnt`` wait. While any lane's tag differs,
    the wave loads again only the pairs whose tag differs: a retry of a
    large poll then carries only the late pairs, and a pair that lands
    during the retry is seen one memory round trip later. The wave traps
    after ``MAX_RETRIES`` retries. A wave whose ``spin`` flag is zero skips
    the loop (``pre_poll`` keeps one wave per CTA spinning).
    """
    groups = len(inputs) - 2
    pairs = [p for group in outputs[:groups] for p in group]
    ptrs = [p for group in inputs[:groups] for p in group]
    (tmp,), (acc,), (retries,), (save,) = outputs[groups : groups + 4]
    (tag,), (spin,) = inputs[groups], inputs[groups + 1]
    # Branch on SCC, not VCCZ: the hardware does not interlock a VALU write
    # of VCC with s_cbranch_vccz. Also, LLVM's hazard recognizer cannot see
    # inside inline asm.
    lines = [
        f"v_cmp_ne_u32 vcc, 0, {spin}",
        "s_cmp_eq_u64 vcc, 0",
        "s_cbranch_scc1 2f",
        f"s_mov_b32 {retries}, 0",
    ]
    lines += [f"global_load_dwordx2 {p}, {a}, off sc1" for p, a in zip(pairs, ptrs)]
    lines += ["1:", "s_waitcnt vmcnt(0)", f"s_mov_b64 {acc}, 0"]
    for p in pairs:
        lines += [
            f"v_lshrrev_b64 {tmp}, 32, {p}",
            f"v_cmp_ne_u64 vcc, {tag}, {tmp}",
            f"s_or_b64 {acc}, {acc}, vcc",
        ]
    lines += [f"s_cmp_eq_u64 {acc}, 0", "s_cbranch_scc1 2f"]
    # Only a retry counts, so a poll that succeeds at once costs nothing more.
    lines += [
        f"s_add_u32 {retries}, {retries}, 1",
        f"s_cmp_gt_u32 {retries}, {MAX_RETRIES}",
        "s_cbranch_scc1 3f",
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
            f"global_load_dwordx2 {p}, {a}, off sc1",
            f"s_mov_b64 exec, {save}",
        ]
    lines += ["s_branch 1b", "3:", "s_trap 2", "2:"]
    return "\n".join(lines)


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
    return pairs + [_lane_type(num_warps), gl.int64, gl.int32, gl.int64]


@gluon.jit
def _poll_n(scratch, offsets, tag, spin):
    """Spin until every pair at the index tensors ``offsets`` (a tuple, any
    layouts) carries ``tag``, in one round trip. Return the tuple of their
    pairs. Waves whose ``spin`` is zero skip the loop."""
    ptrs = ()
    for k in gl.static_range(len(offsets)):
        ptrs += (scratch + offsets[k],)
    out = gl.inline_asm(
        _poll_asm,
        _poll_constraints(len(offsets)),
        ptrs + (tag.to(gl.int64), spin),
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
    return _payload(_poll_n(scratch, (offsets,), tag, _spin_all())[0])


@gluon.jit
def poll_waves(scratch, offsets, tag, spin):
    """:func:`poll` by the waves whose ``spin`` (one value per thread, in
    :func:`lane_layout`) is non-zero. The other waves skip the loop, and
    their payloads are undefined."""
    return _payload(_poll_n(scratch, (offsets,), tag, spin)[0])


@gluon.jit
def poll_many(scratch, offsets, tag):
    """:func:`poll` of a tuple of index tensors (any layouts), four per round
    trip. Returns the tuple of their payloads."""
    payloads = ()
    for first in gl.static_range(0, len(offsets), 4):
        group = ()
        for k in gl.static_range(first, _group_end(first, len(offsets))):
            group += (offsets[k],)
        pairs = _poll_n(scratch, group, tag, _spin_all())
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
    lanes = gl.arange(0, 64 * gl.num_warps(), lane_layout(gl.num_warps()))
    _poll_n(scratch, (hints,), tag, (lanes < 64).to(gl.int32))
    gl.barrier()


@gluon.constexpr_function
def lane_layout(num_warps):
    """One-element-per-thread 1-D layout: thread ``i`` holds element ``i``."""
    return gl.BlockedLayout([1], [64], [num_warps], [0])


# ``plan`` below is a ``PlanKey`` (see plan.py).
@aggregate
class Mailbox:
    """A launch's mailbox buffer, tag, plan and optional LDS stamp table
    (``trace.allocate``).

    A caller addresses a region by name, with pair indices relative to the
    region's row 0. :meth:`row` gives the view of another row, so a
    stage written for one row serves any row through that view.
    """

    scratch: gl.tensor
    tag: gl.tensor
    plan: gl.constexpr
    trace: gl.shared_memory_descriptor | gl.constexpr

    @gluon.constexpr_function
    def __init__(self, scratch, tag, plan, trace_table):
        self.scratch = scratch
        self.tag = tag
        self.plan = gl.constexpr(plan)
        self.trace = gl.constexpr(None) if trace.disabled(trace_table) else trace_table

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
        )

    @gluon.jit
    def stamp(self, stage: gl.constexpr, point: gl.constexpr):
        """Trace ``point`` of ``stage`` for this CTA (see ``trace.py``)."""
        trace.stamp(self.trace, schedule.stage_event(self.plan, stage, point))

    @gluon.jit
    def put(self, region: gl.constexpr, index, payload, mask=None):
        put(
            self.scratch + schedule.region(self.plan, region),
            index,
            payload,
            self.tag,
            mask,
        )

    @gluon.jit
    def poll(self, region: gl.constexpr, index):
        return poll(self.scratch + schedule.region(self.plan, region), index, self.tag)

    @gluon.jit
    def poll_waves(self, region: gl.constexpr, index, spin):
        return poll_waves(
            self.scratch + schedule.region(self.plan, region), index, self.tag, spin
        )

    @gluon.jit
    def poll_many(self, regions: gl.constexpr, indices):
        """Poll ``indices[k]`` (pairs from region ``regions[k]``'s row 0) for
        every ``k``, four index tensors per round trip. Returns the tuple of
        payloads."""
        offsets = ()
        for k in gl.static_range(len(regions)):
            offsets += (self.pairs(regions[k], indices[k]),)
        return poll_many(self.scratch, offsets, self.tag)

    @gluon.jit
    def pairs(self, region: gl.constexpr, index):
        """Mailbox pair offsets of pairs ``index`` of ``region``, for
        :meth:`poll_pairs` (one index tensor may then mix regions)."""
        return index + schedule.region(self.plan, region)

    @gluon.jit
    def poll_pairs(self, offsets):
        """Poll a tuple of tensors of :meth:`pairs` offsets, four per round
        trip."""
        return poll_many(self.scratch, offsets, self.tag)

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
        pre_poll(self.scratch + schedule.region(self.plan, region), hints, self.tag)


__all__ = [
    "LAYER_SLOTS",
    "Mailbox",
    "lane_layout",
    "launch_tag",
    "poll",
    "poll_many",
    "poll_waves",
    "pre_poll",
    "put",
]
