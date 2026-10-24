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

"""Validate LongCat phases and assign tickets, warp groups and shared memory.

Routine calls own their publications. Spare warps prefetch weights and mirror
cluster barriers. The planner checks dependency counts, resource limits and
SMEM lifetimes before the CuTe engine is specialized.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

# The engine's own named barrier: every phase boundary syncs the CTA on it,
# so a routine that uses the same id would break the phase sequence.
SEGMENT_BARRIER = 11
# A CTA is always twelve warps: the routine group takes the first
# ``routine_warps`` of them and the engine group the rest.
WARPS = 12
BLOCK = 32 * WARPS
# Grain of the SMEM arena, and what TMA and the MMA descriptors ask of any
# block in it.
_ALIGN = 128


class PlanError(ValueError):
    """The program is inconsistent; the message names the offending item."""


@dataclass(frozen=True)
class EngineConfig:
    grid: int = 148
    cluster: int = 2
    smem_limit: int = 232448  # SM103 dynamic SMEM per CTA
    tmem_cols: int = 512


@dataclass(frozen=True)
class TicketDecl:
    """Monotonic ticket words with the same arrival count per generation.

    Routines publish; consumers wait for gen*producers_per_gen.
    """

    name: str
    width: int = 1


@dataclass(frozen=True)
class RoutineDecl:
    """What the planner needs to know about a device routine."""

    name: str
    # Routine-group thread count; 384 uses all twelve CTA warps.
    threads: int
    smem_bytes: int
    # Device routine for one tile, receiving the engine's _CallContext.
    invoke: Callable
    # SMEM alignment in bytes. Whole SW128x8 row atoms require 1024;
    # declaring only 128 can produce incorrect descriptor addresses.
    smem_align: int = _ALIGN
    # ``cluster_arrive/wait`` pairs the routine executes per tile; every
    # thread of the physical cluster must join them, so the engine group
    # mirrors them whenever the routines leave one (``routine_warps < 12``).
    cluster_barriers: int = 0
    ctas_per_tile: int = 1  # 2 for pair-coupled (split-K) tiles
    tmem_cols: int = 0
    named_barriers: tuple[int, ...] = ()
    # Concurrent HBM streams per CTA while running the routine.
    streams: int = 1
    # Host-jit operand preparation, returned as ctx.prepared kernel parameters.
    # TMA descriptors must be built here; plain cp.async.bulk needs none.
    host_prepare: Optional[Callable] = None
    # Owns ctx.gate(): prefetch weights before acquiring consumed inputs.
    # Required in issue_before_gate phases when sharing CTAs with engine warps.
    gates_itself: bool = False


@dataclass(frozen=True)
class RoutineCall:
    routine: RoutineDecl
    # Calls sharing a name share one offset and the maximum declared size,
    # live from first to last use. Other names may alias disjoint lifetimes.
    smem: str
    # Tiles per generation on [cta_offset, cta_offset+tiles*ctas_per_tile).
    # Disjoint CTA ranges run concurrently; overlapping calls run in order.
    tiles: int
    cta_offset: int = 0
    # (ticket, arrivals per generation) totals for this whole call, not per
    # tile: publications may vary by CTA. Routines publish; calls choose wiring.
    produces: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True)
class Prefetch:
    """Issue-only L2 warming on engine warps for later phases or a successor.

    Split total_bytes across role CTAs; each warms its own slice without waiting.
    """

    source: str  # workspace field to warm, from its first byte
    total_bytes: int
    # Issue on every CTA warp before the phase gate, for independent weights.
    before_gate: bool = False


@dataclass(frozen=True)
class TicketWait:
    ticket: str


@dataclass(frozen=True)
class Phase:
    name: str
    consumes: tuple[TicketWait, ...] = ()
    # Calls hosted on the routine group; their thread count defines the split.
    routines: tuple[RoutineCall, ...] = ()
    # L2 prefetch work for the engine warps.
    prefetch: tuple[Prefetch, ...] = ()
    # gates_itself routines prefetch into SMEM before acquiring consumes;
    # other routines get their gate immediately before invocation. Keep SMEM
    # disjoint from cluster-coupled predecessors whose DSMEM reads may be live.
    # A ticket-free first phase may start under PDL before its predecessor;
    # its routine or the engine must wait before consuming predecessor data.
    issue_before_gate: bool = False


@dataclass(frozen=True)
class Role:
    name: str
    ctas: int
    phases: tuple[Phase, ...]


@dataclass(frozen=True)
class Program:
    name: str
    config: EngineConfig
    tickets: tuple[TicketDecl, ...]
    roles: tuple[Role, ...]
    # Every phase name of every role, in time order; each role's phase list
    # must be a subsequence.  SMEM lifetimes are measured on this axis.
    phase_order: tuple[str, ...]
    # Operands the kernel takes, in the order the workspace supplies them.
    # The engine passes each as a raw pointer; the routine adapters build the
    # tensors, since the layout is part of a routine's convention.
    operands: tuple[str, ...] = ()
    # Concurrent HBM streams a phase tolerates GPU-wide, by phase name.  It
    # belongs to the time slot rather than to one role's program for it:
    # sparse MLA tolerates none, whoever is running.  Unlisted = unchecked.
    stream_budgets: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True)
class WarpSegment:
    """CTA run sharing routine_warps; remaining warps form the engine group.

    Engine warps mirror barriers, prefetch and host MoE. Idle runs lend all twelve.
    """

    lo: int
    span: int
    routine_warps: int


@dataclass(frozen=True)
class Plan:
    program: Program
    # ticket -> (word offset, width).
    ticket_offsets: dict[str, tuple[int, int]]
    # counter -> arrivals per generation (the consumer threshold multiplier).
    producers_per_gen: dict[str, int]
    # role -> SMEM name -> byte offset.
    smem_offsets: dict[str, dict[str, int]]
    smem_bytes: dict[str, int]
    # SMEM name -> the alignment its offset was placed on.
    smem_align: dict[str, int]
    # (role, phase) -> cluster barriers the engine warps mirror.
    mirrors: dict[tuple[str, str], int]
    # (role, phase) -> the role's CTAs cut into runs sharing a warp split,
    # covering the whole role.
    warp_segments: dict[tuple[str, str], tuple[WarpSegment, ...]]
    # (role,phase) -> (word offset,multiplier,subtrahend), with validated
    # threshold gen*multiplier-subtrahend.
    waits: dict[tuple[str, str], tuple[tuple[int, int, int], ...]]

    @property
    def ticket_words(self) -> int:
        return max(off + width for off, width in self.ticket_offsets.values())

    @property
    def arena_bytes(self) -> int:
        return max(
            (
                off + self.smem_bytes[name]
                for offsets in self.smem_offsets.values()
                for name, off in offsets.items()
            ),
            default=0,
        )

    @property
    def arena_align(self) -> int:
        """Arena-base alignment: the maximum alignment of its SMEM blocks."""
        return max(self.smem_align.values(), default=_ALIGN)


def _align(value: int, multiple: int = _ALIGN) -> int:
    return (value + multiple - 1) // multiple * multiple


@dataclass
class _Tally:
    """What walking the roles and phases accumulates."""

    producers: dict[str, int]
    streams: dict[str, int]
    consumed: set[str] = field(default_factory=set)
    # SMEM name -> bytes, and role -> name -> the phases it is live for.
    smem_need: dict[str, int] = field(default_factory=dict)
    # SMEM name -> the strictest alignment any routine placed there asks for.
    smem_align: dict[str, int] = field(default_factory=dict)
    smem_live: dict[str, dict[str, set[int]]] = field(default_factory=dict)
    # role -> SMEM name -> the CTAs of the role that touch it, as one
    # cluster-aligned interval.  Two blocks whose CTAs never overlap may share
    # an offset however long they are both live, because no CTA holds both.
    smem_ctas: dict[str, dict[str, tuple[int, int]]] = field(default_factory=dict)
    mirrors: dict[tuple[str, str], int] = field(default_factory=dict)
    warp_segments: dict[tuple[str, str], tuple] = field(default_factory=dict)


def _check_launch(program: Program) -> None:
    cfg = program.config
    total = sum(role.ctas for role in program.roles)
    if total != cfg.grid:
        raise PlanError(f"{program.name}: roles cover {total} CTAs, grid is {cfg.grid}")
    for role in program.roles:
        if role.ctas % cfg.cluster:
            raise PlanError(
                f"{program.name}: role {role.name} has {role.ctas} CTAs, "
                f"not a multiple of cluster {cfg.cluster}"
            )


def _phase_index(program: Program) -> dict[str, int]:
    """Position of each phase on the time axis; roles are subsequences of it."""
    order = {name: i for i, name in enumerate(program.phase_order)}
    if len(order) != len(program.phase_order):
        raise PlanError(f"{program.name}: duplicate phase in phase_order")
    for role in program.roles:
        last = -1
        for phase in role.phases:
            if phase.name not in order:
                raise PlanError(
                    f"{program.name}: role {role.name} phase {phase.name} "
                    f"is not in phase_order"
                )
            if order[phase.name] <= last:
                raise PlanError(
                    f"{program.name}: role {role.name} phases are not a "
                    f"subsequence of phase_order at {phase.name}"
                )
            last = order[phase.name]
    return order


def _ticket_index(program: Program) -> dict[str, TicketDecl]:
    tickets = {t.name: t for t in program.tickets}
    if len(tickets) != len(program.tickets):
        raise PlanError(f"{program.name}: duplicate ticket declaration")
    if "GEN" in tickets or "EXIT" in tickets:
        raise PlanError(f"{program.name}: GEN and EXIT are engine tickets")
    return tickets


def _visit_call(
    call: RoutineCall,
    *,
    key: tuple[str, str],
    cfg: EngineConfig,
    role: Role,
    phase_idx: int,
    tickets: dict[str, TicketDecl],
    tally: _Tally,
) -> int:
    """Check one call and tally it; returns the CTAs it occupies."""
    rt = call.routine
    if rt.invoke is None:
        raise PlanError(f"{key}: {rt.name} has no implementation")
    if SEGMENT_BARRIER in rt.named_barriers:
        raise PlanError(
            f"{key}: routine {rt.name} uses named barrier {SEGMENT_BARRIER}, "
            f"which the engine syncs phases on"
        )
    if rt.ctas_per_tile > cfg.cluster:
        raise PlanError(
            f"{key}: routine {rt.name} couples {rt.ctas_per_tile} CTAs per tile, "
            f"cluster is {cfg.cluster}"
        )
    ctas = call.tiles * rt.ctas_per_tile
    if call.cta_offset % rt.ctas_per_tile:
        raise PlanError(
            f"{key}: {rt.name} starts at CTA {call.cta_offset}, not aligned to "
            f"its {rt.ctas_per_tile}-CTA tiles"
        )
    if call.cta_offset % cfg.cluster:
        # DSMEM addressing and SMEM sharing both require cluster-aligned
        # CTA ranges, independently of the number of blocks in the phase.
        raise PlanError(
            f"{key}: {rt.name} starts at CTA {call.cta_offset}, which is not a "
            f"multiple of the {cfg.cluster}-CTA cluster; round it up to "
            f"{-(-call.cta_offset // cfg.cluster) * cfg.cluster}"
        )
    if call.cta_offset + ctas > role.ctas:
        raise PlanError(
            f"{key}: {call.tiles} tiles of {rt.name} from CTA {call.cta_offset} "
            f"need {call.cta_offset + ctas} CTAs, role has {role.ctas}"
        )
    tally.smem_need[call.smem] = max(tally.smem_need.get(call.smem, 0), rt.smem_bytes)
    tally.smem_align[call.smem] = max(
        tally.smem_align.get(call.smem, _ALIGN), rt.smem_align
    )
    tally.smem_live[role.name].setdefault(call.smem, set()).add(phase_idx)
    # Cluster-aligned, because a DSMEM push reaches the peers of a physical
    # cluster at the same offset: a block one CTA of a cluster holds is held
    # by the whole cluster as far as sharing an offset goes.
    lo = call.cta_offset // cfg.cluster * cfg.cluster
    hi = -(-(call.cta_offset + ctas) // cfg.cluster) * cfg.cluster
    seen = tally.smem_ctas.setdefault(role.name, {}).get(call.smem)
    tally.smem_ctas[role.name][call.smem] = (
        (lo, hi) if seen is None else (min(seen[0], lo), max(seen[1], hi))
    )
    for ticket_name, count in call.produces:
        if ticket_name not in tickets:
            raise PlanError(f"{key}: {rt.name} produces unknown ticket {ticket_name}")
        tally.producers[ticket_name] += count
    tally.streams[key[1]] += ctas * rt.streams
    return ctas


def _warp_segments(phase: Phase, key, role_ctas: int) -> tuple[WarpSegment, ...]:
    """Split CTAs by routine warp demand; idle CTAs lend all twelve warps.

    Cluster-barrier phases require one split throughout so every physical-cluster
    thread participates, including engine-group mirrors.
    """
    threads_at: list[Optional[int]] = [None] * role_ctas
    for call in phase.routines:
        rt = call.routine
        if rt.threads % 32 or rt.threads > BLOCK:
            raise PlanError(
                f"{key}: {rt.name} runs on {rt.threads} threads, not a warp "
                f"multiple of at most {BLOCK}"
            )
        if call.cta_offset < 0:
            # Reject negative offsets before Python indexing can wrap to tail CTAs.
            raise PlanError(
                f"{key}: {rt.name} starts at CTA {call.cta_offset}, which is "
                f"before the role"
            )
        for cta in range(
            call.cta_offset,
            min(role_ctas, call.cta_offset + call.tiles * rt.ctas_per_tile),
        ):
            if threads_at[cta] not in (None, rt.threads):
                # Calls sharing a range run back to back in the same warps.
                raise PlanError(
                    f"{key}: routines disagree on the warp split at CTA {cta} "
                    f"({sorted({threads_at[cta], rt.threads})} threads); split "
                    f"the phase or the role"
                )
            threads_at[cta] = rt.threads

    if any(call.routine.cluster_barriers for call in phase.routines):
        split = {t for t in threads_at if t is not None}
        if len(split) > 1:
            raise PlanError(
                f"{key}: routines disagree on the warp split "
                f"({sorted(split)} threads) and one executes cluster "
                f"barriers, which the engine group mirrors phase-wide"
            )

    out, start = [], 0
    for cta in range(1, role_ctas + 1):
        if cta == role_ctas or threads_at[cta] != threads_at[start]:
            out.append(WarpSegment(start, cta - start, (threads_at[start] or 0) // 32))
            start = cta
    return tuple(out)


def _visit_spare(
    phase: Phase,
    *,
    key,
    role: Role,
    rw: int,
    tally: _Tally,
    operands: frozenset,
) -> None:
    """Validate this phase's prefetch work."""
    in_phase = [pf for pf in phase.prefetch if not pf.before_gate]
    if in_phase and rw == WARPS:
        raise PlanError(
            f"{key}: spare work is declared but the routines take all "
            f"{WARPS} warps, so there are no engine warps to run it"
        )
    for pf in phase.prefetch:
        if pf.source not in operands:
            raise PlanError(
                f"{key}: prefetch names {pf.source}, which the program does "
                f"not declare as an operand"
            )
        if pf.total_bytes <= 0 or pf.total_bytes % 16:
            raise PlanError(
                f"{key}: prefetch of {pf.source} is {pf.total_bytes} bytes, "
                f"which is not a positive multiple of 16"
            )
        tally.streams[phase.name] += role.ctas


def _cluster_coupled(phase: Phase) -> bool:
    """Whether the phase declares cluster synchronization for peer SMEM access."""
    return any(c.routine.cluster_barriers for c in phase.routines)


def _visit_phase(
    phase: Phase,
    *,
    role: Role,
    position: int,
    order: dict[str, int],
    cfg: EngineConfig,
    tickets: dict[str, TicketDecl],
    tally: _Tally,
    operands: frozenset,
) -> None:
    key = (role.name, phase.name)
    phase_idx = order[phase.name]
    segments = _warp_segments(phase, key, role.ctas)
    tally.warp_segments[key] = segments
    hosting = {seg.routine_warps for seg in segments if seg.routine_warps}
    rw = hosting.pop() if len(hosting) == 1 else (0 if not hosting else None)
    # Cluster barriers are counted per CTA, not per phase: calls on disjoint
    # ranges are alternatives, and only calls sharing a range run back to back.
    barriers_at = [0] * role.ctas
    # TMEM is per SM: calls on disjoint CTA ranges do not add up.
    tmem_at = [0] * role.ctas
    for call in phase.routines:
        rt = call.routine
        if rt.cluster_barriers and cfg.cluster == 1:
            raise PlanError(
                f"{key}: routine {rt.name} needs cluster barriers but the "
                f"kernel has no cluster"
            )
        ctas = _visit_call(
            call,
            key=key,
            cfg=cfg,
            role=role,
            phase_idx=phase_idx,
            tickets=tickets,
            tally=tally,
        )
        if rt.cluster_barriers and (
            call.cta_offset % cfg.cluster or ctas % cfg.cluster
        ):
            # Every thread of a physical cluster must join a cluster barrier,
            # so a routine that executes one has to own whole clusters --
            # otherwise the CTAs outside its range hang their peers.
            raise PlanError(
                f"{key}: {rt.name} executes cluster barriers on CTAs "
                f"[{call.cta_offset}, {call.cta_offset + ctas}) of the role, "
                f"which is not aligned to {cfg.cluster}-CTA clusters"
            )
        if (
            position > 0
            and phase.issue_before_gate
            and rt.gates_itself
            and _cluster_coupled(role.phases[position - 1])
        ):
            # Early weight copies must not overwrite a peer's live SMEM.
            tally.smem_live[role.name][call.smem].add(
                order[role.phases[position - 1].name]
            )
        for cta in range(call.cta_offset, call.cta_offset + ctas):
            barriers_at[cta] += rt.cluster_barriers
            tmem_at[cta] += rt.tmem_cols

    if phase.issue_before_gate:
        if not phase.consumes and position > 0:
            raise PlanError(
                f"{key}: issue_before_gate hands the routines the phase's "
                f"tickets; only a role's first phase may consume none"
            )
        for call in phase.routines:
            # A routine that does not wait for its own inputs is gated by the
            # engine right before its call -- a CTA-wide wait, so it must be
            # the only thing on its CTAs (all twelve warps).
            if not call.routine.gates_itself and call.routine.threads != BLOCK:
                raise PlanError(
                    f"{key}: {call.routine.name} does not wait for its inputs "
                    f"itself and shares its CTAs with the engine group, so it "
                    f"cannot be gated per call"
                )

    for wait in phase.consumes:
        if wait.ticket not in tickets:
            raise PlanError(f"{key}: phase waits on unknown ticket {wait.ticket}")
        tally.consumed.add(wait.ticket)

    cols = max(tmem_at, default=0)
    if cols > cfg.tmem_cols:
        raise PlanError(
            f"{key}: routines allocate {cols} TMEM columns, limit {cfg.tmem_cols}"
        )

    _visit_spare(
        phase,
        key=key,
        role=role,
        rw=rw,
        tally=tally,
        operands=operands,
    )

    # Mirror routine cluster barriers on all remaining engine warps.
    # Non-uniform splits only occur in phases without cluster barriers.
    tally.mirrors[key] = max(barriers_at) if rw is not None and rw < WARPS else 0


def _check_tickets(program: Program, tickets, tally: _Tally) -> dict[str, int]:
    """Conservation: what is waited on is produced, and in the right shape."""
    for name, ticket in tickets.items():
        produced = tally.producers[name]
        if name in tally.consumed and produced == 0:
            raise PlanError(
                f"{program.name}: ticket {name} is consumed but never produced"
            )
        if produced % ticket.width:
            raise PlanError(
                f"{program.name}: ticket array {name} of width {ticket.width} "
                f"receives {produced} arrivals, not a multiple"
            )
    return {
        name: count // tickets[name].width for name, count in tally.producers.items()
    }


def _entry_gates(
    program: Program,
    tickets: dict[str, TicketDecl],
    offsets: dict[str, tuple[int, int]],
    per_gen: dict[str, int],
) -> dict[tuple[str, str], tuple[tuple[int, int, int], ...]]:
    """Precompute (offset,multiplier,subtrahend) for each phase wait.

    Device thresholds are gen*multiplier-subtrahend.
    """
    gates: dict[tuple[str, str], tuple[tuple[int, int, int], ...]] = {}
    for role in program.roles:
        for phase in role.phases:
            key = (role.name, phase.name)
            waits = []
            for wait in phase.consumes:
                ticket = tickets[wait.ticket]
                if ticket.width != 1:
                    # The gate reads the ticket's base word only, so a wait on
                    # an array would let the phase through on the first word.
                    raise PlanError(
                        f"{key}: waits on {wait.ticket}, which is "
                        f"{ticket.width} words; the entry gate reads one"
                    )
                arrivals = per_gen[wait.ticket]
                waits.append((offsets[wait.ticket][0], arrivals, 0))
            gates[key] = tuple(waits)
    return gates


def _assign_ticket_words(program: Program) -> dict[str, tuple[int, int]]:
    """Word 0 is the generation, word 1 the exit counter; the rest follow."""
    offsets = {"GEN": (0, 1), "EXIT": (1, 1)}
    offset = 2
    for ticket in program.tickets:
        offsets[ticket.name] = (offset, ticket.width)
        offset += ticket.width
    return offsets


def _lay_out_smem(
    program: Program, tally: _Tally
) -> tuple[dict[str, dict[str, int]], dict[str, int], dict[str, int]]:
    """Give every SMEM name an offset, sharing between disjoint lifetimes.

    Blocks need distinct offsets only when both their live phase intervals
    and their owning CTA ranges overlap.
    """
    for name, need in tally.smem_need.items():
        if need <= 0:
            raise PlanError(f"{program.name}: SMEM {name} has no size")
    sizes = {name: _align(need) for name, need in tally.smem_need.items()}
    aligns = {name: tally.smem_align.get(name, _ALIGN) for name in sizes}
    offsets: dict[str, dict[str, int]] = {}
    total = 0
    for role in program.roles:
        live = tally.smem_live[role.name]
        spans = {name: (min(ph), max(ph)) for name, ph in live.items()}
        ctas = tally.smem_ctas.get(role.name, {})
        # (start, end, cta_lo, cta_hi, off, bytes)
        placed: list[tuple[int, int, int, int, int, int]] = []
        role_offsets: dict[str, int] = {}
        # Longest-lived first, then largest: rings and shared scratch sit low
        # and the short-lived routine storage packs above or across them.
        for name in sorted(
            spans, key=lambda n: (-(spans[n][1] - spans[n][0]), -sizes[n], n)
        ):
            start, end = spans[name]
            cta_lo, cta_hi = ctas.get(name, (0, role.ctas))
            size = sizes[name]
            grain = aligns[name]
            offset = 0
            clash = True
            while clash:
                clash = False
                for o_start, o_end, o_lo, o_hi, o_off, o_size in placed:
                    same_time = not (end < o_start or start > o_end)
                    same_ctas = cta_lo < o_hi and o_lo < cta_hi
                    overlaps = (
                        same_time
                        and same_ctas
                        and not (offset + size <= o_off or offset >= o_off + o_size)
                    )
                    if overlaps:
                        offset = _align(o_off + o_size, grain)
                        clash = True
                        break
            placed.append((start, end, cta_lo, cta_hi, offset, size))
            role_offsets[name] = offset
        offsets[role.name] = role_offsets
        total = max(total, max((off + size for *_, off, size in placed), default=0))
    if total > program.config.smem_limit:
        raise PlanError(
            f"{program.name}: SMEM needs {total} B, limit {program.config.smem_limit} B"
        )
    return offsets, sizes, aligns


def plan(program: Program) -> Plan:
    """Validate ``program`` and derive its compile-time constants.

    Raises ``PlanError`` naming the first inconsistency found.
    """
    cfg = program.config
    _check_launch(program)
    order = _phase_index(program)
    tickets = _ticket_index(program)
    tally = _Tally(
        producers={name: 0 for name in tickets},
        streams={name: 0 for name in program.phase_order},
        smem_live={role.name: {} for role in program.roles},
    )
    operands = frozenset(program.operands)
    for role in program.roles:
        for position, phase in enumerate(role.phases):
            _visit_phase(
                phase,
                role=role,
                position=position,
                order=order,
                cfg=cfg,
                tickets=tickets,
                tally=tally,
                operands=operands,
            )

    for phase_name, budget in program.stream_budgets:
        if phase_name not in order:
            raise PlanError(
                f"{program.name}: stream budget names unknown phase {phase_name}"
            )
        if tally.streams[phase_name] > budget:
            raise PlanError(
                f"{program.name}: phase {phase_name} keeps "
                f"{tally.streams[phase_name]} streams in flight, budget {budget}"
            )

    producers_per_gen = _check_tickets(program, tickets, tally)
    smem_offsets, smem_bytes, smem_align = _lay_out_smem(program, tally)
    ticket_offsets = _assign_ticket_words(program)
    return Plan(
        program=program,
        ticket_offsets=ticket_offsets,
        producers_per_gen=producers_per_gen,
        smem_offsets=smem_offsets,
        smem_bytes=smem_bytes,
        smem_align=smem_align,
        mirrors=tally.mirrors,
        warp_segments=tally.warp_segments,
        waits=_entry_gates(program, tickets, ticket_offsets, producers_per_gen),
    )
