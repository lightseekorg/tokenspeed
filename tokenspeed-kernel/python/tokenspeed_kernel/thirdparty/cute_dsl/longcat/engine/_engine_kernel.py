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

"""Persistent LongCat layer execution from a validated static plan.

Each CTA runs its role's phases, waits for producer tickets, invokes device
routines and joins their declared cluster barriers. The last CTA advances
the generation after every phase has retired. Workspaces have fixed
addresses for CUDA Graph replay.
"""

from __future__ import annotations

from typing import Optional

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings.driver import CUstream
from cutlass import Int32, Int64, Uint8, Uint32, utils
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from flashinfer.cute_dsl.utils import make_ptr
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.attention._launch import (
    assert_cluster_residency,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.attention._primitives import (
    make_l2_policy_evict_normal,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.engine._plan import (
    BLOCK,
    SEGMENT_BARRIER,
    WARPS,
    Plan,
    Role,
    WarpSegment,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.routines._primitives import (
    atomic_add_gpu_u32,
    load_acquire_gpu_u32,
    prefetch_bulk_l2,
    store_relaxed_gpu_u32,
)

# One bulk L2 prefetch per participating thread per iteration.
PREFETCH_CHUNK_BYTES = 4096


def _preserve_thread_id_values(_owner, module, _function_name):
    """Keep thread-ID dependencies opaque to backend rematerialization."""
    reads = []

    def collect(op):
        if op.name == "nvvm.read.ptx.sreg.tid.x":
            reads.append(op)
        return ir.WalkResult.ADVANCE

    module.operation.walk(collect)
    for op in reads:
        with ir.InsertionPoint(op):
            read = op.clone()
            value = llvm.inline_asm(
                op.results[0].type,
                [read.results[0]],
                "mov.u32 $0, $1;",
                "=r,r",
                has_side_effects=False,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
                loc=op.location,
            )
            # Preserve the intrinsic's [0, 1024) range through the identity.
            bound = llvm.mlir_constant(
                ir.IntegerAttr.get(value.type, 1024), loc=op.location
            )
            in_range = llvm.icmp(llvm.ICmpPredicate.ult, value, bound, loc=op.location)
            llvm.intr_assume(in_range, [], [], loc=op.location)
        op.results[0].replace_all_uses_with(value)
        op.erase()


def _operand_ptr(addrs, index, name, dtype):
    """Trace an operand pointer from the shared ordered address tuple."""
    return cute.make_ptr(
        dtype, addrs[index[name]], cute.AddressSpace.gmem, assumed_align=16
    )


class _HostContext:
    """Host-preparation operand addresses, without device tile/thread/SMEM state.

    Shares ptr(name,dtype) with _CallContext so host TMA preparation and device
    routines address the same operands.
    """

    def __init__(self, addrs, index):
        self._addrs = addrs
        self._index = index

    def ptr(self, name: str, dtype=Uint8):
        """Operand pointer by name, as the caller's element type."""
        return _operand_ptr(self._addrs, self._index, name, dtype)


class _CallContext:
    """One routine tile: trace-time operand metadata and device tile/thread values."""

    def __init__(
        self,
        addrs,
        index,
        smem,
        tile,
        pair_rank,
        tidx,
        gen,
        tickets,
        plan,
        threads,
        prepared=None,
        gate=None,
    ):
        self._addrs = addrs
        # The ``(offset, mult, sub)`` of each of the phase's tickets when the
        # engine left the gate to the routine (``Phase.issue_before_gate``),
        # or None when the engine owns the phase gate.
        self._gate = gate
        self._index = index
        # Host-prepared values (or None), passed as kernel parameters for TMA.
        self.prepared = prepared
        self._tickets = tickets
        self._plan = plan
        self.smem = smem
        self.tile = tile
        self.pair_rank = pair_rank
        self.tidx = tidx
        self.gen = gen
        # Threads the routine group runs on, so an adapter passing its
        # routine's participant count cannot disagree with the warp split.
        self.threads = threads

    def ptr(self, name: str, dtype=Uint8):
        """Operand pointer by name/dtype, from the program-ordered address tuple."""
        return _operand_ptr(self._addrs, self._index, name, dtype)

    def ticket(self, name: str):
        """Pointer to a ticket word, for a routine that publishes per segment."""
        offset, _ = self._plan.ticket_offsets[name]
        return self._tickets.iterator + offset

    def ticket_target(self, name: str):
        """Return this generation's complete (ticket,target) for name.

        Use for late routine-owned waits not declared as phase consumes; compare
        with signed difference, as in the engine gate.
        """
        offset, _ = self._plan.ticket_offsets[name]
        return (
            self._tickets.iterator + offset,
            self.gen * Uint32(self._plan.producers_per_gen[name]),
        )

    @property
    def gated(self) -> bool:
        """Trace-time: the routine owns its phase gate (``gate()`` is valid)."""
        return self._gate is not None

    def gate(self):
        """Return the gate owned by an issue_before_gate routine; reject other phases.

        Acquire-spin with signed difference before reading consumed inputs.
        """
        (pair,) = self.gates()
        return pair

    def gates(self):
        """``[(ticket, target), ...]``: every ticket of the phase gate, for a
        routine whose phase consumes more than one (see :meth:`gate`)."""
        if self._gate is None:
            raise ValueError("gate() is only handed to issue_before_gate phases")
        return [
            (self._tickets.iterator + offset, self.gen * Uint32(mult) - Uint32(sub))
            for offset, mult, sub in self._gate
        ]


class LongcatEngineKernel:
    """Trace-time specialisation of the engine on one ``Plan``."""

    def __init__(
        self,
        plan: Plan,
        *,
        use_pdl: bool = False,
        pdl_trigger_phase: Optional[str] = None,
    ) -> None:
        """Specialize the validated plan and its PDL launch boundaries.

        Args:
            plan: Validated static program.
            use_pdl: Issue first-phase weight L2 prefetch while the predecessor drains;
                wait on griddepcontrol before reading tickets or operands. Activation
                prefetch is coherent but may waste bandwidth. PDL operations are no-ops
                under an ordinary launch.
                An issue_before_gate first phase without a ticket skips the entry wait
                (pdl_early). Its previous workspace launch must already be complete;
                consecutive PDL launches on that workspace would read stale GEN.
            pdl_trigger_phase: Trigger the dependent grid before this phase's gate.
                Roles without it, or with None, trigger at their final phase. An early
                trigger can occupy SMs needed by side-stream MoE until the tail.
        """
        program = plan.program
        self.plan = plan
        self.program = program
        self.grid = program.config.grid
        self.cluster = program.config.cluster
        self.use_pdl = bool(use_pdl)
        if (
            pdl_trigger_phase is not None
            and pdl_trigger_phase not in program.phase_order
        ):
            raise ValueError(
                f"pdl_trigger_phase {pdl_trigger_phase!r} is not a phase of "
                f"{program.name}: {list(program.phase_order)}"
            )
        # Per role: the named phase if the role has it, else its last one.
        self.pdl_trigger = {
            role.name: (
                pdl_trigger_phase
                if any(p.name == pdl_trigger_phase for p in role.phases)
                else role.phases[-1].name
            )
            for role in program.roles
        }
        # Early routines may prefetch into SMEM before the predecessor retires.
        # They read GEN before griddepcontrol_wait, requiring the previous
        # launch on this workspace to be complete (one workspace per layer).
        self.pdl_early = {
            role.name: self.use_pdl
            and role.phases[0].issue_before_gate
            and not role.phases[0].consumes
            for role in program.roles
        }
        self.operands = tuple(program.operands)
        self.operand_index = {name: i for i, name in enumerate(self.operands)}
        # ``(routine name, callable)`` in program order; the kernel receives
        # their results as one parameter and hands each back by name.
        seen: set[str] = set()
        self.host_prepare: list[tuple[str, object]] = []
        for role in program.roles:
            for phase in role.phases:
                for call in phase.routines:
                    rt = call.routine
                    if rt.host_prepare is None or rt.name in seen:
                        continue
                    seen.add(rt.name)
                    self.host_prepare.append((rt.name, rt.host_prepare))
        self.host_prepare_index = {
            name: i for i, (name, _) in enumerate(self.host_prepare)
        }
        self.role_bounds: list[tuple[Role, int, int]] = []
        lo = 0
        for role in program.roles:
            self.role_bounds.append((role, lo, lo + role.ctas))
            lo += role.ctas
        # Pre-filter calls and engine-group cluster mirrors: traced meta loops
        # cannot use continue. Entries carry the call, CTA range and body.
        self.prefetch: dict[tuple[str, str], list] = {}
        self.emitted: dict[tuple[str, str], list] = {}
        self.mirrored: dict[tuple[str, str], list] = {}
        for role in program.roles:
            for phase in role.phases:
                key = (role.name, phase.name)
                emitted = []
                mirrored = []
                for call in phase.routines:
                    rt = call.routine
                    span = call.tiles * rt.ctas_per_tile
                    if rt.invoke is None:
                        raise ValueError(
                            f"{role.name}/{phase.name}: {rt.name} has no implementation"
                        )
                    emitted.append((call, call.cta_offset, span, (rt.invoke,)))
                    if plan.mirrors[key] and rt.cluster_barriers:
                        mirrored.append((call.cta_offset, span, rt.cluster_barriers))
                self.emitted[key] = emitted
                self.mirrored[key] = mirrored
                # Resolve task kinds before tracing, which visits every branch.
                # Align CTA slices to 16 bytes; device code clamps the final
                # slice to avoid prefetching beyond the operand.
                prefetch = []
                for pf in phase.prefetch:
                    lo, hi = 0, role.ctas
                    per_cta = -(-pf.total_bytes // (hi - lo))  # ceil
                    slice_bytes = -(-per_cta // 16) * 16  # ceil to 16 B
                    prefetch.append(
                        (
                            pf.source,
                            slice_bytes,
                            pf.total_bytes,
                            pf.before_gate,
                            lo,
                            hi,
                        )
                    )
                self.prefetch[key] = prefetch

    # device
    #
    def _segment_warps(self, key: cutlass.Constexpr, lo: int) -> int:
        """Trace-time routine warp count for the CTA run containing lo.

        Idle CTAs lend all twelve warps to engine work. Engine-group entry must
        remain below thread 384 on every CTA with engine work.
        """
        for seg in self.plan.warp_segments.get(key, ()):
            if seg.lo <= lo < seg.lo + seg.span:
                return seg.routine_warps
        raise KeyError(f"{key}: no warp segment covers CTA {lo}")

    # Device helpers need @cute.jit for staged control flow. Phase keys are
    # Constexpr and program loops unroll at trace time.

    @cute.jit
    def _gate(self, key: cutlass.Constexpr, tickets: cute.Tensor, gen, tidx):
        """Wait for the phase's tickets, then let the whole CTA through."""
        if len(self.plan.waits[key]) > 0:
            if tidx == 0:
                for offset, mult, sub in self.plan.waits[key]:
                    target = gen * Uint32(mult) - Uint32(sub)
                    # Signed difference handles ticket wrap: subtract as Uint32
                    # before reinterpreting as Int32. The exit counter also
                    # wraps modulo 2^32.
                    ready = load_acquire_gpu_u32(tickets.iterator + offset)
                    while (ready - target).to(Int32) < Int32(0):
                        ready = load_acquire_gpu_u32(tickets.iterator + offset)
            cute.arch.barrier(barrier_id=SEGMENT_BARRIER, number_of_threads=BLOCK)

    @cute.jit
    def _routines(
        self,
        key: cutlass.Constexpr,
        tickets: cute.Tensor,
        operands,
        arena: Int64,
        gen,
        local,
        warp,
        tidx,
        prepared=(),
    ):
        """Run this CTA's declared device routines."""
        role_name, _ = key
        for call, lo, span, invokes in self.emitted[key]:
            routine = call.routine
            rw = routine.threads // 32
            if local >= Int32(lo) and local < Int32(lo + span):
                # ``rw == 12`` hosts the routines on every warp, so the guard
                # folds away.
                if rw == WARPS or warp < Int32(rw):
                    if cutlass.const_expr(
                        self._gate_for(key) is not None and not routine.gates_itself
                    ):
                        # issue_before_gate left the gate to the routines, and
                        # this one does not wait itself: the CTA-wide gate it
                        # would have had at the entry (it holds all warps).
                        self._gate(key, tickets, gen, tidx)
                        if cutlass.const_expr(self.pdl_early[role_name]):
                            cute.arch.griddepcontrol_wait()
                    for invoke in invokes:
                        invoke(
                            _CallContext(
                                operands,
                                self.operand_index,
                                cute.make_ptr(
                                    Uint8,
                                    arena
                                    + Int64(
                                        self.plan.smem_offsets[role_name][call.smem]
                                    ),
                                    cute.AddressSpace.smem,
                                    assumed_align=self.plan.smem_align[call.smem],
                                ),
                                (local - Int32(lo)) // Int32(routine.ctas_per_tile),
                                (local - Int32(lo)) % Int32(routine.ctas_per_tile),
                                tidx,
                                gen,
                                tickets,
                                self.plan,
                                32 * rw,
                                self._prepared_for(routine.name, prepared),
                                self._gate_for(key),
                            )
                        )

    @cute.jit
    def _prefetch(
        self,
        key: cutlass.Constexpr,
        operands,
        policy: Int64,
        local,
        warp,
        tidx,
        before_gate: cutlass.Constexpr = False,
    ):
        """Issue the declared L2 prefetch ranges without blocking consumers."""
        entries = [e for e in self.prefetch[key] if e[3] == before_gate]
        segments = self.plan.warp_segments.get(key, ())
        if cutlass.const_expr(before_gate):
            # One run spanning the role, no routine warps.
            segments = (WarpSegment(lo=0, span=1 << 30, routine_warps=0),)
        for seg in segments:
            rw = seg.routine_warps
            # A run whose routines hold every warp has no engine group to
            # issue from; the guard folds away at trace time.
            if rw < WARPS:
                # Group-local tid and this CTA run's warp split give contiguous
                # chunks; idle CTAs can use all twelve warps.
                tid = tidx - Int32(32 * rw)
                threads = BLOCK - 32 * rw
                if (
                    local >= Int32(seg.lo)
                    and local < Int32(seg.lo + seg.span)
                    and (rw == 0 or warp >= Int32(rw))
                ):
                    for source, slice_bytes, total, _, lo, hi in entries:
                        base = operands[self.operand_index[source]]
                        # Outside ``[lo, hi)`` the slice is empty: past ``hi``
                        # ``first`` lands beyond ``total``; before ``lo`` the
                        # length is zeroed.
                        first = Int64(local - Int32(lo)) * Int64(slice_bytes)
                        length = Int64(slice_bytes)
                        if Int64(total) - first < length:
                            length = Int64(total) - first
                        if local < Int32(lo):
                            length = Int64(0)
                        offset = Int64(tid) * Int64(PREFETCH_CHUNK_BYTES)
                        while offset < length:
                            span = Int64(PREFETCH_CHUNK_BYTES)
                            if length - offset < span:
                                span = length - offset
                            prefetch_bulk_l2(
                                cute.make_ptr(
                                    Uint8,
                                    base + first + offset,
                                    cute.AddressSpace.gmem,
                                    assumed_align=16,
                                ),
                                Uint32(span),
                                policy,
                            )
                            offset = offset + Int64(threads) * Int64(
                                PREFETCH_CHUNK_BYTES
                            )

    @cute.jit
    def _mirror(self, key: cutlass.Constexpr, local, warp):
        """Mirror only engine warps at the routine's cluster barriers.

        Guard by each CTA range's warp split; routine warps arrive in the routine
        itself and must not arrive twice.
        """
        for lo, span, barriers in self.mirrored[key]:
            rw = self._segment_warps(key, lo)
            if (
                local >= Int32(lo)
                and local < Int32(lo + span)
                and (rw == 0 or warp >= Int32(rw))
            ):
                for _ in range(barriers):
                    cute.arch.cluster_arrive()
                    cute.arch.cluster_wait()

    @cute.jit
    def _finish(self, tickets: cute.Tensor, gen, tidx):
        """Take the exit ticket; the last CTA opens the next generation."""
        gen_offset, _ = self.plan.ticket_offsets["GEN"]
        exit_offset, _ = self.plan.ticket_offsets["EXIT"]
        if tidx == 0:
            # The next launch on this workspace reads GEN after this grid retires.
            arrived = atomic_add_gpu_u32(
                tickets.iterator + exit_offset, Uint32(1), acq_rel=False
            )
            if arrived + Uint32(1) == gen * Uint32(self.grid):
                store_relaxed_gpu_u32(tickets.iterator + gen_offset, gen + Uint32(1))

    def _gate_for(self, key):
        """The ``(offset, mult, sub)`` gates of an ``issue_before_gate``
        phase, which its routines wait for themselves; None for every other
        phase (their gate is the engine's)."""
        role_name, phase_name = key
        for role in self.plan.program.roles:
            if role.name != role_name:
                continue
            for phase in role.phases:
                if phase.name == phase_name and phase.issue_before_gate:
                    if not phase.consumes:
                        # A role's first phase with no ticket: the routines'
                        # gate is the PDL wait alone, and ``GEN`` already
                        # holds ``gen``, so this target is met on first read.
                        offset, _ = self.plan.ticket_offsets["GEN"]
                        return ((offset, 1, 0),)
                    return tuple(self.plan.waits[key])
        return None

    def _prepared_for(self, name: cutlass.Constexpr, prepared: cutlass.Constexpr):
        """This routine's slice of the host-prepare tuple, or None."""
        i = self.host_prepare_index.get(name)
        return None if i is None else prepared[i]

    @cute.jit
    def _host_prepare(self, host_addrs):
        """Prepare routine operands in program order using the shared ptr interface.

        Return one tuple of prepared values, indexed by routine in device calls.
        """
        ctx = _HostContext(host_addrs, self.operand_index)
        out = ()
        for name, fn in self.host_prepare:
            out = out + (fn(ctx),)
        return out

    @cute.kernel
    def kernel(
        self,
        tickets: cute.Tensor,
        operands,
        prepared,
    ):
        cta, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        warp = tidx // 32
        arena = Int64(
            utils.SmemAllocator()
            .allocate(self.plan.arena_bytes, byte_alignment=self.plan.arena_align)
            .toint()
        )
        gen_offset, _ = self.plan.ticket_offsets["GEN"]
        # Normal priority keeps prefetched weights above streaming evict-first data.
        policy = make_l2_policy_evict_normal()
        if cutlass.const_expr(self.use_pdl):
            # First-phase prefetch is issue-only into coherent L2; it reads
            # neither tickets nor operand values before the PDL wait.
            for role, role_lo, role_hi in self.role_bounds:
                if cta >= Int32(role_lo) and cta < Int32(role_hi):
                    for early in (False, True):
                        self._prefetch(
                            (role.name, role.phases[0].name),
                            operands,
                            policy,
                            cta - Int32(role_lo),
                            warp,
                            tidx,
                            before_gate=early,
                        )
            for role, role_lo, role_hi in self.role_bounds:
                if cutlass.const_expr(not self.pdl_early[role.name]):
                    if cta >= Int32(role_lo) and cta < Int32(role_hi):
                        cute.arch.griddepcontrol_wait()
        # Stream order or the PDL wait must retire the prior workspace launch
        # before reading GEN; stale GEN opens every gate. Compile and launch
        # must agree on use_pdl (see the early-role contract above).
        gen = load_acquire_gpu_u32(tickets.iterator + gen_offset)

        for role, role_lo, role_hi in self.role_bounds:
            if cta >= Int32(role_lo) and cta < Int32(role_hi):
                local = cta - Int32(role_lo)
                for phase_i, phase in enumerate(role.phases):
                    key = (role.name, phase.name)
                    # Prefetch and mirrors guard their own CTA ranges; admit
                    # every warp that belongs to any range's engine group.
                    rw = min(
                        (
                            seg.routine_warps
                            for seg in self.plan.warp_segments.get(key, ())
                        ),
                        default=0,
                    )

                    if cutlass.const_expr(
                        self.use_pdl and phase.name == self.pdl_trigger[role.name]
                    ):
                        # Before the gate: a CTA that waits here has nothing
                        # left to say about when its successor may start.
                        cute.arch.griddepcontrol_launch_dependents()
                    # The phase's own weights, while this CTA waits for its
                    # inputs (the first phase's went out in the PDL prologue).
                    if cutlass.const_expr(
                        any(e[3] for e in self.prefetch[key])
                        and not (self.use_pdl and phase_i == 0)
                    ):
                        self._prefetch(
                            key, operands, policy, local, warp, tidx, before_gate=True
                        )
                    # An issue-before-gate phase leaves ticket waits to its routines.
                    if cutlass.const_expr(not phase.issue_before_gate):
                        self._gate(key, tickets, gen, tidx)

                    self._routines(
                        key, tickets, operands, arena, gen, local, warp, tidx, prepared
                    )
                    # Issue prefetch before joining the routines' cluster barriers.
                    if len(self.mirrored[key]) > 0 or any(
                        not e[3] for e in self.prefetch[key]
                    ):
                        # rw is the minimum split, admitting every possible engine warp.
                        # Each helper must still guard its own CTA run.
                        if rw == 0 or warp >= Int32(rw):
                            # The PDL prologue already issued these ranges.
                            if not (self.use_pdl and phase_i == 0):
                                self._prefetch(key, operands, policy, local, warp, tidx)
                            self._mirror(key, local, warp)

                    if cutlass.const_expr(self.pdl_early[role.name] and phase_i == 0):
                        # Every thread past the predecessor grid before the
                        # phases that read its output.
                        cute.arch.griddepcontrol_wait()
                    # SMEM lifetime boundaries require this full-CTA barrier before
                    # routine warps can overwrite the next phase's arena.
                    cute.arch.barrier(
                        barrier_id=SEGMENT_BARRIER, number_of_threads=BLOCK
                    )

        cute.arch.barrier(barrier_id=SEGMENT_BARRIER, number_of_threads=BLOCK)
        self._finish(tickets, gen, tidx)

    @cute.jit
    def launch(
        self,
        tickets_ptr: cute.Pointer,
        host_addrs,
        stream: CUstream,
    ):
        tickets = cute.make_tensor(
            tickets_ptr, cute.make_layout((self.plan.ticket_words,))
        )
        # Build TMA atoms in host jit scope and pass preparation results
        # into the kernel. Plain cp.async.bulk needs no host descriptor.
        prepared = self._host_prepare(host_addrs)
        # Operand addresses are launch parameters: rebind updates eager
        # launches, while a captured graph retains its recorded addresses.
        self.kernel(tickets, host_addrs, prepared).launch(
            grid=(self.grid, 1, 1),
            block=(BLOCK, 1, 1),
            cluster=(self.cluster, 1, 1),
            smem=self.plan.arena_bytes,
            min_blocks_per_mp=1,
            stream=stream,
            use_pdl=self.use_pdl,
        )


class EngineWorkspace:
    """Fixed ticket buffers and rebindable operand references for one engine.

    reset starts generation 1; the caller supplies the program's operands.
    """

    def __init__(
        self,
        kernel: LongcatEngineKernel,
        device: torch.device,
        operands: Optional[dict[str, torch.Tensor]] = None,
    ) -> None:
        self.kernel = kernel
        supplied = dict(operands or {})
        missing = set(kernel.operands) - supplied.keys()
        if missing:
            raise ValueError(f"missing operands: {sorted(missing)}")
        extra = supplied.keys() - set(kernel.operands)
        if extra:
            raise ValueError(f"operands the program does not declare: {sorted(extra)}")
        for name in kernel.operands:
            self._check_operand(name, supplied[name])
        self.operands = supplied
        self.tickets = torch.zeros(
            kernel.plan.ticket_words, dtype=torch.uint32, device=device
        )
        self.ticket_ptr = make_ptr(
            Uint32, self.tickets.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
        )
        self.reset()

    def _check_operand(self, name: str, t: torch.Tensor) -> None:
        # Routines assume contiguous, 16-byte-aligned operand pointers.
        # Validate tensors here; the planner sees only names and byte counts.
        if not t.is_contiguous():
            raise ValueError(f"operand {name} is not contiguous")
        if t.data_ptr() % 16:
            raise ValueError(
                f"operand {name} is not 16-byte aligned: {t.data_ptr():#x}"
            )
        # Warming past the end of an operand reads whatever the allocator
        # put after it.  Contiguity above makes this byte count exact.
        for role in self.kernel.program.roles:
            for phase in role.phases:
                for pf in phase.prefetch:
                    if (
                        pf.source == name
                        and pf.total_bytes > t.numel() * t.element_size()
                    ):
                        raise ValueError(
                            f"{role.name}/{phase.name}: prefetch of {name} "
                            f"warms {pf.total_bytes} bytes of a "
                            f"{t.numel() * t.element_size()}-byte tensor"
                        )

    def rebind(self, **tensors: torch.Tensor) -> None:
        """Bind operand tensors for subsequent eager or capture launches.

        A captured graph retains its recorded addresses. Keep tensors alive through
        their launch; rebind affects subsequent launches, not an existing graph.

        Args:
            tensors: Declared operand names mapped to contiguous, 16-byte-aligned
                tensors covering all declared prefetch ranges.

        Raises:
            ValueError: An undeclared name or an invalid tensor.
        """
        extra = tensors.keys() - set(self.kernel.operands)
        if extra:
            raise ValueError(f"operands the program does not declare: {sorted(extra)}")
        for name, t in tensors.items():
            self._check_operand(name, t)
        self.operands.update(tensors)

    def _host_addrs(self) -> tuple:
        return tuple(
            Int64(self.operands[name].data_ptr()) for name in self.kernel.operands
        ) or (Int64(0),)

    def reset(self) -> None:
        self.tickets.zero_()
        self.tickets[self.kernel.plan.ticket_offsets["GEN"][0]] = 1


class LongcatEngine:
    """Compile once per plan; launch on the current stream."""

    def __init__(self, plan: Plan, **kwargs) -> None:
        self.kernel = LongcatEngineKernel(plan, **kwargs)
        self._compiled = None

    def workspace(
        self,
        device: torch.device,
        operands: Optional[dict[str, torch.Tensor]] = None,
    ) -> EngineWorkspace:
        return EngineWorkspace(self.kernel, device, operands)

    def __call__(self, ws: EngineWorkspace) -> None:
        """Enqueue one generation on the current stream.

        Tickets stay at fixed addresses. Read current operand bindings per launch;
        query residency only at first compilation.
        """
        stream = CUstream(torch.cuda.current_stream(ws.tickets.device).cuda_stream)
        args = (ws.ticket_ptr, ws._host_addrs(), stream)
        if self._compiled is None:
            # Once, on the first call: the query needs a live context and
            # must not run inside a graph capture.
            assert_cluster_residency(
                ws.tickets.device.index,
                self.kernel.cluster,
                self.kernel.grid // self.kernel.cluster,
                f"engine {self.kernel.program.name}",
            )
            self._compiled = cute.compile(
                self.kernel.launch,
                *args,
                trace_finalize_hooks=_preserve_thread_id_values,
            )
        self._compiled(*args)


__all__ = [
    "EngineWorkspace",
    "LongcatEngine",
    "LongcatEngineKernel",
]
