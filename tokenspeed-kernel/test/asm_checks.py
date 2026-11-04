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

"""Text checks on gfx950 assembly that kernel codegen tests share.

Pure functions of the assembly text: no device, no Triton import, so the
offline compile tests can use them on hosts without a GPU. The source
attribution (:func:`source_lines`, :func:`frame_functions`) also reads the
Python files the ``.file`` directives name.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


def polling_cycles(assembly: str) -> list[list[str]]:
    """Return each loop of ``assembly`` from its target label to its back branch."""
    lines = assembly.splitlines()
    labels = {
        match[1]: index
        for index, line in enumerate(lines)
        if (match := re.match(r"^(\.LBB\w+):", line))
    }
    cycles = []
    for index, line in enumerate(lines):
        match = re.search(r"\bs_(?:cbranch_\w+|branch)\s+(\.LBB\w+)", line)
        if match and labels[match[1]] < index:
            cycles.append(lines[labels[match[1]] : index + 1])
    return cycles


def votes_with_subgroup_mask(cycle: list[str]) -> bool:
    """Return whether a polling cycle exits on one v_cmp lane mask.

    A cross-lane reduction (DPP steps and a lane read) or a workgroup
    exchange (LDS and s_barrier) on every retry would delay noticing data.
    """
    text = "\n".join(cycle)
    return (
        len(re.findall(r"v_cmp_ne_u32_e64 s\[\d+:\d+\], 0, v\d+", text)) == 1
        and re.search(r"_dpp|v_readlane|v_readfirstlane|s_barrier|\bds_", text) is None
    )


def _innermost_loop_lines(assembly: str) -> dict[str, list[int]]:
    """Map each innermost loop's header to the indices of its lines.

    A block starts at a label (``.LBB0_7:``) or an unlabeled block comment
    (``; %bb.7:``); its loop annotation is on that line or on the comment
    lines right after it: ``This Inner Loop Header`` for an innermost
    loop's header, ``in Loop: Header=BB0_7`` for a block of that loop.
    """
    lines = assembly.splitlines()
    loops: dict[str, list[int]] = {}
    current = None
    for index, line in enumerate(lines):
        label = re.match(r"^\.L(BB\w+):", line)
        if label or re.match(r"^; %bb\.\d+:", line):
            notes = [line]
            for later in lines[index + 1 :]:
                if not re.match(r"^\s+;", later):
                    break
                notes.append(later)
            comment = " ".join(notes)
            header = re.search(r"in Loop: Header=(BB\w+)", comment)
            if label and "This Inner Loop Header" in comment:
                current = label[1]
            elif header:
                current = header[1]
            else:
                current = None
            if current is not None:
                loops.setdefault(current, [])
        if current is not None and current in loops:
            loops[current].append(index)
    # A block "in" an outer loop's header is not an innermost loop's.
    return {
        name: lines_
        for name, lines_ in loops.items()
        if any("This Inner Loop Header" in assembly.splitlines()[i] for i in lines_)
    }


def innermost_loops(assembly: str) -> list[str]:
    """Return the text of each innermost loop, from LLVM's loop comments.

    A block belongs to the loop whose header it names (``in Loop:
    Header=BB0_7``); the header itself says ``=>This Inner Loop Header``.
    Backward branches that LLVM placed out of line are not loops.
    """
    lines = assembly.splitlines()
    return [
        "\n".join(lines[i] for i in indices)
        for indices in _innermost_loop_lines(assembly).values()
    ]


def _successors(lines: list[str]):
    """Line successors of the assembly: branch targets and fall-through."""
    labels = {}
    for index, line in enumerate(lines):
        label = re.match(r"^(\.L\w+):", line)
        if label:
            labels[label[1]] = index

    def successors(index: int) -> list[int]:
        line = lines[index]
        target = re.search(r"\bs_(c?branch)\w*\s+(\.L\w+)", line)
        nexts = [labels[target[2]]] if target and target[2] in labels else []
        if not re.search(r"\bs_branch\s", line) and "s_endpgm" not in line:
            nexts.append(index + 1)
        return [n for n in nexts if n < len(lines)]

    return successors


def _loop_header(lines: list[str], loop: set[int]) -> int:
    return next(i for i in sorted(loop) if "Inner Loop Header" in lines[i])


def _loop_exits(lines: list[str], loop: set[int], successors) -> list[int]:
    """Successors outside the loop of the loop lines its header reaches
    (lines after an unconditional branch only start other blocks' code)."""
    inside, todo = set(), [_loop_header(lines, loop)]
    while todo:
        index = todo.pop()
        if index in inside or index not in loop:
            continue
        inside.add(index)
        todo.extend(successors(index))
    return [n for i in inside for n in successors(i) if n not in loop]


def _retry_lines(lines: list[str], loop: set[int], successors) -> set[int]:
    """Lines reached from the loop's exits that lead back into the loop: the
    retry path (LLVM moves a check run every N retries out of the loop)."""
    reached: set[int] = set()
    todo = _loop_exits(lines, loop, successors)
    while todo:
        index = todo.pop()
        if index in reached or index in loop:
            continue
        reached.add(index)
        todo.extend(successors(index))
    retry = {i for i in reached if any(n in loop for n in successors(i))}
    changed = True
    while changed:
        changed = False
        for i in reached - retry:
            if any(n in retry for n in successors(i)):
                retry.add(i)
                changed = True
    return retry


def _load_before_barrier(lines: list[str], loop: set[int]) -> bool:
    """Whether a path leaving the loop for good (not its retry path) reaches
    a vector memory load before a workgroup barrier, following branches."""
    successors = _successors(lines)
    retry = _retry_lines(lines, loop, successors)
    seen: set[int] = set()
    todo = _loop_exits(lines, loop, successors)
    while todo:
        index = todo.pop()
        if index in seen or index in loop:
            continue
        seen.add(index)
        line = lines[index]
        if "s_barrier" in line:
            continue
        if index not in retry and re.search(r"\b(global|buffer)_load", line):
            return True
        todo.extend(successors(index))
    return False


def _clock_on_retry_path(lines: list[str], loop: set[int]) -> bool:
    """Whether the loop reads the clock, in its blocks or on its retry path."""
    retry = _retry_lines(lines, loop, _successors(lines))
    return any("s_memrealtime" in lines[i] for i in loop | retry)


def _registers(operand: str) -> set[str]:
    """The registers an operand names: ``v1``, ``s[4:7]`` -> {s4, ..., s7}."""
    single = re.fullmatch(r"([sv])(\d+)", operand)
    if single:
        return {operand}
    span = re.fullmatch(r"([sv])\[(\d+):(\d+)\]", operand)
    if not span:
        return set()
    return {f"{span[1]}{i}" for i in range(int(span[2]), int(span[3]) + 1)}


def _unordered_read_then_write(lines: list[str]) -> list[str]:
    """Global stores to the address of an earlier global load with a scalar
    base (same ``vaddr``, base and offset, neither register redefined since),
    with no full drain and workgroup barrier in between.

    Such a word is one per workgroup: every wave loads it, one lane stores
    it, and a wave that loads after the store reads the new value. The scan
    is linear and textual: loads through the scalar cache (``s_load``) and
    addresses rebuilt in other registers are not covered."""
    problems = []
    loads: dict[tuple[str, str, str], int] = {}
    for index, line in enumerate(lines):
        match = re.match(r"\s*([sv]_\w+|global_\w+|buffer_\w+|ds_\w+)\s+(.*)", line)
        if not match:
            continue
        mnemonic = match[1]
        operands = [
            operand.split()[0] for operand in match[2].split(",") if operand.strip()
        ]
        modifiers = match[2].split(",")[-1].split()[1:]
        address = None
        if re.match(r"global_(load|store)_", mnemonic) and len(operands) >= 3:
            vaddr = operands[1] if "load" in mnemonic else operands[0]
            offset = next((m for m in modifiers if m.startswith("offset:")), "")
            if operands[2].startswith("s["):
                address = (vaddr, operands[2], offset)
        if address and mnemonic.startswith("global_store_") and address in loads:
            between = "\n".join(lines[loads[address] : index])
            if "s_waitcnt vmcnt(0)" not in between or "s_barrier" not in between:
                problems.append(f"store after an unordered read: {line.strip()}")
        # Forget addresses whose registers this instruction redefines: its
        # first operand, except for compares, branches, stores and atomics
        # that return nothing (no sc0).
        reads_only = re.match(
            r"(s_cmp|s_bitcmp|s_cbranch|s_branch|s_waitcnt|s_barrier"
            r"|global_store|buffer_store|ds_write)",
            mnemonic,
        ) or (re.match(r"(global|buffer)_atomic", mnemonic) and "sc0" not in modifiers)
        if operands and not reads_only:
            written = _registers(operands[0])
            for key in [
                key
                for key in loads
                if written & (_registers(key[0]) | _registers(key[1]))
            ]:
                del loads[key]
        if address and mnemonic.startswith("global_load_"):
            loads.setdefault(address, index)
    return problems


def _counted_loop(loop: str) -> bool:
    """Whether a loop's back-edge branch tests only a counter the loop steps
    by a constant (a ``for`` over a runtime trip count), not a loaded value."""
    lines = [line.strip() for line in loop.splitlines() if line.strip()]
    branches = [i for i, line in enumerate(lines) if line.startswith("s_cbranch_scc")]
    if not branches:
        return False
    compares = [line for line in lines[: branches[-1]] if re.match(r"s_cmp\w*\s", line)]
    if not compares:
        return False
    operands = [op.strip() for op in compares[-1].split(None, 1)[1].split(",")]
    for operand in operands:
        step = rf"s_(?:add|sub)k?_[iu]32 {re.escape(operand)},(?: {re.escape(operand)},)? -?(?:0x)?[0-9a-f]+$"
        if any(re.match(step, line) for line in lines):
            return True
    return False


def persistent_codegen_problems(assembly: str, *, entry: str | None) -> list[str]:
    """Return the ordering rules a persistent kernel's assembly breaks.

    - Nothing invalidates or writes back L2.
    - With ``entry`` (a kernel that runs the launch entry, in the form
      ``enter_check`` takes; None for a kernel without it), one check per
      inlined copy of the entry, and nothing waits for the role's loads
      issued before the entry until the claim:
      - ``"vector"``: the epoch store (``global_store_dwordx2``) is the
        first store, and the returning claim exchange the next one.
    - A store to the address of an earlier global load with a scalar base
      (a per-workgroup word such as the epoch) follows a full drain and a
      workgroup barrier after that load.
    - Every join arrival follows a full drain and a workgroup barrier.
    - Every poll (an innermost loop that reloads with ``sc1``, or with a
      scalar ``glc`` load, runs no MFMA, and is not counted: its back edge
      tests a loaded value, not a constant-step counter) reads the clock for its
      deadline and holds no workgroup barrier or LDS exchange; a vector poll
      exits on lane-mask votes or polls from one lane.
    - A vector flag poll (``sc1`` without ``sc0``; Lamport polls read
      their data with ``sc0 sc1``) reaches a workgroup barrier before its
      next load. A scalar poll runs in every wave and may end with a
      per-wave acquire instead.
    """
    problems = []
    if "buffer_inv" in assembly or "buffer_wbl2" in assembly:
        problems.append("L2 invalidate or write-back")
    lines = [
        line
        for line in assembly.splitlines()
        if not line.lstrip().startswith((";", ".loc"))
    ]
    writes = [
        index
        for index, line in enumerate(lines)
        if re.search(r"\b(global|buffer)_(store|atomic)\w*", line)
    ]
    if entry == "vector":
        # The entry: the epoch store, then the returning claim exchange, as
        # the first two writes of each inlined copy; no vmcnt wait before
        # the first epoch store (the exchange's own wait is the first).
        swaps = [
            position
            for position, index in enumerate(writes)
            if re.match(r"\s*global_atomic_swap_x2 v\[", lines[index])
            and " sc0" in lines[index]
        ]
        if not swaps or swaps[0] != 1:
            problems.append("the claim exchange is not the entry's second write")
        else:
            for position in swaps:
                if not re.match(
                    r"\s*global_store_dwordx2 ", lines[writes[position - 1]]
                ):
                    problems.append(
                        "the claim exchange does not follow the epoch store"
                    )
            if any("vmcnt" in line for line in lines[: writes[0]]):
                problems.append("a vmcnt wait precedes the epoch store")
    else:
        assert entry is None, entry
    problems += _unordered_read_then_write(lines)
    for position, index in enumerate(writes):
        if re.search(r"global_atomic_add\s", lines[index]) and " sc0" in lines[index]:
            # A returning add is a join arrival: every store before it drains,
            # then the work-group joins.
            previous = writes[position - 1] if position else 0
            before = "\n".join(lines[previous:index])
            if "s_waitcnt vmcnt(0)" not in before or "s_barrier" not in before:
                problems.append(f"arrival without drain and barrier: {lines[index]}")
    raw = assembly.splitlines()
    for indices in _innermost_loop_lines(assembly).values():
        loop = "\n".join(raw[i] for i in indices)
        # A poll rereads with sc1 and computes nothing; a K loop over data
        # another work-group wrote (".cv" loads) runs MFMAs.
        scalar = re.search(r"s_load_dword\w* .*\bglc\b|s_dcache_inv", loop)
        vector = re.search(r"(global|buffer)_load_\w+ .*sc1\b", loop)
        if not (scalar or vector) or "v_mfma" in loop or _counted_loop(loop):
            continue
        if not scalar and not re.search(r"(global|buffer)_load_\w+ .*\bsc0\b", loop):
            # A vector flag poll reloads in one lane (or for one wave's
            # lanes), and other waves read what the flag guards: the
            # workgroup barrier after it is their acquire. A scalar poll
            # runs in every wave, so each wave's loads after it follow its
            # own poll; it needs only a per-wave acquire, which emits no
            # instruction (and a barrier where other waves' LDS writes
            # are read).
            if _load_before_barrier(raw, set(indices)):
                problems.append("a load follows a flag poll before a barrier")
        if re.search(r"s_barrier|\bds_|_dpp", loop):
            problems.append("workgroup barrier or LDS exchange inside a poll loop")
        # A retry exits on a lane mask written by a compare (one per polled
        # array, or the one polling lane's of gl.atomic_poll), never on a
        # cross-lane reduction. The compare may write VCC (a ballot of it
        # then tests VCC with s_cmp).
        if vector and not re.search(
            r"v_cmp_\w+_e64 s\[\d+:\d+\]|v_readfirstlane_b32"
            r"|v_cmp_\w+_e32 vcc,[^\n]*\n(?:[^\n]*\n)?\s*s_cmp_lg_u64 vcc, 0",
            loop,
        ):
            problems.append("poll loop without a lane-mask exit")
        if not _clock_on_retry_path(raw, set(indices)):
            problems.append("poll loop without a deadline")
    return problems


def scalar_poll_vector_waits(assembly: str) -> list[str]:
    """The ``vmcnt`` waits on scalar polls (innermost loops that reload with
    ``s_load … glc`` or ``s_dcache_inv``) and on their retry paths.

    A scalar poll exists so that a role can wait with its vector loads in
    flight; any ``vmcnt`` wait there (for example from a vector load of the
    abort word) drains those loads.
    """
    raw = assembly.splitlines()
    waits = []
    for indices in _innermost_loop_lines(assembly).values():
        loop = set(indices)
        if not any(
            re.search(r"s_load_dword\w* .*\bglc\b|s_dcache_inv", raw[i]) for i in loop
        ):
            continue
        retry = _retry_lines(raw, loop, _successors(raw))
        waits += [raw[i].strip() for i in sorted(loop | retry) if "vmcnt" in raw[i]]
    return waits


def instruction_lines(assembly: str) -> list[str]:
    """The instruction and label lines of ``assembly``, stripped, without
    comments and directives."""
    return [
        line.strip()
        for line in assembly.splitlines()
        if re.match(r"^\.LBB", line)
        or (line.strip() and not line.lstrip().startswith((";", ".")))
    ]


# ---------------------------------------------------------------------------
# Resource reports and weight-stream rules (gfx950, 8 waves per CU)
# ---------------------------------------------------------------------------

# Unified VGPR + AGPR budget per wave with 8 waves per CU (512 per SIMD
# lane, 2 waves per SIMD).
REGISTERS_PER_WAVE_8_WAVES = 256

_WAIT_VMCNT = re.compile(r"\bs_waitcnt\b.*\bvmcnt\((\d+)\)")
_VMEM_LOAD = re.compile(r"\b(?:global|buffer)_load_(\w+)")
_VMEM_OP = re.compile(r"^(?:global|buffer|flat)_(?:load|store|atomic)\w*")
_SCRATCH = re.compile(r"\bscratch_(?:store|load)_")


def _metadata(assembly: str, name: str) -> int:
    match = re.search(rf"\.{name}:\s+(\d+)", assembly)
    return int(match[1]) if match else 0


@dataclass
class IsaReport:
    """Resource use and size of one compiled gfx950 kernel."""

    vgprs: int
    agprs: int
    sgprs: int
    scratch_bytes: int
    code_bytes: int
    instruction_count: int

    @property
    def registers(self) -> int:
        """VGPR + AGPR per wave: AGPRs start at the 4-aligned end of the VGPRs."""
        if not self.agprs:
            return self.vgprs
        return (self.vgprs + 3) // 4 * 4 + self.agprs


def isa_report(assembly: str) -> IsaReport:
    """Summarize a kernel from its code object metadata and code length."""
    code = re.search(r"codeLenInByte = (\d+)", assembly)
    return IsaReport(
        vgprs=_metadata(assembly, "vgpr_count"),
        agprs=_metadata(assembly, "agpr_count"),
        sgprs=_metadata(assembly, "sgpr_count"),
        scratch_bytes=_metadata(assembly, "private_segment_fixed_size"),
        code_bytes=int(code[1]) if code else 0,
        instruction_count=sum(
            1 for line in source_lines(assembly) if not line.text.endswith(":")
        ),
    )


def register_problems(
    report: IsaReport, budget=REGISTERS_PER_WAVE_8_WAVES
) -> list[str]:
    """Over the per-wave register budget, or any scratch (spills)."""
    problems = []
    if report.registers > budget:
        problems.append(f"{report.registers} VGPR+AGPR > {budget}")
    if report.scratch_bytes:
        problems.append(f"{report.scratch_bytes} B of scratch (register spills)")
    return problems


def banned_instruction_problems(assembly: str) -> list[str]:
    """Instructions a persistent layer must never emit: L2 write-back and
    invalidate (microseconds each, and they drop warm lines), global wave
    sync (not supported on gfx950 although the assembler accepts it) and
    ``s_sleep`` (polls do not back off)."""
    lines = instruction_lines(assembly)
    return [
        why
        for pattern, why in (
            (r"\bbuffer_wbl2\b", "L2 write-back"),
            (r"\bbuffer_inv\b", "L2 invalidate"),
            (r"\bds_gws_", "global wave sync"),
            (r"\bs_sleep\b", "s_sleep"),
        )
        if any(re.search(pattern, line) for line in lines)
    ]


def _waits_zero(line: str) -> bool:
    match = _WAIT_VMCNT.search(line)
    return bool(match) and match[1] == "0"


def serialized_coherent_loads(lines: list[str]) -> list[str]:
    """Coherent (``sc0 sc1``) loads each followed by ``vmcnt(0)``, back to
    back: how ``gl.load(..., cache_modifier=".cv")`` lowers (as a volatile
    load). Each wait is a memory round trip and drains every older load of
    the wave. ``lines``: :func:`instruction_lines`, or a selection of them."""
    chain = longest = 0
    index = 0
    while index < len(lines):
        line = lines[index]
        following = lines[index + 1] if index + 1 < len(lines) else ""
        if _VMEM_LOAD.search(line) and " sc0" in line and " sc1" in line:
            if _waits_zero(following):
                chain += 1
                longest = max(longest, chain)
                index += 2
                continue
        if not line.startswith("s_waitcnt"):
            chain = 0
        index += 1
    return [f"{longest} coherent loads serialized by vmcnt(0)"] if longest >= 2 else []


def write_through_store_drains(lines: list[str]) -> list[str]:
    """Write-through stores (``sc0 sc1``) followed at once by ``vmcnt(0)``:
    ``gl.store(..., cache_modifier=".wt")``, a volatile store whose wait
    drains every older load of the wave (``buffer_store(cache=".wt")`` emits
    the store alone)."""
    count = sum(
        1
        for line, following in zip(lines, lines[1:])
        if re.match(r"(global|buffer)_store", line)
        and " sc0" in line
        and " sc1" in line
        and _waits_zero(following)
    )
    return [f"{count} write-through stores drained by vmcnt(0)"] if count else []


def stream_loop_problems(loop: str, min_in_flight: int) -> list[str]:
    """Rules for an innermost loop that streams weights into MFMAs (``loop``
    as :func:`innermost_loops` returns it): every ``vmcnt(N)`` keeps at
    least ``min_in_flight - 1`` loads in flight (no ``vmcnt(0)``), there is
    no workgroup barrier, every stream load is non-temporal (``nt``) and
    none is narrower than a dword (64 B per wave instruction)."""
    lines = [line.split(";", 1)[0].strip() for line in loop.splitlines()]
    loads = [line for line in lines if _VMEM_LOAD.search(line)]
    problems = []
    waits = [int(m[1]) for line in lines if (m := _WAIT_VMCNT.search(line))]
    if waits and min(waits) < min_in_flight - 1:
        problems.append(f"stream loop waits vmcnt({min(waits)})")
    if "s_barrier" in lines:
        problems.append("s_barrier in a stream loop")
    if any(re.search(r"_load_(?:u|s)?(?:byte|short)", line) for line in loads):
        problems.append("sub-dword loads in a stream loop")
    if not all(re.search(r"\bnt\b", line) for line in loads):
        problems.append("stream load without nt")
    return problems


def store_loop_preheader_drains(assembly: str) -> list[str]:
    """A loop of stores (no loads) entered right after ``vmcnt(0)``.

    LLVM flushes the vector counter before a store-only loop that uses a
    value loaded before it (gfx9 has no separate store counter); the flush
    also waits for every weight load issued earlier. Unrolled pushes avoid
    the loop."""
    lines = assembly.splitlines()
    problems = []
    for indices in _innermost_loop_lines(assembly).values():
        body = [lines[i] for i in indices]
        if not any(re.search(r"\b(global|buffer)_store", b) for b in body) or any(
            _VMEM_LOAD.search(b) for b in body
        ):
            continue
        before = []
        for line in reversed(lines[: min(indices)]):
            text = line.split(";", 1)[0].strip()
            if text and not text.startswith("."):
                before.append(text)
            if len(before) == 3:
                break
        if any(_waits_zero(text) for text in before):
            problems.append("vmcnt(0) before a store-only loop")
    return problems


def split_poll_problems(assembly: str) -> list[str]:
    """Poll loops that reload with more than one coherent load per retry,
    each behind its own ``vmcnt(0)`` (a volatile 16-byte load of a [lane,
    pair, word] tensor is split into two ``dwordx2`` loads): poll the tag
    dword instead."""
    problems = []
    for loop in innermost_loops(assembly):
        lines = [line.split(";", 1)[0].strip() for line in loop.splitlines()]
        if any(line.startswith("v_mfma") for line in lines):
            continue
        drained = sum(
            1
            for line, following in zip(lines, lines[1:])
            if _VMEM_LOAD.search(line)
            and " sc0" in line
            and " sc1" in line
            and _waits_zero(following)
        )
        if drained >= 2:
            problems.append(f"poll retry with {drained} serialized coherent loads")
    return problems


def lds_read_widths(lines: list[str]) -> dict[str, int]:
    """Count LDS reads by mnemonic (an exchange of lane-contiguous partials
    should read ``ds_read_b128``, not ``ds_read2_b32`` pairs)."""
    counts: dict[str, int] = {}
    for line in lines:
        match = re.match(r"(ds_read\w*)", line)
        if match:
            counts[match[1]] = counts.get(match[1], 0) + 1
    return counts


def drains_of(lines: list[str], first: int, last: int) -> list[int]:
    """Indices in ``lines[first + 1 : last]`` of the ``s_waitcnt vmcnt(N)``
    that wait for the vector memory operation at ``first``, following the
    text in order: vector operations retire in issue order, so ``vmcnt(N)``
    completes every operation older than the N newest, and the one at
    ``first`` once N is at most the count issued after it."""
    newer = 0
    drains = []
    for index in range(first + 1, min(last, len(lines))):
        line = lines[index]
        if _VMEM_OP.match(line):
            newer += 1
        match = _WAIT_VMCNT.search(line)
        if match and int(match[1]) <= newer:
            drains.append(index)
    return drains


@dataclass
class SourceLine:
    """One instruction (or label) line of the assembly with the source frames
    it came from: ``(file name, line)`` pairs, innermost first, from the last
    ``.loc`` before it and the inlining chain in that line's comment.
    ``index`` is the line's index in ``assembly.splitlines()``."""

    index: int
    text: str
    frames: list[tuple[str, int]]


def source_lines(assembly: str) -> list[SourceLine]:
    """The kernel's instruction and label lines (as :func:`instruction_lines`
    selects them, without trailing comments, and up to the end of the
    function, not the metadata after it) with their source frames. File names
    are base names, as the ``.loc`` comments print them."""
    result = []
    frames: list[tuple[str, int]] = []
    for index, line in enumerate(assembly.splitlines()):
        if line.startswith(".Lfunc_end"):
            # The code object metadata follows the kernel's code.
            break
        loc = re.match(r"^\s*\.loc\s+\d+\s+\d+\s+\d+.*?;\s*(.*)$", line)
        if loc:
            frames = [
                (name, int(number))
                for name, number in re.findall(r"([\w.]+\.py):(\d+):\d+", loc[1])
            ]
            continue
        if re.match(r"^\.LBB", line) or (
            line.strip() and not line.lstrip().startswith((";", "."))
        ):
            result.append(SourceLine(index, line.split(";", 1)[0].strip(), frames))
    return result


def call_site(line: SourceLine, file_name: str, first: int, last: int) -> int | None:
    """The outermost frame of ``line`` in ``file_name`` within source lines
    ``first..last`` (for example a kernel function's body): the call site
    inside that function the instruction belongs to."""
    sites = [n for f, n in line.frames if f == file_name and first <= n <= last]
    return sites[-1] if sites else None


def frame_functions(assembly: str):
    """Return ``names(frames) -> list[str]``: the Python function names of a
    line's frames, innermost first, from the source files the ``.file``
    directives name (functions found with :mod:`ast`)."""
    import ast
    from pathlib import Path

    spans: dict[str, list[tuple[int, int, str]]] = {}
    for directory, name in re.findall(
        r'^\s*\.file\s+\d+\s+"([^"]*)"\s+"([^"]*)"', assembly, re.M
    ):
        if name in spans or not name.endswith(".py"):
            continue
        path = Path(directory) / name
        if not path.exists():
            continue
        tree = ast.parse(path.read_text())
        spans[name] = [
            (node.lineno, node.end_lineno, node.name)
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]

    def function(name: str, line: int) -> str:
        # The innermost (latest-starting) definition that spans the line.
        best = max(
            (span for span in spans.get(name, ()) if span[0] <= line <= span[1]),
            default=None,
        )
        return best[2] if best else "?"

    def names(frames: list[tuple[str, int]]) -> list[str]:
        return [function(name, line) for name, line in frames]

    return names
