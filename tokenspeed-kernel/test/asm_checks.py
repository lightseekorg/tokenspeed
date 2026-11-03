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
offline compile tests can use them on hosts without a GPU.
"""

from __future__ import annotations

import re


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


def persistent_codegen_problems(assembly: str, *, entry: bool) -> list[str]:
    """Return the ordering rules a persistent kernel's assembly breaks.

    - Nothing invalidates or writes back L2.
    - With ``entry`` (a kernel that runs the launch entry): the two
      returning counter increments are its first atomics, nothing is stored
      before the claim compare, and the compare waits only for the counters,
      not for the role's loads issued after them.
    - A store to the address of an earlier global load with a scalar base
      (a per-workgroup word such as the epoch) follows a full drain and a
      workgroup barrier after that load.
    - Every join arrival follows a full drain and a workgroup barrier.
    - Every poll (an innermost loop that reloads with ``sc1``, or with a
      scalar ``glc`` load, runs no MFMA, and is not counted: its back edge
      tests a loaded value, not a constant-step counter) reads the clock for its
      deadline and holds no workgroup barrier or LDS exchange; a vector poll
      exits on lane-mask votes or polls from one lane.
    - A flag poll (scalar, or ``sc1`` without ``sc0``; Lamport polls read
      their data with ``sc0 sc1``) reaches a workgroup barrier before its
      next load.
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
    if entry:
        # Entry v2: the two returning counter increments come first, the
        # role's first loads after them, and the claim compare waits only for
        # the counters (vmcnt(k), k = loads issued after them).
        counters = [i for i in writes if "global_atomic_add_x2" in lines[i]][:2]
        if len(counters) != 2 or any(i < counters[0] for i in writes):
            problems.append("the entry counters are not the first atomics")
        else:
            loads = [
                i
                for i in range(counters[1], len(lines))
                if re.match(r"\s*(buffer|global)_load", lines[i])
            ]
            check = next(
                (
                    i
                    for i in range(counters[1], len(lines))
                    if "v_cmp_ne_u64_e64" in lines[i]
                ),
                None,
            )
            if check is None:
                problems.append("no claim compare after the entry counters")
            else:
                if any(counters[0] < i < check and i not in counters for i in writes):
                    problems.append("a store or atomic precedes the claim compare")
                issued = sum(1 for i in loads if i < check)
                waits = [
                    int(m[1])
                    for line in lines[counters[1] : check]
                    if (m := re.search(r"s_waitcnt vmcnt\((\d+)\)", line))
                ]
                if not waits or min(waits) < issued:
                    problems.append(
                        f"the claim waits for the role's loads: vmcnt {waits}, "
                        f"{issued} loads issued after the counters"
                    )
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
        scalar = re.search(r"s_load_dword\w* .*\bglc\b", loop)
        vector = re.search(r"(global|buffer)_load_\w+ .*sc1\b", loop)
        if not (scalar or vector) or "v_mfma" in loop or _counted_loop(loop):
            continue
        if scalar or not re.search(r"(global|buffer)_load_\w+ .*\bsc0\b", loop):
            # The barrier after a flag poll is the acquire that orders the
            # payload loads after it.
            if _load_before_barrier(raw, set(indices)):
                problems.append("a load follows a flag poll before a barrier")
        if re.search(r"s_barrier|\bds_|_dpp", loop):
            problems.append("workgroup barrier or LDS exchange inside a poll loop")
        # A retry exits on a lane mask written by a compare (one per polled
        # array, or the one polling lane's of gl.atomic_poll), never on a
        # cross-lane reduction.
        if vector and not re.search(
            r"v_cmp_\w+_e64 s\[\d+:\d+\]|v_readfirstlane_b32", loop
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
