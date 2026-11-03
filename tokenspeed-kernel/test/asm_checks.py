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
    """Map each innermost loop's header to the indices of its lines."""
    loops: dict[str, list[int]] = {}
    current = None
    for index, line in enumerate(assembly.splitlines()):
        label = re.match(r"^\.L(BB\w+):(.*)", line)
        if label:
            name, comment = label.groups()
            if "=>This Inner Loop Header" in comment:
                current = name
            else:
                header = re.search(r"in Loop: Header=(BB\w+)", comment)
                current = header[1] if header else None
            if current is not None:
                loops.setdefault(current, [])
        if current is not None and current in loops:
            loops[current].append(index)
    return loops


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


def _load_before_barrier(lines: list[str], loop: set[int]) -> bool:
    """Whether a path leaving the loop (its line indices) reaches a vector
    memory load before a workgroup barrier, following branches."""
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

    # Exits: successors outside the loop of the loop lines its header reaches
    # (lines after an unconditional branch only start other blocks' code).
    inside, todo = set(), [min(loop)]
    while todo:
        index = todo.pop()
        if index in inside or index not in loop:
            continue
        inside.add(index)
        todo.extend(successors(index))
    seen: set[int] = set()
    todo = [n for i in inside for n in successors(i) if n not in loop]
    while todo:
        index = todo.pop()
        if index in seen or index in loop:
            continue
        seen.add(index)
        line = lines[index]
        if "s_barrier" in line:
            continue
        if re.search(r"\b(global|buffer)_load", line):
            return True
        todo.extend(successors(index))
    return False


def persistent_codegen_problems(assembly: str) -> list[str]:
    """Return the ordering rules a persistent kernel's assembly breaks.

    - Nothing invalidates or writes back L2.
    - Only the epoch store precedes the claim exchange, and the claim's
      compare-and-branch completes before any other store or atomic.
    - Every join arrival follows a full drain and a workgroup barrier.
    - Every poll (an innermost loop that reloads with ``sc1`` and runs no
      MFMA) exits on lane-mask votes or polls from one lane, reads the clock
      for its deadline, and holds no workgroup barrier or LDS exchange.
    - A flag poll (``sc1`` without ``sc0``; Lamport polls read their data
      with ``sc0 sc1``) reaches a workgroup barrier before its next load.
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
    swap = next(i for i in writes if "atomic_swap_x2" in lines[i])
    if sum(1 for i in writes if i < swap) != 1:
        problems.append("a store or atomic other than the epoch precedes the claim")
    after = next(i for i in writes if i > swap)
    between = "\n".join(lines[swap:after])
    if "s_waitcnt vmcnt(0)" not in between or "s_cbranch" not in between:
        problems.append(
            "the claim's compare-and-branch does not precede the next store"
        )
    for position, index in enumerate(writes):
        if "global_atomic_add" in lines[index] and " sc0" in lines[index]:
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
        if not re.search(r"(global|buffer)_load_\w+ .*sc1\b", loop) or "v_mfma" in loop:
            continue
        if not re.search(r"(global|buffer)_load_\w+ .*\bsc0\b", loop):
            # The barrier after a flag poll is the acquire that orders the
            # payload loads after it.
            if _load_before_barrier(raw, set(indices)):
                problems.append("a load follows a flag poll before a barrier")
        if re.search(r"s_barrier|\bds_|_dpp", loop):
            problems.append("workgroup barrier or LDS exchange inside a poll loop")
        # A retry exits on a lane mask written by a compare (one per polled
        # array, or the one polling lane's of gl.atomic_poll), never on a
        # cross-lane reduction.
        if not re.search(r"v_cmp_\w+_e64 s\[\d+:\d+\]|v_readfirstlane_b32", loop):
            problems.append("poll loop without a lane-mask exit")
        if "s_memrealtime" not in loop:
            problems.append("poll loop without a deadline")
    return problems
