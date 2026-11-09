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


"""Checks on inline asm in compiled gfx950 kernels.

The compiler places the wait states between dependent instructions and keeps
values out of the registers an instruction writes only for code it can see.
It sees neither inside an inline asm block, so kernel codegen tests run these
checks on every compiled variant that contains one:

- :func:`inline_asm_helpers`: which source helper each asm block came from,
  so a test can require every block to come from a listed helper;
- :func:`undeclared_asm_writes`: an asm template (from the LLVM IR) that
  writes a register it does not declare (including the implicit SCC and
  VCC), or writes an output that is not early clobber before its last input
  read;
- :func:`inline_asm_input_hazards`: an asm instruction that reads an MFMA
  result, a transcendental result or a VALU-written SGPR before the wait
  states its producer needs (CDNA4 ISA 4.5) have passed.

Pure functions of the assembly and LLVM IR text: no device and no Triton
import. The source attribution reads the Python files the ``.file``
directives name.
"""

from __future__ import annotations

import ast
import collections
import re
from pathlib import Path


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
        # No fall-through past an unconditional branch, the end of the
        # kernel or a function's return.
        if not re.search(r"\bs_(?:branch|endpgm|setpc_b64)\b", line):
            nexts.append(index + 1)
        return [n for n in nexts if n < len(lines)]

    return successors


def frame_functions(assembly: str):
    """Return ``names(frames) -> list[str]``: the Python function names of a
    line's frames, innermost first, from the source files the ``.file``
    directives name (functions found with :mod:`ast`)."""
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


_IR_ASM = re.compile(
    r'\basm (?:sideeffect )?(?:alignstack )?"((?:[^"\\]|\\.)*)", "((?:[^"\\]|\\.)*)"'
)
# SALU instructions that leave SCC alone; every other s_* ALU op writes it.
_SCC_KEEPERS = re.compile(
    r"s_(?:mov|movk|cmov|cselect|load|store|buffer|waitcnt|nop|cbranch|branch"
    r"|getreg|setreg|getpc|sleep|memrealtime|memtime|dcache|barrier|mul_i32|sendmsg"
    r"|trap|setprio|endpgm)"
)
# Instructions whose first operand is read, not written.
_NO_DESTINATION = re.compile(
    r"(?:s_cmp|s_bitcmp|s_cbranch|s_branch|s_waitcnt|s_nop|s_dcache|s_barrier"
    r"|s_sleep|s_trap|s_sendmsg|s_setprio|s_store|ds_write|ds_add_u32"
    r"|buffer_store|global_store)"
)


def _unescape_ir(text: str) -> str:
    return re.sub(r"\\([0-9A-Fa-f]{2})", lambda m: chr(int(m[1], 16)), text)


def inline_asm_templates(llir: str) -> set[tuple[str, str]]:
    """The ``(template, constraints)`` of every inline asm call in LLVM IR,
    with the IR's escapes (``\\0A``, ``\\09``) undone."""
    return {
        (_unescape_ir(template), _unescape_ir(constraints))
        for template, constraints in _IR_ASM.findall(llir)
    }


def _register_names(operand: str) -> set[str]:
    """Physical registers an operand names (``s[64:79]`` -> s64..s79)."""
    span = re.fullmatch(r"([sva])\[(\d+):(\d+)\]", operand)
    if span:
        return {f"{span[1]}{i}" for i in range(int(span[2]), int(span[3]) + 1)}
    if re.fullmatch(r"[sva]\d+|scc|vcc|exec|m0", operand):
        return {operand}
    return set()


def undeclared_asm_writes(template: str, constraints: str) -> list[str]:
    """Registers an inline asm template writes without declaring them.

    An operand ``$k`` may be written only if it is an output (``=``) or an
    input tied to one (a digit constraint). A named register (``s80``,
    ``vcc``, ``scc``, implicit SCC of SALU arithmetic) must be in the
    ``~{...}`` clobbers. An output without early clobber (``=&``) must not be
    written before the template's last read of an input, or the compiler may
    give it the input's register."""
    parts = [part.strip() for part in constraints.split(",")]
    outputs = [part for part in parts if part.startswith("=")]
    inputs = [part for part in parts if not part.startswith(("=", "~"))]
    clobbered: set[str] = set()
    for part in parts:
        if part.startswith("~{") and part.endswith("}"):
            clobbered |= _register_names(part[2:-1]) or {part[2:-1]}
    instructions = []
    for line in template.splitlines():
        line = line.split(";")[0].strip()
        if not line or re.fullmatch(r"\w+:", line):
            continue
        mnemonic, _, rest = line.partition(" ")
        operands = [o.strip() for o in rest.split(",") if o.strip()]
        instructions.append((mnemonic, operands))
    problems = []
    first_write: dict[int, int] = {}
    last_input_read = -1
    for index, (mnemonic, operands) in enumerate(instructions):
        writes = []
        if operands and not _NO_DESTINATION.match(mnemonic):
            writes.append(operands[0])
        if mnemonic.startswith("s_") and not _SCC_KEEPERS.match(mnemonic):
            writes.append("scc")
        if re.match(r"v_(?:add|sub|subrev)_co_|v_(?:addc|subb)", mnemonic):
            writes.append("vcc")
        if mnemonic.startswith("v_cmpx"):
            writes.append("exec")
        reads = operands[1:] if writes[:1] == operands[:1] else operands
        for operand in reads:
            number = re.fullmatch(r"\$(\d+)", operand)
            if number and int(number[1]) >= len(outputs):
                last_input_read = index
        for operand in writes:
            number = re.fullmatch(r"\$(\d+)", operand)
            if number:
                k = int(number[1])
                if k < len(outputs):
                    first_write.setdefault(k, index)
                elif not inputs[k - len(outputs)].isdigit():
                    problems.append(f"{mnemonic} writes input ${k}")
                continue
            names = _register_names(operand)
            if names and not names <= clobbered:
                problems.append(f"{mnemonic} writes undeclared {operand}")
    for k, index in first_write.items():
        if "&" not in outputs[k] and index < last_input_read:
            problems.append(f"output ${k} written before the last input read")
    return problems


# Wait states an inline asm read needs after its producer (gfx950). The
# compiler places these for its own instructions; it does not look inside
# inline asm, so an asm instruction must not read a value this fresh.
_MFMA_READ_WAIT_STATES = 19  # 16-pass XDL write, then a VALU or memory read
_TRANS = re.compile(r"v_(?:exp|log|rcp|rsq|sqrt|sin|cos)_")


def _asm_operands(text: str) -> tuple[str, list[str]]:
    mnemonic, _, rest = text.partition(" ")
    return mnemonic, [o.strip().split()[0] for o in rest.split(",") if o.strip()]


def inline_asm_input_hazards(assembly: str) -> list[tuple[str, str]]:
    """Inline asm instructions that read a register before its producer's
    hazard window has passed, as ``(helper, problem)`` with the helper the
    asm came from: an MFMA result; a transcendental's result read by a VALU;
    an SGPR written by a VALU and read by a memory instruction (5 wait
    states), a VALU (2) or a lane read/write as its lane select (4).

    Producers are found through the control flow, ``s_nop N`` counting N + 1
    wait states."""
    raw = assembly.splitlines()
    # Every function of the code object (Triton keeps noinline helpers as
    # functions of their own); the metadata after the last one is not code.
    ends = [i for i, l in enumerate(raw) if l.startswith(".Lfunc_end")]
    end = ends[-1] if ends else len(raw)
    lines = [l.split(";")[0].rstrip() for l in raw[:end]]
    successors = _successors(lines)
    predecessors = collections.defaultdict(list)
    for index in range(len(lines)):
        for later in successors(index):
            predecessors[later].append(index)
    names = frame_functions(assembly)
    inside: dict[int, int] = {}
    start = None
    for index, line in enumerate(raw[:end]):
        if ";;#ASMSTART" in line:
            start = index
        elif ";;#ASMEND" in line:
            start = None
        elif start is not None and lines[index].strip():
            inside[index] = start
    problems = []
    for index, block in sorted(inside.items()):
        mnemonic, operands = _asm_operands(lines[index].strip())
        if not mnemonic or mnemonic.endswith(":"):
            continue
        writes = bool(operands) and not _NO_DESTINATION.match(mnemonic)
        consumer = (
            "memory" if re.match(r"(?:buffer|global|flat|ds)_", mnemonic)
            else "lane" if mnemonic in ("v_readlane_b32", "v_writelane_b32")
            else "valu" if mnemonic.startswith("v_")
            else "salu"
        )  # fmt: skip
        read = operands[1:] if writes else operands
        for register in sorted(
            set().union(*map(_register_names, read)) if read else ()
        ):
            todo = [(p, 0) for p in predecessors[index]]
            seen = set()
            while todo:
                at, waits = todo.pop()
                if (at, waits) in seen or waits > _MFMA_READ_WAIT_STATES:
                    continue
                seen.add((at, waits))
                producer, outs = _asm_operands(lines[at].strip())
                nop = re.match(r"s_nop (\d+)", producer + " " + " ".join(outs))
                step = (
                    int(nop[1]) + 1
                    if nop
                    else (
                        1
                        if producer
                        and not producer.endswith(":")
                        and not producer.startswith(".")
                        else 0
                    )
                )
                if (
                    outs
                    and not _NO_DESTINATION.match(producer)
                    and register in _register_names(outs[0])
                ):
                    need = 0
                    sgpr = register.startswith("s")
                    if "mfma" in producer:
                        need = _MFMA_READ_WAIT_STATES
                    elif _TRANS.match(producer) and consumer == "valu":
                        need = 1
                    elif producer.startswith("v_") and sgpr:
                        need = {"memory": 5, "valu": 2, "lane": 4}.get(consumer, 0)
                    if waits < need:
                        problems.append((
                            _asm_block_helper(raw, block, names),
                            f"{lines[index].strip()} reads {register} {waits} wait"
                            f" states after {lines[at].strip()} (needs {need})",
                        ))  # fmt: skip
                    continue
                todo.extend((p, waits + step) for p in predecessors[at])
    return problems


def _asm_block_helper(raw: list[str], start: int, names) -> str:
    """The source helper of the inline asm block whose ``;;#ASMSTART`` is
    line ``start``: the innermost function of the ``.loc`` before it."""
    for line in reversed(raw[:start]):
        loc = re.match(r"^\s*\.loc\s+\d+\s+\d+\s+\d+.*?;\s*(.*)$", line)
        if loc:
            frames = [
                (name, int(number))
                for name, number in re.findall(r"([\w.]+\.py):(\d+):\d+", loc[1])
            ]
            return (names(frames)[:1] or ["?"])[0]
    return "?"


def inline_asm_helpers(assembly: str) -> dict[str, int]:
    """How many inline asm blocks each source helper left in the assembly
    (the innermost function of the ``.loc`` before the block)."""
    names = frame_functions(assembly)
    raw = assembly.splitlines()
    counts: dict[str, int] = {}
    for index, line in enumerate(raw):
        if ";;#ASMSTART" in line:
            helper = _asm_block_helper(raw, index, names)
            counts[helper] = counts.get(helper, 0) + 1
    return counts
