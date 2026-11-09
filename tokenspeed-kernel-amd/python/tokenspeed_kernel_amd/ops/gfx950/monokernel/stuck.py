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

"""Stuck-poll records (see ``mailbox``): the host-visible buffer a launch's
polls write into when they give up, and their decoding.

A record is ``mailbox.STUCK_WORDS`` int64 words a CTA: the expected tag, the
stale pair as observed, the pair's address, the realtime clock at the
poll's first check and at the trap, the stale operand's index in the poll,
and its CTA's record offset. A record whose tag word is zero is empty (a tag is
never zero).

The buffer is pinned host memory, or a file mapped and registered as host
memory (``path``): the GPU's system-scope stores land in the file's pages,
so a record outlives a process that the trap took down. Its row ``grid``
(past the CTAs' records) is a header the host writes: a magic number, the
scratch buffer's address, the rank and the process id, so a file decodes
on its own (with the plan).
"""

from __future__ import annotations

import ctypes
import mmap
import os
from dataclasses import dataclass

import torch
from tokenspeed_kernel_amd.ops.gfx950.monokernel.mailbox import STUCK_WORDS
from tokenspeed_kernel_amd.ops.gfx950.monokernel.plan import LAYER_SLOTS, Plan

# The realtime clock that the polls read (s_memrealtime): 100 MHz on gfx950.
TICKS_PER_SECOND = 100_000_000
# The header row's first word: b"STUCKREC".
MAGIC = int.from_bytes(b"STUCKREC", "little")


@dataclass
class StuckBuffer:
    """A stuck-poll record buffer: ``tensor`` (int64 ``[grid, STUCK_WORDS]``,
    CPU, device-accessible) and, for a file-backed buffer, its ``path``."""

    tensor: torch.Tensor
    path: str | None
    _map: mmap.mmap | None


def stuck_buffer(
    grid: int, path: str | None, scratch_ptr: int, rank: int
) -> StuckBuffer:
    """A zeroed record buffer for launches of up to ``grid`` CTAs with the
    header row (``scratch_ptr``: the mailbox scratch's address, ``rank``):
    pinned host memory, or (``path``) a file of that name mapped and
    registered as host memory, which keeps the records after the process
    dies. Pass ``tensor`` to the launches whole."""
    rows = grid + 1
    nbytes = rows * STUCK_WORDS * 8
    if path is None:
        t = torch.zeros((rows, STUCK_WORDS), dtype=torch.int64, pin_memory=True)
        t[grid, :4] = torch.tensor([MAGIC, scratch_ptr, rank, os.getpid()])
        return StuckBuffer(t, None, None)
    with open(path, "wb") as f:
        f.truncate(nbytes)
    fd = os.open(path, os.O_RDWR)
    try:
        m = mmap.mmap(fd, nbytes, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
    finally:
        os.close(fd)
    t = torch.frombuffer(m, dtype=torch.int64).view(rows, STUCK_WORDS)
    t[grid, :4] = torch.tensor([MAGIC, scratch_ptr, rank, os.getpid()])
    ptr = ctypes.c_void_p(t.data_ptr())
    err = torch.cuda.cudart().cudaHostRegister(ptr.value, nbytes, 0)
    if int(err) != 0:
        raise RuntimeError(f"cannot register {path} as host memory: error {int(err)}")
    return StuckBuffer(t, path, m)


def close(buf: StuckBuffer, remove: bool) -> None:
    """Unregister and unmap a file-backed buffer; with ``remove``, also
    delete its file (a run without a trap)."""
    if buf.path is None:
        return
    torch.cuda.cudart().cudaHostUnregister(buf.tensor.data_ptr())
    buf.tensor = torch.empty(0, dtype=torch.int64)
    buf._map.close()
    if remove:
        os.remove(buf.path)


def read_file(path: str) -> torch.Tensor:
    """The records of a file-backed buffer, read after the process died."""
    with open(path, "rb") as f:
        data = bytearray(f.read())
    return torch.frombuffer(data, dtype=torch.int64).view(-1, STUCK_WORDS).clone()


def header(t: torch.Tensor) -> dict:
    """The header row of buffer ``t``: ``scratch_ptr``, ``rank``, ``pid``."""
    w = t[-1].tolist()
    if w[0] != MAGIC:
        raise ValueError("not a stuck-poll record buffer")
    return {"scratch_ptr": w[1], "rank": w[2], "pid": w[3]}


def records(t: torch.Tensor) -> list[dict]:
    """The non-empty records of buffer ``t``, by CTA."""
    out = []
    for cta, w in enumerate(t[:-1].tolist()):
        if w[0] == 0:
            continue
        out.append(
            {
                "cta": cta,
                "expected": w[0] & 0xFFFFFFFF,
                "observed": (w[1] >> 32) & 0xFFFFFFFF,
                "payload": w[1] & 0xFFFFFFFF,
                "address": w[2] & 0xFFFFFFFFFFFFFFFF,
                "start": w[3],
                "trap": w[4],
                "operand": w[5],
            }
        )
    return out


def tag_launch(tag: int) -> tuple[int, int]:
    """``(step, layer)`` of a launch tag (``plan.launch_tag``), modulo 2**32
    of the step's tags; a zero tag (a pair never written) gives (-1, -1)."""
    if tag == 0:
        return -1, -1
    return (tag - 1) // LAYER_SLOTS, (tag - 1) % LAYER_SLOTS


def locate(plan: Plan, pair: int) -> tuple[str, int, int, int] | None:
    """``(region, row, index, copy)`` of scratch pair ``pair`` in ``plan``'s
    layout, or None outside it."""
    if not 0 <= pair < plan.scratch_pairs:
        return None
    row, at = divmod(pair, plan.row_pairs)
    for name, _ in plan.regions:
        start = plan.offset(name)
        size = plan.peer_copy(name) if plan.is_peer(name) else None
        span = 2 * size if size is not None else dict(plan.regions)[name]
        if start <= at < start + span:
            if size is None:
                return name, row, at - start, 0
            return name, row, (at - start) % size, (at - start) // size
    return None


def stages_at(plan: Plan, cta: int, region: str) -> list[str]:
    """The stages of ``plan`` that read ``region`` and run a task on CTA
    ``cta`` (tasks wrap the grid from the stage's base)."""
    return [
        s.name
        for s in plan.stages
        if region in s.reads
        and ((cta - s.base) % plan.grid < s.tasks or s.tasks > plan.grid)
    ]


def tasks_on(plan: Plan, cta: int) -> list[str]:
    """``stage#task`` of every task of ``plan`` that runs on CTA ``cta``
    (tasks wrap the grid from the stage's base)."""
    out = []
    for s in plan.stages:
        for task in range((cta - s.base) % plan.grid, s.tasks, plan.grid):
            out.append(f"{s.name}#{task}")
    return out


def _where(rec: dict, plan: Plan | None, scratch_ptr: int):
    """``(region, row, index, copy)`` of the pair record ``rec`` waited
    for, or None outside ``plan``'s scratch (or with no plan)."""
    pair, rem = divmod(rec["address"] - scratch_ptr, 8)
    return None if rem or plan is None else locate(plan, pair)


def describe(rec: dict, plan: Plan | None, scratch_ptr: int, rank: int) -> str:
    """One line on record ``rec``: where its poll waited and for what. When
    no stage of ``plan`` that reads the region runs on the record's CTA (a
    record decoded under another launch's layout, say), the line names the
    CTA's tasks and the plan instead of a stage."""
    step, layer = tag_launch(rec["expected"])
    ostep, olayer = tag_launch(rec["observed"])
    where = _where(rec, plan, scratch_ptr)
    if where is None:
        place = f"address {rec['address']:#x} (outside the scratch)"
        stage = "?"
    else:
        region, row, index, copy = where
        place = f"region {region} row {row} pair {index}" + (
            f" (copy {copy})" if plan.is_peer(region) else ""
        )
        stage = "/".join(stages_at(plan, rec["cta"], region)) or (
            f"? (no stage that reads {region} runs on CTA {rec['cta']}; its "
            f"tasks under the plan of {plan.rows} rows, chain {plan.chain}, "
            f"{plan.row_pairs} pairs a row: "
            f"{', '.join(tasks_on(plan, rec['cta'])) or 'none'})"
        )
    waited = (rec["trap"] - rec["start"]) / TICKS_PER_SECOND
    seen = (
        "never written"
        if rec["observed"] == 0
        else f"tag {rec['observed']} (step {ostep} layer {olayer}), payload {rec['payload']:#x}"
    )
    return (
        f"rank {rank} CTA {rec['cta']}: stage {stage} waited for "
        f"{place}: expected tag {rec['expected']} (step {step} layer {layer}, "
        f"parity {layer & 1}), saw {seen}; waited {waited:.1f} s from clock "
        f"{rec['start']} (operand {rec['operand']})"
    )


def _plans(rec: dict, plan_of) -> list[Plan | None]:
    _, layer = tag_launch(rec["expected"])
    return plan_of(layer) or [None]


def decode(t: torch.Tensor, plan_of) -> list[str]:
    """:func:`describe` of every record of buffer ``t``, earliest wait first.
    ``plan_of(layer)`` gives the plans a launch of decoder layer ``layer``
    may have run (one per row count): a record is described under each
    whose layout holds its address."""
    h = header(t)
    return [
        describe(rec, plan, h["scratch_ptr"], h["rank"])
        for rec in sorted(records(t), key=lambda r: r["start"])
        for plan in _plans(rec, plan_of)
    ]


def _ranges(values: list[int]) -> str:
    """``1-4, 7`` of sorted distinct ``values``."""
    out, run = [], []
    for v in sorted(set(values)):
        if run and v != run[-1] + 1:
            out.append(run)
            run = []
        run.append(v)
    if run:
        out.append(run)
    return ", ".join(f"{r[0]}-{r[-1]}" if len(r) > 1 else f"{r[0]}" for r in out)


def summarize(t: torch.Tensor, plan_of) -> list[str]:
    """:func:`decode` with the records of one wait grouped: the same plan,
    stage, region, copy and tags (any rows). A group prints its earliest
    record, then how many CTAs waited there in all, on which rows and
    pairs.

    Groups come in plan order of their stages, then of their first check:
    by the deadlock rule (``Plan``) the earliest stage that waits reads only
    regions that earlier stages, on every rank, should have written, so it
    names the cause, and the later groups wait downstream of it. The first
    check alone would not: a stage that polls from the launch's start (an
    expert stage on ``route``) checks first but waits on the stall."""
    h = header(t)
    groups: dict[tuple, list] = {}
    for rec in records(t):
        for i, plan in enumerate(_plans(rec, plan_of)):
            where = _where(rec, plan, h["scratch_ptr"])
            stages = (
                () if where is None else tuple(stages_at(plan, rec["cta"], where[0]))
            )
            region = None if where is None else (where[0], where[3])
            key = (i, stages, region, rec["expected"], rec["observed"])
            groups.setdefault(key, []).append((rec, plan, where))

    def order(members):
        rec, plan, where = members[0]
        names = [] if plan is None else [s.name for s in plan.stages]
        waiting = [] if where is None else stages_at(plan, rec["cta"], where[0])
        first = min((names.index(n) for n in waiting), default=len(names))
        return first, min(m[0]["start"] for m in members)

    lines = []
    for members in sorted(groups.values(), key=order):
        members.sort(key=lambda m: m[0]["start"])
        rec, plan, _ = members[0]
        lines.append(describe(rec, plan, h["scratch_ptr"], h["rank"]))
        if len(members) > 1:
            ctas = _ranges([m[0]["cta"] for m in members])
            located = [m[2] for m in members if m[2] is not None]
            span = (members[-1][0]["start"] - rec["start"]) / TICKS_PER_SECOND * 1e6
            lines.append(
                f"  {len(members)} CTAs in all ({ctas})"
                + (
                    f" on rows {_ranges([w[1] for w in located])}, pairs "
                    f"{_ranges([w[2] for w in located])}"
                    if located
                    else ""
                )
                + f", first checks within {span:.1f} us"
            )
    return lines


__all__ = [
    "StuckBuffer",
    "TICKS_PER_SECOND",
    "close",
    "decode",
    "describe",
    "header",
    "locate",
    "read_file",
    "records",
    "stages_at",
    "stuck_buffer",
    "summarize",
    "tag_launch",
    "tasks_on",
]
