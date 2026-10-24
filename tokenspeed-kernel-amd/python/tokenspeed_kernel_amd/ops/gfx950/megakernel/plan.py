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

"""Static work plan of a persistent decode megakernel.

A megakernel launch keeps ``grid`` CTAs resident (one per CU) and runs a
fixed list of stages. Stage ``X`` has ``tasks`` independent tasks. Task ``t``
runs on CTA ``(base + t) % grid``. Every CTA walks the stages in plan order
and runs its tasks of each stage, if it has any. A stage with more tasks than
CTAs runs in rounds: task ``t + grid`` runs on the CTA of task ``t``, after it.
Stages hand values to each other only through mailbox regions in a scratch
buffer (``mailbox.py``).

Deadlock freedom rests on one rule, which :class:`Plan` checks: a stage reads
only regions that an earlier stage writes. A CTA then never waits on a task
that it, or a CTA still behind it, has not reached yet. The rule needs every
CTA to be resident at once, so a launch must not exceed one CTA per CU.

A kernel takes its plan as the compile-time string :attr:`Plan.key` (the
plan's canonical JSON), not as the object. Triton's compile hooks serialize
every compile-time argument to JSON, and the full text keeps the compile
cache key exact. The key carries the plan object for the kernel's
compile-time helpers (:class:`PlanKey`).

Placement (``base``) is the main tuning knob. Put a stage on CTAs whose
earlier tasks end early. Put a weight-streaming stage on CTAs that are idle
while other stages produce its inputs. Its weight loads then overlap the wait.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

# A region starts on its own 256-byte line, so producers of different regions
# never write the same cache line.
REGION_ALIGN_PAIRS = 32


@dataclass(frozen=True)
class Stage:
    """One stage of a plan.

    Args:
        name: Stage name, unique in the plan.
        tasks: Independent tasks the stage runs per launch (in rounds when
            more than the grid).
        base: CTA that runs task 0.
        reads: Mailbox regions the stage polls.
        writes: Mailbox regions the stage fills.
    """

    name: str
    tasks: int
    base: int
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()


@dataclass(frozen=True)
class Plan:
    """Stages, their placement and the mailbox regions of one megakernel launch.

    Args:
        grid: Resident CTAs of the launch.
        stages: Stages in execution order.
        regions: ``(name, pairs)`` of every mailbox region, per row.
        rows: Token rows the launch serves. Row ``r`` of a region sits
            ``r * row_pairs`` pairs after row 0.
        chain: Consecutive rows that are one request's consecutive tokens
            (1 for decode, ``k + 1`` for a speculative verify of ``k`` draft
            tokens). Row ``r`` is token ``r % chain`` of request
            ``r // chain``. A request's state carries from token to token.
        external: Regions the host fills before the launch (for example in a
            test that runs a suffix of a layer's stages).
    """

    grid: int
    stages: tuple[Stage, ...]
    regions: tuple[tuple[str, int], ...]
    rows: int
    chain: int
    external: tuple[str, ...] = ()
    # __post_init__ derives these fields: the pairs of one row of every
    # region, and the pair offset of each region in row 0.
    row_pairs: int = field(init=False)
    _offsets: dict = field(init=False, compare=False, hash=False, repr=False)

    def __post_init__(self) -> None:
        offsets, offset = {}, 0
        for name, pairs in self.regions:
            if name in offsets:
                raise ValueError(f"duplicate mailbox region {name!r}")
            offsets[name] = offset
            offset += -(-pairs // REGION_ALIGN_PAIRS) * REGION_ALIGN_PAIRS
        object.__setattr__(self, "_offsets", offsets)
        object.__setattr__(self, "row_pairs", offset)
        self._check()

    def _check(self) -> None:
        if self.rows < 1:
            raise ValueError("a plan serves one row at least")
        if self.chain < 1 or self.rows % self.chain:
            raise ValueError(
                f"{self.rows} rows are not requests of {self.chain} tokens"
            )
        written = set(self.external)
        names = set()
        for stage in self.stages:
            if stage.name in names:
                raise ValueError(f"duplicate stage {stage.name!r}")
            names.add(stage.name)
            if stage.tasks < 1 or not 0 <= stage.base < self.grid:
                raise ValueError(
                    f"stage {stage.name!r}: {stage.tasks} tasks at base "
                    f"{stage.base} do not fit a grid of {self.grid}"
                )
            for region in stage.reads + stage.writes:
                if region not in self._offsets:
                    raise ValueError(f"stage {stage.name!r}: unknown region {region!r}")
            # The deadlock-freedom rule: inputs come from earlier stages.
            late = [r for r in stage.reads if r not in written]
            if late:
                raise ValueError(
                    f"stage {stage.name!r} reads {late} before any earlier stage "
                    "writes them"
                )
            written.update(stage.writes)

    def only(self, names: tuple[str, ...]) -> Plan:
        """This plan with only the stages ``names`` (for example, to test a
        module's stages alone). The regions that those stages read before one
        of them writes become ``external``, for the host to fill."""
        stages = tuple(stage for stage in self.stages if stage.name in names)
        missing = set(names) - {stage.name for stage in stages}
        if missing:
            raise KeyError(f"no stages {sorted(missing)}")
        written, external = set(self.external), list(self.external)
        for stage in stages:
            external += [r for r in stage.reads if r not in written]
            written.update(stage.reads + stage.writes)
        return Plan(
            grid=self.grid,
            stages=stages,
            regions=self.regions,
            rows=self.rows,
            chain=self.chain,
            external=tuple(dict.fromkeys(external)),
        )

    def stage(self, name: str) -> Stage:
        for stage in self.stages:
            if stage.name == name:
                return stage
        raise KeyError(name)

    def has_stage(self, name: str) -> bool:
        return any(stage.name == name for stage in self.stages)

    def offset(self, region: str) -> int:
        """Pair offset of row 0 of ``region`` in the scratch buffer."""
        return self._offsets[region]

    @property
    def scratch_pairs(self) -> int:
        """Pairs of the scratch buffer this plan needs."""
        return self.rows * self.row_pairs

    @property
    def key(self) -> PlanKey:
        """Canonical JSON of the plan, the form a kernel takes it in."""
        fields = {
            "grid": self.grid,
            "stages": [asdict(stage) for stage in self.stages],
            "regions": self.regions,
            "rows": self.rows,
            "chain": self.chain,
            "external": self.external,
        }
        return PlanKey(json.dumps(fields, separators=(",", ":")), self)


class PlanKey(str):
    """A plan's canonical JSON text that also carries the plan.

    As a ``str`` it serializes, hashes and compares by its text, so it is a
    valid compile-time kernel argument. The kernel's compile-time helpers read
    the plan from :attr:`plan`.
    """

    plan: Plan

    def __new__(cls, text: str, plan: Plan) -> PlanKey:
        key = super().__new__(cls, text)
        key.plan = plan
        return key


def placement(bases: dict[str, int], names: tuple[str, ...]) -> tuple[int, ...]:
    """The base CTA of each stage ``names`` from ``bases``, a stage module's
    placement table, which must name exactly those stages in that order."""
    if tuple(bases) != names:
        raise ValueError(f"bases must place {names}, got {tuple(bases)}")
    return tuple(bases.values())


# Layers sharing one step counter tell their launches apart by layer slot.
LAYER_SLOTS = 64


def launch_tag(step: int, layer: int) -> int:
    """Host copy of the device tag of ``layer``'s launch at ``step``."""
    return (step * LAYER_SLOTS + layer + 1) & 0xFFFFFFFF


__all__ = ["LAYER_SLOTS", "Plan", "PlanKey", "Stage", "launch_tag", "placement"]
