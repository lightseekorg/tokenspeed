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

"""Plans of a persistent fused-layer launch and the checks that accept them.

A launch runs one workgroup on each of the ``GRID`` CUs, all resident. Each
workgroup, keyed by its physical ``(xcd, slot)``, runs a fixed sequence of
role steps on wave groups of its 8 waves:

- ``Load``: vector loads (weights, KV) issued and still in flight;
- ``Wait``: poll until every listed key has been published;
- ``Publish``: make keys visible (payload-as-flag words, tagged pairs,
  flags, peer pushes);
- ``Clear``: re-arm this workgroup's share of two-copy regions;
- ``Barrier``: a workgroup rendezvous of all its wave groups.

Cross-GPU waits are modelled on one rank: every rank runs the same plan, so a
wait on a peer's publish of a key is a wait on the local publish of it.
Waits whose producers depend on data (MoE routes) list every producer that
may contribute, which over-approximates the edges.

:func:`check_plan` accepts a plan only if a launch whose workgroups are all
resident cannot deadlock and its hand-offs keep their contracts:

1. every awaited key has exactly one publishing step;
2. the wait graph is acyclic: per wave group, steps in order; a barrier joins
   all wave groups of its workgroup; a publish precedes every wait on its
   key. All workgroups are resident, so a topological order is an execution
   order and every wait is eventually satisfied;
3. role counts per XCD match the plan's expected table;
4. each two-copy region is cleared by exactly the workgroups that publish the
   all-to-all key, each before that publish (a peer reaches the next launch
   only after every one of those publishes, so every clear of the copy it
   will write has drained); the all-to-all key is present in every plan;
5. P-QUEUE: a vector poll of another workgroup's key never sits behind the
   group's own loads in flight (a CU returns loads in order), unless the
   wait is marked issue-wait-compute (the loads feed the compute after it);
   scalar polls are exempt;
6. every region spans at least the plan's rows.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass

GRID = 256
XCDS = 8
SLOTS = GRID // XCDS


class PlanError(ValueError):
    """A plan breaks a check of :func:`check_plan`."""


@dataclass(frozen=True)
class Load:
    """Vector loads issued by ``group`` that stay in flight until consumed."""

    role: str
    group: str


@dataclass(frozen=True)
class Wait:
    """``group`` polls until every key in ``keys`` is published.

    Attributes:
        load: ``"vector"`` (``sc1``/``.cv``/Lamport words) or ``"smem"``.
        issue_wait_compute: The loads in flight feed the compute after this
            wait (allowed behind its own loads, P-QUEUE).
    """

    role: str
    group: str
    keys: tuple[str, ...]
    load: str
    issue_wait_compute: bool


@dataclass(frozen=True)
class Publish:
    role: str
    group: str
    keys: tuple[str, ...]


@dataclass(frozen=True)
class Clear:
    """``group`` clears its share of copy ``(L + 1) & 1`` of ``regions``."""

    role: str
    group: str
    regions: tuple[str, ...]


@dataclass(frozen=True)
class Barrier:
    role: str


Step = Load | Wait | Publish | Clear | Barrier


@dataclass(frozen=True)
class Region:
    """A hand-off buffer of the launch state.

    Attributes:
        name: Region name.
        copies: 2 for areas selected by ``epoch & 1`` and re-armed by clears;
            1 for epoch-tagged words, which are never cleared.
        rows: Rows the region is sized for.
    """

    name: str
    copies: int
    rows: int


@dataclass(frozen=True)
class LayerPlan:
    """One plan variant: the step sequence of every workgroup.

    Attributes:
        name: Variant name for messages.
        rows: Rows of the launch.
        groups: Wave groups of every workgroup (a barrier joins all of them).
        programs: ``(xcd, slot)`` -> its steps, in order.
        regions: The launch state's regions.
        all_to_all: The key every rank's launch waits for from every rank
            (the reuse proof of two-copy regions rests on it).
        expected_counts: role -> instances per XCD (a tuple of 8).
    """

    name: str
    rows: int
    groups: tuple[str, ...]
    programs: dict[tuple[int, int], tuple[Step, ...]]
    regions: tuple[Region, ...]
    all_to_all: str
    expected_counts: dict[str, tuple[int, ...]]


def role_counts(plan: LayerPlan) -> dict[str, tuple[int, ...]]:
    """Instances of each role per XCD (a role's steps on one workgroup count
    once)."""
    counts: dict[str, list[int]] = defaultdict(lambda: [0] * XCDS)
    for (xcd, _), steps in plan.programs.items():
        for role in dict.fromkeys(step.role for step in steps):
            counts[role][xcd] += 1
    return {role: tuple(per_xcd) for role, per_xcd in counts.items()}


def check_plan(plan: LayerPlan) -> int:
    """Raise :class:`PlanError` unless ``plan`` passes every check above.

    Returns:
        The number of ordered wait-graph nodes (all of them).
    """
    if sorted(plan.programs) != [(x, s) for x in range(XCDS) for s in range(SLOTS)]:
        raise PlanError(f"{plan.name}: the plan must cover every (xcd, slot)")
    for region in plan.regions:
        if region.rows < plan.rows:
            raise PlanError(
                f"{plan.name}: region {region.name} spans {region.rows} rows, "
                f"the plan runs {plan.rows}"
            )
    counts = role_counts(plan)
    for role, expected in plan.expected_counts.items():
        got = counts.get(role, (0,) * XCDS)
        if got != tuple(expected):
            raise PlanError(
                f"{plan.name}: role {role} per XCD is {got}, expected {tuple(expected)}"
            )
    publishers: dict[str, tuple] = {}
    waits: dict[str, list[tuple]] = defaultdict(list)
    clears: dict[str, list[tuple]] = defaultdict(list)
    edges: dict[tuple, list[tuple]] = defaultdict(list)
    nodes: list[tuple] = []
    all_to_all_publishers: dict[tuple[int, int], int] = {}
    queued: list[tuple[tuple, Wait]] = []
    for cta, steps in plan.programs.items():
        last = dict.fromkeys(plan.groups)
        inflight = dict.fromkeys(plan.groups, False)
        for index, step in enumerate(steps):
            node = (cta, index)
            nodes.append(node)
            if isinstance(step, Barrier):
                for group in plan.groups:
                    if last[group] is not None:
                        edges[last[group]].append(node)
                    last[group] = node
                continue
            if step.group not in last:
                raise PlanError(
                    f"{plan.name}: {cta} step {index}: no group {step.group}"
                )
            if last[step.group] is not None:
                edges[last[step.group]].append(node)
            last[step.group] = node
            if isinstance(step, Load):
                inflight[step.group] = True
            elif isinstance(step, Wait):
                if step.load not in ("vector", "smem"):
                    raise PlanError(
                        f"{plan.name}: {cta} step {index}: load {step.load}"
                    )
                if (
                    step.load == "vector"
                    and inflight[step.group]
                    and not step.issue_wait_compute
                ):
                    queued.append((node, step))
                # The compute after a wait consumes the loads before it.
                inflight[step.group] = False
                for key in step.keys:
                    waits[key].append(node)
            elif isinstance(step, Publish):
                inflight[step.group] = False
                for key in step.keys:
                    if key in publishers:
                        raise PlanError(
                            f"{plan.name}: key {key} published twice: "
                            f"{publishers[key]} and {node}"
                        )
                    publishers[key] = node
                    if key == plan.all_to_all or key.startswith(plan.all_to_all + "["):
                        all_to_all_publishers.setdefault(cta, index)
            elif isinstance(step, Clear):
                for region in step.regions:
                    clears[region].append(node)
    if not all_to_all_publishers:
        raise PlanError(f"{plan.name}: no all-to-all hop ({plan.all_to_all})")
    names = {region.name for region in plan.regions}
    for region in plan.regions:
        if region.copies != 2:
            if clears.get(region.name):
                raise PlanError(f"{plan.name}: tagged region {region.name} is cleared")
            continue
        clearers = {node[0]: node[1] for node in clears.get(region.name, [])}
        if set(clearers) != set(all_to_all_publishers):
            raise PlanError(
                f"{plan.name}: region {region.name} is cleared by {len(clearers)} "
                f"workgroups, not by the {len(all_to_all_publishers)} that publish "
                f"{plan.all_to_all}"
            )
        late = [cta for cta, at in clearers.items() if at > all_to_all_publishers[cta]]
        if late:
            raise PlanError(
                f"{plan.name}: {late[0]} clears {region.name} after its "
                f"{plan.all_to_all} publish"
            )
    unknown = set(clears) - names
    if unknown:
        raise PlanError(f"{plan.name}: clears of unknown regions {sorted(unknown)}")
    for node, step in queued:
        if any(
            key in publishers and publishers[key][0] != node[0] for key in step.keys
        ):
            raise PlanError(
                f"{plan.name}: {step.role} on {node[0]} polls {step.keys[0]} "
                "behind its own loads in flight (P-QUEUE)"
            )
    for key, waiters in waits.items():
        if key not in publishers:
            raise PlanError(f"{plan.name}: key {key} is awaited but never published")
        for node in waiters:
            edges[publishers[key]].append(node)
    indegree = Counter()
    for src in edges:
        for dst in edges[src]:
            indegree[dst] += 1
    ready = deque(node for node in nodes if indegree[node] == 0)
    ordered = 0
    while ready:
        node = ready.popleft()
        ordered += 1
        for dst in edges[node]:
            indegree[dst] -= 1
            if indegree[dst] == 0:
                ready.append(dst)
    if ordered != len(nodes):
        stuck = sorted(
            {
                plan.programs[node[0]][node[1]].role
                for node in nodes
                if indegree[node] > 0
            }
        )
        raise PlanError(f"{plan.name}: wait cycle (deadlock) through roles {stuck}")
    return ordered
