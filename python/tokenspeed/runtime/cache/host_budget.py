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

"""Host memory budgets of a whole deployment, checked offline.

Each rank's ``HostCacheExecutor`` checks its own pinned Host tiers against
an equal share of what its node has available when it starts. That check
sees one process at one moment: it reserves nothing across independent jobs,
prefill/decode worlds or containers, and it does not count Mooncake client
segments, staging buffers or storage providers. This module sums what an
explicit deployment manifest declares, per node and per container, and
reports which limits the declarations exceed (``tokenspeed cache-budget``,
``docs/configuration/host-memory-budget.md``).

The result is a declared budget estimate. It reserves no memory, takes no
part in runtime admission and does not measure or bound live RSS.

The module imports neither torch nor a device runtime, so the check runs on
any CPU-only machine. ``pinned_reservation_bytes`` is also the rounding
contract the runtime's own startup check uses.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, replace
from typing import Any

SCHEMA_VERSION = 1

REPORT_KIND = "declared_host_memory_budget_estimate"

NOTICE = (
    "Declared budget estimate: the sum of the reservations this manifest "
    "declares. It reserves no memory, is not a runtime admission check, and "
    "does not measure or bound live RSS."
)

# Buffers the runtime pins through the device runtime's caching Host
# allocator, which can reserve more than the requested bytes.
PINNED_KINDS = frozenset({"l2", "snapshot_pool", "slot_state"})

# Kinds every process of a role must declare exactly once (zero bytes
# explicitly when absent); any role may also declare any number of "other".
ROLE_REQUIRED_KINDS: dict[str, tuple[str, ...]] = {
    "inference_rank": (
        "l2",
        "snapshot_pool",
        "slot_state",
        "l3_client_segment",
        "l3_client_staging",
        "process_headroom",
    ),
    "storage_provider": ("provider_segment", "process_headroom"),
    "auxiliary": ("process_headroom",),
}

KINDS = frozenset(kind for kinds in ROLE_REQUIRED_KINDS.values() for kind in kinds) | {
    "other"
}

# Every rank pins its own Host tiers and has its own process memory, so these
# can never be one allocation that several processes share.
PRIVATE_ONLY_KINDS = PINNED_KINDS | {
    "process_headroom",
    "l3_client_segment",
    "l3_client_staging",
}

_RESERVATION_FIELDS: dict[str, tuple[str, ...]] = {
    "measured": ("bytes",),
    "declared": ("bytes",),
    "pinned_rounding": ("requested_bytes", "round_threshold_bytes"),
    "included_in": ("allocation",),
}

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def pinned_reservation_bytes(nbytes: int, *, round_limit_bytes: int | None) -> int:
    """Host bytes a pinned buffer reserves under the caching allocator.

    The allocator rounds a request up to a power of two unless it is larger
    than its rounding limit (``pinned_max_round_threshold_mb``, or a
    ``pinned_max_cached_size_mb`` below it), above which the requested size
    is reserved as is.

    Args:
        nbytes: The requested buffer size; zero or less reserves nothing.
        round_limit_bytes: The allocator's rounding limit, or None when it
            rounds every request.

    Returns:
        The reserved bytes.
    """
    if nbytes <= 0:
        return 0
    if round_limit_bytes is not None and nbytes > round_limit_bytes:
        return nbytes
    return 1 << (nbytes - 1).bit_length()


@dataclass(frozen=True)
class ManifestIssue:
    """One reason a manifest cannot be evaluated."""

    path: str
    message: str


class ManifestInvalid(Exception):
    """The manifest is malformed, incomplete or contradictory."""

    def __init__(self, issues: list[ManifestIssue]) -> None:
        super().__init__(f"{len(issues)} manifest error(s)")
        self.issues = issues


def parse_manifest_json(text: str) -> Any:
    """Decode manifest text, refusing duplicate keys and non-finite numbers.

    Raises:
        ManifestInvalid: The text is not one strict JSON document.
    """

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        counts = Counter(key for key, _ in pairs)
        duplicated = sorted(key for key, count in counts.items() if count > 1)
        if duplicated:
            raise ValueError(f"duplicate JSON key(s): {', '.join(duplicated)}")
        return dict(pairs)

    def refuse_constant(name: str) -> Any:
        raise ValueError(f"non-finite number {name} is not allowed")

    try:
        return json.loads(
            text, object_pairs_hook=unique_object, parse_constant=refuse_constant
        )
    except (ValueError, RecursionError) as error:
        raise ManifestInvalid([ManifestIssue(path="$", message=str(error))])


class _Issues:
    def __init__(self) -> None:
        self.items: list[ManifestIssue] = []

    def add(self, path: str, message: str) -> None:
        self.items.append(ManifestIssue(path=path, message=message))


def _fields(
    value: Any,
    path: str,
    *,
    required: tuple[str, ...],
    optional: tuple[str, ...],
    issues: _Issues,
) -> dict[str, Any] | None:
    """``value`` as an object with exactly the known fields, or None."""
    if not isinstance(value, dict):
        issues.add(path, "must be a JSON object")
        return None
    for key in sorted(set(value) - set(required) - set(optional)):
        issues.add(f"{path}.{key}", "unknown field")
    for key in required:
        if key not in value:
            issues.add(f"{path}.{key}", "required field is missing")
    return value


def _integer(
    obj: dict[str, Any], key: str, path: str, *, minimum: int, issues: _Issues
) -> int | None:
    if key not in obj:
        return None
    value = obj[key]
    # bool is an int subclass; a byte count of `true` is a mistake.
    if isinstance(value, bool) or not isinstance(value, int):
        issues.add(f"{path}.{key}", f"must be an integer, got {json.dumps(value)}")
        return None
    if value < minimum:
        issues.add(f"{path}.{key}", f"must be at least {minimum}, got {value}")
        return None
    return value


def _identifier(
    obj: dict[str, Any], key: str, path: str, issues: _Issues
) -> str | None:
    if key not in obj:
        return None
    value = obj[key]
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        issues.add(
            f"{path}.{key}",
            "must be a non-empty identifier of letters, digits, '.', '_' or '-'",
        )
        return None
    return value


def _choice(
    obj: dict[str, Any],
    key: str,
    path: str,
    *,
    choices: frozenset[str] | tuple[str, ...],
    issues: _Issues,
) -> str | None:
    if key not in obj:
        return None
    value = obj[key]
    if not isinstance(value, str) or value not in choices:
        issues.add(
            f"{path}.{key}",
            f"must be one of {', '.join(sorted(choices))}, got {json.dumps(value)}",
        )
        return None
    return value


def _array(
    obj: dict[str, Any], key: str, path: str, *, min_length: int, issues: _Issues
) -> list[Any]:
    if key not in obj:
        return []
    value = obj[key]
    if not isinstance(value, list):
        issues.add(f"{path}.{key}", "must be a JSON array")
        return []
    if len(value) < min_length:
        issues.add(f"{path}.{key}", f"must list at least {min_length} entry")
    return value


def _check_unique(ids: list[tuple[str, str]], what: str, issues: _Issues) -> None:
    """Report every repeated id; ``ids`` holds ``(path, id)`` pairs."""
    seen: set[str] = set()
    for path, identifier in ids:
        if identifier in seen:
            issues.add(path, f"duplicate {what} id {identifier!r}")
        seen.add(identifier)


@dataclass(frozen=True)
class _Node:
    id: str
    memory_bytes: int
    system_reserved_bytes: int


@dataclass(frozen=True)
class _Container:
    id: str
    node: str
    memory_limit_bytes: int


@dataclass(frozen=True)
class _Process:
    key: str
    job: str
    role: str
    count: int
    node: str
    container: str | None


@dataclass(frozen=True)
class _Declaration:
    """One allocation entry of one process entry."""

    path: str
    process: _Process
    id: str
    kind: str
    sharing: str
    basis: str
    # The validated reservation fields, compared across shared declarations.
    spec: tuple[tuple[str, Any], ...]
    reserved_bytes: int
    included_in: str | None
    note: str | None


def _parse_reservation(
    value: Any, path: str, *, kind: str | None, issues: _Issues
) -> tuple[str, tuple[tuple[str, Any], ...], int, str | None] | None:
    """``(basis, spec, reserved_bytes, included_in)`` of one reservation."""
    if not isinstance(value, dict):
        issues.add(path, "must be a JSON object")
        return None
    basis = _choice(
        value, "basis", path, choices=tuple(_RESERVATION_FIELDS), issues=issues
    )
    if "basis" not in value:
        issues.add(f"{path}.basis", "required field is missing")
    if basis is None:
        return None
    obj = _fields(
        value,
        path,
        required=("basis", *_RESERVATION_FIELDS[basis]),
        optional=(),
        issues=issues,
    )
    if obj is None:
        return None
    if kind in PINNED_KINDS and basis == "declared":
        # A nominal tensor size is not what the pinned allocator reserves.
        issues.add(
            f"{path}.basis",
            f"a pinned {kind} buffer needs 'measured' reserved bytes or the "
            "explicit 'pinned_rounding' contract, not 'declared'",
        )
        return None
    if kind is not None and kind not in PINNED_KINDS and basis == "pinned_rounding":
        issues.add(
            f"{path}.basis",
            f"'pinned_rounding' applies to pinned buffers "
            f"({', '.join(sorted(PINNED_KINDS))}), not {kind}",
        )
        return None
    if basis in ("measured", "declared"):
        nbytes = _integer(obj, "bytes", path, minimum=0, issues=issues)
        if nbytes is None:
            return None
        return basis, (("bytes", nbytes),), nbytes, None
    if basis == "pinned_rounding":
        requested = _integer(obj, "requested_bytes", path, minimum=0, issues=issues)
        threshold: int | None = None
        threshold_ok = "round_threshold_bytes" in obj
        if threshold_ok and obj["round_threshold_bytes"] is not None:
            threshold = _integer(
                obj, "round_threshold_bytes", path, minimum=0, issues=issues
            )
            threshold_ok = threshold is not None
        if requested is None or not threshold_ok:
            return None
        reserved = pinned_reservation_bytes(requested, round_limit_bytes=threshold)
        spec = (("requested_bytes", requested), ("round_threshold_bytes", threshold))
        return basis, spec, reserved, None
    target = _identifier(obj, "allocation", path, issues)
    if target is None:
        return None
    return basis, (("allocation", target),), 0, target


def _parse_allocation(
    value: Any, path: str, *, process: _Process, issues: _Issues
) -> _Declaration | None:
    obj = _fields(
        value,
        path,
        required=("id", "kind", "sharing", "reservation"),
        optional=("note",),
        issues=issues,
    )
    if obj is None:
        return None
    alloc_id = _identifier(obj, "id", path, issues)
    kind = _choice(obj, "kind", path, choices=KINDS, issues=issues)
    sharing = _choice(
        obj, "sharing", path, choices=("private", "shared"), issues=issues
    )
    note = obj.get("note")
    if note is not None and not isinstance(note, str):
        issues.add(f"{path}.note", "must be a string")
    reservation = (
        _parse_reservation(
            obj["reservation"], f"{path}.reservation", kind=kind, issues=issues
        )
        if "reservation" in obj
        else None
    )
    if kind is not None and sharing == "shared" and kind in PRIVATE_ONLY_KINDS:
        issues.add(
            f"{path}.sharing",
            f"{kind} is allocated by each process and cannot be shared",
        )
        return None
    if alloc_id is None or kind is None or sharing is None or reservation is None:
        return None
    basis, spec, reserved_bytes, included_in = reservation
    if sharing == "shared" and included_in is not None:
        issues.add(
            f"{path}.reservation.basis",
            "a shared allocation cannot be included in a process's allocation",
        )
        return None
    return _Declaration(
        path=path,
        process=process,
        id=alloc_id,
        kind=kind,
        sharing=sharing,
        basis=basis,
        spec=spec,
        reserved_bytes=reserved_bytes,
        included_in=included_in,
        note=note if isinstance(note, str) else None,
    )


def _check_process_allocations(
    declarations: list[_Declaration],
    raw_allocations: list[Any],
    path: str,
    *,
    role: str,
    issues: _Issues,
) -> None:
    """One process entry's required kinds and ``included_in`` references."""
    # Count kinds from the entries themselves, so an allocation with an
    # invalid reservation is not also reported as missing.
    kinds = [
        raw["kind"]
        for raw in raw_allocations
        if isinstance(raw, dict)
        and isinstance(raw.get("kind"), str)
        and raw["kind"] in KINDS
    ]
    counts = Counter(kinds)
    allowed = set(ROLE_REQUIRED_KINDS[role]) | {"other"}
    for index, raw in enumerate(raw_allocations):
        if (
            isinstance(raw, dict)
            and isinstance(raw.get("kind"), str)
            and raw["kind"] in KINDS - allowed
        ):
            issues.add(
                f"{path}.allocations[{index}].kind",
                f"a {role} process does not have {raw['kind']} memory",
            )
    for kind in ROLE_REQUIRED_KINDS[role]:
        if counts[kind] != 1:
            issues.add(
                f"{path}.allocations",
                f"a {role} process must declare exactly one {kind} allocation "
                f"(use zero bytes when it has none), found {counts[kind]}",
            )
    by_id = {declaration.id: declaration for declaration in declarations}
    for declaration in declarations:
        if declaration.included_in is None:
            continue
        target = by_id.get(declaration.included_in)
        where = f"{declaration.path}.reservation.allocation"
        if target is None:
            issues.add(
                where,
                f"references allocation {declaration.included_in!r}, which this "
                "process does not declare",
            )
        elif target is declaration:
            issues.add(where, "an allocation cannot be included in itself")
        elif target.sharing != "private" or target.basis not in (
            "measured",
            "declared",
        ):
            # A rounding-contract reservation covers its own requested bytes
            # only; just a measured or declared total can contain another
            # buffer.
            issues.add(
                where,
                f"allocation {target.id!r} must be private and 'measured' or "
                "'declared' to include another allocation",
            )


def _parse_process(
    value: Any,
    path: str,
    *,
    job: str,
    replica: str,
    nodes: dict[str, _Node],
    containers: dict[str, _Container],
    issues: _Issues,
) -> tuple[_Process, list[_Declaration]] | None:
    obj = _fields(
        value,
        path,
        required=("id", "role", "count", "placement", "allocations"),
        optional=(),
        issues=issues,
    )
    if obj is None:
        return None
    process_id = _identifier(obj, "id", path, issues)
    role = _choice(obj, "role", path, choices=tuple(ROLE_REQUIRED_KINDS), issues=issues)
    count = _integer(obj, "count", path, minimum=1, issues=issues)
    placement = (
        _parse_placement(
            obj["placement"],
            f"{path}.placement",
            nodes=nodes,
            containers=containers,
            issues=issues,
        )
        if "placement" in obj
        else None
    )
    raw_allocations = _array(obj, "allocations", path, min_length=1, issues=issues)
    if process_id is None or role is None or count is None or placement is None:
        return None
    node, container = placement
    process = _Process(
        key=f"{job}/{replica}/{process_id}",
        job=job,
        role=role,
        count=count,
        node=node,
        container=container,
    )
    declarations = []
    alloc_ids = []
    for index, raw in enumerate(raw_allocations):
        alloc_path = f"{path}.allocations[{index}]"
        declaration = _parse_allocation(raw, alloc_path, process=process, issues=issues)
        if declaration is not None:
            declarations.append(declaration)
            alloc_ids.append((f"{alloc_path}.id", declaration.id))
    _check_unique(alloc_ids, "allocation", issues)
    _check_process_allocations(
        declarations, raw_allocations, path, role=role, issues=issues
    )
    return process, declarations


def _parse_placement(
    value: Any,
    path: str,
    *,
    nodes: dict[str, _Node],
    containers: dict[str, _Container],
    issues: _Issues,
) -> tuple[str, str | None] | None:
    """``(node, container)`` of a placement naming exactly one of them."""
    obj = _fields(
        value, path, required=(), optional=("node", "container"), issues=issues
    )
    if obj is None:
        return None
    if len(set(obj) & {"node", "container"}) != 1:
        # A container already names its node; naming both could disagree.
        issues.add(path, "must name exactly one of 'node' or 'container'")
        return None
    if "container" in obj:
        name = _identifier(obj, "container", path, issues)
        if name is None:
            return None
        if name not in containers:
            issues.add(f"{path}.container", f"references unknown container {name!r}")
            return None
        return containers[name].node, name
    name = _identifier(obj, "node", path, issues)
    if name is None:
        return None
    if name not in nodes:
        issues.add(f"{path}.node", f"references unknown node {name!r}")
        return None
    return name, None


def _parse_job(
    value: Any,
    path: str,
    *,
    nodes: dict[str, _Node],
    containers: dict[str, _Container],
    issues: _Issues,
) -> list[tuple[_Process, list[_Declaration]]]:
    obj = _fields(
        value,
        path,
        required=("id", "replica_count", "processes_per_replica", "replicas"),
        optional=(),
        issues=issues,
    )
    if obj is None:
        return []
    job_id = _identifier(obj, "id", path, issues)
    replica_count = _integer(obj, "replica_count", path, minimum=1, issues=issues)
    expected_roles = (
        _parse_role_counts(
            obj["processes_per_replica"], f"{path}.processes_per_replica", issues
        )
        if "processes_per_replica" in obj
        else None
    )
    replicas = _array(obj, "replicas", path, min_length=1, issues=issues)
    if replica_count is not None and len(replicas) != replica_count:
        issues.add(
            f"{path}.replicas",
            f"lists {len(replicas)} replica(s), but replica_count is {replica_count}",
        )
    if job_id is None:
        return []
    parsed = []
    replica_ids = []
    for replica_index, raw_replica in enumerate(replicas):
        replica_path = f"{path}.replicas[{replica_index}]"
        replica = _fields(
            raw_replica,
            replica_path,
            required=("id", "processes"),
            optional=(),
            issues=issues,
        )
        if replica is None:
            continue
        replica_id = _identifier(replica, "id", replica_path, issues)
        processes = _array(
            replica, "processes", replica_path, min_length=1, issues=issues
        )
        if replica_id is None:
            continue
        replica_ids.append((f"{replica_path}.id", replica_id))
        role_counts: Counter[str] = Counter()
        process_ids = []
        complete = True
        for process_index, raw_process in enumerate(processes):
            process_path = f"{replica_path}.processes[{process_index}]"
            result = _parse_process(
                raw_process,
                process_path,
                job=job_id,
                replica=replica_id,
                nodes=nodes,
                containers=containers,
                issues=issues,
            )
            if result is None:
                complete = False
                continue
            process, declarations = result
            role_counts[process.role] += process.count
            process_ids.append((f"{process_path}.id", process.key))
            parsed.append(result)
        _check_unique(process_ids, "process", issues)
        if complete and expected_roles is not None and role_counts != expected_roles:
            issues.add(
                f"{replica_path}.processes",
                f"replica {replica_id!r} runs {_format_roles(role_counts)}, but "
                f"processes_per_replica is {_format_roles(expected_roles)}",
            )
    _check_unique(replica_ids, f"replica (in job {job_id!r})", issues)
    return parsed


def _parse_role_counts(value: Any, path: str, issues: _Issues) -> Counter[str] | None:
    if not isinstance(value, dict) or not value:
        issues.add(path, "must be a non-empty JSON object of role to process count")
        return None
    counts: Counter[str] = Counter()
    for role in sorted(value):
        if role not in ROLE_REQUIRED_KINDS:
            issues.add(f"{path}.{role}", "unknown process role")
            return None
        count = _integer(value, role, path, minimum=1, issues=issues)
        if count is None:
            return None
        counts[role] = count
    return counts


def _format_roles(counts: Counter[str]) -> str:
    return ", ".join(f"{counts[role]} {role}" for role in sorted(counts)) or "nothing"


@dataclass(frozen=True)
class _Manifest:
    nodes: dict[str, _Node]
    containers: dict[str, _Container]
    processes: list[_Process]
    declarations: list[_Declaration]


def _parse_manifest(document: Any) -> _Manifest:
    issues = _Issues()
    top = _fields(
        document,
        "$",
        required=("schema_version", "nodes", "containers", "jobs"),
        optional=(),
        issues=issues,
    )
    if top is None:
        raise ManifestInvalid(issues.items)
    version = _integer(top, "schema_version", "$", minimum=0, issues=issues)
    if version is not None and version != SCHEMA_VERSION:
        issues.add("$.schema_version", f"must be {SCHEMA_VERSION}, got {version}")

    nodes: dict[str, _Node] = {}
    node_ids = []
    for index, raw in enumerate(_array(top, "nodes", "$", min_length=1, issues=issues)):
        path = f"$.nodes[{index}]"
        obj = _fields(
            raw,
            path,
            required=("id", "memory_bytes", "system_reserved_bytes"),
            optional=(),
            issues=issues,
        )
        if obj is None:
            continue
        node_id = _identifier(obj, "id", path, issues)
        memory = _integer(obj, "memory_bytes", path, minimum=1, issues=issues)
        reserved = _integer(
            obj, "system_reserved_bytes", path, minimum=0, issues=issues
        )
        if node_id is None:
            continue
        node_ids.append((f"{path}.id", node_id))
        # Registered even when a size is invalid, so references to it do not
        # add errors; an invalid manifest is never evaluated.
        nodes.setdefault(node_id, _Node(node_id, memory or 0, reserved or 0))
    _check_unique(node_ids, "node", issues)

    containers: dict[str, _Container] = {}
    container_ids = []
    raw_containers = _array(top, "containers", "$", min_length=0, issues=issues)
    for index, raw in enumerate(raw_containers):
        path = f"$.containers[{index}]"
        obj = _fields(
            raw,
            path,
            required=("id", "node", "memory_limit_bytes"),
            optional=(),
            issues=issues,
        )
        if obj is None:
            continue
        container_id = _identifier(obj, "id", path, issues)
        node = _identifier(obj, "node", path, issues)
        limit = _integer(obj, "memory_limit_bytes", path, minimum=1, issues=issues)
        if container_id is None:
            continue
        container_ids.append((f"{path}.id", container_id))
        if node is not None and node not in nodes:
            issues.add(f"{path}.node", f"references unknown node {node!r}")
        containers.setdefault(
            container_id, _Container(container_id, node or "", limit or 0)
        )
    _check_unique(container_ids, "container", issues)

    processes: list[_Process] = []
    declarations: list[_Declaration] = []
    job_ids = []
    for index, raw in enumerate(_array(top, "jobs", "$", min_length=1, issues=issues)):
        path = f"$.jobs[{index}]"
        if isinstance(raw, dict) and isinstance(raw.get("id"), str):
            job_ids.append((f"{path}.id", raw["id"]))
        for process, process_declarations in _parse_job(
            raw, path, nodes=nodes, containers=containers, issues=issues
        ):
            processes.append(process)
            declarations.extend(process_declarations)
    _check_unique(job_ids, "job", issues)
    _check_sharing(declarations, issues)

    if issues.items:
        raise ManifestInvalid(issues.items)
    return _Manifest(nodes, containers, processes, declarations)


def _check_sharing(declarations: list[_Declaration], issues: _Issues) -> None:
    """Every declaration of one shared id must describe the same memory."""
    private_ids = {d.id for d in declarations if d.sharing == "private"}
    first: dict[str, _Declaration] = {}
    for declaration in declarations:
        if declaration.sharing != "shared":
            continue
        if declaration.id in private_ids:
            issues.add(
                f"{declaration.path}.sharing",
                f"allocation id {declaration.id!r} is declared both shared and "
                "private",
            )
            continue
        original = first.setdefault(declaration.id, declaration)
        if original is declaration:
            continue
        for what, mine, theirs in (
            ("kind", declaration.kind, original.kind),
            (
                "reservation",
                (declaration.basis, declaration.spec),
                (original.basis, original.spec),
            ),
            (
                "memory domain",
                _domain_name(declaration.process),
                _domain_name(original.process),
            ),
        ):
            if mine != theirs:
                issues.add(
                    declaration.path,
                    f"shared allocation {declaration.id!r} has {what} {mine} here "
                    f"but {theirs} at {original.path}",
                )


def _domain_name(process: _Process) -> str:
    if process.container is not None:
        return f"container {process.container}"
    return f"node {process.node} outside any container"


@dataclass(frozen=True)
class _Charge:
    """One counted allocation: once if shared, once per process if private."""

    key: str
    kind: str
    sharing: str
    basis: str
    node: str
    container: str | None
    # The job of a private allocation; None for a shared one.
    job: str | None
    bytes_each: int
    instances: int
    users: tuple[str, ...]
    user_processes: int
    included_in: str | None
    note: str | None

    @property
    def total_bytes(self) -> int:
        return self.bytes_each * self.instances


def _charges(manifest: _Manifest) -> list[_Charge]:
    charges: dict[str, _Charge] = {}
    shared_users: dict[str, list[_Process]] = {}
    for d in manifest.declarations:
        process = d.process
        if d.sharing == "shared":
            shared_users.setdefault(d.id, []).append(process)
            key = f"shared/{d.id}"
            if key in charges:
                continue
            charges[key] = _Charge(
                key=key,
                kind=d.kind,
                sharing="shared",
                basis=d.basis,
                node=process.node,
                container=process.container,
                job=None,
                bytes_each=d.reserved_bytes,
                instances=1,
                users=(),
                user_processes=0,
                included_in=None,
                note=d.note,
            )
            continue
        key = f"{process.key}/{d.id}"
        charges[key] = _Charge(
            key=key,
            kind=d.kind,
            sharing="private",
            basis=d.basis,
            node=process.node,
            container=process.container,
            job=process.job,
            bytes_each=d.reserved_bytes,
            instances=process.count,
            users=(process.key,),
            user_processes=process.count,
            included_in=(
                f"{process.key}/{d.included_in}" if d.included_in is not None else None
            ),
            note=d.note,
        )
    for shared_id, users in shared_users.items():
        key = f"shared/{shared_id}"
        charges[key] = replace(
            charges[key],
            users=tuple(sorted({user.key for user in users})),
            user_processes=sum(user.count for user in users),
        )
    return [charges[key] for key in sorted(charges)]


def _usage(charges: list[_Charge]) -> dict[str, Any]:
    by_kind: Counter[str] = Counter()
    by_job: Counter[str] = Counter()
    shared = 0
    for charge in charges:
        by_kind[charge.kind] += charge.total_bytes
        if charge.job is None:
            shared += charge.total_bytes
        else:
            by_job[charge.job] += charge.total_bytes
    return {
        "allocated_bytes": sum(charge.total_bytes for charge in charges),
        "by_kind": {kind: by_kind[kind] for kind in sorted(by_kind)},
        "by_job": {job: by_job[job] for job in sorted(by_job)},
        "shared_bytes": shared,
    }


def _report(
    *,
    status: str,
    errors: list[ManifestIssue],
    warnings: list[str],
    nodes: list[dict[str, Any]],
    containers: list[dict[str, Any]],
    allocations: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "report_kind": REPORT_KIND,
        "notice": NOTICE,
        "status": status,
        "errors": [{"path": e.path, "message": e.message} for e in errors],
        "warnings": warnings,
        "nodes": nodes,
        "containers": containers,
        "allocations": allocations,
    }


def invalid_report(issues: list[ManifestIssue]) -> dict[str, Any]:
    """The report of a manifest that could not be evaluated."""
    return _report(
        status="invalid",
        errors=issues,
        warnings=[],
        nodes=[],
        containers=[],
        allocations=[],
    )


def evaluate_manifest(document: Any) -> dict[str, Any]:
    """Check a decoded deployment manifest against its node and container limits.

    Every allocation counts once in its memory domain: a private one once per
    process of its entry, a shared one once however many processes declare
    it. A node's total is its system reservation plus every allocation placed
    on it, inside a container or not; a container's total is the allocations
    placed in it. A container's limit therefore never adds to its node's
    total, and checking both never counts an allocation twice.

    Args:
        document: The decoded JSON manifest (``parse_manifest_json``).

    Returns:
        The JSON-serializable report, with ``status`` ``"fits"`` or
        ``"overcommitted"``; lists are in a deterministic order.

    Raises:
        ManifestInvalid: The manifest is malformed, incomplete or
            contradictory; nothing is evaluated.
    """
    manifest = _parse_manifest(document)
    charges = _charges(manifest)
    warnings = []
    overcommitted = False

    containers = []
    for container_id in sorted(manifest.containers):
        container = manifest.containers[container_id]
        mine = [c for c in charges if c.container == container_id]
        usage = _usage(mine)
        free = container.memory_limit_bytes - usage["allocated_bytes"]
        overcommitted |= free < 0
        if not any(p.container == container_id for p in manifest.processes):
            warnings.append(f"container {container_id!r} runs no declared process")
        containers.append(
            {
                "id": container_id,
                "node": container.node,
                "memory_limit_bytes": container.memory_limit_bytes,
                **usage,
                "total_bytes": usage["allocated_bytes"],
                "free_bytes": free,
                "fits": free >= 0,
            }
        )

    nodes = []
    for node_id in sorted(manifest.nodes):
        node = manifest.nodes[node_id]
        usage = _usage([c for c in charges if c.node == node_id])
        total = node.system_reserved_bytes + usage["allocated_bytes"]
        free = node.memory_bytes - total
        overcommitted |= free < 0
        node_containers = sorted(
            c.id for c in manifest.containers.values() if c.node == node_id
        )
        container_limits = sum(
            manifest.containers[c].memory_limit_bytes for c in node_containers
        )
        if container_limits > node.memory_bytes - node.system_reserved_bytes:
            # Not an error: a node's check uses its declared allocations, and
            # container limits are commonly overcommitted.
            warnings.append(
                f"node {node_id!r}: its container limits total {container_limits} "
                "bytes, more than its memory less system_reserved_bytes; the "
                "node check counts declared allocations, not limits"
            )
        if not any(p.node == node_id for p in manifest.processes):
            warnings.append(f"node {node_id!r} runs no declared process")
        nodes.append(
            {
                "id": node_id,
                "memory_bytes": node.memory_bytes,
                "system_reserved_bytes": node.system_reserved_bytes,
                **usage,
                "total_bytes": total,
                "free_bytes": free,
                "fits": free >= 0,
                "containers": node_containers,
                "container_limits_bytes": container_limits,
            }
        )

    allocations = [
        {
            "key": c.key,
            "kind": c.kind,
            "sharing": c.sharing,
            "basis": c.basis,
            "node": c.node,
            "container": c.container,
            "job": c.job,
            "bytes_each": c.bytes_each,
            "instances": c.instances,
            "total_bytes": c.total_bytes,
            "users": list(c.users),
            "user_processes": c.user_processes,
            "included_in": c.included_in,
            "note": c.note,
        }
        for c in charges
    ]
    return _report(
        status="overcommitted" if overcommitted else "fits",
        errors=[],
        warnings=warnings,
        nodes=nodes,
        containers=containers,
        allocations=allocations,
    )
