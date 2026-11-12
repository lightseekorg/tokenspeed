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

"""``tokenspeed cache-budget`` and the deployment Host memory budget it checks."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from tokenspeed.cli import main as cli_main
from tokenspeed.runtime.cache.host_budget import (
    ManifestInvalid,
    evaluate_manifest,
    parse_manifest_json,
    pinned_reservation_bytes,
)

GIB = 1 << 30
EXAMPLE = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "configuration"
    / "host-memory-budget.example.json"
)


def measured(nbytes: Any) -> dict[str, Any]:
    return {"basis": "measured", "bytes": nbytes}


def declared(nbytes: Any) -> dict[str, Any]:
    return {"basis": "declared", "bytes": nbytes}


def allocation(
    alloc_id: str, kind: str, sharing: str, reservation: dict[str, Any]
) -> dict[str, Any]:
    return {
        "id": alloc_id,
        "kind": kind,
        "sharing": sharing,
        "reservation": reservation,
    }


def rank_allocations(
    *,
    l2: int,
    snapshot: int,
    slot_state: int,
    segment: int,
    staging: int,
    headroom: int,
) -> list[dict[str, Any]]:
    """Every allocation an inference rank must declare, all private."""
    return [
        allocation("l2", "l2", "private", measured(l2)),
        allocation("snapshot", "snapshot_pool", "private", measured(snapshot)),
        allocation("slot-state", "slot_state", "private", measured(slot_state)),
        allocation("segment", "l3_client_segment", "private", declared(segment)),
        allocation("staging", "l3_client_staging", "private", declared(staging)),
        allocation("headroom", "process_headroom", "private", declared(headroom)),
    ]


def ranks(
    process_id: str,
    count: int,
    placement: dict[str, str],
    allocations: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "id": process_id,
        "role": "inference_rank",
        "count": count,
        "placement": placement,
        "allocations": allocations,
    }


def job(
    job_id: str, replicas: list[list[dict[str, Any]]], ranks_per_replica: int
) -> dict[str, Any]:
    return {
        "id": job_id,
        "replica_count": len(replicas),
        "processes_per_replica": {"inference_rank": ranks_per_replica},
        "replicas": [
            {"id": str(index), "processes": processes}
            for index, processes in enumerate(replicas)
        ],
    }


def node(node_id: str, memory: int, system_reserved: int) -> dict[str, Any]:
    return {
        "id": node_id,
        "memory_bytes": memory,
        "system_reserved_bytes": system_reserved,
    }


def container(container_id: str, node_id: str, limit: int) -> dict[str, Any]:
    return {"id": container_id, "node": node_id, "memory_limit_bytes": limit}


def manifest(
    nodes: list[dict[str, Any]],
    containers: list[dict[str, Any]],
    jobs: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "nodes": nodes,
        "containers": containers,
        "jobs": jobs,
    }


def one_rank_job(job_id: str, placement: dict[str, str], l2: int) -> dict[str, Any]:
    """A one-rank job whose rank pins ``l2`` bytes plus 1 GiB of headroom."""
    allocations = rank_allocations(
        l2=l2, snapshot=0, slot_state=0, segment=0, staging=0, headroom=GIB
    )
    return job(job_id, [[ranks("rank", 1, placement, allocations)]], 1)


def by_id(entries: list[dict[str, Any]], entry_id: str) -> dict[str, Any]:
    (entry,) = [entry for entry in entries if entry["id"] == entry_id]
    return entry


def issues_of(document: Any) -> list[tuple[str, str]]:
    with pytest.raises(ManifestInvalid) as raised:
        evaluate_manifest(document)
    return [(issue.path, issue.message) for issue in raised.value.issues]


def run_cli(argv: list[str], capsys) -> tuple[int, str]:
    old_argv = sys.argv
    sys.argv = ["tokenspeed", *argv]
    try:
        with pytest.raises(SystemExit) as raised:
            cli_main()
    finally:
        sys.argv = old_argv
    return raised.value.code, capsys.readouterr().out


def write_manifest(tmp_path: Path, document: Any) -> str:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(document))
    return str(path)


# --- the pinned allocator's rounding contract -------------------------------


def test_pinned_rounding_contract():
    threshold = GIB
    # Up to the threshold a request rounds up to a power of two...
    assert pinned_reservation_bytes(400_000_000, round_limit_bytes=threshold) == (
        512 << 20
    )
    assert pinned_reservation_bytes(threshold, round_limit_bytes=threshold) == GIB
    # ...above it the request is reserved as is.
    assert (
        pinned_reservation_bytes(threshold + 1, round_limit_bytes=threshold)
        == threshold + 1
    )
    # An allocator without a limit rounds every request.
    assert pinned_reservation_bytes(140_000_000_000, round_limit_bytes=None) == (
        1 << 38
    )
    assert pinned_reservation_bytes(0, round_limit_bytes=None) == 0


def test_rounding_basis_charges_the_reservation_not_the_tensor_bytes():
    allocations = rank_allocations(
        l2=0, snapshot=0, slot_state=0, segment=0, staging=0, headroom=0
    )
    allocations[1]["reservation"] = {
        "basis": "pinned_rounding",
        "requested_bytes": 400_000_000,
        "round_threshold_bytes": GIB,
    }
    document = manifest(
        [node("n", 4 * GIB, 0)],
        [container("c", "n", 4 * GIB)],
        [job("j", [[ranks("r", 2, {"container": "c"}, allocations)]], 2)],
    )
    report = evaluate_manifest(document)
    assert by_id(report["containers"], "c")["by_kind"]["snapshot_pool"] == 2 * (
        512 << 20
    )


# --- the example and the CLI -----------------------------------------------


def test_example_manifest_fits_and_report_is_deterministic(capsys):
    code, text = run_cli(["cache-budget", str(EXAMPLE), "--format", "text"], capsys)
    assert code == 0
    assert "declared Host memory budget estimate" in text
    assert "Status: fits" in text

    code, first = run_cli(["cache-budget", str(EXAMPLE), "--format", "json"], capsys)
    assert code == 0
    _, second = run_cli(["cache-budget", str(EXAMPLE), "--format", "json"], capsys)
    assert first == second
    report = json.loads(first)
    assert report["status"] == "fits"
    assert report["report_kind"] == "declared_host_memory_budget_estimate"
    assert report["errors"] == []
    assert {n["id"] for n in report["nodes"]} == {"node-a", "node-b"}
    # The shared segment of the prefill container counts once, not per rank.
    shared = [a for a in report["allocations"] if a["sharing"] == "shared"]
    assert [(a["key"], a["instances"], a["user_processes"]) for a in shared] == [
        ("shared/prefill-ipc-shm", 1, 4)
    ]
    # The decode ranks' slot-state arena is inside the measured snapshot.
    slot_state = [
        a for a in report["allocations"] if a["key"] == "chat-decode/0/ranks/slot-state"
    ]
    assert slot_state[0]["total_bytes"] == 0
    assert slot_state[0]["included_in"] == "chat-decode/0/ranks/snapshot"


def test_cli_requires_an_explicit_format(capsys):
    code, _ = run_cli(["cache-budget", str(EXAMPLE)], capsys)
    assert code == 2


def test_cli_overcommit_exits_one(tmp_path, capsys):
    document = manifest(
        [node("n", 100 * GIB, 0)],
        [container("c", "n", 10 * GIB)],
        [one_rank_job("j", {"container": "c"}, 10 * GIB)],
    )
    code, out = run_cli(
        ["cache-budget", write_manifest(tmp_path, document), "--format", "json"],
        capsys,
    )
    assert code == 1
    report = json.loads(out)
    assert report["status"] == "overcommitted"
    assert by_id(report["containers"], "c")["free_bytes"] == -GIB
    assert by_id(report["nodes"], "n")["fits"]

    code, text = run_cli(
        ["cache-budget", write_manifest(tmp_path, document), "--format", "text"],
        capsys,
    )
    assert code == 1
    assert "Container c: OVERCOMMITTED" in text


def test_cli_invalid_and_unreadable_manifests_exit_two(tmp_path, capsys):
    path = tmp_path / "broken.json"
    path.write_text('{"schema_version": 1, "nodes": [')
    code, out = run_cli(["cache-budget", str(path), "--format", "json"], capsys)
    assert code == 2
    report = json.loads(out)
    assert report["status"] == "invalid"
    assert report["nodes"] == [] and report["allocations"] == []
    assert report["errors"][0]["path"] == "$"

    code, out = run_cli(
        ["cache-budget", str(tmp_path / "missing.json"), "--format", "text"], capsys
    )
    assert code == 2
    assert "cannot read manifest" in out


# --- limits at the boundary ------------------------------------------------


@pytest.mark.parametrize("extra, fits", [(0, True), (1, False)])
def test_container_limit_boundary(extra, fits):
    allocations = rank_allocations(
        l2=2 * GIB + 1, snapshot=0, slot_state=0, segment=0, staging=0, headroom=GIB
    )
    allocations.append(allocation("extra", "other", "private", declared(extra)))
    # Two ranks of 2 GiB + 1 byte L2 and 1 GiB headroom each, plus 2 x extra.
    document = manifest(
        [node("n", 100 * GIB, 0)],
        [container("c", "n", 6 * GIB + 2)],
        [job("j", [[ranks("r", 2, {"container": "c"}, allocations)]], 2)],
    )
    report = evaluate_manifest(document)
    entry = by_id(report["containers"], "c")
    assert entry["total_bytes"] == 6 * GIB + 2 + 2 * extra
    assert entry["fits"] is fits
    assert report["status"] == ("fits" if fits else "overcommitted")


@pytest.mark.parametrize("extra, fits", [(0, True), (1, False)])
def test_node_limit_includes_system_reservation(extra, fits):
    document = manifest(
        [node("n", 12 * GIB, 2 * GIB)],
        [],
        [one_rank_job("j", {"node": "n"}, 9 * GIB + extra)],
    )
    report = evaluate_manifest(document)
    entry = by_id(report["nodes"], "n")
    assert entry["total_bytes"] == 12 * GIB + extra
    assert entry["free_bytes"] == -extra
    assert entry["fits"] is fits


# --- node and container accounting -------------------------------------------


def test_container_allocations_count_once_toward_their_node():
    # The node holds exactly what the container's processes allocate: neither
    # the container's (larger) limit nor a second copy of its usage counts.
    document = manifest(
        [node("n", 13 * GIB, 2 * GIB)],
        [container("c", "n", 50 * GIB)],
        [one_rank_job("j", {"container": "c"}, 10 * GIB)],
    )
    report = evaluate_manifest(document)
    entry = by_id(report["nodes"], "n")
    assert entry["allocated_bytes"] == 11 * GIB
    assert entry["total_bytes"] == 13 * GIB
    assert entry["fits"]
    assert by_id(report["containers"], "c")["total_bytes"] == 11 * GIB
    assert report["status"] == "fits"
    # A limit above the node is reported, but only allocations are checked.
    assert any("container limits total" in w for w in report["warnings"])


def test_node_and_container_are_both_enforced():
    # The container fits its limit; the node, holding a bare process too,
    # does not.
    document = manifest(
        [node("n", 20 * GIB, 0)],
        [container("c", "n", 12 * GIB)],
        [
            one_rank_job("in-container", {"container": "c"}, 10 * GIB),
            one_rank_job("bare", {"node": "n"}, 9 * GIB),
        ],
    )
    report = evaluate_manifest(document)
    assert by_id(report["containers"], "c")["fits"]
    node_entry = by_id(report["nodes"], "n")
    assert node_entry["total_bytes"] == 21 * GIB
    assert node_entry["by_job"] == {"bare": 10 * GIB, "in-container": 11 * GIB}
    assert not node_entry["fits"]
    assert report["status"] == "overcommitted"


def test_independent_jobs_that_each_fit_alone_overcommit_their_node():
    # Each job's ranks check only their own share when they start; two
    # independent jobs on one node are only seen together here.
    nodes = [node("n", 24 * GIB, 2 * GIB)]
    containers = [container("a", "n", 16 * GIB), container("b", "n", 16 * GIB)]
    job_a = one_rank_job("job-a", {"container": "a"}, 11 * GIB)
    job_b = one_rank_job("job-b", {"container": "b"}, 11 * GIB)
    for alone in (job_a, job_b):
        assert evaluate_manifest(manifest(nodes, containers, [alone]))["status"] == (
            "fits"
        )
    report = evaluate_manifest(manifest(nodes, containers, [job_a, job_b]))
    assert report["status"] == "overcommitted"
    entry = by_id(report["nodes"], "n")
    assert entry["by_job"] == {"job-a": 12 * GIB, "job-b": 12 * GIB}
    assert entry["free_bytes"] == -2 * GIB
    assert all(c["fits"] for c in report["containers"])


def test_replicas_count_separately_in_their_containers():
    allocations = rank_allocations(
        l2=4 * GIB, snapshot=GIB, slot_state=0, segment=2 * GIB, staging=0, headroom=GIB
    )
    document = manifest(
        [node("n", 100 * GIB, 0)],
        [container("c0", "n", 16 * GIB), container("c1", "n", 16 * GIB)],
        [
            job(
                "j",
                [
                    [ranks("r", 2, {"container": "c0"}, allocations)],
                    [ranks("r", 2, {"container": "c1"}, allocations)],
                ],
                2,
            )
        ],
    )
    report = evaluate_manifest(document)
    for container_id in ("c0", "c1"):
        assert by_id(report["containers"], container_id)["total_bytes"] == 16 * GIB
    assert by_id(report["nodes"], "n")["by_job"] == {"j": 32 * GIB}
    assert report["status"] == "fits"


# --- shared allocations ----------------------------------------------------


def test_shared_allocation_counts_once_in_its_container():
    base = rank_allocations(
        l2=GIB, snapshot=0, slot_state=0, segment=0, staging=0, headroom=0
    )
    shared = allocation("shm", "other", "shared", declared(4 * GIB))
    document = manifest(
        [node("n", 100 * GIB, 0)],
        [container("c", "n", 100 * GIB)],
        [
            job(
                "j",
                [
                    [
                        ranks("a", 2, {"container": "c"}, [*base, shared]),
                        ranks("b", 2, {"container": "c"}, [*base, shared]),
                    ]
                ],
                4,
            )
        ],
    )
    report = evaluate_manifest(document)
    entry = by_id(report["containers"], "c")
    assert entry["total_bytes"] == 4 * GIB + 4 * GIB
    assert entry["shared_bytes"] == 4 * GIB
    assert entry["by_job"] == {"j": 4 * GIB}
    (record,) = [a for a in report["allocations"] if a["key"] == "shared/shm"]
    assert record["users"] == ["j/0/a", "j/0/b"]
    assert record["user_processes"] == 4


def test_contradictory_shared_declarations_are_rejected():
    base = rank_allocations(
        l2=GIB, snapshot=0, slot_state=0, segment=0, staging=0, headroom=0
    )

    def two_users(
        first: dict[str, Any], second: dict[str, Any], second_placement: dict[str, str]
    ) -> dict[str, Any]:
        return manifest(
            [node("n", 100 * GIB, 0)],
            [container("c", "n", 100 * GIB), container("d", "n", 100 * GIB)],
            [
                job(
                    "j",
                    [
                        [
                            ranks("a", 1, {"container": "c"}, [*base, first]),
                            ranks("b", 1, second_placement, [*base, second]),
                        ]
                    ],
                    2,
                )
            ],
        )

    shared = allocation("shm", "other", "shared", declared(GIB))
    bigger = allocation("shm", "other", "shared", declared(2 * GIB))
    as_staging = allocation("shm", "l3_client_staging", "shared", declared(GIB))
    private = allocation("shm", "other", "private", declared(GIB))

    messages = [m for _, m in issues_of(two_users(shared, bigger, {"container": "c"}))]
    assert any("has reservation" in m for m in messages)
    messages = [m for _, m in issues_of(two_users(shared, shared, {"container": "d"}))]
    assert any("has memory domain container d" in m for m in messages)
    messages = [m for _, m in issues_of(two_users(shared, shared, {"node": "n"}))]
    assert any("memory domain node n outside any container" in m for m in messages)
    # A rank-local staging buffer cannot alias shared memory; declaring a
    # second one also breaks the exactly-once rule.
    messages = [
        m for _, m in issues_of(two_users(shared, as_staging, {"container": "c"}))
    ]
    assert any("l3_client_staging is allocated by each process" in m for m in messages)
    messages = [m for _, m in issues_of(two_users(shared, private, {"container": "c"}))]
    assert any("both shared and private" in m for m in messages)


def test_per_process_buffers_cannot_be_shared():
    allocations = rank_allocations(
        l2=GIB, snapshot=0, slot_state=0, segment=0, staging=0, headroom=0
    )
    allocations[0]["sharing"] = "shared"
    document = manifest(
        [node("n", 100 * GIB, 0)],
        [],
        [job("j", [[ranks("r", 2, {"node": "n"}, allocations)]], 2)],
    )
    paths = [path for path, _ in issues_of(document)]
    assert paths[0].endswith(".allocations[0].sharing")


# --- invalid and incomplete manifests ----------------------------------------


def valid_manifest() -> dict[str, Any]:
    allocations = rank_allocations(
        l2=GIB, snapshot=GIB, slot_state=GIB, segment=0, staging=0, headroom=GIB
    )
    return manifest(
        [node("n", 100 * GIB, GIB)],
        [container("c", "n", 50 * GIB)],
        [job("j", [[ranks("r", 2, {"container": "c"}, allocations)]], 2)],
    )


def rank_entry(document: dict[str, Any]) -> dict[str, Any]:
    return document["jobs"][0]["replicas"][0]["processes"][0]


@pytest.mark.parametrize(
    "bad_bytes", [-1, 1.0, 1e9, True, "1073741824", None], ids=repr
)
def test_byte_counts_must_be_nonnegative_integers(bad_bytes):
    document = valid_manifest()
    rank_entry(document)["allocations"][0]["reservation"]["bytes"] = bad_bytes
    paths = [path for path, _ in issues_of(document)]
    assert paths == [
        "$.jobs[0].replicas[0].processes[0].allocations[0].reservation.bytes"
    ]


def test_limits_and_counts_reject_bool_and_zero():
    document = valid_manifest()
    document["nodes"][0]["memory_bytes"] = True
    document["containers"][0]["memory_limit_bytes"] = 0
    rank_entry(document)["count"] = 0
    paths = {path for path, _ in issues_of(document)}
    assert {
        "$.nodes[0].memory_bytes",
        "$.containers[0].memory_limit_bytes",
        "$.jobs[0].replicas[0].processes[0].count",
    } <= paths


def test_missing_memory_is_not_assumed_zero():
    document = valid_manifest()
    del document["nodes"][0]["system_reserved_bytes"]
    allocations = rank_entry(document)["allocations"]
    rank_entry(document)["allocations"] = [
        a for a in allocations if a["kind"] != "l3_client_staging"
    ]
    issues = issues_of(document)
    assert ("$.nodes[0].system_reserved_bytes", "required field is missing") in issues
    assert any("exactly one l3_client_staging" in message for _, message in issues)


def test_nominal_bytes_are_refused_for_pinned_buffers():
    document = valid_manifest()
    rank_entry(document)["allocations"][0]["reservation"] = declared(GIB)
    ((path, message),) = issues_of(document)
    assert path.endswith(".allocations[0].reservation.basis")
    assert "measured" in message and "pinned_rounding" in message

    document = valid_manifest()
    rank_entry(document)["allocations"][0]["reservation"] = {
        "basis": "pinned_rounding",
        "requested_bytes": GIB,
    }
    ((path, message),) = issues_of(document)
    assert path.endswith(".reservation.round_threshold_bytes")


def test_missing_references_are_rejected():
    document = valid_manifest()
    document["containers"][0]["node"] = "elsewhere"
    assert issues_of(document) == [
        ("$.containers[0].node", "references unknown node 'elsewhere'")
    ]

    document = valid_manifest()
    rank_entry(document)["placement"] = {"container": "nowhere"}
    ((path, message),) = issues_of(document)
    assert path.endswith(".placement.container")
    assert "'nowhere'" in message

    document = valid_manifest()
    rank_entry(document)["placement"] = {"node": "n", "container": "c"}
    ((path, _),) = issues_of(document)
    assert path.endswith(".placement")

    document = valid_manifest()
    rank_entry(document)["allocations"][2]["reservation"] = {
        "basis": "included_in",
        "allocation": "nowhere",
    }
    ((_, message),) = issues_of(document)
    assert "'nowhere'" in message


def test_included_in_needs_a_measured_target():
    document = valid_manifest()
    allocations = rank_entry(document)["allocations"]
    allocations[1]["reservation"] = {
        "basis": "pinned_rounding",
        "requested_bytes": GIB,
        "round_threshold_bytes": GIB,
    }
    allocations[2]["reservation"] = {"basis": "included_in", "allocation": "snapshot"}
    ((_, message),) = issues_of(document)
    assert "'measured' or 'declared'" in message


def test_counts_must_match_the_declared_replicas_and_ranks():
    document = valid_manifest()
    document["jobs"][0]["replica_count"] = 2
    rank_entry(document)["count"] = 3
    messages = [message for _, message in issues_of(document)]
    assert any("lists 1 replica(s), but replica_count is 2" in m for m in messages)
    assert any("runs 3 inference_rank" in m for m in messages)


def test_identities_must_be_unique():
    document = valid_manifest()
    document["nodes"].append(copy.deepcopy(document["nodes"][0]))
    document["jobs"].append(copy.deepcopy(document["jobs"][0]))
    rank_entry(document)["allocations"].append(
        allocation("l2", "other", "private", declared(0))
    )
    messages = [message for _, message in issues_of(document)]
    assert "duplicate node id 'n'" in messages
    assert "duplicate job id 'j'" in messages
    assert "duplicate allocation id 'l2'" in messages


def test_unknown_fields_and_roles_are_rejected():
    document = valid_manifest()
    document["nodes"][0]["memroy_bytes"] = 1
    rank_entry(document)["allocations"].append(
        allocation("seg", "provider_segment", "private", declared(GIB))
    )
    issues = issues_of(document)
    assert ("$.nodes[0].memroy_bytes", "unknown field") in issues
    assert any("does not have provider_segment memory" in m for _, m in issues)


@pytest.mark.parametrize(
    "text, message",
    [
        ('{"a": 1, "a": 2}', "duplicate JSON key(s): a"),
        ('{"a": NaN}', "non-finite number NaN"),
        ("[1,", "Expecting value"),
    ],
)
def test_manifest_text_must_be_strict_json(text, message):
    with pytest.raises(ManifestInvalid) as raised:
        parse_manifest_json(text)
    assert message in raised.value.issues[0].message


@pytest.mark.parametrize("value", [[], {}, True, None, 1])
@pytest.mark.parametrize(
    "path",
    [
        ("role",),
        ("allocations", 0, "kind"),
        ("allocations", 0, "sharing"),
        ("allocations", 0, "reservation", "basis"),
    ],
)
def test_cli_rejects_non_string_choices(tmp_path, capsys, path, value):
    document = valid_manifest()
    target = rank_entry(document)
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value

    code, output = run_cli(
        ["cache-budget", write_manifest(tmp_path, document), "--format", "json"],
        capsys,
    )

    report = json.loads(output)
    assert code == 2
    assert report["status"] == "invalid"
    assert report["allocations"] == []
    assert any(error["path"].endswith(f".{path[-1]}") for error in report["errors"])


def test_cli_reports_excessive_json_nesting_as_invalid(tmp_path, capsys):
    path = tmp_path / "nested.json"
    depth = sys.getrecursionlimit() * 2
    path.write_text("[" * depth + "]" * depth)

    code, output = run_cli(["cache-budget", str(path), "--format", "json"], capsys)

    report = json.loads(output)
    assert code == 2
    assert report["status"] == "invalid"
    assert report["errors"][0]["path"] == "$"


@pytest.mark.parametrize("kind", ["l3_client_segment", "l3_client_staging"])
def test_rank_local_client_buffers_cannot_be_deduplicated(kind):
    document = valid_manifest()
    for entry in rank_entry(document)["allocations"]:
        if entry["kind"] == kind:
            entry["sharing"] = "shared"

    assert any(
        f"{kind} is allocated by each process" in m for _, m in issues_of(document)
    )
