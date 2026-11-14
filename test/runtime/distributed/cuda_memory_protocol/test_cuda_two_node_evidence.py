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

"""CPU regression tests for the downloadable serving evidence checker."""

import json
from test.ci_system.ci_register import register_cuda_ci

import pytest
from test_summarize_cuda_two_node_serving import summarize

# CPU regressions use the runtime job's complete dependency environment.
register_cuda_ci(est_time=5, suite="runtime-1gpu")


def write_case(tmp_path, layout, *, mixed=False, protocol=3, nccl=False):
    directory = tmp_path / "case"
    serving = directory / "serving"
    serving.mkdir(parents=True)
    protocols = [2, 3] if mixed else [protocol, protocol]
    tags = ["kv_cache", "weights"] + (["nccl"] if nccl else [])
    rows = []
    for rank in range(4):
        node = rank // 2
        rows.extend(
            [
                {
                    "event": "mapping",
                    "rank": rank,
                    "node": node,
                    "tp": [2 * node, 2 * node + 1],
                    "dp": [rank % 2, rank % 2 + 2],
                    "ep": [0, 1, 2, 3],
                    "mapped_memory_saver_hooks": [
                        "/test/torch_memory_saver_hook_mode_preload_cu12.abi3.so"
                    ],
                },
                {"event": "negotiated", "rank": rank, "protocol": protocols[node]},
                {
                    "event": "rendezvous_refused" if mixed else "service_ready",
                    "rank": rank,
                },
            ]
        )
        if not mixed:
            for release in (True, False):
                phases = [tags]
                if protocol == 3 and nccl:
                    phases = (
                        [["kv_cache", "weights"], ["nccl"]] if release else [[], tags]
                    )
                for index, phase in enumerate(phases):
                    stamp = (100 if release else 200) + index * 4
                    if nccl:
                        rows.append(
                            {
                                "event": "apply_begin",
                                "rank": rank,
                                "release": release,
                                "tags": phase,
                                "ns": stamp,
                            }
                        )
                        if "nccl" in phase:
                            count = 3 if layout == "ep4" else 5
                            operation = 1 if release else 2
                            rows.extend(
                                [
                                    {
                                        "event": "nccl_inventory",
                                        "rank": rank,
                                        "release": release,
                                        "operation": operation,
                                        "backend_ids": [
                                            rank * 100
                                            + i
                                            + 1
                                            + (0 if release else 10000)
                                            for i in range(count)
                                        ],
                                        "backend_group_names": [
                                            str(i) for i in range(count)
                                        ],
                                        # Registered CPU/unused groups need not
                                        # appear in the actual NCCL inventory.
                                        "registered_pg_names": [
                                            str(i) for i in range(count + 2)
                                        ],
                                        "ns": stamp + 1,
                                    },
                                    {
                                        "event": "nccl_complete",
                                        "rank": rank,
                                        "release": release,
                                        "operation": operation,
                                        "count": count,
                                        "ns": stamp + 2,
                                    },
                                ]
                            )
                    rows.append(
                        {
                            "event": "apply_complete",
                            "rank": rank,
                            "release": release,
                            "tags": phase,
                            "ns": stamp + 3 if nccl else stamp,
                        }
                    )
    (serving / "node-0.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    result = {
        "layout": layout,
        "protocols": protocols,
        "status": "rejected-before-ready" if mixed else "passed",
        "launcher_exits": [1, 1] if mixed else [0, 0],
    }
    if mixed:
        result["wire"] = {"HELLO": 0, "READY": 0, "input": 0}
    else:
        ingresses = range(4 if layout == "ep4" else 2)
        tokens = {str(rank): list(range(8)) for rank in ingresses}
        result.update(
            tokens_per_ingress=tokens,
            tokens_after_resume_per_ingress={
                key: value[:] for key, value in tokens.items()
            },
            token_count=8,
            tags=tags,
            release_ack_ns=[110] * len(ingresses),
            resume_ack_ns=[210] * len(ingresses),
        )
    (serving / "result.json").write_text(json.dumps(result))
    (directory / "serving.exit").write_text("0\n")
    return directory, result


@pytest.mark.parametrize("layout", ["ep4", "dp2-tp2-ep4"])
def test_complete_before_after_token_evidence_passes(tmp_path, layout):
    directory, result = write_case(tmp_path, layout)
    summary = summarize(directory)
    assert summary["tokens_per_ingress"] == result["tokens_per_ingress"]
    assert summary["tokens_after_resume_per_ingress"] == result["tokens_per_ingress"]


@pytest.mark.parametrize("layout", ["ep4", "dp2-tp2-ep4"])
@pytest.mark.parametrize("protocol", [2, 3])
def test_nccl_apply_phases_match_protocol(tmp_path, layout, protocol):
    directory, _ = write_case(tmp_path, layout, protocol=protocol, nccl=True)
    summary = summarize(directory)
    assert summary["protocols"] == [protocol, protocol]
    assert summary["nccl_backend_counts_per_rank"] == {
        str(rank): 3 if layout == "ep4" else 5 for rank in range(4)
    }


@pytest.mark.parametrize("protocol", [2, 3])
def test_nccl_inventory_can_vary_by_rank_without_topology_constants(tmp_path, protocol):
    directory, _ = write_case(tmp_path, "ep4", protocol=protocol, nccl=True)
    trace = directory / "serving/node-0.jsonl"
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    counts = {str(rank): rank + 2 for rank in range(4)}
    for row in rows:
        if row["event"] == "nccl_inventory":
            ids = [row["rank"] * 100 + i + 1 for i in range(counts[str(row["rank"])])]
            names = [str(i) for i in range(len(ids))]
            row["backend_ids"] = (
                ids if row["release"] else [value + 10000 for value in ids[::-1]]
            )
            row["backend_group_names"] = names if row["release"] else names[::-1]
            row["registered_pg_names"] = names + ["unused-group"]
        elif row["event"] == "nccl_complete":
            row["count"] = counts[str(row["rank"])]
    trace.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert summarize(directory)["nccl_backend_counts_per_rank"] == counts


@pytest.mark.parametrize("protocol", [2, 3])
def test_nccl_group_identity_allows_python_wrapper_id_reuse(tmp_path, protocol):
    directory, _ = write_case(tmp_path, "ep4", protocol=protocol, nccl=True)
    trace = directory / "serving/node-0.jsonl"
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    for row in rows:
        if row["event"] == "nccl_inventory" and not row["release"]:
            row["backend_ids"] = [value - 10000 for value in row["backend_ids"]]
    trace.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert summarize(directory)["protocols"] == [protocol, protocol]


@pytest.mark.parametrize("protocol", [2, 3])
@pytest.mark.parametrize(
    "damage",
    [
        "missing_rank",
        "legacy_without_inventory",
        "missing_inventory",
        "missing_complete",
        "duplicate_inventory",
        "duplicate_complete",
        "empty_inventory",
        "duplicate_backend",
        "wrong_count",
        "zero_count",
        "non_integer_count",
        "wrong_operation",
        "unpaired_operation",
        "unpaired_group_names",
        "missing_group_names",
        "missing_registry",
        "empty_group_name",
        "duplicate_group_name",
        "unregistered_group_name",
        "truncated_group_names",
        "reused_ids_with_different_groups",
        "wrong_release",
        "wrong_rank",
        "out_of_phase",
        "complete_before_inventory",
        "missing_apply_begin",
    ],
)
def test_incomplete_or_unpaired_nccl_inventory_is_rejected(tmp_path, protocol, damage):
    directory, _ = write_case(tmp_path, "ep4", protocol=protocol, nccl=True)
    trace = directory / "serving/node-0.jsonl"
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    inventory = next(
        row
        for row in rows
        if row["event"] == "nccl_inventory" and row["rank"] == 0 and row["release"]
    )
    complete = next(
        row
        for row in rows
        if row["event"] == "nccl_complete" and row["rank"] == 0 and row["release"]
    )
    if damage == "missing_rank":
        rows = [
            row
            for row in rows
            if row["rank"] != 0
            or row["event"] not in {"nccl_inventory", "nccl_complete"}
        ]
    elif damage == "legacy_without_inventory":
        rows = [
            row
            for row in rows
            if row["event"] not in {"nccl_inventory", "nccl_complete"}
        ]
    elif damage == "missing_inventory":
        rows.remove(inventory)
    elif damage == "missing_complete":
        rows.remove(complete)
    elif damage == "duplicate_inventory":
        rows.append(inventory.copy())
    elif damage == "duplicate_complete":
        rows.append(complete.copy())
    elif damage == "empty_inventory":
        inventory["backend_ids"] = []
    elif damage == "duplicate_backend":
        inventory["backend_ids"][-1] = inventory["backend_ids"][0]
    elif damage == "wrong_count":
        complete["count"] -= 1
    elif damage == "zero_count":
        complete["count"] = 0
    elif damage == "non_integer_count":
        complete["count"] = True
    elif damage == "wrong_operation":
        complete["operation"] += 1
    elif damage == "unpaired_operation":
        for row in rows:
            if row["rank"] == 0 and row["event"] in {"nccl_inventory", "nccl_complete"}:
                row["operation"] = 1
    elif damage == "unpaired_group_names":
        inventory["backend_group_names"][0] = "different-group"
        inventory["registered_pg_names"].append("different-group")
    elif damage == "missing_group_names":
        del inventory["backend_group_names"]
    elif damage == "missing_registry":
        del inventory["registered_pg_names"]
    elif damage == "empty_group_name":
        inventory["backend_group_names"][0] = ""
    elif damage == "duplicate_group_name":
        inventory["backend_group_names"][-1] = inventory["backend_group_names"][0]
    elif damage == "unregistered_group_name":
        inventory["backend_group_names"][0] = "unregistered-group"
    elif damage == "truncated_group_names":
        inventory["backend_group_names"].pop()
    elif damage == "reused_ids_with_different_groups":
        resumed = next(
            row
            for row in rows
            if row["event"] == "nccl_inventory"
            and row["rank"] == 0
            and not row["release"]
        )
        resumed["backend_ids"] = inventory["backend_ids"][:]
        resumed["backend_group_names"][0] = "different-group"
        resumed["registered_pg_names"].append("different-group")
    elif damage == "wrong_release":
        inventory["release"] = False
    elif damage == "wrong_rank":
        inventory["rank"] = 4
    elif damage == "out_of_phase":
        inventory["ns"] = 0
    elif damage == "complete_before_inventory":
        complete["ns"] = inventory["ns"] - 1
    elif damage == "missing_apply_begin":
        rows = [
            row
            for row in rows
            if not (
                row["event"] == "apply_begin"
                and row["rank"] == 0
                and row["release"]
                and "nccl" in row["tags"]
            )
        ]
    trace.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(AssertionError, match="NCCL"):
        summarize(directory)


@pytest.mark.parametrize("protocol", [2, 3])
@pytest.mark.parametrize("release", [True, False])
def test_all_rank_ack_barrier_applies_only_to_v3(tmp_path, protocol, release):
    directory, result = write_case(tmp_path, "dp2-tp2-ep4", protocol=protocol)
    trace = directory / "serving/node-0.jsonl"
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    operation = "release" if release else "resume"
    # A non-ingress rank finishes after another ingress's local v2 ACK.
    late_rank = next(
        row
        for row in rows
        if row["event"] == "apply_complete"
        and row["rank"] == 3
        and row["release"] == release
    )
    late_rank["ns"] = min(result[f"{operation}_ack_ns"]) + 1
    trace.write_text("".join(json.dumps(row) + "\n" for row in rows))
    if protocol == 2:
        assert summarize(directory)["protocols"] == [2, 2]
    else:
        with pytest.raises(AssertionError):
            summarize(directory)


@pytest.mark.parametrize("protocol", [2, 3])
@pytest.mark.parametrize("release", [True, False])
def test_nccl_inventory_does_not_add_a_v2_global_ack_barrier(
    tmp_path, protocol, release
):
    directory, result = write_case(tmp_path, "ep4", protocol=protocol, nccl=True)
    trace = directory / "serving/node-0.jsonl"
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    operation = "release" if release else "resume"
    ack = min(result[f"{operation}_ack_ns"])
    for row in rows:
        if row["rank"] != 3 or row.get("release") != release:
            continue
        if row["event"] == "nccl_inventory":
            row["ns"] = ack
        elif row["event"] == "nccl_complete":
            row["ns"] = ack + 1
        elif row["event"] == "apply_complete" and "nccl" in row["tags"]:
            row["ns"] = ack + 2
    trace.write_text("".join(json.dumps(row) + "\n" for row in rows))
    if protocol == 2:
        assert summarize(directory)["first_ack_after_last_rank_ms"][operation] <= 0
    else:
        with pytest.raises(AssertionError):
            summarize(directory)


@pytest.mark.parametrize("protocol", [2, 3])
@pytest.mark.parametrize("release", [True, False])
@pytest.mark.parametrize("damage", ["missing", "wrong_tags", "other_protocol"])
def test_missing_or_wrong_nccl_apply_phases_are_rejected(
    tmp_path, protocol, release, damage
):
    directory, _ = write_case(tmp_path, "ep4", protocol=protocol, nccl=True)
    trace = directory / "serving/node-0.jsonl"
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    phases = [
        row
        for row in rows
        if row["event"] == "apply_complete"
        and row["rank"] == 0
        and row["release"] == release
    ]
    if damage == "missing":
        rows.remove(phases[0])
    elif damage == "wrong_tags":
        phases[0]["tags"] = ["unexpected"]
    else:
        rows = [row for row in rows if row not in phases]
        tags = ["kv_cache", "weights", "nccl"]
        other_phases = [tags]
        if protocol == 2:
            other_phases = (
                [["kv_cache", "weights"], ["nccl"]] if release else [[], tags]
            )
        for index, tags in enumerate(other_phases):
            rows.append(
                {
                    "event": "apply_complete",
                    "rank": 0,
                    "release": release,
                    "tags": tags,
                    "ns": (100 if release else 200) + index,
                }
            )
    trace.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(AssertionError, match="apply phases"):
        summarize(directory)


@pytest.mark.parametrize("layout", ["ep4", "dp2-tp2-ep4"])
@pytest.mark.parametrize("operation", ["release", "resume"])
@pytest.mark.parametrize(
    "damage", ["missing", "empty", "truncated", "extra", "non_integer"]
)
def test_incomplete_or_invalid_ack_evidence_is_rejected(
    tmp_path, layout, operation, damage
):
    directory, result = write_case(tmp_path, layout)
    key = f"{operation}_ack_ns"
    if damage == "missing":
        del result[key]
    elif damage == "empty":
        result[key] = []
    elif damage == "truncated":
        result[key] = result[key][:1]
    elif damage == "extra":
        result[key].append(result[key][-1])
    elif damage == "non_integer":
        result[key][0] = str(result[key][0])
    (directory / "serving/result.json").write_text(json.dumps(result))
    with pytest.raises(AssertionError, match="ACK"):
        summarize(directory)


@pytest.mark.parametrize(
    "damage",
    [
        "empty",
        "missing_after",
        "wrong_ingress",
        "changed",
        "short",
        "non_integer",
        "wrong_count",
    ],
)
def test_incomplete_or_changed_token_evidence_is_rejected(tmp_path, damage):
    directory, result = write_case(tmp_path, "ep4")
    if damage == "empty":
        result["tokens_per_ingress"] = {}
        result["tokens_after_resume_per_ingress"] = {}
    elif damage == "missing_after":
        del result["tokens_after_resume_per_ingress"]
    elif damage == "wrong_ingress":
        result["tokens_per_ingress"]["4"] = result["tokens_per_ingress"].pop("3")
        result["tokens_after_resume_per_ingress"]["4"] = result[
            "tokens_after_resume_per_ingress"
        ].pop("3")
    elif damage == "changed":
        result["tokens_after_resume_per_ingress"]["0"][0] += 1
    elif damage == "short":
        result["tokens_after_resume_per_ingress"]["0"].pop()
    elif damage == "non_integer":
        result["tokens_per_ingress"]["0"][0] = "0"
        result["tokens_after_resume_per_ingress"]["0"][0] = "0"
    elif damage == "wrong_count":
        result["token_count"] = 7
    (directory / "serving/result.json").write_text(json.dumps(result))
    with pytest.raises(AssertionError):
        summarize(directory)


def test_missing_automatic_preload_evidence_is_rejected(tmp_path):
    directory, _ = write_case(tmp_path, "ep4")
    trace = directory / "serving/node-0.jsonl"
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    rows[0]["mapped_memory_saver_hooks"] = []
    trace.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(AssertionError):
        summarize(directory)


def test_missing_rank_trace_is_rejected(tmp_path):
    directory, _ = write_case(tmp_path, "ep4")
    trace = directory / "serving/node-0.jsonl"
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    trace.write_text(
        "".join(json.dumps(row) + "\n" for row in rows if row["rank"] != 3)
    )
    with pytest.raises(AssertionError):
        summarize(directory)


@pytest.mark.parametrize("layout", ["ep4", "dp2-tp2-ep4"])
def test_mixed_protocol_refusal_needs_no_generation_tokens(tmp_path, layout):
    directory, _ = write_case(tmp_path, layout, mixed=True)
    assert summarize(directory)["wire"] == {"HELLO": 0, "READY": 0, "input": 0}
