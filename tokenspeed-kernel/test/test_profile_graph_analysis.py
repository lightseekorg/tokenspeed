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

"""Synthetic CPU fixtures test serialization and joins, never GPU correctness."""

import copy
import hashlib
import json
import sqlite3
import struct
import subprocess
import sys
from pathlib import Path

import pytest
from tokenspeed_kernel.op_profile import CostRegistry
from tokenspeed_kernel.profile_graph_analysis import (
    analyze_graph_replays,
    decode_replay,
    default_cost_registry,
    file_source,
    native_coverage_receipt,
)
from tokenspeed_kernel.profile_native_report import audit_nvtx_locators


@pytest.fixture
def graph_case(tmp_path):
    capture = json.loads(
        (
            Path(__file__).parent / "data/profile_graph_synthetic_capture.json"
        ).read_text()
    )
    # The collector's original CPU fixture records useful FLOPs but deliberately
    # has no proven compute engine. Tests retain that missing normalization.
    capture_path = tmp_path / "capture.json"
    capture_path.write_text(json.dumps(capture))
    payload = tmp_path / "metadata.bin"
    payload.write_bytes(struct.pack("<i", 7) + bytes(4) + struct.pack("<ii", 2, 2))
    row = dict(
        run_id=capture["run_id"],
        rank=0,
        step_id=f"{capture['run_id']}:r0:s42",
        serving_step=42,
        forward_id=900,
        event="device_results_ready",
        process_id=capture["process_id"],
        replay_thread_id=12345,
        capture_id=capture["capture_id"],
        capture_file=str(capture_path),
        replay_nvtx_label=f"tokenspeed.profile.replay | {capture['run_id']}:r0:s42",
        metadata_lease=dict(
            slot=0, generation=1, step_id=f"{capture['run_id']}:r0:s42"
        ),
        binary_payload={**file_source(payload), "bytes": 16},
        unavailable_reason=None,
    )
    index = tmp_path / "replays.jsonl"
    index.write_text(json.dumps(row) + "\n")
    window = tmp_path / "window.json"
    window.write_text(
        json.dumps(
            dict(
                schema="tokenspeed.profile_window/v1",
                run_id=capture["run_id"],
                rank=0,
                writer_flushed=True,
                writer_error=None,
                unresolved_result_metadata=0,
                dropped_metadata=0,
                selected_steps=1,
                device_results_ready=1,
                persisted_payloads=1,
            )
        )
    )
    report = tmp_path / "synthetic.nsys-rep"
    report.write_bytes(b"synthetic fixture, not an Nsight GPU capture")
    database = tmp_path / "synthetic.sqlite"
    tid = (capture["process_id"] << 24) | 12345
    pid = capture["process_id"] << 24
    with sqlite3.connect(database) as db:
        db.executescript("""
        CREATE TABLE StringIds(id INTEGER PRIMARY KEY,value TEXT);
        CREATE TABLE NVTX_EVENTS(eventId INTEGER PRIMARY KEY,start INTEGER,end INTEGER,
            globalTid INTEGER,endGlobalTid INTEGER,textId INTEGER,text TEXT,uint64Value INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,globalTid INTEGER,
            nameId INTEGER,correlationId INTEGER);
        CREATE TABLE CUDA_GRAPH_NODE_EVENTS(graphNodeId INTEGER,originalGraphNodeId INTEGER,globalTid INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,graphNodeId INTEGER,
            correlationId INTEGER,globalPid INTEGER);
        """)
        db.executemany(
            "INSERT INTO StringIds VALUES (?,?)",
            [(1, row["replay_nvtx_label"]), (2, "cudaGraphLaunch_v10000")],
        )
        db.execute("INSERT INTO NVTX_EVENTS VALUES (1,10,20,?,?,1,NULL,42)", (tid, tid))
        db.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (11,19,?,2,5)", (tid,)
        )
        db.executemany(
            "INSERT INTO CUDA_GRAPH_NODE_EVENTS VALUES (?,?,?)",
            [(1101, 101, tid), (1102, 102, tid), (1201, 201, tid), (1202, 202, tid)],
        )
        db.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?)",
            [
                (100, 130, 1101, 5, pid),
                (120, 125, 1102, 5, pid),
                (140, 180, 1201, 5, pid),
                (170, 175, 1202, 5, pid),
            ],
        )
    return dict(
        capture=capture,
        row=row,
        payload=payload,
        capture_path=capture_path,
        database=database,
        report=report,
        replays=index,
        window=window,
    )


def analyze(case, registry=None):
    return analyze_graph_replays(
        database=case["database"],
        report=case["report"],
        replays=case["replays"],
        window=case["window"],
        registry=(
            default_cost_registry(hbm_policy="logical_io_once")
            if registry is None
            else registry
        ),
        hardware=None,
        evidence_kind="synthetic_cpu_fixture",
    )


def test_actual_replay_values_costs_and_exact_nodes(graph_case):
    result = analyze(graph_case)
    step = result["steps"][0]
    assert step["graph_node_ids"] == [101, 201]
    assert step["observer_graph_node_ids"] == [102, 202]
    assert [row["cost"]["useful_flops"] for row in step["replay_analysis"]["ops"]] == [
        280,
        288,
    ]
    assert step["attributed_scope_ids"] == [0, 1, 2, 3]
    assert step["attributed_layer_ids"] == [0]
    model = next(
        row for row in result["device_analysis"]["scopes"] if row["scope_id"] == 0
    )
    assert (
        model["gpu_duration_ns"],
        model["gpu_activity_sum_ns"],
        model["gpu_activity_union_ns"],
    ) == (80, 70, 70)
    assert model["mfu_total"] is None  # synthetic engine identity is unproven
    assert "compute_normalization_unavailable" in model["critical_unknowns"]
    assert result["quantity_scope"] == "per_rank"


def test_dynamic_values_are_not_capture_time_values(graph_case):
    first = analyze(graph_case)["steps"][0]["replay_analysis"]["ops"][0]["cost"][
        "useful_flops"
    ]
    graph_case["payload"].write_bytes(
        struct.pack("<i", 3) + bytes(4) + struct.pack("<ii", 2, 2)
    )
    graph_case["row"]["binary_payload"].update(file_source(graph_case["payload"]))
    graph_case["replays"].write_text(json.dumps(graph_case["row"]) + "\n")
    second = analyze(graph_case)["steps"][0]["replay_analysis"]["ops"][0]["cost"][
        "useful_flops"
    ]
    assert (first, second) == (280, 120)


@pytest.mark.parametrize("problem", ["sha", "bounds", "generation", "overlap"])
def test_malformed_binary_or_lease_rejected(graph_case, problem):
    capture, row = copy.deepcopy(graph_case["capture"]), copy.deepcopy(
        graph_case["row"]
    )
    if problem == "sha":
        row["binary_payload"]["sha256"] = "0" * 64
    if problem == "bounds":
        capture["integer_snapshots"][0]["offset"] = 32
    if problem == "generation":
        row["metadata_lease"]["generation"] = 0
    if problem == "overlap":
        capture["integer_snapshots"][1]["offset"] = 0
    with pytest.raises(ValueError):
        decode_replay(capture, row, payload_path=graph_case["payload"])


@pytest.mark.parametrize(
    "mutation,reason",
    [
        (
            "DROP TABLE CUPTI_ACTIVITY_KIND_RUNTIME",
            "missing_native_sqlite_provenance_tables",
        ),
        (
            "DELETE FROM CUPTI_ACTIVITY_KIND_RUNTIME",
            "missing_or_ambiguous_exact_cudaGraphLaunch",
        ),
        (
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME",
            "missing_or_ambiguous_exact_cudaGraphLaunch",
        ),
        (
            "UPDATE NVTX_EVENTS SET globalTid=globalTid+1",
            "missing_or_ambiguous_exact_replay_nvtx",
        ),
        (
            "DELETE FROM CUDA_GRAPH_NODE_EVENTS WHERE graphNodeId=1101",
            "correlated_nodes_outside_capture_scopes",
        ),
    ],
)
def test_missing_ambiguous_and_wrong_thread_stay_partial(graph_case, mutation, reason):
    with sqlite3.connect(graph_case["database"]) as db:
        db.execute(mutation)
    result = analyze(graph_case)
    assert reason in result["steps"][0]["critical_unknowns"]
    model = next(
        row for row in result["device_analysis"]["scopes"] if row["scope_id"] == 0
    )
    assert model["mfu_total"] is None


@pytest.mark.parametrize(
    "mutation",
    [
        "INSERT INTO CUDA_GRAPH_NODE_EVENTS SELECT 1101,999,globalTid FROM CUDA_GRAPH_NODE_EVENTS LIMIT 1",
        "INSERT INTO CUDA_GRAPH_NODE_EVENTS SELECT 101,1101,globalTid FROM CUDA_GRAPH_NODE_EVENTS LIMIT 1",
    ],
)
def test_ambiguous_or_cyclic_clone_rejected(graph_case, mutation):
    with sqlite3.connect(graph_case["database"]) as db:
        db.execute(mutation)
    with pytest.raises(ValueError, match="clone"):
        analyze(graph_case)


def test_unknown_cost_rule_is_explicit(graph_case):
    result = analyze(graph_case, CostRegistry())
    assert all(
        row["cost"] is None for row in result["steps"][0]["replay_analysis"]["ops"]
    )
    assert any(
        "cost_coverage_incomplete" in problem
        for problem in result["steps"][0]["critical_unknowns"]
    )


def test_whole_service_forward_set_and_frozen_model_depth(graph_case, tmp_path):
    result = analyze(graph_case)
    config = tmp_path / "model-config.json"
    config.write_text('{"num_layers": 2}')
    step = result["steps"][0]
    metadata = tmp_path / "instrumentation.json"
    request = dict(
        rid="synthetic-rid",
        generation=1,
        first_count=1,
        last_count=2,
        forward_evidence=[
            dict(
                forward_id=900,
                step_id=step["step_id"],
                capture_id=step["capture_id"],
                expected_layer_ids=[0],
                expected_scope_ids=[0, 1, 2, 3],
                capture_file=step["capture_file"],
                binary_payload=step["binary_payload"],
            ),
            dict(
                forward_id=901,
                step_id="missing",
                capture_id=step["capture_id"],
                expected_layer_ids=[0],
                expected_scope_ids=[0, 1, 2, 3],
            ),
        ],
    )
    metadata.write_text(
        json.dumps(
            dict(
                run_id=result["run_id"],
                metadata_status="verified",
                requested_coverage_mode="full",
                model_config_source=file_source(config),
                requests={"0": request},
            )
        )
    )
    receipt = native_coverage_receipt(result, metadata=metadata, model_config=config)
    assert receipt["status"] == "partial"
    assert "synthetic_cpu_fixture_is_not_gpu_evidence" in receipt["critical_unknowns"]
    assert "missing_or_ambiguous_measured_forward" in receipt["critical_unknowns"]
    assert receipt["requests"]["0"]["forward_evidence"][0]["layer_coverage"][
        "expected_ids"
    ] == [0, 1]
    assert len(receipt["requests"]["0"]["forward_evidence"]) == 2


def test_nvtx_event_id_only_rekey_preserves_full_payload(tmp_path):
    connections = [sqlite3.connect(":memory:") for _ in range(2)]
    try:
        for db in connections:
            db.executescript(
                "CREATE TABLE StringIds(id INTEGER,value TEXT); INSERT INTO StringIds VALUES (7,'original');"
                "CREATE TABLE NVTX_EVENTS(eventId INTEGER PRIMARY KEY,start INTEGER,end INTEGER,textId INTEGER,payload INTEGER);"
                "INSERT INTO NVTX_EVENTS VALUES (1,10,20,7,99);"
            )
        connections[1].execute("UPDATE NVTX_EVENTS SET eventId=88")
        assert audit_nvtx_locators(*connections)["changed_locator_group_count"] == 1
        connections[1].execute("UPDATE NVTX_EVENTS SET payload=100")
        with pytest.raises(ValueError, match="multiset"):
            audit_nvtx_locators(*connections)
    finally:
        for db in connections:
            db.close()


def test_public_cli_import_without_cuda_or_triton():
    code = "import sys, torch; import tokenspeed_kernel.profile_nsys; import tokenspeed_kernel.profile_graph_analysis; assert not torch.cuda.is_initialized(); assert not any(x.startswith(('triton','tokenspeed_triton','tokenspeed_kernel._triton')) for x in sys.modules)"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_serialized_pure_and_fused_collectives_preserve_bandwidth(graph_case):
    capture = graph_case["capture"]

    def tensor(shape):
        elements = 1
        for dim in shape:
            elements *= dim
        stride = [shape[1], 1] if len(shape) == 2 else [1]
        return dict(
            shape=shape,
            stride=stride,
            dtype="torch.bfloat16",
            device="cuda:0",
            logical_bytes=elements * 2,
        )

    common = dict(ranks=[0, 1], group_name="attention_tp")
    pure, fused = capture["ops"]
    pure.update(
        cost_rule="nccl_all_reduce_sum.v1",
        op_name="all_reduce",
        inputs={"tensor": tensor([1, 4])},
        outputs={"return": tensor([1, 4])},
        scalar_args={**common, "is_sum": True},
        tensor_values={},
    )
    fused.update(
        cost_rule="all_reduce_residual_rmsnorm.v1",
        op_name="all_reduce_rmsnorm",
        inputs={
            "input_tensor": tensor([1, 4]),
            "residual": tensor([1, 4]),
            "weight": tensor([4]),
        },
        outputs={"return[0]": tensor([1, 4]), "return[1]": tensor([1, 4])},
        scalar_args={**common, "fused": True},
        tensor_values={},
    )
    capture.update(integer_snapshots=[], integer_metadata_bytes=0)
    graph_case["capture_path"].write_text(json.dumps(capture))
    graph_case["payload"].write_bytes(b"")
    graph_case["row"]["binary_payload"] = {
        **file_source(graph_case["payload"]),
        "bytes": 0,
    }
    graph_case["replays"].write_text(json.dumps(graph_case["row"]) + "\n")
    result = analyze(graph_case)
    rows = [row for row in result["device_analysis"]["scopes"] if row["kind"] == "op"]
    assert len(rows) == 2
    for row in rows:
        assert row["collective_count"] == 1
        assert row["communication_world_size"] == 2
        assert row["algbw_bytes_per_second"] > 0
        assert row["busbw_bytes_per_second"] > 0
        assert row["busbw_missing_reason"] is None
    assert rows[0]["useful_flops"] == 0
    assert rows[1]["useful_flops"] > 0


@pytest.mark.parametrize("node_type", [5, 6, 7, 0, None, 4])
def test_typed_dependency_nodes_preserve_full_ownership_and_engine_join(
    graph_case, node_type
):
    capture = graph_case["capture"]
    for scope in capture["graph_scopes"]:
        if 101 in scope["graph_node_ids"]:
            scope["graph_node_ids"].append(301)
    capture["graph_node_type_api"] = "cudaGraphNodeGetType"
    capture["graph_node_types"] = [
        dict(
            graph_node_id=301,
            graph_id=100,
            cuda_graph_node_type=node_type,
            unavailable_reason=None,
        )
    ]
    graph_case["capture_path"].write_text(json.dumps(capture))
    tid = (capture["process_id"] << 24) | 12345
    with sqlite3.connect(graph_case["database"]) as db:
        db.execute("INSERT INTO CUDA_GRAPH_NODE_EVENTS VALUES (1301,301,?)", (tid,))
    contexts = {}
    registry = default_cost_registry(hbm_policy="logical_io_once")

    def resolver(op, cost, context):
        contexts[op.trace_label] = context
        return None

    registry.register_engine_resolver(resolver)
    result = analyze(graph_case, registry)
    row = next(
        row for row in result["device_analysis"]["scopes"] if row["scope_id"] == 2
    )
    context = contexts[capture["ops"][0]["trace_label"]]
    assert row["owned_graph_node_ids"] == context["source_node_ids"] == [101, 301]
    assert context["join_evidence"]["clone_relations"]["1301"] == 301
    dependency = node_type in {5, 6, 7}
    assert context["complete_join"] is dependency
    assert row["missing_graph_node_ids"] == ([] if dependency else [301])
    assert row["dependency_graph_node_ids"] == ([301] if dependency else [])
    assert row["graph_node_ids"] == [101]
