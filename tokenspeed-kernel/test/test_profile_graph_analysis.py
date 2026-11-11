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


def test_native_receipt_keeps_dropped_measured_forward_without_scope_expectations(
    graph_case, tmp_path
):
    original = graph_case["row"]
    dropped = {
        **original,
        "step_id": f"{original['run_id']}:r0:s43",
        "serving_step": 43,
        "forward_id": 901,
        "request_ids": ["synthetic-rid"],
        "device_ready_sequence": 2,
        "metadata_lease": None,
        "unavailable_reason": "metadata_dropped:ring_full",
    }
    del dropped["binary_payload"]
    with graph_case["replays"].open("a") as stream:
        stream.write(json.dumps(dropped) + "\n")
    window = json.loads(graph_case["window"].read_text())
    window.update(selected_steps=2, device_results_ready=2, dropped_metadata=1)
    graph_case["window"].write_text(json.dumps(window))
    result = analyze(graph_case)
    config = tmp_path / "model-config.json"
    config.write_text('{"num_layers": 1}')
    metadata = tmp_path / "instrumentation.json"
    identity = {name: dropped[name] for name in ("forward_id", "step_id", "capture_id")}
    good = result["steps"][0]
    metadata.write_text(
        json.dumps(
            dict(
                run_id=result["run_id"],
                metadata_status="incomplete",
                requested_coverage_mode="decode",
                model_config_source=file_source(config),
                requests={
                    "0": dict(
                        rid="synthetic-rid",
                        generation=1,
                        first_count=1,
                        last_count=2,
                        forward_evidence=[
                            dict(
                                forward_id=good["forward_id"],
                                step_id=good["step_id"],
                                capture_id=good["capture_id"],
                                expected_layer_ids=[0],
                                expected_scope_ids=good["expected_scope_ids"],
                                capture_file=good["capture_file"],
                                binary_payload=good["binary_payload"],
                            ),
                            identity,
                        ],
                    )
                },
            )
        )
    )
    receipt = native_coverage_receipt(result, metadata=metadata, model_config=config)
    assert receipt["status"] == receipt["whole_run_status"] == "partial"
    assert receipt["selected_forward_ids"] == [900, 901]
    assert "collector_metadata_dropped" in receipt["critical_unknowns"]
    complete, partial = receipt["requests"]["0"]["forward_evidence"]
    assert complete["scope_coverage"]["missing_scope_ids"] == []
    assert {key: partial[key] for key in identity} == identity
    assert partial["request_ids"] == ["synthetic-rid"]
    assert partial["device_ready_sequence"] == 2
    assert partial["unavailable_reason"] == "metadata_dropped:ring_full"
    assert "metadata_dropped:ring_full" in partial["critical_unknowns"]
    assert "measured_scope_expectations_unavailable" in partial["critical_unknowns"]
    assert partial["scope_coverage"] == {
        "expected_scope_ids": None,
        "attributed_scope_ids": [],
        "missing_scope_ids": None,
    }
    assert partial["layer_coverage"] == {
        "expected_ids": [0],
        "attributed_ids": [],
        "missing_ids": [0],
    }


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


@pytest.fixture
def mixed_case(graph_case, tmp_path):
    import csv

    case = graph_case
    graph = case["row"]
    graph.update(request_ids=["rid"], request_pool_indices=[0])
    config_sha256 = "a" * 64
    index = json.loads(case["window"].read_text())
    index.update(
        selected_steps=2,
        device_results_ready=2,
        persisted_eager=1,
        profile_config_sha256=config_sha256,
    )
    case["window"].write_text(json.dumps(index))
    fields = dict(
        run_id=graph["run_id"],
        rank=0,
        step_id=f"{graph['run_id']}:r0:s43",
        serving_step=43,
        request_ids=["rid"],
        request_pool_indices=[0],
        forward_id=None,
    )
    eager = dict(
        **fields,
        event="device_results_ready",
        execution_mode="eager",
        process_id=graph["process_id"],
    )
    capture = case["capture"]
    cost = analyze_graph_replays(
        database=case["database"],
        report=case["report"],
        replays=case["replays"],
        window=case["window"],
        registry=default_cost_registry(hbm_policy="logical_io_once"),
        hardware=None,
        evidence_kind="synthetic_cpu_fixture",
    )["steps"][0]["replay_analysis"]["ops"][0]
    assert cost["cost"] is not None
    cost["cost"]["compute_type"] = "fp32_simt"
    cost["iter_id"] = 43
    op = dict(capture["ops"][0], iter_id=43, trace_label="op")
    op["tensor_values"] = {**op["tensor_values"], "cache_seqlens": [7]}
    report = {
        "schema_version": 1,
        "run_id": f"{graph['run_id']}:eager:43",
        "rank": 0,
        "model_id": capture["model_id"],
        "hardware": capture["hardware"],
        "eager_step": dict(
            **fields,
            profile_config_sha256=config_sha256,
            schema="tokenspeed.profile_eager_step/v1",
        ),
        "layers": [
            dict(layer, iter_id=43, trace_label=label)
            for layer, label in zip(capture["layers"], ("model", "layer"), strict=True)
        ],
        "ops": [op],
        "analysis": {"ops": [cost]},
    }
    id_base = (1 << 53) + 1
    labels = ["Model", "L0", "attention.mla"]
    report["device_analysis"] = dict(
        timing_kind="pending_nsight",
        nvtx_payload_type="int64",
        process_id=graph["process_id"],
        thread_id=12345,
        device="cuda:0",
        scopes=[
            dict(trace_label=key, nvtx_label=label, nvtx_id=id_base + offset)
            for offset, (key, label) in enumerate(zip(("model", "layer", "op"), labels))
        ],
    )
    sidecar = tmp_path / "eager.json"
    sidecar.write_text(json.dumps(report))
    eager["eager_sidecar"] = dict(**file_source(sidecar), bytes=sidecar.stat().st_size)
    binding = dict(fields, forward_id=901, event="forward_identity_bound")
    case["replays"].write_text(
        "".join(json.dumps(row) + "\n" for row in (graph, binding, eager))
    )
    tid = (graph["process_id"] << 24) | 12345
    ranges = [(30, 50), (32, 45), (34, 37)]
    with sqlite3.connect(case["database"]) as db:
        db.executescript("""
            ALTER TABLE NVTX_EVENTS ADD COLUMN domainId INTEGER;
            ALTER TABLE NVTX_EVENTS ADD COLUMN eventType INTEGER;
            ALTER TABLE NVTX_EVENTS ADD COLUMN int64Value INTEGER;
            ALTER TABLE CUPTI_ACTIVITY_KIND_RUNTIME ADD COLUMN callchainId INTEGER;
        """)
        db.execute(
            "INSERT INTO NVTX_EVENTS(start,globalTid,text,domainId,eventType) VALUES(0,?,'TokenSpeed',4,75)",
            (tid,),
        )
        db.execute("INSERT INTO StringIds VALUES(3,'cudaLaunchKernel')")
        for offset, ((start, end), label) in enumerate(zip(ranges, labels)):
            db.execute(
                """
                INSERT INTO NVTX_EVENTS(start,end,globalTid,endGlobalTid,text,domainId,eventType,int64Value)
                VALUES(?,?,?,?,?,4,59,?)
                """,
                (start, end, tid, tid, label, id_base + offset),
            )
        db.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME(start,end,globalTid,nameId,correlationId) VALUES(?,?,?,3,?)",
            [(35, 36, tid, 6), (38, 39, tid, 7)],
        )
        db.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?,NULL,?,?)",
            [(120, 135, 6, tid & ~0xFFFFFF), (175, 200, 7, tid & ~0xFFFFFF)],
        )
        db.execute(
            "INSERT INTO NVTX_EVENTS(start,globalTid,text,domainId,eventType) VALUES(0,?,'Foreign',8,75)",
            (tid,),
        )
        db.execute("""
            INSERT INTO NVTX_EVENTS(start,end,globalTid,endGlobalTid,text,domainId,eventType,int64Value)
            SELECT start,end,globalTid,endGlobalTid,text,8,59,int64Value
            FROM NVTX_EVENTS WHERE eventId=3
            """)
    projection = tmp_path / "projection.csv"
    rows = [
        {
            "RangeId": 3 + offset,
            "Name": f"TokenSpeed:{label}",
            "Orig Start (ns)": start,
            "Orig Duration (ns)": end - start,
            "PID": graph["process_id"],
            "TID": 12345,
            "Projected Start (ns)": 120,
            "Projected Duration (ns)": 80 if offset < 2 else 15,
            "NumGPUOps": 2 if offset < 2 else 1,
        }
        for offset, ((start, end), label) in enumerate(zip(ranges, labels))
    ]
    with projection.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    case.update(projection=projection, sidecar=sidecar, eager=eager, binding=binding)
    return case


def analyze_mixed(case, registry=None):
    return analyze_graph_replays(
        database=case["database"],
        report=case["report"],
        replays=case["replays"],
        window=case["window"],
        registry=(
            registry
            if registry is not None
            else default_cost_registry(hbm_policy="logical_io_once")
        ),
        hardware=None,
        evidence_kind="synthetic_cpu_fixture",
        eager_projection=case["projection"],
    )


@pytest.mark.parametrize("copy_kind", [2, 1])
def test_mixed_eager_keeps_profiler_d2h_outside_model_activity(mixed_case, copy_kind):
    import csv

    from tokenspeed_kernel.profile_nvtx import PROFILE_METADATA_COPY_LABEL

    case = mixed_case
    sidecar = json.loads(case["sidecar"].read_text())
    process = sidecar["device_analysis"]["process_id"]
    tid = (process << 24) | 12345
    scope_id = (1 << 53) + 10
    sidecar["device_analysis"].update(
        metadata_snapshot_bytes=16,
        metadata_copy_scopes=[
            {
                "trace_label": f"profiler.metadata_copy/{scope_id}",
                "nvtx_label": PROFILE_METADATA_COPY_LABEL,
                "nvtx_id": scope_id,
                "bytes": 16,
            }
        ],
    )
    case["sidecar"].write_text(json.dumps(sidecar))
    case["eager"]["eager_sidecar"] = dict(
        **file_source(case["sidecar"]), bytes=case["sidecar"].stat().st_size
    )
    case["replays"].write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in (case["row"], case["binding"], case["eager"])
        )
    )
    with sqlite3.connect(case["database"]) as db:
        db.execute("INSERT INTO StringIds VALUES(4,'cudaMemcpyAsync')")
        db.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME(start,end,globalTid,nameId,correlationId) VALUES(32,33,?,4,8)",
            (tid,),
        )
        observer = db.execute(
            """
            INSERT INTO NVTX_EVENTS(start,end,globalTid,endGlobalTid,text,domainId,eventType,int64Value)
            VALUES(32,34,?,?,?,4,59,?)
            """,
            (tid, tid, PROFILE_METADATA_COPY_LABEL, scope_id),
        ).lastrowid
        db.execute("""
            CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(
                start INTEGER, end INTEGER, graphNodeId INTEGER, correlationId INTEGER,
                globalPid INTEGER, copyKind INTEGER, bytes INTEGER
            )
            """)
        db.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES(95,110,NULL,8,?,?,16)",
            (tid & ~0xFFFFFF, copy_kind),
        )
    with case["projection"].open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows[:2]:
        row.update(
            {"Projected Start (ns)": 95, "Projected Duration (ns)": 105, "NumGPUOps": 3}
        )
    rows.append(
        {
            "RangeId": observer,
            "Name": f"TokenSpeed:{PROFILE_METADATA_COPY_LABEL}",
            "Orig Start (ns)": 32,
            "Orig Duration (ns)": 2,
            "PID": process,
            "TID": 12345,
            "Projected Start (ns)": 95,
            "Projected Duration (ns)": 15,
            "NumGPUOps": 1,
        }
    )
    with case["projection"].open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if copy_kind != 2:
        with pytest.raises(ValueError, match="differs from its metadata copy"):
            analyze_mixed(case)
        return
    result = analyze_mixed(case)
    eager = result["steps"][1]
    evidence = eager["eager_native_timing"]
    (copy,) = evidence["metadata_copy_activities"]
    assert copy["nvtx_id"] == scope_id and copy["memory"]["bytes"] == 16
    assert {item["correlationId"] for item in evidence["activities"]} == {6, 7}
    model = next(
        row
        for row in result["device_analysis"]["scopes"]
        if row["trace_label"] == "model"
    )
    assert (
        model["gpu_start_ns"],
        model["gpu_end_ns"],
        model["gpu_activity_count"],
    ) == (120, 200, 2)
    assert "eager_model_activity_outside_costed_ops" in eager["critical_unknowns"]


def test_mixed_scope_and_scheduler_ownership(mixed_case, tmp_path):
    from tokenspeed_kernel.profile_scheduler import analyze_scheduler

    case = mixed_case
    result = analyze_mixed(case)
    graph, eager = result["steps"]
    assert graph["attribution_mode"] == "cuda_graph_node_ids"
    assert eager["attribution_mode"] == "eager_nvtx_gpu_projection"
    assert eager["forward_id"] == 901
    assert eager["scope_kinds"] == ["layer", "model", "op"]
    assert eager["attributed_layer_ids"] == [0]
    assert eager["attributed_scope_ids"] == list(range((1 << 53) + 1, (1 << 53) + 4))
    assert result["collector_window"]["dropped_steps"] == []
    assert result["device_analysis"]["timing_kind"] == "nsight_mixed_graph_eager"
    model, layer, op = result["device_analysis"]["scopes"][-3:]
    assert [item["gpu_duration_ns"] for item in (model, layer, op)] == [80, 80, 15]
    assert op["parent_scope_id"] == layer["scope_id"]
    assert op["mfu"] is None and op["useful_flops"] > 0
    assert len(op["native_activity_rows"]) == 1

    commits = tmp_path / "commits.jsonl"
    commits.write_text(
        "".join(
            json.dumps(
                dict(
                    run_id=result["run_id"],
                    boundary="commit",
                    rid="rid",
                    generation=1,
                    observed_ns=1000 + count,
                    completion_tokens=count,
                    output_ids=[count],
                    forward_id=step["forward_id"],
                    pid=case["row"]["process_id"],
                    graph=dict(
                        run_id=result["run_id"], rank=0, step_id=step["step_id"]
                    ),
                )
            )
            + "\n"
            for count, step in enumerate((graph, eager), 1)
        )
    )
    metadata = tmp_path / "service.json"
    metadata.write_text(
        json.dumps(
            dict(
                schema="tokenspeed.metadata_coverage/v1",
                run_id=result["run_id"],
                window_manifest_source=file_source(case["window"]),
                window_manifest=json.loads(case["window"].read_text()),
                ready_index_source=file_source(case["replays"]),
                service_sources=[file_source(commits)],
                requests={},
            )
        )
    )
    scheduler = analyze_scheduler(
        case["database"],
        selection_mode="collector",
        missing_launch_policy="report_unavailable",
        registry=None,
        service_metadata=metadata,
        eager_steps={eager["step_id"]: eager},
    )
    assert scheduler["identified_graph_launch_count"] == 1
    assert scheduler["eager_marker_count"] == 1
    assert scheduler["gpu_timed_step_count"] == 2
    assert [step["forward_id"] for step in scheduler["steps"]] == [900, 901]
    assert all(step["authoritative_commit"] for step in scheduler["steps"])
    (pair,) = scheduler["consecutive_pairs"]
    assert pair["span_overlap_ns"] == 60
    assert pair["gpu_activity_intersection_ns"] == 15


@pytest.mark.parametrize(
    "failure,reason",
    [
        ("domain", "closed same-thread TokenSpeed push/pop range"),
        ("graph", "graph-owned eager GPU activity"),
        ("projection", "disagree with the official projection"),
        ("binding", "Late ForwardId binding mismatch"),
        ("sidecar", "Eager sidecar hash or size mismatch"),
    ],
)
def test_mixed_eager_rejects_wrong_native_evidence(mixed_case, failure, reason):
    case = mixed_case
    if failure == "domain":
        with sqlite3.connect(case["database"]) as db:
            db.execute("UPDATE NVTX_EVENTS SET domainId=99 WHERE eventId=3")
    elif failure == "graph":
        with sqlite3.connect(case["database"]) as db:
            db.execute(
                "UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET graphNodeId=10 WHERE correlationId=6"
            )
    elif failure == "projection":
        with sqlite3.connect(case["database"]) as db:
            db.execute(
                "UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET end=136 WHERE correlationId=6"
            )
    elif failure == "binding":
        case["binding"]["request_ids"] = ["wrong"]
        case["replays"].write_text(
            "".join(
                json.dumps(row) + "\n"
                for row in (case["row"], case["binding"], case["eager"])
            )
        )
    else:
        with case["sidecar"].open("a") as stream:
            stream.write("\n")
    with pytest.raises(ValueError, match=reason):
        analyze_mixed(case)


@pytest.mark.parametrize(
    "case_name",
    [
        "exact",
        "graph_only",
        "wrong_role",
        "missing_context",
        "conv_host_control",
        "conv_wrong_source",
        "conv_wrong_host_dtype",
        "conv_wrong_gpu_device",
    ],
)
def test_eager_provider_requires_reviewed_exact_native_op_proof(mixed_case, case_name):
    from dataclasses import asdict, replace

    from tokenspeed_kernel.profile_costs import register_selected_engine_resolver
    from tokenspeed_kernel.profile_utilization import HardwareProfile

    case = mixed_case
    digest = "b" * 64
    uuid = "abcdef0123456789" * 2
    hardware = HardwareProfile.from_file(
        Path(__file__).resolve().parents[2]
        / "docs/hardware-profiles/nvidia-h200-sxm-141gb.yaml"
    )
    hardware = replace(
        hardware,
        input_snapshot=None,
        device={**hardware.device, "compute_capability": "9.0"},
    )
    hardware = hardware.validate_device(
        dict(name="NVIDIA H200", partition="full_device", uuid=uuid)
    )
    sidecar = json.loads(case["sidecar"].read_text())
    sidecar["hardware"] = asdict(hardware)
    op = sidecar["ops"][0]
    is_conv = case_name.startswith("conv_")
    expected_flops, engine, peak = 280, "bf16_tensor", 989500000000000
    boundary = {"inputs": {"q": {"shape": op["inputs"]["q"]["shape"]}}}
    if is_conv:
        if case_name != "conv_wrong_source":
            digest = "c42179c6e4c51d784ad9dc259094d9b5177b2ac121eae67874dd7cf966b22d5e"
        x = dict(
            shape=[8, 2],
            stride=[2, 1],
            device="cuda:0",
            dtype="torch.bfloat16",
            logical_bytes=32,
        )
        weight = dict(x, shape=[8, 4], stride=[4, 1], logical_bytes=64)
        lengths = dict(
            shape=[2], stride=[1], device="cuda:0", dtype="torch.int32", logical_bytes=8
        )
        host_lengths = dict(
            lengths,
            device="cpu",
            dtype="torch.int64",
            logical_bytes=16,
            storage_offset=0,
        )
        if case_name == "conv_wrong_host_dtype":
            host_lengths.update(dtype="torch.int32", logical_bytes=8)
        if case_name == "conv_wrong_gpu_device":
            lengths["device"] = "cuda:1"
        op.update(
            op_name="conv.causal_conv1d_prefill",
            implementation="triton_causal_conv1d_prefill",
            cost_rule="causal_conv1d_prefill.v1",
            inputs={
                "x": x,
                "weight": weight,
                "query_start_loc": lengths,
                "prefill_metadata.cu_seqlens_cpu": host_lengths,
            },
            outputs={"return": x},
        )
        expected_flops, engine, peak = 128, "fp32_simt", 67000000000000
        boundary = {"inputs": {"x": {"shape": x["shape"]}}}
    op["scalar_args"]["profile_source_sha256"] = digest
    op["scalar_args"]["profile_config_sha256"] = sidecar["eager_step"][
        "profile_config_sha256"
    ]
    case["sidecar"].write_text(json.dumps(sidecar))
    case["eager"]["eager_sidecar"] = dict(
        **file_source(case["sidecar"]), bytes=case["sidecar"].stat().st_size
    )
    case["replays"].write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in (case["row"], case["binding"], case["eager"])
        )
    )
    window = json.loads(case["window"].read_text())
    window["loaded_library_manifest"] = {
        "schema": "tokenspeed.loaded_libraries/v1",
        "process_id": case["row"]["process_id"],
        "libraries": [],
    }
    case["window"].write_text(json.dumps(window))
    with sqlite3.connect(case["database"]) as db:
        db.execute("CREATE TABLE TARGET_INFO_GPU(id,uuid,computeMajor,computeMinor)")
        db.execute("INSERT INTO TARGET_INFO_GPU VALUES(0,?,9,0)", (uuid,))
        db.execute("INSERT INTO StringIds VALUES(4,'fixture_eager_hopper')")
        values = {
            "deviceId": 0,
            "contextId": 9,
            "streamId": 3,
            "mangledName": 4,
            "gridX": 1,
            "gridY": 2,
            "gridZ": 3,
            "blockX": 32,
            "blockY": 1,
            "blockZ": 1,
            "clusterX": 0,
            "clusterY": 0,
            "clusterZ": 0,
            "registersPerThread": 32,
            "staticSharedMemory": 0,
            "dynamicSharedMemory": 512,
        }
        for field, value in values.items():
            db.execute(
                f"ALTER TABLE CUPTI_ACTIVITY_KIND_KERNEL ADD COLUMN {field} INTEGER"
            )
            db.execute(
                f"UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET {field}=? WHERE correlationId IN (6,7)",
                (value,),
            )
        if case_name == "wrong_role":
            db.execute(
                "UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET gridX=4 WHERE correlationId=6"
            )
        if case_name == "missing_context":
            db.execute(
                "UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET contextId=NULL WHERE correlationId=6"
            )
    native = dict(
        name="fixture_eager_hopper",
        grid=[1, 2, 3],
        block=[32, 1, 1],
        cluster=[0, 0, 0],
        registers_per_thread=32,
        static_shared_memory=0,
        dynamic_shared_memory=512,
    )
    reviewed = {
        "contract_id": "test.eager.live_mla",
        "op_name": op["op_name"],
        "implementation": op["implementation"],
        "source_sha256s": [digest],
        "compute_capability": "9.0",
        "library_prefixes": [],
        "library_sha256s": [],
        "cases": [
            {
                "boundary": boundary,
                "native_launches": [native],
            }
        ],
        "components": [{"numerator": 1, "denominator": 1, "compute_type": engine}],
        "engine_evidence_level": "synthetic_fixture",
        "module_identity": "synthetic_fixture",
    }
    if case_name != "graph_only":
        reviewed["attribution_modes"] = ["eager_nvtx_gpu_projection"]
    definitions = json.dumps(
        {
            "schema": "tokenspeed.reviewed_engine_guards/v1",
            "bindings": [reviewed],
            "semantic_sources": [],
        }
    ).encode()
    registry = default_cost_registry(hbm_policy="logical_io_once")
    register_selected_engine_resolver(
        registry,
        definitions=definitions,
        expected_sha256=hashlib.sha256(definitions).hexdigest(),
    )
    result = analyze_mixed(case, registry)
    row = next(
        item
        for item in result["device_analysis"]["scopes"]
        if item["trace_label"] == "op"
    )
    assert row["cost_provenance"] == "trusted_offline_registry"
    if case_name != "conv_wrong_gpu_device":
        assert row["useful_flops"] == expected_flops
        assert row["estimated_mbu"] > 0
    assert row["gpu_duration_ns"] == 15
    (native_row,) = row["native_op_activity_bindings"]
    assert native_row["source_rowid"] == 5
    assert native_row["runtime_rowid"] == 2
    assert native_row["thread_id"] == 12345
    assert native_row["kernel"]["name"] == "fixture_eager_hopper"
    assert "source" not in native_row and "node" not in native_row
    proof = row["engine_proof"]
    assert proof["eager_sidecar"] == case["eager"]["eager_sidecar"]
    assert proof["eager_step_identity"]["forward_id"] == 901
    assert proof["source_report_sha256"] == file_source(case["report"])["sha256"]
    if case_name in ("exact", "conv_host_control"):
        assert proof["status"] == "bound"
        assert row["mfu"] == pytest.approx(expected_flops / (peak * 15e-9))
    else:
        assert proof["status"] == "unavailable"
        assert row["mfu"] is None
        assert {
            "graph_only": "no reviewed native engine contract",
            "wrong_role": "complete native launch/resource role set differs",
            "missing_context": "CUDA context",
            "conv_wrong_source": "captured tensor device unproven",
            "conv_wrong_host_dtype": "captured tensor device unproven",
            "conv_wrong_gpu_device": "captured tensor device unproven",
        }[case_name] in proof["unavailable_reason"]
