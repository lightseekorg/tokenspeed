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

"""Exact eager scope evidence for a serving window with CUDA graph replay."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path

from tokenspeed_kernel.profile_costs import _eager_host_control_input
from tokenspeed_kernel.profile_native_activity import (
    activity_resource_columns,
    decode_activity_resources,
)
from tokenspeed_kernel.profile_nvtx import PROFILE_METADATA_COPY_LABEL

STEP_FIELDS = (
    "run_id",
    "rank",
    "step_id",
    "serving_step",
    "request_ids",
    "request_pool_indices",
)

GPU_TABLES = {
    "CUPTI_ACTIVITY_KIND_KERNEL": "kernel",
    "CUPTI_ACTIVITY_KIND_MEMCPY": "memcpy",
    "CUPTI_ACTIVITY_KIND_MEMSET": "memset",
    "CUPTI_ACTIVITY_KIND_MEM_DECOMPRESS": "mem_decompress",
}

STEP_SUPPORT_OPS = {"sampling.broadcast_verify", "sampling.apply_stop_conditions"}
CUPTI_MEMCPY_DTOH = 2


def merge_forward_binding(row: dict, binding: dict) -> None:
    """Resolve a late ForwardId while preserving the ready step's identity."""
    forward_id = binding.get("forward_id")
    if (
        type(forward_id) is not int
        or forward_id < 0
        or row.get("forward_id") not in (None, forward_id)
        or any(
            row.get(key) != binding.get(key)
            for key in STEP_FIELDS
            if key in binding or row.get("eager_sidecar") is not None
        )
    ):
        raise ValueError("Late ForwardId binding mismatch")
    row["forward_id"] = forward_id


def read_eager_sidecar(row: dict, *, config_sha256: str) -> dict:
    """Validate the sidecar against the exact device-ready collector record."""
    source = row["eager_sidecar"]
    if (
        row.get("event") != "device_results_ready"
        or row.get("execution_mode") != "eager"
        or any(
            row.get(key) is not None
            for key in (
                "capture_id",
                "capture_file",
                "binary_payload",
                "metadata_lease",
                "replay_thread_id",
                "replay_stream",
                "unavailable_reason",
            )
        )
        or type(row.get("serving_step")) is not int
        or row["serving_step"] < 0
        or type(row.get("rank")) is not int
        or row["step_id"] != f"{row['run_id']}:r{row['rank']}:s{row['serving_step']}"
        or not isinstance(row.get("request_ids"), list)
        or not row["request_ids"]
        or any(not isinstance(rid, str) or not rid for rid in row["request_ids"])
        or not isinstance(row.get("request_pool_indices"), list)
        or len(row["request_ids"]) != len(row["request_pool_indices"])
        or any(
            type(index) is not int or index < 0 for index in row["request_pool_indices"]
        )
        or (
            row.get("forward_id") is not None
            and (type(row["forward_id"]) is not int or row["forward_id"] < 0)
        )
        or not isinstance(config_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", config_sha256) is None
    ):
        raise ValueError("Invalid eager device-ready ownership")
    path = Path(source["path"]).resolve(strict=True)
    raw = path.read_bytes()
    if (
        str(path) != source["path"]
        or len(raw) != source["bytes"]
        or hashlib.sha256(raw).hexdigest() != source["sha256"]
    ):
        raise ValueError("Eager sidecar hash or size mismatch")
    report = json.loads(raw)
    identity = report["eager_step"]
    analysis = report["device_analysis"]
    scopes = analysis["scopes"]
    records = report["layers"] + report["ops"]
    if (
        report.get("schema_version") != 1
        or report.get("run_id") != f"{row['run_id']}:eager:{row['serving_step']}"
        or report.get("rank") != row["rank"]
        or identity.get("schema") != "tokenspeed.profile_eager_step/v1"
        or any(identity.get(key) != row.get(key) for key in STEP_FIELDS)
        or identity.get("profile_config_sha256") != config_sha256
        or identity.get("forward_id") not in (None, row.get("forward_id"))
        or analysis.get("timing_kind") != "pending_nsight"
        or analysis.get("nvtx_payload_type") != "int64"
        or type(analysis.get("process_id")) is not int
        or analysis["process_id"] <= 0
        or analysis["process_id"] != row.get("process_id")
        or type(analysis.get("thread_id")) is not int
        or analysis["thread_id"] <= 0
        or not scopes
        or len(scopes) != len(records)
        or {scope["trace_label"] for scope in scopes}
        != {record["trace_label"] for record in records}
        or any(record["iter_id"] != row["serving_step"] for record in records)
        or any(
            type(scope.get("nvtx_id")) is not int or not 0 <= scope["nvtx_id"] < 1 << 63
            for scope in scopes
        )
        or len({scope["nvtx_id"] for scope in scopes}) != len(scopes)
    ):
        raise ValueError("Eager sidecar run/rank/step/config/payload identity mismatch")
    observer_scopes = analysis.get("metadata_copy_scopes", [])
    if not isinstance(observer_scopes, list) or any(
        not isinstance(scope, dict)
        or scope.get("nvtx_label") != PROFILE_METADATA_COPY_LABEL
        or type(scope.get("nvtx_id")) is not int
        or not 0 <= scope["nvtx_id"] < 1 << 63
        or scope.get("trace_label") != f"profiler.metadata_copy/{scope['nvtx_id']}"
        or type(scope.get("bytes")) is not int
        or scope["bytes"] < 1
        for scope in observer_scopes
    ):
        raise ValueError("Invalid eager integer metadata copy scopes")
    if observer_scopes and (
        len({scope["nvtx_id"] for scope in scopes + observer_scopes})
        != len(scopes) + len(observer_scopes)
        or type(analysis.get("metadata_snapshot_bytes")) is not int
        or sum(scope["bytes"] for scope in observer_scopes)
        > analysis["metadata_snapshot_bytes"]
    ):
        raise ValueError("Eager metadata copy scope identity or size mismatch")
    return report


def _activity_rows(db, ranges, global_tid):
    # Match the Nsight GPU projection contract for enclosed same-thread Runtime
    # calls. Match globalPid with the entire packed process key.
    start = min(item["cpu_start_ns"] for item in ranges)
    end = max(item["cpu_end_ns"] for item in ranges)
    launches = {}
    db.execute("CREATE TEMP TABLE eager_correlations(id INTEGER PRIMARY KEY)")
    for row in db.execute(
        """
        SELECT r.correlationId, r.start, r.end, s.value AS name, r.rowid AS runtime_rowid
        FROM CUPTI_ACTIVITY_KIND_RUNTIME r LEFT JOIN StringIds s ON r.nameId=s.id
        WHERE r.globalTid=? AND r.start>=? AND r.end<=? AND r.callchainId IS NULL
          AND r.correlationId IS NOT NULL
        """,
        (global_tid, start, end),
    ):
        launches.setdefault(row["correlationId"], []).append(tuple(row)[1:])
    db.executemany(
        "INSERT INTO eager_correlations VALUES (?)", ((k,) for k in launches)
    )
    process_key = global_tid >> 24
    activities = []
    strings = dict(db.execute("SELECT id,value FROM StringIds"))
    for table, kind in GPU_TABLES.items():
        columns = {row[1] for row in db.execute(f"PRAGMA table_info({table})")}
        if not columns:
            continue
        required = {"start", "end", "globalPid", "correlationId"}
        if not required <= columns:
            raise ValueError(f"Eager GPU projection table lacks fields: {table}")
        resources = activity_resource_columns(table)
        extra = ",".join(
            f"a.{name}" if name in columns else f"NULL AS {name}"
            for name in ("graphNodeId", *resources)
        )
        for item in db.execute(
            f"""
            SELECT a.rowid AS source_rowid, a.start, a.end, a.correlationId, {extra}
            FROM {table} a JOIN eager_correlations c ON a.correlationId=c.id
            WHERE a.globalPid=?
            """,
            (process_key << 24,),
        ):
            correlation = item["correlationId"]
            entries = launches[correlation]
            if len(entries) != 1:
                raise ValueError("Ambiguous eager runtime correlation")
            cpu_start, cpu_end, name, runtime_rowid = entries[0]
            if (
                type(item["start"]) is not int
                or type(item["end"]) is not int
                or item["end"] <= item["start"]
                or item["graphNodeId"] not in (None, 0)
                or not isinstance(name, str)
                or name.startswith("cudaGraphLaunch")
            ):
                raise ValueError("Invalid or graph-owned eager GPU activity")
            kernel, memory = decode_activity_resources(
                table, resources, (item[key] for key in resources), strings
            )
            resource = kernel if kernel is not None else memory
            activities.append(
                dict(
                    start=item["start"],
                    end=item["end"],
                    correlationId=correlation,
                    source_rowid=item["source_rowid"],
                    table=table,
                    kind=kind,
                    process_key=process_key,
                    process_id=(global_tid >> 24) & 0xFFFFFF,
                    thread_id=global_tid & 0xFFFFFF,
                    runtime_rowid=runtime_rowid,
                    runtime_api=name,
                    cpu_start=cpu_start,
                    cpu_end=cpu_end,
                    deviceId=resource["device_id"],
                    contextId=resource["context_id"],
                    streamId=resource["stream_id"],
                    kernel=kernel,
                    memory=memory,
                )
            )
    return activities


def _engine_context(
    row,
    scope,
    op,
    selected,
    owners,
    *,
    expected_uuid,
    device_uuids,
    selected_hardware,
    loaded_library_manifest,
    config_sha256,
    source_report_sha256,
):
    scope_id = scope["scope_id"]
    owned = [
        item
        for item in selected
        if owners[(item["table"], item["source_rowid"])] == scope_id
    ]
    capture = row["eager_step_identity"]
    devices = {item["deviceId"] for item in owned}
    contexts = {item["contextId"] for item in owned}
    tensor_devices = {
        tensor["device"]
        for name, tensor in op["inputs"].items()
        if not _eager_host_control_input(
            op["op_name"],
            op["implementation"],
            op["scalar_args"].get("profile_source_sha256"),
            name,
            device=tensor["device"],
            dtype=tensor["dtype"],
            shape=tensor["shape"],
            stride=tensor["stride"],
            storage_offset=tensor.get("storage_offset"),
            logical_bytes=tensor["logical_bytes"],
        )
    } | {tensor["device"] for tensor in op["outputs"].values()}
    reason = None
    if not owned or not any(item["kernel"] is not None for item in owned):
        reason = "eager Op has no exactly owned native kernel"
    elif len(owned) != len(selected):
        reason = "eager Op projection includes descendant Op activity"
    elif (
        len(devices) != 1
        or None in devices
        or len(contexts) != 1
        or None in contexts
        or not expected_uuid
        or device_uuids.get(next(iter(devices)), {}).get("uuid") != expected_uuid
        or tensor_devices != {capture["device"]}
    ):
        reason = "eager Op device, CUDA context or captured tensor device unproven"
    elif any(
        item["kernel"] is not None
        and any(
            item["kernel"][field] is None
            or isinstance(item["kernel"][field], list)
            and None in item["kernel"][field]
            for field in (
                "name",
                "grid",
                "block",
                "cluster",
                "registers_per_thread",
                "static_shared_memory",
                "dynamic_shared_memory",
            )
        )
        for item in owned
    ):
        reason = "eager kernel symbol or native launch resource metadata unavailable"
    elif type(capture.get("forward_id")) is not int:
        reason = "eager Op lacks an exact serving ForwardId"
    elif (
        re.fullmatch(
            r"[0-9a-f]{64}", op["scalar_args"].get("profile_source_sha256") or ""
        )
        is None
    ):
        reason = "eager Op implementation source SHA unavailable"
    elif op["scalar_args"].get("profile_config_sha256", config_sha256) != config_sha256:
        reason = "eager Op configuration provenance mismatch"
    return dict(
        attribution_mode="eager_nvtx_gpu_projection",
        complete_join=reason is None,
        incomplete_join_reason=reason,
        run_id=row["run_id"],
        rank=row["rank"],
        eager_step_identity=capture,
        eager_sidecar=row["eager_sidecar"],
        source_report_sha256=source_report_sha256,
        scope_identity={
            "scope_id": scope_id,
            "trace_label": scope["trace_label"],
            "nsight_range_id": scope["nsight_range_id"],
            "nvtx_domain": "TokenSpeed",
            "nvtx_payload_type": "int64",
            "cpu_start_ns": scope["cpu_start_ns"],
            "cpu_end_ns": scope["cpu_end_ns"],
        },
        activity_bindings=owned,
        hardware={
            "device": selected_hardware.device,
            "selection": selected_hardware.selection,
        },
        native_device_uuids=device_uuids,
        profile_config_sha256=config_sha256,
        startup_observation_sha256=op["scalar_args"].get("startup_observation_sha256"),
        loaded_library_manifest=loaded_library_manifest,
    )


def _activity_key(item):
    return item["table"], item["source_rowid"]


def _gpu_projection(activities):
    return (
        min((item["start"] for item in activities), default=None),
        max((item["end"] for item in activities), default=None),
        len({item["correlationId"] for item in activities}),
    )


def _projected_bounds(scope):
    return scope["gpu_start_ns"], scope["gpu_end_ns"], scope["gpu_activity_count"]


def analyze_eager_step(
    row,
    *,
    config_sha256,
    database,
    projection,
    hardware,
    device_uuids,
    registry,
    loaded_library_manifest,
    source_report_sha256,
    sealed_communication_artifacts=None,
):
    """Bind complete CPU scope nesting and official projection to native row IDs."""
    from tokenspeed_kernel.profile_communication import bind_capture_communication
    from tokenspeed_kernel.profile_nsys import _join_nsight_projection
    from tokenspeed_kernel.profile_utilization import (
        analyze_utilization,
        hardware_for_analysis,
    )

    report = read_eager_sidecar(row, config_sha256=config_sha256)
    capture = report["device_analysis"]
    if capture["process_id"] >= 1 << 24 or capture["thread_id"] >= 1 << 24:
        raise ValueError("Eager process/thread exceeds Nsight's packed identity width")
    selected_hardware = hardware_for_analysis(report["hardware"], hardware)
    report["hardware"] = asdict(selected_hardware)
    observer_scopes = capture.get("metadata_copy_scopes", [])
    timings, evidence = _join_nsight_projection(
        {
            **report,
            "device_analysis": {
                **capture,
                "scopes": capture["scopes"] + observer_scopes,
            },
        },
        database=database,
        projection=projection,
    )
    observers = [
        {**scope, **evidence.pop(scope["trace_label"])} for scope in observer_scopes
    ]
    timings = {
        timing.trace_label: timing
        for timing in timings
        if timing.trace_label in evidence
    }
    by_label = {item["trace_label"]: item["nvtx_id"] for item in capture["scopes"]}
    rows = [
        {
            "kind": kind,
            **{key: record[key] for key in ("trace_label", "layer_id", "iter_id")},
            **evidence[record["trace_label"]],
        }
        for kind, records in (("op", report["ops"]), ("layer", report["layers"]))
        for record in records
    ]
    with sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        # The shared projection join has already required closed signed
        # TokenSpeed push/pop ranges. Retain their full packed thread key.
        tids = {
            db.execute(
                "SELECT globalTid FROM NVTX_EVENTS WHERE rowid=?",
                (scope["nsight_range_id"],),
            ).fetchone()[0]
            for scope in rows + observers
        }
        if len(tids) != 1:
            raise ValueError("Eager scopes span multiple native threads")
        global_tid = tids.pop()
        graph_marker = f"tokenspeed.profile.replay | {row['step_id']}"
        if (
            db.execute(
                """
            SELECT 1 FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id
            WHERE (n.globalTid >> 24)=? AND COALESCE(s.value,n.text)=? LIMIT 1
            """,
                (global_tid >> 24, graph_marker),
            ).fetchone()
            is not None
        ):
            raise ValueError("Eager step also has a native Graph replay marker")
        activities = _activity_rows(db, rows + observers, global_tid)

    expected_uuid = (
        (report["hardware"].get("selection") or {}).get("observed", {}).get("uuid")
    )
    if expected_uuid:
        expected_uuid = expected_uuid.lower().removeprefix("gpu-").replace("-", "")
        if any(
            device_uuids.get(item["deviceId"], {}).get("uuid") != expected_uuid
            for item in activities
        ):
            raise ValueError(
                "Eager GPU activity device UUID differs from hardware binding"
            )

    ordered = sorted(
        rows,
        key=lambda item: (
            item["cpu_start_ns"],
            -item["cpu_end_ns"],
            by_label[item["trace_label"]],
        ),
    )
    stack, models, attributed, layers, kinds, unknowns = [], [], [], set(), set(), []
    assignments = {}
    op_activities = {}
    sources = {}
    support = {}
    metadata_copy_activities = {}
    for observer in observers:
        selected = [
            item
            for item in activities
            if observer["cpu_start_ns"] <= item["cpu_start"]
            and item["cpu_end"] <= observer["cpu_end_ns"]
        ]
        if len(selected) != 1 or _gpu_projection(selected) != _projected_bounds(
            observer
        ):
            raise ValueError(
                "Eager observer scope does not own exactly one projected copy"
            )
        (activity,) = selected
        key = _activity_key(activity)
        if (
            key in metadata_copy_activities
            or activity["kind"] != "memcpy"
            or activity["memory"]["copy_kind"] != CUPTI_MEMCPY_DTOH
            or activity["memory"]["bytes"] != observer["bytes"]
        ):
            raise ValueError(
                "Eager observer GPU activity differs from its metadata copy"
            )
        metadata_copy_activities[key] = {
            **activity,
            "nvtx_id": observer["nvtx_id"],
            "nsight_range_id": observer["nsight_range_id"],
        }
    op_records = {op["trace_label"]: op for op in report["ops"]}
    for scope in ordered:
        start, end = scope["cpu_start_ns"], scope["cpu_end_ns"]
        scope_id = by_label[scope["trace_label"]]
        while stack and stack[-1]["cpu_end_ns"] <= start:
            stack.pop()
        if stack and end > stack[-1]["cpu_end_ns"]:
            raise ValueError("Crossing eager NVTX scope boundaries")
        parent = stack[-1]["scope_id"] if stack else None
        kind = scope["kind"]
        is_model = kind == "layer" and parent is None and scope["layer_id"] is None
        selected = [
            item
            for item in activities
            if start <= item["cpu_start"] and item["cpu_end"] <= end
        ]
        if _gpu_projection(selected) != _projected_bounds(scope):
            raise ValueError(
                "Eager native activity rows disagree with the official projection"
            )
        selected = [
            item
            for item in selected
            if _activity_key(item) not in metadata_copy_activities
        ]
        gpu_start, gpu_end, gpu_count = _gpu_projection(selected)
        scope.update(
            gpu_start_ns=gpu_start, gpu_end_ns=gpu_end, gpu_activity_count=gpu_count
        )
        label = scope["trace_label"]
        timing = timings[label]
        timings[label] = replace(
            timing,
            gpu_duration_ns=0 if gpu_start is None else gpu_end - gpu_start,
            unavailable_reason=(
                timing.unavailable_reason
                if selected or timing.unavailable_reason
                else "no model activity remains after excluding profiler metadata copies"
            ),
        )
        identity = [_activity_key(item) for item in selected]
        scope_reasons = []
        if not selected:
            scope_reasons.append("eager_scope_native_activity_unavailable")
        else:
            attributed.append(scope_id)
            kinds.add("model" if is_model else kind)
            if kind == "layer" and type(scope["layer_id"]) is int:
                layers.add(scope["layer_id"])
        if kind == "op":
            op_activities[label] = selected
            ancestors = {item["scope_id"] for item in stack}
            for key in identity:
                previous = assignments.get(key)
                if previous is not None and previous not in ancestors:
                    raise ValueError("Duplicate eager Op GPU activity ownership")
                assignments[key] = scope_id
            if parent is None and op_records[label]["op_name"] in STEP_SUPPORT_OPS:
                support.update(zip(identity, selected, strict=True))
        if is_model:
            models.append(
                {
                    key: scope[key]
                    for key in ("nsight_range_id", "cpu_start_ns", "cpu_end_ns")
                }
            )
            for item, key in zip(selected, identity, strict=True):
                if key in sources:
                    raise ValueError("Duplicate eager Model GPU activity ownership")
                sources[key] = item
        scope.update(
            scope_id=scope_id,
            parent_scope_id=parent,
            step_id=row["step_id"],
            forward_id=row.get("forward_id"),
            rank=row["rank"],
            serving_step=row["serving_step"],
            capture_id=None,
            quantity_scope="per_rank",
            attribution_mode="eager_nvtx_gpu_projection",
            native_activity_rows=[
                {"table": table, "rowid": rid} for table, rid in identity
            ],
            critical_unknowns=scope_reasons,
        )
        unknowns.extend(f"scope={scope_id}:{reason}" for reason in scope_reasons)
        stack.append(scope)
    if capture.get("unavailable_integer_metadata"):
        unknowns.append("eager_integer_metadata_unavailable")
    if not models or not sources:
        unknowns.append("eager_model_native_activity_unavailable")
    if assignments.keys() - sources.keys() - support.keys():
        unknowns.append("eager_op_activity_outside_model")
    unowned = sources.keys() - assignments.keys()
    if "metadata_copy_scopes" in capture and unowned:
        unknowns.append("eager_model_activity_outside_costed_ops")
        for scope in ordered:
            if any(
                (item["table"], item["rowid"]) in unowned
                for item in scope["native_activity_rows"]
            ):
                label = scope["trace_label"]
                timings[label] = replace(
                    timings[label],
                    unavailable_reason="eager_model_activity_outside_costed_ops",
                )
    comm = bind_capture_communication(
        {**report, "run_id": row["run_id"], "process_id": capture["process_id"]},
        Path(row["eager_sidecar"]["path"]).parent,
        sealed_artifacts=sealed_communication_artifacts,
        native_op_activities=op_activities,
    )
    for label, binding in comm.items():
        op_records[label]["scalar_args"].update(binding)
    step_identity = {
        **{key: row[key] for key in STEP_FIELDS},
        "forward_id": row.get("forward_id"),
        **{
            key: capture[key]
            for key in ("process_id", "thread_id", "device", "stream")
            if key in capture
        },
    }
    engine_context = {
        scope["trace_label"]: _engine_context(
            {**row, "eager_step_identity": step_identity},
            scope,
            op_records[scope["trace_label"]],
            op_activities[scope["trace_label"]],
            assignments,
            expected_uuid=expected_uuid,
            device_uuids=device_uuids,
            selected_hardware=selected_hardware,
            loaded_library_manifest=loaded_library_manifest,
            config_sha256=config_sha256,
            source_report_sha256=source_report_sha256,
        )
        for scope in ordered
        if scope["kind"] == "op"
    }
    runtime_costs = report["analysis"]["ops"]
    report["analysis"] = registry.analyze_recorded(
        report, engine_context=engine_context
    )
    costs = {}
    for op, runtime, offline in zip(
        report["ops"], runtime_costs, report["analysis"]["ops"], strict=True
    ):
        if any(
            op.get(key) != runtime.get(key)
            for key in ("iter_id", "layer_id", "inner_id")
        ):
            raise ValueError("Eager runtime cost and OpRecord identities differ")
        offline["cost_provenance"] = "trusted_offline_registry"
        if (
            offline["missing_reason"] == "cost rule unavailable"
            and runtime.get("cost") is not None
        ):
            cost = runtime["cost"]
            # A runtime-only rule can retain accounting. Its engine claim has
            # no reviewed offline binding.
            if cost["useful_flops"] > 0:
                cost.update(
                    compute_type=None,
                    compute_components=[
                        {
                            "useful_flops": cost["useful_flops"],
                            "compute_type": None,
                            "missing_reason": "eager runtime cost has no trusted offline engine rule",
                        }
                    ],
                )
            offline.update(
                cost=cost, missing_reason=None, cost_provenance="recorded_runtime_rule"
            )
        costs[op["trace_label"]] = offline

    metrics = {
        item["trace_label"]: item
        for item in analyze_utilization(
            report, list(timings.values()), selected_hardware
        )
    }
    categories = {}
    for scope in ordered:
        label = scope["trace_label"]
        scope.update(metrics[label])
        reasons = scope["critical_unknowns"]
        if scope["mfu"] is None and scope["mfu_missing_reason"]:
            reasons.append(scope["mfu_missing_reason"])
        if scope["estimated_mbu"] is None and scope["mbu_missing_reason"]:
            reasons.append(scope["mbu_missing_reason"])
        unknowns.extend(f"scope={scope['scope_id']}:{reason}" for reason in reasons)
        if scope["kind"] == "op":
            scope.update(
                engine_proof=costs[label]["engine_proof"],
                cost_provenance=costs[label]["cost_provenance"],
                native_op_activity_bindings=engine_context[label]["activity_bindings"],
            )
            traffic, flops = scope.get("collective_count", 0), scope.get("useful_flops")
            categories[scope["scope_id"]] = (
                "fused_compute_communication"
                if traffic and flops
                else (
                    "pure_communication"
                    if traffic
                    else "compute" if flops else "unclassified"
                )
            )
    groups = defaultdict(list)
    step_activities = {**sources, **support}
    for key, item in step_activities.items():
        groups[categories.get(assignments.get(key), "unclassified")].append(
            (item["start"], item["end"])
        )
    return (
        {
            **row,
            "attribution_mode": "eager_nvtx_gpu_projection",
            "graph_node_ids": [],
            "scope_kinds": sorted(kinds),
            "expected_scope_ids": sorted(by_label.values()),
            "attributed_scope_ids": sorted(attributed),
            "attributed_layer_ids": sorted(layers),
            "critical_unknowns": sorted(set(unknowns)),
            "hardware": report["hardware"],
        },
        ordered,
        {
            "global_tid": global_tid,
            "models": models,
            "activities": [
                {
                    key: value
                    for key, value in item.items()
                    if key not in ("kernel", "memory")
                }
                for item in step_activities.values()
            ],
            "step_support_activity_rows": [
                {"table": table, "rowid": rowid} for table, rowid in support
            ],
            "metadata_copy_activities": list(metadata_copy_activities.values()),
            "semantic_groups": dict(groups),
        },
    )
