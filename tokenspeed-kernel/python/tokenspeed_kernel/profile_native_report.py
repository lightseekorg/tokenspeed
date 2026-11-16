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

"""Append utilization labels to a native Nsight report using its offline writer.

Uses NVIDIA's ``ImportNvtxt`` utility from the matching Nsight installation.
The input is the Nsight-timed JSON produced by ``profile_nsys`` for this capture.
Ranges occupy the correlated GPU spans, including their gaps. They are derived
annotations, not new kernel measurements or replacements for the CPU ranges.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sqlite3
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path


def _scope_message(row: dict) -> str:
    mfu, mfu_reason = _mfu_display(row)
    mbu = (
        "N/A" if row["estimated_mbu"] is None else f"{row['estimated_mbu'] * 100:.3g}%"
    )
    if row["estimated_mbu"] is None and row.get("estimated_mbu_known") is not None:
        mbu = f"{row['estimated_mbu_known'] * 100:.3g}% (partial)"
    collective_only = (
        row.get("collective_count", 0) > 0 and row.get("useful_flops") == 0
    )
    message = f"{row['native_nvtx_label']} | "
    communication = row.get("kind") == "op" and (
        row.get("collective_count") or row.get("comm_unavailable_reason")
    )
    if communication:
        utilization = row.get("estimated_comm_utilization")
        display = "N/A" if utilization is None else f"{100 * utilization:.3g}%"
        message += f"Est.CommBWUtil={display}"
        rates = (
            ("algbw_bytes_per_second", "commAlgBW"),
            ("busbw_bytes_per_second", "commBusBW"),
        )
        directional_rates = (
            ("estimated_comm_tx_payload_bytes_per_second", "Est.TXPayloadBW"),
            ("estimated_comm_rx_payload_bytes_per_second", "Est.RemoteRXPayloadBW"),
        )
        if row.get("communication_payload_model") or any(
            row.get(field) is not None for field, _ in directional_rates
        ):
            rates += directional_rates
        for field, title in rates:
            value = row.get(field)
            rate = "N/A" if value is None else f"{value / 1e9:.3g}GB/s"
            message += f" | {title}={rate}"
        message += f" | GPU={row['gpu_duration_ns'] / 1000:.5g}us | "
    if not (row.get("kind") == "op" and collective_only):
        message += f"MFU_total={mfu} "
    message += f"Est.MBU(HBM)={mbu}"
    message += f" | iter={row['iter_id']} layer={row['layer_id']}"
    if row.get("step_id") is not None:
        message += f" step={row['step_id']} ForwardId={row.get('forward_id')}"
    if row.get("scheduler_status") is not None:
        status = row["scheduler_status"]
        message += f" status={status['completion_status']}"
        message += f" scheduled={status['scheduled']} GPUobserved={status['gpu_observed']} device_ready={status['device_ready']} authoritative_commit={status['authoritative_commit']}"
        for commit in status["token_commits"]:
            first, last = commit["accepted_token_interval"]
            message += f" | rid={commit['rid']} generation={commit['generation']} accepted_tokens=({first},{last}]"
    if row.get("kind") == "op" and row.get("collective_count"):
        efficiency = row.get("estimated_comm_reference_efficiency")
        display = "N/A" if efficiency is None else f"{efficiency / 0.01:.3g}%"
        message += f" CommRefEff={display}"
        model = row.get("communication_payload_model")
        model_name = model.get("id") if isinstance(model, dict) else model
        direction = row.get("communication_bottleneck_direction")
        message += " | comm rates denominator=whole Op GPU span"
        message += " | commAlgBW/commBusBW=NCCL-tests reference"
        message += f" | Est.CommBWUtil model={model_name or 'N/A'} bottleneck={direction or 'N/A'}"
        message += " max(TX/capacity,RX/capacity); excludes unmodeled wire overhead"
        message += " | CommRefEff=normalized bus BW/reference; not physical saturation"
        if row.get("busbw_missing_reason"):
            message += f" | comm N/A: {row['busbw_missing_reason']}"
        if row.get("estimated_comm_utilization_missing_reason"):
            message += (
                f" | comm util N/A: {row['estimated_comm_utilization_missing_reason']}"
            )
        elif row.get("estimated_comm_utilization") is None:
            message += (
                " | comm util N/A: communication normalization metadata unavailable"
            )
        if row.get("estimated_comm_utilization_warning"):
            message += f" | {row['estimated_comm_utilization_warning']}"
        if row.get("estimated_comm_reference_efficiency_warning"):
            message += f" | {row['estimated_comm_reference_efficiency_warning']}"
    if row.get("kind") == "op" and row.get("comm_unavailable_reason"):
        message += f" | Comm=N/A: {row['comm_unavailable_reason']}"

    for compute_type, value in row["mfu_by_compute_type"].items():
        if row["mfu"] is None:
            message += f" | MFU[{compute_type}]={value * 100:.3g}%"
    reasons = {
        reason
        for reason in (mfu_reason, row["mbu_missing_reason"])
        if reason is not None
    }
    if reasons:
        message += f" | N/A: {'; '.join(sorted(reasons))}"
    return message


def _display_groups(rows, default_rank):
    """Separate levels using recorded ancestry, never GPU-time containment."""

    def key(row):
        return row.get("rank", default_rank), row["iter_id"], row.get("scope_id")

    scopes = {key(row): row for row in rows if row.get("scope_id") is not None}

    def ancestors(row):
        rank, iteration, _ = key(row)
        seen = set()
        parent = row.get("parent_scope_id")
        while parent is not None:
            if parent in seen:
                raise ValueError("Cycle in recorded scope ancestry")
            seen.add(parent)
            scope = scopes.get((rank, iteration, parent))
            if scope is None:
                raise ValueError("Missing recorded parent scope")
            yield scope
            parent = scope.get("parent_scope_id")

    layer_ids = defaultdict(set)
    for row in rows:
        if row["kind"] == "layer" and row.get("scope_id") is not None:
            for scope in (row, *ancestors(row)):
                layer_ids[key(scope)].add(row["layer_id"])
    model_keys = {identity for identity, ids in layer_ids.items() if len(ids) > 1}
    model_keys.update(
        key(row)
        for row in rows
        if row["kind"] == "layer"
        and row.get("scope_id") is not None
        and row.get("parent_scope_id") is None
        and row.get("layer_id") is None
    )
    groups = defaultdict(list)
    for row in rows:
        rank = row.get("rank", default_rank)
        if row["kind"] == "op":
            level = (2, 0, "Op")
        elif key(row) in model_keys or rank == "all":
            level = (0, 0, "Model")
        else:
            depth = sum(
                parent["kind"] == "layer" and key(parent) not in model_keys
                for parent in ancestors(row)
            )
            label = "Layer" if depth == 0 else f"Module depth={depth}"
            level = (1, depth, label)
        groups[(str(rank), *level)].append(row)
    return groups


def _mfu_display(row):
    """Use the explicit ideal-time total, retaining older graph report fields."""
    if "mfu_total" in row:
        value, known = row["mfu_total"], row.get("mfu_total_known")
        reason = row.get("mfu_total_missing_reason")
    else:
        value, known = row["mfu"], row.get("mfu_known")
        reason = row["mfu_missing_reason"]
    if value is not None:
        return f"{value * 100:.3g}%", reason
    if known is not None:
        return f"{known * 100:.3g}% (partial)", reason
    return "N/A", reason


def fingerprint(db, table, fields):
    names = ",".join(f'"{field}"' for field in fields)
    sha, count = hashlib.sha256(), 0
    for row in db.execute(f'SELECT {names} FROM "{table}" ORDER BY {names}'):
        sha.update((repr(tuple(row)) + "\n").encode())
        count += 1
    return {"count": count, "sha256": sha.hexdigest(), "fields": fields}


def audit_nvtx_locators(raw, annotated):
    """Preserve full NVTX measurements while auditing export-local eventId.

    Groups use the complete nonlocator row, including all timestamps, thread,
    text IDs and payload fields. Duplicate measurements retain their complete
    locator sets; they are not arbitrarily paired into individual identities.
    """
    raw_schema = raw.execute('PRAGMA table_info("NVTX_EVENTS")').fetchall()
    after_schema = annotated.execute('PRAGMA table_info("NVTX_EVENTS")').fetchall()
    columns = [row[1] for row in raw_schema]
    if columns != [row[1] for row in after_schema]:
        raise ValueError("NVTX measurement schema changed")
    for schema in (raw_schema, after_schema):
        locators = [row for row in schema if row[1] == "eventId"]
        if (
            len(locators) != 1
            or locators[0][2].upper() != "INTEGER"
            or locators[0][5] != 1
        ):
            raise ValueError(
                "NVTX eventId must be the export-local integer primary key"
            )
    fields = [field for field in columns if field != "eventId"]
    names = ",".join(f'"{field}"' for field in fields)

    def groups(db):
        result = {}
        for row in db.execute(f'SELECT "eventId",{names} FROM "NVTX_EVENTS"'):
            # Keep the complete typed tuple representation, not a timestamp/name
            # key or a hash alone, as the equivalence-group key.
            measurement = repr(tuple(row[1:]))
            result.setdefault(measurement, []).append(row[0])
        for ids in result.values():
            ids.sort()
        return result

    before, after = groups(raw), groups(annotated)
    if {key: len(ids) for key, ids in before.items()} != {
        key: len(ids) for key, ids in after.items()
    }:
        raise ValueError("NVTX measurement multiset changed outside eventId")

    def referenced_strings(db):
        refs = set()
        for field in ("textId", "jsonTextId"):
            if field in fields:
                refs.update(
                    row[0]
                    for row in db.execute(
                        f'SELECT DISTINCT "{field}" FROM NVTX_EVENTS WHERE "{field}" IS NOT NULL'
                    )
                )
        if not refs:
            return {}
        values = {
            row[0]: row[1]
            for row in db.execute("SELECT id,value FROM StringIds")
            if row[0] in refs
        }
        if values.keys() != refs:
            raise ValueError("NVTX referenced StringIds are missing")
        return values

    strings = referenced_strings(raw)
    if strings != referenced_strings(annotated):
        raise ValueError("NVTX referenced StringIds mapping changed")
    entries = [
        {
            "complete_measurement_sha256": hashlib.sha256(key.encode()).hexdigest(),
            "measurement_count": len(before[key]),
            "source_event_ids": before[key],
            "export_event_ids": after[key],
        }
        for key in sorted(before)
    ]
    row_count = sum(len(ids) for ids in before.values())
    return {
        "policy": "Only NVTX_EVENTS.eventId is an export-local locator; every other column and referenced StringIds mapping must match exactly as a multiset",
        "measurement_fields": fields,
        "source_row_count": row_count,
        "export_row_count": sum(len(ids) for ids in after.values()),
        "measurement_group_count": len(entries),
        "duplicate_measurement_group_count": sum(
            len(ids) > 1 for ids in before.values()
        ),
        "changed_locator_group_count": sum(before[key] != after[key] for key in before),
        "source_unique_locator_count": len(
            {value for ids in before.values() for value in ids}
        ),
        "export_unique_locator_count": len(
            {value for ids in after.values() for value in ids}
        ),
        "referenced_string_count": len(strings),
        "referenced_strings_sha256": hashlib.sha256(
            repr(sorted(strings.items())).encode()
        ).hexdigest(),
        "measurement_groups": entries,
        "duplicate_identity_policy": "Complete equivalent-row locator sets only; no individual correspondence claimed for duplicate measurements",
    }


def annotate_report(
    *,
    report: Path,
    metrics: Path,
    output: Path,
    importer: Path,
    target_id: str,
    nsys: str,
    source_database: Path | None = None,
):
    """Copy a capture and append native NVTX ranges with estimated utilization.

    All paths must be explicit. ``output`` must be a new ``.nsys-rep`` path;
    ``importer`` is the matching Nsight installation's ImportNvtxt executable.
    GPU timestamps are Nsight nanoseconds relative to the analysis start, as
    exported by ``profile_nsys``. Missing metrics stay N/A with their reasons.
    Optional ``source_database`` reuses the original SQLite export only when its
    SHA256 and the current report SHA256 match the metrics provenance. It is
    opened read-only; the annotated report is still exported and fully audited.
    """
    if re.fullmatch(r"[0-9]+:[0-9]+", target_id) is None:
        raise ValueError("Import target must be the explicit Hw:Vm ID from report info")
    report, metrics, output, importer = (
        path.expanduser().resolve() for path in (report, metrics, output, importer)
    )
    if output.suffix != ".nsys-rep" or output.exists():
        raise ValueError("Output must be a new .nsys-rep file")
    if output in (report, metrics):
        raise ValueError("Output must not overwrite capture or metrics")
    if not report.is_file() or not importer.is_file():
        raise ValueError("Capture and ImportNvtxt executable must exist")
    data = json.loads(metrics.read_text())
    analysis = data["device_analysis"]
    if analysis["timing_kind"] not in {
        "nsight_gpu_projection",
        "nsight_graph_node_ids",
        "nsight_mixed_graph_eager",
    }:
        raise ValueError("Native annotation requires completed Nsight GPU timing")
    rows = [
        row
        for row in analysis["scopes"]
        if row.get("gpu_start_ns") is not None and row.get("gpu_end_ns") is not None
    ]
    omitted_scopes = len(analysis["scopes"]) - len(rows)
    if not rows:
        raise ValueError("No analyzed scopes")
    for row in rows:
        start, end = row["gpu_start_ns"], row["gpu_end_ns"]
        if start is None or end is None or end <= start:
            raise ValueError("Every annotated scope must have correlated GPU work")
        if end - start != row["gpu_duration_ns"]:
            raise ValueError("Scope duration does not match its GPU boundaries")

    def fingerprint(path):
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    original_hash = fingerprint(report)
    if analysis.get("source_report_sha256") != original_hash:
        raise ValueError("Metrics were calculated from a different capture")
    source_database_hash = None
    if source_database is not None:
        source_database = source_database.expanduser().resolve(strict=True)
        if not source_database.is_file():
            raise ValueError("Source database must be an existing SQLite file")
        provenance = data.get("provenance", {})
        if provenance.get("report", {}).get("sha256") != original_hash:
            raise ValueError(
                "Source database reuse requires matching report provenance"
            )
        source_database_hash = fingerprint_file(source_database)
        if provenance.get("database", {}).get("sha256") != source_database_hash:
            raise ValueError("Source database SHA256 differs from metrics provenance")
    # Inspect original measurements and existing derived lanes without writing them.
    with tempfile.TemporaryDirectory(prefix="tokenspeed-native-source-") as directory:
        collision_database = source_database
        if collision_database is None:
            collision_database = Path(directory) / "source.sqlite"
            subprocess.run(
                [
                    nsys,
                    "export",
                    "--type=sqlite",
                    f"--output={collision_database}",
                    str(report),
                ],
                check=True,
            )
        with sqlite3.connect(
            f"{collision_database.resolve().as_uri()}?mode=ro", uri=True
        ) as db:
            occupied = set()
            tables = {
                row[0]
                for row in db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            for table in tables:
                if table not in {
                    "NVTX_EVENTS",
                    "NVTXT_EVENTS",
                } and not table.startswith("CUPTI_ACTIVITY_KIND_"):
                    continue
                fields = {row[1] for row in db.execute(f'PRAGMA table_info("{table}")')}
                for field in fields & {"globalPid", "globalTid"}:
                    occupied.update(
                        (row[0] >> 24) & 0xFFFFFF
                        for row in db.execute(
                            f'SELECT DISTINCT "{field}" FROM "{table}" WHERE "{field}" IS NOT NULL'
                        )
                    )
                for field in fields & {"processId", "processID", "pid"}:
                    occupied.update(
                        row[0]
                        for row in db.execute(
                            f'SELECT DISTINCT "{field}" FROM "{table}" WHERE "{field}" IS NOT NULL'
                        )
                    )
        process_base = max(occupied | {0}) + 1
    output.parent.mkdir(parents=True, exist_ok=True)
    nvtxt = output.with_suffix(".nvtxt")
    if nvtxt.exists() or output.with_suffix(".verification.json").exists():
        raise ValueError("NVTXT sidecar already exists")
    # NVTXT Relative uses the same origin as Nsight's SQLite activity times.
    # Synthetic process/thread IDs identify analysis lanes, not new GPU work.
    lines = [
        "@NameProcess,ProcessId,Name",
        "@NameOsThread,ProcessId,ThreadId,Name",
        "@RangePush,Time,ProcessId,ThreadId,CategoryId,Color,Message",
        "@RangePop,Time,ProcessId,ThreadId",
        "TimeBase=Relative",
        "Payload=0",
    ]
    events = []
    # Imported thread rows can flatten nested ranges in the native viewer.
    # Put each hierarchy level on its own visible lane, and split any overlap
    # within a level so no longer range can cover a shorter one.
    lane_rows = []
    groups = _display_groups(rows, data["rank"])
    processes = {
        rank: process_base + index
        for index, rank in enumerate(sorted({key[0] for key in groups}))
    }
    if max(processes.values()) > 0xFFFFFF:
        raise ValueError("No collision-free synthetic process ID remains")
    for (rank, _, _, level), grouped in sorted(groups.items()):
        tracks = []
        for row in sorted(
            grouped,
            key=lambda item: (item["gpu_start_ns"], -item["gpu_end_ns"]),
        ):
            start, end = row["gpu_start_ns"], row["gpu_end_ns"]
            for assigned in tracks:
                if start >= assigned[-1]["gpu_end_ns"]:
                    assigned.append(row)
                    break
            else:
                tracks.append([row])
        for index, assigned in enumerate(tracks, 1):
            name = f"rank={rank} | {level}"
            if len(tracks) > 1:
                name += f" track={index}"
            # Synthetic process groups are independently collapsible in the
            # native viewer. Each rank uses a new process ID with new threads.
            process = processes[rank]
            lane_rows.append((process, name, assigned))
    process_names = {
        value: f"Rank {rank} | MFU / Est.MBU(HBM) / Est.CommBWUtil | derived GPU analysis"
        for rank, value in processes.items()
    }
    for process in sorted({entry[0] for entry in lane_rows}):
        lines.append(f'NameProcess,{process},"{process_names[process]}"')
    for lane, (process, name, assigned) in enumerate(
        sorted(lane_rows, key=lambda item: item[0]), 1
    ):
        lines.append(f'NameOsThread,{process},{lane},"{name}"')
        for order, row in enumerate(assigned):
            start, end = row["gpu_start_ns"], row["gpu_end_ns"]
            message = _scope_message(row)
            if any(char in message for char in ('"', "\r", "\n")):
                raise ValueError(
                    "NVTXT message contains unsupported quoting characters"
                )
            color = "LightBlue" if row["kind"] == "layer" else "PaleGreen"
            # Preserve parent-first input order for identical GPU spans, and
            # close them in reverse order. Label sorting would invert scopes.
            events.append(
                (
                    start,
                    1,
                    -end,
                    order,
                    f'RangePush,{start},{process},{lane},0,{color},"{message}"',
                )
            )
            events.append((end, 0, -start, -order, f"RangePop,{end},{process},{lane}"))
    lines.extend(event[-1] for event in sorted(events))
    nvtxt.write_text("\n".join(lines) + "\n")
    subprocess.run(
        [
            str(importer),
            "--cmd",
            "merge",
            "-i",
            str(report),
            "-n",
            str(nvtxt),
            "-o",
            str(output),
            "--target",
            target_id,
        ],
        check=True,
    )
    if fingerprint(report) != original_hash:
        raise RuntimeError("Source capture changed during annotation")
    if not output.is_file():
        raise RuntimeError("Writer did not produce a report")
    verification = _verify_annotation(
        report=report,
        output=output,
        nvtxt=nvtxt,
        nsys=nsys,
        source_database=source_database,
    )
    if fingerprint_file(report) != original_hash:
        raise RuntimeError("Source capture changed during verification")
    if (
        source_database is not None
        and fingerprint_file(source_database) != source_database_hash
    ):
        raise RuntimeError("Source database changed during annotation or verification")
    receipt = {
        "source_sha256": original_hash,
        "reused_source_database": (
            None
            if source_database is None
            else {
                "path": str(source_database),
                "sha256": source_database_hash,
            }
        ),
        "output_sha256": fingerprint_file(output),
        "metrics_sha256": fingerprint_file(metrics),
        "annotated_scopes": len(rows),
        "omitted_unavailable_scopes": omitted_scopes,
        "annotation_lanes": len(lane_rows),
        "import_target_id": target_id,
        "synthetic_process_ids": processes,
        **verification,
    }
    verification_path = output.with_suffix(".verification.json")
    with verification_path.open("x") as stream:
        stream.write(json.dumps(receipt, indent=2) + "\n")
    return receipt


def fingerprint_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _nvtxt_ranges(path: Path) -> Counter:
    ranges, stacks = Counter(), {}
    with path.open(newline="") as stream:
        for row in csv.reader(stream):
            if row and row[0] == "RangePush":
                stacks.setdefault((int(row[2]), int(row[3])), []).append(
                    (int(row[1]), row[6])
                )
            elif row and row[0] == "RangePop":
                start, text = stacks[(int(row[2]), int(row[3]))].pop()
                ranges[(start, int(row[1]), text)] += 1
    if any(stacks.values()):
        raise ValueError("Unclosed derived NVTXT ranges")
    return ranges


def _verify_annotation(
    *,
    report: Path,
    output: Path,
    nvtxt: Path,
    nsys: str,
    source_database: Path | None = None,
) -> dict:
    """Audit full CUPTI/clone rows and original NVTX measurement multisets."""
    with tempfile.TemporaryDirectory(prefix="tokenspeed-native-audit-") as directory:
        paths = [Path(directory) / f"{name}.sqlite" for name in ("source", "annotated")]
        if source_database is not None:
            paths[0] = source_database
        for index, (trace, database) in enumerate(
            zip((report, output), paths, strict=True)
        ):
            if index == 0 and source_database is not None:
                continue
            subprocess.run(
                [nsys, "export", "--type=sqlite", f"--output={database}", str(trace)],
                check=True,
            )
        with sqlite3.connect(
            f"{paths[0].resolve().as_uri()}?mode=ro", uri=True
        ) as raw, sqlite3.connect(
            f"{paths[1].resolve().as_uri()}?mode=ro", uri=True
        ) as annotated:
            audit = audit_nvtx_locators(raw, annotated)
            measurements = {}
            tables = {
                row[0]
                for row in raw.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            after_tables = {
                row[0]
                for row in annotated.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            for table in tables | after_tables:
                if table != "CUDA_GRAPH_NODE_EVENTS" and not table.startswith(
                    "CUPTI_ACTIVITY_KIND_"
                ):
                    continue
                if table not in tables or table not in after_tables:
                    raise ValueError(f"Measurement table changed: {table}")
                fields = [
                    row[1] for row in raw.execute(f'PRAGMA table_info("{table}")')
                ]
                if fields != [
                    row[1] for row in annotated.execute(f'PRAGMA table_info("{table}")')
                ]:
                    raise ValueError(f"Measurement schema changed: {table}")
                before = fingerprint(raw, table, fields)
                if before != fingerprint(annotated, table, fields):
                    raise ValueError(f"Original measurement rows changed: {table}")
                measurements[table] = before
            expected = _nvtxt_ranges(nvtxt)
            if "NVTXT_EVENTS" in tables:
                expected.update(raw.execute("SELECT start,end,text FROM NVTXT_EVENTS"))
            if "NVTXT_EVENTS" not in after_tables or expected != Counter(
                annotated.execute("SELECT start,end,text FROM NVTXT_EVENTS")
            ):
                raise ValueError(
                    "Derived NVTXT labels/timestamps differ or previous annotations changed"
                )
    return {
        "validator_passed": True,
        "original_measurements_preserved": measurements,
        "nvtx_locator_audit": audit,
        "all_labels_and_timestamps_match": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--import-nvtxt", type=Path, required=True)
    parser.add_argument(
        "--target-id", required=True, help="Actual Hw:Vm ID from report info"
    )
    parser.add_argument("--nsys", required=True, help="Matching Nsight CLI")
    parser.add_argument(
        "--source-database",
        type=Path,
        help="Reuse original SQLite export verified against metrics report/database SHA256",
    )
    args = parser.parse_args()
    result = annotate_report(
        report=args.report,
        metrics=args.metrics,
        output=args.output,
        importer=args.import_nvtxt,
        target_id=args.target_id,
        nsys=args.nsys,
        source_database=args.source_database,
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
