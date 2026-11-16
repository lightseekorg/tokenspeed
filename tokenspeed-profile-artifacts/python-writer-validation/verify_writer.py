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

"""Compare Writer scopes and experimental PID/TID schemas against a real capture.

Native scope probes use a version-specific Python metadata adapter for the
PID/TID types declared in nvToolsExtPayload.h. They are diagnostics, not a
supported replacement for the active-capture NVTX bridge.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import sqlite3
import subprocess
import sys
from enum import Enum
from pathlib import Path

import numpy as np
import nvtx


class NativeScopeType(Enum):
    # Official NVTX C schema types missing from Python PayloadEntryType 0.2.16.
    PID_UINT32 = 71
    TID_UINT32 = 73
    TID_UINT64 = 74


def native_dtype(mode):
    if importlib.metadata.version("nvtx") != "0.2.16":
        raise RuntimeError("The experimental metadata adapter requires nvtx 0.2.16")
    from nvtx._metadata import NVTX_DTYPE_METADATA_KEY_NAME, _PayloadMetadata

    def scope_dtype(storage, entry_type):
        return np.dtype(
            storage,
            metadata={
                NVTX_DTYPE_METADATA_KEY_NAME: _PayloadMetadata(entry_type=entry_type)
            },
        )

    fields = [
        ("start", nvtx.numpy_dtype("i8", entry_kind=nvtx.EntryKind.RANGE_BEGIN)),
        ("end", nvtx.numpy_dtype("i8", entry_kind=nvtx.EntryKind.RANGE_END)),
    ]
    if mode == "native_pid_tid32":
        fields += [
            ("pid", scope_dtype("u4", NativeScopeType.PID_UINT32)),
            ("tid", scope_dtype("u4", NativeScopeType.TID_UINT32)),
        ]
    else:
        fields.append(("tid", scope_dtype("u8", NativeScopeType.TID_UINT64)))
    fields.append(
        ("message", nvtx.numpy_dtype("U512", entry_kind=nvtx.EntryKind.MESSAGE))
    )
    return np.dtype(fields)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--nsys", type=Path, required=True)
parser.add_argument("--source-report", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument(
    "--modes",
    nargs="+",
    choices=("default_scope", "custom_scope", "native_pid_tid32", "native_tid64"),
    required=True,
)
args = parser.parse_args()
root = args.output_dir.resolve()
root.mkdir(parents=True, exist_ok=False)
nsys = args.nsys.resolve()
sys.path.insert(0, str(nsys.parent / "python/packages"))
os.environ["NSYS_WRITER_BACKEND"] = str(
    nsys.parent.parent / "host-linux-x64/libNvtxwBackend.so"
)
from nsys_writer import Session, TimeBase

source = args.source_report.resolve()
original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
with sqlite3.connect(f"file:{source.with_suffix('.sqlite')}?mode=ro", uri=True) as db:
    events = db.execute(
        "SELECT start,end,globalTid,text FROM NVTX_EVENTS "
        "WHERE text LIKE '%MFU=%' ORDER BY start,end DESC"
    ).fetchall()
assert len(events) == 9

results = {"source_sha256": original_hash, "source_ranges": len(events)}
for mode in args.modes:
    output = root / mode
    if output.with_suffix(".nsys-rep").exists():
        raise FileExistsError(f"Use a fresh experiment directory: {output}")
    with Session(
        str(output), report_merge=source, time_base=TimeBase.RELATIVE
    ) as session:
        domain = session.get_domain("PythonWriter")
        try:
            session.create_stream(
                "original_thread",
                domain=domain,
                scope=nvtx.PredefinedScope.CURRENT_SW_THREAD,
            )
        except (ValueError, TypeError) as error:
            results["current_thread_scope_error"] = str(error)
        else:
            results["current_thread_scope_error"] = None
        scope = domain.get_scope("Derived Analysis") if mode == "custom_scope" else None
        schema = (
            domain.get_schema(native_dtype(mode), kind=nvtx.EventKind.RANGE_PUSHPOP)
            if mode.startswith("native_")
            else None
        )
        with session.create_stream(
            "Python metrics", domain=domain, scope=scope
        ) as stream:
            for start, end, global_tid, label in events:
                message = f"PYTHON_WRITER | {label}"
                if schema is None:
                    stream.write_pushpop(start=start, end=end, message=message)
                else:
                    # Nsight exports VM, process and thread identifiers together.
                    # Probe both OS PID/TID and the full 64-bit exported ID.
                    ids = (
                        ((global_tid >> 24) & 0xFFFFFF, global_tid & 0xFFFFFF)
                        if mode == "native_pid_tid32"
                        else (global_tid,)
                    )
                    if len(message) >= 512:
                        raise ValueError(
                            "Label exceeds the experimental schema capacity"
                        )
                    stream.write_event(schema, (start, end, *ids, message))
    report = output.with_suffix(".nsys-rep")
    subprocess.run(
        [
            str(nsys),
            "stats",
            "--report=nvtx_gpu_proj_trace",
            "--format=csv",
            f"--output={output}",
            str(report),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    with (root / f"{mode}_nvtx_gpu_proj_trace.csv").open() as handle:
        projections = list(csv.DictReader(handle))
    with sqlite3.connect(
        f"file:{output.with_suffix('.sqlite')}?mode=ro", uri=True
    ) as db:
        writer_domain = db.execute(
            "SELECT domainId FROM NVTX_EVENTS WHERE eventType=75 AND text='PythonWriter'"
        ).fetchone()[0]
        tables = [
            row[0]
            for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%NVTX%'"
            )
        ]
        summary = {}
        for table in tables:
            count = db.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0]
            columns = [row[1] for row in db.execute(f'PRAGMA table_info("{table}")')]
            summary[table] = {"rows": count, "columns": columns}
            if "text" in columns:
                db.row_factory = sqlite3.Row
                written = [
                    dict(row)
                    for row in db.execute(
                        f'SELECT * FROM "{table}" WHERE domainId=? AND end IS NOT NULL',
                        (writer_domain,),
                    )
                ]
                db.row_factory = None
                if written:
                    summary[table]["writer_events"] = written
        original_ranges = db.execute(
            "SELECT count(*) FROM NVTX_EVENTS WHERE text LIKE '%MFU=%' "
            "AND domainId!=?",
            (writer_domain,),
        ).fetchone()[0]
        original_tids = {
            row[0]
            for row in db.execute(
                "SELECT DISTINCT globalTid FROM NVTX_EVENTS WHERE text LIKE '%MFU=%' "
                "AND domainId!=?",
                (writer_domain,),
            )
        }
    writer_events = summary["NVTX_EVENTS"]["writer_events"]
    expected = sorted((start, end, label) for start, end, _, label in events)
    actual = sorted(
        (
            row["start"],
            row["end"],
            (
                row["text"].removeprefix("PYTHON_WRITER | ")
                if row["text"] is not None
                else None
            ),
        )
        for row in writer_events
    )
    assert len(writer_events) == len(events) == original_ranges
    if not mode.startswith("native_"):
        assert actual == expected
    # Identical intervals and labels preserve the same containment relationships.
    depths = sorted(
        sum(
            other_start < start and end < other_end
            for other_start, other_end, _ in actual
        )
        for start, end, _ in actual
    )
    assert depths == [0, 0, 0, 1, 1, 1, 1, 1, 1]
    results[mode] = {
        "report": str(report),
        "original_ranges_preserved": original_ranges,
        "writer_ranges": len(writer_events),
        "timestamps_and_labels_match": actual == expected,
        "nesting_depths": depths,
        "source_global_tids": sorted({event[2] for event in events}),
        "merged_original_global_tids": sorted(original_tids),
        "writer_global_tids": sorted({event["globalTid"] for event in writer_events}),
        "same_execution_threads": {event["globalTid"] for event in writer_events}
        == original_tids,
        "projection_rows": len(projections),
        "writer_projection_rows": sum(
            "PYTHON_WRITER" in row["Name"] for row in projections
        ),
        "original_projection_rows": sum(
            "PYTHON_WRITER" not in row["Name"] for row in projections
        ),
        "tables": summary,
    }
    results[mode]["native_thread_target_met"] = (
        results[mode]["timestamps_and_labels_match"]
        and results[mode]["same_execution_threads"]
        and results[mode]["writer_projection_rows"] == len(events)
    )
results["source_unchanged"] = (
    hashlib.sha256(source.read_bytes()).hexdigest() == original_hash
)
(root / "results.json").write_text(json.dumps(results, indent=2) + "\n")
print(
    json.dumps(
        {
            key: (
                {k: v for k, v in value.items() if k != "tables"}
                if isinstance(value, dict)
                else value
            )
            for key, value in results.items()
        },
        indent=2,
    )
)
