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

"""Isolated experiment using the installed official Python Writer API."""

import csv
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import nvtx

root = Path(__file__).resolve().parent
nsys = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(nsys.parent / "python/packages"))
os.environ["NSYS_WRITER_BACKEND"] = str(
    nsys.parent.parent / "host-linux-x64/libNvtxwBackend.so"
)
from nsys_writer import Session, TimeBase

source = root.parent / "native-timeline/timeline.nsys-rep"
original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
with sqlite3.connect(f"file:{source.with_suffix('.sqlite')}?mode=ro", uri=True) as db:
    events = db.execute(
        "SELECT start,end,globalTid,text FROM NVTX_EVENTS "
        "WHERE text LIKE '%MFU=%' ORDER BY start,end DESC"
    ).fetchall()
assert len(events) == 9

results = {"source_sha256": original_hash, "source_ranges": len(events)}
for mode in ("default_scope", "custom_scope"):
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
        with session.create_stream(
            "Python metrics", domain=domain, scope=scope
        ) as stream:
            for start, end, _, label in events:
                stream.write_pushpop(
                    start=start, end=end, message=f"PYTHON_WRITER | {label}"
                )
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
                        f'SELECT * FROM "{table}" WHERE text LIKE ?',
                        ("PYTHON_WRITER%",),
                    )
                ]
                db.row_factory = None
                if written:
                    summary[table]["writer_events"] = written
        original_ranges = db.execute(
            "SELECT count(*) FROM NVTX_EVENTS WHERE text LIKE '%MFU=%' "
            "AND text NOT LIKE 'PYTHON_WRITER%'"
        ).fetchone()[0]
    writer_events = summary["NVTX_EVENTS"]["writer_events"]
    expected = sorted((start, end, label) for start, end, _, label in events)
    actual = sorted(
        (row["start"], row["end"], row["text"].removeprefix("PYTHON_WRITER | "))
        for row in writer_events
    )
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
        "writer_global_tids": sorted({event["globalTid"] for event in writer_events}),
        "projection_rows": len(projections),
        "writer_projection_rows": sum(
            "PYTHON_WRITER" in row["Name"] for row in projections
        ),
        "original_projection_rows": sum(
            "PYTHON_WRITER" not in row["Name"] for row in projections
        ),
        "tables": summary,
    }
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
