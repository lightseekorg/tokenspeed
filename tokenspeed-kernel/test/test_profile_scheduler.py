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

"""SQLite behavior tests runnable without importing CUDA dependencies."""

import importlib.util
import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

# Load the standard-library-only module without running GPU package bootstrap.
_MODULE = Path(__file__).parents[1] / "python/tokenspeed_kernel/profile_scheduler.py"
_SPEC = importlib.util.spec_from_file_location("scheduler_offline_test", _MODULE)
_MODULE_OBJECT = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE_OBJECT
_SPEC.loader.exec_module(_MODULE_OBJECT)
analyze_scheduler = _MODULE_OBJECT.analyze_scheduler


class SchedulerTimelineTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.database = self.root / "trace.sqlite"
        self.pid = 11
        self.tid = (self.pid << 24) | 7
        self.capture = self.root / "capture.json"
        self.capture.write_text(
            json.dumps(
                {
                    "process_id": self.pid,
                    "rank": 0,
                    "graph_scopes": [{"graph_node_ids": [1]}],
                }
            )
        )
        with sqlite3.connect(self.database) as db:
            db.executescript("""
                CREATE TABLE StringIds(id INTEGER,value TEXT);
                INSERT INTO StringIds VALUES(1,'request'),(2,'replay'),(3,'cudaGraphLaunch_v10000');
                CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,globalTid INTEGER,
                    endGlobalTid INTEGER,text TEXT,textId INTEGER);
                CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,
                    globalTid INTEGER,correlationId INTEGER,nameId INTEGER);
                CREATE TABLE CUDA_GRAPH_NODE_EVENTS(graphNodeId INTEGER,
                    originalGraphNodeId INTEGER,globalTid INTEGER);
            """)
            db.execute(
                "INSERT INTO NVTX_EVENTS VALUES(0,500,?,NULL,NULL,1)", (90 << 24,)
            )
            for start, correlation in [(10, 41), (30, 42), (50, 43)]:
                db.execute(
                    "INSERT INTO NVTX_EVENTS VALUES(?,?,?,NULL,NULL,2)",
                    (start, start + 10, self.tid),
                )
                db.execute(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(?,?,?,?,3)",
                    (start + 1, start + 2, self.tid, correlation),
                )
            db.execute(
                "INSERT INTO CUDA_GRAPH_NODE_EVENTS VALUES(101,1,?)", (self.tid,)
            )
            for kind in ("KERNEL", "MEMCPY", "MEMSET"):
                db.execute(
                    f"CREATE TABLE CUPTI_ACTIVITY_KIND_{kind}(start INTEGER,end INTEGER,"
                    "globalPid INTEGER,correlationId INTEGER,graphNodeId INTEGER,"
                    "deviceId INTEGER,contextId INTEGER,streamId INTEGER)"
                )
            self.insert(db, "KERNEL", 100, 140, 11, 41, 101, 0, 1)
            self.insert(db, "MEMCPY", 120, 150, 11, 41, 101, 0, 2)
            self.insert(db, "MEMSET", 155, 160, 11, 41, 101, 0, 1)
            self.insert(db, "KERNEL", 161, 165, 11, 41, 99, 0, 1)
            self.insert(db, "KERNEL", 200, 240, 11, 42, 101, 0, 1)
            self.insert(db, "KERNEL", 230, 260, 11, 43, 101, 0, 1)
            # Same correlation in another process must never enlarge a replay.
            self.insert(db, "KERNEL", 250, 270, 12, 41, 101, 1, 1)
            # Activities in a gap cover time but have no model/op attribution.
            self.insert(db, "MEMCPY", 170, 185, 11, 77, 0, 0, 2)
            self.insert(db, "KERNEL", 180, 195, 12, 78, 0, 1, 3)

    @staticmethod
    def insert(db, kind, start, end, pid, corr, node, device, stream):
        db.execute(
            f"INSERT INTO CUPTI_ACTIVITY_KIND_{kind} VALUES(?,?,?,?,?,?,?,?)",
            (start, end, pid << 24, corr, node, device, 1, stream),
        )

    def analyze(self, *, captures=True, completion=None):
        return analyze_scheduler(
            self.database,
            selection_mode="legacy",
            registry=None,
            process_id=self.pid,
            request_label="request",
            replay_label="replay",
            missing_launch_policy="error",
            captures=(self.capture,) if captures else (),
            completion=completion,
        )

    def test_exact_identity_and_all_stream_gap_coverage(self):
        result = self.analyze(
            completion={
                "completed_verify_count": 2,
                "output_token_count": 3,
                "source": "same-request result",
            }
        )
        first = result["steps"][0]
        self.assertEqual(first["gpu_start_ns"], 100)
        self.assertEqual(first["gpu_end_ns"], 160)
        self.assertEqual(first["gpu_span_ns"], 60)
        self.assertEqual(first["gpu_activity_sum_ns"], 75)
        self.assertEqual(first["gpu_activity_union_ns"], 55)
        self.assertEqual(first["whole_graph_launch"]["gpu_end_ns"], 165)
        gap, overlap = result["consecutive_pairs"]
        self.assertEqual(gap["gpu_period_ns"], 100)
        self.assertEqual(gap["signed_gap_ns"], 40)
        self.assertEqual(gap["all_captured_activity_union_ns"], 29)
        self.assertEqual(gap["no_captured_activity_ns"], 11)
        self.assertEqual(gap["selected_worker_activity_union_ns"], 19)
        self.assertEqual(gap["other_process_activity_union_ns"], 15)
        self.assertEqual(
            gap["by_activity_kind"], {"kernel": 19, "memcpy": 15, "memset": 0}
        )
        devices = {row["device_id"]: row for row in gap["by_device"]}
        self.assertEqual(devices[0]["activity_union_ns"], 19)
        self.assertEqual(devices[1]["activity_union_ns"], 15)
        self.assertEqual(
            {row["activity_kind"] for row in gap["by_process_device_stream_kind"]},
            {"kernel", "memcpy"},
        )
        self.assertEqual(overlap["signed_gap_ns"], -10)
        self.assertEqual(overlap["overlap_ns"], 10)
        self.assertIsNone(overlap["no_captured_activity_ns"])
        self.assertEqual(
            result["completion"]["scheduled_minus_completed_verify_count"], 1
        )
        self.assertTrue(
            all(
                row["completion_status"] == "not_identified_by_trace"
                for row in result["steps"]
            )
        )
        self.assertEqual(result["request"]["global_tid"], 90 << 24)

    def test_original_without_capture_uses_launch_boundary(self):
        result = self.analyze(captures=False)
        self.assertEqual(result["steps"][0]["gpu_end_ns"], 165)
        self.assertEqual(result["consecutive_pairs"][0]["signed_gap_ns"], 35)
        self.assertIsNone(result["completion"])
        self.assertIsNone(result["steps"][0]["rank"])

    def test_missing_gpu_launch_is_not_fabricated(self):
        with sqlite3.connect(self.database) as db:
            db.execute("DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE correlationId=42")
        result = self.analyze()
        self.assertEqual(result["scheduled_replay_count"], 3)
        self.assertEqual(result["gpu_timed_replay_count"], 2)
        self.assertIsNone(result["steps"][1]["gpu_span_ns"])
        self.assertEqual(result["consecutive_pairs"], [])

    def test_rejects_wrong_capture_and_clone_cycle(self):
        self.capture.write_text(
            json.dumps(
                {"process_id": 99, "rank": 0, "graph_scopes": [{"graph_node_ids": [1]}]}
            )
        )
        with self.assertRaisesRegex(ValueError, "another process"):
            self.analyze()
        self.capture.write_text(
            json.dumps(
                {"process_id": 11, "rank": 0, "graph_scopes": [{"graph_node_ids": [1]}]}
            )
        )
        with sqlite3.connect(self.database) as db:
            db.execute(
                "INSERT INTO CUDA_GRAPH_NODE_EVENTS VALUES(1,101,?)", (self.tid,)
            )
        with self.assertRaisesRegex(ValueError, "Cyclic"):
            self.analyze()

    def test_rejects_ambiguous_launch(self):
        with sqlite3.connect(self.database) as db:
            db.execute(
                "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(12,13,?,90,3)",
                (self.tid,),
            )
        with self.assertRaisesRegex(ValueError, "2 graph launches"):
            self.analyze()

    def test_missing_launch_identity_retains_unknown_marker_without_bridging(self):
        with sqlite3.connect(self.database) as db:
            db.execute("DELETE FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE correlationId=42")
        with self.assertRaisesRegex(ValueError, "0 graph launches"):
            self.analyze()
        result = analyze_scheduler(
            self.database,
            selection_mode="legacy",
            registry=None,
            process_id=self.pid,
            request_label="request",
            replay_label="replay",
            missing_launch_policy="report_unavailable",
            captures=(self.capture,),
            completion={
                "completed_verify_count": 2,
                "output_token_count": 3,
                "source": "request",
            },
        )
        self.assertEqual(result["replay_marker_count"], 3)
        self.assertEqual(result["identified_graph_launch_count"], 2)
        self.assertIsNone(result["scheduled_replay_count"])
        self.assertEqual(result["missing_launch_identity_count"], 1)
        self.assertEqual(result["steps"][1]["replay_marker_index"], 1)
        self.assertIsNone(result["steps"][1]["launch_correlation_id"])
        self.assertIsNone(result["steps"][1]["gpu_span_ns"])
        self.assertEqual(result["consecutive_pairs"], [])
        self.assertIsNone(
            result["completion"]["scheduled_minus_completed_verify_count"]
        )
        self.assertIn("partial", result["statistics_coverage"])

    def test_whole_capture_has_no_invented_request_boundary(self):
        result = analyze_scheduler(
            self.database,
            selection_mode="legacy",
            registry=None,
            process_id=self.pid,
            request_label=None,
            replay_label="replay",
            missing_launch_policy="error",
            whole_capture_source="dedicated single-request capture",
        )
        self.assertEqual(result["scheduled_replay_count"], 3)
        self.assertEqual(result["request"]["selection"], "whole_capture")
        self.assertIsNone(result["request"]["start_ns"])
        self.assertIsNone(result["request"]["end_ns"])
        self.assertEqual(result["request"]["replay_cpu_envelope_start_ns"], 10)

    def test_gpu_tail_and_cli_completion_provenance(self):
        with sqlite3.connect(self.database) as db:
            db.execute("UPDATE NVTX_EVENTS SET end=80 WHERE textId=1")
        completion = self.root / "completion.json"
        completion.write_text(
            json.dumps(
                {
                    "completed_verify_count": 2,
                    "output_token_count": 3,
                    "source": "samples.json trace",
                }
            )
        )
        output = self.root / "result.json"
        subprocess.run(
            [
                sys.executable,
                str(_MODULE),
                "--selection-mode",
                "legacy",
                "--database",
                str(self.database),
                "--process-id",
                str(self.pid),
                "--request-label",
                "request",
                "--replay-label",
                "replay",
                "--missing-launch-policy",
                "error",
                "--capture",
                str(self.capture),
                "--completion",
                str(completion),
                "--output",
                str(output),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        result = json.loads(output.read_text())
        self.assertEqual(result["steps"][-1]["gpu_end_ns"], 260)
        self.assertEqual(result["request"]["end_ns"], 80)
        self.assertEqual(
            result["provenance"]["completion_file"]["path"], str(completion.resolve())
        )
        self.assertEqual(len(result["provenance"]["completion_file"]["sha256"]), 64)
        with self.assertRaisesRegex(ValueError, "Database SHA-256"):
            analyze_scheduler(
                self.database,
                selection_mode="legacy",
                registry=None,
                process_id=self.pid,
                request_label="request",
                replay_label="replay",
                missing_launch_policy="error",
                expected_database_sha256="incorrect",
            )


if __name__ == "__main__":
    unittest.main()
