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

"""Behavioral tests using synthetic timings and real local HTTP requests."""

import json
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from test.cache_gds_calibration.calibrate import parser
from test.cache_gds_calibration.data import (
    PrefillCase,
    completed_repeats,
    prefill_samples,
    validate_prefix,
)
from test.cache_gds_calibration.fitting import constrained_fit, features, fit, fit_one
from test.cache_gds_calibration.prefill import import_prefill, request
from test.cache_gds_calibration.wire_collect import parser as wire_parser

import numpy as np


def samples(kind):
    shapes = (
        [(0, 1024), (0, 8192), (0, 65536), (8192, 24576), (65536, 131072)]
        if kind == "prefill"
        else [4096, 65536, 1048576, 12288, 262144]
    )
    rows = []
    coefficients = [2e-5, 4e-6] if kind == "prefill" else [8e-5, 1e-9]
    for index, shape in enumerate(shapes):
        for noise in (0.98, 1.0, 1.03):
            row = {
                "kind": kind,
                "deployment": "synthetic-test",
                "split": "train" if index < 3 else "validation",
            }
            if kind == "prefill":
                row.update(a=shape[0], b=shape[1], timing_source="plain-request-stats")
            else:
                row.update(
                    read_blocks=1,
                    aligned_bytes=shape,
                    time_basis="job-latency",
                    native={
                        "backend": "psync",
                        "inflight": 1,
                        "queue_depth": 1,
                        "max_io_bytes": 4194304,
                        "direct_io": True,
                    },
                )
            row["seconds"] = float(np.dot(features(row, kind), coefficients)) * noise
            rows.append(row)
    return rows


def dump(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


class FitTest(unittest.TestCase):
    def test_invalid_weighting_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "weighting"):
            constrained_fit([[1, 2], [2, 3]], [1, 2], "unknown")

    def test_invalid_cost_model_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "cost model"):
            features({}, "unknown")

    def test_recovers_coefficients_in_seconds(self):
        for kind, coefs in [("prefill", [2e-5, 4e-6]), ("disk", [8e-5, 1e-9])]:
            result = fit_one(samples(kind), kind, "absolute")
            np.testing.assert_allclose(result["coefficients"], coefs, rtol=1e-9)
            self.assertLess(result["validation_max_error"], 1e-12)

    def test_zero_quadratic_and_zero_per_block_are_allowed(self):
        for kind, coefs in [("prefill", [2e-5, 0]), ("disk", [0, 1e-9])]:
            rows = samples(kind)
            for row in rows:
                row["seconds"] = float(np.dot(features(row, kind), coefs))
            fitted = fit_one(rows, kind, "relative")
            np.testing.assert_allclose(fitted["coefficients"], coefs, atol=1e-18)

    def test_rank_deficiency_at_one_block_size(self):
        with self.assertRaisesRegex(ValueError, "unidentifiable"):
            constrained_fit(
                [[n, n * 65536] for n in (1, 2, 4)], [0.1, 0.2, 0.4], "absolute"
            )

    def test_negative_unconstrained_solution_uses_boundary(self):
        coefficients, _ = constrained_fit(
            [[1, 1], [1, 2], [1, 3]], [3, 2, 1], "absolute"
        )
        np.testing.assert_allclose(coefficients, [2, 0], atol=1e-12)

    def test_no_repeat_leakage_into_holdout(self):
        rows = samples("prefill")
        rows.append(dict(rows[0], split="validation"))
        with self.assertRaisesRegex(ValueError, "whole geometries"):
            fit_one(rows, "prefill", "absolute")

    def test_validation_timings_do_not_change_fitted_coefficients(self):
        rows = samples("prefill")
        baseline = fit_one(rows, "prefill", "absolute")
        for row in rows:
            if row["split"] == "validation":
                row["seconds"] *= 2
        changed = fit_one(rows, "prefill", "absolute")
        np.testing.assert_allclose(
            changed["coefficients"], baseline["coefficients"], rtol=1e-12
        )
        self.assertAlmostEqual(changed["validation_mape"], 0.5)

    def test_nonfinite_and_invalid_geometry(self):
        for field, value in [("seconds", float("nan")), ("a", -1), ("b", 0)]:
            rows = samples("prefill")
            rows[0][field] = value
            with self.assertRaises(ValueError):
                fit_one(rows, "prefill", "absolute")

    def test_mixed_disk_queues_rejected(self):
        rows = samples("disk")
        rows[0]["native"]["inflight"] = 32
        with self.assertRaisesRegex(ValueError, "mix Disk"):
            fit_one(rows, "disk", "absolute")

    def test_missing_repeats_rejected(self):
        with self.assertRaisesRegex(ValueError, "repeats"):
            fit_one(samples("disk")[::3], "disk", "absolute")

    def run_fit(self, root, *, bad_validation):
        p, d = samples("prefill"), samples("disk")
        if bad_validation:
            for row in p:
                if row["split"] == "validation":
                    row["seconds"] *= 3
        dump(root / "p.jsonl", p)
        dump(root / "d.jsonl", d)
        args = parser().parse_args(
            [
                "fit",
                "--prefill",
                str(root / "p.jsonl"),
                "--disk",
                str(root / "d.jsonl"),
                "--weighting",
                "absolute",
                "--disk-scale",
                "2",
                "--max-mape",
                "0.15",
                "--max-error",
                "0.3",
                "--max-relative-mad",
                "0.1",
                "--output-dir",
                str(root / "fit"),
            ]
        )
        fit(args)

    def test_output_and_disk_scale(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            self.run_fit(root, bad_validation=False)
            config = json.loads((root / "fit/gds_args.json").read_text())
            np.testing.assert_allclose(
                config["prefix_cache_gds_disk_read_cost"], [1.6e-4, 2e-9]
            )
            np.testing.assert_allclose(
                config["prefix_cache_gds_prefill_cost"], [2e-5, 4e-6]
            )

    def test_bad_fit_has_report_but_no_config(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            with self.assertRaisesRegex(ValueError, "acceptance"):
                self.run_fit(root, bad_validation=True)
            self.assertTrue((root / "fit/report.json").exists())
            self.assertFalse((root / "fit/gds_args.json").exists())


class ImportTest(unittest.TestCase):
    def run_import(
        self,
        root,
        *,
        cache,
        preempt,
        source,
        native_queue_ms,
    ):
        dump(
            root / "requests",
            [
                {
                    "rid": "r1",
                    "warmup": False,
                    "expected_a": 8192,
                    "expected_b": 16384,
                    "split": "train",
                    "deployment": "synthetic-test",
                }
            ],
        )
        received = 100.0 if native_queue_ms is None else 0.0
        dispatch = (
            100.001 if native_queue_ms is None else 100.0 + native_queue_ms / 1000
        )
        arrival = (
            ""
            if native_queue_ms is None
            else '1970-01-01T00:01:40.000000Z INFO event=ts_request_accepted rid="r1"\n'
        )
        stats = (
            f"INFO Req: r1 Finish! RequestStats(status='finished', prompt_tokens=16384, "
            f"cache_tokens={cache}, output_tokens=1, queue_ms=0.2, prefill_ms=120.0, "
            f"decode_wait_ms=0.0, preempt_ms=0.0, preempt_count={preempt}, recv_ts={received}, commit_ts={dispatch})\n"
        )
        trace = {
            "rid": "r1",
            "role": "prefill",
            "status": "first_output",
            "prompt_tokens": 16384,
            "cached_tokens": cache,
            "phases_ms": {"prefill": 80.0, "admission": 0.1, "transfer_observe": 999},
        }
        (root / "server.log").write_text(
            arrival + stats + "TTFTBreakdown " + json.dumps(trace) + "\n"
        )
        args = parser().parse_args(
            [
                "import-prefill",
                "--requests",
                str(root / "requests"),
                "--server-logs",
                str(root / "server.log"),
                "--timing-source",
                source,
                "--max-prefix-shortfall",
                "1",
                "--max-queue-ms",
                "10",
                "--queue-source",
                (
                    "pd-admission"
                    if source == "pd-prefill-trace"
                    else (
                        "server-accepted"
                        if native_queue_ms is not None
                        else "request-stats"
                    )
                ),
                "--output",
                str(root / "accepted"),
                "--rejected",
                str(root / "rejected"),
            ]
        )
        import_prefill(args)
        return json.loads((root / "accepted").read_text())

    def test_use_actual_prefix_and_seconds(self):
        with tempfile.TemporaryDirectory() as name:
            row = self.run_import(
                Path(name),
                cache=8191,
                preempt=0,
                source="plain-request-stats",
                native_queue_ms=None,
            )
            self.assertEqual(row["a"], 8191)
            self.assertEqual(row["seconds"], 0.12)

    def test_pd_transfer_not_in_compute(self):
        with tempfile.TemporaryDirectory() as name:
            row = self.run_import(
                Path(name),
                cache=8192,
                preempt=0,
                source="pd-prefill-trace",
                native_queue_ms=None,
            )
            self.assertEqual(row["seconds"], 0.08)

    def test_cache_miss_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            with self.assertRaisesRegex(ValueError, "invalid"):
                self.run_import(
                    Path(name),
                    cache=0,
                    preempt=0,
                    source="plain-request-stats",
                    native_queue_ms=None,
                )

    def test_preemption_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            with self.assertRaisesRegex(ValueError, "invalid"):
                self.run_import(
                    Path(name),
                    cache=8192,
                    preempt=1,
                    source="plain-request-stats",
                    native_queue_ms=None,
                )

    def test_native_queue_measured_from_acceptance(self):
        with tempfile.TemporaryDirectory() as name:
            row = self.run_import(
                Path(name),
                cache=8192,
                preempt=0,
                source="plain-request-stats",
                native_queue_ms=3,
            )
            self.assertAlmostEqual(row["queue_ms"], 3)

    def test_native_zero_queue_stat_does_not_hide_admission_wait(self):
        with tempfile.TemporaryDirectory() as name:
            with self.assertRaisesRegex(ValueError, "invalid"):
                self.run_import(
                    Path(name),
                    cache=8192,
                    preempt=0,
                    source="plain-request-stats",
                    native_queue_ms=25,
                )

    def test_explicit_queue_source_controls_isolation(self):
        for queue_source, accepted, created, expected in (
            ("server-accepted", True, True, True),
            ("request-stats", True, True, False),
            ("server-accepted", False, True, False),
            ("request-stats", True, False, False),
        ):
            with self.subTest(
                queue_source=queue_source, accepted=accepted, created=created
            ):
                with tempfile.TemporaryDirectory() as name:
                    root = Path(name)
                    dump(
                        root / "requests",
                        [
                            {
                                "rid": "r1",
                                "warmup": False,
                                "expected_a": 8192,
                                "expected_b": 16384,
                                "split": "train",
                                "deployment": "synthetic-test",
                            }
                        ],
                    )
                    # Encoding and transport precede server acceptance. The same
                    # request has 25 ms frontend delay but only 3 ms server wait.
                    created_ts = 99.978 if created else 0.0
                    arrival = (
                        '1970-01-01T00:01:40.000000Z INFO event=ts_request_accepted rid="r1"\n'
                        if accepted
                        else ""
                    )
                    (root / "server.log").write_text(
                        arrival + "Req: r1 Finish! RequestStats(status='finished', "
                        "prompt_tokens=16384, cache_tokens=8192, output_tokens=1, "
                        "queue_ms=25.0, prefill_ms=120.0, decode_wait_ms=0.0, "
                        f"preempt_ms=0.0, preempt_count=0, recv_ts={created_ts}, commit_ts=100.003)\n"
                    )
                    args = parser().parse_args(
                        [
                            "import-prefill",
                            "--requests",
                            str(root / "requests"),
                            "--server-logs",
                            str(root / "server.log"),
                            "--timing-source",
                            "plain-request-stats",
                            "--queue-source",
                            queue_source,
                            "--max-prefix-shortfall",
                            "0",
                            "--max-queue-ms",
                            "20",
                            "--output",
                            str(root / "accepted"),
                            "--rejected",
                            str(root / "rejected"),
                        ]
                    )
                    if expected:
                        import_prefill(args)
                        row = json.loads((root / "accepted").read_text())
                        self.assertAlmostEqual(row["queue_ms"], 3)
                        self.assertEqual(row["seconds"], 0.12)
                        self.assertEqual(row["queue_source"], "server-accepted")
                    else:
                        with self.assertRaisesRegex(ValueError, "invalid"):
                            import_prefill(args)
                        self.assertEqual((root / "accepted").read_text(), "")


class RequestTest(unittest.TestCase):
    def test_native_request_preserves_ids_and_dp_routing(self):
        received = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                payload = json.loads(
                    self.rfile.read(int(self.headers["Content-Length"]))
                )
                received.append(payload)
                body = json.dumps(
                    {
                        "meta_info": {
                            "id": payload["rid"],
                            "finish_reason": {"type": "length"},
                        }
                    }
                ).encode()
                self.send_response(200)
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_):
                pass

        with HTTPServer(("127.0.0.1", 0), Handler) as server:
            thread = threading.Thread(target=server.serve_forever)
            thread.start()
            try:
                request(
                    f"http://127.0.0.1:{server.server_port}",
                    [1, 3, 5],
                    "test-id",
                    7,
                    10,
                )
            finally:
                server.shutdown()
                thread.join()
        self.assertEqual(received[0]["input_ids"], [1, 3, 5])
        self.assertEqual(received[0]["data_parallel_rank"], 7)
        self.assertEqual(received[0]["sampling_params"]["max_new_tokens"], 1)
        self.assertEqual(received[0]["sampling_params"]["ignore_eos"], True)
        self.assertEqual(received[0]["rid"], "test-id")


class SampleTest(unittest.TestCase):
    def test_resume_reuses_only_completed_selected_formal_samples(self):
        case = PrefillCase(prefix_tokens=128, new_tokens=64, split="train")
        row = {
            "deployment": "synthetic-test",
            "split": "train",
            "warmup": False,
            "expected_a": 128,
            "expected_b": 192,
            "repeat": 1,
            "completion": {
                "finish_reason": "length",
                "completion_tokens": 1,
                "cached_tokens": 128,
                "prompt_tokens": 192,
            },
        }
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "requests.jsonl"
            dump(
                path,
                [
                    row,
                    dict(row, warmup=True),
                    dict(row, expected_a=256, expected_b=320),
                ],
            )
            reused = completed_repeats(
                [case],
                [str(path)],
                repeats=3,
                deployment="synthetic-test",
                max_prefix_shortfall=0,
            )
            self.assertEqual(reused, {(128, 192, 1)})
            dump(path, [row, row])
            with self.assertRaisesRegex(ValueError, "duplicate"):
                completed_repeats(
                    [case],
                    [str(path)],
                    repeats=3,
                    deployment="synthetic-test",
                    max_prefix_shortfall=0,
                )
            dump(path, [dict(row, deployment="other-deployment")])
            with self.assertRaisesRegex(ValueError, "deployment"):
                completed_repeats(
                    [case],
                    [str(path)],
                    repeats=3,
                    deployment="synthetic-test",
                    max_prefix_shortfall=0,
                )
            bad = dict(row["completion"], cached_tokens=0)
            dump(path, [dict(row, completion=bad)])
            with self.assertRaisesRegex(ValueError, "prefix/prompt"):
                completed_repeats(
                    [case],
                    [str(path)],
                    repeats=3,
                    deployment="synthetic-test",
                    max_prefix_shortfall=0,
                )

    def test_named_case_generates_prefix_plus_new_prompt_tokens(self):
        case = PrefillCase(
            **json.loads(
                '{"prefix_tokens": 131072, "new_tokens": 32768, "split": "train"}'
            )
        )
        sample = next(
            prefill_samples(
                [case], list(range(64)), repeats=1, warmup_repeats=1, seed=7
            )
        )
        self.assertEqual(len(sample.input_ids), 163840)
        self.assertEqual(len(sample.input_ids[: case.prefix_tokens]), 131072)
        self.assertEqual(len(sample.input_ids[case.prefix_tokens :]), 32768)

    def test_shared_workload_is_reproducible_and_warms_each_shape_first(self):
        cases = [
            PrefillCase(prefix_tokens=0, new_tokens=128, split="train"),
            PrefillCase(prefix_tokens=128, new_tokens=128, split="validation"),
        ]
        first = list(
            prefill_samples(cases, list(range(64)), repeats=3, warmup_repeats=2, seed=7)
        )
        second = list(
            prefill_samples(cases, list(range(64)), repeats=3, warmup_repeats=2, seed=7)
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first), 10)
        self.assertTrue(all(sample.warmup for sample in first[:4]))
        self.assertTrue(all(not sample.warmup for sample in first[4:]))
        for case in cases:
            self.assertEqual(
                sorted(sample.repeat for sample in first if sample.case == case),
                [-2, -1, 0, 1, 2],
            )
        for sample in first:
            self.assertEqual(len(sample.input_ids), sample.case.prompt_tokens)
            self.assertTrue(set(sample.input_ids) <= set(range(64)))
        self.assertEqual(len({tuple(sample.input_ids[:32]) for sample in first}), 10)

    def test_invalid_geometries_and_splits_rejected(self):
        for prefix_tokens, new_tokens, split in [
            (-1, 128, "train"),
            (128, 0, "train"),
            (0, 32, "test"),
            (0, 1.5, "train"),
            (0.0, 128, "train"),
        ]:
            with self.subTest(
                prefix_tokens=prefix_tokens, new_tokens=new_tokens, split=split
            ), self.assertRaises(ValueError):
                PrefillCase(
                    prefix_tokens=prefix_tokens, new_tokens=new_tokens, split=split
                )

    def test_duplicate_and_leaked_shapes_rejected(self):
        for split in ["train", "validation"]:
            with self.subTest(split=split), self.assertRaisesRegex(
                ValueError, "distinct"
            ):
                list(
                    prefill_samples(
                        [
                            PrefillCase(prefix_tokens=0, new_tokens=128, split="train"),
                            PrefillCase(prefix_tokens=0, new_tokens=128, split=split),
                        ],
                        list(range(64)),
                        repeats=3,
                        warmup_repeats=1,
                        seed=7,
                    )
                )

    def test_unrepresentative_corpus_rejected(self):
        with self.assertRaisesRegex(ValueError, "corpus"):
            list(
                prefill_samples(
                    [PrefillCase(prefix_tokens=0, new_tokens=128, split="train")],
                    [1, 2, 3],
                    repeats=3,
                    warmup_repeats=1,
                    seed=7,
                )
            )

    def test_prefix_tolerance_is_explicit(self):
        validate_prefix(
            expected_prefix_tokens=128,
            expected_prompt_tokens=256,
            cached_tokens=127,
            prompt_tokens=256,
            max_prefix_shortfall=1,
        )
        with self.assertRaisesRegex(ValueError, "prefix/prompt"):
            validate_prefix(
                expected_prefix_tokens=128,
                expected_prompt_tokens=256,
                cached_tokens=127,
                prompt_tokens=256,
                max_prefix_shortfall=0,
            )
        with self.assertRaisesRegex(ValueError, "prefix/prompt"):
            validate_prefix(
                expected_prefix_tokens=0,
                expected_prompt_tokens=256,
                cached_tokens=1,
                prompt_tokens=256,
                max_prefix_shortfall=1,
            )


class CliTest(unittest.TestCase):
    def test_native_default_quick_matrix_and_explicit_full_matrix(self):
        for options, geometries in (
            ([], 8),
            (["--cases", str(Path(__file__).with_name("cases.json"))], 44),
        ):
            with self.subTest(options=options):
                args = wire_parser().parse_args(
                    [
                        "--engine-args",
                        "engine-args.json",
                        "--model-path",
                        "model",
                        "--text-file",
                        "corpus.txt",
                        "--deployment",
                        "synthetic-test",
                        "--output-dir",
                        "output",
                        "--num-engines",
                        "1",
                        "--data-parallel-rank",
                        "0",
                        "--repeats",
                        "3",
                        "--warmup-repeats",
                        "1",
                        "--seed",
                        "7",
                        "--max-prefix-shortfall",
                        "0",
                        "--routing",
                        "round_robin",
                        "--trust-remote-code",
                        "no",
                        "--startup-timeout",
                        "60",
                        *options,
                    ]
                )
                cases = [
                    PrefillCase(**case)
                    for case in json.loads(Path(args.cases).read_text())
                ]
                warmups, formal = 0, 0
                for sample in prefill_samples(
                    cases,
                    list(range(64)),
                    repeats=args.repeats,
                    warmup_repeats=args.warmup_repeats,
                    seed=args.seed,
                ):
                    self.assertEqual(len(sample.input_ids), sample.case.prompt_tokens)
                    if sample.warmup:
                        warmups += 1
                    else:
                        formal += 1
                self.assertEqual(warmups, geometries)
                self.assertEqual(formal, geometries * 3)

    def test_fit_from_repository_module_writes_usable_config(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            dump(root / "prefill.jsonl", samples("prefill"))
            dump(root / "disk.jsonl", samples("disk"))
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "test.cache_gds_calibration.calibrate",
                    "fit",
                    "--prefill",
                    str(root / "prefill.jsonl"),
                    "--disk",
                    str(root / "disk.jsonl"),
                    "--weighting",
                    "absolute",
                    "--disk-scale",
                    "1",
                    "--max-mape",
                    "0.15",
                    "--max-error",
                    "0.3",
                    "--max-relative-mad",
                    "0.1",
                    "--output-dir",
                    str(root / "fit"),
                ],
                cwd=Path(__file__).resolve().parents[2],
                check=True,
                capture_output=True,
                text=True,
            )
            result = json.loads((root / "fit/gds_args.json").read_text())
            self.assertEqual(result["prefix_cache_evict_policy"], "gds")
            np.testing.assert_allclose(
                result["prefix_cache_gds_prefill_cost"], [2e-5, 4e-6], rtol=1e-9
            )
            np.testing.assert_allclose(
                result["prefix_cache_gds_disk_read_cost"], [8e-5, 1e-9], rtol=1e-9
            )


if __name__ == "__main__":
    unittest.main()
