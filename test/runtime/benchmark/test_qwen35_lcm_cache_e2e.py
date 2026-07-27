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

import importlib.util
import json
import pathlib
import tempfile
import unittest


_HERE = pathlib.Path(__file__).resolve().parent
_MODULE_PATH = _HERE / "qwen35_lcm_cache_e2e.py"
_TRACE_PATH = _HERE / "qwen35_lcm_cache_trace.json"


def _load_harness():
    if not _MODULE_PATH.is_file():
        raise AssertionError("Qwen3.5 LCM cache E2E harness is missing")
    spec = importlib.util.spec_from_file_location("qwen35_lcm_cache_e2e", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Qwen35LcmCacheE2ETest(unittest.TestCase):
    def setUp(self):
        self.harness = _load_harness()
        self.trace_spec = json.loads(_TRACE_PATH.read_text())

    def test_trace_is_deterministic_and_covers_lcm_boundaries(self):
        trace_a = self.harness.build_trace(
            self.trace_spec,
            radix_capacity_tokens=131_072,
            lcm_capacity_tokens=262_144,
        )
        trace_b = self.harness.build_trace(
            self.trace_spec,
            radix_capacity_tokens=131_072,
            lcm_capacity_tokens=262_144,
        )

        self.assertEqual(trace_a, trace_b)
        self.assertEqual(
            [bucket["prompt_tokens"] for bucket in trace_a["buckets"]],
            [128, 4096, 4224, 32769],
        )
        phases = {phase["name"]: phase for phase in trace_a["phases"]}
        for phase in phases.values():
            for bucket_name in self.trace_spec["boundary_tree"]:
                self.assertEqual(phase["request_counts"][bucket_name], 1)
        self.assertLessEqual(
            phases["no_pressure"]["working_set_tokens"],
            131_072 // 2,
        )
        fixed_target = int(
            131_072 * float(self.trace_spec["fixed_pressure_fraction"])
        )
        self.assertLessEqual(
            phases["fixed_pressure"]["working_set_tokens"], fixed_target
        )
        self.assertLess(
            fixed_target - phases["fixed_pressure"]["working_set_tokens"],
            32_768,
        )
        self.assertTrue(
            131_072
            < phases["capacity_cliff"]["working_set_tokens"]
            < 262_144
        )

        first = next(self.harness.iter_trace_requests(trace_a))
        self.assertEqual(len(first["input_ids"]), first["prompt_tokens"])
        self.assertEqual(first["input_ids"], first["input_ids"].copy())
        boundary_requests = {
            item["bucket"]: item
            for item in self.harness.iter_trace_requests(trace_a)
            if item["phase"] == "no_pressure"
            and item["round"] == "prime"
            and item["request_id"].endswith("/0")
            and item["bucket"] in self.trace_spec["boundary_tree"]
        }
        parent = boundary_requests["full_parent"]["input_ids"]
        child = boundary_requests["parent_plus_child"]["input_ids"]
        block = boundary_requests["cache_block"]["input_ids"]
        self.assertEqual(parent[: len(block)], block)
        self.assertEqual(child[: len(parent)], parent)
        self.assertEqual(
            self.harness.trace_sha256(trace_a),
            self.harness.trace_sha256(trace_b),
        )

    def test_fake_response_stream_reports_real_denominators(self):
        trace = self.harness.build_trace(
            self.trace_spec,
            radix_capacity_tokens=131_072,
            lcm_capacity_tokens=262_144,
        )
        flushed = []

        def generate(input_ids, request):
            cached_tokens = len(input_ids) if request["round"] == "replay" else 0
            return {
                "meta_info": {
                    "prompt_tokens": len(input_ids),
                    "cached_tokens": cached_tokens,
                }
            }

        observations = self.harness.run_trace(
            trace,
            generate=generate,
            flush_cache=lambda phase: flushed.append(phase),
        )
        observations.append(
            {
                "phase": "warmup",
                "bucket": "cache_block",
                "round": "warmup",
                "measured": False,
                "prompt_tokens": 10_000,
                "cached_tokens": 9_999,
                "executed_prefill_tokens": 1,
            }
        )
        summary = self.harness.summarize_observations(observations)

        self.assertEqual(
            flushed,
            ["fixed_pressure", "capacity_cliff"],
        )
        self.assertNotIn("warmup/cache_block/warmup", summary["buckets"])
        replay = summary["buckets"]["no_pressure/cache_block/replay"]
        self.assertEqual(replay["cached_tokens"], replay["prompt_tokens"])
        self.assertEqual(replay["executed_prefill_tokens"], 0)
        self.assertEqual(replay["cache_hit_rate"], 1.0)
        prime = summary["buckets"]["no_pressure/cache_block/prime"]
        self.assertEqual(prime["cached_tokens"], 0)
        self.assertEqual(
            prime["executed_prefill_tokens"],
            prime["prompt_tokens"],
        )
        self.assertEqual(
            summary["phases"]["no_pressure/replay"]["cache_hit_rate"],
            1.0,
        )

    def test_probe_geometry_models_matchable_frontier_capacity(self):
        probes = {
            "main_radix": {
                "configured_cache_bytes": 100,
                "physical_token_capacity": 8_205_440,
            },
            "branch_radix": {
                "configured_cache_bytes": 100,
                "physical_token_capacity": 8_205_440,
            },
            "branch_flat_lcm": {
                "configured_cache_bytes": 100,
                "physical_token_capacity": 15_585_280,
                "geometry": {
                    "logical_block_tokens": 128,
                    "num_lcm_blocks": 3_805,
                    "cache_blocks_per_lcm_block": {
                        "full_attention": 32,
                        "linear_attention_0": 1,
                        "linear_attention_1": 1,
                        "linear_attention_2": 1,
                    },
                },
            },
        }

        trace = self.harness.build_trace_from_probes(self.trace_spec, probes)

        # The one-token tail makes the last completed State checkpoint
        # matchable: 8 Full parents plus one checkpoint in each State group.
        self.assertEqual(trace["capacities"]["flat_lcm"], 327 * 32_769)
        self.assertEqual(
            trace["capacity_model"]["flat_lcm"],
            "bulk_32k_plus_1 with one retained checkpoint per State group "
            "and 198-parent admission reserve",
        )
        self.assertTrue(
            trace["capacities"]["radix"]
            < next(
                phase["working_set_tokens"]
                for phase in trace["phases"]
                if phase["name"] == "capacity_cliff"
            )
            < trace["capacities"]["flat_lcm"]
        )

        aligned_spec = json.loads(json.dumps(self.trace_spec))
        aligned_spec["capacity_prompt_bucket"] = "bulk_32k"
        aligned_spec["buckets"][-1] = {"name": "bulk_32k", "pages": 256}
        aligned_trace = self.harness.build_trace_from_probes(
            aligned_spec, probes
        )
        self.assertEqual(
            aligned_trace["capacities"]["flat_lcm"], 257 * 32_768
        )
        self.assertEqual(
            aligned_trace["capacity_model"]["flat_lcm"],
            "bulk_32k exact replay with two retained checkpoints per State "
            "group and 198-parent admission reserve",
        )

    def test_comparison_rejects_storage_or_bucket_regression(self):
        trace = self.harness.build_trace(
            self.trace_spec,
            radix_capacity_tokens=131_072,
            lcm_capacity_tokens=262_144,
        )
        trace_hash = self.harness.trace_sha256(trace)

        def result(arm, capacity, hit_rate):
            return {
                "arm": arm,
                "trace_sha256": trace_hash,
                "configured_cache_bytes": 10_000,
                # Physical pools include null/padding rows outside the usable
                # scheduler capacity, so actual bytes may slightly exceed the
                # profiled cache budget.
                "allocated_cache_bytes": 10_100,
                "max_total_num_tokens": capacity,
                "physical_token_capacity": capacity,
                "resident_prefix_capacity_tokens": capacity,
                "capacity_source": "lcm_geometry",
                "working_sets": {
                    phase["name"]: phase["working_set_tokens"]
                    for phase in trace["phases"]
                },
                "summary": {
                    "buckets": {
                        "capacity_cliff/full_parent/replay": {
                            "prompt_tokens": 100,
                            "cached_tokens": int(100 * hit_rate),
                            "executed_prefill_tokens": int(100 * (1 - hit_rate)),
                            "cache_hit_rate": hit_rate,
                        }
                    },
                    "phases": {
                        "no_pressure/replay": {
                            "prompt_tokens": 100,
                            "cached_tokens": (
                                95 if arm == "branch_flat_lcm" else 75
                            ),
                            "executed_prefill_tokens": (
                                5 if arm == "branch_flat_lcm" else 25
                            ),
                            "cache_hit_rate": (
                                0.95 if arm == "branch_flat_lcm" else 0.75
                            ),
                        },
                        "capacity_cliff/replay": {
                            "prompt_tokens": 100,
                            "cached_tokens": int(100 * hit_rate),
                            "executed_prefill_tokens": int(100 * (1 - hit_rate)),
                            "cache_hit_rate": hit_rate,
                        },
                    },
                },
            }

        good = {
            "main_radix": result("main_radix", 131_072, 0.2),
            "branch_radix": result("branch_radix", 131_072, 0.2),
            "branch_flat_lcm": result("branch_flat_lcm", 262_144, 0.9),
        }
        report = self.harness.compare_results(good)
        self.assertEqual(report["capacity_ratio"], 2.0)
        self.assertIn("capacity_cliff/full_parent/replay", report["buckets"])
        self.assertIn("2.000", self.harness.render_markdown(report))

        bad_budget = json.loads(json.dumps(good))
        bad_budget["branch_flat_lcm"]["configured_cache_bytes"] += 1
        with self.assertRaisesRegex(ValueError, "configured cache bytes"):
            self.harness.compare_results(bad_budget)

        bad_hit = json.loads(json.dumps(good))
        bad_hit["branch_flat_lcm"]["summary"]["buckets"][
            "capacity_cliff/full_parent/replay"
        ]["cache_hit_rate"] = 0.1
        with self.assertRaisesRegex(ValueError, "regressed"):
            self.harness.compare_results(bad_hit)

        scheduler_only = json.loads(json.dumps(good))
        scheduler_only["branch_flat_lcm"]["capacity_source"] = "scheduler"
        with self.assertRaisesRegex(ValueError, "physical allocation"):
            self.harness.compare_results(scheduler_only)

    def test_probe_run_and_compare_cli_core(self):
        trace = self.harness.build_trace(
            self.trace_spec,
            radix_capacity_tokens=131_072,
            lcm_capacity_tokens=262_144,
        )

        class FakeEngine:
            def __init__(self, **kwargs):
                capacity = int(kwargs["capacity"])
                self.scheduler_info = {
                    "max_total_num_tokens": capacity,
                }
                if not kwargs.get("omit_cache_storage", False):
                    self.scheduler_info["cache_storage"] = {
                        "configured_cache_bytes": 10_000,
                        "allocated_cache_bytes": 9_000,
                        "physical_token_capacity": capacity,
                        "capacity_source": kwargs["capacity_source"],
                    }
                self.flushes = 0
                self.shutdown_called = False

            def flush_cache(self):
                self.flushes += 1

            def generate(self, *, input_ids, sampling_params):
                del sampling_params
                return {
                    "meta_info": {
                        "prompt_tokens": len(input_ids),
                        "cached_tokens": len(input_ids),
                    }
                }

            def shutdown(self):
                self.shutdown_called = True

        engines = []

        def engine_factory(**kwargs):
            engine = FakeEngine(**kwargs)
            engines.append(engine)
            return engine

        probe = self.harness.probe_arm(
            arm="branch_flat_lcm",
            engine_args={
                "capacity": 262_144,
                "capacity_source": "lcm_geometry",
            },
            engine_factory=engine_factory,
        )
        self.assertEqual(probe["physical_token_capacity"], 262_144)
        self.assertTrue(engines[-1].shutdown_called)

        engines_before_run = len(engines)
        result = self.harness.run_arm(
            arm="branch_flat_lcm",
            trace=trace,
            engine_args={
                "capacity": 262_144,
                "capacity_source": "lcm_geometry",
            },
            engine_factory=engine_factory,
        )
        self.assertEqual(result["trace_sha256"], self.harness.trace_sha256(trace))
        self.assertEqual(result["resident_prefix_capacity_tokens"], 262_144)
        phase_engines = engines[engines_before_run:]
        self.assertEqual(len(phase_engines), len(trace["phases"]))
        self.assertTrue(all(engine.flushes == 0 for engine in phase_engines))
        self.assertTrue(all(engine.shutdown_called for engine in phase_engines))

        external_storage = {
            "configured_cache_bytes": 10_000,
            "allocated_cache_bytes": 9_500,
            "physical_token_capacity": 131_072,
            "capacity_source": "allocated_token_rows",
        }
        legacy_probe = self.harness.probe_arm(
            arm="main_radix",
            engine_args={
                "capacity": 131_072,
                "capacity_source": "unused",
                "omit_cache_storage": True,
            },
            cache_storage_override=external_storage,
            engine_factory=engine_factory,
        )
        self.assertEqual(legacy_probe["allocated_cache_bytes"], 9_500)
        self.assertTrue(engines[-1].shutdown_called)

        with tempfile.TemporaryDirectory() as temp_dir:
            output = pathlib.Path(temp_dir) / "probe.json"
            exit_code = self.harness.main(
                [
                    "probe",
                    "--arm",
                    "branch_flat_lcm",
                    "--engine-args-json",
                    json.dumps(
                        {
                            "capacity": 262_144,
                            "capacity_source": "lcm_geometry",
                        }
                    ),
                    "--output",
                    str(output),
                ],
                engine_factory=engine_factory,
            )
            self.assertEqual(exit_code, 0)
            self.assertEqual(
                json.loads(output.read_text())["physical_token_capacity"],
                262_144,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
