"""Pure tests for heterogeneous flat KV-cache pressure metrics."""

from __future__ import annotations

import ast
import importlib.util
import os
import pathlib
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from prometheus_client import CollectorRegistry, generate_latest


def _load_module():
    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    path = os.path.join(
        repo_root, "python", "tokenspeed", "runtime", "flat_kv_metrics.py"
    )
    spec = importlib.util.spec_from_file_location("_flat_kv_metrics", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_metrics = _load_module()


def _stub_module(name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


class _DisaggDecodeExecutor:
    pass


class _DisaggPrefillExecutor:
    def __init__(self):
        self.prefill_prepared = False

    def prepare_prefill(self, _forward_op):
        self.prefill_prepared = True

    @staticmethod
    def store_prefill_token(*_args, **_kwargs):
        return None


def _load_collector_module():
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    path = repo_root / "python" / "tokenspeed" / "runtime" / "metrics" / "collector.py"
    envs = SimpleNamespace(
        TOKENSPEED_TEST_REQUEST_TIME_STATS=SimpleNamespace(get=lambda: False)
    )
    stubs = {
        "tokenspeed": _stub_module("tokenspeed", __path__=[]),
        "tokenspeed.runtime": _stub_module("tokenspeed.runtime", __path__=[]),
        "tokenspeed.runtime.utils": _stub_module(
            "tokenspeed.runtime.utils", __path__=[]
        ),
        "tokenspeed.runtime.utils.env": _stub_module(
            "tokenspeed.runtime.utils.env", envs=envs
        ),
    }
    spec = importlib.util.spec_from_file_location("_flat_metrics_collector", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    with patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module


_collector = _load_collector_module()


def _load_event_loop_metrics_seam():
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    path = repo_root / "python" / "tokenspeed" / "runtime" / "engine" / "event_loop.py"
    tree = ast.parse(path.read_text())
    source_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "EventLoop"
    )
    method_names = {
        "_dispatch_forward",
        "_flat_pool_usage_summary",
        "_flat_pool_usage_state",
        "_get_scheduler_stats",
    }
    methods = [
        node
        for node in source_class.body
        if isinstance(node, ast.FunctionDef) and node.name in method_names
    ]
    assert {node.name for node in methods} == method_names
    seam_class = ast.ClassDef(
        name="EventLoopMetricsSeam",
        bases=[],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    module = ast.fix_missing_locations(
        ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__",
                    names=[ast.alias(name="annotations")],
                    level=0,
                ),
                seam_class,
            ],
            type_ignores=[],
        )
    )
    namespace = {
        "DisaggDecodeExecutor": _DisaggDecodeExecutor,
        "DisaggPrefillExecutor": _DisaggPrefillExecutor,
        "collect_flat_pool_metrics": _metrics.collect_flat_pool_metrics,
    }
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["EventLoopMetricsSeam"]


EventLoopMetricsSeam = _load_event_loop_metrics_seam()


def _snapshot(
    pool_id,
    *,
    usable,
    free,
    active,
    reserved,
    bytes_per_block,
    cached_evictable=0,
    pinned_cached=0,
):
    return {
        "pool_id": pool_id,
        "usable_blocks": usable,
        "free_blocks": free,
        "active_blocks": active,
        "cached_evictable_blocks": cached_evictable,
        "pinned_cached_blocks": pinned_cached,
        "reserved_blocks": reserved,
        "bytes_per_block": bytes_per_block,
    }


class FlatKvMetricsTest(unittest.TestCase):
    def test_equivalent_total_pages_requires_a_non_negative_int(self):
        snapshot = _snapshot(
            "history",
            usable=4,
            free=4,
            active=0,
            reserved=0,
            bytes_per_block=16,
        )
        for invalid in (True, 1.5, "4", None, -1):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "non-negative int"):
                    _metrics.collect_flat_pool_metrics(
                        [snapshot], equivalent_total_pages=invalid
                    )

    def test_byte_utilization_and_bottleneck_pressure_are_distinct(self):
        summary = _metrics.summarize_flat_pool_snapshots(
            [
                _snapshot(
                    "large",
                    usable=9,
                    free=8,
                    active=1,
                    reserved=0,
                    bytes_per_block=100,
                ),
                _snapshot(
                    "tight",
                    usable=1,
                    free=0,
                    active=1,
                    reserved=0,
                    bytes_per_block=1,
                ),
            ],
            equivalent_total_pages=100,
        )

        self.assertAlmostEqual(summary["byte_utilization"], 101 / 901)
        self.assertEqual(summary["pressure"], 1.0)
        self.assertEqual(summary["active_equivalent_pages"], 12)
        self.assertEqual(summary["pressure_equivalent_pages"], 100)

    def test_reservation_contributes_to_pressure_without_becoming_active(self):
        summary = _metrics.summarize_flat_pool_snapshots(
            [
                _snapshot(
                    "reserved",
                    usable=10,
                    free=8,
                    active=2,
                    reserved=3,
                    bytes_per_block=4,
                )
            ],
            equivalent_total_pages=10,
        )
        self.assertEqual(summary["active_equivalent_pages"], 2)
        self.assertEqual(summary["pressure_equivalent_pages"], 5)

    def test_rejects_non_conserving_snapshot(self):
        with self.assertRaisesRegex(ValueError, "conserve"):
            _metrics.summarize_flat_pool_snapshots(
                [
                    _snapshot(
                        "bad",
                        usable=10,
                        free=8,
                        active=3,
                        reserved=0,
                        bytes_per_block=4,
                    )
                ],
                equivalent_total_pages=10,
            )

    def test_collect_returns_source_domain_rows_for_prometheus(self):
        summary, rows = _metrics.collect_flat_pool_metrics(
            [
                _snapshot(
                    "history",
                    usable=8,
                    free=5,
                    active=3,
                    cached_evictable=2,
                    pinned_cached=1,
                    reserved=2,
                    bytes_per_block=128,
                )
            ],
            equivalent_total_pages=16,
        )

        self.assertEqual(summary["active_bytes"], 384)
        self.assertEqual(summary["capacity_bytes"], 1024)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["pool_id"], "history")
        self.assertEqual(rows[0]["cached_evictable_blocks"], 2)
        self.assertAlmostEqual(rows[0]["pressure"], 5 / 8)

    def test_duplicate_pool_ids_fail_closed(self):
        row = _snapshot(
            "same",
            usable=4,
            free=4,
            active=0,
            reserved=0,
            bytes_per_block=16,
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            _metrics.collect_flat_pool_metrics(
                [row, dict(row)],
                equivalent_total_pages=4,
            )

    def test_pool_metric_rows_can_be_skipped_for_the_scalar_fast_path(self):
        snapshot = _snapshot(
            "history",
            usable=4,
            free=3,
            active=1,
            reserved=1,
            bytes_per_block=16,
        )
        snapshot.pop("cached_evictable_blocks")
        snapshot.pop("pinned_cached_blocks")

        summary, rows = _metrics.collect_flat_pool_metrics(
            [snapshot],
            equivalent_total_pages=4,
            include_pool_metrics=False,
        )

        self.assertEqual(rows, ())
        self.assertEqual(summary["active_bytes"], 16)
        self.assertEqual(summary["pressure_equivalent_pages"], 2)

    def test_engine_metrics_exports_real_prometheus_series_and_updates_down(self):
        registry = CollectorRegistry()
        labels = {"model_name": "v4", "app_key": "", "dp_rank": "0"}
        metrics = _collector.EngineMetrics(
            labels=labels,
            enabled=True,
            registry=registry,
        )

        first_summary, first_rows = _metrics.collect_flat_pool_metrics(
            [
                _snapshot(
                    "history",
                    usable=8,
                    free=4,
                    active=4,
                    cached_evictable=2,
                    pinned_cached=1,
                    reserved=2,
                    bytes_per_block=128,
                ),
                _snapshot(
                    "state",
                    usable=4,
                    free=3,
                    active=1,
                    reserved=0,
                    bytes_per_block=32,
                ),
            ],
            equivalent_total_pages=16,
        )
        metrics.record_scheduler_iteration(
            running=2,
            waiting=1,
            num_active_pages=int(first_summary["active_equivalent_pages"]),
            num_total_pages=16,
            num_iteration_tokens=5,
            flat_summary=first_summary,
            flat_pool_metrics=first_rows,
        )

        payload = generate_latest(registry).decode()
        self.assertIn("tokenspeed:flat_kv_pool_blocks", payload)
        self.assertIn('flat_pool_id="history"', payload)
        self.assertIn('flat_pool_id="state"', payload)
        self.assertEqual(
            registry.get_sample_value(
                "tokenspeed:flat_kv_pool_blocks",
                {**labels, "flat_pool_id": "history", "state": "active"},
            ),
            4,
        )
        self.assertEqual(
            registry.get_sample_value(
                "tokenspeed:flat_kv_pool_blocks",
                {
                    **labels,
                    "flat_pool_id": "history",
                    "state": "cached_evictable",
                },
            ),
            2,
        )
        self.assertEqual(
            registry.get_sample_value(
                "tokenspeed:flat_kv_pool_blocks",
                {**labels, "flat_pool_id": "state", "state": "active"},
            ),
            1,
        )
        self.assertEqual(
            registry.get_sample_value("tokenspeed:flat_kv_active_bytes", labels),
            544,
        )
        self.assertEqual(
            registry.get_sample_value("tokenspeed:flat_kv_capacity_bytes", labels),
            1152,
        )
        self.assertAlmostEqual(
            registry.get_sample_value("tokenspeed:flat_kv_bottleneck_pressure", labels),
            0.75,
        )

        second_summary, second_rows = _metrics.collect_flat_pool_metrics(
            [
                _snapshot(
                    "history",
                    usable=8,
                    free=7,
                    active=1,
                    reserved=0,
                    bytes_per_block=128,
                ),
                _snapshot(
                    "state",
                    usable=4,
                    free=4,
                    active=0,
                    reserved=0,
                    bytes_per_block=32,
                ),
            ],
            equivalent_total_pages=16,
        )
        metrics.record_scheduler_iteration(
            running=1,
            waiting=0,
            num_active_pages=int(second_summary["active_equivalent_pages"]),
            num_total_pages=16,
            num_iteration_tokens=1,
            flat_summary=second_summary,
            flat_pool_metrics=second_rows,
        )

        self.assertEqual(
            registry.get_sample_value(
                "tokenspeed:flat_kv_pool_blocks",
                {**labels, "flat_pool_id": "history", "state": "active"},
            ),
            1,
        )
        self.assertEqual(
            registry.get_sample_value(
                "tokenspeed:flat_kv_pool_blocks",
                {**labels, "flat_pool_id": "state", "state": "active"},
            ),
            0,
        )
        self.assertEqual(
            registry.get_sample_value("tokenspeed:flat_kv_active_bytes", labels),
            128,
        )
        self.assertAlmostEqual(
            registry.get_sample_value("tokenspeed:flat_kv_bottleneck_pressure", labels),
            0.125,
        )

    def test_event_loop_skips_pool_rows_when_metrics_are_disabled(self):
        full_snapshot = _snapshot(
            "history",
            usable=4,
            free=3,
            active=1,
            reserved=0,
            bytes_per_block=16,
        )
        summary_only_snapshot = dict(full_snapshot)
        summary_only_snapshot.pop("cached_evictable_blocks")
        summary_only_snapshot.pop("pinned_cached_blocks")
        snapshot_batches = iter(([summary_only_snapshot], [full_snapshot]))

        class Scheduler:
            @staticmethod
            def flat_pool_snapshots():
                return next(snapshot_batches)

            @staticmethod
            def waiting_size():
                return 0

            @staticmethod
            def available_kv_pages():
                raise AssertionError("flat stats must not use the legacy scalar pool")

            @staticmethod
            def active_kv_pages():
                raise AssertionError("flat stats must not use the legacy scalar pool")

        event_loop = EventLoopMetricsSeam()
        event_loop.scheduler = Scheduler()
        event_loop.server_args = SimpleNamespace(block_size=4)
        event_loop.max_total_num_tokens = 16

        event_loop.metrics = SimpleNamespace(enabled=False)
        disabled = event_loop._get_scheduler_stats()
        self.assertEqual(disabled["flat_pool_metrics"], ())
        self.assertEqual(disabled["num_active_pages"], 1)

        event_loop.metrics = SimpleNamespace(enabled=True)
        enabled = event_loop._get_scheduler_stats()
        self.assertEqual(len(enabled["flat_pool_metrics"]), 1)
        self.assertEqual(enabled["flat_pool_metrics"][0]["pool_id"], "history")

    def test_flat_metric_payload_does_not_leak_into_model_forward_routes(self):
        snapshot = _snapshot(
            "history",
            usable=4,
            free=2,
            active=2,
            reserved=1,
            bytes_per_block=16,
        )

        class Scheduler:
            @staticmethod
            def flat_pool_snapshots():
                return [snapshot]

            @staticmethod
            def waiting_size():
                return 3

        class ModelExecutor:
            def __init__(self):
                self.forward_calls = []

            @staticmethod
            def update_block_table(_forward_op):
                return None

            @staticmethod
            def reset_valid_cache_length(_forward_op):
                return None

            def execute_forward_op_with_log(
                self,
                _forward_op,
                _sampling_params_list,
                num_active_pages=0,
                num_cached_pages=0,
                num_queue_reqs=0,
                dp_global_num_tokens=None,
                dp_global_bs=None,
                dp_all_decode_or_idle=False,
                dp_all_extend=False,
                grammar_inputs=None,
                multimodal_context=None,
                capture_next_input_ids=False,
            ):
                self.forward_calls.append(
                    {
                        "stats": (
                            num_active_pages,
                            num_cached_pages,
                            num_queue_reqs,
                        ),
                        "capture_next_input_ids": capture_next_input_ids,
                    }
                )
                return "forward-result"

        class ForwardOp:
            def __init__(self, num_extends):
                self._num_extends = num_extends

            def num_extends(self):
                return self._num_extends

        event_loop = EventLoopMetricsSeam()
        event_loop.scheduler = Scheduler()
        event_loop.server_args = SimpleNamespace(block_size=4)
        event_loop.max_total_num_tokens = 16
        event_loop.metrics = SimpleNamespace(enabled=True)
        event_loop.model_executor = ModelExecutor()
        event_loop._get_multimodal_context_for_forward = lambda _forward_op: None
        event_loop._assert_epd_embeddings_received = lambda _context: None

        stats = event_loop._get_scheduler_stats()
        routes = (
            ("no-pd", None, ForwardOp(1), False, None),
            ("pd-decode", _DisaggDecodeExecutor(), ForwardOp(0), False, None),
            (
                "pd-prefill",
                _DisaggPrefillExecutor(),
                ForwardOp(1),
                True,
                _DisaggPrefillExecutor.store_prefill_token,
            ),
        )
        for (
            route,
            kv_transfer,
            forward_op,
            expected_capture,
            expected_callback,
        ) in routes:
            with self.subTest(route=route):
                event_loop.kv_transfer = kv_transfer
                result = event_loop._dispatch_forward(
                    forward_op,
                    [],
                    stats=stats,
                )

                self.assertEqual(result[0], "forward-result")
                if expected_callback is None:
                    self.assertIsNone(result[1])
                else:
                    self.assertIs(result[1], expected_callback)
                self.assertEqual(
                    event_loop.model_executor.forward_calls[-1],
                    {
                        "stats": (2, 3, 3),
                        "capture_next_input_ids": expected_capture,
                    },
                )
                if isinstance(kv_transfer, _DisaggPrefillExecutor):
                    self.assertTrue(kv_transfer.prefill_prepared)


if __name__ == "__main__":
    unittest.main()
