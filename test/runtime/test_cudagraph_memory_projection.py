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

from __future__ import annotations

import ast
import pathlib
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed.runtime.execution.cudagraph_memory import (
    PROBE_ENTRIES_PER_LADDER,
    estimate_cudagraph_memory,
    probe_cudagraph_memory,
    reserve_and_rebind,
)
from tokenspeed.runtime.execution.memory_delta import (
    NULL_MEMORY_DELTA_OBSERVER,
    DriverMemoryDeltaObserver,
)
from tokenspeed.runtime.layers.attention.utils import (
    profile_available_cache_memory_bytes,
)


def _probe_predicate() -> ast.Return:
    """The body of the one function that decides whether a boot probes."""
    path = (
        pathlib.Path(__file__).resolve().parents[2]
        / "python/tokenspeed/runtime/execution/device.py"
    )
    predicate = next(
        node
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.FunctionDef)
        and node.name == "_can_probe_cudagraph_memory"
    )
    return next(
        node.value for node in ast.walk(predicate) if isinstance(node, ast.Return)
    )


def _build_device_side() -> ast.FunctionDef:
    path = (
        pathlib.Path(__file__).resolve().parents[2]
        / "python/tokenspeed/runtime/execution/device.py"
    )
    return next(
        node
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.FunctionDef) and node.name == "build_device_side"
    )


class TestCudagraphMemoryProjection(unittest.TestCase):
    def test_observer_measure_records_samples(self) -> None:
        class DeviceModule:
            def __init__(self) -> None:
                self.readings = iter((1000, 900, 900, 850))
                self.synchronizations = 0

            def synchronize(self, gpu_id: int) -> None:
                # The device measured, not whichever one happens to be current.
                assert gpu_id == 2
                self.synchronizations += 1

            def mem_get_info(self, gpu_id: int) -> tuple[int, int]:
                self.assert_gpu_id(gpu_id)
                return next(self.readings), 2000

            @staticmethod
            def assert_gpu_id(gpu_id: int) -> None:
                if gpu_id != 2:
                    raise AssertionError(f"unexpected GPU id {gpu_id}")

        device_module = DeviceModule()
        observer = DriverMemoryDeltaObserver(device_module, 2)
        with observer.measure("phase"):
            pass
        with observer.measure("phase"):
            pass
        self.assertEqual(observer.samples["phase"], [100, 50])
        self.assertEqual(device_module.synchronizations, 4)

    def test_null_observer_records_nothing(self) -> None:
        observer = NULL_MEMORY_DELTA_OBSERVER
        with observer.measure("decode"):
            pass
        self.assertEqual(observer.samples, {})

    def test_estimator_sums_disjoint_ladders(self) -> None:
        estimates = estimate_cudagraph_memory(
            {"prefill": (100, 7, 9, 8), "decode": (30, 5, 4)},
            {"prefill": 5, "decode": 3},
        )
        self.assertEqual(
            (
                estimates.series["prefill"].first_capture,
                estimates.series["prefill"].extrapolated_rate,
                estimates.series["prefill"].total,
            ),
            (100, 8, 132),
        )
        self.assertEqual(
            (
                estimates.series["decode"].first_capture,
                estimates.series["decode"].extrapolated_rate,
                estimates.series["decode"].total,
            ),
            (30, 5, 39),
        )
        self.assertEqual(estimates.total, 171)

    def test_a_growth_step_is_amortized_over_the_window_it_was_seen_in(self) -> None:
        """A step recurs down the ladder; the rate spreads it, it is not dropped.

        Taking the middle sample instead would extrapolate 44 more entries at
        the between-steps marginal and project a capture that cannot fit.
        """
        estimates = estimate_cudagraph_memory(
            {"prefill": (2530, 40, 650, 40), "decode": ()},
            {"prefill": 48, "decode": 0},
        )
        first = estimates.series["prefill"].first_capture
        rate = estimates.series["prefill"].extrapolated_rate
        estimate = estimates.series["prefill"].total
        self.assertEqual(first, 2530)
        self.assertEqual(rate, 244, "the step is spread over the sampled window")
        # Bounded by what was measured: 44 entries at the rate would be 10736.
        self.assertEqual(estimate, 2 * (2530 + 730))
        self.assertGreater(estimate, 5020, "the median would under-project here")

    def test_a_variants_own_first_capture_stays_in_its_own_ladder(self) -> None:
        """A variant opens its own buffers; that cost is a one-off, not a rate.

        Pooling the variants into one series puts every later variant's
        opening capture inside the sampled window and none of them in the
        tail, so the mean carries a one-off and multiplies it by every entry
        the probe did not sample -- here 500 would be charged 12 more times.
        """
        estimates = estimate_cudagraph_memory(
            {
                "decode:default": (100, 10, 10, 10),
                "decode:penalties": (500, 10, 10, 10),
            },
            {"decode:default": 8, "decode:penalties": 8},
        )

        self.assertEqual(estimates.series["decode:default"].extrapolated_rate, 10)
        self.assertEqual(estimates.series["decode:penalties"].first_capture, 500)
        self.assertEqual(estimates.series["decode:penalties"].extrapolated_rate, 10)
        # Each ladder carries its own opening capture and its own rate.
        self.assertEqual(estimates.total, (130 + 10 * 4) + (530 + 10 * 4))

    def test_sample_count_contract(self) -> None:
        """Extrapolation needs two samples and rejects excess samples."""
        with self.assertRaises(ValueError):  # one sample cannot give a rate
            estimate_cudagraph_memory(
                {"prefill": (100,), "decode": ()},
                {"prefill": 40, "decode": 0},
            )
        with self.assertRaises(ValueError):  # more samples than entries
            estimate_cudagraph_memory(
                {"prefill": (100, 7, 9), "decode": ()},
                {"prefill": 2, "decode": 0},
            )
        estimates = estimate_cudagraph_memory(
            {"prefill": (100, 7), "decode": ()},
            {"prefill": 40, "decode": 0},
        )
        self.assertEqual(estimates.series["prefill"].total, 2 * (100 + 7))

    def test_allocator_churn_is_dropped_rather_than_credited(self) -> None:
        """A region that gave memory back says nothing about what a pool costs.

        Crediting it would let a churning allocator talk the reserve down to
        nothing on a 40-entry ladder -- the case the reserve exists for.
        """
        estimates = estimate_cudagraph_memory(
            {"prefill": (100, -48, -50, -46), "decode": ()},
            {"prefill": 40, "decode": 0},
        )
        self.assertEqual(estimates.series["prefill"].first_capture, 100)
        self.assertEqual(estimates.series["prefill"].extrapolated_rate, 0)
        self.assertEqual(estimates.series["prefill"].total, 100)

    def test_a_negative_pool_creating_capture_is_floored(self) -> None:
        """The pool-creating capture can read negative; it is not a credit."""
        estimates = estimate_cudagraph_memory(
            {"prefill": (), "decode": (-48, 26, 30, 26)},
            {"prefill": 0, "decode": 10},
        )
        first = estimates.series["decode"].first_capture
        rate = estimates.series["decode"].extrapolated_rate
        estimate = estimates.series["decode"].total
        self.assertEqual(first, 0, "a net-negative pool creation is dropped")
        self.assertEqual(rate, 28)
        # The tail is bounded by the measured 82, not by the dropped first.
        self.assertEqual(estimate, 2 * (26 + 30 + 26))

    def test_estimate_never_goes_negative(self) -> None:
        estimates = estimate_cudagraph_memory(
            {"prefill": (-100, -5, -6, -7), "decode": ()},
            {"prefill": 4, "decode": 0},
        )
        self.assertEqual(estimates.series["prefill"].total, 0)
        self.assertEqual(estimates.total, 0)

    def test_the_budget_leaves_the_headroom_and_the_reserve(self) -> None:
        """The headroom is not graph space, so the reserve adds to it.

        Taking the larger of the two was measured wrong: on a boot whose
        projection sits under the headroom it discards the reserve and
        reproduces the OOM this feature exists to prevent, because the
        headroom is already spoken for by activations and fragmentation.
        """
        config = type("Config", (), {"device": "cuda"})()
        gib = 1 << 30
        with patch(
            "tokenspeed.runtime.layers.attention.utils.get_available_gpu_memory",
            return_value=10.0,
        ):
            # headroom = 20 * (1 - 0.9) = 2 GiB
            without = profile_available_cache_memory_bytes(
                config, 0, 1, 0.9, 20, graph_reserve_bytes=0
            )
            under_headroom = profile_available_cache_memory_bytes(
                config, 0, 1, 0.9, 20, graph_reserve_bytes=1 * gib
            )
            over_headroom = profile_available_cache_memory_bytes(
                config, 0, 1, 0.9, 20, graph_reserve_bytes=3 * gib
            )
        self.assertEqual(without, 8 * gib)
        self.assertEqual(
            without - under_headroom, 1 * gib, "a small reserve still costs"
        )
        self.assertEqual(without - over_headroom, 3 * gib)

    def test_the_operator_flag_gates_the_probe(self) -> None:
        """--disable-cudagraph-memory-reserve has to reach the predicate."""
        probing = _probe_predicate()
        negated = {
            operand.attr
            for node in ast.walk(probing)
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not)
            for operand in [node.operand]
            if isinstance(operand, ast.Attribute)
        }
        self.assertIn("disable_cudagraph_memory_reserve", negated)

    def test_the_boot_step_measures_releases_rebuilds_then_publishes(self) -> None:
        """Order is the contract, and every argument along it is load-bearing.

        Capturing before the rebuild would record the probe arena; rebuilding
        before the release would profile memory a live graph pool still holds;
        publishing a pool the rebuild did not return would serve the probe's.
        """
        order = []
        probe_pool, real_pool = object(), object()
        probe_draft, real_draft = object(), object()
        executor = SimpleNamespace(
            attn_backend="target backend",
            draft_attn_backend="draft backend",
            release_graphs=lambda: order.append(("release",)),
            set_cache_pool=lambda target, draft: order.append(("adopt", target, draft)),
        )

        def build_components(
            *, graph_reserve_bytes, num_lcm_blocks_override, reuse_backends
        ):
            order.append(
                ("build", graph_reserve_bytes, num_lcm_blocks_override, reuse_backends)
            )
            return ("new target", real_pool, "new draft", real_draft, "storage")

        with patch(
            "tokenspeed.runtime.execution.cudagraph_memory.probe_cudagraph_memory",
            side_effect=lambda *args: order.append(("probe", *args[1:])) or 4096,
        ):
            components = reserve_and_rebind(
                executor, build_components, "server args", 3
            )

        self.assertEqual(
            [step[0] for step in order], ["probe", "release", "build", "adopt"]
        )
        self.assertEqual(order[0], ("probe", "server args", 3))
        # The measured reserve is what the rebuild profiles against, the probe
        # override is gone, and the backends are handed back target-first.
        self.assertEqual(
            order[2], ("build", 4096, None, ("target backend", "draft backend"))
        )
        self.assertEqual(order[3], ("adopt", real_pool, real_draft))
        self.assertEqual(components[1], real_pool)
        self.assertNotIn(probe_pool, components)
        self.assertNotIn(probe_draft, components)

    def test_the_serving_capture_takes_the_whole_ladder(self) -> None:
        """``entries`` is all that separates the probe's sample from serving.

        A serving capture left holding the probe's slice records four graphs
        per family and pads every other batch onto an eager forward -- a
        silent throughput regression no boot-time check would catch.
        """
        build = _build_device_side()
        call = next(
            node
            for node in ast.walk(build)
            if isinstance(node, ast.Call)
            and getattr(node.func, "attr", None) == "capture_graphs"
        )
        entries = next(kw.value for kw in call.keywords if kw.arg == "entries")

        self.assertIsInstance(entries, ast.Constant)
        self.assertIsNone(entries.value)

    def test_the_probe_measures_the_device_it_was_given(self) -> None:
        """Every rank measures its own device, not whichever one is current.

        The reserve is also the whole projection: dropping a ladder from the
        sum reads as a smaller capture, never as a missing one.
        """
        seen = {}
        samples = {"prefill": [700, 10, 10, 10], "decode:default": [300, 6, 6, 6]}

        class Observer:
            def __init__(self, device_module, gpu_id):
                seen["gpu_id"] = gpu_id
                self.samples = samples

        executor = SimpleNamespace(
            capture_graphs=lambda entries, observer: seen.update(entries=entries),
            forward_step=SimpleNamespace(capture_entries={"decode:default": 8}),
            prefill_graph=SimpleNamespace(capture_entries={"prefill": 8}),
        )
        server_args = SimpleNamespace(
            device="cuda", mapping=SimpleNamespace(world_size=1, world_group=None)
        )
        with patch(
            "tokenspeed.runtime.execution.cudagraph_memory."
            "DriverMemoryDeltaObserver",
            Observer,
        ):
            reserve = probe_cudagraph_memory(executor, server_args, 3)

        self.assertEqual(seen["gpu_id"], 3)
        # A probe that captures the whole ladder is not a probe.
        self.assertEqual(seen["entries"], PROBE_ENTRIES_PER_LADDER)

        estimate = estimate_cudagraph_memory(
            samples, {"prefill": 8, "decode:default": 8}
        )
        self.assertGreater(estimate.series["prefill"].total, 0)
        self.assertGreater(estimate.series["decode:default"].total, 0)
        self.assertEqual(reserve, estimate.total)

    def test_a_reserve_that_eats_the_budget_names_the_escape_hatch(self) -> None:
        """The one boot failure this feature can cause has to name its flag."""
        config = type("Config", (), {"device": "cuda"})()
        with patch(
            "tokenspeed.runtime.layers.attention.utils.get_available_gpu_memory",
            return_value=2.0,
        ):
            with self.assertRaises(ValueError) as caught:
                profile_available_cache_memory_bytes(
                    config, 0, 1, 0.9, 20, graph_reserve_bytes=4 << 30
                )

        self.assertIn("disable-cudagraph-memory-reserve", str(caught.exception))

    def test_the_boot_path_runs_the_step_once_before_it_captures(self) -> None:
        """The one call site: a second one would re-probe against the real pool."""
        build = _build_device_side()
        lines = {}
        for node in ast.walk(build):
            if not isinstance(node, ast.Call):
                continue
            name = (
                node.func.id
                if isinstance(node.func, ast.Name)
                else getattr(node.func, "attr", None)
            )
            if name in {"reserve_and_rebind", "capture_graphs", "set_random_seed"}:
                lines.setdefault(name, []).append(node.lineno)

        self.assertEqual(len(lines["reserve_and_rebind"]), 1)
        self.assertEqual(len(lines["capture_graphs"]), 1)
        self.assertLess(max(lines["reserve_and_rebind"]), min(lines["capture_graphs"]))
        self.assertLess(max(lines["capture_graphs"]), min(lines["set_random_seed"]))


if __name__ == "__main__":
    unittest.main()
