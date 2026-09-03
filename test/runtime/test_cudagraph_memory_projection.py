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
    estimate_cudagraph_memory,
    probe_batch,
)
from tokenspeed.runtime.execution.memory_delta import memory_delta_observer
from tokenspeed.runtime.layers.attention.utils import (
    profile_available_cache_memory_bytes,
)


class _FakeDeviceModule:
    """Replays paired driver-free and allocator readings for one GPU."""

    def __init__(self, *, free: tuple[int, ...], allocated: tuple[int, ...]) -> None:
        self._free = iter(free)
        self._allocated = iter(allocated)
        self.synchronizations = 0

    def synchronize(self) -> None:
        self.synchronizations += 1

    def mem_get_info(self, gpu_id: int) -> tuple[int, int]:
        self._assert_gpu_id(gpu_id)
        return next(self._free), 2000

    def memory_allocated(self, gpu_id: int) -> int:
        self._assert_gpu_id(gpu_id)
        return next(self._allocated)

    @staticmethod
    def _assert_gpu_id(gpu_id: int) -> None:
        if gpu_id != 2:
            raise AssertionError(f"unexpected GPU id {gpu_id}")


class TestCudagraphMemoryProjection(unittest.TestCase):
    def test_observer_measure_records_samples(self) -> None:
        device_module = _FakeDeviceModule(
            free=(1000, 900, 900, 850), allocated=(0, 0, 0, 0)
        )
        observer = memory_delta_observer(
            record=True, device_module=device_module, gpu_id=2
        )
        with observer.measure("phase"):
            pass
        with observer.measure("phase"):
            pass
        self.assertEqual(observer.samples["phase"], [100, 50])
        self.assertEqual(device_module.synchronizations, 4)

    def test_observer_charges_a_region_the_allocator_alone_paid_for(self) -> None:
        """Reusing a held segment moves no driver memory but still costs."""
        device_module = _FakeDeviceModule(free=(1000, 1000), allocated=(0, 300))
        observer = memory_delta_observer(
            record=True, device_module=device_module, gpu_id=2
        )
        with observer.measure("decode"):
            pass
        self.assertEqual(observer.samples["decode"], [300])

    def test_observer_takes_the_larger_of_the_two_readings(self) -> None:
        device_module = _FakeDeviceModule(free=(1000, 600), allocated=(0, 120))
        observer = memory_delta_observer(
            record=True, device_module=device_module, gpu_id=2
        )
        with observer.measure("prefill"):
            pass
        self.assertEqual(observer.samples["prefill"], [400])

    def test_null_observer_records_nothing(self) -> None:
        observer = memory_delta_observer(record=False)
        with observer.measure("decode"):
            pass
        self.assertEqual(observer.samples, {})

    def test_probe_batch_covers_the_decode_ladder_it_captures(self) -> None:
        """Raw-gate replay reuses pool conv_state rows as its verify scratch."""

        def batch(chunk: int, context_len: int, max_batch_size: int) -> tuple[int, int]:
            b = probe_batch(
                SimpleNamespace(
                    server_args=SimpleNamespace(chunked_prefill_size=chunk),
                    model_config=SimpleNamespace(context_len=context_len),
                    max_batch_size=max_batch_size,
                )
            )
            return b.requests, b.tokens

        # One prefill request, but the decode ladder still climbs to 16.
        self.assertEqual(batch(8192, 102400, 16), (16, 16))
        # A short context makes prefill the wider of the two.
        self.assertEqual(batch(8192, 256, 8), (32, 32))
        # One token per request keeps each row at a single page.
        self.assertEqual(batch(0, 512, 1), (1, 1))

    def test_the_unguarded_paths_keep_ordinary_headroom(self) -> None:
        """A budget nothing reserves against gets headroom, not the ladder."""
        path = (
            pathlib.Path(__file__).resolve().parents[2]
            / "python/tokenspeed/runtime/utils/server_args.py"
        )
        source = path.read_text()
        self.assertIn(
            "unguarded = self.enforce_eager or self.disable_cudagraph_memory_reserve",
            source,
        )
        self.assertIn("0.9 if unguarded else 0.95", source)
        # The topology ladder the projection replaces must be gone for good.
        for stale in ("0.79", "0.81", "0.87", "0.88"):
            self.assertNotIn(f"gpu_memory_utilization = {stale}", source)

    def test_dflash_unwire_drops_the_reference_it_gave_the_target(self) -> None:
        """A probe drafter left wired keeps its buffers out of the KV budget."""
        from tokenspeed.runtime.execution.drafter.dflash import DFlashDrafter

        class Target:
            def __init__(self) -> None:
                self.wiring: dict[str, object] = {}

            def set_dflash_layers_to_capture(
                self, layer_ids, incremental_callback=None, slot_bufs=None
            ) -> None:
                self.wiring = {
                    "layer_ids": layer_ids,
                    "incremental_callback": incremental_callback,
                    "slot_bufs": slot_bufs,
                }

        drafter = DFlashDrafter.__new__(DFlashDrafter)
        drafter.target_layer_ids = [2, 46]
        target = Target()
        target.set_dflash_layers_to_capture(
            [2, 46], incremental_callback=lambda: None, slot_bufs=[object()]
        )
        self.assertIsNotNone(target.wiring["incremental_callback"])

        drafter.unwire_target(target)
        self.assertIsNone(target.wiring["incremental_callback"])
        self.assertIsNone(target.wiring["slot_bufs"])
        self.assertEqual(target.wiring["layer_ids"], [2, 46])

    def test_estimator_sums_disjoint_families(self) -> None:
        estimate = estimate_cudagraph_memory(
            {"prefill": (100, 7, 9, 8), "decode": (30, 5, 4)},
            {"prefill": 5, "decode": 3},
        )
        self.assertEqual(
            (
                estimate.prefill.first_capture,
                estimate.prefill.extrapolated_rate,
                estimate.prefill.total,
            ),
            (100, 8, 132),
        )
        self.assertEqual(
            (
                estimate.decode.first_capture,
                estimate.decode.extrapolated_rate,
                estimate.decode.total,
            ),
            (30, 4.5, 39),
        )
        self.assertEqual(estimate.total, 171)

    def test_one_off_growth_step_is_charged_but_not_extrapolated(self) -> None:
        """A single allocator growth step must not become a per-entry rate."""
        estimates = estimate_cudagraph_memory(
            {"prefill": (2530, 40, 650, 40), "decode": ()},
            {"prefill": 48, "decode": 0},
        )
        first = estimates.prefill.first_capture
        rate = estimates.prefill.extrapolated_rate
        estimate = estimates.prefill.total
        self.assertEqual(first, 2530)
        self.assertEqual(rate, 40, "the growth step must not set the rate")
        self.assertEqual(estimate, 5020)
        self.assertLess(estimate, 33080, "must not extrapolate the step")
        self.assertGreater(estimate, 4670, "must not under-project the actual")

    def test_a_later_variants_first_capture_is_charged_where_it_happens(self) -> None:
        """Decode samples retain pool-creating captures from later variants."""
        variant_major = (100, 10, 10, 10, 500, 10, 10, 10)
        estimates = estimate_cudagraph_memory(
            {"prefill": (), "decode": variant_major},
            {"prefill": 0, "decode": 8},
        )
        first = estimates.decode.first_capture
        rate = estimates.decode.extrapolated_rate
        estimate = estimates.decode.total
        self.assertEqual(rate, 10, "one 500 step must not become the rate")
        self.assertEqual(estimate, sum(variant_major))
        self.assertGreater(estimate, 170)

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
        self.assertEqual(estimates.prefill.total, 100 + 7 + 7 * 38)

    def test_allocator_churn_is_charged_but_never_extrapolated_negative(self) -> None:
        """Negative observations are charged but never extrapolated."""
        estimates = estimate_cudagraph_memory(
            {"prefill": (100, -48, -50, -46), "decode": ()},
            {"prefill": 40, "decode": 0},
        )
        first = estimates.prefill.first_capture
        rate = estimates.prefill.extrapolated_rate
        estimate = estimates.prefill.total
        self.assertEqual(rate, 0, "an all-negative sample must not project a shrink")
        self.assertEqual(estimate, 0)

    def test_pool_creating_capture_may_come_out_negative(self) -> None:
        """The pool-creating capture may have a negative net observation."""
        estimates = estimate_cudagraph_memory(
            {"prefill": (), "decode": (-48, 26, 30, 26)},
            {"prefill": 0, "decode": 10},
        )
        first = estimates.decode.first_capture
        rate = estimates.decode.extrapolated_rate
        estimate = estimates.decode.total
        self.assertEqual(first, -48)
        self.assertEqual(rate, 26)
        self.assertEqual(estimate, -48 + 26 + 30 + 26 + 26 * 6)

    def test_estimate_never_goes_negative(self) -> None:
        estimates = estimate_cudagraph_memory(
            {"prefill": (-100, -5, -6, -7), "decode": ()},
            {"prefill": 4, "decode": 0},
        )
        self.assertEqual(estimates.prefill.total, 0)
        self.assertEqual(estimates.total, 0)

    def test_reserve_defaults_to_zero_so_old_callers_get_main_behaviour(self) -> None:
        """The parameter is public API; omitting it must mean "no projection"."""
        config = type("Config", (), {"device": "cuda"})()
        with patch(
            "tokenspeed.runtime.layers.attention.utils.get_available_gpu_memory",
            return_value=10.0,
        ):
            omitted = profile_available_cache_memory_bytes(config, 0, 1, 0.9, 20)
            explicit_zero = profile_available_cache_memory_bytes(
                config, 0, 1, 0.9, 20, graph_reserve_bytes=0
            )
        self.assertEqual(omitted, explicit_zero)

    def test_profile_subtracts_graph_reserve(self) -> None:
        config = type("Config", (), {"device": "cuda"})()
        with patch(
            "tokenspeed.runtime.layers.attention.utils.get_available_gpu_memory",
            return_value=10.0,
        ):
            without = profile_available_cache_memory_bytes(
                config, 0, 1, 0.9, 20, graph_reserve_bytes=0
            )
            with_reserve = profile_available_cache_memory_bytes(
                config, 0, 1, 0.9, 20, graph_reserve_bytes=12345
            )
        self.assertEqual(without - with_reserve, 12345)

    def test_disabled_gate_dominates_probe_call(self) -> None:
        path = (
            pathlib.Path(__file__).resolve().parents[2]
            / "python/tokenspeed/runtime/execution/device.py"
        )
        tree = ast.parse(path.read_text())
        build = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "build_device_side"
        )
        calls = [
            node
            for node in ast.walk(build)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "measure_cudagraph_reserve"
        ]
        self.assertEqual(len(calls), 1)
        parent_if = next(
            node
            for node in ast.walk(build)
            if isinstance(node, ast.If) and calls[0] in list(ast.walk(node))
        )
        negated = {
            operand.attr
            for node in ast.walk(parent_if.test)
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not)
            for operand in [node.operand]
            if isinstance(operand, ast.Attribute)
        }
        # disable_prefill_graph is deliberately absent: the decode pools still
        # need projecting when only the prefill buckets are turned off.
        self.assertEqual(negated, {"disable_cudagraph_memory_reserve", "enforce_eager"})


if __name__ == "__main__":
    unittest.main()
