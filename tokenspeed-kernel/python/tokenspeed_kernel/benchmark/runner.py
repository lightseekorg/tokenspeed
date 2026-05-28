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

import math
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from typing import Any, Callable

import torch
from tokenspeed_kernel.benchmark.config import BenchmarkConfig
from tokenspeed_kernel.benchmark.result import BenchmarkResult
from tokenspeed_kernel.benchmark.throughput import ThroughputCalculator
from tokenspeed_kernel.numerics.comparison import compare_outputs
from tokenspeed_kernel.numerics.inputs import (
    get_benchmark_shapes,
    get_input_generator,
)
from tokenspeed_kernel.numerics.tolerance import get_family_tolerance
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.profiling import ProfilingConfig, profiling
from tokenspeed_kernel.registry import KernelRegistry, KernelSpec
from tokenspeed_kernel.selection import (
    ref_compatible_with_spec,
    spec_matches_shape_traits,
)

# isort: split
import tokenspeed_kernel.numerics.gemm  # noqa: F401

__all__ = ["BenchmarkRunner"]


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])

    rank = (len(sorted_values) - 1) * (percentile / 100.0)
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return float(sorted_values[low])

    weight = rank - float(low)
    return float(sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight)


class BenchmarkRunner:
    """Benchmarks kernel implementations."""

    def __init__(self, config: BenchmarkConfig | None = None):
        if not torch.cuda.is_available():
            raise RuntimeError("BenchmarkRunner requires CUDA")
        self.config = config or BenchmarkConfig()
        self.config.validate()

    def _resolve_profiling_config(self, default_output: str) -> ProfilingConfig | None:
        if not self.config.proton_profile:
            return None
        if self.config.proton_config is not None:
            return self.config.proton_config
        return ProfilingConfig(output=default_output, data="trace")

    def _profiling_context(self, default_output: str):
        proton_cfg = self._resolve_profiling_config(default_output)
        if proton_cfg is None:
            return nullcontext()
        return profiling(proton_cfg)

    def _time_kernel(
        self, kernel: Callable[..., Any], inputs: dict[str, Any]
    ) -> list[float]:
        with torch.no_grad():
            for _ in range(self.config.warmup_iters):
                kernel(**inputs)
        torch.cuda.synchronize()

        if self.config.use_cuda_events:
            start_events = [
                torch.cuda.Event(enable_timing=True)
                for _ in range(self.config.bench_iters)
            ]
            end_events = [
                torch.cuda.Event(enable_timing=True)
                for _ in range(self.config.bench_iters)
            ]
            with torch.no_grad():
                for i in range(self.config.bench_iters):
                    start_events[i].record()
                    kernel(**inputs)
                    end_events[i].record()
            torch.cuda.synchronize()
            times = [
                start.elapsed_time(end) * 1000.0
                for start, end in zip(start_events, end_events, strict=False)
            ]
        else:
            times: list[float] = []
            with torch.no_grad():
                for _ in range(self.config.bench_iters):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    kernel(**inputs)
                    torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    times.append((t1 - t0) * 1e6)

        times.sort()
        return times

    def _benchmark_one_shape(
        self,
        spec: KernelSpec,
        kernel: Callable[..., Any],
        shape: dict[str, Any],
        dtype: torch.dtype,
    ) -> BenchmarkResult | None:
        if not spec_matches_shape_traits(spec, shape):
            return None

        signature = spec.format_signature_for_primary_storage_dtype(dtype)
        if signature is None:
            return None

        generator = get_input_generator(
            spec.family,
            spec.mode,
            dtype=dtype,
            traits=spec.traits,
            format_signature=signature,
            device="cuda",
            seed=self.config.seed,
        )
        inputs = generator.generate(**shape)
        times = self._time_kernel(kernel, inputs)

        if not times:
            return None

        wanted_percentiles = set(self.config.percentiles)
        wanted_percentiles.update({50.0, 90.0, 99.0})
        percentile_values = {
            percentile: _percentile(times, percentile)
            for percentile in sorted(wanted_percentiles)
        }

        p50 = percentile_values[50.0]
        p90 = percentile_values[90.0]
        p99 = percentile_values[99.0]
        min_latency = float(times[0])
        max_latency = float(times[-1])

        throughput_dtype = inputs.get("out_dtype")
        if not isinstance(throughput_dtype, torch.dtype):
            throughput_dtype = dtype
        tflops, bandwidth = ThroughputCalculator.compute(
            spec.family,
            spec.mode,
            shape,
            p50,
            dtype=throughput_dtype,
        )

        numerics_passed: bool | None = None
        max_abs_diff: float | None = None
        max_rel_diff: float | None = None
        if self.config.verify:
            numerics_passed, max_abs_diff, max_rel_diff = self._verify_one_shape(
                spec,
                kernel,
                shape,
                dtype,
                generator,
            )

        platform = current_platform()
        return BenchmarkResult(
            kernel_name=spec.name,
            op_family=spec.family,
            op_mode=spec.mode,
            solution=spec.solution,
            dtype=str(dtype),
            platform_arch=f"{platform.vendor}:{platform.arch}",
            shape_params=dict(shape),
            median_latency_us=p50,
            p90_latency_us=p90,
            p99_latency_us=p99,
            min_latency_us=min_latency,
            max_latency_us=max_latency,
            tflops=tflops,
            bandwidth_gb_s=bandwidth,
            numerics_passed=numerics_passed,
            max_abs_diff=max_abs_diff,
            max_rel_diff=max_rel_diff,
            timestamp=datetime.now(timezone.utc).isoformat(),
            num_iters=self.config.bench_iters,
        )

    def _verify_one_shape(
        self,
        spec: KernelSpec,
        kernel: Callable[..., Any],
        shape: dict[str, Any],
        dtype: torch.dtype,
        generator: Any,
    ) -> tuple[bool | None, float | None, float | None]:
        if spec.solution == "reference":
            return None, None, None

        registry = KernelRegistry.get()
        signature = spec.format_signature_for_primary_storage_dtype(dtype)
        if signature is None:
            return None, None, None

        ref_specs = registry.get_for_operator(
            spec.family,
            spec.mode,
            format_signature=signature,
            solution="reference",
        )
        if not ref_specs:
            return None, None, None

        ref_spec = None
        for ref in ref_specs:
            if ref.name == spec.name:
                continue
            if ref_compatible_with_spec(ref, spec):
                ref_spec = ref
                break
        if ref_spec is None:
            return None, None, None
        if not spec_matches_shape_traits(ref_spec, shape):
            return None, None, None

        ref_kernel = registry.get_impl(ref_spec.name)
        if ref_kernel is None:
            return None, None, None

        verify_inputs = generator.generate(**shape)
        with torch.no_grad():
            expected = ref_kernel(**verify_inputs)
            actual = kernel(**verify_inputs)

        if not isinstance(actual, torch.Tensor) or not isinstance(
            expected, torch.Tensor
        ):
            return None, None, None

        try:
            tol_fn = get_family_tolerance(spec.family)
        except KeyError:
            return None, None, None

        tolerance = tol_fn(dtype, inputs=verify_inputs, **shape)
        comparison = compare_outputs(actual, expected, tolerance=tolerance)
        return (
            comparison.passed,
            comparison.max_abs_diff,
            comparison.max_rel_diff,
        )

    def _benchmark_kernel_impl(
        self,
        kernel_name: str,
        *,
        shapes: list[dict[str, Any]] | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> list[BenchmarkResult]:
        """Benchmark a single kernel across shapes."""
        registry = KernelRegistry.get()
        spec = registry.get_by_name(kernel_name)
        if spec is None:
            raise ValueError(f"Kernel {kernel_name!r} is not registered")

        if spec.format_signature_for_primary_storage_dtype(dtype) is None:
            raise ValueError(
                f"Kernel {kernel_name!r} does not support primary storage dtype={dtype}"
            )

        platform = current_platform()
        if not spec.capability.satisfied_by(platform):
            raise ValueError(
                f"Kernel {kernel_name!r} is not compatible with platform {platform.device_name}"
            )

        kernel = registry.get_impl(kernel_name)
        if kernel is None:
            raise ValueError(f"Kernel implementation for {kernel_name!r} is missing")

        test_shapes = shapes or get_benchmark_shapes(spec.family, spec.mode)

        results: list[BenchmarkResult] = []
        for shape in test_shapes:
            result = self._benchmark_one_shape(
                spec,
                kernel,
                shape,
                dtype,
            )
            if result is not None:
                results.append(result)
        return results

    def benchmark_kernel(
        self,
        kernel_name: str,
        *,
        shapes: list[dict[str, Any]] | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> list[BenchmarkResult]:
        with self._profiling_context(default_output=f"bench_{kernel_name}"):
            return self._benchmark_kernel_impl(
                kernel_name,
                shapes=shapes,
                dtype=dtype,
            )

    def _benchmark_op_impl(
        self,
        op_family: str,
        op_mode: str,
        *,
        shapes: list[dict[str, Any]] | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> list[BenchmarkResult]:
        """Benchmark all implementations of an op."""
        registry = KernelRegistry.get()
        platform = current_platform()
        specs = [
            spec
            for spec in registry.get_for_operator(
                op_family,
                op_mode,
                platform=platform,
            )
            if spec.format_signature_for_primary_storage_dtype(dtype) is not None
        ]

        results: list[BenchmarkResult] = []
        for spec in sorted(specs, key=lambda item: (item.solution, item.name)):
            results.extend(
                self._benchmark_kernel_impl(spec.name, shapes=shapes, dtype=dtype)
            )
        return results

    def benchmark_op(
        self,
        op_family: str,
        op_mode: str,
        *,
        shapes: list[dict[str, Any]] | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> list[BenchmarkResult]:
        with self._profiling_context(default_output=f"bench_{op_family}_{op_mode}"):
            return self._benchmark_op_impl(
                op_family,
                op_mode,
                shapes=shapes,
                dtype=dtype,
            )

    def _benchmark_all_impl(
        self,
        *,
        dtype: torch.dtype = torch.bfloat16,
    ) -> list[BenchmarkResult]:
        """Benchmark all registered kernels on this platform."""
        registry = KernelRegistry.get()
        results: list[BenchmarkResult] = []
        for family, mode in sorted(registry.list_operators()):
            try:
                op_results = self._benchmark_op_impl(family, mode, dtype=dtype)
            except KeyError:
                continue
            if op_results:
                results.extend(op_results)
        return results

    def benchmark_all(
        self,
        *,
        dtype: torch.dtype = torch.bfloat16,
    ) -> list[BenchmarkResult]:
        with self._profiling_context(default_output="bench_all"):
            return self._benchmark_all_impl(dtype=dtype)
