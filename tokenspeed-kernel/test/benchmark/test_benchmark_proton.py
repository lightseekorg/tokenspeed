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

import json
import math
from pathlib import Path

import pytest
import torch
from tokenspeed_kernel.benchmark import (
    BenchmarkRequest,
    BenchmarkStatus,
    GraphBenchmarkConfig,
    GraphTimer,
    KernelBenchmarkHarness,
)
from tokenspeed_kernel.benchmark.proton import (
    ProtonProfiler,
    aggregate_proton_profiles,
)
from tokenspeed_kernel.platform import current_platform


def _metadata_kernel(marker, name="kernel", **metrics):
    return {
        "cat": "kernel",
        "name": name,
        "args": {"call_stack": [marker], "metrics": metrics},
    }


def _replay(marker, *kernels):
    return {
        "frame": {"name": marker},
        "metrics": {},
        "children": [
            {
                "frame": {"name": name},
                "metrics": {
                    "count": count,
                    "device_type": "HIP",
                    "time (ns)": duration_us * 1000,
                },
            }
            for name, duration_us, count in kernels
        ],
    }


def test_aggregate_reports_timing_and_width_specific_tflops():
    profile = aggregate_proton_profiles(
        [
            {
                "invocation_index": 0,
                "kernels": [
                    {
                        "scope_path": ["op"],
                        "name": "matmul",
                        "occurrence": 0,
                        "device_time_us": 10.0,
                        "flops8": 20_000_000.0,
                        "flops16": 10_000_000.0,
                    },
                    {
                        "scope_path": [],
                        "name": "copy",
                        "occurrence": 0,
                        "device_time_us": 4.0,
                    },
                ],
            },
            {
                "invocation_index": 1,
                "kernels": [
                    {
                        "scope_path": ["op"],
                        "name": "matmul",
                        "occurrence": 0,
                        "device_time_us": 20.0,
                        "flops8": 20_000_000.0,
                        "flops16": 10_000_000.0,
                    },
                    {
                        "scope_path": [],
                        "name": "copy",
                        "occurrence": 0,
                        "device_time_us": 6.0,
                    },
                ],
            },
        ]
    )

    assert profile["provider"] == "proton"
    assert profile["execution_mode"] == "graph_replay"
    assert profile["invocations"] == 2
    matmul, copy = profile["kernels"]
    assert matmul["device_time_us"] == {"mean": 15.0, "p50": 15.0, "p90": 19.0}
    assert matmul["tflops8"] == pytest.approx({"mean": 1.5, "p50": 1.5, "p90": 1.9})
    assert matmul["tflops16"] == pytest.approx({"mean": 0.75, "p50": 0.75, "p90": 0.95})
    assert copy["device_time_us"] == pytest.approx(
        {"mean": 5.0, "p50": 5.0, "p90": 5.8}
    )
    assert not any(key.startswith("tflops") for key in copy)


def test_profile_reconciles_repeated_launches_and_topology_mismatch(tmp_path: Path):
    metadata_path = tmp_path / "metadata.chrome_trace"
    replay_path = tmp_path / "replay.hatchet"
    markers = {
        "repeated_metadata": "__tokenspeed_benchmark_eager_metadata_0",
        "repeated_0": "__tokenspeed_benchmark_graph_replay_1",
        "repeated_1": "__tokenspeed_benchmark_graph_replay_2",
        "mismatch_metadata": "__tokenspeed_benchmark_eager_metadata_3",
        "mismatch_0": "__tokenspeed_benchmark_graph_replay_4",
        "mismatch_1": "__tokenspeed_benchmark_graph_replay_5",
    }
    metadata_path.write_text(
        json.dumps(
            {
                "traceEvents": [
                    _metadata_kernel(markers["repeated_metadata"], flops16=10_000_000),
                    _metadata_kernel(markers["repeated_metadata"], flops16=20_000_000),
                    _metadata_kernel(
                        markers["mismatch_metadata"],
                        name="expected",
                        flops16=10_000_000,
                    ),
                ]
            }
        ),
        encoding="utf-8",
    )
    replay_path.write_text(
        json.dumps(
            [
                {
                    "frame": {"name": "ROOT"},
                    "metrics": {},
                    "children": [
                        _replay(markers["repeated_0"], ("kernel", 30, 2)),
                        _replay(markers["repeated_1"], ("kernel", 60, 2)),
                        _replay(markers["mismatch_0"], ("expected", 10, 1)),
                        _replay(markers["mismatch_1"], ("different", 20, 1)),
                    ],
                },
                {},
            ]
        ),
        encoding="utf-8",
    )

    profiler = ProtonProfiler()
    profiler._scopes = {
        markers["repeated_metadata"]: ("repeated", "eager_metadata", 1),
        markers["repeated_0"]: ("repeated", "graph_replay", 0),
        markers["repeated_1"]: ("repeated", "graph_replay", 1),
        markers["mismatch_metadata"]: ("mismatch", "eager_metadata", 1),
        markers["mismatch_0"]: ("mismatch", "graph_replay", 0),
        markers["mismatch_1"]: ("mismatch", "graph_replay", 1),
    }
    profiles = profiler._read_profiles(metadata_path, replay_path)

    repeated = profiles["repeated"]["kernels"][0]
    assert repeated["launches_per_invocation"] == 2
    assert repeated["device_time_us"]["mean"] == 45.0
    assert repeated["tflops16"]["mean"] == 0.75

    mismatched = {kernel["name"]: kernel for kernel in profiles["mismatch"]["kernels"]}
    assert mismatched["expected"]["device_time_us"]["mean"] == 10.0
    assert "tflops16" not in mismatched["expected"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU is required")
def test_gfx950_gemm_produces_profile_results():
    platform = current_platform()
    if not platform.is_cdna4:
        pytest.skip("gfx950 is required")

    case_id = "proton-integration"
    harness = KernelBenchmarkHarness(
        GraphTimer(
            GraphBenchmarkConfig(
                eager_warmup_iterations=2,
                replay_warmup_iterations=1,
            )
        ),
        platform_provider=current_platform,
    )

    with ProtonProfiler(backend="roctracer") as profiler:
        result = harness.run(
            BenchmarkRequest(
                family="gemm",
                mode="bmm",
                parameters={
                    "batch": 12,
                    "M": 1,
                    "N": 512,
                    "K": 128,
                    "dtype": "bfloat16",
                    "validation": {"runs": 1},
                },
                solution=None,
                registration="gluon_bmm_a16w16_gfx950",
                cold_cache=True,
                seed=42,
            ),
            measurement_blocks=3,
            profile_invocation=lambda phase, index: profiler.profile(
                case_id, phase, index
            ),
        )

    assert result.status is BenchmarkStatus.SUCCESS, result.to_dict()
    profile = profiler.profiles[case_id]
    assert profile["provider"] == "proton"
    assert profile["execution_mode"] == "graph_replay"
    assert profile["invocations"] == 3
    assert len(profile["kernels"]) == 1

    kernel = profile["kernels"][0]
    assert kernel["name"] == "gluon_bmm_a16w16_gfx950"
    assert kernel["samples"] == 3
    for metric in ("device_time_us", "tflops16"):
        assert set(kernel[metric]) == {"mean", "p50", "p90"}
        assert all(
            math.isfinite(value) and value > 0.0 for value in kernel[metric].values()
        )
