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

"""PrimTS benchmarks use the same logical candidates as direct-slot kernels."""

from dataclasses import replace

import pytest
import torch
from tokenspeed_kernel.benchmark.generators.qsa import prepare_qsa_sparse_attention
from tokenspeed_kernel.benchmark.graph import GraphBenchmarkConfig, GraphTimer
from tokenspeed_kernel.benchmark.harness import BenchmarkRequest
from tokenspeed_kernel.platform import ArchVersion, current_platform

pytestmark = pytest.mark.skipif(
    not current_platform().is_nvidia
    or current_platform().arch_version not in (ArchVersion(10, 0), ArchVersion(10, 3)),
    reason="PrimTS QSA requires SM100 or SM103",
)


@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_prims_qsa_benchmark_matches_fa2_and_captures(phase: str) -> None:
    parameters = {
        "model_profile": "qwen38_flash_tp2",
        "phase": phase,
        "batch": 2,
        "context_tokens": 320,
        "dtype": "bfloat16",
        "kv_dtype": "bfloat16",
    }
    if phase == "decode":
        parameters["q_len"] = 4
    request = BenchmarkRequest(
        family="attention",
        mode="qsa_sparse_attention",
        parameters=parameters,
        solution="flashinfer_prims",
        registration=None,
        cold_cache=False,
        seed=42,
    )
    platform = current_platform()
    prepared = prepare_qsa_sparse_attention(request, platform)
    reference = prepare_qsa_sparse_attention(
        replace(request, solution="flashinfer"), platform
    )
    expected = reference.invocation.invoke()
    actual = prepared.invocation.invoke()
    torch.testing.assert_close(actual, expected, atol=4e-3, rtol=1e-2)

    timer = GraphTimer(
        GraphBenchmarkConfig(eager_warmup_iterations=2, replay_warmup_iterations=1)
    )
    measurement = timer.measure(
        prepared.invocation, cold_cache=False, measurement_blocks=2
    )
    assert all(sample > 0 for sample in measurement.samples_us)
