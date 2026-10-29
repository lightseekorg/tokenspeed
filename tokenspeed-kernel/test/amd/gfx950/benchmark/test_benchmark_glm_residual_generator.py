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

import pytest
import torch
import torch.nn.functional as F
from tokenspeed_kernel.benchmark.generators.residual import (
    prepare_mhc_post,
    prepare_mhc_pre,
)
from tokenspeed_kernel.benchmark.harness import BenchmarkRequest, PreparedBenchmark
from tokenspeed_kernel.benchmark.validation import ValidationDatum, validate_output
from tokenspeed_kernel.numerics.torch.residual import torch_mhc_post, torch_mhc_pre
from tokenspeed_kernel.platform import current_platform
from utils import is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required for the mHC benchmarks", allow_module_level=True)


def _prepare(mode: str, tokens: int) -> PreparedBenchmark:
    parameters = {
        "model_profile": "glm53_flash_tp4",
        "tokens": tokens,
        "hc_mult": 4,
        "hidden_size": 4096,
        "dtype": "bfloat16",
        "validation": {"runs": 1, "atol": 2e-2, "rtol": 2e-2},
    }
    if mode == "mhc_pre":
        parameters.update(
            sinkhorn_iters=20,
            rms_eps=1e-5,
            hc_eps=1e-6,
            fused_norm=False,
        )
    request = BenchmarkRequest(
        family="residual",
        mode=mode,
        parameters=parameters,
        solution=None,
        registration=None,
        cold_cache=True,
        seed=123,
    )
    prepare = prepare_mhc_pre if mode == "mhc_pre" else prepare_mhc_post
    return prepare(request, current_platform())


def _outputs(mode: str, value: object) -> tuple[torch.Tensor, ...]:
    return value if mode == "mhc_pre" else (value,)


def _reference(mode: str, inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    if mode == "mhc_pre":
        return torch_mhc_pre(*inputs, 1e-5, 1e-6, 20, None, None)
    return (torch_mhc_post(*inputs),)


def _assert_outputs(
    actual: tuple[torch.Tensor, ...], expected: tuple[torch.Tensor, ...]
) -> None:
    for result, correct in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, correct, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("mode", ["mhc_pre", "mhc_post"])
@pytest.mark.parametrize("tokens", [1, 64, 65, 256, 257, 848, 8192])
def test_glm_residual_benchmark_checks_outputs_and_persistent_inputs(
    mode: str, tokens: int
) -> None:
    prepared = _prepare(mode, tokens)
    validation = prepared.validation
    assert validation is not None
    invocation = validation.prepare_run(0)
    expected = invocation.reference()
    actual = invocation.candidate()
    output_count = 3 if mode == "mhc_pre" else 1

    # Reuse the operation references used by the existing gfx950 mHC tests.
    independent = _reference(mode, expected[output_count:])
    _assert_outputs(actual[:output_count], independent)
    _assert_outputs(expected[:output_count], independent)
    for result, original in zip(
        actual[output_count:], expected[output_count:], strict=True
    ):
        assert torch.count_nonzero(original) > 0
        torch.testing.assert_close(result, original, rtol=0, atol=0)

    for index, (spec, result, correct) in enumerate(
        zip(validation.output_specs, actual, expected, strict=True)
    ):
        assert spec is not None
        outcome = validate_output(spec, [ValidationDatum(result, correct)])
        assert outcome.passed, outcome.diagnostic
        # Missing output checks or permissive input checks must not pass silently.
        corrupted = result.clone()
        first = corrupted.reshape(-1)[0]
        if index < output_count:
            first.add_(1)
        else:
            first.copy_(torch.nextafter(first, torch.full_like(first, torch.inf)))
        outcome = validate_output(spec, [ValidationDatum(corrupted, correct)])
        assert not outcome.passed


@pytest.mark.parametrize("mode", ["mhc_pre", "mhc_post"])
@pytest.mark.parametrize("tokens", [64, 65, 257])
def test_glm_residual_benchmark_timed_invocation_replays_without_state_drift(
    mode: str, tokens: int
) -> None:
    prepared = _prepare(mode, tokens)
    assert prepared.validation is not None
    output_count = 3 if mode == "mhc_pre" else 1
    expected = prepared.validation.prepare_run(0).reference()[:output_count]
    invocation = prepared.invocation
    if invocation.reset is not None:
        invocation.reset()
    _assert_outputs(_outputs(mode, invocation.invoke()), expected)
    torch.cuda.synchronize()

    if invocation.reset is not None:
        invocation.reset()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = _outputs(mode, invocation.invoke())
    for _ in range(3):
        # The timer runs the reset hook before replay, outside the measured graph.
        if invocation.reset is not None:
            invocation.reset()
        graph.replay()
        _assert_outputs(actual, expected)


@pytest.mark.parametrize("mode", ["mhc_pre", "mhc_post"])
def test_glm_residual_benchmark_graph_reads_changed_inputs(mode: str) -> None:
    prepared = _prepare(mode, 65)
    assert prepared.validation is not None
    invocation = prepared.validation.prepare_run(0)
    invocation.candidate()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = invocation.candidate()

    output_count = 3 if mode == "mhc_pre" else 1
    inputs = actual[output_count:]
    for value in inputs:
        value.mul_(1.25)
    inputs[0].mul_(2)
    expected_inputs = tuple(value.clone() for value in inputs)
    expected = _reference(mode, expected_inputs)
    graph.replay()
    _assert_outputs(actual[:output_count], expected)
    for result, correct in zip(inputs, expected_inputs, strict=True):
        torch.testing.assert_close(result, correct, rtol=0, atol=0)

    if mode == "mhc_pre":
        # GLM normalizes after this operation. Scaling the input makes an
        # accidentally fused output normalization observably different.
        normalized = F.rms_norm(expected[0], (4096,), eps=1e-5)
        assert not torch.allclose(actual[0], normalized, rtol=2e-2, atol=2e-2)
