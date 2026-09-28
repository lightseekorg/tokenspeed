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
import tokenspeed_kernel.benchmark.generators.gemm as gemm_generator
import torch
from tokenspeed_kernel.benchmark.graph import GraphBenchmarkConfig, GraphTimer
from tokenspeed_kernel.benchmark.harness import (
    BenchmarkCaseError,
    BenchmarkRequest,
    BenchmarkStatus,
    KernelBenchmarkHarness,
)
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.registry import KernelRegistry, KernelSpec
from tokenspeed_kernel.signature import dense_tensor_format, format_signature
from utils import FakeTimer


def _mxfp8_parameters(**updates):
    parameters = {
        "M": 256,
        "N": 256,
        "K": 512,
        "quant": "mxfp8",
        "block_size": [1, 32],
        "out_dtype": "bfloat16",
    }
    parameters.update(updates)
    return parameters


@pytest.mark.parametrize(
    ("parameters", "match"),
    [
        (_mxfp8_parameters(extra=1), "Unknown"),
        (_mxfp8_parameters(quant="fp8"), "quant"),
        (_mxfp8_parameters(block_size=[128, 128]), "block_size"),
        (_mxfp8_parameters(out_dtype="float16"), "dtype"),
    ],
)
def test_mxfp8_mm_rejects_invalid_generator_parameters(
    mi350_platform, parameters, match
):
    request = BenchmarkRequest(
        family="gemm",
        mode="mm",
        parameters=parameters,
        solution=None,
        registration=None,
        cold_cache=True,
        seed=42,
    )

    with pytest.raises(BenchmarkCaseError, match=match) as raised:
        gemm_generator.prepare_mxfp8_mm(request, mi350_platform)

    assert raised.value.status is BenchmarkStatus.INVALID_CASE


@pytest.mark.parametrize(
    "parameters, match",
    [
        (
            {"batch": 12, "M": 1, "N": 512, "K": 128, "extra": 1},
            "Unknown",
        ),
        ({"batch": 0, "M": 1, "N": 512, "K": 128}, "batch"),
        (
            {"batch": 12, "M": 1, "N": 512, "K": 128, "dtype": "float16"},
            "dtype",
        ),
    ],
)
def test_dense_bmm_rejects_invalid_generator_parameters(
    mi350_platform, parameters, match
):
    request = BenchmarkRequest(
        family="gemm",
        mode="bmm",
        parameters=parameters,
        solution=None,
        registration=None,
        cold_cache=True,
        seed=42,
    )

    with pytest.raises(BenchmarkCaseError, match=match) as raised:
        gemm_generator.prepare_dense_bmm(request, mi350_platform)

    assert raised.value.status is BenchmarkStatus.INVALID_CASE


def test_dense_bmm_validation_configuration_is_opt_in() -> None:
    assert gemm_generator._parse_validation(None, dtype=torch.bfloat16, K=128) is None
    assert gemm_generator._parse_validation({}, dtype=torch.bfloat16, K=128) == {
        "runs": 5,
        "atol": 0.015,
        "rtol": 0.015,
    }
    assert gemm_generator._parse_validation(
        {"runs": 3, "atol": 0.02, "rtol": 0.01},
        dtype=torch.bfloat16,
        K=128,
    ) == {"runs": 3, "atol": 0.02, "rtol": 0.01}


@pytest.mark.parametrize(
    ("validation", "match"),
    [
        ({"runs": 0}, "positive integer"),
        ({"rtol": float("inf")}, "finite and nonnegative"),
        ({"atol": -1.0}, "finite and nonnegative"),
        ({"unexpected": 1}, "Unknown"),
    ],
)
def test_dense_bmm_rejects_invalid_validation_configuration(validation, match):
    with pytest.raises(BenchmarkCaseError, match=match) as raised:
        gemm_generator._parse_validation(
            validation,
            dtype=torch.bfloat16,
            K=128,
        )

    assert raised.value.status is BenchmarkStatus.INVALID_CASE


@pytest.mark.parametrize("corrupt", [False, True])
def test_dense_bmm_uses_registered_reference_for_local_correctness(
    mi350_platform,
    fresh_registry,
    monkeypatch,
    corrupt,
):
    _ = fresh_registry
    signature = format_signature(
        a=dense_tensor_format(torch.bfloat16),
        b=dense_tensor_format(torch.bfloat16),
    )
    candidate_spec = KernelSpec(
        name="unit_candidate_bmm",
        family="gemm",
        mode="bmm",
        solution="unit",
        format_signatures=frozenset({signature}),
    )
    reference_spec = KernelSpec(
        name="unit_reference_bmm",
        family="gemm",
        mode="bmm",
        solution="reference",
        format_signatures=frozenset({signature}),
    )
    calls = {"candidate": 0, "reference": 0}
    candidate_inputs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    reference_inputs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    seeds: list[int] = []

    def product(kwargs):
        return torch.bmm(
            kwargs["A"].float(),
            kwargs["B"].float().transpose(1, 2),
        ).to(torch.bfloat16)

    def candidate(**kwargs):
        calls["candidate"] += 1
        candidate_inputs.append((kwargs["A"], kwargs["B"], kwargs["out"]))
        output = product(kwargs)
        if corrupt:
            output.add_(1)
        kwargs["out"].copy_(output)
        return kwargs["out"]

    def reference(**kwargs):
        calls["reference"] += 1
        reference_inputs.append((kwargs["A"], kwargs["B"], kwargs["out"]))
        kwargs["out"].copy_(product(kwargs))
        return kwargs["out"]

    KernelRegistry.get().register(candidate_spec, candidate)
    KernelRegistry.get().register(reference_spec, reference)

    class InputGenerator:
        def __init__(self, seed):
            self.seed = seed

        def generate(self, *, batch, M, N, K):
            value = float(self.seed + 1)
            A = torch.full((batch, M, K), value, dtype=torch.bfloat16)
            B = torch.ones((batch, K, N), dtype=torch.bfloat16)
            return {"A": A, "B": B.transpose(1, 2), "out_dtype": torch.bfloat16}

    def get_generator(*_args, seed, **_kwargs):
        seeds.append(seed)
        return InputGenerator(seed)

    monkeypatch.setattr(gemm_generator, "get_input_generator", get_generator)
    monkeypatch.setattr(gemm_generator, "load_builtin_kernels", lambda: None)
    timer = FakeTimer()
    result = KernelBenchmarkHarness(
        timer, platform_provider=lambda: mi350_platform
    ).run(
        BenchmarkRequest(
            family="gemm",
            mode="bmm",
            parameters={
                "batch": 2,
                "M": 1,
                "N": 4,
                "K": 8,
                "dtype": "bfloat16",
                "validation": {"runs": 2, "atol": 0.0, "rtol": 0.0},
            },
            solution=None,
            registration=candidate_spec.name,
            cold_cache=True,
            seed=7,
        ),
        measurement_blocks=3,
    )

    assert seeds == [7, 8, 9]
    assert calls["reference"] == 2
    for candidate_args, reference_args in zip(
        candidate_inputs[:2], reference_inputs, strict=True
    ):
        assert candidate_args[0] is reference_args[0]
        assert candidate_args[1] is reference_args[1]
        assert candidate_args[2] is not reference_args[2]
    assert result.correctness is not None
    assert result.correctness["outputs"][0]["validator"] == "close"
    if corrupt:
        assert result.status is BenchmarkStatus.CORRECTNESS_FAILURE
        assert result.correctness["passed"] is False
        assert calls["candidate"] == 2
        assert timer.calls == 0
    else:
        assert result.status is BenchmarkStatus.SUCCESS
        assert result.correctness["passed"] is True
        assert calls["candidate"] == 3
        assert timer.calls == 1


@pytest.mark.parametrize("register_incompatible_reference", [False, True])
def test_dense_bmm_validation_requires_a_compatible_registered_reference(
    mi350_platform,
    fresh_registry,
    monkeypatch,
    register_incompatible_reference,
):
    _ = fresh_registry
    signature = format_signature(
        a=dense_tensor_format(torch.bfloat16),
        b=dense_tensor_format(torch.bfloat16),
    )
    candidate_spec = KernelSpec(
        name="unit_candidate_without_reference",
        family="gemm",
        mode="bmm",
        solution="unit",
        format_signatures=frozenset({signature}),
    )
    KernelRegistry.get().register(
        candidate_spec,
        lambda **kwargs: kwargs["out"],
    )
    if register_incompatible_reference:
        reference_spec = KernelSpec(
            name="unit_incompatible_reference",
            family="gemm",
            mode="bmm",
            solution="reference",
            format_signatures=frozenset({signature}),
            traits={"m": frozenset({99})},
        )
        KernelRegistry.get().register(
            reference_spec,
            lambda **kwargs: kwargs["out"],
        )

    class InputGenerator:
        def generate(self, *, batch, M, N, K):
            A = torch.ones((batch, M, K), dtype=torch.bfloat16)
            B = torch.ones((batch, K, N), dtype=torch.bfloat16).transpose(1, 2)
            return {"A": A, "B": B, "out_dtype": torch.bfloat16}

    monkeypatch.setattr(
        gemm_generator,
        "get_input_generator",
        lambda *_args, **_kwargs: InputGenerator(),
    )
    monkeypatch.setattr(gemm_generator, "load_builtin_kernels", lambda: None)
    timer = FakeTimer()

    result = KernelBenchmarkHarness(
        timer, platform_provider=lambda: mi350_platform
    ).run(
        BenchmarkRequest(
            family="gemm",
            mode="bmm",
            parameters={
                "batch": 2,
                "M": 1,
                "N": 4,
                "K": 8,
                "dtype": "bfloat16",
                "validation": {"runs": 1},
            },
            solution=None,
            registration=candidate_spec.name,
            cold_cache=True,
            seed=42,
        ),
        measurement_blocks=3,
    )

    assert result.status is BenchmarkStatus.REGISTRATION_MISSING
    assert result.error_phase == "preparation"
    assert "No compatible registered reference" in (result.error_message or "")
    assert timer.calls == 0


def test_exact_dense_bmm_rejects_incompatible_shape(
    mi350_platform,
    fresh_registry,
    monkeypatch,
):
    _ = fresh_registry
    signature = format_signature(
        a=dense_tensor_format(torch.bfloat16),
        b=dense_tensor_format(torch.bfloat16),
    )
    spec = KernelSpec(
        name="unit_exact_bmm",
        family="gemm",
        mode="bmm",
        solution="unit",
        format_signatures=frozenset({signature}),
        traits={
            "batch": frozenset({12}),
            "m": frozenset({1}),
            "n": frozenset({512}),
            "k": frozenset({128}),
        },
    )
    KernelRegistry.get().register(spec, lambda **_kwargs: None)
    monkeypatch.setattr(gemm_generator, "load_builtin_kernels", lambda: None)

    result = KernelBenchmarkHarness(
        FakeTimer(), platform_provider=lambda: mi350_platform
    ).run(
        BenchmarkRequest(
            family="gemm",
            mode="bmm",
            parameters={"batch": 12, "M": 2, "N": 512, "K": 128},
            solution=None,
            registration="unit_exact_bmm",
            cold_cache=True,
            seed=42,
        ),
        measurement_blocks=3,
    )

    assert result.status is BenchmarkStatus.INVALID_CASE
    assert result.registration_name is None
    assert "does not support parameters" in (result.error_message or "")


@pytest.mark.parametrize("solution", ["unit", "missing"])
def test_dense_bmm_selection_miss_is_not_applicable(
    mi350_platform,
    fresh_registry,
    monkeypatch,
    solution,
):
    _ = fresh_registry
    signature = format_signature(
        a=dense_tensor_format(torch.bfloat16),
        b=dense_tensor_format(torch.bfloat16),
    )
    spec = KernelSpec(
        name="unit_solution_bmm",
        family="gemm",
        mode="bmm",
        solution="unit",
        format_signatures=frozenset({signature}),
        traits={"m": frozenset({1})},
    )
    KernelRegistry.get().register(spec, lambda **_kwargs: None)
    monkeypatch.setattr(gemm_generator, "load_builtin_kernels", lambda: None)

    result = KernelBenchmarkHarness(
        FakeTimer(), platform_provider=lambda: mi350_platform
    ).run(
        BenchmarkRequest(
            family="gemm",
            mode="bmm",
            parameters={"batch": 12, "M": 2, "N": 512, "K": 128},
            solution=solution,
            registration=None,
            cold_cache=True,
            seed=42,
        ),
        measurement_blocks=3,
    )

    assert result.status is BenchmarkStatus.NOT_APPLICABLE
    assert "No kernel found" in (result.error_message or "")


@pytest.mark.parametrize(
    ("selection", "selection_mode"),
    [
        ({"solution": None, "registration": None}, "normal"),
        ({"solution": "gluon", "registration": None}, "solution"),
        (
            {"solution": None, "registration": "gluon_bmm_a16w16_gfx950"},
            "registration",
        ),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU is required")
def test_dense_bmm_gluon_registration_graph_replay(selection, selection_mode):
    platform = current_platform()
    if not platform.is_cdna4:
        pytest.skip("Gluon dense BMM benchmark requires an AMD CDNA4 GPU")

    harness = KernelBenchmarkHarness(
        GraphTimer(
            GraphBenchmarkConfig(
                eager_warmup_iterations=2,
                replay_warmup_iterations=1,
            )
        ),
        platform_provider=current_platform,
    )
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
                "validation": {"runs": 3},
            },
            cold_cache=True,
            seed=42,
            **selection,
        ),
        measurement_blocks=7,
    )

    assert result.status is BenchmarkStatus.SUCCESS, result.to_dict()
    assert result.registration_name == "gluon_bmm_a16w16_gfx950"
    assert result.solution == "gluon"
    assert result.selection_mode == selection_mode
    assert result.measurement_blocks == 7
    assert result.median_us is not None and result.median_us > 0.0
    assert all(sample > 0.0 for sample in result.samples_us)
    assert result.correctness is not None
    assert result.correctness["passed"] is True
    assert result.correctness["runs"] == 3
