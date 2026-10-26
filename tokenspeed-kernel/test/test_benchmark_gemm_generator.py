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
from tokenspeed_kernel.signature import (
    ScaleFormat,
    dense_tensor_format,
    format_signature,
    tensor_format,
)
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
        (_mxfp8_parameters(block_size=[64, 128]), "block_size"),
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


def _gemm_mm_request(parameters, *, registration=None) -> BenchmarkRequest:
    return BenchmarkRequest(
        family="gemm",
        mode="mm",
        parameters=parameters,
        solution=None,
        registration=registration,
        cold_cache=True,
        seed=42,
    )


@pytest.mark.parametrize(
    "block_size,scale_dtype", [([1, 32], torch.uint8), ([128, 128], torch.float32)]
)
@pytest.mark.parametrize("corrupt", [False, True])
def test_scaled_mm_harness_validates_canonical_scale_formats(
    mi350_platform, fresh_registry, monkeypatch, block_size, scale_dtype, corrupt
):
    from tokenspeed_kernel.numerics.torch.gemm import torch_mm_fp8_blockscale

    scale = ScaleFormat(
        storage_dtype=scale_dtype,
        granularity="block",
        block_shape=tuple(block_size),
    )
    signature = format_signature(
        a=tensor_format("mxfp8", torch.float8_e4m3fn, scale=scale),
        b=tensor_format("mxfp8", torch.float8_e4m3fn, scale=scale),
    )
    candidate_spec = KernelSpec(
        name="unit_scaled_mm",
        family="gemm",
        mode="mm",
        solution="unit",
        format_signatures=frozenset({signature}),
        traits={"out_inner_stride_one": frozenset({True})},
    )
    reference_spec = KernelSpec(
        name="unit_scaled_mm_reference",
        family="gemm",
        mode="mm",
        solution="torch",
        format_signatures=frozenset({signature}),
    )

    def candidate(**inputs):
        assert inputs["A_scales"].dtype == scale_dtype
        assert inputs["B_scales"].dtype == scale_dtype
        assert inputs["A_scales"].shape == (3, 256 // block_size[1])
        assert inputs["B_scales"].shape == (
            (130 + block_size[0] - 1) // block_size[0],
            256 // block_size[1],
        )
        output = torch_mm_fp8_blockscale(**inputs)
        if corrupt:
            output.add_(1)
        return output

    KernelRegistry.get().register(candidate_spec, candidate)
    KernelRegistry.get().register(reference_spec, torch_mm_fp8_blockscale)
    get_generator = gemm_generator.get_input_generator

    def cpu_generator(*args, **kwargs):
        # Only the unavailable GPU allocation boundary is replaced.
        return get_generator(*args, **dict(kwargs, device="cpu"))

    monkeypatch.setattr(gemm_generator, "get_input_generator", cpu_generator)
    monkeypatch.setattr(gemm_generator, "load_builtin_kernels", lambda: None)
    timer = FakeTimer()
    result = KernelBenchmarkHarness(
        timer, platform_provider=lambda: mi350_platform
    ).run(
        _gemm_mm_request(
            _mxfp8_parameters(
                M=3, N=130, K=256, block_size=block_size, validation={"runs": 2}
            ),
            registration=candidate_spec.name,
        ),
        measurement_blocks=3,
    )
    assert result.correctness is not None, result.to_dict()
    assert result.correctness["passed"] is not corrupt
    assert result.status is (
        BenchmarkStatus.CORRECTNESS_FAILURE if corrupt else BenchmarkStatus.SUCCESS
    ), result.to_dict()
    assert timer.calls == (0 if corrupt else 1)
    assert result.parameters["scale_dtype"] == (
        "uint8" if scale_dtype == torch.uint8 else "float32"
    )


def _dense_mm_parameters(**updates):
    parameters = {
        "M": 2816,
        "N": 3072,
        "K": 512,
        "quant": "none",
        "dtype": "bfloat16",
    }
    parameters.update(updates)
    return parameters


def test_mm_requires_explicit_quant(mi350_platform):
    parameters = _dense_mm_parameters()
    del parameters["quant"]

    with pytest.raises(BenchmarkCaseError, match="quant") as raised:
        gemm_generator.prepare_mm(_gemm_mm_request(parameters), mi350_platform)

    assert raised.value.status is BenchmarkStatus.INVALID_CASE


def test_dense_mm_runs_through_harness_with_registered_reference(
    mi350_platform,
    fresh_registry,
    monkeypatch,
):
    _ = fresh_registry
    signature = format_signature(
        a=dense_tensor_format(torch.bfloat16),
        b=dense_tensor_format(torch.bfloat16),
    )
    candidate_spec = KernelSpec(
        name="unit_candidate_mm",
        family="gemm",
        mode="mm",
        solution="unit",
        format_signatures=frozenset({signature}),
        traits={
            "mnk_problem_filter": frozenset({lambda m, n, k: m % 4 == 0}),
            "a_inner_stride_one": frozenset({True}),
            "b_inner_stride_one": frozenset({True}),
            "out_dtype": frozenset({torch.bfloat16}),
        },
    )
    reference_spec = KernelSpec(
        name="unit_reference_mm",
        family="gemm",
        mode="mm",
        solution="torch",
        format_signatures=frozenset({signature}),
    )
    out_shapes: list[tuple[int, ...]] = []

    def kernel(**kwargs):
        out_shapes.append(tuple(kwargs["out"].shape))
        kwargs["out"].copy_(kwargs["A"] @ kwargs["B"].T)
        return kwargs["out"]

    KernelRegistry.get().register(candidate_spec, kernel)
    KernelRegistry.get().register(reference_spec, kernel)

    class InputGenerator:
        def generate(self, *, M, N, K):
            return {
                "A": torch.ones((M, K), dtype=torch.bfloat16),
                "B": torch.ones((N, K), dtype=torch.bfloat16),
                "A_scales": None,
                "B_scales": None,
                "out_dtype": torch.bfloat16,
                "alpha": None,
                "block_size": None,
            }

    monkeypatch.setattr(
        gemm_generator,
        "get_input_generator",
        lambda *_args, **_kwargs: InputGenerator(),
    )
    monkeypatch.setattr(gemm_generator, "load_builtin_kernels", lambda: None)
    # The harness routes quant="none" through prepare_mm to the dense generator,
    # whose traits must satisfy the candidate's problem filter.
    result = KernelBenchmarkHarness(
        FakeTimer(), platform_provider=lambda: mi350_platform
    ).run(
        _gemm_mm_request(
            _dense_mm_parameters(
                M=4,
                N=3,
                K=8,
                validation={"runs": 1, "atol": 0.0, "rtol": 0.0},
            ),
            registration="unit_candidate_mm",
        ),
        measurement_blocks=3,
    )

    assert result.status is BenchmarkStatus.SUCCESS, result.to_dict()
    assert result.registration_name == "unit_candidate_mm"
    assert result.correctness is not None
    assert result.correctness["passed"] is True
    assert set(out_shapes) == {(4, 3)}


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
        solution="torch",
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
            solution="torch",
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


def _grouped_projection_request(**updates) -> BenchmarkRequest:
    parameters = {
        "groups": 2,
        "M": 3,
        "N": 8,
        "K": 16,
        "x_token_stride": 64,
        "dtype": "bfloat16",
    }
    parameters.update(updates)
    return BenchmarkRequest(
        family="gemm",
        mode="grouped_bf16_projection",
        parameters=parameters,
        solution=None,
        registration=None,
        cold_cache=True,
        seed=42,
    )


def _cpu_random_tensors(_seed, dtype, *shapes):
    return [torch.zeros(shape, dtype=dtype) for shape in shapes]


@pytest.mark.parametrize(
    ("updates", "match"),
    [
        ({"extra": 1}, "Unknown"),
        ({"groups": 0}, "groups"),
        ({"x_token_stride": 31}, "x_token_stride"),
        ({"dtype": "float16"}, "dtype"),
    ],
)
def test_grouped_bf16_projection_rejects_invalid_parameters(updates, match):
    with pytest.raises(BenchmarkCaseError, match=match):
        gemm_generator.prepare_grouped_bf16_projection(
            _grouped_projection_request(**updates), None
        )


@pytest.mark.parametrize("x_token_stride", [32, 128])
def test_grouped_bf16_projection_selection_matches_operation_api(
    monkeypatch, x_token_stride
):
    from tokenspeed_kernel.ops import gemm as gemm_ops

    captured = {}

    def capture_operation(_family, _mode, signature, *, traits, **_kwargs):
        captured["operation"] = (signature, traits)
        raise RuntimeError("selection captured")

    def capture_generator(_request, _platform, signature, traits, _shape):
        captured["generator"] = (signature, traits)
        raise RuntimeError("selection captured")

    monkeypatch.setattr(gemm_ops, "select_kernel", capture_operation)
    monkeypatch.setattr(gemm_generator, "_random_tensors", _cpu_random_tensors)
    monkeypatch.setattr(gemm_generator, "load_builtin_kernels", lambda: None)
    monkeypatch.setattr(gemm_generator, "_select_registration", capture_generator)

    rows = torch.zeros((3, x_token_stride), dtype=torch.bfloat16)
    x = rows[:, :32].unflatten(1, (2, 16))
    weight = torch.zeros((2, 8, 16), dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="selection captured"):
        gemm_ops.grouped_bf16_projection(x, weight, None, None)
    with pytest.raises(RuntimeError, match="selection captured"):
        gemm_generator.prepare_grouped_bf16_projection(
            _grouped_projection_request(x_token_stride=x_token_stride), None
        )

    assert captured["generator"] == captured["operation"]


def test_grouped_bf16_projection_times_strided_view_without_out(
    mi350_platform, monkeypatch
):
    spec = KernelSpec(
        name="unit_grouped_projection",
        family="gemm",
        mode="grouped_bf16_projection",
        solution="unit",
        format_signatures=frozenset(),
        traits={},
    )
    calls = []
    monkeypatch.setattr(gemm_generator, "_random_tensors", _cpu_random_tensors)
    monkeypatch.setattr(gemm_generator, "load_builtin_kernels", lambda: None)
    monkeypatch.setattr(
        gemm_generator,
        "_select_registration",
        lambda *_args: (spec, lambda *args: calls.append(args)),
    )

    prepared = gemm_generator.prepare_grouped_bf16_projection(
        _grouped_projection_request(), mi350_platform
    )
    prepared.invocation.invoke()

    ((x, weight, out),) = calls
    assert x.shape == (3, 2, 16)
    assert x.stride() == (64, 16, 1)
    assert weight.shape == (2, 8, 16)
    assert weight.is_contiguous()
    assert out is None
    assert prepared.parameters["x_token_stride"] == 64


def _router_request(**updates) -> BenchmarkRequest:
    return BenchmarkRequest(
        family="gemm",
        mode="prepared_fp32_router_gemv",
        parameters={
            "M": 4,
            "N": 256,
            "K": 1024,
            "multiplier": 2.0,
            "dtype": "float32",
            **updates,
        },
        solution=None,
        registration=None,
        cold_cache=True,
        seed=42,
    )


@pytest.mark.parametrize(
    ("updates", "match"),
    [
        ({"dtype": "bfloat16"}, "float32"),
        ({"multiplier": 2}, "multiplier"),
        ({"extra": 1}, "Unknown"),
    ],
)
def test_prepared_router_gemv_rejects_invalid_parameters(updates, match):
    with pytest.raises(BenchmarkCaseError, match=match):
        gemm_generator.prepare_prepared_fp32_router_gemv(
            _router_request(**updates), None
        )


def test_prepared_router_gemv_refuses_exact_registration():
    request = BenchmarkRequest(
        family="gemm",
        mode="prepared_fp32_router_gemv",
        parameters=_router_request().parameters,
        solution=None,
        registration="triton_prepared_fp32_router_gemv",
        cold_cache=True,
        seed=42,
    )
    with pytest.raises(BenchmarkCaseError, match="solution instead"):
        gemm_generator.prepare_prepared_fp32_router_gemv(request, None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU is required")
def test_prepared_router_gemv_runs_through_harness():
    result = KernelBenchmarkHarness(
        GraphTimer(
            GraphBenchmarkConfig(eager_warmup_iterations=1, replay_warmup_iterations=1)
        ),
        platform_provider=current_platform,
    ).run(_router_request(), measurement_blocks=3)

    assert result.status is BenchmarkStatus.SUCCESS, result.to_dict()
    spec = KernelRegistry.get().get_by_name(result.registration_name)
    assert (spec.family, spec.mode) == ("gemm", "prepared_fp32_router_gemv")
