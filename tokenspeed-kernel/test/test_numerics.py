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
from tokenspeed_kernel.numerics.comparison import compare_outputs, format_comparison
from tokenspeed_kernel.numerics.inputs import get_input_generator
from tokenspeed_kernel.numerics.tolerance import Tolerance
from tokenspeed_kernel.numerics.verify import (
    _verification_signature_and_reference,
    verify_kernel,
)
from tokenspeed_kernel.platform import Platform
from tokenspeed_kernel.registry import KernelRegistry, KernelSpec, load_builtin_kernels
from tokenspeed_kernel.signature import ScaleFormat, format_signatures

_fp8_dtype = Platform.get().fp8e4m3fn.dtype


class TestCompareOutputs:
    def test_treats_nan_as_mismatch(self) -> None:
        actual = torch.tensor([1.0, float("nan")], dtype=torch.float32)
        expected = torch.tensor([1.0, 1.0], dtype=torch.float32)

        result = compare_outputs(
            actual,
            expected,
            tolerance=Tolerance(atol=1e6, rtol=1e6),
        )

        assert not result.passed
        assert result.num_mismatches == 1

    def test_treats_inf_as_mismatch(self) -> None:
        actual = torch.tensor([float("inf"), 1.0], dtype=torch.float32)
        expected = torch.tensor([float("inf"), 1.0], dtype=torch.float32)

        result = compare_outputs(
            actual,
            expected,
            tolerance=Tolerance(atol=1e6, rtol=1e6),
        )

        assert not result.passed
        assert result.num_mismatches == 1


def test_gemm_input_generator_uses_signature_scale_metadata() -> None:
    scale = ScaleFormat(
        storage_dtype=torch.float32,
        granularity="block",
        block_shape=(128, 128),
    )
    signature = next(
        iter(format_signatures(("a", "b"), "mxfp8", {_fp8_dtype}, scale=scale))
    )
    generator = get_input_generator(
        "gemm",
        "mm",
        dtype=_fp8_dtype,
        traits={},
        format_signature=signature,
        device="cpu",
    )

    inputs = generator.generate(M=4, N=256, K=128)

    assert inputs["A"].dtype == _fp8_dtype
    assert inputs["B"].dtype == _fp8_dtype
    assert inputs["A_scales"].shape == (4, 1)
    assert inputs["B_scales"].shape == (2, 1)
    assert inputs["A_scales"].dtype == torch.float32
    assert inputs["B_scales"].dtype == torch.float32
    assert inputs["block_size"] == [128, 128]


def test_gemm_input_generator_requires_mxfp8_block_shape() -> None:
    scale = ScaleFormat(
        storage_dtype=torch.float32,
        granularity="block",
        dynamic_block_shape=True,
    )
    signature = next(
        iter(format_signatures(("a", "b"), "mxfp8", {_fp8_dtype}, scale=scale))
    )
    generator = get_input_generator(
        "gemm",
        "mm",
        dtype=_fp8_dtype,
        traits={},
        format_signature=signature,
        device="cpu",
    )

    with pytest.raises(ValueError, match="requires concrete block_shape"):
        generator.generate(M=4, N=256, K=128)


def test_verification_uses_signature_with_compatible_reference(fresh_registry) -> None:
    tensor_scale = ScaleFormat(storage_dtype=torch.float32, granularity="tensor")
    channel_scale = ScaleFormat(storage_dtype=torch.float32, granularity="channel")
    tensor_signature = next(
        iter(
            format_signatures(
                ("a", "b"), "scaled-fp8", {_fp8_dtype}, scale=tensor_scale
            )
        )
    )
    channel_signature = next(
        iter(
            format_signatures(
                ("a", "b"), "scaled-fp8", {_fp8_dtype}, scale=channel_scale
            )
        )
    )
    ref_spec = KernelSpec(
        name="test_tensor_scale_reference",
        family="gemm",
        mode="mm",
        solution="reference",
        format_signatures=frozenset({tensor_signature}),
        traits={"b_layout": frozenset({"KN"})},
    )
    test_spec = KernelSpec(
        name="test_fp8_scaled",
        family="gemm",
        mode="mm",
        solution="triton",
        format_signatures=frozenset({channel_signature, tensor_signature}),
        traits={"b_layout": frozenset({"KN"})},
    )
    registry = KernelRegistry.get()
    registry.register(ref_spec, lambda **_kwargs: None)
    registry.register(test_spec, lambda **_kwargs: None)

    signature, reference = _verification_signature_and_reference(
        registry, test_spec, _fp8_dtype
    )

    assert signature == tensor_signature
    assert reference is ref_spec


class TestNumericsVerification:
    def _get_verifiable_specs(
        dtype: torch.dtype, family: str | None = None
    ) -> list[KernelSpec]:
        load_builtin_kernels()
        registry = KernelRegistry.get()
        platform = Platform.get()
        specs: list[KernelSpec] = []
        for family_name, mode in registry.list_operators():
            if family and family_name != family:
                continue
            # Only run kernels that have a paired reference for this dtype;
            # otherwise verify_kernel raises ValueError and the test errors.
            op_specs = registry.get_for_operator(family_name, mode)
            dtype_specs = [
                s
                for s in op_specs
                if s.format_signature_for_primary_storage_dtype(dtype) is not None
            ]
            has_reference = any(s.solution == "reference" for s in dtype_specs)
            if not has_reference:
                continue
            for spec in dtype_specs:
                if spec.solution == "reference":
                    continue
                if spec.solution == "deep_gemm":
                    continue
                if not spec.capability.satisfied_by(platform):
                    continue
                specs.append(spec)
        specs.sort(key=lambda s: (s.family, s.mode, s.name))
        return specs

    def _verify(self, spec: KernelSpec, dtype: torch.dtype) -> None:
        if not torch.cuda.is_available():
            pytest.skip("CUDA is required for numerics verification")

        try:
            results = verify_kernel(spec.name, dtype=dtype, verbose=False)
        except Exception as exc:
            pytest.fail(
                f"Kernel {spec.name} raised an exception during verification: {exc}"
            )

        for i, result in enumerate(results):
            if not result.passed:
                pytest.fail(
                    f"Kernel {spec.name} failed numerics verification for shape set {i}:\n"
                    f"{format_comparison(result, kernel_name=spec.name)}"
                )

    @pytest.mark.parametrize(
        "spec",
        _get_verifiable_specs(_fp8_dtype, family="gemm"),
        ids=lambda s: f"{s.family}.{s.mode}:{s.name}",
    )
    def test_gemm_fp8(self, spec: KernelSpec):
        self._verify(spec, _fp8_dtype)

    @pytest.mark.parametrize(
        "spec",
        _get_verifiable_specs(torch.bfloat16, family="gemm"),
        ids=lambda s: f"{s.family}.{s.mode}:{s.name}",
    )
    def test_gemm_bf16(self, spec: KernelSpec):
        self._verify(spec, torch.bfloat16)

    @pytest.mark.parametrize(
        "spec",
        _get_verifiable_specs(torch.bfloat16, family="quantize"),
        ids=lambda s: f"{s.family}.{s.mode}:{s.name}",
    )
    def test_quantize_bf16(self, spec: KernelSpec):
        self._verify(spec, torch.bfloat16)

    @pytest.mark.parametrize(
        "spec",
        _get_verifiable_specs(torch.int32, family="moe"),
        ids=lambda s: f"{s.family}.{s.mode}:{s.name}",
    )
    def test_moe_int32(self, spec: KernelSpec):
        self._verify(spec, torch.int32)
