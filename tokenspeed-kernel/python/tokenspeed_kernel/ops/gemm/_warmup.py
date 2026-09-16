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

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from tokenspeed_kernel.ops.gemm import (
    _select_nvfp4_swiglu_quant,
    nvfp4_gemm_swiglu_nvfp4_quant,
)
from tokenspeed_kernel.platform import ArchVersion, PlatformInfo
from tokenspeed_kernel.registry import KernelRegistry, KernelSpec

__all__ = ["Nvfp4SwigluQuantWarmupConfig"]


def _require_fields(
    raw: Mapping[str, object], required: frozenset[str], context: str
) -> None:
    keys = frozenset(raw)
    missing = sorted(required - keys)
    unknown = sorted(keys - required)
    if missing:
        raise ValueError(f"{context} is missing fields: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{context} has unknown fields: {', '.join(unknown)}")


def _require_int(raw: Mapping[str, object], field: str, context: str) -> int:
    value = raw[field]
    if type(value) is not int:
        raise TypeError(f"{context}.{field} must be an integer")
    return value


def _require_bool(raw: Mapping[str, object], field: str, context: str) -> bool:
    value = raw[field]
    if type(value) is not bool:
        raise TypeError(f"{context}.{field} must be a boolean")
    return value


def _require_str(raw: Mapping[str, object], field: str, context: str) -> str:
    value = raw[field]
    if not isinstance(value, str) or not value:
        raise TypeError(f"{context}.{field} must be a non-empty string")
    return value


def _round_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


@dataclass(frozen=True)
class _Nvfp4SwigluQuantWarmupCase:
    expected_registration: str | None
    maximum_num_tokens: int
    n: int
    k: int
    ab_dtype: str
    sf_dtype: str
    c_dtype: str
    sf_vec_size: int
    use_prefetch: bool
    prefetch_dist: int
    vectorized_f32: bool
    enable_pdl: bool
    random_seed: int

    @classmethod
    def from_json(
        cls, raw: Mapping[str, object], case_index: int
    ) -> _Nvfp4SwigluQuantWarmupCase:
        context = f"gemm.nvfp4_swiglu_quant case {case_index}"
        _require_fields(
            raw,
            frozenset(
                {
                    "expected_registration",
                    "maximum_num_tokens",
                    "n",
                    "k",
                    "ab_dtype",
                    "sf_dtype",
                    "c_dtype",
                    "sf_vec_size",
                    "use_prefetch",
                    "prefetch_dist",
                    "vectorized_f32",
                    "enable_pdl",
                    "random_seed",
                }
            ),
            context,
        )
        expected_registration = raw["expected_registration"]
        if expected_registration is not None and (
            not isinstance(expected_registration, str) or not expected_registration
        ):
            raise TypeError(
                f"{context}.expected_registration must be a non-empty string or null"
            )
        maximum_num_tokens = _require_int(raw, "maximum_num_tokens", context)
        n = _require_int(raw, "n", context)
        k = _require_int(raw, "k", context)
        sf_vec_size = _require_int(raw, "sf_vec_size", context)
        prefetch_dist = _require_int(raw, "prefetch_dist", context)
        random_seed = _require_int(raw, "random_seed", context)
        if maximum_num_tokens <= 0 or n <= 0 or k <= 0 or sf_vec_size <= 0:
            raise ValueError(f"{context} dimensions must be positive")
        if n % (2 * sf_vec_size):
            raise ValueError(
                f"{context}.n must be divisible by 2 * sf_vec_size, got {n}"
            )
        if k % 16:
            raise ValueError(f"{context}.k must be divisible by 16, got {k}")
        if sf_vec_size != 16:
            raise ValueError(f"{context}.sf_vec_size must be 16")
        if prefetch_dist < 0:
            raise ValueError(f"{context}.prefetch_dist cannot be negative")
        if random_seed < 0:
            raise ValueError(f"{context}.random_seed cannot be negative")
        ab_dtype = _require_str(raw, "ab_dtype", context)
        sf_dtype = _require_str(raw, "sf_dtype", context)
        c_dtype = _require_str(raw, "c_dtype", context)
        if ab_dtype != "float4_e2m1fn" or c_dtype != "float4_e2m1fn":
            raise ValueError(f"{context} requires float4_e2m1fn input and output")
        if sf_dtype != "float8_e4m3fn":
            raise ValueError(f"{context} requires float8_e4m3fn scale factors")
        return cls(
            expected_registration=expected_registration,
            maximum_num_tokens=maximum_num_tokens,
            n=n,
            k=k,
            ab_dtype=ab_dtype,
            sf_dtype=sf_dtype,
            c_dtype=c_dtype,
            sf_vec_size=sf_vec_size,
            use_prefetch=_require_bool(raw, "use_prefetch", context),
            prefetch_dist=prefetch_dist,
            vectorized_f32=_require_bool(raw, "vectorized_f32", context),
            enable_pdl=_require_bool(raw, "enable_pdl", context),
            random_seed=random_seed,
        )


@dataclass(frozen=True)
class _Nvfp4SwigluQuantInvocation:
    a: torch.Tensor
    a_scale: torch.Tensor
    b: torch.Tensor
    b_scale: torch.Tensor
    alpha: torch.Tensor
    output_global_scale: torch.Tensor
    ab_dtype: str
    sf_dtype: str
    c_dtype: str
    sf_vec_size: int
    use_prefetch: bool
    prefetch_dist: int
    vectorized_f32: bool
    enable_pdl: bool
    solution: str

    def run(self) -> None:
        nvfp4_gemm_swiglu_nvfp4_quant(
            a=self.a,
            a_scale=self.a_scale,
            b=self.b,
            b_scale=self.b_scale,
            alpha=self.alpha,
            output_global_scale=self.output_global_scale,
            out=None,
            out_scale=None,
            ab_dtype=self.ab_dtype,
            sf_dtype=self.sf_dtype,
            c_dtype=self.c_dtype,
            sf_vec_size=self.sf_vec_size,
            use_prefetch=self.use_prefetch,
            prefetch_dist=self.prefetch_dist,
            vectorized_f32=self.vectorized_f32,
            enable_pdl=self.enable_pdl,
            solution=self.solution,
        )


@dataclass(frozen=True)
class _PreparedNvfp4SwigluQuantWarmup:
    _registration: KernelSpec
    invocations: tuple[_Nvfp4SwigluQuantInvocation, ...]

    @property
    def registration(self) -> KernelSpec:
        return self._registration

    def run(self) -> None:
        for invocation in self.invocations:
            invocation.run()


@dataclass(frozen=True)
class Nvfp4SwigluQuantWarmupConfig:
    version: int
    cases: tuple[_Nvfp4SwigluQuantWarmupCase, ...]

    @classmethod
    def from_json(cls, raw: Mapping[str, object]) -> Nvfp4SwigluQuantWarmupConfig:
        _require_fields(raw, frozenset({"version", "cases"}), "warmup definition")
        version = _require_int(raw, "version", "warmup definition")
        if version != 1:
            raise ValueError(
                "gemm.nvfp4_swiglu_quant warmup definition version must be 1"
            )
        raw_cases = raw["cases"]
        if not isinstance(raw_cases, list) or not raw_cases:
            raise TypeError("warmup definition.cases must be a non-empty list")
        cases = []
        for index, raw_case in enumerate(raw_cases):
            if not isinstance(raw_case, Mapping):
                raise TypeError(
                    f"gemm.nvfp4_swiglu_quant case {index} must be an object"
                )
            cases.append(_Nvfp4SwigluQuantWarmupCase.from_json(raw_case, index))
        return cls(version=version, cases=tuple(cases))

    def prepare(
        self, solution: str, platform: PlatformInfo
    ) -> _PreparedNvfp4SwigluQuantWarmup:
        if platform.vendor != "nvidia" or not (
            ArchVersion(10, 0) <= platform.arch_version <= ArchVersion(10, 3)
        ):
            raise ValueError(
                "gemm.nvfp4_swiglu_quant warmup requires NVIDIA SM100 or SM103"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("gemm.nvfp4_swiglu_quant warmup requires CUDA")

        device = torch.device("cuda", torch.cuda.current_device())
        invocations = []
        registration = None
        for index, case in enumerate(self.cases):
            kernel = _select_nvfp4_swiglu_quant(
                a_dtype=torch.uint8,
                a_scale_dtype=torch.float8_e4m3fn,
                b_dtype=torch.uint8,
                b_scale_dtype=torch.float8_e4m3fn,
                sf_vec_size=case.sf_vec_size,
                solution=solution,
            )
            spec = KernelRegistry.get().get_by_name(kernel.name)
            if spec is None:
                raise RuntimeError(f"selected kernel {kernel.name!r} is not registered")
            if case.expected_registration is not None and (
                spec.name != case.expected_registration
            ):
                raise ValueError(
                    f"gemm.nvfp4_swiglu_quant case {index} selected {spec.name!r}, "
                    f"expected {case.expected_registration!r}"
                )
            if registration is not None and registration.name != spec.name:
                raise ValueError(
                    "one gemm.nvfp4_swiglu_quant definition cannot resolve to "
                    "multiple registrations"
                )
            registration = spec

            generator = torch.Generator(device=device).manual_seed(case.random_seed)
            a = torch.randint(
                0,
                256,
                (case.maximum_num_tokens, case.k // 2),
                dtype=torch.uint8,
                device=device,
                generator=generator,
            )
            a_scale = torch.randint(
                1,
                127,
                (
                    _round_up(case.maximum_num_tokens, 128),
                    _round_up(case.k // case.sf_vec_size, 4),
                ),
                dtype=torch.uint8,
                device=device,
                generator=generator,
            ).view(torch.float8_e4m3fn)
            b = torch.randint(
                0,
                256,
                (case.n, case.k // 2),
                dtype=torch.uint8,
                device=device,
                generator=generator,
            )
            b_scale = torch.randint(
                1,
                127,
                (
                    _round_up(case.n, 128),
                    _round_up(case.k // case.sf_vec_size, 4),
                ),
                dtype=torch.uint8,
                device=device,
                generator=generator,
            ).view(torch.float8_e4m3fn)
            invocations.append(
                _Nvfp4SwigluQuantInvocation(
                    a=a,
                    a_scale=a_scale,
                    b=b,
                    b_scale=b_scale,
                    alpha=torch.ones((1, 1), dtype=torch.float32, device=device),
                    output_global_scale=torch.ones(
                        (1,), dtype=torch.float32, device=device
                    ),
                    ab_dtype=case.ab_dtype,
                    sf_dtype=case.sf_dtype,
                    c_dtype=case.c_dtype,
                    sf_vec_size=case.sf_vec_size,
                    use_prefetch=case.use_prefetch,
                    prefetch_dist=case.prefetch_dist,
                    vectorized_f32=case.vectorized_f32,
                    enable_pdl=case.enable_pdl,
                    solution=solution,
                )
            )

        if registration is None:
            raise RuntimeError("warmup definition has no cases")
        return _PreparedNvfp4SwigluQuantWarmup(
            _registration=registration,
            invocations=tuple(invocations),
        )
