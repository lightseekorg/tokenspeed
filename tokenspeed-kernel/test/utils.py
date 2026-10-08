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

from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from typing import Any
from unittest.mock import patch

import torch
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    PlatformInfo,
    current_platform,
)
from tokenspeed_kernel.registry import KernelRegistry, register_kernel
from tokenspeed_kernel.signature import FormatSignature, format_signatures

SampleRegistration = tuple[dict, Callable]


def detected_platform() -> PlatformInfo | None:
    """Return the current platform, or ``None`` when none is usable.

    ``current_platform()`` raises when PyTorch sees no GPU or unsupported
    architecture that. Tests gate themselves at import time, so turn that into
    a value they can branch on instead of a collection error.
    """
    try:
        return current_platform()
    except RuntimeError:
        return None


def is_amd() -> bool:
    platform = detected_platform()
    return platform is not None and platform.is_amd


def is_nvidia() -> bool:
    platform = detected_platform()
    return platform is not None and platform.is_nvidia


def is_cdna4() -> bool:
    platform = detected_platform()
    return platform is not None and platform.is_cdna4


def is_cdna5() -> bool:
    platform = detected_platform()
    return platform is not None and platform.is_cdna5


def kernel_supported(name: str) -> bool:
    """Whether the registered kernel ``name`` can run on the detected device."""
    platform = detected_platform()
    spec = KernelRegistry.get().get_by_name(name)
    return (
        platform is not None
        and spec is not None
        and spec.capability.satisfied_by(platform)
    )


@contextmanager
def assert_no_triton_compile(*kernels: Any) -> Iterator[None]:
    """Fail if any Triton kernel compiles a new specialization in the block.

    Every ``tl.constexpr`` value is part of the compile-cache key, so a
    per-batch quantity passed as a constexpr recompiles the kernel on every new
    shape. Warm the kernels before entering, covering each integer
    specialization class Triton still keys on for runtime scalars (divisible by
    16 or not), then launch them with shapes that vary the way serving does. A
    serving-path kernel that opts out of those classes (``do_not_specialize``,
    ``do_not_specialize_on_alignment``) warms once and sweeps across them.
    """
    with ExitStack() as stack:
        compiles = [
            stack.enter_context(
                patch.object(kernel, "_do_compile", wraps=kernel._do_compile)
            )
            for kernel in kernels
        ]
        yield
    for kernel, compile_calls in zip(kernels, compiles, strict=True):
        assert compile_calls.call_count == 0, (
            f"{kernel.fn.__name__} compiled {compile_calls.call_count} new "
            "specialization(s); a per-batch value is likely passed as tl.constexpr"
        )


def compiled_kernels(kernel: Any) -> list[Any]:
    """Every binary this process has compiled for a Triton ``kernel``, from its JIT cache."""
    return [
        binary
        for cache in kernel.device_caches.values()
        for binary in cache[0].values()
    ]


def int_specialization_class(value: int) -> str:
    """The class Triton specializes a runtime integer on."""
    return "one" if value == 1 else "div16" if value % 16 == 0 else "other"


def warm_specialization_classes(run, key, sweep, pool) -> None:
    """Run one pool value per specialization key the sweep will hit.

    ``key`` maps a value to everything that selects a binary (integer classes,
    power-of-two buckets, launch configs). Which keys a sweep reaches can
    depend on the device, e.g. split counts follow the SM count; warming from
    a pool keeps the guard meaningful everywhere. A key no pool value reaches
    is warmed with its sweep value.
    """
    needed = {key(value) for value in sweep}
    for value in [*(v for v in pool if v not in sweep), *sweep]:
        if key(value) in needed:
            needed.discard(key(value))
            run(value)


def make_mxfp4_moe_weights(
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    generator: torch.Generator,
    *,
    device: str = "cuda",
    scale_range: tuple[int, int] = (120, 121),
) -> dict[str, torch.Tensor]:
    def scales(*shape: int) -> torch.Tensor:
        return torch.randint(
            *scale_range,
            shape,
            dtype=torch.uint8,
            device=device,
            generator=generator,
        )

    return {
        "w13_weight": torch.randint(
            0,
            256,
            (num_experts, 2 * intermediate_size, hidden_size // 2),
            dtype=torch.uint8,
            device=device,
            generator=generator,
        ),
        "w13_scale": scales(num_experts, 2 * intermediate_size, hidden_size // 32),
        "w2_weight": torch.randint(
            0,
            256,
            (num_experts, hidden_size, intermediate_size // 2),
            dtype=torch.uint8,
            device=device,
            generator=generator,
        ),
        "w2_scale": scales(num_experts, hidden_size, intermediate_size // 32),
    }


def make_fp8_per_channel_gemm_operands(m: int, n: int, k: int, seed: int):
    """Per-token FP8 activations and per-channel FP8 weights for ``A @ B.T``.

    Returns ``(a, a_scales, b, b_scales)``: ``a`` is ``[m, k]`` E4M3 with FP32
    ``[m, 1]`` scales and ``b`` is ``[n, k]`` E4M3 with FP32 ``[n, 1]`` scales.
    The weights are scaled so outputs have roughly unit variance.
    """
    from tokenspeed_kernel.ops.gemm.fp8_utils import per_token_group_quant_fp8

    generator = torch.Generator(device="cuda").manual_seed(seed)
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, generator=generator)
    b = torch.randn(n, k, device="cuda", generator=generator) / k**0.5
    b_scales = b.abs().amax(dim=1, keepdim=True) / 448.0
    b_fp8 = (b / b_scales).to(torch.float8_e4m3fn)
    # One quantization group spanning the row is per-token scaling.
    a_fp8, a_scales = per_token_group_quant_fp8(a, k)
    return a_fp8, a_scales, b_fp8, b_scales


def make_round_robin_topk(
    num_tokens: int,
    num_experts: int,
    top_k: int,
    *,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    ids = (
        torch.arange(num_tokens, device=device)[:, None]
        + torch.arange(top_k, device=device)
    ) % num_experts
    weights = torch.arange(1, top_k + 1, device=device, dtype=torch.float32)
    return (
        (weights / weights.sum()).expand(num_tokens, -1).contiguous(),
        ids.to(torch.int32),
    )


def dummy_impl(name: str) -> Callable:
    def impl(*args, **kwargs):
        return name

    impl.__name__ = name
    return impl


def _sample_registration(
    name: str,
    family: str,
    mode: str,
    solution: str,
    signatures: frozenset[FormatSignature],
    *,
    features: frozenset[str] | None = None,
    capability: CapabilityRequirement | None = None,
    priority: int = 10,
) -> SampleRegistration:
    return (
        {
            "family": family,
            "mode": mode,
            "name": name,
            "solution": solution,
            "features": features,
            "capability": capability,
            "signatures": signatures,
            "priority": priority,
        },
        dummy_impl(name),
    )


def make_sample_specs() -> dict[str, SampleRegistration]:
    return {
        "flashinfer_decode": _sample_registration(
            "flashinfer_decode",
            "attention",
            "decode",
            "flashinfer",
            format_signatures(
                ("q", "k_cache", "v_cache"), "dense", {torch.float16, torch.bfloat16}
            ),
            features=frozenset({"paged"}),
            capability=CapabilityRequirement(
                vendors=frozenset({"nvidia"}),
                min_arch_version=ArchVersion(8, 0),
            ),
            priority=18,
        ),
        "triton_decode": _sample_registration(
            "triton_decode",
            "attention",
            "decode",
            "triton",
            format_signatures(
                ("q", "k_cache", "v_cache"), "dense", {torch.float16, torch.bfloat16}
            ),
            features=frozenset({"paged"}),
            priority=10,
        ),
        "cutlass_prefill": _sample_registration(
            "cutlass_prefill",
            "attention",
            "prefill",
            "cutlass",
            format_signatures(
                ("q", "k", "v"), "dense", {torch.float16, torch.bfloat16}
            ),
            capability=CapabilityRequirement(
                vendors=frozenset({"nvidia"}),
                min_arch_version=ArchVersion(9, 0),
            ),
            priority=16,
        ),
        "reference_decode": _sample_registration(
            "reference_decode",
            "attention",
            "decode",
            "reference",
            format_signatures(
                ("q", "k_cache", "v_cache"),
                "dense",
                {torch.float16, torch.bfloat16, torch.float32},
            ),
            features=frozenset({"paged"}),
            capability=CapabilityRequirement(),
            priority=10,
        ),
        "aiter_decode": _sample_registration(
            "aiter_decode",
            "attention",
            "decode",
            "aiter",
            format_signatures(
                ("q", "k_cache", "v_cache"), "dense", {torch.float16, torch.bfloat16}
            ),
            features=frozenset({"paged"}),
            capability=CapabilityRequirement(vendors=frozenset({"amd"})),
            priority=16,
        ),
        "cutlass_gemm": _sample_registration(
            "cutlass_gemm",
            "gemm",
            "mm",
            "cutlass",
            format_signatures(("a", "b"), "dense", {torch.float16, torch.bfloat16}),
            capability=CapabilityRequirement(
                vendors=frozenset({"nvidia"}),
                min_arch_version=ArchVersion(8, 0),
            ),
            priority=15,
        ),
        "triton_gemm": _sample_registration(
            "triton_gemm",
            "gemm",
            "mm",
            "triton",
            format_signatures(("a", "b"), "dense", {torch.float16, torch.bfloat16}),
            priority=10,
        ),
        "cutlass_grouped_gemm": _sample_registration(
            "cutlass_grouped_gemm",
            "gemm",
            "grouped_mm",
            "cutlass",
            format_signatures(("a", "b"), "dense", {torch.float16, torch.bfloat16}),
            capability=CapabilityRequirement(
                vendors=frozenset({"nvidia"}),
                min_arch_version=ArchVersion(9, 0),
            ),
            priority=16,
        ),
        "triton_grouped_gemm": _sample_registration(
            "triton_grouped_gemm",
            "gemm",
            "grouped_mm",
            "triton",
            format_signatures(("a", "b"), "dense", {torch.float16, torch.bfloat16}),
            priority=10,
        ),
        "triton_fused_moe": _sample_registration(
            "triton_fused_moe",
            "moe",
            "fused",
            "triton",
            format_signatures(
                ("x", "weight"), "dense", {torch.float16, torch.bfloat16}
            ),
            priority=12,
        ),
        "cutlass_fused_moe": _sample_registration(
            "cutlass_fused_moe",
            "moe",
            "fused",
            "cutlass",
            format_signatures(
                ("x", "weight"), "dense", {torch.float16, torch.bfloat16}
            ),
            capability=CapabilityRequirement(
                vendors=frozenset({"nvidia"}),
                min_arch_version=ArchVersion(9, 0),
            ),
            priority=15,
        ),
        "triton_modular_moe": _sample_registration(
            "triton_modular_moe",
            "moe",
            "modular",
            "triton",
            format_signatures("x", "dense", {torch.float16, torch.bfloat16}),
            priority=10,
        ),
        "cutlass_modular_moe": _sample_registration(
            "cutlass_modular_moe",
            "moe",
            "modular",
            "cutlass",
            format_signatures("x", "dense", {torch.float16, torch.bfloat16}),
            capability=CapabilityRequirement(
                vendors=frozenset({"nvidia"}),
                min_arch_version=ArchVersion(8, 0),
            ),
            priority=14,
        ),
    }


def register_all_samples(
    registry: KernelRegistry, samples: dict[str, SampleRegistration]
) -> None:
    if registry is not KernelRegistry.get():
        raise ValueError("sample registrations must target the active KernelRegistry")
    for options, impl in samples.values():
        register_kernel(**options)(impl)


class FakeTimer:
    """Benchmark timer that invokes the operation once and reports fixed samples."""

    def __init__(self) -> None:
        self.calls = 0
        self.cold_cache: list[bool] = []
        self.measurement_blocks: list[int] = []

    def measure(self, prepared, *, cold_cache: bool, measurement_blocks: int):
        # Imported lazily: conftest imports this module for every test.
        from tokenspeed_kernel.benchmark.graph import GraphMeasurement

        self.calls += 1
        self.cold_cache.append(cold_cache)
        self.measurement_blocks.append(measurement_blocks)
        prepared.invoke()
        return GraphMeasurement(
            samples_us=(2.0, 3.0, 4.0),
            median_us=3.0,
            p90_us=3.8,
            min_us=2.0,
            max_us=4.0,
            relative_mad=1.0 / 3.0,
            eager_warmup_iterations=5,
            replay_warmup_iterations=3,
            warmup_time_ms=1.0,
            capture_time_ms=2.0,
            first_replay_time_ms=3.0,
            measurement_time_ms=4.0,
        )
