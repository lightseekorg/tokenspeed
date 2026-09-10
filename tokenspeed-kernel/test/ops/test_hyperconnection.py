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
from tokenspeed_kernel import (
    gated_residual_combine,
    gated_residual_combine_norm,
    gated_residual_mix,
    grouped_gemma_rmsnorm,
    prepare_gated_residual_weight_cache,
)
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.platform import current_platform, pdl_enabled
from tokenspeed_kernel.profiling import ShapeCapture
from tokenspeed_kernel.registry import KernelRegistry

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="a CUDA/ROCm device is required"
)

HC_COUNT = 4
HIDDEN_SIZE = 2560
LOWRANK = 320
WIDE = HC_COUNT * HIDDEN_SIZE


@triton.jit
def _delayed_combine_projection(
    source_block,
    source_inject,
    block_output,
    inject_output,
    H: tl.constexpr,
    G: tl.constexpr,
    BLOCK: tl.constexpr,
    CYCLES: tl.constexpr,
):
    tl.extra.cuda.gdc_launch_dependents()
    tl.extra.cuda.gdc_wait()
    start = tl.inline_asm_elementwise(
        "mov.u64 $0, %clock64;",
        constraints="=l",
        args=[],
        dtype=tl.uint64,
        is_pure=False,
        pack=1,
    )
    now = start
    while now - start < CYCLES:
        now = tl.inline_asm_elementwise(
            "mov.u64 $0, %clock64;",
            constraints="=l",
            args=[],
            dtype=tl.uint64,
            is_pure=False,
            pack=1,
        )
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    value = tl.load(source_block + row * H + offsets, mask=offsets < H, other=0)
    gate = tl.load(source_inject + row * G + offsets, mask=offsets < G, other=0)
    tl.store(block_output + row * H + offsets, value, mask=offsets < H)
    tl.store(inject_output + row * G + offsets, gate, mask=offsets < G)


def _inputs(
    rows: int, dtype: torch.dtype, *, seed: int = 17
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    normalized = torch.randn(
        rows, WIDE, dtype=dtype, device="cuda", generator=generator
    )
    projection = (
        torch.randn(
            LOWRANK + HC_COUNT,
            WIDE,
            dtype=dtype,
            device="cuda",
            generator=generator,
        )
        * 0.01
    )
    up = (
        torch.randn(
            WIDE,
            LOWRANK,
            dtype=dtype,
            device="cuda",
            generator=generator,
        )
        * 0.01
    )
    return normalized, projection, up


def _mix_reference(
    normalized: torch.Tensor,
    projection: torch.Tensor,
    up: torch.Tensor,
    projection_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    # The reference runs on the host. Otherwise the two float64 GEMMs trigger
    # high execution time on simulator-based flows.
    device = normalized.device
    x = normalized.cpu().double()
    projected = x @ projection.cpu().double().T
    down = torch.nn.functional.silu(projected[:, :LOWRANK] * projection_scale)
    gate = down @ up.cpu().double().T
    mixed = (
        torch.sigmoid(gate).unflatten(-1, (HC_COUNT, HIDDEN_SIZE))
        * x.unflatten(-1, (HC_COUNT, HIDDEN_SIZE))
    ).mean(dim=-2)
    inject = projected[:, LOWRANK:] * projection_scale
    return (
        mixed.to(device=device, dtype=normalized.dtype),
        inject.to(device=device, dtype=normalized.dtype),
    )


@pytest.mark.parametrize("rows", [0, 1, 4, 8, 16, 24, 32, 128])
def test_general_triton_mix_matches_fp64_reference(rows: int) -> None:
    normalized, projection, up = _inputs(rows, torch.bfloat16)
    actual, actual_inject = gated_residual_mix(
        normalized,
        projection,
        up,
        HC_COUNT,
        HIDDEN_SIZE,
        LOWRANK,
        override="triton_hyperconnection_mix",
    )
    assert actual.shape == (rows, HIDDEN_SIZE)
    assert actual_inject.shape == (rows, HC_COUNT)
    if rows == 0:
        return
    expected, expected_inject = _mix_reference(normalized, projection, up, 1.0)
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(actual_inject, expected_inject, rtol=3e-2, atol=3e-2)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("rows", [1, 8, 16])
def test_persistent_mix_matches_fp64_reference(rows: int, dtype: torch.dtype) -> None:
    if not current_platform().is_nvidia:
        pytest.skip("the persistent grid barrier is NVIDIA-only")
    normalized, projection, up = _inputs(rows, dtype)
    actual, actual_inject = gated_residual_mix(
        normalized,
        projection,
        up,
        HC_COUNT,
        HIDDEN_SIZE,
        LOWRANK,
        override="triton_persistent_hyperconnection_mix",
    )
    expected, expected_inject = _mix_reference(normalized, projection, up, 1.0)
    tolerance = 4e-2 if dtype is torch.bfloat16 else 8e-3
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(
        actual_inject, expected_inject, rtol=tolerance, atol=tolerance
    )


@pytest.mark.parametrize("rows", [1, 8])
def test_cute_dsl_mix_matches_fp64_reference(rows: int) -> None:
    if not current_platform().is_blackwell:
        pytest.skip("the CuTeDSL HC specialization is Blackwell-only")
    if KernelRegistry.get().get_by_name("cute_dsl_hyperconnection_mix") is None:
        pytest.skip("CuTeDSL dependencies are unavailable")
    normalized, projection, up = _inputs(rows, torch.bfloat16)
    assert prepare_gated_residual_weight_cache(up, LOWRANK)
    actual, actual_inject = gated_residual_mix(
        normalized,
        projection,
        up,
        HC_COUNT,
        HIDDEN_SIZE,
        LOWRANK,
        override="cute_dsl_hyperconnection_mix",
    )
    expected, expected_inject = _mix_reference(normalized, projection, up, 1.0)
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(actual_inject, expected_inject, rtol=3e-2, atol=3e-2)


def test_cute_dsl_graph_replay_observes_reloaded_up_weight() -> None:
    if not current_platform().is_blackwell:
        pytest.skip("the CuTeDSL HC specialization is Blackwell-only")
    if KernelRegistry.get().get_by_name("cute_dsl_hyperconnection_mix") is None:
        pytest.skip("CuTeDSL dependencies are unavailable")
    normalized, projection, up = _inputs(1, torch.bfloat16, seed=23)
    up.zero_()
    assert prepare_gated_residual_weight_cache(up, LOWRANK)

    def mix() -> tuple[torch.Tensor, torch.Tensor | None]:
        return gated_residual_mix(
            normalized,
            projection,
            up,
            HC_COUNT,
            HIDDEN_SIZE,
            LOWRANK,
            override="cute_dsl_hyperconnection_mix",
        )

    before, _ = mix()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual, actual_inject = mix()

    generator = torch.Generator(device="cuda").manual_seed(29)
    up.data.copy_(
        torch.randn(up.shape, dtype=up.dtype, device=up.device, generator=generator)
        * 0.1
    )
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, before, rtol=3e-2, atol=3e-2)

    assert prepare_gated_residual_weight_cache(up, LOWRANK)
    graph.replay()
    torch.cuda.synchronize()

    expected, expected_inject = _mix_reference(normalized, projection, up, 1.0)
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(actual_inject, expected_inject, rtol=3e-2, atol=3e-2)
    assert not torch.allclose(actual, before, rtol=3e-2, atol=3e-2)


def test_cute_dsl_mix_rejects_unprepared_up_weight() -> None:
    if not current_platform().is_blackwell:
        pytest.skip("the CuTeDSL HC specialization is Blackwell-only")
    if KernelRegistry.get().get_by_name("cute_dsl_hyperconnection_mix") is None:
        pytest.skip("CuTeDSL dependencies are unavailable")
    normalized, projection, up = _inputs(1, torch.bfloat16, seed=31)

    with pytest.raises(RuntimeError, match="was not prepared"):
        gated_residual_mix(
            normalized,
            projection,
            up,
            HC_COUNT,
            HIDDEN_SIZE,
            LOWRANK,
            override="cute_dsl_hyperconnection_mix",
        )


def test_cute_dsl_weight_preparation_rejects_capture(monkeypatch) -> None:
    if not current_platform().is_blackwell:
        pytest.skip("the CuTeDSL HC specialization is Blackwell-only")
    if KernelRegistry.get().get_by_name("cute_dsl_hyperconnection_mix") is None:
        pytest.skip("CuTeDSL dependencies are unavailable")
    _, _, up = _inputs(1, torch.bfloat16, seed=37)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    with pytest.raises(RuntimeError, match="outside CUDA Graph capture"):
        prepare_gated_residual_weight_cache(up, LOWRANK)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_combine_accepts_reduce_scatter_row_slices(dtype: torch.dtype) -> None:
    generator = torch.Generator(device="cuda").manual_seed(29)
    block_base = torch.randn(
        19, HIDDEN_SIZE, dtype=dtype, device="cuda", generator=generator
    )
    residual_base = torch.randn(
        19, WIDE, dtype=dtype, device="cuda", generator=generator
    )
    inject_base = torch.randn(
        19, HC_COUNT, dtype=dtype, device="cuda", generator=generator
    )
    block = block_base[3:12]
    residual = residual_base[3:12]
    inject = inject_base[3:12]
    actual = gated_residual_combine(block, residual, inject, HC_COUNT, HIDDEN_SIZE)
    expected = (
        residual.unflatten(-1, (HC_COUNT, HIDDEN_SIZE))
        + block.unsqueeze(-2) * (2 * torch.sigmoid(inject)).unsqueeze(-1)
    ).flatten(-2)
    tolerance = 3e-2 if dtype is torch.bfloat16 else 5e-3
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("rows", [0, 1, 16, 128])
def test_grouped_gemma_rmsnorm_production_shape(rows: int, dtype: torch.dtype) -> None:
    generator = torch.Generator(device="cuda").manual_seed(41)
    x = torch.randn(rows, WIDE, dtype=dtype, device="cuda", generator=generator)
    weight = torch.randn(WIDE, dtype=dtype, device="cuda", generator=generator) * 0.02
    actual = grouped_gemma_rmsnorm(x, weight, HIDDEN_SIZE, 1e-6)
    grouped = x.float().unflatten(-1, (HC_COUNT, HIDDEN_SIZE))
    expected = (
        grouped * torch.rsqrt(grouped.square().mean(dim=-1, keepdim=True) + 1e-6)
    ).flatten(-2) * (1.0 + weight.float())
    tolerance = 2e-2 if dtype is torch.bfloat16 else 3e-3
    torch.testing.assert_close(
        actual, expected.to(dtype), rtol=tolerance, atol=tolerance
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("shared_weight", [False, True])
@pytest.mark.parametrize("preload_residual", [False, True])
@pytest.mark.parametrize(
    "rows,hc_count,hidden_size",
    [(0, 4, 2560), (1, 4, 2560), (17, 4, 2560), (128, 4, 2560), (3, 3, 13)],
)
def test_combine_norm_matches_separate_kernels(
    dtype: torch.dtype,
    shared_weight: bool,
    preload_residual: bool,
    rows: int,
    hc_count: int,
    hidden_size: int,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(43)
    wide = hc_count * hidden_size
    # Nonzero row offsets exercise reduce-scatter views; strided columns also
    # exercise the fused entry point's contiguous input conversion.
    residual = torch.randn(
        rows + 2, wide * 2, dtype=dtype, device="cuda", generator=generator
    )[1 : rows + 1, ::2]
    block = torch.randn(
        rows + 2, hidden_size * 2, dtype=dtype, device="cuda", generator=generator
    )[1 : rows + 1, ::2]
    inject = torch.randn(
        rows + 2, hc_count * 2, dtype=dtype, device="cuda", generator=generator
    )[1 : rows + 1, ::2]
    weight = (
        torch.randn(
            hidden_size if shared_weight else wide,
            dtype=dtype,
            device="cuda",
            generator=generator,
        )
        * 0.02
    )
    original = residual.clone()
    if preload_residual:
        residual = residual.contiguous()
        block = block.contiguous()
        inject = inject.contiguous()
    reference = gated_residual_combine(
        block.contiguous(),
        residual.contiguous(),
        inject.contiguous(),
        hc_count,
        hidden_size,
        override=None,
        solution=None,
    )
    reference_weight = weight.repeat(hc_count) if shared_weight else weight
    reference_norm = grouped_gemma_rmsnorm(
        reference, reference_weight, hidden_size, 1e-6, out=None
    )

    combined, normalized = gated_residual_combine_norm(
        block,
        residual,
        inject,
        weight,
        hc_count,
        hidden_size,
        1e-6,
        preload_residual=preload_residual,
    )

    torch.testing.assert_close(combined, reference, rtol=0, atol=0)
    torch.testing.assert_close(normalized, reference_norm, rtol=0, atol=0)
    torch.testing.assert_close(residual, original, rtol=0, atol=0)
    assert combined.is_contiguous() and normalized.is_contiguous()


@pytest.mark.parametrize("rows", [0, 3])
@pytest.mark.parametrize("field", ["block_output", "inject_logits", "weight"])
@pytest.mark.parametrize("invalid", ["shape", "dtype", "device"])
def test_combine_norm_validates_inputs(rows: int, field: str, invalid: str) -> None:
    shapes = {"block_output": (rows, 8), "inject_logits": (rows, 4), "weight": (32,)}
    args = {
        name: torch.empty(shape, dtype=torch.bfloat16, device="cuda")
        for name, shape in shapes.items()
    }
    shape = (
        (*shapes[field][:-1], shapes[field][-1] + 1)
        if invalid == "shape"
        else shapes[field]
    )
    args[field] = torch.empty(
        shape,
        dtype=torch.float32 if invalid == "dtype" else torch.bfloat16,
        device="cpu" if invalid == "device" else "cuda",
    )
    residual = torch.empty(rows, 32, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match=field):
        gated_residual_combine_norm(
            residual=residual,
            hc_count=4,
            hidden_size=8,
            eps=1e-6,
            preload_residual=False,
            **args,
        )


@pytest.mark.parametrize("enable_pdl", [False, True])
@pytest.mark.parametrize("preload_residual", [False, True])
def test_combine_norm_cuda_graph_replays_changed_inputs(
    enable_pdl: bool, preload_residual: bool
) -> None:
    if enable_pdl and not (
        current_platform().is_nvidia and current_platform().is_hopper_plus
    ):
        pytest.skip("PDL requires NVIDIA Hopper or newer")
    generator = torch.Generator(device="cuda").manual_seed(47)
    residual = torch.randn(
        8, WIDE, dtype=torch.bfloat16, device="cuda", generator=generator
    )
    source_block = torch.randn(
        8, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda", generator=generator
    )
    source_inject = torch.randn(
        8, HC_COUNT, dtype=torch.bfloat16, device="cuda", generator=generator
    )
    block, inject = torch.empty_like(source_block), torch.empty_like(source_inject)
    weight = (
        torch.randn(WIDE, dtype=torch.bfloat16, device="cuda", generator=generator)
        * 0.02
    )

    def chain() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if enable_pdl:
            _delayed_combine_projection[(8,)](
                source_block,
                source_inject,
                block,
                inject,
                H=HIDDEN_SIZE,
                G=HC_COUNT,
                BLOCK=4096,
                CYCLES=100000,
                launch_pdl=True,
            )
        else:
            block.copy_(source_block)
            inject.copy_(source_inject)
        combined, normalized = gated_residual_combine_norm(
            block,
            residual,
            inject,
            weight,
            HC_COUNT,
            HIDDEN_SIZE,
            1e-6,
            preload_residual=preload_residual,
        )
        consumed = grouped_gemma_rmsnorm(
            normalized, weight, HIDDEN_SIZE, 1e-6, out=None
        )
        return combined, normalized, consumed

    previous_pdl = pdl_enabled()
    try:
        pdl_enabled(enable_pdl)
        chain()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs = chain()
        for _ in range(3):
            residual.normal_(generator=generator)
            source_block.normal_(generator=generator)
            source_inject.normal_(generator=generator)
            # Reading either activation before the producer completes sees NaNs.
            block.fill_(float("nan"))
            inject.fill_(float("nan"))
            graph.replay()
            reference = gated_residual_combine(
                source_block,
                residual,
                source_inject,
                HC_COUNT,
                HIDDEN_SIZE,
                override=None,
                solution=None,
            )
            reference_norm = grouped_gemma_rmsnorm(
                reference, weight, HIDDEN_SIZE, 1e-6, out=None
            )
            reference_consumed = grouped_gemma_rmsnorm(
                reference_norm, weight, HIDDEN_SIZE, 1e-6, out=None
            )
            for actual, expected in zip(
                outputs, (reference, reference_norm, reference_consumed), strict=True
            ):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        torch.cuda.synchronize()
        pdl_enabled(previous_pdl)


def test_grouped_gemma_rmsnorm_validates_out_for_zero_rows() -> None:
    x = torch.empty(0, WIDE, dtype=torch.bfloat16, device="cuda")
    weight = torch.empty(WIDE, dtype=x.dtype, device=x.device)
    invalid_outputs = (
        torch.empty(1, WIDE, dtype=x.dtype, device=x.device),
        torch.empty(x.shape, dtype=torch.float16, device=x.device),
        torch.empty(x.shape, dtype=x.dtype, device="cpu"),
    )
    for out in invalid_outputs:
        with pytest.raises(ValueError, match="out must match"):
            grouped_gemma_rmsnorm(x, weight, HIDDEN_SIZE, 1e-6, out=out)

    out = torch.empty_like(x)
    assert grouped_gemma_rmsnorm(x, weight, HIDDEN_SIZE, 1e-6, out=out) is out


@pytest.mark.parametrize("enable_pdl", [False, True])
def test_persistent_mix_full_chain_cuda_graph_replays(enable_pdl: bool) -> None:
    if not current_platform().is_nvidia:
        pytest.skip("the persistent grid barrier is NVIDIA-only")
    if enable_pdl and not current_platform().is_hopper_plus:
        pytest.skip("PDL requires NVIDIA Hopper or newer")
    residual, projection, up = _inputs(8, torch.bfloat16)
    generator = torch.Generator(device="cuda").manual_seed(71)
    norm_weight = (
        torch.randn(WIDE, dtype=torch.bfloat16, device="cuda", generator=generator)
        * 0.02
    )

    def chain() -> tuple[torch.Tensor, ...]:
        normalized = grouped_gemma_rmsnorm(residual, norm_weight, HIDDEN_SIZE, 1e-6)
        mixed, inject = gated_residual_mix(
            normalized,
            projection,
            up,
            HC_COUNT,
            HIDDEN_SIZE,
            LOWRANK,
            override="triton_persistent_hyperconnection_mix",
        )
        combined = gated_residual_combine(
            mixed, residual, inject, HC_COUNT, HIDDEN_SIZE
        )
        return normalized, mixed, inject, combined

    previous_pdl = pdl_enabled()
    try:
        assert pdl_enabled(enable_pdl) is enable_pdl
        # Compile every PDL variant and populate the stream-private workspace.
        chain()
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            normalized, mixed, inject, combined = chain()
        for _ in range(10):
            graph.replay()
        torch.cuda.synchronize()

        grouped = residual.float().unflatten(-1, (HC_COUNT, HIDDEN_SIZE))
        expected_normalized = (
            grouped * torch.rsqrt(grouped.square().mean(dim=-1, keepdim=True) + 1e-6)
        ).flatten(-2) * (1.0 + norm_weight.float())
        expected_normalized = expected_normalized.to(residual.dtype)
        expected_mixed, expected_inject = _mix_reference(
            expected_normalized, projection, up, 1.0
        )
        expected_combined = (
            residual.unflatten(-1, (HC_COUNT, HIDDEN_SIZE))
            + expected_mixed.unsqueeze(-2)
            * (2 * torch.sigmoid(expected_inject)).unsqueeze(-1)
        ).flatten(-2)
        torch.testing.assert_close(
            normalized, expected_normalized, rtol=2e-2, atol=2e-2
        )
        torch.testing.assert_close(mixed, expected_mixed, rtol=4e-2, atol=4e-2)
        torch.testing.assert_close(inject, expected_inject, rtol=4e-2, atol=4e-2)
        torch.testing.assert_close(combined, expected_combined, rtol=4e-2, atol=4e-2)
    finally:
        pdl_enabled(previous_pdl)


def test_persistent_mix_uses_stream_private_barriers() -> None:
    if not current_platform().is_nvidia:
        pytest.skip("the persistent grid barrier is NVIDIA-only")
    inputs_a = _inputs(8, torch.bfloat16, seed=53)
    inputs_b = _inputs(8, torch.bfloat16, seed=59)
    streams = (torch.cuda.Stream(), torch.cuda.Stream())

    # First calls compile and create one barrier workspace per stream.
    for stream, values in zip(streams, (inputs_a, inputs_b), strict=True):
        with torch.cuda.stream(stream):
            gated_residual_mix(
                *values,
                HC_COUNT,
                HIDDEN_SIZE,
                LOWRANK,
                override="triton_persistent_hyperconnection_mix",
            )
    for stream in streams:
        stream.synchronize()

    outputs = []
    for stream, values in zip(streams, (inputs_a, inputs_b), strict=True):
        with torch.cuda.stream(stream):
            outputs.append(
                gated_residual_mix(
                    *values,
                    HC_COUNT,
                    HIDDEN_SIZE,
                    LOWRANK,
                    override="triton_persistent_hyperconnection_mix",
                )
            )
    for stream in streams:
        stream.synchronize()
    for values, (mixed, inject) in zip((inputs_a, inputs_b), outputs, strict=True):
        expected, expected_inject = _mix_reference(*values, 1.0)
        torch.testing.assert_close(mixed, expected, rtol=4e-2, atol=4e-2)
        torch.testing.assert_close(inject, expected_inject, rtol=4e-2, atol=4e-2)


@pytest.mark.parametrize(
    ("rows", "expected_kernel"),
    [
        (8, "triton_persistent_hyperconnection_mix"),
        (24, "triton_hyperconnection_mix"),
    ],
)
def test_default_dispatch_uses_persistent_only_for_low_m(
    rows: int, expected_kernel: str
) -> None:
    if not current_platform().is_nvidia:
        pytest.skip("the persistent grid barrier is NVIDIA-only")
    normalized, projection, up = _inputs(rows, torch.bfloat16, seed=67 + rows)
    capture = ShapeCapture.get()
    previous_capture = capture.enabled
    capture.clear()
    capture.enabled = True
    try:
        gated_residual_mix(
            normalized,
            projection,
            up,
            HC_COUNT,
            HIDDEN_SIZE,
            LOWRANK,
        )
        torch.cuda.synchronize()
        assert capture._records[-1].kernel_name == expected_kernel
    finally:
        capture.enabled = previous_capture
        capture.clear()


def test_deterministic_mode_filters_atomic_persistent_mix() -> None:
    if not current_platform().is_nvidia:
        pytest.skip("the persistent grid barrier is NVIDIA-only")
    normalized, projection, up = _inputs(8, torch.bfloat16, seed=61)
    capture = ShapeCapture.get()
    previous_capture = capture.enabled
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    capture.clear()
    capture.enabled = True
    try:
        torch.use_deterministic_algorithms(True)
        gated_residual_mix(
            normalized,
            projection,
            up,
            HC_COUNT,
            HIDDEN_SIZE,
            LOWRANK,
        )
        torch.cuda.synchronize()
        assert capture._records[-1].kernel_name == "triton_hyperconnection_mix"
    finally:
        torch.use_deterministic_algorithms(previous_deterministic)
        capture.enabled = previous_capture
        capture.clear()
