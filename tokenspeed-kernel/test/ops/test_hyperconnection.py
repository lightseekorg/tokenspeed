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
        weights_independent=False,
    )
    assert actual.shape == (rows, HIDDEN_SIZE)
    assert actual_inject.shape == (rows, HC_COUNT)
    if rows == 0:
        return
    expected, expected_inject = _mix_reference(normalized, projection, up, 1.0)
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(actual_inject, expected_inject, rtol=3e-2, atol=3e-2)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("rows", [1, 3, 4, 8, 15, 16])
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
        weights_independent=False,
    )
    expected, expected_inject = _mix_reference(normalized, projection, up, 1.0)
    tolerance = 4e-2 if dtype is torch.bfloat16 else 8e-3
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(
        actual_inject, expected_inject, rtol=tolerance, atol=tolerance
    )


@pytest.mark.parametrize(
    ("rows", "projection_scale"),
    [(1, 1.0), (3, 0.25), (4, 1.0), (8, 0.25), (16, 1.0), (32, 0.25)],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("has_inject", [False, True])
@pytest.mark.parametrize("enable_pdl", [False, True])
def test_cute_dsl_mix_matches_fp64_reference(
    rows: int,
    projection_scale: float,
    dtype: torch.dtype,
    has_inject: bool,
    enable_pdl: bool,
) -> None:
    if not current_platform().is_blackwell:
        pytest.skip("the CuTeDSL HC specialization is Blackwell-only")
    if KernelRegistry.get().get_by_name("cute_dsl_hyperconnection_mix") is None:
        pytest.skip("CuTeDSL dependencies are unavailable")
    normalized, projection, up = _inputs(rows, dtype, seed=17)
    if not has_inject:
        projection = projection[:LOWRANK]
    assert prepare_gated_residual_weight_cache(up, LOWRANK)
    previous_pdl = pdl_enabled(None)
    try:
        pdl_enabled(enable_pdl)
        actual, actual_inject = gated_residual_mix(
            normalized,
            projection,
            up,
            HC_COUNT,
            HIDDEN_SIZE,
            LOWRANK,
            projection_scale=projection_scale,
            override="cute_dsl_hyperconnection_mix",
            solution=None,
            weights_independent=False,
        )
        expected, expected_inject = _mix_reference(
            normalized, projection, up, projection_scale
        )
        tolerance = 3e-2 if dtype is torch.bfloat16 else 8e-3
        torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
        if has_inject:
            torch.testing.assert_close(
                actual_inject, expected_inject, rtol=tolerance, atol=tolerance
            )
        else:
            assert actual_inject is None
    finally:
        pdl_enabled(previous_pdl)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("inference_mode", [False, True])
def test_cute_dsl_model_parameters_support_inference_and_capture(
    dtype: torch.dtype, inference_mode: bool
) -> None:
    _require_cute_hc()
    normalized, projection_data, up_data = _inputs(4, dtype, seed=43)
    projection = torch.nn.Parameter(projection_data, requires_grad=True)
    up = torch.nn.Parameter(up_data, requires_grad=True)
    assert prepare_gated_residual_weight_cache(up, LOWRANK)

    def mix() -> tuple[torch.Tensor, torch.Tensor | None]:
        return gated_residual_mix(
            normalized,
            projection,
            up,
            HC_COUNT,
            HIDDEN_SIZE,
            LOWRANK,
            projection_scale=1.0,
            override="cute_dsl_hyperconnection_mix",
            solution=None,
            weights_independent=False,
        )

    tolerance = 3e-2 if dtype is torch.bfloat16 else 8e-3
    with torch.inference_mode() if inference_mode else torch.no_grad():
        eager, eager_inject = mix()
        expected, expected_inject = _mix_reference(normalized, projection, up, 1.0)
        torch.testing.assert_close(eager, expected, rtol=tolerance, atol=tolerance)
        torch.testing.assert_close(
            eager_inject, expected_inject, rtol=tolerance, atol=tolerance
        )
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual, actual_inject = mix()

        # Export must preserve storage aliasing for in-place weight reloads.
        projection.mul_(0.5)
        up.mul_(0.5)
        assert prepare_gated_residual_weight_cache(up, LOWRANK)
        graph.replay()
        torch.cuda.synchronize()
        expected, expected_inject = _mix_reference(normalized, projection, up, 1.0)
        torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
        torch.testing.assert_close(
            actual_inject, expected_inject, rtol=tolerance, atol=tolerance
        )
    assert projection.requires_grad and up.requires_grad
    assert projection.grad is None and up.grad is None


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
            weights_independent=False,
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
            weights_independent=False,
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


@pytest.mark.parametrize("rows", [1, 4, 8, 16])
@pytest.mark.parametrize("enable_pdl", [False, True])
@pytest.mark.parametrize(
    "backend",
    [
        "triton_persistent_hyperconnection_mix",
        "cute_dsl_hyperconnection_mix",
        "cute_fused_hyperconnection_mix",
    ],
)
def test_hc_full_chain_cuda_graph_replays(
    rows: int, enable_pdl: bool, backend: str
) -> None:
    if not current_platform().is_nvidia:
        pytest.skip("the persistent grid barrier is NVIDIA-only")
    if enable_pdl and not current_platform().is_hopper_plus:
        pytest.skip("PDL requires NVIDIA Hopper or newer")
    residual, projection, up = _inputs(rows, torch.bfloat16, seed=17)
    if backend == "cute_dsl_hyperconnection_mix":
        _require_cute_hc()
        assert prepare_gated_residual_weight_cache(up, LOWRANK)
    generator = torch.Generator(device="cuda").manual_seed(71)
    norm_weight = (
        torch.randn(WIDE, dtype=torch.bfloat16, device="cuda", generator=generator)
        * 0.02
    )

    def chain(value: torch.Tensor) -> tuple[torch.Tensor, ...]:
        normalized = grouped_gemma_rmsnorm(
            value, norm_weight, HIDDEN_SIZE, 1e-6, out=None
        )
        mixed, inject = gated_residual_mix(
            normalized,
            projection,
            up,
            HC_COUNT,
            HIDDEN_SIZE,
            LOWRANK,
            override=backend,
            weights_independent=backend == "cute_fused_hyperconnection_mix",
        )
        combined = gated_residual_combine(
            mixed, value, inject, HC_COUNT, HIDDEN_SIZE, override=None, solution=None
        )
        return normalized, mixed, inject, combined

    previous_pdl = pdl_enabled()
    try:
        assert pdl_enabled(enable_pdl) is enable_pdl
        # Compile every PDL variant and populate the stream-private workspace.
        chain(residual)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            value = residual
            for _ in range(3):
                previous = value
                normalized, mixed, inject, combined = chain(previous)
                value = combined
        for _ in range(10):
            graph.replay()
        torch.cuda.synchronize()

        grouped = previous.float().unflatten(-1, (HC_COUNT, HIDDEN_SIZE))
        expected_normalized = (
            grouped * torch.rsqrt(grouped.square().mean(dim=-1, keepdim=True) + 1e-6)
        ).flatten(-2) * (1.0 + norm_weight.float())
        expected_normalized = expected_normalized.to(residual.dtype)
        expected_mixed, expected_inject = _mix_reference(
            expected_normalized, projection, up, 1.0
        )
        expected_combined = (
            previous.unflatten(-1, (HC_COUNT, HIDDEN_SIZE))
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
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            gated_residual_mix(
                *values,
                HC_COUNT,
                HIDDEN_SIZE,
                LOWRANK,
                override="triton_persistent_hyperconnection_mix",
                weights_independent=False,
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
                    weights_independent=False,
                )
            )
    for stream in streams:
        stream.synchronize()
    for values, (mixed, inject) in zip((inputs_a, inputs_b), outputs, strict=True):
        expected, expected_inject = _mix_reference(*values, 1.0)
        torch.testing.assert_close(mixed, expected, rtol=4e-2, atol=4e-2)
        torch.testing.assert_close(inject, expected_inject, rtol=4e-2, atol=4e-2)

    graphs = []
    outputs = []
    for stream, values in zip(streams, (inputs_a, inputs_b), strict=True):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(3):
                result = gated_residual_mix(
                    *values,
                    HC_COUNT,
                    HIDDEN_SIZE,
                    LOWRANK,
                    projection_scale=1.0,
                    override="triton_persistent_hyperconnection_mix",
                    solution=None,
                    weights_independent=False,
                )
        graphs.append(graph)
        outputs.append(result)
    for _ in range(16):
        for stream, graph in zip(streams, graphs, strict=True):
            with torch.cuda.stream(stream):
                graph.replay()
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
            weights_independent=False,
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
            weights_independent=False,
        )
        torch.cuda.synchronize()
        assert capture._records[-1].kernel_name == "triton_hyperconnection_mix"
    finally:
        torch.use_deterministic_algorithms(previous_deterministic)
        capture.enabled = previous_capture
        capture.clear()


@pytest.mark.parametrize("has_inject", [False, True])
@pytest.mark.parametrize("enable_pdl", [False, True])
def test_persistent_mix_reuses_workspace_across_shapes(
    has_inject: bool, enable_pdl: bool
) -> None:
    if not current_platform().is_nvidia:
        pytest.skip("the persistent grid barrier is NVIDIA-only")
    if enable_pdl and not current_platform().is_hopper_plus:
        pytest.skip("PDL requires NVIDIA Hopper or newer")
    previous_pdl = pdl_enabled(None)
    try:
        pdl_enabled(enable_pdl)
        # These calls share the same stream workspace, including across dtypes.
        # Every call must leave scratch ready for a different subsequent shape.
        for index, rows in enumerate((4, 1, 16, 3, 8, 15, 4)):
            dtype = torch.bfloat16 if index % 2 == 0 else torch.float16
            normalized, projection, up = _inputs(rows, dtype, seed=101 + index)
            if not has_inject:
                projection = projection[:LOWRANK]
            mixed, inject = gated_residual_mix(
                normalized,
                projection,
                up,
                HC_COUNT,
                HIDDEN_SIZE,
                LOWRANK,
                projection_scale=0.25,
                override="triton_persistent_hyperconnection_mix",
                solution=None,
                weights_independent=False,
            )
            expected, expected_inject = _mix_reference(normalized, projection, up, 0.25)
            tolerance = 4e-2 if dtype is torch.bfloat16 else 8e-3
            torch.testing.assert_close(mixed, expected, rtol=tolerance, atol=tolerance)
            if has_inject:
                torch.testing.assert_close(
                    inject, expected_inject, rtol=tolerance, atol=tolerance
                )
            else:
                assert inject is None
    finally:
        pdl_enabled(previous_pdl)


@pytest.mark.parametrize("calls_per_replay", [1, 3, 4])
@pytest.mark.parametrize("enable_pdl", [False, True])
def test_persistent_mix_graph_observes_changed_inputs(
    enable_pdl: bool, calls_per_replay: int
) -> None:
    if not current_platform().is_nvidia:
        pytest.skip("the persistent grid barrier is NVIDIA-only")
    if enable_pdl and not current_platform().is_hopper_plus:
        pytest.skip("PDL requires NVIDIA Hopper or newer")
    normalized, projection, up = _inputs(4, torch.bfloat16, seed=113)

    def mix() -> tuple[torch.Tensor, torch.Tensor | None]:
        return gated_residual_mix(
            normalized,
            projection,
            up,
            HC_COUNT,
            HIDDEN_SIZE,
            LOWRANK,
            projection_scale=1.0,
            override="triton_persistent_hyperconnection_mix",
            solution=None,
            weights_independent=False,
        )

    previous_pdl = pdl_enabled(None)
    try:
        pdl_enabled(enable_pdl)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        # Create the workspace before capture, so graph replay advances the
        # device generation rather than replaying workspace initialization.
        with torch.cuda.stream(stream):
            mix()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            outputs = [mix() for _ in range(calls_per_replay)]
        for _ in range(4):
            normalized.mul_(-0.75)
            projection.mul_(0.5)
            up.mul_(-0.5)
            graph.replay()
            expected, expected_inject = _mix_reference(normalized, projection, up, 1.0)
            for mixed, inject in outputs:
                torch.testing.assert_close(mixed, expected, rtol=4e-2, atol=4e-2)
                torch.testing.assert_close(
                    inject, expected_inject, rtol=4e-2, atol=4e-2
                )
    finally:
        pdl_enabled(previous_pdl)


@pytest.mark.parametrize("enable_pdl", [False, True])
def test_persistent_mix_generation_crosses_uint32(enable_pdl: bool) -> None:
    if not current_platform().is_nvidia:
        pytest.skip("the persistent grid barrier is NVIDIA-only")
    if enable_pdl and not current_platform().is_hopper_plus:
        pytest.skip("PDL requires NVIDIA Hopper or newer")
    from tokenspeed_kernel.ops.residual.triton import _persistent_workspace

    normalized, projection, up = _inputs(4, torch.bfloat16, seed=127)
    raw, counters = _persistent_workspace(normalized.device, LOWRANK + HC_COUNT)
    num_ctas = torch.cuda.get_device_properties(normalized.device).multi_processor_count
    generation = (1 << 32) - 1
    previous_pdl = pdl_enabled(None)
    try:
        pdl_enabled(enable_pdl)
        raw.zero_()
        counters.copy_(
            torch.tensor(
                [generation * num_ctas, generation],
                device=counters.device,
                dtype=torch.int64,
            )
        )
        expected, expected_inject = _mix_reference(normalized, projection, up, 1.0)
        for step in range(3):
            mixed, inject = gated_residual_mix(
                normalized,
                projection,
                up,
                HC_COUNT,
                HIDDEN_SIZE,
                LOWRANK,
                projection_scale=1.0,
                override="triton_persistent_hyperconnection_mix",
                solution=None,
                weights_independent=False,
            )
            torch.testing.assert_close(mixed, expected, rtol=4e-2, atol=4e-2)
            torch.testing.assert_close(inject, expected_inject, rtol=4e-2, atol=4e-2)
            next_generation = generation + step + 1
            assert counters.tolist() == [next_generation * num_ctas, next_generation]
            assert torch.count_nonzero(raw[next_generation & 1]).item() == 0
    finally:
        torch.cuda.synchronize()
        raw.zero_()
        counters.zero_()
        pdl_enabled(previous_pdl)


@triton.jit
def _publish_mix_inputs_kernel(
    x_source,
    projection_source,
    up_source,
    x_target,
    projection_target,
    up_target,
    X_ELEMENTS: tl.constexpr,
    PROJECTION_ELEMENTS: tl.constexpr,
    UP_ELEMENTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    stride = tl.num_programs(0) * BLOCK
    offsets = tl.arange(0, BLOCK)
    tl.extra.cuda.gdc_wait()
    # Intentionally allow mix to prefetch the old destination contents while
    # this resident producer is still publishing all three new inputs.
    tl.extra.cuda.gdc_launch_dependents()
    for start in range(pid * BLOCK, PROJECTION_ELEMENTS, stride):
        indices = start + offsets
        mask = indices < PROJECTION_ELEMENTS
        value = tl.load(projection_source + indices, mask=mask, other=0.0)
        tl.store(projection_target + indices, value, mask=mask)
    for start in range(pid * BLOCK, UP_ELEMENTS, stride):
        indices = start + offsets
        mask = indices < UP_ELEMENTS
        value = tl.load(up_source + indices, mask=mask, other=0.0)
        tl.store(up_target + indices, value, mask=mask)
    for start in range(pid * BLOCK, X_ELEMENTS, stride):
        indices = start + offsets
        mask = indices < X_ELEMENTS
        value = tl.load(x_source + indices, mask=mask, other=0.0)
        tl.store(x_target + indices, value, mask=mask)


@pytest.mark.parametrize("rows", [1, 4, 16])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("has_inject", [False, True])
def test_persistent_prefetch_observes_pdl_producer_updates(
    rows: int, dtype: torch.dtype, has_inject: bool
) -> None:
    if not current_platform().is_hopper_plus:
        pytest.skip("bulk weight prefetch and PDL require NVIDIA Hopper or newer")
    x_source, projection_source, up_source = _inputs(rows, dtype, seed=139)
    if not has_inject:
        projection_source = projection_source[:LOWRANK]
    projection_source.mul_(2.0)
    up_source.mul_(4.0)
    normalized = torch.zeros_like(x_source)
    projection = torch.zeros_like(projection_source)
    up = torch.zeros_like(up_source)
    num_ctas = torch.cuda.get_device_properties(normalized.device).multi_processor_count

    def publish_and_mix() -> tuple[torch.Tensor, torch.Tensor | None]:
        _publish_mix_inputs_kernel[(num_ctas,)](
            x_source,
            projection_source,
            up_source,
            normalized,
            projection,
            up,
            X_ELEMENTS=normalized.numel(),
            PROJECTION_ELEMENTS=projection.numel(),
            UP_ELEMENTS=up.numel(),
            BLOCK=1024,
            num_warps=4,
            launch_pdl=True,
        )
        return gated_residual_mix(
            normalized,
            projection,
            up,
            HC_COUNT,
            HIDDEN_SIZE,
            LOWRANK,
            projection_scale=1.0,
            override="triton_persistent_hyperconnection_mix",
            solution=None,
            weights_independent=False,
        )

    previous_pdl = pdl_enabled(None)
    try:
        pdl_enabled(True)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            publish_and_mix()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            outputs = [publish_and_mix() for _ in range(3)]
        for _ in range(3):
            x_source.mul_(-0.875)
            projection_source.mul_(-1.125)
            up_source.mul_(-1.0)
            graph.replay()
            expected, expected_inject = _mix_reference(
                x_source, projection_source, up_source, 1.0
            )
            tolerance = 4e-2 if dtype is torch.bfloat16 else 8e-3
            for mixed, inject in outputs:
                torch.testing.assert_close(
                    mixed, expected, rtol=tolerance, atol=tolerance
                )
                if has_inject:
                    torch.testing.assert_close(
                        inject, expected_inject, rtol=tolerance, atol=tolerance
                    )
                else:
                    assert inject is None
    finally:
        torch.cuda.synchronize()
        pdl_enabled(previous_pdl)


@pytest.mark.parametrize("input_index", [0, 1, 2])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_persistent_prefetch_accepts_unaligned_contiguous_views(
    input_index: int, dtype: torch.dtype
) -> None:
    if not current_platform().is_hopper_plus:
        pytest.skip("bulk weight prefetch and PDL require NVIDIA Hopper or newer")
    inputs = list(_inputs(4, dtype, seed=149))
    original = inputs[input_index]
    storage = torch.empty(original.numel() + 1, dtype=dtype, device=original.device)
    unaligned = storage[1:].view_as(original)
    unaligned.copy_(original)
    assert unaligned.is_contiguous()
    assert unaligned.data_ptr() % 16 != 0
    inputs[input_index] = unaligned
    normalized, projection, up = inputs
    previous_pdl = pdl_enabled(None)
    try:
        pdl_enabled(True)
        mixed, inject = gated_residual_mix(
            normalized,
            projection,
            up,
            HC_COUNT,
            HIDDEN_SIZE,
            LOWRANK,
            projection_scale=1.0,
            override="triton_persistent_hyperconnection_mix",
            solution=None,
            weights_independent=False,
        )
        expected, expected_inject = _mix_reference(normalized, projection, up, 1.0)
        tolerance = 4e-2 if dtype is torch.bfloat16 else 8e-3
        torch.testing.assert_close(mixed, expected, rtol=tolerance, atol=tolerance)
        torch.testing.assert_close(
            inject, expected_inject, rtol=tolerance, atol=tolerance
        )
    finally:
        pdl_enabled(previous_pdl)


def _require_cute_hc() -> None:
    if not current_platform().is_blackwell:
        pytest.skip("warp-specialized HC requires Blackwell")
    if KernelRegistry.get().get_by_name("cute_dsl_hyperconnection_mix") is None:
        pytest.skip("CuTeDSL dependencies are unavailable")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    ("rows", "enable_pdl", "unaligned_input", "expected_kernel"),
    [
        (1, True, None, "cute_dsl_hyperconnection_mix"),
        (4, True, None, "cute_dsl_hyperconnection_mix"),
        (8, True, None, "cute_dsl_hyperconnection_mix"),
        (9, True, None, "triton_persistent_hyperconnection_mix"),
        (16, True, None, "triton_persistent_hyperconnection_mix"),
        (24, True, None, "triton_hyperconnection_mix"),
        (4, False, None, "triton_persistent_hyperconnection_mix"),
        (4, True, 0, "triton_persistent_hyperconnection_mix"),
        (4, True, 1, "triton_persistent_hyperconnection_mix"),
        (4, True, 2, "cute_dsl_hyperconnection_mix"),
    ],
)
def test_prepared_hc_default_dispatch(
    dtype: torch.dtype,
    rows: int,
    enable_pdl: bool,
    unaligned_input: int | None,
    expected_kernel: str,
) -> None:
    _require_cute_hc()
    inputs = list(_inputs(rows, dtype, seed=157))
    if unaligned_input is not None:
        original = inputs[unaligned_input]
        storage = torch.empty(original.numel() + 1, dtype=dtype, device=original.device)
        inputs[unaligned_input] = storage[1:].view_as(original).copy_(original)
    normalized, projection, up = inputs
    assert prepare_gated_residual_weight_cache(up, LOWRANK)
    capture = ShapeCapture.get()
    previous_capture = capture.enabled
    previous_pdl = pdl_enabled(None)
    try:
        pdl_enabled(enable_pdl)
        capture.enabled = True
        capture.clear()
        mixed, inject = gated_residual_mix(
            normalized,
            projection,
            up,
            HC_COUNT,
            HIDDEN_SIZE,
            LOWRANK,
            projection_scale=1.0,
            override=None,
            solution=None,
            weights_independent=False,
        )
        assert capture._records[-1].kernel_name == expected_kernel
        expected, expected_inject = _mix_reference(normalized, projection, up, 1.0)
        tolerance = 4e-2 if dtype is torch.bfloat16 else 8e-3
        torch.testing.assert_close(mixed, expected, rtol=tolerance, atol=tolerance)
        torch.testing.assert_close(
            inject, expected_inject, rtol=tolerance, atol=tolerance
        )
    finally:
        capture.enabled = previous_capture
        capture.clear()
        pdl_enabled(previous_pdl)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_cute_hc_up_prefetch_waits_for_ancestor_weight_updates(
    dtype: torch.dtype,
) -> None:
    _require_cute_hc()
    from tokenspeed_kernel.ops.residual.cute_dsl import (
        _get_prepared_padded_up_weight,
    )

    x_source, projection_source, up_source = _inputs(4, dtype, seed=163)
    projection_source.mul_(2.0)
    up_source.mul_(4.0)
    normalized = torch.empty_like(x_source)
    projection = torch.empty_like(projection_source)
    up = torch.zeros_like(up_source)
    assert prepare_gated_residual_weight_cache(up, LOWRANK)
    assert prepare_gated_residual_weight_cache(up_source, LOWRANK)
    padded_up = _get_prepared_padded_up_weight(up)
    padded_source = _get_prepared_padded_up_weight(up_source)
    num_ctas = torch.cuda.get_device_properties(normalized.device).multi_processor_count

    def publish_and_mix() -> tuple[torch.Tensor, torch.Tensor | None]:
        # Also publish the prepared up cache from an ancestor PDL kernel. Down
        # must finish its wait before triggering up's independent weight warp.
        _publish_mix_inputs_kernel[(num_ctas,)](
            x_source,
            projection_source,
            padded_source,
            normalized,
            projection,
            padded_up,
            X_ELEMENTS=normalized.numel(),
            PROJECTION_ELEMENTS=projection.numel(),
            UP_ELEMENTS=padded_up.numel(),
            BLOCK=1024,
            num_warps=4,
            launch_pdl=True,
        )
        return gated_residual_mix(
            normalized,
            projection,
            up,
            HC_COUNT,
            HIDDEN_SIZE,
            LOWRANK,
            projection_scale=1.0,
            override="cute_dsl_hyperconnection_mix",
            solution=None,
            weights_independent=False,
        )

    previous_pdl = pdl_enabled(None)
    try:
        pdl_enabled(True)
        publish_and_mix()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs = [publish_and_mix() for _ in range(3)]
        for _ in range(3):
            x_source.mul_(-0.875)
            projection_source.mul_(-1.125)
            up_source.mul_(-1.0)
            assert prepare_gated_residual_weight_cache(up_source, LOWRANK)
            graph.replay()
            expected, expected_inject = _mix_reference(
                x_source, projection_source, up_source, 1.0
            )
            tolerance = 4e-2 if dtype is torch.bfloat16 else 8e-3
            for mixed, inject in outputs:
                torch.testing.assert_close(
                    mixed, expected, rtol=tolerance, atol=tolerance
                )
                torch.testing.assert_close(
                    inject, expected_inject, rtol=tolerance, atol=tolerance
                )
    finally:
        torch.cuda.synchronize()
        pdl_enabled(previous_pdl)


def test_cute_hc_concurrent_graphs_keep_outputs_independent() -> None:
    _require_cute_hc()
    cases = [_inputs(rows, torch.bfloat16, seed=173 + rows) for rows in (1, 4)]
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    graphs = []
    outputs = []
    previous_pdl = pdl_enabled(None)
    try:
        pdl_enabled(True)
        for inputs, stream in zip(cases, streams, strict=True):
            assert prepare_gated_residual_weight_cache(inputs[2], LOWRANK)
            stream.wait_stream(torch.cuda.current_stream())

            def mix() -> tuple[torch.Tensor, torch.Tensor | None]:
                return gated_residual_mix(
                    *inputs,
                    HC_COUNT,
                    HIDDEN_SIZE,
                    LOWRANK,
                    projection_scale=1.0,
                    override="cute_dsl_hyperconnection_mix",
                    solution=None,
                    weights_independent=False,
                )

            with torch.cuda.stream(stream):
                mix()
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                outputs.append([mix() for _ in range(3)])
            graphs.append(graph)
        for _ in range(5):
            for graph, stream in zip(graphs, streams, strict=True):
                with torch.cuda.stream(stream):
                    graph.replay()
        torch.cuda.synchronize()
        for inputs, results in zip(cases, outputs, strict=True):
            expected, expected_inject = _mix_reference(*inputs, 1.0)
            for mixed, inject in results:
                torch.testing.assert_close(mixed, expected, rtol=4e-2, atol=4e-2)
                torch.testing.assert_close(
                    inject, expected_inject, rtol=4e-2, atol=4e-2
                )
    finally:
        torch.cuda.synchronize()
        pdl_enabled(previous_pdl)


def test_cute_hc_compilation_is_device_private() -> None:
    _require_cute_hc()
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two Blackwell devices")
    if torch.cuda.get_device_capability(1)[0] != 10:
        pytest.skip("requires two Blackwell devices")
    inputs = _inputs(4, torch.bfloat16, seed=181)
    for device_index in (0, 1, 0):
        values = tuple(t.to(device=torch.device("cuda", device_index)) for t in inputs)
        assert prepare_gated_residual_weight_cache(values[2], LOWRANK)
        mixed, inject = gated_residual_mix(
            *values,
            HC_COUNT,
            HIDDEN_SIZE,
            LOWRANK,
            projection_scale=1.0,
            override="cute_dsl_hyperconnection_mix",
            solution=None,
            weights_independent=False,
        )
        expected, expected_inject = _mix_reference(*values, 1.0)
        torch.testing.assert_close(mixed, expected, rtol=4e-2, atol=4e-2)
        torch.testing.assert_close(inject, expected_inject, rtol=4e-2, atol=4e-2)


def test_prepared_cute_hc_is_deterministic() -> None:
    _require_cute_hc()
    inputs = _inputs(4, torch.bfloat16, seed=191)
    assert prepare_gated_residual_weight_cache(inputs[2], LOWRANK)
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_pdl = pdl_enabled(None)
    try:
        torch.use_deterministic_algorithms(True)
        pdl_enabled(True)
        outputs = [
            gated_residual_mix(
                *inputs,
                HC_COUNT,
                HIDDEN_SIZE,
                LOWRANK,
                projection_scale=1.0,
                override=None,
                solution=None,
                weights_independent=False,
            )
            for _ in range(4)
        ]
        for mixed, inject in outputs[1:]:
            assert torch.equal(mixed, outputs[0][0])
            assert torch.equal(inject, outputs[0][1])
    finally:
        torch.use_deterministic_algorithms(previous_deterministic)
        pdl_enabled(previous_pdl)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("rows", [1, 3, 4, 8, 9, 16])
@pytest.mark.parametrize("has_inject", [False, True])
@pytest.mark.parametrize("enable_pdl", [False, True])
@pytest.mark.parametrize("weights_independent", [False, True])
def test_fused_cute_mix_matches_fp64_and_graph(
    dtype, rows, has_inject, enable_pdl, weights_independent
):
    from tokenspeed_kernel.ops.residual.cute_fused import supports_fused_hc

    if not supports_fused_hc(torch.device("cuda")):
        pytest.skip("requires a full Blackwell device and CuTe")
    x, w, u = _inputs(rows, dtype, seed=3450 + rows)
    if not has_inject:
        w = w[:LOWRANK]
    scale = 0.25 if rows % 2 else 1.0
    previous_pdl = pdl_enabled(None)
    try:
        pdl_enabled(enable_pdl)

        def call():
            return gated_residual_mix(
                x,
                w,
                u,
                HC_COUNT,
                HIDDEN_SIZE,
                LOWRANK,
                weights_independent=weights_independent,
                projection_scale=scale,
                override="cute_fused_hyperconnection_mix",
                solution=None,
            )

        actual = call()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(3):
                actual = call()
        for _ in range(5):
            graph.replay()
        torch.cuda.synchronize()
        expected = _mix_reference(x, w, u, scale)
        tolerance = 0.04 if dtype == torch.bfloat16 else 0.008
        torch.testing.assert_close(
            actual[0], expected[0], rtol=tolerance, atol=tolerance
        )
        if has_inject:
            torch.testing.assert_close(
                actual[1], expected[1], rtol=tolerance, atol=tolerance
            )
        else:
            assert actual[1] is None
    finally:
        pdl_enabled(previous_pdl)


@pytest.mark.parametrize("rows", [1, 8, 9, 16, 24])
@pytest.mark.parametrize("enable_pdl", [False, True])
@pytest.mark.parametrize("weights_independent", [False, True])
def test_fused_cute_dispatch_requires_explicit_weight_contract(
    rows, enable_pdl, weights_independent
):
    from tokenspeed_kernel.ops.residual.cute_fused import supports_fused_hc

    if not supports_fused_hc(torch.device("cuda")):
        pytest.skip("requires a full Blackwell device and CuTe")
    x, w, u = _inputs(rows, torch.bfloat16, seed=4511)
    previous_pdl = pdl_enabled(None)
    capture = ShapeCapture.get()
    previous_capture = capture.enabled
    try:
        pdl_enabled(enable_pdl)
        capture.enabled = True
        capture.clear()
        actual = gated_residual_mix(
            x,
            w,
            u,
            HC_COUNT,
            HIDDEN_SIZE,
            LOWRANK,
            weights_independent=weights_independent,
            projection_scale=1.0,
            override=None,
            solution=None,
        )
        expected_name = (
            "cute_fused_hyperconnection_mix"
            if weights_independent and rows <= 16
            else (
                "triton_persistent_hyperconnection_mix"
                if rows <= 16
                else "triton_hyperconnection_mix"
            )
        )
        assert capture._records[-1].kernel_name == expected_name
        expected = _mix_reference(x, w, u, 1.0)
        for got, ref in zip(actual, expected):
            torch.testing.assert_close(got, ref, rtol=0.04, atol=0.04)
    finally:
        capture.enabled = previous_capture
        capture.clear()
        pdl_enabled(previous_pdl)


@pytest.mark.parametrize("input_index", [0, 1, 2])
@pytest.mark.parametrize("offset", [1, 8])
def test_fused_cute_dispatch_checks_all_tma_alignments(input_index, offset):
    from tokenspeed_kernel.ops.residual.cute_fused import supports_fused_hc

    if not supports_fused_hc(torch.device("cuda")):
        pytest.skip("requires a full Blackwell device and CuTe")
    values = list(_inputs(4, torch.bfloat16, seed=4731))
    original = values[input_index]
    storage = torch.empty(
        original.numel() + offset, device="cuda", dtype=original.dtype
    )
    values[input_index] = storage[offset:].view_as(original).copy_(original)
    capture = ShapeCapture.get()
    previous_capture = capture.enabled
    try:
        capture.enabled = True
        capture.clear()
        actual = gated_residual_mix(
            *values,
            HC_COUNT,
            HIDDEN_SIZE,
            LOWRANK,
            weights_independent=True,
            projection_scale=1.0,
            override=None,
            solution=None,
        )
        expected_name = (
            "cute_fused_hyperconnection_mix"
            if offset == 8
            else "triton_persistent_hyperconnection_mix"
        )
        assert capture._records[-1].kernel_name == expected_name
        expected = _mix_reference(*values, 1.0)
        for got, ref in zip(actual, expected):
            torch.testing.assert_close(got, ref, rtol=0.04, atol=0.04)
    finally:
        capture.enabled = previous_capture
        capture.clear()


@pytest.mark.parametrize("has_inject", [False, True])
def test_fused_cute_graphs_share_only_stream_private_generations(has_inject):
    from tokenspeed_kernel.ops.residual.cute_fused import (
        _persistent_workspace,
        supports_fused_hc,
    )

    if not supports_fused_hc(torch.device("cuda")):
        pytest.skip("requires a full Blackwell device and CuTe")
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    graphs = []
    outputs = []
    expected = []
    counters = []
    input_banks = []
    projection_rows = LOWRANK + (HC_COUNT if has_inject else 0)
    clusters = (projection_rows + 63) // 64
    # Graphs with odd call counts, mixed rows and retained outputs exercise the
    # runtime wrapper's workspace key and on-device generation selection.
    for index, stream in enumerate(streams):
        values = [
            _inputs(rows, dtype, seed=4800 + index * 100 + rows)
            for rows, dtype in (
                (1, torch.bfloat16),
                (9, torch.float16),
                (16, torch.bfloat16),
            )
        ]
        if not has_inject:
            values = [(x, w[:LOWRANK], u) for x, w, u in values]
        input_banks.append(values)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for x, w, u in values:
                gated_residual_mix(
                    x,
                    w,
                    u,
                    HC_COUNT,
                    HIDDEN_SIZE,
                    LOWRANK,
                    weights_independent=True,
                    projection_scale=1.0,
                    override="cute_fused_hyperconnection_mix",
                    solution=None,
                )
            raw, count = _persistent_workspace(values[0][0].device, projection_rows)
            generation = 2**32 - 1
            # All consumed projection elements must be overwritten even when
            # the workspace is uninitialized and row counts change in a graph.
            # 0xffff is NaN in both BF16 and FP16; storage is intentionally opaque.
            raw.fill_(-1)
            count.fill_(generation)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            result = [
                gated_residual_mix(
                    x,
                    w,
                    u,
                    HC_COUNT,
                    HIDDEN_SIZE,
                    LOWRANK,
                    weights_independent=True,
                    projection_scale=1.0,
                    override="cute_fused_hyperconnection_mix",
                    solution=None,
                )
                for x, w, u in values
            ]
        graphs.append(graph)
        outputs.append(result)
        expected.append([_mix_reference(x, w, u, 1.0) for x, w, u in values])
        counters.append(count)
    assert counters[0].data_ptr() != counters[1].data_ptr()
    for _ in range(5):
        for stream, graph in zip(streams, graphs):
            with torch.cuda.stream(stream):
                graph.replay()
    torch.cuda.synchronize()
    for results, refs, counter in zip(outputs, expected, counters):
        for result, ref in zip(results, refs):
            for got, want in zip(result, ref):
                if want is None or want.numel() == 0:
                    assert got is None
                else:
                    torch.testing.assert_close(got, want, rtol=0.04, atol=0.04)
        assert counter.cpu().tolist() == [generation + 15] * clusters


@pytest.mark.parametrize("rows", [1, 16])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("has_inject", [False, True])
@pytest.mark.parametrize("weights_independent", [False, True])
def test_fused_cluster_mix_observes_pdl_updates(
    rows, dtype, has_inject, weights_independent
):
    from tokenspeed_kernel.ops.residual.cute_fused import supports_fused_hc

    if not supports_fused_hc(torch.device("cuda")):
        pytest.skip("requires six resident Blackwell clusters and CuTe")
    x_source, w_source, u_source = _inputs(rows, dtype, seed=5107 + rows)
    if not has_inject:
        w_source = w_source[:LOWRANK]
    w_source.mul_(2.0)
    u_source.mul_(4.0)
    x = torch.zeros_like(x_source)
    w = w_source.clone() if weights_independent else torch.zeros_like(w_source)
    u = u_source.clone() if weights_independent else torch.zeros_like(u_source)
    ctas = torch.cuda.get_device_properties(x.device).multi_processor_count

    def call():
        # Independent weights are updated before forward. Otherwise the PDL
        # ancestor publishes them after it has already triggered the consumer.
        _publish_mix_inputs_kernel[(ctas,)](
            x_source,
            w_source,
            u_source,
            x,
            w,
            u,
            X_ELEMENTS=x.numel(),
            PROJECTION_ELEMENTS=0 if weights_independent else w.numel(),
            UP_ELEMENTS=0 if weights_independent else u.numel(),
            BLOCK=1024,
            num_warps=4,
            launch_pdl=True,
        )
        return gated_residual_mix(
            x,
            w,
            u,
            HC_COUNT,
            HIDDEN_SIZE,
            LOWRANK,
            projection_scale=0.25,
            override="cute_fused_hyperconnection_mix",
            solution=None,
            weights_independent=weights_independent,
        )

    previous_pdl = pdl_enabled(None)
    try:
        pdl_enabled(True)
        call()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            results = [call() for _ in range(3)]
        for _ in range(3):
            x_source.mul_(-0.875)
            w_source.mul_(-1.125)
            u_source.mul_(-1.0)
            if weights_independent:
                w.copy_(w_source)
                u.copy_(u_source)
            graph.replay()
            expected = _mix_reference(x_source, w_source, u_source, 0.25)
            tolerance = 0.04 if dtype == torch.bfloat16 else 0.008
            for result in results:
                for got, want in zip(result, expected):
                    if want is None or want.numel() == 0:
                        assert got is None
                    else:
                        torch.testing.assert_close(
                            got, want, rtol=tolerance, atol=tolerance
                        )
    finally:
        torch.cuda.synchronize()
        pdl_enabled(previous_pdl)


@pytest.mark.parametrize("rows", [4, 16])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fused_cluster_mix_is_deterministic(rows, dtype):
    from tokenspeed_kernel.ops.residual.cute_fused import supports_fused_hc

    if not supports_fused_hc(torch.device("cuda")):
        pytest.skip("requires six resident Blackwell clusters and CuTe")
    values = _inputs(rows, dtype, seed=5301 + rows)
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        outputs = [
            gated_residual_mix(
                *values,
                HC_COUNT,
                HIDDEN_SIZE,
                LOWRANK,
                weights_independent=True,
                projection_scale=1.0,
                override=None,
                solution=None,
            )
            for _ in range(8)
        ]
        expected = _mix_reference(*values, 1.0)
        tolerance = 0.04 if dtype == torch.bfloat16 else 0.008
        for got, want in zip(outputs[0], expected):
            torch.testing.assert_close(got, want, rtol=tolerance, atol=tolerance)
        for result in outputs[1:]:
            for got, want in zip(result, outputs[0]):
                assert torch.equal(got, want)
    finally:
        torch.use_deterministic_algorithms(previous)


def test_fused_cluster_grid_checks_cluster_capacity(monkeypatch):
    from tokenspeed_kernel.ops.residual import cute_fused

    if not cute_fused.supports_fused_hc(torch.device("cuda")):
        pytest.skip("requires six resident Blackwell clusters and CuTe")
    # A large SM count alone does not establish cluster residency, for example
    # when a stream belongs to a reduced green context.
    monkeypatch.setattr(cute_fused, "_resident_clusters", lambda index, stream: 5)
    assert not cute_fused.supports_fused_hc(torch.device("cuda"))


def test_fused_cluster_split_k_does_not_exceed_sixteen():
    from tokenspeed_kernel.ops.residual.cute_fused import supports_fused_hc

    if not supports_fused_hc(torch.device("cuda")):
        pytest.skip("requires six resident Blackwell clusters and CuTe")
    from tokenspeed_kernel.thirdparty.cute_dsl.hc_fused import FusedGatedResidualKernel

    with pytest.raises(ValueError, match="split-K=16"):
        FusedGatedResidualKernel(4, 324, 32, True, 1.0, True)


@pytest.mark.parametrize("rows", [1, 4, 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("has_inject", [False, True])
@pytest.mark.parametrize("down_stages", [2, 5])
def test_fused_down_consumes_every_k128_stage_on_graph_replay(
    rows, dtype, has_inject, down_stages, monkeypatch
):
    from tokenspeed_kernel.ops.residual import cute_fused
    from tokenspeed_kernel.ops.residual.cute_fused import (
        _persistent_workspace,
        supports_fused_hc,
    )

    if not supports_fused_hc(torch.device("cuda")):
        pytest.skip("requires six resident Blackwell clusters and CuTe")

    class StagedKernel(cute_fused.FusedGatedResidualKernel):
        def __init__(
            self, rows, projection_rows, split_k, use_pdl, scale, weights_independent
        ):
            super().__init__(
                rows, projection_rows, split_k, use_pdl, scale, weights_independent
            )
            self.down_stages = down_stages

    # Two buffers must recycle and wrap their phases while consuming all five
    # K128 tiles; five buffers must consume the same data without recycling.
    # Isolate compilation because the public dispatch key fixes the tactic.
    monkeypatch.setattr(cute_fused, "FusedGatedResidualKernel", StagedKernel)
    monkeypatch.setattr(cute_fused, "_COMPILED", {})
    x, w, u = _inputs(rows, dtype, seed=6023 + rows)
    if not has_inject:
        w = w[:LOWRANK]
    positions = torch.arange(WIDE, device="cuda")
    tokens = torch.arange(rows, device="cuda")
    x.copy_((positions[None, :] % 17 - 8 + tokens[:, None]) / 16)
    w.zero_()
    columns = (torch.arange(w.shape[0], device="cuda") % 13 - 6).to(dtype)
    split_pattern = (torch.arange(16, device="cuda") % 5 - 2).to(dtype)
    split_offsets = torch.arange(16, device="cuda") * 640
    scale = 0.25

    def call():
        return gated_residual_mix(
            x,
            w,
            u,
            HC_COUNT,
            HIDDEN_SIZE,
            LOWRANK,
            projection_scale=scale,
            weights_independent=True,
            solution=None,
            override="cute_fused_hyperconnection_mix",
        )

    previous = pdl_enabled(None)
    try:
        pdl_enabled(True)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            call()
            storage, _ = _persistent_workspace(x.device, w.shape[0])
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            actual = call()
        for active_stages in ((0,), (1,), (2,), (3,), (4,), (0, 1, 2, 3, 4)):
            w.zero_()
            # Distinct endpoints catch omitted/repeated stages and incorrect
            # activation/weight strides, including the last K128 tile. Reload
            # weights between forwards without changing the captured pointers.
            for stage in active_stages:
                offsets = split_offsets + stage * 128
                w[:, offsets] = columns[:, None] * split_pattern[None, :] / 16
                w[:, offsets + 127] = (
                    columns[:, None] * split_pattern.flip(0)[None, :] / 32
                )
            storage.fill_(-1)
            graph.replay()
            projected = x.cpu().double() @ w.cpu().double().T * scale
            expected_activation = torch.nn.functional.silu(projected[:, :LOWRANK])
            torch.testing.assert_close(
                storage.view(dtype)[:rows].cpu(),
                expected_activation.to(dtype),
                rtol=0.0,
                atol=0.002,
            )
            expected = _mix_reference(x, w, u, scale)
            tolerance = 0.04 if dtype == torch.bfloat16 else 0.008
            torch.testing.assert_close(
                actual[0], expected[0], rtol=tolerance, atol=tolerance
            )
            if has_inject:
                torch.testing.assert_close(actual[1], expected[1], rtol=0.0, atol=0.0)
            else:
                assert actual[1] is None
    finally:
        pdl_enabled(previous)


@pytest.mark.parametrize("rows", [1, 7, 16])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("has_inject", [False, True])
def test_fused_distributed_reduce_applies_activation_after_complete_sum(
    rows, dtype, has_inject
):
    from tokenspeed_kernel.ops.residual.cute_fused import (
        _persistent_workspace,
        supports_fused_hc,
    )

    if not supports_fused_hc(torch.device("cuda")):
        pytest.skip("requires six resident Blackwell clusters and CuTe")
    x, w, u = _inputs(rows, dtype, seed=5711 + rows)
    if not has_inject:
        w = w[:LOWRANK]
    x.fill_(1.0)
    w.zero_()
    # One exact nonzero per K slice. Large alternating partials cancel to a
    # signed small sum, making SiLU-before-reduction observably incorrect.
    pattern = torch.tensor(
        [8, -8, 4, -4, 2, -2, 1, -1, 8, -8, 4, -4, 2, -2, 1, -0.5],
        device="cuda",
        dtype=dtype,
    )
    columns = (torch.arange(w.shape[0], device="cuda") % 13 - 6).to(dtype)
    w[:, ::640] = columns[:, None] * pattern[None, :]
    scale = 0.25
    previous = pdl_enabled(None)
    try:
        pdl_enabled(True)
        storage, epochs = _persistent_workspace(x.device, w.shape[0])
        storage.fill_(-1)
        actual = gated_residual_mix(
            x,
            w,
            u,
            HC_COUNT,
            HIDDEN_SIZE,
            LOWRANK,
            projection_scale=scale,
            weights_independent=True,
            solution=None,
            override="cute_fused_hyperconnection_mix",
        )
        projected = x.cpu().double() @ w.cpu().double().T * scale
        expected_activation = torch.nn.functional.silu(projected[:, :LOWRANK])
        typed = storage.view(dtype)[:rows]
        torch.testing.assert_close(
            typed.cpu(), expected_activation.to(dtype), rtol=0.0, atol=0.002
        )
        expected = _mix_reference(x, w, u, scale)
        tolerance = 0.04 if dtype == torch.bfloat16 else 0.008
        torch.testing.assert_close(
            actual[0], expected[0], rtol=tolerance, atol=tolerance
        )
        if has_inject:
            torch.testing.assert_close(actual[1], expected[1], rtol=0.0, atol=0.0)
        else:
            assert actual[1] is None
        assert epochs.numel() == (w.shape[0] + 63) // 64
        assert len(set(epochs.cpu().tolist())) == 1
    finally:
        pdl_enabled(previous)
