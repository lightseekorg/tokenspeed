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

"""HC numerical, dispatch and graph regressions with per-case state isolation."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
import torch
from tokenspeed_kernel import (
    gated_residual_combine,
    gated_residual_combine_norm,
    gated_residual_mix,
    grouped_gemma_rmsnorm,
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


def _fused_diagnostic_workspace(device: torch.device, projection_rows: int):
    from tokenspeed_kernel.ops.residual.cute_fused import _workspace_for_plan

    return _workspace_for_plan(
        device,
        projection_rows,
        1,
        (projection_rows + 63) // 64,
        16,
    )


@pytest.fixture(autouse=True)
def restore_kernel_state():
    previous_pdl = pdl_enabled()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    capture = ShapeCapture.get()
    previous_capture = capture.enabled
    yield
    torch.cuda.synchronize()
    pdl_enabled(previous_pdl)
    torch.use_deterministic_algorithms(
        previous_deterministic, warn_only=previous_warn_only
    )
    capture.enabled = previous_capture
    capture.clear()


def _require_fused_hc() -> None:
    from tokenspeed_kernel.ops.residual.cute_fused import supports_fused_hc

    if not supports_fused_hc(torch.device("cuda")):
        pytest.skip("requires six resident Blackwell clusters and CuTe")


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
    tl.extra.cuda.gdc_wait()
    tl.extra.cuda.gdc_launch_dependents()
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
    rows: int, dtype: torch.dtype, *, seed: int
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


def _mix(
    inputs: Sequence[torch.Tensor],
    *,
    override: str | None,
    projection_scale: float,
    weights_independent: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Use the production HC4/H2560/R320 dimensions with explicit dispatch options."""
    return gated_residual_mix(
        *inputs,
        HC_COUNT,
        HIDDEN_SIZE,
        LOWRANK,
        override=override,
        solution=None,
        projection_scale=projection_scale,
        weights_independent=weights_independent,
    )


def _mix_reference(
    normalized: torch.Tensor,
    projection: torch.Tensor,
    up: torch.Tensor,
    projection_scale: float,
) -> tuple[torch.Tensor, torch.Tensor | None]:
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
    inject = (
        (projected[:, LOWRANK:] * projection_scale).to(
            device=device, dtype=normalized.dtype
        )
        if projection.shape[0] > LOWRANK
        else None
    )
    return (
        mixed.to(device=device, dtype=normalized.dtype),
        inject,
    )


def _assert_mix_close(
    actual: tuple[torch.Tensor, torch.Tensor | None],
    inputs: Sequence[torch.Tensor],
    projection_scale: float,
    tolerance: float,
) -> None:
    expected = _mix_reference(*inputs, projection_scale)
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("rows", [0, 1, 4, 8, 16, 24, 32, 128])
def test_general_triton_mix_matches_fp64_reference(rows: int) -> None:
    normalized, projection, up = _inputs(rows, torch.bfloat16, seed=17)
    actual, actual_inject = _mix(
        (normalized, projection, up),
        override="triton_hyperconnection_mix",
        projection_scale=1.0,
        weights_independent=False,
    )
    assert actual.shape == (rows, HIDDEN_SIZE)
    assert actual_inject.shape == (rows, HC_COUNT)
    if rows == 0:
        return
    _assert_mix_close((actual, actual_inject), (normalized, projection, up), 1.0, 0.03)


@pytest.mark.parametrize(
    (
        "rows",
        "dtype",
        "has_inject",
        "enable_pdl",
        "weights_independent",
        "projection_scale",
    ),
    [
        (1, torch.bfloat16, True, False, False, 0.25),
        (4, torch.float16, False, True, True, 1.0),
        (8, torch.bfloat16, False, True, True, 1.0),
        (9, torch.float16, True, False, True, 0.25),
        (16, torch.bfloat16, True, True, False, 1.0),
    ],
)
def test_fused_cute_mix_matches_fp64_and_graph(
    rows: int,
    dtype: torch.dtype,
    has_inject: bool,
    enable_pdl: bool,
    weights_independent: bool,
    projection_scale: float,
) -> None:
    _require_fused_hc()
    normalized, projection, up = _inputs(rows, dtype, seed=3450 + rows)
    if not has_inject:
        projection = projection[:LOWRANK]
    pdl_enabled(enable_pdl)

    def mix() -> tuple[torch.Tensor, torch.Tensor | None]:
        return _mix(
            (normalized, projection, up),
            override="cute_fused_hyperconnection_mix",
            projection_scale=projection_scale,
            weights_independent=weights_independent,
        )

    tolerance = 4e-2 if dtype is torch.bfloat16 else 8e-3
    expected = _mix_reference(normalized, projection, up, projection_scale)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        eager = mix()
    stream.synchronize()
    torch.testing.assert_close(eager, expected, rtol=tolerance, atol=tolerance)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        outputs = [mix() for _ in range(3)]
    for _ in range(5):
        graph.replay()
    for actual in outputs:
        torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)


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


@pytest.mark.parametrize(
    ("dtype", "shared_weight", "preload_residual", "rows", "hc_count", "hidden_size"),
    [
        (torch.bfloat16, False, False, 0, 4, 2560),
        (torch.bfloat16, True, True, 1, 4, 2560),
        (torch.float16, False, True, 17, 4, 2560),
        (torch.float32, True, False, 3, 3, 13),
        (torch.bfloat16, False, False, 128, 4, 2560),
    ],
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

    torch.testing.assert_close(
        (combined, normalized, residual),
        (reference, reference_norm, original),
        rtol=0,
        atol=0,
    )
    assert combined.is_contiguous() and normalized.is_contiguous()


@pytest.mark.parametrize(
    ("field", "invalid"),
    [("block_output", "shape"), ("inject_logits", "device"), ("weight", "dtype")],
)
def test_combine_norm_validates_inputs(field: str, invalid: str) -> None:
    rows = 3
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


@pytest.mark.parametrize(
    ("enable_pdl", "preload_residual"),
    [(False, False), (True, False), (True, True)],
)
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
        torch.testing.assert_close(
            outputs, (reference, reference_norm, reference_consumed), rtol=0, atol=0
        )


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


@pytest.mark.parametrize(
    ("rows", "enable_pdl", "backend"),
    [
        (1, False, "triton_hyperconnection_mix"),
        (8, True, "triton_hyperconnection_mix"),
        (4, False, "cute_fused_hyperconnection_mix"),
        (16, True, "cute_fused_hyperconnection_mix"),
        (33, True, "cute_fused_hyperconnection_mix"),
        (1024, True, "cute_fused_hyperconnection_mix"),
    ],
)
def test_hc_full_chain_cuda_graph_replays(
    rows: int, enable_pdl: bool, backend: str
) -> None:
    if not current_platform().is_nvidia:
        pytest.skip("CUDA graph coverage requires NVIDIA")
    if enable_pdl and not current_platform().is_hopper_plus:
        pytest.skip("PDL requires NVIDIA Hopper or newer")
    residual, projection, up = _inputs(rows, torch.bfloat16, seed=17)
    if backend == "cute_fused_hyperconnection_mix":
        _require_fused_hc()
    generator = torch.Generator(device="cuda").manual_seed(71)
    norm_weight = (
        torch.randn(WIDE, dtype=torch.bfloat16, device="cuda", generator=generator)
        * 0.02
    )

    def chain(value: torch.Tensor) -> tuple[torch.Tensor, ...]:
        normalized = grouped_gemma_rmsnorm(
            value, norm_weight, HIDDEN_SIZE, 1e-6, out=None
        )
        mixed, inject = _mix(
            (normalized, projection, up),
            override=backend,
            projection_scale=1.0,
            weights_independent=backend == "cute_fused_hyperconnection_mix",
        )
        combined = gated_residual_combine(
            mixed, value, inject, HC_COUNT, HIDDEN_SIZE, override=None, solution=None
        )
        return normalized, mixed, inject, combined

    assert pdl_enabled(enable_pdl) is enable_pdl
    # Compile every PDL variant and populate the shared layout workspace.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        chain(residual)
    stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
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
    torch.testing.assert_close(normalized, expected_normalized, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(
        (mixed, inject, combined),
        (expected_mixed, expected_inject, expected_combined),
        rtol=0.04,
        atol=0.04,
    )


def test_general_fallback_supports_deterministic_mode() -> None:
    if not current_platform().is_nvidia:
        pytest.skip("CUDA graph coverage requires NVIDIA")
    normalized, projection, up = _inputs(8, torch.bfloat16, seed=61)
    capture = ShapeCapture.get()
    capture.clear()
    capture.enabled = True
    torch.use_deterministic_algorithms(True, warn_only=False)
    _mix(
        (normalized, projection, up),
        override=None,
        projection_scale=1.0,
        weights_independent=False,
    )
    torch.cuda.synchronize()
    assert capture._records[-1].kernel_name == "triton_hyperconnection_mix"


def test_hc_mix_has_only_fused_and_general_implementations() -> None:
    from tokenspeed_kernel.ops.residual import cute_fused

    expected = {"triton_hyperconnection_mix"}
    if cute_fused._AVAILABLE:
        expected.add("cute_fused_hyperconnection_mix")
    kernels = KernelRegistry.get().list_kernels("residual", "hyperconnection_mix")
    assert {kernel.name for kernel in kernels} == expected


@pytest.mark.parametrize("reason", ["dtype", "strides", "capacity", "unavailable"])
def test_unsupported_fused_contract_uses_general_fallback(monkeypatch, reason) -> None:
    from tokenspeed_kernel.ops.residual import cute_fused

    dtype = torch.float32 if reason == "dtype" else torch.bfloat16
    values = list(_inputs(4, dtype, seed=73))
    if reason == "strides":
        up = values[2]
        storage = up.new_empty((WIDE, LOWRANK * 2))
        values[2] = storage[:, ::2].copy_(up)
    elif reason == "capacity":
        monkeypatch.setattr(cute_fused, "_resident_clusters", lambda index, stream: 5)
    elif reason == "unavailable":
        monkeypatch.setattr(cute_fused, "_AVAILABLE", False)
    capture = ShapeCapture.get()
    capture.enabled = True
    actual = _mix(values, override=None, projection_scale=1.0, weights_independent=True)
    assert capture._records[-1].kernel_name == "triton_hyperconnection_mix"
    _assert_mix_close(actual, values, 1.0, 0.04)


@triton.jit
def _publish_mix_inputs_kernel(
    sources,
    targets,
    sizes: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tl.extra.cuda.gdc_wait()
    # Let mix prefetch old contents while this producer publishes new inputs.
    tl.extra.cuda.gdc_launch_dependents()
    for tensor in tl.static_range(len(sizes)):
        for start in range(
            tl.program_id(0) * BLOCK, sizes[tensor], tl.num_programs(0) * BLOCK
        ):
            indices = start + tl.arange(0, BLOCK)
            mask = indices < sizes[tensor]
            value = tl.load(sources[tensor] + indices, mask=mask, other=0.0)
            tl.store(targets[tensor] + indices, value, mask=mask)


@pytest.mark.parametrize(
    ("rows", "dtype", "has_inject", "backend", "weights_independent", "scale"),
    [
        (4, torch.bfloat16, True, "cute_fused", False, 0.25),
        (1, torch.float16, False, "cute_fused", True, 0.25),
        (16, torch.bfloat16, True, "cute_fused", True, 0.25),
        (33, torch.bfloat16, True, "cute_fused", False, 0.25),
        (33, torch.bfloat16, True, "cute_fused", True, 0.25),
        (1024, torch.bfloat16, True, "cute_fused", False, 0.25),
        (1024, torch.bfloat16, True, "cute_fused", True, 0.25),
    ],
)
def test_mix_prefetch_observes_pdl_producer_updates(
    rows: int,
    dtype: torch.dtype,
    has_inject: bool,
    backend: str,
    weights_independent: bool,
    scale: float,
) -> None:
    if not current_platform().is_hopper_plus:
        pytest.skip("bulk weight prefetch and PDL require NVIDIA Hopper or newer")
    if backend == "cute_fused":
        _require_fused_hc()

    x_source, projection_source, up_source = _inputs(rows, dtype, seed=139 + rows)
    if not has_inject:
        projection_source = projection_source[:LOWRANK]
    projection_source.mul_(2.0)
    up_source.mul_(4.0)
    normalized = torch.full_like(x_source, float("nan"))
    projection = (
        projection_source.clone()
        if weights_independent
        else torch.full_like(projection_source, float("nan"))
    )
    up = (
        up_source.clone()
        if weights_independent
        else torch.full_like(up_source, float("nan"))
    )
    sources = (projection_source, up_source, x_source)
    targets = (projection, up, normalized)
    sizes = tuple(
        0 if weights_independent and index < 2 else tensor.numel()
        for index, tensor in enumerate(targets)
    )
    num_ctas = torch.cuda.get_device_properties(normalized.device).multi_processor_count

    def publish_and_mix() -> tuple[torch.Tensor, torch.Tensor | None]:
        _publish_mix_inputs_kernel[(num_ctas,)](
            sources,
            targets,
            sizes,
            BLOCK=1024,
            num_warps=4,
            launch_pdl=True,
        )
        return _mix(
            (normalized, projection, up),
            override=f"{backend}_hyperconnection_mix",
            projection_scale=scale,
            weights_independent=weights_independent,
        )

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
        normalized.fill_(float("nan"))
        if weights_independent:
            projection.copy_(projection_source)
            up.copy_(up_source)
        else:
            projection.fill_(float("nan"))
            up.fill_(float("nan"))
        graph.replay()
        expected = _mix_reference(x_source, projection_source, up_source, scale)
        tolerance = 4e-2 if dtype is torch.bfloat16 else 8e-3
        for actual in outputs:
            torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize(
    ("rows", "enable_pdl", "weights_independent"),
    [(1, True, True), (16, False, True), (9, True, False), (24, True, True)],
)
def test_fused_cute_dispatch_requires_explicit_weight_contract(
    rows, enable_pdl, weights_independent
):
    _require_fused_hc()
    x, w, u = _inputs(rows, torch.bfloat16, seed=4511)
    capture = ShapeCapture.get()
    pdl_enabled(enable_pdl)
    capture.enabled = True
    capture.clear()
    actual = _mix(
        (x, w, u),
        override=None,
        projection_scale=1.0,
        weights_independent=weights_independent,
    )
    expected_name = (
        "cute_fused_hyperconnection_mix"
        if weights_independent
        else "triton_hyperconnection_mix"
    )
    assert capture._records[-1].kernel_name == expected_name
    _assert_mix_close(actual, (x, w, u), 1.0, 0.04)


@pytest.mark.parametrize(("input_index", "offset"), [(0, 1), (1, 1), (2, 1), (0, 8)])
def test_fused_cute_dispatch_checks_all_tma_alignments(input_index, offset):
    _require_fused_hc()
    values = list(_inputs(4, torch.bfloat16, seed=4731))
    original = values[input_index]
    storage = torch.empty(
        original.numel() + offset, device="cuda", dtype=original.dtype
    )
    values[input_index] = storage[offset:].view_as(original).copy_(original)
    capture = ShapeCapture.get()
    capture.enabled = True
    capture.clear()
    actual = _mix(values, override=None, projection_scale=1.0, weights_independent=True)
    expected_name = (
        "cute_fused_hyperconnection_mix"
        if offset == 8
        else "triton_hyperconnection_mix"
    )
    assert capture._records[-1].kernel_name == expected_name
    _assert_mix_close(actual, values, 1.0, 0.04)


@pytest.mark.parametrize("has_inject", [False, True])
def test_fused_cute_graphs_share_ordered_generations(has_inject):
    _require_fused_hc()
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    graphs = []
    outputs = []
    expected = []
    counters = []
    input_banks = []
    projection_rows = LOWRANK + (HC_COUNT if has_inject else 0)
    clusters = (projection_rows + 63) // 64
    # Graphs with odd call counts, mixed rows and retained outputs exercise the
    # shared workspace key and on-device generation selection.
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
                _mix(
                    (x, w, u),
                    override="cute_fused_hyperconnection_mix",
                    projection_scale=1.0,
                    weights_independent=True,
                )
            raw, count = _fused_diagnostic_workspace(
                values[0][0].device, projection_rows
            )
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
                _mix(
                    (x, w, u),
                    override="cute_fused_hyperconnection_mix",
                    projection_scale=1.0,
                    weights_independent=True,
                )
                for x, w, u in values
            ]
        graphs.append(graph)
        outputs.append(result)
        expected.append([_mix_reference(x, w, u, 1.0) for x, w, u in values])
        counters.append(count)
    assert counters[0].data_ptr() == counters[1].data_ptr()
    for _ in range(5):
        for stream, graph in zip(streams, graphs):
            with torch.cuda.stream(stream):
                graph.replay()
            # The shared persistent epochs require an ordering edge between
            # streams, matching the runtime's single model execution lane.
            stream.synchronize()
    torch.cuda.synchronize()
    for results, refs in zip(outputs, expected):
        torch.testing.assert_close(results, refs, rtol=0.04, atol=0.04)
    counter = counters[0]
    assert counter[:clusters].cpu().tolist() == [generation + 30] * clusters
    # Single-microtile invocations do not recycle their slot, so the separate
    # consumer-completion epochs remain untouched.
    assert counter[clusters:].cpu().tolist() == [generation] * clusters


@pytest.mark.parametrize(("rows", "dtype"), [(4, torch.bfloat16), (16, torch.float16)])
def test_fused_cluster_mix_is_deterministic(rows, dtype):
    _require_fused_hc()
    values = _inputs(rows, dtype, seed=5301 + rows)
    torch.use_deterministic_algorithms(True, warn_only=False)
    outputs = [
        _mix(values, override=None, projection_scale=1.0, weights_independent=True)
        for _ in range(8)
    ]
    expected = _mix_reference(*values, 1.0)
    tolerance = 0.04 if dtype == torch.bfloat16 else 0.008
    torch.testing.assert_close(outputs[0], expected, rtol=tolerance, atol=tolerance)
    for result in outputs[1:]:
        for got, want in zip(result, outputs[0]):
            assert torch.equal(got, want)


def test_fused_cluster_grid_checks_cluster_capacity(monkeypatch):
    from tokenspeed_kernel.ops.residual import cute_fused

    _require_fused_hc()
    # A large SM count alone does not establish cluster residency, for example
    # when a stream belongs to a reduced green context.
    monkeypatch.setattr(cute_fused, "_resident_clusters", lambda index, stream: 5)
    assert not cute_fused.supports_fused_hc(torch.device("cuda"))


def test_fused_cluster_split_k_does_not_exceed_sixteen():
    _require_fused_hc()
    from tokenspeed_kernel.thirdparty.cute_dsl.hc_fused import FusedGatedResidualKernel

    with pytest.raises(ValueError, match="split-K"):
        FusedGatedResidualKernel(
            projection_tile=64,
            token_tile=8,
            projection_rows=324,
            split_k=32,
            projection_tiles=1,
            batch_tiles=1,
            workers=1,
            stages=2,
            final_tile=32,
            rounds=1,
            use_pdl=True,
            scale=1.0,
            weights_independent=True,
            single_tile=True,
            full_tiles=True,
        )


@pytest.mark.parametrize(
    ("rows", "dtype", "has_inject", "down_stages"),
    [
        (1, torch.bfloat16, False, 2),
        (4, torch.float16, True, 2),
        (8, torch.bfloat16, True, 5),
        (4, torch.float16, False, 5),
    ],
)
def test_fused_down_consumes_every_k128_stage_on_graph_replay(
    rows, dtype, has_inject, down_stages, monkeypatch
):
    from tokenspeed_kernel.ops.residual import cute_fused

    _require_fused_hc()

    # Two buffers must recycle and wrap their phases while consuming all five
    # K128 tiles; five buffers must consume the same data without recycling.
    # Isolate compilation because the public dispatch key fixes the tactic.
    monkeypatch.setattr(
        cute_fused,
        "_tactic",
        lambda rows, projection_rows: (
            16,
            64,
            8 if rows <= 8 else 16,
            1,
            1,
            down_stages,
            32,
        ),
    )
    monkeypatch.setattr(cute_fused, "_PLANS", {})
    monkeypatch.setattr(cute_fused, "_CAPACITIES", {})
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
        return _mix(
            (x, w, u),
            override="cute_fused_hyperconnection_mix",
            projection_scale=scale,
            weights_independent=True,
        )

    pdl_enabled(True)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        call()
        storage, _ = _fused_diagnostic_workspace(x.device, w.shape[0])
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
            w[:, offsets + 127] = columns[:, None] * split_pattern.flip(0)[None, :] / 32
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
        torch.testing.assert_close(actual[1], expected[1], rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    ("rows", "dtype", "has_inject"),
    [
        (1, torch.bfloat16, False),
        (7, torch.float16, True),
        (16, torch.bfloat16, True),
        (16, torch.float16, False),
    ],
)
def test_fused_distributed_reduce_applies_activation_after_complete_sum(
    rows, dtype, has_inject
):
    _require_fused_hc()
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
    pdl_enabled(True)
    storage, epochs = _fused_diagnostic_workspace(x.device, w.shape[0])
    storage.fill_(-1)
    actual = _mix(
        (x, w, u),
        override="cute_fused_hyperconnection_mix",
        projection_scale=scale,
        weights_independent=True,
    )
    projected = x.cpu().double() @ w.cpu().double().T * scale
    expected_activation = torch.nn.functional.silu(projected[:, :LOWRANK])
    typed = storage.view(dtype)[:rows]
    torch.testing.assert_close(
        typed.cpu(), expected_activation.to(dtype), rtol=0.0, atol=0.002
    )
    expected = _mix_reference(x, w, u, scale)
    tolerance = 0.04 if dtype == torch.bfloat16 else 0.008
    torch.testing.assert_close(actual[0], expected[0], rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(actual[1], expected[1], rtol=0.0, atol=0.0)
    assert epochs.numel() == 2 * ((w.shape[0] + 63) // 64)
    clusters = (w.shape[0] + 63) // 64
    assert len(set(epochs[:clusters].cpu().tolist())) == 1
