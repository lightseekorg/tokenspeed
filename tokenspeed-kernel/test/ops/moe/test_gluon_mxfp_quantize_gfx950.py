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
from utils import is_cdna4

if not is_cdna4():
    pytest.skip(
        "AMD CDNA4 is required for Gluon MXFP quantization tests",
        allow_module_level=True,
    )

import tokenspeed_kernel  # noqa: E402
from tokenspeed_kernel_amd._triton import gl, gluon  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused._layouts import (  # noqa: E402
    _mxfp4_swiglu_reduce,
    _swiglu_reduce,
    get_mfma_layout,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.quantize_gluon import (  # noqa: E402
    _mxfp4_quantize_tile,
    _mxfp4_quantize_tile_in_layout,
    _mxfp8_quantize_tile,
    quantize_mxfp8_sorted_routes,
    scaled_downcast_layout,
)


def _unswizzle_cdna4_route_scales(
    scales: torch.Tensor,
    *,
    rows: int,
    cols: int,
) -> torch.Tensor:
    logical_m = torch.arange(rows, device=scales.device)[:, None]
    logical_k = torch.arange(cols, device=scales.device)[None, :]
    m_in_block = logical_m % 32
    m_hi = m_in_block // 16
    m_lo = m_in_block % 16
    k_block = logical_k // 8
    k_hi = (logical_k % 8) // 4
    k_lo = logical_k % 4
    swizzled_k = (((k_block * 4 + k_lo) * 16 + m_lo) * 2 + k_hi) * 2 + m_hi
    offsets = swizzled_k + (logical_m // 32) * (cols * 32)
    return scales.view(torch.uint8).flatten()[offsets]


def test_sorted_route_mxfp8_quantization_matches_standard_gfx950() -> None:
    generator = torch.Generator(device="cuda").manual_seed(20260830)
    hidden_states = torch.randn(
        (7, 256),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    source_rows = torch.tensor([5, 1, 6, 0, 3], dtype=torch.int32, device="cuda")
    slots = torch.tensor([2, 4, 1, 0, 3], dtype=torch.int32, device="cuda")
    valid_rows = int(source_rows.numel())
    sorted_ids = torch.full((32,), 7, dtype=torch.int32, device="cuda")
    sorted_ids[:valid_rows] = source_rows | (slots << 24)
    num_valid_ids = torch.tensor([valid_rows], dtype=torch.int32, device="cuda")

    actual, actual_scales = quantize_mxfp8_sorted_routes(
        hidden_states,
        sorted_ids,
        num_valid_ids,
    )
    expected, expected_scales = tokenspeed_kernel.quantize_mxfp8(
        hidden_states[source_rows.long()],
        solution="triton",
    )

    torch.testing.assert_close(
        actual[:valid_rows].view(torch.uint8),
        expected.view(torch.uint8),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        _unswizzle_cdna4_route_scales(
            actual_scales,
            rows=valid_rows,
            cols=hidden_states.shape[1] // 32,
        ),
        expected_scales.view(torch.uint8),
        atol=0,
        rtol=0,
    )


@gluon.jit
def _quantize_tile_probe_kernel(
    x_ptr,
    out_ptr,
    scale_ptr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    OUTPUT_FP8: gl.constexpr,
):
    """Run one quantize tile over a dense BLOCK_M x BLOCK_N tile."""

    layout: gl.constexpr = scaled_downcast_layout(BLOCK_M, BLOCK_N, gl.num_warps())
    rows = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, layout))
    cols = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, layout))
    values = gl.load(x_ptr + rows[:, None] * BLOCK_N + cols[None, :])

    if OUTPUT_FP8:
        quantized, scale_byte = _mxfp8_quantize_tile(values)
        OUT_COLS: gl.constexpr = BLOCK_N
    else:
        quantized, scale_byte = _mxfp4_quantize_tile(values)
        OUT_COLS: gl.constexpr = BLOCK_N // 2

    out_layout: gl.constexpr = quantized.type.layout
    out_rows = gl.arange(0, BLOCK_M, gl.SliceLayout(1, out_layout))
    out_cols = gl.arange(0, OUT_COLS, gl.SliceLayout(0, out_layout))
    gl.store(out_ptr + out_rows[:, None] * OUT_COLS + out_cols[None, :], quantized)

    scale_layout: gl.constexpr = scale_byte.type.layout
    scale_rows = gl.arange(0, BLOCK_M, gl.SliceLayout(1, scale_layout))
    scale_cols = gl.arange(0, BLOCK_N // 32, gl.SliceLayout(0, scale_layout))
    gl.store(
        scale_ptr + scale_rows[:, None] * (BLOCK_N // 32) + scale_cols[None, :],
        scale_byte,
    )


def _quantize_tile_probe(
    hidden_states: torch.Tensor,
    *,
    output_fp8: bool,
):
    """Return the quantized tile, its scales, and the compiled kernel."""

    block_m, block_n = hidden_states.shape
    out_cols = block_n if output_fp8 else block_n // 2
    out_dtype = torch.float8_e4m3fn if output_fp8 else torch.uint8
    out = torch.empty((block_m, out_cols), dtype=out_dtype, device="cuda")
    scales = torch.empty((block_m, block_n // 32), dtype=torch.uint8, device="cuda")
    compiled = _quantize_tile_probe_kernel[(1,)](
        hidden_states,
        out,
        scales,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        OUTPUT_FP8=output_fp8,
        num_warps=4,
    )
    return out, scales, compiled


def _quantize_tile_probe_input() -> torch.Tensor:
    """A tile spanning normal, tiny, tie-rounded, and all-zero groups."""

    generator = torch.Generator(device="cuda").manual_seed(271828)
    values = torch.randn(
        (32, 256),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    values[0] = 0.0
    for row, exponent in enumerate((-100, -120, -125, -126, -127, -130, -133), 1):
        values[row] = torch.full_like(values[row], 2.0**exponent)
    # Ties between two E2M1 codes must round half-to-even the same way.
    values[8] = torch.tensor(
        [0.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0] * 32,
        dtype=torch.bfloat16,
        device="cuda",
    )
    return values


@pytest.mark.parametrize("output_fp8", [False, True])
def test_quantize_tile_matches_reference_quantizer_gfx950(output_fp8: bool) -> None:
    values = _quantize_tile_probe_input()
    actual, actual_scales, _ = _quantize_tile_probe(values, output_fp8=output_fp8)

    if output_fp8:
        expected, expected_scales = tokenspeed_kernel.quantize_mxfp8(
            values,
            enable_pdl=False,
            override=None,
            solution="triton",
        )
    else:
        expected, expected_scales = tokenspeed_kernel.quantize_mxfp4(
            values,
            global_scale=None,
            scale_size=32,
            scale_layout="linear",
            enable_pdl=False,
            override=None,
            solution="triton",
        )

    torch.testing.assert_close(
        actual.view(torch.uint8),
        expected.view(torch.uint8),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        actual_scales,
        expected_scales.view(torch.uint8),
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize("output_fp8", [False, True])
def test_quantize_tile_uses_hardware_scaled_downcast_gfx950(output_fp8: bool) -> None:
    """The tiles must lower to v_cvt_scalef32_*, not the software bit path."""

    values = _quantize_tile_probe_input()
    _, _, compiled = _quantize_tile_probe(values, output_fp8=output_fp8)
    amdgcn = compiled.asm["amdgcn"]
    suffix = "fp8" if output_fp8 else "fp4"
    assert f"v_cvt_scalef32_pk_{suffix}_f32" in amdgcn


@gluon.jit
def _swiglu_quantize_layout_probe_kernel(
    x_ptr,
    out_ptr,
    scale_ptr,
    BLOCK_M: gl.constexpr,
    BLOCK_N_FULL: gl.constexpr,
    OPTIMIZED: gl.constexpr,
):
    """Quantize a SwiGLU tile through the reference or optimized layout path."""

    OUT_BLOCK_N: gl.constexpr = BLOCK_N_FULL // 2
    acc_layout: gl.constexpr = get_mfma_layout(
        gl.num_warps(),
        True,
        scale_preshuffle=False,
        block_m=BLOCK_M,
        w_via_vgpr=False,
    )
    rows = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, acc_layout))
    cols = gl.arange(0, BLOCK_N_FULL, layout=gl.SliceLayout(0, acc_layout))
    acc = gl.load(x_ptr + rows[:, None] * BLOCK_N_FULL + cols[None, :]).to(gl.float32)
    if OPTIMIZED:
        activated = _mxfp4_swiglu_reduce(
            acc,
            1.0,
            0.0,
            0.0,
            OUT_BLOCK_N,
        )
        quantized, scale_byte = _mxfp4_quantize_tile_in_layout(activated)
    else:
        activated = _swiglu_reduce(
            acc,
            1.0,
            0.0,
            0.0,
            OUT_BLOCK_N,
        )
        quantized, scale_byte = _mxfp4_quantize_tile(activated)
    quantized = quantized.reshape((BLOCK_M, OUT_BLOCK_N // 2))

    out_layout: gl.constexpr = quantized.type.layout
    out_rows = gl.arange(0, BLOCK_M, gl.SliceLayout(1, out_layout))
    out_cols = gl.arange(0, OUT_BLOCK_N // 2, gl.SliceLayout(0, out_layout))
    gl.store(
        out_ptr + out_rows[:, None] * (OUT_BLOCK_N // 2) + out_cols[None, :],
        quantized,
    )

    scale_layout: gl.constexpr = scale_byte.type.layout
    scale_rows = gl.arange(0, BLOCK_M, gl.SliceLayout(1, scale_layout))
    scale_cols = gl.arange(0, OUT_BLOCK_N // 32, gl.SliceLayout(0, scale_layout))
    gl.store(
        scale_ptr + scale_rows[:, None] * (OUT_BLOCK_N // 32) + scale_cols[None, :],
        scale_byte,
    )


def _swiglu_quantize_layout_probe(
    values: torch.Tensor,
    *,
    optimized: bool,
    num_warps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    block_m, block_n_full = values.shape
    out_block_n = block_n_full // 2
    out = torch.empty(
        (block_m, out_block_n // 2),
        dtype=torch.uint8,
        device="cuda",
    )
    scales = torch.empty(
        (block_m, out_block_n // 32),
        dtype=torch.uint8,
        device="cuda",
    )
    _swiglu_quantize_layout_probe_kernel[(1,)](
        values,
        out,
        scales,
        BLOCK_M=block_m,
        BLOCK_N_FULL=block_n_full,
        OPTIMIZED=optimized,
        num_warps=num_warps,
    )
    return out, scales


@pytest.mark.parametrize("num_warps", [4, 8])
def test_swiglu_quantize_layout_specialization_is_bit_exact_gfx950(
    num_warps: int,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(161803)
    values = torch.randn(
        (64, 128),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )

    expected, expected_scales = _swiglu_quantize_layout_probe(
        values,
        optimized=False,
        num_warps=num_warps,
    )
    actual, actual_scales = _swiglu_quantize_layout_probe(
        values,
        optimized=True,
        num_warps=num_warps,
    )

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(actual_scales, expected_scales, atol=0, rtol=0)
