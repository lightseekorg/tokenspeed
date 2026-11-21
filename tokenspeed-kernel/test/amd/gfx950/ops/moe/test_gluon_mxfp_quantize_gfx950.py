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

from tokenspeed_kernel.numerics.torch.quantization import (  # noqa: E402
    dequantize_mxfp4,
    fake_quantize_mxfp4_even,
)
from tokenspeed_kernel.ops.quantization import quantize_mxfp4 as kernel_quantize_mxfp4
from tokenspeed_kernel.ops.quantization import quantize_mxfp8 as kernel_quantize_mxfp8
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

# noqa: E402


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
    expected, expected_scales = kernel_quantize_mxfp8(
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
        expected, expected_scales = kernel_quantize_mxfp8(
            values,
            enable_pdl=False,
            override=None,
            solution="triton",
        )
    else:
        expected, expected_scales = kernel_quantize_mxfp4(
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


@pytest.mark.parametrize(
    "num_tokens",
    # 32 is the Kimi-K3 decode width at concurrency 8 with EAGLE3, and 64 at
    # concurrency 16. Everything from 49 up used to abort the dispatch with
    # HSA_STATUS_ERROR_OUT_OF_RESOURCES, because the scale was reduced by a
    # single workgroup materialising next_power_of_2(numel) elements; 32 and 48
    # survived only by spilling, at up to 50x the cost of the small widths.
    # 4 and 5 straddle the single-pass bound at hidden 3584 (16,384 vs 32,768 padded).
    [1, 4, 5, 8, 16, 18, 19, 32, 48, 49, 64, 128, 1024],
)
def test_dynamic_fp8_quantize_scales_past_one_workgroup_gfx950(
    num_tokens: int,
) -> None:
    from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused.quantize import (
        _dynamic_fp8_quantize,
    )

    # The routed-expert width Kimi-K3 feeds the MoE (routed_expert_hidden_size).
    hidden = 3584
    generator = torch.Generator(device="cuda").manual_seed(20260914 + num_tokens)
    activations = (
        torch.randn(
            (num_tokens, hidden),
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        * 3.0
    )

    quantized, scale = _dynamic_fp8_quantize(activations)

    assert quantized.shape == activations.shape
    assert quantized.dtype == torch.float8_e4m3fn
    assert scale.shape == (1,)

    # The scale is a property of the whole tensor, so a reduction that missed
    # any block would show up here rather than as a tolerance failure.
    expected_scale = activations.float().abs().max() / 448.0
    torch.testing.assert_close(scale, expected_scale.reshape(1), atol=0.0, rtol=1e-6)

    dequantized = quantized.float() * scale
    torch.testing.assert_close(
        dequantized,
        activations.float(),
        atol=float(expected_scale) * 0.5,
        rtol=6e-2,
    )


def test_dynamic_fp8_quantize_single_pass_bound_matches_spill_cliff() -> None:
    """Pin which strategy each size takes.

    The bound is a measured property of the one-workgroup form under graph
    replay, not a tuning knob: it keeps the Kimi-K3 one-request verify window
    (M=4) on the single launch, 0.9 us cheaper there, and moves every wider
    window onto the grid reduction, which is already ahead at M=5 and 1.9x
    ahead at M=9-18.
    """
    from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused.quantize import (
        _DYNAMIC_FP8_SINGLE_PASS_MAX_ELEMENTS,
        _dynamic_fp8_use_single_pass,
    )

    routed_hidden = 3584

    # One request's EAGLE3 verify window keeps the one-launch form.
    assert _dynamic_fp8_use_single_pass(4 * routed_hidden)
    # 2-4 requests pad to 32,768-65,536, where one workgroup loses to two launches.
    assert not _dynamic_fp8_use_single_pass(5 * routed_hidden)
    assert not _dynamic_fp8_use_single_pass(16 * routed_hidden)
    assert not _dynamic_fp8_use_single_pass(32 * routed_hidden)
    # Concurrency 16 is M=64, where the one-workgroup form used to abort the
    # dispatch outright.
    assert not _dynamic_fp8_use_single_pass(64 * routed_hidden)

    # Exactly on the bound is still single-pass; one element past it is not.
    assert _dynamic_fp8_use_single_pass(_DYNAMIC_FP8_SINGLE_PASS_MAX_ELEMENTS)
    assert not _dynamic_fp8_use_single_pass(_DYNAMIC_FP8_SINGLE_PASS_MAX_ELEMENTS + 1)


def test_linear_mxfp4_quantize_writes_quarks_even_weight_format() -> None:
    """The linear-layout ``quantize_mxfp4`` writes the format of Quark's MXFP4
    weights (Muse Spark encodes its BF16 expert layers with it at load,
    ``--quantize-unquantized-experts mxfp4``): its codes and UE8M0 scales
    decode to the "even"-mode round trip exactly, including ties to the even
    E2M1 code, saturation past 6 and all-zero groups."""
    torch.manual_seed(11)
    rows = torch.randn(64, 4096, device="cuda") * torch.logspace(
        -6, 3, 64, device="cuda"
    ).unsqueeze(1)
    special = torch.zeros(4, 4096, device="cuda")
    # Group 0: amax 6, every E2M1 midpoint (ties) and its neighbours.
    ties = torch.tensor(
        [
            6.0,
            0.25,
            0.75,
            1.25,
            1.75,
            2.5,
            3.5,
            5.0,
            -0.25,
            -0.75,
            -1.25,
            -1.75,
            -2.5,
            -3.5,
            -5.0,
            0.26,
            0.74,
            1.26,
            1.74,
            2.6,
            3.4,
            5.1,
            4.9,
            0.0,
        ],
        device="cuda",
    )
    special[0, : ties.numel()] = ties
    # Group 1: amax just under 1.75 x a power of two and one just past it.
    special[1, :32] = 1.7499
    special[1, 32:64] = 1.7501
    # Groups that saturate: amax 7 x 2^k scales 7 / 4 > 1.75 down to 4.
    special[2] = torch.linspace(-7.0, 7.0, 4096, device="cuda")
    values = torch.cat((rows, special)).to(torch.bfloat16)
    codes, scales = kernel_quantize_mxfp4(values, scale_layout="linear")
    assert codes.dtype == torch.uint8 and codes.shape == (68, 2048)
    assert scales.dtype == torch.uint8 and scales.shape == (68, 128)
    decoded = dequantize_mxfp4(codes, scales)
    expected = fake_quantize_mxfp4_even(values.float())
    assert torch.equal(decoded, expected)


_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _quark_even_mxfp4_bytes(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quark's "even"-mode MXFP4 encoding of ``x`` ([rows, columns], columns a
    multiple of 32), written from its definition, bit by bit: the block amax
    rounded at the E8M0 boundary (+2**21 then truncated to the exponent), the
    scale exponent floor(log2) - 2 clamped to [-127, 127] and stored biased by
    127, E2M1 codes rounded to nearest with ties to the even code and saturated
    at 6, two codes a byte with the even column in the low nibble."""
    values = x.float()
    rows, columns = values.shape
    blocks = values.reshape(rows, columns // 32, 32)
    amax_bits = blocks.abs().amax(-1).contiguous().view(torch.int32).to(torch.int64)
    rounded = ((amax_bits + 0x200000) & 0xFF800000).to(torch.int32).view(torch.float32)
    exponent = (torch.log2(rounded).floor() - 2).clamp(-127, 127)
    scales = (exponent + 127).to(torch.uint8)
    magnitude = (blocks.abs() / torch.exp2(exponent).unsqueeze(-1)).clamp(max=6.0)
    grid = torch.tensor(_E2M1_VALUES, device=x.device)
    distance = (magnitude.unsqueeze(-1) - grid).abs()
    nearest = distance == distance.amin(-1, keepdim=True)
    even = torch.arange(len(_E2M1_VALUES), device=x.device) % 2 == 0
    # A tie sits between an even and an odd code: take the even one.
    pick = torch.where((nearest & even).any(-1, keepdim=True), nearest & even, nearest)
    codes = pick.to(torch.uint8).argmax(-1).to(torch.uint8)
    codes = codes | (torch.signbit(blocks).to(torch.uint8) << 3)
    codes = codes.reshape(rows, columns)
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return packed, scales


def test_linear_mxfp4_quantize_matches_quarks_even_bytes() -> None:
    """The linear-layout ``quantize_mxfp4`` writes Quark's MXFP4 bytes exactly:
    codes and UE8M0 scale bytes equal an independent bit-level reference of the
    even mode, on blocks whose amax spans 2**-20 to 2**10 (negative scale
    exponents included), an all-zero block (scale byte 0), E2M1 ties, and
    saturation past 6."""
    torch.manual_seed(5)
    rows = torch.randn(32, 1024, device="cuda") * torch.logspace(
        -20, 10, 32, base=2.0, device="cuda"
    ).unsqueeze(1)
    special = torch.zeros(3, 1024, device="cuda")
    ties = torch.tensor(
        [6.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -0.25, -1.25, -2.5, -5.0],
        device="cuda",
    )
    special[0, : ties.numel()] = ties
    special[1, :32] = torch.linspace(-40.0, 40.0, 32, device="cuda")  # saturates
    # special[2] stays zero: every block of that row is all-zero.
    x = torch.cat((rows, special)).to(torch.bfloat16)
    codes, scales = kernel_quantize_mxfp4(x, scale_size=32, scale_layout="linear")
    expected_codes, expected_scales = _quark_even_mxfp4_bytes(x)
    assert torch.equal(scales.view(torch.uint8), expected_scales)
    assert torch.equal(codes.view(torch.uint8), expected_codes)
    assert torch.equal(
        scales.view(torch.uint8)[-1], torch.zeros_like(scales[-1]).view(torch.uint8)
    )
