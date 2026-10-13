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

"""iHC boundaries against FP64 references on BF16 and FP32 residual channels,
and the operand checks of the public entry points, which need no device."""

from __future__ import annotations

import re

import pytest
import torch
from tokenspeed_kernel.ops.residual import ihc_head, ihc_post, ihc_pre

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="a CUDA/ROCm device is required"
)

MAGNITUDE = 2.0
RMS_EPS = 1e-5
HC_EPS = 1e-6
NORM_EPS = 1e-5
SOLUTIONS = ("torch", "triton")

# (hc_mult, hidden_size, rows). Prefill rows over 256 projection columns take a
# single K split; three channels pad six gates to eight lanes.
SHAPES = [
    pytest.param(4, 1536, 3, id="decode"),
    pytest.param(4, 1536, 300, id="prefill"),
    pytest.param(4, 64, 300, id="single_k_split"),
    pytest.param(3, 512, 37, id="odd_channels"),
]


def _randn(*shape: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(*shape, generator=generator, device="cuda")


def _params(
    hc_mult: int, hidden_size: int, gate_count: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    fn = _randn(gate_count, hc_mult * hidden_size, seed=1) * 0.02
    hc_scale = torch.tensor([0.7, 1.3][: gate_count // hc_mult], device="cuda")
    hc_base = _randn(gate_count, seed=2)
    return fn, hc_scale, hc_base


def _fp64_gates(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    index: int,
) -> torch.Tensor:
    hc_mult = residual.shape[-2]
    flat = residual.double().flatten(1)
    rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + RMS_EPS)
    mixes = (flat @ fn.double().T) * rsqrt
    cols = slice(index * hc_mult, (index + 1) * hc_mult)
    return torch.sigmoid(mixes[:, cols] * float(hc_scale[index]) + hc_base[cols])


def _check_ihc_pre_fp32_residual(
    solution: str, hc_mult: int, hidden_size: int, rows: int
) -> None:
    residual = _randn(rows, hc_mult, hidden_size, seed=rows) * 3
    fn, hc_scale, hc_base = _params(hc_mult, hidden_size, 2 * hc_mult)
    norm_weight = _randn(hidden_size, seed=3).bfloat16()
    pre = _fp64_gates(residual, fn, hc_scale, hc_base, 0) + HC_EPS
    post = MAGNITUDE * _fp64_gates(residual, fn, hc_scale, hc_base, 1) + HC_EPS
    reduced = (pre.unsqueeze(-1) * residual.double()).sum(1)
    rstd = torch.rsqrt(reduced.square().mean(-1, keepdim=True) + NORM_EPS)
    expected = reduced * rstd * norm_weight.double()

    layer_input, gates = ihc_pre(
        residual,
        fn,
        hc_scale,
        hc_base,
        MAGNITUDE,
        RMS_EPS,
        HC_EPS,
        norm_weight=norm_weight,
        norm_eps=NORM_EPS,
        solution=solution,
    )

    assert layer_input.dtype == torch.bfloat16
    torch.testing.assert_close(gates.double(), post, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(layer_input, expected.bfloat16(), rtol=2**-7, atol=1e-2)


@requires_gpu
@pytest.mark.parametrize("solution", SOLUTIONS)
@pytest.mark.parametrize("hc_mult,hidden_size,rows", SHAPES)
def test_ihc_pre_fp32_residual_matches_fp64(
    solution: str, hc_mult: int, hidden_size: int, rows: int
) -> None:
    _check_ihc_pre_fp32_residual(solution, hc_mult, hidden_size, rows)


# HY V4's 24576 projection columns: the most K splits at decode, the longest K
# walk at prefill. The portable path's FP32 GEMM rounds in the BLAS's summation
# order, which over this many columns can exceed the gate bounds.
@requires_gpu
@pytest.mark.parametrize(
    "rows", [pytest.param(4, id="decode"), pytest.param(4096, id="prefill")]
)
def test_triton_ihc_pre_fp32_residual_at_production_k_matches_fp64(rows: int) -> None:
    _check_ihc_pre_fp32_residual("triton", 4, 6144, rows)


@requires_gpu
@pytest.mark.parametrize("solution", SOLUTIONS)
@pytest.mark.parametrize(
    "hc_mult,hidden_size,rows",
    [
        pytest.param(4, 1536, 1, id="decode"),
        pytest.param(4, 1536, 300, id="prefill"),
        pytest.param(4, 64, 300, id="single_k_split"),
        pytest.param(3, 512, 37, id="odd_channels"),
    ],
)
def test_ihc_pre_bf16_residual_matches_fp64(
    solution: str, hc_mult: int, hidden_size: int, rows: int
) -> None:
    residual = (_randn(rows, hc_mult, hidden_size, seed=rows + 1) * 3).bfloat16()
    fn, hc_scale, hc_base = _params(hc_mult, hidden_size, 2 * hc_mult)
    norm_weight = _randn(hidden_size, seed=3)
    pre = _fp64_gates(residual, fn, hc_scale, hc_base, 0) + HC_EPS
    post = MAGNITUDE * _fp64_gates(residual, fn, hc_scale, hc_base, 1) + HC_EPS
    # The norm reads the reduction rounded to the residual dtype and applies
    # the FP32 weight before the single BF16 rounding of the output.
    reduced = (pre.unsqueeze(-1) * residual.double()).sum(1).bfloat16().double()
    rstd = torch.rsqrt(reduced.square().mean(-1, keepdim=True) + NORM_EPS)

    plain, gates = ihc_pre(
        residual,
        fn,
        hc_scale,
        hc_base,
        MAGNITUDE,
        RMS_EPS,
        HC_EPS,
        norm_weight=None,
        norm_eps=None,
        solution=solution,
    )
    normed, _ = ihc_pre(
        residual,
        fn,
        hc_scale,
        hc_base,
        MAGNITUDE,
        RMS_EPS,
        HC_EPS,
        norm_weight=norm_weight,
        norm_eps=NORM_EPS,
        solution=solution,
    )

    torch.testing.assert_close(gates.double(), post, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(plain, reduced.bfloat16(), rtol=2**-7, atol=1e-2)
    torch.testing.assert_close(
        normed,
        (reduced * rstd * norm_weight.double()).bfloat16(),
        rtol=2**-7,
        atol=1e-2,
    )


@requires_gpu
@pytest.mark.parametrize("solution", SOLUTIONS)
def test_ihc_post_keeps_fp32_channels(solution: str) -> None:
    residual = _randn(300, 4, 1536, seed=4) * 3
    block_output = _randn(300, 1536, seed=5).bfloat16()
    gates = _randn(300, 4, seed=6).abs()

    actual = ihc_post(block_output, residual, gates, solution=solution)

    assert actual.dtype == torch.float32
    expected = (
        gates.double().unsqueeze(-1) * block_output.double().unsqueeze(-2)
        + residual.double()
    )
    torch.testing.assert_close(actual.double(), expected, rtol=1e-6, atol=1e-5)


@requires_gpu
@pytest.mark.parametrize("solution", SOLUTIONS)
@pytest.mark.parametrize(
    "hc_mult,hidden_size",
    [pytest.param(4, 1536, id="hy_v4"), pytest.param(3, 512, id="odd_channels")],
)
def test_ihc_head_fp32_residual_matches_fp64(
    solution: str, hc_mult: int, hidden_size: int
) -> None:
    residual = _randn(37, hc_mult, hidden_size, seed=7) * 3
    fn, hc_scale, hc_base = _params(hc_mult, hidden_size, hc_mult)
    gates = _fp64_gates(residual, fn, hc_scale, hc_base, 0) + HC_EPS
    expected = (gates.unsqueeze(-1) * residual.double()).sum(1)

    actual = ihc_head(
        residual, fn, hc_scale, hc_base, RMS_EPS, HC_EPS, solution=solution
    )

    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected.bfloat16(), rtol=2**-7, atol=1e-2)


@requires_gpu
def test_ihc_pre_beyond_the_triton_gate_limit_selects_torch() -> None:
    residual = _randn(5, 9, 64, seed=8) * 3
    fn, hc_scale, hc_base = _params(9, 64, 18)
    norm_weight = _randn(64, seed=3)
    args = (residual, fn, hc_scale, hc_base, MAGNITUDE, RMS_EPS, HC_EPS)
    norm = dict(norm_weight=norm_weight, norm_eps=NORM_EPS)

    selected = ihc_pre(*args, **norm)
    portable = ihc_pre(*args, **norm, solution="torch")

    assert all(torch.equal(a, b) for a, b in zip(selected, portable))


@pytest.mark.parametrize("solution", SOLUTIONS)
def test_ihc_outputs_keep_leading_dimensions(solution: str) -> None:
    if solution == "triton" and not torch.cuda.is_available():
        pytest.skip("the Triton boundaries need a CUDA/ROCm device")
    device = "cuda" if solution == "triton" else "cpu"
    generator = torch.Generator().manual_seed(11)
    hc_mult, hidden_size = 4, 64
    residual = torch.randn(2, 3, hc_mult, hidden_size, generator=generator)
    fn = torch.randn(2 * hc_mult, hc_mult * hidden_size, generator=generator) * 0.02
    hc_base = torch.randn(2 * hc_mult, generator=generator)
    norm_weight = torch.randn(hidden_size, generator=generator)
    residual, fn, hc_base, norm_weight = (
        t.to(device) for t in (residual, fn, hc_base, norm_weight)
    )
    hc_scale = torch.tensor([0.7, 1.3], device=device)
    norm = dict(norm_weight=norm_weight, norm_eps=NORM_EPS, solution=solution)

    layer_input, gates = ihc_pre(
        residual, fn, hc_scale, hc_base, MAGNITUDE, RMS_EPS, HC_EPS, **norm
    )
    flat_input, flat_gates = ihc_pre(
        residual.flatten(0, 1),
        fn,
        hc_scale,
        hc_base,
        MAGNITUDE,
        RMS_EPS,
        HC_EPS,
        **norm,
    )
    updated = ihc_post(layer_input, residual, gates, solution=solution)
    merged = ihc_head(
        residual,
        fn[:hc_mult],
        hc_scale[:1],
        hc_base[:hc_mult],
        RMS_EPS,
        HC_EPS,
        solution=solution,
    )

    assert layer_input.shape == (2, 3, hidden_size)
    assert gates.shape == (2, 3, hc_mult)
    assert torch.equal(layer_input.flatten(0, 1), flat_input)
    assert torch.equal(gates.flatten(0, 1), flat_gates)
    assert updated.shape == residual.shape
    assert merged.shape == (2, 3, hidden_size)


def _pre_operands() -> dict[str, torch.Tensor]:
    return {
        "residual": torch.zeros(3, 4, 8),
        "fn": torch.zeros(8, 32),
        "hc_scale": torch.ones(2),
        "hc_base": torch.zeros(8),
        "norm_weight": torch.ones(8),
    }


_PRE_MISMATCHES = {
    "one_channel": ({"residual": torch.zeros(3, 1, 32)}, "hc_mult > 1"),
    "head_projection": ({"fn": torch.zeros(4, 32)}, "fn must have shape (8, 32)"),
    "strided_projection": ({"fn": torch.zeros(32, 8).t()}, "fn must be contiguous"),
    "head_scale": ({"hc_scale": torch.ones(1)}, "hc_scale must have shape (2,)"),
    "short_bias": ({"hc_base": torch.zeros(4)}, "hc_base must have shape (8,)"),
    "short_norm_weight": (
        {"norm_weight": torch.ones(4)},
        "norm_weight must have shape (8,)",
    ),
    "bias_off_device": (
        {"hc_base": torch.zeros(8, device="meta")},
        "hc_base device must match the residual",
    ),
}


@pytest.mark.parametrize("case", sorted(_PRE_MISMATCHES))
def test_ihc_pre_rejects_mismatched_operands(case: str) -> None:
    changes, message = _PRE_MISMATCHES[case]
    operands = {**_pre_operands(), **changes}
    with pytest.raises(ValueError, match=re.escape(message)):
        ihc_pre(
            operands["residual"],
            operands["fn"],
            operands["hc_scale"],
            operands["hc_base"],
            MAGNITUDE,
            RMS_EPS,
            HC_EPS,
            norm_weight=operands["norm_weight"],
            norm_eps=NORM_EPS,
        )


def _post_operands() -> dict[str, torch.Tensor]:
    return {
        "block_output": torch.zeros(3, 8, dtype=torch.bfloat16),
        "residual": torch.zeros(3, 4, 8),
        "gates": torch.zeros(3, 4),
    }


_POST_MISMATCHES = {
    "one_channel": ({"residual": torch.zeros(3, 1, 8)}, "hc_mult > 1"),
    "short_block_output": (
        {"block_output": torch.zeros(2, 8, dtype=torch.bfloat16)},
        "block_output must have shape (3, 8)",
    ),
    "broadcast_block_output": (
        {"block_output": torch.zeros(1, 8, dtype=torch.bfloat16)},
        "block_output must have shape (3, 8)",
    ),
    "flat_gates": ({"gates": torch.zeros(12)}, "gates must have shape (3, 4)"),
    "gates_off_device": (
        {"gates": torch.zeros(3, 4, device="meta")},
        "gates device must match the residual",
    ),
}


@pytest.mark.parametrize("case", sorted(_POST_MISMATCHES))
def test_ihc_post_rejects_mismatched_operands(case: str) -> None:
    changes, message = _POST_MISMATCHES[case]
    operands = {**_post_operands(), **changes}
    with pytest.raises(ValueError, match=re.escape(message)):
        ihc_post(operands["block_output"], operands["residual"], operands["gates"])


def _head_operands() -> dict[str, torch.Tensor]:
    return {
        "residual": torch.zeros(3, 4, 8),
        "fn": torch.zeros(4, 32),
        "hc_scale": torch.ones(1),
        "hc_base": torch.zeros(4),
    }


_HEAD_MISMATCHES = {
    "one_channel": ({"residual": torch.zeros(3, 1, 32)}, "hc_mult > 1"),
    "pre_projection": ({"fn": torch.zeros(8, 32)}, "fn must have shape (4, 32)"),
    "pre_scale": ({"hc_scale": torch.ones(2)}, "hc_scale must have shape (1,)"),
    "short_bias": ({"hc_base": torch.zeros(3)}, "hc_base must have shape (4,)"),
}


@pytest.mark.parametrize("case", sorted(_HEAD_MISMATCHES))
def test_ihc_head_rejects_mismatched_operands(case: str) -> None:
    changes, message = _HEAD_MISMATCHES[case]
    operands = {**_head_operands(), **changes}
    with pytest.raises(ValueError, match=re.escape(message)):
        ihc_head(
            operands["residual"],
            operands["fn"],
            operands["hc_scale"],
            operands["hc_base"],
            RMS_EPS,
            HC_EPS,
        )
