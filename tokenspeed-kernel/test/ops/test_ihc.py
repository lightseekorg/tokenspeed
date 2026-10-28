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

"""iHC numerical boundaries, public operand checks and graph replay."""

from __future__ import annotations

from functools import partial

import pytest
import torch
from tokenspeed_kernel.ops.residual import ihc_head, ihc_post, ihc_pre
from tokenspeed_kernel.ops.residual.torch import (
    torch_ihc_head,
    torch_ihc_post,
    torch_ihc_pre,
)
from utils import assert_no_triton_compile

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="a CUDA/ROCm device is required"
)

MAGNITUDE = 2.0
RMS_EPS = 1e-5
HC_EPS = 1e-6
NORM_EPS = 1e-5
VALUE_RTOL = torch.finfo(torch.bfloat16).eps
VALUE_ATOL = VALUE_RTOL / 8


@pytest.fixture(autouse=True)
def _ieee_fp32_matmul():
    """The independent FP32 oracle must not silently become TF32."""
    precision = torch.get_float32_matmul_precision()
    allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        yield
    finally:
        torch.set_float32_matmul_precision(precision)
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32


def _inputs(
    rows: int,
    hc_mult: int,
    hidden: int,
    dtype: torch.dtype,
    device: str,
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(173)
    residual = torch.randn(rows, hc_mult, hidden, generator=generator) * 3
    fn = torch.randn(2 * hc_mult, hc_mult * hidden, generator=generator) * 0.02
    scale = torch.tensor([0.7, 1.3])
    base = torch.randn(2 * hc_mult, generator=generator)
    norm = torch.randn(hidden, generator=generator)
    return (
        residual.to(device=device, dtype=dtype),
        *(value.to(device) for value in (fn, scale, base, norm)),
    )


def _reference_gates(residual, fn, scale, base, *, dtype):
    channels, hidden = residual.shape[-2:]
    flat = residual.to(dtype).reshape(-1, channels * hidden)
    projected = flat @ fn.to(dtype).T
    projected *= (flat.square().mean(-1, keepdim=True) + RMS_EPS).rsqrt()
    scale, base = scale.to(dtype), base.to(dtype)
    pre = torch.sigmoid(projected[:, :channels] * scale[0] + base[:channels]) + HC_EPS
    post = (
        MAGNITUDE * torch.sigmoid(projected[:, channels:] * scale[1] + base[channels:])
        + HC_EPS
    )
    return (
        pre.reshape(*residual.shape[:-2], channels),
        post.reshape(*residual.shape[:-2], channels),
    )


def _rms_reference(values, weight):
    values = values.double()
    return (
        values
        * (values.square().mean(-1, keepdim=True) + NORM_EPS).rsqrt()
        * weight.double()
    ).bfloat16()


def _reference_pre(residual, fn, scale, base, norm):
    pre, post = _reference_gates(residual, fn, scale, base, dtype=torch.float64)
    reduced = (pre.unsqueeze(-1) * residual.double()).sum(-2).to(residual.dtype)
    return (reduced.bfloat16() if norm is None else _rms_reference(reduced, norm)), post


def _reduction_bounds(residual, gates):
    """FP32 product/sum error, then the specified residual-dtype store.

    For these finite normal fixtures, n rounded operations have relative error
    gamma_n = n*u/(1-n*u), u=eps/2. At most 2*channels products/additions cover
    either separate arithmetic or FMA, without assuming a particular tree.
    Gates have already passed independent FP32/FP64 checks; using the observed
    gates here isolates the channel reduction instead of counting gate error
    again. The FP64 calculation receives its own conservative rounding guard.
    """
    channels = residual.shape[-2]
    products = gates.double().unsqueeze(-1) * residual.double()
    center, magnitude = products.sum(-2), products.abs().sum(-2)
    error = torch.zeros_like(center)
    for dtype in (torch.float32, torch.float64):
        nu = 2 * channels * torch.finfo(dtype).eps / 2
        error += (nu / (1 - nu)) * magnitude
    return (
        (center - error).to(residual.dtype).double(),
        (center + error).to(residual.dtype).double(),
    )


def _output_bounds(lower, upper, norm):
    """Compose first-store uncertainty with the unchanged norm-stage contract.

    Exact RMSNorm is enclosed by interval arithmetic: a range crossing zero has
    minimum square zero; reciprocal RMS decreases with the row's square sum.
    Endpoint products account for negative values and negative norm weights.
    After BF16 rounding, only the original norm-stage tolerance is added.
    This is a composition of two independently checked stages, not a claim
    that an arbitrary native rsqrt is correctly rounded or bitwise identical.
    """
    if norm is None:
        return lower.bfloat16().double(), upper.bfloat16().double()
    minimum_square = torch.minimum(lower.square(), upper.square())
    minimum_square = torch.where((lower <= 0) & (upper >= 0), 0.0, minimum_square)
    maximum_square = torch.maximum(lower.square(), upper.square())
    rstd_lower = (maximum_square.mean(-1, keepdim=True) + NORM_EPS).rsqrt()
    rstd_upper = (minimum_square.mean(-1, keepdim=True) + NORM_EPS).rsqrt()
    products = (
        torch.stack(
            (
                lower * rstd_lower,
                lower * rstd_upper,
                upper * rstd_lower,
                upper * rstd_upper,
            )
        )
        * norm.double()
    )
    lo, hi = products.amin(0), products.amax(0)
    # Outward guard for the independent FP64 mean/rsqrt/product calculation.
    nu = (lower.shape[-1] + 16) * torch.finfo(torch.float64).eps / 2
    guard = (nu / (1 - nu)) * torch.maximum(lo.abs(), hi.abs())
    lo, hi = (lo - guard).bfloat16().double(), (hi + guard).bfloat16().double()
    tolerance = VALUE_ATOL + VALUE_RTOL * torch.maximum(lo.abs(), hi.abs())
    return lo - tolerance, hi + tolerance


def _assert_inside(actual, bounds):
    lower, upper = bounds
    assert (
        torch.isfinite(actual).all()
        and torch.isfinite(lower).all()
        and torch.isfinite(upper).all()
    )
    assert (lower <= upper).all()
    value = actual.double()
    torch.testing.assert_close(
        value, torch.maximum(lower, torch.minimum(value, upper)), rtol=0, atol=0
    )


def _assert_pre_stages(actual, operation, residual, fn, scale, base, norm):
    """Check gates, the first store, RMSNorm, and their end-to-end composition.

    An FP32 reduction can cross a BF16 midpoint. A later RMSNorm and another
    BF16 store can then give two output ULPs of difference from an FP64 path.
    Even the portable Torch FP32 implementation exhibits this, so a universal
    one-output-ULP bound against the FP64 endpoint is not a valid oracle.
    """
    values, post = actual
    assert values.dtype == torch.bfloat16 and post.dtype == torch.float32
    channels = residual.shape[-2]
    plain, _ = operation(
        residual,
        fn,
        scale,
        base,
        MAGNITUDE,
        RMS_EPS,
        HC_EPS,
        norm_weight=None,
        norm_eps=None,
    )
    # Preserve the projection geometry while exposing H_pre through H_post.
    probe_fn, probe_base, probe_scale = fn.clone(), base.clone(), scale.clone()
    probe_fn[channels:].copy_(probe_fn[:channels])
    probe_base[channels:].copy_(probe_base[:channels])
    probe_scale[1].copy_(probe_scale[0])
    probe_plain, pre = operation(
        residual,
        probe_fn,
        probe_scale,
        probe_base,
        1.0,
        RMS_EPS,
        HC_EPS,
        norm_weight=None,
        norm_eps=None,
    )
    assert plain.dtype == torch.bfloat16 and pre.dtype == torch.float32
    torch.testing.assert_close(probe_plain, plain, rtol=0, atol=0)
    for dtype in (torch.float32, torch.float64):
        expected_pre, expected_post = _reference_gates(
            residual, fn, scale, base, dtype=dtype
        )
        torch.testing.assert_close(
            pre.double(), expected_pre.double(), rtol=1e-5, atol=1e-6
        )
        torch.testing.assert_close(
            post.double(), expected_post.double(), rtol=1e-5, atol=1e-6
        )
    first_bounds = _reduction_bounds(residual, pre)
    _assert_inside(plain, _output_bounds(*first_bounds, None))
    if norm is None:
        torch.testing.assert_close(values, plain, rtol=0, atol=0)
    elif residual.dtype == torch.bfloat16:
        torch.testing.assert_close(
            values, _rms_reference(plain, norm), rtol=VALUE_RTOL, atol=VALUE_ATOL
        )
    else:
        # FP32 residuals retain FP32 before normalization; plain is BF16 and
        # therefore cannot be used as the norm's actual intermediate operand.
        torch.testing.assert_close(
            values,
            _reference_pre(residual, fn, scale, base, norm)[0],
            rtol=VALUE_RTOL,
            atol=VALUE_ATOL,
        )
    _assert_inside(values, _output_bounds(*first_bounds, norm))
    reference = _reference_pre(residual, fn, scale, base, norm)[0].double()
    delta = values.double() - reference
    return {
        "fp64_max_abs": delta.abs().max().item(),
        "fp64_relative_rmse": (
            delta.square().mean().sqrt()
            / reference.square().mean().sqrt().clamp_min(1e-30)
        ).item(),
        "fp64_original_tolerance_violations": int(
            (delta.abs() > VALUE_ATOL + VALUE_RTOL * reference.abs()).sum()
        ),
    }


def _reference_head(
    residual: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
) -> torch.Tensor:
    channels, hidden = residual.shape[-2:]
    flat = residual.double().reshape(-1, channels * hidden)
    logits = flat @ fn.double().T
    logits *= (flat.square().mean(-1, keepdim=True) + RMS_EPS).rsqrt()
    gates = torch.sigmoid(logits * scale[0] + base.double()) + HC_EPS
    merged = (
        gates.unsqueeze(-1) * residual.double().reshape(-1, channels, hidden)
    ).sum(1)
    return merged.bfloat16().reshape(*residual.shape[:-2], hidden)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("with_norm", [False, True])
def test_torch_ihc_pre_matches_fp64_on_cpu(dtype: torch.dtype, with_norm: bool) -> None:
    residual, fn, scale, base, norm = _inputs(7, 3, 64, dtype, "cpu")
    norm = norm if with_norm else None
    actual = torch_ihc_pre(
        residual,
        fn,
        scale,
        base,
        MAGNITUDE,
        RMS_EPS,
        HC_EPS,
        norm,
        NORM_EPS if with_norm else None,
    )
    _assert_pre_stages(actual, torch_ihc_pre, residual, fn, scale, base, norm)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_torch_ihc_post_and_head_match_fp64_on_cpu(dtype: torch.dtype) -> None:
    residual, fn, scale, base, _ = _inputs(7, 3, 64, dtype, "cpu")
    output = torch.linspace(-3, 3, 7 * 64).reshape(7, 64).bfloat16()
    gates = torch.linspace(0.1, 1.9, 7 * 3).reshape(7, 3)
    actual = torch_ihc_post(output, residual, gates)
    expected = (
        residual.double() + gates.double().unsqueeze(-1) * output.double().unsqueeze(1)
    ).to(dtype)
    torch.testing.assert_close(
        actual,
        expected,
        rtol=2**-7 if dtype == torch.bfloat16 else 1e-6,
        atol=2**-10 if dtype == torch.bfloat16 else 1e-6,
    )
    head = torch_ihc_head(residual, fn[:3], scale[:1], base[:3], RMS_EPS, HC_EPS)
    torch.testing.assert_close(
        head,
        _reference_head(residual, fn[:3], scale[:1], base[:3]),
        rtol=2**-7,
        atol=2**-10,
    )


@pytest.mark.parametrize(
    "device,solution", [("cpu", "torch"), ("cuda", "torch"), ("cuda", "triton")]
)
def test_ihc_pre_rounds_bf16_reduction_before_norm(device, solution) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("a CUDA/ROCm device is required")
    residual = torch.tensor(
        [[[1.0, 1.0], [1.0078125, 1.03125]]], dtype=torch.bfloat16, device=device
    )
    norm = torch.ones(2, device=device)
    operation = (
        torch_ihc_pre if device == "cpu" else partial(ihc_pre, solution=solution)
    )
    result, gates = operation(
        residual,
        torch.zeros(4, 4, device=device),
        torch.ones(2, device=device),
        torch.zeros(4, device=device),
        2.0,
        RMS_EPS,
        0.0,
        norm_weight=norm,
        norm_eps=NORM_EPS,
    )
    # An exact fixture, independent of the random-case interval: omitting the
    # first store yields 0.99609375 in element zero and must be rejected.
    expected = torch.tensor(
        [[0.9921875, 1.0078125]], dtype=torch.bfloat16, device=device
    )
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    torch.testing.assert_close(gates, torch.ones(1, 2, device=device), rtol=0, atol=0)
    unrounded = residual.double().mean(-2)
    assert not torch.equal(_rms_reference(unrounded, norm), expected)


def test_norm_interval_rejects_wrong_variance_and_handles_signed_values():
    values = torch.tensor([[-1.0, 1.015625]], dtype=torch.bfloat16)
    weight = torch.tensor([-1.25, 0.75])
    bounds = _output_bounds(values.double(), values.double(), weight)
    _assert_inside(_rms_reference(values, weight), bounds)
    wrong_variance = (
        values.double()
        * (values.double().square().sum(-1, keepdim=True) + NORM_EPS).rsqrt()
        * weight.double()
    ).bfloat16()
    with pytest.raises(AssertionError):
        _assert_inside(wrong_variance, bounds)
    crossing = _output_bounds(
        torch.tensor([[-1.0, 0.9]], dtype=torch.float64),
        torch.tensor([[1.0, 1.1]], dtype=torch.float64),
        weight,
    )
    for first in (-1.0, 0.0, 1.0):
        _assert_inside(_rms_reference(torch.tensor([[first, 1.0]]), weight), crossing)


@requires_gpu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("solution", ["torch", "triton"])
@pytest.mark.parametrize(
    "rows,channels,hidden", [(3, 4, 1536), (300, 4, 64), (37, 3, 512)]
)
def test_ihc_gpu_boundaries_match_fp64(
    dtype: torch.dtype,
    solution: str,
    rows: int,
    channels: int,
    hidden: int,
) -> None:
    residual, fn, scale, base, norm = _inputs(rows, channels, hidden, dtype, "cuda")
    actual = ihc_pre(
        residual,
        fn,
        scale,
        base,
        MAGNITUDE,
        RMS_EPS,
        HC_EPS,
        norm_weight=norm,
        norm_eps=NORM_EPS,
        solution=solution,
    )
    _assert_pre_stages(
        actual, partial(ihc_pre, solution=solution), residual, fn, scale, base, norm
    )
    updated = ihc_post(actual[0], residual, actual[1], solution=solution)
    expected = (
        residual.double()
        + actual[1].double().unsqueeze(-1) * actual[0].double().unsqueeze(1)
    ).to(dtype)
    torch.testing.assert_close(
        updated,
        expected,
        rtol=2**-7 if dtype == torch.bfloat16 else 1e-6,
        atol=2**-10 if dtype == torch.bfloat16 else 1e-5,
    )
    head = ihc_head(
        updated,
        fn[:channels],
        scale[:1],
        base[:channels],
        RMS_EPS,
        HC_EPS,
        solution=solution,
    )
    torch.testing.assert_close(
        head,
        _reference_head(updated, fn[:channels], scale[:1], base[:channels]),
        rtol=2**-7,
        atol=2**-10,
    )


@requires_gpu
@pytest.mark.parametrize("rows", [1, 1024])
@pytest.mark.parametrize("solution", ["torch", "triton"])
def test_ihc_pre_at_hy4_projection_width(
    rows: int, solution: str, record_property
) -> None:
    residual, fn, scale, base, norm = _inputs(rows, 4, 6144, torch.bfloat16, "cuda")
    actual = ihc_pre(
        residual,
        fn,
        scale,
        base,
        MAGNITUDE,
        RMS_EPS,
        HC_EPS,
        norm_weight=norm,
        norm_eps=NORM_EPS,
        solution=solution,
    )
    diagnostics = _assert_pre_stages(
        actual, partial(ihc_pre, solution=solution), residual, fn, scale, base, norm
    )
    for name, value in diagnostics.items():
        record_property(name, value)


@requires_gpu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_ihc_startup_prepares_split_buckets_for_serving_rows(dtype):
    from tokenspeed_kernel.ops.residual.triton import _ihc_gate_reduce_kernel
    from tokenspeed_kernel.ops.tuning import kernel_warmup

    residual, fn, scale, base, norm = _inputs(512, 4, 6144, dtype, "cuda")
    original = residual.clone()

    def execute(rows):
        pre = ihc_pre(
            residual[:rows],
            fn,
            scale,
            base,
            MAGNITUDE,
            RMS_EPS,
            HC_EPS,
            norm_weight=norm,
            norm_eps=NORM_EPS,
            solution="triton",
        )
        head = ihc_head(
            residual[:rows],
            fn[:4],
            scale[:1],
            base[:4],
            RMS_EPS,
            HC_EPS,
            solution="triton",
        )
        return pre, head

    ordinary = execute(512)
    with kernel_warmup(enabled=True):
        warmed = execute(512)
    torch.testing.assert_close(residual, original, rtol=0, atol=0)
    for expected, actual in zip(ordinary[0], warmed[0], strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(warmed[1], ordinary[1], rtol=0, atol=0)

    results = []
    with assert_no_triton_compile(_ihc_gate_reduce_kernel):
        for rows in (1, 3, 16, 17, 32, 33, 64, 65, 128, 257, 511):
            results.append((rows, execute(rows)))
    for rows, (pre, head) in results:
        _assert_pre_stages(
            pre,
            partial(ihc_pre, solution="triton"),
            residual[:rows],
            fn,
            scale,
            base,
            norm,
        )
        torch.testing.assert_close(
            head,
            _reference_head(residual[:rows], fn[:4], scale[:1], base[:4]),
            rtol=2**-7,
            atol=2**-10,
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_torch_ihc_empty_rows(dtype: torch.dtype) -> None:
    residual, fn, scale, base, norm = _inputs(0, 4, 64, dtype, "cpu")
    values, gates = torch_ihc_pre(
        residual, fn, scale, base, MAGNITUDE, RMS_EPS, HC_EPS, norm, NORM_EPS
    )
    updated = torch_ihc_post(values, residual, gates)
    merged = torch_ihc_head(updated, fn[:4], scale[:1], base[:4], RMS_EPS, HC_EPS)
    assert values.shape == (0, 64) and values.dtype == torch.bfloat16
    assert gates.shape == (0, 4) and gates.dtype == torch.float32
    assert updated.shape == residual.shape and updated.dtype == dtype
    assert merged.shape == (0, 64) and merged.dtype == torch.bfloat16


@requires_gpu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_triton_ihc_empty_rows(dtype: torch.dtype) -> None:
    residual, fn, scale, base, norm = _inputs(0, 4, 64, dtype, "cuda")
    values, gates = ihc_pre(
        residual,
        fn,
        scale,
        base,
        MAGNITUDE,
        RMS_EPS,
        HC_EPS,
        norm_weight=norm,
        norm_eps=NORM_EPS,
        solution="triton",
    )
    updated = ihc_post(values, residual, gates, solution="triton")
    merged = ihc_head(
        updated, fn[:4], scale[:1], base[:4], RMS_EPS, HC_EPS, solution="triton"
    )
    assert values.shape == (0, 64) and values.dtype == torch.bfloat16
    assert gates.shape == (0, 4) and gates.dtype == torch.float32
    assert updated.shape == residual.shape and updated.dtype == dtype
    assert merged.shape == (0, 64) and merged.dtype == torch.bfloat16


@requires_gpu
def test_triton_ihc_leading_dimensions_and_strided_residual() -> None:
    source, fn, scale, base, norm = _inputs(12, 3, 64, torch.bfloat16, "cuda")
    residual = source.reshape(2, 6, 3, 64)[:, ::2]
    actual = ihc_pre(
        residual,
        fn,
        scale,
        base,
        MAGNITUDE,
        RMS_EPS,
        HC_EPS,
        norm_weight=norm,
        norm_eps=NORM_EPS,
        solution="triton",
    )
    _assert_pre_stages(
        actual, partial(ihc_pre, solution="triton"), residual, fn, scale, base, norm
    )
    assert actual[0].shape == (2, 3, 64)
    assert actual[1].shape == (2, 3, 3)


def _public_pre(
    operands: dict[str, torch.Tensor], *, norm_eps: float | None = NORM_EPS
):
    return ihc_pre(
        operands["residual"],
        operands["fn"],
        operands["scale"],
        operands["base"],
        MAGNITUDE,
        RMS_EPS,
        HC_EPS,
        norm_weight=operands["norm"],
        norm_eps=norm_eps,
        solution="torch",
    )


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("residual", torch.zeros(3, 1, 32)),
        ("residual", torch.zeros(3, 4, 8, dtype=torch.float16)),
        ("fn", torch.zeros(4, 32)),
        ("fn", torch.zeros(32, 8).T),
        ("fn", torch.zeros(8, 32, dtype=torch.bfloat16)),
        ("scale", torch.zeros(1)),
        ("scale", torch.zeros(2, dtype=torch.bfloat16)),
        ("base", torch.zeros(8, device="meta")),
        ("base", torch.zeros(8, dtype=torch.bfloat16)),
        ("norm", torch.ones(4)),
    ],
)
def test_ihc_pre_rejects_invalid_operands(
    field: str, replacement: torch.Tensor
) -> None:
    residual, fn, scale, base, norm = _inputs(3, 4, 8, torch.bfloat16, "cpu")
    operands = dict(residual=residual, fn=fn, scale=scale, base=base, norm=norm)
    operands[field] = replacement
    with pytest.raises((ValueError, TypeError)):
        _public_pre(operands)


def test_ihc_pre_rejects_unpaired_norm_arguments() -> None:
    residual, fn, scale, base, norm = _inputs(3, 4, 8, torch.bfloat16, "cpu")
    with pytest.raises(ValueError, match="norm_weight and norm_eps"):
        _public_pre(
            dict(residual=residual, fn=fn, scale=scale, base=base, norm=norm),
            norm_eps=None,
        )


@pytest.mark.parametrize(
    "output,gates",
    [
        (torch.zeros(1, 8, dtype=torch.bfloat16), torch.zeros(3, 4)),
        (torch.zeros(3, 8), torch.zeros(3, 4)),
        (
            torch.zeros(3, 8, dtype=torch.bfloat16),
            torch.zeros(3, 4, dtype=torch.bfloat16),
        ),
        (torch.zeros(3, 8, dtype=torch.bfloat16), torch.zeros(3, 4, device="meta")),
    ],
)
def test_ihc_post_rejects_invalid_operands(
    output: torch.Tensor, gates: torch.Tensor
) -> None:
    with pytest.raises((ValueError, TypeError)):
        ihc_post(
            output, torch.zeros(3, 4, 8, dtype=torch.bfloat16), gates, solution="torch"
        )


def test_ihc_head_rejects_pre_projection() -> None:
    residual, fn, scale, base, _ = _inputs(3, 4, 8, torch.bfloat16, "cpu")
    with pytest.raises(ValueError):
        ihc_head(residual, fn, scale, base, RMS_EPS, HC_EPS, solution="torch")


@requires_gpu
def test_triton_ihc_graph_replay_uses_changed_values() -> None:
    residual, fn, scale, base, norm = _inputs(33, 4, 64, torch.bfloat16, "cuda")

    def forward():
        values, gates = ihc_pre(
            residual,
            fn,
            scale,
            base,
            MAGNITUDE,
            RMS_EPS,
            HC_EPS,
            norm_weight=norm,
            norm_eps=NORM_EPS,
            solution="triton",
        )
        updated = ihc_post(values, residual, gates, solution="triton")
        merged = ihc_head(
            updated, fn[:4], scale[:1], base[:4], RMS_EPS, HC_EPS, solution="triton"
        )
        return values, gates, updated, merged

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            forward()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = forward()
    for factor in (0.5, -1.75):
        residual.mul_(factor)
        base.add_(0.125)
        expected = forward()
        graph.replay()
        torch.cuda.synchronize()
        for actual, reference in zip(captured, expected, strict=True):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        _assert_pre_stages(
            captured[:2],
            partial(ihc_pre, solution="triton"),
            residual,
            fn,
            scale,
            base,
            norm,
        )


@requires_gpu
def test_triton_ihc_token_counts_reuse_specializations() -> None:
    from tokenspeed_kernel.ops.residual.triton import (
        _ihc_gate_reduce_kernel,
        _ihc_post_kernel,
        _ihc_project_kernel,
    )

    def forward(rows: int) -> None:
        residual, fn, scale, base, norm = _inputs(rows, 4, 64, torch.bfloat16, "cuda")
        values, gates = ihc_pre(
            residual,
            fn,
            scale,
            base,
            MAGNITUDE,
            RMS_EPS,
            HC_EPS,
            norm_weight=norm,
            norm_eps=NORM_EPS,
            solution="triton",
        )
        ihc_post(values, residual, gates, solution="triton")
        ihc_head(
            residual, fn[:4], scale[:1], base[:4], RMS_EPS, HC_EPS, solution="triton"
        )

    # This token ladder shares the projection tile and split bound. Warm both
    # runtime-scalar alignment classes, then vary the actual number of rows.
    forward(33)
    forward(48)
    with assert_no_triton_compile(
        _ihc_project_kernel, _ihc_gate_reduce_kernel, _ihc_post_kernel
    ):
        for rows in (35, 47, 63):
            forward(rows)
