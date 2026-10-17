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

"""The route epilogue's arithmetic for ``triton_router_route_lanepar``
(:mod:`tokenspeed_kernel.ops.moe._triton.router_select`), and the Torch
reduction order it reproduces.

Under ``renormalize`` the route divides the eight selected FP32 scores by their
sum plus ``1e-20`` with the sum spelled in the order Torch's CUDA reduction uses
for a contiguous FP32 ``[rows, 8]`` ``sum(dim=-1)``: ``((v0+v4)+(v2+v6)) +
((v1+v5)+(v3+v7))``, then ``+ 1e-20``, then a correctly rounded division -- every
step one round-to-nearest FP32 operation (``add.rn.f32`` by inline PTX,
``tl.div_rn``). With a routed ``scale`` (rounded to FP32 as Torch rounds a Python
scalar) the weights then take one ``mul.rn.f32``, the multiply TopK applies in
place after the route. The output equals the route's Torch statements
(``torch_route_epilogue`` in :mod:`tokenspeed_kernel.ops.moe.grouped_route`)
byte for byte on a Torch build whose CUDA K8 sum follows that tree.

The Torch side of the contract (torch ``2.13.0+cu130``,
``ATen/native/cuda/Reduce.cuh``): the sum over the fastest-striding dimension
of 8 elements takes no input vectorization (that needs ``dim0 >= 128``), gets
``block.x = 8`` lanes per output row with one element per lane, and combines the
lanes with ``shfl_down`` offsets 4, 2, 1, so lane 0 holds ``((v0+v4)+(v2+v6)) +
((v1+v5)+(v3+v7))``. A Torch change to that order voids the contract without
touching the kernel, so the GPU test runs :func:`reduction_order_mismatch` first
and fails on a mismatch (never a skip).
"""

from __future__ import annotations

import math

import torch
from tokenspeed_kernel._triton import tl, triton

__all__ = [
    "k8_reduction_tree",
    "reduction_order_fixture",
    "reduction_order_mismatch",
    "route_scale_fp32",
    "route_scale_rejection",
]

_TOPK = 8

# Fixture rows of the reduction-order pin (module doc): the
# rounding row (its sum's bits depend on the summation order), the subnormal
# row, an all-equal row, a row with ``-inf`` and seeded rows of log-uniform
# magnitudes ``2^U[-149, 0] * rand`` with random signs.
_PIN_ROUNDING_ROW = (
    1.0,
    2.0**-24,
    2.0**-25,
    2.0**-26,
    0.5,
    2.0**-23,
    2.0**-24,
    2.0**-25,
)
_PIN_SUBNORMAL_ROW = (
    2.0**-149,
    2.0**-140,
    2.0**-130,
    2.0**-126,
    0.0,
    2.0**-145,
    2.0**-135,
    2.0**-128,
)
_PIN_EQUAL_ROW = (0.25,) * _TOPK
_PIN_NEG_INF_ROW = (1.0, 2.0, float("-inf"), 4.0, 0.5, 8.0, 16.0, 32.0)
_PIN_SEED = 913
_PIN_RANDOM_ROWS = 128


@triton.jit
def _add_rn(left, right):
    """One correctly rounded FP32 add (``add.rn.f32``) that the compiler can
    neither reassociate nor fuse."""
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[left, right],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _mul_rn(left, right):
    """One correctly rounded FP32 multiply (``mul.rn.f32``, non-FTZ) that the
    compiler cannot fuse."""
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[left, right],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


def route_scale_fp32(scale: float) -> float:
    """``scale`` rounded to FP32 the way Torch rounds a Python scalar operand of an
    FP32 multiply (one round-to-nearest conversion), as a Python float."""
    return torch.tensor(scale, dtype=torch.float32).item()


def route_scale_rejection(scale: float | None) -> str:
    """Why a Triton route kernel cannot take ``scale`` as its compile-time
    ``SCALE``; empty for ``None`` or a real number whose FP32 rounding is finite
    and zero or normal (Triton keeps such a constant FP32)."""
    if scale is None:
        return ""
    if isinstance(scale, bool) or not isinstance(scale, (int, float)):
        return f"scale must be a real number, got {type(scale).__name__}"
    rounded = route_scale_fp32(scale)
    if not math.isfinite(rounded) or 0 < abs(rounded) < 2.0**-126:
        return f"scale must round to a finite zero or normal FP32, got {scale!r}"
    return ""


def reduction_order_fixture() -> torch.Tensor:
    """The contiguous FP32 ``[n, 8]`` CPU rows the reduction-order pin compares on.

    The rounding row and the mixed-magnitude seeded rows are the ones on which
    different summation orders give different bits; the subnormal, all-equal and
    ``-inf`` rows cover the special-value paths. Deterministic (seed
    ``_PIN_SEED``); the caller moves the rows to the device under test.

    Returns:
        A fresh contiguous FP32 tensor of ``4 + _PIN_RANDOM_ROWS`` rows.
    """
    generator = torch.Generator().manual_seed(_PIN_SEED)
    exponents = torch.randint(
        -149, 1, (_PIN_RANDOM_ROWS, _TOPK), generator=generator
    ).to(torch.float64)
    mantissas = torch.rand(
        _PIN_RANDOM_ROWS, _TOPK, generator=generator, dtype=torch.float64
    )
    negative = torch.rand(_PIN_RANDOM_ROWS, _TOPK, generator=generator) < 0.5
    magnitudes = mantissas * torch.exp2(exponents)
    random_rows = torch.where(negative, -magnitudes, magnitudes).to(torch.float32)
    fixed_rows = torch.tensor(
        [_PIN_ROUNDING_ROW, _PIN_SUBNORMAL_ROW, _PIN_EQUAL_ROW, _PIN_NEG_INF_ROW],
        dtype=torch.float32,
    )
    return torch.cat([fixed_rows, random_rows]).contiguous()


def k8_reduction_tree(values: torch.Tensor) -> torch.Tensor:
    """Sum the last dimension (size 8) in the pinned Torch CUDA order.

    ``((v0+v4)+(v2+v6)) + ((v1+v5)+(v3+v7))`` with one FP32 round-to-nearest add
    per ``+`` (torch elementwise adds: a lone add is never fused, so the order
    is the point and the device is not). This is the reference the GPU pin
    compares ``torch.sum`` against and the tree the kernel spells in
    ``add.rn.f32``.

    Args:
        values: FP32 tensor whose last dimension has size 8.

    Returns:
        FP32 tensor of shape ``values.shape[:-1] + (1,)``.

    Raises:
        ValueError: ``values`` is not FP32 or its last dimension is not 8.
    """
    if values.dtype != torch.float32:
        raise ValueError(f"k8_reduction_tree needs float32 values, got {values.dtype}")
    if values.ndim == 0 or values.shape[-1] != _TOPK:
        raise ValueError(
            f"k8_reduction_tree needs a last dimension of {_TOPK}, "
            f"got shape {tuple(values.shape)}"
        )
    v = values.unbind(-1)
    even = (v[0] + v[4]) + (v[2] + v[6])
    odd = (v[1] + v[5]) + (v[3] + v[7])
    return (even + odd).unsqueeze(-1)


def reduction_order_mismatch(device: torch.device) -> str:
    """Run ``torch.sum`` over the pinned fixture rows on ``device`` and compare it
    bit for bit with :func:`k8_reduction_tree`.

    The GPU test runs it first and fails the route's byte-equality tests on a
    mismatch. Never called by the kernel (the route runs under CUDA-graph
    capture, where the host sync this needs is illegal).

    Args:
        device: The CUDA device whose ``torch.sum`` order is the contract.

    Returns:
        ``""`` when every row agrees, else a reason naming ``torch.__version__``
        and the first differing row.

    Raises:
        ValueError: ``device`` is not a CUDA device (a CPU ``torch.sum`` reduces
            in another order and is not the contract).
    """
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError(
            "reduction_order_mismatch needs a CUDA device: the contract is the CUDA "
            f"K8 reduction order, a CPU torch.sum reduces differently (got {device})"
        )
    rows = reduction_order_fixture().to(device)
    torch_sum = rows.sum(dim=-1, keepdim=True).contiguous()
    tree = k8_reduction_tree(rows).contiguous()
    same = torch_sum.view(torch.int32) == tree.view(torch.int32)
    if bool(same.all()):
        return ""
    first = int((~same).flatten().nonzero()[0].item())
    return (
        f"torch {torch.__version__}: the CUDA K8 sum does not follow the pinned "
        "shuffle-down tree ((v0+v4)+(v2+v6))+((v1+v5)+(v3+v7)); "
        "the one-launch route's bitwise contract is void on this build "
        f"(first differing row {first}: values {rows[first].tolist()}, "
        f"torch.sum {torch_sum[first].item()!r}, tree {tree[first].item()!r})"
    )
