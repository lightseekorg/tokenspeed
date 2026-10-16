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

"""Metadata-only public admission and early argument rejection."""

import pytest
import torch
from tokenspeed_kernel.ops.layernorm.gluon import NAME, gluon_qk_rmsnorm_ssmax_gfx950
from tokenspeed_kernel.ops.layernorm.qk_rmsnorm_ssmax import (
    QK_RMSNORM_SSMAX_SIGNATURES,
    TORCH_QK_RMSNORM_SSMAX,
    qk_rmsnorm_ssmax_traits,
)
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature


def arguments(rows=16, stride=1792):
    packed = torch.empty((rows, stride), dtype=torch.bfloat16, device="meta")
    q = packed[:, :768]
    k = (
        packed[:, 768:896]
        if stride >= 896
        else torch.empty((rows, 128), dtype=torch.bfloat16, device="meta")
    )
    return (
        q,
        k,
        torch.empty(128, dtype=torch.bfloat16, device="meta"),
        torch.empty(128, dtype=torch.bfloat16, device="meta"),
        1e-6,
        torch.empty(rows, dtype=torch.int64, device="meta"),
    )


def selected(platform, args, *, dequant, changed_traits=None):
    q, k, qw, kw, eps, pos = args
    traits = qk_rmsnorm_ssmax_traits(*args)
    traits["is_cuda"] = True  # Only the missing device facility is simulated.
    if changed_traits:
        traits.update(changed_traits)
    signature = format_signature(
        **{
            name: dense_tensor_format(t.dtype)
            for name, t in (
                ("q", q),
                ("k", k),
                ("q_weight", qw),
                ("k_weight", kw),
                ("positions", pos),
            )
        }
    )
    assert signature in QK_RMSNORM_SSMAX_SIGNATURES
    return select_kernel(
        "layernorm",
        "qk_rmsnorm_ssmax",
        signature,
        traits=traits,
        features=frozenset({"dequant"}) if dequant else None,
        platform=platform,
    ).name


@pytest.mark.parametrize("rows", [1, 2, 4, 8, 16, 32, 64, 128, 256, 848, 896, 8192])
@pytest.mark.parametrize("stride", [768, 1792])
def test_narrow_automatic_admission_and_dequant_fallback(mi350_platform, rows, stride):
    args = arguments(rows, stride)
    assert selected(mi350_platform, args, dequant=False) == NAME
    assert selected(mi350_platform, args, dequant=True) == TORCH_QK_RMSNORM_SSMAX


@pytest.mark.parametrize(
    "trait,value",
    [
        ("heads", (48, 8)),
        ("head_dim", 64),
        ("tokens", 0),
        ("ndim", 3),
        ("is_cuda", False),
        ("positions_contiguous", False),
        ("rows_disjoint", False),
        ("unit_stride", False),
        ("weights_contiguous", False),
        ("eps_ok", False),
    ],
)
def test_unsupported_metadata_keeps_portable_fallback(mi350_platform, trait, value):
    assert (
        selected(
            mi350_platform, arguments(), dequant=False, changed_traits={trait: value}
        )
        == TORCH_QK_RMSNORM_SSMAX
    )


@pytest.mark.parametrize(
    "change", ["dtype", "positions", "eps", "rows", "shape", "dequant"]
)
def test_by_name_rejects_before_any_device_launch(change):
    args = list(arguments())
    extra = {}
    if change == "dtype":
        args[0] = args[0].float()
    elif change == "positions":
        args[5] = args[5].float()
    elif change == "eps":
        args[4] = 0.0
    elif change == "rows":
        args = list(arguments(0))
    elif change == "shape":
        args[0] = args[0][:, :640]
    else:
        extra["dequant"] = object()
    with pytest.raises(ValueError):
        gluon_qk_rmsnorm_ssmax_gfx950(*args, **extra)


def test_fixture_distinguishes_required_affine_bf16_rounding():
    # Exact unit inputs avoid making this negative control about reduction order.
    weight = (torch.arange(128).float() / 128 + 0.5).bfloat16().float()
    affine = torch.rsqrt(torch.tensor(1.5)) * weight
    scale = torch.exp(0.1 * torch.log1p(torch.tensor(65536.0) / 2048))
    required = (affine.bfloat16().float() * scale).bfloat16()
    skipped_boundary = (affine * scale).bfloat16()
    assert (required != skipped_boundary).any()
