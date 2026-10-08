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

import importlib
import math
import sys
import types
from types import SimpleNamespace

import pytest
import tokenspeed_kernel.ops.layernorm as layernorm
import tokenspeed_kernel.ops.layernorm.triton as triton_layernorm
import torch


def _platform(vendor: str) -> SimpleNamespace:
    return SimpleNamespace(
        is_nvidia=vendor == "nvidia",
        is_amd=vendor == "amd",
        is_npu=vendor == "ascend",
    )


@pytest.mark.parametrize(
    ("vendor", "message"),
    [
        ("nvidia", "fused_add_rmsnorm does not support out"),
        ("amd", "fused add rmsnorm does not support out"),
        ("ascend", "rmsnorm does not support residual and out together"),
    ],
)
def test_residual_and_out_are_mutually_exclusive(
    monkeypatch, vendor: str, message: str
) -> None:
    monkeypatch.setattr(layernorm, "_platform", _platform(vendor))

    with pytest.raises(ValueError, match=message):
        layernorm.rmsnorm(object(), object(), 1e-6, residual=object(), out=object())


class _KernelLaunches:
    """Stand-in for a Triton kernel that records each launch."""

    def __init__(self) -> None:
        self.launches: list[tuple[tuple, dict]] = []

    def __getitem__(self, grid):
        return lambda *args, **kwargs: self.launches.append((args, kwargs))


def _record_rmsnorm_launches(monkeypatch, vendor: str) -> _KernelLaunches:
    kernel = _KernelLaunches()
    monkeypatch.setattr(triton_layernorm, "_rmsnorm_kernel", kernel)
    monkeypatch.setattr(triton_layernorm, "current_platform", lambda: _platform(vendor))
    monkeypatch.setattr(triton_layernorm, "pdl_enabled", lambda: True)
    return kernel


@pytest.mark.parametrize(
    ("options", "changes", "scales"),
    [
        ({}, {}, (1.0, 1.0)),
        ({"residual": None}, {"HAS_RESIDUAL": False}, (1.0, 1.0)),
        ({"enable_pdl": None}, {"ENABLE_PDL": True, "launch_pdl": True}, (1.0, 1.0)),
        ({"enable_pdl": True}, {"ENABLE_PDL": True, "launch_pdl": True}, (1.0, 1.0)),
        (
            {"round_residual_sum_bf16": True},
            {"ROUND_RESIDUAL_SUM_BF16": True},
            (1.0, 1.0),
        ),
        (
            {"round_residual_sum_bf16": True, "x_scale": 0.5},
            {"ROUND_RESIDUAL_SUM_BF16": True, "SCALE_INPUTS_BF16": True},
            (0.5, 1.0),
        ),
    ],
)
def test_triton_rmsnorm_launch_keywords(monkeypatch, options, changes, scales) -> None:
    """Each option sets only its own launch keywords. Without options every new
    constexpr is False, which compiles the kernel body as it was before the
    options existed, and ``launch_pdl`` is not passed."""
    kernel = _record_rmsnorm_launches(monkeypatch, "nvidia")
    x = torch.zeros(3, 96, dtype=torch.bfloat16)
    weight = torch.ones(96, dtype=torch.bfloat16)

    triton_layernorm.rmsnorm(x, weight, 1e-6, **{"residual": x.clone(), **options})

    ((args, kwargs),) = kernel.launches
    assert args[5:7] == scales and all(type(scale) is float for scale in args[5:7])
    assert kwargs == {
        "BLOCK": 128,
        "HAS_RESIDUAL": True,
        "ROUND_RESIDUAL_SUM_BF16": False,
        "SCALE_INPUTS_BF16": False,
        "ENABLE_PDL": False,
        **changes,
    }


def test_triton_rmsnorm_rejects_invalid_options(monkeypatch) -> None:
    kernel = _record_rmsnorm_launches(monkeypatch, "nvidia")
    x = torch.zeros(2, 64, dtype=torch.bfloat16)
    weight = torch.ones(64, dtype=torch.bfloat16)
    rounded = {"residual": x, "round_residual_sum_bf16": True}
    cases = [
        ("needs BF16 x and residual", {"round_residual_sum_bf16": True}),
        ("needs BF16 x and residual", {**rounded, "residual": x.half()}),
        ("need round_residual_sum_bf16=True", {"residual": x, "x_scale": 0.5}),
        ("x_scale must be a finite float", {**rounded, "x_scale": 1}),
        ("residual_scale must be a finite", {**rounded, "residual_scale": math.inf}),
    ]
    for message, options in cases:
        with pytest.raises(ValueError, match=message):
            triton_layernorm.rmsnorm(x, weight, 1e-6, **options)
    monkeypatch.setattr(triton_layernorm, "current_platform", lambda: _platform("amd"))
    with pytest.raises(ValueError, match="need an NVIDIA GPU"):
        triton_layernorm.rmsnorm(x, weight, 1e-6, **rounded, x_scale=0.5)
    assert kernel.launches == []


def test_nvidia_rmsnorm_preserves_fused_residual_contract(monkeypatch) -> None:
    calls = []
    x, residual, weight = object(), object(), object()

    def fused(x_arg, residual_arg, weight_arg, eps, **kwargs):
        calls.append((x_arg, residual_arg, weight_arg, eps, kwargs))

    monkeypatch.setattr(layernorm, "_platform", _platform("nvidia"))
    monkeypatch.setattr(layernorm, "_fused_add_rmsnorm", fused, raising=False)

    result = layernorm.rmsnorm(x, weight, 1e-6, residual=residual)

    assert result == (x, residual)
    assert calls == [(x, residual, weight, 1e-6, {})]


def test_nvidia_rmsnorm_defers_pdl_policy_to_backend(monkeypatch) -> None:
    calls = []
    result, x, weight, out = object(), object(), object(), object()

    def backend(x_arg, weight_arg, eps, **kwargs):
        calls.append((x_arg, weight_arg, eps, kwargs))
        return result

    monkeypatch.setattr(layernorm, "_platform", _platform("nvidia"))
    monkeypatch.setattr(layernorm, "_rmsnorm", backend, raising=False)

    assert layernorm.rmsnorm(x, weight, 1e-6, out=out) is result
    assert calls == [(x, weight, 1e-6, {"out": out})]


def test_amd_rmsnorm_preserves_triton_call_contract(monkeypatch) -> None:
    calls = []
    result, x, weight, residual, out = (object() for _ in range(5))

    def backend(x_arg, weight_arg, eps, **kwargs):
        calls.append((x_arg, weight_arg, eps, kwargs))
        return result

    monkeypatch.setattr(layernorm, "_platform", _platform("amd"))
    monkeypatch.setattr(layernorm, "triton_rmsnorm", backend, raising=False)

    assert layernorm.rmsnorm(x, weight, 1e-6, residual=residual) is result
    assert layernorm.rmsnorm(x, weight, 1e-6, out=out) is result
    assert calls == [
        (x, weight, 1e-6, {"residual": residual}),
        (x, weight, 1e-6, {"out": out}),
    ]


def test_ascend_rmsnorm_forwards_residual_or_out(monkeypatch) -> None:
    calls = []
    result, x, weight, residual, out = (object() for _ in range(5))

    def backend(x_arg, weight_arg, eps, **kwargs):
        calls.append((x_arg, weight_arg, eps, kwargs))
        return result

    monkeypatch.setattr(layernorm, "_platform", _platform("ascend"))
    monkeypatch.setattr(layernorm, "_rmsnorm", backend, raising=False)

    assert layernorm.rmsnorm(x, weight, 1e-6, residual=residual) is result
    assert layernorm.rmsnorm(x, weight, 1e-6, out=out) is result
    assert calls == [
        (x, weight, 1e-6, {"residual": residual}),
        (x, weight, 1e-6, {"out": out}),
    ]


@pytest.mark.parametrize("vendor", ["nvidia", "amd", "ascend"])
def test_qk_rmsnorm_has_one_platform_contract(monkeypatch, vendor: str) -> None:
    calls = []
    result = (object(), object())
    q, k, q_weight, k_weight = (object() for _ in range(4))

    def backend(q_arg, k_arg, qw_arg, kw_arg, eps, **kwargs):
        calls.append((q_arg, k_arg, qw_arg, kw_arg, eps, kwargs))
        return result

    monkeypatch.setattr(layernorm, "_platform", _platform(vendor))
    monkeypatch.setattr(layernorm, "_qk_rmsnorm", backend)

    assert (
        layernorm.qk_rmsnorm(q, k, q_weight, k_weight, 1e-6, weight_offset=1.0)
        == result
    )
    assert calls == [(q, k, q_weight, k_weight, 1e-6, {"weight_offset": 1.0})]


def test_ascend_forms_the_offset_weight_in_the_weight_dtype(monkeypatch):
    calls = []
    npu = types.ModuleType("tokenspeed_kernel_npu.ops.layernorm")
    npu.qk_rmsnorm = lambda *args: calls.append(args)
    npu.rmsnorm = None
    for name in ("tokenspeed_kernel_npu", "tokenspeed_kernel_npu.ops"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, npu.__name__, npu)
    monkeypatch.delitem(sys.modules, "tokenspeed_kernel.ops.layernorm.ascend", False)
    ascend = importlib.import_module("tokenspeed_kernel.ops.layernorm.ascend")

    # 1 + 2^-9 is not a bf16 value: formed in bf16 it rounds to 1.0.
    q_weight = torch.tensor([2.0**-9, 0.5, -0.25], dtype=torch.bfloat16)
    k_weight = torch.tensor([3.0, 2.0**-9, 0.125], dtype=torch.bfloat16)
    q, k = object(), object()
    ascend.qk_rmsnorm(q, k, q_weight, k_weight, 1e-5, weight_offset=1.0)
    ((got_q, got_k, got_q_weight, got_k_weight, eps),) = calls
    assert got_q is q and got_k is k and eps == 1e-5
    for got, weight in ((got_q_weight, q_weight), (got_k_weight, k_weight)):
        assert got.dtype == torch.bfloat16
        assert torch.equal(got, (weight.float() + 1.0).to(torch.bfloat16))
    # A zero offset hands the stored weights through untouched.
    ascend.qk_rmsnorm(q, k, q_weight, k_weight, 1e-5, weight_offset=0.0)
    assert calls[1][2] is q_weight and calls[1][3] is k_weight
