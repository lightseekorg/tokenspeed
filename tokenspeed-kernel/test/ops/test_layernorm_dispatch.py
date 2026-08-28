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

from types import SimpleNamespace

import pytest
import tokenspeed_kernel.ops.layernorm as layernorm


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

    assert layernorm.qk_rmsnorm(q, k, q_weight, k_weight, 1e-6) == result
    assert calls == [(q, k, q_weight, k_weight, 1e-6, {})]
