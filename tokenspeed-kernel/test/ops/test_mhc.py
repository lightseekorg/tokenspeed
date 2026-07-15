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

import importlib

import pytest
import tokenspeed_kernel.ops.mhc as mhc_ops
import tokenspeed_kernel.ops.mhc.deep_gemm as deep_gemm_mhc
import torch
import torch.nn.functional as F
from tokenspeed_kernel.platform import Platform, current_platform
from tokenspeed_kernel.registry import KernelRegistry


class _SelectedKernel:
    name = "test_mhc_kernel"

    def __init__(self, result: object) -> None:
        self.result = result
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return self.result


@pytest.mark.parametrize(
    "platform_fixture",
    [
        "h100_platform",
        "b200_platform",
        "sm120_platform",
    ],
)
def test_mhc_plan_preserves_existing_architecture_solutions(
    platform_fixture: str,
    request,
    fresh_registry,
) -> None:
    platform = request.getfixturevalue(platform_fixture)
    real_platform = Platform.get()
    try:
        Platform.override(platform)
        importlib.reload(deep_gemm_mhc)
        KernelRegistry.get().clear_cache()
        if deep_gemm_mhc.deep_gemm is None:
            pytest.skip("DeepGEMM is unavailable")

        plan = mhc_ops.mhc_plan(hc_mult=4)

        assert plan == {
            "hc_mult": 4,
            "pre_kernel_name": "deep_gemm_mhc_pre",
            "post_kernel_name": "deep_gemm_mhc_post",
            "solution": "deep_gemm",
        }
    finally:
        Platform.override(real_platform)
        KernelRegistry.get().clear_cache()


def test_mhc_pre_dispatches_through_registered_kernel(monkeypatch) -> None:
    residual = torch.randn(2, 4, 8, dtype=torch.bfloat16)
    fn = torch.randn(24, 32, dtype=torch.float32)
    scale = torch.ones(3, dtype=torch.float32)
    base = torch.zeros(24, dtype=torch.float32)
    expected = (
        torch.empty(2, 8, dtype=torch.bfloat16),
        torch.empty(2, 4, 1, dtype=torch.float32),
        torch.empty(2, 4, 4, dtype=torch.float32),
    )
    selected = _SelectedKernel(expected)
    selection: dict[str, object] = {}

    def fake_select_kernel(family, mode, signature, **kwargs):
        selection.update(
            family=family,
            mode=mode,
            signature=signature,
            kwargs=kwargs,
        )
        return selected

    monkeypatch.setattr(mhc_ops, "select_kernel", fake_select_kernel)

    actual = mhc_ops.mhc_pre(
        residual,
        fn,
        scale,
        base,
        rms_eps=1e-6,
        hc_eps=2e-6,
        sinkhorn_iters=3,
        solution="deep_gemm",
    )

    assert actual is expected
    assert selection["family"] == "mhc"
    assert selection["mode"] == "pre"
    signature = selection["signature"]
    assert signature.storage_dtype_for("residual") == torch.bfloat16
    assert signature.storage_dtype_for("fn") == torch.float32
    assert signature.storage_dtype_for("mhc_scale") == torch.float32
    assert signature.storage_dtype_for("mhc_base") == torch.float32
    assert selection["kwargs"] == {
        "traits": {"hc_mult": 4},
        "solution": "deep_gemm",
        "override": None,
    }
    assert selected.calls == [
        {
            "residual": residual,
            "fn": fn,
            "mhc_scale": scale,
            "mhc_base": base,
            "rms_eps": 1e-6,
            "mhc_eps": 2e-6,
            "sinkhorn_iters": 3,
        }
    ]


def test_mhc_post_dispatches_through_registered_kernel(monkeypatch) -> None:
    hidden_states = torch.randn(2, 8, dtype=torch.bfloat16)
    residual = torch.randn(2, 4, 8, dtype=torch.bfloat16)
    post = torch.randn(2, 4, 1, dtype=torch.float32)
    comb = torch.randn(2, 4, 4, dtype=torch.float32)
    expected = torch.empty_like(residual)
    selected = _SelectedKernel(expected)
    selection: dict[str, object] = {}

    def fake_select_kernel(family, mode, signature, **kwargs):
        selection.update(
            family=family,
            mode=mode,
            signature=signature,
            kwargs=kwargs,
        )
        return selected

    monkeypatch.setattr(mhc_ops, "select_kernel", fake_select_kernel)

    actual = mhc_ops.mhc_post(
        hidden_states,
        residual,
        post,
        comb,
        solution="deep_gemm",
    )

    assert actual is expected
    assert selection["family"] == "mhc"
    assert selection["mode"] == "post"
    signature = selection["signature"]
    assert signature.storage_dtype_for("hidden_states") == torch.bfloat16
    assert signature.storage_dtype_for("residual") == torch.bfloat16
    assert signature.storage_dtype_for("post") == torch.float32
    assert signature.storage_dtype_for("comb") == torch.float32
    assert selection["kwargs"] == {
        "traits": {"hc_mult": 4},
        "solution": "deep_gemm",
        "override": None,
    }
    assert selected.calls == [
        {
            "hidden_states": hidden_states,
            "residual": residual,
            "post": post,
            "comb": comb,
        }
    ]


def test_mhc_pre_empty_input_preserves_output_contract(monkeypatch) -> None:
    residual = torch.empty(0, 4, 8, dtype=torch.bfloat16)
    fn = torch.empty(24, 32, dtype=torch.float32)
    scale = torch.ones(3, dtype=torch.float32)
    base = torch.zeros(24, dtype=torch.float32)

    def fail_select(*args, **kwargs):
        raise AssertionError("empty input must not dispatch a kernel")

    monkeypatch.setattr(mhc_ops, "select_kernel", fail_select)

    layer_input, post, comb = mhc_ops.mhc_pre(
        residual,
        fn,
        scale,
        base,
        rms_eps=1e-6,
        hc_eps=1e-6,
        sinkhorn_iters=2,
    )

    assert layer_input.shape == (0, 8)
    assert layer_input.dtype == torch.bfloat16
    assert post.shape == (0, 4, 1)
    assert post.dtype == torch.float32
    assert comb.shape == (0, 4, 4)
    assert comb.dtype == torch.float32


def _mhc_reference(
    residual: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    *,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens, hc_mult, hidden_size = residual.shape
    x = residual.reshape(tokens, hc_mult * hidden_size).float()
    rms = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + rms_eps)
    mixes = F.linear(x, fn) * rms
    pre_raw, post_raw, comb_raw = torch.split(
        mixes, [hc_mult, hc_mult, hc_mult * hc_mult], dim=-1
    )
    pre_base, post_base, comb_base = torch.split(
        base, [hc_mult, hc_mult, hc_mult * hc_mult]
    )
    pre = torch.sigmoid(pre_raw * scale[0] + pre_base) + hc_eps
    post = (torch.sigmoid(post_raw * scale[1] + post_base) * 2.0).unsqueeze(-1)
    comb = torch.softmax(
        comb_raw.reshape(tokens, hc_mult, hc_mult) * scale[2]
        + comb_base.reshape(1, hc_mult, hc_mult),
        dim=-1,
    )
    comb = comb + hc_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + hc_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
    layer_input = torch.sum(pre.unsqueeze(-1) * residual.float(), dim=1)
    return layer_input.to(torch.bfloat16), post, comb


@pytest.mark.skipif(
    not current_platform().is_nvidia,
    reason="mHC GPU kernels require an NVIDIA GPU",
)
def test_mhc_pre_and_post_match_reference(
    device: str,
    fresh_registry,
) -> None:
    importlib.reload(deep_gemm_mhc)
    if deep_gemm_mhc.deep_gemm is None:
        pytest.skip("DeepGEMM mHC is unavailable")

    from tokenspeed_kernel import mhc_post, mhc_pre

    torch.manual_seed(1234)
    residual = torch.randn(3, 4, 64, device=device, dtype=torch.bfloat16)
    fn = torch.randn(24, 256, device=device, dtype=torch.float32)
    scale = torch.tensor([0.7, 1.1, 0.5], device=device, dtype=torch.float32)
    base = torch.randn(24, device=device, dtype=torch.float32)

    actual = mhc_pre(
        residual,
        fn,
        scale,
        base,
        rms_eps=1e-6,
        hc_eps=1e-6,
        sinkhorn_iters=3,
        solution="deep_gemm",
    )
    expected = _mhc_reference(
        residual,
        fn,
        scale,
        base,
        rms_eps=1e-6,
        hc_eps=1e-6,
        sinkhorn_iters=3,
    )

    torch.testing.assert_close(actual[0], expected[0], rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(actual[1], expected[1], rtol=1e-2, atol=5e-3)
    torch.testing.assert_close(actual[2], expected[2], rtol=1e-2, atol=5e-3)

    updated = mhc_post(
        actual[0],
        residual,
        actual[1],
        actual[2],
        solution="deep_gemm",
    )
    expected_updated = torch.einsum("tnm,tnh->tmh", actual[2].float(), residual.float())
    expected_updated += actual[1].float() * actual[0].float().unsqueeze(1)
    torch.testing.assert_close(
        updated,
        expected_updated.to(torch.bfloat16),
        rtol=2e-2,
        atol=2e-2,
    )
