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

from unittest import mock

import pytest
import tokenspeed_kernel.ops.moe.gluon.sigmoid_topk as gluon_sigmoid_topk
import torch
from tokenspeed_kernel.ops.moe import moe_sigmoid_bias_topk
from utils import is_cdna4

if not is_cdna4():
    pytest.skip(
        "AMD CDNA4 is required for Gluon MXFP4 routing tests",
        allow_module_level=True,
    )


from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4 import routing  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused import (  # noqa: E402
    _biased_grouped_topk_reference,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.routing import (  # noqa: E402
    gluon_topk_route_supported,
    invoke_sigmoid_bias_topk_route_gluon,
    invoke_sigmoid_bias_topk_route_prefill_gluon,
)

_ROUTE_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _route_reference(
    logits: torch.Tensor,
    correction_bias: torch.Tensor,
    *,
    routed_scaling_factor: float = 2.827,
    normalize_topk_weights: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _biased_grouped_topk_reference(
        logits,
        correction_bias,
        8,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=routed_scaling_factor,
        normalize_topk_weights=normalize_topk_weights,
    )


def test_gluon_topk_route_rejects_cpu_tensor() -> None:
    router = torch.empty((16, 384), dtype=torch.bfloat16)

    assert not gluon_topk_route_supported(router, 8)


def test_public_sigmoid_bias_topk_dispatch_uses_gluon_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logits = torch.zeros((8, 384), device="cuda", dtype=torch.bfloat16)
    correction_bias = torch.zeros(384, device="cuda", dtype=torch.float32)
    sentinel_ids = torch.empty((8, 8), device="cuda", dtype=torch.int32)
    sentinel_weights = torch.empty((8, 8), device="cuda", dtype=torch.float32)

    def launch(route_input, bias, topk, **kwargs):
        assert route_input is logits
        assert bias is correction_bias
        assert topk == 8
        return sentinel_ids, sentinel_weights

    monkeypatch.setattr(
        gluon_sigmoid_topk,
        "invoke_sigmoid_bias_topk_route_gluon",
        launch,
    )
    actual_weights, actual_ids = moe_sigmoid_bias_topk(
        logits,
        correction_bias,
        8,
        routed_scaling_factor=2.827,
        normalize_topk_weights=True,
    )

    assert actual_weights is sentinel_weights
    assert actual_ids is sentinel_ids


def test_public_sigmoid_bias_topk_uses_per_token_float32_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logits = torch.zeros((2, 896), device="cuda", dtype=torch.float32)
    correction_bias = torch.zeros(896, device="cuda", dtype=torch.float32)
    sentinel_ids = torch.empty((2, 16), device="cuda", dtype=torch.int32)
    sentinel_weights = torch.empty((2, 16), device="cuda", dtype=torch.float32)

    def launch(route_input, bias, topk, **kwargs):
        assert route_input is logits
        assert bias is correction_bias
        assert topk == 16
        return sentinel_ids, sentinel_weights

    monkeypatch.setattr(
        gluon_sigmoid_topk,
        "invoke_sigmoid_bias_topk_route_prefill_gluon",
        launch,
    )
    actual_weights, actual_ids = moe_sigmoid_bias_topk(
        logits,
        correction_bias,
        16,
        routed_scaling_factor=2.827,
        normalize_topk_weights=True,
    )

    assert actual_weights is sentinel_weights
    assert actual_ids is sentinel_ids


def test_public_sigmoid_bias_topk_retains_decode_route_above_prefill_topk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logits = torch.zeros((2, 64), device="cuda", dtype=torch.float32)
    correction_bias = torch.zeros(64, device="cuda", dtype=torch.float32)
    sentinel_ids = torch.empty((2, 32), device="cuda", dtype=torch.int32)
    sentinel_weights = torch.empty((2, 32), device="cuda", dtype=torch.float32)

    def launch(route_input, bias, topk, **kwargs):
        assert route_input is logits
        assert bias is correction_bias
        assert topk == 32
        return sentinel_ids, sentinel_weights

    monkeypatch.setattr(
        gluon_sigmoid_topk,
        "invoke_sigmoid_bias_topk_route_gluon",
        launch,
    )
    monkeypatch.setattr(
        gluon_sigmoid_topk,
        "invoke_sigmoid_bias_topk_route_prefill_gluon",
        mock.Mock(side_effect=AssertionError("top-k 32 must use the decode route")),
    )
    actual_weights, actual_ids = moe_sigmoid_bias_topk(
        logits,
        correction_bias,
        32,
        routed_scaling_factor=2.827,
        normalize_topk_weights=True,
        solution="gluon",
    )

    assert actual_weights is sentinel_weights
    assert actual_ids is sentinel_ids


@pytest.mark.parametrize("dtype", _ROUTE_DTYPES)
def test_sigmoid_bias_topk_route_gluon_fuses_sigmoid(
    monkeypatch: pytest.MonkeyPatch,
    dtype: torch.dtype,
) -> None:
    logits = torch.zeros((1, 8), device="cuda", dtype=dtype)
    correction_bias = torch.zeros(8, device="cuda", dtype=torch.float32)
    sentinel_ids = torch.empty((1, 2), device="cuda", dtype=torch.int32)
    sentinel_weights = torch.empty((1, 2), device="cuda", dtype=torch.float32)

    def launch(route_input, bias, topk, **kwargs):
        assert route_input is logits
        assert bias is correction_bias
        assert topk == 2
        return sentinel_ids, sentinel_weights

    monkeypatch.setattr(routing, "_launch_sigmoid_bias_topk_route_gluon", launch)
    actual_ids, actual_weights = invoke_sigmoid_bias_topk_route_gluon(
        logits, correction_bias, 2
    )

    assert actual_ids is sentinel_ids
    assert actual_weights is sentinel_weights


@pytest.mark.parametrize("dtype", _ROUTE_DTYPES)
def test_sigmoid_bias_topk_route_gluon_matches_supported_dtype_reference(
    dtype: torch.dtype,
) -> None:
    logits = torch.zeros((16, 16), device="cuda", dtype=dtype)
    correction_bias = torch.arange(16, device="cuda", dtype=torch.float32)
    expected_weights, expected_ids = _route_reference(
        logits, correction_bias, routed_scaling_factor=2.0
    )

    actual_ids, actual_weights = invoke_sigmoid_bias_topk_route_gluon(
        logits,
        correction_bias,
        8,
        routed_scaling_factor=2.0,
        normalize_topk_weights=True,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(actual_ids, expected_ids, atol=0, rtol=0)
    torch.testing.assert_close(actual_weights, expected_weights, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", _ROUTE_DTYPES)
@pytest.mark.parametrize(
    "logit_value",
    [pytest.param(float("nan"), id="nan"), pytest.param(-float("inf"), id="neg-inf")],
)
@pytest.mark.parametrize("normalize_topk_weights", [False, True])
def test_sigmoid_bias_topk_route_gluon_matches_nan_and_neg_inf_reference(
    dtype: torch.dtype,
    logit_value: float,
    normalize_topk_weights: bool,
) -> None:
    logits = torch.full((16, 16), logit_value, device="cuda", dtype=dtype)
    correction_bias = torch.zeros(16, device="cuda", dtype=torch.float32)
    expected_weights, expected_ids = _route_reference(
        logits,
        correction_bias,
        routed_scaling_factor=2.0,
        normalize_topk_weights=normalize_topk_weights,
    )

    actual_ids, actual_weights = invoke_sigmoid_bias_topk_route_gluon(
        logits,
        correction_bias,
        8,
        routed_scaling_factor=2.0,
        normalize_topk_weights=normalize_topk_weights,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(actual_ids, expected_ids, atol=0, rtol=0)
    torch.testing.assert_close(
        actual_weights,
        expected_weights,
        atol=0,
        rtol=0,
        equal_nan=True,
    )


@pytest.mark.parametrize(
    ("num_tokens", "seed"),
    [(8, 17), (8, 990611), (8, 20260731), (16, 17), (16, 990611), (16, 20260731)],
)
def test_sigmoid_bias_topk_route_gluon_matches_bf16_e384(
    num_tokens: int,
    seed: int,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn(
        (num_tokens, 384),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    correction_bias = (
        torch.randn(384, device="cuda", dtype=torch.float32, generator=generator) * 0.01
    )
    expected_weights, expected_ids = _route_reference(logits, correction_bias)

    actual_ids, actual_weights = invoke_sigmoid_bias_topk_route_gluon(
        logits,
        correction_bias,
        8,
        routed_scaling_factor=2.827,
        normalize_topk_weights=True,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(actual_ids, expected_ids, atol=0, rtol=0)
    torch.testing.assert_close(actual_weights, expected_weights, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", _ROUTE_DTYPES)
def test_sigmoid_bias_topk_route_gluon_handles_strided_inputs(
    dtype: torch.dtype,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(31973)
    logits = torch.randn(
        (16, 32),
        device="cuda",
        dtype=dtype,
        generator=generator,
    )[:, ::2]
    correction_bias_base = (
        torch.randn(32, device="cuda", dtype=torch.float32, generator=generator) * 0.01
    )
    correction_bias = correction_bias_base[::2]
    assert not logits.is_contiguous()
    assert not correction_bias.is_contiguous()
    expected_ids, expected_weights = invoke_sigmoid_bias_topk_route_gluon(
        logits.contiguous(),
        correction_bias.contiguous(),
        8,
        routed_scaling_factor=2.827,
        normalize_topk_weights=True,
    )

    actual_ids, actual_weights = invoke_sigmoid_bias_topk_route_gluon(
        logits,
        correction_bias,
        8,
        routed_scaling_factor=2.827,
        normalize_topk_weights=True,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(actual_ids, expected_ids, atol=0, rtol=0)
    torch.testing.assert_close(actual_weights, expected_weights, atol=0, rtol=0)


def test_prefill_sigmoid_bias_topk_route_gluon_handles_strided_inputs() -> None:
    generator = torch.Generator(device="cuda").manual_seed(61129)
    logits = torch.randn(
        (32, 32),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )[:, ::2]
    correction_bias_base = (
        torch.randn(32, device="cuda", dtype=torch.float32, generator=generator) * 0.01
    )
    correction_bias = correction_bias_base[::2]
    assert not logits.is_contiguous()
    assert not correction_bias.is_contiguous()
    expected_ids, expected_weights = invoke_sigmoid_bias_topk_route_prefill_gluon(
        logits.contiguous(),
        correction_bias.contiguous(),
        8,
        routed_scaling_factor=2.827,
        normalize_topk_weights=True,
    )

    actual_ids, actual_weights = invoke_sigmoid_bias_topk_route_prefill_gluon(
        logits,
        correction_bias,
        8,
        routed_scaling_factor=2.827,
        normalize_topk_weights=True,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(actual_ids, expected_ids, atol=0, rtol=0)
    torch.testing.assert_close(actual_weights, expected_weights, atol=0, rtol=0)


def test_sigmoid_bias_topk_route_gluon_matches_bf16_extremes() -> None:
    logits = torch.tensor(
        [[-float("inf"), -100.0, -10.0, -1.0, -0.0, 0.0, 1.0, float("inf")]],
        device="cuda",
        dtype=torch.bfloat16,
    )
    correction_bias = torch.zeros(8, device="cuda", dtype=torch.float32)
    expected_weights = logits.sigmoid().float()

    actual_ids, actual_weights = invoke_sigmoid_bias_topk_route_gluon(
        logits, correction_bias, 8, normalize_topk_weights=False
    )
    torch.cuda.synchronize()
    actual_weights_by_id = torch.empty_like(actual_weights).scatter(
        1, actual_ids.long(), actual_weights
    )

    torch.testing.assert_close(
        actual_ids.sort(dim=1).values,
        torch.arange(8, device="cuda", dtype=torch.int32).unsqueeze(0),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(actual_weights_by_id, expected_weights, atol=0, rtol=0)


def test_sigmoid_bias_topk_route_gluon_matches_bf16_rounding_boundaries() -> None:
    logits = torch.full((2, 16), -float("inf"), device="cuda", dtype=torch.bfloat16)
    logits[0, :8] = torch.tensor(
        [
            float("inf"),
            -5.53125,
            -5.53125,
            -float("inf"),
            -float("inf"),
            -float("inf"),
            -float("inf"),
            -float("inf"),
        ],
        device="cuda",
        dtype=torch.bfloat16,
    )
    logits[1, :8] = torch.tensor(
        [
            -2.03125,
            2.03125,
            -float("inf"),
            -float("inf"),
            -float("inf"),
            -float("inf"),
            -float("inf"),
            -float("inf"),
        ],
        device="cuda",
        dtype=torch.bfloat16,
    )
    correction_bias = torch.full((16,), -100.0, device="cuda")
    correction_bias[:8] = torch.arange(8, 0, -1, device="cuda")
    scores = logits.sigmoid()
    choice = scores + correction_bias.unsqueeze(0)
    _, expected_ids = torch.topk(choice, 8, dim=-1, sorted=True)
    expected_weights = scores.gather(1, expected_ids)
    expected_weights = expected_weights / expected_weights.sum(dim=-1, keepdim=True)
    expected_weights *= 2.827

    actual_ids, actual_weights = invoke_sigmoid_bias_topk_route_gluon(
        logits,
        correction_bias,
        8,
        routed_scaling_factor=2.827,
        normalize_topk_weights=True,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(actual_ids, expected_ids.to(torch.int32), atol=0, rtol=0)
    torch.testing.assert_close(actual_weights, expected_weights.float(), atol=0, rtol=0)


@pytest.mark.parametrize("dtype", _ROUTE_DTYPES)
def test_sigmoid_bias_topk_route_gluon_graph_replay_uses_updated_inputs(
    dtype: torch.dtype,
) -> None:
    logits = torch.zeros((16, 16), device="cuda", dtype=dtype)
    correction_bias = torch.arange(16, device="cuda", dtype=torch.float32)
    invoke_sigmoid_bias_topk_route_gluon(
        logits,
        correction_bias,
        8,
        routed_scaling_factor=2.0,
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_ids, graph_weights = invoke_sigmoid_bias_topk_route_gluon(
            logits,
            correction_bias,
            8,
            routed_scaling_factor=2.0,
            normalize_topk_weights=True,
        )

    correction_bias.copy_(torch.arange(15, -1, -1, device="cuda", dtype=torch.float32))
    expected_weights, expected_ids = _route_reference(
        logits,
        correction_bias,
        routed_scaling_factor=2.0,
    )
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(graph_ids, expected_ids, atol=0, rtol=0)
    torch.testing.assert_close(graph_weights, expected_weights, atol=0, rtol=0)
