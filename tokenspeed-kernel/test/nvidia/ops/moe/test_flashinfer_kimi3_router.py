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

"""Optional Kimi-K3 routing through the public output-buffer contract."""

import pytest
import torch
from tokenspeed_kernel.ops.moe import _moe_sigmoid_bias_topk
from tokenspeed_kernel.platform import ArchVersion, current_platform
from tokenspeed_kernel.selection import NoKernelFoundError, kernel_override
from tokenspeed_kernel.thirdparty.cuda.flashinfer_kimi_k3 import fused_router_available

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or current_platform().arch_version != ArchVersion(10, 3),
    reason="requires SM103",
)
native = pytest.mark.skipif(
    not fused_router_available(), reason="FlashInfer Kimi-K3 sources unavailable"
)


def check_routes(actual, expected):
    aw, ai = actual
    ew, ei = expected
    ai, a_order = ai.sort(-1)
    ei, e_order = ei.sort(-1)
    torch.testing.assert_close(ai.to(torch.int64), ei.to(torch.int64), atol=0, rtol=0)
    torch.testing.assert_close(
        aw.gather(1, a_order).float(),
        ew.gather(1, e_order).float(),
        atol=2e-7,
        rtol=2e-6,
    )


@native
@pytest.mark.parametrize("rows", [3072, 4097, 8192])
@pytest.mark.parametrize("pattern", ["random", "ties", "saturated"])
def test_fused_router_buffers_and_graph(rows, pattern):
    torch.manual_seed(rows)
    logits = torch.randn(rows, 896, device="cuda", dtype=torch.float32)
    bias = torch.randn(896, device="cuda", dtype=torch.float32) * 0.01
    if pattern == "ties":
        logits.zero_()
        bias.zero_()
    elif pattern == "saturated":
        logits.copy_(logits.sign() * 100)
        bias.zero_()
    out = (
        torch.empty(rows, 16, device="cuda", dtype=torch.float32),
        torch.empty(rows, 16, device="cuda", dtype=torch.int32),
    )
    reference = _moe_sigmoid_bias_topk(logits, bias, 16, solution="triton")
    actual = _moe_sigmoid_bias_topk(
        logits, bias, 16, weights_dtype=torch.bfloat16, out=out
    )
    assert actual[0] is out[0] and actual[1] is out[1]
    check_routes(actual, reference)
    check_routes(
        _moe_sigmoid_bias_topk(logits, bias, 16, solution="flashinfer"), reference
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _moe_sigmoid_bias_topk(logits, bias, 16, out=out)
    logits.normal_()
    graph.replay()
    check_routes(captured, _moe_sigmoid_bias_topk(logits, bias, 16, solution="triton"))
    scores = logits.sigmoid().gather(1, captured[1].long())
    torch.testing.assert_close(
        captured[0], scores / scores.sum(-1, keepdim=True), atol=2e-7, rtol=2e-6
    )


@pytest.mark.parametrize(
    "case",
    [
        "small",
        "large",
        "int64",
        "bf16_weights",
        "scale",
        "unnormalized",
        "bf16_logits",
        "bf16_bias",
        "strided",
        "topk",
        "experts",
    ],
)
def test_other_contracts_keep_existing_router(case):
    rows = 16 if case == "small" else 8193 if case == "large" else 3072
    experts = 128 if case == "experts" else 896
    topk = 8 if case == "topk" else 16
    logits = torch.randn(
        rows, experts + (8 if case == "strided" else 0), device="cuda"
    )[:, :experts]
    bias = torch.randn(experts, device="cuda") * 0.01
    if case == "bf16_logits":
        logits = logits.bfloat16()
    if case == "bf16_bias":
        bias = bias.bfloat16()
    dtype = torch.bfloat16 if case == "bf16_weights" else torch.float32
    kwargs = dict(
        routed_scaling_factor=1.5 if case == "scale" else 1.0,
        normalize_topk_weights=case != "unnormalized",
        weights_dtype=dtype,
    )
    out = (
        torch.empty(rows, topk, device="cuda", dtype=dtype),
        torch.empty(
            rows,
            topk,
            device="cuda",
            dtype=torch.int64 if case == "int64" else torch.int32,
        ),
    )
    actual = _moe_sigmoid_bias_topk(logits, bias, topk, out=out, **kwargs)
    expected = _moe_sigmoid_bias_topk(logits, bias, topk, solution="triton", **kwargs)
    check_routes(actual, expected)
    if fused_router_available():
        with pytest.raises((NoKernelFoundError, ValueError)):
            _moe_sigmoid_bias_topk(
                logits, bias, topk, solution="flashinfer", out=out, **kwargs
            )


@native
def test_explicit_override_and_mapping_preserve_outputs():
    logits = torch.randn(4096, 896, device="cuda")
    bias = torch.randn(896, device="cuda") * 0.01
    with kernel_override(
        "moe", "sigmoid_bias_topk", "triton_minimax_sigmoid_bias_topk"
    ):
        chosen = _moe_sigmoid_bias_topk(logits, bias, 16)
    check_routes(chosen, _moe_sigmoid_bias_topk(logits, bias, 16, solution="triton"))
    out = (
        torch.empty(4096, 16, device="cuda"),
        torch.empty(4096, 16, device="cuda", dtype=torch.int32),
    )
    mapping = torch.arange(895, -1, -1, device="cuda", dtype=torch.int64)
    actual = _moe_sigmoid_bias_topk(
        logits, bias, 16, logical_to_physical_map=mapping, out=out
    )
    expected = (chosen[0], mapping[chosen[1].long()].int())
    check_routes(actual, expected)
    assert actual[0] is out[0] and actual[1] is out[1]
