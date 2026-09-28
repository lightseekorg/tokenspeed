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

"""Shifted MegaMHC arithmetic, fallback contracts and graph input refresh."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from tokenspeed_kernel.ops.residual import try_mhc_shifted_post_pre_norm
from tokenspeed_kernel.platform import current_platform

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not current_platform().is_nvidia
    or current_platform().arch_version.major != 10,
    reason="requires an SM100-family GPU",
)


def _inputs(tokens, hidden, iters):
    torch.manual_seed(41)

    def rand(*shape, dtype):
        return torch.randn(*shape, dtype=dtype, device="cuda")

    return dict(
        x=rand(tokens, hidden, dtype=torch.bfloat16),
        residual=rand(tokens, 4, hidden, dtype=torch.bfloat16),
        pre=torch.rand(tokens, 4, device="cuda"),
        post=torch.rand(tokens, 4, device="cuda") * 2,
        comb=torch.rand(tokens, 4, 4, device="cuda").softmax(-2),
        weight=rand(24, 4 * hidden, dtype=torch.float32) * 0.03,
        scale=torch.tensor([0.7, 1.1, 0.5], device="cuda"),
        base=rand(24, dtype=torch.float32) * 0.1,
        rms_eps=1e-20,
        hc_eps=1e-6,
        sinkhorn_iters=iters,
        norm_weight=rand(hidden, dtype=torch.bfloat16) * 0.1 + 1,
        norm_eps=1e-5,
    )


def _reference(a):
    # Preserve both BF16 rounding boundaries before RMSNorm.
    r = (
        (a["comb"][..., None] * a["residual"].float()[:, :, None, :]).sum(1)
        + a["post"][..., None] * a["x"].float()[:, None, :]
    ).bfloat16()
    y = (r.float() * a["pre"][..., None]).sum(1).bfloat16().float()
    y = (
        y
        * torch.rsqrt(y.square().mean(-1, keepdim=True) + a["norm_eps"])
        * a["norm_weight"].float()
    ).bfloat16()
    flat = r.float().flatten(1)
    logits = F.linear(flat, a["weight"]) * torch.rsqrt(
        flat.square().mean(-1, keepdim=True) + a["rms_eps"]
    )
    scale, base, eps = a["scale"], a["base"], a["hc_eps"]
    pre = (logits[:, :4] * scale[0] + base[:4]).sigmoid() + eps
    post = (logits[:, 4:8] * scale[1] + base[4:8]).sigmoid() * 2
    comb = (logits[:, 8:] * scale[2] + base[8:]).reshape(-1, 4, 4).softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(a["sinkhorn_iters"] - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return r, y, pre, post, comb


@pytest.mark.parametrize(
    "tokens,hidden,iters",
    [
        (1, 1024, 1),
        (16, 5120, 20),
        (17, 5120, 20),
        (64, 1024, 3),
        (257, 5120, 20),
        (2048, 5120, 20),
    ],
)
def test_shifted_mhc_matches_reference(tokens, hidden, iters):
    args = _inputs(tokens, hidden, iters)
    before = {k: v.clone() for k, v in args.items() if isinstance(v, torch.Tensor)}
    actual = try_mhc_shifted_post_pre_norm(**args)
    assert actual is not None
    for got, ref in zip(actual, _reference(args), strict=True):
        torch.testing.assert_close(got, ref, atol=0.02, rtol=0.01)
    for name, original in before.items():
        torch.testing.assert_close(args[name], original, atol=0, rtol=0)


@pytest.mark.parametrize("tokens", [33, 512, 8192])
def test_shifted_mhc_graph_refresh_on_two_streams(tokens):
    args = _inputs(tokens, 5120, 20)
    # A fresh stream needs its own native barrier initialization.
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            try_mhc_shifted_post_pre_norm(**args)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            actual = try_mhc_shifted_post_pre_norm(**args)
        for _ in range(3):
            args["x"].normal_()
            args["residual"].normal_()
            args["pre"].uniform_()
            graph.replay()
            for got, ref in zip(actual, _reference(args), strict=True):
                torch.testing.assert_close(got, ref, atol=0.02, rtol=0.01)
        del graph


@pytest.mark.parametrize(
    "case",
    [
        "cpu",
        "empty",
        "hidden",
        "residual_dtype",
        "norm_dtype",
        "strided",
        "mix_shape",
        "iterations",
        "dependency",
        "architecture",
    ],
)
def test_shifted_mhc_unsupported_returns_none(monkeypatch, case):
    from tokenspeed_kernel.ops.residual import deep_gemm as impl
    from tokenspeed_kernel.thirdparty.cuda import mega_mhc as native

    args = _inputs(17, 1024, 20)
    if case == "cpu":
        args = {
            k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in args.items()
        }
    elif case in ("empty", "hidden"):
        args = _inputs(
            0 if case == "empty" else 17, 128 if case == "hidden" else 1024, 20
        )
    elif case == "residual_dtype":
        args["residual"] = args["residual"].float()
    elif case == "norm_dtype":
        args["norm_weight"] = args["norm_weight"].float()
    elif case == "strided":
        args["residual"] = args["residual"].transpose(1, 2).contiguous().transpose(1, 2)
    elif case == "mix_shape":
        args["pre"] = args["pre"].unsqueeze(-1)
    elif case == "iterations":
        args["sinkhorn_iters"] = 0
    elif case == "dependency":
        monkeypatch.setattr(native, "is_mega_mhc_available", lambda: False)
    elif case == "architecture":
        monkeypatch.setattr(
            impl,
            "platform",
            SimpleNamespace(is_nvidia=True, arch_version=SimpleNamespace(major=9)),
        )
    assert try_mhc_shifted_post_pre_norm(**args) is None
