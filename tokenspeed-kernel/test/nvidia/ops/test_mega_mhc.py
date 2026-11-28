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
from tokenspeed_kernel.ops.residual import (
    try_mhc_shifted_post_pre_norm,
    try_mhc_shifted_post_pre_norm_quantize_fp8,
)
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


def _group32_reference(y):
    """UE8M0 exponents and E4M3 codes the group-32 quantizers write for y."""
    groups = y.float().unflatten(-1, (-1, 32))
    raw = groups.abs().amax(-1).clamp_min(1e-4) * (1.0 / 448.0)
    bits = raw.view(torch.int32)
    exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    inverse = (exponent << 23).view(torch.float32)
    codes = (groups / inverse[..., None]).flatten(-2).to(torch.float8_e4m3fn)
    return exponent, codes


@pytest.mark.parametrize(
    "tokens,hidden", [(1, 1024), (17, 5120), (257, 5120), (2048, 5120)]
)
def test_shifted_mhc_quantize_fp8_matches_group32_reference(tokens, hidden):
    args = _inputs(tokens, hidden, 20)
    plain = try_mhc_shifted_post_pre_norm(**args)
    actual = try_mhc_shifted_post_pre_norm_quantize_fp8(**args)
    assert plain is not None and actual is not None
    for got, ref in zip(actual[:5], plain, strict=True):
        torch.testing.assert_close(got, ref, atol=0.02, rtol=0.01)
    y, codes, packed = actual[1], actual[5], actual[6]
    assert codes.dtype == torch.float8_e4m3fn and codes.shape == (tokens, hidden)
    assert packed.dtype == torch.int32 and packed.shape == (tokens, hidden // 128)
    # MN-major words, the token count padded to four for TMA.
    assert packed.stride() == (1, (tokens + 3) // 4 * 4)
    shifts = torch.arange(4, device=packed.device) * 8
    exponent = ((packed[..., None] >> shifts) & 255).flatten(-2).to(torch.int32)
    expected_exponent, expected_codes = _group32_reference(y)
    torch.testing.assert_close(exponent, expected_exponent, atol=0, rtol=0)
    torch.testing.assert_close(
        codes.view(torch.uint8), expected_codes.view(torch.uint8), atol=0, rtol=0
    )


@pytest.mark.parametrize("tokens", [33, 512, 8192])
def test_shifted_mhc_refresh_on_two_streams(tokens):
    args = _inputs(tokens, 5120, 20)
    # A fresh stream needs its own native barrier initialization. CUDA graph
    # replay is unsupported: split barriers can read the previous replay's
    # partials.
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                args["x"].normal_()
                args["residual"].normal_()
                args["pre"].uniform_()
                actual = try_mhc_shifted_post_pre_norm(**args)
                for got, ref in zip(actual, _reference(args), strict=True):
                    torch.testing.assert_close(got, ref, atol=0.02, rtol=0.01)
        stream.synchronize()


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


@pytest.mark.parametrize("quantize_fp8", [False, True])
@pytest.mark.parametrize("tokens", [48, 1024])
def test_shifted_mhc_cudagraph_replay_after_stream_warmup(tokens, quantize_fp8):
    from tokenspeed_kernel.ops.residual import is_mega_mhc_capture_safe

    if not is_mega_mhc_capture_safe():
        pytest.skip("requires DeepGEMM >= 2.8.3 capture-safe MegaMHC")
    args = _inputs(tokens, 5120, 20)
    call = (
        try_mhc_shifted_post_pre_norm_quantize_fp8
        if quantize_fp8
        else try_mhc_shifted_post_pre_norm
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        # First eager call on ``stream`` initializes DeepGEMM per-stream barriers.
        for _ in range(2):
            assert call(**args) is not None
    stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = call(**args)
    assert captured is not None

    for _ in range(3):
        args["x"].normal_()
        args["residual"].normal_()
        args["pre"].uniform_()
        args["post"].uniform_()
        args["comb"].copy_(torch.rand_like(args["comb"]).softmax(-2))
        graph.replay()
        torch.cuda.synchronize(stream)
        for got, ref in zip(captured[:5], _reference(args), strict=True):
            torch.testing.assert_close(got, ref, atol=0.02, rtol=0.01)
        if quantize_fp8:
            exp_sf, exp_codes = _group32_reference(captured[1])
            packed = captured[6]
            shifts = torch.arange(4, device=packed.device) * 8
            got_sf = ((packed[..., None] >> shifts) & 255).flatten(-2).to(torch.int32)
            torch.testing.assert_close(got_sf, exp_sf, atol=0, rtol=0)
            torch.testing.assert_close(
                captured[5].view(torch.uint8),
                exp_codes.view(torch.uint8),
                atol=0,
                rtol=0,
            )


def test_is_mega_mhc_capture_safe_version_gate(monkeypatch):
    from tokenspeed_kernel.ops.residual import is_mega_mhc_capture_safe
    from tokenspeed_kernel.thirdparty.cuda import mega_mhc as native

    monkeypatch.setattr(native, "_api", lambda: SimpleNamespace(__version__="2.8.1"))
    assert not is_mega_mhc_capture_safe()
    monkeypatch.setattr(native, "_api", lambda: SimpleNamespace(__version__="2.8.2"))
    assert not is_mega_mhc_capture_safe()
    monkeypatch.setattr(
        native, "_api", lambda: SimpleNamespace(__version__="2.8.2.post20261115")
    )
    assert not is_mega_mhc_capture_safe()
    monkeypatch.setattr(native, "_api", lambda: SimpleNamespace(__version__="2.8.3"))
    assert is_mega_mhc_capture_safe()
    monkeypatch.setattr(native, "_api", lambda: SimpleNamespace(__version__=None))
    assert not is_mega_mhc_capture_safe()
    monkeypatch.setattr(
        native, "_api", lambda: SimpleNamespace(__version__="not-semver")
    )
    assert not is_mega_mhc_capture_safe()

    def _missing():
        raise ImportError("no deep_gemm")

    monkeypatch.setattr(native, "is_mega_mhc_available", lambda: True)
    monkeypatch.setattr(native, "_api", _missing)
    assert not is_mega_mhc_capture_safe()
    monkeypatch.setattr(native, "is_mega_mhc_available", lambda: False)
    monkeypatch.setattr(native, "_api", lambda: SimpleNamespace(__version__="2.8.2"))
    assert not is_mega_mhc_capture_safe()


@pytest.mark.parametrize("quantize_fp8", [False, True])
def test_shifted_mhc_two_graphs_on_same_stream_replay(quantize_fp8):
    from tokenspeed_kernel.ops.residual import is_mega_mhc_capture_safe

    if not is_mega_mhc_capture_safe():
        pytest.skip("requires DeepGEMM >= 2.8.3 capture-safe MegaMHC")
    call = (
        try_mhc_shifted_post_pre_norm_quantize_fp8
        if quantize_fp8
        else try_mhc_shifted_post_pre_norm
    )
    args_a = _inputs(1024, 5120, 20)
    args_b = _inputs(4096, 5120, 20)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            assert call(**args_a) is not None
            assert call(**args_b) is not None
    stream.synchronize()

    graph_a = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph_a, stream=stream):
        cap_a = call(**args_a)
    assert cap_a is not None

    graph_b = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph_b, stream=stream):
        cap_b = call(**args_b)
    assert cap_b is not None

    for tag in ("A", "B", "A", "B"):
        graph, args, cap = (
            (graph_a, args_a, cap_a) if tag == "A" else (graph_b, args_b, cap_b)
        )
        args["x"].normal_()
        args["residual"].normal_()
        args["pre"].uniform_()
        args["post"].uniform_()
        args["comb"].copy_(torch.rand_like(args["comb"]).softmax(-2))
        graph.replay()
        torch.cuda.synchronize(stream)
        for got, ref in zip(cap[:5], _reference(args), strict=True):
            torch.testing.assert_close(got, ref, atol=0.02, rtol=0.01)
        if quantize_fp8:
            exp_sf, exp_codes = _group32_reference(cap[1])
            packed = cap[6]
            shifts = torch.arange(4, device=packed.device) * 8
            got_sf = ((packed[..., None] >> shifts) & 255).flatten(-2).to(torch.int32)
            torch.testing.assert_close(got_sf, exp_sf, atol=0, rtol=0)
            torch.testing.assert_close(
                cap[5].view(torch.uint8),
                exp_codes.view(torch.uint8),
                atol=0,
                rtol=0,
            )
