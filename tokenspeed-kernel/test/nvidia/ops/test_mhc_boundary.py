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

"""Blackwell 16-CTA DSMEM cluster shifted mHC boundary tests."""

from __future__ import annotations

import pytest
import torch
from utils import assert_no_cute_compile, is_nvidia

if not is_nvidia():
    pytest.skip("requires an NVIDIA GPU", allow_module_level=True)
pytest.importorskip("cutlass")

from tokenspeed_kernel import platform  # noqa: E402
from tokenspeed_kernel.ops.residual import (  # noqa: E402
    mhc_boundary_max_decode_tokens,
    prepare_mhc_boundary_weight,
    try_mhc_shifted_post_pre_boundary,
    try_mhc_shifted_post_pre_norm,
    warmup_mhc_boundary,
)
from tokenspeed_kernel.thirdparty.cuda.mega_mhc import (  # noqa: E402
    is_mega_mhc_available,
)
from tokenspeed_kernel.thirdparty.cute_dsl import mhc_boundary as cute_mhc  # noqa: E402

if not (
    platform.ArchVersion(10, 0)
    <= platform.current_platform().arch_version
    <= platform.ArchVersion(10, 9)
):
    pytest.skip("requires an SM100-family GPU", allow_module_level=True)


def _inputs(tokens: int, hidden: int, iters: int, *, seed: int = 41) -> dict:
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape, dtype):
        return torch.randn(*shape, dtype=dtype, device="cuda", generator=gen)

    weight = randn(24, 4 * hidden, dtype=torch.float32) * 0.03
    boundary_weight = prepare_mhc_boundary_weight(weight)
    assert boundary_weight is not None
    return {
        "x": randn(tokens, hidden, dtype=torch.bfloat16),
        "residual": randn(tokens, 4, hidden, dtype=torch.bfloat16),
        "pre": torch.rand(tokens, 4, device="cuda", generator=gen),
        "post": torch.rand(tokens, 4, device="cuda", generator=gen) * 2,
        "comb": torch.rand(tokens, 4, 4, device="cuda", generator=gen).softmax(-2),
        "raw_weight": weight,
        "boundary_weight": boundary_weight,
        "scale": torch.tensor([0.7, 1.1, 0.5], device="cuda"),
        "base": randn(24, dtype=torch.float32) * 0.1,
        "rms_eps": 1e-20,
        "hc_eps": 1e-6,
        "sinkhorn_iters": iters,
        "norm_weight": randn(hidden, dtype=torch.bfloat16) * 0.1 + 1,
        "norm_eps": 1e-5,
    }


def _call_boundary(args: dict):
    return try_mhc_shifted_post_pre_boundary(
        args["x"],
        args["residual"],
        args["pre"],
        args["post"],
        args["comb"],
        args["boundary_weight"],
        args["scale"],
        args["base"],
        args["rms_eps"],
        args["hc_eps"],
        args["sinkhorn_iters"],
        args["norm_weight"],
        args["norm_eps"],
    )


def _reference(args: dict):
    new_residual = (
        (args["comb"][..., None] * args["residual"].float()[:, :, None, :]).sum(1)
        + args["post"][..., None] * args["x"].float()[:, None, :]
    ).bfloat16()
    r = new_residual.float()
    collapsed = (args["pre"][..., None] * r).sum(1).bfloat16()
    cf = collapsed.float()
    norm_rstd = torch.rsqrt(cf.square().mean(-1, keepdim=True) + args["norm_eps"])
    layer_input = (cf * norm_rstd * args["norm_weight"].float()).bfloat16()

    hidden = r.shape[-1]
    flat = r.reshape(-1, 4 * hidden)
    hc_rstd = torch.rsqrt(flat.square().mean(-1, keepdim=True) + args["rms_eps"])
    proj = (flat @ args["raw_weight"].T) * hc_rstd
    scale = args["scale"]
    base = args["base"]
    new_pre = torch.sigmoid(proj[:, :4] * scale[0] + base[:4]) + args["hc_eps"]
    new_post = torch.sigmoid(proj[:, 4:8] * scale[1] + base[4:8]) * 2.0
    m = (proj[:, 8:] * scale[2] + base[8:]).view(-1, 4, 4)
    m = m - m.amax(-1, keepdim=True)
    m = torch.exp(m)
    m = m / m.sum(-1, keepdim=True) + args["hc_eps"]
    m = m / (m.sum(-2, keepdim=True) + args["hc_eps"])
    for _ in range(1, args["sinkhorn_iters"]):
        m = m / (m.sum(-1, keepdim=True) + args["hc_eps"])
        m = m / (m.sum(-2, keepdim=True) + args["hc_eps"])
    return new_residual, layer_input, new_pre, new_post, m


def _reference_fp64(args: dict):
    x64 = args["x"].double()
    res64 = args["residual"].double()
    pre64 = args["pre"].double()
    post64 = args["post"].double()
    comb64 = args["comb"].double()
    new_residual = (
        (comb64[..., None] * res64[:, :, None, :]).sum(1)
        + post64[..., None] * x64[:, None, :]
    ).bfloat16()
    # Match the kernel's BF16 updated-residual input fed to the mix GEMM.
    r = new_residual.double()
    collapsed = (pre64[..., None] * r).sum(1).bfloat16()
    cf = collapsed.double()
    norm_rstd = torch.rsqrt(cf.square().mean(-1, keepdim=True) + args["norm_eps"])
    layer_input = (cf * norm_rstd * args["norm_weight"].double()).bfloat16()

    hidden = r.shape[-1]
    flat = r.reshape(-1, 4 * hidden)
    hc_rstd = torch.rsqrt(flat.square().mean(-1, keepdim=True) + args["rms_eps"])
    proj = (flat @ args["raw_weight"].double().T) * hc_rstd
    scale = args["scale"].double()
    base = args["base"].double()
    new_pre = torch.sigmoid(proj[:, :4] * scale[0] + base[:4]) + args["hc_eps"]
    new_post = torch.sigmoid(proj[:, 4:8] * scale[1] + base[4:8]) * 2.0
    m = (proj[:, 8:] * scale[2] + base[8:]).view(-1, 4, 4)
    m = m - m.amax(-1, keepdim=True)
    m = torch.exp(m)
    m = m / m.sum(-1, keepdim=True) + args["hc_eps"]
    m = m / (m.sum(-2, keepdim=True) + args["hc_eps"])
    for _ in range(1, args["sinkhorn_iters"]):
        m = m / (m.sum(-1, keepdim=True) + args["hc_eps"])
        m = m / (m.sum(-2, keepdim=True) + args["hc_eps"])
    return new_residual, layer_input, new_pre.float(), new_post.float(), m.float()


@pytest.mark.parametrize(
    "tokens", [1, 14, 15, 17, 33, 64, 128, 224, 336, 337, 512, 1024, 2048]
)
@pytest.mark.parametrize("hidden", [2560, 5120])
@pytest.mark.parametrize("iters", [1, 3, 20])
def test_mhc_boundary_matches_reference_and_preserves_inputs(tokens, hidden, iters):
    args = _inputs(tokens, hidden, iters)
    snapshots = {
        k: args[k].clone()
        for k in (
            "x",
            "residual",
            "pre",
            "post",
            "comb",
            "scale",
            "base",
            "norm_weight",
        )
    }
    actual = _call_boundary(args)
    assert actual is not None
    assert len(actual) == 5
    expected_specs = (
        ((tokens, 4, hidden), torch.bfloat16),
        ((tokens, hidden), torch.bfloat16),
        ((tokens, 4), torch.float32),
        ((tokens, 4), torch.float32),
        ((tokens, 4, 4), torch.float32),
    )
    for tensor, (shape, dtype) in zip(actual, expected_specs, strict=True):
        assert tensor.shape == shape
        assert tensor.dtype == dtype
        assert tensor.is_cuda and tensor.is_contiguous()
    for name, before in snapshots.items():
        assert torch.equal(args[name], before)

    for got, ref in zip(actual, _reference(args), strict=True):
        torch.testing.assert_close(got, ref, atol=0.02, rtol=0.01)
    ref64 = _reference_fp64(args)
    for got, ref in zip(actual[2:], ref64[2:], strict=True):
        torch.testing.assert_close(got, ref, atol=5e-4, rtol=5e-4)

    if hidden % 1024 == 0 and is_mega_mhc_available():
        mega_args = {
            k: v for k, v in args.items() if k not in ("raw_weight", "boundary_weight")
        }
        mega_args["weight"] = args["raw_weight"]
        mega = try_mhc_shifted_post_pre_norm(**mega_args)
        assert mega is not None
        for got, ref in zip(actual, mega, strict=True):
            torch.testing.assert_close(got, ref, atol=0.02, rtol=0.01)


def test_warmup_mhc_boundary_leaves_no_cute_compile():
    hidden, iters = 5120, 20
    warmup_mhc_boundary(hidden, iters, device="cuda")
    torch.cuda.synchronize()
    args = _inputs(777, hidden, iters)
    with assert_no_cute_compile():
        for tokens in (3, 17, 48, 100, 336, 512, 777):
            sub = {
                **args,
                "x": args["x"][:tokens],
                "residual": args["residual"][:tokens],
                "pre": args["pre"][:tokens],
                "post": args["post"][:tokens],
                "comb": args["comb"][:tokens],
            }
            out = _call_boundary(sub)
            assert out is not None and out[1].shape == (tokens, hidden)
    torch.cuda.synchronize()


def test_cold_capture_returns_none_without_compiling():
    args = _inputs(16, 2560, 20)
    saved = dict(cute_mhc._KERNELS)
    cute_mhc._KERNELS.clear()
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=stream):
            out = _call_boundary(args)
        assert out is None
        assert not cute_mhc._KERNELS
    finally:
        del graph
        torch.cuda.synchronize()
        cute_mhc._KERNELS.clear()
        cute_mhc._KERNELS.update(saved)


def test_mhc_boundary_cudagraph_replay_and_interleaved_graphs():
    hidden, iters = 5120, 20
    warmup_mhc_boundary(hidden, iters, device="cuda")
    a16 = _inputs(16, hidden, iters, seed=11)
    a512 = _inputs(512, hidden, iters, seed=12)

    stream = torch.cuda.Stream()
    g16, g512 = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    stream.wait_stream(torch.cuda.current_stream())
    with assert_no_cute_compile():
        with torch.cuda.graph(g16, stream=stream):
            out16 = _call_boundary(a16)
        with torch.cuda.graph(g512, stream=stream):
            out512 = _call_boundary(a512)
    assert out16 is not None and out512 is not None

    for step in range(3):
        for args, graph, captured in ((a16, g16, out16), (a512, g512, out512)):
            args["x"].normal_()
            args["residual"].normal_()
            args["pre"].uniform_()
            args["post"].uniform_()
            args["comb"].copy_(torch.rand_like(args["comb"]).softmax(-2))
            graph.replay()
            torch.cuda.synchronize(stream)
            eager = _call_boundary(args)
            assert eager is not None
            for got, ref in zip(captured, eager, strict=True):
                assert torch.equal(
                    got.view(torch.uint8) if got.dtype == torch.bfloat16 else got,
                    ref.view(torch.uint8) if ref.dtype == torch.bfloat16 else ref,
                ), f"step={step}"


@pytest.mark.parametrize(
    "case",
    [
        "cpu",
        "empty",
        "hidden",
        "unaligned_x",
        "non_contiguous_residual",
        "bad_weight_shape",
        "bad_weight_dtype",
        "zero_sinkhorn",
    ],
)
def test_mhc_boundary_unsupported_inputs_return_none(case):
    args = _inputs(17, 2560, 20)
    if case == "cpu":
        args = {
            k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in args.items()
        }
    elif case == "empty":
        args = _inputs(0, 2560, 20)
    elif case == "hidden":
        assert (
            prepare_mhc_boundary_weight(torch.randn(24, 4 * 1024, device="cuda"))
            is None
        )
        return
    elif case == "unaligned_x":
        raw = torch.empty(17 * 2560 + 4, dtype=torch.bfloat16, device="cuda").normal_()
        args["x"] = raw[1 : 1 + 17 * 2560].view(17, 2560)
        assert args["x"].is_contiguous() and args["x"].data_ptr() % 16 != 0
    elif case == "non_contiguous_residual":
        args["residual"] = args["residual"].transpose(1, 2).contiguous().transpose(1, 2)
    elif case == "bad_weight_shape":
        args["boundary_weight"] = args["boundary_weight"][:-1].contiguous()
    elif case == "bad_weight_dtype":
        args["boundary_weight"] = args["boundary_weight"].float()
    elif case == "zero_sinkhorn":
        args["sinkhorn_iters"] = 0
    assert _call_boundary(args) is None


def test_mhc_boundary_reads_and_writes_past_2gb():
    tokens, hidden, iters = 110_000, 2560, 20
    # [110_000, 4, 2560] BF16 is ~2.25 GB (> 2^31 bytes), exercising int64 offsets.
    try:
        args = _inputs(tokens, hidden, iters, seed=91)
        assert args["residual"].numel() * args["residual"].element_size() > (1 << 31)
        actual = _call_boundary(args)
        assert actual is not None
        tail = slice(-8, None)
        head = slice(0, 8)
        for sl in (head, tail):
            sub = {
                **args,
                "x": args["x"][sl],
                "residual": args["residual"][sl],
                "pre": args["pre"][sl],
                "post": args["post"][sl],
                "comb": args["comb"][sl],
            }
            for got, ref in zip((t[sl] for t in actual), _reference(sub), strict=True):
                torch.testing.assert_close(got, ref, atol=0.02, rtol=0.01)
    finally:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def test_mhc_boundary_hi_lo_split_fp64_precision_and_cluster_limit():
    props = torch.cuda.get_device_properties("cuda")
    expected_clusters = max(1, (2 * props.multi_processor_count) // cute_mhc.CLUSTER)
    assert cute_mhc.max_decode_wave_clusters("cuda") == expected_clusters
    expected_max_tokens = (
        0
        if platform.current_platform().arch_version == platform.ArchVersion(10, 3)
        else expected_clusters * cute_mhc.TILE
    )
    assert mhc_boundary_max_decode_tokens("cuda") == expected_max_tokens

    args = _inputs(33, 5120, 20, seed=107)
    raw = args["raw_weight"] * 8.0
    bw = prepare_mhc_boundary_weight(raw)
    assert bw is not None
    args["raw_weight"] = raw
    args["boundary_weight"] = bw
    actual = _call_boundary(args)
    assert actual is not None
    ref64 = _reference_fp64(args)
    for got, ref in zip(actual[:2], ref64[:2], strict=True):
        torch.testing.assert_close(got, ref, atol=0.02, rtol=0.01)
    # Without the hi+lo BF16 weight split, 8x weights shift projections by ~5e-3+.
    for got, ref in zip(actual[2:], ref64[2:], strict=True):
        torch.testing.assert_close(got, ref, atol=5e-4, rtol=5e-4)


def test_mhc_boundary_max_decode_tokens_disabled_on_sm103(monkeypatch):
    from types import SimpleNamespace

    from tokenspeed_kernel.ops.residual._cute_dsl import mhc_boundary as backend

    props = torch.cuda.get_device_properties("cuda")
    expected_clusters = max(1, (2 * props.multi_processor_count) // cute_mhc.CLUSTER)

    monkeypatch.setattr(
        backend,
        "current_platform",
        lambda: SimpleNamespace(
            is_nvidia=True, arch_version=platform.ArchVersion(10, 3)
        ),
    )
    assert mhc_boundary_max_decode_tokens("cuda") == 0

    monkeypatch.setattr(
        backend,
        "current_platform",
        lambda: SimpleNamespace(
            is_nvidia=True, arch_version=platform.ArchVersion(10, 0)
        ),
    )
    assert mhc_boundary_max_decode_tokens("cuda") == expected_clusters * cute_mhc.TILE
