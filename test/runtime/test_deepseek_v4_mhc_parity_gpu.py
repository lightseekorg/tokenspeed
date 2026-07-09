# Copyright (c) 2026 LightSeek Foundation

"""GPU parity for fused-mHC allinone vs composed routing at production dims.

Compares all four outputs (residual, layer_input, post, comb) of the allinone
kernel against the composed post_mapping + prenorm-GEMM + big-fuse path on
identical inputs, across the routing boundary shapes and over an 8-layer
chained iteration, with absolute and relative error bounds.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

# CI registration (AST-parsed, runtime no-op). The TRT-LLM mHC kernels only
# support NVIDIA sm100, so restrict to B200-class runners.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(
    est_time=120,
    suite="runtime-1gpu",
    disabled_on_runners=["h100-*", "b300-*", "*mi3*", "linux-mi*"],
    disabled_on_runners_reason="TRT-LLM mHC kernels are sm100-only",
)

os.environ.setdefault("TOKENSPEED_V4_MHC_BACKEND", "trtllm")

from tokenspeed.runtime.layers import deepseek_v4_mhc as mhc  # noqa: E402


def _is_sm100() -> bool:
    return (
        torch.cuda.is_available()
        and torch.version.hip is None
        and torch.cuda.get_device_capability(0) == (10, 0)
    )


pytestmark = pytest.mark.skipif(not _is_sm100(), reason="requires NVIDIA sm100")

HC, HID = 4, 7168
MIX = (2 + HC) * HC
RMS_EPS, HC_EPS, SINK = 1e-6, 1e-2, 4
REL_TOL = 0.03


def _inputs(m, seed):
    torch.manual_seed(seed)
    dev = "cuda:0"
    return (
        torch.randn(m, HID, device=dev, dtype=torch.bfloat16),
        torch.randn(m, HC, HID, device=dev, dtype=torch.bfloat16) * 0.5,
        torch.rand(m, HC, 1, device=dev),
        torch.rand(m, HC, HC, device=dev) / HC,
        torch.randn(MIX, HC * HID, device=dev) * 0.02,
        torch.ones(3, device=dev),
        torch.zeros(MIX, device=dev),
    )


def _allinone(x, res, post, comb, fn, scale, base, ws):
    if res.shape[0] > mhc._MHC_FUSED_ALLINONE_MAX_TOKENS:
        pytest.skip("allinone reference only defined for B<=threshold")
    return mhc._trtllm_mhc_fused_hc(
        x, res, post, comb, fn, scale, base, RMS_EPS, HC_EPS, SINK, ws
    )


def _composed(x, res, post, comb, fn, scale, base):
    res_cur = mhc._trtllm_mhc_post(x, res, post, comb)
    layer_in, post_cur, comb_cur = mhc._trtllm_mhc_pre(
        res_cur, fn, scale, base, RMS_EPS, HC_EPS, SINK
    )
    return res_cur, layer_in, post_cur, comb_cur


def _check(oa, ob, m, layer):
    names = ("residual", "layer_input", "post", "comb")
    for name, ta, tb in zip(names, oa, ob):
        fa, fb = ta.float(), tb.float()
        abs_err = (fa - fb).abs().max().item()
        rel_err = ((fa - fb).norm() / fb.norm().clamp_min(1e-9)).item()
        if name in ("residual", "layer_input"):
            # bf16 outputs: bound by bf16 ULP at the reference magnitude
            # (the two paths quantize to bf16 at different points), floored
            # for near-zero references and growing with chain depth since
            # ULP-level drift compounds per chained layer.
            abs_tol = max(0.02, fb.abs().max().item() * 2**-7 * (2 + layer))
        else:
            # fp32 post/comb: O(1) sigmoid-scale values computed through
            # bf16-quantized intermediates; fixed small bound.
            abs_tol = 0.02 * (1 + 0.5 * layer)
        assert (
            abs_err <= abs_tol
        ), f"M={m} L{layer} {name} abs={abs_err:.4f} tol={abs_tol:.4f}"
        assert rel_err < REL_TOL, f"M={m} L{layer} {name} rel={rel_err:.4f}"


@pytest.mark.parametrize("m", [16, 32])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_mhc_allinone_vs_composed_chained_parity(m, seed):
    # At/below the threshold both paths are reachable; the composed path must
    # agree with allinone through an 8-layer chained iteration.
    x, res, post, comb, fn, scale, base = _inputs(m, seed)
    ws = mhc.MhcFusedWorkspace()
    ws.reset()
    fa = (x, res, post, comb)
    fb = (x, res, post, comb)
    for layer in range(8):
        oa = tuple(t.clone() for t in _allinone(*fa[:4], fn, scale, base, ws))
        ob = _composed(*fb[:4], fn, scale, base)
        _check(oa, ob, m, layer)

        # Normalize the re-injected sublayer output like a real network's
        # norms would, so synthetic magnitudes stay bounded across the chain.
        def _renorm(t):
            return (t.float() / t.float().std().clamp_min(1e-3)).to(torch.bfloat16)

        fa = (
            _renorm(oa[1]),
            oa[0],
            oa[2].view(m, HC, 1),
            oa[3].view(m, HC, HC),
        )
        fb = (
            _renorm(ob[1]),
            ob[0],
            ob[2].view(m, HC, 1),
            ob[3].view(m, HC, HC),
        )


@pytest.mark.parametrize("m", [36, 40, 64])
def test_mhc_routed_path_matches_composed_reference(m):
    # Above the threshold the wrapper must return exactly the composed-path
    # results (routing correctness at the boundary shapes).
    x, res, post, comb, fn, scale, base = _inputs(m, 0)
    ws = mhc.MhcFusedWorkspace()
    ws.reset()
    routed = mhc._trtllm_mhc_fused_hc(
        x, res, post, comb, fn, scale, base, RMS_EPS, HC_EPS, SINK, ws
    )
    ref = _composed(x, res, post, comb, fn, scale, base)
    for ta, tb in zip(routed, ref):
        assert torch.equal(ta.float(), tb.float())


@pytest.mark.parametrize("m", [36, 40, 64])
@pytest.mark.parametrize("seed", [0, 1])
def test_mhc_large_m_allinone_vs_composed_chained_parity(m, seed):
    # The path being REPLACED: force the allinone kernel above the routing
    # threshold (it is ~10x slower there but numerically defined) and compare
    # against the composed replacement over an 8-layer chained iteration, so
    # cross-layer drift on the affected shapes is bounded too.
    from unittest.mock import patch

    x, res, post, comb, fn, scale, base = _inputs(m, seed)
    ws = mhc.MhcFusedWorkspace()
    ws.reset()
    fa = (x, res, post, comb)
    fb = (x, res, post, comb)

    def _renorm(t):
        return (t.float() / t.float().std().clamp_min(1e-3)).to(torch.bfloat16)

    for layer in range(8):
        with patch.object(mhc, "_MHC_FUSED_ALLINONE_MAX_TOKENS", 1 << 20):
            oa = tuple(
                t.clone()
                for t in mhc._trtllm_mhc_fused_hc(
                    *fa[:4], fn, scale, base, RMS_EPS, HC_EPS, SINK, ws
                )
            )
        ob = _composed(*fb[:4], fn, scale, base)
        _check(oa, ob, m, layer)
        fa = (_renorm(oa[1]), oa[0], oa[2].view(m, HC, 1), oa[3].view(m, HC, HC))
        fb = (_renorm(ob[1]), ob[0], ob[2].view(m, HC, 1), ob[3].view(m, HC, HC))


@pytest.mark.parametrize("m", [36, 64])
def test_mhc_composed_graph_capture_replay(m):
    # The composed branch allocates temporaries inside CUDA graph capture;
    # replay must be stable, respond to static-input mutation, match eager,
    # and not grow the graph pool across replays.
    x, res, post, comb, fn, scale, base = _inputs(m, 0)
    ws = mhc.MhcFusedWorkspace()
    ws.reset()

    def call():
        return mhc._trtllm_mhc_fused_hc(
            x, res, post, comb, fn, scale, base, RMS_EPS, HC_EPS, SINK, ws
        )

    for _ in range(3):
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = call()
    for _ in range(10):
        graph.replay()
    torch.cuda.synchronize()
    eager = _composed(x, res, post, comb, fn, scale, base)
    for ta, tb in zip(out, eager):
        assert torch.equal(ta.float(), tb.float())
    # Mutate a static input and replay again; the captured graph must track it.
    x.copy_(torch.randn_like(x))
    graph.replay()
    torch.cuda.synchronize()
    eager2 = _composed(x, res, post, comb, fn, scale, base)
    assert torch.equal(out[1].float(), eager2[1].float())
    reserved = torch.cuda.memory_reserved()
    for _ in range(50):
        graph.replay()
    torch.cuda.synchronize()
    assert torch.cuda.memory_reserved() == reserved, "graph pool grew on replay"
