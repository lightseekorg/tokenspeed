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

"""The measured decode-GEMV route: dispatch reaches it, and it computes right.

The routed backends are registered from sm100 up (GB200/B200 and GB300); below
that the whole module must be a no-op and the generic selection must be
untouched -- both directions are asserted here. The table is keyed per shape,
so an arch that never runs a listed shape simply never matches.
"""

from __future__ import annotations

import pytest
import tokenspeed_kernel.ops.gemm  # noqa: F401  (registration side effects)
import torch
from tokenspeed_kernel.ops.gemm.routed_gemv import MEASURED_ROUTE
from tokenspeed_kernel.ops.gemm.triton_gemv import _select, decode_gemv
from tokenspeed_kernel.platform import current_platform

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _is_routed_arch() -> bool:
    # Mirrors _CAPABILITY in routed_gemv: sm100 and up, not sm103 exactly.
    return (
        current_platform().vendor == "nvidia"
        and torch.cuda.get_device_capability() >= (10, 0)
    )


def _is_add3_arch() -> bool:
    # ADD3_ROUTE stores sm103-tuned TILE CONFIGS, not just a backend choice, so
    # it stays gated where it was swept -- mirrors _is_measured_arch. Widening
    # it would run another architecture's tuning parameters unmeasured.
    return (
        current_platform().vendor == "nvidia"
        and torch.cuda.get_device_capability() >= (10, 3)
    )


def _routed_cases():
    return sorted(MEASURED_ROUTE.items())


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _is_routed_arch(),
    reason="route is registered for sm100 and up",
)
@pytest.mark.parametrize("shape,backend", _routed_cases())
def test_dispatch_picks_the_measured_backend(shape, backend):
    m, n, k = shape
    _select.cache_clear()
    impl = _select(m, n, k, True)
    assert backend in getattr(
        impl, "__name__", ""
    ), f"M={m} N={n} K={k} resolved {impl} instead of the measured {backend}"


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _is_routed_arch(),
    reason="route is registered for sm100 and up",
)
@pytest.mark.parametrize("shape,backend", _routed_cases())
def test_routed_backend_matches_torch(shape, backend):
    m, n, k = shape
    torch.manual_seed(0)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    got = decode_gemv(x, w).float()
    ref = (x @ w.t()).float()
    # Accumulation order differs per kernel; a couple of ulp at ~sqrt(K).
    assert torch.allclose(got, ref, atol=0.5, rtol=2e-2), (
        f"M={m} N={n} K={k} via {backend}: "
        f"max abs err {(got - ref).abs().max().item():.4f}"
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _is_routed_arch(),
    reason="route is registered for sm100 and up",
)
def test_non_bf16_inputs_fall_back_to_torch():
    from tokenspeed_kernel.ops.gemm.routed_gemv import skinny_gemv, tgv_gemv

    x = torch.randn(1, 7168, device="cuda", dtype=torch.float16)
    w = torch.randn(768, 7168, device="cuda", dtype=torch.float16)
    got = skinny_gemv(x, w)
    assert torch.allclose(got.float(), (x @ w.t()).float(), atol=0.5, rtol=2e-2)

    x = torch.randn(1, 1536, device="cuda", dtype=torch.float16)
    w = torch.randn(7168, 1536, device="cuda", dtype=torch.float16)
    got = tgv_gemv(x, w)
    assert torch.allclose(got.float(), (x @ w.t()).float(), atol=0.5, rtol=2e-2)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _is_routed_arch(),
    reason="route is registered for sm100 and up",
)
def test_capture_of_a_warmed_shape_replays_correctly():
    """Warm eagerly, then capture: the routed kernel must be capturable and the
    replay must actually compute -- inputs change and outputs are poisoned
    between capture and replay, so a graph that recorded nothing (or a replay
    that writes nothing) fails the comparison. An unwarmed shape inside the
    same capture must fall back rather than JIT."""
    from tokenspeed_kernel.ops.gemm import routed_gemv as route

    m, n, k = 1, 3648, 7168  # skinny-routed
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    decode_gemv(x, w)  # warm

    dev = x.device.index
    assert dev is not None
    cold = ("skinny", dev, 1, 3584, 7168)
    with route._warmed_lock:
        route._warmed.discard(cold)
    xc = torch.randn(1, 7168, device="cuda", dtype=torch.bfloat16)
    wc = torch.randn(3584, 7168, device="cuda", dtype=torch.bfloat16)

    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        decode_gemv(x, w, out=out)
    torch.cuda.current_stream().wait_stream(s)
    from tokenspeed_kernel.thirdparty.cute_dsl.skinny_gemm import (
        shape_dynamic_skinny_gemm,
    )

    compiles: list[tuple] = []
    real_compile = shape_dynamic_skinny_gemm._compile
    shape_dynamic_skinny_gemm._compile = lambda *a, **kw: compiles.append(a)
    try:
        with torch.cuda.graph(g):
            decode_gemv(x, w, out=out)
            cold_out = route.skinny_gemv(xc, wc)  # unwarmed: falls back
    finally:
        shape_dynamic_skinny_gemm._compile = real_compile
    assert not compiles  # nothing may JIT inside the capture

    # Poisoned outputs: only a replay that really runs the kernels can pass.
    x.copy_(torch.randn_like(x))
    xc.copy_(torch.randn_like(xc))
    out.fill_(float("nan"))
    cold_out.fill_(float("nan"))
    g.replay()
    torch.cuda.synchronize()
    assert torch.allclose(out.float(), (x @ w.t()).float(), atol=0.5, rtol=2e-2)
    assert torch.allclose(cold_out.float(), (xc @ wc.t()).float(), atol=0.5, rtol=2e-2)
    assert cold not in route._warmed  # capture must not mark anything warmed


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _is_routed_arch(),
    reason="add3 route is measured for sm103 only",
)
@pytest.mark.parametrize("m", [1, 2])
def test_skinny_add3_matches_reference(m):
    from tokenspeed_kernel.ops.gemm.routed_gemv import skinny_gemv_add3

    torch.manual_seed(0)
    n, k = 7168, 3584
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) / 8
    a = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
    # Column slice of a wider tensor, as the serving call site passes it.
    c_wide = torch.randn(m, 2 * n, device="cuda", dtype=torch.bfloat16)
    c = c_wide[:, n:]
    got = skinny_gemv_add3(x, w, a, c).float()
    ref = a.float() + x.float() @ w.float().t() + c.float()
    assert torch.allclose(got, ref, atol=0.5, rtol=2e-2)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _is_add3_arch(),
    reason="ADD3_ROUTE holds sm103-tuned tile configs; unswept below that",
)
def test_kimi3_add3_auto_selects_the_skinny_epilogue():
    from tokenspeed_kernel.ops.gemm.kimi3 import kimi3_latent_projection_add3

    torch.manual_seed(1)
    m, n, k = 1, 7168, 3584
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) / 8
    a = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
    c = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
    auto = kimi3_latent_projection_add3(x, w, a, c).float()
    forced = kimi3_latent_projection_add3(x, w, a, c, solution="skinny_add3").float()
    composed = kimi3_latent_projection_add3(x, w, a, c, solution="composed").float()
    assert torch.allclose(auto, forced, atol=0.0, rtol=0.0)
    assert torch.allclose(auto, composed, atol=0.5, rtol=2e-2)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _is_routed_arch(),
    reason="add3 route is measured for sm103 only",
)
def test_skinny_add3_unwarmed_capture_falls_back(monkeypatch):
    import tokenspeed_kernel.ops.gemm.routed_gemv as route
    from tokenspeed_kernel.thirdparty.cute_dsl import skinny_gemm

    torch.manual_seed(2)
    m, n, k = 1, 7168, 3584
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) / 8
    a = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
    c = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
    dev = x.device.index or 0
    with route._warmed_lock:
        route._warmed.discard(("skinny_add3", dev, m, n, k))

    def _no_jit(*args, **kwargs):
        raise AssertionError("JIT compile attempted inside capture")

    monkeypatch.setattr(skinny_gemm.shape_dynamic_skinny_gemm, "_compile", _no_jit)
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        ref = a.float() + x.float() @ w.float().t() + c.float()
        with torch.cuda.graph(g):
            out = route.skinny_gemv_add3(x, w, a, c)
        g.replay()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    assert torch.allclose(out.float(), ref, atol=0.5, rtol=2e-2)


def test_unlisted_shapes_keep_the_generic_selection():
    _select.cache_clear()
    impl = _select(1, 999, 4096, True)
    assert "rowcta" in getattr(impl, "__name__", "")
    impl = _select(4, 3216, 7168, True)
    assert "torch" in getattr(impl, "__name__", "")
    impl = _select(3, 6288, 7168, True)
    assert "torch" in getattr(impl, "__name__", "")
    impl = _select(1, 2304, 1536, True)
    assert "rowcta" in getattr(impl, "__name__", "")


@pytest.mark.parametrize("m", [1, 2, 4])
def test_shared_projections_route_and_match_torch(m):
    """The K3 shared gate_up/down call sites take the measured route on
    covered shapes and stay bit-compatible with the torch composition."""
    from tokenspeed_kernel.ops.gemm.kimi3 import (
        kimi3_shared_down_projection,
        kimi3_shared_situ_projection,
    )

    torch.manual_seed(m)
    x = torch.randn(m, 7168, device="cuda", dtype=torch.bfloat16)
    gate_up_w = torch.randn(1536, 7168, device="cuda", dtype=torch.bfloat16)
    act = kimi3_shared_situ_projection(x, gate_up_w, beta=4.0, linear_beta=25.0)
    ref = kimi3_shared_situ_projection(
        x, gate_up_w, beta=4.0, linear_beta=25.0, solution="torch"
    )
    torch.testing.assert_close(act, ref, atol=5e-2, rtol=2e-2)

    y = torch.randn(m, 768, device="cuda", dtype=torch.bfloat16)
    down_w = torch.randn(7168, 768, device="cuda", dtype=torch.bfloat16)
    got = kimi3_shared_down_projection(y, down_w)
    want = kimi3_shared_down_projection(y, down_w, solution="torch")
    torch.testing.assert_close(got, want, atol=5e-2, rtol=2e-2)


def test_forced_torch_solution_is_not_routed():
    """solution="torch" must stay the vendor-BLAS baseline even for shapes the
    measured route covers, or A/B comparisons silently measure the route."""
    from unittest.mock import patch

    from tokenspeed_kernel.ops.gemm import kimi3

    x = torch.randn(1, 7168, device="cuda", dtype=torch.bfloat16)
    latent_w = torch.randn(3584, 7168, device="cuda", dtype=torch.bfloat16)
    down_w = torch.randn(7168, 768, device="cuda", dtype=torch.bfloat16)
    y = torch.randn(1, 768, device="cuda", dtype=torch.bfloat16)

    with patch(
        "tokenspeed_kernel.ops.gemm.triton_gemv.decode_gemv",
        side_effect=AssertionError("forced torch path must not route"),
    ):
        kimi3.kimi3_latent_projection(x, latent_w, solution="torch")
        kimi3.kimi3_shared_down_projection(y, down_w, solution="torch")
    # Drain the queued vendor-BLAS work here: under emulation it otherwise keeps
    # executing into the next test and charges its runtime against that test.
    torch.cuda.synchronize()


def test_route_predicate_admits_the_registered_arch_floor():
    """MEASURED_ROUTE registers at sm100 and documents a GB200 re-sweep, so the
    predicate must not gate on ADD3_ROUTE's stricter sm103 floor."""
    from unittest.mock import patch

    from tokenspeed_kernel.ops.gemm import routed_gemv

    x = torch.randn(1, 7168, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(3584, 7168, device="cuda", dtype=torch.bfloat16)
    nvidia = type("P", (), {"vendor": "nvidia"})()
    for capability, expected in (((10, 0), True), ((9, 0), False)):
        routed_gemv._is_routed_arch.cache_clear()
        with (
            patch("torch.cuda.get_device_capability", return_value=capability),
            patch("tokenspeed_kernel.platform.current_platform", return_value=nvidia),
        ):
            assert routed_gemv.decode_gemv_routed(x, w) is expected
    routed_gemv._is_routed_arch.cache_clear()
    torch.cuda.synchronize()
