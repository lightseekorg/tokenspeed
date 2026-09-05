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
    from tokenspeed_kernel.ops.gemm.routed_gemv import (
        ll_bf16_gemv,
        skinny_gemv,
        tgv_gemv,
    )

    x = torch.randn(1, 7168, device="cuda", dtype=torch.float16)
    w = torch.randn(768, 7168, device="cuda", dtype=torch.float16)
    got = skinny_gemv(x, w)
    assert torch.allclose(got.float(), (x @ w.t()).float(), atol=0.5, rtol=2e-2)

    x = torch.randn(1, 1536, device="cuda", dtype=torch.float16)
    w = torch.randn(7168, 1536, device="cuda", dtype=torch.float16)
    got = tgv_gemv(x, w)
    assert torch.allclose(got.float(), (x @ w.t()).float(), atol=0.5, rtol=2e-2)

    x = torch.randn(1, 1536, device="cuda", dtype=torch.float16)
    w = torch.randn(2560, 1536, device="cuda", dtype=torch.float16)
    got = ll_bf16_gemv(x, w)
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
    # A width no call site produces.
    impl = _select(3, 6289, 7168, True)
    assert "torch" in getattr(impl, "__name__", "")
    impl = _select(1, 2304, 1536, True)
    assert "rowcta" in getattr(impl, "__name__", "")


def test_qwen38_route_keeps_unstable_shapes_on_fallback():
    assert MEASURED_ROUTE[(2, 12800, 2560)] == "skinny"
    assert {m for m, n, k in MEASURED_ROUTE if (n, k) == (12800, 2560)} == {2, 4}
    for shape in (
        (2, 512, 2560),
        (4, 640, 2560),
        (17, 2560, 320),
        (21, 2560, 320),
        (23, 2560, 320),
        (1, 6656, 2560),
    ):
        assert shape not in MEASURED_ROUTE


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


def test_measured_route_source_has_no_duplicate_keys():
    """A dict literal silently resolves duplicate keys last-wins, so a re-added
    entry would shadow an existing one with no error anywhere. Count key
    occurrences in the SOURCE text, where duplicates are still visible."""
    import collections
    import inspect
    import re

    from tokenspeed_kernel.ops.gemm import routed_gemv

    src = inspect.getsource(routed_gemv)
    keys = re.findall(r'\((\d+), (\d+), (\d+)\): "\w+"', src)
    dupes = [k for k, n in collections.Counter(keys).items() if n > 1]
    assert not dupes, f"duplicate MEASURED_ROUTE keys in source: {dupes}"
    # Parsed size must match source count, else a duplicate collapsed.
    assert len(routed_gemv.MEASURED_ROUTE) == len(keys)
    # Tuple-valued tables need their own pattern, scanned per table: the two
    # config tables are independent key spaces, so a shared key is legal.
    for name, table in (
        ("SKINNY_CONFIG_ROUTE", routed_gemv.SKINNY_CONFIG_ROUTE),
        ("ADD3_ROUTE", routed_gemv.ADD3_ROUTE),
    ):
        start = src.index(f"{name}: MappingProxyType")
        span = src[start : src.index(")", src.index("}", start))]
        cfg_keys = re.findall(r"\((\d+), (\d+), (\d+)\): \(", span)
        cfg_dupes = [k for k, n in collections.Counter(cfg_keys).items() if n > 1]
        assert not cfg_dupes, f"duplicate {name} keys in source: {cfg_dupes}"
        assert len(table) == len(cfg_keys), name
    # Exact-M keying: entries may only exist in the gap-free swept range. The
    # bound is the widest M any routed backend serves, which the split-K
    # tactics carry to 64; nothing above that has been swept.
    assert all(m <= 64 for m, _, _ in routed_gemv.MEASURED_ROUTE)
    for name, table in (
        ("SKINNY_CONFIG_ROUTE", routed_gemv.SKINNY_CONFIG_ROUTE),
        ("SPLITK_TACTIC_ROUTE", routed_gemv.SPLITK_TACTIC_ROUTE),
    ):
        start = src.index(f"{name}: MappingProxyType")
        span = src[start : src.index(")", src.index("}", start))]
        cfg_keys = re.findall(r"\((\d+), (\d+), (\d+)\): \(", span)
        assert len(table) == len(cfg_keys), name


def test_skinny_config_route_entries_are_valid():
    """Every tuned skinny config must target a shape the route actually sends
    to skinny, and must satisfy the kernel's geometry contract; a typo here
    would otherwise fall back (wrong M) or crash at compile time."""
    from tokenspeed_kernel.ops.gemm.routed_gemv import SKINNY_CONFIG_ROUTE
    from tokenspeed_kernel.thirdparty.cute_dsl.skinny_gemm import (
        SkinnyGemmConfig,
        shape_dynamic_skinny_gemm,
    )

    for (m, n, k), tuned in SKINNY_CONFIG_ROUTE.items():
        assert MEASURED_ROUTE.get((m, n, k)) == "skinny", (m, n, k)
        # supports() does not know the kernel's warp-multiple block rule.
        assert tuned[0] % 32 == 0, (m, n, k)
        config = SkinnyGemmConfig(m, *tuned)
        assert shape_dynamic_skinny_gemm.supports(config, m, n, k), (m, n, k)


# What each FlashInfer BF16 backend does here, per PDL setting: None when it
# runs, else the substring its refusal must carry. The route's candidate set
# rests on this. A sweep that catches every exception and scores it as "no
# result" cannot tell "this backend lost" from "this backend was never asked
# properly" -- which is how three of these were missed: they reject pdl=True
# outright rather than ignoring it.
_BF16_BACKEND_SUPPORT = {
    "cudnn": {True: None, False: None},
    "tgv": {True: None, False: None},
    "tinygemm": {True: None, False: None},
    "cute-dsl": {True: None, False: None},
    "cutlass": {True: "does not support PDL", False: None},
    "cublaslt": {True: "does not support PDL", False: None},
    "cutile": {True: "ignores `pdl`", False: "No valid config found"},
}


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _is_routed_arch(),
    reason="route is registered for sm100 and up",
)
@pytest.mark.parametrize("backend", sorted(_BF16_BACKEND_SUPPORT))
def test_bf16_backend_support_is_what_the_route_was_tuned_against(backend):
    """Pin the candidate set the measured tables were chosen from.

    A wheel that makes a refused backend work, or breaks one that worked, moves
    the set the tuner should have searched -- and neither shows up as a wrong
    answer anywhere, only as a table that is quietly no longer the best pick.
    """
    from flashinfer import mm_bf16

    m, n, k = 8, 1792, 7168  # a real draft projection
    torch.manual_seed(0)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    ref = x.float() @ w.float().t()
    for pdl, refusal in _BF16_BACKEND_SUPPORT[backend].items():
        try:
            got = mm_bf16(x, w.t(), pdl=pdl, backend=backend)
        except Exception as exc:  # noqa: BLE001  (any refusal is the signal)
            assert refusal is not None, (
                f"{backend} pdl={pdl} was usable when the tables were tuned and "
                f"now raises {exc!r}; re-run test/gemm_tuning/tune_route.py"
            )
            assert refusal in str(
                exc
            ), f"{backend} pdl={pdl} refuses for a new reason: {exc!r}"
            continue
        assert refusal is None, (
            f"{backend} pdl={pdl} now runs but was excluded from the sweep; "
            f"re-run test/gemm_tuning/tune_route.py -- it may win a shape"
        )
        rel = (got.float() - ref).abs().max().item() / ref.abs().max().item()
        assert rel < 0.02, f"{backend} pdl={pdl} rel err {rel:.4f}"


def test_splitk_tactic_route_entries_are_valid():
    """Every measured tactic must target a shape the route sends to splitk and
    be one the vendor kernel accepts; a typo would silently fall back."""
    from tokenspeed_kernel.ops.gemm.routed_gemv import SPLITK_TACTIC_ROUTE
    from tokenspeed_kernel.thirdparty.cute_dsl import flashinfer_splitk

    if not flashinfer_splitk.is_available():
        pytest.skip("flashinfer split-K BF16 GEMM is not the measured build")
    for (m, n, k), tactic in SPLITK_TACTIC_ROUTE.items():
        assert MEASURED_ROUTE.get((m, n, k)) == "splitk", (m, n, k)
        assert flashinfer_splitk.supports(m, n, k, tactic), (m, n, k)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _is_routed_arch(),
    reason="route is registered for sm100 and up",
)
@pytest.mark.parametrize("m", [33, 47, 64])
def test_splitk_serves_m_past_the_vendor_cutover(m):
    """What the table above M == 32 rests on.

    Public M rides the kernel's MMA-N axis, so the vendor's cutover is a
    heuristic rather than a contract and M is tiled, not truncated. A kernel
    that computed only the first tile would leave whole rows zero.
    """
    from tokenspeed_kernel.thirdparty.cute_dsl import flashinfer_splitk

    if not flashinfer_splitk.is_available():
        pytest.skip("flashinfer split-K BF16 GEMM is not the measured build")
    torch.manual_seed(0)
    n, k, tactic = 1792, 7168, (64, 32, 2, 9)
    assert flashinfer_splitk.supports(m, n, k, tactic)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    out = torch.zeros(m, n, device="cuda", dtype=torch.bfloat16)
    flashinfer_splitk.splitk_mm(x, w, tactic, out)
    ref = x.float() @ w.float().t()
    assert int((out.abs().sum(dim=1) == 0).sum()) == 0
    assert (out.float() - ref).abs().max() / ref.abs().max() < 0.02


def _splitk_cases():
    from tokenspeed_kernel.ops.gemm.routed_gemv import SPLITK_TACTIC_ROUTE

    return sorted(SPLITK_TACTIC_ROUTE)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _is_routed_arch(),
    reason="route is registered for sm100 and up",
)
@pytest.mark.parametrize("shape", _splitk_cases())
def test_splitk_captures_and_replays(shape):
    """The drafter runs entirely inside a CUDA graph, so a routed backend that
    cannot be captured would silently fall back there and nowhere else -- and
    the fallback is correct, so nothing else in the suite would notice."""
    m, n, k = shape
    torch.manual_seed(0)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    decode_gemv(x, w, out=out)  # warm
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        decode_gemv(x, w, out=out)
    torch.cuda.current_stream().wait_stream(s)
    from tokenspeed_kernel.ops.gemm import routed_gemv as route

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        decode_gemv(x, w, out=out)
    # Warming happens eagerly only, so the capture must have found it already
    # warm rather than marking it warm from inside.
    assert ("splitk", x.device.index or 0, m, n, k) in route._warmed
    # Poisoned output and fresh input: only a replay that really runs passes.
    x.copy_(torch.randn_like(x))
    out.fill_(float("nan"))
    g.replay()
    torch.cuda.synchronize()
    assert torch.allclose(out.float(), (x @ w.t()).float(), atol=0.5, rtol=2e-2)


def test_skinny_config_prefers_the_measured_table_over_the_heuristic():
    """Measured entries serve; unmeasured shapes fall through to the heuristic."""
    from tokenspeed_kernel.ops.gemm import routed_gemv
    from tokenspeed_kernel.thirdparty.cute_dsl.skinny_gemm import (
        shape_dynamic_skinny_gemm,
    )

    routed_gemv._skinny_config.cache_clear()
    m, n, k = 2, 768, 1536
    config = routed_gemv._skinny_config(m, n, k)
    assert (
        config.block_size,
        config.outputs_per_block,
        config.k_unroll,
        config.vector_width,
    ) == routed_gemv.SKINNY_CONFIG_ROUTE[(m, n, k)]

    unmeasured = (7, 768, 1536)
    assert unmeasured not in routed_gemv.SKINNY_CONFIG_ROUTE
    assert routed_gemv._skinny_config(*unmeasured) == (
        shape_dynamic_skinny_gemm.default_config(*unmeasured)
    )
    routed_gemv._skinny_config.cache_clear()


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _is_routed_arch(),
    reason="route is registered for sm100 and up",
)
@pytest.mark.parametrize("misalign,offset", [("x", 4), ("weight", 4), ("x", 8)])
def test_under_aligned_operands_fall_back_to_torch(monkeypatch, misalign, offset):
    """vw-16 needs 32-byte pointers and supports() cannot see alignment, so the
    guard falls back; offset 8 is the 16B case a vw-8 config would accept."""
    from tokenspeed_kernel.ops.gemm import routed_gemv
    from tokenspeed_kernel.thirdparty.cute_dsl import skinny_gemm

    m, n, k = 2, 768, 1536  # tuned entry (96, 2, 1, 16)
    xbuf = torch.randn(m * k + offset, device="cuda", dtype=torch.bfloat16)
    wbuf = torch.randn(n * k + offset, device="cuda", dtype=torch.bfloat16)
    x = (xbuf[offset:] if misalign == "x" else xbuf[:-offset]).view(m, k)
    w = (wbuf[offset:] if misalign == "weight" else wbuf[:-offset]).view(n, k)
    assert (x if misalign == "x" else w).data_ptr() % 32
    monkeypatch.setattr(
        skinny_gemm.ShapeDynamicSkinnyGemm,
        "__call__",
        lambda *a, **kw: pytest.fail("under-aligned input must not launch vw-16"),
    )
    got = routed_gemv.skinny_gemv(x, w)
    assert torch.allclose(got.float(), (x @ w.t()).float(), atol=0.5, rtol=2e-2)
