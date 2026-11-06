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

"""Decode-sized BF16 GEMMs on small weights run pinned FlashInfer TGV tiles."""

from __future__ import annotations

import logging
from unittest import mock

import pytest
import torch
from tokenspeed_kernel import compile_monitor
from tokenspeed_kernel.ops.gemm import flashinfer as fi
from tokenspeed_kernel.ops.gemm import mm
from tokenspeed_kernel.ops.layernorm import rmsnorm
from tokenspeed_kernel.ops.tuning import kernel_warmup
from tokenspeed_kernel.platform import pdl_enabled
from tokenspeed_kernel.profiling import ShapeCapture
from utils import assert_no_cute_compile

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or fi.platform.arch_version not in (fi.ArchVersion(10, 0), fi.ArchVersion(10, 3))
    or not fi.has_flashinfer_cute_dsl_bf16(),
    reason="pinned TGV tiles need FlashInfer's BF16 GEMMs on SM100/SM103",
)

TILE16, TILE32, TILE64 = fi.TGV_DECODE_TILES
TGV = "flashinfer_tgv_decode_gemm"


@pytest.fixture(autouse=True)
def startup_with_pdl(monkeypatch):
    """Run as startup does, with PDL on, whatever an earlier test set."""
    monkeypatch.setattr(compile_monitor, "_serving", False)
    previous_pdl = pdl_enabled()
    pdl_enabled(True)
    capture = ShapeCapture.get()
    enabled = capture.enabled
    yield
    capture.enabled = enabled
    capture.clear()
    pdl_enabled(previous_pdl)


@pytest.fixture
def tgv_compiles(monkeypatch):
    """FlashInfer's TGV module and its compile cache, emptied for this test.

    The cache is process-wide and keyed by tile, PDL setting and operand
    layout, not by shape, so tiles an earlier test compiled would hide this
    test's compiles. Each entry is one compiled TGV kernel.
    """
    module = pytest.importorskip("flashinfer.gemm.kernels.tgv_gemm_cute_ext")
    compiled = {}
    monkeypatch.setattr(module, "_TGV_CUTE_EXT_COMPILE_CACHE", compiled)
    return module, compiled


def inputs(m, n, k, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, k, device="cuda", generator=generator).bfloat16()
    w = (torch.randn(n, k, device="cuda", generator=generator) / k**0.5).bfloat16()
    return x, w


def check_result(got, x, w):
    """TGV rounds its FP32 sum once to BF16, a relative error of at most
    2**-8; the bound allows twice that, and FP32 error near zero."""
    reference = x.double() @ w.double().T
    torch.testing.assert_close(got.double(), reference, rtol=2**-7, atol=3e-5)


def routed_kernel(x, w, **kwargs):
    """Run mm() and return the kernel it recorded with its result."""
    capture = ShapeCapture.get()
    capture.clear()
    capture.enabled = True
    try:
        result = mm(x, w, **kwargs)
    finally:
        capture.enabled = False
    return capture._records[-1].kernel_name, result


def guarded_out(m, n):
    """An ``[m, n]`` output followed by sentinels that catch writes past it."""
    storage = torch.full((m * n + 64,), 42.0, device="cuda", dtype=torch.bfloat16)
    return storage, storage[: m * n].view(m, n)


def test_installed_flashinfer_lists_the_pinned_tiles():
    # A deliberate upgrade canary on FlashInfer's tactic table: a FlashInfer
    # without the tiles only logs a warning and keeps cuBLASLt, so without
    # this assertion a FlashInfer bump could turn the route off unnoticed.
    # _flashinfer_tgv is this package's binding (module, launcher arguments,
    # lookup by shape), which a FlashInfer change can also break.
    tgv = fi._flashinfer_tgv()
    assert tgv is not None
    assert set(tgv.tactics) == set(fi.TGV_DECODE_TILES)
    x, w = inputs(48, 512, 2560, 1)
    unpinned = next(i for i in range(64) if i not in tgv.tactics.values())
    with pytest.raises(ValueError, match="Unsupported input"):
        fi.flashinfer_tgv_decode_gemm(x, w, None, unpinned)
    with pytest.raises(ValueError, match="Unsupported input"):
        fi.flashinfer_tgv_decode_gemm(x[:, ::2], w[:, ::2], None, tgv.tactics[TILE16])


def test_pinned_tiles_start_where_the_joint_path_ends():
    # select_tgv_decode_tile pins its lower row bound instead of reading
    # BF16_GEMM_MAX_M, which follows the host GPU. On SM100 and SM103 the two
    # agree, so mm() hands every row count above the joint FlashInfer path to
    # the tile policy.
    assert fi.TGV_DECODE_MIN_M == fi.BF16_GEMM_MAX_M


@pytest.mark.parametrize(
    "m,n,k,routed",
    [
        (32, 512, 2560, False),  # rows of the joint FlashInfer path
        (33, 512, 2560, True),
        (512, 512, 2560, True),
        (513, 512, 2560, False),
        (64, 4096, 2048, True),  # exactly 16 MiB of weight
        (64, 4096, 2056, False),
        (48, 2560, 2080, True),  # a partial last K tile
        # Weight rows ending inside a 64-row tile; for 160 rows, the second
        # CTA of the last 2-CTA cluster lies wholly past the weight.
        (48, 160, 2560, True),
        (400, 1000, 2560, True),
        (192, 2560, 2560, True),
        (288, 2560, 2560, False),  # no tile's grid fits in one wave
        (
            160,
            2560,
            6144,
            False,
        ),  # a 30 MiB attention output projection (also outside the K window)
        # Small weights whose grid fits in one wave, with K outside
        # TGV_DECODE_MIN_K..TGV_DECODE_MAX_K: they keep the existing GEMM.
        (64, 2560, 640, False),
        (64, 160, 7168, False),
        (64, 3072, 512, False),
    ],
)
def test_mm_routes_decode_rows_by_the_tile_policy(m, n, k, routed):
    if not 128 <= fi.platform.sm_count < 200:
        pytest.skip("these shapes route as listed on GPUs with 128 to 199 SMs")
    x, w = inputs(m, n, k, seed=m + n + k)
    out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    tactic = fi.flashinfer_tgv_decode_tactic(x, w, out)
    assert (tactic is not None) is routed
    kernel, result = routed_kernel(x, w, out=out)
    assert result is out
    if routed:
        assert kernel == TGV
        assert torch.equal(out, fi.flashinfer_tgv_decode_gemm(x, w, None, tactic))
        check_result(out, x, w)
    else:
        assert kernel != TGV
        if m > fi.BF16_GEMM_MAX_M:
            assert torch.equal(out, x @ w.T)


@pytest.mark.parametrize("pdl", [False, True])
@pytest.mark.parametrize(
    "n,k",
    [(512, 2560), (960, 2560), (2560, 2080), (2560, 160), (160, 2560), (1000, 2560)],
)
def test_pinned_tiles_match_fp64(pdl, n, k):
    """Every tile, with full and partial last activation, weight and K tiles:
    160 and 1000 weight rows end inside a 64-row tile."""
    tgv = fi._flashinfer_tgv()
    assert tgv is not None
    x, w = inputs(fi.TGV_DECODE_MAX_M, n, k, seed=n + k)
    pdl_enabled(pdl)  # startup_with_pdl restores it
    for tile, tactic in tgv.tactics.items():
        for m in (33, 3 * tile.cta_n, 3 * tile.cta_n + 1, 255, x.shape[0]):
            storage, out = guarded_out(m, n)
            assert fi.flashinfer_tgv_decode_gemm(x[:m], w, out, tactic) is out
            check_result(out, x[:m], w)
            assert torch.all(storage[m * n :] == 42)


@pytest.mark.parametrize("m,tile", [(33, TILE16), (96, TILE32), (192, TILE64)])
def test_captured_graph_replays_the_tile_bit_for_bit(m, tile):
    n, k = 2560, 2560
    x, w = inputs(m, n, k, seed=m)
    chosen = fi.select_tgv_decode_tile(
        m=m,
        n=n,
        k=k,
        weight_bytes=2 * n * k,
        arch=fi.platform.arch_version,
        sm_count=fi.platform.sm_count,
        pdl=pdl_enabled(),
    )
    if chosen != tile:
        pytest.skip(f"{fi.platform.sm_count} SMs give {m} rows another tile")
    storage, out = guarded_out(m, n)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        mm(x, w, out=out)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        mm(x, w, out=out)
    for scale in (0.125, -8.0, 0.75):
        x.mul_(scale)
        kernel, expected = routed_kernel(x, w)
        assert kernel == TGV
        for _ in range(3):
            out.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(out, expected)
        check_result(out, x, w)
        assert torch.all(storage[m * n :] == 42)


@pytest.mark.parametrize("m", [33, 96, 192])
def test_pdl_chain_replays_bit_for_bit(m):
    """A norm feeds a GEMM whose output feeds a second GEMM, all launched with
    PDL, as in a decode layer. Each TGV launch may start while its producer
    still runs and load its weight then; only its activation loads wait for
    the producer. Buffers hold NaN before each replay, so an activation read
    that ran ahead of its producer shows in the result."""
    hidden, inner, out_features = 2560, 2560, 512
    x_in, w1 = inputs(m, inner, hidden, seed=m)
    _, w2 = inputs(m, out_features, inner, seed=m + 1)
    generator = torch.Generator(device="cuda").manual_seed(m + 2)
    gamma = (1 + 0.1 * torch.randn(hidden, device="cuda", generator=generator)).to(
        torch.bfloat16
    )
    x = torch.empty(m, hidden, device="cuda", dtype=torch.bfloat16)
    y = torch.empty(m, inner, device="cuda", dtype=torch.bfloat16)
    storage, z = guarded_out(m, out_features)
    if any(
        fi.flashinfer_tgv_decode_tactic(a, w, out) is None
        for a, w, out in ((x, w1, y), (y, w2, z))
    ):
        pytest.skip(f"{fi.platform.sm_count} SMs leave a GEMM of the chain on cuBLASLt")

    def chain():
        rmsnorm(x_in, gamma, 1e-6, residual=None, out=x)
        mm(x, w1, out=y)
        mm(y, w2, out=z)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        chain()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        chain()
    for seed in (1, 2, 3):
        # RMSNorm undoes a rescale, so each round draws new activations.
        x_in.copy_(inputs(m, 1, hidden, seed=1000 * m + seed)[0])
        expected_x = rmsnorm(x_in, gamma, 1e-6, residual=None, out=None)
        kernel, expected_y = routed_kernel(expected_x, w1)
        assert kernel == TGV
        kernel, expected_z = routed_kernel(expected_y, w2)
        assert kernel == TGV
        for _ in range(3):
            for buffer in (x, y, z):
                buffer.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(x, expected_x)
            assert torch.equal(y, expected_y)
            assert torch.equal(z, expected_z)
        check_result(y, x, w1)
        check_result(z, y, w2)
        assert torch.all(storage[m * out_features :] == 42)


def test_serving_and_bias_keep_cublaslt(monkeypatch):
    x, w = inputs(fi.TGV_DECODE_MAX_M, 512, 2560, 7)
    bias = torch.randn(512, device="cuda", dtype=torch.bfloat16)
    assert fi.flashinfer_tgv_decode_tactic(x[:48], w, None) is not None
    kernel, result = routed_kernel(x[:48], w, bias=bias)
    assert kernel != TGV
    assert torch.equal(result, torch.nn.functional.linear(x[:48], w, bias))
    monkeypatch.setattr(compile_monitor, "_serving", True)
    with assert_no_cute_compile():
        for m in (33, 48, 200, fi.TGV_DECODE_MAX_M):
            kernel, result = routed_kernel(x[:m], w)
            assert kernel != TGV
            assert torch.equal(result, x[:m] @ w.T)


def test_without_pdl_mm_keeps_cublaslt_and_warms_no_tile(tgv_compiles):
    _, compiled = tgv_compiles
    x, w = inputs(fi.TGV_DECODE_MAX_M, 512, 2560, 19)
    assert fi.flashinfer_tgv_decode_tactic(x[:48], w, None) is not None
    pdl_enabled(False)  # startup_with_pdl restores it
    with kernel_warmup(enabled=True):
        mm(x[:64], w)
    for m in (33, 48, 200, fi.TGV_DECODE_MAX_M):
        assert fi.flashinfer_tgv_decode_tactic(x[:m], w, None) is None
        kernel, result = routed_kernel(x[:m], w)
        assert kernel != TGV
        assert torch.equal(result, x[:m] @ w.T)
    assert not compiled


def test_warmup_compiles_the_tiles_for_every_row_and_weight(tgv_compiles):
    """TGV compiles per tile, PDL setting and operand layout, so warmup on one
    weight compiles the three tiles, and no row count of that weight or of
    another routed weight compiles again."""
    module, compiled = tgv_compiles
    pairs = [
        inputs(fi.TGV_DECODE_MAX_M, 512, 2560, 11),
        inputs(fi.TGV_DECODE_MAX_M, 2560, 2080, 13),
    ]
    rows = range(fi.BF16_GEMM_MAX_M + 1, fi.TGV_DECODE_MAX_M + 1)
    warmed_x, warmed_w = pairs[0]
    with kernel_warmup(enabled=True):
        mm(warmed_x[:64], warmed_w)
    # 64 rows of this weight need one tile; warmup compiled all three.
    assert len(compiled) == len(fi.TGV_DECODE_TILES)
    warmed = set(compiled)
    capture = ShapeCapture.get()
    capture.clear()
    capture.enabled = True
    try:
        # TGV compiles through cute.experimental; fail on any such compile.
        with mock.patch.object(
            module.cute_ext,
            "compile",
            side_effect=AssertionError("a TGV tile compiled after warmup"),
        ), assert_no_cute_compile():
            for x, w in pairs:
                for m in rows:
                    mm(x[:m], w)
    finally:
        capture.enabled = False
    assert set(compiled) == warmed
    routed = sum(
        fi.flashinfer_tgv_decode_tactic(x[:m], w, None) is not None
        for x, w in pairs
        for m in rows
    )
    assert routed >= len(rows)
    assert [record.kernel_name for record in capture._records].count(TGV) == routed


def test_flashinfer_without_a_pinned_tile_keeps_cublaslt(monkeypatch, caplog):
    module = pytest.importorskip("flashinfer.gemm.kernels.tgv_gemm_cute_ext")
    x, w = inputs(48, 512, 2560, 17)
    table = [entry for entry in module._TGV_CUTE_EXT_TACTIC_CONFIGS if entry != TILE32]
    monkeypatch.setattr(module, "_TGV_CUTE_EXT_TACTIC_CONFIGS", table)
    fi._flashinfer_tgv.cache_clear()
    try:
        with caplog.at_level(logging.WARNING, logger=fi.logger.name):
            assert fi.flashinfer_tgv_decode_tactic(x, w, None) is None
            kernel, result = routed_kernel(x, w)
        assert kernel != TGV
        assert torch.equal(result, x @ w.T)
        assert "keep cuBLASLt" in caplog.text
    finally:
        monkeypatch.undo()
        fi._flashinfer_tgv.cache_clear()
    assert fi._flashinfer_tgv() is not None
