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

"""Correctness of the research 3-stage, Stream-K and persistent Gluon GEMMs (gfx950)."""

from __future__ import annotations

import re

import pytest
import torch
from utils import assert_no_triton_compile, is_cdna4

if not is_cdna4():
    pytest.skip(
        "AMD CDNA4 is required for dense16 Gluon GEMM tests",
        allow_module_level=True,
    )

from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.mm_3stage import (  # noqa: E402
    config_fits as gl3_config_fits,
)
from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.mm_3stage import (  # noqa: E402
    gluon_mm_a16w16_3stage_gfx950,
    launch_gluon_mm_a16w16_3stage_gfx950,
)
from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.mm_persistent import (  # noqa: E402
    config_fits as pers_config_fits,
)
from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.mm_persistent import (  # noqa: E402
    gluon_mm_a16w16_persistent_gfx950,
    launch_gluon_mm_a16w16_persistent_gfx950,
    tail_pieces,
)
from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.mm_streamk import (  # noqa: E402
    config_fits as sk_config_fits,
)
from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.mm_streamk import (  # noqa: E402
    gluon_mm_a16w16_streamk_gfx950,
    launch_gluon_mm_a16w16_streamk_gfx950,
)

# Kernels and reference accumulate in FP32 and round once to BF16, so outputs
# differ by at most one BF16 ULP (2**-7 relative); split-K changes only the
# FP32 summation order. atol covers FP32 accumulation-order noise near zero.
_ATOL = 1e-5
_RTOL = 2**-7


def _inputs(m: int, n: int, k: int):
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    w = torch.randn((n, k), device="cuda", dtype=torch.bfloat16) * 0.05
    ref = (a.float() @ w.float().t()).to(torch.bfloat16)
    return a, w, ref


# cfg = (BM, BN, BK, WM, WN, NS, SPLIT_K, XCD); ragged M and N exercise masking.
@pytest.mark.parametrize(
    "mnk,cfg,warp_pipe,variant",
    [
        pytest.param(
            (64, 1024, 1024), (64, 128, 64, 2, 2, 3, 1, 8), False, "base", id="base"
        ),
        pytest.param(
            (100, 1120, 1024),
            (64, 128, 64, 2, 2, 3, 1, 8),
            False,
            "base",
            id="ragged-mn",
        ),
        pytest.param(
            (128, 512, 2048), (128, 64, 128, 2, 2, 3, 2, 8), False, "base", id="splitk2"
        ),
        pytest.param(
            (64, 640, 1536), (64, 64, 128, 2, 2, 3, 3, 8), False, "base", id="splitk3"
        ),
        pytest.param(
            (256, 1024, 1024),
            (256, 128, 64, 4, 2, 3, 1, 8),
            True,
            "base",
            id="warp-pipe",
        ),
        pytest.param(
            (128, 1024, 1024),
            (128, 128, 64, 2, 2, 4, 1, 1),
            False,
            "sched",
            id="sched-ns4",
        ),
        pytest.param(
            (128, 1024, 1024),
            (128, 128, 64, 2, 2, 3, 2, 8),
            False,
            "sched+pin",
            id="sched-pin-splitk",
        ),
        # Non-power-of-two tiles: BM / BN = lo + hi, run as separate MFMA chains.
        pytest.param(
            (100, 1100, 2048),
            (64, 144, 128, 4, 1, 2, 1, 8),
            False,
            "base",
            id="split-n-ragged",
        ),
        pytest.param(
            (300, 1000, 1024),
            (192, 128, 64, 4, 2, 3, 1, 8),
            False,
            "base",
            id="split-m-ragged",
        ),
        pytest.param(
            (200, 1000, 1024),
            (192, 160, 64, 4, 1, 3, 2, 8),
            False,
            "base",
            id="split-mn-splitk2",
        ),
        pytest.param(
            (200, 1000, 1024),
            (192, 160, 64, 4, 1, 3, 1, 8),
            True,
            "base",
            id="split-mn-warp-pipe",
        ),
        pytest.param(
            (64, 600, 1536),
            (64, 80, 128, 4, 1, 3, 3, 8),
            False,
            "sched",
            id="split-n-splitk3",
        ),
        pytest.param(
            (130, 1000, 2048),
            (96, 96, 128, 2, 2, 3, 4, 8),
            False,
            "sched+pin",
            id="split-mn-pin-splitk4",
        ),
    ],
)
def test_3stage_matches_reference(mnk, cfg, warp_pipe, variant) -> None:
    torch.manual_seed(0)
    m, n, k = mnk
    assert gl3_config_fits(cfg, m, n, k)
    a, w, ref = _inputs(m, n, k)
    c = torch.full((m, n), float("nan"), device="cuda", dtype=torch.bfloat16)
    kw = dict(
        schedule_hint="mfma-schedule" if "sched" in variant else "",
        pin_acc="pin" in variant,
    )
    # Run twice: split-K programs must leave their per-tile counters reset.
    for _ in range(2):
        c.fill_(float("nan"))
        launch_gluon_mm_a16w16_3stage_gfx950(a, w, c, cfg, warp_pipe, **kw)
        torch.testing.assert_close(c, ref, atol=_ATOL, rtol=_RTOL)


def test_3stage_column_slice_output() -> None:
    """Writing into a column slice of a wider C leaves the other columns untouched."""
    torch.manual_seed(0)
    m, n, k = 64, 512, 1024
    a, w, ref = _inputs(m, n, k)
    c = torch.zeros((m, n + 256), device="cuda", dtype=torch.bfloat16)
    launch_gluon_mm_a16w16_3stage_gfx950(
        a, w, c[:, 128 : 128 + n], (64, 128, 64, 2, 2, 3, 1, 8), False
    )
    torch.testing.assert_close(c[:, 128 : 128 + n], ref, atol=_ATOL, rtol=_RTOL)
    assert not c[:, :128].any() and not c[:, 128 + n :].any()


# cfg = (BM, BN, BK, WM, WN, NS, GRID, XCD); GRID below and above the tile count.
@pytest.mark.parametrize(
    "mnk,cfg",
    [
        pytest.param((64, 1024, 2048), (64, 128, 64, 2, 2, 3, 24, 8), id="grid>tiles"),
        pytest.param(
            (128, 1056, 1024), (128, 64, 128, 2, 2, 3, 8, 8), id="grid<tiles-ragged-n"
        ),
        pytest.param((96, 512, 4096), (64, 64, 128, 2, 2, 3, 40, 8), id="ragged-m"),
    ],
)
def test_streamk_matches_reference(mnk, cfg) -> None:
    torch.manual_seed(0)
    m, n, k = mnk
    assert sk_config_fits(cfg, m, n, k)
    a, w, ref = _inputs(m, n, k)
    c = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    for _ in range(2):
        c.fill_(float("nan"))
        launch_gluon_mm_a16w16_streamk_gfx950(a, w, c, cfg)
        torch.testing.assert_close(c, ref, atol=_ATOL, rtol=_RTOL)


# Configs whose kernel used scratch (52-320 bytes per lane) while the Stream-K
# reduction ran inside the main loop. N is unique per case and ragged (masked
# B loads) except for 256x128-w4x2-pin, which keeps 12 bytes of segment-end
# spills with a ragged N.
@pytest.mark.parametrize(
    "n,cfg,pin_acc",
    [
        pytest.param(2144, (256, 128, 64, 4, 1, 3, 24, 8), True, id="256x128-w4x1-pin"),
        pytest.param(2272, (128, 256, 64, 2, 2, 3, 24, 8), True, id="128x256-w2x2-pin"),
        pytest.param(2304, (256, 128, 64, 4, 2, 3, 24, 8), True, id="256x128-w4x2-pin"),
        pytest.param(2528, (256, 128, 64, 4, 2, 3, 24, 8), False, id="256x128-w4x2"),
        pytest.param(2656, (128, 128, 64, 2, 2, 4, 24, 8), False, id="128x128-w2x2"),
    ],
)
def test_streamk_does_not_spill(n, cfg, pin_acc) -> None:
    """Stream-K kernels with large accumulators keep everything in registers."""
    torch.manual_seed(0)
    m, k = 256, 2048
    a, w, ref = _inputs(m, n, k)
    c = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    cache = gluon_mm_a16w16_streamk_gfx950.device_caches[torch.cuda.current_device()][0]
    before = set(cache)
    launch_gluon_mm_a16w16_streamk_gfx950(a, w, c, cfg, pin_acc=pin_acc)
    torch.testing.assert_close(c, ref, atol=_ATOL, rtol=_RTOL)
    # N is compile-time and unique per case, so this launch compiled one kernel.
    (kernel,) = [cache[key] for key in cache if key not in before]
    scratch = re.search(
        r"\.private_segment_fixed_size:\s+(\d+)", kernel.asm["amdgcn"]
    ).group(1)
    assert int(scratch) == 0


@pytest.mark.parametrize(
    "cfg",
    [
        pytest.param((64, 128, 64, 2, 2, 3, 2, 8), id="pow2"),
        pytest.param((192, 160, 64, 4, 1, 3, 2, 8), id="split-mn"),
    ],
)
def test_3stage_row_count_does_not_recompile(cfg) -> None:
    """M is a runtime argument: new row counts reuse the compiled kernel."""
    torch.manual_seed(0)
    n, k = 512, 1024
    w = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)

    def run(m: int) -> None:
        a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
        c = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
        launch_gluon_mm_a16w16_3stage_gfx950(a, w, c, cfg, False)

    run(64)
    with assert_no_triton_compile(gluon_mm_a16w16_3stage_gfx950):
        for m in (16, 33, 100, 128, 200):
            run(m)


@pytest.mark.parametrize(
    "block,legal",
    [(192, True), (144, True), (80, True), (96, True), (224, False), (176, False)],
)
def test_split_tile_sizes(block, legal) -> None:
    """A tile is legal when it is lo + hi with both parts powers of two."""
    cfg = (64, block, 128, 4, 1, 2, 1, 8)
    assert gl3_config_fits(cfg, 64, 4096, 4096) == legal
    assert pers_config_fits(cfg[:6] + (256, 8), 64, 4096, 4096) == legal


# cfg = (BM, BN, BK, WM, WN, NS, GRID, XCD). A small GRID gives several
# data-parallel rounds plus a split-K tail on small shapes.
@pytest.mark.parametrize(
    "mnk,cfg,max_pieces,expect_pieces",
    [
        pytest.param(
            (64, 1152, 1024), (64, 64, 64, 2, 2, 3, 16, 8), 32, 5, id="round+tail"
        ),
        pytest.param(
            (64, 512, 4096), (64, 128, 64, 2, 2, 3, 16, 8), 32, 4, id="tail-only"
        ),
        pytest.param(
            (128, 2048, 1024), (128, 128, 64, 2, 2, 3, 16, 8), 32, 1, id="no-tail"
        ),
        pytest.param(
            (100, 1120, 1024),
            (64, 128, 64, 2, 2, 3, 8, 8),
            32,
            4,
            id="ragged-mn-2rounds",
        ),
        pytest.param(
            (64, 1152, 1024), (64, 64, 64, 2, 2, 3, 16, 8), 1, 1, id="tail-unsplit"
        ),
        # Non-power-of-two tiles in the rounds and in the split-K tail.
        pytest.param(
            (300, 1000, 2048),
            (192, 160, 64, 4, 1, 3, 12, 4),
            32,
            6,
            id="split-mn-round+tail",
        ),
        pytest.param(
            (200, 1100, 2048),
            (64, 144, 128, 4, 1, 2, 24, 8),
            32,
            3,
            id="split-n-ragged-round+tail",
        ),
        pytest.param(
            (130, 900, 1024),
            (96, 96, 128, 2, 2, 3, 8, 8),
            32,
            2,
            id="split-mn-2rounds+tail",
        ),
    ],
)
def test_persistent_matches_reference(mnk, cfg, max_pieces, expect_pieces) -> None:
    torch.manual_seed(0)
    m, n, k = mnk
    assert pers_config_fits(cfg, m, n, k)
    assert tail_pieces(cfg, m, n, k, max_pieces) == expect_pieces
    a, w, ref = _inputs(m, n, k)
    c = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    # Run three times: the tail pieces must leave their counters reset.
    for _ in range(3):
        c.fill_(float("nan"))
        launch_gluon_mm_a16w16_persistent_gfx950(a, w, c, cfg, max_pieces=max_pieces)
        torch.testing.assert_close(c, ref, atol=_ATOL, rtol=_RTOL)


def test_persistent_row_count_does_not_recompile() -> None:
    """M and the tail split are runtime arguments: no recompile across M."""
    torch.manual_seed(0)
    n, k = 1152, 1024
    cfg = (64, 64, 64, 2, 2, 3, 16, 8)
    w = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)

    def run(m: int) -> None:
        a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
        c = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
        launch_gluon_mm_a16w16_persistent_gfx950(a, w, c, cfg)

    run(64)
    with assert_no_triton_compile(gluon_mm_a16w16_persistent_gfx950):
        for m in (16, 33, 100, 128, 200):
            run(m)
