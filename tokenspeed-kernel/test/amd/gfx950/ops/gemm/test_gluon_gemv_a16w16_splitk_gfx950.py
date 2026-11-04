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

"""gfx950 K-split decode GEMV task bodies, through their standalone launch."""

from __future__ import annotations

import pytest
import torch
from utils import assert_no_triton_compile, is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)

from tokenspeed_kernel_amd._triton import gl, gluon  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.gemv_splitk import (  # noqa: E402
    gemv_a16w16_splitk_sum_gfx950,
    gluon_gemv_a16w16_splitk_gfx950,
    launch_gluon_gemv_a16w16_splitk_gfx950,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.persistent import (  # noqa: E402
    ERROR_ABORT,
    Launch,
    persistent_compile_options,
)

_ABORT = gl.constexpr(ERROR_ABORT)

_SITE = gl.constexpr(77)


@gluon.jit
def _record_expiry(launch, site, detail):
    # Test handler: record the site and return (kernel handlers never return).
    gl.store(launch.error_ptr + 8, site)


@gluon.jit
def _reader_kernel(
    part_ptr,
    y_ptr,
    error_ptr,
    M,
    tag,
    deadline_ticks,
    N: gl.constexpr,
    BLOCK: gl.constexpr,
):
    """One token's BLOCK outputs through the in-launch reader."""
    layout: gl.constexpr = gl.BlockedLayout(
        [BLOCK // (64 * gl.num_warps())], [64], [gl.num_warps()], [0]
    )
    token = gl.program_id(0)
    rows = gl.program_id(1) * BLOCK + gl.arange(0, BLOCK, layout)
    zero = gl.program_id(0) * 0
    launch = Launch(
        gl.program_id(0), zero, zero, zero.to(gl.int64), zero, zero, zero, error_ptr,
        error_ptr + _ABORT, deadline_ticks, error_ptr,
    )  # fmt: skip
    total = gemv_a16w16_splitk_sum_gfx950(
        part_ptr,
        tag,
        token * N + rows,
        rows < N,
        M,
        N,
        16,
        launch,
        _record_expiry,
        _SITE,
    )
    gl.store(y_ptr + token * N + rows, total.to(gl.bfloat16), mask=rows < N)


def _read(partials, rows, n, tag, deadline_ticks):
    y = torch.empty(rows, n, dtype=torch.bfloat16, device="cuda")
    error = torch.zeros(64, dtype=torch.int32, device="cuda")
    _reader_kernel[(rows, (n + 1023) // 1024)](
        partials,
        y,
        error,
        rows,
        tag,
        deadline_ticks,
        N=n,
        BLOCK=1024,
        num_warps=8,
        **persistent_compile_options(),
    )
    torch.cuda.synchronize()
    return y, int(error[8])


# QKV + output gate of an 8192-wide attention at TP8, and a 256-multiple tail case.
SHAPES = [(2304, 8192), (512, 4096)]


def _problem(rows: int, n: int, k: int, seed: int):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, k, device="cuda", generator=generator).bfloat16()
    w = (torch.randn(n, k, device="cuda", generator=generator) * 0.02).bfloat16()
    return x, w


def _run(x, w, tag, partials=None):
    rows, n = x.shape[0], w.shape[0]
    if partials is None:
        partials = torch.zeros(16 * rows * n, dtype=torch.int64, device="cuda")
    out = torch.empty(rows, n, dtype=torch.bfloat16, device="cuda")
    return launch_gluon_gemv_a16w16_splitk_gfx950(x, w, out, partials, tag)


# 1, 8 and 16 rows: one row, a verify step, two requests' verify rows.
@pytest.mark.parametrize("rows", [1, 8, 16])
@pytest.mark.parametrize(("n", "k"), SHAPES)
def test_matches_fp64_within_one_bf16_step(rows: int, n: int, k: int) -> None:
    x, w = _problem(rows, n, k, rows + n)
    out = _run(x, w, 7)
    expected = x.double() @ w.double().T
    # FP32 partials of 16 K slices summed in order, rounded once to BF16.
    torch.testing.assert_close(out.double(), expected, rtol=2**-7, atol=2e-3)


def test_rows_are_independent_and_runs_bitwise_stable() -> None:
    x, w = _problem(16, 2304, 8192, 3)
    full = _run(x, w, 11)
    assert torch.equal(_run(x, w, 12), full)
    for rows in (1, 7, 8):
        assert torch.equal(_run(x[:rows].contiguous(), w, 13), full[:rows]), rows


def test_reader_waits_for_this_launchs_tag() -> None:
    """Words left by an earlier use carry an older tag; the sum must not take them."""
    x, w = _problem(8, 2304, 8192, 5)
    expected = _run(x, w, 21)
    stale = torch.full(
        (16 * 8 * 2304,), (20 << 32) | 0x7F800000, dtype=torch.int64, device="cuda"
    )
    assert torch.equal(_run(x, w, 21, stale), expected)


def test_row_count_and_tag_do_not_recompile() -> None:
    x, w = _problem(16, 2304, 8192, 9)
    for rows in (1, 16):
        _run(x[:rows].contiguous(), w, 30 + rows)
    with assert_no_triton_compile(gluon_gemv_a16w16_splitk_gfx950):
        for rows in range(1, 17):
            _run(x[:rows].contiguous(), w, 100 + rows)


# A small tag, and one with its top bit set (a launch tag past 2**31).
@pytest.mark.parametrize("tag", [41, -2])
def test_in_launch_reader_sums_like_the_reduce_and_expires_on_stale_tags(
    tag: int,
) -> None:
    x, w = _problem(8, 2304, 8192, 17)
    partials = torch.zeros(16 * 8 * 2304, dtype=torch.int64, device="cuda")
    expected = _run(x, w, tag, partials)
    got, site = _read(partials, 8, 2304, tag, 200_000_000)
    assert site == 0
    assert torch.equal(got, expected)
    # Every word carries this tag; waiting for the next must stop at the
    # deadline.
    _, site = _read(partials, 8, 2304, tag + 1, 100_000)
    assert site == _SITE
