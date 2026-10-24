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

"""Megakernel BF16 x BF16 decode GEMV on gfx950."""

from __future__ import annotations

import pytest
import torch
from utils import is_cdna4

if is_cdna4():
    from tokenspeed_kernel_amd._triton import gl, gluon, triton
    from tokenspeed_kernel_amd.ops.gfx950.megakernel import bf16
    from tokenspeed_kernel_amd.ops.gfx950.megakernel.weights import bf16_tiles

    @gluon.jit
    def _rows_to_lds(x, k0, K: gl.constexpr, KS: gl.constexpr, R: gl.constexpr):
        """Values ``k0 .. k0 + KS - 1`` of BF16 rows ``x [R, K]`` into the
        GEMV's LDS operand (as :func:`bf16.load_rows` places them)."""
        KI: gl.constexpr = KS // 32
        KP: gl.constexpr = triton.next_power_of_2(KI)
        RP: gl.constexpr = triton.next_power_of_2(R)
        flat: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [2, 1, 0])
        x_smem = gl.allocate_shared_memory(gl.bfloat16, [RP, KP, 32], flat)
        lx: gl.constexpr = gl.BlockedLayout(
            [1, 1, 8], [1, 16, 4], [1, gl.num_warps(), 1], [2, 1, 0]
        )
        r = gl.arange(0, RP, gl.SliceLayout(1, gl.SliceLayout(2, lx)))
        c = gl.arange(0, KP, gl.SliceLayout(0, gl.SliceLayout(2, lx)))
        k = gl.arange(0, 32, gl.SliceLayout(0, gl.SliceLayout(1, lx)))
        rr = gl.minimum(r, R - 1)[:, None, None]
        x_smem.store(
            gl.load(
                x + rr * K + k0 + c[None, :, None] * 32 + k[None, None, :],
                mask=(c < KI)[None, :, None],
                other=0.0,
            )
        )
        gl.barrier()
        return x_smem

    @gluon.jit
    def _gemv_kernel(
        x,
        w,
        out,
        K: gl.constexpr,
        N: gl.constexpr,
        KSPLIT: gl.constexpr,
        SPLIT: gl.constexpr,
        NW: gl.constexpr,
        R: gl.constexpr,
    ):
        """``R`` BF16 rows ``x [R, K]`` times ``W [N, K]`` into FP32 partial
        sums ``out [KSPLIT, R, N]``: task ``(t, j)`` takes the rows' 16 * NW
        columns ``t ..`` and K part ``j`` of ``KSPLIT``."""
        task = gl.program_id(0)
        part = gl.program_id(1)
        KS: gl.constexpr = K // KSPLIT
        RP: gl.constexpr = triton.next_power_of_2(R)
        tiles = bf16.load_weight_tiles(
            w,
            nb0=task * NW,
            K=K,
            kit0=part * (KS // 32),
            KITS=KS // 32,
            SPLIT=SPLIT,
            NW=NW,
            GROUP=NW,
            GROUP_STRIDE=0,
        )
        x_smem = _rows_to_lds(x, part * KS, K, KS, R)
        y = bf16.gemv(x_smem, tiles, FIRST=0, ROWS=R)
        row = gl.arange(0, RP, gl.SliceLayout(1, y.type.layout))[:, None]
        n = gl.arange(0, 16 * NW, gl.SliceLayout(0, y.type.layout))[None, :]
        gl.store(out + (part * R + row) * N + task * 16 * NW + n, y, mask=row < R)

    def _gemv(x, w, ksplit, split, nw):
        rows, k = x.shape
        n = w.shape[0]
        out = torch.empty(ksplit, rows, n, device="cuda")
        _gemv_kernel[(n // (16 * nw), ksplit)](
            x, bf16_tiles(w), out, k, n, ksplit, split, nw, rows, num_warps=split * nw
        )
        return out

    @pytest.mark.parametrize(
        "k, n, ksplit, split, nw",
        [
            (2560, 256, 1, 4, 1),
            (2560, 16480, 1, 2, 2),
            (6144, 2560, 1, 4, 1),
            (10240, 320, 4, 4, 1),
            (320, 10240, 1, 2, 4),
        ],
    )
    def test_gemv_matches_the_fp32_reference(k, n, ksplit, split, nw):
        torch.manual_seed(k + n)
        x = torch.randn(1, k, device="cuda").bfloat16()
        w = (torch.randn(n, k, device="cuda") * 0.05).bfloat16()
        out = _gemv(x, w, ksplit, split, nw).sum(0)
        ref = x.float() @ w.float().T
        torch.testing.assert_close(
            out, ref, rtol=1e-5, atol=1e-5 * ref.abs().max().item()
        )

    @pytest.mark.parametrize("rows", [2, 3, 4])
    @pytest.mark.parametrize("k, n, split, nw", [(2560, 512, 2, 4), (6144, 256, 4, 1)])
    def test_gemv_rows_equal_one_row_launches(rows, k, n, split, nw):
        """Each request row of a multi-row GEMV is bitwise its own one-row GEMV."""
        torch.manual_seed(rows)
        x = torch.randn(rows, k, device="cuda").bfloat16()
        w = (torch.randn(n, k, device="cuda") * 0.05).bfloat16()
        out = _gemv(x, w, 1, split, nw)[0]
        for r in range(rows):
            single = _gemv(x[r : r + 1], w, 1, split, nw)[0]
            assert torch.equal(out[r], single[0])
