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

"""Megakernel MXFP4 x MXFP8 GEMV and MXFP8 quantization on gfx950."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel_amd.ops.gfx950.megakernel.weights import (
    dequantize_mxfp4,
    mxfp4_scales,
    mxfp4_tiles,
    quantize_mxfp4_exact,
)
from utils import is_cdna4


def test_exact_mxfp4_round_trip_and_rejection():
    torch.manual_seed(0)
    codes = torch.randint(0, 256, (32, 64), dtype=torch.uint8)
    scales = torch.randint(118, 130, (32, 4), dtype=torch.uint8)
    values = dequantize_mxfp4(codes, scales)
    packed, got_scales = quantize_mxfp4_exact(values.bfloat16())
    torch.testing.assert_close(
        dequantize_mxfp4(packed, got_scales), values, rtol=0, atol=0
    )
    with pytest.raises(ValueError, match="not on an MXFP4 grid"):
        quantize_mxfp4_exact(values.bfloat16() + 0.0078125 * values.abs().max())


def _mxfp8_reference(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-32 E4M3 with the exponent of amax/448 rounded up (TokenSpeed rule)."""
    blocks = x.float().view(-1, 32)
    ratio = (blocks.abs().amax(-1) * (1.0 / 448.0)).view(torch.int32)
    exponent = (ratio >> 23) & 255
    exponent = exponent + ((exponent < 255) & ((ratio & 0x7FFFFF) != 0)).int()
    q = (blocks / torch.exp2(exponent.float() - 127)[:, None]).to(torch.float8_e4m3fn)
    return q.view(x.shape), exponent.to(torch.uint8)


if is_cdna4():
    from tokenspeed_kernel_amd._triton import gl, gluon, triton
    from tokenspeed_kernel_amd.ops.gfx950.megakernel import mx

    @gluon.jit
    def _quantize_kernel(x, q, e, G: gl.constexpr):
        layout: gl.constexpr = gl.BlockedLayout(
            [1, 16], [32, 2], [gl.num_warps(), 1], [1, 0]
        )
        g = gl.arange(0, G, gl.SliceLayout(1, layout))
        j = gl.arange(0, 32, gl.SliceLayout(0, layout))
        values = gl.load(x + g[:, None] * 32 + j[None, :])
        fp8, exponent = mx.quantize_mxfp8(values)
        gl.store(q + g[:, None] * 32 + j[None, :], fp8.to(gl.uint8, bitcast=True))
        gl.store(e + g, exponent)

    def test_quantize_mxfp8_matches_the_tokenspeed_rule():
        torch.manual_seed(0)
        x = (
            torch.randn(64, 32, device="cuda")
            * torch.logspace(-8, 8, 64, device="cuda")[:, None]
        )
        x[0] = 0
        q = torch.empty(64, 32, dtype=torch.uint8, device="cuda")
        e = torch.empty(64, dtype=torch.uint8, device="cuda")
        _quantize_kernel[(1,)](x, q, e, 64, num_warps=4)
        ref_q, ref_e = _mxfp8_reference(x)
        assert torch.equal(e, ref_e)
        assert torch.equal(q, ref_q.view(torch.uint8))

    @gluon.jit
    def _rows_to_lds(x8, xs, K: gl.constexpr, R: gl.constexpr):
        """MXFP8 rows ``x8 [R, K]`` and exponents ``xs`` into the GEMV's LDS
        operands."""
        KI: gl.constexpr = K // 128
        KP: gl.constexpr = triton.next_power_of_2(KI)
        RP: gl.constexpr = triton.next_power_of_2(R)
        flat: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [2, 1, 0])
        x_smem = gl.allocate_shared_memory(gl.uint8, [RP, KP, 128], flat)
        xs_smem = gl.allocate_shared_memory(gl.uint8, [RP, KP, 4], flat)
        lx: gl.constexpr = gl.BlockedLayout(
            [1, 1, 16], [1, 8, 8], [1, gl.num_warps(), 1], [2, 1, 0]
        )
        r = gl.arange(0, RP, gl.SliceLayout(1, gl.SliceLayout(2, lx)))
        c = gl.arange(0, KP, gl.SliceLayout(0, gl.SliceLayout(2, lx)))
        k = gl.arange(0, 128, gl.SliceLayout(0, gl.SliceLayout(1, lx)))
        rr = gl.minimum(r, R - 1)[:, None, None]
        x_smem.store(
            gl.load(
                x8 + rr * K + c[None, :, None] * 128 + k[None, None, :],
                mask=(c < KI)[None, :, None],
                other=0,
            )
        )
        ls: gl.constexpr = gl.BlockedLayout(
            [1, 1, 4], [1, 64, 1], [1, gl.num_warps(), 1], [2, 1, 0]
        )
        rs = gl.arange(0, RP, gl.SliceLayout(1, gl.SliceLayout(2, ls)))
        cs = gl.arange(0, KP, gl.SliceLayout(0, gl.SliceLayout(2, ls)))
        g = gl.arange(0, 4, gl.SliceLayout(0, gl.SliceLayout(1, ls)))
        rrs = gl.minimum(rs, R - 1)[:, None, None]
        xs_smem.store(
            gl.load(
                xs + rrs * (K // 32) + cs[None, :, None] * 4 + g[None, None, :],
                mask=(cs < KI)[None, :, None],
                other=127,
            )
        )
        gl.barrier()
        return x_smem, xs_smem

    @gluon.jit
    def _gemv_kernel(
        x8,
        xs,
        w,
        ws,
        out,
        K: gl.constexpr,
        N: gl.constexpr,
        SPLIT: gl.constexpr,
        NW: gl.constexpr,
        R: gl.constexpr,
    ):
        """``R`` activation rows ``x8 [R, K]`` times ``W``, into ``out [R, N]``."""
        task = gl.program_id(0)
        RP: gl.constexpr = triton.next_power_of_2(R)
        tiles, scales = mx.load_weight_tiles(
            w,
            ws,
            nb0=task * NW,
            K=K,
            kit0=0,
            KITS=K // 128,
            SPLIT=SPLIT,
            NW=NW,
            GROUP=NW,
            GROUP_STRIDE=0,
            SCALES="k4",
        )
        x_smem, xs_smem = _rows_to_lds(x8, xs, K, R)
        y = mx.gemv(
            x_smem,
            xs_smem,
            tiles,
            scales,
            SPLIT=SPLIT,
            NW=NW,
            SEG=0,
            SEG_STRIDE=0,
            FIRST=0,
            ROWS=R,
        )
        row = gl.arange(0, RP, gl.SliceLayout(1, y.type.layout))[:, None]
        n = gl.arange(0, 16 * NW, gl.SliceLayout(0, y.type.layout))[None, :]
        gl.store(out + row * N + task * 16 * NW + n, y, mask=row < R)

    def _gemv(q, e, packed, scales, k, n, split, nw):
        rows = q.shape[0]
        out = torch.empty(rows, n, device="cuda")
        _gemv_kernel[(n // (16 * nw),)](
            q.view(torch.uint8),
            e,
            mxfp4_tiles(packed, k),
            mxfp4_scales(scales, k),
            out,
            k,
            n,
            split,
            nw,
            rows,
            num_warps=split * nw,
        )
        return out

    @pytest.mark.parametrize(
        "k, n, split, nw",
        [
            (2560, 256, 4, 1),
            (6144, 2560, 4, 1),
            (10240, 64, 4, 2),
            (384, 512, 1, 8),
            (2560, 512, 2, 4),
        ],
    )
    def test_gemv_matches_dequantized_reference(k, n, split, nw):
        torch.manual_seed(k + n)
        x = torch.randn(k, device="cuda")
        q, e = _mxfp8_reference(x)
        packed = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="cuda")
        scales = torch.randint(118, 130, (n, k // 32), dtype=torch.uint8, device="cuda")
        out = _gemv(q[None], e[None], packed, scales, k, n, split, nw)[0]
        x_deq = q.float() * torch.exp2(e.float() - 127).repeat_interleave(32)
        ref = dequantize_mxfp4(packed, scales) @ x_deq
        torch.testing.assert_close(
            out, ref, rtol=1e-5, atol=1e-5 * ref.abs().max().item()
        )

    @pytest.mark.parametrize("rows", [2, 3, 4])
    @pytest.mark.parametrize("k, n, split, nw", [(2560, 512, 2, 4), (6144, 256, 4, 1)])
    def test_gemv_rows_equal_one_row_launches(rows, k, n, split, nw):
        """Each request row of a multi-row GEMV is bitwise its own one-row GEMV."""
        torch.manual_seed(rows)
        q, e = zip(
            *(_mxfp8_reference(torch.randn(k, device="cuda")) for _ in range(rows))
        )
        q, e = torch.stack(q), torch.stack(e)
        packed = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="cuda")
        scales = torch.randint(118, 130, (n, k // 32), dtype=torch.uint8, device="cuda")
        out = _gemv(q, e, packed, scales, k, n, split, nw)
        for r in range(rows):
            single = _gemv(q[r : r + 1], e[r : r + 1], packed, scales, k, n, split, nw)
            assert torch.equal(out[r], single[0])

    @gluon.jit
    def _gemv_row_kernel(
        x8,
        xs,
        w,
        ws,
        out,
        K: gl.constexpr,
        N: gl.constexpr,
        SPLIT: gl.constexpr,
        NW: gl.constexpr,
        R: gl.constexpr,
    ):
        """Row ``r`` of ``x8 [R, K]`` times its own weight ``W_r`` (``w`` and
        ``ws`` hold ``R`` stacked weights) through ``mx.gemv_row``."""
        task = gl.program_id(0)
        RP: gl.constexpr = triton.next_power_of_2(R)
        x_smem, xs_smem = _rows_to_lds(x8, xs, K, R)
        acc = mx.gemv_zero(SPLIT, NW)
        for r in gl.static_range(R):
            tiles, scales = mx.load_weight_tiles(
                w + r * (N * K // 2),
                ws + r * (N // 16 * ((K // 128 + 3) // 4) * 64),
                nb0=task * NW,
                K=K,
                kit0=0,
                KITS=K // 128,
                SPLIT=SPLIT,
                NW=NW,
                GROUP=NW,
                GROUP_STRIDE=0,
                SCALES="k4",
            )
            acc = mx.gemv_row(acc, x_smem, xs_smem, tiles, scales, r)
        y = mx.gemv_result(acc, RP)
        row = gl.arange(0, RP, gl.SliceLayout(1, y.type.layout))[:, None]
        n = gl.arange(0, 16 * NW, gl.SliceLayout(0, y.type.layout))[None, :]
        gl.store(out + row * N + task * 16 * NW + n, y, mask=row < R)

    @pytest.mark.parametrize("rows", [1, 2, 3, 4])
    def test_gemv_row_gives_each_row_its_own_gemv(rows):
        """Rows with their own weights, gathered into one accumulator: each
        row is bitwise its one-row GEMV with its weights."""
        k, n, split, nw = 2560, 256, 2, 4
        torch.manual_seed(rows)
        q, e = zip(
            *(_mxfp8_reference(torch.randn(k, device="cuda")) for _ in range(rows))
        )
        q, e = torch.stack(q), torch.stack(e)
        packed = torch.randint(
            0, 256, (rows, n, k // 2), dtype=torch.uint8, device="cuda"
        )
        scales = torch.randint(
            118, 130, (rows, n, k // 32), dtype=torch.uint8, device="cuda"
        )
        out = torch.empty(rows, n, device="cuda")
        _gemv_row_kernel[(n // (16 * nw),)](
            q.view(torch.uint8),
            e,
            torch.stack([mxfp4_tiles(packed[r], k) for r in range(rows)]),
            torch.stack([mxfp4_scales(scales[r], k) for r in range(rows)]),
            out,
            k,
            n,
            split,
            nw,
            rows,
            num_warps=split * nw,
        )
        for r in range(rows):
            single = _gemv(
                q[r : r + 1], e[r : r + 1], packed[r], scales[r], k, n, split, nw
            )
            assert torch.equal(out[r], single[0])

    @gluon.jit
    def _bank_gemv_kernel(
        x8, xs, w13, s13, out, expert, STRIDE: gl.constexpr, SSTRIDE: gl.constexpr
    ):
        """Gate/up rows of one expert of an N16 MoE bank (``"n32"`` scales)."""
        task = gl.program_id(0)
        e = expert.to(gl.int64)
        tiles, scales = mx.load_weight_tiles(
            w13 + e * STRIDE,
            s13 + e * SSTRIDE,
            nb0=task * 4,
            K=2560,
            kit0=0,
            KITS=20,
            SPLIT=2,
            NW=4,
            GROUP=4,
            GROUP_STRIDE=0,
            SCALES="n32",
        )
        flat: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
        x_smem = gl.allocate_shared_memory(gl.uint8, [32, 128], flat)
        xs_smem = gl.allocate_shared_memory(gl.uint8, [32, 4], flat)
        lx: gl.constexpr = gl.BlockedLayout(
            [1, 16], [8, 8], [gl.num_warps(), 1], [1, 0]
        )
        c = gl.arange(0, 32, gl.SliceLayout(1, lx))
        k = gl.arange(0, 128, gl.SliceLayout(0, lx))
        x_smem.store(
            gl.load(x8 + c[:, None] * 128 + k[None, :], mask=(c < 20)[:, None], other=0)
        )
        ls: gl.constexpr = gl.BlockedLayout(
            [1, 4], [64, 1], [gl.num_warps(), 1], [1, 0]
        )
        cs = gl.arange(0, 32, gl.SliceLayout(1, ls))
        g = gl.arange(0, 4, gl.SliceLayout(0, ls))
        xs_smem.store(
            gl.load(
                xs + cs[:, None] * 4 + g[None, :], mask=(cs < 20)[:, None], other=127
            )
        )
        gl.barrier()
        y = mx.gemv(
            x_smem.reshape([1, 32, 128]),
            xs_smem.reshape([1, 32, 4]),
            tiles,
            scales,
            SPLIT=2,
            NW=4,
            SEG=0,
            SEG_STRIDE=0,
            FIRST=0,
            ROWS=1,
        )
        y = gl.reshape(y, [64])
        gl.store(out + task * 64 + gl.arange(0, 64, y.type.layout), y)

    def test_gemv_reads_tokenspeed_moe_banks():
        from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.n16_weights import (
            pad_mxfp4_moe_intermediate,
            preprocess_n16_mxfp4_weights,
        )

        torch.manual_seed(3)
        experts, inter, hidden = 8, 640, 2560
        w13 = torch.randint(
            0, 256, (experts, 2 * inter, hidden // 2), dtype=torch.uint8, device="cuda"
        )
        s13 = torch.randint(
            118,
            130,
            (experts, 2 * inter, hidden // 32),
            dtype=torch.uint8,
            device="cuda",
        )
        w2 = torch.zeros(experts, hidden, inter // 2, dtype=torch.uint8, device="cuda")
        s2 = torch.full(
            (experts, hidden, inter // 32), 127, dtype=torch.uint8, device="cuda"
        )
        bank = torch.nn.Module()
        for name, value in zip(
            ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale"),
            (w13.clone(), s13.clone(), w2, s2),
        ):
            bank.register_parameter(
                name, torch.nn.Parameter(value, requires_grad=False)
            )
        pad_mxfp4_moe_intermediate(bank, 256)
        preprocess_n16_mxfp4_weights(bank)
        x = torch.randn(hidden, device="cuda")
        q, e = _mxfp8_reference(x)
        x_deq = q.float() * torch.exp2(e.float() - 127).repeat_interleave(32)
        expert = 5
        out = torch.empty(2 * 768, device="cuda")
        _bank_gemv_kernel[(2 * 768 // 64,)](
            q.view(torch.uint8),
            e,
            bank.w13_weight,
            bank.w13_weight_scale,
            out,
            expert,
            2 * 768 * hidden // 2,
            768 // 16 * 10 * 256,
            num_warps=8,
        )
        # Bank row-block b holds gate rows (b even) or up rows (b odd) of
        # block b // 2.
        ref = dequantize_mxfp4(w13[expert], s13[expert]) @ x_deq
        blocks = out.view(-1, 2, 16)[: inter // 16]
        torch.testing.assert_close(
            blocks[:, 0].flatten(),
            ref[:inter],
            rtol=1e-4,
            atol=1e-4 * ref.abs().max().item(),
        )
        torch.testing.assert_close(
            blocks[:, 1].flatten(),
            ref[inter:],
            rtol=1e-4,
            atol=1e-4 * ref.abs().max().item(),
        )
