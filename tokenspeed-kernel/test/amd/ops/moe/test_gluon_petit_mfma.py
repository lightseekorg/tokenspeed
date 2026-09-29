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

"""Compare the tensor MFMA layout against the original native instructions."""

import pytest
import torch
from tokenspeed_kernel.thirdparty.gluon_petit import load_petit_kernel

load_petit_kernel()

import triton.experimental.gluon as g
from lib.gemm.rocm.cdna4_ops import scaled_mfma_tile
from lib.gemm.rocm.intrinsics import _native_call
from triton.experimental.gluon import language as l

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx950"),
    reason="requires GFX950",
)


HAS_AMD_SCALE_FP4_MFMA = l.constexpr(True)


@g.jit
def mma_scale_m16n16k128_fp4_fp4_f32(
    fa, scale_a, fb, scale_b, c, kOpSelA: l.constexpr, kOpSelB: l.constexpr
):
    if HAS_AMD_SCALE_FP4_MFMA:
        return _native_call(
            "llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32",
            "v4f32",
            ("v8i32", "v8i32", "v4f32", "#i32", "#i32", "#i32", "i32", "#i32", "i32"),
            (
                fa + (0, 0, 0, 0),
                fb + (0, 0, 0, 0),
                c,
                4,
                4,
                kOpSelA,
                scale_a,
                kOpSelB,
                scale_b,
            ),
            True,
        )
    else:
        return c


@g.jit
def ScaledMxFp4Mfma(
    opsel_a: l.constexpr, opsel_b: l.constexpr, a, scale_a, b, scale_b, acc
):
    # The native dispatch lambdas specialize both immediate selectors.
    if opsel_a == 0:
        kOpSelA: l.constexpr = 0
    elif opsel_a == 1:
        kOpSelA: l.constexpr = 1
    elif opsel_a == 2:
        kOpSelA: l.constexpr = 2
    else:
        kOpSelA: l.constexpr = 3
    if opsel_b == 0:
        kOpSelB: l.constexpr = 0
    elif opsel_b == 1:
        kOpSelB: l.constexpr = 1
    elif opsel_b == 2:
        kOpSelB: l.constexpr = 2
    else:
        kOpSelB: l.constexpr = 3
    return mma_scale_m16n16k128_fp4_fp4_f32(
        a, scale_a, b, scale_b, acc, kOpSelA, kOpSelB
    )


@g.jit
def _native_reference(acc, w, x, sx, sw, M: l.constexpr, N: l.constexpr):
    for k in l.static_range(2):
        for n in l.static_range(N):
            for m in l.static_range(M):
                i: l.constexpr
                i = n * M + m
                result = ScaledMxFp4Mfma(
                    2 * k + (n & 1),
                    2 * k + (m & 1),
                    w[k][n],
                    sw[n // 2],
                    x[m * 2 + k],
                    sx[m // 2],
                    acc[i],
                )
                acc = acc[:i] + (result,) + acc[i + 1 :]
    return acc


@g.jit
def _read_fragments(ptr, count: l.constexpr, tid, threads: l.constexpr):
    fragments = ()
    for i in l.static_range(count):
        words = ()
        for j in l.static_range(4):
            words += (l.load(ptr + (i * 4 + j) * threads + tid),)
        fragments += (words,)
    return fragments


@g.jit
def _compare_mfma(X, W, SX, SW, C, actual, expected, M: l.constexpr, N: l.constexpr):
    T: l.constexpr = l.num_warps() * 64
    tid = l.arange(0, T, layout=l.BlockedLayout([1], [64], [l.num_warps()], [0]))
    x = _read_fragments(X, M * 2, tid, T)
    w = (_read_fragments(W, N, tid, T), _read_fragments(W + N * 4 * T, N, tid, T))
    acc = _read_fragments(C, M * N, tid, T)
    sx = ()
    sw = ()
    for i in l.static_range(M // 2):
        sx += (l.load(SX + i * T + tid),)
    for i in l.static_range(N // 2):
        sw += (l.load(SW + i * T + tid),)
    a = scaled_mfma_tile(acc, w, x, sx, sw, M, N)
    b = _native_reference(acc, w, x, sx, sw, M, N)
    for i in l.static_range(M * N):
        for j in l.static_range(4):
            offset = (i * 4 + j) * T + tid
            l.store(actual + offset, a[i][j])
            l.store(expected + offset, b[i][j])


@pytest.mark.parametrize(
    "tile", ((32, 32), (32, 64), (32, 128), (32, 256), (64, 32), (64, 64))
)
@pytest.mark.parametrize("warps", (4, 8))
def test_scaled_mfma_matches_native(tile, warps):
    m, n = (value // 16 for value in tile)
    threads = warps * 64
    generator = torch.Generator(device="cuda").manual_seed(42)

    def packed(shape):
        return torch.randint(
            0, 2**32, shape, generator=generator, device="cuda", dtype=torch.int64
        ).to(torch.int32)

    def scales(count):
        words = torch.zeros((count, threads), device="cuda", dtype=torch.int32)
        for byte in range(4):
            words |= torch.randint(
                123,
                130,
                words.shape,
                generator=generator,
                device="cuda",
                dtype=torch.int32,
            ) << (byte * 8)
        return words

    x, w = packed((m * 2, 4, threads)), packed((2, n, 4, threads))
    sx, sw = scales(m // 2), scales(n // 2)
    acc = torch.randn((m * n, 4, threads), generator=generator, device="cuda")
    actual, expected = torch.empty_like(acc), torch.empty_like(acc)
    _compare_mfma[(1,)](
        x,
        w,
        sx,
        sw,
        acc,
        actual,
        expected,
        m,
        n,
        num_warps=warps,
        enable_fp_fusion=False,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
