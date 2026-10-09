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

"""FP32 decode GEMM on CUDA cores for 17 to 96 rows (``decode_gemv``).

A narrow FP32 projection such as an expert router sees a few dozen rows per
decode step. Torch serves those rows with an FP32 SGEMM, after an FP32 copy
when the activations are BF16, and the cuBLAS kernel and its summation order
change with the row count. This kernel computes every output in one fixed FP32
order, the same for any row count, tile, activation dtype, graph replay or
eager launch:

* K is cut into 4 chunks of K / 4. In chunk ``s`` there are 128 lanes; lane
  ``4 t + v`` (``t`` the warp lane, ``v`` = 0..3) is a sequential FP32 FMA
  chain from +0 over ``k = s K / 4 + 128 i + 4 t + v``. BF16 activations widen
  exactly as they load.
* Each warp lane adds its 4 chains in order, ``((a0 + a1) + a2) + a3``, then
  the warp adds across lanes with an xor butterfly (offsets 16, 8, 4, 2, 1).
* The 4 chunk sums are added in chunk order from +0.

A product passes at most ``K / 512 + 11`` roundings, so the error is at most
``gamma_(K / 512 + 11) * sum_k |x_k w_k|`` plus underflow. There is no tensor
core and no fused multiply-add beyond the explicit chains, so the result does
not depend on Torch's matmul precision setting.

One CTA of ``4 * WM * WN`` warps computes a ``[BM_W * WM, BN_W * WN]`` output
tile, warp group ``s`` taking chunk ``s``: every thread holds ``BM_W * BN_W * 4``
chain accumulators and loads 4 activation elements per row and 4 weight
elements per output per step. The chunk sums meet in shared memory. One launch,
no workspace, atomics or split-K.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import gl, gluon, triton
from tokenspeed_kernel.ops.gemm.triton_gemv import _BF16_FP32_SIG
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel

__all__ = ["gluon_simt_gemm_fp32"]

# 4 chunks of 128 lanes: K must be a multiple of 512.
_K_ALIGN = 512
# The K loop is unrolled; this bounds the code size.
_MAX_K = 8192
# Launch tiles (BM_W, BN_W, WM, WN). The tile does not change the bits.
_TILE_8X2 = (8, 2, 1, 1)
_TILE_8X4 = (8, 4, 1, 1)
_TILE_4X4 = (4, 4, 1, 1)
# BF16 activations, more than 128 outputs, K from 4608: (largest row count,
# tile) per row range below K 6144 and from K 6144 on; [8, 4] past the last.
_WIDE_TILES = ((24, _TILE_8X2), (32, _TILE_8X4), (48, _TILE_4X4))
_WIDE_LONG_K_TILES = ((32, _TILE_8X4), (48, _TILE_4X4))


@gluon.jit
def _simt_gemm_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    M,
    K: gl.constexpr,
    N: gl.constexpr,
    BM_W: gl.constexpr,
    BN_W: gl.constexpr,
    WM: gl.constexpr,
    WN: gl.constexpr,
):
    """The fixed order of the module doc on a ``[BM_W * WM, BN_W * WN]`` tile."""
    CHUNKS: gl.constexpr = 4
    LANES: gl.constexpr = 128
    CHUNK: gl.constexpr = K // CHUNKS
    STEPS: gl.constexpr = CHUNK // LANES
    BM: gl.constexpr = BM_W * WM
    BN: gl.constexpr = BN_W * WN
    # [chunk, row, output, lane]: lane 4 t + v is thread t's element v.
    acc_l: gl.constexpr = gl.BlockedLayout(
        [1, BM_W, BN_W, 4], [1, 1, 1, 32], [CHUNKS, WM, WN, 1], [3, 2, 1, 0]
    )
    x_l: gl.constexpr = gl.SliceLayout(2, acc_l)
    w_l: gl.constexpr = gl.SliceLayout(1, acc_l)
    smem_l: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [2, 1, 0])
    out_l: gl.constexpr = gl.BlockedLayout(
        [1, 1, 1], [1, 1, 32], [1, CHUNKS * WM * WN, 1], [2, 1, 0]
    )

    m0 = gl.program_id(1) * BM
    n0 = gl.program_id(0) * BN

    xs = gl.arange(0, CHUNKS, layout=gl.SliceLayout(1, gl.SliceLayout(2, x_l)))
    xr = gl.arange(0, BM, layout=gl.SliceLayout(0, gl.SliceLayout(2, x_l)))
    xk = gl.arange(0, LANES, layout=gl.SliceLayout(0, gl.SliceLayout(1, x_l)))
    xs3 = gl.expand_dims(gl.expand_dims(xs, 1), 2)
    rows = m0 + gl.expand_dims(gl.expand_dims(xr, 0), 2)
    xk3 = gl.expand_dims(gl.expand_dims(xk, 0), 1)
    x_mask = (rows < M) & (xk3 >= 0) & (xs3 >= 0)
    x_ptrs = x_ptr + rows.to(gl.int64) * K + xs3 * CHUNK + xk3

    ws = gl.arange(0, CHUNKS, layout=gl.SliceLayout(1, gl.SliceLayout(2, w_l)))
    wn = gl.arange(0, BN, layout=gl.SliceLayout(0, gl.SliceLayout(2, w_l)))
    wk = gl.arange(0, LANES, layout=gl.SliceLayout(0, gl.SliceLayout(1, w_l)))
    ws3 = gl.expand_dims(gl.expand_dims(ws, 1), 2)
    cols = n0 + gl.expand_dims(gl.expand_dims(wn, 0), 2)
    wk3 = gl.expand_dims(gl.expand_dims(wk, 0), 1)
    w_ptrs = w_ptr + cols.to(gl.int64) * K + ws3 * CHUNK + wk3

    partials = gl.allocate_shared_memory(gl.float32, [CHUNKS, BM, BN], smem_l)

    acc = gl.zeros([CHUNKS, BM, BN, LANES], gl.float32, acc_l)
    for i in gl.static_range(STEPS):
        xv = gl.load(x_ptrs + i * LANES, mask=x_mask, other=0.0).to(gl.float32)
        wv = gl.load(w_ptrs + i * LANES)
        xb, _ = gl.broadcast(gl.expand_dims(xv, 2), acc)
        wb, _ = gl.broadcast(gl.expand_dims(wv, 1), acc)
        acc = gl.fma(xb, wb, acc)
    # Lane 4 t + v -> (t, v // 2, v % 2): split the 4 chains out of each thread
    # and add them in order (gl.sum over them would pair (a0 + a2) + (a1 + a3)),
    # then gl.sum over the 32 warp lanes is the xor butterfly.
    even, odd = gl.split(gl.reshape(acc, [CHUNKS, BM, BN, 32, 2, 2]))
    a0, a2 = gl.split(even)
    a1, a3 = gl.split(odd)
    partials.store(gl.sum(((a0 + a1) + a2) + a3, axis=3))
    gl.barrier()

    total = gl.zeros([1, BM, BN], gl.float32, out_l)
    for s in gl.static_range(CHUNKS):
        total = total + partials.slice(s, 1, dim=0).load(out_l)
    orow = m0 + gl.arange(0, BM, layout=gl.SliceLayout(0, gl.SliceLayout(2, out_l)))
    ocol = n0 + gl.arange(0, BN, layout=gl.SliceLayout(0, gl.SliceLayout(1, out_l)))
    orow3 = gl.expand_dims(gl.expand_dims(orow, 0), 2)
    ocol3 = gl.expand_dims(gl.expand_dims(ocol, 0), 1)
    out_ptrs = out_ptr + orow3.to(gl.int64) * N + ocol3
    gl.store(out_ptrs, total, mask=(orow3 < M) & (ocol3 >= 0))


def _simt_tile(
    m: int, n: int, k: int, x_dtype: torch.dtype
) -> tuple[int, int, int, int]:
    """``(BM_W, BN_W, WM, WN)`` of the launch for ``[m, k]`` x ``[n, k]``.

    From GB200 graph timings of each tile at 24 to 96 rows: BF16 activations
    with 128 x 4096 and with 256 outputs at K 3584 to 7168, FP32 activations
    with three of those weights. [8, 2] slows down by K 6144 (1.9 to 2.4x
    slower than [8, 4] at 256 x 7168). With BF16 activations [8, 4] was
    up to 1.9x slower than [8, 2] with 128 outputs and than [4, 4] at K 3584,
    and with 256 outputs from K 4608 the fastest tile changes with the rows.
    FP32 activations hold twice the registers per element and spill in
    [8, 4] from K 3584.
    """
    if x_dtype != torch.bfloat16 or n <= 128:
        return _TILE_8X2 if k < 6144 else _TILE_4X4
    if k < 4608:
        return _TILE_4X4
    for max_rows, tile in _WIDE_TILES if k < 6144 else _WIDE_LONG_K_TILES:
        if m <= max_rows:
            return tile
    return _TILE_8X4


# Registered where it beat Torch's FP32 path at every point of a GB200 module
# bench (CUDA graphs of 49 calls over cold weights, 291 points): BF16
# activations, 132 points with N 4 to 256 and K 512 to 8192 from 17 to 96
# rows, at 1.23x to 5.17x (median 2.72x). FP32 activations, and N 512 or 1024,
# were slower than Torch at some points in those rows (28 of 105 and 3 of 36)
# and are left to Torch. Against the BF16x3 split kernel on 256-row weights (K
# 3072 to 7168, separate runs), this kernel was faster at every K up to 48
# rows, while at 64 and 96 rows the split kernel was faster at some K from
# 4096, by up to 1.29x; the band is not cut there, since without a split this
# kernel is the only fast path. One step above the band start, so a broader
# FP32 kernel registered at the band start for the same rows does not take
# them.
@register_kernel(
    "gemm",
    "decode_gemv",
    name="gluon_simt_gemm_fp32",
    solution="gluon",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    signatures=_BF16_FP32_SIG,
    traits={
        "m": frozenset(range(17, 97)),
        "n_align": frozenset({4}),
        "n_max": frozenset({256}),
        "k_align": frozenset({_K_ALIGN}),
        "k_min": frozenset({_K_ALIGN}),
        "k_max": frozenset({_MAX_K}),
    },
    priority=Priority.SPECIALIZED + 1,
)
def gluon_simt_gemm_fp32(
    x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor:
    """``x @ weight.T`` in FP32 in the fixed order of the module doc.

    Args:
        x: ``[M, K]`` contiguous FP32 or BF16 activations, K a multiple of 512
            up to 8192.
        weight: ``[N, K]`` contiguous FP32 weight, N a multiple of 4.
        out: optional contiguous ``[M, N]`` FP32 destination.

    Returns:
        ``[M, N]`` FP32 output.
    """
    m, k = x.shape
    n = weight.shape[0]
    assert x.is_contiguous() and weight.is_contiguous()
    assert k % _K_ALIGN == 0 and k <= _MAX_K and n % 4 == 0
    if out is None:
        out = torch.empty(m, n, dtype=torch.float32, device=x.device)
    if m == 0 or n == 0 or k == 0:
        return out.zero_()
    bm_w, bn_w, wm, wn = _simt_tile(m, n, k, x.dtype)
    _simt_gemm_kernel[(n // (bn_w * wn), triton.cdiv(m, bm_w * wm))](
        x,
        weight,
        out,
        m,
        K=k,
        N=n,
        BM_W=bm_w,
        BN_W=bn_w,
        WM=wm,
        WN=wn,
        num_warps=4 * wm * wn,
        enable_fp_fusion=False,
    )
    return out
